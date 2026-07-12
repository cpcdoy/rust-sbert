//! `SentenceTransformer<T>` — the pipeline driver.
//!
//! Replaces the original `SBert<T>` (which was hard-coded to DistilBERT +
//! mandatory Dense). The new driver reads `modules.json` at the model root
//! and composes the post-transformer pipeline from the manifest. The
//! transformer backend is also `modules.json`-driven and dispatches on the
//! config's `model_type` (`bert`, `distilbert`, ...).
//!
//! `SBert<T>` is preserved as a `pub type` alias in [`crate::lib`] for
//! backward compatibility — existing call sites (`SBertRT::new(...).forward(...)`)
//! keep working unchanged.

use std::convert::TryFrom;
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;
use tch::{nn, Device, Tensor};

use crate::models::pad_sort;
use crate::modules::{
    self, manifest, transformer as transformer_mod, Features, Module, TransformerBackend,
};
use crate::tokenizers::Tokenizer;
use crate::{att, Attentions, Embeddings, Error};

pub struct SentenceTransformer<T> {
    transformer: Box<dyn TransformerBackend>,
    /// Ordered post-transformer pipeline (Pooling, optional Dense, optional
    /// Normalize, ...) built from `modules.json`. The transformer is NOT in
    /// this list — it has a different I/O shape and is held separately.
    post: Vec<Box<dyn Module>>,
    tokenizer: Arc<T>,
    device: Device,
}

impl<T> SentenceTransformer<T>
where
    T: Tokenizer + Send + Sync,
{
    /// Load a sentence-transformers checkpoint from `root`.
    ///
    /// `root` must contain `modules.json`. The subdirectory each module
    /// lives in is resolved by [`manifest::resolve_module_dir`], which
    /// tolerates three real-world layouts:
    ///
    /// * `"path": ""` (transformer files at the model root — modern HF
    ///   convention, e.g. `all-MiniLM-L6-v2`).
    /// * `"path": "0_BERT"` / `"path": "1_Pooling"` etc. (truthful manifest).
    /// * `"path": "0_Transformer"` but on-disk dir is e.g. `0_DistilBERT`
    ///   (legacy UKPabs distiluse export — falls back to `<idx>_*` scan).
    pub fn new<P>(root: P, device: Option<Device>) -> Result<Self, Error>
    where
        P: Into<PathBuf>,
    {
        let root = root.into();
        let device = device.unwrap_or_else(Device::cuda_if_available);
        log::info!("Using device {:?}", device);

        let entries = manifest::parse(&root)?;

        let mut transformer_entry: Option<&manifest::ModuleEntry> = None;
        let mut transformer_dir: Option<PathBuf> = None;
        let mut post: Vec<Box<dyn Module>> = Vec::new();

        // Walk the manifest in declared order; the first Transformer entry
        // becomes the backend, everything else gets pushed into the
        // post-pipeline. Module directories are resolved with fallback logic
        // because the manifest's `path` field is unreliable in the wild.
        for entry in &entries {
            let module_dir = manifest::resolve_module_dir(&root, entry);
            match entry.short_type() {
                "Transformer" => {
                    if transformer_entry.is_some() {
                        return Err(Error::Encoding(
                            "modules.json declares more than one Transformer module",
                        ));
                    }
                    transformer_entry = Some(entry);
                    transformer_dir = Some(module_dir);
                }
                "Pooling" => {
                    post.push(Box::new(modules::Pooling::new(&module_dir)?));
                }
                "Dense" => {
                    post.push(Box::new(modules::Dense::new(&module_dir, device)?));
                }
                "Normalize" => {
                    post.push(Box::new(modules::Normalize::new()));
                }
                other => {
                    log::warn!(
                        "skipping modules.json entry {:?}: unsupported type \"{}\"",
                        entry.name,
                        other
                    );
                }
            }
        }

        let _transformer_entry = transformer_entry.ok_or_else(|| {
            Error::Encoding("modules.json has no Transformer entry — cannot build pipeline")
        })?;
        let transformer_dir = transformer_dir.ok_or_else(|| {
            // Unreachable: transformer_entry being Some implies transformer_dir is too.
            Error::Encoding("internal: transformer_dir not set")
        })?;

        let mut vs = nn::VarStore::new(device);
        let (transformer, vocab_dir) =
            transformer_mod::load(&transformer_dir, &vs.root(), device)?;
        let tokenizer = Arc::new(T::new(&vocab_dir.join("vocab.txt"))?);

        // Load the transformer weights AFTER the VarStore paths have been
        // wired up by the backend constructor.
        let weights_file = transformer_dir.join("model.ot");
        vs.load(weights_file)?;

        Ok(SentenceTransformer {
            transformer,
            post,
            tokenizer,
            device,
        })
    }

    pub fn forward<S, B>(&self, input: &[S], batch_size: B) -> Result<Vec<Embeddings>, Error>
    where
        S: AsRef<str>,
        B: Into<Option<usize>>,
    {
        let input = input.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
        let batch_size = batch_size.into().unwrap_or_else(|| 64);

        let _guard = tch::no_grad_guard();

        let sorted_pad_input_idx = pad_sort(&input.iter().map(|s| s.len()).collect::<Vec<usize>>());
        let sorted_pad_input = sorted_pad_input_idx
            .iter()
            .map(|i| input[*i])
            .collect::<Vec<&str>>();

        let input_len = sorted_pad_input.len();
        let tokenizer = self.tokenizer.clone();
        let device = self.device;

        // Tokenize
        let tokenized_batches = (0..input_len)
            .into_par_iter()
            .step_by(batch_size)
            .map(|batch_i| {
                let max_range = std::cmp::min(batch_i + batch_size, input_len);
                let range = batch_i..max_range;

                log::info!(
                    "Batch {}/{}, size {}",
                    (batch_i as f64 / batch_size as f64).ceil() as usize + 1,
                    (input_len as f64 / batch_size as f64).ceil() as usize,
                    max_range - batch_i
                );

                let (tokenized_input, attention) = tokenizer.tokenize(&sorted_pad_input[range]);

                let batch_tensor = Tensor::stack(&tokenized_input, 0).to(device);
                let batch_attention = Tensor::stack(&attention, 0).to(device);

                (batch_tensor, batch_attention)
            })
            .collect::<Vec<(Tensor, Tensor)>>();

        // Embed + run pipeline
        let mut batch_tensors = Vec::<Embeddings>::with_capacity(input_len);

        for (batch_tensor, batch_attention) in tokenized_batches.into_iter() {
            let batch_attention_c = batch_attention.shallow_clone();

            let out = self
                .transformer
                .forward(&batch_tensor, &batch_attention, false)?;

            let mut features = Features::Token {
                token_embeddings: out.hidden_state,
                attention_mask: batch_attention_c,
            };
            for module in &self.post {
                module.forward(&mut features)?;
            }

            let embedding = match features {
                Features::Sentence { embedding } => embedding,
                Features::Token { .. } => {
                    return Err(Error::Encoding(
                        "pipeline ended without producing a sentence embedding (no Pooling module?)",
                    ));
                }
            };
            batch_tensors.extend(Vec::<Embeddings>::try_from(embedding).unwrap());
        }

        // Sort results
        let sorted_pad_input_idx = pad_sort(&sorted_pad_input_idx);
        let batch_tensors = sorted_pad_input_idx
            .into_iter()
            .map(|i| mem::replace(&mut batch_tensors[i], vec![]))
            .collect::<Vec<_>>();

        Ok(batch_tensors)
    }

    pub fn forward_with_attention<S, B>(
        &self,
        input: &[S],
        batch_size: B,
    ) -> Result<(Vec<Embeddings>, Attentions), Error>
    where
        S: AsRef<str>,
        B: Into<Option<usize>>,
    {
        let input = input.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
        let batch_size = batch_size.into().unwrap_or_else(|| 64);

        let _guard = tch::no_grad_guard();

        let sorted_pad_input_idx = pad_sort(&input.iter().map(|s| s.len()).collect::<Vec<usize>>());
        let sorted_pad_input = sorted_pad_input_idx
            .iter()
            .map(|i| input[*i])
            .collect::<Vec<&str>>();

        let input_len = sorted_pad_input.len();
        let tokenizer = self.tokenizer.clone();
        let device = self.device;

        let tokenized_batches = (0..input_len)
            .into_par_iter()
            .step_by(batch_size)
            .map(|batch_i| {
                let max_range = std::cmp::min(batch_i + batch_size, input_len);
                let range = batch_i..max_range;

                log::info!(
                    "Batch {}/{}, size {}",
                    (batch_i as f64 / batch_size as f64).ceil() as usize + 1,
                    (input_len as f64 / batch_size as f64).ceil() as usize,
                    max_range - batch_i
                );

                let (tokenized_input, attention) = tokenizer.tokenize(&sorted_pad_input[range]);

                let batch_len = max_range - batch_i;
                let batch_tensor = Tensor::stack(&tokenized_input, 0).to(device);
                let batch_attention = Tensor::stack(&attention, 0).to(device);

                (batch_len, batch_tensor, batch_attention)
            })
            .collect::<Vec<(usize, Tensor, Tensor)>>();

        let nb_layers = self.transformer.nb_layers();
        let nb_heads = self.transformer.nb_heads();

        let mut batch_attention_tensors = Attentions::with_capacity(nb_layers);
        let mut batch_tensors = Vec::<Embeddings>::with_capacity(input_len);

        for (batch_len, batch_tensor, batch_attention) in tokenized_batches.into_iter() {
            let batch_attention_c = batch_attention.shallow_clone();

            let out = self
                .transformer
                .forward(&batch_tensor, &batch_attention, false)?;

            let attention = out.all_attentions;

            let mut features = Features::Token {
                token_embeddings: out.hidden_state,
                attention_mask: batch_attention_c,
            };
            for module in &self.post {
                module.forward(&mut features)?;
            }
            let embedding = match features {
                Features::Sentence { embedding } => embedding,
                Features::Token { .. } => {
                    return Err(Error::Encoding(
                        "pipeline ended without producing a sentence embedding (no Pooling module?)",
                    ));
                }
            };
            batch_tensors.extend(Vec::<Embeddings>::try_from(embedding).unwrap());

            let attention = attention.ok_or_else(|| Error::Encoding("No attention"))?;
            for i in 0..batch_len as i64 {
                let mut layers_att = att::Layers::with_capacity(nb_layers);

                for layer in attention.iter() {
                    let mut heads_att = att::Heads::with_capacity(nb_heads);

                    for head in 0..nb_heads {
                        let att_slice = layer
                            .slice(0, i, i + 1, 1)
                            .slice(1, head as i64, head as i64 + 1, 1)
                            .squeeze();

                        let head_att = att::Attention2D::try_from(att_slice).unwrap();
                        heads_att.push(head_att);
                    }
                    layers_att.push(heads_att);
                }
                batch_attention_tensors.push(layers_att);
            }
        }

        // Sort results
        let sorted_pad_input_idx = pad_sort(&sorted_pad_input_idx);
        let batch_tensors = sorted_pad_input_idx
            .iter()
            .map(|i| mem::replace(&mut batch_tensors[*i], vec![]))
            .collect::<Vec<_>>();
        let batch_attention_tensors = sorted_pad_input_idx
            .iter()
            .map(|i| mem::replace(&mut batch_attention_tensors[*i], vec![]))
            .collect::<Vec<_>>();

        Ok((batch_tensors, batch_attention_tensors))
    }

    pub fn tokenizer(&self) -> Arc<T> {
        self.tokenizer.clone()
    }
}
