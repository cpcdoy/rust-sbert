//! `SentenceTransformer<T>` — the pipeline driver, libtorch backend
//! (cargo feature `torch`, the default).
//!
//! Reads `modules.json` at the model root and composes the pipeline from
//! the manifest: a rust-bert TorchScript transformer (weights from
//! `model.ot` VarStore archives) followed by the declared post-transformer
//! modules (`Pooling`, optional `Dense`, optional `Normalize`), all tch.
//!
//! `SBert<T>` is preserved as a `pub type` alias ([`crate::SBert`]) for
//! backward compatibility — existing call sites
//! (`SBertRT::new(...).forward(...)`) keep working unchanged.

use std::convert::TryFrom;
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;
use tch::{nn, Tensor};

use crate::models::pad_sort;
use crate::models::settings::resolve_tokenizer_settings;
use crate::modules::{
    self, manifest, transformer as transformer_mod, Features, Module, TransformerBackend,
};
use crate::tokenizers::Tokenizer;
use crate::{att, Attentions, Device, Embeddings, Error};

pub struct SentenceTransformer<T> {
    transformer: Box<dyn TransformerBackend>,
    /// Ordered post-transformer pipeline (Pooling, optional Dense, optional
    /// Normalize, ...) built from `modules.json`. The transformer is NOT in
    /// this list — it has a different I/O shape and is held separately.
    post: Vec<Box<dyn Module>>,
    tokenizer: Arc<T>,
    device: tch::Device,
}

/// Stack per-row token ids / masks (as produced by the tokenizers, which pad
/// within a batch) into `[batch, seq]` tensors on `device`.
fn stack_ids(ids: &[Vec<i64>], device: tch::Device) -> Tensor {
    Tensor::stack(
        &ids.iter()
            .map(|r| Tensor::from_slice(r))
            .collect::<Vec<_>>(),
        0,
    )
    .to(device)
}

impl<T> SentenceTransformer<T>
where
    T: Tokenizer + Send + Sync,
{
    /// Load a sentence-transformers checkpoint from `root`.
    ///
    /// `root` must contain `modules.json`; the transformer module dir must
    /// contain a rust-bert VarStore archive (`model.ot`), `config.json` and
    /// `vocab.txt` — see `utils/prepare_models.py --backend torch`.
    ///
    /// The subdirectory each module lives in is resolved by
    /// [`manifest::resolve_module_dir`], which tolerates three real-world
    /// layouts:
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
        let device: tch::Device = device.unwrap_or_else(Device::cuda_if_available).into();
        log::info!("Using device {:?}", device);

        let entries = manifest::parse(&root)?;

        let mut transformer_dir: Option<PathBuf> = None;
        let mut post: Vec<Box<dyn Module>> = Vec::new();

        // Walk the manifest in declared order; the first Transformer entry
        // becomes the backend, everything else gets pushed into the
        // post-pipeline. Module directories are resolved with fallback logic
        // because the manifest's `path` field is unreliable in the wild.
        for entry in &entries {
            let module_dir = manifest::resolve_module_dir(&root, entry);
            // `Transformer` is the modern (sentence-transformers >= 0.3) class
            // name. Legacy pre-0.3 exports declared one class per
            // architecture — e.g. the UKP v0.2 distiluse export's
            // `modules.json` says `"type":
            // "sentence_transformers.models.DistilBERT"` — so those names are
            // accepted here too. Dispatch to a concrete backend happens later
            // in `transformer::load`, on `config.json`'s `model_type`.
            match entry.short_type().as_str() {
                "transformer" | "bert" | "distilbert" => {
                    if transformer_dir.is_some() {
                        return Err(Error::Encoding(
                            "modules.json declares more than one Transformer module",
                        ));
                    }
                    transformer_dir = Some(module_dir);
                }
                "pooling" => {
                    post.push(Box::new(modules::Pooling::new(&module_dir)?));
                }
                "dense" => {
                    post.push(Box::new(modules::Dense::new(&module_dir, device)?));
                }
                "normalize" => {
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

        let transformer_dir = transformer_dir.ok_or_else(|| {
            Error::Encoding("modules.json has no Transformer entry — cannot build pipeline")
        })?;

        let mut vs = nn::VarStore::new(device);
        let transformer = transformer_mod::load(&transformer_dir, &vs.root(), device)?;

        // Tokenizer settings are resolved from the checkpoint's config files
        // (see `settings::resolve_tokenizer_settings` for the precedence
        // rules).
        let settings = resolve_tokenizer_settings(&transformer_dir, &root);
        log::info!(
            "tokenizer settings: do_lower_case = {}, max_seq_length = {}",
            settings.do_lower_case,
            settings.max_seq_length
        );

        let tokenizer = Arc::new(T::new(
            &transformer_dir.join("vocab.txt"),
            settings.do_lower_case,
            settings.max_seq_length,
        )?);

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

                let batch_tensor = stack_ids(&tokenized_input, device);
                let batch_attention = stack_ids(&attention, device);

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

    /// Embeddings + per-layer/head attention maps. TorchScript-only: the
    /// ONNX backend runs bare-backbone exports without attention outputs.
    /// Requires the transformer `config.json` to set
    /// `"output_attentions": true`, otherwise rust-bert returns
    /// `all_attentions: None` and this errors with `Encoding("No attention")`.
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
                let batch_tensor = stack_ids(&tokenized_input, device);
                let batch_attention = stack_ids(&attention, device);

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
            .map(|&i| mem::replace(&mut batch_tensors[i], vec![]))
            .collect::<Vec<_>>();
        let batch_attention_tensors = sorted_pad_input_idx
            .into_iter()
            .map(|i| mem::replace(&mut batch_attention_tensors[i], vec![]))
            .collect::<Vec<_>>();

        Ok((batch_tensors, batch_attention_tensors))
    }

    pub fn tokenizer(&self) -> Arc<T> {
        self.tokenizer.clone()
    }
}
