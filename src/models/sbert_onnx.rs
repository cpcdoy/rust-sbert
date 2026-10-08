//! `SentenceTransformer<T>` — the pipeline driver, ONNX Runtime backend
//! (cargo feature `onnx`).
//!
//! Reads `modules.json` at the model root and composes the pipeline from
//! the manifest: an [`OnnxBackend`] transformer over a bare-backbone
//! `model.onnx` export, followed by the declared post-transformer modules
//! (Pooling, optional Dense, optional Normalize) running in pure ndarray.
//!
//! `SBert<T>` is preserved as a `pub type` alias ([`crate::SBert`]) for
//! backward compatibility — existing call sites
//! (`SBertRT::new(...).forward(...)`) keep working unchanged.

use std::mem;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use rayon::prelude::*;

use crate::models::pad_sort;
use crate::models::settings::resolve_tokenizer_settings;
use crate::modules::manifest;
use crate::modules_nd::{self, Features, Module, OnnxBackend, TransformerBackend as _};
use crate::tokenizers::Tokenizer;
use crate::{Device, Embeddings, Error};

pub struct SentenceTransformer<T> {
    transformer: OnnxBackend,
    /// Ordered post-transformer pipeline (Pooling, optional Dense, optional
    /// Normalize, ...) built from `modules.json`. The transformer is NOT in
    /// this list — it has a different I/O shape and is held separately.
    post: Vec<Box<dyn Module>>,
    tokenizer: Arc<T>,
}

impl<T> SentenceTransformer<T>
where
    T: Tokenizer + Send + Sync,
{
    /// Load a sentence-transformers checkpoint from `root`.
    ///
    /// `root` must contain `modules.json`, and the transformer module dir
    /// must contain a bare-backbone `model.onnx` (see
    /// `utils/prepare_models.py --backend onnx`) plus `vocab.txt` and
    /// `config.json`.
    ///
    /// The subdirectory each module lives in is resolved by
    /// [`manifest::resolve_module_dir`], which tolerates three real-world
    /// layouts (modern root layout, truthful manifest paths, legacy
    /// `<idx>_*` fallback).
    ///
    /// `device` selects the ONNX Runtime execution provider (`Cuda(i)` or
    /// `Cpu`; defaults to [`Device::cuda_if_available`], which is CPU unless
    /// a CUDA-enabled build is available). Requires an onnxruntime library
    /// at runtime (`ORT_DYLIB_PATH`).
    pub fn new<P>(root: P, device: Option<Device>) -> Result<Self, Error>
    where
        P: Into<PathBuf>,
    {
        let root = root.into();
        let device = device.unwrap_or_else(Device::cuda_if_available);
        log::info!("Using device {:?}", device);

        let entries = manifest::parse(&root)?;

        let mut transformer_dir: Option<PathBuf> = None;
        let mut post: Vec<Box<dyn Module>> = Vec::new();

        // Walk the manifest in declared order; the first Transformer entry
        // becomes the backend, everything else gets pushed into the
        // post-pipeline.
        for entry in &entries {
            let module_dir = manifest::resolve_module_dir(&root, entry);
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
                    post.push(Box::new(modules_nd::Pooling::new(&module_dir)?));
                }
                "dense" => {
                    post.push(Box::new(modules_nd::Dense::new(&module_dir)?));
                }
                "normalize" => {
                    post.push(Box::new(modules_nd::Normalize::new()));
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

        let transformer_dir = transformer_dir.ok_or({
            Error::Encoding("modules.json has no Transformer entry — cannot build pipeline")
        })?;

        let transformer = OnnxBackend::new(&transformer_dir, None, device)?;

        let settings = resolve_tokenizer_settings(&transformer_dir, &root);
        log::info!(
            "tokenizer settings: do_lower_case = {}, max_seq_length = {}",
            settings.do_lower_case,
            settings.max_seq_length
        );

        let tokenizer = Arc::new(T::new(
            transformer_dir.join("vocab.txt"),
            settings.do_lower_case,
            settings.max_seq_length,
        )?);

        Ok(SentenceTransformer {
            transformer,
            post,
            tokenizer,
        })
    }

    /// Embed `input`, processing at most `batch_size` sentences at a time
    /// (`None`/default 64). Sentences are sorted by length for efficient
    /// padding and results are returned in input order; row `i` of the
    /// output equals the embedding of `input[i]` encoded on its own.
    pub fn forward<S, B>(&self, input: &[S], batch_size: B) -> Result<Vec<Embeddings>, Error>
    where
        S: AsRef<str>,
        B: Into<Option<usize>>,
    {
        let input = input.iter().map(AsRef::as_ref).collect::<Vec<&str>>();
        let batch_size = batch_size.into().unwrap_or(64);

        let sorted_pad_input_idx = pad_sort(&input.iter().map(|s| s.len()).collect::<Vec<usize>>());
        let sorted_pad_input = sorted_pad_input_idx
            .iter()
            .map(|i| input[*i])
            .collect::<Vec<&str>>();

        let input_len = sorted_pad_input.len();
        let tokenizer = self.tokenizer.clone();

        // Tokenize (rayon-parallel over batches)
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

                tokenizer.tokenize(&sorted_pad_input[range])
            })
            .collect::<Vec<_>>();

        // Embed + run pipeline
        let mut batch_tensors = Vec::<Embeddings>::with_capacity(input_len);

        for (ids, mask) in tokenized_batches.into_iter() {
            let (batch, seq) = (ids.len(), ids.first().map(|r| r.len()).unwrap_or(0));
            let batch_tensor = Array2::from_shape_vec((batch, seq), ids.concat())
                .map_err(|_| Error::Encoding("token id rows of unequal length"))?;
            let batch_attention = Array2::from_shape_vec((batch, seq), mask.concat())
                .map_err(|_| Error::Encoding("attention mask rows of unequal length"))?;

            let out = self.transformer.forward(&batch_tensor, &batch_attention)?;

            let mut features = Features::Token {
                token_embeddings: out.hidden_state,
                attention_mask: batch_attention,
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
            for row in embedding.outer_iter() {
                batch_tensors.push(row.to_vec());
            }
        }

        // Sort results
        let sorted_pad_input_idx = pad_sort(&sorted_pad_input_idx);
        let batch_tensors = sorted_pad_input_idx
            .into_iter()
            .map(|i| mem::take(&mut batch_tensors[i]))
            .collect::<Vec<_>>();

        Ok(batch_tensors)
    }

    pub fn tokenizer(&self) -> Arc<T> {
        self.tokenizer.clone()
    }
}
