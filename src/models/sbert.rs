//! `SentenceTransformer<T>` — the pipeline driver.
//!
//! Replaces the original `SBert<T>` (which was hard-coded to DistilBERT +
//! mandatory Dense). The new driver reads `modules.json` at the model root
//! and composes the post-transformer pipeline from the manifest. The
//! transformer backend is also `modules.json`-driven and dispatches on the
//! config's `model_type` (`bert`, `distilbert`, ...).
//!
//! `SBert<T>` is preserved as a `pub type` alias ([`SBert`](crate::SBert))
//! for backward compatibility — existing call sites
//! (`SBertRT::new(...).forward(...)`) keep working unchanged.

use std::convert::TryFrom;
use std::mem;
use std::path::{Path, PathBuf};
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
            // `Transformer` is the modern (sentence-transformers >= 0.3) class
            // name. Legacy pre-0.3 exports declared one class per
            // architecture — e.g. the UKP v0.2 distiluse export's
            // `modules.json` says `"type":
            // "sentence_transformers.models.DistilBERT"` — so those names are
            // accepted here too. Dispatch to a concrete backend happens later
            // in `transformer::load`, on `config.json`'s `model_type`.
            match entry.short_type().as_str() {
                "transformer" | "bert" | "distilbert" => {
                    if transformer_entry.is_some() {
                        return Err(Error::Encoding(
                            "modules.json declares more than one Transformer module",
                        ));
                    }
                    transformer_entry = Some(entry);
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

        let _transformer_entry = transformer_entry.ok_or_else(|| {
            Error::Encoding("modules.json has no Transformer entry — cannot build pipeline")
        })?;
        let transformer_dir = transformer_dir.ok_or_else(|| {
            // Unreachable: transformer_entry being Some implies transformer_dir is too.
            Error::Encoding("internal: transformer_dir not set")
        })?;

        let mut vs = nn::VarStore::new(device);
        let transformer = transformer_mod::load(&transformer_dir, &vs.root(), device)?;

        // Tokenizer settings are resolved from the checkpoint's config files
        // (see `resolve_tokenizer_settings` for the precedence rules).
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

/// Tokenizer settings resolved from a checkpoint's config files.
#[derive(Debug, PartialEq, Eq)]
struct TokenizerSettings {
    do_lower_case: bool,
    max_seq_length: usize,
}

/// Resolve `do_lower_case` and `max_seq_length` for a checkpoint laid out as
/// `transformer_dir` (the module dir of the Transformer entry) plus the model
/// `root`.
///
/// Resolution order — first match wins:
///
/// * `do_lower_case`:
///   1. `tokenizer_config.json` (transformer dir, then model root) — the
///      authoritative HF tokenizer flag,
///   2. `sentence_bert_config.json` (model root, then transformer dir),
///   3. legacy `sentence_distilbert_config.json` (v0.2-era exports, in the
///      transformer subdir),
///   4. `false` (conservative: cased vocabs like distiluse must not be
///      folded).
///
///   `tokenizer_config.json` must take precedence: e.g. all-MiniLM-L6-v2 has
///   `do_lower_case: true` there but `do_lower_case: false` in
///   `sentence_bert_config.json`. Its vocab is lowercase-only, so the `true`
///   is required — tokenizing cased text case-sensitively maps every
///   capitalized word to `[UNK]`, collapsing distinct inputs
///   ("Tech. root" vs "Browsers. root") to byte-identical embeddings.
///
/// * `max_seq_length`:
///   1. `max_seq_length` of `sentence_bert_config.json` / legacy
///      `sentence_distilbert_config.json` — e.g. 256 for all-MiniLM-L6-v2;
///      truncating at a smaller length diverges from Python
///      sentence-transformers on long inputs,
///   2. `128` — the historical default of this crate (also what the v0.2
///      distiluse export declares).
fn resolve_tokenizer_settings(transformer_dir: &Path, root: &Path) -> TokenizerSettings {
    let read_json = |dir: &Path, file: &str| -> Option<serde_json::Value> {
        let path = dir.join(file);
        let value = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        if value.is_some() {
            log::info!("reading tokenizer settings from {}", path.display());
        }
        value
    };

    let tokenizer_config = [&transformer_dir, &root]
        .iter()
        .find_map(|dir| read_json(dir, "tokenizer_config.json"));

    let sbert_config = [&root, &transformer_dir]
        .iter()
        .find_map(|dir| read_json(dir, "sentence_bert_config.json"))
        .or_else(|| read_json(transformer_dir, "sentence_distilbert_config.json"));

    let do_lower_case = tokenizer_config
        .as_ref()
        .and_then(|v| v.get("do_lower_case").and_then(|b| b.as_bool()))
        .or_else(|| {
            sbert_config
                .as_ref()
                .and_then(|v| v.get("do_lower_case").and_then(|b| b.as_bool()))
        })
        .unwrap_or(false);

    let max_seq_length = sbert_config
        .as_ref()
        .and_then(|v| v.get("max_seq_length").and_then(|n| n.as_u64()))
        .map(|n| n as usize)
        .unwrap_or(128);

    TokenizerSettings {
        do_lower_case,
        max_seq_length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmpdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sbert-settings-test-{}", name));
        fs::remove_dir_all(&dir).ok();
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, file: &str, json: &str) {
        fs::write(dir.join(file), json).unwrap();
    }

    #[test]
    fn defaults_when_no_config_present() {
        let root = tmpdir("defaults-root");
        let tdir = root.join("0_BERT");
        fs::create_dir_all(&tdir).unwrap();
        assert_eq!(
            resolve_tokenizer_settings(&tdir, &root),
            TokenizerSettings {
                do_lower_case: false,
                max_seq_length: 128,
            }
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn tokenizer_config_takes_precedence_over_sentence_bert_config() {
        // The all-MiniLM-L6-v2 situation: tokenizer_config.json says
        // do_lower_case=true, sentence_bert_config.json says false (and
        // carries max_seq_length=256). The HF tokenizer flag must win.
        let root = tmpdir("minilm-root");
        write(&root, "tokenizer_config.json", r#"{"do_lower_case": true}"#);
        write(
            &root,
            "sentence_bert_config.json",
            r#"{"max_seq_length": 256, "do_lower_case": false}"#,
        );
        // Modern layout: `path: ""` → transformer dir IS the model root.
        let s = resolve_tokenizer_settings(&root, &root);
        assert!(s.do_lower_case);
        assert_eq!(s.max_seq_length, 256);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn sentence_bert_config_used_when_no_tokenizer_config() {
        let root = tmpdir("sbertcfg-root");
        write(
            &root,
            "sentence_bert_config.json",
            r#"{"max_seq_length": 256, "do_lower_case": false}"#,
        );
        let s = resolve_tokenizer_settings(&root, &root);
        assert!(!s.do_lower_case);
        assert_eq!(s.max_seq_length, 256);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn legacy_v02_layout_reads_transformer_subdir_configs() {
        // UKP v0.2 distiluse: no root tokenizer_config.json; the transformer
        // subdir carries tokenizer_config.json + the legacy
        // sentence_distilbert_config.json.
        let root = tmpdir("v02-root");
        let tdir = root.join("0_DistilBERT");
        fs::create_dir_all(&tdir).unwrap();
        write(
            &tdir,
            "tokenizer_config.json",
            r#"{"do_lower_case": false, "max_len": 512}"#,
        );
        write(
            &tdir,
            "sentence_distilbert_config.json",
            r#"{"max_seq_length": 96, "do_lower_case": false}"#,
        );
        let s = resolve_tokenizer_settings(&tdir, &root);
        assert!(!s.do_lower_case);
        // 96 is deliberately non-default: 128 would be indistinguishable
        // from the fallback default, so the test would pass even if the
        // legacy file were never read.
        assert_eq!(s.max_seq_length, 96);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn legacy_sentence_distilbert_config_is_last_resort() {
        let root = tmpdir("legacy-root");
        let tdir = root.join("0_X");
        fs::create_dir_all(&tdir).unwrap();
        write(
            &tdir,
            "sentence_distilbert_config.json",
            r#"{"max_seq_length": 96, "do_lower_case": true}"#,
        );
        let s = resolve_tokenizer_settings(&tdir, &root);
        assert!(s.do_lower_case);
        assert_eq!(s.max_seq_length, 96);
        fs::remove_dir_all(&root).ok();
    }
}
