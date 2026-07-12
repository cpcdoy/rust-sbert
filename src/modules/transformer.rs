//! Transformer backends for the embedding pipeline.
//!
//! A [`TransformerBackend`] wraps a rust-bert encoder model and exposes a
//! uniform `forward(input_ids, attention_mask, train)` returning
//! [`TransformerOutput`]. The [`load`] factory inspects
//! `<module_dir>/config.json`'s `model_type` field and dispatches to the
//! appropriate backend (`bert`, `distilbert`, ...).

use std::path::{Path, PathBuf};

use rust_bert::bert::{BertConfig, BertEmbeddings, BertModel};
use rust_bert::distilbert::{DistilBertConfig, DistilBertModel};
use rust_bert::Config;
use serde::Deserialize;
use tch::{nn, Device, Tensor};

use crate::Error;

/// Uniform output of any transformer backend.
pub struct TransformerOutput {
    pub hidden_state: Tensor,
    pub all_attentions: Option<Vec<Tensor>>,
}

/// Object-safe backend trait. The pipeline holds a `Box<dyn TransformerBackend>`.
///
/// `Send` only (not `Sync`) — see [`crate::modules::Module`] for the rationale.
pub trait TransformerBackend: Send {
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        train: bool,
    ) -> Result<TransformerOutput, Error>;

    /// Number of encoder layers — used by `forward_with_attention` to size
    /// the attention-output container.
    fn nb_layers(&self) -> usize;
    /// Number of attention heads per layer.
    fn nb_heads(&self) -> usize;
}

// ___________________________________________________________________________
// DistilBERT
//

pub struct DistilBertBackend {
    model: DistilBertModel,
    nb_layers: usize,
    nb_heads: usize,
}

impl DistilBertBackend {
    /// `module_dir` must point at the `0_DistilBERT/` directory containing
    /// `config.json`, `model.ot`, and `vocab.txt`.
    pub fn new(module_dir: &Path, vs: &nn::Path, _device: Device) -> Result<Self, Error> {
        let config_file = module_dir.join("config.json");
        let config = DistilBertConfig::from_file(&config_file);
        let model = DistilBertModel::new(vs, &config);
        Ok(DistilBertBackend {
            model,
            nb_layers: config.n_layers as usize,
            nb_heads: config.n_heads as usize,
        })
    }
}

impl TransformerBackend for DistilBertBackend {
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        train: bool,
    ) -> Result<TransformerOutput, Error> {
        let out = self
            .model
            .forward_t(Some(input_ids), Some(attention_mask), None, train)?;
        Ok(TransformerOutput {
            hidden_state: out.hidden_state,
            all_attentions: out.all_attentions,
        })
    }
    fn nb_layers(&self) -> usize {
        self.nb_layers
    }
    fn nb_heads(&self) -> usize {
        self.nb_heads
    }
}

// ___________________________________________________________________________
// BERT (covers MiniLM-L6-v2, bert-base, etc.)
//

pub struct BertBackend {
    model: BertModel<BertEmbeddings>,
    nb_layers: usize,
    nb_heads: usize,
}

impl BertBackend {
    /// `module_dir` must point at the `0_BERT/` directory containing
    /// `config.json`, `model.ot`, and `vocab.txt`.
    pub fn new(module_dir: &Path, vs: &nn::Path, _device: Device) -> Result<Self, Error> {
        let config_file = module_dir.join("config.json");
        let config = BertConfig::from_file(&config_file);
        let model = BertModel::<BertEmbeddings>::new(vs, &config);
        Ok(BertBackend {
            model,
            nb_layers: config.num_hidden_layers as usize,
            nb_heads: config.num_attention_heads as usize,
        })
    }
}

impl TransformerBackend for BertBackend {
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        train: bool,
    ) -> Result<TransformerOutput, Error> {
        // BertModel::forward_t takes 8 args (input_ids, mask, token_type_ids,
        // position_ids, input_embeds, encoder_hidden_states, encoder_mask, train).
        // None for the optionals lets BertModel derive sensible defaults
        // (token_type_ids=0, position_ids=arange).
        let out = self.model.forward_t(
            Some(input_ids),
            Some(attention_mask),
            None,
            None,
            None,
            None,
            None,
            train,
        )?;
        Ok(TransformerOutput {
            hidden_state: out.hidden_state,
            all_attentions: out.all_attentions,
        })
    }
    fn nb_layers(&self) -> usize {
        self.nb_layers
    }
    fn nb_heads(&self) -> usize {
        self.nb_heads
    }
}

// ___________________________________________________________________________
// Factory: dispatch on config.json's `model_type` field
//

/// Minimal slice of `config.json` — enough to read `model_type` without
/// pulling every model-specific schema. Add fields as needed.
#[derive(Debug, Deserialize)]
struct ModelTypeProbe {
    /// Lowercased HF model_type tag: `"bert"`, `"distilbert"`, `"roberta"`, ...
    /// Some older checkpoints (e.g. the original sentence-transformers
    /// distiluse export) omit it; we fall back to `"distilbert"` since the
    /// only known-omitted case is exactly that one.
    #[serde(default)]
    model_type: Option<String>,
}

impl Config for ModelTypeProbe {}

/// Load the transformer backend declared by `<module_dir>/config.json`.
///
/// `module_dir` should already be resolved by [`crate::modules::manifest::resolve_module_dir`]
/// — this function does no path fallback of its own. Dispatches on the
/// config's `model_type` field (`bert`, `distilbert`).
///
/// Returns the backend plus the tokenizer directory (the same module dir —
/// rust-bert reads `vocab.txt` from there).
pub fn load(
    module_dir: &Path,
    vs: &nn::Path,
    device: Device,
) -> Result<(Box<dyn TransformerBackend>, PathBuf), Error> {
    let config_file = module_dir.join("config.json");

    let probe = ModelTypeProbe::from_file(&config_file);
    let model_type = probe
        .model_type
        .as_deref()
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "distilbert".to_string());

    log::info!(
        "Loading transformer backend (model_type={}) from {}",
        model_type,
        module_dir.display()
    );

    let backend: Box<dyn TransformerBackend> = match model_type.as_str() {
        "distilbert" => Box::new(DistilBertBackend::new(module_dir, vs, device)?),
        "bert" => Box::new(BertBackend::new(module_dir, vs, device)?),
        other => {
            log::error!(
                "unsupported transformer model_type '{}' (only 'bert' and 'distilbert' are implemented)",
                other
            );
            return Err(Error::Encoding(
                "unsupported transformer model_type in config.json (only 'bert' and 'distilbert' are implemented)",
            ));
        }
    };

    Ok((backend, module_dir.to_path_buf()))
}
