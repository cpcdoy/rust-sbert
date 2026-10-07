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

#[cfg(feature = "onnx")]
use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
#[cfg(feature = "onnx")]
use rust_bert::pipelines::onnx::ONNXEncoder;

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
// Source selection
//

/// Where the transformer weights come from.
#[derive(Clone, Debug)]
pub enum TransformerSource {
    /// TorchScript `model.ot` loaded into the driver's `VarStore` (default).
    TorchScript,
    /// ONNX graph. `None` resolves to `<transformer_dir>/model.onnx`.
    ///
    /// The user `Device` selects only the ONNX Runtime execution provider
    /// (`Cuda(i)` → CUDA EP, everything else → CPU EP; ORT has no Metal
    /// provider). All tch-side tensors stay on CPU — see
    /// [`crate::SentenceTransformer::new_with_source`].
    #[cfg(feature = "onnx")]
    Onnx(Option<PathBuf>),
}

/// A transformer backend plus the driver-side weight bookkeeping it implies.
pub struct LoadedTransformer {
    pub backend: Box<dyn TransformerBackend>,
    /// `Some(path)` for TorchScript: the driver must `vs.load(path)` after
    /// the backend constructor wired up its VarStore paths. `None` for ONNX
    /// — the weights live inside the graph.
    pub torchscript_weights: Option<PathBuf>,
}

// ___________________________________________________________________________
// ONNX (optional; reuses rust-bert's ONNXEncoder for tch⇄ndarray marshalling
// and session I/O name mapping)
//

/// Transformer backend backed by an ONNX Runtime session.
///
/// Wraps [`ONNXEncoder`]; inputs/outputs are `tch::Tensor`. Requires the
/// `onnx` cargo feature and, at runtime, an onnxruntime library located via
/// `ORT_DYLIB_PATH` (or a `libonnxruntime.dylib` next to the executable /
/// on the system loader paths).
#[cfg(feature = "onnx")]
pub struct OnnxBackend {
    encoder: ONNXEncoder,
    nb_layers: usize,
    nb_heads: usize,
}

#[cfg(feature = "onnx")]
impl OnnxBackend {
    /// `module_dir` must contain `config.json` (for `model_type` validation
    /// and the layer/head counts) and, unless `onnx_file` is given,
    /// `model.onnx`.
    pub fn new(
        module_dir: &Path,
        onnx_file: Option<PathBuf>,
        device: Device,
    ) -> Result<Self, Error> {
        let model_file = onnx_file.unwrap_or_else(|| module_dir.join("model.onnx"));
        if !model_file.exists() {
            log::error!("ONNX model file not found: {}", model_file.display());
            return Err(Error::Encoding(
                "ONNX model file not found (expected `model.onnx` in the transformer module dir, or an explicit path)",
            ));
        }
        if std::env::var("ORT_DYLIB_PATH").map_or(true, |v| v.is_empty()) {
            log::warn!(
                "ORT_DYLIB_PATH is not set; ort will try a bare `libonnxruntime.dylib` \
                 (executable dir / system paths). Set ORT_DYLIB_PATH for an explicit runtime."
            );
        }

        let probe = ModelTypeProbe::from_file(&module_dir.join("config.json"));
        // Only consumed by `forward_with_attention`, which ONNX does not
        // support anyway — absence is a warning, not an error.
        let nb_layers = probe.num_hidden_layers.map(|n| n as usize).unwrap_or(0);
        let nb_heads = probe.num_attention_heads.map(|n| n as usize).unwrap_or(0);
        if nb_layers == 0 || nb_heads == 0 {
            log::warn!(
                "config.json carries no layer/head counts; reporting 0 \
                 (only affects forward_with_attention, unsupported on ONNX anyway)"
            );
        }

        // Environment-level execution providers do reach sessions in ort 1.15
        // (session creation chains `env.execution_providers`). Cuda(i) →
        // CUDA EP, everything else → CPU EP.
        let onnx_config = ONNXEnvironmentConfig::from_device(device);
        let environment = onnx_config.get_environment()?;
        let encoder = ONNXEncoder::new(model_file, &environment, &onnx_config)?;

        Ok(OnnxBackend {
            encoder,
            nb_layers,
            nb_heads,
        })
    }
}

#[cfg(feature = "onnx")]
impl TransformerBackend for OnnxBackend {
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _train: bool,
    ) -> Result<TransformerOutput, Error> {
        // Always supply token_type_ids=zeros: ONNXEncoder only pulls the
        // input names the session declares, so an extra entry is ignored
        // (DistilBERT exports don't declare it; BERT exports do and need it).
        // Graphs requiring position_ids/input_embeds error — documented limit.
        let token_type_ids = input_ids.zeros_like();
        let out = self.encoder.forward(
            Some(input_ids),
            Some(attention_mask),
            Some(&token_type_ids),
            None,
            None,
        )?;
        Ok(TransformerOutput {
            hidden_state: out.last_hidden_state.ok_or_else(|| {
                log::error!(
                    "ONNX export has no `last_hidden_state` output (export with optimum defaults)"
                );
                Error::Encoding(
                    "ONNX export has no `last_hidden_state` output (export with optimum defaults)",
                )
            })?,
            all_attentions: out.attentions,
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

/// Minimal slice of `config.json` — enough to read `model_type` and the
/// layer/head counts without pulling every model-specific schema.
#[derive(Debug, Deserialize)]
struct ModelTypeProbe {
    /// Lowercased HF model_type tag: `"bert"`, `"distilbert"`, `"roberta"`, ...
    /// Every known sentence-transformers export carries it (including the
    /// v0.2 distiluse export, whose config says `"model_type": "distilbert"`).
    /// A missing field is a hard error: guessing a default would surface only
    /// later as an unrelated `vs.load` shape error.
    #[serde(default)]
    model_type: Option<String>,
    /// Encoder depth under either naming convention: BERT configs say
    /// `num_hidden_layers`, DistilBERT configs say `n_layers`.
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    #[serde(default, alias = "n_layers")]
    num_hidden_layers: Option<i64>,
    /// Attention heads per layer: `num_attention_heads` (BERT) or `n_heads`
    /// (DistilBERT).
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    #[serde(default, alias = "n_heads")]
    num_attention_heads: Option<i64>,
}

impl Config for ModelTypeProbe {}

/// Load the transformer backend declared by `<module_dir>/config.json`.
///
/// `module_dir` should already be resolved by [`crate::modules::manifest::resolve_module_dir`]
/// — this function does no path fallback of its own. Dispatches on the
/// config's `model_type` field (`bert`, `distilbert`). Missing `model_type`
/// is an error (the offending path is logged).
///
/// The `TransformerSource::Onnx` source builds an `OnnxBackend` (requires the
/// `onnx` feature) and returns no TorchScript weights; otherwise the
/// backend's `model.ot` path is returned for the caller to `vs.load`.
pub fn load(
    module_dir: &Path,
    vs: &nn::Path,
    device: Device,
    source: TransformerSource,
) -> Result<LoadedTransformer, Error> {
    let config_file = module_dir.join("config.json");

    let probe = ModelTypeProbe::from_file(&config_file);
    let model_type = probe
        .model_type
        .as_deref()
        .map(|s| s.to_lowercase())
        .ok_or_else(|| {
            log::error!(
                "config.json at {} has no model_type field",
                config_file.display()
            );
            Error::Encoding("transformer config.json has no model_type field")
        })?;

    log::info!(
        "Loading transformer backend (model_type={}, source={:?}) from {}",
        model_type,
        source,
        module_dir.display()
    );

    #[cfg(feature = "onnx")]
    if let TransformerSource::Onnx(file) = source {
        // model_type is still validated for ONNX: it proves the checkpoint
        // is an architecture we understand and feeds the layer/head probe.
        if model_type != "bert" && model_type != "distilbert" {
            log::error!(
                "unsupported transformer model_type '{}' (only 'bert' and 'distilbert' are implemented)",
                model_type
            );
            return Err(Error::Encoding(
                "unsupported transformer model_type in config.json (only 'bert' and 'distilbert' are implemented)",
            ));
        }
        let backend = Box::new(OnnxBackend::new(module_dir, file, device)?);
        return Ok(LoadedTransformer {
            backend,
            torchscript_weights: None,
        });
    }
    #[cfg(not(feature = "onnx"))]
    let _ = source;

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

    Ok(LoadedTransformer {
        backend,
        torchscript_weights: Some(module_dir.join("model.ot")),
    })
}
