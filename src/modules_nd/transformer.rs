//! Transformer backend for the embedding pipeline — ONNX Runtime.
//!
//! Wraps rust-bert's [`ONNXEncoder`] (ndarray-shaped, tch-free in the Luxbit
//! fork) over a bare-backbone ONNX export: inputs `input_ids` /
//! `attention_mask`, output `last_hidden_state`. Graphs that additionally
//! declare `token_type_ids` / `position_ids` get zeros supplied by the
//! encoder wrapper; graphs with other required inputs error loudly.
//!
//! Requires an onnxruntime shared library at runtime, located via
//! `ORT_DYLIB_PATH` (manual setup — see the crate README).

use std::path::{Path, PathBuf};

use ndarray::{Array2, Array3, ArrayD, Ix3};
use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
use rust_bert::pipelines::onnx::ONNXEncoder;
use rust_bert::{Config, Device};
use serde::Deserialize;

use crate::Error;

/// Output of the transformer backend: per-token hidden states.
///
/// No attention outputs: bare-backbone optimum exports don't carry them
/// (see `utils/prepare_onnx.py`), so there is no `forward_with_attention`
/// in the ort-only pipeline.
pub struct TransformerOutput {
    pub hidden_state: Array3<f32>,
}

pub trait TransformerBackend: Send {
    fn forward(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> Result<TransformerOutput, Error>;
}

/// Transformer backend backed by an ONNX Runtime session.
pub struct OnnxBackend {
    encoder: ONNXEncoder,
}

impl OnnxBackend {
    /// `module_dir` must contain `config.json` (for the `model_type`
    /// sanity check) and, unless `onnx_file` is given, `model.onnx`.
    ///
    /// `device` selects the execution provider: `Cuda(i)` → CUDA EP
    /// (requires rust-bert's `cuda` feature downstream) then CPU fallback;
    /// `Cpu` → CPU EP only.
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

        // model_type is validated to prove the checkpoint is an architecture
        // we understand (bert / distilbert backbones exported as encoders).
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
        if model_type != "bert" && model_type != "distilbert" {
            log::error!(
                "unsupported transformer model_type '{}' (only 'bert' and 'distilbert' are implemented)",
                model_type
            );
            return Err(Error::Encoding(
                "unsupported transformer model_type in config.json (only 'bert' and 'distilbert' are implemented)",
            ));
        }

        let onnx_config = ONNXEnvironmentConfig::from_device(device);
        let encoder = ONNXEncoder::new(model_file.clone(), &onnx_config)?;
        log::info!(
            "Loaded ONNX transformer backend (model_type={model_type}) from {}",
            model_file.display()
        );

        Ok(OnnxBackend { encoder })
    }
}

impl TransformerBackend for OnnxBackend {
    fn forward(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> Result<TransformerOutput, Error> {
        let ids: ArrayD<i64> = input_ids.to_owned().into_dyn();
        let mask: ArrayD<i64> = attention_mask.to_owned().into_dyn();
        let out = self
            .encoder
            .forward(Some(&ids), Some(&mask), None, None, None)?;
        let hidden = out.last_hidden_state.ok_or_else(|| {
            log::error!(
                "ONNX export has no `last_hidden_state` output (export with optimum defaults)"
            );
            Error::Encoding(
                "ONNX export has no `last_hidden_state` output (export with optimum defaults)",
            )
        })?;
        let hidden_state = hidden
            .into_dimensionality::<Ix3>()
            .map_err(|_| Error::Encoding("`last_hidden_state` is not [batch, seq, hidden]"))?;
        Ok(TransformerOutput { hidden_state })
    }
}

/// Minimal slice of `config.json` — enough to read `model_type` without
/// pulling every model-specific schema.
#[derive(Debug, Deserialize)]
struct ModelTypeProbe {
    /// Lowercased HF model_type tag: `"bert"`, `"distilbert"`, `"roberta"`, ...
    /// A missing field is a hard error: guessing a default would surface only
    /// later as an unrelated session-input error.
    #[serde(default)]
    model_type: Option<String>,
}

impl Config for ModelTypeProbe {}
