//! `sentence_transformers.models.Dense` — linear+tanh projection applied to
//! a sentence embedding. Optional: only models with `2_Dense/` in their
//! `modules.json` instantiate this (e.g. `distiluse-base-multilingual-cased`).
//! Plain BERT/MiniLM checkpoints omit it.
//!
//! The ONNX graph only covers the transformer stage, so the Dense weights
//! are read directly from `2_Dense/weights.safetensors` (`linear.weight`
//! `[out, in]`, `linear.bias` `[out]` — see `utils/prepare_models.py`).

use std::path::Path;

use ndarray::{Array1, Array2};
use serde::{de, Deserialize, Deserializer};
use std::str::FromStr;
use strum_macros::EnumString;

use crate::modules_nd::safetensors::SafetensorsFile;
use crate::modules_nd::{Features, Module};
use crate::Error;

/// Activation applied after the linear projection. `config.json` names it as
/// the fully-qualified Python class (e.g. `torch.nn.Tanh`); only the last
/// segment is compared, case-insensitively.
#[derive(Debug, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Activation {
    Tanh,
    Relu,
    Gelu,
}

impl Activation {
    fn apply(&self, t: Array2<f32>) -> Array2<f32> {
        match self {
            Activation::Tanh => t.mapv(f32::tanh),
            Activation::Relu => t.mapv(|v| v.max(0.0)),
            Activation::Gelu => t.mapv(gelu_erf),
        }
    }
}

/// Exact (erf-based) GELU, `torch.nn.gelu(approximate="none")`'s formula.
/// Rust std has no `erf`, so the Abramowitz & Stegun 7.1.26 rational
/// approximation is used (|absolute error| < 1.5e-7 — below f32 rounding
/// for embedding workloads).
fn gelu_erf(x: f32) -> f32 {
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.3275911;

    let erf = |x: f32| {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + P * x);
        let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
        sign * y
    };

    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

#[derive(Debug, Deserialize)]
pub struct DenseConfig {
    pub in_features: i64,
    pub out_features: i64,
    /// Defaults to `true` — older exports (e.g. the UKP v0.2 distiluse
    /// `2_Dense/config.json`) omit the field but do carry a bias.
    #[serde(default = "default_bias")]
    pub bias: bool,
    #[serde(deserialize_with = "last_part")]
    pub activation_function: Activation,
}

fn default_bias() -> bool {
    true
}

pub struct Dense {
    /// Transposed kernel `[in_features, out_features]` so the forward pass
    /// is a single `dot` on contiguous arrays.
    w_t: Array2<f32>,
    b: Array1<f32>,
    conf: DenseConfig,
}

impl Dense {
    /// `module_dir` is the path to the `2_Dense/` directory itself (not the
    /// model root). Must contain `config.json` and `weights.safetensors`.
    pub fn new<P: AsRef<Path>>(module_dir: P) -> Result<Dense, Error> {
        let module_dir = module_dir.as_ref();
        log::info!("Loading Dense from {}", module_dir.display());

        let config_file = module_dir.join("config.json");
        let weights_file = module_dir.join("weights.safetensors");

        // Parsed manually (not via `Config::from_file`) so an unsupported
        // `activation_function` surfaces as an `Err` instead of a panic.
        let content = std::fs::read_to_string(&config_file).map_err(|e| {
            log::error!("{} not readable: {}", config_file.display(), e);
            Error::Encoding("Dense config.json not readable")
        })?;
        let conf: DenseConfig = serde_json::from_str(&content).map_err(|e| {
            log::error!("invalid Dense config.json: {}", e);
            Error::Encoding("invalid Dense config.json (unsupported activation_function?)")
        })?;

        let st = SafetensorsFile::read(&weights_file)?;
        let w = st.tensor_f32("linear.weight")?;
        let b = st.tensor_f32("linear.bias")?;

        let w = w
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| Error::Encoding("linear.weight is not 2-D"))?;
        let b = b
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|_| Error::Encoding("linear.bias is not 1-D"))?;

        if w.shape() != [conf.out_features as usize, conf.in_features as usize] {
            return Err(Error::Encoding(
                "linear.weight shape does not match Dense config.json in/out features",
            ));
        }
        if b.shape() != [conf.out_features as usize] {
            return Err(Error::Encoding(
                "linear.bias shape does not match Dense config.json out_features",
            ));
        }

        Ok(Dense {
            w_t: w.t().to_owned(),
            b,
            conf,
        })
    }
}

impl Module for Dense {
    fn forward(&self, features: &mut Features) -> Result<(), Error> {
        let embedding = match features {
            Features::Sentence { embedding } => embedding,
            _ => {
                return Err(Error::Encoding(
                    "Dense received non-sentence features (Pooling must run first)",
                ));
            }
        };
        // [batch, in] x [in, out] + [out] -> [batch, out], then the
        // configured activation.
        let projected = embedding.dot(&self.w_t) + &self.b;
        *embedding = self.conf.activation_function.apply(projected);
        Ok(())
    }
}

/// Split the given string on `.` and try to construct an `Activation` from the last part
fn last_part<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Activation, D::Error> {
    let activation = String::deserialize(deserializer)?;
    activation
        .split('.')
        .next_back()
        .map(Activation::from_str)
        .transpose()
        .map_err(de::Error::custom)?
        .ok_or_else(|| format!("Invalid Activation: {}", activation))
        .map_err(de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn write_checkpoint(dir: &Path, w: &[[f32; 2]], b: &[f32]) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"in_features": 2, "out_features": 2, "bias": true,
                "activation_function": "torch.nn.Tanh"}"#,
        )
        .unwrap();
        // hand-built safetensors: header + 2x2 weight + 2 bias, f32 LE
        let w_flat: Vec<f32> = w.iter().flat_map(|r| r.iter().copied()).collect();
        let n = (w_flat.len() + b.len()) * 4;
        let header = format!(
            r#"{{"linear.weight":{{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}},"linear.bias":{{"dtype":"F32","shape":[2],"data_offsets":[16,{n}]}}}}"#
        );
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for v in w_flat.iter().chain(b.iter()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("weights.safetensors"), bytes).unwrap();
    }

    /// identity kernel + zero bias under tanh maps inputs to tanh(input).
    #[test]
    fn dense_applies_linear_then_activation() {
        let dir = std::env::temp_dir().join("sbert-dense-test");
        std::fs::remove_dir_all(&dir).ok();
        write_checkpoint(&dir, &[[1.0, 0.0], [0.0, 1.0]], &[0.0, 0.0]);

        let dense = Dense::new(&dir).unwrap();
        let mut features = Features::Sentence {
            embedding: Array2::from_shape_vec((2, 2), vec![0.5, 1.0, -0.5, 2.0]).unwrap(),
        };
        dense.forward(&mut features).unwrap();
        match features {
            Features::Sentence { embedding } => {
                let got = embedding
                    .outer_iter()
                    .map(|r| r.to_vec())
                    .collect::<Vec<_>>();
                for (row, input) in got.iter().zip([[0.5f32, 1.0], [-0.5, 2.0]]) {
                    assert!((row[0] - input[0].tanh()).abs() < 1e-6);
                    assert!((row[1] - input[1].tanh()).abs() < 1e-6);
                }
            }
            _ => panic!("expected sentence features"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gelu_matches_reference_points() {
        // reference: exact erf gelu at 0, 1, -1, 2
        let points = [
            (0.0f32, 0.0f32),
            (1.0, 0.8413447),
            (-1.0, -0.1586553),
            (2.0, 1.9544997),
        ];
        for (x, want) in points {
            assert!(
                (gelu_erf(x) - want).abs() < 1e-5,
                "gelu({x}) = {}",
                gelu_erf(x)
            );
        }
    }
}
