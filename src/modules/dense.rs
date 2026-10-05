//! `sentence_transformers.models.Dense` — linear+tanh projection applied to
//! a sentence embedding. Optional: only models with `2_Dense/` in their
//! `modules.json` instantiate this (e.g. `distiluse-base-multilingual-cased`).
//! Plain BERT/MiniLM checkpoints omit it.

use std::path::Path;

use serde::{de, Deserialize, Deserializer};
use std::str::FromStr;
use strum_macros::EnumString;
use tch::{nn, Device, Tensor};

use crate::modules::{Features, Module};
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
    fn apply(&self, t: &Tensor) -> Tensor {
        match self {
            Activation::Tanh => t.tanh(),
            Activation::Relu => t.relu(),
            Activation::Gelu => t.gelu("none"),
        }
    }
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
    linear: nn::Linear,
    conf: DenseConfig,
}

impl Dense {
    /// `module_dir` is the path to the `2_Dense/` directory itself (not the
    /// model root). Must contain `config.json` and `model.ot`.
    pub fn new<P: AsRef<Path>>(module_dir: P, device: Device) -> Result<Dense, Error> {
        let module_dir = module_dir.as_ref();
        log::info!("Loading Dense from {}", module_dir.display());

        let mut vs_dense = nn::VarStore::new(device);

        let config_file = module_dir.join("config.json");
        let weights_file = module_dir.join("model.ot");

        // Parsed manually (not via `Config::from_file`) so an unsupported
        // `activation_function` surfaces as an `Err` instead of a panic.
        let content = std::fs::read_to_string(&config_file).map_err(|e| {
            log::error!("{} not readable: {}", config_file.display(), e);
            Error::Encoding("Dense config.json not readable")
        })?;
        let conf: DenseConfig = serde_json::from_str(&content).map_err(|e| {
            log::error!("invalid Dense config.json: {}", e);
            Error::Encoding(
                "invalid Dense config.json (unsupported activation_function?)",
            )
        })?;

        let init_conf = nn::LinearConfig {
            ws_init: nn::Init::Const(0.),
            bs_init: Some(nn::Init::Const(0.)),
            bias: conf.bias,
        };

        let linear = nn::linear(&vs_dense.root(), conf.in_features, conf.out_features, init_conf);

        vs_dense.load(weights_file)?;

        Ok(Dense { linear, conf })
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
        // `embedding` is `&mut Tensor` — apply produces a fresh Tensor we
        // write back, then the configured activation is applied.
        let projected = self.conf.activation_function.apply(&embedding.apply(&self.linear));
        *embedding = projected;
        Ok(())
    }
}

/// Split the given string on `.` and try to construct an `Activation` from the last part
fn last_part<'de, D>(deserializer: D) -> Result<Activation, D::Error>
where
    D: Deserializer<'de>,
{
    let activation = String::deserialize(deserializer)?;
    activation
        .split('.')
        .last()
        .map(Activation::from_str)
        .transpose()
        .map_err(de::Error::custom)?
        .ok_or_else(|| format!("Invalid Activation: {}", activation))
        .map_err(de::Error::custom)
}
