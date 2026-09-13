//! `sentence_transformers.models.Dense` — linear+tanh projection applied to
//! a sentence embedding. Optional: only models with `2_Dense/` in their
//! `modules.json` instantiate this (e.g. `distiluse-base-multilingual-cased`).
//! Plain BERT/MiniLM checkpoints omit it.

use std::path::Path;

use rust_bert::Config;
use serde::{de, Deserialize, Deserializer};
use std::str::FromStr;
use strum_macros::EnumString;
use tch::{nn, Device};

use crate::modules::{Features, Module};
use crate::Error;

#[derive(Debug, Deserialize, EnumString)]
pub enum Activation {
    Tanh,
}

#[derive(Debug, Deserialize)]
pub struct DenseConfig {
    pub in_features: i64,
    pub out_features: i64,
    #[serde(deserialize_with = "last_part")]
    pub activation_function: Activation,
}

impl Config for DenseConfig {}

pub struct Dense {
    linear: nn::Linear,
    _conf: DenseConfig,
}

impl Dense {
    /// `module_dir` is the path to the `2_Dense/` directory itself (not the
    /// model root). Must contain `config.json` and `model.ot`.
    pub fn new<P: AsRef<Path>>(module_dir: P, device: Device) -> Result<Dense, Error> {
        let module_dir = module_dir.as_ref();
        log::info!("Loading Dense from {}", module_dir.display());

        let mut vs_dense = nn::VarStore::new(device);

        let init_conf = nn::LinearConfig {
            ws_init: nn::Init::Const(0.),
            bs_init: Some(nn::Init::Const(0.)),
            bias: true,
        };

        let config_file = module_dir.join("config.json");
        let weights_file = module_dir.join("model.ot");

        let conf = DenseConfig::from_file(&config_file);
        let linear = nn::linear(&vs_dense.root(), conf.in_features, conf.out_features, init_conf);

        vs_dense.load(weights_file)?;

        Ok(Dense {
            linear,
            _conf: conf,
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
        // `embedding` is `&mut Tensor` — apply produces a fresh Tensor we
        // write back. Tanh activation is hard-coded (the only variant
        // `Activation` currently models).
        let projected = embedding.apply(&self.linear).tanh();
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
