use std::path::PathBuf;

use rust_bert::Config;
use serde::{de, Deserialize, Deserializer};
use std::str::FromStr;
use strum_macros::EnumString;
use tch::{nn, Device, Tensor};

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
    pub fn new<P: Into<PathBuf>>(root: P) -> Result<Dense, Error> {
        let dense_dir = root.into().join("2_Dense");
        log::info!("Loading conf {:?}", dense_dir);

        let device = Device::cuda_if_available();
        //let device = Device::Cpu;
        let mut vs_dense = nn::VarStore::new(device);

        let init_conf = nn::LinearConfig {
            ws_init: nn::Init::Const(0.),
            bs_init: Some(nn::Init::Const(0.)),
            bias: true,
        };

        let config_file = dense_dir.join("config.json");
        let weights_file = dense_dir.join("model.ot");

        let conf = DenseConfig::from_file(&config_file);
        let linear = nn::linear(
            &vs_dense.root(),
            conf.in_features,
            conf.out_features,
            init_conf,
        );

        vs_dense.load(weights_file)?;

        Ok(Dense {
            linear,
            _conf: conf,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.apply(&self.linear).tanh()
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
