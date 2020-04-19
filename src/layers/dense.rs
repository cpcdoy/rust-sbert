use rust_bert::Config;

use tch::{nn, Device, Tensor};

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[allow(non_camel_case_types)]
#[derive(Debug, Serialize, Deserialize)]
pub enum Activation {
    tanh,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DenseConfig {
    pub in_features: i64,
    pub out_features: i64,
    pub activation_function: Activation,
}

impl Config<DenseConfig> for DenseConfig {}

pub struct Dense {
    linear: nn::Linear,
    conf: DenseConfig,
}

impl Dense {
    pub fn new(path: &str) -> Dense {
        // Dense
        let mut conf_copy = PathBuf::from(path);
        conf_copy.push("2_Dense");
        println!("Loading conf {:?}", conf_copy);

        let device = Device::cuda_if_available();
        let mut vs_dense = nn::VarStore::new(device);
        let init_conf = nn::LinearConfig {
            ws_init: nn::Init::Const(0.),
            bs_init: Some(nn::Init::Const(0.)),
        };
        let config_path = &conf_copy.as_path().join("config.json");
        let weights_path = &conf_copy.as_path().join("model.ot");

        let conf = DenseConfig::from_file(config_path);

        let linear = nn::linear(&vs_dense.root(), conf.in_features, conf.out_features, init_conf);
        vs_dense.load(weights_path).unwrap();


        Dense { linear, conf }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.apply(&self.linear).tanh()
    }
}
