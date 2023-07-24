use std::path::PathBuf;

use rust_bert::Config;
use serde::Deserialize;
use tch::{Kind, Tensor};

#[derive(Debug, Deserialize)]
pub struct PoolingConfig {
    pub word_embedding_dimension: i64,
    pub pooling_mode_cls_token: bool,
    pub pooling_mode_mean_tokens: bool,
    pub pooling_mode_max_tokens: bool,
    pub pooling_mode_mean_sqrt_len_tokens: bool,
}

impl Config for PoolingConfig {}

pub struct Pooling {
    _conf: PoolingConfig,
}

impl Pooling {
    pub fn new<P: Into<PathBuf>>(root: P) -> Pooling {
        let pooling_dir = root.into().join("1_Pooling");
        log::info!("Loading conf {:?}", pooling_dir);

        let config_file = pooling_dir.join("config.json");
        let _conf = PoolingConfig::from_file(&config_file);

        Pooling { _conf }
    }

    pub fn forward(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Tensor {
        let input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(&token_embeddings);

        let mut output_vectors = Vec::new();

        let mut sum_mask = input_mask_expanded.copy();
        sum_mask = sum_mask.sum_dim_intlist(1, false, Kind::Float);
        let sum_embeddings =
            (token_embeddings * input_mask_expanded).sum_dim_intlist(1, false, Kind::Float);

        output_vectors.push(sum_embeddings / sum_mask);

        Tensor::cat(&output_vectors, 1)
    }
}
