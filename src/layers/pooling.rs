use rust_bert::Config;

use tch::{Kind, Tensor};

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct PoolingConfig {
    pub word_embedding_dimension: i64,
    pub pooling_mode_cls_token: bool,
    pub pooling_mode_mean_tokens: bool,
    pub pooling_mode_max_tokens: bool,
    pub pooling_mode_mean_sqrt_len_tokens: bool,
}

impl Config<PoolingConfig> for PoolingConfig {}

pub struct Pooling {
    conf: PoolingConfig,
}

impl Pooling {
    pub fn new(path: &str) -> Pooling {
        // Pooling
        let mut conf_copy = PathBuf::from(path);
        conf_copy.push("1_Pooling");
        println!("Loading conf {:?}", conf_copy);

        let config_path = &conf_copy.as_path().join("config.json");
        let conf = PoolingConfig::from_file(config_path);

        Pooling { conf }
    }

    pub fn forward(&self, token_embeddings: &Tensor) -> Tensor {
        let attention_mask = token_embeddings.ones_like();
        let input_mask_expanded = attention_mask.expand_as(&token_embeddings);

        let mut output_vectors = Vec::new();
        let dim = [1];
        let mut sum_mask = input_mask_expanded.copy();
        sum_mask = sum_mask.sum1(&dim, false, Kind::Float);
        //println!("sum_mask {:?}", sum_mask);
        //println!("sum_mask {:?}", sum_mask.get(0).get(0));
        //println!("sum_mask {:?}", sum_mask.get(0).get(1));
        //println!("sum_mask {:?}", sum_mask.get(0).get(2));
        let sum_embeddings =
            (token_embeddings * input_mask_expanded).sum1(&dim, false, Kind::Float);

        output_vectors.push(sum_embeddings / sum_mask);

        let output_vector = Tensor::cat(&output_vectors, 1);

        output_vector
    }
}
