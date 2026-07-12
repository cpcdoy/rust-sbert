//! `sentence_transformers.models.Pooling` — converts per-token transformer
//! output into a single sentence embedding via (currently) mean pooling.
//!
//! Configured by `<module_dir>/config.json` whose schema includes
//! `word_embedding_dimension` and the `pooling_mode_*_tokens` booleans. The
//! original rust-sbert implementation honors mean-pooling only; the booleans
//! are read for forward compatibility but only mean-pool is actually applied.

use std::path::Path;

use rust_bert::Config;
use serde::Deserialize;
use tch::Kind;

use crate::modules::{Features, Module};
use crate::Error;

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
    /// `module_dir` is the path to the `1_Pooling/` directory itself (not the
    /// model root). Must contain `config.json`.
    pub fn new<P: AsRef<Path>>(module_dir: P) -> Result<Pooling, Error> {
        let module_dir = module_dir.as_ref();
        let config_file = module_dir.join("config.json");
        log::info!("Loading Pooling config from {}", module_dir.display());
        let _conf = PoolingConfig::from_file(&config_file);
        Ok(Pooling { _conf })
    }
}

impl Module for Pooling {
    fn forward(&self, features: &mut Features) -> Result<(), Error> {
        // Reborrow the &mut Tensor fields as &Tensor for the arithmetic.
        // (tch's Mul/Div impls are on `&Tensor`, not `&mut Tensor`.)
        let (token_embeddings, attention_mask) = match features {
            Features::Token {
                token_embeddings,
                attention_mask,
            } => (&*token_embeddings, &*attention_mask),
            _ => {
                return Err(Error::Encoding(
                    "Pooling received non-token features",
                ));
            }
        };

        // Mean pooling with attention-mask weighting (matches the canonical
        // sentence-transformers Pooling for `pooling_mode_mean_tokens=true`).
        let input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings);

        let mut sum_mask = input_mask_expanded.copy();
        sum_mask = sum_mask.sum_dim_intlist(1, false, Kind::Float);
        let sum_embeddings =
            (token_embeddings * &input_mask_expanded).sum_dim_intlist(1, false, Kind::Float);

        // Clamp sum_mask to avoid div-by-zero on all-pad batches.
        let sum_mask = sum_mask.clamp_min(1e-9);

        let sentence = sum_embeddings / sum_mask;

        *features = Features::Sentence { embedding: sentence };
        Ok(())
    }
}
