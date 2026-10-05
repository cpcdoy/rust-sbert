//! `sentence_transformers.models.Pooling` — converts per-token transformer
//! output into a single sentence embedding.
//!
//! Implements the four pooling modes of the Python `Pooling` module — `cls`,
//! `max`, `mean` and `mean_sqrt_len` — selected by the `pooling_mode_*`
//! booleans in `<module_dir>/config.json`. Multiple active modes concatenate
//! their `[batch, hidden]` outputs in the Python implementation's order
//! (cls, max, mean, mean_sqrt_len).
//!
//! A config with no active mode, or requesting a mode this crate does not
//! implement (`weightedmean`, `lasttoken`), is a hard error: silently
//! falling back to mean would produce wrong vectors for e.g. CLS-pooled
//! `multi-qa-*` / `msmarco-*` checkpoints.

use std::path::Path;

use serde::Deserialize;
use tch::{Kind, Tensor};

use crate::modules::{Features, Module};
use crate::Error;

#[derive(Debug, Deserialize)]
pub struct PoolingConfig {
    pub word_embedding_dimension: i64,
    #[serde(default)]
    pub pooling_mode_cls_token: bool,
    #[serde(default)]
    pub pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pub pooling_mode_max_tokens: bool,
    #[serde(default)]
    pub pooling_mode_mean_sqrt_len_tokens: bool,
    /// Declared by newer sentence-transformers exports; not implemented —
    /// `Pooling::new` hard-errors if it is set.
    #[serde(default)]
    pub pooling_mode_weightedmean_tokens: bool,
    /// Declared by newer sentence-transformers exports; not implemented —
    /// `Pooling::new` hard-errors if it is set.
    #[serde(default)]
    pub pooling_mode_lasttoken: bool,
}

pub struct Pooling {
    cls: bool,
    max: bool,
    mean: bool,
    mean_sqrt_len: bool,
}

impl Pooling {
    /// `module_dir` is the path to the `1_Pooling/` directory itself (not the
    /// model root). Must contain `config.json`.
    pub fn new<P: AsRef<Path>>(module_dir: P) -> Result<Pooling, Error> {
        let module_dir = module_dir.as_ref();
        let config_file = module_dir.join("config.json");
        log::info!("Loading Pooling config from {}", module_dir.display());

        // Parsed manually (not via `Config::from_file`) so a malformed or
        // unsupported config surfaces as an `Err` instead of a panic.
        let content = std::fs::read_to_string(&config_file).map_err(|e| {
            log::error!("{} not readable: {}", config_file.display(), e);
            Error::Encoding("Pooling config.json not readable")
        })?;
        let conf: PoolingConfig = serde_json::from_str(&content).map_err(|e| {
            log::error!("invalid Pooling config.json: {}", e);
            Error::Encoding("invalid Pooling config.json")
        })?;

        if conf.pooling_mode_weightedmean_tokens || conf.pooling_mode_lasttoken {
            return Err(Error::Encoding(
                "unsupported pooling mode in config.json (weightedmean/lasttoken are not implemented)",
            ));
        }
        if !(conf.pooling_mode_cls_token
            || conf.pooling_mode_mean_tokens
            || conf.pooling_mode_max_tokens
            || conf.pooling_mode_mean_sqrt_len_tokens)
        {
            return Err(Error::Encoding(
                "no pooling mode active in Pooling config.json (cls/mean/max/mean_sqrt_len all false)",
            ));
        }

        Ok(Pooling {
            cls: conf.pooling_mode_cls_token,
            max: conf.pooling_mode_max_tokens,
            mean: conf.pooling_mode_mean_tokens,
            mean_sqrt_len: conf.pooling_mode_mean_sqrt_len_tokens,
        })
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

        // Mirrors sentence_transformers.models.Pooling.forward: each active
        // mode appends one [batch, hidden] vector, concatenated below in the
        // Python implementation's fixed order.
        let input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings);
        let mut vectors: Vec<Tensor> = Vec::new();

        if self.cls {
            // Take first token by default.
            vectors.push(token_embeddings.select(1, 0));
        }

        if self.max {
            // Set padding tokens to a large negative value so they never win
            // the max.
            let padded = token_embeddings.masked_fill(&input_mask_expanded.eq(0), -1e9);
            vectors.push(padded.max_dim(1, false).0);
        }

        if self.mean || self.mean_sqrt_len {
            let sum_embeddings = (token_embeddings * &input_mask_expanded)
                .sum_dim_intlist(1, false, Kind::Float);
            let sum_mask = input_mask_expanded.sum_dim_intlist(1, false, Kind::Float);

            if self.mean {
                // Clamp sum_mask to avoid div-by-zero on all-pad batches.
                vectors.push(&sum_embeddings / &sum_mask.clamp_min(1e-9));
            }
            if self.mean_sqrt_len {
                vectors.push(sum_embeddings / sum_mask.sqrt().clamp_min(1e-9));
            }
        }

        let sentence = match vectors.len() {
            0 => return Err(Error::Encoding("no pooling mode produced output")),
            1 => vectors.remove(0),
            _ => Tensor::cat(&vectors, 1),
        };

        *features = Features::Sentence { embedding: sentence };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;
    use tch::Device;

    const B: i64 = 2;
    const S: i64 = 3;
    const H: i64 = 4;

    /// Synthetic token embeddings `[2, 3, 4]` filled with 0..24 and a mask
    /// of `[[1, 1, 0], [1, 0, 0]]` — 2 real tokens in row 0, 1 in row 1.
    fn sample_features() -> Features {
        Features::Token {
            token_embeddings: Tensor::arange(B * S * H, (Kind::Float, Device::Cpu)).view([B, S, H]),
            attention_mask: Tensor::from_slice(&[1i64, 1, 0, 1, 0, 0]).view([B, S]),
        }
    }

    fn pooling(cls: bool, max: bool, mean: bool, mean_sqrt_len: bool) -> Pooling {
        Pooling {
            cls,
            max,
            mean,
            mean_sqrt_len,
        }
    }

    fn embed(pooling: &Pooling) -> Vec<Vec<f32>> {
        let mut features = sample_features();
        pooling.forward(&mut features).unwrap();
        match features {
            Features::Sentence { embedding } => Vec::<Vec<f32>>::try_from(embedding).unwrap(),
            _ => panic!("expected sentence features"),
        }
    }

    fn assert_close(got: &[Vec<f32>], want: &[&[f32]]) {
        assert_eq!(got.len(), want.len());
        for (row_got, row_want) in got.iter().zip(want.iter()) {
            assert_eq!(row_got.len(), row_want.len());
            for (g, w) in row_got.iter().zip(row_want.iter()) {
                assert!(
                    (g - w).abs() < 1e-6,
                    "got {:?}, want {:?}",
                    got,
                    want
                );
            }
        }
    }

    #[test]
    fn cls_takes_first_token() {
        assert_close(
            &embed(&pooling(true, false, false, false)),
            &[&[0.0, 1.0, 2.0, 3.0], &[12.0, 13.0, 14.0, 15.0]],
        );
    }

    #[test]
    fn max_ignores_padding() {
        assert_close(
            &embed(&pooling(false, true, false, false)),
            &[&[4.0, 5.0, 6.0, 7.0], &[12.0, 13.0, 14.0, 15.0]],
        );
    }

    #[test]
    fn mean_ignores_padding() {
        assert_close(
            &embed(&pooling(false, false, true, false)),
            &[&[2.0, 3.0, 4.0, 5.0], &[12.0, 13.0, 14.0, 15.0]],
        );
    }

    #[test]
    fn mean_sqrt_divides_by_sqrt_token_count() {
        // Row 0: sum [4, 6, 8, 10] / sqrt(2); row 1: sum [12..15] / sqrt(1).
        let sqrt2 = 2.0f32.sqrt();
        assert_close(
            &embed(&pooling(false, false, false, true)),
            &[
                &[4.0 / sqrt2, 6.0 / sqrt2, 8.0 / sqrt2, 10.0 / sqrt2],
                &[12.0, 13.0, 14.0, 15.0],
            ],
        );
    }

    #[test]
    fn multiple_modes_concatenate_in_python_order() {
        // cls first, then mean — same order as sentence-transformers.
        let rows = embed(&pooling(true, false, true, false));
        assert_eq!(rows[0].len(), 2 * H as usize, "outputs are concatenated");
        assert_close(&vec![rows[0][..H as usize].to_vec()], &[&[0.0, 1.0, 2.0, 3.0]]);
        assert_close(&vec![rows[0][H as usize..].to_vec()], &[&[2.0, 3.0, 4.0, 5.0]]);
    }

    #[test]
    fn new_rejects_config_without_any_mode() {
        let dir = std::env::temp_dir().join("sbert-pooling-test-no-mode");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"word_embedding_dimension": 4, "pooling_mode_cls_token": false,
                "pooling_mode_mean_tokens": false, "pooling_mode_max_tokens": false,
                "pooling_mode_mean_sqrt_len_tokens": false}"#,
        )
        .unwrap();
        assert!(Pooling::new(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn new_rejects_unimplemented_mode() {
        let dir = std::env::temp_dir().join("sbert-pooling-test-lasttoken");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"word_embedding_dimension": 4, "pooling_mode_cls_token": false,
                "pooling_mode_mean_tokens": true, "pooling_mode_max_tokens": false,
                "pooling_mode_mean_sqrt_len_tokens": false,
                "pooling_mode_lasttoken": true}"#,
        )
        .unwrap();
        assert!(Pooling::new(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
