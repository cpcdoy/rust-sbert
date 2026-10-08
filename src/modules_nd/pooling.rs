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

use ndarray::{Array2, Axis};
use serde::Deserialize;

use crate::modules_nd::{Features, Module};
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
        let (token_embeddings, attention_mask) = match features {
            Features::Token {
                token_embeddings,
                attention_mask,
            } => (&*token_embeddings, &*attention_mask),
            _ => {
                return Err(Error::Encoding("Pooling received non-token features"));
            }
        };

        let (batch, seq, hidden) = (
            token_embeddings.shape()[0],
            token_embeddings.shape()[1],
            token_embeddings.shape()[2],
        );

        // f32 mask expanded to [batch, seq, hidden] — the same layout the
        // tch implementation used (mask.unsqueeze(-1).expand_as).
        let mask3 = attention_mask
            .mapv(|m| m as f32)
            .insert_axis(Axis(2))
            .broadcast((batch, seq, hidden))
            .expect("mask broadcast")
            .to_owned();

        // Mirrors sentence_transformers.models.Pooling.forward: each active
        // mode appends one [batch, hidden] vector, concatenated below in the
        // Python implementation's fixed order.
        let mut vectors: Vec<Array2<f32>> = Vec::new();

        if self.cls {
            // Take first token by default.
            vectors.push(token_embeddings.index_axis(Axis(1), 0).into_owned());
        }

        if self.max {
            // Set padding tokens to a large negative value so they never win
            // the max: emb*mask + (1-mask)*(-1e9).
            let neg_fill = mask3.mapv(|m| (1.0 - m) * -1e9);
            let padded = token_embeddings * &mask3 + &neg_fill;
            // Max over the sequence axis (ndarray 0.17 keeps min/max in
            // ndarray-stats; fold_axis is equivalent for f32).
            vectors.push(padded.fold_axis(
                Axis(1),
                f32::NEG_INFINITY,
                |&a, &b| {
                    if a > b {
                        a
                    } else {
                        b
                    }
                },
            ));
        }

        if self.mean || self.mean_sqrt_len {
            let sum_embeddings = (token_embeddings * &mask3).sum_axis(Axis(1));
            // Expanded-mask sum over the sequence axis: [batch, hidden],
            // every entry the real-token count (tch's sum_mask).
            let sum_mask = mask3.sum_axis(Axis(1));

            if self.mean {
                // Clamp to avoid div-by-zero on all-pad batches.
                vectors.push(&sum_embeddings / &sum_mask.mapv(|c| c.max(1e-9)));
            }
            if self.mean_sqrt_len {
                vectors.push(&sum_embeddings / &sum_mask.mapv(|c| c.sqrt().max(1e-9)));
            }
        }

        let sentence = match vectors.len() {
            0 => return Err(Error::Encoding("no pooling mode produced output")),
            1 => vectors.remove(0),
            _ => ndarray::concatenate(
                Axis(1),
                &vectors.iter().map(|v| v.view()).collect::<Vec<_>>(),
            )
            .map_err(|_| Error::Encoding("pooling mode concatenation failed"))?,
        };

        *features = Features::Sentence {
            embedding: sentence,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// Synthetic token embeddings `[2, 3, 4]` filled with 0..24 and a mask
    /// of `[[1, 1, 0], [1, 0, 0]]` — 2 real tokens in row 0, 1 in row 1.
    fn sample_features() -> Features {
        Features::Token {
            token_embeddings: Array3::from_shape_fn((2, 3, 4), |(i, j, k)| {
                ((i * 3 + j) * 4 + k) as f32
            }),
            attention_mask: Array2::from_shape_vec((2, 3), vec![1i64, 1, 0, 1, 0, 0]).unwrap(),
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
            Features::Sentence { embedding } => {
                embedding.outer_iter().map(|r| r.to_vec()).collect()
            }
            _ => panic!("expected sentence features"),
        }
    }

    fn assert_close(got: &[Vec<f32>], want: &[&[f32]]) {
        assert_eq!(got.len(), want.len());
        for (row_got, row_want) in got.iter().zip(want.iter()) {
            assert_eq!(row_got.len(), row_want.len());
            for (g, w) in row_got.iter().zip(row_want.iter()) {
                assert!((g - w).abs() < 1e-6, "got {:?}, want {:?}", got, want);
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
        assert_eq!(rows[0].len(), 2 * 4, "outputs are concatenated");
        assert_close(&vec![rows[0][..4].to_vec()], &[&[0.0, 1.0, 2.0, 3.0]]);
        assert_close(&vec![rows[0][4..].to_vec()], &[&[2.0, 3.0, 4.0, 5.0]]);
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
