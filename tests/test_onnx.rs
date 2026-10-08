//! Integration tests for the ort-only pipeline.
//!
//! Require an onnxruntime library at runtime (`ORT_DYLIB_PATH` — the manual
//! setup) and prepared checkpoints under `models/`:
//!
//! ```sh
//! ORT_DYLIB_PATH=<path>/libonnxruntime.dylib \
//!   cargo test --test test_onnx -- --nocapture
//! ```
//!
//! Tests skip themselves (with a note) when the checkpoint they need is not
//! present, so a bare `cargo test` on a fresh clone only runs the hermetic
//! `--lib` unit tests plus whatever fixtures exist locally. CI prepares the
//! distiluse checkpoint via `utils/prepare_models.py`.

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    use sbert::{SBertHF, SBertRT, Tokenizer as _};

    fn model_home(name: &str) -> PathBuf {
        env::current_dir().unwrap().join("models").join(name)
    }

    /// Skip gracefully when a checkpoint is not prepared locally.
    fn require_checkpoint(name: &str) -> Option<PathBuf> {
        let home = model_home(name);
        if home.exists() {
            Some(home)
        } else {
            println!("skipping: models/{name} not present (see utils/prepare_models.py)");
            None
        }
    }

    const TEXTS: [&str; 4] = [
        "TTThis player needs tp be reported lolz.",
        "A completely different sentence about ONNX runtime.",
        "Short one.",
        "Another sentence of ordinary length for the batch.",
    ];

    /// Smoke-check an embedding vector without baking in stale hardcoded
    /// floats. Asserts:
    /// * dimension matches `expected_dim`
    /// * every component is finite (no NaN / infinity)
    /// * L2 norm is strictly positive (the model actually produced output)
    /// * L2 norm is within a sane upper bound (catches e.g. a missing
    ///   pooling step that would leave raw per-token magnitudes)
    fn assert_embedding_sane(emb: &[f32], expected_dim: usize, model_name: &str) {
        assert_eq!(
            emb.len(),
            expected_dim,
            "{} embedding dimension (got {}, expected {})",
            model_name,
            emb.len(),
            expected_dim
        );
        assert!(
            emb.iter().all(|v| v.is_finite()),
            "{} embedding contains NaN or infinity",
            model_name
        );
        let norm_sq: f32 = emb.iter().map(|v| v * v).sum();
        assert!(norm_sq > 0.0, "{} embedding must be non-zero", model_name);
        let norm = norm_sq.sqrt();
        // For sentence-transformers checkpoints the L2 norm of an embedding
        // is typically O(1) — well below 100. If we see something huge, the
        // pipeline is probably missing a step (e.g. raw token embeddings
        // rather than pooled, or missing Dense/Normalize).
        assert!(
            norm < 100.0,
            "{} embedding L2 norm {} is implausibly large",
            model_name,
            norm
        );
    }

    /// distiluse-base-multilingual-cased: exercises the full pipeline —
    /// DistilBERT backbone (no `token_type_ids` input), mean Pooling, Dense
    /// (768 -> 512, tanh, safetensors weights), no Normalize module (the
    /// v0.2 export declares none).
    #[test]
    fn test_distiluse_encode() {
        let Some(home) = require_checkpoint("distiluse-base-multilingual-cased") else {
            return;
        };
        let texts = ["Hello, how are you?", "Bonjour, comment \u{e7}a va ?"];
        let model = SBertRT::new(&home, None).unwrap();
        let out = model.forward(&texts, 2).unwrap();
        assert_eq!(out.len(), 2);
        for emb in &out {
            assert_embedding_sane(emb, 512, "distiluse-base-multilingual-cased");
        }
    }

    /// Determinism: a second forward on the same input must match
    /// bit-for-bit.
    #[test]
    fn test_forward_is_deterministic() {
        let Some(home) = require_checkpoint("distiluse-base-multilingual-cased") else {
            return;
        };
        let model = SBertRT::new(&home, None).unwrap();
        let a = model.forward(&TEXTS, 4).unwrap();
        let b = model.forward(&TEXTS, 4).unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(ea, eb)| {
                ea.iter()
                    .zip(eb.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_diff == 0.0,
            "forward must be deterministic across calls; max diff = {}",
            max_diff
        );
    }

    /// Batch-order invariant: row i must equal the solo embedding of
    /// text i (padding must not leak across rows, order restore must be
    /// correct). Checked with descending-length inputs so the length sort
    /// actually permutes.
    #[test]
    fn test_batch_row_equals_solo_embedding() {
        let Some(home) = require_checkpoint("distiluse-base-multilingual-cased") else {
            return;
        };
        let model = SBertRT::new(&home, None).unwrap();
        let batched = model.forward(&TEXTS, 4).unwrap();
        for (i, text) in TEXTS.iter().enumerate() {
            let solo = model.forward(&[text], 1).unwrap();
            let max_diff = batched[i]
                .iter()
                .zip(solo[0].iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-5,
                "sentence {}: batched vs solo max abs diff {}",
                i,
                max_diff
            );
        }
    }

    /// all-MiniLM-L6-v2: BERT backbone (declares `token_type_ids` — the
    /// encoder's zeros default must satisfy it), Pooling + Normalize.
    /// 384-dim unit-norm output.
    #[test]
    fn test_minilm_encode() {
        let Some(home) = require_checkpoint("all-MiniLM-L6-v2") else {
            return;
        };
        let model = SBertRT::new(&home, None).unwrap();
        let out = model.forward(&TEXTS, 2).unwrap();
        assert_eq!(out.len(), TEXTS.len());
        for emb in &out {
            assert_embedding_sane(emb, 384, "all-MiniLM-L6-v2");
            let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "MiniLM declares Normalize; L2 norm must be ~1 (got {})",
                norm
            );
        }
    }

    /// The Hugging Face tokenizer backend must produce the same embeddings
    /// as the rust_tokenizers backend on identical inputs.
    #[test]
    fn test_hf_tokenizer_alias() {
        let Some(home) = require_checkpoint("distiluse-base-multilingual-cased") else {
            return;
        };
        let rt = SBertRT::new(&home, None).unwrap();
        let hf = SBertHF::new(&home, None).unwrap();
        let a = rt.forward(&TEXTS[..2], 2).unwrap();
        let b = hf.forward(&TEXTS[..2], 2).unwrap();
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(ea, eb)| {
                ea.iter()
                    .zip(eb.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "rust_tokenizers vs HF tokenizers max abs diff {}",
            max_diff
        );
    }

    /// Reverse guard: cased checkpoints must NOT be lowercased — distiluse's
    /// config says `do_lower_case: false` and its vocab is cased. If the
    /// resolution ever defaulted wrongly to `true`, "TTThis" would become
    /// "ttthis" -> subword garbage instead of `TT` `##T` `##his`.
    #[test]
    fn test_distiluse_remains_cased() {
        let Some(home) = require_checkpoint("distiluse-base-multilingual-cased") else {
            return;
        };
        let model = SBertHF::new(&home, None).unwrap();
        let tokens = model
            .tokenizer()
            .pre_tokenize(&["TTThis player needs tp be reported lolz."]);
        assert_eq!(
            tokens[0],
            [
                "[CLS]", "TT", "##T", "##his", "player", "needs", "t", "##p", "be", "reported",
                "lo", "##lz", ".", "[SEP]"
            ]
        );
    }
}
