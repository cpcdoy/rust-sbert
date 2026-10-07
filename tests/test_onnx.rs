//! ONNX backend tests (cargo feature `onnx`).
//!
//! Require the local `models/all-MiniLM-L6-v2` checkpoint with a
//! `model.onnx` backbone export (see `utils/prepare_onnx.py`) and an
//! onnxruntime library at runtime (`ORT_DYLIB_PATH`). `#[ignore]`d like
//! the other MiniLM tests because the checkpoint is not in CI:
//!
//! ```sh
//! ORT_DYLIB_PATH=<path>/libonnxruntime.dylib \
//!   cargo test --features onnx --test test_onnx -- --ignored --nocapture
//! ```

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    use torch_sys::dummy_cuda_dependency;

    use sbert::{SBertHF, SBertRT};

    fn minilm_home() -> PathBuf {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("all-MiniLM-L6-v2");
        home
    }

    const TEXTS: [&str; 4] = [
        "TTThis player needs tp be reported lolz.",
        "A completely different sentence about ONNX runtime.",
        "Short one.",
        "Another sentence of ordinary length for the batch.",
    ];

    /// Parity: ONNX transformer + tch Pooling/Normalize must match the
    /// all-tch path on identical inputs (same tokenizer, same weight
    /// provenance — `rust_model.ot` and the optimum export come from the
    /// same HF checkpoint).
    #[test]
    #[ignore]
    fn test_onnx_parity_minilm() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack

        let home = minilm_home();
        let tch_model = SBertRT::new(&home, None).unwrap();
        let onnx_model = SBertRT::new_onnx(&home, None).unwrap();

        let a = tch_model.forward(&TEXTS, 4).unwrap();
        let b = onnx_model.forward(&TEXTS, 4).unwrap();

        assert_eq!(a.len(), b.len(), "same number of embeddings");
        let mut worst = 0.0f32;
        for (i, (ea, eb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ea.len(), eb.len(), "dim mismatch at sentence {i}");
            assert_eq!(ea.len(), 384, "MiniLM-L6-v2 embedding must be 384-dim");
            let max_diff = ea
                .iter()
                .zip(eb.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            worst = worst.max(max_diff);
            assert!(
                max_diff < 1e-3,
                "sentence {}: max abs diff {} exceeds 1e-3\n  tch: {:?}\n onnx: {:?}",
                i,
                max_diff,
                &ea[..5],
                &eb[..5]
            );
        }
        println!(
            "ONNX parity OK ({} sentences, worst max-abs-diff = {worst:.2e})",
            a.len()
        );
    }

    /// `new_onnx` through the Hugging Face tokenizer alias.
    #[test]
    #[ignore]
    fn test_onnx_hf_alias() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack
        let model = SBertHF::new_onnx(minilm_home(), None).unwrap();
        let out = model.forward(&TEXTS[..2], 2).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 384);
    }

    /// Parity on `distiluse-base-multilingual-cased` — exercises the Dense
    /// module (still tch) downstream of the ONNX transformer, and the
    /// DistilBERT backbone whose exports declare no `token_type_ids` (the
    /// backend's always-pass-zeros strategy must tolerate that).
    /// 512-dim output (768 -> Dense 512), normalized.
    #[test]
    #[ignore]
    fn test_onnx_parity_distiluse() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack

        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        let texts = ["Hello, how are you?", "Bonjour, comment \u{e7}a va ?"];
        let tch_model = SBertRT::new(&home, None).unwrap();
        let onnx_model = SBertRT::new_onnx(&home, None).unwrap();

        let a = tch_model.forward(&texts, 2).unwrap();
        let b = onnx_model.forward(&texts, 2).unwrap();
        assert_eq!(a.len(), b.len());
        let mut worst = 0.0f32;
        for (i, (ea, eb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                ea.len(),
                512,
                "distiluse embedding must be 512-dim (Dense 768->512)"
            );
            let max_diff = ea
                .iter()
                .zip(eb.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            worst = worst.max(max_diff);
            assert!(
                max_diff < 1e-3,
                "sentence {}: max abs diff {} exceeds 1e-3",
                i,
                max_diff
            );
        }
        println!("ONNX distiluse parity OK (worst max-abs-diff = {worst:.2e})");
    }

    /// `forward_with_attention` is unsupported on the ONNX backend (exports
    /// carry no attention outputs) and must return `Err`, not panic.
    #[test]
    #[ignore]
    fn test_onnx_attention_unsupported() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack
        let model = SBertRT::new_onnx(minilm_home(), None).unwrap();
        let r = model.forward_with_attention(&TEXTS[..1], 1);
        assert!(r.is_err(), "forward_with_attention must error on ONNX");
    }
}
