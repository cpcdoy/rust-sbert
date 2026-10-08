//! Dump torch-backend embeddings for cross-backend validation
//! (`cargo run --example dump_torch -- <model-dir> <out.json>`).
//!
//! Encodes a fixed corpus twice — batched (3 per batch, mixed lengths) and
//! one sentence at a time — and writes both to JSON. The onnx-feature twin
//! `dump_onnx.rs` produces the byte-comparable file; a compare script then
//! checks backend-vs-backend parity and batch-vs-solo invariance.

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let home = args.get(1).map(String::as_str).unwrap_or("models");
    let out = args.get(2).map(String::as_str).unwrap_or("dump_torch.json");

    let texts: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
    let model = sbert::SBertRT::new(home, None)?;

    let batched = model.forward(&texts, 3)?;
    let solo: Vec<sbert::Embeddings> = texts
        .iter()
        .map(|t| model.forward(&[t], 1).unwrap().remove(0))
        .collect();

    let json = serde_json::json!({
        "backend": "torch",
        "model": home,
        "texts": texts,
        "batched": batched,
        "solo": solo,
    });
    fs::write(out, serde_json::to_string(&json)?)?;
    println!("wrote {out} ({} texts)", texts.len());
    Ok(())
}

/// Fixed corpus — the twin example must keep this byte-identical. Covers:
/// cased English, Western languages (distiluse is multilingual), emoji/CJK,
/// empty input, and a >max_seq_length input to exercise truncation.
pub const CORPUS: [&str; 9] = [
    "TTThis player needs tp be reported lolz.",
    "Hello, how are you?",
    "Bonjour, comment \u{e7}a va ?",
    "Hola, \u{bf}c\u{f3}mo est\u{e1}s?",
    "Ein kurzer Satz auf Deutsch.",
    "Short one.",
    "emoji \u{1f680} and CJK \u{4e2d}\u{6587}\u{6d4b}\u{8bd5} mixed into one sentence",
    "",
    "This sentence is deliberately very long so that the tokenizer has to truncate it at the checkpoint's max_seq_length which means the two backends must agree not only on ordinary encoding but also on the truncation boundary and everything after it keeps going and going and going with many many words",
];
