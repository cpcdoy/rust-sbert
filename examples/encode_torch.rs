//! Minimal end-to-end encode with the libtorch backend
//! (`cargo run --example encode_torch -- <model-dir>`).
//!
//! Mirrors `encode_onnx.rs` (the onnx-feature twin) so the two backends'
//! outputs can be compared: same input, same checkpoint, printed with full
//! f32 precision.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let home = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("models/distiluse-base-multilingual-cased");

    let model = sbert::SBertRT::new(home, None)?;
    let out = model.forward(&["TTThis player needs tp be reported lolz."], 1)?;

    println!("{:?}", out[0]);
    Ok(())
}
