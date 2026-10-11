pub mod settings;

#[cfg(feature = "torch")]
pub mod distilroberta;
#[cfg(feature = "onnx")]
pub mod sbert_onnx;
#[cfg(feature = "torch")]
pub mod sbert_torch;

// Both drivers expose the same public shape (`SentenceTransformer<T>` with
// `new(path, Option<Device>)` / `forward`), selected by cargo feature; the
// torch one additionally has `forward_with_attention`. The `not(...)`
// guards keep the names unique when both features are (erroneously) on, so
// the compile_error! in lib.rs is the failure the user sees.
#[cfg(all(feature = "onnx", not(feature = "torch")))]
pub use self::sbert_onnx::SentenceTransformer;
#[cfg(all(feature = "torch", not(feature = "onnx")))]
pub use self::sbert_torch::SentenceTransformer;

#[cfg(any(
    all(feature = "torch", not(feature = "onnx")),
    all(feature = "onnx", not(feature = "torch")),
))]
pub use self::SentenceTransformer as SBert;

// Utils
pub fn pad_sort<O: Ord>(arr: &[O]) -> Vec<usize> {
    let mut idx = (0..arr.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| arr[i].cmp(&arr[j]));
    idx
}
