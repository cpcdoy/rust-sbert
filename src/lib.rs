//! # sbert — sentence embeddings, libtorch or ONNX Runtime
//!
//! Rust implementation of Sentence-BERT style embedding pipelines
//! (`sentence-transformers` checkpoints driven by `modules.json`), with two
//! interchangeable inference backends selected at compile time:
//!
//! * `torch` (**default**) — rust-bert TorchScript transformer + tch
//!   Pooling/Dense/Normalize; weights from `model.ot` VarStore archives.
//!   Needs a libtorch at build/run time. The only backend with
//!   `forward_with_attention` and MPS support.
//! * `onnx` — bare-backbone `model.onnx` via ONNX Runtime (rust-bert's
//!   ndarray `ONNXEncoder`), Pooling/Dense/Normalize in pure ndarray.
//!   **No libtorch anywhere in the graph**; needs an onnxruntime shared
//!   library at runtime via `ORT_DYLIB_PATH` (see README).
//!
//! The two features are mutually exclusive by design ("you either use ort
//! or tch"); pick with `--features torch` (default) or
//! `--no-default-features --features onnx`.
//!
//! Both drivers expose the same shape:
//! `SentenceTransformer::new(path, Option<Device>)` / `forward(texts, batch)`
//! with `SBert`/`SBertRT`/`SBertHF` aliases.

// Backend exclusivity — the whole point of the split is one tensor library
// per build (see ORT_ONLY_PLAN.md §3.1). Loud compile errors beat silent
// feature unification picking a winner.
#[cfg(all(feature = "torch", feature = "onnx"))]
compile_error!("features `torch` and `onnx` are mutually exclusive backends — pick one");
#[cfg(not(any(feature = "torch", feature = "onnx")))]
compile_error!("no backend selected — enable `torch` (default) or `onnx`");

pub mod models;
pub mod modules;
#[cfg(feature = "onnx")]
pub mod modules_nd;
pub mod tokenizers;

use rust_bert::RustBertError;
use rust_tokenizers::error::TokenizerError;
use thiserror::Error;

/// Inference device. Re-exported from rust-bert so both backends share one
/// definition: under `torch` it converts to `tch::Device` (Cuda(i) / Cpu;
/// Mps and Vulkan map to Cpu — see the fork's conversion), under `onnx` it
/// selects the ONNX Runtime execution provider.
pub use rust_bert::Device;

#[cfg(feature = "torch")]
pub use crate::models::distilroberta::DistilRobertaForSequenceClassification;
pub use crate::models::SentenceTransformer;
// Backward-compat: existing code refers to `SBert<T>`, `SBertRT`, `SBertHF`.
// The underlying type is the feature-selected `SentenceTransformer<T>`.
pub use crate::models::SBert;
pub use crate::modules::{manifest, parse_manifest};
#[cfg(all(feature = "torch", not(feature = "onnx")))]
pub use crate::modules::{
    transformer, BertBackend, Dense, DistilBertBackend, Features, Module, Normalize, Pooling,
    TransformerBackend, TransformerOutput,
};
#[cfg(all(feature = "onnx", not(feature = "torch")))]
pub use crate::modules_nd::{
    transformer, Dense, Features, Module, Normalize, OnnxBackend, Pooling, TransformerBackend,
    TransformerOutput,
};
pub use crate::tokenizers::{HFTokenizer, RustTokenizers, RustTokenizersSentencePiece, Tokenizer};

pub mod att {
    pub type Attention = Vec<f32>;
    pub type Attention2D = Vec<Vec<f32>>;
    pub type Heads = Vec<Attention2D>;
    pub type Layers = Vec<Heads>;
}

pub type Embeddings = Vec<f32>;
pub type Attentions = Vec<att::Layers>;

pub type SBertRT = SentenceTransformer<RustTokenizers>;
pub type SBertHF = SentenceTransformer<HFTokenizer>;

#[cfg(feature = "torch")]
pub type DistilRobertaForSequenceClassificationRT =
    DistilRobertaForSequenceClassification<RustTokenizersSentencePiece>;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[cfg(feature = "torch")]
    #[error("Torch error: {0}")]
    Torch(#[from] tch::TchError),
    #[error("Encoding error: {0}")]
    Encoding(&'static str),
    #[error("Tokenizer error: {0}")]
    RustTokenizers(#[from] TokenizerError),
    #[error("Rust bert error: {0}")]
    RustBert(#[from] RustBertError),
    #[cfg(feature = "onnx")]
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
}
