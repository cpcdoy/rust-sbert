//! Tensor-free pipeline module system for sentence-transformers models —
//! ndarray implementation (cargo feature `onnx`).
//!
//! Mirror of [`crate::modules`] (the tch version): same `Features`/`Module`
//! shape and the same Pooling/Normalize/Dense semantics, over `ndarray`
//! arrays. The transformer backend wraps rust-bert's `ONNXEncoder`. The
//! manifest parser is shared ([`crate::modules::manifest`]).

pub use crate::modules::manifest;

pub mod dense;
pub mod normalize;
pub mod pooling;
pub mod safetensors;
pub mod transformer;

pub use dense::Dense;
pub use normalize::Normalize;
pub use pooling::Pooling;
pub use transformer::{OnnxBackend, TransformerBackend, TransformerOutput};

use ndarray::{Array2, Array3};

use crate::Error;

/// Mutable feature bag passed through the post-transformer pipeline.
///
/// - [`Features::Token`] is produced by the transformer backend and consumed
///   by `Pooling` (which converts it to `Sentence`).
/// - [`Features::Sentence`] is produced by `Pooling` and consumed/produced by
///   `Dense` and `Normalize`.
///
/// Modules that receive a variant they don't expect should return
/// `Error::Encoding(...)`.
pub enum Features {
    /// Output of the transformer: per-token embeddings + the attention mask
    /// that produced them. Shapes: `[batch, seq, hidden]` and `[batch, seq]`.
    Token {
        token_embeddings: Array3<f32>,
        attention_mask: Array2<i64>,
    },
    /// Output of `Pooling` (and downstream modules). Shape: `[batch, hidden]`.
    Sentence { embedding: Array2<f32> },
}

/// A post-transformer pipeline module.
///
/// Implementations mutate `features` in place. The trait is object-safe so
/// modules can be stored as `Vec<Box<dyn Module>>`.
pub trait Module: Send {
    fn forward(&self, features: &mut Features) -> Result<(), Error>;
}
