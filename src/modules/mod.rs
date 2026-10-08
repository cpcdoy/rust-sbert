//! Pipeline module system for sentence-transformers models — libtorch
//! implementation (cargo feature `torch`).
//!
//! Mirrors the structure of Python `sentence_transformers.SentenceTransformer`:
//! a model is a transformer backend followed by an ordered list of
//! post-processing modules, composed by parsing the model's `modules.json`.
//! The tch-free mirror of this module system lives in `crate::modules_nd`
//! (feature `onnx`, not present in torch-only builds); the shared manifest
//! parser lives here in [`manifest`] and is cfg-free.

pub mod manifest;

#[cfg(feature = "torch")]
pub mod dense;
#[cfg(feature = "torch")]
pub mod normalize;
#[cfg(feature = "torch")]
pub mod pooling;
#[cfg(feature = "torch")]
pub mod transformer;

#[cfg(feature = "torch")]
pub use dense::Dense;
pub use manifest::{parse as parse_manifest, resolve_module_dir, ModuleEntry};
#[cfg(feature = "torch")]
pub use normalize::Normalize;
#[cfg(feature = "torch")]
pub use pooling::Pooling;
#[cfg(feature = "torch")]
pub use transformer::{BertBackend, DistilBertBackend, TransformerBackend, TransformerOutput};

#[cfg(feature = "torch")]
use tch::Tensor;

#[cfg(feature = "torch")]
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
#[cfg(feature = "torch")]
pub enum Features {
    /// Output of the transformer: per-token embeddings + the attention mask
    /// that produced them. Shape: `[batch, seq, hidden]` and `[batch, seq]`.
    Token {
        token_embeddings: Tensor,
        attention_mask: Tensor,
    },
    /// Output of `Pooling` (and downstream modules). Shape: `[batch, hidden]`.
    Sentence { embedding: Tensor },
}

/// A post-transformer pipeline module.
///
/// Implementations mutate `features` in place. The trait is object-safe so
/// modules can be stored as `Vec<Box<dyn Module>>`.
///
/// Bound is `Send` only (not `Sync`) because the underlying `tch::Tensor`
/// contains a raw `*mut C_tensor` which is `!Sync`. The pipeline is meant to
/// live behind a `Mutex` (e.g. a shared model singleton in a server), which
/// only requires the inner type to be `Send`.
#[cfg(feature = "torch")]
pub trait Module: Send {
    fn forward(&self, features: &mut Features) -> Result<(), Error>;
}
