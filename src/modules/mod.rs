//! Pipeline module system for sentence-transformers models.
//!
//! Mirrors the structure of Python `sentence_transformers.SentenceTransformer`:
//! a model is a transformer backend followed by an ordered list of
//! post-processing modules, composed by parsing the model's `modules.json`.
//!
//! Each post-processing module implements [`Module`] and mutates a [`Features`]
//! bag in place. The transformer backend implements [`TransformerBackend`] and
//! is the entry point that converts token ids into the first `Features` value.

pub mod dense;
pub mod manifest;
pub mod normalize;
pub mod pooling;
pub mod transformer;

pub use dense::Dense;
pub use manifest::{parse as parse_manifest, resolve_module_dir, ModuleEntry};
pub use normalize::Normalize;
pub use pooling::Pooling;
pub use transformer::{
    BertBackend, DistilBertBackend, TransformerBackend, TransformerOutput,
};

use tch::Tensor;

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
/// live behind a `Mutex` (as in linkmesh's `SBERT_MODEL`), which only
/// requires the inner type to be `Send`.
pub trait Module: Send {
    fn forward(&self, features: &mut Features) -> Result<(), Error>;
}
