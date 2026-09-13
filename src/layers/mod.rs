//! Deprecated. The pooling/dense modules now live in [`crate::modules`].
//!
//! This module is kept only as a re-export shim so external code that
//! references `sbert::layers::{Dense, Pooling}` continues to compile. The
//! shims will be removed in a future release.

#[deprecated(note = "use `sbert::modules::Dense` instead")]
pub use crate::modules::Dense;

#[deprecated(note = "use `sbert::modules::Pooling` instead")]
pub use crate::modules::Pooling;
