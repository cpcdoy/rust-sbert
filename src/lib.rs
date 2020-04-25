pub mod layers;
mod sbert;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Torch VarStore error: {0}")]
    VarStore(failure::Error),
    #[error("Encoding error: {0}")]
    Encoding(&'static str),
}

pub use sbert::SBert;
