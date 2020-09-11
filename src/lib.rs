pub mod layers;
pub mod sbert;
pub mod tokenizers;

use rust_tokenizers::preprocessing::error::TokenizerError;
use tch::TchError;
use thiserror::Error;

pub use crate::sbert::SBert;
pub use crate::tokenizers::{HFTokenizer, RustTokenizers, Tokenizer};

pub mod att {
    pub type Attention = Vec<f32>;
    pub type Attention2D = Vec<Vec<f32>>;
    pub type Heads = Vec<Attention2D>;
    pub type Layers = Vec<Heads>;
}

pub type Embeddings = Vec<f32>;
pub type Attentions = Vec<att::Layers>;

pub type SBertRT = SBert<RustTokenizers>;
pub type SBertHF = SBert<HFTokenizer>;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("Torch error: {0}")]
    Torch(#[from] TchError),
    #[error("Encoding error: {0}")]
    Encoding(&'static str),
    #[error("Tokenizer error: {0}")]
    RustTokenizers(#[from] TokenizerError),
}
