pub mod layers;
pub mod sbert;
pub mod tokenizers;

use tch::TchError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Torch VarStore error: {0}")]
    VarStore(TchError),
    #[error("Encoding error: {0}")]
    Encoding(&'static str),
    #[error("Multithreading issue")]
    Multithreading(Box<dyn std::any::Any + 'static + Send>),
}

impl From<Box<dyn std::any::Any + 'static + Send>> for Error {
    fn from(source: Box<dyn std::any::Any + 'static + Send>) -> Self {
        Self::Multithreading(source)
    }
}

pub use crate::sbert::SBert;
pub use crate::sbert::SafeSBert;
pub use crate::tokenizers::hf_tokenizers_impl::HFTokenizer;
pub use crate::tokenizers::rust_tokenizers_impl::RustTokenizers;
pub use crate::tokenizers::Tokenizer;

pub type SBertRT = SBert<RustTokenizers>;
pub type SBertHF = SBert<HFTokenizer>;

pub type SafeSBertRT = SafeSBert<RustTokenizers>;
pub type SafeSBertHF = SafeSBert<HFTokenizer>;

pub type Embeddings = Vec<f32>;
pub type Attentions = Vec<att::Layers>;

pub mod att {
    pub type Attention = Vec<f32>;
    pub type Attention2D = Vec<Vec<f32>>;
    pub type Heads = Vec<Attention2D>;
    pub type Layers = Vec<Heads>;
}
