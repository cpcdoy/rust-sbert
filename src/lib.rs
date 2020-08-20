pub mod layers;
pub mod sbert;
pub mod tokenizers;

use thiserror::Error;
use tch::TchError;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Torch VarStore error: {0}")]
    VarStore(TchError),
    #[error("Encoding error: {0}")]
    Encoding(&'static str),
}

pub use crate::tokenizers::Tokenizer;
pub use crate::tokenizers::hf_tokenizers_impl::HFTokenizer;
pub use crate::tokenizers::rust_tokenizers_impl::RustTokenizers;
pub use crate::sbert::SBert;
pub use crate::sbert::SafeSBert;

pub type SBertRT = SBert<RustTokenizers>;
pub type SBertHF = SBert<HFTokenizer>;

pub type SafeSBertRT = SafeSBert<RustTokenizers>;
pub type SafeSBertHF = SafeSBert<HFTokenizer>;
