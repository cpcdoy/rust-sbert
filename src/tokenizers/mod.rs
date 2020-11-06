mod hf_tokenizers;
mod rust_tokenizers;
mod rust_tokenizers_sentencepiece;

use std::path::PathBuf;

use tch::Tensor;

pub trait Tokenizer {
    fn new<P: Into<PathBuf>>(path: P) -> Result<Self, crate::Error>
    where
        Self: Sized;
    fn pre_tokenize<S: AsRef<str>>(&self, input: &[S]) -> Vec<Vec<String>>;
    fn tokenize<S: AsRef<str>>(&self, input: &[S]) -> (Vec<Tensor>, Vec<Tensor>);
}

pub use self::hf_tokenizers::HFTokenizer;
pub use self::rust_tokenizers::RustTokenizers;
pub use self::rust_tokenizers_sentencepiece::RustTokenizersSentencePiece;
