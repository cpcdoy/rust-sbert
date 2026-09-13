mod hf_tokenizers;
mod rust_tokenizers;
mod rust_tokenizers_sentencepiece;

use std::path::PathBuf;

use tch::Tensor;

pub trait Tokenizer {
    /// `path` is tokenizer-impl specific (a vocab file for the BERT impls,
    /// a model directory for sentencepiece). `do_lower_case` comes from the
    /// checkpoint's tokenizer_config.json and MUST match the vocab the
    /// model ships: a lowercase-only vocab (e.g. all-MiniLM-L6-v2) silently
    /// maps every capitalized word to [UNK] when tokenized case-sensitively,
    /// collapsing distinct inputs to byte-identical embeddings.
    fn new<P: Into<PathBuf>>(path: P, do_lower_case: bool) -> Result<Self, crate::Error>
    where
        Self: Sized;
    fn pre_tokenize<S: AsRef<str>>(&self, input: &[S]) -> Vec<Vec<String>>;
    fn tokenize<S: AsRef<str>>(&self, input: &[S]) -> (Vec<Tensor>, Vec<Tensor>);
}

pub use self::hf_tokenizers::HFTokenizer;
pub use self::rust_tokenizers::RustTokenizers;
pub use self::rust_tokenizers_sentencepiece::RustTokenizersSentencePiece;
