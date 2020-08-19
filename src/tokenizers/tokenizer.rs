use std::path::PathBuf;

use tch::Tensor;

use crate::Error;

pub trait Tokenizer {
    fn new<P: Into<PathBuf>>(path: P) -> Result<Self, Error>
    where
        Self: Sized;
    fn pre_tokenize<S: AsRef<str>>(&self, input: &[S]) -> Vec<Vec<String>>;
    fn tokenize<S: AsRef<str>>(&self, input: &[S]) -> (Vec<Tensor>, Vec<Tensor>);
}
