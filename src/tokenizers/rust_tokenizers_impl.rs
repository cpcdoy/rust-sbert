use std::path::PathBuf;

use tch::Tensor;
use rust_tokenizers::bert_tokenizer::BertTokenizer;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::TruncationStrategy;

use crate::tokenizers::{Tokenizer};
use crate::Error;

pub struct RustTokenizers {
    tokenizer: BertTokenizer,
}

impl Tokenizer for RustTokenizers {
    fn new<P: Into<PathBuf>>(path: P) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let tokenizer = BertTokenizer::from_file(&path.into().to_string_lossy(), false);

        Ok(Self { tokenizer })
    }

    fn tokenize<S: AsRef<str>>(&self, input: &[S]) -> (Vec<Tensor>, Vec<Tensor>) {
        use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::Tokenizer;

        let tokenized_input = self.tokenizer.encode_list(
            input.iter().map(|v| v.as_ref()).collect::<Vec<_>>(),
            128,
            &TruncationStrategy::LongestFirst,
            0,
        );

        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap_or_else(|| 0);

        let tokenized_input = tokenized_input
            .into_iter()
            .map(|input| input.token_ids)
            .map(|mut input| {
                input.extend(vec![0; max_len - input.len()]);
                input
            })
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();

        (tokenized_input, Vec::new())
    }
}
