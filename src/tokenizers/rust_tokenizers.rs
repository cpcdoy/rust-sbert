use std::path::PathBuf;

use rust_tokenizers::bert_tokenizer::BertTokenizer;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::TruncationStrategy;
use tch::Tensor;

use crate::tokenizers::Tokenizer;
use crate::Error;

pub struct RustTokenizers {
    tokenizer: BertTokenizer,
}

impl Tokenizer for RustTokenizers {
    fn new<P: Into<PathBuf>>(path: P) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let tokenizer = BertTokenizer::from_file(&path.into().to_string_lossy(), false, false)?;

        Ok(Self { tokenizer })
    }

    fn pre_tokenize<S: AsRef<str>>(&self, _input: &[S]) -> Vec<Vec<String>> {
        Vec::new()
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
            .map(|input| {
                let mut token_ids = input.token_ids;
                token_ids.extend(vec![0; max_len - token_ids.len()]);
                token_ids
            })
            .collect::<Vec<_>>();

        let attention_mask = tokenized_input
            .iter()
            .map(|input| {
                Tensor::of_slice(
                    &input
                        .into_iter()
                        .map(|e| match *e {
                            0 => 0 as i64,
                            _ => 1 as i64,
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let tokenized_input = tokenized_input
            .into_iter()
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();
        (tokenized_input, attention_mask)
    }
}
