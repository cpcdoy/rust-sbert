use std::path::PathBuf;

use tch::Tensor;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer;
use tokenizers::tokenizer::{
    EncodeInput, PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams,
    TruncationStrategy,
};

use crate::tokenizers::Tokenizer;
use crate::Error;

pub struct HFTokenizer {
    tokenizer: tokenizer::Tokenizer,
}

impl Tokenizer for HFTokenizer {
    fn new<P: Into<PathBuf>>(path: P) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let mut tokenizer = tokenizer::Tokenizer::new(Box::new(
            WordPiece::from_files(&path.into().to_string_lossy())
                .build()
                .expect("Files not found."),
        ));
        let bert_normalizer = BertNormalizer::new(false, false, false, false);
        tokenizer.with_normalizer(Box::new(bert_normalizer));
        tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));
        tokenizer.with_post_processor(Box::new(BertProcessing::new(
            (
                String::from("[SEP]"),
                tokenizer.get_model().token_to_id("[SEP]").unwrap(),
            ),
            (
                String::from("[CLS]"),
                tokenizer.get_model().token_to_id("[CLS]").unwrap(),
            ),
        )));

        let strategy = PaddingStrategy::BatchLongest;
        let direction = PaddingDirection::Right;
        let pad_token = String::from("[PAD]");
        let pad_id = tokenizer.get_model().token_to_id("[PAD]").unwrap();
        let pad_type_id = 0;
        tokenizer.with_padding(Some(PaddingParams {
            strategy,
            direction,
            pad_id,
            pad_type_id,
            pad_token,
        }));

        let max_length = 128;
        let stride = 0;
        let strategy = TruncationStrategy::LongestFirst;
        tokenizer.with_truncation(Some(TruncationParams {
            max_length,
            stride,
            strategy,
        }));

        Ok(Self { tokenizer })
    }

    fn pre_tokenize<S: AsRef<str>>(&self, input: &[S]) -> Vec<Vec<String>> {
        let input = input.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
        let encode_input = input
            .into_iter()
            .map(|s| EncodeInput::Single(s.into()))
            .collect();
        let encoding = self.tokenizer.encode_batch(encode_input, true).unwrap();

        encoding
            .into_iter()
            .map(|input| input.get_tokens().iter().map(String::from).collect())
            .collect()
    }

    fn tokenize<S: AsRef<str>>(&self, input: &[S]) -> (Vec<Tensor>, Vec<Tensor>) {
        let input = input.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
        let encode_input = input
            .into_iter()
            .map(|s| EncodeInput::Single(s.into()))
            .collect();
        let encoding = self.tokenizer.encode_batch(encode_input, true).unwrap();

        let attention_mask = encoding
            .iter()
            .map(|input| {
                Tensor::of_slice(
                    &input
                        .get_ids()
                        .iter()
                        .map(|e| match *e {
                            0 => 0 as i64,
                            _ => 1 as i64,
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let tokenized_input = encoding
            .into_iter()
            .map(|input| {
                Tensor::of_slice(
                    &input
                        .get_ids()
                        .iter()
                        .map(|e| *e as i64)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        (tokenized_input, attention_mask)
    }
}
