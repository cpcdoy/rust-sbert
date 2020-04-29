#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;

use std::env;
use std::path::PathBuf;

use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer;

use rust_tokenizers::bert_tokenizer::BertTokenizer;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{Tokenizer, TruncationStrategy};

use sbert_rs::SBert;

fn bench_sbert(c: &mut Criterion) {
    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("models");
    home.push("distiluse-base-multilingual-cased");

    println!("Loading sbert_rs ...");
    let sbert_model = SBert::new(home).unwrap();

    let text = "TTThis player needs tp be reported lolz.";
    let texts = vec![text; 1000];

    c.bench_function("Encode batch 1", |b| {
        b.iter(|| sbert_model.encode(black_box(&[text])).unwrap())
    });
    c.bench_function("Encode batch 1000", |b| {
        b.iter(|| sbert_model.encode(black_box(&texts)).unwrap())
    });
}

pub fn get_bert(path: &str) -> tokenizer::Tokenizer {
    let mut tokenizer = tokenizer::Tokenizer::new(Box::new(
        WordPiece::from_files(path)
            .build()
            .expect("Files not found, run `make test` to download these files"),
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

    tokenizer
}

fn tokenize_rust_tokenizers(tokenizer: &BertTokenizer, input: Vec<&str>) -> Vec<Vec<i64>> {
    let tokenized_input = tokenizer.encode_list(input, 128, &TruncationStrategy::LongestFirst, 0);

    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap_or_else(|| 0);

    tokenized_input
        .into_iter()
        .map(|input| input.token_ids)
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .collect::<Vec<_>>()
}

fn bench_tokenizers(c: &mut Criterion) {
    let mut root: PathBuf = env::current_dir().unwrap();
    root.push("models");
    root.push("distiluse-base-multilingual-cased");

    let model_dir = root.join("0_DistilBERT");

    let vocab_file = model_dir.join("vocab.txt");

    let input = vec!["TTThis player needs tp be reported lolz."; 1000];

    let tokenizer_rust_tokenizers = BertTokenizer::from_file(&vocab_file.to_string_lossy(), false);
    let tokenizer_hugging_face = get_bert(vocab_file.to_str().unwrap());

    let encode_input = input.clone()
        .into_iter()
        .map(|s| tokenizer::EncodeInput::Single(s.into()))
        .collect::<Vec<_>>();

    let closure = || {
        tokenizer_hugging_face
            .encode_batch(encode_input.clone(), true)
            .unwrap()
            .into_iter()
            .map(|e| Box::new((*e.get_ids()).to_vec()))
            .collect::<Vec<_>>()
    };

    let closure_2 = || tokenize_rust_tokenizers(&tokenizer_rust_tokenizers, input.clone());
    c.bench_function("Tokenizer: Hugging Face, batch 1000", |b| b.iter(closure));
    c.bench_function("Tokenizer: Rust tokenizers, batch 1000", |b| {
        b.iter(closure_2)
    });
}

fn sample_size_10() -> Criterion {
    Criterion::default().sample_size(10)
}

criterion_group!(
    name = benches;
    config = sample_size_10();
    targets = bench_sbert,
    bench_tokenizers
);
criterion_main!(benches);
