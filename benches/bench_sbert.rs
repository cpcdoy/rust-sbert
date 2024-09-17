use std::env;
use std::path::PathBuf;

use criterion::black_box;
use criterion::Criterion;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer;
use tokenizers::Model;

use sbert::{SBertHF, SBertRT};

fn rand_string(r: &mut impl Rng) -> String {
    (0..(r.gen::<f32>() * 100.0) as usize)
        .map(|_| (0x20u8 + (r.gen::<f32>() * 96.0) as u8) as char)
        .collect()
}

fn bench_sbert_rust_tokenizers(c: &mut Criterion) {
    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("models");
    home.push("distiluse-base-multilingual-cased");

    println!("Loading sbert ...");
    let sbert_model = SBertRT::new(home, None).unwrap();

    let text = "TTThis player needs tp be reported lolz.";
    c.bench_function("Encode batch, safe sbert rust tokenizer, total 1", |b| {
        b.iter(|| sbert_model.forward(black_box(&[text]), None).unwrap())
    });

    let mut r = StdRng::seed_from_u64(42);
    let texts = (0..1000).map(|_| rand_string(&mut r)).collect::<Vec<_>>();
    for batch_size in (0..7).map(|p| 2.0f32.powi(p)).collect::<Vec<f32>>().iter() {
        let batch_size = *batch_size as usize;
        let s = format!(
            "Encode batch_size {}, safe sbert, rust tokenizer, total 1000",
            batch_size
        );
        c.bench_function(&s, |b| {
            b.iter(|| black_box(sbert_model.forward(&texts, batch_size)).unwrap())
        });
    }
}

fn bench_sbert_hugging_face_tokenizers(c: &mut Criterion) {
    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("models");
    home.push("distiluse-base-multilingual-cased");

    println!("Loading sbert ...");
    let sbert_model = SBertHF::new(home, None).unwrap();

    let text = "TTThis player needs tp be reported lolz.";
    c.bench_function(
        "Encode batch, safe sbert, hugging face tokenizer, total 1",
        |b| b.iter(|| sbert_model.forward(black_box(&[text]), None).unwrap()),
    );

    let mut r = StdRng::seed_from_u64(42);
    let texts = (0..1000).map(|_| rand_string(&mut r)).collect::<Vec<_>>();
    for batch_size in (0..7).map(|p| 2.0f32.powi(p)).collect::<Vec<f32>>().iter() {
        let batch_size = *batch_size as usize;
        let s = format!(
            "Encode batch_size {}, safe sbert, hugging face tokenizer, total 1000",
            batch_size
        );
        c.bench_function(&s, |b| {
            b.iter(|| black_box(sbert_model.forward(&texts, batch_size)).unwrap())
        });
    }
}

pub fn get_bert(path: &str) -> tokenizer::Tokenizer {
    let mut tokenizer = tokenizer::Tokenizer::new(
        WordPiece::from_file(path)
            .build()
            .expect("Files not found, run `make test` to download these files"),
    );
    let bert_normalizer = BertNormalizer::new(false, false, None, false);
    tokenizer.with_normalizer(bert_normalizer);
    tokenizer.with_pre_tokenizer(BertPreTokenizer);
    let bert_processing = BertProcessing::new(
        (
            String::from("[SEP]"),
            tokenizer.get_model().token_to_id("[SEP]").unwrap(),
        ),
        (
            String::from("[CLS]"),
            tokenizer.get_model().token_to_id("[CLS]").unwrap(),
        ),
    );
    tokenizer.with_post_processor(bert_processing);

    tokenizer
}

fn bench_tokenizers(c: &mut Criterion) {
    let mut root: PathBuf = env::current_dir().unwrap();
    root.push("models");
    root.push("distiluse-base-multilingual-cased");

    let model_dir = root.join("0_DistilBERT");

    let vocab_file = model_dir.join("vocab.txt");

    let texts = vec!["TTThis player needs tp be reported lolz."; 1000];

    let tokenizer_rust_tokenizers =
        BertTokenizer::from_file(&vocab_file.to_string_lossy(), false, false).unwrap();
    let tokenizer_hugging_face = get_bert(vocab_file.to_str().unwrap());

    let encode_input = texts
        .clone()
        .into_iter()
        .map(|s| tokenizer::EncodeInput::Single(s.into()))
        .collect::<Vec<_>>();

    let hf_bench = || {
        tokenizer_hugging_face
            .encode_batch(encode_input.clone(), true)
            .unwrap()
            .into_iter()
            .map(|e| Box::new((*e.get_ids()).to_vec()))
            .collect::<Vec<_>>()
    };

    let rt_bench = || {
        let tokenized_input = tokenizer_rust_tokenizers.encode_list(
            &texts,
            128,
            &TruncationStrategy::LongestFirst,
            0,
        );

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
    };

    c.bench_function("Tokenizer: Hugging Face, batch 1000", |b| b.iter(hf_bench));
    c.bench_function("Tokenizer: Rust tokenizers, batch 1000", |b| {
        b.iter(rt_bench)
    });
}

fn sample_size_10() -> Criterion {
    Criterion::default().sample_size(10)
}

criterion::criterion_group!(
    name = benches;
    config = sample_size_10();
    targets = bench_sbert_rust_tokenizers, bench_sbert_hugging_face_tokenizers, bench_tokenizers
);
criterion::criterion_main!(benches);
