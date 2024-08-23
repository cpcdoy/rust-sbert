use std::env;
use std::path::PathBuf;

use criterion::black_box;
use criterion::Criterion;
use rand::{rngs::StdRng, Rng, SeedableRng};

use sbert::DistilRobertaForSequenceClassificationRT;

fn rand_string(r: &mut impl Rng) -> String {
    (0..(r.gen::<f32>() * 100.0) as usize)
        .map(|_| (0x20u8 + (r.gen::<f32>() * 96.0) as u8) as char)
        .collect()
}

fn bench_distilroberta_rust_tokenizers_sentencepiece(c: &mut Criterion) {
    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("models");
    home.push("distilroberta_toxicity");

    println!("Loading distilroberta ...");
    let sbert_model = DistilRobertaForSequenceClassificationRT::new(home, None).unwrap();

    let text = "TTThis player needs tp be reported lolz.";
    c.bench_function(
        "Encode batch, distilroberta, rust tokenizers sentencepiece, total 1",
        |b| b.iter(|| sbert_model.forward(black_box(&[text]), None).unwrap()),
    );

    let mut r = StdRng::seed_from_u64(42);
    let texts = (0..1000).map(|_| rand_string(&mut r)).collect::<Vec<_>>();
    for batch_size in (0..7).map(|p| 2.0f32.powi(p)).collect::<Vec<f32>>().iter() {
        let batch_size = *batch_size as usize;
        let s = format!(
            "Encode batch_size {}, distilroberta, rust tokenizers sentencepiece, total 1000",
            batch_size
        );
        c.bench_function(&s, |b| {
            b.iter(|| black_box(sbert_model.forward(&texts, batch_size)).unwrap())
        });
    }
}

fn sample_size_10() -> Criterion {
    Criterion::default().sample_size(10)
}

criterion::criterion_group!(
    name = benches;
    config = sample_size_10();
    targets = bench_distilroberta_rust_tokenizers_sentencepiece
);
criterion::criterion_main!(benches);
