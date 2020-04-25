use std::env;
use std::path::PathBuf;
use std::time::Instant;

use sbert_rs::SBert;

fn main() {
    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("models");
    home.push("distiluse-base-multilingual-cased");

    println!("Loading sbert_rs ...");
    let sbert_model = SBert::new(home).unwrap();

    let texts = ["TTThis player needs tp be reported lolz."];

    println!("Encoding {:?}...", texts);
    let before = Instant::now();
    let output = sbert_model.encode(&texts).unwrap();
    println!("Elapsed time: {:.2?}", before.elapsed());

    println!("Embeddings: {:?}", output);
    println!("Embeddings: {:?}", output.get(0).get(0));
    println!("Embeddings: {:?}", output.get(0).get(1));
    println!("Embeddings: {:?}", output.get(0).get(2));
}
