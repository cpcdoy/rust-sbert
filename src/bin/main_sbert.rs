use std::env;
use std::path::PathBuf;
use std::time::Instant;

use rSBert::models::sbert::SBert;

fn main() -> failure::Fallible<()> {
    let mut home: PathBuf = env::current_dir().unwrap();
    home.push("models");
    home.push("distiluse-base-multilingual-cased");

    println!("Loading rSBert...");
    let sbert_model = SBert::new(home.to_str().unwrap());

    let texts = ["TTThis player needs tp be reported lolz."];

    println!("Encoding {:?}...", texts);
    let before = Instant::now();
    let output = sbert_model.encode(texts.to_vec()).unwrap();
    println!("Elapsed time: {:.2?}", before.elapsed());

    println!("Embeddings: {:?}", output);
    println!("Embeddings: {:?}", output.get(0).get(0));
    println!("Embeddings: {:?}", output.get(0).get(1));
    println!("Embeddings: {:?}", output.get(0).get(2));

    Ok(())
}
