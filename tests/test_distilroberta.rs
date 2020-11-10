#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    use torch_sys::dummy_cuda_dependency;

    use sbert::Tokenizer as TraitTokenizer;
    use sbert::{DistilRobertaForSequenceClassificationRT, RustTokenizersSentencePiece};

    const BATCH_SIZE: usize = 64;

    #[test]
    fn test_rust_tokenizers_sentencepiece() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distilroberta_toxicity");

        let tok = RustTokenizersSentencePiece::new(home).unwrap();

        let mut texts = Vec::new();
        texts.push(String::from("Omg you are so bad at this game!"));
        texts.push(String::from(
            "wow it's a nice day todayyyyyyyyyyyyyyyyyyyy!!!",
        ));

        let toks = tok.pre_tokenize(&texts);
        println!("Pretokenize {:?}", toks);

        assert_eq!(
            toks[0],
            vec!["O", "mg", "Ġyou", "Ġare", "Ġso", "Ġbad", "Ġat", "Ġthis", "Ġgame", "!"]
        );
    }

    #[test]
    fn test_distilroberta_for_classification_rust_tokenizers_sentencepiece() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack

        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distilroberta_toxicity");

        println!("Loading distilroberta ...");
        let before = Instant::now();
        let sbert_model = DistilRobertaForSequenceClassificationRT::new(home).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("Omg you are so bad at this game!"));
        texts.push(String::from(
            "wow it's a nice day todayyyyyyyyyyyyyyyyyyyy!!!",
        ));

        println!("Encoding {} sentences...", texts.len());
        let before = Instant::now();
        let output = &sbert_model.forward(&texts, BATCH_SIZE).unwrap();
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis() / 10);
        println!("Vec: {:?}", output);

        let v = output[0][..2]
            .iter()
            .map(|f| (f * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>();
        let v2 = output[1][..2]
            .iter()
            .map(|f| (f * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>();

        assert_eq!(v, [-1.057, 1.993]);
        assert_eq!(v2, [3.055, -2.810]);
    }
}
