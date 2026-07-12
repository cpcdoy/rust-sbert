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
        let sbert_model = DistilRobertaForSequenceClassificationRT::new(home, None).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("Omg you are so bad at this game!"));
        texts.push(String::from(
            "wow it's a nice day todayyyyyyyyyyyyyyyyyyyy!!!",
        ));
        texts.push(String::from("lollll!!!"));

        println!("Encoding {} sentences...", texts.len());
        let before = Instant::now();
        let output = &sbert_model.forward(&texts, BATCH_SIZE).unwrap();
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis() / 10);
        println!("Vec: {:?}", output);

        // Structural checks (replaces stale hardcoded-float assertions that
        // were written for a 2-class variant of this model; the on-disk
        // `distilroberta_toxicity` checkpoint is a 4-class classifier — see
        // `id2label` in its `config.json`).
        //
        // Per-sentence row is a softmax over class logits: every component
        // finite and in [0, 1], summing to ~1.0.
        const EXPECTED_CLASSES: usize = 4;
        for (i, row) in output.iter().enumerate() {
            assert_eq!(
                row.len(),
                EXPECTED_CLASSES,
                "sentence {} produced {} class probabilities (expected {})",
                i,
                row.len(),
                EXPECTED_CLASSES
            );
            assert!(
                row.iter().all(|v| v.is_finite()),
                "sentence {} contains NaN/inf in its probabilities",
                i
            );
            assert!(
                row.iter().all(|v| *v >= 0.0 && *v <= 1.0),
                "sentence {} has a probability outside [0, 1]: {:?}",
                i,
                row
            );
            let sum: f32 = row.iter().copied().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "sentence {} probabilities sum to {} (expected 1.0)",
                i,
                sum
            );
        }
    }
}
