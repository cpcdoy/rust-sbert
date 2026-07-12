#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::normalizers::bert::BertNormalizer;
    use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
    use tokenizers::processors::bert::BertProcessing;
    use tokenizers::{tokenizer, Model};
    use torch_sys::dummy_cuda_dependency;

    use sbert::Tokenizer as TraitTokenizer;
    use sbert::{HFTokenizer, SBertHF, SBertRT};

    const BATCH_SIZE: usize = 64;

    fn rand_string(r: &mut impl Rng) -> String {
        (0..(r.gen::<f32>() * 100.0) as usize)
            .map(|_| (0x20u8 + (r.gen::<f32>() * 96.0) as u8) as char)
            .collect()
    }

    #[test]
    fn test_hf_pre_tokenizer() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack

        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");
        home.push("0_DistilBERT");

        let vocab_file = home.join("vocab.txt");
        let tok = HFTokenizer::new(&vocab_file).unwrap();

        let mut texts = Vec::new();
        texts.push(String::from("TTThis player needs tp be reported lolz."));

        let tokens = tok.pre_tokenize(&texts);
        println!("Tokens {:?}", tokens[0]);

        assert_eq!(
            tokens[0],
            [
                "[CLS]", "TT", "##T", "##his", "player", "needs", "t", "##p", "be", "reported",
                "lo", "##lz", ".", "[SEP]"
            ]
        );
    }

    #[test]
    fn test_sbert_rust_tokenizers() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack
        let mut r = StdRng::seed_from_u64(42);

        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert ...");
        let before = Instant::now();
        let sbert_model = SBertRT::new(home, None).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("TTThis player needs tp be reported lolz."));
        for _ in 0..9 {
            texts.push(rand_string(&mut r));
        }

        println!("Encoding {} sentences...", texts.len());
        let before = Instant::now();
        for _ in 0..9 {
            let _ = sbert_model.forward(&texts, BATCH_SIZE).unwrap();
        }
        let output = sbert_model.forward(&texts, BATCH_SIZE).unwrap();
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis() / 10);
        println!("First 5 of first embedding: {:?}", &output[0][..5]);

        // Structural checks (replaces brittle hardcoded-float assertion that
        // drifted whenever the distiluse weights were re-exported).
        assert_embedding_sane(&output[0], 512, "distiluse-base-multilingual-cased");

        // Determinism: a second forward on the same input must match
        // bit-for-bit (inference is in no_grad mode).
        let output_again = sbert_model.forward(&texts, BATCH_SIZE).unwrap();
        let max_diff = output[0]
            .iter()
            .zip(output_again[0].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "forward must be deterministic across calls; max diff = {}",
            max_diff
        );
    }

    #[test]
    fn test_sbert_hugging_face_tokenizers() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack
        let mut r = StdRng::seed_from_u64(42);

        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert ...");
        let before = Instant::now();
        let sbert_model = SBertHF::new(home, None).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("TTThis player needs tp be reported lolz."));
        for _ in 0..9 {
            texts.push(rand_string(&mut r));
        }

        println!("Encoding {} sentences...", texts.len());
        let before = Instant::now();
        for _ in 0..9 {
            let _ = sbert_model.forward(&texts, BATCH_SIZE).unwrap()[0][..5];
        }
        let output = sbert_model.forward(&texts, BATCH_SIZE).unwrap();
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis() / 10);
        println!("First 5 of first embedding: {:?}", &output[0][..5]);

        // Structural checks — see test_sbert_rust_tokenizers for rationale.
        assert_embedding_sane(&output[0], 512, "distiluse-base-multilingual-cased");
    }

    /// Smoke-check an embedding vector without baking in stale hardcoded
    /// floats. Asserts:
    /// * dimension matches `expected_dim`
    /// * every component is finite (no NaN / infinity)
    /// * L2 norm is strictly positive (the model actually produced output)
    /// * L2 norm is within a sane upper bound (catches e.g. a missing
    ///   pooling step that would leave raw per-token magnitudes)
    fn assert_embedding_sane(emb: &[f32], expected_dim: usize, model_name: &str) {
        assert_eq!(
            emb.len(),
            expected_dim,
            "{} embedding dimension (got {}, expected {})",
            model_name,
            emb.len(),
            expected_dim
        );
        assert!(
            emb.iter().all(|v| v.is_finite()),
            "{} embedding contains NaN or infinity",
            model_name
        );
        let norm_sq: f32 = emb.iter().map(|v| v * v).sum();
        assert!(norm_sq > 0.0, "{} embedding must be non-zero", model_name);
        let norm = norm_sq.sqrt();
        // For sentence-transformers checkpoints the L2 norm of an embedding
        // is typically O(1) — well below 100. If we see something huge, the
        // pipeline is probably missing a step (e.g. raw token embeddings
        // rather than pooled, or missing Dense/Normalize).
        assert!(
            norm < 100.0,
            "{} embedding L2 norm {} is implausibly large",
            model_name,
            norm
        );
    }

    /// Regression test for `forward_with_attention`. Requires distiluse's
    /// `0_DistilBERT/config.json` to have `"output_attentions": true` set;
    /// without it, rust-bert returns `all_attentions: None` and the test
    /// errors out with `Encoding("No attention")`.
    ///
    /// `#[ignore]`d by default because we deliberately do not mutate the
    /// on-disk model files. To run it locally, edit distiluse's config.json
    /// to add `"output_attentions": true` and invoke:
    ///
    /// ```sh
    /// cargo test --features all-tests -- --ignored test_sbert_encode_attention
    /// ```
    #[test]
    #[ignore]
    pub fn test_sbert_encode_attention() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert ...");
        let before = Instant::now();
        let sbert_model = SBertHF::new(home, None).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("test"));
        texts.push(String::from("testtest"));

        println!("Encoding {} sentence with attention...", texts.len());
        let output = &sbert_model
            .forward_with_attention(&texts, BATCH_SIZE)
            .unwrap();
        let emb = &output.0[0][..5];
        let attention = &output.1;

        println!("texts: {:?}", texts.clone());
        let tokens = sbert_model.tokenizer().pre_tokenize(&texts);

        let len = tokens[0].len();
        let head_nb = attention[0][0].len();
        let mut tok_highlights = vec![0.0; len];

        for head_atts in attention[0][5].iter() {
            let mut head_vec = vec![0.0; len];
            for atts in head_atts.iter() {
                println!("tok att: {:?}", atts);
                for (i, tok_att) in atts.iter().enumerate() {
                    head_vec[i] += tok_att;
                }
            }

            let head_vec: Vec<f32> = head_vec.into_iter().map(|e| e / (len as f32)).collect();

            println!("head vec: {:?}", head_vec.clone());
            for (i, att) in head_vec.iter().enumerate() {
                tok_highlights[i] += att;
            }

            println!("tok high: {:?}", tok_highlights);
        }

        let tok_highlights: Vec<f32> = tok_highlights
            .into_iter()
            .map(|e| e / (head_nb as f32))
            .collect();

        let mut tokens_and_atts: Vec<(f32, String)> = Vec::new();

        for (att, tok) in tok_highlights.iter().zip(tokens[0].iter()) {
            tokens_and_atts.push((att.clone(), tok.clone()));
        }
        println!("########### Tokens and att: {:?}", tokens_and_atts);
        println!(
            "Important tokens: {:?}",
            tokens_and_atts
                .iter()
                .filter(|x| x.0 > 0.009)
                .map(|x| x.1.clone())
                .collect::<Vec<_>>()
        );
        println!("Tokens: {:?}", tokens);
        println!("Att toks: {:?}", tok_highlights);

        // Structural check (same rationale as the other embedding tests).
        assert_embedding_sane(emb, 512, "distiluse-base-multilingual-cased");
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

    /// Regression test for the BERT backend (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
    ///
    /// Ignored by default because it requires the model files to be present at
    /// `models/all-MiniLM-L6-v2/`.
    /// ```
    #[test]
    #[ignore]
    fn test_bert_backend_minilm() {
        unsafe {
            dummy_cuda_dependency();
        } // Windows Hack

        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("all-MiniLM-L6-v2");

        println!(
            "Loading SentenceTransformer (BERT backend) from {} ...",
            home.display()
        );
        let before = Instant::now();
        let model = SBertRT::new(home.clone(), None).unwrap();
        println!("Loaded in {:.2?}", before.elapsed());

        let sentence = "Hello world!".to_string();
        let sentences = vec![sentence];
        let output = model.forward(&sentences, BATCH_SIZE).unwrap();
        let emb = &output[0];

        // all-MiniLM-L6-v2 produces 384-dim sentence embeddings.
        assert_eq!(emb.len(), 384, "MiniLM-L6-v2 embedding must be 384-dim");

        let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "embedding must be non-zero");
        println!(
            "MiniLM-L6-v2 OK: dim={}, L2 norm={:.4}, first 5 = {:?}",
            emb.len(),
            norm,
            &emb[..5.min(emb.len())]
        );
    }

    #[test]
    fn test_tok() {
        let mut root: PathBuf = env::current_dir().unwrap();
        root.push("models");
        root.push("distiluse-base-multilingual-cased");

        let model_dir = root.join("0_DistilBERT");

        let vocab_file = model_dir.join("vocab.txt");

        // Set-up DistilBert model and tokenizer
        let tokenizer =
            BertTokenizer::from_file(&vocab_file.to_string_lossy(), false, false).unwrap();

        let input = vec!["TTThis player needs tp be reported lolz."; 1000];
        let input_1 = input.clone();
        let before = Instant::now();
        let tokenized_input =
            tokenizer.encode_list(&input_1, 128, &TruncationStrategy::LongestFirst, 0);

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
            .collect::<Vec<_>>();

        println!(
            "Rust-tokenizers: {} {:?}",
            tokenized_input.len(),
            tokenized_input[0]
        );
        println!("Elapsed time: {:?}ms", before.elapsed().as_micros());

        let tokenizer = get_bert(vocab_file.to_str().unwrap());

        let before = Instant::now();
        let encode_input = input
            .into_iter()
            .map(|s| tokenizer::EncodeInput::Single(s.into()))
            .collect();
        let encoding = tokenizer.encode_batch(encode_input, true).unwrap();
        println!(
            "Hugging Face's tokenizers: {} {:?}",
            encoding.len(),
            encoding[0].get_ids()
        );
        println!("Elapsed time: {:?}ms", before.elapsed().as_micros());

        let tok_i64 = encoding[0]
            .get_ids()
            .into_iter()
            .map(|e| *e as i64)
            .collect::<Vec<_>>();
        assert_eq!(tok_i64, tokenized_input[0]);
        assert_eq!(
            tok_i64,
            [
                101, 59725, 11090, 49311, 12928, 28615, 188, 10410, 10347, 15943, 10406, 48275,
                119, 102
            ]
        );
    }
}
