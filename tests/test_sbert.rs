#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rust_tokenizers::bert_tokenizer::BertTokenizer;
    use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{
        Tokenizer, TruncationStrategy,
    };
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::normalizers::bert::BertNormalizer;
    use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
    use tokenizers::processors::bert::BertProcessing;
    use tokenizers::tokenizer;
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
        let model_path = home.clone();

        home.push("0_DistilBERT");

        let vocab_file = home.join("vocab.txt");
        let tok = HFTokenizer::new(&vocab_file).unwrap();

        let mut texts = Vec::new();
        texts.push(String::from("TTThis player needs tp be reported lolz."));

        let tokens = tok.pre_tokenize(&texts);
        println!("Tokens {:?}", tokens[0]);

        let sbert_model = SBertHF::new(model_path).unwrap();
        let output = sbert_model
            .encode_with_attention(&texts, BATCH_SIZE)
            .unwrap();

        println!("att {:?}", output.1[0][0]);
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
        let sbert_model = SBertRT::new(home).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("TTThis player needs tp be reported lolz."));
        for _ in 0..9 {
            texts.push(rand_string(&mut r));
        }

        println!("Encoding {} sentences...", texts.len());
        let before = Instant::now();
        for _ in 0..9 {
            &sbert_model.encode(&texts, BATCH_SIZE).unwrap();
        }
        let output = &sbert_model.encode(&texts, BATCH_SIZE).unwrap()[0][..5];
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis() / 10);
        println!("Vec: {:?}", output);

        let v = output
            .iter()
            .map(|f| (f * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>();
        assert_eq!(v, [-0.0227, -0.006, 0.0552, 0.0185, -0.0754]);
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
        let sbert_model = SBertHF::new(home).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("TTThis player needs tp be reported lolz."));
        for _ in 0..9 {
            texts.push(rand_string(&mut r));
        }

        println!("Encoding {} sentences...", texts.len());
        let before = Instant::now();
        for _ in 0..9 {
            &sbert_model.encode(&texts, BATCH_SIZE).unwrap()[0][..5];
        }
        let output = &sbert_model.encode(&texts, BATCH_SIZE).unwrap()[0][..5];
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis() / 10);
        println!("Vec: {:?}", output);

        let v = output
            .iter()
            .map(|f| (f * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>();
        assert_eq!(v, [-0.0227, -0.006, 0.0552, 0.0185, -0.0754]);
    }

    #[test]
    pub fn test_sbert_encode_attention() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert ...");
        let before = Instant::now();
        let sbert_model = SBertHF::new(home).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let mut texts = Vec::new();
        texts.push(String::from("test"));
        texts.push(String::from("testtest"));

        println!("Encoding {} sentence with attention...", texts.len());
        let output = &sbert_model
            .encode_with_attention(&texts, BATCH_SIZE)
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

        let v = emb
            .iter()
            .map(|f| (f * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>();
        assert_eq!(v, [0.033, 0.0469, -0.0579, 0.0181, -0.0693]);
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
            tokenizer.encode_list(input_1, 128, &TruncationStrategy::LongestFirst, 0);

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
