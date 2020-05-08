#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;
    use std::time::Instant;

    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::normalizers::bert::BertNormalizer;
    use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
    use tokenizers::processors::bert::BertProcessing;
    use tokenizers::tokenizer;

    use rust_tokenizers::bert_tokenizer::BertTokenizer;
    use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{
        Tokenizer, TruncationStrategy,
    };

    use sbert_rs::{SBertHF, SBertRT};

    #[test]
    fn test_sbert_hugging_face_tokenizers() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert_rs ...");
        let before = Instant::now();
        let sbert_model = SBertHF::new(home).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let texts = vec!["TTThis player needs tp be reported lolz."; 100];

        println!("Encoding {:?}...", texts[0]);
        let before = Instant::now();
        let output = sbert_model.encode(&texts, None).unwrap();
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis());

        let r = output.get(0).slice(0, 0, 5, 1);
        r.print();

        let v = (r / 0.01)
            .iter::<f64>()
            .unwrap()
            .map(|f| (f * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>();
        assert_eq!(v, [-2.2717, -0.6020, 5.5196, 1.8546, -7.5385]);
    }

    #[test]
    fn test_sbert_rust_tokenizers() {
        let mut home: PathBuf = env::current_dir().unwrap();
        home.push("models");
        home.push("distiluse-base-multilingual-cased");

        println!("Loading sbert_rs ...");
        let before = Instant::now();
        let sbert_model = SBertRT::new(home).unwrap();
        println!("Elapsed time: {:.2?}", before.elapsed());

        let texts = vec!["TTThis player needs tp be reported lolz."; 100];

        println!("Encoding {:?}...", texts[0]);
        let before = Instant::now();
        let output = sbert_model.encode(&texts, None).unwrap();
        println!("Elapsed time: {:?}ms", before.elapsed().as_millis());

        let r = output.get(0).slice(0, 0, 5, 1);
        r.print();

        let v = (r / 0.01)
            .iter::<f64>()
            .unwrap()
            .map(|f| (f * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>();
        assert_eq!(v, [-2.2717, -0.6020, 5.5196, 1.8546, -7.5385]);
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

        let tokenizer = BertTokenizer::from_file(&vocab_file.to_string_lossy(), false);

        //let input = ["TTThis player needs tp be reported lolz."].to_vec();
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
