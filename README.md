# Rust SBert

Rust port of [sentence-transformers](https://github.com/UKPLab/sentence-transformers) using [rust-bert](https://github.com/guillaume-be/rust-bert), [tch-rs](https://github.com/LaurentMazare/tch-rs) and [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers).

Might consider replacing [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers) with Hugging Face's [tokenizers](https://github.com/huggingface/tokenizers/tree/master/tokenizers).

## Usage

The API is made to be very easy to use and enables you to create a sentence embedding very simply;

Load SBert model with weights:

```Rust
let mut home: PathBuf = env::current_dir().unwrap();
home.push("path-to-model");

let sbert_model = SBert::new(home.to_str().unwrap());
```

Encode a sentence and get its sentence embedding:

```Rust
let texts = ["You can encode",
             "As many sentences",
             "As you want",
             "Enjoy ;)"];

let output = sbert_model.encode(texts.to_vec()).unwrap();
```

Then you can use the `output` sentence embedding in any application you want. 
