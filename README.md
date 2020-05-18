# Rust SBert

Rust port of [sentence-transformers](https://github.com/UKPLab/sentence-transformers) using [rust-bert](https://github.com/guillaume-be/rust-bert), [tch-rs](https://github.com/LaurentMazare/tch-rs) and [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers).

Might consider replacing [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers) with Hugging Face's [tokenizers](https://github.com/huggingface/tokenizers/tree/master/tokenizers).

## Supported models

### Multilingual Models

- **distiluse-base-multilingual-cased**: Supported languages: Arabic, Chinese, Dutch, English, French, German,  Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. Performance on the extended STS2017: 80.1

## Usage

### Example

The API is made to be very easy to use and enables you to create quality multilingual sentence embeddings in a straightforward way.

Load SBert model with weights by specifying the directory of the model:

```Rust
let mut home: PathBuf = env::current_dir().unwrap();
home.push("path-to-model");
```

You can use different versions of the models that use different tokenizers:

```Rust
// To use Hugging Face tokenizer
let sbert_model = SBertHF::new(home.to_str().unwrap());

// To use Rust-tokenizers
let sbert_model = SBertRT::new(home.to_str().unwrap());
```

It is also possible to use a threaded version of the model called `SafeSbert`:

```Rust
// To use Hugging Face tokenizer
let sbert_model = SafeSBertHF::new(home.to_str().unwrap());

// To use Rust-tokenizers
let sbert_model = SafeSBertRT::new(home.to_str().unwrap());
```

Now, you can encode your sentences:

```Rust
let texts = ["You can encode",
             "As many sentences",
             "As you want",
             "Enjoy ;)"];

let batch_size = 64;

let output = sbert_model.encode(texts.to_vec(), batch_size).unwrap();
```

The parameter `batch_size` can be left to `None` to let the model use its default value.

Then you can use the `output` sentence embedding in any application you want. 

### Convert models from Python to Rust

To be able to use the models provided [here](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/) by UKPLabs, you need to run this script to convert the model in a suitable format:

```Bash
cd model-path/
python3 utils/prepare_distilbert.py
```
