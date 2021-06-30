# Rust SBert [![Latest Version]][crates.io] [![Latest Doc]][docs.rs] ![Build Status]

[latest version]: https://img.shields.io/crates/v/sbert.svg
[crates.io]: https://crates.io/crates/sbert
[latest doc]: https://docs.rs/sbert/badge.svg
[docs.rs]: https://docs.rs/sbert
[build status]: https://travis-ci.com/cpcdoy/rust-sbert.svg?branch=master

Rust port of [sentence-transformers][] using [rust-bert][] and [tch-rs][].

Supports both [rust-tokenizers][] and Hugging Face's [tokenizers][].

## Supported models

- **distiluse-base-multilingual-cased**: Supported languages: Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish. Performance on the extended STS2017: 80.1

- **DistilRoBERTa**-based classifiers

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

Now, you can encode your sentences:

```Rust
let texts = ["You can encode",
             "As many sentences",
             "As you want",
             "Enjoy ;)"];

let batch_size = 64;

let output = sbert_model.forward(texts.to_vec(), batch_size).unwrap();
```

The parameter `batch_size` can be left to `None` to let the model use its default value.

Then you can use the `output` sentence embedding in any application you want.

### Convert models from Python to Rust

Firstly, get a model provided by UKPLabs (all models are [here][models]):

```Bash
mkdir -p models/distiluse-base-multilingual-cased

wget -P models https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distiluse-base-multilingual-cased.zip

unzip models/distiluse-base-multilingual-cased.zip -d models/distiluse-base-multilingual-cased
```

Then, you need to convert the model in a suitable format (requires [pytorch][]):

```Bash
python utils/prepare_distilbert.py models/distiluse-base-multilingual-cased
```

A dockerized environment is also available for running the conversion script:

```Bash
docker build -t tch-converter -f utils/Dockerfile .

docker run \
  -v $(pwd)/models/distiluse-base-multilingual-cased:/model \
  tch-converter:latest \
  python prepare_distilbert.py /model
```

Finally, set `"output_attentions": true` in `distiluse-base-multilingual-cased/0_distilbert/config.json`.

[sentence-transformers]: https://github.com/UKPLab/sentence-transformers
[rust-bert]: https://github.com/guillaume-be/rust-bert
[tch-rs]: https://github.com/LaurentMazare/tch-rs
[rust-tokenizers]: https://github.com/guillaume-be/rust-tokenizers
[tokenizers]: https://github.com/huggingface/tokenizers/tree/master/tokenizers
[models]: https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
[pytorch]: https://pytorch.org/get-started/locally
