# Rust SBert [![Latest Version]][crates.io] [![Latest Doc]][docs.rs] ![Build Status]

[Latest Version]: https://img.shields.io/crates/v/sbert.svg
[crates.io]: https://crates.io/crates/sbert
[Latest Doc]: https://docs.rs/sbert/badge.svg
[docs.rs]: https://docs.rs/sbert
[Build Status]: https://travis-ci.com/cpcdoy/rust-sbert.svg?branch=master

Rust port of [sentence-transformers][] using [rust-bert][] and [tch-rs][].

The model pipeline is driven by a checkpoint's `modules.json` manifest: the
library reads it, picks the transformer backend from `<0_*/config.json>`'s
`model_type`, and composes the post-transformer modules (`Pooling`,
optional `Dense`, optional `Normalize`) in declared order. Adding a new
transformer backbone (RoBERTa, MPNet, …) is one trait impl + one match arm
in the factory.

Supports both [rust-tokenizers][] and Hugging Face's [tokenizers][].

## Supported models

Any sentence-transformers checkpoint whose `modules.json` references one of
the implemented backends and module types loads out of the box:

- **Transformer backends**: `bert` (e.g. `all-MiniLM-L6-v2`,
  `paraphrase-MiniLM-L6-v2`, `bert-base-nli-mean-tokens`),
  `distilbert` (e.g. `distiluse-base-multilingual-cased`).
- **Post-transformer modules**: `Pooling` (mean), `Dense` (linear+tanh,
  optional), `Normalize` (L2, optional).
- **DistilRoBERTa**-based sequence classifiers (unchanged from prior
  releases — see `src/models/distilroberta.rs`).

For `distiluse-base-multilingual-cased` the supported languages are: Arabic,
Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese,
Russian, Spanish, Turkish. Performance on the extended STS2017: 80.1.

## Usage

### Example

The API is made to be very easy to use and enables you to create quality
multilingual sentence embeddings in a straightforward way.

Load a model by pointing at its directory (must contain `modules.json` and
the subdirectories/paths it references):

```Rust
let mut home: PathBuf = env::current_dir().unwrap();
home.push("path-to-model");
```

You can use different versions of the models that use different tokenizers:

```Rust
// To use Hugging Face tokenizer
let sbert_model = SBertHF::new(home, None).unwrap();

// To use Rust-tokenizers
let sbert_model = SBertRT::new(home, None).unwrap();
```

`SBert<T>` is a backward-compat alias for the underlying
`SentenceTransformer<T>` struct — both names refer to the same type, so
existing call sites keep working unchanged.

Now, you can encode your sentences:

```Rust
let texts = vec![
    "You can encode".to_string(),
    "As many sentences".to_string(),
    "As you want".to_string(),
    "Enjoy ;)".to_string(),
];

let batch_size = 64;

let output = sbert_model.forward(&texts, batch_size).unwrap();
```

The parameter `batch_size` can be left to `None` to let the model use its
default value.

Then you can use the `output` sentence embedding in any application you want.

### Loading a `bert`-backbone checkpoint (e.g. `all-MiniLM-L6-v2`)

For most modern sentence-transformers BERT checkpoints rust-bert already
publishes a pre-converted `rust_model.ot`, so no Python toolchain is needed.
The HF `modules.json` for `all-MiniLM-L6-v2` declares the Transformer at
`path: ""` (files at the model root):

```Bash
mkdir -p models/all-MiniLM-L6-v2/1_Pooling
HF=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main

curl -L -o models/all-MiniLM-L6-v2/model.ot                $HF/rust_model.ot
curl -L -o models/all-MiniLM-L6-v2/config.json             $HF/config.json
curl -L -o models/all-MiniLM-L6-v2/vocab.txt               $HF/vocab.txt
curl -L -o models/all-MiniLM-L6-v2/tokenizer_config.json   $HF/tokenizer_config.json
curl -L -o models/all-MiniLM-L6-v2/special_tokens_map.json $HF/special_tokens_map.json
curl -L -o models/all-MiniLM-L6-v2/1_Pooling/config.json   $HF/1_Pooling/config.json
curl -L -o models/all-MiniLM-L6-v2/modules.json            $HF/modules.json
```

Then load as usual: `SBertRT::new("models/all-MiniLM-L6-v2", None)`.

### Convert `distilbert` models from Python to Rust

For checkpoints that don't ship a pre-converted `.ot` (e.g. the original
`distiluse-base-multilingual-cased` export), convert the PyTorch weights
manually.

Firstly, get a model provided by UKPLabs (all models are [here][models]):

```Bash
mkdir -p models/distiluse-base-multilingual-cased

wget -P models https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/distiluse-base-multilingual-cased.zip

unzip models/distiluse-base-multilingual-cased.zip -d models/distiluse-base-multilingual-cased
```

Then, you need to convert the model in a suitable format (requires [pytorch][]):

``` Bash
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

Finally, set `"output_attentions": true` in
`distiluse-base-multilingual-cased/0_distilbert/config.json`.

## Pipeline architecture

`SentenceTransformer<T>` holds:

- `transformer: Box<dyn TransformerBackend>` — dispatches on
  `config.json`'s `model_type` field. Implemented: `DistilBertBackend`,
  `BertBackend`.
- `post: Vec<Box<dyn Module>>` — built from `modules.json` in declared
  order. The forward pass threads a `Features` enum (`Token { .. }` →
  `Sentence { .. }`) through each module; `Pooling` does the token→sentence
  reduction.

To extend:

- **New backbone** (e.g. RoBERTa): add a `RobertaBackend` struct + impl
  `TransformerBackend` for it + add a match arm in
  `modules::transformer::load`.
- **New post-transformer module** (e.g. `WeightedLayerPooling`): add a
  module struct + impl `Module` for it + add a match arm on `short_type`
  in `SentenceTransformer::new`.

See `src/modules/` for the building blocks.

[sentence-transformers]: https://github.com/UKPLab/sentence-transformers
[rust-bert]: https://github.com/guillaume-be/rust-bert
[tch-rs]: https://github.com/LaurentMazare/tch-rs
[rust-tokenizers]: https://github.com/guillaume-be/rust-tokenizers
[tokenizers]: https://github.com/huggingface/tokenizers/tree/master/tokenizers
[models]: https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
[pytorch]: https://pytorch.org/get-started/locally
