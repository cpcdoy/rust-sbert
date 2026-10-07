# Rust SBert [![Latest Version]][crates.io] [![Latest Doc]][docs.rs] [![Build Status]][ci]

[Latest Version]: https://img.shields.io/crates/v/sbert.svg
[crates.io]: https://crates.io/crates/sbert
[Latest Doc]: https://docs.rs/sbert/badge.svg
[docs.rs]: https://docs.rs/sbert
[Build Status]: https://github.com/Luxbit/rust-sbert/actions/workflows/ci.yml/badge.svg
[ci]: https://github.com/Luxbit/rust-sbert/actions/workflows/ci.yml

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

rust-bert publishes pre-converted `rust_model.ot` weights for a handful of
well-known checkpoints (those its own sentence-embeddings pipeline lists —
`all-MiniLM-L6-v2` is one of them), so no Python toolchain is needed for
those. For anything else you must convert the weights yourself (see the next
section); note that hub checkpoints shipping `2_Dense/model.safetensors` are
not covered by `utils/prepare_models.py` either. The HF `modules.json` for
`all-MiniLM-L6-v2` declares the Transformer at `path: ""` (files at the
model root):

```Bash
mkdir -p models/all-MiniLM-L6-v2/1_Pooling
HF=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main

curl -L -o models/all-MiniLM-L6-v2/model.ot                $HF/rust_model.ot
curl -L -o models/all-MiniLM-L6-v2/config.json             $HF/config.json
curl -L -o models/all-MiniLM-L6-v2/vocab.txt               $HF/vocab.txt
curl -L -o models/all-MiniLM-L6-v2/tokenizer_config.json   $HF/tokenizer_config.json
curl -L -o models/all-MiniLM-L6-v2/sentence_bert_config.json $HF/sentence_bert_config.json
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

## ONNX backend (optional)

The transformer stage can run through ONNX Runtime instead of libtorch:
Pooling/Dense/Normalize and the tokenizers stay exactly the same, and the
libtorch (TorchScript) path remains the default — MPS on Apple Silicon is
only available through it.

Enable the feature and load with `new_onnx`:

```Rust
let sbert_model = SBertRT::new_onnx("models/all-MiniLM-L6-v2", None)?;
let output = sbert_model.forward(&texts, 64)?;
// or an explicit graph path:
// SBertRT::new_onnx_with_file("models/all-MiniLM-L6-v2", "model.onnx", None)?
```

Export a backbone graph with `utils/prepare_onnx.py` (requires python with
`optimum==1.25.x`, `transformers<5`, `torch`, `onnx`, `onnxscript` — see the
script docstring; it handles opset/IR-version clamping for ONNX Runtime
1.15):

```Bash
# PREFERRED: export from the checkpoint's own model.ot — the ONNX weights
# then match the TorchScript path bit-for-bit (what tests/test_onnx.rs checks):
python utils/prepare_onnx.py models/distiluse-base-multilingual-cased/0_DistilBERT \
                             models/distiluse-base-multilingual-cased/0_DistilBERT

# or from a hub repo (writes <out_dir>/model.onnx):
python utils/prepare_onnx.py sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
```

Runtime notes:

- onnxruntime is loaded at runtime — ort never downloads it. Set
  `ORT_DYLIB_PATH` to a `libonnxruntime.dylib`/`.so`/`.dll` (onnxruntime
  1.15.1 pairs with ort 1.15) or install it on the system loader path —
  the same manual workflow as `LIBTORCH` for tch.
- **ort 1.15.x is yanked on crates.io**, and `Cargo.lock` does not ship
  with libraries — so dependents enabling `onnx` need a patch source until
  rust-bert moves to ort 2.x:

  ```toml
  [patch.crates-io]
  ort = { git = "https://github.com/pykeio/ort", tag = "v1.15.3" }
  ```

- **CUDA execution provider**: the `onnx` feature enables ort's `cuda`
  feature, so CUDA inference needs CUDA-11-era user-space libraries at
  runtime (`libcudart.so.11.0`, cuDNN 8.9, cuBLAS 11.x — the
  `nvidia-*-cu11` pip wheels provide them). Without them, ONNX Runtime
  silently falls back to the CPU provider.
- `device` semantics differ from the TorchScript path: it selects the ORT
  **execution provider** only (`Device::Cuda(i)` → CUDA, anything else →
  CPU). All tch-side tensors stay on CPU, so a CPU-only libtorch build is
  enough for ONNX deployments. There is no Metal provider — use the default
  backend for MPS.
- Limits: `forward_with_attention` errors (exports carry no attention
  outputs); graphs must expose a `last_hidden_state` output (optimum's
  `feature-extraction` export of the bare backbone does — the
  sentence_transformers-style export with `token_embeddings`/
  `sentence_embedding` outputs is not supported); fp16 graphs are not
  supported.

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
  `modules::transformer::load`. Note this alone is not enough: tokenization
  is hard-coded to `vocab.txt` + WordPiece in `SentenceTransformer::new`, so
  a backbone with a different tokenizer (RoBERTa's SentencePiece/BPE) also
  needs a new `tokenizers::Tokenizer` implementation and corresponding
  tokenizer-file resolution.
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
