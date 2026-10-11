# Rust SBert [![Latest Version]][crates.io] [![Latest Doc]][docs.rs] [![Build Status]][ci]

[Latest Version]: https://img.shields.io/crates/v/sbert.svg
[crates.io]: https://crates.io/crates/sbert
[Latest Doc]: https://docs.rs/sbert/badge.svg
[docs.rs]: https://docs.rs/sbert
[Build Status]: https://github.com/Luxbit/rust-sbert/actions/workflows/ci.yml/badge.svg
[ci]: https://github.com/Luxbit/rust-sbert/actions/workflows/ci.yml

Rust port of [sentence-transformers][] with two interchangeable inference
backends selected at compile time via cargo features: **libtorch**
([tch-rs][], the default) or **ONNX Runtime** (libtorch-free, via the
[Luxbit fork][luxbit-rust-bert] of [rust-bert][] where `tch` is optional and
the ONNX stack is ndarray-based). The features are mutually exclusive by
design, one tensor library per build:

|  | Cargo | Weights | Runtime needs |
|---|---|---|---|
| libtorch (default) | `sbert = "0.7"` (feature `torch`) | `model.ot` VarStore archives | libtorch (torch-sys can download it; MPS/Vulkan supported) |
| ONNX Runtime, no libtorch | `sbert = { version = "0.7", default-features = false, features = ["onnx"] }` | `model.onnx` + `2_Dense/weights.safetensors` | onnxruntime via `ORT_DYLIB_PATH` |


The model pipeline is driven by a checkpoint's `modules.json` manifest: the
library reads it, runs the transformer stage through the selected backend,
and composes the post-transformer modules (`Pooling`, optional `Dense`,
optional `Normalize`) in declared order: tch tensors under `torch`, pure
`ndarray` under `onnx`.

Supports both [rust-tokenizers][] and Hugging Face's [tokenizers][].

## Requirements

- `torch` (default): point `LIBTORCH` at a local
  extraction, as usual for tch crates. With no `LIBTORCH*` env
  vars set, torch-sys downloads PyTorch itself (the `torch`
  feature enables the [rust-bert fork](https://github.com/Luxbit/rust-bert)'s
  `libtorch-download` flag).
- `onnx`: an **onnxruntime** shared library (>= 1.23) at runtime, located
  via `ORT_DYLIB_PATH`, the manual setup documented by the
  [rust-bert fork](https://github.com/Luxbit/rust-bert).

  1. Download a release for your platform from the
     [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
     (e.g. `onnxruntime-osx-arm64-1.23.2.tgz`).
  2. Extract it and point `ORT_DYLIB_PATH` at the library:

     ```bash
     export ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.dylib # .so / .dll
     ```

## Supported models

Any sentence-transformers checkpoint whose `modules.json` references the
implemented module types loads out of the box on either backend:

- **Transformer backbones** (`model_type` in `config.json`): `bert`
  (e.g. `all-MiniLM-L6-v2`, `paraphrase-MiniLM-L6-v2`,
  `bert-base-nli-mean-tokens`), `distilbert`
  (e.g. `distiluse-base-multilingual-cased`).
- **Post-transformer modules**: `Pooling` (cls / mean / max / mean_sqrt_len,
  concatenated in the Python order), `Dense` (linear + tanh/relu/gelu,
  optional), `Normalize` (L2, optional).
- torch only: `DistilRoBERTa`-based sequence classifiers, and
  `forward_with_attention` (bare-backbone ONNX exports carry no attention
  outputs).

For `distiluse-base-multilingual-cased` the supported languages are: Arabic,
Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese,
Russian, Spanish, Turkish. Performance on the extended STS2017: 80.1.

## Usage

Same API under both features:

```Rust
// To use Hugging Face tokenizer
let sbert_model = SBertHF::new("path-to-model", None).unwrap();

// To use Rust-tokenizers
let sbert_model = SBertRT::new("path-to-model", None).unwrap();

let texts = vec![
    "You can encode".to_string(),
    "As many sentences".to_string(),
    "As you want".to_string(),
    "Enjoy ;)".to_string(),
];

let output = sbert_model.forward(&texts, 64).unwrap(); // batch size, or None
```

`SBert<T>` is a backward-compat alias for the underlying
`SentenceTransformer<T>` struct; both names refer to the same type.

The `device` argument is the re-exported `rust_bert::Device`; `None` defaults
to `Device::cuda_if_available()`.

Also look at the minimal examples `examples/encode_torch.rs` and `examples/encode_onnx.rs`.

## Preparing a checkpoint

One command per checkpoint, per backend (see `utils/prepare_models.py`;
deps in `utils/requirements-torch.txt` / `utils/requirements-onnx.txt`):

```Bash
# torch weights (scaffolding + .ot conversion via the convert-tensor bin):
python utils/prepare_models.py sentence-transformers/distiluse-base-multilingual-cased \
    models/distiluse-base-multilingual-cased --backend torch

# onnx weights (scaffolding + optimum bare-backbone export + Dense safetensors):
python utils/prepare_models.py sentence-transformers/distiluse-base-multilingual-cased \
    models/distiluse-base-multilingual-cased --backend onnx

# a bare RoBERTa-family classifier (no modules.json; torch only):
python utils/prepare_models.py cross-encoder/stsb-roberta-base \
    models/distilroberta_toxicity --backend torch
```

The transformer's rust-bert varstore prefix (`distilbert.` / `roberta.`) is
detected from `config.json`'s `model_type` / `architectures`; override with
`--prefix` for anything exotic. Checkpoints must ship `model.safetensors`
(legacy `pytorch_model.bin`-only repos are rejected with a pointer).

Then run the tests:

```Bash
cargo test --test sbert_test                                   # torch (default)
ORT_DYLIB_PATH=.../libonnxruntime.dylib \
  cargo test --test test_onnx --no-default-features --features onnx -- --nocapture
```

(Integration tests skip themselves when `models/` fixtures are absent)

## Pipeline architecture

`SentenceTransformer<T>` (torch: `src/models/sbert_torch.rs`, onnx:
`src/models/sbert_onnx.rs`) holds:

- **the transformer backend**: the one stage with a different I/O shape, so
  it is held separately rather than in the module list. Both backends expose
  the same `forward(input_ids, attention_mask)` contract to the driver:
  - `torch`: a `Box<dyn TransformerBackend>` over rust-bert's `BertModel` /
    `DistilBertModel`, weights loaded from the `model.ot` VarStore archive.
    The `load` factory in `src/modules/transformer.rs` dispatches on
    `config.json`'s `model_type`.
  - `onnx`: an `OnnxBackend` wrapping rust-bert's ndarray `ONNXEncoder` over
    the checkpoint's bare-backbone `model.onnx` (`src/modules_nd/transformer.rs`).
    The graph *is* the model, so there is no per-architecture Rust code;
    `OnnxBackend::new` only validates `model_type` as a sanity check.
- **`post: Vec<Box<dyn Module>>`**: the post-transformer stages, built from
  `modules.json` in declared order:
  - the forward pass threads a `Features` enum (`Token { .. }` →
    `Sentence { .. }`) through each module; `Pooling` does the token→sentence
    reduction.
  - the tch implementations live in `src/modules/`, their ndarray mirrors in
    `src/modules_nd/` (including a dependency-free safetensors reader for
    `Dense` weights).
  - the manifest parser (`src/modules/manifest.rs`) and tokenizer-settings
    resolution (`src/models/settings.rs`) are shared and cfg-free.

### To extend

- **New backbone** (e.g. RoBERTa):
  - `torch`: add a backend struct + `impl TransformerBackend` in
    `src/modules/transformer.rs`, plus a match arm in the `load` factory.
    `utils/prepare_models.py` must also know the backbone's rust-bert varstore
    prefix (`distilbert.`, `roberta.`, ...) so `model.safetensors` converts to
    `model.ot`.
  - `onnx`: no Rust backend struct is needed: backbones are graphs. Just
    accept the new `model_type` in `OnnxBackend::new` (currently an explicit
    `bert` / `distilbert` allow-list) and export the graph with
    `utils/prepare_onnx.py` (optimum handles the architecture).
  - **both**: the drivers hard-code the tokenizer file to
    `transformer_dir.join("vocab.txt")`, and the built-in tokenizers construct
    WordPiece from it. A backbone with a different tokenizer (e.g. RoBERTa's
    BPE / SentencePiece) therefore needs a `tokenizers::Tokenizer`
    implementation *and* matching tokenizer-file resolution. The existing
    `RustTokenizersSentencePiece` reads `vocab.json` + `merges.txt`, not
    `vocab.txt`, so the driver's path plumbing has to select it.
- **New post-transformer module** (e.g. `WeightedLayerPooling`):
  - add a struct + `impl Module` under `src/modules/` (tch) and its ndarray
    mirror under `src/modules_nd/`, respecting the `Features` transition
    (`Token` → `Sentence` at `Pooling`).
  - add a match arm on `entry.short_type()` in **both** drivers' manifest walks
    (`sbert_torch.rs` and `sbert_onnx.rs`). Extend the shared
    `transformer`/`bert`/`distilbert` arm too if the checkpoint uses a legacy
    per-architecture transformer class name.

See `src/modules/` and `src/modules_nd/` for the building blocks.

[sentence-transformers]: https://github.com/UKPLab/sentence-transformers
[luxbit-rust-bert]: https://github.com/Luxbit/rust-bert
[rust-bert]: https://github.com/guillaume-be/rust-bert
[tch-rs]: https://github.com/LaurentMazare/tch-rs
[rust-tokenizers]: https://github.com/guillaume-be/rust-tokenizers
[tokenizers]: https://github.com/huggingface/tokenizers/tree/master/tokenizers
