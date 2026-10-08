# Rust SBert [![Latest Version]][crates.io] [![Latest Doc]][docs.rs] [![Build Status]][ci]

[Latest Version]: https://img.shields.io/crates/v/sbert.svg
[crates.io]: https://crates.io/crates/sbert
[Latest Doc]: https://docs.rs/sbert/badge.svg
[docs.rs]: https://docs.rs/sbert
[Build Status]: https://github.com/Luxbit/rust-sbert/actions/workflows/ci.yml/badge.svg
[ci]: https://github.com/Luxbit/rust-sbert/actions/workflows/ci.yml

Rust port of [sentence-transformers][] with two interchangeable inference
backends selected at compile time via cargo features — **libtorch**
([tch-rs][], the default) or **ONNX Runtime** (libtorch-free, via the
[Luxbit fork][luxbit-rust-bert] of [rust-bert][] where `tch` is optional and
the ONNX stack is ndarray-based). The features are mutually exclusive by
design — one tensor library per build:

| You want... | Cargo | Weights | Runtime needs |
|---|---|---|---|
| libtorch (default) | `sbert = "0.8"` (feature `torch`) | `model.ot` VarStore archives | libtorch (torch-sys can download it; MPS/Vulkan supported) |
| ONNX Runtime, no libtorch | `sbert = { version = "0.8", default-features = false, features = ["onnx"] }` | `model.onnx` + `2_Dense/weights.safetensors` | onnxruntime via `ORT_DYLIB_PATH` |
| Both at once | — | `compile_error!` (mutually exclusive) | — |

The model pipeline is driven by a checkpoint's `modules.json` manifest: the
library reads it, runs the transformer stage through the selected backend,
and composes the post-transformer modules (`Pooling`, optional `Dense`,
optional `Normalize`) in declared order — tch tensors under `torch`, pure
`ndarray` under `onnx`.

Supports both [rust-tokenizers][] and Hugging Face's [tokenizers][].

## Requirements

- `torch` (default): a libtorch at build/run time. With no `LIBTORCH*` env
  vars set, torch-sys downloads PyTorch itself (the rust-bert fork's tch
  dependency enables `download-libtorch`). Or point `LIBTORCH` at a local
  extraction, as usual for tch crates.
- `onnx`: an **onnxruntime** shared library (>= 1.17) at runtime, located
  via `ORT_DYLIB_PATH` — the manual setup documented by the rust-bert fork:

  1. Download a release for your platform from the
     [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
     (e.g. `onnxruntime-osx-arm64-1.20.1.tgz`).
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
`SentenceTransformer<T>` struct — both names refer to the same type.

The `device` argument (`None` = `Device::cuda_if_available()`) is the
re-exported `rust_bert::Device`: under `torch` it maps to `tch::Device`
(`Cuda(i)` / `Cpu`; MPS and Vulkan map to CPU — request them via tch at the
call site if needed), under `onnx` it selects the execution provider
(`Cuda(i)` → CUDA EP, which additionally requires rust-bert's `cuda`
feature downstream; `Cpu` → CPU EP; there is no Metal provider).

`examples/encode_torch.rs` and `examples/encode_onnx.rs` are twin minimal
programs — run both on the same checkpoint to see the backends agree to
float rounding (~1e-7 max abs diff on distiluse).

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

# a bare RoBERTa-family classifier (no modules.json — torch only):
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

(Integration tests skip themselves when `models/` fixtures are absent;
`cargo test --lib` is hermetic under both features.)

## Pipeline architecture

`SentenceTransformer<T>` (torch: `src/models/sbert_torch.rs`, onnx:
`src/models/sbert_onnx.rs`) holds:

- the transformer backend — under `torch`, rust-bert `BertModel` /
  `DistilBertModel` with `.ot` weights, dispatched on `config.json`'s
  `model_type` (`src/modules/transformer.rs`); under `onnx`, an
  `OnnxBackend` wrapping rust-bert's ndarray `ONNXEncoder` over the
  checkpoint's bare-backbone `model.onnx`
  (`src/modules_nd/transformer.rs`).
- `post: Vec<Box<dyn Module>>` — built from `modules.json` in declared
  order; the forward pass threads a `Features` enum
  (`Token { .. }` → `Sentence { .. }`) through each module; `Pooling` does
  the token→sentence reduction. The tch implementations live in
  `src/modules/`, their ndarray mirrors in `src/modules_nd/` (including a
  dependency-free safetensors reader for Dense weights); the manifest
  parser and tokenizer-settings resolution are shared and cfg-free.

To extend:

- **New backbone**: under `torch`, add a backend struct + impl
  `TransformerBackend` + a match arm in `modules::transformer::load`; under
  `onnx`, backbones are graphs — accept the new `model_type` in
  `OnnxBackend::new`. Either way a tokenizer compatible with the vocab is
  needed (tokenization assumes `vocab.txt` + WordPiece except for the
  SentencePiece impl).
- **New post-transformer module** (e.g. `WeightedLayerPooling`): add a
  module struct + impl `Module` + a match arm on `short_type` in both
  drivers' manifest walks.

## Migration from 0.7

- The hybrid `new_onnx*` / `new_with_source` constructors are gone — the
  fork's `ONNXEncoder` is ndarray-typed, so "ORT transformer + tch pooling"
  no longer exists. The ONNX pipeline is now the `onnx` feature's own
  full ndarray implementation; `SBertRT::new_onnx(path, device)` becomes
  `--features onnx` + `SBertRT::new(path, device)`.
- `Option<tch::Device>` → `Option<sbert::Device>` (`Device::Cpu` /
  `Device::Cuda(i)`; `Some(tch_dev.into())` at call sites that still have
  tch in their own graph).
- onnx checkpoints need `model.onnx` + `2_Dense/weights.safetensors`
  instead of `.ot` — `utils/prepare_models.py --backend onnx` produces
  them.
- tch is now 0.17 (libtorch 2.4) — was 0.15 / libtorch 2.2.

[sentence-transformers]: https://github.com/UKPLab/sentence-transformers
[luxbit-rust-bert]: https://github.com/Luxbit/rust-bert
[rust-bert]: https://github.com/guillaume-be/rust-bert
[tch-rs]: https://github.com/LaurentMazare/tch-rs
[rust-tokenizers]: https://github.com/guillaume-be/rust-tokenizers
[tokenizers]: https://github.com/huggingface/tokenizers/tree/master/tokenizers
