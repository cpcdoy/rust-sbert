#!/usr/bin/env python3
"""Prepare a checkpoint for sbert (both backends) — two layouts:

Usage:
    python utils/prepare_models.py <hf_repo_id> <output_dir> [--backend torch|onnx|both] [--prefix P.]

    python utils/prepare_models.py sentence-transformers/distiluse-base-multilingual-cased \
        models/distiluse-base-multilingual-cased --backend both
    python utils/prepare_models.py cardiffnlp/twitter-roberta-base-sentiment-latest \
        models/distilroberta_toxicity --backend torch

Sentence-transformers pipelines (modules.json present) or bare classifier
checkpoints like `models/distilroberta_toxicity` (no modules.json —
transformer at root, torch backend only). Per backend this produces:

* shared scaffolding from the hub (modules.json, configs, vocab / tokenizer
  files) — pure download, no torch;
* `--backend torch` (or both): VarStore `model.ot` archives — for the
  transformer with the rust-bert backbone prefix (`distilbert.` for
  DistilBERT, `roberta.` for RoBERTa/DistilRoBERTa — detected from
  config.json, override with --prefix) and for `2_Dense` (bare
  `weight`/`bias`). Conversion is safetensors -> npz in Python, then npz ->
  .ot via the `convert-tensor` cargo bin (which needs a libtorch; with
  default features torch-sys downloads one);
* `--backend onnx` (or both; pipelines only): the bare-backbone `model.onnx`
  transformer export (optimum — recipe notes in utils/prepare_onnx.py) plus
  `2_Dense/weights.safetensors` (the Dense projection lives outside the
  ONNX graph and is read directly by the crate).

Dependencies: utils/requirements-torch.txt for the torch backend,
utils/requirements-onnx.txt additionally for the onnx export.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors.numpy import load_file

REPO_ROOT = Path(__file__).resolve().parent.parent

# Root files a checkpoint commonly carries; missing ones are skipped
# (e.g. vocab.json/merges.txt only exist for sentencepiece checkpoints).
ROOT_FILES = [
    "modules.json",
    "config.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentence_bert_config.json",
]


def fetch(repo_id: str, filename: str, dest: Path, optional: bool = False) -> bool:
    try:
        cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except EntryNotFoundError:
        if optional:
            print(f"  (optional) {filename}: not on the hub, skipping")
            return False
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(cached.read_bytes())
    print(f"  {dest}")
    return True


def fetch_safetensors(repo_id: str) -> Path:
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors"))
    except EntryNotFoundError:
        raise SystemExit(
            f"{repo_id}: no model.safetensors on the hub — legacy "
            "pytorch_model.bin-only checkpoints are not supported (convert "
            "to safetensors first, e.g. via torch/safetensors save_file)"
        )


def detect_prefix(out_dir: Path, transformer_rel: str) -> str:
    """Map the checkpoint's backbone to its rust-bert varstore prefix.

    rust-bert namespaces the backbone in the varstore (`distilbert.` for
    DistilBertModel, `roberta.` for RoBERTa-family models). Sentence-
    transformers pipelines store the bare submodule state dict (needs the
    prefix added); full classifier checkpoints already carry it (keys pass
    through unchanged).
    """
    for cfg_path in (out_dir / transformer_rel / "config.json",
                     out_dir / "config.json"):
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        model_type = (cfg.get("model_type") or "").lower()
        archs = " ".join(str(a).lower() for a in cfg.get("architectures") or [])
        if "distilbert" in model_type or "distilbert" in archs:
            return "distilbert."
        if model_type in ("roberta", "distilroberta") or "roberta" in archs:
            return "roberta."
        raise SystemExit(
            f"{cfg_path}: unsupported backbone (model_type={model_type!r}, "
            f"architectures={cfg.get('architectures')!r}); pass --prefix manually"
        )
    raise SystemExit(
        "no config.json found to detect the backbone prefix from; "
        "pass --prefix manually"
    )


def convert_to_ot(safetensors_path: Path, ot_path: Path, transform_key) -> None:
    tensors = load_file(str(safetensors_path))
    nps = {}
    for key, value in tensors.items():
        if value.dtype == np.float16:
            print(f"  note: {key} is float16, casting to float32")
            value = value.astype(np.float32)
        elif value.dtype != np.float32:
            if np.issubdtype(value.dtype, np.integer):
                # Constant buffers (e.g. roberta.embeddings.position_ids),
                # not weights: rust-bert computes them at forward time and
                # the VarStore .ot carries only trainable tensors.
                print(f"  skipping non-weight buffer {key} ({value.dtype})")
                continue
            raise RuntimeError(
                f"{safetensors_path}: tensor {key!r} is {value.dtype}, expected float32"
            )
        nps[transform_key(key)] = np.ascontiguousarray(value)

    npz_path = ot_path.with_suffix(".npz")
    np.savez(npz_path, **nps)

    cmd = [
        "cargo", "run", "--bin=convert-tensor",
        f"--manifest-path={REPO_ROOT / 'Cargo.toml'}",
        "--", str(npz_path), str(ot_path),
    ]
    print("Converting:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    npz_path.unlink()


def prepare_torch(repo_id: str, out_dir: Path, modules: list, prefix: str) -> None:
    if modules:
        # Pipeline: rust-bert's varstore prefixes the backbone; ST repos ship
        # the bare submodule state dict, so add the prefix (no-op if present).
        print(f"torch backend: converting .ot weights (prefix {prefix!r})")
        transform = lambda k: k if k.startswith(prefix) else prefix + k
    else:
        # Classifier: a full RobertaForSequenceClassification state dict is
        # already namespaced (`roberta.*` + `classifier.*`) — pass through.
        print("torch backend: converting .ot weights (full classifier state "
              "dict, keys pass through unchanged)")
        transform = lambda k: k
    st = fetch_safetensors(repo_id)
    convert_to_ot(st, out_dir / "model.ot", transform)
    # Dense: hub keys are linear.{weight,bias}; nn::linear expects bare names.
    for entry in modules:
        if entry.get("path") and entry["type"].rsplit(".", 1)[-1].lower() == "dense":
            st = Path(hf_hub_download(repo_id=repo_id, filename=f"{entry['path']}/model.safetensors"))
            convert_to_ot(st, out_dir / entry["path"] / "model.ot",
                          lambda k: k.split(".")[-1])


def prepare_onnx(repo_id: str, out_dir: Path, modules: list) -> None:
    print("onnx backend: exporting bare-backbone graph")
    for entry in modules:
        if not entry.get("path"):
            continue
        if entry["type"].rsplit(".", 1)[-1].lower() == "dense":
            # linear.weight / linear.bias, read directly by sbert's ndarray Dense.
            fetch(repo_id, f"{entry['path']}/model.safetensors",
                  out_dir / entry["path"] / "weights.safetensors")

    from prepare_onnx import main as export_backbone

    transformer = next(
        (e.get("path") or "" for e in modules
         if e["type"].rsplit(".", 1)[-1].lower() in ("transformer", "bert", "distilbert")),
        "",
    )
    export_backbone(repo_id, out_dir / transformer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="Hugging Face repo id, e.g. sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--backend", choices=["torch", "onnx", "both"], default="both")
    parser.add_argument("--prefix", metavar="P.",
                        help="rust-bert varstore prefix for the transformer "
                             "(e.g. 'roberta.'); default: detected from config.json")
    args = parser.parse_args()

    print(f"Preparing {args.repo_id} -> {args.out_dir} (backend: {args.backend})")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in ROOT_FILES:
        fetch(args.repo_id, name, args.out_dir / name, optional=True)

    modules_path = args.out_dir / "modules.json"
    if modules_path.exists():
        modules = json.loads(modules_path.read_text())
        for entry in modules:
            rel = entry.get("path") or ""
            if rel == "":
                continue  # transformer at root: files already fetched
            fetch(args.repo_id, f"{rel}/config.json", args.out_dir / rel / "config.json")
    else:
        # Bare classifier checkpoint (e.g. models/distilroberta_toxicity):
        # transformer at root, no pipeline modules, torch backend only.
        print("no modules.json — preparing as a bare classifier checkpoint "
              "(torch backend only; see src/models/distilroberta.rs)")
        modules = []
        if args.backend == "onnx":
            raise SystemExit(
                "--backend onnx requires a sentence-transformers pipeline "
                "(modules.json); the classifier path is torch-only"
            )
        if args.backend == "both":
            args.backend = "torch"

    transformer_rel = next(
        (e.get("path") or "" for e in modules
         if e["type"].rsplit(".", 1)[-1].lower()
         in ("transformer", "bert", "distilbert", "roberta")),
        "",
    )
    prefix = args.prefix or detect_prefix(args.out_dir, transformer_rel)

    if args.backend in ("torch", "both"):
        prepare_torch(args.repo_id, args.out_dir, modules, prefix)
    if args.backend in ("onnx", "both"):
        prepare_onnx(args.repo_id, args.out_dir, modules)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
