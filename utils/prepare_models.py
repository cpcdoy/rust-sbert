"""Prepare the distiluse-base-multilingual-cased checkpoint for integration
tests (`.github/workflows/ci.yml` → `integration-tests`).

Replaces the old TU-Darmstadt-zip + `torch.load` pipeline: the checkpoint is
downloaded from the Hugging Face hub and converted **torch-free** —
`model.safetensors` files are read with the `safetensors` package (numpy
backend), keys are renamed to this crate's VarStore layout, and the npz →
`.ot` step shells out to the existing `convert-tensor` cargo bin (which does
need libtorch; CI gets it via `--features tch/download-libtorch`).

Key renames (mirroring the old torch-based conversion, verified against the
crate's loaders):

* transformer (rust-bert `DistilBertModel` varstore): the hub weights are the
  bare submodule state dict (`embeddings.*`, `transformer.layer.*`) and must
  be prefixed with `distilbert.` — see `DistilBertBackend::new` /
  rust-bert's `DistilBertStack` path in `src/modules/transformer.rs`;
* `2_Dense` (`nn::linear` varstore): hub keys are `linear.{weight,bias}`, the
  crate expects bare `{weight,bias}` — see `Dense::new` in
  `src/modules/dense.rs`.

Output layout (same `0_DistilBERT` directory shape the tests and the legacy
UKP export use; `manifest::resolve_module_dir` tolerates it):

    models/distiluse-base-multilingual-cased/
    ├── modules.json                      (authored below)
    ├── sentence_bert_config.json         (max_seq_length / do_lower_case)
    ├── 0_DistilBERT/{config.json, vocab.txt, model.ot}
    ├── 1_Pooling/config.json
    └── 2_Dense/{config.json, model.ot}

Env overrides:
  SBERT_MODELS_DIR       output root        (default: ./models)
  SBERT_CONVERT_FEATURES cargo features for convert-tensor
                         (default: tch/download-libtorch; empty to use the
                         ambient LIBTORCH* environment instead)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO_ID = "sentence-transformers/distiluse-base-multilingual-cased"
MODEL_NAME = "distiluse-base-multilingual-cased"

REPO_ROOT = Path(__file__).resolve().parent.parent

MODULES_JSON = """[
  { "idx": 0, "name": "0", "path": "0_DistilBERT", "type": "sentence_transformers.models.DistilBERT" },
  { "idx": 1, "name": "1", "path": "1_Pooling",    "type": "sentence_transformers.models.Pooling" },
  { "idx": 2, "name": "2", "path": "2_Dense",      "type": "sentence_transformers.models.Dense" }
]
"""

# hub filename -> destination relative to the model dir
SMALL_FILES = {
    "config.json": "0_DistilBERT/config.json",
    "vocab.txt": "0_DistilBERT/vocab.txt",
    "sentence_bert_config.json": "sentence_bert_config.json",
    "1_Pooling/config.json": "1_Pooling/config.json",
    "2_Dense/config.json": "2_Dense/config.json",
}

# safetensors source -> (destination .ot, key transform)
WEIGHTS = {
    "model.safetensors": ("0_DistilBERT/model.ot",
                          lambda k: k if k.startswith("distilbert.") else "distilbert." + k),
    "2_Dense/model.safetensors": ("2_Dense/model.ot",
                                  lambda k: k.split(".")[-1]),
}


def download(filename: str) -> Path:
    print(f"Downloading {REPO_ID}/{filename} ...")
    return Path(hf_hub_download(repo_id=REPO_ID, filename=filename))


def convert_to_ot(safetensors_path: Path, ot_path: Path, transform_key) -> None:
    tensors = load_file(str(safetensors_path))
    nps = {}
    for key, value in tensors.items():
        if value.dtype != np.float32:
            raise RuntimeError(
                f"{safetensors_path}: tensor {key!r} is {value.dtype}, expected float32"
            )
        nps[transform_key(key)] = np.ascontiguousarray(value)

    npz_path = ot_path.with_suffix(".npz")
    np.savez(npz_path, **nps)

    features = os.environ.get("SBERT_CONVERT_FEATURES", "tch/download-libtorch")
    cmd = [
        "cargo", "run", "--bin=convert-tensor",
        f"--manifest-path={REPO_ROOT / 'Cargo.toml'}",
    ]
    if features:
        cmd += ["--features", features]
    cmd += ["--", str(npz_path), str(ot_path)]
    print("Converting:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    npz_path.unlink()


def main() -> int:
    models_root = Path(os.environ.get("SBERT_MODELS_DIR", "models"))
    model_dir = models_root / MODEL_NAME
    print(f"Preparing checkpoint at {model_dir} ...")

    for relative in [".", "0_DistilBERT", "1_Pooling", "2_Dense"]:
        (model_dir / relative).mkdir(parents=True, exist_ok=True)

    (model_dir / "modules.json").write_text(MODULES_JSON)

    for hub_file, relative in SMALL_FILES.items():
        shutil.copyfile(download(hub_file), model_dir / relative)

    for hub_file, (relative, transform_key) in WEIGHTS.items():
        cached = download(hub_file)
        convert_to_ot(cached, model_dir / relative, transform_key)
        print(f"OK: {model_dir / relative}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
