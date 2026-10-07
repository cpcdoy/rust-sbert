#!/usr/bin/env python3
"""Export sentence-transformer backbones to ONNX for the sbert `onnx` feature.

Usage:
    python utils/prepare_onnx.py <hf_repo_id_or_dir> <output_dir>

    # from the hub (writes models/all-MiniLM-L6-v2/model.onnx):
    python utils/prepare_onnx.py sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2

    # PREFERRED for an existing checkpoint: export from the transformer
    # module dir that already carries a converted `model.ot` — the ONNX
    # weights then match the tch path bit-for-bit, which is what the parity
    # test (tests/test_onnx.rs) compares:
    python utils/prepare_onnx.py models/distiluse-base-multilingual-cased/0_DistilBERT models/distiluse-base-multilingual-cased/0_DistilBERT

Recipe notes (hard-won):

- optimum 2.x removed ONNX export from the CLI/core. Pin optimum 1.25.x
  with transformers <5 (this repo's .venv313 does).
- TorchScript mode: sbert's `model.ot` is a rust-bert VarStore archive with
  pipe-separated keys (`distilbert|embeddings|…`). We rebuild a HF-style
  `pytorch_model.bin` (replace `|` -> `.`, strip the `distilbert.` prefix)
  and let optimum export that — guaranteeing identical weights.
- `library_name="transformers"` is essential for hub exports: it exports
  the bare backbone (output `last_hidden_state`). The default
  sentence_transformers loader bakes pooling+normalize into the graph and
  names outputs token_embeddings/sentence_embedding, which rust-bert's
  ONNXEncoder cannot consume.
- opset 17 (ONNX Runtime 1.15.1 — the version ort 1.15.2 pairs with —
  supports up to 18).
- optimum 1.25 + torch >= 2.9 crashes during post-export cleanup
  (`os.remove` of `model.onnx.data` vs the actually-written
  `model.onnx_data`) — after the files are already on disk. Expected and
  ignored below; the script then inlines external weights into a single
  self-contained `model.onnx`.
- modern `onnx` packages stamp IR version 10+; ONNX Runtime 1.15.1 caps at
  IR 9 — clamped (independent of the opset).
"""

import shutil
import sys
import tempfile
from pathlib import Path

import onnx
from optimum.exporters.onnx import main_export

TORCHSCRIPT_PREFIX = "distilbert."


def export_from_hub(repo: str, tmp: Path) -> None:
    try:
        main_export(
            repo,
            str(tmp),
            task="feature-extraction",
            opset=17,
            library_name="transformers",
        )
    except FileNotFoundError as e:
        # optimum 1.25 post-export cleanup bug with torch >= 2.9; the model
        # files are already written by this point.
        print(f"note: optimum cleanup crashed post-export (expected): {e}")


def export_from_torchscript(ts_dir: Path, export_in: Path, tmp: Path) -> None:
    import torch  # heavy import: only needed in this mode

    ts = torch.jit.load(str(ts_dir / "model.ot"), map_location="cpu")
    sd = dict(ts.state_dict())
    hf_sd = {}
    stripped = 0
    for k, v in sd.items():
        k2 = k.replace("|", ".")
        if k2.startswith(TORCHSCRIPT_PREFIX):
            k2 = k2[len(TORCHSCRIPT_PREFIX):]
            stripped += 1
        hf_sd[k2] = v.contiguous()
    if stripped == 0:
        print("note: no keys carried the rust-bert prefix "
              f"'{TORCHSCRIPT_PREFIX}' (expected for distiluse-style checkpoints)")
    torch.save(hf_sd, str(export_in / "pytorch_model.bin"))
    for f in ("config.json", "vocab.txt"):
        shutil.copy(ts_dir / f, export_in / f)
    print(f"rehydrated {len(hf_sd)} params from {ts_dir / 'model.ot'}")
    try:
        main_export(
            str(export_in),
            str(tmp),
            task="feature-extraction",
            opset=17,
            library_name="transformers",
        )
    except FileNotFoundError as e:
        print(f"note: optimum cleanup crashed post-export (expected): {e}")


def main(source: str, out_dir: Path) -> None:
    src_dir = Path(source)
    use_torchscript = src_dir.is_dir() and (src_dir / "model.ot").exists()

    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        if use_torchscript:
            export_from_torchscript(src_dir, Path(input_dir), Path(output_dir))
        else:
            export_from_hub(source, Path(output_dir))

        onnx_path = Path(output_dir) / "model.onnx"
        assert onnx_path.exists(), "export did not produce model.onnx"
        model = onnx.load_model(str(onnx_path), load_external_data=True)

    # Modern `onnx` packages stamp IR version 10+; ONNX Runtime 1.15.1 (what
    # ort 1.15.2 pairs with) supports IR <= 9. IR and opset are independent —
    # clamping is safe for an opset-17 graph.
    if model.ir_version > 9:
        print(f"note: clamping IR version {model.ir_version} -> 9 (ORT 1.15.1 ceiling)")
        model.ir_version = 9

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "model.onnx"
    onnx.save_model(model, str(dest), save_as_external_data=False)

    ins = [i.name for i in model.graph.input]
    outs = [o.name for o in model.graph.output]
    print(f"wrote {dest} (opset {model.opset_import[0].version})")
    print(f"  inputs : {ins}")
    print(f"  outputs: {outs}")
    if outs != ["last_hidden_state"]:
        raise SystemExit(
            f"unexpected outputs {outs} — export used the wrong library "
            "(need library_name='transformers'); see module docstring"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]))
