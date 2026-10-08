#!/usr/bin/env python3
"""Cross-validate the torch and onnx backends on dump files.

Compares the JSON dumps produced by the twin examples:

    cargo run --example dump_torch -- models/<name> /tmp/<name>.torch.json
    ORT_DYLIB_PATH=... cargo run --no-default-features --features onnx \
        --example dump_onnx -- models/<name> /tmp/<name>.onnx.json
    python utils/compare_backends.py /tmp/<name>.torch.json /tmp/<name>.onnx.json

Checks, per text row:
* cross-backend max/mean absolute difference (threshold 1e-4) and cosine
  similarity (threshold 1 - 1e-6);
* within-backend batched-vs-solo invariance (threshold 1e-5 — catches
  padding leaking across batch rows or a broken order restore).
"""

import json
import math
import sys

MAX_ABS = 1e-4
MIN_COS = 1.0 - 1e-6
MAX_INVARIANCE = 1e-5


def stats(a, b):
    diffs = [abs(x - y) for x, y in zip(a, b)]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
    return max(diffs), sum(diffs) / len(diffs), cos


def main(torch_path, onnx_path):
    t = json.load(open(torch_path))
    o = json.load(open(onnx_path))
    assert t["texts"] == o["texts"], "corpora differ between dumps"
    n = len(t["texts"])
    print(f"{n} texts, dim {len(t['batched'][0])}")
    print(f"{'i':>2} {'max_abs':>10} {'mean_abs':>10} {'cosine':>12}  text")

    failures = 0
    worst = (0.0, -1)
    for i in range(n):
        m, mean, cos = stats(t["batched"][i], o["batched"][i])
        worst = max(worst, (m, i))
        flag = ""
        if m >= MAX_ABS or cos < MIN_COS:
            flag = "  <-- FAIL"
            failures += 1
        print(f"{i:>2} {m:10.3e} {mean:10.3e} {cos:12.10f}  {t['texts'][i][:38]!r}{flag}")

    inv_t = max(
        max(abs(x - y) for x, y in zip(t["batched"][i], t["solo"][i]))
        for i in range(n)
    )
    inv_o = max(
        max(abs(x - y) for x, y in zip(o["batched"][i], o["solo"][i]))
        for i in range(n)
    )
    print(f"batch-vs-solo invariance: torch {inv_t:.3e}  onnx {inv_o:.3e}")
    if inv_t >= MAX_INVARIANCE or inv_o >= MAX_INVARIANCE:
        failures += 1
        print("  <-- INVARIANCE FAIL")

    print(f"worst cross-backend: text {worst[1]} max_abs {worst[0]:.3e}")
    print("PASS" if failures == 0 else f"FAIL ({failures} failures)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
