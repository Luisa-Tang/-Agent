"""
evaluate.py — Task: Circle Packing in a Unit Square (n=26)

Usage:
    python evaluate.py                  # evaluate baseline.py
    python evaluate.py solution.py      # evaluate solution.py
"""

import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np

# ── Fixed evaluation parameters ─────────────────────────────────────────────
NUM_CIRCLES  = 26
RANDOM_SEED  = 42
TIMEOUT      = 1000          # wall-clock limit per call (seconds)
INNER_LIMIT  = TIMEOUT - 5   # time limit injected into the solution
TARGET       = 2.635990        # score denominator
# ────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

_RUNNER = """\
import sys, os, pickle, traceback, numpy as np
os.environ["PACKING_RANDOM_SEED"] = "{seed}"
os.environ["PACKING_TIME_LIMIT"]  = "{limit}"
sys.path.insert(0, os.path.dirname({path!r}))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_solution", {path!r})
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    centers, radii, sum_radii = mod.run_packing({n})
    payload = dict(
        centers=np.asarray(centers, dtype=float),
        radii=np.asarray(radii, dtype=float),
        sum_radii=float(sum_radii),
    )
    pickle.dump({{"ok": True, "payload": payload}}, open({result!r}, "wb"))
except Exception:
    pickle.dump({{"ok": False, "error": traceback.format_exc()}}, open({result!r}, "wb"))
"""


def _run(path):
    """Run path in a subprocess. Returns (payload | None, error | None, elapsed)."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        runner = f.name
        result = runner + ".pkl"
        f.write(_RUNNER.format(
            seed=RANDOM_SEED, limit=INNER_LIMIT,
            path=path, n=NUM_CIRCLES, result=result,
        ))
    t0 = time.time()
    try:
        proc = subprocess.Popen([sys.executable, runner],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            proc.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
            return None, f"Timeout after {TIMEOUT}s", time.time() - t0

        elapsed = time.time() - t0
        if not os.path.exists(result):
            return None, "No result file produced", elapsed
        data = pickle.load(open(result, "rb"))
        if not data["ok"]:
            return None, data["error"], elapsed
        return data["payload"], None, elapsed
    finally:
        for p in (runner, result):
            try: os.unlink(p)
            except FileNotFoundError: pass


def _validate(centers, radii):
    """Return (is_valid, message)."""
    n = NUM_CIRCLES
    if centers.shape != (n, 2):
        return False, f"centers.shape {centers.shape} != ({n}, 2)"
    if radii.shape != (n,):
        return False, f"radii.shape {radii.shape} != ({n},)"
    if not (np.isfinite(centers).all() and np.isfinite(radii).all()):
        return False, "non-finite values in output"
    if not (radii >= 0).all():
        return False, "negative radius detected"
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or \
           y - r < -1e-9 or y + r > 1 + 1e-9:
            return False, f"circle {i} extends outside unit square"
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if radii[i] + radii[j] > d + 1e-9:
                return False, (f"circles {i} and {j} overlap "
                               f"(dist={d:.6f}, r_i+r_j={radii[i]+radii[j]:.6f})")
    return True, ""


def evaluate(path):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Circle Packing in Unit Square  (n={NUM_CIRCLES})")
    print(f"  File : {path}")
    print(sep)

    payload, err, elapsed = _run(path)
    print(f"  Elapsed : {elapsed:.2f}s")

    if payload is None:
        print(f"  [FAILED]   {err}")
        print(f"  Score    : 0.000000")
        print(f"{sep}\n")
        return 0.0

    valid, msg = _validate(payload["centers"], payload["radii"])
    if not valid:
        print(f"  [INVALID]  {msg}")
        print(f"  Score    : 0.000000")
        print(f"{sep}\n")
        return 0.0

    sum_r = float(np.sum(payload["radii"]))
    score = sum_r / TARGET
    print(f"  sum_radii : {sum_r:.6f}")
    print(f"  Target    : {TARGET:.6f}")
    print(f"  Score     : {score:.6f}")
    print(f"{sep}\n")
    return score


if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "baseline.py")
    evaluate(os.path.abspath(target_file))
