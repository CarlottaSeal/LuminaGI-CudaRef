"""validate.py <scene.json> [--open]  —  runs cuda_ref.exe then diff.py on the engine PNG next to it."""

import argparse
import os
import pathlib
import subprocess
import sys
import time


REPO = pathlib.Path(__file__).resolve().parent.parent
CUDA_REF = REPO / "build" / "bin" / "cuda_ref.exe"
DIFF_PY  = REPO / "tools" / "diff.py"
OUT_DIR  = REPO / "output"


def run(cmd, label):
    print(f"--- {label} ---")
    print("  " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.stdout: print(r.stdout.rstrip())
    if r.stderr: print(r.stderr.rstrip(), file=sys.stderr)
    if r.returncode != 0:
        sys.exit(f"{label} failed (exit {r.returncode})")
    print(f"  ({dt:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_json")
    ap.add_argument("--open", action="store_true", help="open HTML report after running")
    args = ap.parse_args()

    scene_json = pathlib.Path(args.scene_json).resolve()
    if not scene_json.is_file():
        sys.exit(f"no such file: {scene_json}")

    engine_png = scene_json.with_suffix(".png")
    if not engine_png.is_file():
        sys.exit(f"engine screenshot missing: {engine_png}")

    OUT_DIR.mkdir(exist_ok=True)
    stamp = scene_json.stem   # e.g. manual_20260417_091230
    ref_png  = OUT_DIR / f"ref_{stamp}.png"
    report   = OUT_DIR / f"report_{stamp}.html"

    run([str(CUDA_REF), str(scene_json), str(ref_png)], "CUDA reference render")
    run([sys.executable, str(DIFF_PY), str(engine_png), str(ref_png), "--out", str(report)], "image diff")

    print()
    print(f"report: {report}")
    if args.open:
        os.startfile(report)


if __name__ == "__main__":
    main()
