"""gen_dataset.py <scene.json> <out_dir> [--n N] [--noisy-spp 8] [--clean-spp 256] [--bounces 2]"""

import argparse
import json
import math
import pathlib
import random
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
CUDA_REF = REPO / "build" / "bin" / "cuda_ref.exe"


def random_camera_around(anchor, radius_xy=2.5, radius_z=0.4, yaw_deg=30, pitch_deg=15):
    ax, ay, az = anchor
    r = random.uniform(0.0, radius_xy)
    ang = random.uniform(0.0, 2.0 * math.pi)
    pos = (ax + r * math.cos(ang),
           ay + r * math.sin(ang),
           az + random.uniform(-radius_z, radius_z))

    yaw   = math.radians(random.uniform(-yaw_deg,   yaw_deg))
    pitch = math.radians(random.uniform(-pitch_deg, pitch_deg))
    fx = math.cos(pitch) * math.cos(yaw)
    fy = math.cos(pitch) * math.sin(yaw)
    fz = math.sin(pitch)
    target = (pos[0] + fx, pos[1] + fy, pos[2] + fz)
    return pos, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_json")
    ap.add_argument("out_dir")
    ap.add_argument("--n",          type=int, default=50)
    ap.add_argument("--noisy-spp",  type=int, default=8)
    ap.add_argument("--clean-spp",  type=int, default=256)
    ap.add_argument("--bounces",    type=int, default=2)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--gbuffer",    action="store_true",
                    help="also write per-pose albedo/normal/depth (next to clean PNG)")
    args = ap.parse_args()

    random.seed(args.seed)

    scene_json = pathlib.Path(args.scene_json).resolve()
    anchor = json.loads(scene_json.read_text())["camera"]["pos"]

    out = pathlib.Path(args.out_dir).resolve()
    (out / "noisy").mkdir(parents=True, exist_ok=True)
    (out / "clean").mkdir(parents=True, exist_ok=True)

    batch_path = out / "batch.txt"
    with open(batch_path, "w") as f:
        f.write("# out_png spp bounces  camx camy camz  tgtx tgty tgtz\n")
        for i in range(args.n):
            pos, tgt = random_camera_around(anchor)
            name = f"{i:04d}.png"
            for subdir, spp in (("noisy", args.noisy_spp), ("clean", args.clean_spp)):
                p = (out / subdir / name).as_posix()
                f.write(f"{p} {spp} {args.bounces} "
                        f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} "
                        f"{tgt[0]:.4f} {tgt[1]:.4f} {tgt[2]:.4f}\n")

    print(f"wrote {batch_path}  ({2 * args.n} render jobs)")

    t0 = time.perf_counter()
    cmd = [str(CUDA_REF), str(scene_json), "--batch", str(batch_path)]
    if args.gbuffer:
        cmd.append("--gbuffer")
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        sys.exit("cuda_ref batch failed")
    elapsed = time.perf_counter() - t0
    print(f"done: {2*args.n} renders in {elapsed:.1f}s  ({elapsed / (2*args.n):.2f}s per render)")


if __name__ == "__main__":
    main()
