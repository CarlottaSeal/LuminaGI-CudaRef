"""denoise.py <checkpoint.pt> <noisy.png> <out.png>"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ml"))
from train import UNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("noisy_png")
    ap.add_argument("out_png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    base = ckpt.get("base", 32)
    model = UNet(base=base).to(device).eval()
    model.load_state_dict(ckpt["model"])

    img = np.asarray(Image.open(args.noisy_png).convert("RGB"), dtype=np.float32) / 255.0
    H, W = img.shape[:2]

    # Pad to multiple of 8 (three 2× downsamples in the U-Net).
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    y = y[:, :, :H, :W].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    y = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(y).save(args.out_png)

    print(f"inference: {dt_ms:.1f} ms ({W}x{H}, base={base})  wrote {args.out_png}")


if __name__ == "__main__":
    main()
