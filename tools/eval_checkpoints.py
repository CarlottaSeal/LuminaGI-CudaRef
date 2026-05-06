"""eval_checkpoints.py noisy.png gt.png ckpt1.pt [ckpt2.pt ...]"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ml"))
from train import UNet  # noqa: E402


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))


def denoise_one(model, device, img_np):
    H, W = img_np.shape[:2]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    x = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    y = y[:, :, :H, :W].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), dt_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("noisy_png")
    ap.add_argument("gt_png")
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noisy = np.asarray(Image.open(args.noisy_png).convert("RGB"), dtype=np.float32) / 255.0
    gt    = np.asarray(Image.open(args.gt_png).convert("RGB"), dtype=np.uint8)

    print("| checkpoint | infer ms | PSNR (dB) | SSIM |")
    print("|---|--:|--:|--:|")

    raw_u8 = (np.clip(noisy, 0, 1) * 255 + 0.5).astype(np.uint8)
    print(f"| (raw input) | — | {psnr(raw_u8, gt):.2f} | {ssim(raw_u8, gt, channel_axis=2, data_range=255):.4f} |")

    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        base  = ckpt.get("base", 32)
        in_ch = ckpt.get("in_ch", 3)
        if in_ch != 3:
            print(f"| {pathlib.Path(ckpt_path).name} | — | — | — |  # in_ch={in_ch}, skipped")
            continue
        model = UNet(base=base, in_ch=in_ch).to(device).eval()
        model.load_state_dict(ckpt["model"])
        # warmup + best of 3
        denoise_one(model, device, noisy)
        best_dt = float("inf")
        for _ in range(3):
            out, dt = denoise_one(model, device, noisy)
            best_dt = min(best_dt, dt)
        p = psnr(out, gt)
        s = ssim(out, gt, channel_axis=2, data_range=255)
        print(f"| {pathlib.Path(ckpt_path).name} | {best_dt:.1f} | {p:.2f} | {s:.4f} |")


if __name__ == "__main__":
    main()
