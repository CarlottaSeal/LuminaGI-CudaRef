"""
Compare a LuminaGI screenshot against the CUDA reference render.

    diff.py engine.png ref.png [--out report.html]

Produces an HTML report with PSNR/SSIM numbers and an abs-diff heatmap.
If the two images differ in size, the engine image is resized to match.
"""

import argparse
import base64
import io
import pathlib
import sys

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8), img.size  # size = (w, h)


def match_size(a, size_b):
    # size_b is (w, h)
    w_a, h_a = a.shape[1], a.shape[0]
    if (w_a, h_a) == size_b:
        return a
    img = Image.fromarray(a).resize(size_b, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


# 256-step viridis-ish colormap, enough for a single-channel heatmap.
def viridis_lut():
    x = np.linspace(0.0, 1.0, 256)
    r = np.clip(-0.2 + 2.4 * x - 1.3 * x ** 2, 0, 1)
    g = np.clip(0.1 + 0.8 * x + 0.1 * x ** 2, 0, 1)
    b = np.clip(0.4 + 0.8 * x - 1.1 * x ** 2, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def heatmap(a, b, gain=4.0):
    d = np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)), axis=2)
    d = np.clip(d * gain, 0, 255).astype(np.uint8)
    lut = viridis_lut()
    return lut[d]


def png_base64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Diff Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 1.5em; color: #eee; background: #222; }}
h1 {{ margin-top: 0; }}
table {{ border-collapse: collapse; margin-bottom: 1em; }}
td, th {{ padding: 0.3em 1em; border-bottom: 1px solid #555; text-align: left; }}
th {{ color: #aaa; font-weight: normal; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1em; }}
.grid figure {{ margin: 0; }}
.grid figcaption {{ margin-top: 0.3em; color: #aaa; font-size: 0.9em; }}
.grid img {{ width: 100%; display: block; border: 1px solid #444; }}
</style></head><body>
<h1>LuminaGI vs CUDA Reference</h1>
<table>
<tr><th>Resolution</th><td>{w}x{h}</td></tr>
<tr><th>PSNR</th><td>{psnr_val:.2f} dB</td></tr>
<tr><th>SSIM</th><td>{ssim_val:.4f}</td></tr>
<tr><th>Mean abs diff (per channel)</th><td>{mad:.2f} / 255</td></tr>
<tr><th>Engine</th><td>{engine_path}</td></tr>
<tr><th>Reference</th><td>{ref_path}</td></tr>
</table>
<div class="grid">
<figure><img src="data:image/png;base64,{eng_b64}"><figcaption>Engine (LuminaGI)</figcaption></figure>
<figure><img src="data:image/png;base64,{ref_b64}"><figcaption>Reference (CUDA)</figcaption></figure>
<figure><img src="data:image/png;base64,{heat_b64}"><figcaption>Abs diff &times; 4</figcaption></figure>
</div>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", help="LuminaGI screenshot PNG")
    ap.add_argument("ref", help="CUDA reference PNG")
    ap.add_argument("--out", default="diff_report.html")
    args = ap.parse_args()

    engine, eng_size = load_rgb(args.engine)
    ref, ref_size = load_rgb(args.ref)

    if eng_size != ref_size:
        print(f"resizing engine {eng_size} -> {ref_size}", file=sys.stderr)
        engine = match_size(engine, ref_size)

    psnr_val = psnr(engine, ref)
    ssim_val = ssim(engine, ref, channel_axis=2, data_range=255)
    mad = float(np.mean(np.abs(engine.astype(np.int16) - ref.astype(np.int16))))

    heat = heatmap(engine, ref, gain=4.0)
    heat_png = pathlib.Path(args.out).with_suffix(".heatmap.png")
    Image.fromarray(heat).save(heat_png)

    html = HTML.format(
        w=ref_size[0], h=ref_size[1],
        psnr_val=psnr_val, ssim_val=ssim_val, mad=mad,
        engine_path=pathlib.Path(args.engine).resolve().as_posix(),
        ref_path=pathlib.Path(args.ref).resolve().as_posix(),
        eng_b64=png_base64(engine),
        ref_b64=png_base64(ref),
        heat_b64=png_base64(heat),
    )
    pathlib.Path(args.out).write_text(html, encoding="utf-8")

    print(f"PSNR {psnr_val:.2f} dB | SSIM {ssim_val:.4f} | MAD {mad:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
