"""denoise_onnx.py <model.onnx> <noisy.png> <out.png> [--check checkpoint.pt]"""

import argparse
import os
import pathlib
import site
import sys
import time

import numpy as np
import torch  # import first — its bundled cuDNN must win the loader race
from PIL import Image

# onnxruntime-gpu 1.25 needs CUDA 12 / cuDNN 9 DLLs that the system CUDA 13
# install does not provide. We pip-installed the nvidia-* wheels that ship
# them; expose those bins to the loader. add_dll_directory alone is not
# enough because ort's provider DLL transitively dlopens cublasLt64_12.dll,
# which Windows resolves via PATH. Append (don't prepend) so PyTorch's
# already-loaded cuDNN keeps priority.
_extra_paths = []
for sp in site.getsitepackages():
    for sub in ("cublas", "cudnn", "cuda_runtime", "cufft",
                "cuda_nvrtc", "nvjitlink"):
        p = pathlib.Path(sp) / "nvidia" / sub / "bin"
        if p.is_dir():
            _extra_paths.append(str(p))
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(p))
if _extra_paths:
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.pathsep.join(_extra_paths)

import onnxruntime as ort  # noqa: E402


def pad_to_8(x):
    H, W = x.shape[2], x.shape[3]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return x, H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("noisy_png")
    ap.add_argument("out_png")
    ap.add_argument("--check", help="PyTorch .pt to compare against")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs",   type=int, default=5)
    args = ap.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx_path, providers=providers)
    print(f"providers: {sess.get_providers()}")

    img = np.asarray(Image.open(args.noisy_png).convert("RGB"), dtype=np.float32) / 255.0
    x = img.transpose(2, 0, 1)[None]
    x_pad, H, W = pad_to_8(x)

    for _ in range(args.warmup):
        sess.run(None, {"noisy": x_pad})

    t0 = time.perf_counter()
    for _ in range(args.runs):
        y = sess.run(None, {"noisy": x_pad})[0]
    dt_ms = (time.perf_counter() - t0) * 1000.0 / args.runs

    y = y[0, :, :H, :W].transpose(1, 2, 0)
    out = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(args.out_png)
    print(f"onnx inference: {dt_ms:.1f} ms ({W}x{H}, {args.runs} runs avg)  wrote {args.out_png}")

    if args.check:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ml"))
        from train import UNet  # noqa: E402

        ckpt = torch.load(args.check, map_location="cuda", weights_only=True)
        base = ckpt.get("base", 32)
        model = UNet(base=base).to("cuda").eval()
        model.load_state_dict(ckpt["model"])

        with torch.no_grad():
            xt = torch.from_numpy(x_pad).to("cuda")
            yt = model(xt).cpu().numpy()

        max_abs = float(np.max(np.abs(yt - sess.run(None, {"noisy": x_pad})[0])))
        print(f"parity vs PyTorch: max abs diff = {max_abs:.2e}")


if __name__ == "__main__":
    main()
