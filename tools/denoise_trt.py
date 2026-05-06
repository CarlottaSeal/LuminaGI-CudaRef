"""denoise_trt.py <engine.trt> <noisy.png> <out.png> [--check ckpt.pt]"""

import argparse
import pathlib
import sys
import time

import numpy as np
import tensorrt as trt
import torch
from PIL import Image


def pad_to_8(x):
    H, W = x.shape[2], x.shape[3]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return x, H, W


class TRTRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(pathlib.Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name  = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.stream = torch.cuda.Stream()

    def __call__(self, x_np):
        x_t = torch.from_numpy(x_np).contiguous().to("cuda")
        self.ctx.set_input_shape(self.in_name, tuple(x_t.shape))
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        y_t = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.in_name,  x_t.data_ptr())
        self.ctx.set_tensor_address(self.out_name, y_t.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return y_t.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_path")
    ap.add_argument("noisy_png")
    ap.add_argument("out_png")
    ap.add_argument("--check", help="PyTorch .pt to compare against")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs",   type=int, default=10)
    args = ap.parse_args()

    runner = TRTRunner(args.engine_path)

    img = np.asarray(Image.open(args.noisy_png).convert("RGB"), dtype=np.float32) / 255.0
    x = img.transpose(2, 0, 1)[None]
    x_pad, H, W = pad_to_8(x)

    for _ in range(args.warmup):
        runner(x_pad)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.runs):
        y = runner(x_pad)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0 / args.runs

    y = y[0, :, :H, :W].transpose(1, 2, 0)
    out = (np.clip(y, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(args.out_png)
    print(f"trt inference: {dt_ms:.2f} ms ({W}x{H}, {args.runs} runs avg)  wrote {args.out_png}")

    if args.check:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ml"))
        from train import UNet  # noqa: E402

        ckpt = torch.load(args.check, map_location="cuda", weights_only=True)
        base = ckpt.get("base", 32)
        model = UNet(base=base).to("cuda").eval()
        model.load_state_dict(ckpt["model"])
        with torch.no_grad():
            yt = model(torch.from_numpy(x_pad).to("cuda")).cpu().numpy()

        diff = np.abs(yt - runner(x_pad))
        print(f"parity vs PyTorch FP32: max abs diff = {diff.max():.2e}  mean = {diff.mean():.2e}")


if __name__ == "__main__":
    main()
