"""export_onnx.py <checkpoint.pt> <out.onnx> [--opset 17] [--height H --width W]"""

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ml"))
from train import UNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--width",  type=int, default=1728)
    args = ap.parse_args()

    if args.height % 8 or args.width % 8:
        sys.exit("height and width must be multiples of 8")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base  = ckpt.get("base", 32)
    in_ch = ckpt.get("in_ch", 3)
    model = UNet(base=base, in_ch=in_ch).eval()
    model.load_state_dict(ckpt["model"])

    dummy = torch.zeros(1, in_ch, args.height, args.width)

    torch.onnx.export(
        model,
        dummy,
        args.out_onnx,
        input_names=["noisy"],
        output_names=["denoised"],
        opset_version=args.opset,
        dynamic_axes={
            "noisy":    {0: "N", 2: "H", 3: "W"},
            "denoised": {0: "N", 2: "H", 3: "W"},
        },
    )

    sz = pathlib.Path(args.out_onnx).stat().st_size / 1e6
    print(f"exported: {args.out_onnx}  ({sz:.2f} MB, opset {args.opset}, base={base}, in_ch={in_ch})")


if __name__ == "__main__":
    main()
