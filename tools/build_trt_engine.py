"""build_trt_engine.py <model.onnx> <out.trt> [--fp16] [--height H --width W]"""

import argparse
import pathlib
import sys

import tensorrt as trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_path")
    ap.add_argument("out_trt")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--height", type=int, default=864,  help="optimisation height")
    ap.add_argument("--width",  type=int, default=1728, help="optimisation width")
    ap.add_argument("--workspace-mb", type=int, default=2048)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(args.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            sys.exit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    H, W = args.height, args.width
    profile.set_shape("noisy",
                      min=(1, 3, 8, 8),
                      opt=(1, 3, H, W),
                      max=(1, 3, H, W))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        sys.exit("engine build failed")

    out = pathlib.Path(args.out_trt)
    out.write_bytes(engine_bytes)
    sz = out.stat().st_size / 1e6
    fp = "FP16" if args.fp16 else "FP32"
    print(f"built: {out}  ({sz:.2f} MB, {fp}, opt H={H} W={W})")


if __name__ == "__main__":
    main()
