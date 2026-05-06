# ONNX export

The UNet denoiser is a 1.95 M-parameter fully-convolutional model. ONNX
export gives it a runtime-agnostic file that can be served by ONNX Runtime
(CPU / CUDA / DirectML EPs), TensorRT, or any other ONNX-compatible
runtime without depending on PyTorch at inference time.

## Export

```
python tools/export_onnx.py ml/runs/denoiser.pt ml/runs/denoiser.onnx
```

- Opset 17, dynamic batch / H / W axes (must be multiples of 8 at runtime
  due to the three 2× downsamples in the U-Net).
- Output: 7.80 MB. Same numerical content as the .pt; the difference is
  in metadata + a different graph encoding.
- Op set used: Conv, LeakyRelu, MaxPool, Resize, Concat, Add, Clip,
  Constant. All standard, no custom ops, no opset-specific shims.

## Numeric parity vs PyTorch

| EP | Inference (1728×864, 5-run avg) | Max abs diff vs PyTorch CUDA |
|---|--:|--:|
| PyTorch CUDA (reference) | 46.1 ms | 0 |
| ONNX Runtime CUDA EP | 56.3 ms | **0.00e+00** (bitwise) |
| ONNX Runtime CPU EP | 590.6 ms | 9.09e-06 |

Bitwise zero against PyTorch on the CUDA EP is the strict-correctness
check — same kernels, same numerics. The CPU EP's ~1e-5 difference
comes from a different math library and operation order, well below
the 1/255 visual quantisation floor.

## Inference

```
python tools/denoise_onnx.py ml/runs/denoiser.onnx noisy.png out.png
python tools/denoise_onnx.py ml/runs/denoiser.onnx noisy.png out.png --check ml/runs/denoiser.pt
```

Without `--check`, only ONNX Runtime is loaded. With `--check`, also
runs the matching PyTorch checkpoint and prints max abs diff.

## CUDA / cuDNN DLL pinning (Windows)

ONNX Runtime 1.25 (`onnxruntime-gpu`) ships against CUDA 12 / cuDNN 9.
The dev box here has CUDA 13 installed system-wide, so the CUDA EP fails
to load with `cublasLt64_12.dll missing`.

The fix in `denoise_onnx.py` is local-only — no system CUDA changes:

1. `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12`
   — these wheels carry the CUDA-12 DLLs as Python-package data.
2. Append the `nvidia/*/bin` directories to PATH **after** PyTorch's
   already-loaded cuDNN. Order matters: prepending breaks PyTorch
   because Windows then resolves PyTorch's transitive cuDNN imports
   to the v9.21 wheels, which are ABI-incompatible with the cuDNN
   PyTorch was built against.
3. `os.add_dll_directory` alone is not enough — ORT's provider DLL
   transitively dlopens `cublasLt64_12.dll`, which Windows resolves
   through PATH, not through Python's added dll directories.

`import torch` happens at the top of the script so its bundled cuDNN
wins the load race; the CUDA-12 wheels then back-fill anything ORT
needs that PyTorch's environment did not provide.

## Out of scope

- TensorRT engine build — the obvious next step. ONNX → trtexec would
  almost certainly drop CUDA-EP latency further (FP16, fused kernels)
  but adds a second build dependency.
- Quantisation (INT8). Not worth doing at 56 ms / 1.95 M params.
