# ECB block + super-resolution sweep

Adds two things to the denoiser, both behind flags so the baseline is untouched:

- `--ecb` — replace the plain 3x3 convs in every `ConvBlock` with an **Edge-oriented
  Convolution Block** (ECBSR, Zhang et al. MM '21). Train time runs 4 branches
  (3x3 + expand-squeeze + scaled Sobel-x/-y + scaled Laplacian); eval/deploy folds
  them into a single 3x3 via re-parameterization, so inference cost is identical to a
  plain conv. Implemented in [`ml/ecb.py`](../ml/ecb.py); `python ml/ecb.py` runs the
  train-vs-folded equivalence check (max abs diff ~1e-6).
- `--scale N` — joint **denoise + super-resolution**. The UNet body runs at low res and
  a sub-pixel PixelShuffle head upsamples by `N`, learning detail on top of a bilinear
  upscale of the input. The LR input is the clean target's noisy crop bicubic-downscaled
  by `N` (the DIV2K / ECBSR protocol). `--scale 1` (default) is the original denoiser.

Deploy: `tools/export_onnx.py` calls `convert_ecb_to_plain()` before export, so the ONNX
graph holds folded 3x3 convs, not the branch arithmetic (verified bit-exact).

## Experiment matrix

The point of the sweep is that ECB's edge prior should be **task-dependent** — little to
no help on denoising (removing zero-mean noise needs no edge prior), a real gain on the
SR sub-task (recovering high-frequency detail), matching the paper's "ECB helps SR, not
high-level vision" finding.

|                 | plain 3x3            | ECB                  |
|-----------------|----------------------|----------------------|
| denoise (x1)    | PSNR __ / SSIM __    | PSNR __ / SSIM __    |
| denoise+SR (x2) | PSNR __ / SSIM __    | PSNR __ / SSIM __    |

Reference baselines for the SR row: bilinear upscale of the LR input, and a cascade
(existing x1 denoiser → bilinear x2).

## Commands

```bat
:: denoise (x1) — plain vs ECB
python ml\train.py ml\data_v3 --optim adamw_cosine --out ml\runs\dn_plain.pt
python ml\train.py ml\data_v3 --ecb --optim adamw_cosine --out ml\runs\dn_ecb.pt

:: denoise + SR x2 — plain vs ECB
python ml\train.py ml\data_v3 --scale 2 --optim adamw_cosine --out ml\runs\sr_plain.pt
python ml\train.py ml\data_v3 --scale 2 --ecb --optim adamw_cosine --out ml\runs\sr_ecb.pt

:: compare (x1 row; eval harness for the SR row compares against HR ground truth)
python tools\eval_checkpoints.py ml\runs\dn_plain.pt ml\runs\dn_ecb.pt

:: deploy an ECB model (folds to single 3x3 in the ONNX graph)
python tools\export_onnx.py ml\runs\sr_ecb.pt ml\runs\sr_ecb.onnx
```

## Notes / open items

- The SR pairs use bicubic-downscaled renders (DIV2K/ECBSR protocol). A rendering-specific
  variant — native low-res renders from `cuda_ref` (different aliasing/noise than a
  downscaled HR frame) — needs a `--res` override in the renderer and a per-job resolution
  column in `batch.txt`; not done here.
- `eval_checkpoints.py` compares same-size images; the SR row needs the LR-in / HR-GT
  pairing wired into the eval harness before it reports the SR table.
- ECB + `--gbuffer` not supported together yet (guarded with an error).
