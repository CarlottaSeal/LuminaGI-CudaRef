# v3 sweep — bigger dataset + optimizer upgrade

The v2 baseline (`v2_rgb.pt`) was 50 poses, 256-spp clean targets, Adam lr=1e-3.
This sweep changed three things at once and ended with a clean +0.91 dB / +0.013 SSIM
win on the same held-out frame, plus one debugging story and one
already-known negative reinforced.

## Held-out frame (1920×1200, vs 1024-spp ground truth, `ml/eval_v2/gt_1024.png`)

| ckpt | data | optim | PSNR (dB) | SSIM |
|---|---|---|--:|--:|
| (raw 8-spp input) | — | — | 34.31 | 0.8030 |
| v2_rgb | 50p / 256-spp | Adam | 41.17 | 0.9530 |
| v2_gb | 50p / 256-spp | Adam | 40.36 | 0.9400 |
| v3_rgb | 150p / 1024-spp | AdamW + cos + clip + **EMA(0.999)** | 35.38 | 0.9143 |
| v3_adam | 150p / 1024-spp | Adam | 41.85 | 0.9657 |
| v3_awc | 150p / 1024-spp | AdamW + cos + clip, no EMA | 42.06 | 0.9658 |
| **v3_ema099** | 150p / 1024-spp | AdamW + cos + clip + **EMA(0.99)** | **42.08** | **0.9659** |
| v3_gb | 150p / 1024-spp | AdamW + cos + clip, +gbuffer | 40.91 | 0.9471 |

## What changed in the dataset

- **150 poses** (was 50). Same `random_camera_around` seed=42, so the first 50 poses
  are byte-identical to `data_gb`; 100 new poses appended.
- **Clean targets at 1024-spp** (was 256-spp). Noisy stays at 8-spp, the eval
  target.
- Both 1920×1200, same scene `manual_20260506_183748.json` (1.12 M tris).
- Render time: 300 jobs in 12189 s = 40.6 s/render avg, dominated by 150 ×
  1024-spp clean. Per-frame variance large (22–104 s) by view complexity.

## What changed in the optimizer

`train.py --optim adamw_cosine` swaps:

- `Adam` → `AdamW(weight_decay=1e-4)`
- fixed `lr=1e-3` → `CosineAnnealingLR(T_max=epochs)` (1e-3 → 0)
- `loss.backward(); opt.step()` → `... ; clip_grad_norm_(max_norm=1.0); opt.step()`
- optional EMA shadow used for eval and the saved checkpoint

Vanilla `--optim adam` path unchanged so old sweeps still reproduce.

## The v3_rgb regression and the diagnostic

First combined attempt (`v3_rgb`: all three changes, EMA decay 0.999): −5.79 dB.
Worse than the raw input on PSNR by only 1.07 dB, well below v2 on every metric.

A/B by holding the dataset constant:

- `v3_adam` (vanilla Adam on `data_v3`) → 41.85 dB. Better than v2 on the same
  optimizer, so the new dataset is fine.
- `v3_awc` (AdamW + cos + clip, no EMA) → 42.06 dB. AdamW + cos + clip are also
  fine.
- `v3_ema099` (same as v3_awc, EMA decay 0.99) → 42.08 dB. EMA at decay 0.99 is
  also fine, marginal gain over no-EMA.

So the regression was specifically EMA decay 0.999. Arithmetic: 80 epochs ×
30 batches/epoch = 2400 update steps; `0.999 ** 2400 ≈ 0.0907`. The shadow was
initialized as a copy of the random-init model, so ~9% of the saved weights at
the end of training were still that random initialization. The shadow was
undertrained, not the model.

This is a parameter-vs-scale mismatch. Decay 0.999 is the published default in
papers that train for ≥ 1e5 steps, where `0.999 ** 1e5 ≈ 4.5e-44`. At 2.4 k steps
it doesn't forget the init.

`v3_ema099` (decay 0.99, `0.99 ** 2400 ≈ 5e-11`) recovers and very slightly
beats no-EMA. With cosine LR hitting zero in the last few epochs, the late-
training averaging that EMA usually buys is mostly already done by the
schedule, so the additional gain is small.

## v3_adam diverged at epoch 70 — grad_clip in v3_awc prevented it

Same late-Adam dynamic as the epoch=120 sweep that diverged at epoch 83
(`epoch_sweep.md`):

```
v3_adam:
epoch  69/80  lr=1.00e-03  train=0.0048  val=0.0046
epoch  70/80  lr=1.00e-03  train=0.0898  val=0.1106
... no recovery through epoch 80
best val L1: 0.0041 saved (from epoch ~31, before divergence)
```

`v3_awc` and `v3_ema099` (both have `clip_grad_norm_(max_norm=1.0)` + cosine
LR decay) trained the full 80 epochs with no divergence and saved best
checkpoints in the last few epochs (ep 76–78). The combination of grad clip
plus end-of-schedule LR shrinkage is what ate the divergence — neither alone
would have, since `v3_adam` had constant lr but no clip and `v3_awc` had cos
LR that would have eventually reached the divergence regime if clip were off.

## Win decomposition

Over the +0.91 dB / +0.013 SSIM total improvement (`v2_rgb` → `v3_ema099`):

| component | Δ PSNR | Δ SSIM | source |
|---|--:|--:|---|
| data (1024-spp + 150 poses) | +0.68 | +0.0127 | v3_adam − v2_rgb |
| optimizer (AdamW + cos + clip) | +0.21 | +0.0001 | v3_awc − v3_adam |
| EMA at decay 0.99 | +0.02 | +0.0001 | v3_ema099 − v3_awc |

Most of the gain is from the cleaner training targets and the larger pose
distribution. The optimizer upgrade is real but small. EMA at our scale is
within noise.

## G-buffer retry — original negative was not a small-data artifact

The `gbuffer_sweep.md` regression (40.36 vs 41.17 RGB-only) was on 50 poses /
256-spp; cause #1 listed there was "40 train pairs too few for the 12-ch input
adapter". The v3 retry (`v3_gb`: 150 poses, 1024-spp, AdamW + cos + clip) lets
us test that hypothesis directly.

| variant | train pairs | PSNR (dB) | SSIM | Δ vs same-setup RGB |
|---|--:|--:|--:|--:|
| v2_gb | 40 | 40.36 | 0.9400 | −0.81 vs v2_rgb |
| v3_gb | 120 | 40.91 | 0.9471 | −1.17 vs v3_awc |

3× more data and a 4× higher-spp clean target: the gap got *larger*, not
smaller. The "too few pairs" hypothesis is no longer supported by the data.
The other two `gbuffer_sweep.md` causes survive:

- the model takes the shortcut of fitting deterministic aux features instead
  of denoising the noisy RGB
- a vanilla concat-input UNet does not implicitly learn the kernel-prediction
  / cross-bilateral structure that NRD and SVGF make explicit

`v3_gb` also diverged (epoch 35 → 36, train 0.0095 → 0.1323; best ckpt from
ep 31). The grad clip with `max_norm=1.0` and cosine LR were not enough on
the 12-ch model. RGB-only at the same setup (`v3_awc`) trained cleanly through
all 80 epochs, so this is a 12-ch-specific instability rather than a global
optimizer issue.

## Reproduce

```
# dataset (3.4 h on RTX 4080 Laptop)
python ml/gen_dataset.py <scene.json> ml/data_v3 --n 150 --clean-spp 1024 --gbuffer

# diagnostic ladder (each ~50 min)
python ml/train.py ml/data_v3 --out ml/runs/v3_rgb.pt      --optim adamw_cosine
python ml/train.py ml/data_v3 --out ml/runs/v3_adam.pt
python ml/train.py ml/data_v3 --out ml/runs/v3_awc.pt      --optim adamw_cosine --ema-decay 0
python ml/train.py ml/data_v3 --out ml/runs/v3_ema099.pt   --optim adamw_cosine --ema-decay 0.99
python ml/train.py ml/data_v3 --out ml/runs/v3_gb.pt       --optim adamw_cosine --ema-decay 0 --gbuffer

python tools/eval_checkpoints.py ml/eval_v2/noisy_8.png ml/eval_v2/gt_1024.png \
    ml/runs/v2_rgb.pt ml/runs/v2_gb.pt \
    ml/runs/v3_rgb.pt ml/runs/v3_adam.pt ml/runs/v3_awc.pt ml/runs/v3_ema099.pt ml/runs/v3_gb.pt
```

Logs in `ml/runs/log_v3_*.txt`. Determinism: seed=42 fixed; reruns reproduce
these numbers exactly.
