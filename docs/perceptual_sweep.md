# Perceptual loss sweep — honest negative result

The L1-only baseline (`denoiser.pt`) gets PSNR 42.04 dB / SSIM 0.964 on
the held-out 1728×864 frame. The hypothesis going in was that adding a
VGG perceptual term would trade a small PSNR loss for a visible SSIM /
edge-quality gain, since pure L1 is known to over-smooth.

This sweep tested that. It did not work on this dataset.

## Setup

VGG16 (ImageNet weights), L1 between features at relu1\_2, relu2\_2, and
relu3\_3, summed. Total loss `L = L1 + λ · L_perceptual`. All else held
fixed against the baseline (Adam lr=1e-3, 80 epochs, seed 42, 40/10
train/val split, 256×256 random crops). Best checkpoint selected on val
L1 — the same metric the baseline uses, so checkpoints are comparable.

| Run | λ | Best ckpt epoch | Best val L1 | Notes |
|---|--:|--:|--:|---|
| baseline | 0   | 80 | 0.0054 | reference run |
| perc05   | 0.05 | 65 | 0.0060 | diverged at epoch ~72 |
| perc10   | 0.1  | 78 | 0.0062 | clean, no divergence |
| perc20   | 0.2  | 15 | 0.0146 | diverged at epoch 16, barely trained |

## Held-out frame (1728×864, vs 1024-spp ground truth)

| Variant | PSNR (dB) | SSIM | Comment |
|---|--:|--:|---|
| 8 spp raw       | 34.52 | 0.815 | input |
| **L1 baseline** | **42.04** | **0.964** | |
| L1 + 0.05·perc  | 41.08 | 0.951 | |
| L1 + 0.10·perc  | 40.92 | 0.951 | |
| L1 + 0.20·perc  | 34.17 | 0.834 | worse than raw input |

**Both PSNR and SSIM regressed under every perceptual weight tested.**
The expected SSIM gain did not appear; instead the perceptual term moved
the optimum away from L1 and the L1 metric got worse without anything
showing up to compensate.

## Why this likely failed (best guess for interview)

Three reasons, in order of suspicion:

1. **The "clean" target still has visible Monte Carlo noise.** It is
   256-spp, not converged. VGG feature maps are sensitive to high-frequency
   structure; matching feature L1 with a noisy target means the model is
   asked to reproduce that noise, not just the signal. L1 in pixel space
   averages noise out; perceptual L1 in feature space does not.
2. **Tiny training set.** 40 train pairs. Perceptual loss is normally
   useful as a regularizer that biases the network toward "natural-looking"
   outputs given enough data. Here it just adds gradient noise.
3. **Domain mismatch.** VGG16 was trained on ImageNet (natural photos).
   The scene here is a single indoor CG render with simple shading and
   no real-world textures. The features VGG cares about are not
   necessarily the features that distinguish a good denoise from a bad
   one in this domain.

A real fix for any of these is more work than the sweep itself: render
an actually-converged target (4096+ spp), grow the dataset by 10×, or
fine-tune VGG features on path-traced renders. None are in scope.

## Divergence note (cross-references the lr/epoch sweep)

`perc05` diverged at epoch ~72 and `perc20` at epoch 16. Same dynamic as
the `lr=2e-3` and `epoch=120` runs in `epoch_sweep.md` and `lr_sweep.md`:
late-training Adam run-away once the second-moment estimate `v` lags a
sudden gradient spike. The best-checkpoint mechanism in `train.py` was
load-bearing for both perceptual runs — final-epoch weights would be
useless. Higher λ shifts the divergence earlier, consistent with the
larger effective gradient magnitude the perceptual term introduces.

## Reproduce

```
python ml/train.py ml/data --epochs 80 --perceptual-weight 0.05 --out ml/runs/perc05.pt
python ml/train.py ml/data --epochs 80 --perceptual-weight 0.1  --out ml/runs/perc10.pt
python ml/train.py ml/data --epochs 80 --perceptual-weight 0.2  --out ml/runs/perc20.pt
python tools/eval_checkpoints.py ml/eval/noisy_8.png ml/eval/gt_1024.png \
    ml/runs/denoiser.pt ml/runs/perc05.pt ml/runs/perc10.pt ml/runs/perc20.pt
```

Logs in `ml/runs/log_perc{05,10,20}.txt`, checkpoints in `ml/runs/perc{05,10,20}.pt`.
Determinism: seed=42 fixed; reruns reproduce these numbers exactly.
