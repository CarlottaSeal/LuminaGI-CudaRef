# G-buffer aux input — honest negative result

NRD-style production denoisers feed the network albedo / normal / depth
(or world position) alongside the noisy color. The standard expectation
is that aux features factor out shading, expose surface boundaries, and
let the network do something close to a learned cross-bilateral filter
— so PSNR / SSIM should improve, especially at material edges.

This sweep tested that. It did not work on this dataset.

## Setup

Channels match what LuminaGI's deferred path actually exposes (see
`Run/Data/Shaders/CompositeShader.hlsl` registers t200–t203):

- `g_GBufferAlbedo`   — RGB
- `g_GBufferNormal`   — RGB encoded `(n+1)/2`
- `g_GBufferWorldPos` — RGB, AABB-normalised so an 8-bit PNG round-trips

The `g_GBufferMaterial` slot (roughness / metallic / ao) is not
modelled — the CUDA reference is Lambertian-only, so there's no PBR
data to populate it.

UNet input: 12 channels (3 noisy RGB + 3 albedo + 3 normal + 3 worldpos),
otherwise identical to the RGB-only baseline (4-level UNet, 32 base ch,
1.95 M params, residual on the noisy RGB only). Same training recipe:
50-pose dataset (40 train / 10 val), Adam lr=1e-3, 80 epochs, seed 42.
G-buffer kernel runs once per render at primary hit.

## Results on a held-out 1920×1200 frame (vs 1024-spp ground truth)

The held-out camera is the scene's original F9-dump pose, not in the
training distribution.

| Variant | infer ms | PSNR (dB) | SSIM |
|---|--:|--:|--:|
| 8 spp raw | — | 34.31 | 0.803 |
| **RGB-only baseline** | **75.8** | **41.17** | **0.953** |
| RGB + G-buffer (12 ch) | 73.7 | 40.36 | 0.940 |

**Both PSNR (−0.81 dB) and SSIM (−0.013) regressed.** Same shape as
the perceptual-loss sweep — the standard recipe does not help on this
dataset at this scale.

Visually the two denoised outputs are nearly indistinguishable; the
gap shows up in the diff heatmap as slightly higher mean abs diff
(1.72 vs 1.54 / 255).

## Why this likely failed (best guess for interview)

Three reasons, in order of suspicion:

1. **Dataset is too small for the input adapter.** First-conv weights
   went from 3·32·9 = 864 to 12·32·9 = 3456 — 4× the parameters in the
   layer that has to learn what to do with the new channels. With 40
   train pairs, that's not constrained enough to learn aux integration
   from scratch.

2. **Aux channels are noise-free, the model overweights them.** Albedo
   / normal / worldpos at primary hit are deterministic (1-spp suffices,
   no Monte Carlo randomness). For each (noisy, clean) pair the aux is
   the *same* in both. The network can drive train loss down by
   matching aux features rather than denoising the noisy RGB — the
   wrong shortcut.

3. **A vanilla UNet doesn't do what NRD does.** Production aux-input
   denoisers are kernel-prediction networks: they output filter
   weights and the actual filtering is a cross-bilateral convolution
   driven by the aux. A direct-prediction UNet has to discover that
   structure implicitly. With 40 pairs, it doesn't.

Real fixes are scope-sized: kernel-prediction architecture, aux-aware
loss (e.g. extra L1 on `pred · (1 + edge_weight(N, P))`), or an order
of magnitude more data. None are in scope for a one-scene reference.

## Reproduce

```
python ml/gen_dataset.py <scene.json> ml/data_gb --gbuffer
python ml/train.py ml/data_gb --out ml/runs/v2_rgb.pt
python ml/train.py ml/data_gb --gbuffer --out ml/runs/v2_gb.pt
python tools/eval_checkpoints.py ml/eval_v2/noisy_8.png ml/eval_v2/gt_1024.png \
    ml/runs/v2_rgb.pt ml/runs/v2_gb.pt
```

Logs in `ml/runs/log_v2_{rgb,gb}.txt`. Determinism: seed=42 fixed;
reruns reproduce these numbers exactly.
