# Epoch sweep — UNet denoiser

Why is the training capped at 80 epochs? This sweep checks both directions:
40 (could we stop earlier?) and 120 (could we train longer?). Same seed (42)
and otherwise identical config to the baseline (Adam 1e-3, 4-level UNet,
base 32, batch 4, L1 loss, 50 poses with 80/20 split). With the seed fixed
the runs are deterministic, so the first N epochs of any run reproduce the
baseline's first N exactly.

## Results

| epochs | best val L1 | best @ epoch | final-epoch state                       |
| ------ | ----------- | ------------ | --------------------------------------- |
| 40     | 0.0070      | 33           | still improving (final val 0.0083)      |
| **80** | **0.0054**  | **74**       | **train 0.0063 / val 0.0070, no diverge** |
| 120    | 0.0054      | 74           | **diverged at epoch 83**, train 0.10    |

Best checkpoints: `ml/runs/ep40.pt`, `ml/runs/denoiser_rerun.pt` (baseline),
`ml/runs/ep120.pt`. Logs: `ml/runs/log_ep40.txt`, `ml/runs/train_log.txt`,
`ml/runs/log_ep120.txt`.

## What each run shows

**40 epochs.** Best val L1 0.0070 at epoch 33, then 7 more epochs of
fluctuation that never reach a new best. Stops with the model still some
distance from the baseline floor — about 30% worse on val L1. Confirmation
that 40 is genuinely too short, not just an arbitrary halving.

**80 epochs (baseline).** Documented in `denoiser.md` and `lr_sweep.md`.
Best val L1 = 0.0054 at epoch 74, last 6 epochs no further improvement.
Train continues to descend (0.0063 at epoch 80) while val fluctuates in
0.0054–0.0094 — mild overfitting beginning, but model still healthy.

**120 epochs.** Through epoch 82 the run tracks baseline exactly (seed
determinism). Then at epoch 83 something abrupt happens:

```
epoch  82/120  train=0.0074  val=0.0064
epoch  83/120  train=0.1059  val=0.0802
epoch  84/120  train=0.0998  val=0.0764
epoch  85/120  train=0.1193  val=0.1124
...
epoch 120/120  train=0.1005  val=0.1093
```

Train loss jumps roughly 15× in a single epoch and the model never
recovers — 37 more epochs of oscillation between train 0.09 and 0.12, val
0.07 and 0.14. Same dynamic as the lr=2e-3 failure in `lr_sweep.md`, only
arriving late instead of from the start.

## Why this happens (best guess)

Adam normalises each parameter's update by `sqrt(v) + ε`, where `v` is an
EMA of past squared gradients with `β2 ≈ 0.999`. Late in training, `v` for
many parameters has decayed toward the typical near-converged gradient
magnitude. A rare batch producing an unusually large gradient on some
parameter then yields a disproportionately large update step (because `v`
is still small for that parameter relative to the new gradient), which can
push the parameter outside the local minimum. Once outside, gradients grow,
`v` lags, the next steps stay too large, and the run cascades into the
oscillating regime above. The fixed-lr Adam recipe has no warmup or decay
schedule to dampen this; published mitigations include AdamW with weight
decay, a cosine lr schedule, or gradient clipping.

This is a known late-training instability in vanilla Adam, not specific to
this model.

## Implications for the saved checkpoint

The training loop saves the *best* validation checkpoint, not the *latest*
weights (`train.py:144-148`). Without that, the 120-epoch run would have
ended with epoch-120 weights — train loss 0.10, completely useless for
inference. The 80-epoch run was effectively the safe cutoff: short enough
to stay inside Adam's stability region, long enough to reach the val L1
floor. 50–80 would all give roughly the same result; 100+ starts gambling.

## How the sweep was run

```
python ml/train.py ml/data --epochs 40  --out ml/runs/ep40.pt  2>&1 | tee ml/runs/log_ep40.txt
python ml/train.py ml/data --epochs 80  --out ml/runs/denoiser_rerun.pt 2>&1 | tee ml/runs/train_log.txt
python ml/train.py ml/data --epochs 120 --out ml/runs/ep120.pt 2>&1 | tee ml/runs/log_ep120.txt
```

Total wall time: under 5 minutes on RTX 4080 Laptop.
