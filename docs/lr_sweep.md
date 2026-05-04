# Learning rate sweep — UNet denoiser

Sanity check on the choice of `lr=1e-3` for the final denoiser model. Three
runs, identical seed (42) and otherwise identical config (4-level UNet, base
32, batch 4, L1 loss, 80 epochs, 50 poses with 80/20 split), only the
optimizer's learning rate varying.

## Results

| lr     | best val L1 | best @ epoch | final-epoch train | shape                                |
| ------ | ----------- | ------------ | ----------------- | ------------------------------------ |
| 5e-4   | 0.0058      | 78           | 0.0080            | converges ~2× slower, similar floor  |
| **1e-3** | **0.0054** | **74**     | **0.0063**        | **smooth descent, plateau ~ epoch 50** |
| 2e-3   | 0.0634      | 17           | 0.1051            | unstable, never converges            |

Best checkpoints saved as `ml/runs/lr5e-4.pt`, `ml/runs/denoiser_rerun.pt`
(baseline), and `ml/runs/lr2e-3.pt`. Training logs in `ml/runs/log_lr5e-4.txt`,
`ml/runs/train_log.txt`, `ml/runs/log_lr2e-3.txt`.

## What each run shows

**lr=5e-4 (half).** Smooth descent like baseline but lagged by roughly a
factor of two early on — baseline reaches val L1 = 0.0096 by epoch 16, the
half-lr run takes until epoch 27 to reach the same point. By epoch 50 the
two are basically tied (0.0064 vs 0.0065). Final 0.0058 vs baseline 0.0054
is a marginal regression, not a meaningful one. Conclusion: halving lr buys
no quality, costs convergence speed.

**lr=1e-3 (baseline).** Documented in `denoiser.md`. Val L1 drops from
0.0206 at epoch 1 → 0.0149 by epoch 4 (rapid initial descent), → 0.0096 by
epoch 16 (dropping below 0.01), → 0.0075 by epoch 23, → ~0.0065 by epoch 50
(plateau begins), → best 0.0054 at epoch 74. Last 6 epochs produce no
further `*` (no new best); train continues to drop (0.0063) while val
fluctuates 0.0054–0.0094, suggesting mild overfitting begins around epoch 75.
80 epochs is a reasonable cutoff; 50 would also be defensible.

**lr=2e-3 (double).** Diverges. Train loss oscillates wildly:
0.066 → 0.178 → 0.079 → 0.111 → 0.085 → 0.091 → ... never settling.
No `*` after epoch 17 (val=0.0634, the "best" reached only by chance during
oscillation). At final epoch 80, train=0.105 — actively worse than epoch 1.
Adam at this scale + batch + model size cannot keep updates inside the
stable region with lr=2e-3. Validates that 1e-3 is on the right side of
Adam's stability boundary, not just a literature default.

## How the sweep was run

Each run identical except `--lr`:

```
python ml/train.py ml/data --lr 5e-4 --out ml/runs/lr5e-4.pt 2>&1 | tee ml/runs/log_lr5e-4.txt
python ml/train.py ml/data --lr 1e-3 --out ml/runs/denoiser_rerun.pt 2>&1 | tee ml/runs/train_log.txt
python ml/train.py ml/data --lr 2e-3 --out ml/runs/lr2e-3.pt 2>&1 | tee ml/runs/log_lr2e-3.txt
```

Total wall time across the three runs: under 3 minutes on RTX 4080 Laptop.
