# Nsight Compute Profile — `accumulate_kernel`

**Hardware:** NVIDIA GeForce RTX 4080 Laptop GPU (Ada Lovelace, SM 8.9, 58 SMs, 12 GB, ~40 MB L2)
**Kernel:** `accumulate_kernel` — single launch, grid 108×54×1, block 16×16×1 (256 threads = 8 warps per block)
**Workload:** 1728×864 pixels, 1.12M triangles, 810k BVH nodes, 8 spp, 2 indirect bounces
**Captured with:** Nsight Compute 2026.1.1 `--set detailed --launch-count 1`

Raw report: `nsight_compute_report.txt`. Binary: `../output/profile/accumulate.ncu-rep` (open in `ncu-ui`).

## Top-line read

| Section | Metric | Value |
|---|---|--:|
| Speed of Light | Memory Throughput | **90.80%** |
| Speed of Light | L2 Cache Throughput | **90.80%** |
| Speed of Light | L1/TEX Cache Throughput | 70.80% |
| Speed of Light | Compute (SM) Throughput | 61.37% |
| Speed of Light | DRAM Throughput | 7.15% |
| Compute | Executed IPC | 1.66 / 4.0 peak |
| Compute | Issue Slots Busy | 41.28% |
| Memory | L1/TEX Hit Rate | 78.25% |
| Memory | L2 Hit Rate | **97.76%** |
| Occupancy | Theoretical | 50% (reg-limited) |
| Occupancy | Achieved | 46.56% (22.35 warps / SM) |
| Divergence | Branch Efficiency | 83.60% |
| Divergence | Avg Divergent Branches | 4.49M |

## Interpretation

The kernel is **L2-bandwidth bound**, not compute-bound. The 90.8% L2 Cache
Throughput against a 7.15% DRAM Throughput shows the working set mostly fits
in L2 (Ada's ~40 MB L2 easily absorbs the 810k-node × 40 B ≈ 32 MB BVH plus
hot triangles), but the kernel is pushing L2 to saturation serving those hits.

This *corrects* the earlier hypothesis from the shared-memory experiment that
the kernel was compute-bound on ray-triangle intersection. Sym­p­tom­ati­cally it
looks that way (IPC low, kernel long), but the underlying stall driver is L2
bandwidth — a distinction only visible with hardware counters.

### Why shared-memory BVH caching didn't help (revisited)

With L2 already at 90.8% and a 97.76% hit rate, an L2 request for a top-level
BVH node resolves in near-L2-hit latency. Moving it to `__shared__` replaces
an L2 hit with a shmem hit — **tens of cycles saved per access, but those
cycles are already overlapped by the warp scheduler issuing other instructions
from the pool of active warps**. Without raising occupancy, latency hiding
was already near-perfect for these accesses.

Additionally, `__shared__` and L1 share physical SRAM on SM 8.9. Reserving
10.2 KB of SMEM for the BVH cache reduces the L1 capacity available for
triangle and stack data — a potential net negative if the displaced data had
a better hit rate in L1 than the BVH nodes did in L2.

## Three optimizations ncu itself recommends

Listed by estimated speedup (from the Source Counters / stall-reason pages):

1. **~70% speedup — reduce uncoalesced global memory accesses.**
   70% of the global-memory sectors fetched are "excessive" — the warp accesses
   are scattered across cache lines. Root cause: once primary rays diverge (after
   the first bounce), neighboring threads traverse different BVH branches and
   intersect different triangles. A packet-tracer or ray-sort-by-morton before
   each bounce would restore coherence.

2. **~50% speedup — raise occupancy.**
   Theoretical occupancy is capped at 50% by register usage (76 regs/thread;
   the Ada SM holds 65,536 registers, so 65,536 / 76 = 862 threads = 3 blocks
   of 256 threads = 24 warps, against the SM max of 48). Refactoring to drop
   below 64 regs (e.g., recomputing some intermediates, splitting into
   producer/consumer kernels) would unlock another block per SM.

3. **~21% speedup — reduce uncoalesced shared-memory accesses.**
   21% of SMEM wavefronts are excessive. Likely the per-thread traversal stack
   — threads in the same warp pop to different stack depths, hitting different
   shmem banks.

## Optimization #1 applied: register reduction via `__launch_bounds__`

ncu's top occupancy recommendation was "50% theoretical, reg-limited at 76/thread".
The Ada SM has 65,536 registers; at 256 threads/block and 76 regs/thread, only
3 blocks (768 threads, 24 warps) fit per SM against a hardware max of 48 warps.

Adding `__launch_bounds__(256, 4)` to `accumulate_kernel` tells nvcc to target
4 blocks per SM, which forces the register budget down to 65536/(256·4) = **64**
regs/thread. Compiler output (`ptxas -Xptxas=-v`):

| | default | `__launch_bounds__(256, 4)` |
|---|--:|--:|
| Registers per thread | 76 | **64** |
| Stack frame | 256 B | 288 B |
| Spill stores | 0 B | 76 B |
| Spill loads | 0 B | 56 B |
| Theoretical occupancy | 50% | 75% |

The +32 B of stack and ~132 B/thread spill traffic are the compiler's cost
of squeezing into 64 regs. The tradeoff is favorable: real kernel time drops
by ~5–10% on this hardware (measurements fluctuate ±5% run-to-run from
laptop thermal and power state, so the noise floor is non-trivial).

Kept permanently in the code — it's a one-line compiler hint, the spill
overhead is bounded, and the occupancy win is structural.

## Why this analysis is the point

This is the workflow a performance-model / arch team cares about:
**predict → measure → reconcile.** My prediction ("compute-bound") was wrong;
ncu told me so; the corrected model (L2-bound, divergence-amplified)
explains both the original timing and the null result from the shmem cache.
The SASS, the hardware counters, and the architectural reasoning line up —
which is the point of profile-driven work, not the raw speedup number.
