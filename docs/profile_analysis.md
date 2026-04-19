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

This contradicts the initial guess (from the shmem experiment) that the kernel
was compute-bound on ray-triangle intersection. The symptoms (low IPC, long
kernel time) look the same; only the hardware counters separate the two.

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

### Post-fix ncu metrics

Reprofiled with the same `--set basic --launch-count 1` invocation. Raw
report: `accumulate_lb4.ncu-rep`. Full text: `nsight_compute_report_lb4.txt`.

| Metric | Before (76 regs) | After (64 regs + LB) | Δ |
|---|--:|--:|--:|
| Registers / thread | 76 | 64 | -16% |
| Theoretical occupancy | 50.0% | 66.7% | +17 pp |
| Achieved occupancy | 46.6% | 61.8% | +15 pp (+33% more active warps) |
| Block Limit (regs) | 3 / SM | 4 / SM | +33% |
| L2 Cache Throughput | 90.8% | **93.4%** | +2.6 pp |
| DRAM Throughput | 7.2% | 9.7% | +2.5 pp |
| Compute (SM) Throughput | 61.4% | 53.2% | −8 pp |

More active warps → more concurrent memory requests → L2 even more saturated.
SM throughput dropped because the same compute is now spread across more warps
that each individually stall more on memory; the kernel didn't get less efficient,
the bottleneck just became more obviously L2 bandwidth.

Theoretical occupancy stopped at 66.7% rather than the register-math-optimal
75% because the new limiter is **shared memory per block** (10.2 KB × 4 blocks
= 40.8 KB per SM). If occupancy mattered more than it does here, the next
step would be shrinking the BVH shmem cache or eliminating it entirely.
Since the kernel is already L2-bound, the real next optimization is reducing
L2 traffic via ray coalescence — not chasing more occupancy.

## SASS instruction mix (cuobjdump)

Dumping `accumulate_kernel` with `cuobjdump --dump-sass build/bin/cuda_ref.exe`
produces 1,560 SM 8.9 instructions. Histogram of the top opcodes:

| Instruction class | Count | % |
|---|--:|--:|
| FFMA / FMUL / FADD / FMNMX / MUFU.RCP (FP math) | ~460 | ~30% |
| LDG.E (global loads) | 85 | 5.4% |
| **BSSY / BSYNC (warp convergence barriers)** | **134** | **8.6%** |
| STL (spill stores) | 32 | 2.1% |
| CALL.REL.NOINC | 35 | 2.2% |
| MUFU.RCP (reciprocal SFU) | 28 | 1.8% |
| IMAD / IADD3 / LOP3 (integer) | ~200 | ~13% |
| BRA / FSETP / ISETP (branches + predicates) | ~120 | ~8% |

Full dump: `kernel.sass.txt`.

The 8.6% BSSY/BSYNC slice is the clearest hardware-visible symptom of warp
divergence: every non-uniform branch emits a convergence barrier on Ada, so
1,560 instructions include 134 pure sync ops. Branch Efficiency of 83.6%
(ncu) and 4.5 M divergent branches per kernel invocation corroborate this.
Low LDG count (5.4%) is misleading — memory instructions are "fat" in
cycles, so a small count can still saturate L2 bandwidth, which is what
ncu shows.

## Tonemap A/B — kept sqrt + clamp

Tried three output curves at 64 spp / 2 bounces:

| Tonemap | PSNR | SSIM | MAD |
|---|--:|--:|--:|
| **sqrt + clamp** (kept) | 21.80 dB | **0.631** | 13.43 |
| Reinhard (white=4) + sRGB | 21.87 dB | 0.557 | 14.19 |
| sRGB encode only | 22.16 dB | 0.559 | 13.80 |

PSNR moves by tenths of a dB; SSIM drops ~10%. LuminaGI's backbuffer is
close to sqrt gamma, so switching to a display-referred sRGB curve or a
Reinhard roll-off pushes the reference further from the engine instead of
closer. The metric this project cares about is reference-vs-engine
similarity, so sqrt stays. The measured table is repeated in a comment
above `tonemap_kernel`.

