# Ada Lovelace SM 8.9 — quick reference

Notes I keep when tuning or reading ncu. Written against the RTX 4080 Laptop
GPU (AD104 cut-down, the hardware this project profiles on), but most of it
generalises across the Ada RTX 40-series.

## Die-level (RTX 4080 Laptop, AD104)

| | |
|---|--:|
| SMs | 58 |
| FP32 lanes | 7,424 (128 per SM) |
| Boost clock | ~2,280 MHz |
| L2 cache | ~40 MB (shared across SMs) |
| Memory | 12 GB GDDR6, 192-bit, ~384 GB/s peak |
| RT cores | 3rd-gen, 58 (one per SM) |
| Tensor cores | 4th-gen, 232 (four per SM) |

Desktop AD102 (RTX 4090) is a bigger version: 128 SMs, 72 MB L2, 384-bit GDDR6X.
Same SM microarch — code tuned for SM 8.9 on a Laptop part scales up.

## Inside an SM (SM 8.9)

```
┌── SM ────────────────────────────────────────────────────────────┐
│  ┌─ SMSP 0 ─┐ ┌─ SMSP 1 ─┐ ┌─ SMSP 2 ─┐ ┌─ SMSP 3 ─┐              │
│  │ 32 FP32  │ │ 32 FP32  │ │ 32 FP32  │ │ 32 FP32  │              │
│  │ 16 INT32 │ │ 16 INT32 │ │ 16 INT32 │ │ 16 INT32 │              │
│  │ 1 SFU    │ │ 1 SFU    │ │ 1 SFU    │ │ 1 SFU    │              │
│  │ 1 LD/ST  │ │ 1 LD/ST  │ │ 1 LD/ST  │ │ 1 LD/ST  │              │
│  │ 1 Tensor │ │ 1 Tensor │ │ 1 Tensor │ │ 1 Tensor │              │
│  │ scheduler│ │ scheduler│ │ scheduler│ │ scheduler│              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
│                                                                   │
│  Register file: 65,536 × 32-bit  (256 KB, shared across SMSPs)    │
│  L1 / shmem:    128 KB unified   (split configurable at kernel)   │
│  RT core:       1 (shared)                                        │
└───────────────────────────────────────────────────────────────────┘
```

**Peak instruction throughput:** 4 insts / cycle / SM — one per SMSP scheduler.
Anything higher is a counter artifact (or a fused op).

**Warp limits:**

| | |
|---|--:|
| Max warps per SM | 48 |
| Max warps per SMSP scheduler | 12 |
| Max blocks per SM | 24 |
| Max threads per block | 1,024 |
| Warp size | 32 |

## Occupancy math (the part you do in your head)

Theoretical occupancy = `min(reg_limit, shmem_limit, block_limit, warp_limit)`.

For a block of `T` threads using `R` registers and `S` bytes of shmem:

| Limit | Formula |
|---|---|
| Registers | `floor(65536 / (R · T))` blocks per SM |
| Shared memory | `floor(102400 / S)` blocks per SM (100 KB usable) |
| Block count | up to 24 blocks per SM |
| Warp count | up to `48 / (T / 32)` blocks per SM |

Occupancy = `blocks_per_SM · T / 32 / 48`.

**This project's `accumulate_kernel`:**
- T = 256, R = 64 (after `__launch_bounds__(256, 4)`), S = 10,240 B (shmem BVH cache)
- Reg limit: floor(65536 / (64·256)) = **4 blocks**
- Shmem limit: floor(102400 / 10240) = **10 blocks**
- Warp limit: floor(48 / 8) = **6 blocks**
- Block limit: 24
- `min = 4` → 4 blocks × 8 warps / 48 = **66.7% theoretical**

Without `__launch_bounds__` (R = 76) the reg limit drops to 3 blocks → 50%.

## Memory hierarchy — typical latencies

| Level | Size | Latency (cycles) |
|---|--:|--:|
| Registers | 256 KB / SM | 1 |
| Shared memory | up to 100 KB / SM | ~25 |
| L1 (same pool as shmem) | same | ~30 |
| L2 | ~40 MB die-wide | ~250 |
| GDDR6 | 12 GB | ~500 |

L2 on Ada is ~12× bigger than Ampere (RTX 3080 had 5 MB). Workloads that
looked memory-bound on Ampere often become L2-resident on Ada — which is
exactly why the shmem BVH cache in this project gave zero measurable win:
the top-level nodes were already in L2.

## Reading ncu — which number to look at first

1. **Speed of Light / Compute vs Memory Throughput**: the bigger of the two
   tells you which side the kernel leans on.
2. **DRAM vs L2 Throughput**: if L2 is high and DRAM is low, the working
   set fits in L2 — the bottleneck is L2 bandwidth, not HBM BW. (This is
   this project's kernel, 90.8% L2 vs 7.2% DRAM.)
3. **Executed IPC vs peak (4)**: if IPC ≪ 4, look at stall reasons next.
4. **Warp Stall Reasons** (Source Counters page):
   - `Short Scoreboard` / `Long Scoreboard` — memory latency
   - `Wait` — pipeline dependency (MUFU / Tensor chains)
   - `Not Selected` — occupancy
5. **Theoretical vs Achieved occupancy**: if Achieved ≪ Theoretical, launch
   imbalance; if Theoretical is already low, the kernel is reg- or
   shmem-limited — fix those first, don't chase launch-config.
6. **Branch Efficiency**: < 90% means real divergence. On Ada the cost
   shows up as `BSSY`/`BSYNC` convergence barriers in SASS.

## Ada-specific features (mention-worthy in interview)

- **Shader Execution Reordering (SER)** — hardware sorts in-flight shader
  invocations by some key (material ID, ray type) before dispatch to reduce
  divergence. Exposed through OptiX 7.6+. Conceptually what a ray-sort pass
  does in software.
- **Opacity / Displacement micromaps** — RT core extensions that let the
  hardware skip transparent fragments or trace against sub-triangle detail
  without blowing up BVH size.
- **4th-gen Tensor cores with FP8** (E4M3 / E5M2) — inference workhorse.
- **Larger L2** — see above; changes what "memory-bound" means in practice.

## Glossary

- **SM** — streaming multiprocessor
- **SMSP** — SM sub-partition (1 of 4 per SM; owns a warp scheduler)
- **SFU** — special function unit (rcp, rsq, sin, cos, log, exp)
- **LD/ST** — load/store unit
- **ITS** — independent thread scheduling (Volta+), needs BSSY/BSYNC for convergence
- **TDR** — Windows timeout detection & recovery (~2 s default)
