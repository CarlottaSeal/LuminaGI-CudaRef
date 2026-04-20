# Ray sort between bounces — implementation plan

ncu's top speedup lead for `accumulate_kernel` is reducing uncoalesced
global memory accesses (70% of sectors are excessive). The root cause is
warp divergence after the first bounce: neighboring pixels traverse
different BVH branches, so their triangle / material fetches scatter.

The textbook fix is to **reorder rays by direction coherence between
bounces** so each warp processes rays that mostly visit the same nodes.

## Status

- [x] thrust dependency verified — `thrust_smoke.cu` builds and runs under
      the project's nvcc + MSVC setup, with `-Xcompiler /Zc:preprocessor`
- [ ] Multi-kernel refactor (primary + shade passes split by buffer)
- [ ] Sort step between passes
- [ ] Measure against single-kernel baseline

## Sketch

**Ray queue record (40 B):**

```cpp
struct QueuedRay {
    float3   origin;
    float3   dir;
    float3   throughput;
    uint32_t pixel_idx;
    uint32_t sort_key;   // Morton-encoded quantized dir
};
```

For 1728 × 864 = 1.5 M rays, a double buffer is ~120 MB — fits on a 12 GB
GPU with room to spare.

**Per-spp pipeline:**

```
primary_kernel        → writes QueuedRay[bounce=1]
thrust::sort_by_key   on sort_key
bounce_kernel         → reads sorted QueuedRay[bounce=1]
                       → writes QueuedRay[bounce=2]
thrust::sort_by_key
bounce_kernel
...
```

**Sort key:** Morton-encode the direction vector. With 10 bits per axis,
quantized against a unit cube mapping, you get a 30-bit key that clusters
nearby directions to nearby indices. `thrust::sort_by_key` on an int[] key
runs in low ms for 1.5 M entries.

## What to measure

- Kernel time with / without sort (same spp, same bounces)
- ncu L2 Cache Throughput, Branch Efficiency, Uncoalesced Global Access
  — expect L2 drops below 90%, uncoalesced sectors drop meaningfully
- Per-bounce kernel time breakdown — the sort overhead needs to be amortised

## What could make this a null result

- Ada's 40 MB L2 already absorbs most of the repeat-access cost, same way
  it ate the shmem BVH cache. The 70% ncu estimate is computed assuming
  memory-latency-limited execution; if the kernel is actually
  memory-BW-limited on already-cached data, sort buys nothing.
- Multi-kernel launch overhead + extra global reads/writes for the queue
  could cancel the coherence win.

Worth trying because the ceiling is high; worth measuring because it's a
known "obvious optimization that sometimes doesn't help" in GPU raytracing.
