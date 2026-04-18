# LuminaGI-CudaRef

A CUDA reference path tracer used as ground truth for validating
[LuminaGI](https://github.com/CarlottaSeal)'s real-time global illumination output.

LuminaGI is a DX12 GI engine using screen-space probes, voxel lighting, and a
surface card cache. Its output is always an *approximation*. This project provides
a brute-force, physically-straight renderer on CUDA so any LuminaGI frame can
be diffed against a reference image — giving PSNR/SSIM numbers and per-pixel
heatmaps to tell you *where* and *how much* the approximation diverges.

| LuminaGI (engine) | CUDA reference (this repo) | Abs-diff heatmap |
|---|---|---|
| ![engine](docs/engine.png) | ![reference](docs/reference.png) | ![heatmap](docs/heatmap.png) |

*Test scene: 62 meshes, 1.12M triangles, 3 lights (1 sun + 2 point)*

## Pipeline

```
LuminaGI (DX12)           │   LuminaGI-CudaRef (this repo)
──────────────────────────┼────────────────────────────────────────
F9 ─┬─► screenshot.png    │
    └─► scene.json ───────┼──► LoadScene  ─► Build LBVH (150 ms)
                          │                      │
                          │                      ▼
                          │                  CUDA kernel (4 ms)
                          │                      │
                          │                      ▼
                          │                  reference.png
                          │                      │
        screenshot.png ───┼──► tools/diff.py ◄───┘
                          │          │
                          │          ▼
                          │     HTML report (PSNR / SSIM / heatmap)
```

`Scene::DumpToJSON()` on the engine side serialises transformed triangle soup +
lights + camera. `validate.py` orchestrates the render + diff end-to-end.

## Numbers

Hardware: RTX 4080 Laptop (SM 8.9, 58 SMs, 12 GB).

| Stage | Time |
|---|--:|
| JSON load (1.1M tris, ~300 MB) | 12 s (host) |
| LBVH build (Morton + radix split) | 150 ms (CPU) |
| Render kernel, direct only (1 spp) | **4 ms** |
| Render kernel, full GI (256 spp, 2 bounces) | **8.7 s** |
| Image diff (PSNR / SSIM / heatmap) | 1.5 s |

Current numbers against LuminaGI screenshot (same scene, 1.1M tris, 64 spp, 2 bounces):

| Metric | Value |
|---|--:|
| PSNR | **21.8 dB** |
| SSIM | **0.631** |
| Mean abs diff | 13.4 / 255 |

Remaining gap is systematic: LuminaGI has an ambient term, normal-map detail,
and a specific tonemap this reference doesn't model. That's intentional — the
point is physically-correct ground truth, not a pixel clone of the engine.

Two silent bugs the diff pipeline has caught so far:
1. `Scene::DumpToJSON` double-applied the mesh transform because
   `MeshObject::GetWorldMatrix()` already includes it; meshes rendered at origin
   in the reference but not in the engine. Fix lifted PSNR from 12.7 dB to 20.5 dB.
2. `DX12Renderer::CreateTextureFromImage` never propagated the source image's
   path to the resulting `Texture`, so GLB-embedded diffuse textures (chess piece
   materials) dumped as empty paths. Fixing the name propagation plus a PNG-export
   step on GLB load lifted PSNR from 20.5 dB to 21.8 dB.

## Build

Requires CUDA Toolkit 13.x and MSVC v143 (Visual Studio 2022). RTX card with
CC ≥ 7.0 for `sm_89` arch.

```bat
REM CUDA smoke test
build_hello.bat

REM Full reference renderer
build_cuda_ref.bat
```

## Run

```bat
REM in LuminaGI, press F9 during gameplay — produces:
REM   Run/Screenshots/manual_<stamp>.png
REM   Run/Screenshots/manual_<stamp>.json

REM one-shot validation
python tools/validate.py path\to\manual_<stamp>.json --open
```

Or manually:

```bat
REM direct lighting only, 1 spp — fast preview
build\bin\cuda_ref.exe scene.json -o ref.png --spp 1 --bounces 0

REM full GI, 256 spp with 2 indirect bounces
build\bin\cuda_ref.exe scene.json -o ref.png --spp 256 --bounces 2

python tools\diff.py engine.png ref.png --out report.html
```

## Layout

```
include/
  scene.h        host-side scene struct + LoadSceneJSON
  bvh.h          BvhNode (shared host/device)
  vecmath.h      vec3 + ray, __host__ __device__ math
src/
  scene_loader.cpp   JSON parse (nlohmann::json)
  bvh.cpp            LBVH build: Morton codes → radix split
  pathtracer.cu      CUDA kernel: camera setup, BVH traversal, NEE shading
  main.cpp           driver (argv → render → PNG)
  test_load_scene.cpp / test_bvh.cpp   pure-host sanity tests
tools/
  diff.py       PSNR / SSIM / heatmap → HTML report
  validate.py   end-to-end: cuda_ref.exe + diff.py on a scene.json
third_party/
  nlohmann/json.hpp
  stb_image_write.h
```

## Status

- [x] Scene JSON dump from LuminaGI (F9 key or `--screenshot` flag)
- [x] Host-side scene loader + BVH build (Morton + radix split, 150 ms for 1.1M tris)
- [x] CUDA path tracer: primary rays, BVH traversal, shadow rays
- [x] Diffuse texture sampling (CUDA texture objects, sRGB decode, bilinear)
- [x] Indirect bounces: cosine-weighted hemisphere, Russian roulette, progressive accumulation
- [x] Image diff tool (PSNR / SSIM / heatmap)
- [x] One-shot validation pipeline
- [ ] Binary scene format (JSON is slow at 300 MB / 12 s parse)
- [ ] Variance-aware accumulation / adaptive sampling
- [ ] HDR Reinhard tonemap + proper sRGB encode

## Nsight Compute profile (RTX 4080, Ada Lovelace SM 8.9)

Full analysis: [`docs/profile_analysis.md`](docs/profile_analysis.md). Raw
report: [`docs/accumulate.ncu-rep`](docs/accumulate.ncu-rep) (open in `ncu-ui`).

Top-line numbers for `accumulate_kernel`:

| Section | Metric | Value |
|---|---|--:|
| Speed of Light | L2 Cache Throughput | **90.8%** |
| Speed of Light | Compute (SM) Throughput | 61.4% |
| Speed of Light | DRAM Throughput | 7.2% |
| Memory | L1 Hit Rate | 78.3% |
| Memory | L2 Hit Rate | 97.8% |
| Occupancy | Theoretical / Achieved | 50% / 46.6% (reg-limited at 76 regs/thread) |
| Divergence | Branch Efficiency | 83.6% |

The kernel is **L2-bandwidth bound**, not compute-bound (correcting my earlier
guess). Working set fits in Ada's ~40 MB L2 — DRAM barely touched — but L2
itself is saturated serving hits. ncu identifies three optimization paths:
reduce uncoalesced global accesses (~70% potential speedup via ray sorting),
raise occupancy (~50% via register reduction), and fix uncoalesced SMEM
stack accesses (~21%).

## Performance notes

A BFS relayout + `__shared__` cache of the top 255 BVH nodes (top 8 levels)
is implemented behind `-DBVH_USE_SHMEM`. A/B benchmark on the test scene
(64 spp, 2 bounces, 3 runs each): **3336 ms off, 3343 ms on — essentially zero
difference**.

Why it didn't help:
- RTX 4080 has ~40 MB L2 cache. The top BVH levels are permanent L2 residents
  under path-tracing workloads, so a manual shmem cache duplicates what L2
  already does for free.
- The kernel is compute-bound on ray-triangle intersection, `cosf`/`sinf`/`sqrtf`
  in hemisphere sampling, and RNG — not memory-bound on BVH loads.
- Path tracing has high warp divergence after the first bounce; neighboring
  pixels don't traverse the same nodes coherently, which weakens the spatial
  locality a shmem cache relies on.

Kept in the tree as a toggle because "implemented and measured to not help"
is a better answer than "didn't try".

## Notes on the engine side

Three changes to LuminaGI / Engine to make the dump work:
- `Camera::GetPerspectiveFOV/Aspect/Near/Far` getters (4 lines)
- `Scene::DumpToJSON(path, camera, w, h)` (~140 lines)
- F9 handler in `App::RunFrame` — deferred to *after* `EndFrame` because
  `CaptureScreenshot` stomps the D3D12 command list state if run mid-frame
- `AutomatedTesting` auto-emits a `.json` whenever `--screenshot` fires

The deferred-capture bug was the one surprise of the integration — the first
version crashed on the next frame's `BindConstantBuffer` call because
`CaptureScreenshot` internally closes and resets the command list, discarding
any state the normal render path expected to be intact. Moving the call to
after `EndFrame()` fixed it.
