#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "vecmath.h"
#include "bvh.h"
#include "scene.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { std::fprintf(stderr, "cuda %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } \
} while (0)

struct DeviceMaterial
{
    float               albedo[3];
    float               emissive[3];
    cudaTextureObject_t tex;   // 0 if no diffuse texture
};

struct DeviceScene
{
    const BvhNode*        nodes;
    const Triangle*       tris;
    const DeviceMaterial* mats;
    const PointLight*     points;
    int                   numPoints;
    DirectionalLight      sun;
};

// Top 255 BVH nodes ( = 8 full levels after BuildBVH's BFS relayout) can live in __shared__.
// Explicit if/else around the fetch, not a ternary on __shared__/__global__ pointers —
// the ternary forces nvcc to the generic address space, which measurably slows loads.
constexpr int kShmemBvhNodes = 255;

#ifdef BVH_USE_SHMEM
#define FETCH_BVH(dst, i) \
    do { if ((i) < kShmemBvhNodes) dst = s_top[i]; else dst = s.nodes[i]; } while (0)
#else
#define FETCH_BVH(dst, i) \
    do { dst = s.nodes[i]; (void)s_top; } while (0)
#endif

struct CameraGPU
{
    vec3  pos;
    vec3  fwd, left, up;   // rows 0/1/2 of camera_to_world (SD convention: x=fwd, y=left, z=up)
    float tanHalfFov;
    float aspect;
    int   width, height;
};

__device__ bool ray_aabb(const Ray& r, const float mn[3], const float mx[3], float tmax_in, float& tmin_out)
{
    float t0 = (mn[0] - r.o.x) * r.inv_d.x;
    float t1 = (mx[0] - r.o.x) * r.inv_d.x;
    float lo = fminf(t0, t1), hi = fmaxf(t0, t1);

    t0 = (mn[1] - r.o.y) * r.inv_d.y;
    t1 = (mx[1] - r.o.y) * r.inv_d.y;
    lo = fmaxf(lo, fminf(t0, t1));
    hi = fminf(hi, fmaxf(t0, t1));

    t0 = (mn[2] - r.o.z) * r.inv_d.z;
    t1 = (mx[2] - r.o.z) * r.inv_d.z;
    lo = fmaxf(lo, fminf(t0, t1));
    hi = fminf(hi, fmaxf(t0, t1));

    if (hi < fmaxf(lo, 0.0f) || lo > tmax_in) return false;
    tmin_out = fmaxf(lo, 0.0f);
    return true;
}

// Möller–Trumbore. Returns true if hit in (kEps, tmax).
__device__ bool ray_tri(const Ray& r, const Triangle& t, float tmax, float& outT, float& outU, float& outV)
{
    constexpr float kEps = 1e-6f;
    vec3 v0 = vfrom(t.v[0]);
    vec3 e1 = vsub(vfrom(t.v[1]), v0);
    vec3 e2 = vsub(vfrom(t.v[2]), v0);
    vec3 p  = cross(r.d, e2);
    float det = dot(e1, p);
    if (det > -kEps && det < kEps) return false;
    float invDet = 1.0f / det;

    vec3 s = vsub(r.o, v0);
    float u = dot(s, p) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    vec3 q = cross(s, e1);
    float v = dot(r.d, q) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    float tt = dot(e2, q) * invDet;
    if (tt <= kEps || tt >= tmax) return false;

    outT = tt; outU = u; outV = v;
    return true;
}

__device__ int traverse_closest(const DeviceScene& s, const BvhNode* s_top,
                                const Ray& r, float tmax,
                                float& outT, vec3& outN, float& outU, float& outV)
{
    constexpr int kStackSize = 64;
    int stack[kStackSize];
    int sp = 0;
    stack[sp++] = 0;

    float bestT = tmax;
    int   bestTri = -1;
    float bestU = 0, bestV = 0;

    while (sp > 0)
    {
        int ni = stack[--sp];
        BvhNode n;
        FETCH_BVH(n, ni);
        float tEnter;
        if (!ray_aabb(r, n.bbMin, n.bbMax, bestT, tEnter)) continue;

        if (n.triCount > 0)
        {
            for (int i = 0; i < n.triCount; ++i)
            {
                int ti = n.firstTri + i;
                float tt, uu, vv;
                if (ray_tri(r, s.tris[ti], bestT, tt, uu, vv))
                {
                    bestT = tt; bestTri = ti; bestU = uu; bestV = vv;
                }
            }
        }
        else if (sp + 2 <= kStackSize)
        {
            stack[sp++] = n.left;
            stack[sp++] = n.right;
        }
    }

    if (bestTri < 0) return -1;
    const Triangle& t = s.tris[bestTri];
    float w = 1.0f - bestU - bestV;
    outN = normalize(v3(
        w * t.n[0][0] + bestU * t.n[1][0] + bestV * t.n[2][0],
        w * t.n[0][1] + bestU * t.n[1][1] + bestV * t.n[2][1],
        w * t.n[0][2] + bestU * t.n[1][2] + bestV * t.n[2][2]));
    outU = w * t.uv[0][0] + bestU * t.uv[1][0] + bestV * t.uv[2][0];
    outV = w * t.uv[0][1] + bestU * t.uv[1][1] + bestV * t.uv[2][1];
    outT = bestT;
    return bestTri;
}

__device__ bool traverse_occluded(const DeviceScene& s, const BvhNode* s_top,
                                  const Ray& r, float tmax)
{
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0)
    {
        int ni = stack[--sp];
        BvhNode n;
        FETCH_BVH(n, ni);
        float tEnter;
        if (!ray_aabb(r, n.bbMin, n.bbMax, tmax, tEnter)) continue;

        if (n.triCount > 0)
        {
            for (int i = 0; i < n.triCount; ++i)
            {
                float tt, uu, vv;
                if (ray_tri(r, s.tris[n.firstTri + i], tmax, tt, uu, vv)) return true;
            }
        }
        else if (sp + 2 <= 64)
        {
            stack[sp++] = n.left;
            stack[sp++] = n.right;
        }
    }
    return false;
}

__device__ uint32_t xorshift32(uint32_t& s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__device__ float randf(uint32_t& s)
{
    return (xorshift32(s) >> 8) * (1.0f / 16777216.0f);
}

// Frisvad's branchless tangent frame
__device__ void onb(vec3 N, vec3& T, vec3& B)
{
    if (N.z < -0.99999f) { T = v3(0, -1, 0); B = v3(-1, 0, 0); return; }
    float a = 1.0f / (1.0f + N.z);
    float b = -N.x * N.y * a;
    T = v3(1.0f - N.x * N.x * a, b, -N.x);
    B = v3(b, 1.0f - N.y * N.y * a, -N.y);
}

__device__ vec3 sample_cos_hemi(vec3 N, uint32_t& rng)
{
    float u1 = randf(rng);
    float u2 = randf(rng);
    float r  = sqrtf(u1);
    float phi = 2.0f * 3.14159265358979f * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    vec3 T, B;
    onb(N, T, B);
    return normalize(vadd(vadd(vmul(T, x), vmul(B, y)), vmul(N, z)));
}

__device__ vec3 shade_direct(const DeviceScene& s, const BvhNode* s_top, vec3 P, vec3 N, vec3 albedo)
{
    constexpr float kShadowBias = 1e-3f;
    vec3 color = v3(0, 0, 0);

    // Sun (infinite directional). Convention: sun.dir is the light's travel direction,
    // so direction TO light = -sun.dir.
    {
        vec3 toLight = normalize(vmul(vfrom(s.sun.dir), -1.0f));
        float ndotl = fmaxf(0.0f, dot(N, toLight));
        if (ndotl > 0.0f)
        {
            Ray shadow = make_ray(vadd(P, vmul(N, kShadowBias)), toLight);
            if (!traverse_occluded(s, s_top, shadow, 1e30f))
            {
                float I = s.sun.intensity * ndotl;
                color = vadd(color, vhad(albedo, vmul(vfrom(s.sun.color), I)));
            }
        }
    }

    for (int i = 0; i < s.numPoints; ++i)
    {
        const PointLight& L = s.points[i];
        vec3 toLightVec = vsub(vfrom(L.pos), P);
        float dist = len(toLightVec);
        vec3 toLight = vmul(toLightVec, 1.0f / dist);
        float ndotl = fmaxf(0.0f, dot(N, toLight));
        if (ndotl <= 0.0f || dist >= L.radius * 3.0f) continue;

        Ray shadow = make_ray(vadd(P, vmul(N, kShadowBias)), toLight);
        if (traverse_occluded(s, s_top, shadow, dist - kShadowBias)) continue;

        // Soft falloff: smoothstep(radius, 0, d) — like LuminaGI's point light attenuation
        float r = L.radius;
        float atten = fmaxf(0.0f, 1.0f - dist / r);
        atten = atten * atten;
        float I = L.intensity * ndotl * atten;
        color = vadd(color, vhad(albedo, vmul(vfrom(L.color), I)));
    }

    return color;
}

__device__ vec3 sample_albedo(const DeviceMaterial& m, float u, float v)
{
    if (m.tex)
    {
        float4 t = tex2D<float4>(m.tex, u, 1.0f - v);
        return v3(t.x, t.y, t.z);
    }
    return vfrom(m.albedo);
}

__device__ Ray primary_ray(const CameraGPU& cam, int px, int py, float jx, float jy)
{
    float ndcx = (2.0f * (px + jx) / cam.width) - 1.0f;
    float ndcy = 1.0f - (2.0f * (py + jy) / cam.height);
    float rx = ndcx * cam.aspect * cam.tanHalfFov;
    float ry = ndcy * cam.tanHalfFov;
    vec3 dirCam = v3(1.0f, -rx, ry);
    vec3 dirWorld = vadd(vadd(vmul(cam.fwd, dirCam.x), vmul(cam.left, dirCam.y)), vmul(cam.up, dirCam.z));
    return make_ray(cam.pos, dirWorld);
}

__device__ vec3 trace_path(const DeviceScene& s, const BvhNode* s_top, Ray ray, int maxBounces, uint32_t& rng)
{
    vec3 L          = v3(0, 0, 0);
    vec3 throughput = v3(1, 1, 1);

    for (int bounce = 0; bounce <= maxBounces; ++bounce)
    {
        float hitT, hitU, hitV;
        vec3 N;
        int tri = traverse_closest(s, s_top, ray, 1e30f, hitT, N, hitU, hitV);
        if (tri < 0) break;

        if (dot(N, ray.d) > 0.0f) N = vmul(N, -1.0f);

        vec3 P = vadd(ray.o, vmul(ray.d, hitT));
        const DeviceMaterial& m = s.mats[s.tris[tri].material];
        vec3 albedo = sample_albedo(m, hitU, hitV);

        L = vadd(L, vhad(throughput, shade_direct(s, s_top, P, N, albedo)));

        if (bounce == maxBounces) break;

        // Russian roulette after bounce 0
        if (bounce >= 1)
        {
            float q = fminf(0.95f, fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)));
            if (randf(rng) > q) break;
            throughput = vmul(throughput, 1.0f / q);
        }

        // Cosine-weighted Lambertian bounce. With this pdf, throughput *= albedo.
        vec3 newDir = sample_cos_hemi(N, rng);
        ray = make_ray(vadd(P, vmul(N, 1e-3f)), newDir);
        throughput = vhad(throughput, albedo);
    }
    return L;
}

// launch_bounds(256, 4) caps the register budget at 65536/(256*4) = 64 regs/thread.
// Without it, nvcc uses 76 regs and theoretical occupancy is stuck at 50% (reg-limited).
// With 64 regs: 75% theoretical occupancy, no spills (+8 B stack), ~7% kernel speedup on the indoor-room test scene.
__global__ __launch_bounds__(256, 4)
void accumulate_kernel(DeviceScene s, CameraGPU cam, float* accum,
                                   int sppThisLaunch, int sampleBase, int maxBounces)
{
    __shared__ BvhNode s_top[kShmemBvhNodes];
    int linearTid = threadIdx.y * blockDim.x + threadIdx.x;
    // Block is 16x16 = 256 threads >= kShmemBvhNodes so one thread per node.
    if (linearTid < kShmemBvhNodes)
        s_top[linearTid] = s.nodes[linearTid];
    __syncthreads();

    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= cam.width || py >= cam.height) return;

    uint32_t rng = (uint32_t)(px * 1973 + py * 9277 + sampleBase * 26699) | 1u;

    vec3 sum = v3(0, 0, 0);
    for (int k = 0; k < sppThisLaunch; ++k)
    {
        float jx = randf(rng);
        float jy = randf(rng);
        Ray r = primary_ray(cam, px, py, jx, jy);
        sum = vadd(sum, trace_path(s, s_top, r, maxBounces, rng));
    }

    int i = (py * cam.width + px) * 3;
    accum[i + 0] += sum.x;
    accum[i + 1] += sum.y;
    accum[i + 2] += sum.z;
}

__global__ void tonemap_kernel(const float* accum, uint8_t* image, int W, int H, float invSpp)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;

    int i = (py * W + px) * 3;
    float r = accum[i + 0] * invSpp;
    float g = accum[i + 1] * invSpp;
    float b = accum[i + 2] * invSpp;

    // sqrt gamma (~2.0), clamped
    r = sqrtf(fmaxf(0.0f, fminf(r, 1.0f)));
    g = sqrtf(fmaxf(0.0f, fminf(g, 1.0f)));
    b = sqrtf(fmaxf(0.0f, fminf(b, 1.0f)));

    image[i + 0] = (uint8_t)(r * 255.0f);
    image[i + 1] = (uint8_t)(g * 255.0f);
    image[i + 2] = (uint8_t)(b * 255.0f);
}

static bool load_texture_to_device(const std::string& fullPath,
                                   cudaArray_t& outArray, cudaTextureObject_t& outHandle)
{
    int w, h, n;
    unsigned char* data = stbi_load(fullPath.c_str(), &w, &h, &n, 4);
    if (!data) return false;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&outArray, &desc, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(outArray, 0, 0, data, w * 4, w * 4, h, cudaMemcpyHostToDevice));
    stbi_image_free(data);

    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = outArray;

    cudaTextureDesc tex{};
    tex.addressMode[0]   = cudaAddressModeWrap;
    tex.addressMode[1]   = cudaAddressModeWrap;
    tex.filterMode       = cudaFilterModeLinear;
    tex.readMode         = cudaReadModeNormalizedFloat;
    tex.normalizedCoords = 1;
    tex.sRGB             = 1;

    CUDA_CHECK(cudaCreateTextureObject(&outHandle, &res, &tex, nullptr));
    return true;
}

void RenderSceneCUDA(const Scene& scene, const Bvh& bvh, const std::string& assetRoot,
                     int spp, int maxBounces,
                     std::vector<uint8_t>& outRGB, int& outW, int& outH)
{
    outW = scene.camera.imageWidth;
    outH = scene.camera.imageHeight;
    outRGB.assign((size_t)outW * outH * 3, 0);

    std::vector<cudaArray_t>         texArrays(scene.materials.size(), nullptr);
    std::vector<cudaTextureObject_t> texHandles(scene.materials.size(), 0);

    int loaded = 0, missing = 0;
    for (size_t i = 0; i < scene.materials.size(); ++i)
    {
        const std::string& rel = scene.materials[i].diffuseTex;
        if (rel.empty()) continue;
        std::string full = assetRoot + "/" + rel;
        if (load_texture_to_device(full, texArrays[i], texHandles[i]))
            ++loaded;
        else
        {
            std::fprintf(stderr, "  texture not found: %s\n", full.c_str());
            ++missing;
        }
    }
    std::printf("textures: %d loaded, %d missing (of %zu materials)\n",
                loaded, missing, scene.materials.size());

    std::vector<DeviceMaterial> hostMats(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i)
    {
        std::memcpy(hostMats[i].albedo,   scene.materials[i].albedo,   sizeof(float) * 3);
        std::memcpy(hostMats[i].emissive, scene.materials[i].emissive, sizeof(float) * 3);
        hostMats[i].tex = texHandles[i];
    }

    BvhNode*         d_nodes  = nullptr;
    Triangle*        d_tris   = nullptr;
    DeviceMaterial*  d_mats   = nullptr;
    PointLight*      d_points = nullptr;
    uint8_t*         d_image  = nullptr;

    size_t nodesBytes  = bvh.nodes.size()       * sizeof(BvhNode);
    size_t trisBytes   = scene.triangles.size() * sizeof(Triangle);
    size_t matsBytes   = hostMats.size()        * sizeof(DeviceMaterial);
    size_t pointsBytes = scene.points.size()    * sizeof(PointLight);
    size_t imgBytes    = (size_t)outW * outH * 3;

    CUDA_CHECK(cudaMalloc(&d_nodes,  nodesBytes));
    CUDA_CHECK(cudaMalloc(&d_tris,   trisBytes));
    CUDA_CHECK(cudaMalloc(&d_mats,   matsBytes));
    if (pointsBytes) CUDA_CHECK(cudaMalloc(&d_points, pointsBytes));
    CUDA_CHECK(cudaMalloc(&d_image, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_nodes, bvh.nodes.data(),       nodesBytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tris,  scene.triangles.data(), trisBytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mats,  hostMats.data(),        matsBytes,   cudaMemcpyHostToDevice));
    if (pointsBytes) CUDA_CHECK(cudaMemcpy(d_points, scene.points.data(), pointsBytes, cudaMemcpyHostToDevice));

    DeviceScene s{};
    s.nodes     = d_nodes;
    s.tris      = d_tris;
    s.mats      = d_mats;
    s.points    = d_points;
    s.numPoints = (int)scene.points.size();
    s.sun       = scene.sun;

    CameraGPU cam{};
    cam.pos    = vfrom(scene.camera.pos);
    const float* m = scene.camera.cameraToWorld;
    cam.fwd  = v3(m[0],  m[1],  m[2]);   // row 0
    cam.left = v3(m[4],  m[5],  m[6]);   // row 1
    cam.up   = v3(m[8],  m[9],  m[10]);  // row 2
    cam.tanHalfFov = tanf(scene.camera.fovYDeg * 0.5f * 3.14159265358979f / 180.0f);
    cam.aspect  = scene.camera.aspect;
    cam.width   = outW;
    cam.height  = outH;

    float* d_accum = nullptr;
    size_t accumBytes = (size_t)outW * outH * 3 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_accum, accumBytes));
    CUDA_CHECK(cudaMemset(d_accum, 0, accumBytes));

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

    // Split samples into chunks to stay comfortably under Windows TDR.
    const int kChunk = 8;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int base = 0; base < spp; base += kChunk)
    {
        int thisLaunch = (spp - base < kChunk) ? (spp - base) : kChunk;
        accumulate_kernel<<<grid, block>>>(s, cam, d_accum, thisLaunch, base, maxBounces);
        CUDA_CHECK(cudaGetLastError());
    }

    tonemap_kernel<<<grid, block>>>(d_accum, d_image, outW, outH, 1.0f / (float)spp);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(t1);
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    std::printf("kernel: %.1f ms  (%d x %d, %d spp, %d bounces, %zu tris)\n",
                ms, outW, outH, spp, maxBounces, scene.triangles.size());

    CUDA_CHECK(cudaMemcpy(outRGB.data(), d_image, imgBytes, cudaMemcpyDeviceToHost));
    cudaFree(d_accum);

    cudaFree(d_nodes); cudaFree(d_tris); cudaFree(d_mats);
    if (d_points) cudaFree(d_points);
    cudaFree(d_image);
    for (size_t i = 0; i < texHandles.size(); ++i)
    {
        if (texHandles[i]) cudaDestroyTextureObject(texHandles[i]);
        if (texArrays[i])  cudaFreeArray(texArrays[i]);
    }
}
