#include "bvh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace
{
    constexpr int kLeafMaxTris = 4;

    // 10 bits -> 30 bits with 2 zeros between each
    uint32_t spread3(uint32_t v)
    {
        v = (v | (v << 16)) & 0x030000FFu;
        v = (v | (v << 8))  & 0x0300F00Fu;
        v = (v | (v << 4))  & 0x030C30C3u;
        v = (v | (v << 2))  & 0x09249249u;
        return v;
    }

    uint32_t morton3(float x, float y, float z)
    {
        auto q = [](float f) {
            f = f * 1024.0f;
            if (f < 0.0f) f = 0.0f;
            if (f > 1023.0f) f = 1023.0f;
            return (uint32_t)f;
        };
        return (spread3(q(x)) << 2) | (spread3(q(y)) << 1) | spread3(q(z));
    }

    void fit(float mn[3], float mx[3], const float p[3])
    {
        if (p[0] < mn[0]) mn[0] = p[0];
        if (p[1] < mn[1]) mn[1] = p[1];
        if (p[2] < mn[2]) mn[2] = p[2];
        if (p[0] > mx[0]) mx[0] = p[0];
        if (p[1] > mx[1]) mx[1] = p[1];
        if (p[2] > mx[2]) mx[2] = p[2];
    }

    void fit_union(float mn[3], float mx[3], const float a_mn[3], const float a_mx[3])
    {
        if (a_mn[0] < mn[0]) mn[0] = a_mn[0];
        if (a_mn[1] < mn[1]) mn[1] = a_mn[1];
        if (a_mn[2] < mn[2]) mn[2] = a_mn[2];
        if (a_mx[0] > mx[0]) mx[0] = a_mx[0];
        if (a_mx[1] > mx[1]) mx[1] = a_mx[1];
        if (a_mx[2] > mx[2]) mx[2] = a_mx[2];
    }

    // Lowest i in [first, last) where morton[i] has bit `mask` set.
    int find_split(const uint32_t* morton, int first, int last, uint32_t mask)
    {
        int lo = first, hi = last;
        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (morton[mid] & mask) hi = mid;
            else                    lo = mid + 1;
        }
        return lo;
    }
}

struct BuildCtx
{
    std::vector<BvhNode>& nodes;
    Scene& scene;
    const uint32_t* morton;
};

static int build_recursive(BuildCtx& c, int first, int last, int bit)
{
    int idx = (int)c.nodes.size();
    c.nodes.emplace_back();
    BvhNode& n = c.nodes[idx];

    int count = last - first;

    // Find a usable split bit
    int split = first;
    while (bit >= 0)
    {
        split = find_split(c.morton, first, last, 1u << bit);
        if (split != first && split != last) break;
        --bit;
    }

    if (count <= kLeafMaxTris || bit < 0)
    {
        n.left = -1; n.right = -1;
        n.firstTri = first; n.triCount = count;
        float mn[3] = { +INFINITY, +INFINITY, +INFINITY };
        float mx[3] = { -INFINITY, -INFINITY, -INFINITY };
        for (int i = first; i < last; ++i)
        {
            for (int k = 0; k < 3; ++k) fit(mn, mx, c.scene.triangles[i].v[k]);
        }
        n.bbMin[0] = mn[0]; n.bbMin[1] = mn[1]; n.bbMin[2] = mn[2];
        n.bbMax[0] = mx[0]; n.bbMax[1] = mx[1]; n.bbMax[2] = mx[2];
        return idx;
    }

    int L = build_recursive(c, first, split, bit - 1);
    int R = build_recursive(c, split, last, bit - 1);

    // Re-fetch reference in case vector reallocated
    BvhNode& self = c.nodes[idx];
    self.left = L; self.right = R;
    self.firstTri = -1; self.triCount = 0;

    self.bbMin[0] = c.nodes[L].bbMin[0]; self.bbMin[1] = c.nodes[L].bbMin[1]; self.bbMin[2] = c.nodes[L].bbMin[2];
    self.bbMax[0] = c.nodes[L].bbMax[0]; self.bbMax[1] = c.nodes[L].bbMax[1]; self.bbMax[2] = c.nodes[L].bbMax[2];
    fit_union(self.bbMin, self.bbMax, c.nodes[R].bbMin, c.nodes[R].bbMax);

    return idx;
}

void BuildBVH(Scene& scene, Bvh& out)
{
    const int N = (int)scene.triangles.size();
    out.nodes.clear();
    if (N == 0) return;

    float sceneMin[3] = { +INFINITY, +INFINITY, +INFINITY };
    float sceneMax[3] = { -INFINITY, -INFINITY, -INFINITY };
    for (const Triangle& t : scene.triangles)
        for (int c = 0; c < 3; ++c) fit(sceneMin, sceneMax, t.v[c]);

    float ext[3] = { sceneMax[0] - sceneMin[0],
                     sceneMax[1] - sceneMin[1],
                     sceneMax[2] - sceneMin[2] };
    for (int a = 0; a < 3; ++a) if (ext[a] <= 0.0f) ext[a] = 1.0f;

    std::vector<uint32_t> mortonCode(N);
    std::vector<int>      order(N);
    for (int i = 0; i < N; ++i)
    {
        const Triangle& t = scene.triangles[i];
        float cx = (t.v[0][0] + t.v[1][0] + t.v[2][0]) * (1.0f / 3.0f);
        float cy = (t.v[0][1] + t.v[1][1] + t.v[2][1]) * (1.0f / 3.0f);
        float cz = (t.v[0][2] + t.v[1][2] + t.v[2][2]) * (1.0f / 3.0f);
        mortonCode[i] = morton3((cx - sceneMin[0]) / ext[0],
                                (cy - sceneMin[1]) / ext[1],
                                (cz - sceneMin[2]) / ext[2]);
        order[i] = i;
    }

    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return mortonCode[a] < mortonCode[b]; });

    std::vector<Triangle> triSorted(N);
    std::vector<uint32_t> mortonSorted(N);
    for (int i = 0; i < N; ++i)
    {
        triSorted[i]    = scene.triangles[order[i]];
        mortonSorted[i] = mortonCode[order[i]];
    }
    scene.triangles = std::move(triSorted);

    out.nodes.reserve(2 * N);
    BuildCtx ctx{ out.nodes, scene, mortonSorted.data() };
    out.rootIdx = build_recursive(ctx, 0, N, 29);
}
