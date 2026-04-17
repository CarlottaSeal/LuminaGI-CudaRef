// Build a BVH over a scene dump and print stats.
//   test_bvh <scene.json>

#include <chrono>
#include <cstdio>

#include "bvh.h"
#include "scene.h"

namespace
{
    void depth_stats(const Bvh& bvh, int idx, int d, int& maxD, double& sumD, int& leafCount)
    {
        const BvhNode& n = bvh.nodes[idx];
        if (n.triCount > 0)
        {
            if (d > maxD) maxD = d;
            sumD += d;
            ++leafCount;
            return;
        }
        depth_stats(bvh, n.left,  d + 1, maxD, sumD, leafCount);
        depth_stats(bvh, n.right, d + 1, maxD, sumD, leafCount);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: %s <scene.json>\n", argv[0]); return 1; }

    Scene scene;
    if (!LoadSceneJSON(argv[1], scene)) return 2;

    auto t0 = std::chrono::steady_clock::now();
    Bvh bvh;
    BuildBVH(scene, bvh);
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    int maxD = 0, leaves = 0;
    double sumD = 0.0;
    depth_stats(bvh, bvh.rootIdx, 0, maxD, sumD, leaves);

    std::printf("bvh built in %.1f ms\n", ms);
    std::printf("  %zu nodes (%zu MB)\n", bvh.nodes.size(),
                (bvh.nodes.size() * sizeof(BvhNode)) >> 20);
    std::printf("  %d leaves, depth max=%d avg=%.1f\n",
                leaves, maxD, sumD / leaves);
    std::printf("  root bounds: (%.2f,%.2f,%.2f) .. (%.2f,%.2f,%.2f)\n",
                bvh.nodes[bvh.rootIdx].bbMin[0], bvh.nodes[bvh.rootIdx].bbMin[1], bvh.nodes[bvh.rootIdx].bbMin[2],
                bvh.nodes[bvh.rootIdx].bbMax[0], bvh.nodes[bvh.rootIdx].bbMax[1], bvh.nodes[bvh.rootIdx].bbMax[2]);
    return 0;
}
