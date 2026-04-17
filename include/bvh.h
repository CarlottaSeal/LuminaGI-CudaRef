#pragma once
#include <cstdint>
#include <vector>

#include "scene.h"

// 40 bytes, 4-byte aligned. Fine for CUDA global memory.
struct BvhNode
{
    float bbMin[3];
    float bbMax[3];
    int   left;      // -1 for leaf
    int   right;     // -1 for leaf
    int   firstTri;  // leaf only (triangle index after Build reorders scene.triangles)
    int   triCount;  // 0 = internal, >0 = leaf
};

struct Bvh
{
    std::vector<BvhNode> nodes;
    int rootIdx = 0;
};

// Builds a BVH over scene.triangles using Morton-ordered radix split.
// Triangles are reordered in place so each leaf refers to a contiguous range.
void BuildBVH(Scene& scene, Bvh& out);
