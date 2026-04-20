// cuda_ref <scene.json> [output.png]

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "bvh.h"
#include "scene.h"

void RenderSceneCUDA(const Scene& scene, const Bvh& bvh, const std::string& assetRoot,
                     int spp, int maxBounces,
                     std::vector<uint8_t>& outRGB, int& outW, int& outH);
void RenderSceneCUDASorted(const Scene& scene, const Bvh& bvh, const std::string& assetRoot,
                           int spp, int maxBounces,
                           std::vector<uint8_t>& outRGB, int& outW, int& outH);

int main(int argc, char** argv)
{
    const char* jsonPath = nullptr;
    const char* pngPath  = "output/reference.png";
    std::string assetRoot;
    int spp = 64;
    int bounces = 2;
    bool useSort = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--spp" && i + 1 < argc)           spp = std::atoi(argv[++i]);
        else if (a == "--bounces" && i + 1 < argc)  bounces = std::atoi(argv[++i]);
        else if (a == "--asset-root" && i + 1 < argc) assetRoot = argv[++i];
        else if (a == "-o" && i + 1 < argc)         pngPath = argv[++i];
        else if (a == "--sort")                     useSort = true;
        else if (a[0] != '-' && !jsonPath)          jsonPath = argv[i];
        else if (a[0] != '-')                       pngPath = argv[i];
    }

    if (!jsonPath)
    {
        std::fprintf(stderr,
            "usage: %s <scene.json> [output.png] [--spp N] [--bounces N] [--asset-root DIR]\n",
            argv[0]);
        return 1;
    }

    if (assetRoot.empty())
        assetRoot = std::filesystem::path(jsonPath).parent_path().parent_path().string();

    Scene scene;
    if (!LoadSceneJSON(jsonPath, scene)) return 2;

    auto t0 = std::chrono::steady_clock::now();
    Bvh bvh;
    BuildBVH(scene, bvh);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("bvh: %.1f ms (%zu nodes)\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                bvh.nodes.size());

    std::vector<uint8_t> rgb;
    int W = 0, H = 0;
    auto t2 = std::chrono::steady_clock::now();
    if (useSort)
        RenderSceneCUDASorted(scene, bvh, assetRoot, spp, bounces, rgb, W, H);
    else
        RenderSceneCUDA(scene, bvh, assetRoot, spp, bounces, rgb, W, H);
    auto t3 = std::chrono::steady_clock::now();
    std::printf("render: %.1f ms (%d x %d)\n",
                std::chrono::duration<double, std::milli>(t3 - t2).count(), W, H);

    if (!stbi_write_png(pngPath, W, H, 3, rgb.data(), W * 3))
    {
        std::fprintf(stderr, "failed to write %s\n", pngPath);
        return 3;
    }
    std::printf("wrote %s\n", pngPath);
    return 0;
}
