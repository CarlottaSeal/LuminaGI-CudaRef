// cuda_ref <scene.json> [output.png]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
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

static void overwrite_camera(Scene& scene, const float camPos[3], const float camTgt[3])
{
    // SD convention: I = forward, J = left, K = up, stored column-major
    // as [Ix Iy Iz Iw | Jx Jy Jz Jw | Kx Ky Kz Kw | Tx Ty Tz Tw] in the float[16].
    float fx = camTgt[0] - camPos[0];
    float fy = camTgt[1] - camPos[1];
    float fz = camTgt[2] - camPos[2];
    float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
    if (fl < 1e-8f) return;
    fx /= fl; fy /= fl; fz /= fl;

    // J = left = cross(worldUp, forward), worldUp = (0,0,1)
    float lx = -fy, ly = fx, lz = 0.0f;
    float ll = std::sqrt(lx*lx + ly*ly + lz*lz);
    if (ll < 1e-6f) { lx = 0; ly = 1; lz = 0; ll = 1; }
    lx /= ll; ly /= ll; lz /= ll;

    // K = up = cross(forward, left)
    float ux = fy*lz - fz*ly;
    float uy = fz*lx - fx*lz;
    float uz = fx*ly - fy*lx;

    float* m = scene.camera.cameraToWorld;
    m[0]  = fx; m[1]  = fy; m[2]  = fz; m[3]  = 0;
    m[4]  = lx; m[5]  = ly; m[6]  = lz; m[7]  = 0;
    m[8]  = ux; m[9]  = uy; m[10] = uz; m[11] = 0;
    m[12] = camPos[0]; m[13] = camPos[1]; m[14] = camPos[2]; m[15] = 1;
    scene.camera.pos[0] = camPos[0];
    scene.camera.pos[1] = camPos[1];
    scene.camera.pos[2] = camPos[2];
}

// Batch file format: one render job per line, space-separated, # starts a comment.
//   out_path spp bounces camx camy camz tgtx tgty tgtz
struct BatchJob
{
    std::string out_png;
    int   spp, bounces;
    float cam_pos[3];
    float cam_tgt[3];
};

static std::vector<BatchJob> read_batch_file(const std::string& path)
{
    std::vector<BatchJob> jobs;
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "can't open batch file: %s\n", path.c_str()); return jobs; }
    std::string line;
    while (std::getline(in, line))
    {
        // strip comments / whitespace
        auto hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        std::istringstream ss(line);
        BatchJob j{};
        if (!(ss >> j.out_png >> j.spp >> j.bounces
                 >> j.cam_pos[0] >> j.cam_pos[1] >> j.cam_pos[2]
                 >> j.cam_tgt[0] >> j.cam_tgt[1] >> j.cam_tgt[2]))
            continue;
        jobs.push_back(j);
    }
    return jobs;
}

int main(int argc, char** argv)
{
    const char* jsonPath = nullptr;
    const char* pngPath  = "output/reference.png";
    std::string assetRoot;
    std::string batchFile;
    int spp = 64;
    int bounces = 2;
    bool useSort = false;

    bool  haveCamPos = false, haveCamTgt = false;
    float camPos[3] = {0,0,0};
    float camTgt[3] = {0,0,0};

    auto read_vec3 = [&](int& i, float out[3]) -> bool {
        if (i + 3 >= argc) return false;
        out[0] = (float)std::atof(argv[++i]);
        out[1] = (float)std::atof(argv[++i]);
        out[2] = (float)std::atof(argv[++i]);
        return true;
    };

    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--spp" && i + 1 < argc)           spp = std::atoi(argv[++i]);
        else if (a == "--bounces" && i + 1 < argc)  bounces = std::atoi(argv[++i]);
        else if (a == "--asset-root" && i + 1 < argc) assetRoot = argv[++i];
        else if (a == "-o" && i + 1 < argc)         pngPath = argv[++i];
        else if (a == "--sort")                     useSort = true;
        else if (a == "--cam-pos")                  haveCamPos = read_vec3(i, camPos);
        else if (a == "--cam-target")               haveCamTgt = read_vec3(i, camTgt);
        else if (a == "--batch" && i + 1 < argc)    batchFile = argv[++i];
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

    if (haveCamPos && haveCamTgt)
    {
        overwrite_camera(scene, camPos, camTgt);
        std::printf("camera override: pos (%.2f,%.2f,%.2f) -> target (%.2f,%.2f,%.2f)\n",
                    camPos[0], camPos[1], camPos[2], camTgt[0], camTgt[1], camTgt[2]);
    }

    auto t_bvh0 = std::chrono::steady_clock::now();
    Bvh bvh_batch;
    if (!batchFile.empty()) BuildBVH(scene, bvh_batch);
    auto t_bvh1 = std::chrono::steady_clock::now();

    if (!batchFile.empty())
    {
        auto jobs = read_batch_file(batchFile);
        std::printf("batch: %zu jobs  (scene+bvh loaded once in %.1f ms)\n",
                    jobs.size(),
                    std::chrono::duration<double, std::milli>(t_bvh1 - t_bvh0).count());
        for (size_t k = 0; k < jobs.size(); ++k)
        {
            const BatchJob& j = jobs[k];
            overwrite_camera(scene, j.cam_pos, j.cam_tgt);
            std::vector<uint8_t> rgb; int W = 0, H = 0;
            auto t0 = std::chrono::steady_clock::now();
            RenderSceneCUDA(scene, bvh_batch, assetRoot, j.spp, j.bounces, rgb, W, H);
            auto t1 = std::chrono::steady_clock::now();
            std::filesystem::create_directories(std::filesystem::path(j.out_png).parent_path());
            stbi_write_png(j.out_png.c_str(), W, H, 3, rgb.data(), W * 3);
            std::printf("[%zu/%zu] %s  spp=%d  render=%.1f ms\n",
                        k + 1, jobs.size(), j.out_png.c_str(), j.spp,
                        std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return 0;
    }

    auto t0 = std::chrono::steady_clock::now();
    Bvh bvh;
    BuildBVH(scene, bvh);
    auto t1 = std::chrono::steady_clock::now();
    (void)t0; (void)t1;
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
