#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct CameraDesc
{
    float pos[3]            = {0, 0, 0};
    float cameraToWorld[16] = {};
    float fovYDeg           = 60.0f;
    float aspect            = 1.0f;
    float nearZ             = 0.1f;
    float farZ              = 1000.0f;
    int   imageWidth        = 1920;
    int   imageHeight       = 1080;
};

struct DirectionalLight
{
    float dir[3]   = {0, 0, -1};
    float color[3] = {1, 1, 1};
    float intensity = 1.0f;
};

struct PointLight
{
    float pos[3]   = {0, 0, 0};
    float color[3] = {1, 1, 1};
    float intensity = 1.0f;
    float radius    = 1.0f;
};

struct Material
{
    float       albedo[3]   = {0.8f, 0.8f, 0.8f};
    float       emissive[3] = {0, 0, 0};
    std::string diffuseTex;   // empty = untextured
};

struct Triangle
{
    float v[3][3];
    float n[3][3];
    float uv[3][2];
    int   material = 0;
};

struct Scene
{
    CameraDesc                    camera;
    DirectionalLight              sun;
    std::vector<DirectionalLight> directionals;
    std::vector<PointLight>       points;
    std::vector<Material>         materials;
    std::vector<Triangle>         triangles;
};

bool LoadSceneJSON(const std::string& path, Scene& out);
