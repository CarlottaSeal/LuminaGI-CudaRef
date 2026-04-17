#include "scene.h"

#include <cstdio>
#include <fstream>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace
{
    void read_vec3(const json& j, float out[3])
    {
        out[0] = j[0].get<float>();
        out[1] = j[1].get<float>();
        out[2] = j[2].get<float>();
    }

    void read_vec2(const json& j, float out[2])
    {
        out[0] = j[0].get<float>();
        out[1] = j[1].get<float>();
    }
}

bool LoadSceneJSON(const std::string& path, Scene& out)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::fprintf(stderr, "[scene] failed to open %s\n", path.c_str());
        return false;
    }

    json j;
    try
    {
        f >> j;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[scene] JSON parse error: %s\n", e.what());
        return false;
    }

    // Version gate — keeps schema compatible going forward
    int version = j.value("version", 0);
    if (version != 1)
    {
        std::fprintf(stderr, "[scene] unsupported version %d (expected 1)\n", version);
        return false;
    }

    // Camera
    {
        const json& c = j.at("camera");
        read_vec3(c.at("pos"), out.camera.pos);
        const json& m = c.at("camera_to_world");
        for (int i = 0; i < 16; ++i) out.camera.cameraToWorld[i] = m[i].get<float>();
        out.camera.fovYDeg     = c.at("fov_y_deg").get<float>();
        out.camera.aspect      = c.at("aspect").get<float>();
        out.camera.nearZ       = c.at("near").get<float>();
        out.camera.farZ        = c.at("far").get<float>();
        out.camera.imageWidth  = c.at("image_width").get<int>();
        out.camera.imageHeight = c.at("image_height").get<int>();
    }

    // Sun (scene-global)
    {
        const json& s = j.at("sun");
        read_vec3(s.at("direction"), out.sun.dir);
        read_vec3(s.at("color"),     out.sun.color);
        out.sun.intensity = s.at("intensity").get<float>();
    }

    // Lights (per-object)
    for (const json& L : j.at("lights"))
    {
        const std::string type = L.at("type").get<std::string>();
        if (type == "directional")
        {
            DirectionalLight d;
            read_vec3(L.at("direction"), d.dir);
            read_vec3(L.at("color"),     d.color);
            d.intensity = L.at("intensity").get<float>();
            out.directionals.push_back(d);
        }
        else if (type == "point")
        {
            PointLight p;
            read_vec3(L.at("pos"),   p.pos);
            read_vec3(L.at("color"), p.color);
            p.intensity = L.at("intensity").get<float>();
            p.radius    = L.at("radius").get<float>();
            out.points.push_back(p);
        }
    }

    // Materials
    for (const json& M : j.at("materials"))
    {
        Material m;
        read_vec3(M.at("albedo"),   m.albedo);
        read_vec3(M.at("emissive"), m.emissive);
        if (M.contains("diffuse_tex"))
        {
            m.diffuseTex = M.at("diffuse_tex").get<std::string>();
        }
        out.materials.push_back(m);
    }

    // Triangles
    const json& tris = j.at("triangles");
    out.triangles.reserve(tris.size());
    for (const json& T : tris)
    {
        Triangle t;
        const json& v  = T.at("v");
        const json& n  = T.at("n");
        const json& uv = T.at("uv");
        for (int c = 0; c < 3; ++c)
        {
            read_vec3(v[c],  t.v[c]);
            read_vec3(n[c],  t.n[c]);
            read_vec2(uv[c], t.uv[c]);
        }
        t.material = T.at("mat").get<int>();
        out.triangles.push_back(t);
    }

    std::printf("[scene] loaded %s\n", path.c_str());
    std::printf("  camera: %dx%d, fov %.1f deg, pos (%.2f,%.2f,%.2f)\n",
                out.camera.imageWidth, out.camera.imageHeight, out.camera.fovYDeg,
                out.camera.pos[0], out.camera.pos[1], out.camera.pos[2]);
    std::printf("  %zu directional + %zu point lights, %zu materials, %zu triangles\n",
                out.directionals.size(), out.points.size(),
                out.materials.size(), out.triangles.size());
    return true;
}
