// Standalone loader smoke test.
//   test_load_scene <scene.json>
// Parses the JSON dump and prints counts + sanity-check ranges.
// No CUDA, pure host code — fast to iterate on the schema.

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "scene.h"

namespace
{
    void update_bounds(float mn[3], float mx[3], const float p[3])
    {
        for (int i = 0; i < 3; ++i)
        {
            if (p[i] < mn[i]) mn[i] = p[i];
            if (p[i] > mx[i]) mx[i] = p[i];
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <scene.json>\n", argv[0]);
        return 1;
    }

    Scene s;
    if (!LoadSceneJSON(argv[1], s)) return 2;

    // World bounds sanity check
    float mn[3] = { std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity() };
    float mx[3] = { -std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity() };
    for (const Triangle& t : s.triangles)
        for (int c = 0; c < 3; ++c)
            update_bounds(mn, mx, t.v[c]);

    std::printf("world bounds:\n");
    std::printf("  min (%.3f, %.3f, %.3f)\n", mn[0], mn[1], mn[2]);
    std::printf("  max (%.3f, %.3f, %.3f)\n", mx[0], mx[1], mx[2]);
    std::printf("  size (%.3f, %.3f, %.3f)\n", mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);

    std::printf("sun: dir (%.2f,%.2f,%.2f) color (%.2f,%.2f,%.2f) intensity %.2f\n",
                s.sun.dir[0], s.sun.dir[1], s.sun.dir[2],
                s.sun.color[0], s.sun.color[1], s.sun.color[2], s.sun.intensity);

    for (size_t i = 0; i < s.points.size(); ++i)
    {
        const PointLight& p = s.points[i];
        std::printf("point[%zu]: pos (%.2f,%.2f,%.2f) color (%.2f,%.2f,%.2f) intensity %.2f radius %.2f\n",
                    i, p.pos[0], p.pos[1], p.pos[2],
                    p.color[0], p.color[1], p.color[2], p.intensity, p.radius);
    }

    std::printf("OK\n");
    return 0;
}
