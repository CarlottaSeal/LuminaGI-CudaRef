#pragma once

#ifdef __CUDACC__
  #define VM_HD __host__ __device__ inline
#else
  #define VM_HD inline
#endif

struct vec3 { float x, y, z; };

VM_HD vec3 v3(float x, float y, float z)      { return {x, y, z}; }
VM_HD vec3 vfrom(const float p[3])            { return {p[0], p[1], p[2]}; }
VM_HD vec3 vadd(vec3 a, vec3 b)               { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
VM_HD vec3 vsub(vec3 a, vec3 b)               { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
VM_HD vec3 vmul(vec3 a, float s)              { return {a.x*s, a.y*s, a.z*s}; }
VM_HD vec3 vhad(vec3 a, vec3 b)               { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
VM_HD float dot(vec3 a, vec3 b)               { return a.x*b.x + a.y*b.y + a.z*b.z; }
VM_HD vec3 cross(vec3 a, vec3 b)
{
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}
VM_HD float len(vec3 a)                       { return sqrtf(dot(a, a)); }
VM_HD vec3 normalize(vec3 a)
{
    float l = len(a);
    return (l > 1e-20f) ? vmul(a, 1.0f / l) : v3(0, 0, 1);
}

struct Ray
{
    vec3 o, d;
    vec3 inv_d;   // precomputed 1/d for slab tests
};

VM_HD Ray make_ray(vec3 o, vec3 d)
{
    Ray r;
    r.o = o;
    r.d = normalize(d);
    r.inv_d = { 1.0f / r.d.x, 1.0f / r.d.y, 1.0f / r.d.z };
    return r;
}
