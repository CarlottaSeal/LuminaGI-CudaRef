# v5 速记卡（Path Tracing 间接光关键概念）

---

## 卡 1：xorshift32 — 极简伪随机数生成器（Pseudo-Random Number Generator, PRNG）

```cpp
struct RNG { uint32_t state; };

__device__ uint32_t xorshift32(RNG& r)
{
    uint32_t x = r.state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    r.state = x ? x : 0x9E3779B9u;   // 防 0 吸收态 (absorbing state)
    return r.state;
}

__device__ float rand01(RNG& r)
{
    return xorshift32(r) * (1.0f / 4294967296.0f);
}
```

- **state 32 bit，周期 (period) 2³² − 1**，跑遍除 0 外所有值
- 3 条 shift + XOR (异或)：左 13、右 17、左 5。Marsaglia 2003 经验组合
- **0 是吸收态** → 一旦命中永远 0；用黄金比例常数 (golden-ratio constant) `0x9E3779B9` 兜底
- 不是密码学安全 (cryptographically secure)，但路径追踪每像素一独立 stream 完全够用
- 替代品：`curand` 质量更好但 state 大几十倍

---

## 卡 2：ONB = Orthonormal Basis（正交规范基）

3 个互相垂直 (orthogonal)、长度为 1 (unit length) 的向量 {T, B, N}，把局部方向 (local direction) 映射到世界方向 (world direction)。
- T = tangent（切线）
- B = bitangent（副切线）
- N = normal（法线）

```
              N (= local Z+)
              ↑
              |     采样到的方向 (sampled direction)
              |    ╱
              |   ╱
              |  ╱  θ
              |╱_______________ B (= local Y+)
             ╱
            ╱
           ╱
          T (= local X+)

worldDir = x·T + y·B + z·N
```

**为什么需要**：cosine-weighted 半球采样 (cosine-weighted hemisphere sampling) 自然定义在"表面是 XY 平面、N 朝上"的局部坐标系 (local frame)；光线 (ray) 要走世界空间，得有基底变换 (basis change)。

**用 Frisvad 2012 无分支 (branch-free) 版本**：

```cpp
__device__ void build_onb(vec3 N, vec3& T, vec3& B)
{
    float sign = copysignf(1.0f, N.z);
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    T = v3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = v3(b, sign + N.y * N.y * a, -N.y);
}
```

**naive cross 那版的坑**：选辅助向量得分 `if (|N.z| < 0.9) up else right`，这条 if 在边界上 T 会突变，画面里出现接缝带 (visible seam)。Frisvad 用 `copysignf` 把南北半球公式合一。

---

## 卡 3：Lambertian + cosine 采样的 cos / pdf 抵消

**渲染方程 (Rendering Equation)** — half-space 版本：
$$L_o = \int_\Omega L_i(\omega) \cdot f \cdot \cos\theta \, d\omega$$

- L_o = 出射辐亮度 (outgoing radiance)
- L_i = 入射辐亮度 (incoming radiance)
- f = BRDF (Bidirectional Reflectance Distribution Function，双向反射分布函数)
- cos θ = Lambert 余弦律 (Lambert's cosine law)
- Ω = 法线上方的半球 (upper hemisphere)

**Monte Carlo (蒙特卡洛) 估计 (estimator)**：
$$L_o \approx \frac{1}{N}\sum \frac{L_i \cdot f \cdot \cos\theta}{p(\omega)}$$

**Lambertian + cosine-weighted 重要性采样 (importance sampling)**：
- BRDF：`f = albedo / π`（常数）
- 选 pdf (probability density function，概率密度函数)：`p = cos θ / π`

代入消去：

$$\text{贡献 (contribution)} = \frac{L_i \cdot (albedo/\pi) \cdot \cos\theta}{\cos\theta/\pi} = L_i \cdot albedo$$

**代码里只剩一行**：

```cpp
throughput *= albedo;                       // cos 与 π 都消掉了
nextDir   = sample_cosine_hemisphere(N);    // pdf 已经隐式是 cos/π
```

**Malley's method (Malley 法) 直觉**（不解积分）：在单位圆盘 (unit disk) 上均匀采样 (uniform sampling) → 投影 (projection) z = √(1−x²−y²) → 自动等价于半球 cosine-weighted 分布。靠近圆盘边缘的样本投影到很倾斜的弧面，单位立体角密度 (per-solid-angle density) 变小，正好是 cos θ 分布。

```
   均匀采样圆盘                       投影到上半球
   uniform on disk                  project to hemisphere
   ◯ ◯ ◯ ◯ ◯ ◯                          ◯
     ◯ ◯ ◯ ◯                         ◯ ◯ ◯
       ◯ ◯                           ◯ ◯ ◯ ◯
                                     ─────────  ←── 圆盘 (disk)
        ↓                              
   每个点垂直投影到上方的半球

   z = √(1 − x² − y²) = cos θ
```

**一句话总结**：cosine 采样让 Lambertian 估计的 `f/p` 退化为常数 `albedo`，理论方差 (variance) 为 0（不计 Li 自身方差）—— 这是把"重要性采样降方差 (variance reduction)"用到极致的教科书例子。
