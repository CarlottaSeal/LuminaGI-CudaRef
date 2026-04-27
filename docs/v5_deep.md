# v5 深度推导（Path Tracing 间接光）

涵盖 xorshift32 的代数背景、Frisvad ONB (Orthonormal Basis) 的几何动机、cosine-weighted 重要性采样 (importance sampling) 的完整数学推导，以及 Malley's method 的几何证明。

---

## 1. xorshift32 — 为什么这样能"伪随机 (pseudo-random)"

### 1.1 GF(2) 上的线性变换 (linear transformation) 视角

把 32-bit state 看作 GF(2)³² (Galois field of order 2) 上的列向量（每位 0/1 元素）。三条操作：

- `x ^= x << k`：等价于 `x = (I + L^k) · x`，L 是左移矩阵 (left-shift matrix)
- `x ^= x >> k`：等价于 `x = (I + R^k) · x`，R 是右移矩阵 (right-shift matrix)
- 整体一步：`x_new = M · x_old`，M = (I + L⁵)(I + R¹⁷)(I + L¹³)

所以 xorshift 是 32 维 GF(2) 上的可逆线性变换 (invertible linear transformation)，状态空间是 2³² 个向量。

### 1.2 周期性 (Periodicity)

线性变换的轨道 (orbit) 由 M 的"阶 (order)"——最小 n 使 Mⁿ = I——决定。Marsaglia 寻找的是 M 的特征多项式 (characteristic polynomial) 恰为 GF(2) 上次数 32 的**本原多项式 (primitive polynomial)**——这种情况下，M 在非零向量上的轨道是单一长 2³² − 1 的环 (cycle)，绕一遍才回原点。

(13, 17, 5) 是 81 组通过此条件的 triplet 之一，Marsaglia 还跑了 Diehard 等统计测试 (statistical tests) 筛选出经验上"看起来更随机"的几组。

### 1.3 吸收态 (Absorbing State) 详解

`x = 0` 时所有移位+异或都是 0，永远跳不出。代码里的兜底 (fallback)：

```cpp
r.state = x ? x : 0x9E3779B9u;
```

这里 `0x9E3779B9` = `(√5 − 1)/2 × 2³² ≈ 2654435769`，没什么神秘——只是个非零常数。也可以用任何 magic number 比如 `0xDEADBEEF`、`1`、`42`。

### 1.4 实战注意

**坏种子 (bad seed)**：如果所有线程都用 `pixelIndex` 当种子 (seed)，相邻像素的 state 高度相关 (highly correlated)，前几次 xorshift 输出会出现网格条纹 (grid pattern)。修复：用 PCG / Wang hash 把 pixelIndex 先做一次混淆 (hashing)。

```cpp
// 简化的 Wang hash 种子混淆
__device__ uint32_t hash_seed(uint32_t k)
{
    k = (k ^ 61) ^ (k >> 16);
    k *= 9;
    k ^= k >> 4;
    k *= 0x27d4eb2d;
    k ^= k >> 15;
    return k;
}
```

每像素初始化时调用一次，后面再用 xorshift 就稳了。

---

## 2. ONB = OrthoNormal Basis（正交规范基）

### 2.1 数学定义 (Mathematical Definition)

一个 ONB {T, B, N} 满足：

- ‖T‖ = ‖B‖ = ‖N‖ = 1（unit length，单位长度）
- T · B = T · N = B · N = 0（orthogonal，正交）
- T × B = N（右手系 right-handed system，可选）

把 [T B N] 排成列写成矩阵 R = [T | B | N]，那么 R 是正交矩阵 (orthogonal matrix)：R⁻¹ = Rᵀ。

**作用**：把局部坐标向量 (local coordinate vector) v_local 转到世界向量 (world vector) v_world：

$$v_{world} = R \cdot v_{local} = x_l T + y_l B + z_l N$$

逆变换 (inverse transform)：v_local = Rᵀ v_world。

### 2.2 为什么不能 naive cross

最直观的写法：

```cpp
vec3 aux = (fabs(N.y) < 0.9f) ? v3(0,1,0) : v3(1,0,0);
T = normalize(cross(aux, N));
B = cross(N, T);
```

问题：

1. **接缝 (seam)**：在 |N.y| = 0.9 这条线上 aux 突变 → T 突变 → cosine 半球的旋转角度突变 → 渲染图能看到一条亮 / 暗带
2. **退化 (degeneracy)**：如果选 aux 时正好 aux ∥ N，cross 是 0 向量，normalize 爆炸
3. **不连续不可导 (discontinuous, non-differentiable)**：在做 mipmapping、自动微分 (auto-diff / automatic differentiation) 这类需要梯度 (gradient) 的任务时不能用

### 2.3 Frisvad 2012 的闭式公式 (Closed-Form Formula)

Jeppe Revall Frisvad 在 *Journal of Graphics Tools* (JGT) 2012 上的 "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization" 给出：

```cpp
// 北半球版 (N.z >= 0)
float a = 1.0f / (1.0f + N.z);
float b = -N.x * N.y * a;
T = v3(1.0f - N.x * N.x * a, b, -N.x);
B = v3(b, 1.0f - N.y * N.y * a, -N.y);
```

**几何动机 (Geometric Motivation)**：把法线从北极 (north pole) (0,0,1) 用最短弧 (shortest arc) 旋转到 N，T、B 跟着这个旋转走。结果 T、B 是 N 的有理函数 (rational function)，处处连续可导 (continuous and differentiable)，除了 N = (0,0,-1) 这个奇点 (singularity)。

**南极 (south pole) 问题**：N.z = -1 时 `1/(1+N.z)` 爆炸。Pixar 2017 提出用 `copysignf` 把南北两个公式合一：

```cpp
float sign = copysignf(1.0f, N.z);   // +1 if N.z >= 0, -1 if N.z < 0
float a = -1.0f / (sign + N.z);
float b = N.x * N.y * a;
T = v3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
B = v3(b, sign + N.y * N.y * a, -N.y);
```

只有精确等于南极那一个点会爆，实战中 `N.z = -1.0f` 概率为 0（浮点）。

### 2.4 验证 ONB 性质（手算一例）

取 N = (0, 0, 1)（北极）：

- sign = +1, a = -1/(1+1) = -0.5, b = 0
- T = (1 + 1·0·(-0.5), 0, 0) = (1, 0, 0) ✓
- B = (0, 1 + 1·0·(-0.5), 0) = (0, 1, 0) ✓

取 N = (1, 0, 0)（赤道）：

- sign = +1, a = -1/1 = -1, b = 0
- T = (1 + 1·1·(-1), 0, -1) = (0, 0, -1)
- B = (0, 1 + 1·0·(-1), 0) = (0, 1, 0)

验证：T·B = 0 ✓，T·N = 0·1 = 0 ✓，T × B = (0,0,-1) × (0,1,0) = (0·0 - (-1)·1, (-1)·0 - 0·0, 0·1 - 0·0) = (1, 0, 0) = N ✓

---

## 3. 渲染方程 + Monte Carlo + 重要性采样

### 3.1 渲染方程 (Rendering Equation, Kajiya 1986)

某点 x 沿出射方向 (outgoing direction) ωₒ 的辐亮度 (radiance)：

$$L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} L_i(x, \omega_i) \cdot f_r(x, \omega_i, \omega_o) \cdot \cos\theta_i \, d\omega_i$$

- L_e 自发光项 (emitted radiance)，光源本身的发光
- 积分项：从所有入射方向 (incoming direction) ωᵢ 来的入射光，被 BRDF (Bidirectional Reflectance Distribution Function) 散射 (scatter) 后贡献到 ωₒ
- cos θᵢ = N · ωᵢ：Lambert 余弦律 (Lambert's cosine law)。物理来源是斜射时光"被涂在更大的面积上"，单位面积接收能量减少
- Ω：N 上方的半球 (upper hemisphere)，光只从上面来

### 3.2 Monte Carlo (蒙特卡洛) 估计

积分没法解析解 (analytical solution)（场景几何太复杂），用 MC 估计 (estimator)：随便抽 N 个方向 ωᵢ，按密度 (density) p(ωᵢ) 加权：

$$L_o \approx L_e + \frac{1}{N}\sum_{k=1}^{N} \frac{L_i(\omega_k) \cdot f_r \cdot \cos\theta_k}{p(\omega_k)}$$

**性质 (Properties)**：
- 无偏 (unbiased)：E[估计] = 真值，无论 p 怎么选（只要 p > 0 处覆盖被积函数 (integrand) 的支撑集 (support)）
- 收敛速率 (convergence rate)：误差 (error) ∝ 1/√N，跟维度无关 (dimension-independent)，准蒙特卡洛 (Quasi-Monte Carlo, QMC) 例外
- 方差 (variance)：Var[估计] = (1/N) · Var[单样本贡献]

### 3.3 重要性采样降方差 (Variance Reduction) 的数学

单样本贡献 X = f(ω) / p(ω)，方差：

$$\text{Var}[X] = E[X^2] - E[X]^2 = \int \frac{f^2}{p} d\omega - \left(\int f \, d\omega\right)^2$$

第二项 = 真积分值² 是常数。要降方差，最小化第一项 ∫ f²/p。

**最优 p (optimal pdf)**：拉格朗日乘子法 (Lagrange multipliers)，约束 ∫ p = 1：

$$p^*(\omega) = \frac{|f(\omega)|}{\int |f| \, d\omega}$$

也就是 **p 正比于 |f|** (p proportional to |f|)。代入回去 Var[X] = 0（如果 f 不变号 sign-definite）。

**实战意义**：你不知道 f 长什么样（这就是要算积分的原因），但**部分** f 形状是已知的。比如 Lambertian 的 BRDF 部分 cos θ × albedo/π 完全已知，未知的只有入射光 L_i。所以选 p 匹配已知部分即可——`p = cos θ / π`。

### 3.4 数值例子 (Numerical Example) — 一维

估计 ∫₀¹ f(x) dx，f 是以 x = 0.5 为中心的尖峰 (peak)（峰值 10，两侧迅速衰减）。

**均匀采样 (uniform sampling)**（p = 1，4 个样本）：
- 大概率抽不到 0.5 附近
- 估计 ≈ 0.5 ~ 5（看运气）
- 100 次平均：结果在 1.5 ± 0.8

**重要性采样**（p ∝ f，4 个样本）：
- 几乎每次都打到 0.4–0.6
- f/p ≈ 常数（积分值本身）
- 100 次平均：结果在 1.5 ± 0.05

样本数相同，重要性采样**方差小 16 倍**（标准差 1/16）。

### 3.5 Lambertian + cosine 重要性采样

```
被积函数 (integrand):     L_i × albedo/π × cos θ
选 pdf:                          cos θ / π
比值 (ratio) f/p:        L_i × albedo  (cos 与 π 完全消掉)
```

**throughput (能量通量) 累乘解读**：每次反弹 (bounce) 乘 albedo，物理上对应于"反射这一步把光强按 albedo 比例衰减"。第 k 次反弹后 throughput = albedo₁ × albedo₂ × ... × albedo_k。最终像素颜色 = throughput × L_e（撞到光源 light source 时）。

---

## 4. cosine 半球采样的 √u1 推导

### 4.1 立体角微元 (Solid Angle Differential)

球面坐标 (spherical coordinates) (θ, φ)，立体角元 (solid angle element) dω = sin θ dθ dφ。直观：在球面上 (θ, φ) 处取小矩形，θ 方向边长 dθ（沿经线 meridian），φ 方向边长 sin θ dφ（沿纬线 parallel，圆周长按纬度 latitude 缩放）。

### 4.2 把 pdf 从立体角换到 (θ, φ) — 雅可比 (Jacobian) 换元

立体角 pdf p_ω(ω) = cos θ / π。换元 (change of variables) 到 (θ, φ)：

$$p(\theta, \phi) = p_\omega \cdot \left|\frac{d\omega}{d\theta \, d\phi}\right| = \frac{\cos\theta}{\pi} \cdot \sin\theta = \frac{\cos\theta \sin\theta}{\pi}$$

验证归一性：

$$\int_0^{2\pi}\!\!\int_0^{\pi/2} \frac{\cos\theta \sin\theta}{\pi} d\theta d\phi = \frac{2\pi}{\pi}\int_0^{\pi/2}\cos\theta\sin\theta \, d\theta = 2 \cdot \frac{1}{2} = 1 ✓$$

### 4.3 逆累积分布函数法 (Inverse-CDF Method)

θ 和 φ 解耦 (decouple)。

**φ 边缘 (marginal)**：∫₀^{π/2} cos θ sin θ / π dθ = 1/(2π)（常数）→ φ 在 [0, 2π) 上均匀分布 (uniform distribution)。
$$\phi = 2\pi u_2$$

**θ 边缘 pdf**：

$$p(\theta) = \int_0^{2\pi} \frac{\cos\theta\sin\theta}{\pi} d\phi = 2\cos\theta\sin\theta = \sin(2\theta)$$

CDF (Cumulative Distribution Function，累积分布函数)：

$$F(\theta) = \int_0^{\theta} \sin(2t) \, dt = \frac{1 - \cos(2\theta)}{2} = \sin^2\theta$$

反演 (invert)：u₁ = sin²θ → **sin θ = √u₁**, **cos θ = √(1 − u₁)**

### 4.4 转笛卡尔坐标 (Cartesian Coordinates)

```
x = sin θ · cos φ = √u₁ · cos(2π u₂)
y = sin θ · sin φ = √u₁ · sin(2π u₂)
z = cos θ        = √(1 − u₁)
```

代码里：

```cpp
float u1  = rand01(rng);
float u2  = rand01(rng);
float r   = sqrtf(u1);                  // = sin θ
float phi = 2.0f * M_PI * u2;
float x   = r * cosf(phi);
float y   = r * sinf(phi);
float z   = sqrtf(fmaxf(0.0f, 1.0f - u1));   // = cos θ
```

`fmaxf(0, ...)` 防 u1 浮点超过 1 时 sqrt 出 NaN。

---

## 5. Malley's method — 几何证明 cosine 采样

### 5.1 命题 (Proposition)

在单位圆盘 (unit disk) 上**均匀采样** (x, y)，然后令 z = √(1 − x² − y²)，得到的 (x, y, z) 在单位上半球 (unit upper hemisphere) 上服从 **cosine-weighted distribution**（即 pdf = cos θ / π）。

### 5.2 证明 (Proof)

设半球点 ω 对应圆盘点 (x, y) 的关系：x = sin θ cos φ, y = sin θ sin φ。

圆盘面积元 (area element)：

$$dA = dx \, dy = (sin\theta \, d\theta)(\sin\theta \, d\phi) = \sin^2\theta \, d\theta \, d\phi$$

但等下——这个换元 Jacobian 不是直接乘。重做：

(x, y) 极坐标 (r, φ)，r = sin θ，dx dy = r dr dφ。
r dr = sin θ d(sin θ) = sin θ cos θ dθ。
所以 dx dy = sin θ cos θ dθ dφ。

立体角元：dω = sin θ dθ dφ。

比较：

$$dx \, dy = \cos\theta \cdot d\omega$$

或者反向：dω = (1 / cos θ) dA。

### 5.3 推 pdf (Density Derivation)

圆盘均匀分布 (uniform on disk)：p_disk(x, y) = 1/π（圆盘面积 area 是 π）。

转半球密度 (hemisphere density)：

$$p_\omega(\omega) = p_{disk} \cdot \left|\frac{dA}{d\omega}\right| = \frac{1}{\pi} \cdot \cos\theta = \frac{\cos\theta}{\pi}$$

正是要的 cosine pdf。✓

### 5.4 几何直觉 (Geometric Intuition)

圆盘上每个等大小的小圆 → 投影 (project) 到半球时，靠近边缘（θ 大）的小圆变成很倾斜的弧面 (oblique arc patch)，"摊开"占了更大的立体角；靠近中心（θ ≈ 0）的小圆变成几乎垂直于视线的弧面，立体角小。所以**单位立体角内**，靠中心的样本密度大、靠边缘小——正好是 cos θ 形状。

### 5.5 实战 vs 4.4 的 inverse-CDF

两条路径殊途同归。Malley 法在代码层面更短（不用 sin/cos）：

```cpp
// Malley's 直接版
float u1 = rand01(rng), u2 = rand01(rng);
float a = 2.0f * u1 - 1.0f;         // [-1, 1]
float b = 2.0f * u2 - 1.0f;
// 把方形 (a,b) 映射到单位圆盘 (Shirley & Chiu 1997 concentric)... 然后投影
```

但需要 box-to-disk 映射额外几行。我们 v5 还是用 inverse-CDF 那版（5 行清楚明了）。

---

## 6. 整合：v5 路径追踪 (Path Tracing) 算法骨架

```
for each pixel (px, py):
    rng.state = hash_seed(pixel_index ^ frame_seed)
    accum = (0, 0, 0)                        // 累加器 accumulator
    for sample = 0 to spp-1:                  // spp = samples per pixel
        ray = primary_ray(cam, px, py, jitter())
        throughput = (1, 1, 1)                // path throughput
        radiance   = (0, 0, 0)                // accumulated radiance
        for bounce = 0 to maxBounces-1:
            tri = traverse_closest(ray)       // closest-hit BVH traversal
            if no hit:
                radiance += throughput * sky_color
                break
            P, N, mat = unpack(tri)
            if dot(N, ray.d) > 0: N = -N     // 双面 (double-sided) 法线翻转 normal flip
            albedo = sample_albedo(mat, uv)

            // 直接光 (direct lighting) — Next Event Estimation (NEE)
            radiance += throughput * shade_direct(scene, P, N, albedo)

            // 间接光 (indirect lighting) 准备：cosine 采样下一根 ray
            throughput *= albedo              // cos/pdf 抵消，只剩 albedo

            // 俄罗斯轮盘赌 (Russian Roulette, RR) — bounce >= 1 之后
            if bounce >= 1:
                p_continue = max(throughput.x, throughput.y, throughput.z)
                if rand01() > p_continue: break
                throughput /= p_continue      // 无偏补偿 (unbiased compensation)

            ray = make_ray(P + N * 1e-3, sample_cosine_hemisphere(N, rng))
        accum += radiance
    image[px, py] = tonemap(accum / spp)      // 色调映射 tone mapping
```

要点：
- **NEE + indirect 同时用**：每个反弹 (bounce) 都查一次直接光（短路径 short path，方差低），然后才采样间接方向（长路径 long path 补全）
- **俄罗斯轮盘 (Russian Roulette)**：随机以概率 1−p 终止 path，活下来的乘 1/p。无偏 (unbiased)，但能"提前止损"暗淡路径
- **shadow bias (阴影偏移)**：1e-3 沿法线方向 offset，防自相交 (self-intersection)。`make_ray` 内部会再 normalize 方向

---

## 参考文献 (References)

- Marsaglia, "Xorshift RNGs", *Journal of Statistical Software* 8(14), 2003
- Frisvad, "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization", *Journal of Graphics Tools* (JGT) 16(3), 2012
- Pixar / Duff et al., "Building An Orthonormal Basis, Revisited", *Journal of Computer Graphics Techniques* (JCGT) 6(1), 2017
- Kajiya, "The Rendering Equation", *SIGGRAPH 1986*
- Pharr / Jakob / Humphreys, *PBRT (Physically Based Rendering) 4th ed.* — Chap. 13–14 covers everything above with rigor
