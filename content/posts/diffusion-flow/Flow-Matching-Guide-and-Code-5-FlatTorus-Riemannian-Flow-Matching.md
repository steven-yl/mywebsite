---
title: "Flow Matching Guide and Code 第5章解读：FlatTorus Riemannian Flow Matching 训练逻辑技术文档"
date: 2024-06-10T00:00:00+08:00
draft: false
authors: [Steven]
description: "解析 examples/2d_riemannian_flow_matching_flat_torus.ipynb 中平坦环面 RFM 的训练逻辑：超参、单步训练循环、测地线路径与条件速度拟合的数学与代码调用链。"
summary: "平坦环面 M=[0,2π)² 上 Riemannian Flow Matching 的训练目标、概率路径、损失形式及实现细节的技术文档。"
tags: [diffusion/flow, flow matching, Riemannian, 环面, 代码解读]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 10
---

本文档详细说明 `examples/2d_riemannian_flow_matching_flat_torus.ipynb` 中**训练块**（超参数设置 + 单步训练循环）的代码逻辑、数学原理与函数调用链。

---

## 1. 总览

训练目标：在**平坦环面** $M = [0, 2\pi)^2$ 上学习一个**速度场** $v_\theta(x, t)$，使得从先验 $p_0$ 沿 ODE $\dot{x} = v_\theta(x, t)$ 积分到 $t=1$ 时，得到数据分布 $p_1$。

- **概率路径**：测地线路径，$X_t = \exp_{X_0}(\alpha_t \cdot \log_{X_0}(X_1))$，$\alpha_t$ 由调度器给出（本示例中 `CondOTScheduler` 取 $\alpha_t = t$）。
- **损失**：Flow matching 的 L2 损失，拟合条件速度 $\dot{X}_t$。

---

## 2. 代码结构概览

### 2.0 模块与对象层次

| 层级 | 名称 | 说明 |
|------|------|------|
| 流形 | `manifold` | `FlatTorus()`，定义 $[0,2\pi)^2$ 上的 `projx` / `proju` / `expmap` / `logmap` |
| 调度器 | `scheduler` | `CondOTScheduler()`，提供 $\alpha_t=t$、$\sigma_t=1-t$ 等 |
| 概率路径 | `path` | `GeodesicProbPath(scheduler, manifold)`，封装测地线路径与 `sample(t, x_0, x_1)` |
| 速度场 | `vf` | `ProjectToTangent(MLP, manifold)`，训练与推理时统一入口 `vf(x, t)` |
| 优化器 | `optim` | `torch.optim.Adam(vf.parameters(), lr)`，只更新 `vf` 参数 |

**训练循环入口**：单步顺序为  
采样端点 → `wrap` 上流形 → 采样 $t$ → `path.sample` 得 $(x_t, \dot{x}_t)$ → `vf(x_t, t)` 与 `dx_t` 算 L2 损失 → `backward` / `step`。

**数据流**：`inf_train_gen` / `randn_like` → `wrap(manifold, ·)` → `path.sample` → `PathSample(x_t, dx_t, ...)` → `vf(x_t, t)` 与 `dx_t` 求差、均方 → `loss`。

---

## 3. 超参数与对象初始化

### 3.1 超参数

| 变量 | 含义 | 本示例取值 |
|------|------|------------|
| `lr` | 学习率 | 0.001 |
| `batch_size` | 每步采样的 $(X_0, X_1)$ 对数 | 4096 |
| `iterations` | 总迭代步数 | 5001 |
| `print_every` | 打印间隔 | 1000 |
| `manifold` | 流形实例 | `FlatTorus()`，表示 $[0, 2\pi)^2$ |
| `dim` | 流形维度 | 2 |
| `hidden_dim` | MLP 隐藏层维度 | 512 |

### 3.2 速度场模型 `vf`

```text
vf = ProjectToTangent(MLP(...), manifold=manifold)
vf.to(device)
```

- **MLP**：输入 $(x, t)$（流形坐标 + 时间），输出 $\mathbb{R}^{\mathrm{dim}}$ 的向量（欧氏速度）。
- **ProjectToTangent**：对输入 $x$ 做 `manifold.projx(x)`，再对 MLP 输出做 `manifold.proju(x, v)`，保证输出是 $x$ 处切空间中的向量。
- **调用链**（推理/训练时）：
  - `vf(x, t)` → `ProjectToTangent.forward(x, t)`
  - → `x' = manifold.projx(x)`（环面：$x' = x \bmod 2\pi$）
  - → `v = MLP(x', t)`
  - → `v' = manifold.proju(x', v)`（FlatTorus 上恒等）
  - → 返回 `v'`

### 3.3 概率路径 `path`

```text
path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)
```

- **GeodesicProbPath**：基于流形测地线的概率路径；需要 `ConvexScheduler` 提供 $\alpha_t$。
- **CondOTScheduler**：$\alpha_t = t$，$\sigma_t = 1 - t$，$\dot{\alpha}_t = 1$，$\dot{\sigma}_t = -1$。
- **path.sample(t, x_0, x_1)** 的数学与实现见下文 §5。

### 3.4 优化器

```text
optim = torch.optim.Adam(vf.parameters(), lr=lr)
```

仅优化 `vf`（即 `ProjectToTangent` 内的 MLP）的参数。

---

## 4. 单步训练循环：逻辑与数据流

每一步迭代完成以下流程（顺序与代码一致）。

### 4.1 清空梯度

```python
optim.zero_grad()
```

为当前步的 `loss.backward()` 做准备。

### 4.2 从耦合 $\pi(X_0, X_1)$ 采样

- **x_1**：数据端  
  - 调用 `inf_train_gen(batch_size=batch_size, device=device)`  
  - 在平面区域上生成棋盘格状样本，形状 `(batch_size, 2)`，数值约在 $[-2,2]^2$ 或类似范围。

- **x_0**：先验端  
  - `x_0 = torch.randn_like(x_1).to(device)`  
  - 即 $X_0 \sim \mathcal{N}(0, I)$（与 x_1 同 shape）。

### 4.3 将端点投影到流形 $[0, 2\pi)^2$

```python
x_1 = wrap(manifold, x_1)
x_0 = wrap(manifold, x_0)
```

- **wrap(manifold, samples)** 实现：  
  - `center = zeros_like(samples)`  
  - `return manifold.expmap(center, samples)`  
- 对 **FlatTorus**：`expmap(0, u) = u % (2π)`，因此  
  - `x_1`、`x_0` 被映射到 $[0, 2\pi)^2$，保证路径两端都在流形上。

**调用关系**：  
`wrap` → `manifold.expmap(0, samples)` → FlatTorus 上等价于 `samples % (2π)`。

### 4.4 采样时间 $t$

```python
t = torch.rand(x_1.shape[0]).to(device)
```

- 每个样本一个 $t \in [0,1]$，shape `(batch_size,)`。

### 4.5 沿概率路径采样 $(X_t, \dot{X}_t)$

```python
path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
```

- **path**：`GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)`  
- **path.sample(x_0, x_1, t)**（即 `GeodesicProbPath.sample(x_0, x_1, t)`）返回 `PathSample`，包含：
  - `x_t`：路径在时间 $t$ 上的点，流形坐标 $[0,2\pi)^2$
  - `dx_t`：路径在 $t$ 处对时间的导数（条件目标速度）
  - `x_0`, `x_1`, `t`（同输入）

**内部函数调用链**（见 §5）：

1. `expand_tensor_like(t, x_1[..., 0:1])`，使 `t` 与 batch 维度对齐。
2. 对每个 $(x_0^{(i)}, x_1^{(i)}, t^{(i)})$（通过 `vmap`）：
   - `geodesic(manifold, x_0, x_1)` → 得到可调用对象 `path(τ)`，满足  
     $path(\tau) = \exp_{x_0}(\tau \cdot \log_{x_0}(x_1))$，$\tau \in [0,1]$。
   - `scheduler(t)` → `SchedulerOutput(alpha_t=t, sigma_t=1-t, d_alpha_t=1, d_sigma_t=-1)`（CondOT 下 $\alpha_t = t$）。
   - `path(alpha_t)` = `path(t)` = 测地线在 $t$ 处的点 $X_t$。
   - `jvp`：对 $\tau \mapsto path(\mathrm{scheduler}(\tau).\mathrm{alpha\_t})$ 在 $\tau=t$ 处求导（CondOT 下即对 $path(\tau)$ 在 $\tau=t$ 求导），得到 $X_t$ 与 $\frac{\partial}{\partial t}X_t = \dot{X}_t$（即 `dx_t`）。
3. 将 `vmap` 结果 reshape 成与 `x_1` 相同 shape，构造 `PathSample(x_t, dx_t, x_1, x_0, t)` 并返回。

### 4.6 计算 Flow Matching L2 损失

```python
loss = torch.pow(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()
```

- **path_sample.x_t**：流形上点，坐标在 $[0, 2\pi)^2$（与 FlatTorus 定义一致）。  
- **path_sample.t**：与 `x_t` 一一对应的时间，shape 与 batch 兼容。  
- **vf(path_sample.x_t, path_sample.t)**：  
  - 调用 `ProjectToTangent.forward(x_t, t)`  
  - 内部：`projx(x_t)` → MLP → `proju(x_t, v)`，输出切空间中的预测速度，shape 与 `x_t` 一致。  
- **path_sample.dx_t**：测地线在 $t$ 处的真实条件速度 $\dot{X}_t$（目标）。  
- **损失**：$\mathcal{L} = \mathbb{E}\bigl[\|v_\theta(X_t, t) - \dot{X}_t\|^2\bigr]$，即 L2 拟合速度场。

### 4.7 反向传播与参数更新

```python
loss.backward()
optim.step()
```

- 梯度只流入 `vf`（MLP + ProjectToTangent 的 projx/proju 若可微则参与，FlatTorus 的 projx/proju 可微）。

### 4.8 日志

每隔 `print_every` 步打印当前迭代数、平均每步耗时和当前 `loss.item()`。

---

## 5. 关键函数调用说明

### 5.1 函数签名与角色速查

| 函数 / 方法 | 签名（要点） | 调用者 | 返回值 / 作用 |
|-------------|----------------|--------|----------------|
| `inf_train_gen` | `(batch_size, device)` | 训练循环 | 数据端样本 `x_1`，形状 `(B, 2)` |
| `wrap` | `(manifold, samples)` | 训练循环 | 投影到流形后的样本，$[0,2\pi)^2$ |
| `path.sample` | `(t, x_0, x_1)` | 训练循环 | `PathSample(x_t, dx_t, x_0, x_1, t)` |
| `vf` | `(x, t)` | 训练循环、推理 | 切空间速度向量，与 `x` 同 shape |
| `geodesic` | `(manifold, x_0, x_1)` | `GeodesicProbPath.sample` 内 | 可调用 `path(τ)`，$\tau\in[0,1]$ |
| `scheduler(t)` | `(t)` | `GeodesicProbPath.sample` 内 | `SchedulerOutput(alpha_t, sigma_t, ...)` |
| `manifold.projx` | `(x)` | `ProjectToTangent.forward` | 投影到流形，FlatTorus 为 `x % (2π)` |
| `manifold.proju` | `(x, v)` | `ProjectToTangent.forward` | 投影到切空间，FlatTorus 恒等 |
| `manifold.expmap` | `(x, u)` | `wrap`、`geodesic` 返回的 path | 指数映射，FlatTorus 为 `(x+u) % (2π)` |
| `manifold.logmap` | `(x, y)` | `geodesic` 内部 | 对数映射，FlatTorus 为 `atan2(sin(y-x), cos(y-x))` |

### 5.2 `geodesic(manifold, x_0, x_1)`（`flow_matching.utils.manifolds.utils`）

- **作用**：构造从 $x_0$ 到 $x_1$ 的测地线，参数化在 $\tau \in [0,1]$。
- **数学**：
  - $u = \mathrm{logmap}(x_0, x_1)$（从 $x_0$ 指向 $x_1$ 的切向量）
  - $path(\tau) = \mathrm{expmap}(x_0, \tau \cdot u)$
- **FlatTorus**：
  - `logmap(x, y) = atan2(sin(y-x), cos(y-x))`（最短方向，考虑周期）
  - `expmap(x, u) = (x + u) % (2π)`
- **返回**：可调用对象 `path(τ)`，输入 `τ` 的 shape 与 batch 兼容，输出流形上点的 shape 与 `x_0`/`x_1` 一致。

### 5.3 `CondOTScheduler.__call__(t)`

- **输入**：`t`，shape `(batch_size,)` 或可广播。
- **输出**：`SchedulerOutput`：
  - `alpha_t = t`
  - `sigma_t = 1 - t`
  - `d_alpha_t = 1`
  - `d_sigma_t = -1`
- 本示例中只用到 `alpha_t`，用于测地线参数：$X_t = path(\alpha_t) = path(t)$。

### 5.4 `GeodesicProbPath.sample(x_0, x_1, t)` 内部

- **assert_sample_shape**：检查 `x_0, x_1, t` 的 batch 等维度兼容性。
- **t** 经 `expand_tensor_like` 与 `x_1[..., 0:1]` 对齐，便于与 `x_0, x_1` 一起做 `vmap`。
- **cond_u(x_0, x_1, t)**（单样本逻辑，再被 vmap）：
  1. `path = geodesic(manifold, x_0, x_1)`  
  2. `alpha_t = scheduler(t).alpha_t`（CondOT 下即为 `t`）  
  3. `x_t, dx_t = jvp(lambda t: path(scheduler(t).alpha_t), (t,), (1,))`  
     - 对 $\tau \mapsto path(\tau)$ 在 $\tau = t$ 处求导，得到 $path(t)$ 和 $\frac{d}{d\tau}path(\tau)\big|_{\tau=t}$；由于 $\tau=t$ 时 $\frac{d}{dt}\alpha_t = 1$，这里得到的就是 $\dot{X}_t$。
- **vmap(cond_u)(x_0, x_1, t)**：对整批做上述计算。
- **reshape_as(x_1)**：保证 `x_t`、`dx_t` 的 shape 与 `x_1` 一致。
- **返回**：`PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)`。

### 5.5 `vf(x, t)` = `ProjectToTangent.forward(x, t)`

- `x = manifold.projx(x)`：FlatTorus 上为 `x % (2π)`，保证 $x$ 在 $[0, 2\pi)^2$。
- `v = vecfield(x, t)`：MLP 在欧氏空间预测速度向量。
- `v = manifold.proju(x, v)`：FlatTorus 上切空间即 $\mathbb{R}^2$，故恒等。
- 返回的 `v` 即模型给出的切空间速度，与 `path_sample.dx_t` 的坐标与物理意义一致。

### 5.6 函数调用关系总览

```
训练循环 (每步)
├── inf_train_gen(batch_size, device)     → x_1
├── torch.randn_like(x_1)                 → x_0
├── wrap(manifold, x_1), wrap(manifold, x_0)
│   └── manifold.expmap(0, samples)
├── torch.rand(...)                       → t
├── path.sample(t, x_0, x_1)
│   ├── expand_tensor_like(t, x_1)
│   └── vmap(cond_u)(x_0, x_1, t)
│       ├── geodesic(manifold, x_0, x_1)  → path(τ)，内部用 logmap/expmap
│       ├── scheduler(t)                  → alpha_t, sigma_t, ...
│       └── jvp(path(scheduler(·).alpha_t), t, 1) → x_t, dx_t
├── vf(path_sample.x_t, path_sample.t)
│   ├── manifold.projx(x)
│   ├── MLP(x, t)
│   └── manifold.proju(x, v)
└── loss = mean((vf(...) - dx_t)²); loss.backward(); optim.step()
```

| 调用方向 | 说明 |
|----------|------|
| 训练循环 → `wrap` | 将端点投影到 $[0,2\pi)^2$ |
| 训练循环 → `path.sample` | 得到路径上点与条件速度 $(x_t, \dot{x}_t)$ |
| 训练循环 → `vf` | 预测速度，与 `dx_t` 求 L2 损失 |
| `path.sample` → `geodesic` | 构造单条测地线可调用对象 |
| `path.sample` → `scheduler` | 取 $\alpha_t$ 与路径参数化 |
| `path.sample` → `jvp` | 对路径在 $t$ 处求导得 $\dot{X}_t$ |
| `vf` → `manifold.projx` / `proju` | 保证输入输出在流形与切空间上 |

---

## 6. 数据与坐标约定

- **流形坐标**：FlatTorus 全程使用 $[0, 2\pi)^2$。  
  - `inf_train_gen` 生成的数据经 `wrap` 后落入该域；  
  - `x_0` 经 `wrap` 后也在该域；  
  - `path.sample` 得到的 `x_t`、`dx_t` 以及 `vf` 的输入/输出均在该坐标下。

- **时间**：$t \in [0,1]$，$t=0$ 对应先验端，$t=1$ 对应数据端；CondOT 下测地线参数 $\alpha_t = t$，故 $X_t$ 即为从 $x_0$ 到 $x_1$ 的测地线在 $t$ 处的点。

---

## 7. 小结：单步训练的数据与调用顺序

```text
inf_train_gen(batch_size, device)  →  x_1 (平面棋盘格)
torch.randn_like(x_1)              →  x_0 (高斯)
wrap(manifold, x_1), wrap(manifold, x_0)  →  x_1, x_0 ∈ [0,2π)²
torch.rand(...)                    →  t ∈ [0,1]

path.sample(t, x_0, x_1)
  → geodesic(manifold, x_0, x_1)   →  path(τ)
  → scheduler(t).alpha_t           →  α_t = t
  → jvp(path(α_t), t, 1)           →  x_t, dx_t
  → PathSample(x_t, dx_t, x_1, x_0, t)

vf(path_sample.x_t, path_sample.t)
  → projx(x_t) → MLP(x_t, t) → proju(x_t, v)  →  v_θ(x_t, t)

loss = mean(|v_θ(x_t,t) - dx_t|²)
loss.backward(); optim.step()
```

以上即平坦环面示例中训练块的完整逻辑与函数调用说明。
