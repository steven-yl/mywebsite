---
title: "Flow Matching Guide and Code(项目解析)"
subtitle: ""
date: 2026-02-28T19:37:39+08:00
# lastmod: 2026-02-28T19:37:39+08:00
draft: false
authors: [Steven]
description: "Flow Matching 技术文档"

tags: [diffusion/flow, tutorial]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 1
series_weight: 1

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: "/mywebsite/posts/images/flow-matching-guide-and-code-项目解析.webp"
---                                                                                  


> 本文档为 `flow_matching` 项目的完整技术文档，涵盖项目结构、算法原理、核心代码解析及使用方式。
> 项目地址：https://github.com/facebookresearch/flow_matching
> 论文：[Flow Matching Guide and Code (arXiv:2412.06264)](https://arxiv.org/abs/2412.06264)

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目结构](#2-项目结构)
3. [安装与环境配置](#3-安装与环境配置)
4. [算法原理](#4-算法原理)
   - 4.1 [连续 Flow Matching](#41-连续-flow-matching)
   - 4.2 [离散 Flow Matching](#42-离散-flow-matching)
   - 4.3 [黎曼 Flow Matching](#43-黎曼-flow-matching)
5. [核心模块详解](#5-核心模块详解)
   - 5.1 [概率路径 (path)](#51-概率路径-path)
   - 5.2 [调度器 (scheduler)](#52-调度器-scheduler)
   - 5.3 [损失函数 (loss)](#53-损失函数-loss)
   - 5.4 [求解器 (solver)](#54-求解器-solver)
   - 5.5 [工具模块 (utils)](#55-工具模块-utils)
6. [使用指南与代码示例](#6-使用指南与代码示例)
7. [依赖关系](#7-依赖关系)

---

## 1. 项目概述

`flow_matching` 是由 Meta (Facebook Research) 开发的 PyTorch 库，实现了 Flow Matching 系列生成模型算法。
该库支持三种核心范式：

- **连续 Flow Matching**：在欧几里得空间中通过仿射概率路径学习速度场
- **离散 Flow Matching**：在离散状态空间（如文本 token）中通过混合概率路径学习转移概率
- **黎曼 Flow Matching**：在流形（球面、环面等）上通过测地线路径学习速度场

核心依赖：Python ≥ 3.9、PyTorch ≥ 2.1、torchdiffeq、numpy。当前版本：`1.0.10`。

---

## 2. 项目结构

```
flow_matching/                     # 核心库
├── __init__.py                    # 版本号定义 (__version__ = "1.0.10")
├── path/                          # 概率路径模块
│   ├── path.py                    # ProbPath 抽象基类
│   ├── path_sample.py             # PathSample / DiscretePathSample 数据类
│   ├── affine.py                  # AffineProbPath（仿射路径）、CondOTProbPath
│   ├── geodesic.py                # GeodesicProbPath（测地线路径）
│   ├── mixture.py                 # MixtureDiscreteProbPath（离散混合路径）
│   └── scheduler/                 # 调度器子模块
│       ├── scheduler.py           # Scheduler 基类及多种调度器实现
│       └── schedule_transform.py  # ScheduleTransformedModel（调度器变换）
├── loss/                          # 损失函数模块
│   └── generalized_loss.py        # MixturePathGeneralizedKL（广义 KL 损失）
├── solver/                        # 求解器模块
│   ├── solver.py                  # Solver 抽象基类
│   ├── ode_solver.py              # ODESolver（连续 ODE 求解器）
│   ├── discrete_solver.py         # MixtureDiscreteEulerSolver（离散 Euler 求解器）
│   ├── riemannian_ode_solver.py   # RiemannianODESolver（黎曼 ODE 求解器）
│   └── utils.py                   # 求解器工具函数
└── utils/                         # 工具模块
    ├── utils.py                   # expand_tensor_like, gradient, unsqueeze_to_match
    ├── model_wrapper.py           # ModelWrapper 抽象基类
    ├── categorical_sampler.py     # categorical 采样器
    └── manifolds/                 # 流形子模块
        ├── manifold.py            # Manifold 抽象基类、Euclidean
        ├── sphere.py              # Sphere（超球面）
        ├── torus.py               # FlatTorus（平坦环面）
        └── utils.py               # geodesic 测地线工具函数

examples/                          # 示例代码
├── 2d_flow_matching.ipynb                          # 2D 连续 Flow Matching
├── 2d_discrete_flow_matching.ipynb                 # 2D 离散 Flow Matching
├── 2d_riemannian_flow_matching_flat_torus.ipynb    # 2D 黎曼 FM（环面）
├── 2d_riemannian_flow_matching_sphere.ipynb        # 2D 黎曼 FM（球面）
├── 2d_cnf_maximum_likelihood.ipynb                 # 2D CNF 最大似然
├── standalone_flow_matching.ipynb                  # 独立连续 FM 示例
├── standalone_discrete_flow_matching.ipynb         # 独立离散 FM 示例
├── image/                         # 图像生成示例（CIFAR10 / ImageNet）
│   ├── train.py                   # 训练入口
│   ├── models/                    # UNet / Discrete UNet 模型
│   └── training/                  # 训练循环、数据变换、分布式等
└── text/                          # 文本生成示例
    ├── train.py                   # 训练入口
    ├── model/                     # Transformer + Rotary Embedding
    ├── data/                      # 数据加载与 Tokenizer
    └── logic/                     # 训练/生成/评估逻辑
```

### 模块依赖关系图

```
utils (基础工具)
  ├── manifolds (流形定义)
  └── model_wrapper (模型封装)
       │
path (概率路径)  ←── scheduler (调度器)
  │
  ├── loss (损失函数，依赖 path)
  └── solver (求解器，依赖 utils.model_wrapper)
```

---

## 3. 安装与环境配置

### 快速安装（pip）

```bash
pip install flow_matching
```

### 开发环境（conda）

```bash
conda env create -f environment.yml
conda activate flow_matching
pip install -e .          # 可编辑模式安装
pre-commit install        # 安装代码规范检查钩子
```

### 核心依赖

| 依赖 | 用途 |
|------|------|
| `torch` (≥2.1) | 张量计算与自动微分 |
| `numpy` | 数值计算 |
| `torchdiffeq` | ODE 数值积分（Euler、Dopri5 等） |

---

## 4. 算法原理

### 4.1 连续 Flow Matching

Flow Matching 的核心思想是学习一个时间依赖的速度场 $u_t(x)$，使得沿该速度场的 ODE 流能将源分布 $p_0$（通常为高斯噪声）变换为目标分布 $p_1$（数据分布）。

#### 核心公式

给定源样本 $X_0 \sim p_0$ 和目标样本 $X_1 \sim p_1$，定义仿射条件概率路径：

$$X_t = \alpha_t X_1 + \sigma_t X_0$$

其中 $\alpha_t$ 和 $\sigma_t$ 由调度器（Scheduler）控制。条件速度场为：

$$\dot{X}_t = \dot{\alpha}_t X_1 + \dot{\sigma}_t X_0$$

训练目标是最小化模型预测速度与条件速度之间的 MSE：

$$\mathcal{L} = \mathbb{E}_{t, X_0, X_1} \left[ \| v_\theta(X_t, t) - \dot{X}_t \|^2 \right]$$

#### 对应代码

仿射路径的核心实现在 `flow_matching/path/affine.py` 的 `AffineProbPath.sample()` 中：

```python
# AffineProbPath.sample() 核心逻辑
scheduler_output = self.scheduler(t)  # 获取 α_t, σ_t 及其导数

alpha_t = expand_tensor_like(input_tensor=scheduler_output.alpha_t, expand_to=x_1)
sigma_t = expand_tensor_like(input_tensor=scheduler_output.sigma_t, expand_to=x_1)
d_alpha_t = expand_tensor_like(input_tensor=scheduler_output.d_alpha_t, expand_to=x_1)
d_sigma_t = expand_tensor_like(input_tensor=scheduler_output.d_sigma_t, expand_to=x_1)

# 构造 X_t = σ_t * X_0 + α_t * X_1
x_t = sigma_t * x_0 + alpha_t * x_1
# 条件速度 dX_t = dσ_t * X_0 + dα_t * X_1
dx_t = d_sigma_t * x_0 + d_alpha_t * x_1

return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)
```

#### 条件最优传输路径（CondOT）

最简单也最常用的调度器是条件最优传输调度器，定义为：

$$\alpha_t = t, \quad \sigma_t = 1 - t$$

此时路径为直线插值 $X_t = (1-t)X_0 + tX_1$，速度为常数 $\dot{X}_t = X_1 - X_0$。

```python
class CondOTScheduler(ConvexScheduler):
    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )
```

#### 表示转换

`AffineProbPath` 提供了六种表示之间的相互转换方法，这在不同训练目标之间切换时非常有用：

| 方法 | 输入 → 输出 | 公式 |
|------|-------------|------|
| `target_to_velocity` | $X_1 \to \dot{X}_t$ | $\dot{X}_t = \frac{\dot{\sigma}_t}{\sigma_t} X_t + \frac{\dot{\alpha}_t \sigma_t - \dot{\sigma}_t \alpha_t}{\sigma_t} X_1$ |
| `epsilon_to_velocity` | $\epsilon \to \dot{X}_t$ | $\dot{X}_t = \frac{\dot{\alpha}_t}{\alpha_t} X_t + \frac{\dot{\sigma}_t \alpha_t - \dot{\alpha}_t \sigma_t}{\alpha_t} \epsilon$ |
| `velocity_to_target` | $\dot{X}_t \to X_1$ | 上式的逆变换 |
| `epsilon_to_target` | $\epsilon \to X_1$ | $X_1 = \frac{1}{\alpha_t} X_t - \frac{\sigma_t}{\alpha_t} \epsilon$ |
| `velocity_to_epsilon` | $\dot{X}_t \to \epsilon$ | 速度到噪声的转换 |
| `target_to_epsilon` | $X_1 \to \epsilon$ | $\epsilon = \frac{1}{\sigma_t} X_t - \frac{\alpha_t}{\sigma_t} X_1$ |

### 4.2 离散 Flow Matching

离散 Flow Matching 将 Flow Matching 框架扩展到离散状态空间（如文本 token），使用连续时间马尔可夫链（CTMC）代替 ODE。

#### 核心公式

在离散空间 $\mathcal{S} = [K]^d$ 上，混合概率路径定义为：

$$P(X_t = X_0) = \sigma_t, \quad P(X_t = X_1) = 1 - \sigma_t$$

即在时间 $t$，每个坐标以概率 $\sigma_t$ 保持为源值 $X_0$，以概率 $1-\sigma_t$ 翻转为目标值 $X_1$。

条件概率速度场为：

$$u_t^i(x^i, y^i | x_1^i) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta_{x_1^i}(x^i) - \delta_{y^i}(x^i) \right]$$

#### 对应代码

离散路径采样在 `flow_matching/path/mixture.py` 中实现：

```python
class MixtureDiscreteProbPath(ProbPath):
    def sample(self, x_0, x_1, t) -> DiscretePathSample:
        sigma_t = self.scheduler(t).sigma_t
        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        # 每个坐标独立地以概率 σ_t 保持为 X_0，否则翻转为 X_1
        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t
        x_t = torch.where(condition=source_indices, input=x_0, other=x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)
```

后验到速度的转换：

```python
def posterior_to_velocity(self, posterior_logits, x_t, t):
    posterior = torch.softmax(posterior_logits, dim=-1)
    x_t = F.one_hot(x_t, num_classes=vocabulary_size)

    scheduler_output = self.scheduler(t)
    kappa_t = scheduler_output.alpha_t
    d_kappa_t = scheduler_output.d_alpha_t

    # u_t = (dκ_t / (1 - κ_t)) * (posterior - x_t)
    return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)
```

#### 广义 KL 损失

离散 Flow Matching 使用广义 KL 散度作为训练损失（`flow_matching/loss/generalized_loss.py`）：

$$\ell_i(x_1, x_t, t) = -\frac{\dot{\kappa}_t}{1-\kappa_t} \left[ p_{1|t}(x_t^i|x_t) - \delta_{x_1^i}(x_t^i) + (1-\delta_{x_1^i}(x_t^i)) \log p_{1|t}(x_1^i|x_t) \right]$$

```python
class MixturePathGeneralizedKL(_Loss):
    def forward(self, logits, x_1, x_t, t):
        # 提取 log p_{1|t}(x_1|x_t)
        log_p_1t = torch.log_softmax(logits, dim=-1)
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1.unsqueeze(-1))

        # 提取 p_{1|t}(x_t|x_t)
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t.unsqueeze(-1))

        # 计算跳跃系数 dκ_t / (1 - κ_t)
        scheduler_output = self.path.scheduler(t)
        jump_coefficient = scheduler_output.d_alpha_t / (1 - scheduler_output.alpha_t)

        delta_x1_xt = (x_t == x_1).to(log_p_1t.dtype)

        loss = -jump_coefficient * (
            p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1
        )
        return torch.mean(loss)  # 默认 reduction='mean'
```

### 4.3 黎曼 Flow Matching

黎曼 Flow Matching 将 Flow Matching 扩展到非欧几里得流形上，使用测地线插值代替线性插值。

#### 核心公式

在流形 $\mathcal{M}$ 上，测地线概率路径定义为：

$$X_t = \psi_t(X_0 | X_1) = \exp_{X_1}(\kappa_t \log_{X_1}(X_0))$$

其中 $\exp$ 和 $\log$ 分别是流形上的指数映射和对数映射，$\kappa_t$ 是调度器参数。

#### 对应代码

测地线路径在 `flow_matching/path/geodesic.py` 中实现：

```python
class GeodesicProbPath(ProbPath):
    def __init__(self, scheduler: ConvexScheduler, manifold: Manifold):
        self.scheduler = scheduler
        self.manifold = manifold

    def sample(self, x_0, x_1, t) -> PathSample:
        def cond_u(x_0, x_1, t):
            # 构造测地线路径函数
            path = geodesic(self.manifold, x_0, x_1)
            # 用 JVP（Jacobian-Vector Product）自动计算速度
            x_t, dx_t = jvp(
                lambda t: path(self.scheduler(t).alpha_t),
                (t,),
                (torch.ones_like(t).to(t),),
            )
            return x_t, dx_t

        # 使用 vmap 对 batch 维度进行向量化
        x_t, dx_t = vmap(cond_u)(x_0, x_1, t)
        return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)
```

测地线工具函数（`flow_matching/utils/manifolds/utils.py`）：

```python
def geodesic(manifold, start_point, end_point):
    """生成参数化的测地线曲线函数"""
    shooting_tangent_vec = manifold.logmap(start_point, end_point)

    def path(t):
        tangent_vecs = torch.einsum("i,...k->...ik", t, shooting_tangent_vec)
        points_at_time_t = manifold.expmap(start_point.unsqueeze(-2), tangent_vecs)
        return points_at_time_t

    return path
```

#### 支持的流形

| 流形 | 类名 | 空间 | 指数映射 | 对数映射 |
|------|------|------|----------|----------|
| 欧几里得空间 | `Euclidean` | $\mathbb{R}^D$ | $\exp_x(u) = x + u$ | $\log_x(y) = y - x$ |
| 超球面 | `Sphere` | $S^{D-1}$ | 球面指数映射 | 球面对数映射 |
| 平坦环面 | `FlatTorus` | $[0, 2\pi]^D$ | $(x + u) \mod 2\pi$ | $\text{atan2}(\sin(y-x), \cos(y-x))$ |

球面的指数映射实现（`flow_matching/utils/manifolds/sphere.py`）：

```python
class Sphere(Manifold):
    def expmap(self, x, u):
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.projx(x + u)
        cond = norm_u > self.EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)  # 数值稳定：小范数时退化为投影

    def logmap(self, x, y):
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(self.EPS[x.dtype])
        return torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(self.EPS[x.dtype]), u)

    def projx(self, x):
        return x / x.norm(dim=-1, keepdim=True)  # 投影到单位球面

    def proju(self, x, u):
        return u - (x * u).sum(dim=-1, keepdim=True) * x  # 投影到切平面
```

---

## 5. 核心模块详解

### 5.1 概率路径 (path)

概率路径是 Flow Matching 的核心抽象，定义了如何在源分布和目标分布之间构造插值。

#### 类继承关系

```
ProbPath (抽象基类)
├── AffineProbPath (仿射路径)
│   └── CondOTProbPath (条件最优传输路径)
├── GeodesicProbPath (测地线路径)
└── MixtureDiscreteProbPath (离散混合路径)
```

#### PathSample 数据结构

所有路径的 `sample()` 方法返回 `PathSample` 或 `DiscretePathSample`：

```python
@dataclass
class PathSample:
    x_1: Tensor   # 目标样本 X_1
    x_0: Tensor   # 源样本 X_0
    t: Tensor      # 时间 t
    x_t: Tensor    # 路径样本 X_t ~ p_t
    dx_t: Tensor   # 条件速度 dX_t/dt

@dataclass
class DiscretePathSample:
    x_1: Tensor   # 目标样本 X_1
    x_0: Tensor   # 源样本 X_0
    t: Tensor      # 时间 t
    x_t: Tensor    # 路径样本 X_t ~ p_t（无速度，因为是离散空间）
```

### 5.2 调度器 (scheduler)

调度器控制概率路径的时间演化参数 $\alpha_t$ 和 $\sigma_t$。

#### 调度器类型一览

| 调度器 | $\alpha_t$ | $\sigma_t$ | 特点 |
|--------|-----------|-----------|------|
| `CondOTScheduler` | $t$ | $1-t$ | 最简单的线性插值，条件最优传输 |
| `PolynomialConvexScheduler(n)` | $t^n$ | $1-t^n$ | 多项式调度，$n$ 控制曲线形状 |
| `VPScheduler` | $e^{-T/2}$ | $\sqrt{1-e^{-T}}$ | 方差保持调度（VP-SDE 等价） |
| `LinearVPScheduler` | $t$ | $\sqrt{1-t^2}$ | 线性方差保持 |
| `CosineScheduler` | $\sin(\frac{\pi}{2}t)$ | $\cos(\frac{\pi}{2}t)$ | 余弦调度 |

所有调度器都实现了 `snr_inverse` 方法，用于从信噪比 $\text{SNR} = \alpha_t / \sigma_t$ 反推时间 $t$，这在调度器变换中至关重要。

#### 调度器变换（Schedule Transform）

`ScheduleTransformedModel` 允许在训练后更换调度器而无需重新训练模型。其核心是尺度-时间（ST）变换：

$$\bar{X}_r = s_r X_{t_r}, \quad t_r = \rho^{-1}(\bar{\rho}(r)), \quad s_r = \frac{\bar{\sigma}_r}{\sigma_{t_r}}$$

变换后的速度场为：

$$\bar{u}_r(x) = \frac{\dot{s}_r}{s_r} x + s_r \dot{t}_r u_{t_r}\left(\frac{x}{s_r}\right)$$

```python
class ScheduleTransformedModel(ModelWrapper):
    def forward(self, x, t, **extras):
        r = t
        # 新调度器参数
        r_out = self.new_scheduler(t=r)
        # 通过 SNR 反推原始时间
        t = self.original_scheduler.snr_inverse(r_out.alpha_t / r_out.sigma_t)
        # 原始调度器参数
        t_out = self.original_scheduler(t=t)
        # 计算尺度因子和时间导数
        s_r = r_out.sigma_t / t_out.sigma_t
        dt_r = ...  # 时间映射的导数
        ds_r = ...  # 尺度因子的导数
        # 变换速度场
        u_t = self.model(x=x / s_r, t=t, **extras)
        u_r = ds_r * x / s_r + dt_r * s_r * u_t
        return u_r
```

### 5.3 损失函数 (loss)

#### MixturePathGeneralizedKL

专为离散 Flow Matching 设计的广义 KL 散度损失。假设模型以 x-prediction 方式训练（即模型输出 $p_{1|t}(\cdot|x_t)$）。

接口：

```python
loss_fn = MixturePathGeneralizedKL(path=my_discrete_path, reduction='mean')
loss = loss_fn(logits=model_output, x_1=target, x_t=path_sample, t=time)
```

参数说明：
- `logits`：模型输出的 logits，形状 `(batch, d, K)`，其中 K 为词表大小
- `x_1`：目标数据，形状 `(batch, d)`
- `x_t`：路径采样点，形状 `(batch, d)`
- `t`：时间，形状 `(batch,)`

对于连续 Flow Matching，通常直接使用 `torch.nn.MSELoss` 计算速度匹配损失。

### 5.4 求解器 (solver)

求解器负责在推理阶段从源分布生成目标分布的样本。

#### 类继承关系

```
Solver (抽象基类, nn.Module)
├── ODESolver (连续 ODE 求解器)
├── MixtureDiscreteEulerSolver (离散 Euler 求解器)
└── RiemannianODESolver (黎曼 ODE 求解器)
```

#### ODESolver

基于 `torchdiffeq` 的通用 ODE 求解器，支持多种数值方法。

```python
solver = ODESolver(velocity_model=my_model)

# 基本采样：从 X_0 ~ p_0 生成 X_1 ~ p_1
x_1 = solver.sample(
    x_init=x_0,                              # 初始噪声
    step_size=1/1000,                         # 步长
    method="euler",                           # 积分方法
    time_grid=torch.tensor([0.0, 1.0]),       # 时间区间
)
```

支持的积分方法：

| 方法 | 类型 | 说明 |
|------|------|------|
| `euler` | 固定步长 | 一阶 Euler 方法，最快但精度最低 |
| `midpoint` | 固定步长 | 二阶中点法 |
| `heun3` | 固定步长 | 三阶 Heun 方法 |
| `dopri5` | 自适应步长 | 五阶 Dormand-Prince，精度高但较慢 |

#### 似然计算

`ODESolver.compute_likelihood()` 通过反向积分 ODE 并计算雅可比行列式的迹来计算精确的 log 似然：

$$\log p_1(x_1) = \log p_0(x_0) + \int_1^0 \text{div}(u_t(x_t)) \, dt$$

```python
x_0, log_likelihood = solver.compute_likelihood(
    x_1=data_samples,
    log_p0=lambda x: -0.5 * x.pow(2).sum(-1),  # 标准高斯的 log 概率
    step_size=1/1000,
    time_grid=torch.tensor([1.0, 0.0]),          # 必须从 1 积到 0
    exact_divergence=False,                       # 使用 Hutchinson 估计器
)
```

散度计算有两种模式：
- `exact_divergence=True`：精确计算 $\text{div}(u_t) = \sum_i \frac{\partial u_t^i}{\partial x^i}$，计算量为 $O(D)$ 次反向传播
- `exact_divergence=False`：Hutchinson 估计器 $\text{div}(u_t) \approx z^T \nabla_x (u_t \cdot z)$，仅需 1 次反向传播

#### MixtureDiscreteEulerSolver

离散空间的 CTMC 模拟器，实现了带无散度项的 Euler 步进：

```python
solver = MixtureDiscreteEulerSolver(
    model=my_model,
    path=my_discrete_path,
    vocabulary_size=256,
    source_distribution_p=uniform_dist,  # 可选：用于无散度项
)

x_1 = solver.sample(
    x_init=x_0,
    step_size=1/1000,
    div_free=0.0,          # 无散度项系数，0 表示不使用
    time_grid=torch.tensor([0.0, 1.0]),
)
```

每步的核心逻辑：
1. 从模型采样 $X_1 \sim p_{1|t}(\cdot|X_t)$
2. 计算条件速度 $u_t(x|X_t, X_1)$
3. 计算跳跃强度 $\lambda = \sum_{x \neq X_t} u_t(x|X_t, X_1)$
4. 以概率 $1 - e^{-h\lambda}$ 发生跳跃，跳跃目标按 $u_t$ 归一化后采样

#### RiemannianODESolver

流形上的 ODE 求解器，支持 Euler、中点法和 RK4，每步都可选择性地将状态投影回流形、将速度投影到切平面：

```python
solver = RiemannianODESolver(
    manifold=Sphere(),
    velocity_model=my_model,
)

x_1 = solver.sample(
    x_init=x_0,
    step_size=0.01,
    method="rk4",     # euler / midpoint / rk4
    projx=True,       # 每步投影到流形
    proju=True,       # 速度投影到切平面
)
```

RK4 步进的流形版本（`_rk4_step`）：

```python
def _rk4_step(velocity_model, xt, t0, dt, manifold, projx=True, proju=True):
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    k1 = velocity_fn(xt, t0)
    k2 = velocity_fn(projx_fn(xt + dt * k1 / 3), t0 + dt / 3)
    k3 = velocity_fn(projx_fn(xt + dt * (k2 - k1 / 3)), t0 + dt * 2 / 3)
    k4 = velocity_fn(projx_fn(xt + dt * (k1 - k2 + k3)), t0 + dt)

    return projx_fn(xt + (k1 + 3 * (k2 + k3) + k4) * dt * 0.125)
```

### 5.5 工具模块 (utils)

#### ModelWrapper

所有速度场模型必须继承 `ModelWrapper`，统一接口为 `forward(x, t, **extras)`：

```python
class ModelWrapper(ABC, nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        return self.model(x=x, t=t, **extras)
```

自定义模型示例：

```python
class MyVelocityModel(ModelWrapper):
    def __init__(self, net):
        super().__init__(net)

    def forward(self, x, t, **extras):
        # 可在此添加时间编码、条件拼接等自定义逻辑
        t_embed = self.time_embedding(t)
        return self.model(torch.cat([x, t_embed], dim=-1))
```

#### 张量工具函数

```python
# expand_tensor_like: 将 1D 张量扩展到与目标张量相同的形状
# 用途：将 (batch_size,) 的时间/调度器参数扩展到 (batch_size, C, H, W) 等
alpha_t = expand_tensor_like(input_tensor=scheduler_output.alpha_t, expand_to=x_1)

# unsqueeze_to_match: 自动添加维度使源张量与目标张量维度匹配
t = unsqueeze_to_match(source=t, target=x_t)

# gradient: 计算梯度的封装，用于似然计算中的散度估计
grad = gradient(output, x, create_graph=True)
```

#### categorical 采样器

基于 `torch.multinomial` 的分类采样器，支持任意形状的概率张量：

```python
def categorical(probs: Tensor) -> Tensor:
    return torch.multinomial(
        probs.flatten(0, -2), 1, replacement=True
    ).view(*probs.shape[:-1])
```

---

## 6. 使用指南与代码示例

### 6.1 连续 Flow Matching 完整训练流程

```python
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

# 1. 定义模型
class SimpleVelocityNet(ModelWrapper):
    def __init__(self, dim):
        net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, dim),
        )
        super().__init__(net)

    def forward(self, x, t, **extras):
        t_expanded = t.unsqueeze(-1) if t.dim() == 1 else t
        inp = torch.cat([x, t_expanded], dim=-1)
        return self.model(inp)

# 2. 初始化路径和模型
dim = 2
path = AffineProbPath(scheduler=CondOTScheduler())
model = SimpleVelocityNet(dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse_loss = torch.nn.MSELoss()

# 3. 训练循环
for epoch in range(1000):
    # 源分布：标准高斯
    x_0 = torch.randn(256, dim)
    # 目标分布：你的数据
    x_1 = sample_from_data(batch_size=256)
    # 随机时间
    t = torch.rand(256)

    # 采样条件路径
    path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

    # 计算速度匹配损失
    predicted_velocity = model(path_sample.x_t, path_sample.t)
    loss = mse_loss(predicted_velocity, path_sample.dx_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. 推理：生成样本
solver = ODESolver(velocity_model=model)
x_0 = torch.randn(100, dim)
x_1 = solver.sample(
    x_init=x_0,
    step_size=0.01,
    method="midpoint",
    time_grid=torch.tensor([0.0, 1.0]),
)
```

### 6.2 离散 Flow Matching 训练流程

```python
import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver

# 1. 初始化
vocab_size = 256
scheduler = PolynomialConvexScheduler(n=1.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)
loss_fn = MixturePathGeneralizedKL(path=path)

# 2. 训练循环
for x_1 in dataloader:  # x_1: (batch, seq_len) 的整数 token
    x_0 = torch.randint(0, vocab_size, x_1.shape)  # 均匀随机源
    t = torch.rand(x_1.shape[0])

    path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

    # 模型输出 logits: (batch, seq_len, vocab_size)
    logits = model(path_sample.x_t, path_sample.t)
    loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=t)

    loss.backward()
    optimizer.step()

# 3. 推理
solver = MixtureDiscreteEulerSolver(
    model=model, path=path, vocabulary_size=vocab_size
)
x_0 = torch.randint(0, vocab_size, (batch_size, seq_len))
x_1 = solver.sample(x_init=x_0, step_size=1/1000)
```

### 6.3 黎曼 Flow Matching（球面）

```python
import torch
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils.manifolds import Sphere

# 1. 初始化
manifold = Sphere()
scheduler = CondOTScheduler()
path = GeodesicProbPath(scheduler=scheduler, manifold=manifold)

# 2. 训练
for x_1 in dataloader:  # x_1: 球面上的数据点
    x_0 = manifold.projx(torch.randn_like(x_1))  # 投影到球面
    t = torch.rand(x_1.shape[0])

    path_sample = path.sample(x_0=x_0, x_1=x_1, t=t)

    predicted_velocity = model(path_sample.x_t, path_sample.t)
    loss = mse_loss(predicted_velocity, path_sample.dx_t)
    loss.backward()

# 3. 推理
solver = RiemannianODESolver(manifold=manifold, velocity_model=model)
x_0 = manifold.projx(torch.randn(100, 3))
x_1 = solver.sample(x_init=x_0, step_size=0.01, method="rk4")
```

### 6.4 训练后更换调度器

```python
from flow_matching.path.scheduler import CondOTScheduler, CosineScheduler, ScheduleTransformedModel

# 模型原本用 CondOT 调度器训练
original_scheduler = CondOTScheduler()
new_scheduler = CosineScheduler()

# 无需重新训练，直接包装模型
transformed_model = ScheduleTransformedModel(
    velocity_model=trained_model,
    original_scheduler=original_scheduler,
    new_scheduler=new_scheduler,
)

# 用新调度器进行推理
solver = ODESolver(velocity_model=transformed_model)
x_1 = solver.sample(x_init=x_0, step_size=1/1000)
```

### 6.5 计算精确 log 似然

```python
import torch
from flow_matching.solver import ODESolver

solver = ODESolver(velocity_model=trained_model)

# 定义源分布的 log 概率（标准高斯）
log_p0 = lambda x: -0.5 * (x.pow(2).sum(-1) + x.shape[-1] * torch.log(torch.tensor(2 * torch.pi)))

# 计算数据点的 log 似然
x_0_recovered, log_likelihood = solver.compute_likelihood(
    x_1=data_samples,
    log_p0=log_p0,
    step_size=1/1000,
    time_grid=torch.tensor([1.0, 0.0]),
    exact_divergence=False,  # Hutchinson 估计器更高效
)
# log_likelihood: 每个样本的 log p_1(x_1)
```

---

## 7. 依赖关系

### 核心库依赖

```
flow_matching
├── numpy          # 数值计算基础
├── torch (≥2.1)   # 张量计算、自动微分、神经网络
└── torchdiffeq    # ODE 数值积分（odeint 接口）
```

### 开发依赖

```
dev:
├── pre-commit     # Git 提交前代码检查
├── black          # 代码格式化
├── usort / ufmt   # import 排序与格式化
├── flake8         # 代码风格检查
└── pydoclint      # 文档字符串检查
```

### 示例额外依赖

- 图像示例：`torchvision`, `submitit`（分布式训练）
- 文本示例：`hydra-core`, `wandb`（实验管理）
- Notebook：`matplotlib`, `jupyter`, `scikit-learn`, `tqdm`

---

> 本文档基于 `flow_matching v1.0.10` 源码生成。
> 论文引用：Lipman et al., "Flow Matching Guide and Code", arXiv:2412.06264, 2024.
