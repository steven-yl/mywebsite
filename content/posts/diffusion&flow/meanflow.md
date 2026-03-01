---
title: "MeanFlow"
subtitle: ""
date: 2026-02-28T10:26:59+08:00
# lastmod: 2026-02-28T10:26:59+08:00
draft: false
authors: [Steven]
description: ""

tags: [diffusion/flow]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 4
series_weight: 4

hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: "/mywebsite/posts/images/meanflow.webp"

---
MeanFlow 算法完整技术文档（原理+数学推导+训练推理+工程代码）
本文档为正式技术规格文档，完整复现 MeanFlow 核心理论、数学推导、训练流程、采样逻辑与可运行 Demo。

## 1 文档基本信息

- 算法名称：MeanFlow
- 提出团队：Facebook AI Research（何恺明团队）
- 发表时间：2025 年
- 核心定位：一步生成（1‑NFE） 生成式建模框架
- 基础依托：流匹配（Flow Matching）+ 常微分方程（ODE）
- 核心创新：从学习瞬时速度场 $v$ 改为学习区间平均速度场 $u$，实现单步高质量生成

## 2 背景与动机

### 2.1 传统生成模型的痛点

- 扩散模型 / 流匹配：学习瞬时速度场，必须多步欧拉积分才能生成高质量图像，步数越少质量越差。
- Consistency Models：需要蒸馏、课程学习、多阶段训练，流程复杂且理论不闭合。
- 一步模型：以往 1 步模型生成质量远低于多步模型。

### 2.2 MeanFlow 解决思路

直接建模时间区间 $[r, t]$ 上的平均速度，让模型一次性学习“从噪声到数据”的完整位移，而非逐点瞬时变化。

## 3 核心数学定义（完整推导）

### 3.1 基本符号定义

- $z_t \in \mathbb{R}^d$：时间 $t$ 处的隐变量
- $x_0 \sim p_0$：初始高斯噪声
- $x_1 \sim p_{\text{data}}$：真实数据分布
- $v(z_t, t)$：瞬时速度场（传统 Flow Matching 学习目标）
- $u(z_t, r, t)$：平均速度场（MeanFlow 学习目标）
- ODE 动力学：$\frac{dz_t}{dt} = v(z_t, t)$

### 3.2 平均速度场定义

平均速度是瞬时速度在区间 $[r, t]$ 上的积分平均：

$$
u(z_t, r, t) = \frac{1}{t - r} \int_{r}^{t} v(z_\tau, \tau) d\tau
$$

物理意义：

- $v$ = 某一时刻的瞬间速度
- $u$ = 从 $r$ 走到 $t$ 的整体平均移动速度

### 3.3 MeanFlow 核心恒等式（关键理论）

对平均速度定义式关于 $t$ 求导，利用链式法则与微积分基本定理，可推导出平均速度与瞬时速度的严格恒等式。

$$
v(z_t, t) - u(z_t, r, t) = (t - r) \left( \frac{\partial u}{\partial t} + \nabla_{z_t} u \cdot v(z_t, t) \right)
$$

推导如下。

**步骤 1**：将定义式两边同乘 $(t - r)$，得
$$
(t - r)\, u(z_t, r, t) = \int_{r}^{t} v(z_\tau, \tau)\, d\tau.
$$

**步骤 2**：对等式两边关于 $t$ 求导。

- **右边**：由微积分基本定理，积分对上限 $t$ 的导数为被积函数在 $\tau = t$ 处的值，即
$$
\frac{d}{dt} \int_{r}^{t} v(z_\tau, \tau)\, d\tau = v(z_t, t).
$$

- **左边**：$(t - r)\, u(z_t, r, t)$ 是 $t$ 的函数，且 $z_t$ 也随 $t$ 变化（满足 $\frac{d z_t}{d t} = v(z_t, t)$）。由乘积法则，
$$
\frac{d}{dt}\bigl[ (t - r)\, u(z_t, r, t) \bigr] = u(z_t, r, t) + (t - r)\, \frac{d}{dt} u(z_t, r, t).
$$
对 $u(z_t, r, t)$ 关于 $t$ 求全导数时，$u$ 既直接依赖 $t$，又通过 $z_t$ 依赖 $t$，故
$$
\frac{d}{dt} u(z_t, r, t) = \frac{\partial u}{\partial t} + \nabla_{z_t} u \cdot \frac{d z_t}{d t} = \frac{\partial u}{\partial t} + \nabla_{z_t} u \cdot v(z_t, t).
$$
因此左边等于
$$
u + (t - r) \left( \frac{\partial u}{\partial t} + \nabla_{z_t} u \cdot v(z_t, t) \right).
$$

**步骤 3**：左右两边相等，故
$$
u(z_t, r, t) + (t - r) \left( \frac{\partial u}{\partial t} + \nabla_{z_t} u \cdot v(z_t, t) \right) = v(z_t, t).
$$
移项即得**核心恒等式**：
$$
v(z_t, t) - u(z_t, r, t) = (t - r) \left( \frac{\partial u}{\partial t} + \nabla_{z_t} u \cdot v(z_t, t) \right)
$$

该式无近似、无假设、完全严格，是 MeanFlow 训练的理论基础。

### 3.4 训练目标推导

从恒等式中解出模型需要拟合的目标平均速度：

$$
u_{\text{tgt}} = v_t - (t - r) \left( \frac{\partial u_\theta}{\partial t} + \nabla_{z_t}u_\theta \cdot v_t \right)
$$

训练损失为模型输出与目标的 L2 距离：

$$
\mathcal{L}(\theta) = \mathbb{E}_{r<t, z_t, v_t} \left\| u_\theta(z_t, r, t) - \text{sg}(u_{\text{tgt}}) \right\|_2^2
$$

- $\text{sg}(\cdot)$：停止梯度，保证目标固定
- $v_t = x_1 - x_0$：线性路径下的解析瞬时速度

**$\text{sg}(\cdot)$ 详细说明与实现**

- **定义**：前向时 $\text{sg}(x) = x$（数值不变）；反向时 $\dfrac{\partial \,\text{sg}(x)}{\partial x} = 0$，即该节点不向输入传梯度，在计算图中被当作常数。
- **为何必须用**：目标 $u_{\text{tgt}} = v_t - (t - r)\big(\partial u_\theta/\partial t + \nabla_{z_t}u_\theta \cdot v_t\big)$ 依赖 $u_\theta$ 及其导数，即依赖 $\theta$。若不做 sg，$\nabla_\theta \mathcal{L}$ 会包含“通过 $u_{\text{tgt}}$ 再对 $\theta$ 求导”的项，目标会随参数更新而变（移动目标）；用 sg 后只对损失里的 $u_\theta(z_t,r,t)$ 关于 $\theta$ 求导，目标在本次更新中固定，等价于监督学习：拟合给定向量 $u_{\text{tgt}}$。
- **实现要点**：先按公式算出 $u_{\text{tgt}}$（需要自动微分得到 $\partial u_\theta/\partial t$ 和 $\nabla_{z_t} u_\theta$），再对 $u_{\text{tgt}}$ 做 stop-gradient，最后算 MSE。下面给出 PyTorch 写法。

**PyTorch**：用 `.detach()` 把目标从计算图剥离，反向时梯度不会穿过目标。$(\nabla_{z_t} u_\theta)\cdot v_t$ 为 Jacobian–向量积，用 `torch.autograd.functional.jvp` 一次算出。

```python
from torch.autograd.functional import jvp

# z_t, t 需 requires_grad=True 以便算 u_tgt 中的导数
u = model(z_t, r, t)   # u_theta(z_t, r, t)

# 时间导数：\partial u / \partial t（保持 z_t 不变）
du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=False, allow_unused=True)[0]
if du_dt is None:
    du_dt = torch.zeros_like(u)
# 空间 Jacobian–向量积：(\nabla_{z_t} u) · v_t
_, jvp_z = jvp(lambda z: model(z, r, t), z_t, v_t)

u_tgt = v_t - (t - r) * (du_dt + jvp_z)
u_tgt = u_tgt.detach()   # stop-gradient：loss 反向不传到 u_tgt
loss = F.mse_loss(u, u_tgt)
loss.backward()
```

### 3.5 一步采样公式

推理阶段仅需一次前向传播，直接从噪声映射到数据：

$$
z_1 = z_0 + u_\theta(z_0, 0, 1)
$$

- $0 \to 1$ 代表完整生成过程
- 无迭代、无积分、1-NFE 完成
 
 
 
## 4 算法完整流程

### 4.1 训练流程（逐步骤）
 
1. 采样噪声 $x_0 \sim \mathcal{N}(0, I)$  \
2. 采样数据 $x_1 \sim p_{\text{data}}$  \
3. 随机采样时间对 $0 \le r < t \le 1$  \
4. 构造线性插值路径：$z_t = (1-t)x_0 + t x_1$  \
5. 计算解析瞬时速度：$v_t = x_1 - x_0$  \
6. 前向计算模型输出 $u_\theta(z_t, r, t)$  \
7. 利用自动微分计算：
- $\partial u_\theta / \partial t$
- $\nabla_{z_t} u_\theta$

8. 构造目标速度 $u_{\text{tgt}}$  \
9. 最小化 MSE 损失更新参数
 
### 4.2 推理（采样）流程
 
1. 采样高斯噪声 $z_0 \sim \mathcal{N}(0, I)$  \
2. 前向计算平均速度 $u = u_\theta(z_0, 0, 1)$  \
3. 一步生成：$z_1 = z_0 + u$  \
4. 输出 $z_1$ 为最终样本
 
 
 
## 5 关键技术特性
 
1. 理论完全闭合
从定义直接推导，无启发式、无蒸馏、无课程学习。
2. 一步生成（1‑NFE）
速度与 GAN 相当，质量逼近多步扩散模型。
3. 训练稳定
损失函数平滑，最优解唯一存在。
4. 天然支持条件生成
可直接嵌入 CFG（无分类器引导），无需修改结构。
5. 兼容所有 DiT / U-Net 架构
只需将输出从瞬时速度 $v$ 改为平均速度 $u$。
 
 
 
## 6 MeanFlow 完整 PyTorch 实现（工程级 Demo）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


# ==============================================================================
# 1. MeanFlow 核心模型：支持 z_t + r + t 输入
# ==============================================================================
class MeanFlowModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256):
        super().__init__()
        # 输入：z + 时间r + 时间t → 输出：平均速度场 u
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, z, r, t):
        """
        z:  [B, D]
        r:  [B]
        t:  [B]
        return: u [B, D]
        """
        t = t.unsqueeze(1)
        r = r.unsqueeze(1)
        x = torch.cat([z, r, t], dim=1)
        return self.net(x)


# ==============================================================================
# 2. 核心函数：计算 u_tgt（目标平均速度）
# ==============================================================================
def compute_meanflow_target(model, z_t, r, t, v_t):
    """
    实现论文核心公式：
    u_tgt = v_t - (t - r) * (du/dt + ∇z u · v_t)
    """
    B, D = z_t.shape

    # 开启微分
    z_t = z_t.detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)

    # 前向
    u = model(z_t, r, t)

    # 1. 计算 du/dt
    du_dt = grad(u.sum(), t, create_graph=True)[0]  # [B]
    du_dt = du_dt.view(B, 1).expand(B, D)

    # 2. 计算 ∇_z u · v_t
    du_dz = grad(u.sum(), z_t, create_graph=True)[0]  # [B, D]
    du_dz_v = du_dz * v_t

    # 3. 目标平均速度
    delta_t = (t - r).view(B, 1)
    u_tgt = v_t - delta_t * (du_dt + du_dz_v)

    # 停止梯度，保证目标不变
    return u_tgt.detach()


# ==============================================================================
# 3. 单步训练逻辑
# ==============================================================================
def train_one_step(model, optimizer, x0, x1):
    B = x0.shape[0]
    device = x0.device

    # 采样 r < t
    t = torch.rand(B, device=device)
    r = torch.rand(B, device=device) * t

    # 线性路径
    z_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    v_t = x1 - x0  # 瞬时速度

    # 前向
    u_pred = model(z_t, r, t)

    # 计算目标
    u_tgt = compute_meanflow_target(model, z_t, r, t, v_t)

    # 损失
    loss = F.mse_loss(u_pred, u_tgt)

    # 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


# ==============================================================================
# 4. 一步采样（核心！1-NFE）
# ==============================================================================
@torch.no_grad()
def meanflow_sample(model, n_samples, dim, device):
    z0 = torch.randn(n_samples, dim, device=device)
    r = torch.zeros(n_samples, device=device)
    t = torch.ones(n_samples, device=device)
    u = model(z0, r, t)
    z1 = z0 + u
    return z1


# ==============================================================================
# 5. Toy 实验：双高斯分布生成（可直接运行）
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型
    model = MeanFlowModel(input_dim=2, hidden_dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 真实数据分布：两个高斯聚类
    def data_sampler(batch):
        x = torch.randn(batch, 2, device=device)
        c = torch.randint(0, 2, (batch,), device=device)
        x[c == 0] += torch.tensor([3.0, 3.0], device=device)
        x[c == 1] -= torch.tensor([3.0, 3.0], device=device)
        return x

    # 训练
    print("Start training...")
    for step in range(10000):
        x0 = torch.randn(256, 2, device=device)
        x1 = data_sampler(256)
        loss = train_one_step(model, opt, x0, x1)
        if step % 500 == 0:
            print(f"Step {step:05d} | Loss {loss:.4f}")

    # 一步采样（仅1次前向）
    samples = meanflow_sample(model, n_samples=1000, dim=2, device=device)
    print("\nSampled points (first 5):")
    print(samples[:5])
```

## 7 与其他模型的对比总结

| 模型 | 学习目标 | 采样步数 | 训练复杂度 | 理论 |
|------|----------|----------|------------|------|
| Flow Matching | 瞬时速度 $v$ | ≥20 步 | 低 | 干净 |
| Consistency Model | 自一致性 | 1 步 | 极高（蒸馏） | 启发式 |
| MeanFlow | 平均速度 $u$ | 1 步 | 低 | 完全严格 |

## 8 适用场景

- 图像/视频/音频生成
- 蛋白质结构生成
- 高分辨率实时 AIGC
- 端侧部署（低算力、低延迟）
- 需要单步快速生成的工业场景
