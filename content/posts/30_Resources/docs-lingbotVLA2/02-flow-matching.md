---
title: "2. Flow Matching 动作生成"
subtitle: ""
date: 2026-07-13T14:00:00+08:00
draft: false
authors: [Steven]
description: "本章详解 LingBot-VLA 2.0 的动作生成算法：Conditional Flow Matching（条件流匹配），包括数学原理、训练目标、推理积分，以及与 Diffusion 的对比。"
summary: "本章详解 LingBot-VLA 2.0 的动作生成算法：Conditional Flow Matching（条件流匹配），包括数学原理、训练目标、推理积分，…"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA2, robots]
series: [lingbotVLA2-docs]
weight: 2
series_weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 2. Flow Matching 动作生成

本章详解 LingBot-VLA 2.0 的动作生成算法：**Conditional Flow Matching（条件流匹配）**，包括数学原理、训练目标、推理积分，以及与 Diffusion 的对比。

---

## 2.1 问题定义

给定观测 \(o_t\)（图像、状态、语言），学习条件分布：

\[
p(a_{t:t+H} \mid o_t)
\]

其中 \(a_{t:t+H}\) 为长度 \(H\) 的动作 chunk（默认 \(H=50\)），每步动作 \(a \in \mathbb{R}^{55}\)。

**模仿学习（Behavior Cloning）** 直接回归 \( \hat{a} = f_\theta(o) \)，但对多模态分布（同一观测对应多种合理动作）建模困难。**生成式方法**（Diffusion、Flow Matching）通过随机性覆盖多模态。

---

## 2.2 Flow Matching 核心思想

### 2.2.1 连续归一化流（CNF）

Flow Matching 学习一个时变向量场 \(v_\theta(x, t)\)，使得从简单先验 \(p_1\)（如标准高斯）到数据分布 \(p_0\) 的 ODE：

\[
\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]
\]

将 \(t=1\) 处的噪声 \(x_1 \sim \mathcal{N}(0, I)\) 沿场积分到 \(t=0\) 即得动作样本。

### 2.2.2 条件线性插值路径

本仓库采用 **Optimal Transport 直线路径**（与 π₀、Stable Diffusion 3 的 FM 变体一致）：

**训练时采样：**

\[
\epsilon \sim \mathcal{N}(0, I), \quad t \sim \text{Beta}(1.5, 1.0)
\]

\[
t' = 0.001 + 0.999 \cdot t \quad \text{（避免边界数值问题）}
\]

\[
x_t = t' \cdot \epsilon + (1 - t') \cdot a
\]

\[
u_t = \epsilon - a \quad \text{（目标速度场，常数于该直线路径）}
\]

**直觉**：\(t'=0\) 时 \(x_t = a\)（干净动作），\(t'=1\) 时 \(x_t = \epsilon\)（纯噪声）。速度场指向「从数据走向噪声」的方向。

### 2.2.3 训练损失

模型 \(v_\theta(x_t, t, o)\) 预测速度，损失为：

**MSE（`loss_type: fm`）：**

\[
\mathcal{L}_{\text{FM}} = \mathbb{E}_{a,\epsilon,t} \left[ \| v_\theta(x_t, t, o) - u_t \|_2^2 \right]
\]

**L1（`loss_type: L1_fm`，RoboTwin 常用）：**

\[
\mathcal{L}_{\text{L1-FM}} = \mathbb{E} \left[ \| v_\theta - u_t \|_1 \right]
\]

对无效关节维度，通过 `joint_mask` 屏蔽：

```python
mask_expanded = joint_mask.unsqueeze(1).expand(-1, losses.size(1), -1)
losses = losses * mask_expanded
loss = losses.sum() / mask_expanded.sum()
```

---

## 2.3 代码实现对照

### 2.3.1 训练前向

```python
# modeling_lingbot_vla.py - FlowMatching.forward（v1/v2 共用逻辑）

# 1. 采样噪声与时间
if noise is None:
    noise = torch.randn_like(actions)
if time is None:
    time = self.sample_time(bsize, device)  # Beta(1.5, 1) * 0.999 + 0.001

# 2. 构造插值状态与目标速度
time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions

# 3. 前缀+后缀联合前向
v_t = model_forward(images, lang, state, x_t, time)  # action_out_proj 输出

# 4. 损失
losses = F.mse_loss(u_t, v_t, reduction="none")  # 或 l1_loss
```

### 2.3.2 时间采样

```python
def sample_time(self, bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)  # Beta 分布
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)
```

**为何 Beta(1.5, 1)**：偏向较大 \(t\)（更多噪声），使模型更多练习「从强噪声恢复」，提升推理鲁棒性。

### 2.3.3 时间嵌入

```python
def create_sinusoidal_pos_embedding(timestep, dimension, min_period=4e-3, max_period=4.0, device="cpu"):
    # 对数间隔频率的正弦/余弦编码，与 Transformer 位置编码类似
    # timestep ∈ [0, 1] 映射到不同频率的 sin/cos
```

`min_period=4e-3, max_period=4.0` 控制对 \(t\) 的敏感度范围。

---

## 2.4 推理：Euler ODE 积分

### 2.4.1 算法

从 \(t=1\)（纯噪声）出发，固定步长 Euler 积分至 \(t=0\)：

\[
\Delta t = -\frac{1}{N_{\text{steps}}}, \quad N_{\text{steps}} = 10 \text{（默认）}
\]

\[
x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, o)
\]

循环条件：`while time >= -dt / 2`（共执行 10 次）。

### 2.4.2 KV Cache 优化

```python
# sample_actions 流程
# Step 1: 仅编码 prefix，填充 KV cache
_, past_key_values, _ = qwenvl_with_expert.forward(
    inputs_embeds=[prefix_embs, None],
    fill_kv_cache=True,
)

# Step 2: 每步去噪只 forward suffix
for step in range(num_steps):
    v_t = predict_velocity(state, past_key_values, x_t, time)
    x_t += dt * v_t
    time += dt
```

前缀（图像+语言+query）**只计算一次**，10 步去噪仅重复 suffix 前向 → 推理延迟约 **130ms**（RTX 4090D）。

### 2.4.3 代码片段

```python
# modeling_lingbot_vla_v2.py - sample_actions
dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
x_t = noise  # 标准高斯
time = torch.tensor(1.0, dtype=dtype, device=device)

while time >= -dt / 2:
    v_t = predict_velocity_fn(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
    x_t += dt * v_t
    time += dt
return x_t  # 去噪后的动作 chunk
```

---

## 2.5 与 Diffusion (DDPM) 对比

| 方面 | DDPM | Flow Matching（本仓库） |
|------|------|-------------------------|
| 前向过程 | 马尔可夫噪声调度 \(q(x_t\|x_0)\) | 确定性线性插值 |
| 训练目标 | 预测噪声 \(\epsilon\) 或 \(x_0\) | 预测速度 \(v = \epsilon - x\) |
| 时间变量 | 离散 \(t \in \{1,\ldots,T\}\) | 连续 \(t \in [0,1]\) |
| 推理 | DDIM/DDPM 采样，步数敏感 | Euler ODE，默认 10 步 |
| 调度器 | 需要 `FlowMatchScheduler` 等 | 均匀步长即可 |
| 代码路径 | `schedulers/flow_match.py`（部分遗留） | 主路径在 `modeling_*.py` 内联 |

> **注意**：`lingbotvla/schedulers/flow_match.py` 中的 `FlowMatchScheduler` 来自图像生成生态（Stable Diffusion 3 风格），**VLA 主路径未使用**该调度器，而是直接 Euler 积分。

---

## 2.6 数学推导补充

### 2.6.1 直线路径下的速度场

设 \(x_t = (1-t)a + t\epsilon\)，对 \(t\) 求导：

\[
\frac{dx}{dt} = \epsilon - a = u_t
\]

该速度场与 \(x_t\) 无关（**常数场**），但神经网络仍需条件于 \(o\) 以生成正确的 \(v_\theta \approx u_t\)。

### 2.6.2 与 Score Matching 的关系

在特定参数化下，Flow Matching 与 denoising score matching 等价。速度场与 score 的关系：

\[
v(x, t) = \sigma_t' \cdot s(x, t) + \frac{\sigma_t'}{\sigma_t}(x - \mu_t)
\]

本仓库采用 OT 直线路径，实现更简洁。

---

## 2.7 超参数影响

| 参数 | 默认 | 影响 |
|------|------|------|
| `num_steps` | 10 | 推理步数 ↑ → 质量可能 ↑，延迟线性 ↑ |
| `chunk_size` / `n_action_steps` | 50 | 预测视野长度 |
| `loss_type` | `fm` / `L1_fm` | L1 对异常值更鲁棒，仿真常用 |
| `action_fp32` / `enable_fp32` | true | 动作分支 FP32，提升数值稳定 |
| `adanorm_time` | true | 时间条件注入 Norm 层 |

---

## 2.8 可运行示例：Flow Matching 玩具演示

以下脚本**不依赖完整 VLA 权重**，演示核心插值与积分逻辑：

```python
"""Flow Matching 一维玩具示例"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 简单 MLP 预测速度场
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x_t, t):
        return self.net(torch.cat([x_t, t], dim=-1))

def sample_beta(alpha, beta, n):
    return torch.distributions.Beta(alpha, beta).sample((n, 1))

# 训练数据：单峰高斯动作 a ~ N(2, 0.5)
model = VelocityNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(2000):
    a = torch.randn(64, 1) * 0.5 + 2.0          # 目标动作
    eps = torch.randn(64, 1)                      # 噪声
    t = sample_beta(1.5, 1.0, 64) * 0.999 + 0.001
    x_t = t * eps + (1 - t) * a
    u_t = eps - a
    v_t = model(x_t, t)
    loss = F.mse_loss(v_t, u_t)
    opt.zero_grad(); loss.backward(); opt.step()

# 推理：从噪声积分
with torch.no_grad():
    x = torch.randn(500, 1)
    time = torch.ones(500, 1)
    dt = -0.1
    for _ in range(10):
        x = x + dt * model(x, time)
        time = time + dt
    print(f"Generated mean: {x.mean():.2f} (target ~2.0)")

# plt.hist(x.numpy(), bins=30); plt.title("Flow Matching samples"); plt.show()
```

---

## 2.9 相关论文与资源

| 资源 | 链接 |
|------|------|
| Flow Matching for Generative Modeling | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) |
| π₀ (Physical Intelligence VLA) | [π₀ 博客](https://www.physicalintelligence.company/blog/pi0) |
| LingBot-VLA 2.0 技术报告 | [arXiv:2607.06403](https://arxiv.org/pdf/2607.06403) |
| OpenVLA | [GitHub](https://github.com/openvla/openvla) |

---

## 2.10 章节边界

| 本文档 | 其他章节 |
|--------|----------|
| 动作生成数学与算法 | 模型如何实现条件 \(v_\theta\) → [01-model-architecture.md](./01-model-architecture.md) |
| 训练循环如何调用 forward | [05-training-system.md](./05-training-system.md) |
| 部署时 `sample_actions` | [06-inference-deployment.md](./06-inference-deployment.md) |
