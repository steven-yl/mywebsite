---
title: "2. 流匹配与核心算法原理"
subtitle: ""
date: 2026-07-13T12:00:00+08:00
draft: false
authors: [Steven]
description: "本章详解 LingBot-VLA 动作生成的数学基础、注意力机制、位置编码，以及代码中的完整实现对应关系。"
summary: "本章详解 LingBot-VLA 动作生成的数学基础、注意力机制、位置编码，以及代码中的完整实现对应关系。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA, robots]
series: [lingbotVLA-docs]
weight: 2
series_weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 2. 流匹配与核心算法原理

本章详解 LingBot-VLA 动作生成的数学基础、注意力机制、位置编码，以及代码中的完整实现对应关系。

---

## 2.1 Flow Matching 基础

### 2.1.1 问题定义

给定机器人观测 $o$（图像、语言、本体状态），学习条件分布 $p(a \mid o)$，其中 $a \in \mathbb{R}^{T \times D}$ 为动作块（$T=$ `chunk_size`，$D=$ `max_action_dim`）。

Flow Matching（FM）将生成建模转化为学习**速度场** $v_\theta(x, t \mid o)$，沿概率路径从噪声 $x_0 \sim \mathcal{N}(0, I)$ 流向数据 $x_1 = a$。

**核心论文：**

- [Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)](https://arxiv.org/abs/2210.02747)
- [π₀: A Vision-Language-Action Flow Model (Black et al., 2024)](https://arxiv.org/abs/2410.24164)

### 2.1.2 Conditional Flow Matching (CFM)

边际路径难以计算，CFM 使用**逐样本条件路径**。对固定数据点 $x_1$ 与噪声 $\epsilon$：

**线性插值路径（LingBot-VLA / PI0 采用）：**

$$
x_t = t \cdot \epsilon + (1 - t) \cdot x_1, \quad t \in [0, 1]
$$

对 $t$ 求导得条件速度场：

$$
u_t = \frac{\mathrm{d} x_t}{\mathrm{d} t} = \epsilon - x_1
$$

注意 $u_t$ **与 $t$ 无关**（线性 OT 路径的性质）。

**训练目标：**

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \epsilon, x_1, o}\left[\| v_\theta(x_t, t, o) - u_t \|^2\right]
$$

代码实现（`FlowMatching.forward`）：

```python
# modeling_lingbot_vla.py
time_expanded = time[:, None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions
v_t = self.action_out_proj(suffix_out)
losses = F.mse_loss(u_t, v_t, reduction="none")  # 或 L1_fm
```

### 2.1.3 时间采样

$t$ 从 **Beta(1.5, 1.0)** 采样后线性映射到 $[0.001, 0.999]$：

```python
def sample_time(self, bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time
```

`sample_beta` 实现（Gamma 比 trick）：

$$
\text{Beta}(\alpha,\beta) = \frac{U_1^{1/\alpha}}{U_1^{1/\alpha} + U_2^{1/\beta}}, \quad U_i \sim \mathrm{Uniform}(0,1)
$$

**为何偏 Beta 而非 Uniform？** 与 PI0 一致，使 $t$ 分布略偏向 0/1 两端，覆盖路径首尾区域更多样本。

### 2.1.4 推理：Euler ODE 求解

从 $t=1$（纯噪声）积分到 $t=0$（动作）：

$$
x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, o), \quad \Delta t = -\frac{1}{N_{\text{steps}}}
$$

```python
# sample_actions()
dt = torch.tensor(-1.0 / num_steps, ...)
x_t = noise
time = torch.tensor(1.0, ...)
while time >= -dt / 2:
    v_t = self.predict_velocity(state, prefix_pad_masks, past_key_values, x_t, time)
    x_t += dt * v_t
    time += dt
```

**KV Cache 优化：** prefix（图像+语言）只算一次 `fill_kv_cache=True`；每步仅 forward suffix，复用 `past_key_values`。

### 2.1.5 与 `FlowMatchScheduler` 的关系

`lingbotvla/schedulers/flow_match.py` 实现了更通用的 **shifted sigma 调度**：

$$
\sigma' = \frac{\text{shift} \cdot \sigma}{1 + (\text{shift}-1)\sigma}
$$

| 方法 | 路径 | 训练目标 | 使用位置 |
|------|------|----------|----------|
| 内联线性 FM | $x_t = (1-t)a + t\epsilon$ | $\epsilon - a$ | `FlowMatching` ✅ |
| FlowMatchScheduler | sigma 调度 + 高斯 timestep 权重 | `noise - sample` | 仓库内**未被 PI0 导入** |

若需 SD3 风格 shift 调度，可替换 `sample_time` / `step` 为 scheduler 接口。

---

## 2.2 共享跨流注意力（Mixture-of-Transformers）

### 2.2.1 动机

PI0 / LingBot 采用 **双塔共享注意力**：VLM 流（2048d）与 Action Expert 流（768d）在每一层分别投影 Q/K/V，然后在序列维 concat，做**一次**全局注意力，再 split 回各流。

**优势：** VLM 语义 token 与 action token 直接交互，无需 cross-attention 模块。  
**代价：** 序列长度 = prefix + suffix，注意力 $O(L^2)$；需 Flex/Flash 优化。

### 2.2.2 单层算法步骤

```
输入: inputs_embeds = [prefix_emb, suffix_emb]
For layer_idx in 0..35:
  1. VLM layer: compute_kqv(prefix) → Q_v, K_v, V_v
  2. Expert layer: compute_kqv(suffix, ada_cond?) → Q_e, K_e, V_e
  3. Q = cat(Q_v, Q_e, dim=seq); 同理 K, V
  4. Q, K = apply_rope(Q, K, position_ids)
  5. KV cache update (if use_cache)
  6. att_out = Attention(Q, K, V, mask)
  7. Split att_out → output_atten per stream + MLP
Final: per-stream RMSNorm (Expert 可选 AdaRMSNorm)
```

### 2.2.3 注意力计算

**Scaled Dot-Product：**

$$
\text{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V
$$

`our_eager_attention_forward`（eager 模式）与 `flex_attention_forward`（flex 模式）均实现 GQA：K/V head 数少于 Q 时 repeat。

**Flex Attention**（PyTorch ≥ 2.5）：block size 128，`torch.compile(max-autotune-no-cudagraphs)`。

---

## 2.3 二维注意力掩码 `make_att_2d_masks`

### 2.3.1 语义

`att_masks[b, i] ∈ {0, 1}` 定义**注意力块**：

- `0`：与前一 token 同块（共享双向 mask）
- `1`：新块开始

```python
cumsum = torch.cumsum(att_masks, dim=1)
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
att_2d_masks = att_2d_masks & pad_2d_masks
```

**规则：** token $i$ 可 attend 到 token $j$ 当且仅当 $\text{cumsum}(j) \le \text{cumsum}(i)$ 且两者均非 pad。

### 2.3.2 LingBot-VLA 典型配置

| 区域 | att_masks | 效果 |
|------|-----------|------|
| Prefix (img+lang) | 全 0 | 双向 |
| Suffix state | 0 | 与 state 块内双向 |
| Suffix actions | 0（首 action 前 :2 为 1） | state 不可看 action；action 间双向 |

**Prefix-Suffix 隔离：** 推理时 `full_att_2d_masks = [prefix_pad_2d | suffix_att_2d]`，suffix 可看 prefix，反向不可。

### 2.3.3 Depth Query 专用掩码 `make_att_2d_masks_with_query`

为每相机 depth query token 设置：

- query 仅 attend 对应相机 image tokens
- query 自注意力
- 语言/other 不看 query（蒸馏专用表示）

---

## 2.4 位置编码

### 2.4.1 RoPE（Rotary Position Embedding）

对 Q/K 应用：

$$
\begin{aligned}
\text{RoPE}(x, m) &= x \cos(m\theta) + \text{rotate\_half}(x) \sin(m\theta) \\
\theta_i &= \text{base}^{-2i/d}
\end{aligned}
$$

`apply_rope(x, positions, max_wavelength=10000)` — 用于 Expert 共享注意力。

Qwen2.5-VL 语言部分使用 **M-RoPE**（3D：temporal, height, width），见 `qwenvl_in_vla.py` 的 `get_rope_index`。

### 2.4.2 正弦时间嵌入 `create_sinusoidal_pos_embedding`

标量 diffusion time $t$ 映射到 `proj_width` 维：

$$
\begin{aligned}
\text{period}_i &= \text{min\_period} \cdot \left(\frac{\text{max\_period}}{\text{min\_period}}\right)^{i/(d/2)} \\
\text{emb}(t) &= [\sin(2\pi t / \text{period}_i), \cos(2\pi t / \text{period}_i)]_{i=1}^{d/2}
\end{aligned}
$$

默认 `min_period=4e-3`, `max_period=4.0`，使 $t \in [0,1]$ 各频段可区分。

**融合方式：**

- **默认：** `concat(action_emb, time_emb)` → 2-layer MLP
- **separate_time_proj：** time 单独 MLP，action 不加 time concat；AdaRMSNorm 用 time 条件

---

## 2.5 AdaRMSNorm 时间条件

$$
\text{AdaRMSNorm}(x, c) = \frac{x}{\|x\|_{\text{rms}}} \cdot (1 + \gamma(c)) + \beta(c)
$$

其中 $\gamma, \beta$ 由 time embedding 经线性层产生。用于 Expert 各层，使去噪步 $t$ 调制归一化尺度。

---

## 2.6 `utils.py` 函数完整说明

| 函数 | 输入/输出 | 作用 |
|------|-----------|------|
| `find_next_divisible_by_8_numpy(n)` | ndarray → ndarray | 向上取 8 倍数（padding 对齐） |
| `create_sinusoidal_pos_embedding(time, dim, min_period, max_period)` | `(B,)` → `(B, dim)` | 扩散时间正弦编码 |
| `sample_beta(alpha, beta, bsize, device)` | — → `(B,)` | Beta 分布采样 |
| `make_att_2d_masks(pad_masks, att_masks)` | `(B,N)` × 2 → `(B,N,N)` bool | 块稀疏注意力掩码 |
| `resize_with_pad(img, w, h, pad_value=-1)` | `(B,C,H,W)` → padded | 保持比例 resize + 左上 pad |
| `our_eager_attention_forward(Q,K,V,mask)` | `(B,L,H,D)` → `(B,L,H*D)` | GQA + einsum 注意力 |
| `apply_rope(x, positions, max_wavelength)` | `(B,L,H,D)` → 同形 | 应用 RoPE |

---

## 2.7 可运行示例：线性 Flow Matching 玩具

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyVelocityNet(nn.Module):
    """预测 u = eps - x1，仅依赖 x_t 与 t"""
    def __init__(self, dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, dim),
        )

    def forward(self, x_t, t):
        return self.net(torch.cat([x_t, t.unsqueeze(-1)], dim=-1))

def train_step(model, x1, opt):
    B, D = x1.shape
    eps = torch.randn_like(x1)
    t = torch.rand(B)
    x_t = t[:, None] * eps + (1 - t[:, None]) * x1
    u_t = eps - x1
    v_t = model(x_t, t)
    loss = F.mse_loss(v_t, u_t)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

def sample(model, steps=10, dim=7):
    x = torch.randn(1, dim)
    dt = -1.0 / steps
    t = 1.0
    while t >= -dt / 2:
        v = model(x, torch.tensor([t]))
        x = x + dt * v
        t += dt
    return x

# 训练拟合随机目标分布
dim = 7
model = ToyVelocityNet(dim)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.randn(256, dim) * 0.5

for _ in range(500):
    idx = torch.randint(0, 256, (32,))
    train_step(model, target[idx], opt)

print("Sample:", sample(model, steps=20, dim=dim))
```

此示例与 `FlowMatching.forward` / `sample_actions` 的数学一致，便于脱离大模型验证 FM 逻辑。

---

## 2.8 延伸阅读

| 资源 | 链接 |
|------|------|
| Conditional Flow Matching 教程 | [Hugging Face Diffusers FM docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/flow_match) |
| PI0 开源 | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| LeRobot PI0 配置 | [lerobot PI0Config](https://github.com/huggingface/lerobot) |
| Flex Attention | [PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) |
