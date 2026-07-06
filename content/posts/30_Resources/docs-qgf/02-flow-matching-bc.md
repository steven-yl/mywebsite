---
title: 02 — Flow Matching 与行为克隆（BC）策略
subtitle: ""
date: 2026-06-17T10:26:59+08:00
# lastmod: 2026-06-17T10:26:59+08:00
draft: false
authors: [Steven]
description: ""
tags: [diffusion/flow, qgf]
categories: [docs qgf]
series: [qgf-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 02 — Flow Matching 与行为克隆（BC）策略

## 1. 本章边界

- **涵盖**：连续流匹配数学、训练时 BC 目标、`ActorFlowField` 前向、`QGFAgent.policy_loss` / `BCAgent` 等价逻辑、推理 Euler 积分。
- **不涵盖**：IQL 价值学习（见 [03-iql-critic-value.md](./03-iql-critic-value.md)）、测试时 Q 引导（见 [04-qgf-core.md](./04-qgf-core.md)）。

---

## 2. 关键概念

### 2.1 什么是 Flow Matching？

Flow Matching 学习一个时间依赖的速度场 $v_\theta(s, a, t)$，使得从简单源分布（标准高斯噪声 $a_0$）沿 ODE 积分到数据动作 $a_1$：

$$
\frac{da}{dt} = v_\theta(s, a, t), \quad t \in [0, 1]
$$

**为什么需要**：离线 RL 数据常多模态；高斯 policy 难以覆盖。Flow 通过 BC 拟合数据分布，推理时从噪声采样得到多样化动作。

**解决什么问题**：在**不**做 RL 梯度的情况下，获得表达力强的生成式策略 $\pi_{\text{BC}}$。

### 2.2 线性插值路径（本仓库实现）

训练时对每个样本 $(s, a_1)$：

1. 采样 $a_0 \sim \mathcal{N}(0, I)$
2. 采样离散时间 $t \in \{0, 1/N, \ldots, 1\}$（`denoise_steps = N`）
3. 构造插值 $a_t = (1-t)\,a_0 + t\,a_1$
4. 目标速度 $v^* = a_1 - a_0$（与 $t$ 无关的常向量场，对应 rectified flow / OT 直线路径）

**损失**（均方误差）：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{a_0, t}\big[\| v_\theta(s, a_t, t) - (a_1 - a_0) \|^2\big]
$$

对应代码 `QGFAgent.policy_loss`：

```python
a_t = a0 * (1 - tv) + a1 * tv
vel = a1 - a0
pred_vel = self.policy(batch["observations"], a_t, t, params=policy_params)
bc_loss = jnp.mean((vel - pred_vel) ** 2)
```

### 2.3 Action Chunking 下的 BC

当 `action_chunking=True` 且 `horizon_length=H`：

- 数据集 `actions` 形状 `(B, H, d_a)`
- BC 将 $H$ 步动作 **展平** 为 `(B, H·d_a)` 的向量，流场输出同维
- 与 critic 的 Q(s, a_{1:H}) 维度一致

---

## 3. 网络：`ActorFlowField`

**文件**：`utils/networks.py`

| 组件 | 作用 |
|------|------|
| `embed_time(t, "sinusoidal")` | 正弦位置编码，将标量 $t$ 映射到 16 维 |
| 输入拼接 | `[obs, noised_action, time_emb]` |
| `MLP` trunk | 4×512 GELU + LayerNorm（默认配置） |
| 输出 `Dense(action_dim)` | 速度向量 $v$ |

```python
# 概念性前向（与源码一致）
concat = jnp.concatenate([obs, noised_action, embed_time(t)], axis=-1)
v = Dense(action_dim)(MLP(concat))
```

**`time_embedding` 选项**：
- `"sinusoidal"`（默认）：与扩散模型类似的时间编码
- `"raw"`：直接拼接标量 $t$

---

## 4. 推理：Euler 去噪

无 Q 引导时（纯 BC，`guidance_weight=0`）：

$$
a_{k+1} = a_k + v_\theta(s, a_k, t_k)\,\Delta t, \quad \Delta t = 1/N
$$

初始化 $a \sim \mathcal{N}(0,I)$，$t_k = k/N$，最后 `clip(-1, 1)`。

`BCAgent.sample_actions`（`agents/bc.py`）即此逻辑；`QGFAgent` 在 BC 速度上叠加 Q 梯度（见第 4 章）。

---

## 5. `BCAgent` 与 `QGFAgent.policy_loss` 对比

| 类 | 文件 | policy_loss | 其他模块 |
|----|------|-------------|----------|
| `BCAgent` | `agents/bc.py` | 相同 flow matching | 无 critic/value |
| `QGFAgent` | `agents/qgf.py` | 相同 | + IQL critic/value |

`BCAgent` 额外提供 `DiffusionPolicy`（DDPM 版 BC，见 [08-networks-and-modules.md](./08-networks-and-modules.md)）。

---

## 6. 可运行示例（JAX 伪流程）

下列片段展示 BC 训练一步的核心计算（需在本仓库环境中加载 agent）：

```python
import jax
import jax.numpy as jnp
from agents.qgf import QGFAgent, get_config

# 假设已有 ex_observations (B, obs_dim), ex_actions (B, action_dim)
config = get_config()
agent = QGFAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=config)

# 模拟 batch（单步 transition）
batch = {
    "observations": ex_obs,
    "actions": ex_act[:, None, :],  # (B, 1, d_a) 当 H=1
    "next_observations": ex_obs,
    "rewards": jnp.zeros((ex_obs.shape[0], 1)),
    "masks": jnp.ones((ex_obs.shape[0], 1)),
    "valid": jnp.ones((ex_obs.shape[0], 1)),
}

agent, info = agent.update(batch)
print(info["bc_loss"])  # policy BC loss
```

---

## 7. 超参数说明

| 配置项 | 默认 | 含义 |
|--------|------|------|
| `denoise_steps` | 10 | 训练/推理离散步数 $N$ |
| `bc_lr` | 3e-4 | 策略 Adam 学习率 |
| `actor_hidden_dims` | (512,)×4 | 流场 MLP 宽度 |
| `use_layer_norm` | 1 | trunk 是否 LayerNorm |
| `activation` | gelu | 激活函数 |

---

## 8. 优缺点与适用场景

| | 说明 |
|---|------|
| **优点** | 训练稳定；多模态；与 IQL 解耦后易复用 checkpoint |
| **缺点** | 推理 $O(N)$ 次网络前向；纯 BC 无法超越数据集支持的动作 |
| **适用** | 作为 QGF / FQL / EDP 等方法的 **行为先验** 底座 |
