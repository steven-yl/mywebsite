---
title: 04 — QGF 核心：`QGFAgent` 完整解读
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


# 04 — QGF 核心：`QGFAgent` 完整解读

## 1. 类结构

**文件**：`agents/qgf.py`  
**基类**：`flax.struct.PyTreeNode`（JAX 可 JIT 的 immutable agent）

### 1.1 字段

| 字段 | 类型 | 作用 |
|------|------|------|
| `rng` | PRNGKey | 全局随机数 |
| `policy` | `TrainState` | BC 流场 $v_\theta$ |
| `critic` | `TrainState` | IQL Q ensemble |
| `target_critic` | `TrainState` | 目标 Q（推理梯度用） |
| `value` | `TrainState` | IQL V |
| `config` | `FrozenDict` | 超参（非 pytree） |

### 1.2 类属性

- `support_guidance = True`：评估时可传 `guidance_weight`（见 `evaluation._is_test_time_guidance_agent`）

---

## 2. 方法索引（无遗漏）

| 方法 | 阶段 | 功能摘要 |
|------|------|----------|
| `_aggregate_q` | 工具 | → `aggregate_q` |
| `_get_flat_batch` | 工具 | → `get_flat_batch` |
| `policy_loss` | 训练 | Flow matching BC |
| `critic_loss` | 训练 | IQL TD |
| `value_loss` | 训练 | IQL expectile |
| `total_loss` | 训练 | 三 loss 求和（诊断用） |
| `update` | 训练 | 分模块 Adam + target 更新 |
| `sample_actions` | 推理 | Q 引导 Euler 去噪 |
| `create` | 构造 | 初始化网络与 optim |
| `get_config` | 配置 | 默认超参 |

---

## 3. `policy_loss` — 逐行逻辑

1. `split` RNG → `eps_rng`, `time_rng`
2. **取干净动作** `a1`：chunking 则 reshape `(B,-1)`，否则 `actions[...,0,:]`
3. `a0 ~ N(0,I)`，`t ~ Uniform{0,1/N,...,1}`
4. `a_t = (1-t)*a0 + t*a1`，`vel = a1 - a0`
5. MSE(`policy(obs,a_t,t)`, `vel`)

返回 `(bc_loss, {"bc_loss": bc_loss})`

---

## 4. `critic_loss` / `value_loss`

见 [03-iql-critic-value.md](./03-iql-critic-value.md)。  
`critic_loss` 可选 `critic_params` 覆盖参数（用于外部 grad）；默认 `self.critic.params`。

---

## 5. `update` — 训练一步

```
new_policy    ← Adam(policy_loss)
new_critic    ← Adam(critic_loss)
new_target    ← τ·critic + (1-τ)·target_critic
new_value     ← Adam(value_loss)
new_rng       ← split
```

**关键设计**：三个模块 **独立** 反向，policy **从不** 接收 Q/V 的梯度。

---

## 6. `sample_actions` — 推理核心

### 6.1 签名

```python
def sample_actions(
    self,
    observations,
    *,
    seed,
    guidance_weight: float = 1.0,
    rejection_sampling: int = 1,
) -> actions
```

### 6.2 算法步骤

```
输入: s, seed, w=guidance_weight, N=denoise_steps
a ← N(0,I)  # 形状 (B, full_action_dim)
Δt ← 1/N

for t_idx = 0 .. N-1:
    t ← t_idx / N
    v_bc ← policy(s, a, t)

    # --- 干净动作近似 a' ---
    if denoised_action_approx == "noisy":
        a' ← a                          # QFQL 基线
    elif denoised_action_approx == "one_euler_step_approx":
        a' ← clip(a + (1-t) * stop_grad(v_bc), -1, 1)   # QGF 默认

    # --- Q 梯度 ---
    qgrad ← ∇_{a'} aggregate_q(target_critic(s, ·)).sum()
    # 注意: a' 传入 grad 前 stop_gradient

    # --- 可选 Jacobian 链式法则 ---
    if apply_jacobian:
        J ← ∂a'/∂a  (对 one_euler_step_approx 的 map)
        qgrad ← qgrad @ J   # einsum("bi,bij->bj", ...)

    a ← a + (v_bc + w * qgrad) * Δt

a ← clip(a, -1, 1)

# --- 可选 rejection sampling ---
若 rejection_sampling > 1:
    采样 K 条轨迹，取 Q(s,a) 最大的动作
```

### 6.3 数学形式（QGF 默认）

单步 Euler 近似干净动作：

$$
a' = \mathrm{clip}\big(a_t + (1-t)\,v_{\text{BC}}(s,a_t,t),\,-1,\,1\big)
$$

Q 梯度（**丢弃** $\partial a'/\partial a_t$，即 Jacobian=I）：

$$
\nabla_{a_t} Q \approx \nabla_{a'} Q(s, a')
$$

引导速度：

$$
v = v_{\text{BC}} + w \cdot \nabla_{a_t} Q
$$

### 6.4 三种推理模式对比

| 模式 | `denoised_action_approx` | `apply_jacobian` | 评估 Q 的点 | 梯度到 $a_t$ |
|------|------------------------|------------------|-------------|----------------|
| **QGF** | `one_euler_step_approx` | False | 近似干净 $a'$ | $\nabla_{a'}Q$（近似） |
| **QFQL** | `noisy` | False | 噪声 $a_t$ | $\nabla_{a_t}Q$（OOD） |
| **QGF-Jacobian** | `one_euler_step_approx` | True | $a'$ | $J^\top \nabla_{a'}Q$ |

### 6.5 `rejection_sampling`

- 将 batch 复制 K 份，独立噪声去噪
- 用 `target_critic` 的 aggregate Q 选最优
- 类似 Best-of-N，无需改训练

---

## 7. `create` — 初始化

1. 计算 `full_action_dim = action_dim * H`（若 chunking）
2. `ActorFlowField` → `policy`（lr=`bc_lr`）
3. `Value(num_ensembles=num_qs)` → `critic` / `target_critic`（lr=`critic_lr`）
4. `Value(num_ensembles=1)` → `value`（lr=`value_lr`）
5. 写入 `config["action_dim"]`

---

## 8. `get_config` 默认项

```python
denoised_action_approx = "one_euler_step_approx"
apply_jacobian = False
denoise_steps = 10
q_aggregation = "min"
expectile = 0.9
horizon_length = 1
action_chunking = False
```

---

## 9. 示意图：单步去噪

```
     a_t (噪声侧)                    a' (近似干净)
        │                                ▲
        │  v_bc                          │  (1-t)*v_bc
        ├──────────────────────────────►│
        │                                │
        │         Q(s, ·)                │
        └──────── ∇Q ──► 叠加到 v ──────┘
                    v_guided = v_bc + w·∇Q
```

---

## 10. 调用关系

```
main.py / scripts
    └── agent.update(batch)          # 训练
    └── agent.sample_actions(        # 评估
            obs, seed=..., 
            guidance_weight=w,
            rejection_sampling=K,
        )

evaluation.run_episodes
    └── _prepare_actor → partial(sample_actions, guidance_weight=w)
```

---

## 11. 设计动机（论文对齐）

| 问题 | QGF 做法 |
|------|----------|
| BPTT 穿过去噪链 | 训练不用 RL；推理不用 BPTT |
| QFQL 在 OOD $a_t$ 上求 Q | 在单步 Euler 近干净 $a'$ 上求 Q |
| 完整 Jacobian 方差大 | 默认丢弃 $J$，可选 `apply_jacobian` |

---

## 12. 最小推理示例

```python
import jax.numpy as jnp
from utils.flax_utils import supply_rng

act_fn = supply_rng(agent.sample_actions)
obs = jnp.zeros((1, obs_dim))  # batch=1
action = act_fn(obs, guidance_weight=0.08, rejection_sampling=1)
# action shape: (1, action_dim) 或 (1, H*action_dim)
```
