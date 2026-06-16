---
title: 03 — IQL 价值函数与 Critic
subtitle: ""
date: 2026-06-17T10:26:59+08:00
# lastmod: 2026-06-17T10:26:59+08:00
draft: false
authors: [Steven]
description: ""
tags: [diffusion/flow, qgf]
categories: [qgf]
series: [qgf-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 03 — IQL 价值函数与 Critic

## 1. 本章边界

- **涵盖**：Implicit Q-Learning (IQL) 的 $Q$、$V$ 训练；`QGFAgent.critic_loss` / `value_loss`；`aggregate_q`、`get_flat_batch`；n-step / action chunking 的 TD 目标。
- **不涵盖**：Actor 的 AWR/DDPG 更新（见 `agents/iql.py`）、测试时引导。

---

## 2. IQL 核心思想

IQL 在**不**对 OOD 动作做 max-Q 的前提下学习价值：

1. **Critic** $Q_\phi(s,a)$：标准 TD，bootstrap $V_\psi(s')$
2. **Value** $V_\psi(s)$：对 **in-distribution** 的 $Q(s,a_{\text{data}})$ 做 **期望分位回归**（expectile），使 $V$ 逼近数据动作上的 Q 上尾

**为什么需要 expectile**：离线 RL 中 max over actions 会选中 OOD 高 Q。Expectile 只向上拟合 $Q-V$ 的正部分（由 `expectile > 0.5` 控制），等价于估计 **行为策略下动作的 Q 的较高分位**，避免 OOD 过估计。

---

## 3. 损失函数

### 3.1 Critic（`critic_loss`）

$$
y = r_{\text{cum}} + \gamma^H \cdot m \cdot V_\psi(s_{t+H})
$$
$$
\mathcal{L}_Q = \mathbb{E}\Big[\sum_{i=1}^{K} (Q_{\phi,i}(s, a) - y)^2 \cdot w_{\text{valid}}\Big]
$$

- $H$ = `horizon_length`（n-step 或 chunk 长度）
- $r_{\text{cum}}$ = `sample_sequence` 返回的累积折扣回报（见第 9 章）
- `num_qs=2`：双 Q ensemble，**训练**用全部 head；**目标/聚合**见下
- `target_critic`：软更新 `tau=0.005`

### 3.2 Value（`value_loss`）

$$
\mathcal{L}_V = \mathbb{E}\big[\rho_\tau(Q_{\bar\phi}(s,a) - V_\psi(s)) \cdot w_{\text{valid}}\big]
$$

期望分位权重（`expectile_loss`）：

$$
\rho_\tau(\delta) = \begin{cases}
\tau \cdot \delta^2 & \delta > 0 \\
(1-\tau) \cdot \delta^2 & \delta \le 0
\end{cases}
$$

默认 $\tau = 0.9$（`expectile`）。

### 3.3 Q 聚合（`aggregate_q`）

```python
def aggregate_q(qs, config):
    aggregation_fn = getattr(jnp, config.get("q_aggregation", "min"))
    return aggregation_fn(qs, axis=0)
```

- `q_aggregation="min"`：Clipped Double-Q，降低过估计（**推理 Q 梯度默认用 target_critic + min**）
- `"mean"`：ensemble 平均

---

## 4. 辅助函数完整解读

### 4.1 `get_flat_batch(batch, config)`

**作用**：从 `sample_sequence` 的序列 batch 中提取 critic/value 所需的标量/向量字段。

| 返回值 | 形状 / 含义 |
|--------|-------------|
| `batch_actions` | chunking 时 `(B, H·d_a)`，否则 `(B, d_a)` |
| `next_obs` | `next_observations[..., -1, :]` 序列末状态 |
| `rewards` | `rewards[..., -1]` 累积 n-step 回报 |
| `masks` | `masks[..., -1]` bootstrap 掩码 |
| `valid_w` | `valid[..., -1]`  episode 截断后步权重为 0 |

**为什么需要**：`sample_sequence` 输出 `(B, H, ·)` 结构；IQL TD 只需 **整条序列末尾** 的 bootstrap 与 **整段动作块** 作为 Q 的输入。

### 4.2 `aggregate_q(qs, config)`

**作用**：将 `(num_qs, B)` 的多 head Q 压成 `(B,)` 标量，用于 value expectile 目标与推理时的 Q 梯度。

---

## 5. `QGFAgent` 中的 IQL 方法清单

| 方法 | 功能 |
|------|------|
| `_aggregate_q(qs)` | 调用 `aggregate_q` |
| `_get_flat_batch(batch)` | 调用 `get_flat_batch` |
| `critic_loss(batch, critic_params=None)` | TD MSE，记录 `critic_loss`, `q` |
| `value_loss(batch, value_params=None)` | expectile，记录 `value_loss`, `v` |
| `update` 内 `target_update(critic, target_critic, tau)` | 软更新 target |

**注意**：`QGFAgent.update` 对 policy / critic / value **分别** `apply_loss_fn`，**不是** `total_loss` 的联合梯度（`total_loss` 存在但主循环用 `update`）。

---

## 6. 网络：`Value`

**文件**：`utils/networks.py`

- 输入：`concat(obs [, actions])`
- `num_ensembles=2`（critic）或 `1`（value）
- `network_class`：`"MLP"`（默认 4×512）或 `"BroNet"`（残差块 critic 放大）

```python
# Critic 前向
qs = Value(num_ensembles=2)(observations, batch_actions)  # shape (2, B)

# Value 前向
v = Value(num_ensembles=1)(observations)  # shape (B,)
```

---

## 7. 与 `IQLAgent` 的差异

| | `QGFAgent` | `IQLAgent` (`agents/iql.py`) |
|---|-----------|------------------------------|
| Actor | Flow BC，**无** RL loss | `GaussianActor` + AWR 或 DDPG+BC |
| Critic/Value | 相同 IQL 结构 | 相同 |
| 推理 | Flow denoise + 可选 Q 引导 | 高斯采样 |

QGF **刻意不**调用 `actor_loss`，这是「测试时 RL」与标准 IQL 的分界点。

---

## 8. 公式示意：n-step TD

对 `horizon_length=H`，`sample_sequence` 计算：

$$
R_t^{(H)} = \sum_{i=0}^{H-1} \gamma^i r_{t+i} \cdot \prod_{j<i} \text{valid}_{t+j}
$$

Critic 目标：

$$
y = R_t^{(H)} + \gamma^H \cdot \mathbb{1}[\text{未终止}] \cdot V(s_{t+H})
$$

---

## 9. 可运行示例

```python
# 使用 sample_sequence 的 batch 更新 QGF（需真实 Dataset）
batch = train_dataset.sample_sequence(
    batch_size=256,
    sequence_length=config.horizon_length,
    discount=config.discount,
)
agent, info = agent.update(batch)
# info 含 bc_loss, critic_loss, value_loss, q, v 等
```

---

## 10. 超参数

| 键 | 默认 | 说明 |
|----|------|------|
| `discount` | 0.99 | $\gamma$ |
| `expectile` | 0.9 | 期望分位 $\tau$ |
| `tau` | 0.005 | target 软更新 |
| `num_qs` | 2 | Q ensemble 大小 |
| `critic_lr` / `value_lr` | 3e-4 | 学习率 |
| `horizon_length` | 1 | n-step / chunk 长度 |
| `action_chunking` | False | 是否展平 H 步动作 |
