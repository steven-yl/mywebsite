---
title: 05 — 测试时引导变体（QFQL / Jacobian / Best-of-N）
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
# 05 — 测试时引导变体（QFQL / Jacobian / Best-of-N）

## 1. 本章边界

本章专讲 **同一 `QGFAgent` checkpoint** 上，通过配置与 `sample_actions` 参数切换的推理变体。训练代码不变。

---

## 2. 配置驱动的变体

### 2.1 QGF（默认）

```bash
--agent.denoised_action_approx=one_euler_step_approx \
--agent.apply_jacobian=False
```

- 在 $a' = \mathrm{clip}(a_t + (1-t)v_{\text{BC}}, -1, 1)$ 上求 $\nabla Q$
- 将梯度当作 $\nabla_{a_t} Q$（忽略 $\partial a'/\partial a_t$）

**适用**：论文主方法；低方差、OOD 风险小。

### 2.2 QFQL 基线

```bash
--agent.denoised_action_approx=noisy \
--agent.apply_jacobian=False
```

- $a' = a_t$（仍在噪声流形上）
- $\nabla_{a_t} Q(s, a_t)$ 常落在 critic 训练分布外 → **有偏、不稳定**

**脚本**：`scripts/exp_qfql_test_time_eval.py`

### 2.3 QGF-Jacobian

```bash
--agent.denoised_action_approx=one_euler_step_approx \
--agent.apply_jacobian=True
```

对单步 map $f(a_t) = \mathrm{clip}(a_t + (1-t)v(a_t), -1, 1)$ 计算 Jacobian：

$$
\nabla_{a_t} Q \leftarrow \big(\nabla_{a'} Q\big)^\top \frac{\partial f}{\partial a_t}
$$

实现（`sample_actions` 内）：

```python
def map_single(a_i, obs_i, tv_i):
    v = self.policy(obs_i[None], a_i[None], tv_i)[0]
    return jnp.clip(a_i + (1 - tv_i[0]) * v, -1, 1)

jac = jax.vmap(jax.jacrev(map_single, argnums=0))(a, observations, tv)
qgrad = jnp.einsum("bi,bij->bj", qgrad, jac)
```

**注意**：`v` 在 Jacobian 中 **不** stop_gradient（与 $a'$ 计算不同），链式法则更完整但方差更高。

**脚本**：`scripts/exp_qgf_jacobian_test_time_eval.py`

---

## 3. `guidance_weight` 扫描

测试时 RL 不重新训练，通过 $w$ 控制引导强度：

$$
v = v_{\text{BC}} + w \cdot \nabla Q
$$

| 现象 | 解释 |
|------|------|
| $w=0$ | 纯 BC，与训练时 `guidance_weights=0.0` 评估一致 |
| $w$ 过小 | 接近 BC，提升有限 |
| $w$ 过大 | Q 梯度主导，可能破坏流场稳定性、动作 OOD |

**评估入口**：`utils/evaluation.eval_with_test_time_guidance` 对 `guidance_weights` 列表逐个跑 episode，选验证回报最高者。

```bash
--guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12
```

---

## 4. Rejection Sampling（Best-of-N）

`sample_actions(..., rejection_sampling=K)`：

1. 将同一 `obs` 复制 K 份
2. 独立噪声，完整去噪
3. 用 `aggregate_q(target_critic(s, a))` 选最大 Q 的动作

**与 $w$ 正交**：可同时使用 `guidance_weight>0` 与 `K>1`。

**代价**：推理时间 × K。

---

## 5. 变体对比表

| 变体 | 训练 | 推理 Q 评估点 | 链式法则 | 典型 $w$ 调参 |
|------|------|---------------|----------|----------------|
| BC | BC | 无 Q | — | — |
| QGF | BC+IQL | $a'$ 近似干净 | 否 | 需要 |
| QFQL | BC+IQL | $a_t$ 噪声 | 否 | 需要 |
| QGF-J | BC+IQL | $a'$ | 是 | 需要 |
| + Best-of-N | 同上 | 同上 | 同上 | + `rejection_sampling` |

---

## 6. 实验脚本与 checkpoint 共享

`scripts/bc_iql_train.py` 训练 **单一** QGF agent（`guidance_weight` 在训练评估时为 0）。  
各 `exp_*_test_time_eval.py` **加载同一 checkpoint**，仅改：

- `denoised_action_approx`
- `apply_jacobian`
- `guidance_weights` 列表

这体现了「测试时 scaling」：**一次训练，多种推理策略**。

---

## 7. 概念：Classifier Guidance 类比

扩散模型中 classifier guidance：

$$
\tilde{\epsilon} = \epsilon_\theta(x_t,t) - w \cdot \sigma_t \nabla_x \log p(c|x_t)
$$

QGF 将 $\log p(c|\cdot)$ 替换为 $Q(s,\cdot)$，将 $\epsilon$ 替换为 flow 速度 $v$，在 **动作空间** 做逐步引导。

---

## 8. 调试建议

1. 先固定 `guidance_weight=0` 确认 BC 回报正常
2. 从小到大扫 $w$，观察是否单调提升（非保证）
3. 对比 `noisy` vs `one_euler_step_approx` 在同一 $w$ 下的稳定性
4. 若 GPU 内存允许，试 `rejection_sampling=4` 或 `8`
