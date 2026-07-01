---
title: 06 — 测试时基线：GradStep 与 RobustQ
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
# 06 — 测试时基线：GradStep 与 RobustQ

## 1. 本章边界

- **GradStep**（`agents/grad_step.py`）：后处理梯度上升
- **RobustQ**（`agents/robust_q.py`）：噪声条件 Q_robust
- **CFGRL**（`agents/cfgrl.py`）：条件流 + guidance 插值（测试时可用 `guidance_weight`）

均与 QGF 共享 **BC + IQL** 训练范式或部分组件，推理机制不同。

---

## 2. GradStepAgent

**继承**：`QGFAgent`  
**`support_guidance = False`**：评估 API 不传 `guidance_weight`

### 2.1 思想

1. **阶段 A**：完整 BC Euler 去噪 → 干净动作 $a_1$
2. **阶段 B**：在干净动作空间做 L 步梯度上升  
   $a \leftarrow \mathrm{clip}(a + \alpha \nabla_a Q(s,a), -1, 1)$

Q 梯度 **不** 进入去噪内环（与 QGF 每步引导对比）。

### 2.2 方法清单

| 方法 | 功能 |
|------|------|
| `sample_actions(obs, seed, rejection_sampling=1)` | BC 去噪 + Q  refine |
| `get_config()` | 继承 QGF + `qgrad_step_size`, `qgrad_steps` |

### 2.3 内部函数

**`_bc_denoise(obs, noise)`**
- `jax.lax.scan` over `denoise_steps`
- 每步 `x += policy(obs, x, t) * dt`

**`_qgrad_refine(obs, actions)`**
- `q_fn(a) = aggregate_q(target_critic(obs, a)).sum()`
- scan `qgrad_steps` 次，`step_size = qgrad_step_size`（默认 0.1）
- 可选 `use_sign_gradient`：用 `sign(g)` 代替 `g`

### 2.4 与 QGF 对比

| | QGF | GradStep |
|---|-----|----------|
| Q 梯度时机 | 每 denoise 步 | 仅去噪后 |
| 计算 | N × (policy + grad Q) | N × policy + L × grad Q |
| OOD | 通过 $a'$ 缓解 | 在 $a_1$ 上（较干净） |

---

## 3. RobustQAgent

**继承**：`QGFAgent`  
**额外字段**：`robust_critic: TrainState`

### 3.1 训练思想

学习 $Q_{\text{robust}}(s, a_t, t)$ 回归干净 IQL 的 $Q(s, a_1)$：

$$
\mathcal{L}_{\text{robust}} = \mathbb{E}\big[(Q_{\text{robust}}(s, a_t, t) - \mathrm{sg}(Q(s,a_1)))^2\big]
$$

其中 $a_t = (1-t)a_0 + t a_1$，与 flow 路径一致。

**为什么**：标准 $Q(s,a)$ 只在数据动作上训练；$Q_{\text{robust}}$ 在 **整条去噪路径** 上监督，使 $\nabla_{a_t} Q_{\text{robust}}(s,a_t,t)$ 在各 $t$ 分布内。

### 3.2 方法清单

| 方法 | 功能 |
|------|------|
| `robust_critic_loss(batch, robust_critic_params, rng)` | 噪声动作 + 时间嵌入，MSE 到 target Q |
| `update(batch)` | policy + critic + value + **robust_critic** + target |
| `total_loss(...)` | 含 robust loss（诊断） |
| `sample_actions(obs, seed, rejection_sampling)` | BC + $Q_{\text{robust}}$ 梯度引导 |
| `create(...)` | 额外初始化 `robust_critic` |
| `get_config()` | + `robust_critic_lr`, `robust_critic_t_emb_size` |

### 3.3 `robust_critic_loss` 步骤

1. 取干净 `actions`（与 policy_loss 相同 reshape）
2. 采样 `x0`, `t ~ U(0,1)`，`a_t = (1-t)*x0 + t*actions`
3. `t_emb = timestep_embedding(t, emb_size)`
4. 输入 critic：`concat(a_t, t_emb)` 作为 **action 侧输入**
5. 目标：`stop_gradient(aggregate_q(target_critic(s, actions)))`

### 3.4 推理 `sample_actions`

每步：

$$
v = v_{\text{BC}} + w_{\text{cfg}} \cdot \nabla_{a_t} Q_{\text{robust}}(s, a_t, t)
$$

其中 $Q_{\text{robust}}$ 输入为 `concat(a_t, t_emb)`。

### 3.5 实现注意：未定义 `cfg`

当前 `sample_actions` 第 148 行使用变量 **`cfg`**，函数内 **未定义** 该符号：

```python
return x + (v_bc + cfg * qgrad) * dt, None
```

**预期行为**：应使用配置项（如 `guidance_weight` 或 `config["cfg"]`）与 QGF 的 `guidance_weight` 对齐。使用前需修复或在外部 wrapper 中 patch；文档记录此为已知缺陷。

**`support_guidance`**：类未设为 True，评估脚本默认 **不能** 通过 `eval_with_test_time_guidance` 扫 `guidance_weight`（除非改类属性或修复后添加参数）。

---

## 4. CFGRLAgent（概要）

**文件**：`agents/cfgrl.py`  
**`support_guidance = True`**

### 4.1 网络

- `ConditionalFlowField`：输入 `(obs, is_positive, a_t, t)`
- `is_positive`：二值嵌入，标记 advantage 正负

### 4.2 训练 `actor_loss`

- 标准 flow matching + **advantage 加权** 条件分支
- 需 `additional_agents` 提供 `value` agent 估计 advantage

### 4.3 推理 `sample_actions`

Classifier-free guidance 式组合：

$$
v = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})
$$

在 Euler 每步对 `is_positive=0/1` 两次前向。

### 4.4 主要方法

| 方法 | 功能 |
|------|------|
| `critic_loss` / `value_loss` | 同 QGF IQL |
| `actor_loss` | 条件 + 无条件 flow + advantage 权重 |
| `update` | critic, value, actor 独立更新 |
| `sample_actions` | CFG 式引导，`guidance_weight` |
| `create` | ConditionalFlowField + IQL |

---

## 5. 三方法推理对比

```
QGF:     每步  v_bc + w·∇Q(s, a'(a_t))
GradStep: 先 BC → a_1，再 L 步 ∇Q(s, a)
RobustQ: 每步  v_bc + w·∇Q_robust(s, a_t, t)
CFGRL:   每步  v_u + w·(v_c - v_u)
```

| 方法 | 额外表 | 测试时超参 |
|------|--------|------------|
| QGF | 无 | `guidance_weight` |
| GradStep | 无 | `qgrad_step_size`, `qgrad_steps` |
| RobustQ | `robust_critic` | 应为 `guidance_weight`（待修复 `cfg`） |
| CFGRL | 条件 actor | `guidance_weight` |

---

## 6. GradStep 配置示例

```python
from agents.grad_step import GradStepAgent, get_config
config = get_config()
config.qgrad_step_size = 0.1
config.qgrad_steps = 3
```

---

## 7. RobustQ 配置示例

```python
from agents.robust_q import RobustQAgent, get_config
config = get_config()
config.robust_critic_lr = 3e-4
config.robust_critic_t_emb_size = 16
```

训练时 `update` 比 QGF 多一步 `robust_critic` 更新；推理前需确保 `cfg`/guidance 权重可用。
