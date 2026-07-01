---
title: 07 — 训练时方法（FQL / EDP / QAM / IQL / BC 等）
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
# 07 — 训练时方法（FQL / EDP / QAM / IQL / BC 等）

## 1. 本章边界

列出 `agents/` 下 **在训练阶段** 用 RL 信号更新 actor（或等价策略网络）的方法，与 QGF「仅 BC 训练 actor」对比。

---

## 2. 方法总览

| Agent | 文件 | Actor 类型 | RL 信号 | Critic |
|-------|------|------------|---------|--------|
| **BC** | `bc.py` | Flow | 无 | 无 |
| **IQL** | `iql.py` | Gaussian | AWR / DDPG+BC | IQL |
| **EDP** | `edp.py` | Flow | 单步 Q 最大化 + BC | IQL |
| **FQL** | `fql.py` | 一步流 + BC 流 | max Q + 蒸馏 | DDPG 或 IQL |
| **QAM** | `qam.py` | Flow | Adjoint matching | DDPG 或 IQL |
| **CFGRL** | `cfgrl.py` | 条件 Flow | Advantage 加权 BC | IQL |
| **QGF** | `qgf.py` | Flow | **无**（仅 BC） | IQL |
| **SAC** | `sac.py` | 高斯 / 流 | 最大熵 RL | Twin Q |
| 其他 | `bptt.py`, `dsrl.py`, `dcgql.py`, `fbrac.py`, `fawac.py`, `ifql.py`, `iql_diffusion.py` | 各论文复现 | — | — |

---

## 3. BCAgent (`agents/bc.py`)

### 方法

| 方法 | 功能 |
|------|------|
| `policy_loss` | Flow matching（同 QGF） |
| `total_loss` | 仅 BC |
| `update` | 只更新 policy |
| `sample_actions` | Euler 去噪，无 Q |
| `create` | 初始化 `ActorFlowField` 或 `DiffusionPolicy` |

**用途**：纯模仿基线；QGF 的 policy 训练与之相同。

---

## 4. IQLAgent (`agents/iql.py`)

### 方法

| 方法 | 功能 |
|------|------|
| `expectile_loss` | 静态 expectile |
| `value_loss` | V ← expectile(Q) |
| `critic_loss` | IQL TD |
| `actor_loss` | `exp(adv/β)*log π(a|s)`（AWR）或 DDPG+BC |
| `total_loss` / `update` | 联合或分模块 |
| `sample_actions` | 高斯 actor 采样 |
| `target_update` | target critic |
| `create` | GaussianActor + Value |

**与 QGF**：共享 IQL critic/value 思想；IQL **更新 actor**，QGF **不更新**。

---

## 5. EDPAgent (`agents/edp.py`)

**Exploration via Denoising Policy** 风格：训练时 **直接** 最大化 Q。

### `policy_loss`（核心）

1. 标准 BC flow loss
2. 单步近似：$a_{\text{eval}} = \mathrm{clip}(x_t + (1-t)v_{\text{pred}}, -1, 1)$
3. **`q_loss = -mean(Q(s, a_eval))`** — 梯度 **穿过** `v_pred` 到 policy

$$
\mathcal{L} = \mathcal{L}_{\text{BC}} \cdot \lambda_{\text{bc}} - \mathbb{E}[Q(s, a_{\text{eval}})]
$$

### 其他方法

| 方法 | 功能 |
|------|------|
| `critic_loss` / `value_loss` | 同 QGF |
| `update` | policy + critic + value + target |
| `sample_actions` | 纯 BC Euler（**无**测试时 Q 引导） |
| `create` | 同 QGF 结构 |

**与 QGF 对比**：EDP 在 **训练** 用 Q；QGF 在 **推理** 用 Q。EDP 的 `a_eval` 与 QGF 的 `a'` 形式相同，但 EDP 对 policy 反传 Q 梯度。

---

## 6. FQLAgent (`agents/fql.py`)

**Flow Q-Learning**：蒸馏多步 BC 流为 **一步** 策略，并 max Q。

### 网络（`ModuleDict` / `TrainState` 包装）

- `actor_bc_flow`：多步 BC 流（teacher）
- `actor_onestep_flow`：$a = f_\theta(s, \xi)$，$\xi \sim \mathcal{N}$
- `critic` / `target_critic` / `value`

### 方法

| 方法 | 功能 |
|------|------|
| `_get_batch_actions` | chunk 展平 |
| `ddpg_critic_loss` | bootstrap `Q(s', π(s'))` |
| `iql_critic_loss` | bootstrap V |
| `value_loss` | expectile |
| `critic_loss` | 按 `critic_loss_type` 分发 |
| `actor_loss` | BC flow + α·distill + **(-Q)** |
| `compute_flow_actions` | teacher 多步积分 |
| `sample_actions` | **一步** `actor_onestep_flow` |
| `update` / `target_update` | 联合 loss + 软更新 |
| `create` | 构建 ModuleDict |

### `actor_loss` 组成

$$
\mathcal{L} = \mathcal{L}_{\text{bc\_flow}} + \alpha \mathcal{L}_{\text{distill}} - \mathbb{E}[Q(s, \pi_{\text{1step}}(s))]
$$

**推理快**：单前向，无 denoise 循环。

---

## 7. QAMAgent (`agents/qam.py`)

**Q-learning with Adjoint Matching**：无 BPTT 的 Q 感知流训练。

### 主要方法

| 方法 | 功能 |
|------|------|
| `_flat_batch_for_critic` | batch 动作展平 |
| `ddpg_critic_loss` / `iql_critic_loss` | 同 FQL |
| `value_loss` / `critic_loss` | 同 FQL |
| `adj_matching` | 伴随匹配：沿 flow 路径估计 $\partial Q/\partial x$ 并匹配速度场 |
| `actor_loss` | BC + adjoint 项 + 可选 BC 权重 |
| `compute_flow_actions` | 多步 teacher |
| `sample_actions` | 多步或一步（依配置） |
| `update` / `batch_update` | 训练 |
| `create` | 网络初始化 |

**思想**：用伴随 ODE / 匹配条件把 Q 梯度注入 flow **训练**，避免对整个 denoise 链反传。

---

## 8. CFGRLAgent

见 [06-test-time-baselines.md](./06-test-time-baselines.md) §4。训练时 `actor_loss` 含 advantage 加权；亦属训练时改进 actor 的一类。

---

## 9. 训练时 vs QGF 决策树

```
需要推理时扫 guidance_weight、共享 BC+IQL checkpoint？
  └─ 是 → QGF / QFQL / GradStep / RobustQ

需要最快推理（单步）？
  └─ 是 → FQL

愿意训练时调 bc_weight 与 Q 平衡？
  └─ 是 → EDP

需要 SOTA 流训练 Q 感知且无 BPTT？
  └─ 是 → QAM

经典高斯 + 离线 RL？
  └─ IQL / SAC
```

---

## 10. 损失对比公式

| 方法 | Actor 目标 |
|------|------------|
| BC / QGF train | $\|v - (a_1-a_0)\|^2$ |
| EDP | $\lambda \|v-\cdot\|^2 - Q(s, x_t+(1-t)v)$ |
| FQL | BC_flow + α·\|π_1 - flow\|² - Q(s, π_1) |
| QAM | BC + adjoint_match(Q) |
| IQL | AWR: $-w \log π(a|s)$ |

---

## 11. SACAgent（`agents/sac.py`）详解

在线 RL / RLPD 基线：高斯 actor + 双 Q + 可学习温度 $\alpha$。

### 方法（完整）

| 方法 | 功能 |
|------|------|
| `critic_loss(batch, grad_params, rng)` | SAC TD；`q_agg` 取 min/mean；可选 `backup_entropy` 在 target 中减 $\alpha\log\pi$ |
| `actor_loss(batch, grad_params, rng)` | $\mathbb{E}[\alpha\log\pi - Q]$ + 温度损失 + `bc_loss_weight`·MSE$(a_\pi, a_{\text{data}})$ |
| `total_loss` | critic + actor 联合 |
| `target_update(network, "critic")` | 软更新 `modules_target_critic` |
| `update(batch)` | 单次联合优化 |
| `sample_actions(obs, seed, temperature)` | 高斯采样 |
| `create` / `get_config` | ModuleDict 初始化 |

### 与 QGF 关系

- **离线**：QGF 不用 SAC；SAC 用于 `online_steps > 0` 时与 replay 联合微调（若选用该 agent）
- **策略类**：SAC 为高斯；QGF 为 flow，无熵正则

网络细节见 [08-networks-and-modules.md](./08-networks-and-modules.md) §16。

---

## 12. 其他 agents（索引）

| 文件 | 说明 |
|------|------|
| `bptt.py` | 通过 denoise BPTT 的 RL |
| `dsrl.py` | Diffusion / RL 相关 |
| `dcgql.py` | 扩散 + Q-learning 变体 |
| `fbrac.py`, `fawac.py` | 行为正则 actor-critic |
| `ifql.py` | Implicit FQL 变体 |
| `iql_diffusion.py` | IQL + DDPM 扩散策略 |

复现论文时通过 `--agent=agents/xxx.py` 切换；各文件含 `get_config()`。

---

## 13. 运行示例

```bash
# EDP 训练
python main.py --agent=agents/edp.py --agent.bc_weight=1.0 ...

# FQL 训练
python main.py --agent=agents/fql.py --agent.alpha=0.1 ...

# 纯 BC
python main.py --agent=agents/bc.py ...
```
