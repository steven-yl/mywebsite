---
title: "10 — Agents 算法全景：原理、特点与优缺点对比"
subtitle: ""
date: 2026-07-14T18:00:00+08:00
draft: false
authors: [Steven]
description: "本文档对 agents/ 目录下全部算法实现做统一解读：数学原理、训练/推理机制、各自特点，以及横向优缺点对比。适合在读完 01-algorithm-overview.md 后，作为「方法选型手册」使用。"
summary: "本文档对 agents/ 目录下全部算法实现做统一解读：数学原理、训练/推理机制、各自特点，以及横向优缺点对比。适合在读完 01-algorithm-over…"
tags: [diffusion/flow, qgf]
categories: [docs qgf]
series: [qgf-docs]
weight: 10
series_weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 10 — Agents 算法全景：原理、特点与优缺点对比

本文档对 `agents/` 目录下全部算法实现做**统一解读**：数学原理、训练/推理机制、各自特点，以及横向优缺点对比。适合在读完 [01-algorithm-overview.md](./01-algorithm-overview.md) 后，作为「方法选型手册」使用。

---

## 1. 问题背景与公共设定

### 1.1 离线强化学习

给定固定数据集 $\mathcal{D} = \{(s, a, r, s')\}$，在不与环境在线交互的前提下学习策略 $\pi(a|s)$，使期望回报高于数据收集策略。

本仓库约定：

- 动作归一化到 $[-1, 1]$
- 支持 **action chunking**：将 $H$ 步动作拼接为 $H \times d_a$ 维向量
- Critic 使用 **n-step TD**：目标为 $r_{\text{cum}} + \gamma^H V(s_{t+H})$

### 1.2 生成式策略：Flow Matching

多数 agent 用连续归一化流（CNF）思想表达多模态行为分布：

$$
\frac{da}{dt} = v_\theta(s, a, t), \quad t: 0 \to 1
$$

训练（BC 目标）：在直线路径 $a_t = (1-t)a_0 + t a_1$ 上回归速度 $v = a_1 - a_0$：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{a_0 \sim \mathcal{N}, t}\big[\|v_\theta(s, a_t, t) - (a_1 - a_0)\|^2\big]
$$

推理：Euler 离散 $N$ 步，从 $a_0 \sim \mathcal{N}(0,I)$ 积分到干净动作。

### 1.3 IQL 价值学习（共享底座）

多个 agent 共享 **Implicit Q-Learning** 的 critic/value 训练，**不在 OOD 动作上 bootstrap**：

| 模块 | 损失 |
|------|------|
| Critic $Q_\phi$ | $\mathbb{E}[(Q(s,a) - r - \gamma^H V(s'))^2]$ |
| Value $V_\psi$ | Expectile 回归：$V \leftarrow \mathbb{E}_\tau[Q(s,a)]$ |

关键超参：`expectile`（默认 0.9）、`q_aggregation`（`min` / `mean`）、`tau`（target 软更新）。

---

## 2. 方法分类体系

```
                         agents/ 算法全景
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
  纯模仿 / 经典离线 RL    训练时 Q 感知流策略      测试时策略改进
       │                       │                       │
   BC, IQL, SAC          FQL, EDP, QAM,          QGF, GradStep,
   IQL-Diffusion         CFGRL, FBRAC,           RobustQ, IFQL,
   FAWAC, DSRL,          DCGQL                   BPTT, CFGRL
   DCGQL
```

**核心对比轴**：

| 维度 | 训练时方法 | 测试时方法 |
|------|-----------|-----------|
| Actor 是否接受 RL 梯度 | 是 | 否（BC 仅 flow matching） |
| 策略改进阶段 | 训练 | 推理 |
| 典型超参 | `alpha`, `bc_weight` | `guidance_weight` |
| Checkpoint 复用 | 各方法独立训练 | QGF 族可共享 BC+IQL 底座 |

---

## 3. 基础与模仿方法

### 3.1 BC — 纯行为克隆

**文件**：`agents/bc.py`  
**论文**：Flow Matching for BC（无 RL 改进）

#### 原理

仅用 flow matching 模仿数据分布 $\pi_{\text{data}}(a|s)$，不训练 critic，不做价值估计。

#### 训练

$$
\mathcal{L} = \mathcal{L}_{\text{BC}}
$$

仅更新 `policy`（`ActorFlowField` / `FlowMatchingPolicy`）。

#### 推理

标准 Euler 去噪，无 Q 引导：

$$
a_{t+\Delta t} = a_t + v_\theta(s, a_t, t)\,\Delta t
$$

#### 特点

| 优点 | 缺点 |
|------|------|
| 训练最稳定、实现最简单 | 无法超越数据策略（无 RL 信号） |
| 推理逻辑清晰 | 对低质量数据无纠偏能力 |
| 是 QGF 等方法的 policy 训练基线 | 多步去噪，推理慢于一步策略 |

**适用场景**：模仿基线、验证数据/环境 pipeline、作为 QGF 训练阶段的 policy 模块。

---

### 3.2 IQL — 隐式 Q 学习（高斯策略）

**文件**：`agents/iql.py`  
**论文**：Kostrikov et al., 2021

#### 原理

通过 expectile 回归学习 $V(s)$，使 Q 仅在**数据集动作**上 bootstrap，避免 OOD Q 查询。策略提取两种方式：

1. **AWR**（默认）：优势加权回归  
   $\mathcal{L}_\pi = -\mathbb{E}[\exp((Q-V)/\beta) \log \pi(a|s)]$
2. **DDPG+BC**：最大化 Q + BC 正则  
   $\mathcal{L}_\pi = -Q(s, \pi(s))/\|Q\| + \alpha \log \pi(a_{\text{data}}|s)$

#### 训练 / 推理

- **Actor**：`GaussianActor`（单峰高斯）
- **训练**：critic + value + actor 联合或分模块更新
- **推理**：从高斯分布采样 `dist.sample()`

#### 特点

| 优点 | 缺点 |
|------|------|
| 经典离线 RL，理论成熟 | 高斯策略难以表达多模态行为 |
| 无需生成式去噪，推理快 | 在复杂操纵任务上常弱于 flow/diffusion |
| 实现简洁、调参文献丰富 | Actor 需 RL 梯度，训练稳定性依赖 $\beta$ |

**适用场景**：低维动作、单模态行为、需要快速推理的经典离线 RL 基线。

---

### 3.3 SAC — 软 Actor-Critic（在线 RL / RLPD）

**文件**：`agents/sac.py`

#### 原理

最大熵 RL：$\max_\pi \mathbb{E}[Q(s,a) - \alpha \log \pi(a|s)]$。支持 **RLPD**（离线数据 + 在线 replay 混合训练）。

#### 训练

$$
\mathcal{L}_\pi = \mathbb{E}[\alpha \log \pi - Q] + \mathcal{L}_\alpha + w_{\text{bc}} \|a_\pi - a_{\text{data}}\|^2
$$

Twin Q + 可学习温度 $\alpha$。

#### 特点

| 优点 | 缺点 |
|------|------|
| 在线探索能力强 | 纯离线场景非主方法 |
| RLPD 可微调离线策略 | 高斯 actor，表达力有限 |
| 熵正则提升鲁棒性 | 需环境交互 |

**适用场景**：`online_steps > 0` 时的在线微调基线，非 QGF 论文主对比对象。

---

## 4. 测试时策略改进（QGF 族）

这类方法的核心思想：**训练阶段 actor 只做 BC + IQL；策略改进发生在推理阶段**。

### 4.1 QGF — Q-Guided Flow（论文主方法）

**文件**：`agents/qgf.py`  
**论文**：[Q-Guided Flow](https://arxiv.org/pdf/2606.11087)

#### 原理

在 Euler 去噪的每一步，用 Q 函数梯度修正 BC 速度场，类比扩散模型的 classifier guidance：

$$
v_{\text{guided}} = v_{\text{BC}} + w \cdot \nabla_{a_t} Q(s, \hat{a})
$$

其中 $\hat{a}$ 是「干净动作」的单步 Euler 近似：

$$
\hat{a} = \mathrm{clip}\big(a_t + (1-t)\,v_{\text{BC}}(s,a_t,t),\,-1,\,1\big)
$$

**关键设计**：在 $\hat{a}$（近干净动作）上求 $\nabla Q$，而非在 OOD 噪声 $a_t$ 上；默认**丢弃 Jacobian** $\partial \hat{a}/\partial a_t$（设为 I），降低方差。

#### 训练

三个模块**独立**更新（policy 不接收 Q 梯度）：

1. `policy_loss` → Flow Matching BC
2. `critic_loss` → IQL TD
3. `value_loss` → Expectile 回归

#### 推理超参

- `guidance_weight` $w$：引导强度，验证集扫描
- `rejection_sampling` $K$：Best-of-N，选 Q 最大的样本
- `denoised_action_approx` / `apply_jacobian`：切换变体（见 §4.5）

#### 特点

| 优点 | 缺点 |
|------|------|
| 训练稳定（BC + IQL 解耦） | 推理每步需 `jax.grad(Q)`，比纯 BC 慢 |
| 同一 checkpoint 可扫 $w$ 做 test-time scaling | $w$ 需验证集调参 |
| 避免 BPTT 与 OOD Q 查询 | 丢弃 Jacobian 是对链式法则的有偏近似 |
| `support_guidance = True`，评估 API 完善 | — |

**适用场景**：已有高质量 BC flow + IQL；希望在**不改训练**的前提下提升策略。

---

### 4.2 QFQL — 在噪声动作上求 Q（QGF 有偏基线）

**实现**：`QGFAgent` + `denoised_action_approx=noisy`

#### 原理

$\hat{a} = a_t$，直接在噪声流形上求 $\nabla_{a_t} Q(s, a_t)$。

#### 特点

| 优点 | 缺点 |
|------|------|
| 实现最简单，无需近似干净动作 | Q 在 OOD 噪声动作上评估 → **有偏、不稳定** |
| 与 DQL/QFQL 类方法对齐 | 实验上通常弱于 QGF 默认配置 |

---

### 4.3 QGF-Jacobian — 完整链式法则变体

**实现**：`QGFAgent` + `apply_jacobian=True`

#### 原理

对 map $f(a_t) = \mathrm{clip}(a_t + (1-t)v(a_t), -1, 1)$ 计算 Jacobian，链式法则：

$$
\nabla_{a_t} Q \leftarrow J^\top \nabla_{\hat{a}} Q
$$

#### 特点

| 优点 | 缺点 |
|------|------|
| 链式法则更完整 | 方差更高，每步需 `jacrev` |
| 理论上梯度更准确 | 计算开销显著增加 |

---

### 4.4 BPTT — 穿过去噪链反传（DQL 风格基线）

**文件**：`agents/bptt.py`（继承 `QGFAgent`，仅改 `sample_actions`）

#### 原理

每步从 $a_t$ 运行**完整** BC 去噪链得到 $a_{\text{clean}} = \text{ODE}(a_t)$，再对 $Q(s, a_{\text{clean}})$ 关于 $a_t$ 求梯度。JAX 自动 BPTT 穿过去噪链。

#### 特点

| 优点 | 缺点 |
|------|------|
| 梯度理论上最「完整」 | 每步 BPTT 穿 $N$ 步去噪 → **极慢** |
| 与 DQL 类方法可比 | 长链反传方差大、易不稳定 |
| 训练同 QGF（BC+IQL） | 未注册到 `agents` 字典，需手动指定 |

---

### 4.5 GradStep — 去噪后梯度上升

**文件**：`agents/grad_step.py`（继承 `QGFAgent`）

#### 原理

两阶段推理：

1. **阶段 A**：完整 BC Euler 去噪 → $a_1$
2. **阶段 B**：在干净动作空间做 $L$ 步梯度上升  
   $a \leftarrow \mathrm{clip}(a + \alpha \nabla_a Q(s,a), -1, 1)$

Q 梯度**不进入**去噪内环。

#### 特点

| 优点 | 缺点 |
|------|------|
| Q 在干净动作上评估，OOD 风险低 | 引导信号与去噪过程解耦 |
| 实现简单，继承 QGF 训练 | 无法在去噪路径上逐步纠偏 |
| 可选 `use_sign_gradient` | 超参：`qgrad_step_size`, `qgrad_steps` |

**与 QGF 对比**：

| | QGF | GradStep |
|---|-----|----------|
| Q 梯度时机 | 每 denoise 步 | 仅去噪后 |
| 计算 | $N \times (\text{policy} + \text{grad } Q)$ | $N \times \text{policy} + L \times \text{grad } Q$ |

---

### 4.6 RobustQ — 噪声条件 Q 网络

**文件**：`agents/robust_q.py`（继承 `QGFAgent`）

#### 原理

额外训练 $Q_{\text{robust}}(s, a_t, t)$，在整条 flow 路径上回归干净 IQL 的 $Q(s, a_1)$：

$$
\mathcal{L}_{\text{robust}} = \mathbb{E}\big[(Q_{\text{robust}}(s, a_t, t) - \mathrm{sg}(Q(s,a_1)))^2\big]
$$

推理时每步：

$$
v = v_{\text{BC}} + w \cdot \nabla_{a_t} Q_{\text{robust}}(s, a_t, t)
$$

#### 特点

| 优点 | 缺点 |
|------|------|
| Q 梯度在**各 $t$ 的分布内**，无需 Euler 近似 | 需额外训练 `robust_critic` |
| 无需 Jacobian 修正 | 实现中 `sample_actions` 使用未定义变量 `cfg`（已知 bug） |
| 训练仍共享 BC+IQL 底座 | `support_guidance` 未设为 True |

---

### 4.7 CFGRL — Classifier-Free Guidance RL

**文件**：`agents/cfgrl.py`  
**论文**：Frans et al., 2025

#### 原理

训练**条件流** $v(s, a_t, t, o)$，$o \in \{0,1\}$ 表示 advantage 正负：

- 无条件分支：标准 BC flow
- 条件分支：仅对 $Q - V > \text{threshold}$ 的样本加权 flow matching

推理时 CFG 式插值：

$$
v = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})
$$

每步需两次 actor 前向（$o=0$ 和 $o=1$）。

#### 特点

| 优点 | 缺点 |
|------|------|
| 引导信号内嵌于网络，非显式 Q 梯度 | Actor 训练含 advantage 加权，比 QGF 复杂 |
| 与扩散 CFG 概念对齐，直觉清晰 | 每步 2× actor 前向 |
| `support_guidance = True` | 依赖 advantage 阈值 `adv_threshold` |

**与 QGF 对比**：QGF 用**外部** IQL Q 梯度；CFGRL 用**内部**条件速度场差分。

---

### 4.8 IFQL — 隐式 Flow Q-Learning（Best-of-N）

**文件**：`agents/ifql.py`  
**论文**：IDQL 的 flow 变体（Hansen-Estruch et al., 2023）

#### 原理

- **训练**：BC flow + IQL（actor 无 RL 梯度）
- **推理**：从 BC flow 采样 $N$ 个候选动作，选 Q 最高者（implicit policy extraction）

#### 特点

| 优点 | 缺点 |
|------|------|
| 无 Q 梯度引导，实现稳健 | 推理时间 × $N$（`num_samples`） |
| 避免 OOD Q 梯度 | 无连续引导，可能不如 QGF 精细 |
| 与 IDQL 思想一致 | 需调 `num_samples` |

**适用场景**：希望测试时改进但**不愿**在每步求 Q 梯度的场景。

---

## 5. 训练时 Q 感知流策略

这类方法在**训练阶段**将 Q 信号注入 actor，推理时通常为标准去噪（或一步采样）。

### 5.1 EDP — Efficient Diffusion Policy

**文件**：`agents/edp.py`  
**论文**：Kang et al., 2023

#### 原理

训练时用单步 Euler 近似最大化 Q，**梯度穿过** $v_\theta$：

$$
a_{\text{eval}} = \mathrm{clip}(a_t + (1-t)v_\theta, -1, 1)
$$
$$
\mathcal{L}_\pi = -Q(s, a_{\text{eval}}) + \lambda_{\text{bc}} \mathcal{L}_{\text{BC}}
$$

#### 推理

标准 BC Euler 去噪（Q 最大化已「烘焙」进 policy 权重）。

#### 特点

| 优点 | 缺点 |
|------|------|
| 推理与 BC 同速（无测试时 Q grad） | `bc_weight` 与 Q 项需精细平衡 |
| 单步近似避免完整 BPTT | Q 梯度反传到 policy，训练可能不稳定 |
| 与 QGF 共享 $a_{\text{eval}}$ 形式 | 不如 QGF 灵活（无法扫 $w$） |

**与 QGF**：EDP 在**训练**用 Q；QGF 在**推理**用 Q。EDP 对 policy 反传 Q 梯度。

---

### 5.2 FQL — Flow Q-Learning

**文件**：`agents/fql.py`  
**论文**：Park et al., 2025

#### 原理

蒸馏多步 BC 流为**一步**策略 $\Omega(s, z)$，并最大化 Q：

$$
\mathcal{L} = \mathcal{L}_{\text{bc\_flow}} + \alpha \mathcal{L}_{\text{distill}} - \mathbb{E}[Q(s, \Omega(s,z))]
$$

Teacher：`actor_bc_flow`（多步积分）；Student：`actor_onestep_flow`（单前向）。

#### 推理

**一步采样**：$a = \Omega(s, \xi), \xi \sim \mathcal{N}$ — 无 denoise 循环。

#### 特点

| 优点 | 缺点 |
|------|------|
| **推理最快**（单前向） | 需维护 teacher + student 两套网络 |
| 完全避免 BPTT | 蒸馏质量影响最终性能 |
| Critic 可选 DDPG 或 IQL | $\alpha$ 调参敏感 |

**适用场景**：生产部署需低延迟推理；可接受较长训练调参。

---

### 5.3 QAM — Q-learning with Adjoint Matching

**文件**：`agents/qam.py`  
**论文**：Li et al., 2026

#### 原理

目标策略：$\pi^*(a|s) \propto \pi_{\text{BC}}(a|s) \exp(\tau Q(s,a))$。

通过**伴随匹配**将 Q 梯度注入 flow 训练，**无需 BPTT**：

1. 沿**固定** BC 流（`actor_slow`）前向积分，得路径 $\{x_t\}$
2. 在终点初始化伴随状态 $g_T = -\tau \nabla_a Q(s, x_T)$
3. 反向传播伴随 ODE 得各时刻 $g_t$
4. 训练 `actor_fast` 匹配：$v_{\text{fast}} \approx v_{\text{base}} + \sigma_t^2 g_t / 2$

网络结构：`actor_slow`（BC）+ `actor_fast`（Q 感知修正），可选 residual、FQL 蒸馏、edit actor。

#### 特点

| 优点 | 缺点 |
|------|------|
| 无 BPTT 的 Q 感知流训练，理论优雅 | 实现最复杂（slow/fast、伴随 ODE） |
| 避免在 evolving policy 上反传（病态） | 超参多：`inv_temp`, `flow_steps`, `residual` 等 |
| 可选集成 FQL 一步蒸馏 | 训练计算量高于 EDP/FQL |
| SOTA 潜力的训练时方法 | — |

**适用场景**：追求训练时 Q 感知、不愿接受测试时额外开销的研究/生产场景。

---

### 5.4 FBRAC — Flow BRAC（BPTT Q 最大化）

**文件**：`agents/fbrac.py`

#### 原理

BC flow + 通过 **Euler 积分 BPTT** 最大化 Q：

$$
\mathcal{L}_\pi = \mathcal{L}_{\text{BC}} + \mathcal{L}_{Q\text{-max}} / \alpha
$$

$\mathcal{L}_{Q\text{-max}}$ 对完整去噪链反传。

#### 特点

| 优点 | 缺点 |
|------|------|
| 训练时直接 Q 最大化 | BPTT 训练不稳定、慢 |
| 与 BRAC 思想结合 flow | 方差大，实践中常不如 EDP/QAM |

---

### 5.5 FAWAC — Flow AWAC

**文件**：`agents/fawac.py`

#### 原理

AWAC 的 flow 变体：flow matching 损失按 $\exp(\text{advantage} \cdot \text{inv\_temp})$ 加权：

$$
\mathcal{L}_\pi = \mathbb{E}\big[e^{(Q-V)/\beta} \cdot \|v_\theta - v_{\text{target}}\|^2\big]
$$

Critic 用 DDPG 式 bootstrap + value 回归。

#### 特点

| 优点 | 缺点 |
|------|------|
| 优势加权 BC，简单直观 | 仅 re-weight BC，无显式 Q 梯度注入 |
| 无需测试时引导 | 高 advantage 样本稀少时有效 batch 小 |
| 推理为标准 flow 去噪 | 对极端优势样本敏感 |

---

### 5.6 CFGRL（训练视角）

见 §4.7。从训练角度，CFGRL 也属于「训练时改进 actor」：advantage 加权条件 flow + 无条件 flow 联合训练。

---

## 6. 扩散与其他变体

### 6.1 DCGQL — Diffusion Classifier-Guidance Q-Learning

**文件**：`agents/dcgql.py`

#### 原理

统一两种扩散 RL 变体（`actor_loss_type` 切换）：

| 模式 | 思想 |
|------|------|
| **QSM** | Actor 损失匹配 Q 梯度到预测噪声（score matching） |
| **DAC** | Q 梯度与预测噪声的加权点积，按噪声水平缩放 |

使用 DDPM 扩散策略 + IQL critic/value。

#### 特点

| 优点 | 缺点 |
|------|------|
| 涵盖 QSM/DAC 两种 SOTA 扩散 RL | DDPM 推理步数多，慢于 flow |
| 统一实现便于对比 | 扩散训练调参复杂 |
| IQL 价值学习稳定 | 与 QGF flow 路线不同 |

---

### 6.2 IQL-Diffusion — IQL + 扩散 Actor

**文件**：`agents/iql_diffusion.py`（继承 `IQLAgent`）

#### 原理

将 IQL 的 AWR 应用于扩散策略：用 flow matching 损失近似 $\log \pi(a|s)$，按 $\exp(\text{adv}/\beta)$ 加权：

$$
\mathcal{L}_\pi = \mathbb{E}\big[e^{(Q-V)/\beta} \cdot \|v_\theta - v_{\text{target}}\|^2\big]
$$

#### 特点

| 优点 | 缺点 |
|------|------|
| 多模态扩散 actor + IQL 理论 | 扩散推理慢 |
| 优势加权，避免纯 BC | 与 QAM/EDP 比表达力/效率需实测 |

---

### 6.3 DSRL — Diffusion SAC RL

**文件**：`agents/dsrl.py`  
**论文**：https://arxiv.org/abs/2506.15799

#### 原理

结合扩散/flow 与 SAC 风格 actor-critic：

- `actor_bc_flow`：BC flow teacher
- `actor`：高斯 latent noise policy
- `z_critic`：在 latent noise 上蒸馏 $Q(s, a)$

Actor 损失：BC flow + 最大熵 RL on latent space。

#### 特点

| 优点 | 缺点 |
|------|------|
| Latent space RL，避免动作空间直接 max Q | 网络组件多（flow + gaussian + z_critic） |
| 结合探索（熵正则） | 调参复杂 |
| 可一步或 flow 采样 | 论文较新，生态不如 FQL/QAM |

---

## 7. 全面对比表

### 7.1 训练机制对比

| Agent | Actor 类型 | Actor RL 梯度 | Critic | 额外表 |
|-------|-----------|--------------|--------|--------|
| **BC** | Flow | 无 | 无 | — |
| **IQL** | Gaussian | AWR/DDPG+BC | IQL Twin Q | — |
| **SAC** | Gaussian | Max-entropy | Twin Q | $\alpha$ |
| **QGF** | Flow | **无**（仅 BC） | IQL | — |
| **GradStep** | Flow | 无 | IQL | — |
| **RobustQ** | Flow | 无 | IQL | `robust_critic` |
| **BPTT** | Flow | 无 | IQL | — |
| **IFQL** | Flow | 无 | IQL | — |
| **CFGRL** | 条件 Flow | Advantage 加权 BC | IQL | — |
| **EDP** | Flow | 单步 Q max | IQL | — |
| **FQL** | 一步流 + BC 流 | max Q + 蒸馏 | DDPG/IQL | teacher flow |
| **QAM** | Slow+Fast Flow | Adjoint matching | DDPG/IQL | fast/slow actor |
| **FBRAC** | Flow | BPTT Q max | DDPG | — |
| **FAWAC** | Flow | Advantage 加权 BC | DDPG + V | — |
| **DCGQL** | DDPM 扩散 | QSM/DAC | IQL | — |
| **IQL-Diffusion** | Flow/Diffusion | AWR 加权 | IQL | — |
| **DSRL** | Flow + Gaussian | SAC on latent | DDPG | `z_critic` |

### 7.2 推理机制对比

| Agent | 推理方式 | 测试时超参 | 每步额外开销 |
|-------|---------|-----------|-------------|
| **BC** | $N$ 步 Euler | — | 0 |
| **IQL** | 高斯采样 | — | 0 |
| **QGF** | $N$ 步 Euler + Q 引导 | `guidance_weight`, `rejection_sampling` | `grad Q` |
| **QFQL** | 同上（噪声 Q） | 同上 | `grad Q` |
| **QGF-J** | 同上 + Jacobian | 同上 | `grad Q` + `jacrev` |
| **BPTT** | 每步 BPTT + Q 引导 | `guidance_weight` | BPTT × `grad Q` |
| **GradStep** | BC 去噪 → Q refine | `qgrad_steps`, `qgrad_step_size` | $L \times$ `grad Q` |
| **RobustQ** | Euler + $Q_{\text{robust}}$ 引导 | `cfg`（待修复） | `grad Q_robust` |
| **CFGRL** | CFG 式双前向 | `guidance_weight` | 2× actor |
| **IFQL** | BC 采样 × N，选 max Q | `num_samples` | $N \times$ 去噪 |
| **EDP** | 标准 Euler | — | 0 |
| **FQL** | **一步**前向 | — | 0 |
| **QAM** | 多步/一步（配置） | — | 0 |
| **FBRAC/FAWAC** | 标准 Euler | — | 0 |
| **DCGQL** | DDPM 采样 | — | 扩散步数 |
| **DSRL** | Flow 或 latent 采样 | — | 0~N |

### 7.3 优缺点一句话总结

| Agent | 最大优势 | 主要短板 |
|-------|---------|---------|
| **BC** | 最稳基线 | 无 RL 提升 |
| **IQL** | 经典、快 | 单峰高斯 |
| **QGF** | 训练稳 + 测试时 scaling | 推理慢 + 调 $w$ |
| **GradStep** | Q 在干净空间 | 引导与去噪解耦 |
| **RobustQ** | 路径内 Q 梯度 | 额外网络 + bug |
| **CFGRL** | 引导内嵌网络 | 训练/推理均 2× 前向 |
| **IFQL** | 无 Q 梯度引导 | 推理 × N |
| **EDP** | 训练时 Q、推理快 | 训练平衡难 |
| **FQL** | **推理最快** | 蒸馏 + 调参 |
| **QAM** | 无 BPTT 的 SOTA 训练 | 最复杂 |
| **FBRAC** | 直接 Q max | BPTT 不稳定 |
| **FAWAC** | 简单优势加权 | 提升有限 |
| **DCGQL** | 扩散 RL 统一 | 推理慢 |
| **DSRL** | Latent SAC | 组件多 |

---

## 8. 选型决策树

```
你的首要目标是什么？
│
├─ 不改训练、同一 checkpoint 扫参提升？
│   └─ QGF（默认）/ CFGRL / IFQL（Best-of-N）
│
├─ 推理延迟最重要（单步）？
│   └─ FQL
│
├─ 训练时 Q 感知、推理无额外开销？
│   ├─ 愿接受复杂实现 → QAM
│   ├─ 简单直接 → EDP
│   └─ 优势加权即可 → FAWAC / CFGRL
│
├─ 纯模仿 / pipeline 验证？
│   └─ BC
│
├─ 经典离线 RL、低维动作？
│   └─ IQL
│
├─ 在线微调？
│   └─ SAC（RLPD）
│
└─ 扩散策略研究？
    └─ DCGQL / IQL-Diffusion / DSRL
```

### 8.1 QGF 族内部选择

```
需要在去噪每步引导？
├─ 是 → QGF（单步 Euler 近似）> RobustQ > QFQL（有偏基线）
└─ 否 → GradStep（去噪后 refine）或 IFQL（Best-of-N）

愿为更准确梯度付计算代价？
├─ 是 → QGF-Jacobian 或 BPTT
└─ 否 → QGF 默认（丢弃 Jacobian）

引导信号来源？
├─ 外部 Q 梯度 → QGF / GradStep / RobustQ
└─ 网络内条件场 → CFGRL
```

---

## 9. 核心公式速查

| 方法 | Actor / 推理目标 |
|------|-----------------|
| BC / QGF train | $\|v - (a_1-a_0)\|^2$ |
| QGF infer | $v = v_{\text{BC}} + w \nabla Q(s, \hat{a})$ |
| EDP | $\lambda \|v-\cdot\|^2 - Q(s, x_t+(1-t)v)$ |
| FQL | $\mathcal{L}_{\text{bc}} + \alpha \mathcal{L}_{\text{distill}} - Q(s, \pi_{\text{1step}})$ |
| QAM | $\mathcal{L}_{\text{BC}} + \mathcal{L}_{\text{adj match}}(g_t)$ |
| IQL AWR | $-\mathbb{E}[e^{(Q-V)/\beta} \log \pi(a|s)]$ |
| CFGRL infer | $v = v_u + w(v_c - v_u)$ |
| GradStep | $a_1 = \text{BC-ODE}(a_0);\; a \leftarrow a + \alpha \nabla Q$ |
| IFQL infer | $a = \arg\max_{a_i \sim \pi_{\text{BC}}} Q(s, a_i)$ |

---

## 10. 代码入口

| Agent | 加载方式 | `support_guidance` |
|-------|---------|-------------------|
| BC | `--agent=agents/bc.py` | False |
| IQL | `--agent=agents/iql.py` | False |
| QGF | `--agent=agents/qgf.py` | **True** |
| GradStep | `--agent=agents/grad_step.py` | False |
| RobustQ | `--agent=agents/robust_q.py` | False |
| CFGRL | `--agent=agents/cfgrl.py` | **True** |
| IFQL | `--agent=agents/ifql.py` | False |
| EDP | `--agent=agents/edp.py` | False |
| FQL | `--agent=agents/fql.py` | False |
| QAM | `--agent=agents/qam.py` | False |
| FBRAC | `--agent=agents/fbrac.py` | False |
| FAWAC | `--agent=agents/fawac.py` | False |
| DCGQL | `--agent=agents/dcgql.py` | False |
| IQL-Diffusion | `--agent=agents/iql_diffusion.py` | False |
| DSRL | `--agent=agents/dsrl.py` | False |
| SAC | `--agent=agents/sac.py` | False |
| BPTT | `--agent=agents/bptt.py` | True（继承 QGF） |

注册表见 `agents/__init__.py` 的 `agents` 字典（BPTT 未注册，需直接指定路径）。

---

## 11. 延伸阅读

| 主题 | 文档 |
|------|------|
| 全局架构 | [README.md](./README.md) |
| Flow Matching 数学 | [02-flow-matching-bc.md](./02-flow-matching-bc.md) |
| IQL Critic/Value | [03-iql-critic-value.md](./03-iql-critic-value.md) |
| QGF 实现细节 | [04-qgf-core.md](./04-qgf-core.md) |
| QGF 推理变体 | [05-inference-guidance-variants.md](./05-inference-guidance-variants.md) |
| 测试时基线 | [06-test-time-baselines.md](./06-test-time-baselines.md) |
| 训练时方法 | [07-train-time-methods.md](./07-train-time-methods.md) |
| 网络结构 | [08-networks-and-modules.md](./08-networks-and-modules.md) |
| 数据与评估 | [09-data-training-evaluation.md](./09-data-training-evaluation.md) |
