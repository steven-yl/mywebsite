---
title: "Consistency Model"
subtitle: ""
date: 2026-02-27T00:00:00+08:00
# lastmod: 2026-02-27T00:00:00+08:00
draft: false
authors: [Steven]
description: "Consistency Model 技术文档：自洽性、一致性蒸馏与一致性训练、一步采样与多步采样、与扩散模型及 MeanFlow 的对比。"

tags: [diffusion/flow, Consistency Model]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 5
series_weight: 5

hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: "/mywebsite/posts/images/consistency-model.webp"
---

Consistency Model 技术文档（原理 + 自洽性 + 蒸馏/直接训练 + 采样 + 对比）

本文档为正式技术规格文档，系统介绍 Consistency Model 的核心思想、自洽性定义、两种训练范式（Consistency Distillation / Consistency Training）、一步与多步采样，以及与扩散模型、MeanFlow 的对比。

## 1 文档基本信息

- **算法名称**：Consistency Model（一致性模型）
- **提出者**：Song Yang, Prafulla Dhariwal 等（OpenAI / 独立研究）
- **发表时间**：2023 年（ICML 2023）
- **核心定位**：一步或极少步采样的生成模型
- **基础依托**：扩散模型轨迹上的自洽性（self-consistency）
- **核心创新**：从学习瞬时速度/得分改为学习「轨迹上任意点到数据」的一致性映射 $f_\theta(x_t, t) \to x_1$，实现 1-NFE 或少量 NFE 采样

## 2 背景与动机

### 2.1 多步扩散的瓶颈

- 扩散模型与流匹配通过 ODE/SDE 从噪声 $x_0$ 生成数据 $x_1$，需对向量场或得分进行多步数值积分（常需 20～50+ 步），推理延迟高。
- 步数减少会明显损害样本质量，难以在保持质量的前提下做到真正的一步生成。

### 2.2 Consistency Model 的解决思路

不直接学瞬时速度 $v_t$ 或得分 $\nabla \log p_t$，而是学一个**一致性函数** $f_\theta$：将轨迹上任意点 $(x_t, t)$ 映射到**同一轨迹的终点** $x_1$。若该映射在整条轨迹上「自洽」，则从噪声 $x_0$ 出发只需一次前向即可得到 $x_1 = f_\theta(x_0, 0)$，实现一步生成。

## 3 核心概念：自洽性（Self-Consistency）

### 3.1 轨迹与终点

- 设数据 $x_1 \sim p_{\text{data}}$，噪声 $x_0 \sim p_{\text{noise}}$（如 $\mathcal{N}(0, I)$）。扩散或流匹配定义了一条从 $x_0$ 到 $x_1$ 的概率轨迹 $t \mapsto x_t$（$t \in [0, 1]$ 或 $[0, T]$）。
- **一致性函数** $f_\theta(x, t)$：输入当前状态 $x$ 与时间 $t$，输出该轨迹应到达的**终点**（即数据端样本）。理想情况下，对同一轨迹上的任意 $(x_t, t)$，有 $f_\theta(x_t, t) = x_1$。

### 3.2 自洽性定义

**自洽性**：轨迹上任意两个时刻 $t, t'$ 的状态 $x_t, x_{t'}$ 经 $f_\theta$ 映射到同一终点，即

$$
f_\theta(x_t, t) = f_\theta(x_{t'}, t') = x_1.
$$

因此，模型学的是「从轨迹上任意点指向终点」的映射，并约束其在同一条轨迹上**一致**。

### 3.3 边界条件

为与生成过程兼容，通常要求：

- $f_\theta(x_1, 1) = x_1$（在 $t=1$ 时已是数据，终点即自身）；
- $f_\theta(x_0, 0) = x_1$（从噪声 $x_0$ 一步得到数据 $x_1$）。

因此**一步采样**形式为：$x_1 = f_\theta(x_0, 0)$，$x_0 \sim p_{\text{noise}}$。

## 4 数学形式化

### 4.1 符号约定

- $x_t \in \mathbb{R}^d$：时刻 $t$ 的状态（$t \in [\varepsilon, 1]$ 或 $[0, T]$，$\varepsilon$ 为小正数避免奇异性）。
- $x_1 \sim p_{\text{data}}$：数据分布。
- $x_0$ 或 $x_\varepsilon$：噪声端（如高斯）。
- $f_\theta(x, t)$：一致性模型，输入 $(x, t)$，输出与 $x$ 同维的向量，表示「从 $(x, t)$ 所在轨迹的终点」。
- 教师模型（仅蒸馏时需要）：多步扩散/流 ODE 的向量场或得分，用于从 $x_1$ 生成轨迹并给出「目标终点」估计。

### 4.2 一致性轨迹（Consistency Trajectory）

设教师（或某个前向过程）给出从 $x_1$ 到噪声的轨迹 $\{ x_t \}_{t \in [\varepsilon, 1]}$（例如 $x_t$ 为 $x_1$ 加噪后的状态）。若 $f$ 是理想的一致性函数，则对轨迹上任意两点 $(x_t, t)$ 与 $(x_{t'}, t')$，有

$$
f(x_t, t) = f(x_{t'}, t') = x_1.
$$

训练目标即让 $f_\theta$ 在数据与噪声生成的轨迹上逼近这一性质。

## 5 两种训练范式

### 5.1 Consistency Distillation（CD，一致性蒸馏）

**前提**：已有训练好的扩散模型（或概率流 ODE）作为教师。

**思路**：用教师从 $x_1$ 出发，沿 ODE 积分得到轨迹上的点 $(x_t, t)$；再用教师从 $x_t$ 积分到终点，得到「目标终点」$\hat{x}_1$（通常用多步数值积分近似）。学生 $f_\theta(x_t, t)$ 拟合该目标：

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}_{x_1, t, x_t} \left[ d\bigl( f_\theta(x_t, t),\, \hat{x}_1 \bigr) \right],
$$

其中 $d(\cdot, \cdot)$ 为距离（如 L2），$x_t$ 由教师前向过程从 $x_1$ 采样得到，$\hat{x}_1$ 由教师从 $x_t$ 反向积分得到。

**特点**：

- 依赖预训练扩散/流模型，**两阶段**（先训教师，再训一致性模型）。
- 目标 $\hat{x}_1$ 来自数值积分，存在离散化误差，理论上是**近似目标**。

### 5.2 Consistency Training（CT，一致性训练）

**前提**：无需教师，仅需数据 $x_1 \sim p_{\text{data}}$。

**思路**：用某种**前向过程**从 $x_1$ 得到 $x_t$（例如加噪 $x_t = \alpha_t x_1 + \sigma_t \epsilon$），并构造「目标终点」。一种常见做法是：对同一 $x_1$，取两个不同时间 $t > t'$，得到 $x_t, x_{t'}$，约束 $f_\theta(x_t, t)$ 与 $f_\theta(x_{t'}, t')$ 一致（都应为 $x_1$），即

$$
\mathcal{L}_{\text{CT}} = \mathbb{E}_{x_1, t, t', x_t, x_{t'}} \left[ d\bigl( f_\theta(x_t, t),\, f_\theta(x_{t'}, t') \bigr) \right],
$$

或直接约束 $f_\theta(x_t, t) \to x_1$（若 $x_1$ 可作为目标）。实际实现中常配合**课程学习**：先在小范围 $t$（接近数据）上训练，再逐步扩大到更大 $t$（接近噪声）。

**特点**：

- **单阶段**，不依赖教师。
- 目标构造和课程设计需要较多设计，理论**偏启发式**。

## 6 采样方式

### 6.1 一步采样（1-NFE）

$$
x_0 \sim p_{\text{noise}},\quad x_1 = f_\theta(x_0, 0).
$$

一次前向即得样本，延迟最低。

### 6.2 多步采样（可选）

可将区间 $[0, 1]$ 划分为 $0 = t_0 < t_1 < \cdots < t_K = 1$，从 $x_{t_0} = x_0 \sim p_{\text{noise}}$ 出发，迭代：

$$
x_{t_{k+1}} = f_\theta(x_{t_k}, t_k) + \text{（可选）修正项},
$$

或使用论文中的多步更新公式，用更多步数换取更高样本质量。

## 7 算法流程小结

### 7.1 CD 训练（简化）

1. 采样数据 $x_1 \sim p_{\text{data}}$。
2. 采样时间 $t$，用教师前向过程得到 $x_t$。
3. 用教师从 $x_t$ 积分到终点，得到目标 $\hat{x}_1$。
4. 前向 $f_\theta(x_t, t)$，最小化 $d(f_\theta(x_t, t), \hat{x}_1)$ 更新 $\theta$。

### 7.2 CT 训练（简化）

1. 采样数据 $x_1 \sim p_{\text{data}}$。
2. 采样时间 $t, t'$，用前向过程得到 $x_t, x_{t'}$。
3. 前向 $f_\theta(x_t, t)$ 与 $f_\theta(x_{t'}, t')$，最小化二者差异（或与 $x_1$ 的差异）更新 $\theta$。

### 7.3 推理（一步）

1. $x_0 \sim p_{\text{noise}}$。
2. $x_1 = f_\theta(x_0, 0)$，输出 $x_1$。

## 8 与其他模型的对比

| 维度 | 扩散 / 流匹配 | Consistency Model | MeanFlow |
|------|----------------|-------------------|----------|
| **学习目标** | 瞬时速度 $v_t$ 或得分 $\nabla \log p_t$ | 一致性映射 $f_\theta(x_t, t) \to x_1$ | 区间平均速度 $u(z_t, r, t)$ |
| **采样步数** | 多步积分（常 20+ 步） | 1 步或少量多步 | 1 步 $z_1 = z_0 + u_\theta(z_0, 0, 1)$ |
| **训练** | 单阶段，目标解析（如 flow matching） | CD 依赖教师两阶段；CT 单阶段但需课程等设计 | 单阶段，目标由恒等式解析给出 |
| **理论** | 基于 ODE/SDE、概率路径，较闭合 | 自洽性 + 蒸馏/启发式目标，非完全闭合 | 从平均速度定义严格推导，完全闭合 |

## 9 适用场景

- 需要**极低延迟**的生成（一步或少数几步）。
- 有现成扩散/流模型时可做 **CD** 压缩为一致性模型；无教师时可考虑 **CT**。
- 与 MeanFlow 相比：Consistency Model 更依赖蒸馏或启发式训练；MeanFlow 用平均速度 + 严格恒等式，单阶段、理论闭合，同样支持一步生成。

## 10 参考文献

- Song Yang, Prafulla Dhariwal, et al., **Consistency Models**, ICML 2023.
- 相关：Flow Matching、DDPM、Probability Flow ODE、MeanFlow（一步生成、平均速度场）。
