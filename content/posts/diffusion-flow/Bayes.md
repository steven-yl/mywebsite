---
title: "Bayes：先验与后验"
subtitle: ""
date: 2026-03-27T00:00:00+08:00
draft: true
authors: [Steven]
description: "用直观方式解释贝叶斯法则中先验、似然、后验与边际概率的关系。"
summary: "通过公式和硬币示例理解先验分布与后验分布的区别，以及贝叶斯更新流程。"

tags: [diffusion/flow, Bayes, 统计]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 10
series_weight: 10

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

### 贝叶斯法则

**先验 = 在没看到数据前，对参数的看法；后验 = 看到数据后，用贝叶斯更新后的看法。**

---
### 贝叶斯里的角色

- **参数 / 未知量**：记成 $\theta$（例如真实概率、均值、模型参数）。
- **观测数据**：记成 $y$ 或 $D$。
- 贝叶斯公式（密度版）：
  $$
  \underbrace{p(\theta \mid y)}_{\text{后验}}
  = \frac{\overbrace{p(y \mid \theta)}^{\text{似然}}\,
        \overbrace{p(\theta)}^{\text{先验}}}{\underbrace{p(y)}_{\text{边际}}}
  $$

---

### 先验分布 $p(\theta)$

- **是什么**：在**还没看到数据 $y$** 时，对 $\theta$ 的分布假设。
- **怎么理解**：基于经验、文献、问题背景给出的“$\theta$ 可能长什么样”的量化。
- **例子**：估计一枚硬币正面概率 $\theta$，若觉得“大概在 0.5 附近”，可用先验 $\theta \sim \text{Beta}(2,2)$。

---

### 后验分布 $p(\theta \mid y)$

- **是什么**：**在已经看到数据 $y$ 之后**，对 $\theta$ 的更新后的分布。
- **怎么理解**：用“数据 $y$”通过贝叶斯公式，把先验 $p(\theta)$ 更新成后验 $p(\theta\mid y)$；既包含先验信息，也包含数据信息。
- **例子**：抛 10 次出现 7 次正面，后验会在 0.7 附近更集中，同时仍受先验影响。

---

### 一句话对照

| 概念 | 何时 | 含义 |
|------|------|------|
| **先验** $p(\theta)$ | 观测**之前** | 对 $\theta$ 的初始信念（分布） |
| **后验** $p(\theta \mid y)$ | 观测**之后** | 用数据 $y$ 更新后的信念（分布） |

**记法**：先验 = prior（先于数据）；后验 = posterior（在数据之后）。  
**流程**：先验 × 似然（数据在给定 $\theta$ 下的分布）→ 归一化 → 得到后验。