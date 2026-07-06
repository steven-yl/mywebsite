---
title: "Formula"
subtitle: ""
date: 2026-03-04T12:22:25+08:00
# lastmod: 
draft: false
authors: [Steven]
description: ""
tags: [CS492D]
categories: [docs CS492D]
series: [CS492D系列]
---

---
### Sampling
概率密度函数(连续)PDF、概率质量函数(离散)PMF、累计分布函数CDF
$$
\begin{array}{cc}
\text{PDF满足}:\displaystyle\sum_{i = 1}^{n} p_i = 1 &
\text{CDF}:P_i = \displaystyle\sum_{j=1}^{i} p_j
\end{array}
$$


#### Inverse Transform Sampling
- CDF 需要可逆
- 从均匀分布中采样$u \sim \mathcal{U}(0, 1)$。
- 选择$x_i$，使得$P_{i-1} \leq u \leq P_i$。
- 若已知 CDF 的逆函数$F_X^{-1}$，则：

$$
x = F_X^{-1}(u)
$$


##### 示例

PDF：
$$
p(x) = \frac{3}{8} x^2 , \quad x \in [0, 2]
$$

CDF：
$$
F_X(x) = \int_0^x p(x) \, dx = \frac{1}{8} x^3
$$

CDF 的逆函数：
$$
F_X^{-1}(x) = 2\sqrt[3]{x}
$$

采样步骤：
1. 采样$u \sim \mathcal{U}(0, 1)$。
2. 计算$x = 2\sqrt[3]{u}$。

#### Rejection Sampling


---
### Reparameterization

---
### VAE
#### ELBO

---
### DDPM
#### VAE2DDPM
---
### DDIM

---
### CG\CFG\ControlNet\LoRA

---
### Zero Shot
#### SDEdit
#### RePaint

---
### DDIM Inversion
#### Null-Text Inversion

---
### Score Distillation
#### SDS
#### VSD
#### SDI
#### DDS
#### Instruct-NeRF2NeRF
#### PDS

---
### SyncDiffusion

---
### InversionProblam


---
### ODE


---
### DPM-Solver


---
### Flow Matching