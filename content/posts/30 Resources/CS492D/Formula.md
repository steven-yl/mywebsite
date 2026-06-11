---
title: "Formula"
subtitle: ""
date: 2026-03-04T12:22:25+08:00
# lastmod: 
draft: false
authors: [Steven]
description: ""
tags: [CS492D]
categories: [CS492D]
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
**特殊设定**：
马尔可夫分层 VAE 的一个特例：
- 隐变量维度与数据维度相同
- 变分后验$q_{\phi}(x_{t+1}|x_t)$不是学习的，而是预定义的：
 $$
  q_{\phi}(x_{t+1}|x_t) \rightarrow q(x_{t+1}|x_t)
 $$
---
$$
\boxed{
\begin{aligned}
&-\log p(x_0) \\
&\qquad \leq \mathbb{E}_{q(x_{1:T}|x_0)}\left[-\log\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}\right] 
&&\text{(变分下界)}\\[6pt]
&\qquad = \mathbb{E}_{q(x_{1:T}|x_0)}\left[ -\log\left(p(x_T)\prod_{t=1}^{T} p_{\theta}(x_{t-1}|x_t)\right) + \log\left(\prod_{t=1}^{T} q(x_t|x_{t-1})\right) \right] 
&&\text{(代入 }p_{\theta}(x_{0:T})=p(x_T)\prod p_{\theta},\; q(x_{1:T}|x_0)=\prod q\text{)}\\[6pt]
&\qquad = \mathbb{E}_{q(x_{1:T}|x_0)}\left[ -\log p(x_T) - \sum_{t=1}^{T}\log p_{\theta}(x_{t-1}|x_t) + \sum_{t=1}^{T}\log q(x_t|x_{t-1}) \right] 
&&\text{(对数乘积化为求和)}\\[6pt]
&\qquad = \mathbb{E}_{q}\left[-\log p(x_T) - \log p_{\theta}(x_0|x_1) - \sum_{t=2}^{T}\log p_{\theta}(x_{t-1}|x_t) + \log q(x_1|x_0) + \sum_{t=2}^{T}\log q(x_t|x_{t-1})\right] 
&&\text{(分离 }t=1\text{ 项)}\\[6pt]
&\qquad = \mathbb{E}_{q}\Bigg[-\log p(x_T) - \log p_{\theta}(x_0|x_1) - \sum_{t=2}^{T}\log p_{\theta}(x_{t-1}|x_t) + \log q(x_1|x_0) \\
&\qquad\qquad + \sum_{t=2}^{T}\Bigl(\log q(x_{t-1}|x_t,x_0) - \log q(x_{t-1}|x_0) + \log q(x_t|x_0)\Bigr)\Bigg] 
&&\text{(对 }t\ge2\text{ 用贝叶斯：}q(x_t|x_{t-1})=\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}\text{)}\\[6pt]
&\qquad = \mathbb{E}_{q}\Bigg[-\log p_{\theta}(x_0|x_1) + \sum_{t=2}^{T}\Bigl(\log q(x_{t-1}|x_t,x_0) - \log p_{\theta}(x_{t-1}|x_t)\Bigr) \\
&\qquad\qquad + \Bigl(-\log p(x_T) + \log q(x_T|x_0)\Bigr) + \Bigl(\log q(x_1|x_0) - \log q(x_1|x_0)\Bigr)\Bigg] 
&&\text{(重组，telescoping 和消去 }+\log q(x_1|x_0)-\log q(x_1|x_0)\text{)}\\[6pt]
&\qquad = \underbrace{-\mathbb{E}_{q(x_1|x_0)}\bigl[\log p_{\theta}(x_0|x_1)\bigr]}_{\text{重构损失：从 }x_1\text{ 重建 }x_0\text{ 的负对数似然}}
+ \underbrace{D_{KL}\bigl(q(x_T|x_0)\,\|\,p(x_T)\bigr)}_{\text{先验匹配项：最终潜变量分布与先验的 KL 散度}}
+ \underbrace{\sum_{t=2}^{T}\mathbb{E}_{q(x_t|x_0)}\Bigl[ D_{KL}\bigl(q(x_{t-1}|x_t,x_0)\,\|\,p_{\theta}(x_{t-1}|x_t)\bigr) \Bigr]}_{\text{去噪匹配项：每个时间步真实后验与模型逆向分布的 KL 散度期望}}
\end{aligned}
}
$$

#### $u、x_0、\varepsilon_t$
#### Connection to Score-Based Models (SMLD)

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