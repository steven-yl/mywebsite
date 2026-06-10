---
title: "VAE_ELBO_formula"
subtitle: "VAE_ELBO_formula"
date: 2026-06-10T12:22:25+08:00
draft: false
authors: [Steven]
tags: [CS492D]
categories: [CS492D]
---

## VAE
### ELBO
**Jensen 不等式推导 ELBO**：

$$
\boxed{
\begin{aligned}
\log p(x) 
&= \log \int p(x,z) \, dz \\
&= \log \int q_{\phi}(z|x) \, \frac{p(x,z)}{q_{\phi}(z|x)} \, dz \\
&\geq \int q_{\phi}(z|x) \, \log \frac{p(x,z)}{q_{\phi}(z|x)} \, dz \quad \text{(Jensen不等式，因log为凹函数)} \\
&= \mathbb{E}_{q_{\phi}(z|x)}\bigl[\log p(x|z)\bigr] - \mathbb{E}_{q_{\phi}(z|x)}\left[\log \frac{q_{\phi}(z|x)}{p(z)}\right] \\
&= \underbrace{\mathbb{E}_{q_{\phi}(z|x)}\bigl[\log p(x|z)\bigr]}_{\text{重建项}} \;-\; \underbrace{D_{KL}\bigl(q_{\phi}(z|x) \parallel p(z)\bigr)}_{\text{先验匹配项}} \\
&\triangleq \text{ELBO},
\end{aligned}
}
$$

等式关系由詹森差距给出：

$$
\log p(x) - \text{ELBO} = D_{KL}\bigl(q_{\phi}(z|x) \parallel p(z|x)\bigr) \geq 0.
$$

上式表明，最大化 ELBO 等价于最小化变分后验与真实后验之间的 KL 散度。

---
**条件概率恒等式推导ELBO**：
$$
\boxed{
\begin{aligned}
\log p_\phi(x) 
&\stackrel{(1)}{=} \int q(z|x) \log p_\phi(x) \, dz \\[6pt]
&\stackrel{(2)}{=} \int q(z|x) \log \frac{p_\phi(x,z)}{p_\phi(z|x)} \, dz \\[6pt]
&\stackrel{(3)}{=} \int q(z|x) \log \left( \frac{p_\phi(x,z)}{q(z|x)} \cdot \frac{q(z|x)}{p_\phi(z|x)} \right) dz \\[6pt]
&\stackrel{(4)}{=} \int q(z|x) \log \frac{p_\phi(x,z)}{q(z|x)} \, dz \;+\; \int q(z|x) \log \frac{q(z|x)}{p_\phi(z|x)} \, dz \\[6pt]
&\stackrel{(5)}{=} \underbrace{\int q(z|x) \log \frac{p_\phi(x,z)}{q(z|x)} \, dz}_{\text{ELBO}} \;+\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p_\phi(z|x)\bigr) \\[12pt]
&\stackrel{(6)}{=} \int q(z|x) \log p_\phi(x|z) \, dz \;-\; \int q(z|x) \log \frac{q(z|x)}{p(z)} \, dz \;+\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p_\phi(z|x)\bigr) \\[6pt]
&\stackrel{(7)}{=} \underbrace{\int q(z|x) \log p_\phi(x|z) \, dz \;-\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p(z)\bigr)}_{\text{ELBO 常用形式}} \;+\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p_\phi(z|x)\bigr)
\end{aligned}
}
$$
$$
\boxed{
\begin{aligned}
\log p_\phi(x) 
&\stackrel{(1)}{=} \mathbb{E}_{z \sim q(z|x)}\bigl[\log p_\phi(x)\bigr] \\[4pt]
&\stackrel{(2)}{=} \mathbb{E}_{z \sim q(z|x)}\left[\log \frac{p_\phi(x,z)}{p_\phi(z|x)}\right] \\[4pt]
&\stackrel{(3)}{=} \mathbb{E}_{z \sim q(z|x)}\left[\log \left(\frac{p_\phi(x,z)}{q(z|x)} \cdot \frac{q(z|x)}{p_\phi(z|x)}\right)\right] \\[4pt]
&\stackrel{(4)}{=} \mathbb{E}_{z \sim q(z|x)}\left[\log \frac{p_\phi(x,z)}{q(z|x)}\right] + \mathbb{E}_{z \sim q(z|x)}\left[\log \frac{q(z|x)}{p_\phi(z|x)}\right] \\[4pt]
&\stackrel{(5)}{=} \underbrace{\mathbb{E}_{z \sim q(z|x)}\left[\log \frac{p_\phi(x,z)}{q(z|x)}\right]}_{\text{ELBO}} \;+\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p_\phi(z|x)\bigr) \\[8pt]
&\stackrel{(6)}{=} \mathbb{E}_{z \sim q(z|x)}\bigl[\log p_\phi(x|z)\bigr] \;-\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p(z)\bigr) \;+\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p_\phi(z|x)\bigr) \\[4pt]
&\stackrel{(7)}{=} \underbrace{\mathbb{E}_{z \sim q(z|x)}\bigl[\log p_\phi(x|z)\bigr] \;-\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p(z)\bigr)}_{\text{ELBO 的常用形式}} \;+\; D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p_\phi(z|x)\bigr)
\end{aligned}
}
$$

其中 KL 散度的积分定义为：
$$
D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p(z)\bigr) = \int q(z|x) \log \frac{q(z|x)}{p(z)} \, dz,
$$
$$
D_{\mathrm{KL}}\bigl(q(z|x)\,\|\,p_\phi(z|x)\bigr) = \int q(z|x) \log \frac{q(z|x)}{p_\phi(z|x)} \, dz.
$$

**步骤说明**  
(1) 将常数 $\log p_\phi(x)$ 写成关于 $q(z|x)$ 的期望。  
(2) 利用条件概率 $p_\phi(z|x)=p_\phi(x,z)/p_\phi(x)$ 代入。  
(3) 分子分母同乘 $q(z|x)$（乘1技巧）。  
(4) 对数乘积拆分为和，并利用期望线性性。  
(5) 第一项定义为 ELBO，第二项为 KL 散度。  
(6) 将 $p_\phi(x,z)=p_\phi(x|z)p(z)$ 代入 ELBO，并展开期望：$\mathbb{E}[\log p_\phi(x,z)/q] = \mathbb{E}[\log p_\phi(x|z)] + \mathbb{E}[\log p(z)/q] = \mathbb{E}[\log p_\phi(x|z)] - D_{\mathrm{KL}}(q\|p)$。  
(7) 整理得到最终表达式，其中 ELBO 的常用形式为重构项减 KL 正则项，整个式子表明 $\log p_\phi(x)$ 等于该 ELBO 加上近似后验与真实后验的 KL 散度。

---


### VAE：总结

- 数据分布：$p(x)$
- 编码器：$q_{\phi}(z|x) = \mathcal{N}(z; \mu_{\phi}(x), \sigma_{\phi}^2(x)I)$
- 隐分布：$p(z) = \mathcal{N}(z; 0, I)$
- 解码器：$p_{\theta}(x|z) = \mathcal{N}(x; D_{\theta}(z), \sigma^2 I)$

---

### VAE：训练

最大化 ELBO：

$$
\arg\max_{\theta,\phi} \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{KL}\left(q_{\phi}(z|x) \parallel p(z)\right)
$$

使用蒙特卡洛估计近似：

$$
\arg\max_{\theta,\phi} \frac{1}{N}\sum_{i=1}^{N} \log p_{\theta}(x|z^{(i)}) - D_{KL}\left(q_{\phi}(z|x) \parallel p(z)\right)
$$
其中$z^{(i)} = \mu_{\phi}(x) + \sigma_{\phi}(x)\epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

---

### VAE：重建项计算

$$
p_{\theta}(x|z) = \mathcal{N}(x; D_{\theta}(z), \sigma^2 I)
$$
$$
\log p_{\theta}(x|z^{(i)}) = \log \left( \frac{1}{\sqrt{(2\pi\sigma^2)^d}} \exp\left( -\frac{\|x - D_{\theta}(z)\|^2}{2\sigma^2} \right) \right)
$$
$$
= -\frac{1}{2\sigma^2} \underbrace{\|x - D_{\theta}(z)\|^2}_{=\text{这就是为什么称为“重建损失”}}
$$

---

### VAE：训练步骤

1. 将数据点$x$输入编码器，预测$\mu_{\phi}(x)$和$\sigma_{\phi}^2(x)$。
2. 从$q_{\phi}(z|x) = \mathcal{N}(z; \mu_{\phi}(x), \sigma_{\phi}^2(x)I)$中采样隐变量$z$。
3. 将$z$输入解码器，预测$\hat{x} = D_{\theta}(z)$。
4. 通过负 ELBO 计算梯度下降。
---


### 分层 VAE 的 ELBO 推导
$$
\boxed{
\begin{aligned}
\log p(x_0) 
&= \log \int p_{\theta}(x_{0:T}) \, dx_{1:T} \\[6pt]
&= \log \int p_{\theta}(x_{0:T}) \cdot \frac{q_{\phi}(x_{1:T} \mid x_0)}{q_{\phi}(x_{1:T} \mid x_0)} \, dx_{1:T} \\[6pt]
&= \log \int q_{\phi}(x_{1:T} \mid x_0) \cdot \frac{p_{\theta}(x_{0:T})}{q_{\phi}(x_{1:T} \mid x_0)} \, dx_{1:T} \\[6pt]
&\geq \int q_{\phi}(x_{1:T} \mid x_0) \, \log \frac{p_{\theta}(x_{0:T})}{q_{\phi}(x_{1:T} \mid x_0)} \, dx_{1:T} \quad \text{(Jensen 不等式，$\log$ 为凹函数)}
\end{aligned}
}
$$

其中：
- $x_{0:T} = (x_0, x_1, \dots, x_T)$，$x_0$ 为观测数据，$x_{1:T}$ 为隐变量序列；
- $dx_{1:T} = dx_1 dx_2 \cdots dx_T$ 表示多重积分；
- 联合分布：$p_{\theta}(x_{0:T}) = p_{\theta}(x_T) \prod_{t=1}^{T} p_{\theta}(x_{t-1} \mid x_t)$（马尔可夫结构）；
- 变分后验：$q_{\phi}(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q_{\phi}(x_t \mid x_{t-1})$