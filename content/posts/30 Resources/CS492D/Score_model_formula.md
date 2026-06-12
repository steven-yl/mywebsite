---
title: "Score_model_formula"
subtitle: "Score_model_formula"
date: 2026-06-10T12:22:25+08:00
draft: false
authors: [Steven]
tags: [CS492D]
categories: [CS492D]
series: [CS492D系列]
---


## 基于得分的模型

### 得分函数

得分函数定义为对数似然函数关于数据点的梯度：

$$
\nabla_{x}\log p(x)
$$

对于分布$q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})$，其得分为：

$$
\nabla_{x_t} \log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon_t}{\sqrt{1 - \bar{\alpha}_t}}
$$

因此，噪声预测器$\hat{\epsilon}_\theta(x_t, t)$可以解释为预测得分函数（带有缩放因子）。


### Tweedie 公式推导：
$$
\boxed{
\begin{aligned}
&\because\; \nabla_x \log p(x\mid\mu) = -\Sigma^{-1}(x-\mu) 
\quad & \text{(高斯条件分布的分数函数)}\\[4pt]
&\Rightarrow\; \mu = x + \Sigma\, \nabla_x \log p(x\mid\mu) 
\quad & \text{(由上式解出 }\mu\text{)}\\[4pt]
&\Rightarrow\; \mathbb{E}[\mu\mid x] 
= x + \Sigma\; \mathbb{E}\bigl[\nabla_x \log p(x\mid\mu) \mid x\bigr] 
\quad & \text{(对后验分布 }p(\mu\mid x)\text{ 取期望)}\\[4pt]
&= x + \Sigma\; \frac{1}{p(x)} \int \nabla_x p(x\mid\mu)\, p(\mu)\, d\mu 
\quad & \text{(代入条件期望定义及 }p(\mu\mid x)=\frac{p(x\mid\mu)p(\mu)}{p(x)}\text{)}\\[4pt]
&= x + \Sigma\; \frac{1}{p(x)} \nabla_x \int p(x\mid\mu)p(\mu)\, d\mu 
\quad & \text{(交换梯度与积分，假定正则条件成立)}\\[4pt]
&= x + \Sigma\; \frac{\nabla_x p(x)}{p(x)} 
\quad & \bigl(p(x)=\int p(x\mid\mu)p(\mu)d\mu\bigr)\\[4pt]
&= x + \Sigma\; \nabla_x \log p(x) 
\quad & \text{(边际分布的分数函数)}\\[4pt]
&\therefore\; \boxed{\mathbb{E}[\mu\mid x] = x + \Sigma\,\nabla_x\log p(x)} \quad & \text{(Tweedie 公式)}
\end{aligned}
}
$$
### Tweedie 公式在扩散模型中的作用
$$
\boxed{
\begin{aligned}
&\mathbb{E}[\mu \mid x] = x + \Sigma \nabla_x \log p(x) 
\quad &&\text{(1) Tweedie 公式的一般形式（高斯观测模型）} \\[6pt]
&\overset{\mu = \sqrt{\bar{\alpha}_t}x_0,\; \Sigma = (1-\bar{\alpha}_t)I}{=} 
\sqrt{\bar{\alpha}_t}\,\mathbb{E}[x_0 \mid x_t] = x_t + (1-\bar{\alpha}_t) \nabla_{x_t}\log q(x_t) 
\quad &&\text{(2) 代入扩散模型的前向噪声分布} \\[6pt]
&\Rightarrow \mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1-\bar{\alpha}_t)\nabla_{x_t}\log q(x_t)}{\sqrt{\bar{\alpha}_t}} 
\quad &&\text{(3) 从分数函数恢复干净数据的后验均值} \\[6pt]
&\Rightarrow \nabla_{x_t}\log q(x_t) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\,\mathbb{E}[x_0\mid x_t]}{1-\bar{\alpha}_t} 
\quad &&\text{(4) 从后验均值恢复分数函数（双向桥梁）} \\[6pt]
&\overset{\text{后验期望}}{=} \mathbb{E}_{x_0\mid x_t}\!\left[ \nabla_{x_t}\log q(x_t\mid x_0) \right] 
\quad &&\text{(4b) 无条件分数 = 条件分数的后验期望} \\[6pt]
&\overset{\text{给定 }x_0}{=} -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t} 
= -\frac{\epsilon_t}{\sqrt{1-\bar{\alpha}_t}},\quad \epsilon_t \sim \mathcal{N}(0,\mathbf{I}) 
\quad &&\text{(5) 条件分数函数（训练时的回归目标）} \\[6pt]
&\Rightarrow \min_\theta \mathbb{E}_{x_0,\epsilon_t,t}\left[\bigl\| s_\theta(x_t,t) - \nabla_{x_t}\log q(x_t\mid x_0) \bigr\|^2\right] 
\quad &&\text{(6) 得分匹配训练目标} \\[6pt]
&\overset{\text{由 (4)(5) 等价}}{=} \min_\theta \mathbb{E}_{x_0,\epsilon_t,t}\left[\bigl\| \epsilon_\theta(x_t,t) - \epsilon_t \bigr\|^2\right] 
\quad &&\text{(7) 等价于 DDPM 的噪声预测损失} \\[6pt]
&\Rightarrow x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t,t) \right) + \sigma_t z,\quad z\sim\mathcal{N}(0,\mathbf{I}) 
\quad &&\text{(8) DDPM 反向采样步骤（依赖 Tweedie 关系）}
\end{aligned}
}
$$
解释：
- 最终需要的是无条件分数 $\nabla_{x_t}\log q(x_t)$，但根据 Tweedie 公式，无条件分数等于条件分数关于后验分布 $q(x_0\mid x_t)$ 的期望：$\nabla_{x_t}\log q(x_t) = \mathbb{E}_{x_0\mid x_t}\bigl[ \nabla_{x_t}\log q(x_t\mid x_0) \bigr]$。
- 当我们用大量 $(x_0,\epsilon_t)$ 对训练网络拟合条件分数时，网络在每一个 $(x_t,t)$ 处学到的实际上是条件分数的平均（因为数据中会有很多不同的 $x_0$ 产生同一个 $x_t$）。这个平均正好就是无条件分数。所以，训练网络拟合条件分数，等价于训练它拟合无条件分数。

### 得分模型&Langevin采样&退火策略Langevin采样推导
$$
\boxed{
\begin{aligned}
\nabla_{x_t}\log q(x_t|x_0) &= -\frac{x_t - \sqrt{\bar{\alpha}_t}\,x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon_t}{\sqrt{1 - \bar{\alpha}_t}} \quad &\text{(得分函数与噪声预测器)} \\
&\stackrel{\text{Tweedie}}{=} \frac{x_t - \sqrt{\bar{\alpha}_t}\,\mathbb{E}[x_0|x_t]}{1 - \bar{\alpha}_t} \quad &\text{(Tweedie公式估计真实均值)} \\
&\approx s_\theta(x_t, t) \quad &\text{(噪声条件得分模型估计)} \\
&\Longrightarrow \min_\theta \mathbb{E}_{x_0, x_t}\left[ \| s_\theta(x_t, t) - \nabla_{x_t}\log q(x_t|x_0) \|^2 \right] \quad &\text{(训练目标，等价于DDPM损失)} \\
&\Longrightarrow x_{t-1} = x_t + \eta \nabla_{x_t}\log q(x_t) + \sqrt{2\eta}\,\epsilon,\; \epsilon\sim\mathcal{N}(0,\mathbf{I}) \quad &\text{(Langevin动力学采样)} \\
&\Longrightarrow \sigma_1 > \sigma_2 > \cdots > \sigma_T \;\text{且}\; x \leftarrow x + \eta \nabla_{x}\log q_\sigma(x) + \sqrt{2\eta}\,\epsilon \quad &\text{(退火Langevin动力学，逐渐减小噪声尺度)}
\end{aligned}
}
$$

朗之万动力学从 **随机微分方程** → **Fokker‑Planck 方程** → **稳态分布** → **离散采样公式** 的完整推导：
$$
\boxed{
\begin{aligned}
& dx(t) = \nabla_x \log q(x(t))\, dt + \sqrt{2}\, dW(t) 
\quad & \text{(1) 连续时间朗之万 SDE，}dW(t)\sim\mathcal{N}(0,dt\,\mathbf{I}) \\[4pt]
&\Rightarrow\; \frac{\partial p(x,t)}{\partial t} = -\nabla_x\!\cdot\!\bigl(p(x,t)\,\nabla_x\log q(x)\bigr) + \nabla_x^2 p(x,t) 
\quad & \text{(2) 对应的 Fokker‑Planck 方程（概率密度演化）} \\[4pt]
&\Rightarrow\; 0 = -\nabla_x\!\cdot\!\bigl(p_{\text{ss}}(x)\,\nabla_x\log q(x)\bigr) + \nabla_x^2 p_{\text{ss}}(x) 
\quad & \text{(3) 稳态条件：}\partial_t p=0\;\Rightarrow\;p_{\text{ss}}(x)\propto q(x)\text{（验证得解）} \\[4pt]
&\Rightarrow\; x(t+\Delta t) = x(t) + \nabla_x\log q(x(t))\,\Delta t + \sqrt{2\Delta t}\;\epsilon,\quad \epsilon\sim\mathcal{N}(0,\mathbf{I}) 
\quad & \text{(4) 欧拉‑丸山离散化（步长 }\eta=\Delta t\text{）} \\[4pt]
&\Rightarrow\; \boxed{x \leftarrow x + \eta\,\nabla_x\log q(x) + \sqrt{2\eta}\,\epsilon} 
\quad & \text{(5) 最终采样迭代公式}
\end{aligned}
}
$$