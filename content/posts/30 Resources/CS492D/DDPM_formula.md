---
title: "DDPM_Formula"
subtitle: "DDPM_Formula"
date: 2026-06-10T12:22:25+08:00
draft: false
authors: [Steven]
tags: [CS492D]
categories: [CS492D]
series: [CS492D系列]
---

## VAE 2 DDPM
**特殊设定**：
马尔可夫分层 VAE 的一个特例：
- 隐变量维度与数据维度相同
- 变分后验$q_{\phi}(x_{t+1}|x_t)$不是学习的，而是预定义的：
 $$
  q_{\phi}(x_{t+1}|x_t) \rightarrow q(x_{t+1}|x_t)
 $$
---

## DDPM的ELBO推导
$$
\boxed{
\begin{aligned}
&-\log p(x_0) \\
&\qquad \leq \mathbb{E}_{q(x_{1:T}|x_0)}\left[-\log\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}\right] 
&&\text{(变分下界)}\\[6pt]
&\qquad = \mathbb{E}_{q(x_{1:T}|x_0)}\left[ -\log\left(p(x_T)\prod_{t=1}^{T} p_{\theta}(x_{t-1}|x_t)\right) + \log\left(\prod_{t=1}^{T} q(x_t|x_{t-1})\right) \right] 
&&\text{(代入 }p_{\theta}(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_{\theta}(x_{t-1} \mid x_t), \qquad 
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})\text{)}\\[6pt]
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
&\qquad = \underbrace{-\mathbb{E}_{q(x_1|x_0)}\bigl[\log p_{\theta}(x_0|x_1)\bigr]}_{\mathcal{L}_0 \text{重构损失：从 }x_1\text{ 重建 }x_0\text{ 的负对数似然}}
+ \underbrace{D_{KL}\bigl(q(x_T|x_0)\,\|\,p(x_T)\bigr)}_{\mathcal{L}_T \text{先验匹配项：最终潜变量分布与先验的 KL 散度}}
+ \underbrace{\sum_{t=2}^{T}\mathbb{E}_{q(x_t|x_0)}\Bigl[ D_{KL}\bigl(q(x_{t-1}|x_t,x_0)\,\|\,p_{\theta}(x_{t-1}|x_t)\bigr) \Bigr]}_{\mathcal{L}_{t-1} \text{去噪匹配项：每个时间步真实后验与模型逆向分布的 KL 散度期望}}
\end{aligned}
}
$$

- $\mathcal{L}_0$：与 VAE 相同，通常可忽略。
- $\mathcal{L}_T$：当$T \to \infty$时趋于零。与 VAE 中的 KL 散度项相同。$q(x_T|x_0)$和$p(x_T)$都是预定义的，此项无需优化。
- $\mathcal{L}_{t-1}$：核心去噪匹配项。
---

## 去噪匹配项$L_{t-1}$计算实现

$$
\sum_{t=2}^{T} \mathbb{E}_{q(x_t|x_0)} \left[ D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p_{\theta}(x_{t-1}|x_t)) \right]
$$

变分分布$p_{\theta}(x_{t-1}|x_t)$应接近$q(x_{t-1}|x_t, x_0)$。

---

### 目标分布$q(x_{t-1}|x_t, x_0)$的计算建模

利用贝叶斯法则：
$$
q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}) \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}
$$

- 详细推导：
  $$
  \begin{aligned}
  q(x_{t-1} \mid x_t, x_0) 
  &\stackrel{\text{(1) 条件概率定义}}{=} 
  \frac{q(x_{t-1}, x_t \mid x_0)}{q(x_t \mid x_0)} \\[8pt]
  &\stackrel{\text{(2) 链式法则拆分子}}{=} 
  \frac{q(x_t \mid x_{t-1}, x_0) \; q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} \\[8pt]
  &\stackrel{\text{(3) 马尔可夫性质}}{=} 
  \frac{q(x_t \mid x_{t-1}) \; q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}.
  \end{aligned}
  $$
  **注释说明**：
  - **(1)** 给定 $x_0$ 的条件概率定义：$q(A|B,C) = \frac{q(A,B|C)}{q(B|C)}$，这里 $A=x_{t-1}, B=x_t, C=x_0$。
  - **(2)** 将联合分布 $q(x_{t-1}, x_t|x_0)$ 写成 $q(x_t|x_{t-1},x_0)\, q(x_{t-1}|x_0)$，这是概率的链式乘法。
  - **(3)** 前向扩散过程是马尔可夫链：给定当前状态 $x_{t-1}$，下一步 $x_t$ 与历史 $x_0$ 条件独立，因此 $q(x_t|x_{t-1},x_0) = q(x_t|x_{t-1})$。

已知：
- $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)I)$
- $q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1})I)$
- $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$

经过推导：
$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde{\mu}(x_t, x_0), \tilde{\sigma}_t^2 I)
$$
其中：
$$
\tilde{\mu}(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0
$$
$$
\tilde{\sigma}_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$

- $q(x_{t-1}|x_t, x_0)$的均值和协方差
  - 均值是$x_t$和$x_0$的函数。
  - 协方差是预定义的（仅依赖于$\beta_t$序列）。

---

#### 从$x_t$和$\epsilon_t$重写均值：

由$q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$，有：
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$
定义：
$$
\epsilon_t = \frac{1}{\sqrt{1-\bar{\alpha}_t}} x_t - \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{1-\bar{\alpha}_t}} x_0
$$
代入均值表达式得：
$$
\tilde{\mu}(x_t, \epsilon_t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_t \right)
$$

---

### 预测分布$p_{\theta}(x_{t-1}|x_t)$的计算建模

如何建模$p_{\theta}(x_{t-1}|x_t)$？

- 由于$q(x_{t-1}|x_t, x_0)$的方差$\tilde{\sigma}_t^2$与$x_t, x_0$无关，定义：
 $$
  p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(\mu_{\theta}(x_t, t), \tilde{\sigma}_t^2 I)
 $$
  其中$\mu_{\theta}(x_t, t)$是均值预测器。

---

### KL 散度的计算（去噪匹配）

对于两个同方差的高斯分布$p(x) = \mathcal{N}(x; \mu_p, \sigma^2 I)$和$q(x) = \mathcal{N}(x; \mu_q, \sigma^2 I)$，
$$
D_{KL}(p \parallel q) = \frac{1}{2\sigma^2} \| \mu_q - \mu_p \|^2
$$

- 详细推导：
  $$
  \begin{aligned}
  D_{\mathrm{KL}}(p \parallel q) 
  &\stackrel{\text{(1) 定义}}{=} \mathbb{E}_{x\sim p}\left[\log p(x) - \log q(x)\right] \\[4pt]
  &\stackrel{\text{(2) 代入高斯密度}}{=} \mathbb{E}_{x\sim p}\left[ \left(-\frac{d}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|x-\mu_p\|^2\right) - \left(-\frac{d}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|x-\mu_q\|^2\right) \right] \\[4pt]
  &\stackrel{\text{(3) 消去常数项}}{=} \mathbb{E}_{x\sim p}\left[ \frac{1}{2\sigma^2}\left( \|x-\mu_q\|^2 - \|x-\mu_p\|^2 \right) \right] \\[4pt]
  &\stackrel{\text{(4) 展开平方差}}{=} \frac{1}{2\sigma^2}\, \mathbb{E}_{x\sim p}\left[ -2x^\top(\mu_q-\mu_p) + \|\mu_q\|^2 - \|\mu_p\|^2 \right] \\[4pt]
  &\stackrel{\text{(5) 线性性求期望}}{=} \frac{1}{2\sigma^2}\left( -2\mu_p^\top(\mu_q-\mu_p) + \|\mu_q\|^2 - \|\mu_p\|^2 \right) \\[4pt]
  &\stackrel{\text{(6) 合并同类项}}{=} \frac{1}{2\sigma^2}\left( \|\mu_q\|^2 - 2\mu_p^\top\mu_q + \|\mu_p\|^2 \right) \\[4pt]
  &\stackrel{\text{(7) 配成平方范数}}{=} \frac{1}{2\sigma^2}\, \|\mu_q - \mu_p\|^2.
  \end{aligned}
  $$
因此：
$$
\mathbb{E}_{q(x_t|x_0)} \left[ D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p_{\theta}(x_{t-1}|x_t)) \right]
= \frac{1}{2\tilde{\sigma}_t^2} \mathbb{E}_{q(x_t|x_0)} \left[ \| \mu_{\theta}(x_t, t) - \tilde{\mu}(x_t, x_0) \|^2 \right]
$$

---
##### $x_0$预测器

若使用$x_0$预测器$\hat{x}_{\theta}(x_t, t)$代替均值预测器，则：
$$
\mathbb{E}_{q(x_t|x_0)} \left[ D_{KL}(\cdot) \right]
= \frac{1}{2\tilde{\sigma}_t^2} \frac{\bar{\alpha}_{t-1} \beta_t^2}{(1 - \bar{\alpha}_t)^2} \mathbb{E}_{q(x_t|x_0)} \left[ \| \hat{x}_{\theta}(x_t, t) - x_0 \|^2 \right]
$$
$$
= \omega_t \mathbb{E}_{q(x_t|x_0)} \left[ \| \hat{x}_{\theta}(x_t, t) - x_0 \|^2 \right]
$$

- $x_0$预测器的用途
  - 目标是从标准正态样本$x_T$通过隐变量$x_{T-1}, \dots, x_1$采样得到$x_0$。
  - 对于每个$x_t$，直接预测$x_0$的期望值。

---

##### $\epsilon_t$预测器

若使用$\epsilon_t$预测器$\hat{\epsilon}_{\theta}(x_t, t)$代替均值预测器，利用：
$$
\tilde{\mu}(x_t, \epsilon_t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_t \right)
$$

则：
$$
\mathbb{E}_{q(x_t|x_0)} \left[ D_{KL}(\cdot) \right]
= \frac{1}{2\tilde{\sigma}_t^2} \frac{(1 - \bar{\alpha}_t)^2}{\bar{\alpha}_t (1 - \bar{\alpha}_t)} \mathbb{E}_{q_{\phi}(x_t|x_0)} \left[ \| \hat{\epsilon}_{\theta}(x_t, t) - \epsilon_t \|^2 \right]
$$
$$
= \omega_t' \mathbb{E}_{q(x_t|x_0)} \left[ \| \hat{\epsilon}_{\theta}(x_t, t) - \epsilon_t \|^2 \right]
$$


- $\epsilon_t$预测器的含义
  - 从$x_t$预测$\epsilon_t$的期望值，该$\epsilon_t$是通过前向跳步从$x_0$采样得到$x_t$时所用的噪声。


### 变分分布$p_\theta(x_{t-1}|x_t)$的三种形式
已知：
$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde{\mu}(x_t, x_0), \tilde{\sigma}_t^2 \mathbf{I})
$$

- 选项1：均值预测器$\mu_\theta(x_t, t)$

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \tilde{\sigma}_t^2 \mathbf{I})
$$

- 选项2：$x_0$预测器$\hat{x}_\theta(x_t, t)$

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}\left(\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \hat{x}_\theta(x_t, t), \tilde{\sigma}_t^2 \mathbf{I}\right)
$$

- 选项3：$\varepsilon_t$预测器$\hat{\varepsilon}_\theta(x_t, t)$

## DDIM 完整推导
$$
\boxed{
\begin{aligned}
&\text{DDPM 前向（马尔可夫）: } q(x_{1:T}|x_0)=\prod_{t=1}^T q(x_t|x_{t-1}) 
\quad &&\text{(1) 边际 } q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I) \\[6pt]
&\text{DDIM 前向（非马尔可夫）: } q_\sigma(x_{1:T}|x_0)=q_\sigma(x_T|x_0)\prod_{t=2}^T q_\sigma(x_{t-1}|x_t,x_0) 
\quad &&\text{(2) 引入条件后向核，保持边际不变} \\[6pt]
&\text{假设: } q_\sigma(x_{t-1}|x_t,x_0)=\mathcal{N}\bigl(\omega_0 x_0+\omega_t x_t+b,\;\sigma_t^2 I\bigr) 
\quad &&\text{(3) 待定参数 } \omega_0,\omega_t,b,\sigma_t \\[6pt]
&\text{约束: } q_\sigma(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I),\quad q_\sigma(x_{t-1}|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})I) 
\quad &&\text{(4) 边际分布必须与 DDPM 一致} \\[6pt]
&\text{推导: } x_{t-1}=\omega_0 x_0+\omega_t x_t+b+\sigma_t\epsilon',\;\epsilon'\sim\mathcal{N}(0,I) 
\quad &&\text{(5) 从 (3) 采样表达式} \\[6pt]
&\text{代入 } x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,\;\epsilon\sim\mathcal{N}(0,I): \\ 
&x_{t-1}=\omega_0 x_0+\omega_t(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon)+b+\sigma_t\epsilon' 
= (\omega_0+\omega_t\sqrt{\bar{\alpha}_t})x_0 + b + \omega_t\sqrt{1-\bar{\alpha}_t}\,\epsilon + \sigma_t\epsilon' 
\quad &&\text{(6) 展开并合并项} \\[6pt]
&\text{要求: } \mathbb{E}[x_{t-1}|x_0] = \sqrt{\bar{\alpha}_{t-1}}x_0 \;\Rightarrow\; \omega_0+\omega_t\sqrt{\bar{\alpha}_t}= \sqrt{\bar{\alpha}_{t-1}},\quad b=0 
\quad &&\text{(7) 均值条件} \\[6pt]
&\text{要求: } \operatorname{Cov}(x_{t-1}|x_0) = (1-\bar{\alpha}_{t-1})I \\ 
&\text{且 } \epsilon,\epsilon' \text{ 独立} \;\Rightarrow\; \omega_t^2(1-\bar{\alpha}_t) + \sigma_t^2 = 1-\bar{\alpha}_{t-1} 
\quad &&\text{(8) 方差条件} \\[6pt]
&\text{由 (7): } \omega_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}-\sigma_t^2}{1-\bar{\alpha}_t}} 
\quad &&\text{(9) 解出 } \omega_t \\[6pt]
&\text{由 (7): } \omega_0 = \sqrt{\bar{\alpha}_{t-1}} - \sqrt{\bar{\alpha}_t}\sqrt{\frac{1-\bar{\alpha}_{t-1}-\sigma_t^2}{1-\bar{\alpha}_t}} 
\quad &&\text{(10) 解出 } \omega_0 \\[6pt]
&\Rightarrow q_\sigma(x_{t-1}|x_t,x_0)=\mathcal{N}\!\left(\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\frac{x_t-\sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}},\;\sigma_t^2 I\right) 
\quad &&\text{(11) 最终后验核表达式} \\[6pt]
&\text{若 } \sigma_t=0: \text{过程变为确定性（隐式模型）} 
\quad &&\text{(12) DDIM 的核心：可快速采样} \\[6pt]
&\text{训练损失: } \mathbb{E}_{q(x_t|x_0)}\bigl[D_{\text{KL}}(q_\sigma(x_{t-1}|x_t,x_0)\parallel p_\theta(x_{t-1}|x_t))\bigr] 
\quad &&\text{(13) 与 DDPM 的 ELBO 形式相同（仅差常数）} \\[6pt]
&\Rightarrow \text{训练目标等价于噪声预测: } \min_\theta \mathbb{E}_{x_0,\epsilon,t}\bigl[\|\epsilon_\theta(x_t,t)-\epsilon\|^2\bigr] 
\quad &&\text{(14) 因此 DDPM 训练好的 } \epsilon_\theta \text{ 可直接用于 DDIM 采样} \\[6pt]
&\boxed{\text{重要结论：DDPM 的噪声预测网络无需重新训练，即可在 DDIM 的确定性或随机采样中使用。}}
\end{aligned}
}
$$

**参数求解步骤说明**：  
- 步骤 (5) 写出 $x_{t-1}$ 关于 $x_0$、$x_t$ 和噪声的表达式。  
- 步骤 (6) 代入 \(x_t\) 与 \(x_0\) 的关系，将 \(x_{t-1}\) 表示为 \(x_0\) 与两个独立高斯噪声的线性组合。  
- 步骤 (7) 要求条件期望与 DDPM 边际一致，得到均值的两个方程（$\omega_0+\omega_t\sqrt{\bar{\alpha}_t}=\sqrt{\bar{\alpha}_{t-1}}$ 且 $b=0$）。  
- 步骤 (8) 要求条件方差与 DDPM 边际一致，得到方差方程 $\omega_t^2(1-\bar{\alpha}_t)+\sigma_t^2=1-\bar{\alpha}_{t-1}$。  
- 步骤 (9)(10) 解出 $\omega_t$ 和 $\omega_0$，代入即得步骤 (11)。