---
title: "DDPM反向过程"
subtitle: ""
date: 2026-02-28T10:26:59+08:00
# lastmod: 2026-02-28T10:26:59+08:00
draft: false
authors: [Steven]
description: ""

tags: [diffusion/flow]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 3
series_weight: 3

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: "/mywebsite/posts/images/ddpm反向过程.webp"
---


# DDPM 反向过程

反向过程从 $x^{(T)} \sim \mathcal{N}(0, \mathbf{I})$ 出发，逐步采样 $x^{(T-1)}, \ldots, x^{(0)}$，得到生成样本。目标是用神经网络拟合**反向转移** $p_\theta(x^{(t-1)} \mid x^{(t)})$。由于不给定 $x^{(0)}$ 时真实反向 $q(x^{(t-1)} \mid x^{(t)})$ 不可解析，我们利用**给定 $x^{(0)}$ 时可解析**的后验 $q(x^{(t-1)} \mid x^{(t)}, x^{(0)})$ 做推导与训练，再以 $\epsilon_\theta$ 参数化均值，得到最终的反向采样公式。

记号与前向一致：$\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$。

---

## 1. 反向后验的贝叶斯形式

在给定 $x^{(t)}$ 与 $x^{(0)}$ 时，由贝叶斯公式（前向转移用 $q$ 表示）：

$$
q(x^{(t-1)} \mid x^{(t)}, x^{(0)}) = \frac{q(x^{(t)} \mid x^{(t-1)})\, q(x^{(t-1)} \mid x^{(0)})}{q(x^{(t)} \mid x^{(0)})}.
$$

三项均为前向过程的高斯，有闭式：

- $q(x^{(t)} \mid x^{(t-1)}) = \mathcal{N}(x^{(t)}; \sqrt{\alpha_t}\, x^{(t-1)}, \beta_t \mathbf{I})$
- $q(x^{(t-1)} \mid x^{(0)}) = \mathcal{N}(x^{(t-1)}; \sqrt{\bar\alpha_{t-1}}\, x^{(0)}, (1-\bar\alpha_{t-1})\mathbf{I})$
- $q(x^{(t)} \mid x^{(0)}) = \mathcal{N}(x^{(t)}; \sqrt{\bar\alpha_t}\, x^{(0)}, (1-\bar\alpha_t)\mathbf{I})$

因此上式右边可算出，且**后验仍为高斯**（高斯的条件仍为高斯）。下面推导其均值 $\tilde\mu_t$ 与方差 $\tilde\beta_t$。

---

## 2. 后验均值 $\tilde\mu_t$ 与方差 $\tilde\beta_t$ 的推导

记
$$
q(x^{(t-1)} \mid x^{(t)}, x^{(0)}) = \mathcal{N}(x^{(t-1)}; \tilde\mu_t(x^{(t)}, x^{(0)}), \tilde\beta_t \mathbf{I}).
$$

对高斯密度取对数、只保留与 $x^{(t-1)}$ 有关的项（其余并入常数），有
$$
\log q(x^{(t-1)} \mid x^{(t)}, x^{(0)}) = -\frac{1}{2\beta_t}\big\| x^{(t)} - \sqrt{\alpha_t}\, x^{(t-1)} \big\|^2 - \frac{1}{2(1-\bar\alpha_{t-1})}\big\| x^{(t-1)} - \sqrt{\bar\alpha_{t-1}}\, x^{(0)} \big\|^2 + \text{const}.
$$

这是 $x^{(t-1)}$ 的二次型，故后验为高斯。展开并合并 $x^{(t-1)}$ 的二次项与一次项即可得到 $\tilde\beta_t$ 与 $\tilde\mu_t$。

### 2.1 方差 $\tilde\beta_t$

$x^{(t-1)}$ 的二次项系数为
$$
\frac{\alpha_t}{2\beta_t} + \frac{1}{2(1-\bar\alpha_{t-1})} = \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{2\beta_t(1-\bar\alpha_{t-1})}.
$$

后验方差满足 $1/\tilde\beta_t = \alpha_t/\beta_t + 1/(1-\bar\alpha_{t-1})$，故
$$
\tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}.
$$

利用 $\alpha_t = 1 - \beta_t$，分母为
$$
\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = (1-\beta_t)(1-\bar\alpha_{t-1}) + \beta_t = (1-\bar\alpha_{t-1}) - \beta_t(1-\bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t.
$$

因此
$$
\boxed{\tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{1 - \bar\alpha_t}.}
$$

### 2.2 均值 $\tilde\mu_t$

由二次型配方法或直接写高斯条件均值，可得
$$
\tilde\mu_t(x^{(t)}, x^{(0)}) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x^{(0)} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x^{(t)}.
$$

即
$$
\boxed{\tilde\mu_t = \frac{1}{1-\bar\alpha_t}\Big( \sqrt{\bar\alpha_{t-1}}\,\beta_t\, x^{(0)} + \sqrt{\alpha_t}(1-\bar\alpha_{t-1})\, x^{(t)} \Big).}
$$

---

## 3. 用 $\epsilon$（噪声）表示均值并引入 $\epsilon_\theta$

前向重参数化有 $x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\,\epsilon$，故
$$
x^{(0)} = \frac{x^{(t)} - \sqrt{1-\bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}.
$$

代入 $\tilde\mu_t$ 的表达式，将 $x^{(0)}$ 用 $x^{(t)}$ 与 $\epsilon$ 替换，可化简为仅含 $x^{(t)}$ 与 $\epsilon$ 的形式（推导见下），得到
$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left( x^{(t)} - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon \right).
$$

**化简步骤**：将 $x^{(0)} = (x^{(t)} - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ 代入
$$
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x^{(0)} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x^{(t)},
$$
第一项变为
$$
\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}}\big( x^{(t)} - \sqrt{1-\bar\alpha_t}\,\epsilon \big).
$$
利用 $\bar\alpha_t = \alpha_t \bar\alpha_{t-1}$ 得 $\sqrt{\bar\alpha_{t-1}}/\sqrt{\bar\alpha_t} = 1/\sqrt{\alpha_t}$，故第一项为
$$
\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)}\, x^{(t)} - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\,\epsilon.
$$
第二项为 $\sqrt{\alpha_t}(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)\, x^{(t)}$。两者相加，$x^{(t)}$ 的系数为
$$
\frac{\beta_t + \alpha_t(1-\bar\alpha_{t-1})}{\sqrt{\alpha_t}(1-\bar\alpha_t)} = \frac{1-\bar\alpha_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}},
$$
因此
$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\, x^{(t)} - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\,\epsilon = \frac{1}{\sqrt{\alpha_t}}\left( x^{(t)} - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon \right).
$$

**参数化**：采样时没有 $\epsilon$ 与 $x^{(0)}$，用神经网络 $\epsilon_\theta(x^{(t)}, t)$ 预测噪声，得到可用的均值
$$
\mu_\theta(x^{(t)}, t) = \frac{1}{\sqrt{\alpha_t}}\left( x^{(t)} - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x^{(t)}, t) \right).
$$

---

## 4. 最终反向过程公式

**模型反向转移**（DDPM 中方差取固定 $\tilde\beta_t$，不学习）：
$$
p_\theta(x^{(t-1)} \mid x^{(t)}) = \mathcal{N}\big(x^{(t-1)};\ \mu_\theta(x^{(t)}, t),\ \tilde\beta_t \mathbf{I}\big),
$$
其中
$$
\mu_\theta(x^{(t)}, t) = \frac{1}{\sqrt{\alpha_t}}\left( x^{(t)} - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x^{(t)}, t) \right), \qquad \tilde\beta_t = \frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}.
$$

**采样**：从 $x^{(T)} \sim \mathcal{N}(0, \mathbf{I})$ 开始，对 $t = T, T-1, \ldots, 1$ 采样
$$
x^{(t-1)} = \mu_\theta(x^{(t)}, t) + \sqrt{\tilde\beta_t}\,\zeta, \qquad \zeta \sim \mathcal{N}(0, \mathbf{I}).
$$

**训练目标**：在给定 $x^{(0)}$ 与 $t$ 时，按前向采样 $x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\,\epsilon$，令网络 $\epsilon_\theta(x^{(t)}, t)$ 预测 $\epsilon$，最小化例如 $\|\epsilon - \epsilon_\theta(x^{(t)}, t)\|^2$（或加权 MSE），等价于拟合 $q(x^{(t-1)} \mid x^{(t)}, x^{(0)})$ 的均值。

---

## 5. 小结

| 量 | 公式 |
|----|------|
| 后验方差 | $\tilde\beta_t = \dfrac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}$ |
| 后验均值（含 $x^{(0)}$） | $\tilde\mu_t = \dfrac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x^{(0)} + \dfrac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x^{(t)}$ |
| 后验均值（含 $\epsilon$） | $\tilde\mu_t = \dfrac{1}{\sqrt{\alpha_t}}\left( x^{(t)} - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon \right)$ |
| 模型均值 | $\mu_\theta(x^{(t)}, t) = \dfrac{1}{\sqrt{\alpha_t}}\left( x^{(t)} - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x^{(t)}, t) \right)$ |
| 反向采样 | $x^{(t-1)} = \mu_\theta(x^{(t)}, t) + \sqrt{\tilde\beta_t}\,\zeta,\ \zeta\sim\mathcal{N}(0,\mathbf{I})$ |

推导链条：**贝叶斯后验** → **高斯闭式 $\tilde\mu_t,\, \tilde\beta_t$** → **用 $x^{(t)},\epsilon$ 表出 $\tilde\mu_t$** → **用 $\epsilon_\theta$ 替代 $\epsilon$** → **得到 $p_\theta(x^{(t-1)}\mid x^{(t)})$ 与采样式**。

---

$\tilde\mu_t$（以及 $\mu_\theta$）只是反向条件分布的均值，不是最终的 $x^{(t-1)}$ 本身。
真正采样时是从高斯里抽一个样本，即
均值 + 标准差×标准正态：
$$
x^{(t-1)} = \mu_\theta(x^{(t)}, t) + \sqrt{\tilde\beta_t}\,\zeta, \qquad \zeta \sim \mathcal{N}(0, \mathbf{I}).
$$
这里的 $+\sqrt{\tilde\beta_t}\,\zeta$ 就是“后面加的噪声”。
所以：
公式 $\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\big( x^{(t)} - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon \big)$ 给出的是分布的均值；
实际更新是：先算均值 $\mu_\theta$，再加上 $\sqrt{\tilde\beta_t}\,\zeta$ 得到 $x^{(t-1)}$。
第 4 节「采样」和第 5 节小结表里已经写了带 $\sqrt{\tilde\beta_t}\,\zeta$ 的采样式；均值公式和采样式是配套的：前者定义均值，后者在均值基础上加噪声完成一步采样。