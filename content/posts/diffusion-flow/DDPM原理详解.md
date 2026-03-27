---
title: "DDPM原理详解"
subtitle: ""
date: 2026-02-28T10:26:59+08:00
## lastmod: 2026-02-28T10:26:59+08:00
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

## 一、前向过程

前向过程将数据 $x^{(0)} \sim p_\text{data}$ 逐步加噪，得到 $x^{(1)}, \ldots, x^{(T)}$，最终 $x^{(T)}$ 近似标准高斯。下面给出定义与**单步转移**、**多步边际** $q(x^{(t)} \mid x^{(0)})$ 的闭式推导，以及**重参数化**形式。

---

### 1. 定义与记号

- **前向过程**是马尔可夫链：
  $$x^{(0)} \rightarrow x^{(1)} \rightarrow \cdots \rightarrow x^{(T)}.$$
- 固定方差序列 $\beta_1, \ldots, \beta_T \in (0,1)$，令
  $$\alpha_t = 1 - \beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t} \alpha_s.$$
  （约定 $\bar\alpha_0 = 1$。）

---

### 2. 单步转移（定义）

前向的**单步转移**取为均值缩小、方差固定的高斯：

$$
q(x^{(t)} \mid x^{(t-1)}) = \mathcal{N}\big(x^{(t)};\ \sqrt{1-\beta_t}\, x^{(t-1)},\ \beta_t \mathbf{I}\big) = \mathcal{N}\big(x^{(t)};\ \sqrt{\alpha_t}\, x^{(t-1)},\ \beta_t \mathbf{I}\big).
$$

等价地，可写成**重参数化**形式（便于采样与推导）：

$$
x^{(t)} = \sqrt{\alpha_t}\, x^{(t-1)} + \sqrt{\beta_t}\, \varepsilon_{t}, \qquad \varepsilon_t \sim \mathcal{N}(0, \mathbf{I}),\ \text{i.i.d.}
$$

---

### 3. 多步边际 $q(x^{(t)} \mid x^{(0)})$ 的闭式

我们希望对中间步积分，得到**从 $x^{(0)}$ 一步到 $x^{(t)}$** 的分布 $q(x^{(t)} \mid x^{(0)})$，并证明它仍是**单高斯**且**有闭式**。

#### 3.1 递推：$x^{(t)}$ 用 $x^{(0)}$ 与噪声表示

由单步形式反复代入：

$$
\begin{aligned}
x^{(1)} &= \sqrt{\alpha_1}\, x^{(0)} + \sqrt{\beta_1}\, \varepsilon_1, \\
x^{(2)} &= \sqrt{\alpha_2}\, x^{(1)} + \sqrt{\beta_2}\, \varepsilon_2 = \sqrt{\alpha_2\alpha_1}\, x^{(0)} + \sqrt{\alpha_2\beta_1}\, \varepsilon_1 + \sqrt{\beta_2}\, \varepsilon_2, \\
&\vdots
\end{aligned}
$$

一般地，$x^{(t)}$ 可写成 $x^{(0)}$ 与 $\varepsilon_1,\ldots,\varepsilon_t$ 的线性组合。由于各 $\varepsilon_s$ 独立且与 $x^{(0)}$ 独立，该线性组合仍为**高斯**，只需求其均值与方差。下面推导中会自然出现 $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$。

#### 3.2 均值与 $\sqrt{\bar\alpha_t}$ 的推导

记 $x^{(t)}$ 中 $x^{(0)}$ 的系数为 $c_t$。由递推：
- $x^{(1)} = \sqrt{\alpha_1}\, x^{(0)} + \cdots$，故 $c_1 = \sqrt{\alpha_1}$；
- $x^{(t)} = \sqrt{\alpha_t}\, x^{(t-1)} + \sqrt{\beta_t}\, \varepsilon_t$，若 $x^{(t-1)}$ 中 $x^{(0)}$ 的系数为 $c_{t-1}$，则 $x^{(t)}$ 中 $x^{(0)}$ 的系数为 $c_t = \sqrt{\alpha_t}\, c_{t-1}$。

因此
$$
c_t = \sqrt{\alpha_t}\, c_{t-1} = \sqrt{\alpha_t\,\alpha_{t-1}}\, c_{t-2} = \cdots = \sqrt{\alpha_t \cdots \alpha_1} = \sqrt{\prod_{s=1}^{t}\alpha_s} = \sqrt{\bar\alpha_t}.
$$

$\mathbb{E}[\varepsilon_s]=0$，故
$$
\mathbb{E}[x^{(t)} \mid x^{(0)}] = \sqrt{\bar\alpha_t}\, x^{(0)}.
$$

#### 3.3 方差与 $(1-\bar\alpha_t)$ 的推导

记 $v_t = \mathrm{Var}(x^{(t)} \mid x^{(0)})$（标量方差，各维度独立且相同）。由 $x^{(t)} = \sqrt{\alpha_t}\, x^{(t-1)} + \sqrt{\beta_t}\, \varepsilon_t$，且 $x^{(t-1)}$ 与 $\varepsilon_t$ 在给定 $x^{(0)}$ 下独立，故
$$
v_t = \alpha_t\, v_{t-1} + \beta_t.
$$

利用 $\beta_t = 1 - \alpha_t$，代入得
$$
v_t = \alpha_t\, v_{t-1} + (1 - \alpha_t).
$$

**递推初值**：$x^{(0)}$ 给定无随机性，$v_0 = 0$。可验证 $v_1 = \beta_1 = 1 - \alpha_1 = 1 - \bar\alpha_1$。

**归纳**：设 $v_{t-1} = 1 - \bar\alpha_{t-1}$，则
$$
v_t = \alpha_t\, (1 - \bar\alpha_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t\bar\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\bar\alpha_{t-1} = 1 - \bar\alpha_t.
$$

因此
$$
\mathrm{Var}(x^{(t)} \mid x^{(0)}) = (1 - \bar\alpha_t)\, \mathbf{I}.
$$

于是：

$$
\boxed{
q(x^{(t)} \mid x^{(0)}) = \mathcal{N}\big(x^{(t)};\ \sqrt{\bar\alpha_t}\, x^{(0)},\ (1-\bar\alpha_t)\,\mathbf{I}\big).
}
$$

即：**给定 $x^{(0)}$ 时，$x^{(t)}$ 是单高斯，均值 $\sqrt{\bar\alpha_t}\, x^{(0)}$，方差 $(1-\bar\alpha_t)\mathbf{I}$**，与中间步无关，**有闭式、可采样、可求密度**。

---

### 4. 重参数化形式（采样与训练用）

将 $x^{(t)}$ 写成**仅依赖 $x^{(0)}$ 与一个标准高斯噪声 $\epsilon$** 的形式，便于实现采样与后续对 $\epsilon$ 的回归：

$$
x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I}).
$$

等价性：右边均值为 $\sqrt{\bar\alpha_t}\, x^{(0)}$，方差为 $(1-\bar\alpha_t)\mathbf{I}$，与 $q(x^{(t)} \mid x^{(0)})$ 一致；且**单步加噪**与**多步一次加噪**在分布上等价（给定 $x^{(0)}$），因此训练时可对 $(x^{(0)}, t)$ 随机采样，再按上式生成 $x^{(t)}$，让网络预测对应的 $\epsilon$（即 $\epsilon_\theta(x^{(t)}, t)$）。

---

### 5. 小结

| 量 | 形式 |
|----|------|
| 单步转移 | $q(x^{(t)} \mid x^{(t-1)}) = \mathcal{N}(\sqrt{\alpha_t}\, x^{(t-1)},\ \beta_t \mathbf{I})$ |
| 多步边际 | $q(x^{(t)} \mid x^{(0)}) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x^{(0)},\ (1-\bar\alpha_t)\mathbf{I})$ |
| 重参数化 | $x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\, \epsilon,\ \epsilon\sim\mathcal{N}(0,\mathbf{I})$ |

- $\bar\alpha_t$ 随 $t$ 增大而减小，故 $\sqrt{\bar\alpha_t}$ 变小、$\sqrt{1-\bar\alpha_t}$ 变大，$x^{(t)}$ 中噪声占比增加；当 $t=T$ 且 $\bar\alpha_T \approx 0$ 时，$x^{(T)}$ 近似 $\mathcal{N}(0,\mathbf{I})$。
- 前向过程**不包含可学习参数**；反向过程才用神经网络拟合 $q(x^{(t-1)} \mid x^{(t)}, x^{(0)})$ 的近似 $p_\theta(x^{(t-1)} \mid x^{(t)})$。

---


## 二. 反向过程
反向过程从 $x^{(T)} \sim \mathcal{N}(0, \mathbf{I})$ 出发，逐步采样 $x^{(T-1)}, \ldots, x^{(0)}$，得到生成样本。目标是用神经网络拟合**反向转移** $p_\theta(x^{(t-1)} \mid x^{(t)})$。由于不给定 $x^{(0)}$ 时真实反向 $q(x^{(t-1)} \mid x^{(t)})$ 不可解析，我们利用**给定 $x^{(0)}$ 时可解析**的后验 $q(x^{(t-1)} \mid x^{(t)}, x^{(0)})$ 做推导与训练，再以 $\epsilon_\theta$ 参数化均值，得到最终的反向采样公式。

记号与前向一致：$\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$。

---

### 1. 反向后验的贝叶斯形式

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

### 2. 后验均值 $\tilde\mu_t$ 与方差 $\tilde\beta_t$ 的推导

记
$$
q(x^{(t-1)} \mid x^{(t)}, x^{(0)}) = \mathcal{N}(x^{(t-1)}; \tilde\mu_t(x^{(t)}, x^{(0)}), \tilde\beta_t \mathbf{I}).
$$

对高斯密度取对数、只保留与 $x^{(t-1)}$ 有关的项（其余并入常数），有
$$
\log q(x^{(t-1)} \mid x^{(t)}, x^{(0)}) = -\frac{1}{2\beta_t}\big\| x^{(t)} - \sqrt{\alpha_t}\, x^{(t-1)} \big\|^2 - \frac{1}{2(1-\bar\alpha_{t-1})}\big\| x^{(t-1)} - \sqrt{\bar\alpha_{t-1}}\, x^{(0)} \big\|^2 + \text{const}.
$$

这是 $x^{(t-1)}$ 的二次型，故后验为高斯。展开并合并 $x^{(t-1)}$ 的二次项与一次项即可得到 $\tilde\beta_t$ 与 $\tilde\mu_t$。

#### 2.1 方差 $\tilde\beta_t$

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

#### 2.2 均值 $\tilde\mu_t$

由二次型配方法或直接写高斯条件均值，可得
$$
\tilde\mu_t(x^{(t)}, x^{(0)}) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\, x^{(0)} + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x^{(t)}.
$$

即
$$
\boxed{\tilde\mu_t = \frac{1}{1-\bar\alpha_t}\Big( \sqrt{\bar\alpha_{t-1}}\,\beta_t\, x^{(0)} + \sqrt{\alpha_t}(1-\bar\alpha_{t-1})\, x^{(t)} \Big).}
$$

---

### 3. 用 $\epsilon$（噪声）表示均值并引入 $\epsilon_\theta$

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

### 4. 最终反向过程公式

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

### 5. 小结

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