---
title: 扩散桥（Diffusion Bridge）原理推导
date: 2026-06-24
tags: [diffusion/flow, DiffusionBridge]
categories: [diffusion/flow,DiffusionBridge]
series: [diffusion/flow系列]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

## 摘要

扩散桥是一种在随机过程理论中用于 **将无约束的扩散过程约束为固定起点和固定终点的条件随机过程** 的数学工具。本文档系统性地阐述了扩散桥的定义、核心原理、数学推导、漂移项的来源与修正，以及完整的求解框架，旨在为理论研究和工程应用提供自洽的参考。

**关键词**：扩散桥；Doob's h-变换；随机微分方程；Girsanov定理；测度变换；条件过程

---

## 1. 引言

### 1.1 背景与动机

在物理、金融、生物和人工智能等领域，我们经常面临这样一个问题：给定一个随机过程，如何在保持其随机性的同时，强制其从已知起点$X_0 = x$出发，并在固定时刻$T$精确到达目标终点$X_T = y$？

普通的扩散过程（如布朗运动、几何布朗运动）虽然能够描述随机演化，但无法保证终点的确定性。扩散桥（Diffusion Bridge）正是为解决这一问题而提出的数学构造。

### 1.2 直观理解

扩散桥可以被形象地理解为 **“受控的随机旅程”** 。它像是给一颗随机滚动的弹珠装上了隐形的导航系统，引导它在充满随机扰动的环境中，从入口精确地到达预定的出口。

---

## 2. 预备知识

### 2.1 随机微分方程（SDE）

设$X_t$为$\mathbb{R}$-值随机过程，其演化由如下随机微分方程描述：

$$
dX_t = f(t, X_t)dt + \sigma(t, X_t)dW_t, \quad X_0 = x
$$

其中：
-$f(t, x)$：**漂移项（Drift）**，刻画过程的确定性趋势；
-$\sigma(t, x)$：**扩散项（Diffusion）**，刻画随机扰动的强度；
-$W_t$：标准维纳过程（布朗运动）。

### 2.2 转移概率密度

记$p(t, x; T, y)$为从时刻$t$的位置$x$出发，在时刻$T$到达位置$y$的转移概率密度。它满足如下性质：

$$
\mathbb{P}(X_T \in dy \mid X_t = x) = p(t, x; T, y) \, dy
$$

---

## 3. 扩散桥的定义

给定一个满足 SDE 的随机过程$X_t$，其 **扩散桥**$X_t^{bridge}$定义为在同时固定起点和终点条件下的原始过程：

$$
X_t^{bridge} \triangleq \big( X_t \mid X_0 = x, \; X_T = y \big), \quad 0 \le t \le T
$$

该条件过程具有与原始过程相同的随机扰动结构，但其漂移项被修正，以确保终点约束得到满足。

---

## 4. 核心工具：Doob's h-变换

### 4.1 向导函数的定义

定义 **向导函数（Guide Function）**$h(t, x)$为从当前状态$(t, x)$出发，最终到达目标终点$y$的转移概率密度：

$$
h(t, x) \equiv p(t, x; T, y) = \mathbb{P}(X_T = y \mid X_t = x)
$$

在严格意义下，该表达式应理解为概率密度而非概率质量。

### 4.2 Doob's h-变换定理

Doob's h-变换指出：将原始过程的漂移项$f$替换为修正漂移项$\tilde{f}$，即可得到以$y$为终点的扩散桥：

$$
\tilde{f}(t, x) = f(t, x) + \sigma^2(t, x) \frac{\partial}{\partial x} \log h(t, x)
$$

于是，扩散桥的 SDE 为：

$$
dX_t^{bridge} = \left[ f(t, X_t^{bridge}) + \sigma^2(t, X_t^{bridge}) \frac{\partial}{\partial x} \log h(t, X_t^{bridge}) \right] dt + \sigma(t, X_t^{bridge}) dW_t
$$

其中终点条件为$X_T^{bridge} = y$。

---

## 5. 向导函数$h(t, x)$的求解

### 5.1 控制方程：Kolmogorov 向后方程

向导函数$h(t, x)$满足如下偏微分方程（PDE），即 **Kolmogorov 向后方程**：

$$
\frac{\partial h}{\partial t} + f(t, x) \frac{\partial h}{\partial x} + \frac{1}{2} \sigma^2(t, x) \frac{\partial^2 h}{\partial x^2} = 0
$$

终端条件为：

$$
h(T, x) = \delta(x - y)
$$

其中$\delta(\cdot)$为狄拉克 delta 函数。

### 5.2 线性扩散过程的显式解

对于如下线性形式的 SDE：

$$
dX_t = \big[\alpha(t) + \beta(t)X_t\big]dt + \sigma(t)dW_t
$$

即漂移项关于$x$线性、扩散项与$x$无关，向导函数具有高斯形式的闭式解：

$$
h(t, x) = \frac{1}{\sqrt{2\pi V(t)}} \exp\left[ -\frac{\big( y - \mu(t; x) \big)^2}{2V(t)} \right]
$$

其中条件期望$\mu(t; x)$和条件方差$V(t)$分别由下式给出：

$$
\mu(t; x) = \mathbb{E}[X_T \mid X_t = x] = \Phi(t, T)x + \int_t^T \Phi(s, T)\alpha(s) \, ds
$$

$$
V(t) = \text{Var}[X_T \mid X_t = x] = \int_t^T \Phi(s, T)^2 \cdot \sigma^2(s) \, ds
$$

而状态转移矩阵$\Phi(t, T)$定义为：

$$
\Phi(t, T) \equiv \exp\left( \int_t^T \beta(s) \, ds \right)
$$

### 5.3 求解链条汇总

$$
\boxed{
\begin{aligned}
& \text{【定义】} & h(t, x) &= p(t, x; T, y) \\
& \text{【控制方程】} & \frac{\partial h}{\partial t} + f\frac{\partial h}{\partial x} + \frac{1}{2}\sigma^2\frac{\partial^2 h}{\partial x^2} &= 0, \quad h(T, x) = \delta(x-y) \\
& \text{【线性化假设】} & f(t, x) &= \alpha(t) + \beta(t)x, \quad \sigma(t, x) = \sigma(t) \\
& \text{【条件期望】} & \mu(t; x) &= \Phi(t, T)x + \int_t^T \Phi(s, T)\alpha(s) \, ds \\
& \text{【条件方差】} & V(t) &= \int_t^T \Phi(s, T)^2 \sigma^2(s) \, ds \\
& \text{【显式闭式解】} & h(t, x) &= \frac{1}{\sqrt{2\pi V(t)}} \exp\left[ -\frac{(y - \mu(t; x))^2}{2V(t)} \right]
\end{aligned}
}
$$

---

## 6. 漂移项的来源与推导

### 6.1 漂移项的数学起源

漂移项$f(t, x)$在数学上被定义为随机过程的**瞬时条件期望变化率**：

$$
\boxed{
f(t, x) = \lim_{\Delta t \to 0} \frac{\mathbb{E}[X_{t+\Delta t} - X_t \mid X_t = x]}{\Delta t}
}
$$

这一极限刻画了在剔除随机噪声后，过程在单位时间内的确定性运动速度。

### 6.2 不同应用场景中的漂移项来源

#### （1）物理与金融：先验建模

在经典应用中，漂移项来源于物理定律或经济原理，由领域专家直接给定：

- **物理（Ornstein-Uhlenbeck 过程）**：
 $$
  f(t, x) = -\gamma x \quad \text{（粘性阻力，均值回归）}
 $$

- **金融（几何布朗运动）**：
 $$
  f(t, x) = \mu x \quad \text{（期望收益率，指数增长）}
 $$

#### （2）AI 生成模型：数据驱动学习

在扩散生成模型中，逆向过程的漂移项由数据分布的对数梯度（得分函数）决定：

$$
\tilde{f}(t, x) = -f(t, x) + g(t)^2 \cdot \nabla_x \log p_t(x)
$$

其中$\nabla_x \log p_t(x)$由神经网络从数据中估计得到。

### 6.3 扩散桥中修正漂移项的严格推导

扩散桥中的修正漂移项$\tilde{f}$可以通过 **Girsanov 定理（测度变换）** 严格推导得出。

#### 推导链条：

$$
\boxed{
\begin{aligned}
& \text{【原始过程（测度 } \mathbb{P} \text{）】} & 
dX_t &= f(t, X_t)\,dt + \sigma(t, X_t)\,dW_t, \quad X_0 = x \\[6pt]
& \text{【向导函数（转移密度）】} & 
h(t, x) &\equiv p(t, x; T, y) = \text{从 }(t,x)\text{ 到 }(T,y)\text{ 的转移密度} \\[6pt]
& \text{【Kolmogorov 向后方程】} & 
\frac{\partial h}{\partial t} + f\frac{\partial h}{\partial x} + \frac{1}{2}\sigma^2\frac{\partial^2 h}{\partial x^2} &= 0, \quad h(T, x) = \delta(x-y) \\[6pt]
& \text{【测度变换权重（Radon-Nikodym）】} & 
\left. \frac{d\mathbb{Q}}{d\mathbb{P}} \right|_t &= \frac{h(t, X_t)}{h(0, x)} \quad (\text{因为 } X_0 = x) \\[6pt]
& \text{【对 } M_t = h(t, X_t) \text{ 应用伊藤引理】} & 
dM_t &= \left( \frac{\partial h}{\partial t} + f\frac{\partial h}{\partial x} + \frac{1}{2}\sigma^2\frac{\partial^2 h}{\partial x^2} \right)dt + \sigma\frac{\partial h}{\partial x}dW_t \\
& & &= \sigma\frac{\partial h}{\partial x}dW_t \quad (\text{向后方程消去 } dt \text{ 项}) \\
& & &= M_t \cdot \left( \sigma(t, X_t) \frac{\partial \log h}{\partial x}(t, X_t) \right) dW_t \\[6pt]
& \text{【Girsanov 定理】} & 
d\tilde{W}_t &= dW_t - \sigma(t, X_t) \frac{\partial \log h}{\partial x}(t, X_t)\,dt \\
& & & \text{其中 } \tilde{W}_t \text{ 是测度 } \mathbb{Q} \text{ 下的标准布朗运动} \\[6pt]
& \text{【将 } dW_t \text{ 代回原始 SDE】} & 
dX_t &= f\,dt + \sigma \left( d\tilde{W}_t + \sigma \frac{\partial \log h}{\partial x}\,dt \right) \\
& &= \left[ f(t, X_t) + \sigma^2(t, X_t) \frac{\partial \log h}{\partial x}(t, X_t) \right] dt + \sigma(t, X_t)\,d\tilde{W}_t \\[6pt]
& \text{【扩散桥的漂移项（最终结论）】} & 
\boxed{ \tilde{f}(t, x) = f(t, x) + \sigma^2(t, x) \frac{\partial}{\partial x} \log h(t, x) }
\end{aligned}
}
$$

---

## 7. 完整推导汇总

以下将整个扩散桥的理论框架汇总为一个完整的多等式链条：

$$
\boxed{
\begin{aligned}
& \text{【1. 原始扩散过程】} & dX_t &= f(t, X_t)dt + \sigma(t, X_t)dW_t, \quad X_0 = x \\
& \text{【2. 定义向导函数】} & h(t, x) &= \mathbb{P}(X_T = y \mid X_t = x) \\
& \text{【3. 向导函数控制方程】} & \frac{\partial h}{\partial t} + f\frac{\partial h}{\partial x} + \frac{1}{2}\sigma^2\frac{\partial^2 h}{\partial x^2} &= 0, \quad h(T, x) = \delta(x-y) \\
& \text{【4. 测度变换（Girsanov）】} & d\tilde{W}_t &= dW_t - \sigma(t, X_t)\frac{\partial}{\partial x}\log h(t, X_t) \, dt \\
& \text{【5. 扩散桥的 SDE】} & dX_t^{bridge} &= \left[ f(t, X_t^{bridge}) + \sigma^2(t, X_t^{bridge})\frac{\partial}{\partial x}\log h(t, X_t^{bridge}) \right] dt \\
& & &\quad + \sigma(t, X_t^{bridge}) d\tilde{W}_t \\
& \text{【6. 边界条件】} & X_0^{bridge} &= x, \quad X_T^{bridge} = y \\
& \text{【7. 线性情形下的显式向导函数】} & h(t, x) &= \frac{1}{\sqrt{2\pi V(t)}} \exp\left[ -\frac{\left( y - \mu(t; x) \right)^2}{2V(t)} \right] \\
& & \mu(t; x) &= \Phi(t, T)x + \int_t^T \Phi(s, T)\alpha(s) \, ds, \quad \Phi(t, T) = \exp\left( \int_t^T \beta(s)ds \right) \\
& & V(t) &= \int_t^T \Phi(s, T)^2 \sigma^2(s) \, ds
\end{aligned}
}
$$

---

## 8. 关键结论

| 要素 | 结论 |
|------|------|
| **核心构造工具** | Doob's h-变换 |
| **额外漂移项（引导项）** |$\sigma^2(t, x) \frac{\partial}{\partial x} \log h(t, x)$|
| **向导函数满足的方程** | Kolmogorov 向后方程 |
| **推导修正漂移的数学工具** | Girsanov 定理（测度变换） |
| **线性扩散的向导函数** | 高斯密度函数（闭式解） |
| **漂移项的根源** | 瞬时条件期望变化率 / 物理先验 / 数据驱动学习 |

---

## 9. 应用场景

扩散桥在多个领域具有重要应用价值：

- **AI 生成模型**：将纯噪声图像引导至真实数据分布（如 Stable Diffusion 的逆向过程）；
- **金融工程**：模拟离散观测数据之间的连续价格路径，用于风险评估与期权定价；
- **生物物理**：模拟分子构象变化的最可能路径，研究蛋白质折叠等动力学过程；
- **时间序列插值**：在缺失数据点之间生成符合随机动力学的最可能轨迹。

---

## 附录：符号表

| 符号 | 含义 |
|------|------|
|$X_t$| 时刻$t$的随机状态 |
|$f(t, x)$| 原始漂移项 |
|$\tilde{f}(t, x)$| 扩散桥修正后的漂移项 |
|$\sigma(t, x)$| 扩散项 |
|$W_t$| 标准维纳过程（测度$\mathbb{P}$） |
|$\tilde{W}_t$| 标准维纳过程（测度$\mathbb{Q}$） |
|$h(t, x)$| 向导函数（转移概率密度） |
|$p(t, x; T, y)$| 从$(t,x)$到$(T,y)$的转移密度 |
|$\Phi(t, T)$| 状态转移矩阵 |
|$\mu(t; x)$| 条件期望$\mathbb{E}[X_T \mid X_t = x]$|
|$V(t)$| 条件方差$\text{Var}[X_T \mid X_t = x]$|

---

## 参考文献

1. Rogers, L. C. G., & Williams, D. (2000). *Diffusions, Markov Processes and Martingales*. Cambridge University Press.
2. Øksendal, B. (2003). *Stochastic Differential Equations: An Introduction with Applications*. Springer.
3. Särkkä, S., & Solin, A. (2019). *Applied Stochastic Differential Equations*. Cambridge University Press.
4. Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. NeurIPS.
5. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.