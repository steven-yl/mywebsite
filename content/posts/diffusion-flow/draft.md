---
title: "draft"
subtitle: ""
date: 2026-03-06T00:00:00+08:00
draft: false
authors: [Steven]
description: ""

tags: [draft]
categories: [draft]
series: [draft系列]
weight: 0
series_weight: 0

hiddenFromHomePage: false
hiddenFromSearch: false
---

## 推导图（Mermaid）

流程图内公式使用 Mermaid 10.9+ 支持的 `$$...$$` 语法；若主题未启用或版本较低，图下方有完整公式对照（由页面 KaTeX 渲染）。

{{< mermaid >}}
graph TD
    A["CTMC 转移概率<br>$$P(X_{t+h}=y|X_t=x) = \delta(y,x) + h\,u_t(y,x) + o(h)$$"] --> B["代入因子化速度<br>$$u_t(y,x) = \sum_k \delta(y^{\neg k}, x^{\neg k})\,u_t^k(y^k,x)$$"]
    B --> C["得到<br>$$P = \delta(y,x) + h \sum_k \delta(y^{\neg k}, x^{\neg k})\,u_t^k(y^k,x) + o(h)$$"]

    C --> D["利用 $$\delta=\prod\delta$$ 及展开式"]
    D --> E["因此<br>$$P = \prod_k \big[\delta(y^k,x^k) + h\,u_t^k(y^k,x) + o(h)\big]$$"]

    E --> F["边缘化求单坐标<br>$$P(X_{t+h}^i = y^i|X_t=x) = \sum_{y^{\neg i}} P$$"]
    F --> G["代入乘积并拆分因子"]

    G --> H["对 $$j\neq i$$ 求和"]
    H --> I["速率条件 ⇒ 求和 $$= 1+o(h)$$"]
    I --> J["乘积求和 $$= (1+o(h))^{d-1} = 1+o(h)$$"]

    J --> K["最终<br>$$P(X_{t+h}^i = y^i|X_t=x) = \delta(y^i,x^i) + h\,u_t^i(y^i,x) + o(h)$$"]

    K --> L["结论: $$o(h)$$ 精度下各坐标独立，可独立采样"]
{{< /mermaid >}}

### 公式对照（KaTeX 渲染）

若图中公式未显示，可对照以下步骤（由页面数学引擎渲染）：

| 步骤 | 公式 |
|------|------|
| **A** | $$P(X_{t+h}=y \mid X_t=x) = \delta(y,x) + h\,u_t(y,x) + o(h)$$ |
| **B** | $$u_t(y,x) = \sum_i \delta(y^{\bar{i}}, x^{\bar{i}})\, u_t^i(y^i,x)$$ |
| **C** | $$P = \delta(y,x) + h \sum_i \delta(y^{\bar{i}}, x^{\bar{i}})\, u_t^i(y^i,x) + o(h)$$ |
| **D** | $$\delta(y,x) = \prod_i \delta(y^i,x^i)$$；展开式 $$\prod_i[\delta(y^i,x^i) + h\, u_t^i(y^i,x)] = \delta(y,x) + h \sum_i \delta(y^{\bar{i}}, x^{\bar{i}}) u_t^i(y^i,x) + o(h)$$ |
| **E** | $$P = \prod_i [\delta(y^i,x^i) + h\, u_t^i(y^i,x) + o(h)]$$ |
| **F** | $$P(X_{t+h}^i = y^i \mid X_t=x) = \sum_{y^{\bar{i}}} P$$ |
| **G** | $$[\delta(y^i,x^i)+h\, u_t^i+o(h)] \times \sum_{y^{\bar{i}}} \prod_{j\neq i} [\delta(y^j,x^j)+h\, u_t^j+o(h)]$$ |
| **H–J** | $$\sum_{y^j} [\delta(y^j,x^j)+h\, u_t^j+o(h)] = 1+o(h)$$；$$\sum_{y^j} u_t^j(y^j,x)=0$$ ⇒ 乘积 $$(1+o(h))^{d-1}=1+o(h)$$ |
| **K** | $$P(X_{t+h}^i = y^i \mid X_t=x) = \delta(y^i,x^i) + h\, u_t^i(y^i,x) + o(h)$$ |
| **L** | 在 $$o(h)$$ 精度下，各坐标演化独立，可独立采样。 |


---


---

# VPScheduler 公式推导

## 1. 约定与目标

**Variance Preserving (VP)**：路径写成  
$x_t = \alpha_t x_1 + \sigma_t \varepsilon$，$\varepsilon\sim\mathcal{N}(0,I)$，并要求**单位方差**：
$$
\alpha_t^2 + \sigma_t^2 = 1.
$$
- $t=0$：源（噪声）→ $\sigma_0$ 大、$\alpha_0$ 小  
- $t=1$：目标（数据）→ $\sigma_1=0$、$\alpha_1=1$

所以 $\alpha_t$ 从 0 附近单调增到 1，$\sigma_t$ 从 1 附近单调减到 0。

---

## 2. 用“累积噪声”$T(t)$ 参数化

设
$$
\alpha_t^2 = e^{-T(t)} \Rightarrow \alpha_t = e^{-T(t)/2},\quad \sigma_t = \sqrt{1 - e^{-T(t)}}.
$$
- $T(1)=0$ ⇒ $\alpha_1=1,\,\sigma_1=0$（纯数据）  
- $T(0)$ 最大 ⇒ $\alpha_0$ 最小、$\sigma_0$ 最大（噪声多）

$T(t)$ 表示从 $t$ 到 $1$ 的“剩余/累积噪声”量，由噪声调度 $\beta(s)$ 的积分给出。

---

## 3. $T(t)$ 的公式：线性 $\beta$ 的积分

取**关于 $(1-s)$ 的线性**调度（$s$ 为 flow 时间，$s=0$ 噪声大、$s=1$ 噪声小）：
$$
\beta(s) = b + (B - b)(1 - s),\quad s\in[0,1].
$$
- $s=0$：$\beta(0)=B$（`beta_max`）  
- $s=1$：$\beta(1)=b$（`beta_min`）

定义**从当前时间 $t$ 到 $s=1$ 的积分**（即“到数据端还剩多少噪声”）：
$$
T(t) = \int_t^1 \beta(s)\,\mathrm{d}s.
$$

代入 $\beta(s) = b + (B-b)(1-s)$：
$$
\int_t^1 \bigl[b + (B-b)(1-s)\bigr]\mathrm{d}s
= b(1-t) + (B-b)\int_t^1 (1-s)\,\mathrm{d}s
= b(1-t) + (B-b)\,\frac{(1-t)^2}{2}.
$$
所以
$$
\boxed{T(t) = \frac{1}{2}(1-t)^2(B-b) + (1-t)b.}
$$
对应代码：
```python
T = 0.5 * (1 - t) ** 2 * (B - b) + (1 - t) * b
```

对 $t$ 求导：
$$
\frac{\mathrm{d}T}{\mathrm{d}t} = (1-t)(-1)(B-b) + (-1)b = -(1-t)(B-b) - b.
$$
对应：
```python
dT = -(1 - t) * (B - b) - b
```

---

## 4. $\alpha_t$、$\sigma_t$ 及其导数

- $\alpha_t = e^{-T/2}$  
  $$
  \frac{\mathrm{d}\alpha_t}{\mathrm{d}t} = -\frac{1}{2}\,\frac{\mathrm{d}T}{\mathrm{d}t}\,e^{-T/2}.
  $$
  → `alpha_t=exp(-0.5*T)`, `d_alpha_t=-0.5*dT*exp(-0.5*T)`.

- $\sigma_t = \sqrt{1 - e^{-T}}$  
  记 $f(T)=1-e^{-T}$，则 $\sigma_t = f^{1/2}$，
  $$
  \frac{\mathrm{d}\sigma_t}{\mathrm{d}t} = \frac{1}{2}f^{-1/2}\,\frac{\mathrm{d}f}{\mathrm{d}T}\,\frac{\mathrm{d}T}{\mathrm{d}t}
  = \frac{1}{2}\,\frac{e^{-T}}{\sqrt{1-e^{-T}}}\,\frac{\mathrm{d}T}{\mathrm{d}t}.
  $$
  → `sigma_t=sqrt(1-exp(-T))`, `d_sigma_t=0.5*dT*exp(-T)/sqrt(1-exp(-T))`.

注意代码里的 `dT` 是**对 $t$ 的导数**（即 $\mathrm{d}T/\mathrm{d}t$），所以上面用 $\frac{\mathrm{d}T}{\mathrm{d}t}$ 代入就得到和实现一致的导数。

---

## 5. 小结（公式来源一览）

| 量 | 来源 |
|----|------|
| $T(t)$ | $\int_t^1 \beta(s)\mathrm{d}s$，$\beta(s)=b+(B-b)(1-s)$ 积分得 $\frac{1}{2}(1-t)^2(B-b)+(1-t)b$ |
| $\mathrm{d}T/\mathrm{d}t$ | 对上面关于 $t$ 求导得 $-(1-t)(B-b)-b$ |
| $\alpha_t$ | VP 约定 $\alpha_t^2=e^{-T}$ ⇒ $\alpha_t=e^{-T/2}$ |
| $\sigma_t$ | $\alpha_t^2+\sigma_t^2=1$ ⇒ $\sigma_t=\sqrt{1-e^{-T}}$ |
| $\mathrm{d}\alpha_t/\mathrm{d}t$ | 链式法则 $-0.5\,(\mathrm{d}T/\mathrm{d}t)\,e^{-T/2}$ |
| $\mathrm{d}\sigma_t/\mathrm{d}t$ | 链式法则 $0.5\,e^{-T}(1-e^{-T})^{-1/2}\,(\mathrm{d}T/\mathrm{d}t)$ |

因此，这些公式是由 **VP 约束 + 线性 $\beta(s)=b+(B-b)(1-s)$ 的积分 $T(t)$** 推导出来的。



---


这个关系来自**扩散过程里“信号系数”随时间的演化方程**，下面按步骤说明。

---

# 为什么 α_t = exp(−T/2)

## 1. 扩散里“信号”怎么衰减

在连续时间扩散里，我们经常把过程写成：

- 信号部分：系数 \(\alpha_t\)（对应“干净数据”的权重）
- 噪声部分：系数 \(\sigma_t\)（对应噪声的权重）
- VP 约束：\(\alpha_t^2 + \sigma_t^2 = 1\)

扩散的物理含义是：**随时间不断往信号里加噪声**。数学上通常假设“信号功率”按 \(\beta(t)\) 衰减：

\[
\frac{\mathrm{d}(\alpha_t^2)}{\mathrm{d}t} = -\beta(t)\,\alpha_t^2.
\]

也就是说：**信号系数的平方** 的衰减率由 \(\beta(t)\) 决定。

---

## 2. 解这个微分方程

把 \(\alpha_t^2\) 当成一个函数，上式是标准的一阶线性 ODE：

\[
\frac{\mathrm{d}(\alpha_t^2)}{\alpha_t^2} = -\beta(t)\,\mathrm{d}t
\;\Rightarrow\;
\ln(\alpha_t^2) = -\int \beta(t)\,\mathrm{d}t + \mathrm{const}
\;\Rightarrow\;
\alpha_t^2 = C\cdot e^{-\int \beta(t)\,\mathrm{d}t}.
\]

在 flow matching 的约定里，\(t=1\) 是“纯数据”、\(t=0\) 是“纯噪声”，所以用的是**从当前时间 \(t\) 积到 1** 的“剩余噪声量”：

\[
T(t) = \int_t^1 \beta(s)\,\mathrm{d}s.
\]

取边界 \(\alpha_{t=1}^2 = 1\)（\(t=1\) 时全是信号），就得到：

\[
\alpha_t^2 = e^{-T(t)}.
\]

再开方（\(\alpha_t \ge 0\)）：

\[
\boxed{\alpha_t = e^{-T(t)/2} = \exp(-T/2).}
\]

所以 **α_t = exp(−T/2)** 就是上面这个 ODE 的解在 VP 约定下的写法。

---

## 3. 直观理解

- **T**：从当前时刻 \(t\) 到终点 1 之间“累积加的噪声”多少（\(\int_t^1 \beta\)）。
- **α_t² = exp(−T)**：T 越大 → 加的噪声越多 → 信号系数平方越小 → \(\alpha_t\) 越小。
- **α_t = exp(−T/2)**：只是对 \(\alpha_t^2 = \exp(-T)\) 开根号，因为代码里需要的是系数 \(\alpha_t\) 而不是 \(\alpha_t^2\)。

所以：**“为什么是 exp(−T/2)”** —— 因为先由扩散的衰减律得到 **α_t² = exp(−T)**，再开方就得到 **α_t = exp(−T/2)**。T 和 β 的积分关系在 VPScheduler 里已经由 `T = 0.5*(1-t)^2*(B-b) + (1-t)*b` 等实现好了。