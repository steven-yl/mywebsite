---
title: "Flow-Matching-Formula"
subtitle: ""
date: 2026-03-04T12:22:25+08:00
# lastmod: 
draft: false
authors: [Steven]
description: ""

tags: [diffusion/flow]
categories: [diffusion/flow]
series: [diffusion/flow系列]

weight: 0
series_weight: 0

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
featuredImageCaption: ""

header:
    number:
        enable: true

summary: ""
---


## 推导图

{{< mermaid >}}graph LR;
    A[Flow Matching] --> B("条件概率\边际概率")
    A[Flow Matching] --> C("条件速度场\边际速度场")
    A[Flow Matching] --> D("速度调度器变换")
    A[Flow Matching] --> E("高斯路径下边际速度场的参数化(速度\x_0\x_1\score之间的转换)")
    A[Flow Matching] --> F("边际概率的计算(微分同胚\推前映射\变量替换)")
    A[Flow Matching] --> G("条件引导")
{{< /mermaid >}}


## 关键公式推导
### 联合概率密度与边际概率密度
- 随机向量 $X, Y$，联合PDF $p_{X,Y}(x,y)$ 满足边际化性质：
  - $p_X(x) = \int p_{X,Y}(x,y) dy$ 
  - $p_Y(y) = \int p_{X,Y}(x,y) dx$

### 条件概率密度与贝叶斯法则
- 条件 PDF 定义：$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$（要求 $p_Y(y) > 0$）

### 条件概率密度和边际概率密度
- z：样本数据，x：采样数据
- 条件概率路径：$p_{t|Z}(x|z)$（生成 $Z=z$ 时的条件路径）；
- 边际概率路径：$p_t(x) = \int p_{t|Z}(x|z) p_Z(z) dz$；


### 条件期望与全期望性质
- 条件期望 $\mathbb{E}[X \mid Y = y] = \int x p_{X \mid Y}(x \mid y) dx$，是“给定 $Y = y$ 时，最小二乘意义下最接近 $X$ 的函数”；
- 全期望性质（Tower Property）：$\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$——多层期望可简化为单层期望，是后续边际速度场推导的关键工具。

**全期望性质：**
记 $\mu(Y) = \mathbb{E}[X \mid Y]$（给定 $Y$ 时 $X$ 的条件期望），它是 $Y$ 的函数（随机变量）。

- **内层 $\mathbb{E}[X \mid Y]$**：对 **$X$** 取平均。在 $Y$ 固定为某值 $y$ 时，用条件分布 $p_{X|Y}(x|y)$ 算期望，即 $\mathbb{E}[X \mid Y=y] = \int x\, p_{X|Y}(x|y)\, dx$。因此内层的结果是 $Y$ 的函数 $\mu(Y)$。
- **外层 $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[\mu(Y)]$**：对 **$Y$** 取平均。用 $Y$ 的边际分布 $p_Y(y)$ 对 $\mu(y)$ 求期望，即 $\int \mu(y)\, p_Y(y)\, dy$。
- **右边 $\mathbb{E}[X]$**：对 **$(X,Y)$ 的联合**（或等价地对 $X$ 的边际）取平均，即 $\int x\, p_X(x)\, dx = \iint x\, p_{X,Y}(x,y)\, dx\, dy$。

因此：**先对 $X$ 在“给定 $Y$”下求期望，再对 $Y$ 求期望，等于直接对 $X$ 求期望**；全期望性质说的是“先条件后边际”与“直接边际”一致。

### 条件速度场和边际速度场
- 条件速度场：$u_t(x|z)$ 由条件路径 $p_{t|Z}(x|z)$ 唯一确定（满足连续性方程，生成该路径）；线性条件流时为 $u_t(x|z) = \frac{z-x}{1-t}$（从当前 $x$ 指向目标 $z$）；
- 边际速度场：$u_t(x) = \int u_t(x|z)\, p_{Z|t}(z|x)\, dz = \int u_t(x|z)\, \frac{p_{t|Z}(x|z)\, p_Z(z)}{p_t(x)}\, dz = \mathbb{E_z}[u_t(x|Z) \mid X_t=x]$（第二式将后验 $p_{Z|t}(z|x)$ 用贝叶斯展开；末式为条件期望形式，便于理解和计算）。
- 边际速度场具体计算公式:
$$
u_t(x)
\approx
\frac{
\sum_{k=1}^K u_t(x\mid z^{(k)}) \cdot \underbrace{p_{t\mid Z}(x\mid z^{(k)})}_{\text{权重}w_k}
}{
\sum_{k=1}^K w_k
}
$$
其中
$
z^{(k)} \sim p_Z(z)
$


{{< admonition tip "边际速度场数学推导：把期望换成可计算形式" false >}}
边际速度场数学推导：把期望换成可计算形式
你要的积分：
$$
u_t(x) = \int u_t(x\mid z)\,\color{red}{p_{Z\mid t}(z\mid x)}\,dz
$$

把贝叶斯代入：
$$
\color{red}{p_{Z\mid t}(z\mid x)}
= \frac{p_{t\mid Z}(x\mid z)\,p_Z(z)}{p_t(x)}
$$

所以：
$$
u_t(x)
= \int u_t(x\mid z)
\cdot \frac{p_{t\mid Z}(x\mid z)\,p_Z(z)}{p_t(x)}
dz
$$

把分母提出来：
$$
u_t(x)
= \frac{1}{p_t(x)}
\int u_t(x\mid z)\,p_{t\mid Z}(x\mid z)\,\color{red}{p_Z(z)}\,dz
$$

注意红色部分：
$$
\int (\cdots) \color{red}{p_Z(z)} dz
= \mathbb{E}_{z\sim p}\big[\,\cdots\,\big]
$$

所以：
$$
u_t(x)
= \frac{1}{p_t(x)}\;
\mathbb{E}_{z\sim p}\big[\,u_t(x\mid z)\,p_{t\mid Z}(x\mid z)\,\big]
$$

---

分母 $p_t(x)$ 也能写成期望
$$
p_t(x) = \int p_{t\mid Z}(x\mid z)\,p_Z(z)\,dz
$$

也是对 $p(z)$ 的期望：
$$
p_t(x) = \mathbb{E}_{z\sim p}\big[\,p_{t\mid Z}(x\mid z)\,\big]
$$

---

合起来：**重要采样公式**
把两个期望合并：
$$
u_t(x) =
\frac{\;\mathbb{E}_{z\sim p}\big[\,u_t(x\mid z)\cdot p_{t\mid Z}(x\mid z)\,\big]\;}
{\;\mathbb{E}_{z\sim p}\big[\,p_{t\mid Z}(x\mid z)\,\big]\;}
$$

---

离散化：变成**加权平均**
期望用**样本平均**近似：
$$
\mathbb{E}[\cdots] \approx \frac{1}{K}\sum_{k=1}^K (\cdots)
$$

代入：
$$
u_t(x)
\approx
\frac{
\sum_{k=1}^K u_t(x\mid z^{(k)}) \cdot \underbrace{p_{t\mid Z}(x\mid z^{(k)})}_{\text{权重}w_k}
}{
\sum_{k=1}^K w_k
}
$$

其中
$
z^{(k)} \sim p_Z(z)
$
{{< /admonition >}}

### 微分同胚&推前映射
todo


### 条件引导
通过预测score计算速度场：
$$
u_t(x|y) = a_t x + b_t \nabla \log p_{t|Y}(x|y). \tag{4.87}
$$
#### 分类器引导
$$
p_{t|Y}(x|y) = \frac{p_{Y|t}(y|x) p_t(x)}{p_Y(y)}. \tag{4.88}
$$
$$
\underbrace{\nabla \log p_{t|Y}(x|y)}_{\text{条件分数}} = \underbrace{\nabla \log p_{Y|t}(y|x)}_{\text{分类器}} + \underbrace{\nabla \log p_t(x)}_{\text{无条件分数}}, \tag{4.89}
$$
$$
\tilde{u}_t^{\theta,\phi}(x|y) = a_t x + b_t \bigl( \nabla \log p_{Y|t}^\phi(y|x) + \nabla \log p_t^\theta(x) \bigr) = u_t^\theta(x) + b_t \nabla \log p_{Y|t}^\phi(y|x), \tag{4.90}
$$
$$
\tilde{u}_t^{\theta,\phi}(x|y) = u_t^\theta(x) + b_t w \nabla \log p_{Y|t}^\phi(y|x), \tag{4.91}
$$

#### 无分类器引导
$$
\underbrace{\nabla \log p_{Y|t}(y|x)}_{\text{分类器}} = \underbrace{\nabla \log p_{t|Y}(x|y)}_{\text{条件分数}} - \underbrace{\nabla \log p_t(x)}_{\text{无条件分数}}, \tag{4.92}
$$

$\nabla \log p_{t|Y}(x|y) = \frac{u_t^\theta(x|y) - a_t x}{b_t}$，$\nabla \log p_t(x) = \frac{u_t^\theta(x|\emptyset) - a_t x}{b_t}$。代入上式：
$$
\tilde{u}_t^\theta(x|y) = u_t^\theta(x|\emptyset) + b_t w\,\frac{u_t^\theta(x|y) - u_t^\theta(x|\emptyset)}{b_t} = (1-w)\, u_t^\theta(x|\emptyset) + w\, u_t^\theta(x|y).
$$