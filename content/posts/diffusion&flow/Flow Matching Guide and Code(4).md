---
title: "Flow Matching Guide and Code(4)(译)"
subtitle: ""
date: 2026-03-02T14:20:25+08:00
# lastmod: 2026-03-02T14:20:25+08:00
draft: false
authors: [Steven]
description: ""

tags: [diffusion/flow, tutorial]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 1
series_weight: 1

hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""

summary: "Flow Matching Guide and Code 第4章原文翻译"
---

## 4 流匹配（Flow Matching）
给定源分布 \(p\) 和目标分布 \(q\)，流匹配（Flow Matching, FM）（Lipman 等人, 2022；Liu 等人, 2022；Albergo 和 Vanden-Eijnden, 2022）是一种用于训练流模型的可扩展方法，该模型由带可学习参数 \(\theta\) 的速度场 \(u_{t}^{\theta}\) 定义，其核心是解决流匹配问题：找到生成概率路径 \(p_{t}\) 的 \(u_{t}^{\theta}\)，其中 \(p_{0}=p\) 且 \(p_{1}=q\)（4.1）。

上述定义中，“生成”的含义与式（3.24）一致。回顾图 2 中的流匹配框架蓝图，FM 框架的步骤为：（a）确定已知的源分布 \(p\) 和未知的数据目标分布 \(q\)；（b）设计一条从 \(p_{0}=p\) 插值到 \(p_{1}=q\) 的概率路径 \(p_{t}\)；（c）学习一个由神经网络实现的速度场 \(u_{t}^{\theta}\)，该速度场能生成路径 \(p_{t}\)；（d）通过求解由 \(u_{t}^{\theta}\) 定义的常微分方程（ODE）从学习到的模型中采样。为了在步骤（c）中学习速度场 \(u_{t}^{\theta}\)，FM 最小化以下回归损失：
\[
\mathcal{L}_{FM}(\theta)=\mathbb{E}_{X_{t} \sim p_{t}} D\left(u_{t}\left(X_{t}\right), u_{t}^{\theta}\left(X_{t}\right)\right), \quad(4.2)
\]
其中 \(D\) 是向量间的不相似性度量，例如平方 \(\ell_{2}\) 范数 \(D(u, v)=\|u-v\|^{2}\)。直观上，FM 损失促使可学习速度场 \(u_{t}^{\theta}\) 与已知能生成期望概率路径 \(p_{t}\) 的真实速度场 \(u_{t}\) 相匹配。图 9 展示了流匹配框架中的主要对象及其依赖关系。接下来，我们将从如何构建 \(p_{t}\) 和 \(u_{t}\)，以及损失函数（4.2）的实际实现入手，展开对流匹配的阐述。

### 4.1 数据
重申一下，设源样本为服从分布 \(p\) 的随机变量 \(X_{0} \sim p\)，目标样本为服从分布 \(q\) 的随机变量 \(X_{1} \sim q\)。通常，源样本遵循易于采样的已知分布，而目标样本则以有限大小的数据集形式给出。根据应用场景的不同，目标样本可能是图像、视频、音频片段或其他类型的高维、结构丰富的数据。源样本和目标样本可以是相互独立的，也可以来自一个被称为耦合（coupling）的一般联合分布：
\[
(X_{0},X_{1})\sim \pi _{0,1}(X_{0},X_{1}), (4.3)
\]
若没有已知的耦合关系，则源-目标样本遵循独立耦合 \(\pi_{0,1}(X_{0}, X_{1})=p(X_{0}) q(X_{1})\)。独立源-目标分布的一个常见例子是从随机高斯噪声向量 \(X_{0} \sim \mathcal{N}(0, I)\) 生成图像 \(X_{1}\)。而依赖耦合的例子包括从低分辨率图像 \(X_{0}\) 生成高分辨率图像 \(X_{1}\)，或从灰度视频 \(X_{0}\) 生成彩色视频 \(X_{1}\)。

### 4.2 构建概率路径
流匹配通过采用条件策略，极大地简化了设计概率路径 \(p_{t}\) 及其对应速度场 \(u_{t}\) 的问题。作为第一个例子，考虑基于单个目标样本 \(X_{1}=x_{1}\) 设计条件概率路径 \(p_{t | 1}(x | x_{1})\)（如图 3a 所示）。然后，我们可以通过聚合这些条件概率路径 \(p_{t | 1}\) 来构建整体的边缘概率路径 \(p_{t}\)：
\[
p_{t}(x)=\int p_{t | 1}\left(x | x_{1}\right) q\left(x_{1}\right) d x_{1},
\]
如图 3b 所示。为了解决流匹配问题，我们希望 \(p_{t}\) 满足以下边界条件：
\[
p_{0}=p, p_{1}=q, (4.5)
\]
即边缘概率路径 \(p_{t}\) 在时间 \(t=0\) 时插值到源分布 \(p\)，在时间 \(t=1\) 时插值到目标分布 \(q\)。这些边界条件可以通过要求条件概率路径满足以下约束来实现：
\[
p_{0 | 1}\left(x | x_{1}\right)=\pi_{0 | 1}\left(x | x_{1}\right), \text{ 且 } p_{1 | 1}\left(x | x_{1}\right)=\delta_{x_{1}}(x), \quad(4.6)
\]
其中条件耦合 \(\pi_{0 | 1}(x_{0} | x_{1})=\pi_{0,1}(x_{0}, x_{1}) / q(x_{1})\)，\(\delta_{x_{1}}\) 是中心在 \(x_{1}\) 的狄拉克测度（delta measure）。对于独立耦合 \(\pi_{0,1}(x_{0}, x_{1})=p(x_{0}) q(x_{1})\)，上述第一个约束简化为 \(p_{0 | 1}(x | x_{1})=p(x)\)。由于狄拉克测度没有密度，第二个约束应理解为：对于连续函数 \(f\)，当 \(t \to 1\) 时，\(\int p_{t | 1}(x | y) f(y) d y \to f(x)\)。需要注意的是，将式（4.6）代入式（4.4）可以验证边界条件（4.5）是否成立。

式（2.2）给出了一个满足式（4.6）条件的常用条件概率路径示例：
\[
\mathcal{N}\left(\cdot | t x_{1},(1-t)^{2} I\right) \to \delta_{x_{1}}(\cdot) \text{ 当 } t \to 1 \text{ 时。}
\]

### 4.3 推导生成速度场
在得到边缘概率路径 \(p_{t}\) 后，我们现在构建生成 \(p_{t}\) 的速度场 \(u_{t}\)。该生成速度场 \(u_{t}\) 是多个条件速度场 \(u_{t}(x | x_{1})\) 的平均值（如图 3c 所示），且满足：
\[
u_{t}(\cdot | x_{1}) \text{ 生成 } p_{t | 1}\left(\cdot | x_{1}\right). (4.7)
\]
然后，生成边缘路径 \(p_{t}(x)\) 的边缘速度场 \(u_{t}(x)\)（如图 3d 所示）通过对所有目标样本的条件速度场 \(u_{t}(x | x_{1})\) 求平均得到：
\[
u_{t}(x)=\int u_{t}\left(x | x_{1}\right) p_{1 | t}\left(x_{1} | x\right) d x_{1} .
\]

为了用已知项表示上述方程，回顾贝叶斯法则：
\[
p_{1 | t}\left(x_{1} | x\right)=\frac{p_{t | 1}\left(x | x_{1}\right) q\left(x_{1}\right)}{p_{t}(x)}, (4.9)
\]
该式对所有满足 \(p_{t}(x)>0\) 的 \(x\) 成立。式（4.8）可以解释为条件速度 \(u_{t}(x | x_{1})\) 的加权平均，权重 \(p_{1 | t}(x_{1} | x)\) 表示给定当前样本 \(x\) 时目标样本 \(x_{1}\) 的后验概率。利用条件期望（见 3.2 节）还可以对式（4.8）给出另一种解释：即如果 \(X_{t}\) 是任意满足 \(X_{t} \sim p_{t | 1}(\cdot | X_{1})\) 的随机变量（等价地，\((X_{t}, X_{1})\) 的联合分布密度为 \(p_{t, 1}(x, x_{1})=p_{t | 1}(x | x_{1}) q(x_{1})\)），则利用式（3.12）可将式（4.8）表示为条件期望：
\[
u_{t}(x)=\mathbb{E}\left[u_{t}(X_{t}|X_{1}) \mid X_{t}=x\right], (4.10)
\]
这表明 \(u_{t}(x)\) 是给定 \(X_{t}=x\) 时 \(u_{t}(X_{t} | X_{1})\) 的最小二乘逼近（见 3.2 节）。需要注意的是，式（4.10）中的 \(X_{t}\) 与最终流模型（3.16）中定义的 \(X_{t}\) 通常是不同的随机变量，尽管它们具有相同的边缘概率 \(p_{t}(x)\)。

### 4.4 一般条件化与边缘化技巧
为了证明上述构建的合理性，我们需要说明在温和假设下，式（4.8）和式（4.10）中的边缘速度场 \(u_{t}\) 能够生成式（4.4）中的边缘概率路径 \(p_{t}\)。证明这一点的数学工具是质量守恒定理（定理 2）。接下来，我们考虑一个更一般的场景，这将在后续章节中发挥作用。具体来说，通过条件化到 \(X_{1}=x_{1}\) 来构建条件概率路径和速度场并没有特殊之处。正如 Tong 等人（2023）所指出的，前一节的分析可以推广到条件化到任何具有概率密度函数 \(p_{Z}\) 的任意随机变量 \(Z \in \mathbb{R}^{m}\)。这将得到边缘概率路径：
\[
p_{t}(x)=\int p_{t | Z}(x | z) p_{Z}(z) d z,
\]
而生成该路径的边缘速度场为：
\[
u_{t}(x)=\int u_{t}(x | z) p_{Z | t}(z | x) d z=\mathbb{E}\left[u_{t}\left(X_{t} | Z\right) | X_{t}=x\right],
\]
其中 \(u_{t}(\cdot | z)\) 生成 \(p_{t | Z}(\cdot | z)\)，\(p_{Z | t}(z | x)=\frac{p_{t | Z}(x | z) p_{Z}(z)}{p_{t}(x)}\) 由贝叶斯法则给出（满足 \(p_{t}(x)>0\)），且 \(X_{t} \sim p_{t | Z}(\cdot | Z)\)。显然，令 \(Z=X_{1}\) 可以恢复前一节中的构建结果。在证明主要结果之前，我们需要以下正则性假设：

**假设 1**：\(p_{t | Z}(x | z)\) 作为 \((t, x)\) 的函数是 \(C^{1}([0,1) ×\mathbb{R}^{d})\) 光滑的，\(u_{t}(x | z)\) 作为 \((t, x)\) 的函数是 \(C^{1}([0,1) ×\mathbb{R}^{d}, \mathbb{R}^{d})\) 光滑的。此外，\(p_{Z}\) 具有有界支撑，即存在 \(\mathbb{R}^{m}\) 中的有界集，使得在该集合外 \(p_{Z}(x)=0\)。最后，对于所有 \(x \in \mathbb{R}^{d}\) 和 \(t \in[0,1)\)，有 \(p_{t}(x)>0\)。

这些假设是温和的。例如，要证明 \(p_{t}(x)>0\)，只需找到一个条件 \(z\) 使得 \(p_{Z}(z)>0\) 且 \(p_{t | Z}(\cdot | z)>0\)。在实践中，可以通过考虑 \((1-(1-t) \epsilon) p_{t | Z}+(1-t) \epsilon \mathcal{N}(0, I)\)（其中 \(\epsilon>0\) 是任意小的常数）来满足这一条件。式（2.2）中的路径（令 \(Z=X_{1}\)）就是满足该假设的一个例子。现在我们可以给出主要结果：

**定理 3（边缘化技巧）**：在假设 1 下，如果 \(u_{t}(x | z)\) 是条件可积的且生成条件概率路径 \(p_{t}(\cdot | z)\)，则边缘速度场 \(u_{t}\) 生成边缘概率路径 \(p_{t}\)（对所有 \(t \in[0,1)\)）。

上述定理中，“条件可积”指的是质量守恒定理（3.26）中可积性条件的条件版本，即：
\[
\int_{0}^{1} \iint\left\| u_{t}(x | z)\right\| p_{t | Z}(x | z) p_{Z}(x) d z d x d t<\infty .
\]

**证明**：该结果可通过验证质量守恒定理（定理 2）的两个条件来获得。首先，我们验证 \((u_{t}, p_{t})\) 满足连续性方程（3.25）。由于 \(u_{t}(\cdot | x_{1})\) 生成 \(p_{t}(\cdot | x_{1})\)，我们有：
\[
\frac{d}{d t} p_{t}(x) \stackrel{(i)}{=} \int \frac{d}{d t} p_{t | Z}(x | z) p_{Z}(x) d z (4.14)
\]
\[
\stackrel{(ii)}{=}-\int div_{x}\left[u_{t}(x | z) p_{t | Z}(x | z)\right] p_{Z}(z) d z
\]
\[
\stackrel{(i)}{=}-div_{x} \int u_{t}(x | z) p_{t | Z}(x | z) p_{Z}(z) d z
\]
\[
\stackrel{(iii)}{=}-div_{x}\left[u_{t}(x) p_{t}(x)\right] . (4.17)
\]

等式 (i) 成立的依据是莱布尼茨法则（Leibniz's rule），即 \(p_{t | Z}(x | z)\) 和 \(u_{t}(x | z)\) 在 \(t\) 上是 \(C^{1}\) 光滑的，且 \(p_{Z}\) 具有有界支撑（因此所有被积函数都是有界集上的连续函数，从而可积）。等式 (ii) 成立的依据是 \(u_{t}(\cdot | z)\) 生成 \(p_{t | Z}(\cdot | z)\) 以及定理 2。等式 (iii) 成立的依据是将表达式乘以并除以 \(p_{t}(x)\)（由假设可知其严格为正），并利用式（4.12）中 \(u_{t}\) 的定义。

为了验证定理 2 的第二个也是最后一个条件，我们需要证明 \(u_{t}\) 是可积的且局部 Lipschitz 连续的。由于 \(C^{1}\) 函数是局部 Lipschitz 连续的，因此只需验证 \(u_{t}(x)\) 对所有 \((t, x)\) 都是 \(C^{1}\) 光滑的。这一点可由假设中 \(u_{t}(x | z)\) 和 \(p_{t | Z}(x | z)\) 是 \(C^{1}\) 光滑的且 \(p_{t}(x)>0\) 直接推出。此外，由于 \(u_{t}(x | z)\) 是条件可积的，因此 \(u_{t}(x)\) 是可积的：
\[
\int_{0}^{1} \int\left\| u_{t}(x)\right\| p_{t}(x) d x d t \leq \int_{0}^{1} \iint\left\| u_{t}(x | z)\right\| p_{t | Z}(x | z) p_{Z}(z) d z d x d t<\infty,
\]
其中第一个不等式由向量 Jensen 不等式推出。

### 4.5 流匹配损失
在确定目标速度场 \(u_{t}\) 能生成从 \(p\) 到 \(q\) 的指定概率路径 \(p_{t}\) 后，接下来需要一个易于处理的损失函数来学习尽可能接近目标 \(u_{t}\) 的速度场模型 \(u_{t}^{\theta}\)。直接给出该损失函数的主要障碍在于，计算目标 \(u_{t}\) 是不可行的，因为它需要对整个训练集进行边缘化（即对式（4.8）中的 \(x_{1}\) 或式（4.12）中的 \(z\) 进行积分）。幸运的是，一类被称为 Bregman 散度的损失函数能够提供无偏梯度，仅通过条件速度 \(u_{t}(x | z)\) 就能学习 \(u_{t}^{\theta}(x)\)。

Bregman 散度用于度量两个向量 \(u, v \in \mathbb{R}^{d}\) 之间的不相似性，其定义为：
\[
D(u, v):=\Phi(u)-[\Phi(v)+< u-v, \nabla \Phi(v)>], \quad(4.19)
\]
其中 \(\Phi: \mathbb{R}^{d} \to \mathbb{R}\) 是定义在凸集 \(\Omega \subset \mathbb{R}^{d}\) 上的严格凸函数。如图 10 所示，Bregman 散度衡量的是 \(\Phi(u)\) 与在 \(v\) 处展开并在 \(u\) 处求值的 \(\Phi\) 的线性近似之间的差异。由于线性近似是凸函数的全局下界，因此有 \(D(u, v) \geq 0\)。此外，由于 \(\Phi\) 是严格凸的，当且仅当 \(u=v\) 时，\(D(u, v)=0\)。最基本的 Bregman 散度是平方欧氏距离 \(D(u, v)=\|u-v\|^{2}\)，它由选择 \(\Phi(u)=\|u\|^{2}\) 得到。Bregman 散度对流匹配有用的关键性质是，其关于第二个参数的梯度具有仿射不变性（Holderrieth 等人, 2024）：
\[
\nabla_{v} D(a u_{1}+b u_{2}, v)=a \nabla_{v} D(u_{1}, v)+b \nabla_{v} D(u_{2}, v), \text{ 对任意 } a+b=1, \eqno (4.20)
\]
这一点可通过式（4.19）直接验证。仿射不变性允许我们如下交换期望和梯度的顺序：
\[
\nabla_{v} D(\mathbb{E}[Y], v)=\mathbb{E}\left[\nabla_{v} D(Y, v)\right] \text{ 对任意随机变量 } Y \in \mathbb{R}^{d}. (4.21)
\]

流匹配损失采用 Bregman 散度，沿着概率路径 \(p_{t}\) 将可学习速度 \(u_{t}^{\theta}(x)\) 回归到目标速度 \(u_{t}(x)\)：
\[
\mathcal{L}_{FM}(\theta)=\mathbb{E}_{t, X_{t} \sim p_{t}} D\left(u_{t}\left(X_{t}\right), u_{t}^{\theta}\left(X_{t}\right)\right), \quad(4.22)
\]
其中时间 \(t \sim U[0,1]\)（均匀分布）。然而，如前所述，目标速度 \(u_{t}\) 是难以处理的，因此上述损失无法直接计算。取而代之，我们考虑更简单且易于处理的条件流匹配（Conditional Flow Matching, CFM）损失：
\[
\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{t, Z, X_{t} \sim p_{t | Z}(\cdot | Z)} D\left(u_{t}\left(X_{t} | Z\right), u_{t}^{\theta}\left(X_{t}\right)\right) . (4.23)
\]

这两种损失在学习层面是等价的，因为它们的梯度是一致的（Holderrieth 等人, 2024）：

**定理 4**：流匹配损失和条件流匹配损失的梯度一致：
\[
\nabla_{\theta} \mathcal{L}_{FM}(\theta)=\nabla_{\theta} \mathcal{L}_{CFM}(\theta). (4.24)
\]
特别地，条件流匹配损失的极小值点是边缘速度 \(u_{t}(x)\)。

**证明**：证明过程如下直接计算所示：
\[
\begin{aligned}
\nabla_{\theta} \mathcal{L}_{FM}(\theta) & =\nabla_{\theta} \mathbb{E}_{t, X_{t} \sim p_{t}} D\left(u_{t}\left(X_{t}\right), u_{t}^{\theta}\left(X_{t}\right)\right) \\
& =\mathbb{E}_{t, X_{t} \sim p_{t}} \nabla_{\theta} D\left(u_{t}\left(X_{t}\right), u_{t}^{\theta}\left(X_{t}\right)\right) \\
& \stackrel{(i)}{=} \mathbb{E}_{t, X_{t} \sim p_{t}} \nabla_{v} D\left(u_{t}\left(X_{t}\right), u_{t}^{\theta}\left(X_{t}\right)\right) \nabla_{\theta} u_{t}^{\theta}\left(X_{t}\right) \\
& \stackrel{(4.12)}{=} \mathbb{E}_{t, X_{t} \sim p_{t}} \nabla_{v} D\left(\mathbb{E}_{Z \sim p_{Z | t}\left(\cdot | X_{t}\right)}\left[u_{t}\left(X_{t} | Z\right)\right], u_{t}^{\theta}\left(X_{t}\right)\right) \nabla_{\theta} u_{t}^{\theta}\left(X_{t}\right) \\
& \stackrel{(ii)}{=} \mathbb{E}_{t, X_{t} \sim p_{t}} \mathbb{E}_{Z \sim p_{Z | t}\left(\cdot | X_{t}\right)}\left[\nabla_{v} D\left(u_{t}\left(X_{t} | Z\right), u_{t}^{\theta}\left(X_{t}\right)\right) \nabla_{\theta} u_{t}^{\theta}\left(X_{t}\right)\right] \\
& \stackrel{(iii)}{=} \mathbb{E}_{t, X_{t} \sim p_{t}} \mathbb{E}_{Z \sim p_{Z | t}\left(\cdot | X_{t}\right)}\left[\nabla_{\theta} D\left(u_{t}\left(X_{t} | Z\right), u_{t}^{\theta}\left(X_{t}\right)\right)\right] \\
& \stackrel{(iv)}{=} \mathbb{E}_{t, Z, X_{t} \sim p_{t | Z}(\cdot | Z)} \nabla_{\theta} D\left(u_{t}\left(X_{t} | Z\right), u_{t}^{\theta}\left(X_{t}\right)\right) \\
& =\nabla_{\theta} \mathcal{L}_{CFM}(\theta),
\end{aligned}
\]
其中 (i) 和 (iii) 使用了链式法则；(ii) 由式（4.21）在 \(X_{t}\) 上条件应用得到；(iv) 使用了贝叶斯法则。

#### 用于学习条件期望的 Bregman 散度
定理 4 是一个更一般结果的特例，该结果利用 Bregman 散度学习条件期望，具体如下。这一结果将贯穿全文，为流匹配背后所有可扩展损失提供理论基础：

**命题 1（用于学习条件期望的 Bregman 散度）**：设 \(X \in S_{X}\)、\(Y \in S_{Y}\) 是定义在状态空间 \(S_{X}\)、\(S_{Y}\) 上的随机变量，\(g: \mathbb{R}^{p} ×S_{X} \to \mathbb{R}^{n}\)，\((\theta, x) \mapsto g^{\theta}(x)\)，其中 \(\theta \in \mathbb{R}^{p}\) 表示可学习参数。设 \(D_{x}(u, v)\)（\(x \in S_{X}\)）是定义在凸集 \(\Omega \subset \mathbb{R}^{n}\) 上的 Bregman 散度，且该凸集包含 \(f\) 的像。则：
\[
\nabla_{\theta} \mathbb{E}_{X, Y} D_{X}\left(Y, g^{\theta}(X)\right)=\nabla_{\theta} \mathbb{E}_{X} D_{X}\left(\mathbb{E}[Y | X], g^{\theta}(X)\right) . (4.25)
\]
特别地，对于所有满足 \(p_{X}(x)>0\) 的 \(x\)，\(g^{\theta}(x)\) 关于 \(\theta\) 的全局最小值满足：
\[
g^{\theta}(x)=\mathbb{E}[Y | X=x] . \quad(4.26)
\]

**证明**：假设 \(g^{\theta}\) 关于 \(\theta\) 可微，且 \(X\) 和 \(Y\) 的分布、\(D_{x}\) 以及 \(g\) 允许交换微分和积分的顺序，则有：
\[
\begin{aligned}
\nabla_{\theta} \mathbb{E}_{X, Y} D_{X}\left(Y, g^{\theta}(X)\right) & \stackrel{(i)}{=} \mathbb{E}_{X}\left[\mathbb{E}\left[\nabla_{v} D_{X}\left(Y, g^{\theta}(X)\right) \nabla_{\theta} g^{\theta}(X) | X\right]\right] \\
& \stackrel{(ii)}{=} \mathbb{E}_{X}\left[\nabla_{v} D_{X}\left(\mathbb{E}[Y | X], g^{\theta}(X)\right) \nabla_{\theta} g^{\theta}(X)\right] \\
& \stackrel{(iii)}{=} \mathbb{E}_{X}\left[\nabla_{\theta} D_{X}\left(\mathbb{E}[Y | X], g^{\theta}(X)\right)\right] \\
& =\nabla_{\theta} \mathbb{E}_{X} D_{X}\left(\mathbb{E}[Y | X], g^{\theta}(X)\right),
\end{aligned}
\]
其中 (i) 由链式法则和期望的塔性质（3.11）得到；等式 (ii) 由式（4.21）得到；等式 (iii) 再次使用了链式法则。最后，对于每个满足 \(p_{X}(x)>0\) 的 \(x \in S_{X}\)，我们可以选择 \(g^{\theta}(x)=\mathbb{E}[Y | X=x]\)，此时 \(\mathbb{E}_{X} D_{X}(\mathbb{E}[Y | X], g^{\theta}(X))=0\)，这必然是关于 \(\theta\) 的全局最小值。

令 \(X=X_{t}\)、\(Y=u_{t}(X_{t} | Z)\)、\(g^{\theta}(x)=u_{t}^{\theta}(x)\)，并对 \(t \sim U[0,1]\) 求期望，即可由命题 1 直接推出定理 4。

#### 一般时间分布
FM 损失的一个有用变体是从均匀分布以外的其他分布中采样时间 \(t\)。具体来说，考虑 \(t \sim \omega(t)\)，其中 \(\omega(t)\) 是 [0, 1] 上的概率密度函数（PDF）。这将得到以下加权目标函数：
\[
\mathcal{L}_{CFM}(\theta)=\mathbb{E}_{t \sim \omega, Z, X_{t}} D\left(u_{t}\left(X_{t} | Z\right), u_{t}^{\theta}\left(X_{t}\right)\right)=\mathbb{E}_{t \sim U, Z, X_{t}} \omega(t) D\left(u_{t}\left(X_{t} | Z\right), u_{t}^{\theta}\left(X_{t}\right)\right) .
\]
尽管在数学上是等价的，但在大规模图像生成任务中，从 \(\omega\) 中采样 \(t\) 比使用权重 \(\omega(t)\) 能获得更好的性能（Esser 等人, 2024）。

### 4.6 利用条件流解决条件生成问题
到目前为止，我们已将训练流模型 \(u_{t}^{\theta}\) 的问题简化为：(i) 找到条件概率路径 \(p_{t | Z}(x | z)\)，使其产生的边缘概率路径 \(p_{t}(x)\) 满足式（4.5）中的边界条件；(ii) 找到生成该条件概率路径的条件速度场 \(u_{t}(x | z)\)；(iii) 使用条件流匹配损失（见式（4.23））进行训练。现在，我们将讨论如何具体执行步骤 (i) 和 (ii)，即设计此类条件概率路径和速度场。

我们提出一种灵活的方法，通过条件流的特定构造来设计条件概率路径和速度场。其核心思想如下：定义一个满足边界条件（4.6）的流模型 \(X_{t | 1}\)（类似于式（3.16）），并通过微分（3.20）从 \(X_{t | 1}\) 中提取速度场。这一过程同时定义了 \(p_{t | 1}(x | x_{1})\) 和 \(u_{t}(x | x_{1})\)。更详细地说，定义条件流模型：
\[
X_{t | 1}=\psi_{t}(X_{0} | x_{1}), \text{ 其中 } X_{0}\sim \pi_{0 | 1}(\cdot | x_{1}), (4.28)
\]
其中 \(\psi:[0,1) ×\mathbb{R}^{d} ×\mathbb{R}^{d} \to \mathbb{R}^{d}\) 是一个条件流，定义为：
\[
\psi_{t}\left(x | x_{1}\right)= \begin{cases}x & t=0 \\ x_{1} & t=1\end{cases},
\]
该函数在 \((t, x)\) 上光滑，且在 \(x\) 上是微分同胚（diffeomorphism）。（这里的“光滑”指的是 \(\psi_{t}(x | x_{1})\) 关于 \(t\) 和 \(x\) 的所有导数都存在且连续，即 \(C^{\infty}([0,1) ×\mathbb{R}^{d}, \mathbb{R}^{d})\)。为简化起见，这些条件可以进一步放宽到 \(C^{2}([0,1) ×\mathbb{R}^{d}, \mathbb{R}^{d})\)。）推送算子公式（3.15）将 \(X_{t | 1}\) 的概率密度定义为：
\[
p_{t | 1}\left(x | x_{1}\right):=\left[\psi_{t}\left(\cdot | x_{1}\right)_{\sharp} \pi_{0 | 1}\left(\cdot | x_{1}\right)\right](x), \quad(4.30)
\]
尽管在 CFM 损失的实际优化中我们不需要这个表达式，但它在理论上用于证明 \(p_{t | 1}\) 满足两个边界条件（4.6）。首先，根据式（4.29），\(\psi_{0}(\cdot | x_{1})\) 是恒等映射，在时间 \(t=0\) 时保持 \(\pi_{0 | 1}(\cdot | x_{1})\) 不变。其次，\(\psi_{1}(\cdot | x_{1})=x_{1}\) 是常数映射，当 \(t \to 1\) 时将所有概率质量集中在 \(x_{1}\) 处。此外，注意到对于 \(t \in[0,1)\)，\(\psi_{t}(\cdot | x_{1})\) 是光滑的微分同胚。因此，根据流和速度场的等价性（3.4.1 节），存在唯一的光滑条件速度场（见式（3.20）），其形式为：
\[
u_{t}\left(x | x_{1}\right)=\dot{\psi}_{t}\left(\psi_{t}^{-1}\left(x | x_{1}\right) | x_{1}\right) . (4.31)
\]

总结：我们已将寻找条件路径和相应生成速度的任务进一步简化为构建满足式（4.29）的条件流 \(\psi_{t}(\cdot | x_{1})\)。在 4.7 节中，我们将选择一个特别简单且具有理想性质的 \(\psi_{t}(x | x_{1})\)（条件最优传输流），它将导出 1 节中介绍的标准流匹配算法；在 4.8 节中，我们将讨论一类特定且知名的条件流，即仿射流（affine flows），其中包括扩散模型文献中的一些已知示例。在 5 节中，我们将利用条件流在流形上定义流匹配，以展示该方法的灵活性。

#### 重新审视条件流匹配损失
通过令 \(Z=X_{1}\)，并利用条件流定义条件概率路径和速度的方式，重新审视 CFM 损失（4.23）：
\[
\begin{aligned}
\mathcal{L}_{CFM}(\theta) & =\mathbb{E}_{t, X_{1}, X_{t} \sim p_{t}\left(\cdot | X_{1}\right)} D\left(u_{t}\left(X_{t} | X_{1}\right), u_{t}^{\theta}\left(X_{t}\right)\right) \\
& \stackrel{(3.4)}{=} \mathbb{E}_{t,\left(X_{0}, X_{1}\right) \sim \pi_{0,1}} D\left(\dot{\psi}_{t}\left(X_{0} | X_{1}\right), u_{t}^{\theta}\left(X_{t}\right)\right)
\end{aligned}
\]
其中第二个等式利用了无意识统计学家法则（Law of the Unconscious Statistician），且 \(X_{t}=\psi_{t}(X_{0} | X_{1})\)，同时：
\[
u_{t}\left(X_{t} | X_{1}\right) \stackrel{(4.31)}{=} \dot{\psi}_{t}\left(\psi_{t}^{-1}\left(\psi_{t}\left(X_{0} | X_{1}\right) | X_{1}\right) | X_{1}\right)=\dot{\psi}_{t}\left(X_{0} | X_{1}\right) . (4.33)
\]

根据命题 1，损失（4.32）的极小值点具有 Liu 等人（2022）中的形式：
\[
u_{t}(x)=\mathbb{E}\left[\dot{\psi}_{t}\left(X_{0} | X_{1}\right) \mid X_{t}=x\right] .
\]

在 flow_matching 库中，ProbPath 对象定义了一条概率路径。该概率路径可以在 \((t, X_{0}, X_{1})\) 处采样，以获得 \(X_{t}\) 和 \(\dot{\psi}_{t}(X_{0} | X_{1})\)。然后，可以计算 CFM 损失 \(L_{CFM}(\theta)\) 的蒙特卡洛估计。代码 4 展示了一个使用 CFM 目标函数的训练循环示例。

**代码 4：使用条件流匹配（CFM）损失进行训练**
```python
import torch
from flow_matching.path import ProbPath
from flow_matching.path.path_sample import PathSample

path: ProbPath = ...  ## flow_matching 库实现了最常用的概率路径
velocity_model: torch.nn.Module = ...  ## 初始化速度模型
optimizer = torch.optim.Adam(velocity_model.parameters())

for x_0, x_1 in dataloader:  ## 从 π0,1 采样，形状为 [batch_size, *data_dim]
    t = torch.rand(batch_size)  ## 随机化时间 t ∼ U[0, 1]
    sample: PathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
    x_t = sample.x_t
    dx_t = sample.dx_t  ## dX_t 是 ˙ψt(X0|X1)
    ## 如果 D 是欧氏距离，则 CFM 目标函数对应于均方误差
    cfm_loss = torch.pow(velocity_model(x_t, t) - dx_t, 2).mean()  ## 蒙特卡洛估计
    optimizer.zero_grad()
    cfm_loss.backward()
    optimizer.step()
```

#### 基于条件流构建的概率路径的边缘化技巧
接下来，我们介绍一种适用于基于条件流构建的概率路径的边缘化技巧。为此，需要注意的是，如果 \(\pi_{0 | 1}(\cdot | x_{1})\) 是 \(C^{1}\) 光滑的，则 \(p_{t}(x | x_{1})\) 通过构造也是 \(C^{1}\) 光滑的；此外，如果：
\[
\mathbb{E}_{t,\left(X_{0}, X_{1}\right) \sim \pi_{0,1}}\left\| \dot{\psi}_{t}\left(X_{0} | X_{1}\right)\right\| <\infty . (4.35)
\]
则 \(u_{t}(x | x_{1})\) 是条件可积的。

因此，令 \(Z=X_{1}\)，可得到定理 3 的如下推论：

**推论 1**：假设 \(q\) 具有有界支撑，\(\pi_{0 | 1}(\cdot | x_{1})\) 是 \(C^{1}(\mathbb{R}^{d})\) 光滑的且对于某些满足 \(q(x_{1})>0\) 的 \(x_{1}\) 严格为正，且 \(\psi_{t}(x | x_{1})\) 是满足式（4.29）和式（4.35）的条件流。则式（4.30）和式（4.31）中分别定义的 \(p_{t | 1}(x | x_{1})\) 和 \(u_{t}(x | x_{1})\) 定义了一个边缘速度场 \(u_{t}(x)\)，该速度场生成插值 \(p\) 和 \(q\) 的边缘概率路径 \(p_{t}(x)\)。

**证明**：如果对于某些满足 \(q(x_{1})>0\) 的 \(x_{1} \in \mathbb{R}^{d}\)，\(\pi_{0 | 1}(\cdot | x_{1})>0\)，则对于所有 \(x \in \mathbb{R}^{d}\)，\(p_{t | 1}(x | x_{1})>0\) 且是 \(C^{1}([0,1) ×\mathbb{R}^{d})\) 光滑的（见式（4.30）和式（3.15）的定义）。此外，式（4.31）中定义的 \(u_{t}(x | x_{1})\) 是光滑的，且满足：
\[
\begin{aligned}
\int_{0}^{1} \int\left\| u_{t}\left(x | x_{1}\right)\right\| p_{t | 1}\left(x | x_{1}\right) q\left(x_{1}\right) d x_{1} d x d t & =\mathbb{E}_{t, X_{1} \sim q, X_{t} \sim p_{t | 1}\left(\cdot | X_{1}\right)}\left\| u_{t}\left(X_{t} | X_{1}\right)\right\| \\
& \stackrel{(3.4)}{=} \mathbb{E}_{t, X_{1} \sim q, X_{0} \sim \pi_{0 | 1}\left(\cdot | X_{1}\right)}\left\| u_{t}\left(\psi_{t}\left(X_{0} | X_{1}\right) | X_{1}\right)\right\| \\
& \stackrel{(4.33)}{=} \mathbb{E}_{t,\left(X_{0}, X_{1}\right) \sim \pi_{0,1}}\left\| \dot{\psi}_{t}\left(X_{0} | X_{1}\right)\right\| \\
& <\infty .
\end{aligned}
\]
因此，\(u_{t}(x | x_{1})\) 是条件可积的（见式（4.13））。根据定理 3，边缘速度场 \(u_{t}\) 生成 \(p_{t}\)。由于式（4.30）中定义的 \(p_{t | 1}(x | x_{1})\) 满足式（4.6），因此 \(p_{t}\) 插值 \(p\) 和 \(q\)。

该定理将用作证明特定条件流选择能产生生成边缘概率路径 \(p_{t}(x)\) 的边缘速度 \(u_{t}(x)\) 的工具。

#### 其他条件的条件流
存在多种不同的条件化选择 \(Z\)，但本质上它们都是等价的。如图 11 所示，主要选项包括固定目标样本 \(Z=X_{1}\)（Lipman 等人, 2022）、固定源样本 \(Z=X_{0}\)（Esser 等人, 2024），或双向条件化 \(Z=(X_{0}, X_{1})\)（Albergo 和 Vanden-Eijnden, 2022；Liu 等人, 2022；Pooladian 等人, 2023；Tong 等人, 2023）。

我们重点关注双向条件化 \(Z=(X_{0}, X_{1})\)。按照上述 FM 蓝图，我们现在希望构建条件概率路径 \(p_{t | 0,1}(x | x_{0}, x_{1})\) 和相应的生成速度 \(u_{t}(x | x_{0}, x_{1})\)，使得：
\[
p_{0|0,1}(x|x_{0},x_{1})=\delta_{x_{0}}(x), \text{ 且 } p_{1|0,1}(x|x_{0},x_{1})=\delta_{x_{1}}(x).\eqno (4.36)
\]

我们将保持这一讨论的形式化，因为它需要使用狄拉克函数 \(\delta\)，而我们之前的推导仅涉及概率密度（而非一般分布）。为了构建这样的路径，我们可以考虑一个插值函数（Albergo 和 Vanden-Eijnden, 2022），其定义为 \(X_{t | 0,1}=\psi_{t}(x_{0}, x_{1})\)，其中函数 \(\psi:[0,1] ×\mathbb{R}^{d} ×\mathbb{R}^{d} \to \mathbb{R}^{d}\) 满足类似于式（4.29）的条件：
\[
\psi_{t}\left(x_{0}, x_{1}\right)= \begin{cases}x_{0} & t=0 \\ x_{1} & t=1 .\end{cases}
\]
因此，\(\psi_{t}(\cdot, x_{1})\) 将 \(\delta_{x_{0}}(x)\) 推送至 \(\delta_{x_{1}}(x)\)。类似地，我们定义条件概率路径为：
\[
p_{t | 0,1}\left(\cdot | x_{0}, x_{1}\right):=\psi_{t}\left(\cdot, x_{1}\right)_{\sharp} \delta_{x_{0}}(\cdot),
\]
该路径满足式（4.36）中的边界约束。Albergo 和 Vanden-Eijnden（2022）的随机插值函数定义为：
\[
X_{t}=\psi_{t}\left(X_{0}, X_{1}\right) \sim p_{t}(\cdot)=\int p_{t | 0,1}\left(\cdot | x_{0}, x_{1}\right) \pi_{0,1}\left(x_{0}, x_{1}\right) d x_{0} d x_{1} . (4.39)
\]
接下来，该路径上的条件速度也可以通过式（3.20）计算得到：
\[
u_{t}\left(x | x_{0}, x_{1}\right)=\dot{\psi}_{t}\left(x_{0}, x_{1}\right) \quad(4.40)
\]
该速度仅在 \(x=\psi_{t}(x_{0}, x_{1})\) 处有定义。暂时忽略额外条件，定理 3 似乎意味着生成 \(p_{t}(x)\) 的边缘速度为：
\[
\begin{aligned}
u_{t}(x) & =\mathbb{E}\left[u_{t}\left(X_{t} | X_{0}, X_{1}\right) | X_{t}=x\right] \\
& =\mathbb{E}\left[\dot{\psi}_{t}\left(X_{0}, X_{1}\right) | X_{t}=x\right],
\end{aligned}
\]
这与 \(X_{1}\) 条件化情形（4.34）得到的边缘公式相同，但此时条件流 \(\psi_{t}(x_{0}, x_{1})\) 仅需是插值函数，放宽了更严格的微分同胚条件。然而，更仔细的分析表明，要使 \(u_{t}(x)\) 成为 \(p_{t}(x)\) 的生成速度，还需要一些额外条件，仅靠简单的插值（如式（4.37）所定义）是不够的，即使加上额外的光滑性条件（如定理 3 所要求的）也不行。为了说明这一点，考虑：
\[
\psi_{t}\left(x_{0}, x_{1}\right)=(1-2 t)_{+}^{\tau} x_{0}+(2 t-1)_{+}^{\tau} x_{1}, \text{ 其中 } (s)_{+}=ReLU(s), \tau>2,
\]
这是一个在时间上 \(C^{2}([0,1])\) 光滑的插值函数，对于所有 \(x_{0}\)、\(x_{1}\)，在时间 \(t=0.5\) 时将所有概率质量集中在位置 0 处。即 \(\mathbb{P}(X_{\frac{1}{2}}=0)=1\)。因此，假设 \(u_{t}(x)\) 确实生成 \(p_{t}(x)\)，则其在 \(t=\frac{1}{2}\) 处的边缘分布为 \(\delta_{0}\)；并且由于流既是马尔可夫的（如式（3.17）所示）又是确定性的，其在所有 \(t>0.5\) 处的边缘分布都必须是狄拉克函数，这与 \(X_{1}=\psi_{1}(X_{0}, X_{1}) \sim q\)（通常不是狄拉克函数）矛盾。Albergo 和 Vanden-Eijnden（2022）以及 Liu 等人（2022）提供了一些额外条件，以保证 \(u_{t}(x)\) 确实生成 \(p_{t}(x)\)，但这些条件与定理 3 的条件相比更难验证。下面我们将展示如何实际检查定理 3 的条件，以验证特定感兴趣的路径是否确实由相应的边缘速度生成。

尽管如此，当 \(\psi_{t}(x_{0}, x_{1})\) 对于固定的 \(x_{1}\) 在 \(x_{0}\) 上是微分同胚，且对于固定的 \(x_{0}\) 在 \(x_{1}\) 上也是微分同胚时，这三种构建方式将得到相同的边缘速度（由式（4.34）定义）和相同的边缘概率路径 \(p_{t}\)（由 \(X_{t}=\psi_{t}(X_{0}, X_{1})=\psi_{t}(X_{0} | X_{1})=\psi_{t}(X_{1} | X_{0})\) 定义），如图 11 所示。

### 4.7 最优传输与线性条件流
现在我们要问：如何找到有用的条件流 \(\psi_{t}(x | x_{1}) ?\) 一种方法是将其选择为某个自然成本泛函的极小值点，理想情况下该泛函应具有一些理想性质。此类成本泛函的一个常见示例是带二次成本的动态最优传输（dynamic Optimal Transport）问题（Villani 等人, 2009；Villani, 2021；Peyré 等人, 2019），其形式化为：
\[
\left( p_{t}^{* },u_{t}^{* }\right) =\underset {p_{t},u_{t}}{arg min }\int _{0}^{1}\int \left\| u_{t}(x)\right\| ^{2}p_{t}(x)dxdt \quad \text{（动能）} (4.41a)
\]
\[
s.t. p_{0}=p, p_{1}=q \quad \text{（插值）} (4.41b)
\]
\[
\frac{d}{d t} p_{t}+div\left(p_{t} u_{t}\right)=0 . \quad \text{（连续性方程）} (4.41c)
\]
上述 \((p_{t}^{*}, u_{t}^{*})\) 通过式（3.19）定义了一个流，其形式为：
\[
\psi_{t}^{*}(x)=t \phi(x)+(1-t) x, \quad(4.42)
\]
该流被称为 OT 位移插值函数（OT displacement interpolant）（McCann, 1997），其中 \(\phi: \mathbb{R}^{d} \to \mathbb{R}^{d}\) 是最优传输映射（Optimal Transport map）。OT 位移插值函数还通过定义随机变量 \(X_{t}=\psi_{t}^{*}(X_{0}) \sim p_{t}^{*}\)（当 \(X_{0} \sim p\) 时）解决了流匹配问题（4.1）。

最优传输公式促进了样本的直线轨迹：
\[
X_{t}=\psi_{t}^{*}(X_{0})=X_{0}+t(\phi(X_{0})-X_{0}),
\]
该轨迹具有恒定速度 \(\phi(X_{0})-X_{0}\)，通常更容易使用 ODE 求解器进行采样——特别是，目标样本 \(X_{1}\) 在这里可以通过欧拉方法（3.21）的单步完美求解。

现在，我们尝试将边缘速度公式（式（4.34））代入最优传输问题（4.41），并寻找最优的 \(\psi_{t}(x | x_{1})\)。尽管这看起来具有挑战性，但我们可以转而寻找动能的一个界，该界的极小值点很容易找到（Liu 等人, 2022）：
\[
\int_{0}^{1} \mathbb{E}_{X_{t} \sim p_{t}}\left\| u_{t}\left(X_{t}\right)\right\| ^{2} d t=\int_{0}^{1} \mathbb{E}_{X_{t} \sim p_{t}}\left\| \mathbb{E}\left[\dot{\psi}_{t}\left(X_{0} | X_{1}\right) | X_{t}\right]\right\| ^{2} d t
\]
\[
\stackrel{(i)}{\leq} \int_{0}^{1} \mathbb{E}_{X_{t} \sim p_{t}} \mathbb{E}\left[\left\| \dot{\psi}_{t}\left(X_{0} | X_{1}\right)\right\| ^{2} | X_{t}\right] d t (4.45)
\]
\[
\stackrel{(ii)}{=} \mathbb{E}_{\left(X_{0}, X_{1}\right) \sim \pi_{0,1}} \int_{0}^{1}\left\| \dot{\psi}_{t}\left(X_{0} | X_{1}\right)\right\| ^{2} d t,
\]

其中不等式 (i) 由 Jensen 不等式给出，等式 (ii) 由交换期望与积分顺序得到。这一边界的极小值点可以通过选择**线性条件流**实现：
\[
\psi_t(x_0 \mid x_1) = (1-t)x_0 + t x_1,
\]
该式定义了从 \(x_0\) 到 \(x_1\) 的线性插值。对时间求导得到条件速度：
\[
\dot\psi_t(x_0 \mid x_1) = x_1 - x_0.
\]
将其代入边界 (4.45)，得到：
\[
\mathbb{E}_{(X_0,X_1)\sim\pi_{0,1}} \int_0^1 \|X_1-X_0\|^2 dt
= \mathbb{E}_{(X_0,X_1)} \|X_1-X_0\|^2,
\]
该式与时间无关，因此是常数。这表明线性条件流能最小化动能上界，从而得到一种简单、稳定且广泛使用的 Flow Matching 形式。

将线性流代入条件速度的定义，得到：
\[
u_t(x \mid x_1) = x_1 - x_0 = x_1 - \frac{x-tx_1}{1-t}
= \frac{x_1 - x}{1-t}.
\]
对应的边缘速度场为：
\[
u_t(x) = \mathbb{E}\left[\frac{X_1 - X_t}{1-t} \,\Big|\, X_t = x\right],
\]
这正是经典 Flow Matching 所使用的速度场。

### 4.8 仿射条件流与扩散模型

我们现在考虑一类更广泛的条件流：**仿射条件流**，其形式为：
\[
\psi_t(x_0 \mid x_1) = \alpha_t x_0 + \beta_t x_1,
\]
其中 \(\alpha_t, \beta_t\) 是满足边界条件的标量函数：
\[
\alpha_0=1,\ \beta_0=0,\quad \alpha_1=0,\ \beta_1=1.
\]
对时间求导得到条件速度：
\[
\dot\psi_t(x_0 \mid x_1) = \dot\alpha_t x_0 + \dot\beta_t x_1.
\]
利用 \(x_0 = \frac{\psi_t(x_0\mid x_1)-\beta_t x_1}{\alpha_t}\)，可以将条件速度表示为当前状态 \(x=\psi_t(x_0\mid x_1)\) 的函数：
\[
u_t(x\mid x_1)
= \dot\alpha_t \cdot \frac{x-\beta_t x_1}{\alpha_t} + \dot\beta_t x_1
= \frac{\dot\alpha_t}{\alpha_t}x + \left(\dot\beta_t - \frac{\dot\alpha_t \beta_t}{\alpha_t}\right)x_1.
\]
这是关于 \(x\) 和 \(x_1\) 的仿射函数，因此被称为**仿射条件流**。

许多经典扩散模型与采样方法都可以被纳入这一框架：
- 当选择 \(\alpha_t = e^{-\int_0^t \beta_s ds}\)、\(\beta_t = 1-\alpha_t\) 时，对应**方差爆炸型扩散模型**。
- 当选择相应的积分尺度与噪声调度时，EDM、DDPM、DDIM 等都可以表示为特定仿射流下的 Flow Matching。

这表明，Flow Matching 不仅是一种独立的生成建模方法，更是一个**统一框架**，可以将扩散模型、流模型、最优传输模型纳入同一套理论体系中。

### 4.9 流匹配与分数匹配的联系

分数模型（Score Matching）学习的是数据分布的对数梯度 \(\nabla\log q(x)\)。我们可以证明，Flow Matching 在特定路径下会**等价于学习分数函数**。

考虑概率路径为**带噪声的数据分布**：
\[
p_t(x) = \int q(x_1) \mathcal{N}(x \mid x_1, \sigma_t^2 I) dx_1,
\]
其中 \(\sigma_t\) 是随时间衰减的噪声水平。对应的速度场可以写成：
\[
u_t(x) = -\sigma_t \nabla\log p_t(x) + \dot\sigma_t \frac{x-\mathbb{E}[X_1\mid X_t=x]}{\sigma_t}.
\]
当噪声满足 \(\dot\sigma_t = 0\) 或第二项可以忽略时，Flow Matching 就退化为**分数匹配与去噪分数匹配**。

这一结论揭示了：
- 扩散模型本质上是在学习**分数函数**；
- Flow Matching 是更广义的框架，分数匹配只是其中一个特例。

### 4.10 本章小结

本章系统介绍了 **Flow Matching** 的完整理论框架：
1. 将生成建模转化为**学习概率路径的速度场**；
2. 通过**条件化 + 边缘化技巧**，将难以计算的边缘速度场转化为可训练的条件目标；
3. 证明 **Flow Matching 损失与条件 Flow Matching 损失梯度等价**，从而可以高效训练；
4. 引入**条件流**作为统一构造方式，覆盖线性流、仿射流、最优传输流；
5. 将扩散模型、分数模型、OT 流模型统一在同一套理论下。

Flow Matching 兼具**训练稳定、采样快速、理论清晰**的优点，已成为当前连续时间生成模型的主流训练范式之一。

---
