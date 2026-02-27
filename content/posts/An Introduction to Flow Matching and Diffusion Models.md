+++
date = '2026-02-27T08:40:58+08:00'
draft = false
title = 'An Introduction to Flow Matching and Diffusion Models'
+++



# 2 Flow and Diffusion Models
## 2.1 Flow Models(ODE)
我们首先从常微分方程（ODE）的定义入手。ODE 的解是一条轨迹，形式为：
$X:[0,1] \to \mathbb{R}^d, \quad t \mapsto X_t$
该轨迹将时间 $t$ 映射到空间 $\mathbb{R}^d$ 中的某个位置。每个 ODE 由一个向量场 $u$ 定义，向量场的形式为：
$u:\mathbb{R}^d \times [0,1] \to \mathbb{R}^d, \quad (x,t) \mapsto u_t(x)$
即对于每个时间 $t$ 和空间位置 $x$，向量场会给出一个向量 $u_t(x) \in \mathbb{R}^d$，该向量指定了空间中的瞬时速度方向（见图 1）。ODE 对轨迹施加了一个约束：我们需要找到一条轨迹 $X$，它能“沿着”向量场 $u_t$ 的方向演化，并从初始点 $x_0$ 出发。这条轨迹可形式化为以下方程的解：
$$
\frac{d}{dt} X_t = u_t(X_t) \tag{1a} \quad \text{（ODE 核心方程）}
$$
$$
X_0 = x_0 \tag{1b} \quad \text{（初始条件）}
$$

方程 (1a) 要求轨迹 $X_t$ 的导数由向量场 $u_t$ 给出的方向决定；方程 (1b) 要求轨迹在时间 $t=0$ 时从初始点 $x_0$ 出发。由此我们可以提出一个关键问题：若在 $t=0$ 时从 $X_0=x_0$ 出发，那么在时间 $t$ 时轨迹会到达何处（即 $X_t$ 的值是多少）？这个问题的答案由一个名为“流”的函数给出，该函数是 ODE 的解，形式为：
$$
\psi: \mathbb{R}^d \times [0,1] \mapsto \mathbb{R}^d, \quad (x_0, t) \mapsto \psi_t(x_0) \tag{2a}
$$
$$
\frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0)) \tag{2b} \quad \text{（流的 ODE 方程）}
$$
$$
\psi_0(x_0) = x_0 \tag{2c} \quad \text{（流的初始条件）}
$$

对于给定的初始条件 $X_0=x_0$，ODE 的轨迹可通过 $X_t = \psi_t(X_0)$ 恢复。因此，向量场、ODE 和流在直观上是对同一对象的三种描述：向量场定义了 ODE，而 ODE 的解就是流。

与所有方程一样，我们需要关注 ODE 的解是否存在且唯一。数学中的一个基本结论给出了肯定答案，只要对向量场 $u_t$ 施加较弱的假设：

### 定理 3（流的存在性与唯一性）
若向量场 $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ 是连续可微的且导数有界，则方程 (2) 定义的 ODE 存在唯一解，该解由流 $\psi_t$ 给出。此时，对于所有时间 $t$，$\psi_t$ 是一个微分同胚，即 $\psi_t$ 是连续可微的，且其逆函数 $\psi_t^{-1}$ 也是连续可微的。

需要注意的是，在机器学习中，我们使用神经网络对 $u_t(x)$ 进行参数化，而神经网络的导数始终有界，因此流的存在性和唯一性假设几乎总能满足。这一结论对我们而言是利好消息：在我们关注的生成建模场景中，流是存在且唯一的 ODE 解（相关证明可参考 [20, 4]）。

### 示例 4（线性向量场）
考虑一个简单的线性向量场 $u_t(x) = -\theta x$（其中 $\theta > 0$），则函数：
$$
\psi_t(x_0) = \exp(-\theta t) x_0 \tag{3}
$$
定义了一个满足方程 (2) 的流。我们可以自行验证：首先，$\psi_0(x_0) = x_0$ 满足初始条件；其次，对 $\psi_t(x_0)$ 求导：
$$
\frac{d}{dt} \psi_t(x_0) \stackrel{(3)}{=} \frac{d}{dt}\left(\exp(-\theta t) x_0\right) \stackrel{(i)}{=} -\theta \exp(-\theta t) x_0 \stackrel{(3)}{=} -\theta \psi_t(x_0) = u_t(\psi_t(x_0))
$$
其中步骤 (i) 使用了链式法则。图 3 展示了这种形式的流，其轨迹会指数级收敛到原点。

### ODE 的模拟
一般而言，若向量场 $u_t$ 不像线性函数那样简单，我们无法显式计算流 $\psi_t$，此时需要使用数值方法来模拟 ODE。幸运的是，这是数值分析中一个经典且研究充分的领域，存在多种高效的数值方法 [11]。其中最简单、最直观的是欧拉法（Euler method）：初始化 $X_0 = x_0$，然后通过以下公式迭代更新：
$$
X_{t+h} = X_t + h u_t(X_t) \quad (t=0, h, 2h, 3h, \dots, 1-h) \tag{4}
$$
其中 $h = n^{-1} > 0$ 是步长超参数（$n \in \mathbb{N}$ 为总步数）。对于本课程而言，欧拉法已能满足需求。为了让你了解更复杂的数值方法，我们简要介绍休恩法（Heun’s method），其更新规则如下：
1. 初始猜测新状态：$X_{t+h}' = X_t + h u_t(X_t)$
2. 利用当前状态和猜测状态的向量场平均值更新：
$$
X_{t+h} = X_t + \frac{h}{2}\left(u_t(X_t) + u_{t+h}(X_{t+h}')\right)
$$
直观来看，休恩法的核心是：先对下一步状态做出初步猜测，再通过该猜测修正初始方向，从而获得更准确的更新结果。

### Flow Models的定义与采样
我们现在可以通过 ODE 构建生成模型。回顾生成建模的核心目标：将简单分布 $p_{\text{init}}$ 转换为复杂的目标分布 $p_{\text{data}}$。ODE 的模拟自然适合这一转换任务——流模型就是通过神经网络参数化向量场 $u_t^\theta$ 构建的 ODE 系统，形式为：
$$
X_0 \sim p_{\text{init}} \quad \text{（随机初始化）}
$$
$$
\frac{d}{dt} X_t = u_t^\theta(X_t) \quad \text{（ODE 核心方程）}
$$
其中 $u_t^\theta$ 是带有参数 $\theta$ 的神经网络。目前，我们可将 $u_t^\theta$ 视为一个通用的神经网络，即一个连续函数 $u_t^\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$；后续章节将详细讨论具体的神经网络架构设计。

流模型的核心目标是让轨迹的终点 $X_1$ 服从目标数据分布 $p_{\text{data}}$，即：
$$
X_1 \sim p_{\text{data}} \quad \Leftrightarrow \quad \psi_t^\theta(X_0) \sim p_{\text{data}}
$$
其中 $\psi_t^\theta$ 表示由向量场 $u_t^\theta$ 诱导的流。需要注意的是：尽管该模型被称为“流模型”，但神经网络实际参数化的是向量场而非流本身——流需要通过模拟 ODE 来计算。

### 算法 1：基于欧拉法的流模型采样流程
**输入**：神经网络向量场 $u_t^\theta$、采样步数 $n$
1. 初始化时间 $t = 0$
2. 设置步长 $h = \frac{1}{n}$
3. 从先验分布 $p_{\text{init}}$ 中采样初始状态 $X_0$
4. 对于 $i = 1, 2, \dots, n-1$：
   - 计算 $X_{t+h} = X_t + h \cdot u_t^\theta(X_t)$
   - 更新时间 $t \leftarrow t + h$
5. 输出终点 $X_1$（即从 $p_{\text{data}}$ 中采样的样本）

## 2.2 Diffusion Models(SDE)
随机微分方程（SDE）在 ODE 的确定性轨迹基础上，引入了随机轨迹，这类随机轨迹通常被称为随机过程 $(X_t)_{0 \leq t \leq 1}$。其核心特征是：对于每个 $0 \leq t \leq 1$，$X_t$ 是一个随机变量；而映射 $t \mapsto X_t$ 是一条随机轨迹——即使模拟同一个 SDE 两次，也可能得到不同的结果，因为其动力学过程包含随机性。

### 布朗运动（Brownian Motion）
SDE 的构建依赖于布朗运动——这是一个源于物理扩散过程研究的基础随机过程，可理解为连续时间下的随机游走。布朗运动的严格定义如下：

一个布朗运动 $W = (W_t)_{0 \leq t \leq 1}$ 是满足以下条件的随机过程：
1. 初始条件：$W_0 = 0$
2. 轨迹连续性：映射 $t \mapsto W_t$ 是连续的
3. 正态增量：对于所有 $0 \leq s < t$，增量 $W_t - W_s \sim \mathcal{N}(0, (t-s) I_d)$（即增量服从高斯分布，方差与时间差成正比，$I_d$ 为单位矩阵）
4. 独立增量：对于任意 $0 \leq t_0 < t_1 < \dots < t_n = 1$，增量 $W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}}$ 是相互独立的随机变量

布朗运动也被称为维纳过程（Wiener process），这也是我们用字母“W”表示它的原因。我们可以通过步长 $h > 0$ 近似模拟布朗运动：初始化 $W_0 = 0$，然后通过以下公式迭代更新：
$$
W_{t+h} = W_t + \sqrt{h} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad (t=0, h, 2h, \dots, 1-h) \tag{5}
$$
图 2 展示了一维布朗运动（$d=1$）的几条示例轨迹。

布朗运动在随机过程研究中的地位，堪比高斯分布在概率论中的地位，其应用范围远超机器学习，涵盖金融、统计物理、流行病学等多个领域。例如，在金融领域，布朗运动被用于建模复杂金融工具的价格波动；从纯数学角度来看，布朗运动也极具魅力——其轨迹是连续的（可以不抬笔一笔画完），但长度却是无限的（永远无法画完）。

### 从 ODE 到 SDE 的扩展
SDE 的核心思想是在 ODE 的确定性动力学基础上，加入由布朗运动驱动的随机动力学。由于 SDE 包含随机性，我们无法再像 ODE 那样直接使用导数形式（式 (1a)），因此需要找到一种不依赖导数的等价表述。

首先，我们将 ODE 的轨迹 $(X_t)_{0 \leq t \leq 1}$ 重写为：
$$
\frac{d}{dt} X_t = u_t(X_t) \quad \text{（导数形式）}
$$
$$
\stackrel{(i)}{\Leftrightarrow} \frac{1}{h}(X_{t+h} - X_t) = u_t(X_t) + R_t(h) \quad \Leftrightarrow \quad X_{t+h} = X_t + h u_t(X_t) + h R_t(h)
$$
其中 $R_t(h)$ 是小步长 $h$ 下的可忽略函数（即 $\lim_{h \to 0} R_t(h) = 0$），步骤 (i) 直接使用了导数的定义。上述推导本质上重申了 ODE 轨迹的核心特征：在每个时间步，轨迹会沿着向量场 $u_t(X_t)$ 的方向迈出一小步。

基于此，我们可以将 ODE 扩展为 SDE：SDE 的轨迹在每个时间步不仅会沿着 $u_t(X_t)$ 的方向迈出确定性一步，还会叠加布朗运动带来的随机步，形式为：
$$
X_{t+h} = X_t + \underbrace{h u_t(X_t)}_{确定性项} + \sigma_t \underbrace{(W_{t+h} - W_t)}_{随机项} + \underbrace{h R_t(h)}_{误差项} \tag{6}
$$
其中：
- $\sigma_t \geq 0$ 是扩散系数，控制随机项的强度
- $R_t(h)$ 是随机误差项，满足 $\mathbb{E}[\|R_t(h)\|^2]^{1/2} \to 0$（当 $h \to 0$ 时）

上述方程即为随机微分方程（SDE），其标准符号表示为：
$$
dX_t = u_t(X_t) dt + \sigma_t dW_t \tag{7a} \quad \text{（SDE 核心方程）}
$$
$$
X_0 = x_0 \tag{7b} \quad \text{（初始条件）}
$$
需要注意的是，上述符号中的 $dX_t$ 是式 (6) 的简化表示，不具备严格的数学导数含义。

与 ODE 不同，SDE 不再存在流映射 $\phi_t$——因为轨迹的演化包含随机性，终点 $X_t$ 无法仅由初始状态 $X_0 \sim p_{\text{init}}$ 完全确定。但与 ODE 类似，SDE 的解也满足存在性和唯一性：

### 定理 5（SDE 解的存在性与唯一性）
若向量场 $u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ 是连续可微的且导数有界，且扩散系数 $\sigma_t$ 是连续的，则方程 (7) 定义的 SDE 存在唯一解，该解由随机过程 $(X_t)_{0 \leq t \leq 1}$ 给出（满足式 (6)）。

若这是一门随机微积分课程，我们会用多个课时严格证明该定理、从基础构造布朗运动，并通过随机积分构建过程 $X_t$。但由于本课程聚焦机器学习，我们仅提供核心结论，感兴趣的读者可参考 [18] 进行更深入的技术学习。

最后需要强调的是：每个 ODE 都可以视为 SDE 的特例——只需令扩散系数 $\sigma_t = 0$（即移除随机项）。因此，在后续章节中，当我们讨论 SDE 时，默认将 ODE 包含在内作为特殊情况。

### 示例 6（奥恩斯坦-乌伦贝克过程，Ornstein-Uhlenbeck Process）
考虑常数扩散系数 $\sigma_t = \sigma \geq 0$ 和常数线性漂移项 $u_t(x) = -\theta x$（其中 $\theta > 0$），对应的 SDE 为：
$$
dX_t = -\theta X_t dt + \sigma dW_t \tag{8}
$$
该 SDE 的解 $(X_t)_{0 \leq t \leq 1}$ 被称为奥恩斯坦-乌伦贝克（OU）过程，其轨迹如图 3 所示。其中，向量场 $-\theta x$ 会将过程“拉回”中心原点（始终朝着与当前位置相反的方向演化），而扩散系数 $\sigma$ 则持续引入噪声。当 $t \to \infty$ 时，该过程会收敛到高斯分布 $\mathcal{N}(0, \frac{\sigma^2}{2\theta})$；特别地，当 $\sigma = 0$ 时，该过程退化为我们在式 (3) 中研究的线性向量场流。

### SDE 的模拟
如果目前你对 SDE 的抽象定义感到困惑，无需担心——理解 SDE 的最直观方式是学习其模拟方法。与 ODE 的欧拉法相对应，SDE 最简单的模拟方法是欧拉-马尔可夫法（Euler-Maruyama method）：初始化 $X_0 = x_0$，然后通过以下公式迭代更新：
$$
X_{t+h} = X_t + h u_t(X_t) + \sqrt{h} \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)
$$
其中 $h = n^{-1} > 0$ 是步长超参数（$n \in \mathbb{N}$ 为总步数）。直观来看，欧拉-马尔可夫法的核心是：在每个时间步，既沿着向量场 $u_t(X_t)$ 的方向迈出确定性小步，又叠加一个由高斯噪声驱动的随机小步（噪声强度由 $\sqrt{h} \sigma_t$ 缩放）。在本课程的 SDE 模拟任务中（例如配套实验），我们通常会使用欧拉-马尔可夫法。

### 扩散模型的定义与采样
与流模型类似，我们可以通过 SDE 构建生成模型——核心目标仍是将简单分布 $p_{\text{init}}$ 的样本转换为目标分布 $p_{\text{data}}$ 的样本。扩散模型的本质是通过神经网络参数化 SDE 的核心组件（向量场 $u_t$），形式为：
$$
X_0 \sim p_{\text{init}} \quad \text{（随机初始化）}
$$
$$
dX_t = u_t^\theta(X_t) dt + \sigma_t dW_t \quad \text{（SDE 核心方程）}
$$
其中 $u_t^\theta$ 是带有参数 $\theta$ 的神经网络，$\sigma_t$ 是固定的扩散系数。

### 算法 2：基于欧拉-马尔可夫法的扩散模型采样流程
**输入**：神经网络向量场 $u_t^\theta$、采样步数 $n$、扩散系数 $\sigma_t$
1. 初始化时间 $t = 0$
2. 设置步长 $h = \frac{1}{n}$
3. 从先验分布 $p_{\text{init}}$ 中采样初始状态 $X_0$
4. 对于 $i = 1, 2, \dots, n-1$：
   - 从高斯分布 $\mathcal{N}(0, I_d)$ 中采样噪声 $\epsilon$
   - 计算 $X_{t+h} = X_t + h \cdot u_t^\theta(X_t) + \sigma_t \cdot \sqrt{h} \epsilon$
   - 更新时间 $t \leftarrow t + h$
5. 输出终点 $X_1$（即从 $p_{\text{data}}$ 中采样的样本）

### 总结 7（SDE 生成模型核心总结）
在本文档中，扩散模型由两部分组成：带有参数 $\theta$、用于参数化向量场的神经网络 $u_t^\theta$，以及固定的扩散系数 $\sigma_t$，具体定义如下：
- 神经网络：$u^\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$，即 $(x, t) \mapsto u_t^\theta(x)$（带有参数 $\theta$）
- 固定扩散系数：$\sigma_t: [0,1] \to [0, \infty)$，即 $t \mapsto \sigma_t$

从 SDE 模型中生成样本（即生成目标对象）的流程如下：
1. 初始化：$X_0 \sim p_{\text{init}}$（从简单分布如高斯分布中采样初始状态）
2. SDE 模拟：$dX_t = u_t^\theta(X_t) dt + \sigma_t dW_t$（从 $t=0$ 到 $t=1$ 模拟 SDE）
3. 目标：$X_1 \sim p_{\text{data}}$（让轨迹终点服从目标数据分布）

特别地，当 $\sigma_t = 0$ 时，扩散模型退化为流模型。


# 3 构建训练目标 Constructing the Training Target
在上一章中，我们构建了流模型和扩散模型，通过模拟常微分方程（ODE）或随机微分方程（SDE）得到轨迹 $(X_t)_{0 \leq t \leq 1}$，具体形式如下：
$$
X_0 \sim p_{\text{init}},\ dX_t = u_t^\theta(X_t) dt \quad (\text{流模型}) \tag{10}
$$
$$
X_0 \sim p_{\text{init}},\ dX_t = u_t^\theta(X_t) dt + \sigma_t dW_t \quad (\text{扩散模型}) \tag{11}
$$
其中 $u_t^\theta$ 是神经网络，$\sigma_t$ 是固定的扩散系数。显然，若随机初始化神经网络的参数 $\theta$，模拟 ODE/SDE 只会得到无意义的结果。与所有机器学习模型一样，我们需要训练神经网络——通过最小化损失函数 $\mathcal{L}(\theta)$（例如均方误差）来实现：
$$
\mathcal{L}(\theta) = \left\| u_t^\theta(x) - \underbrace{u_t^{\text{target}}(x)}_{\text{训练目标}} \right\|^2
$$
其中 $u_t^{\text{target}}(x)$ 是我们希望模型逼近的训练目标。为了推导训练算法，我们分两步进行：本章的目标是推导训练目标 $u_t^{\text{target}}$ 的表达式；下一章将介绍逼近该训练目标的训练算法。自然地，与神经网络 $u_t^\theta$ 类似，训练目标本身也应是一个向量场 $u_t^{\text{target}}: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$，且需具备将噪声转换为数据的核心功能。因此，本章的核心任务是推导训练目标 $u_t^{\text{target}}$ 的公式，使得对应的 ODE/SDE 能将先验分布 $p_{\text{init}}$ 转换为目标数据分布 $p_{\text{data}}$。在此过程中，我们将接触到物理学和随机微积分中的两个核心结果：连续性方程（Continuity Equation）和福克-普朗克方程（Fokker-Planck Equation）。与前文一致，我们将先针对 ODE 阐述核心思想，再将其推广到 SDE。

> 注：推导流模型和扩散模型训练目标的方法有多种，本章介绍的方法兼具通用性和简洁性，与最新的主流模型保持一致，但可能与你见过的早期扩散模型表述有所不同。后续章节将讨论其他替代形式。

## 3.1 条件概率路径与边际概率路径
构建训练目标 $u_t^{\text{target}}$ 的第一步是定义概率路径。直观来看，概率路径描述了从噪声分布 $p_{\text{init}}$ 到数据分布 $p_{\text{data}}$ 的渐变插值过程（见图 4）。本节将详细解释其构建方式。

对于数据点 $z \in \mathbb{R}^d$，我们用 $\delta_z$ 表示狄拉克（Dirac）“分布”——这是最简单的分布：从 $\delta_z$ 中采样永远只会得到 $z$（即确定性分布）。

**条件（插值）概率路径** 是一系列定义在 $\mathbb{R}^d$ 上的分布 $p_t(x | z)$，满足：
$$
p_0(\cdot | z) = p_{\text{init}},\ p_1(\cdot | z) = \delta_z \quad \forall z \in \mathbb{R}^d. \tag{12}
$$
换句话说，条件概率路径描述了“单个数据点逐步转换为噪声分布”的过程（见图 4）。我们可以将概率路径理解为“分布空间中的轨迹”：每个条件概率路径 $p_t(x | z)$ 会诱导出一个**边际概率路径** $p_t(x)$，其定义为：先从数据分布中采样数据点 $z \sim p_{\text{data}}$，再从条件分布 $p_t(\cdot | z)$ 中采样，最终得到的分布即为边际概率路径（采样过程见式 (13)），其概率密度为：
$$
p_t(x) = \int p_t(x | z) p_{\text{data}}(z) dz \tag{14}
$$

需要注意的是：我们知道如何从 $p_t$ 中采样，但由于上述积分难以直接计算，无法直接得到其密度值 $p_t(x)$。结合式 (12) 中对条件概率路径的约束，不难验证边际概率路径会在 $p_{\text{init}}$ 和 $p_{\text{data}}$ 之间插值：
$$
p_0 = p_{\text{init}},\ p_1 = p_{\text{data}} \quad (\text{噪声-数据插值}) \tag{15}
$$

### 示例 9（高斯条件概率路径）
最常用的概率路径之一是高斯条件概率路径，也是去噪扩散模型中使用的概率路径。设 $\alpha_t$ 和 $\beta_t$ 为噪声调度器（noise schedulers）——它们是连续可微的单调函数，满足 $\alpha_0 = \beta_1 = 0$ 且 $\alpha_1 = \beta_0 = 1$。我们定义条件概率路径：
$$
p_t(\cdot | z) = \mathcal{N}\left(\alpha_t z, \beta_t^2 I_d\right) \quad (\text{高斯条件路径}) \tag{16}
$$
根据 $\alpha_t$ 和 $\beta_t$ 的约束条件，该路径满足：
$$
p_0(\cdot | z) = \mathcal{N}\left(\alpha_0 z, \beta_0^2 I_d\right) = \mathcal{N}(0, I_d),\ p_1(\cdot | z) = \mathcal{N}\left(\alpha_1 z, \beta_1^2 I_d\right) = \delta_z
$$
其中利用了“方差为零、均值为 $z$ 的正态分布等价于狄拉克分布 $\delta_z$”这一性质。因此，对于先验分布 $p_{\text{init}} = \mathcal{N}(0, I_d)$，上述 $p_t(x | z)$ 是合法的条件插值路径。高斯条件概率路径具有诸多实用性质，非常契合我们的需求，因此本节后续将以它作为条件概率路径的典型示例。图 4 展示了其在图像上的应用效果。

从边际路径 $p_t$ 中采样的过程可表示为：
$$
z \sim p_{\text{data}},\ \epsilon \sim p_{\text{init}} = \mathcal{N}(0, I_d) \implies x = \alpha_t z + \beta_t \epsilon \sim p_t
$$
直观来看，该过程的核心是：时间 $t$ 越小，添加的噪声越多；当 $t=0$ 时，采样结果完全是噪声。图 5 展示了高斯噪声与简单数据分布之间的插值路径示例。

## 3.2 条件向量场与边际向量场
现在，我们利用前文定义的概率路径 $p_t$，为流模型构建训练目标 $u_t^{\text{target}}$。核心思路是：从可通过解析方法手动推导的简单组件出发，构造训练目标。

### 定理 10（边缘化技巧）
对于每个数据点 $z \in \mathbb{R}^d$，设 $u_t^{\text{target}}(\cdot | z)$ 为条件向量场，其对应的 ODE 满足条件概率路径 $p_t(\cdot | z)$，即：
$$
X_0 \sim p_{\text{init}},\ \frac{d}{dt}X_t = u_t^{\text{target}}(X_t | z) \implies X_t \sim p_t(\cdot | z) \quad (0 \leq t \leq 1). \tag{18}
$$
则由下式定义的边际向量场 $u_t^{\text{target}}(x)$：
$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz, \tag{19}
$$
将遵循边际概率路径，即：
$$
X_0 \sim p_{\text{init}},\ \frac{d}{dt}X_t = u_t^{\text{target}}(X_t) \implies X_t \sim p_t \quad (0 \leq t \leq 1). \tag{20}
$$
特别地，该 ODE 的终点 $X_1 \sim p_{\text{data}}$，因此可以说“$u_t^{\text{target}}$ 将噪声 $p_{\text{init}}$ 转换为数据 $p_{\text{data}}$”。

图 6 直观展示了该定理的核心思想。在证明该定理之前，我们先说明其价值：边缘化技巧允许我们通过条件向量场构造边际向量场，极大简化了训练目标的推导——因为我们通常可以通过解析方法手动推导满足式 (18) 的条件向量场 $u_t^{\text{target}}(\cdot | z)$（仅需简单代数运算）。下面以高斯条件概率路径为例，推导对应的条件向量场 $u_t(x | z)$。

### 示例 11（高斯概率路径的目标 ODE）
如前所述，设高斯条件概率路径为 $p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$（$\alpha_t, \beta_t$ 为噪声调度器，见式 (16)）。用 $\dot{\alpha}_t = \partial_t \alpha_t$ 和 $\dot{\beta}_t = \partial_t \beta_t$ 分别表示 $\alpha_t$ 和 $\beta_t$ 的时间导数。我们需要证明：条件高斯向量场
$$
u_t^{\text{target}}(x | z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x \tag{21}
$$
是定理 10 意义下的合法条件向量场模型——若 $X_0 \sim \mathcal{N}(0, I_d)$，则其 ODE 轨迹 $X_t$ 满足 $X_t \sim p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$。图 6 通过可视化验证了这一点：条件概率路径的真实采样结果（ ground truth）与该流的 ODE 轨迹采样结果完全匹配。以下是具体证明过程：

**证明**：首先构造条件流模型 $\psi_t^{\text{target}}(x | z)$，定义为：
$$
\psi_t^{\text{target}}(x | z) = \alpha_t z + \beta_t x. \tag{22}
$$
若 $X_t$ 是 $\psi_t^{\text{target}}(\cdot | z)$ 对应的 ODE 轨迹，且 $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I_d)$，则根据定义：
$$
X_t = \psi_t^{\text{target}}(X_0 | z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot | z).
$$
由此可得出：轨迹的分布与条件概率路径一致（即满足式 (18)）。接下来需要从 $\psi_t^{\text{target}}(x | z)$ 中提取条件向量场 $u_t^{\text{target}}(x | z)$。根据流的定义（式 (2b)），对所有 $x, z \in \mathbb{R}^d$，有：
$$
\frac{d}{dt} \psi_t^{\text{target}}(x | z) = u_t^{\text{target}}\left( \psi_t^{\text{target}}(x | z) | z \right)
$$
$$
\stackrel{(i)}{\Leftrightarrow} \dot{\alpha}_t z + \dot{\beta}_t x = u_t^{\text{target}}\left( \alpha_t z + \beta_t x | z \right) \quad \forall x, z \in \mathbb{R}^d
$$
$$
\stackrel{(ii)}{\Leftrightarrow} \dot{\alpha}_t z + \dot{\beta}_t \left( \frac{x - \alpha_t z}{\beta_t} \right) = u_t^{\text{target}}(x | z) \quad \forall x, z \in \mathbb{R}^d
$$
$$
\stackrel{(iii)}{\Leftrightarrow} \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x = u_t^{\text{target}}(x | z) \quad \forall x, z \in \mathbb{R}^d
$$
其中步骤 (i) 利用了 $\psi_t^{\text{target}}(x | z)$ 的定义（式 (22)）；步骤 (ii) 通过变量替换 $x \to \frac{x - \alpha_t z}{\beta_t}$ 重新参数化；步骤 (iii) 仅进行了代数整理。最终得到的等式即为式 (21) 定义的条件高斯向量场，证明完成。

> 注：也可通过将该向量场代入后续介绍的连续性方程，验证其正确性。

本节剩余部分将通过**连续性方程**（数学和物理学中的核心结果）证明定理 10。为了理解该方程，我们先定义散度算子（divergence operator）$div$：
$$
div(v_t)(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v_t(x)
$$

### 定理 12（连续性方程）
考虑流模型的向量场为 $u_t^{\text{target}}$，且 $X_0 \sim p_{\text{init}}$。则对所有 $0 \leq t \leq 1$，$X_t \sim p_t$ 的充要条件是：
$$
\partial_t p_t(x) = -div\left( p_t u_t^{\text{target}} \right)(x) \quad \forall x \in \mathbb{R}^d, 0 \leq t \leq 1, \tag{24}
$$
其中 $\partial_t p_t(x) = \frac{d}{dt} p_t(x)$ 表示 $p_t(x)$ 的时间导数。式 (24) 即为连续性方程。

对于有数学基础的读者，附录 B 提供了连续性方程的完整证明。在此之前，我们先直观理解其含义：左侧 $\partial_t p_t(x)$ 描述了位置 $x$ 处的概率 $p_t(x)$ 随时间的变化率；直观来看，这个变化率应等于概率质量的净流入量。在流模型中，粒子 $X_t$ 会沿着向量场 $u_t^{\text{target}}$ 运动——从物理学角度，散度 $div$ 衡量向量场的“净流出量”，因此负散度 $-div$ 衡量“净流入量”。将其与当前位置 $x$ 处的总概率质量相乘，$-div(p_t u_t)$ 即表示概率质量的总净流入量。由于概率质量守恒，方程左右两侧必然相等。

**定理 10 的证明**：根据定理 12，我们需要证明式 (19) 定义的边际向量场 $u_t^{\text{target}}$ 满足连续性方程。通过直接计算可验证：
$$
\begin{aligned}
\partial_t p_t(x) & \stackrel{(i)}{=} \partial_t \int p_t(x | z) p_{\text{data}}(z) dz \\
& = \int \partial_t p_t(x | z) p_{\text{data}}(z) dz \\
& \stackrel{(ii)}{=} \int -div\left( p_t(\cdot | z) u_t^{\text{target}}(\cdot | z) \right)(x) p_{\text{data}}(z) dz \\
& \stackrel{(iii)}{=} -div\left( \int p_t(x | z) u_t^{\text{target}}(x | z) p_{\text{data}}(z) dz \right) \\
& \stackrel{(iv)}{=} -div\left( p_t(x) \int u_t^{\text{target}}(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz \right)(x) \\
& \stackrel{(v)}{=} -div\left( p_t u_t^{\text{target}} \right)(x),
\end{aligned}
$$
其中步骤 (i) 利用了式 (13) 中 $p_t(x)$ 的定义；步骤 (ii) 对条件概率路径 $p_t(\cdot | z)$ 应用连续性方程；步骤 (iii) 利用式 (23) 交换积分与散度算子的顺序；步骤 (iv) 对积分项乘以并除以 $p_t(x)$；步骤 (v) 利用了式 (19) 中边际向量场的定义。上述等式链的首尾表明，$u_t^{\text{target}}$ 满足连续性方程；根据定理 12，这足以证明式 (20)，证明完成。

## 3.3 条件得分函数与边际得分函数
我们已经成功为流模型构建了训练目标，现在将这一思路推广到 SDE。为此，我们定义 $p_t$ 的边际得分函数为 $\nabla \log p_t(x)$——利用该函数，可将上一节的 ODE 扩展为 SDE，具体结果如下：

### 定理 13（SDE 扩展技巧）
设条件向量场 $u_t^{\text{target}}(x | z)$ 和边际向量场 $u_t^{\text{target}}(x)$ 定义如前所述。则对于扩散系数 $\sigma_t \geq 0$，可构造如下 SDE，其将遵循相同的概率路径：
$$
\begin{array}{rl}
X_0 \sim & p_{\text{init}},\ dX_t = \left[ u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \right] dt + \sigma_t dW_t \\
\implies X_t \sim & p_t \quad (0 \leq t \leq 1)
\end{array}
$$
特别地，该 SDE 的终点 $X_1 \sim p_{\text{data}}$。若将边际概率 $p_t(x)$ 和边际向量场 $u_t^{\text{target}}(x)$ 分别替换为条件概率路径 $p_t(x | z)$ 和条件向量场 $u_t^{\text{target}}(x | z)$，上述等式依然成立。

图 7 直观展示了该定理的效果。定理 13 中的公式之所以实用，是因为与前文类似，边际得分函数可通过条件得分函数 $\nabla \log p_t(x | z)$ 表示：
$$
\nabla \log p_t(x) = \frac{\nabla p_t(x)}{p_t(x)} = \frac{\nabla \int p_t(x | z) p_{\text{data}}(z) dz}{p_t(x)} = \frac{\int \nabla p_t(x | z) p_{\text{data}}(z) dz}{p_t(x)} = \int \nabla \log p_t(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz \tag{27}
$$
而条件得分函数 $\nabla \log p_t(x | z)$ 通常可通过解析方法求解，以下示例将具体说明。

### 示例 14（高斯概率路径的得分函数）
对于高斯条件概率路径 $p_t(x | z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)$，利用高斯概率密度的形式（见式 (81)），可推导条件得分函数：
$$
\nabla \log p_t(x | z) = \nabla \log \mathcal{N}\left( x; \alpha_t z, \beta_t^2 I_d \right) = -\frac{x - \alpha_t z}{\beta_t^2}. \tag{28}
$$
需要注意的是，该得分函数是 $x$ 的线性函数——这是高斯分布的独特性质。

本节剩余部分将通过**福克-普朗克方程**（将连续性方程从 ODE 扩展到 SDE）证明定理 13。为此，我们先定义拉普拉斯算子（Laplacian operator）$\Delta$：
$$
\Delta w_t(x) = \sum_{i=1}^d \frac{\partial^2}{\partial x_i^2} w_t(x) = div\left( \nabla w_t \right)(x). \tag{29}
$$

### 定理 15（福克-普朗克方程）
设 $p_t$ 为概率路径，考虑 SDE：
$$
X_0 \sim p_{\text{init}},\ dX_t = u_t(X_t) dt + \sigma_t dW_t.
$$
则对所有 $0 \leq t \leq 1$，$X_t$ 服从分布 $p_t$ 的充要条件是福克-普朗克方程成立：
$$
\partial_t p_t(x) = -div\left( p_t u_t \right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \quad \forall x \in \mathbb{R}^d, 0 \leq t \leq 1. \tag{30}
$$

附录 B 提供了福克-普朗克方程的完整证明。需要注意的是：当 $\sigma_t = 0$ 时，福克-普朗克方程退化为连续性方程。新增的拉普拉斯项 $\Delta p_t$ 初看难以理解，但熟悉物理学的读者会发现，它与热传导方程中的项完全一致（热传导方程本质是福克-普朗克方程的特例）——热量会在介质中扩散，而我们在模型中引入了数学意义上的扩散过程，因此需要添加该拉普拉斯项。

**定理 13 的证明**：根据定理 15，我们需要证明式 (25) 定义的 SDE 满足 $p_t$ 对应的福克-普朗克方程。通过直接计算可验证：
$$
\begin{aligned}
\partial_t p_t(x) & \stackrel{(i)}{=} -div\left( p_t u_t^{\text{target}} \right)(x) \\
& \stackrel{(ii)}{=} -div\left( p_t u_t^{\text{target}} \right)(x) - \frac{\sigma_t^2}{2} \Delta p_t(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \\
& \stackrel{(iii)}{=} -div\left( p_t u_t^{\text{target}} \right)(x) - div\left( \frac{\sigma_t^2}{2} \nabla p_t \right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \\
& \stackrel{(iv)}{=} -div\left( p_t u_t^{\text{target}} \right)(x) - div\left( p_t \left[ \frac{\sigma_t^2}{2} \nabla \log p_t \right] \right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \\
& \stackrel{(v)}{=} -div\left( p_t \left[ u_t^{\text{target}} + \frac{\sigma_t^2}{2} \nabla \log p_t \right] \right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x),
\end{aligned}
$$
其中步骤 (i) 利用了连续性方程；步骤 (ii) 对等式添加并减去相同项；步骤 (iii) 利用了拉普拉斯算子的定义（式 (29)）；步骤 (iv) 利用了 $\nabla \log p_t = \frac{\nabla p_t}{p_t}$；步骤 (v) 利用了散度算子的线性性质。上述推导表明，式 (25) 定义的 SDE 满足 $p_t$ 对应的福克-普朗克方程；根据定理 15，可推出 $0 \leq t \leq 1$ 时 $X_t \sim p_t$，证明完成。

### 注 16（朗之万动力学）
上述构造有一个著名的特例：当概率路径为静态（即 $p_t = p$，$p$ 为固定分布）时，令 $u_t^{\text{target}} = 0$，可得到 SDE：
$$
dX_t = \frac{\sigma_t^2}{2} \nabla \log p(X_t) dt + \sigma_t dW_t, \tag{31}
$$
这一过程被称为朗之万动力学（Langevin dynamics）。由于 $p_t$ 是静态的，有 $\partial_t p_t(x) = 0$；根据定理 13，这些动力学满足静态路径 $p_t = p$ 对应的福克-普朗克方程，因此 $p$ 是朗之万动力学的平稳分布，即：
$$
X_0 \sim p \implies X_t \sim p \quad (t \geq 0)
$$
与许多马尔可夫链类似，在相当一般的条件下，朗之万动力学会收敛到平稳分布 $p$（见 3.3 节）：若初始分布 $X_0 \sim p' \neq p$，则其诱导的分布 $p_t'$ 会在温和条件下收敛到 $p$。这一性质使得朗之万动力学极具实用价值，成为分子动力学模拟、贝叶斯统计及自然科学中多种马尔可夫链蒙特卡洛（MCMC）方法的基础。

### 本节总结 17（训练目标的推导）
流模型的训练目标是边际向量场 $u_t^{\text{target}}$，其构造步骤如下：
1. 选择条件概率路径 $p_t(x | z)$，满足 $p_0(\cdot | z) = p_{\text{init}}$、$p_1(\cdot | z) = \delta_z$；
2. 找到条件向量场 $u_t^{\text{target}}(x | z)$，使其对应的流 $\psi_t^{\text{target}}(x | z)$ 满足：$X_0 \sim p_{\text{init}} \implies X_t = \psi_t^{\text{target}}(X_0 | z) \sim p_t(\cdot | z)$（或等价地，$u_t^{\text{target}}$ 满足连续性方程）；
3. 边际向量场由下式定义：
$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz, \tag{32}
$$
该向量场将遵循边际概率路径，即：
$$
X_0 \sim p_{\text{init}},\ dX_t = u_t^{\text{target}}(X_t) dt \implies X_t \sim p_t \quad (0 \leq t \leq 1). \tag{33}
$$
特别地，该 ODE 的终点 $X_1 \sim p_{\text{data}}$，即 $u_t^{\text{target}}$ 实现了“从噪声到数据的转换”。

**扩展到 SDE**：对于随时间变化的扩散系数 $\sigma_t \geq 0$，可将上述 ODE 扩展为遵循相同边际概率路径的 SDE：
$$
X_0 \sim p_{\text{init}},\ dX_t = \left[ u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \right] dt + \sigma_t dW_t \tag{34}
$$
$$
\implies X_t \sim p_t \quad (0 \leq t \leq 1), \tag{35}
$$
其中边际得分函数为：
$$
\nabla \log p_t(x) = \int \nabla \log p_t(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz. \tag{36}
$$
特别地，该 SDE 的轨迹终点 $X_1 \sim p_{\text{data}}$，即实现了“从噪声到数据的转换”。

**重要示例：高斯概率路径**
对于高斯条件概率路径，相关公式如下：
$$
p_t(x | z) = \mathcal{N}\left( x; \alpha_t z, \beta_t^2 I_d \right) \tag{37}
$$
$$
u_t^{\text{target}}(x | z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x \tag{38}
$$
$$
\nabla \log p_t(x | z) = -\frac{x - \alpha_t z}{\beta_t^2}, \tag{39}
$$
其中 $\alpha_t, \beta_t \in \mathbb{R}$ 为噪声调度器，满足：连续可微、单调，且 $\alpha_0 = \beta_1 = 0$、$\alpha_1 = \beta_0 = 1$。


# 4 训练生成模型
在前两章中，我们已经构建了基于神经网络向量场 $u_t^\theta$ 的生成模型，并推导了训练目标 $u_t^{\text{target}}$ 的表达式。本章将详细说明如何训练神经网络 $u_t^\theta$ 以逼近该训练目标：首先聚焦常微分方程（ODE），引出流匹配（Flow Matching）训练方法；其次将该方法扩展到随机微分方程（SDE），介绍得分匹配（Score Matching）；最后以高斯概率路径为特例，还原去噪扩散模型（Denoising Diffusion Models）的训练逻辑。通过这些内容，我们将获得一套完整的、基于 ODE/SDE 的生成模型训练与采样端到端流程。

## 4.1 流匹配（Flow Matching）
我们先回顾流模型的核心形式：
$$
X_0 \sim p_{\text{init}}, \quad dX_t = u_t^\theta(X_t) dt \quad \text{（流模型）} \tag{40}
$$
训练的核心目标是让神经网络向量场 $u_t^\theta$ 逼近第 3 章推导的边际向量场 $u_t^{\text{target}}$，即找到参数 $\theta$ 使得 $u_t^\theta \approx u_t^{\text{target}}$。

### 流匹配损失函数
为实现这一目标，最直观的方式是使用均方误差构建损失函数，即**流匹配损失**：
$$
\begin{aligned}
\mathcal{L}_{FM}(\theta) & = \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[\left\| u_t^\theta(x) - u_t^{\text{target}}(x) \right\|^2\right] \\
& \stackrel{(i)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| u_t^\theta(x) - u_t^{\text{target}}(x) \right\|^2\right],
\end{aligned}
$$
其中 $p_t(x) = \int p_t(x | z) p_{\text{data}}(z) dz$ 是边际概率路径，步骤 (i) 利用了式 (13) 给出的边际路径采样流程。

该损失的直观逻辑的是：
1. 随机采样一个时间 $t \in [0,1]$；
2. 从数据分布中采样一个真实样本 $z$，并基于条件概率路径 $p_t(\cdot | z)$ 生成中间样本 $x$（例如通过添加噪声实现）；
3. 计算神经网络输出 $u_t^\theta(x)$ 与边际目标向量场 $u_t^{\text{target}}(x)$ 的均方误差。

然而，直接计算该损失存在困难：根据定理 10，边际目标向量场 $u_t^{\text{target}}(x)$ 的表达式为：
$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz, \tag{43}
$$
其中包含难以直接计算的积分项。为解决这一问题，我们利用“条件向量场 $u_t^{\text{target}}(x | z)$ 可通过解析方法求解”的特性，定义**条件流匹配损失**：
$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| u_t^\theta(x) - u_t^{\text{target}}(x | z) \right\|^2\right]. \tag{44}
$$

与流匹配损失相比，条件流匹配损失的核心差异是用可解析计算的条件目标向量场 $u_t^{\text{target}}(x | z)$ 替代了难以计算的边际目标向量场 $u_t^{\text{target}}(x)$，从而使损失函数具备可训练性。

### 定理 18（损失等价性）
边际流匹配损失与条件流匹配损失仅相差一个与参数 $\theta$ 无关的常数，即：
$$
\mathcal{L}_{FM}(\theta) = \mathcal{L}_{CFM}(\theta) + C
$$
因此，两者的梯度完全一致：
$$
\nabla_\theta \mathcal{L}_{FM}(\theta) = \nabla_\theta \mathcal{L}_{CFM}(\theta).
$$
这意味着，使用随机梯度下降（SGD）等优化算法最小化 $\mathcal{L}_{CFM}(\theta)$，与最小化 $\mathcal{L}_{FM}(\theta)$ 是等价的。特别地，若神经网络具有足够的表达能力，$\mathcal{L}_{CFM}(\theta)$ 的最小值点 $\theta^*$ 会满足 $u_t^{\theta^*} = u_t^{\text{target}}$。

#### 证明过程
通过将均方误差展开为三项并分离常数项，可完成证明：
$$
\begin{aligned}
\mathcal{L}_{FM}(\theta) & \stackrel{(i)}{=} \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[\left\| u_t^\theta(x) - u_t^{\text{target}}(x) \right\|^2\right] \\
& \stackrel{(ii)}{=} \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[\left\| u_t^\theta(x) \right\|^2 - 2 u_t^\theta(x)^T u_t^{\text{target}}(x) + \left\| u_t^{\text{target}}(x) \right\|^2\right] \\
& \stackrel{(iii)}{=} \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[\left\| u_t^\theta(x) \right\|^2\right] - 2 \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[u_t^\theta(x)^T u_t^{\text{target}}(x)\right] + \underbrace{\mathbb{E}_{t \sim \text{Unif}_{[0,1]}, x \sim p_t}\left[\left\| u_t^{\text{target}}(x) \right\|^2\right]}_{=: C_1} \\
& \stackrel{(iv)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| u_t^\theta(x) \right\|^2\right] - 2 \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[u_t^\theta(x)^T u_t^{\text{target}}(x)\right] + C_1,
\end{aligned}
$$
其中步骤 (i) 为损失定义，步骤 (ii) 利用了平方差展开公式 $\|a - b\|^2 = \|a\|^2 - 2a^T b + \|b\|^2$，步骤 (iii) 定义常数项 $C_1$，步骤 (iv) 利用了边际路径的采样流程。

接下来重新表达第二项期望：
$$
\begin{aligned}
\mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[u_t^\theta(x)^T u_t^{\text{target}}(x)\right] & \stackrel{(i)}{=} \int_0^1 \int p_t(x) u_t^\theta(x)^T u_t^{\text{target}}(x) dx dt \\
& \stackrel{(ii)}{=} \int_0^1 \int p_t(x) u_t^\theta(x)^T \left[\int u_t^{\text{target}}(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz\right] dx dt \\
& \stackrel{(iii)}{=} \int_0^1 \iint u_t^\theta(x)^T u_t^{\text{target}}(x | z) p_t(x | z) p_{\text{data}}(z) dz dx dt \\
& \stackrel{(iv)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[u_t^\theta(x)^T u_t^{\text{target}}(x | z)\right],
\end{aligned}
$$
其中步骤 (i) 将期望转化为积分，步骤 (ii) 代入式 (43) 中 $u_t^{\text{target}}(x)$ 的表达式，步骤 (iii) 利用积分线性性，步骤 (iv) 重新转化为期望形式。这一步的核心是将边际向量场的期望转化为条件向量场的期望。

将上述结果代入原损失表达式：
$$
\begin{aligned}
\mathcal{L}_{FM}(\theta) & \stackrel{(i)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| u_t^\theta(x) \right\|^2\right] - 2 \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[u_t^\theta(x)^T u_t^{\text{target}}(x | z)\right] + C_1 \\
& \stackrel{(ii)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| u_t^\theta(x) \right\|^2 - 2 u_t^\theta(x)^T u_t^{\text{target}}(x | z) + \left\| u_t^{\text{target}}(x | z) \right\|^2 - \left\| u_t^{\text{target}}(x | z) \right\|^2\right] + C_1 \\
& \stackrel{(iii)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| u_t^\theta(x) - u_t^{\text{target}}(x | z) \right\|^2\right] + \underbrace{\mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[-\left\| u_t^{\text{target}}(x | z) \right\|^2\right]}_{=: C_2} + C_1 \\
& \stackrel{(iv)}{=} \mathcal{L}_{CFM}(\theta) + \underbrace{C_2 + C_1}_{=: C},
\end{aligned}
$$
步骤 (ii) 添加并减去 $\left\| u_t^{\text{target}}(x | z) \right\|^2$，步骤 (iii) 重新组合为平方差形式并定义常数 $C_2$，步骤 (iv) 合并常数项 $C = C_1 + C_2$，最终证明了两者的等价性。

### 训练与采样流程
当 $u_t^\theta$ 训练完成后，可通过模拟 ODE（例如使用算法 1 的欧拉法）生成样本 $X_1 \sim p_{\text{data}}$。这一整套“定义条件损失→训练向量场→模拟 ODE 采样”的流程，即为文献中所说的**流匹配**[14, 16, 1, 15]，其训练过程可总结为算法 3，可视化效果如图 9 所示。

### 示例 19（高斯条件概率路径的流匹配）
以高斯条件概率路径 $p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ 为例（$\alpha_t, \beta_t$ 为噪声调度器），其采样过程为：
$$
\epsilon \sim \mathcal{N}(0, I_d) \implies x_t = \alpha_t z + \beta_t \epsilon \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot | z). \tag{46}
$$
结合第 3 章推导的条件目标向量场（式 (21)）：
$$
u_t^{\text{target}}(x | z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t\right) z + \frac{\dot{\beta}_t}{\beta_t} x, \tag{47}
$$
其中 $\dot{\alpha}_t = \partial_t \alpha_t$、$\dot{\beta}_t = \partial_t \beta_t$ 为时间导数，可将条件流匹配损失具体化为：
$$
\begin{aligned}
\mathcal{L}_{CFM}(\theta) & = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d)}\left[\left\| u_t^\theta(x) - \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t\right) z - \frac{\dot{\beta}_t}{\beta_t} x \right\|^2\right] \\
& \stackrel{(i)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[\left\| u_t^\theta(\alpha_t z + \beta_t \epsilon) - \left(\dot{\alpha}_t z + \dot{\beta}_t \epsilon\right) \right\|^2\right],
\end{aligned}
$$
步骤 (i) 代入了 $x = \alpha_t z + \beta_t \epsilon$ 的采样形式，简化后损失仅需采样数据、噪声和时间，计算极为简便。

针对 $\alpha_t = t$、$\beta_t = 1 - t$ 的特殊情况（该概率路径也被称为高斯 CondOT 路径），$\dot{\alpha}_t = 1$、$\dot{\beta}_t = -1$，损失进一步简化为：
$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[\left\| u_t^\theta(t z + (1 - t) \epsilon) - (z - \epsilon) \right\|^2\right].
$$
许多主流模型（如 Stable Diffusion 3、Meta 的 Movie Gen Video）均采用这种简洁有效的训练方式。

### 算法 3 流匹配训练流程（基于高斯 CondOT 路径 $p_t(x | z) = \mathcal{N}(t z, (1 - t)^2)$）
**输入**：数据样本 $z \sim p_{\text{data}}$、神经网络 $u_t^\theta$
1. 对于每个迷你批次数据：
   - 从数据集中采样一个样本 $z$；
   - 从均匀分布 $\text{Unif}[0,1]$ 中采样时间 $t$；
   - 从高斯分布 $\mathcal{N}(0, I_d)$ 中采样噪声 $\epsilon$；
   - 构造中间样本 $x = t z + (1 - t) \epsilon$（通用情况：$x \sim p_t(\cdot | z)$）；
   - 计算损失：$\mathcal{L}(\theta) = \left\| u_t^\theta(x) - (z - \epsilon) \right\|^2$（通用情况：$\left\| u_t^\theta(x) - u_t^{\text{target}}(x | z) \right\|^2$）；
   - 通过梯度下降更新模型参数 $\theta$。
2. 重复上述步骤直至训练收敛。

## 4.2 得分匹配（Score Matching）
接下来将流匹配的训练思路扩展到 SDE。回顾第 3 章的结论，目标 ODE 可扩展为具有相同边际概率路径的 SDE：
$$
dX_t = \left[ u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \right] dt + \sigma_t dW_t, \tag{48}
$$
$$
X_0 \sim p_{\text{init}}, \quad \Rightarrow X_t \sim p_t \quad (0 \leq t \leq 1), \tag{49, 50}
$$
其中 $u_t^{\text{target}}$ 是边际向量场，$\nabla \log p_t(x)$ 是边际得分函数，其表达式为：
$$
\nabla \log p_t(x) = \int \nabla \log p_t(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz. \tag{51}
$$

为逼近边际得分函数 $\nabla \log p_t$，我们引入**得分网络** $s_t^\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$，并类比流匹配的思路，设计得分匹配损失与条件得分匹配损失：
$$
\mathcal{L}_{SM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| s_t^\theta(x) - \nabla \log p_t(x) \right\|^2\right] \quad \text{（得分匹配损失）},
$$
$$
\mathcal{L}_{CSM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| s_t^\theta(x) - \nabla \log p_t(x | z) \right\|^2\right] \quad \text{（条件得分匹配损失）}.
$$
两者的核心差异在于：得分匹配损失使用难以计算的边际得分函数，而条件得分匹配损失使用可解析求解的条件得分函数 $\nabla \log p_t(x | z)$。

### 定理 20（得分损失等价性）
得分匹配损失与条件得分匹配损失仅相差一个与参数 $\theta$ 无关的常数：
$$
\mathcal{L}_{SM}(\theta) = \mathcal{L}_{CSM}(\theta) + C,
$$
因此两者的梯度完全一致：
$$
\nabla_\theta \mathcal{L}_{SM}(\theta) = \nabla_\theta \mathcal{L}_{CSM}(\theta).
$$
特别地，若得分网络具有足够表达能力，其最小值点 $\theta^*$ 会满足 $s_t^{\theta^*} = \nabla \log p_t$。

#### 证明过程
由于边际得分函数 $\nabla \log p_t(x)$ 的表达式（式 (51)）与边际向量场 $u_t^{\text{target}}(x)$ 的表达式（式 (43)）形式完全一致，因此可直接复用定理 18 的证明逻辑，仅需将 $u_t^{\text{target}}$ 替换为 $\nabla \log p_t$ 即可完成证明。

### 扩散模型的训练与采样
上述流程即为扩散模型的基础训练方法。训练完成后，可选择任意扩散系数 $\sigma_t \geq 0$，通过模拟以下 SDE 生成样本 $X_1 \sim p_{\text{data}}$：
$$
X_0 \sim p_{\text{init}}, \quad dX_t = \left[ u_t^\theta(X_t) + \frac{\sigma_t^2}{2} s_t^\theta(X_t) \right] dt + \sigma_t dW_t. \tag{52}
$$
理论上，在完美训练的情况下，任意 $\sigma_t$ 都能生成符合目标分布的样本，但实际中会存在两类误差：
1. SDE 数值模拟带来的误差；
2. 模型 $u_t^\theta$ 与真实目标 $u_t^{\text{target}}$ 之间的训练误差。

因此，最优扩散系数 $\sigma_t$ 需通过实验验证确定[1, 12, 17]。值得注意的是，虽然扩散模型需要同时学习向量场 $u_t^\theta$ 和得分网络 $s_t^\theta$，但可将两者整合到一个网络中（输出两个分支），额外计算开销极小。更重要的是，对于高斯概率路径，向量场与得分函数可相互转换，无需单独训练。

### 注 21（去噪扩散模型）
如果你熟悉扩散模型，大概率接触过去噪扩散模型（Denoising Diffusion Models）——这类模型因应用广泛，如今常被直接称为“扩散模型”。在本文档的框架中，去噪扩散模型本质是采用高斯概率路径 $p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$ 的扩散模型。需要注意的是，早期去噪扩散模型的文献采用了不同的时间约定（时间反转），且通过“前向过程”（forward process）构建概率路径，后续 4.3 节将详细讨论这些差异。

### 示例 22（高斯概率路径的得分匹配：去噪扩散模型）
针对高斯条件概率路径 $p_t(x | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$，第 3 章已推导其条件得分函数（式 (28)）：
$$
\nabla \log p_t(x | z) = -\frac{x - \alpha_t z}{\beta_t^2}. \tag{53}
$$
将其代入条件得分匹配损失，可得：
$$
\begin{aligned}
\mathcal{L}_{CSM}(\theta) & = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot | z)}\left[\left\| s_t^\theta(x) + \frac{x - \alpha_t z}{\beta_t^2} \right\|^2\right] \\
& \stackrel{(i)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[\left\| s_t^\theta(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t} \right\|^2\right] \\
& = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[\frac{1}{\beta_t^2} \left\| \beta_t s_t^\theta(\alpha_t z + \beta_t \epsilon) + \epsilon \right\|^2\right],
\end{aligned}
$$
步骤 (i) 代入了 $x = \alpha_t z + \beta_t \epsilon$ 的采样形式。此时，得分网络 $s_t^\theta$ 本质上是在学习预测污染数据样本 $z$ 所使用的噪声 $\epsilon$，因此该损失也被称为**去噪得分匹配损失**，是早期扩散模型的核心训练方法。

但实践发现，当 $\beta_t \approx 0$ 时，上述损失会出现数值不稳定问题（即仅当添加足够噪声时，去噪得分匹配才有效）。因此，早期去噪扩散模型（如 Denoising Diffusion Probabilistic Models [9]）提出移除损失中的 $\frac{1}{\beta_t^2}$ 项，并将得分网络 $s_t^\theta$ 重新参数化为噪声预测网络 $\epsilon_t^\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$，满足：
$$
-\beta_t s_t^\theta(x) = \epsilon_t^\theta(x) \implies \mathcal{L}_{DDPM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[\left\| \epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon \right\|^2\right].
$$
此时，网络 $\epsilon_t^\theta$ 直接学习预测污染样本所用的噪声 $\epsilon$，该形式简洁且数值稳定，成为去噪扩散模型的标准训练目标。

### 算法 4 高斯概率路径的得分匹配训练流程
**输入**：数据样本 $z \sim p_{\text{data}}$、得分网络 $s_t^\theta$ 或噪声预测网络 $\epsilon_t^\theta$
1. 对于每个迷你批次数据：
   - 从数据集中采样一个样本 $z$；
   - 从均匀分布 $\text{Unif}[0,1]$ 中采样时间 $t$；
   - 从高斯分布 $\mathcal{N}(0, I_d)$ 中采样噪声 $\epsilon$；
   - 构造中间样本 $x_t = \alpha_t z + \beta_t \epsilon$（通用情况：$x_t \sim p_t(\cdot | z)$）；
   - 计算损失：$\mathcal{L}(\theta) = \left\| s_t^\theta(x_t) + \frac{\epsilon}{\beta_t} \right\|^2$（通用情况：$\left\| s_t^\theta(x_t) - \nabla \log p_t(x_t | z) \right\|^2$）；
     或替代损失：$\mathcal{L}(\theta) = \left\| \epsilon_t^\theta(x_t) - \epsilon \right\|^2$；
   - 通过梯度下降更新模型参数 $\theta$。
2. 重复上述步骤直至训练收敛。

### 命题 1（高斯概率路径的转换公式）
对于高斯条件概率路径 $p_t(x | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$，条件（或边际）向量场与条件（或边际）得分函数可相互转换：
$$
u_t^{\text{target}}(x | z) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \nabla \log p_t(x | z) + \frac{\dot{\alpha}_t}{\alpha_t} x,
$$
$$
u_t^{\text{target}}(x) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \nabla \log p_t(x) + \frac{\dot{\alpha}_t}{\alpha_t} x.
$$
其中，边际向量场对应的 ODE 在文献中被称为**概率流 ODE**（probability flow ODE）。

#### 证明过程
以条件向量场和条件得分函数为例，通过代数变换可推导：
$$
u_t^{\text{target}}(x | z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x \stackrel{(i)}{=} \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \left( \frac{\alpha_t z - x}{\beta_t^2} \right) + \frac{\dot{\alpha}_t}{\alpha_t} x = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \nabla \log p_t(x | z) + \frac{\dot{\alpha}_t}{\alpha_t} x,
$$
步骤 (i) 仅进行了代数整理。对边际向量场和边际得分函数，通过积分运算可证明同样的恒等式：
$$
\begin{aligned}
u_t^{\text{target}}(x) & = \int u_t^{\text{target}}(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz \\
& = \int \left[ \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \nabla \log p_t(x | z) + \frac{\dot{\alpha}_t}{\alpha_t} x \right] \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz \\
& \stackrel{(i)}{=} \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \nabla \log p_t(x) + \frac{\dot{\alpha}_t}{\alpha_t} x,
\end{aligned}
$$
步骤 (i) 利用了式 (51) 中边际得分函数的定义。

### 向量场与得分网络的相互参数化
基于上述转换公式，可将得分网络 $s_t^\theta$ 与向量场网络 $u_t^\theta$ 相互参数化：
$$
u_t^\theta(x) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t} x. \tag{54}
$$
同理，只要 $\beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t \neq 0$（对 $t \in [0,1)$ 恒成立），可得：
$$
s_t^\theta(x) = \frac{\alpha_t u_t^\theta(x) - \dot{\alpha}_t x}{\beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t}. \tag{55}
$$
这一参数化表明，去噪得分匹配损失与条件流匹配损失仅相差一个常数，因此无需同时训练得分网络和向量场网络——掌握其中一个，即可通过公式推导得到另一个。图 10 直观对比了通过得分匹配直接学习的得分场，与通过向量场参数化得到的得分场，两者高度一致。

若已训练好得分网络 $s_t^\theta$，可通过以下 SDE 生成样本 $X_1 \sim p_{\text{data}}$（存在训练误差和模拟误差）：
$$
X_0 \sim p_{\text{init}}, \quad dX_t = \left[ \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t + \frac{\sigma_t^2}{2} \right) s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t} x \right] dt + \sigma_t dW_t, \tag{56}
$$
这一过程即为去噪扩散模型的随机采样。

## 4.3 扩散模型文献指南
围绕扩散模型和流匹配，学术界存在一系列相关框架。不同文献的表述方式可能不同（但核心等价），这给阅读带来了一定困惑。本节将梳理这些框架的差异、历史背景，帮助读者更好地理解相关文献（本节内容不影响对后续章节的理解，仅作为文献阅读参考）。

### 离散时间 vs 连续时间
早期去噪扩散模型文献[28, 29, 9]并未使用 SDE，而是基于离散时间步 $t=0,1,2,\dots$ 构建马尔可夫链。这种离散时间形式虽简洁直观，但存在两个缺点：
1. 训练前需固定时间离散化方式；
2. 损失函数需通过证据下界（ELBO）近似，而 ELBO 只是我们真正想要最小化的损失的下界。

后来，Song 等人[32]证明，这些离散时间构造本质上是连续时间 SDE 的近似；且在连续时间下，ELBO 损失会变得“紧致”（即不再是下界，而是与真实损失相等）——这使得 SDE 构造因数学上更“干净”，且能通过 ODE/SDE 采样器控制模拟误差，逐渐成为主流。需注意的是，两种形式采用的损失本质相同，并非根本性差异。

### “前向过程”（forward process）vs 概率路径
早期去噪扩散模型[28, 29, 9, 32]并未使用“概率路径”这一术语，而是通过“前向过程”构造数据的加噪流程。前向过程是一种 SDE，形式为：
$$
\bar{X}_0 = z, \quad d\bar{X}_t = u_t^{\text{forw}}(\bar{X}_t) dt + \sigma_t^{\text{forw}} d\bar{W}_t, \tag{57}
$$
其核心思想是：从数据点 $z \sim p_{\text{data}}$ 出发，通过模拟前向过程对数据加噪，使得当 $t \to \infty$ 时，$\bar{X}_t$ 的分布收敛到高斯分布 $\mathcal{N}(0, I_d)$（即当 $T \gg 0$ 时，$\bar{X}_T \approx \mathcal{N}(0, I_d)$）。

前向过程本质上对应一种概率路径：给定 $\bar{X}_0 = z$ 时，$\bar{X}_t$ 的条件分布是条件概率路径 $\bar{p}_t(\cdot | z)$；对 $z \sim p_{\text{data}}$ 边缘化后，$\bar{X}_t$ 的分布是边际概率路径 $\bar{p}_t$。但需注意：
1. 扩散模型的“前向过程”从未被实际模拟（训练时仅需从 $\bar{p}_t(\cdot | z)$ 中采样）；
2. 前向过程需在 $t \to \infty$ 时才收敛到先验分布 $p_{\text{init}}$，无法在有限时间内到达。

而“概率路径”这一术语由流匹配[14]提出，其优势在于：简化构造过程的同时，具备更强的通用性——无需依赖前向过程，且能在有限时间内完成从 $p_{\text{init}}$ 到 $p_{\text{data}}$ 的插值。此外，前向过程的向量场 $u_t^{\text{forw}}$ 需满足“条件分布 $\bar{X}_t | \bar{X}_0 = z$ 可解析求解”，这限制了其只能采用仿射形式 $u_t^{\text{forw}}(x) = a_t x$（对应高斯概率路径）；而概率路径则无此限制。

### 时间反转（Time-Reversals）vs 福克-普朗克方程（Fokker-Planck Equation）
早期扩散模型并未通过福克-普朗克方程（或连续性方程）构造训练目标 $u_t^{\text{target}}$ 或 $\nabla \log p_t$，而是通过前向过程的时间反转[2]。时间反转 $(X_t)_{0 \leq t \leq T}$ 是一种 SDE，其轨迹的时间反转后分布与原分布相同，即：
$$
\mathbb{P}\left[X_{t_1} \in A_1, ..., X_{t_n} \in A_n\right] = \mathbb{P}\left[X_{T - t_1} \in A_1, ..., X_{T - t_n} \in A_n\right]
$$
对所有 $0 \leq t_1, ..., t_n \leq T$ 和 $A_1, ..., A_n \subset S$ 成立[59]。

根据 Anderson[2]的研究，满足上述条件的时间反转 SDE 为：
$$
dX_t = \left[ -u_t(X_t) + \sigma_t^2 \nabla \log p_t(X_t) \right] dt + \sigma_t dW_t, \quad u_t(x) = u_{T - t}^{\text{forw}}(x), \quad \sigma_t = \bar{\sigma}_{T - t}.
$$
由于 $u_t(X_t) = a_t X_t$，该形式本质上是命题 1 中训练目标的特例（需注意时间约定的差异，具体推导见[15]）。但对于生成建模而言，我们通常仅关注马尔可夫过程的终点 $X_1$（如生成的图像），因此“是否为真实时间反转”对许多应用而言并不重要；且时间反转往往导致次优结果（例如概率流 ODE 通常表现更优[12, 17]）。如今，除时间反转外，所有扩散模型的采样方法均依赖福克-普朗克方程——这也是当前学术界更倾向于直接通过福克-普朗克方程构造训练目标的原因（该方法由[14, 16, 1]开创，与本章采用的思路一致）。

### 流匹配（Flow Matching）与随机插值器（Stochastic Interpolants）
本文档介绍的框架与流匹配和随机插值器（Stochastic Interpolants, SIs）最为接近：
- 流匹配仅聚焦于流模型（ODE），其核心创新是证明：无需通过前向过程和 SDE，仅流模型即可实现大规模训练；需注意的是，流匹配模型的采样是确定性的（仅初始状态 $X_0 \sim p_{\text{init}}$ 具有随机性）。
- 随机插值器则同时包含纯流模型和基于朗之万动力学（Langevin dynamics）的 SDE 扩展（即本章定理 13 的形式），其名称源于“插值函数 $I(t, x, z)$”——该函数用于在两个分布之间插值，对应本文档中“条件概率路径与边际概率路径”的构造（形式略有不同，但核心等价）。

与扩散模型相比，流匹配和随机插值器的优势在于简洁性与通用性：训练框架简单，且能实现任意先验分布 $p_{\text{init}}$ 到任意数据分布 $p_{\text{data}}$ 的转换；而去噪扩散模型仅适用于高斯先验分布和高斯概率路径——这为生成建模开辟了新的可能性。

### 本节总结 23（生成模型训练核心）
- 流匹配的核心是：通过最小化条件流匹配损失训练神经网络 $u_t^\theta$，损失形式为：
$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{z \sim p_{\text{data}}, t \sim \text{Unif}, x \sim p_t(\cdot | z)}\left[\left\| u_t^\theta(x) - u_t^{\text{target}}(x | z) \right\|^2\right], \tag{60}
$$
训练完成后，通过模拟 ODE 生成样本（见算法 1）。

- 扩散模型的扩展：引入得分网络 $s_t^\theta$，通过条件得分匹配损失训练，损失形式为：
$$
\mathcal{L}_{CSM}(\theta) = \mathbb{E}_{z \sim p_{\text{data}}, t \sim \text{Unif}, x \sim p_t(\cdot | z)}\left[\left\| s_t^\theta(x) - \nabla \log p_t(x | z) \right\|^2\right], \tag{61}
$$
对任意扩散系数 $\sigma_t \geq 0$，模拟 SDE（见算法 2）即可生成近似样本 $X_1 \sim p_{\text{data}}$，最优 $\sigma_t$ 需通过实验确定。

- 高斯概率路径特例：此时条件得分匹配也被称为去噪得分匹配，两类损失具体为：
$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[\left\| u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \right\|^2\right],
$$
$$
\mathcal{L}_{CSM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[\left\| s_t^\theta(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t} \right\|^2\right].
$$
该场景下，向量场与得分函数可通过以下公式相互转换，无需单独训练：
$$
u_t^\theta(x) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t} x.
$$

- 去噪扩散模型：本质是采用高斯概率路径的扩散模型，仅需学习 $u_t^\theta$ 或 $s_t^\theta$ 之一（可相互转换）；其采样可选择确定性的概率流 ODE 或随机性的 SDE 采样；但与流匹配/随机插值器不同，去噪扩散模型仅适用于高斯先验分布和高斯概率路径。

### 文献中常见的扩散模型替代表述
1. 离散时间形式：通过离散时间马尔可夫链近似 SDE；
2. 时间反转约定：将 $t=0$ 对应数据分布 $p_{\text{data}}$（与本文档 $t=0$ 对应先验分布 $p_{\text{init}}$ 相反）；
3. 前向过程：通过加噪过程构造高斯概率路径；
4. 基于时间反转的训练目标：通过 SDE 的时间反转构造训练目标（本质是本文档构造的特例，需调整时间约定）。


# 5 构建图像生成器
在前几章中，我们已经学习了如何训练流匹配模型或扩散模型以从数据分布 $p_{\text{data}}(x)$ 中采样。这套方法具有通用性，可应用于多种数据类型和场景。本章将聚焦如何将该框架应用于构建图像或视频生成器（例如 Stable Diffusion 3 和 Meta 的 Movie Gen Video）。要实现这一目标，我们还需补充两个核心要素：一是条件生成（引导）的形式化定义，即如何生成符合特定文本提示的图像，以及如何调整现有目标函数以适配该任务（将重点介绍提升条件生成质量的主流技术——无分类器引导）；二是适用于图像和视频的常见神经网络架构。最后，我们将深入剖析上述两款最先进的图像和视频生成模型，让你了解大规模生成模型的实际构建方式。

## 5.1 引导（Guidance）
到目前为止，我们讨论的生成模型均为无条件生成模型——例如图像生成模型会随机生成一张图像。但实际应用中，我们往往需要生成符合特定附加信息的对象。例如，输入文本提示 $y=$“一只在积雪覆盖的山坡上奔跑、背景有山脉的狗”，生成对应的图像 $x$。对于固定提示 $y$，我们的目标是从条件数据分布 $p_{\text{data}}(x | y)$ 中采样。形式上，引导变量 $y$ 存在于特定空间 $\mathcal{Y}$：若 $y$ 是文本提示，通常对应连续空间（如 $\mathbb{R}^{d_y}$）；若 $y$ 是离散类别标签，则对应离散空间。在实验部分，我们将使用 MNIST 数据集，此时 $Y=\{0,1,...,9\}$ 对应手写数字的类别。

为避免与“基于 $z \sim p_{\text{data}}$ 的条件化（条件概率路径/条件向量场）”在术语上混淆，我们专门使用“引导（guided）”一词指代基于 $y$ 的条件化。

### 注 24（引导与条件的术语区分）
本笔记中，“引导（guided）”特指基于 $y$ 的条件化，例如引导向量场 $u_t^{\text{target}}(x | y)$；而“条件（conditional）”仍用于描述基于 $z \sim p_{\text{data}}$ 的条件化，例如条件向量场 $u_t^{\text{target}}(x | z)$。该术语用法与其他相关研究一致[15]。

引导式生成建模的核心目标是：能够从任意 $y$ 对应的条件分布 $p_{\text{data}}(x | y)$ 中采样。结合流匹配和得分匹配的框架（生成模型通过模拟常微分方程/随机微分方程实现），可将引导式扩散模型定义如下：

### 核心思想 5（引导式生成模型）
引导式扩散模型由**引导向量场** $u_t^\theta(\cdot | y)$（通过神经网络参数化）和**时变扩散系数** $\sigma_t$ 组成，具体形式为：
- 神经网络：$u^\theta: \mathbb{R}^d \times \mathcal{Y} \times [0,1] \to \mathbb{R}^d$，即 $(x, y, t) \mapsto u_t^\theta(x | y)$（含参数 $\theta$）
- 固定扩散系数：$\sigma_t: [0,1] \to [0, \infty)$，即 $t \mapsto \sigma_t$

与总结 7 相比，核心差异是神经网络额外引入了引导变量 $y \in \mathcal{Y}$。对于任意 $y \in \mathbb{R}^{d_y}$，从该模型中生成样本的流程如下：
1. 初始化：$X_0 \sim p_{\text{init}}$（从简单分布如高斯分布中采样初始状态）
2. SDE 模拟：$dX_t = u_t^\theta(X_t | y) dt + \sigma_t dW_t$（从 $t=0$ 到 $t=1$ 模拟 SDE）
3. 目标：$X_1 \sim p_{\text{data}}(\cdot | y)$（使轨迹终点服从条件数据分布）

当 $\sigma_t = 0$ 时，该模型称为**引导式流模型**。

### 5.1.1 流模型的引导
若固定引导变量 $y$，并将数据分布视为 $p_{\text{data}}(x | y)$，则可复用无条件生成的框架，通过条件流匹配目标构建生成模型：
$$
\mathbb{E}_{z \sim p_{\text{data}}(\cdot | y), x \sim p_t(\cdot | z)} \left\| u_t^\theta(x | y) - u_t^{\text{target}}(x | z) \right\|^2 \tag{63}
$$
需要注意的是，标签 $y$ 不会影响条件概率路径 $p_t(\cdot | z)$ 或条件向量场 $u_t^{\text{target}}(x | z)$（原则上可设计为依赖，但此处无需）。将期望扩展到所有 $y$ 和时间 $t \in \text{Unif}[0,1)$，可得到**引导式条件流匹配目标**：
$$
\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta) = \mathbb{E}_{(z, y) \sim p_{\text{data}}(z, y), t \sim \text{Unif}[0,1), x \sim p_t(\cdot | z)} \left\| u_t^\theta(x | y) - u_t^{\text{target}}(x | z) \right\|^2
$$

该目标与无条件流匹配目标（式 (44)）的核心差异是：采样对象从 $z \sim p_{\text{data}}$ 变为 $(z, y) \sim p_{\text{data}}$（即数据分布变为图像 $z$ 与文本提示 $y$ 的联合分布）。在实际实现中（如 PyTorch），这意味着数据加载器（dataloader）需返回包含 $z$ 和 $y$ 的批次数据。通过该目标训练的模型，能够忠实生成符合 $p_{\text{data}}(\cdot | y)$ 的样本。

#### 无分类器引导（Classifier-Free Guidance）
上述条件训练方法在理论上成立，但实验发现，生成的图像与目标标签 $y$ 的契合度往往不足。研究人员发现，通过人工增强引导变量 $y$ 的影响，可显著提升生成结果的感知质量——这一洞察最终形成了无分类器引导（CFG）技术，该技术已被广泛应用于最先进的扩散模型中。为简化说明，我们以高斯条件概率路径为例展开。回顾式 (16)，高斯条件概率路径的形式为：
$$
p_t(\cdot | z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)
$$
其中噪声调度器 $\alpha_t$ 和 $\beta_t$ 满足连续可微、单调，且 $\alpha_0 = \beta_1 = 0$、$\alpha_1 = \beta_0 = 1$。结合命题 1，可利用引导得分函数 $\nabla \log p_t(x | y)$，将引导向量场 $u_t^{\text{target}}(x | y)$ 重写为：
$$
u_t^{\text{target}}(x | y) = a_t x + b_t \nabla \log p_t(x | y) \tag{65}
$$
其中：
$$
(a_t, b_t) = \left( \frac{\dot{\alpha}_t}{\alpha_t}, \frac{\dot{\alpha}_t \beta_t^2 - \dot{\beta}_t \beta_t \alpha_t}{\alpha_t} \right) \tag{66}
$$

根据贝叶斯法则，引导得分函数可分解为：
$$
\nabla \log p_t(x | y) = \nabla \log \left( \frac{p_t(x) p_t(y | x)}{p_t(y)} \right) = \nabla \log p_t(x) + \nabla \log p_t(y | x)
$$
其中梯度 $\nabla$ 针对变量 $x$ 计算，因此 $\nabla \log p_t(y) = 0$。将其代入式 (65)，可得：
$$
u_t^{\text{target}}(x | y) = a_t x + b_t \left( \nabla \log p_t(x) + \nabla \log p_t(y | x) \right) = u_t^{\text{target}}(x) + b_t \nabla \log p_t(y | x)
$$

该式表明：引导向量场 $u_t^{\text{target}}(x | y)$ 是无条件向量场与引导得分项 $\nabla \log p_t(y | x)$ 的叠加。由于生成图像与提示 $y$ 的契合度不足，自然的改进思路是放大 $\nabla \log p_t(y | x)$ 项的贡献，得到：
$$
\tilde{u}_t(x | y) = u_t^{\text{target}}(x) + w b_t \nabla \log p_t(y | x)
$$
其中 $w > 1$ 称为**引导尺度（guidance scale）**。需要注意的是，这是一种启发式方法：当 $w \neq 1$ 时，$\tilde{u}_t(x | y) \neq u_t^{\text{target}}(x | y)$（即不再是真实的引导向量场），但实验表明，$w > 1$ 时能得到更符合预期的生成结果。

### 注 25（“分类器”的由来）
项 $\log p_t(y | x)$ 可视为带噪声数据的“分类器”（即给定带噪声数据 $x$，输出标签 $y$ 的似然）。早期扩散模型研究中，确实通过训练独立分类器实现引导（称为分类器引导[5, 30]），但目前该方法已基本被无分类器引导取代，因此本笔记不再展开。

再次利用得分分解公式：
$$
\nabla \log p_t(x | y) = \nabla \log p_t(x) + \nabla \log p_t(y | x)
$$
可将缩放后的引导向量场重写为：
$$
\begin{aligned}
\tilde{u}_t(x | y) & = u_t^{\text{target}}(x) + w b_t \nabla \log p_t(y | x) \\
& = u_t^{\text{target}}(x) + w b_t \left( \nabla \log p_t(x | y) - \nabla \log p_t(x) \right) \\
& = u_t^{\text{target}}(x) - \left( w a_t x + w b_t \nabla \log p_t(x) \right) + \left( w a_t x + w b_t \nabla \log p_t(x | y) \right) \\
& = (1 - w) u_t^{\text{target}}(x) + w u_t^{\text{target}}(x | y)
\end{aligned}
$$

可见，缩放后的引导向量场 $\tilde{u}_t(x | y)$ 是无条件向量场 $u_t^{\text{target}}(x)$ 与引导向量场 $u_t^{\text{target}}(x | y)$ 的线性组合。理论上，可分别训练无条件模型（使用式 (44)）和引导模型（使用式 (64)），再在推理时组合两者得到 $\tilde{u}_t(x | y)$——但这需要训练两个独立模型，成本较高。实际解决方案是：在标签集中新增一个“无引导”标签 $\emptyset$，将无条件向量场 $u_t^{\text{target}}(x)$ 视为 $u_t^{\text{target}}(x | \emptyset)$，从而在单个模型中同时训练无条件和引导两种模式。这种“单模型训练、推理时增强引导权重”的方法，即为**无分类器引导（CFG）**[10]。

### 注 26（通用概率路径的推导）
上述线性组合构造 $\tilde{u}_t(x | y) = (1 - w) u_t^{\text{target}}(x) + w u_t^{\text{target}}(x | y)$ 适用于任意概率路径，而非仅高斯路径。当 $w=1$ 时，显然 $\tilde{u}_t(x | y) = u_t^{\text{target}}(x | y)$；本笔记通过高斯路径推导，仅为直观说明“放大分类器项贡献”的核心思路。

#### 无分类器引导的训练目标
为适配“无引导标签 $\emptyset$”，需修改引导式条件流匹配目标（式 (64)）。由于采样 $(z, y) \sim p_{\text{data}}$ 时无法自然获得 $y = \emptyset$，需人工引入该情况：定义超参数 $\eta$ 为“丢弃原始标签 $y$ 并替换为 $\emptyset$”的概率，最终得到**无分类器引导的条件流匹配目标（CFG-CFM）**：
$$
\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta) = \mathbb{E}_{\square} \left\| u_t^\theta(x | y) - u_t^{\text{target}}(x | z) \right\|^2 \tag{68}
$$
其中 $\square = (z, y) \sim p_{\text{data}}(z, y), t \sim \text{Unif}[0,1), x \sim p_t(\cdot | z), \text{以概率 } \eta \text{ 将 } y \text{ 替换为 } \emptyset$ (69)。

### 总结 27（流模型的无分类器引导）
给定无条件边际向量场 $u_t^{\text{target}}(x | \emptyset)$、引导边际向量场 $u_t^{\text{target}}(x | y)$ 和引导尺度 $w > 1$，无分类器引导向量场定义为：
$$
\tilde{u}_t(x | y) = (1 - w) u_t^{\text{target}}(x | \emptyset) + w u_t^{\text{target}}(x | y) \tag{70}
$$

通过单个神经网络同时逼近 $u_t^{\text{target}}(x | \emptyset)$ 和 $u_t^{\text{target}}(x | y)$，可使用上述无分类器引导的条件流匹配目标（CFG-CFM），其直观流程为：
1. 从数据分布中采样 $(z, y)$；
2. 从 $[0,1]$ 均匀分布中采样时间 $t$；
3. 从条件概率路径 $p_t(x | z)$ 中采样 $x$；
4. 以概率 $\eta$ 将 $y$ 替换为 $\emptyset$；
5. 计算损失：$\mathcal{L}(\theta) = \left\| u_t^\theta(x | y) - u_t^{\text{target}}(x | z) \right\|^2$（即让模型逼近条件向量场）。

推理时，对于固定引导变量 $y$，生成流程为：
1. 初始化：$X_0 \sim p_{\text{init}}$（从简单分布如高斯分布中采样）；
2. ODE 模拟：$dX_t = \tilde{u}_t^\theta(X_t | y) dt$（从 $t=0$ 到 $t=1$ 模拟 ODE）；
3. 输出样本 $X_1$（目标是让 $X_1$ 符合引导变量 $y$）。

需要注意的是，当 $w > 1$ 时，$X_1$ 的分布不再严格服从 $p_{\text{data}}(\cdot | y)$，但实验表明其与引导条件的契合度显著提升。图 11 展示了在 128×128 ImageNet 数据集上基于类别的无分类器引导效果[10]：引导尺度 $w=4$ 时，生成样本与“柯基犬”类别的契合度明显高于无引导（$w=1$）的情况。图 12 则展示了在 MNIST 数据集上不同引导尺度的效果，后续实验中你将亲自复现类似结果。

### 算法 5 高斯概率路径的无分类器引导训练（$p_t(x | z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)$）
**输入**：配对数据集 $(z, y) \sim p_{\text{data}}$、神经网络 $u_t^\theta$
1. 对于每个迷你批次数据：
   - 从数据集中采样样本 $(z, y)$；
   - 从均匀分布 $\text{Unif}[0,1]$ 中采样时间 $t$；
   - 从高斯分布 $\mathcal{N}(0, I_d)$ 中采样噪声 $\epsilon$；
   - 构造中间样本 $x = \alpha_t z + \beta_t \epsilon$；
   - 以概率 $p$ 丢弃标签：$y \leftarrow \emptyset$；
   - 计算损失：$\mathcal{L}(\theta) = \left\| u_t^\theta(x | y) - \left( \dot{\alpha}_t \epsilon + \dot{\beta}_t z \right) \right\|^2$；
   - 通过梯度下降更新模型参数 $\theta$。
2. 重复上述步骤直至训练收敛。

### 5.1.2 扩散模型的引导
将无分类器引导的思路扩展到扩散模型：首先，类比式 (64)，将条件得分匹配目标（式 (61)）推广为**引导式条件得分匹配目标**：
$$
\mathcal{L}_{\text{CSM}}^{\text{guided}}(\theta) = \mathbb{E}_{\square} \left\| s_t^\theta(x | y) - \nabla \log p_t(x | z) \right\|^2 \tag{73}
$$
其中 $\square = (z, y) \sim p_{\text{data}}(z, y), t \sim \text{Unif}, x \sim p_t(\cdot | z)$ (74)。

训练得到引导式得分网络 $s_t^\theta(x | y)$ 后，可结合引导式向量场 $u_t^\theta(x | y)$ 模拟 SDE：
$$
X_0 \sim p_{\text{init}}, \quad dX_t = \left[ u_t^\theta(X_t | y) + \frac{\sigma_t^2}{2} s_t^\theta(X_t | y) \right] dt + \sigma_t dW_t
$$

#### 无分类器引导的扩展
结合贝叶斯法则（式 (67)）：
$$
\nabla \log p_t(x | y) = \nabla \log p_t(x) + \nabla \log p_t(y | x)
$$
对于引导尺度 $w > 1$，定义**无分类器引导得分**：
$$
\begin{aligned}
\tilde{s}_t(x | y) & = \nabla \log p_t(x) + w \nabla \log p_t(y | x) \\
& = \nabla \log p_t(x) + w \left( \nabla \log p_t(x | y) - \nabla \log p_t(x) \right) \\
& = (1 - w) \nabla \log p_t(x) + w \nabla \log p_t(x | y) \\
& = (1 - w) \nabla \log p_t(x | \emptyset) + w \nabla \log p_t(x | y)
\end{aligned}
$$

进而得到适配无分类器引导的目标（即允许 $y = \emptyset$）：
$$
\mathcal{L}_{\text{CSM}}^{\text{CFG}}(\theta) = \mathbb{E}_{\square} \left\| s_t^\theta(x | y) - \nabla \log p_t(x | z) \right\|^2 \tag{75}
$$
其中 $\square = (z, y) \sim p_{\text{data}}(z, y), t \sim \text{Unif}[0,1), x \sim p_t(\cdot | z), \text{以概率 } \eta \text{ 将 } y \text{ 替换为 } \emptyset$ (76)，$\eta$ 为超参数（标签替换概率）。该目标称为**引导式条件得分匹配目标**。

### 总结 28（扩散模型的无分类器引导）
给定无条件边际得分 $\nabla \log p_t(x | \emptyset)$、引导边际得分场 $\nabla \log p_t(x | y)$ 和引导尺度 $w > 1$，无分类器引导得分定义为：
$$
\tilde{s}_t(x | y) = (1 - w) \nabla \log p_t(x | \emptyset) + w \nabla \log p_t(x | y) \tag{77}
$$

通过单个神经网络 $s_t^\theta(x | y)$ 同时逼近 $\nabla \log p_t(x | \emptyset)$ 和 $\nabla \log p_t(x | y)$，可使用**无分类器引导的条件得分匹配目标（CFG-CSM）**：
$$
\mathcal{L}_{\text{CSM}}^{\text{CFG}}(\theta) = \mathbb{E}_{\square} \left\| s_t^\theta(x | y) - \nabla \log p_t(x | z) \right\|^2 \tag{78}
$$
其中 $\square = (z, y) \sim p_{\text{data}}(z, y), t \sim \text{Unif}[0,1), x \sim p_t(\cdot | z), \text{以概率 } \eta \text{ 将 } y \text{ 替换为 } \emptyset$ (79)。

其直观流程为：
1. 从数据分布中采样 $(z, y)$；
2. 从 $[0,1]$ 均匀分布中采样时间 $t$；
3. 从条件路径 $p_t(x | z)$ 中采样 $x$；
4. 以概率 $\eta$ 将 $y$ 替换为 $\emptyset$；
5. 计算损失：$\mathcal{L}(\theta) = \left\| s_t^\theta(x | y) - \nabla \log p_t(x | z) \right\|^2$（让模型逼近条件得分）。

推理时，对于固定引导尺度 $w > 1$，结合 $s_t^\theta(x | y)$ 和引导式向量场 $u_t^\theta(x | y)$，定义：
$$
\tilde{s}_t^\theta(x | y) = (1 - w) s_t^\theta(x | \emptyset) + w s_t^\theta(x | y),
$$
$$
\tilde{u}_t^\theta(x | y) = (1 - w) u_t^\theta(x | \emptyset) + w u_t^\theta(x | y).
$$

生成流程为：
1. 初始化：$X_0 \sim p_{\text{init}}$（从简单分布如高斯分布中采样）；
2. SDE 模拟：$dX_t = \left[ \tilde{u}_t^\theta(X_t | y) + \frac{\sigma_t^2}{2} \tilde{s}_t^\theta(X_t | y) \right] dt + \sigma_t dW_t$（从 $t=0$ 到 $t=1$ 模拟 SDE）；
3. 输出样本 $X_1$（目标是让 $X_1$ 符合引导变量 $y$）。

## 5.2 神经网络架构
接下来讨论流模型和扩散模型的神经网络设计，核心问题是：如何构建参数化（引导式）向量场 $u_t^\theta(x | y)$ 的架构？该神经网络需满足：输入为向量 $x \in \mathbb{R}^d$、引导变量 $y \in \mathcal{Y}$ 和时间 $t \in [0,1]$，输出为向量 $u_t^\theta(x | y) \in \mathbb{R}^d$。

对于低维分布（如前文的玩具分布），使用多层感知机（MLP，又称全连接神经网络）即可：将 $x$、$y$、$t$ 拼接后输入 MLP。但对于图像、视频、蛋白质结构等复杂高维分布，MLP 难以胜任，需使用专门的任务适配架构。本节聚焦图像（及视频）生成，重点介绍两种主流架构：U-Net[25] 和扩散Transformer（DiT）。

### 5.2.1 U-Net 与扩散 Transformer
首先回顾：图像可表示为向量 $x \in \mathbb{R}^{C_{\text{image}} \times H \times W}$，其中 $C_{\text{image}}$ 为通道数（RGB 图像通常 $C_{\text{image}} = 3$），$H$ 为图像高度（像素数），$W$ 为图像宽度（像素数）。

#### U-Net
U-Net 是一种卷积神经网络（CNN），最初为图像分割任务设计，其核心优势是**输入与输出的形状一致**（可能通道数不同）——这使其非常适合参数化向量场 $x \mapsto u_t^\theta(x | y)$（固定 $y$ 和 $t$ 时，输入和输出均为图像形状）。因此，U-Net 在扩散模型的发展中被广泛应用。

U-Net 的结构由一系列编码器 $E_i$、解码器 $D_i$ 以及中间的 latent 处理块（称为 midcoder，非通用术语）组成。以输入图像 $x_t \in \mathbb{R}^{3 \times 256 \times 256}$（即 $C_{\text{input}}=3, H=256, W=256$）为例，其处理流程如下：
1. 输入：$x_t^{\text{input}} \in \mathbb{R}^{3 \times 256 \times 256}$（U-Net 的输入图像）；
2. 编码：通过编码器得到 latent 表示 $x_t^{\text{latent}} = \mathcal{E}(x_t^{\text{input}}) \in \mathbb{R}^{512 \times 32 \times 32}$（通道数增加，高度和宽度减小）；
3. 中间处理：通过 midcoder 处理 latent 表示 $x_t^{\text{latent}} = \mathcal{M}(x_t^{\text{latent}}) \in \mathbb{R}^{512 \times 32 \times 32}$；
4. 解码：通过解码器得到输出 $x_t^{\text{output}} = \mathcal{D}(x_t^{\text{latent}}) \in \mathbb{R}^{3 \times 256 \times 256}$（通道数还原，高度和宽度恢复）。

需要补充两点细节：一是输入 $x_t^{\text{input}}$ 通常先经过初始预编码块，增加通道数后再输入第一个编码器；二是编码器与解码器之间常通过残差连接（residual connection）传递信息（完整结构见图 13）。

从宏观上看，大多数 U-Net 均遵循上述核心流程，但具体实现可能存在差异。例如，本笔记描述的是纯卷积架构，而实际应用中常在编码器和解码器中加入注意力层（attention layer）。U-Net 的名称源于其“U”形结构——编码器沿“左-中”方向通道数递增、尺寸递减，解码器沿“中-右”方向通道数递减、尺寸递增，整体形成 U 形（见图 13）。

#### 扩散 Transformer（Diffusion Transformers, DiT）
U-Net 的一种替代方案是扩散 Transformer（DiT），其摒弃了卷积操作，完全基于注意力机制[35, 19]。DiT 基于视觉Transformer（ViT），核心思路是：将图像分割为多个补丁（patch），对每个补丁进行嵌入（embed），然后通过补丁间的注意力交互捕捉全局信息[6]。Stable Diffusion 3 采用条件流匹配训练，其速度场 $u_t^\theta(x)$ 即通过改进后的 DiT 实现（见 5.3 节[7]）。

### 注 29（ latent 空间训练）
大规模应用中面临的一个共性问题是数据维度过高，导致内存消耗巨大。例如，生成 1000×1000 像素的高分辨率图像，其维度可达百万级。为解决这一问题，主流方案是在 latent 空间中训练——将流模型或扩散模型与（变分）自编码器（autoencoder）结合[24]：
1. 训练阶段：先通过自编码器将训练数据编码到低分辨率的 latent 空间，再在 latent 空间中训练流模型或扩散模型；
2. 生成阶段：先在 latent 空间中通过训练好的模型采样，再通过自编码器的解码器将 latent 表示还原为图像。

直观来看，训练良好的自编码器能过滤掉语义无关的细节，让生成模型聚焦于重要的感知相关特征[24]。目前，几乎所有最先进的图像和视频生成模型均采用“latent 空间+自编码器”的架构（称为 latent 扩散模型[24, 34]）。需要注意的是，训练扩散模型前需先训练自编码器，且模型性能很大程度上依赖自编码器的压缩质量和图像还原能力。

### 5.2.2 引导变量的编码
前文未详细说明引导变量 $y$ 如何输入神经网络 $u_t^\theta(x | y)$。该过程大致分为两步：一是将原始输入 $y_{\text{raw}}$（如文本提示“一只吹生日蜡烛的猫，照片级真实感”）嵌入为向量形式 $y$；二是将 $y$ 注入到模型中。以下详细介绍这两步的实现方式。

#### 原始输入的嵌入
分两种场景讨论：
1. 离散类别标签：若 $y_{\text{raw}} \in Y \triangleq \{0, ..., N\}$ 是离散标签，最简单的方式是为每个标签学习一个独立的嵌入向量，将 $y$ 设为该向量。嵌入向量的参数属于 $u_t^\theta(x | y)$ 的一部分，随模型一起训练。
2. 文本提示：若 $y_{\text{raw}}$ 是文本提示，需依赖预训练的冻结模型将其转换为连续向量（捕捉文本语义）。常用模型如 CLIP（Contrastive Language-Image Pretraining）：CLIP 通过对比学习训练，构建了图像和文本的共享嵌入空间——图像嵌入与对应文本提示的嵌入距离较近，与其他文本的嵌入距离较远[22]。因此，可直接使用冻结的预训练 CLIP 模型，将文本提示 $y_{\text{raw}}$ 转换为嵌入向量 $y = \text{CLIP}(y_{\text{raw}}) \in \mathbb{R}^{d_{\text{CLIP}}}$。

部分场景下，不希望将整个文本序列压缩为单个向量，此时可使用预训练 Transformer 对文本提示进行嵌入，得到序列形式的嵌入向量。此外，为充分利用不同预训练模型的优势，条件生成中常组合多种预训练嵌入（如 CLIP 嵌入+Transformer 序列嵌入）[7, 21]。

#### 嵌入向量的注入
得到嵌入向量 $y \in \mathbb{R}^{d_y}$ 后，需将其注入图像生成架构的各个子组件。以实验中使用的 U-Net 实现为例（见图 13），注入流程可通过以下 PyTorch 风格的伪代码实现：
1. $y = \text{MLP}(y) \in \mathbb{R}^C$（将 $y$ 从 $\mathbb{R}^{d_y}$ 映射到与当前网络层通道数 $C$ 一致的向量）；
2. $y = \text{reshape}(y) \in \mathbb{R}^{C \times 1 \times 1}$（将 $y$ 重塑为“图像形状”，便于广播）；
3. $x_t^{\text{intermediate}} = \text{broadcast\_add}(x_t^{\text{intermediate}}, y) \in \mathbb{R}^{C \times H \times W}$（通过广播加法将 $y$ 的信息注入当前特征图）。

若嵌入向量是序列形式（如预训练 Transformer 输出），则需采用跨注意力（cross-attention）机制：将图像补丁化（patchify）后，与文本嵌入序列进行跨注意力交互。5.3 节将给出多个此类实例。

## 5.3 大规模图像和视频模型综述
本节将深入剖析两款大规模生成模型：图像生成模型 Stable Diffusion 3 和视频生成模型 Meta 的 Movie Gen Video[7, 21]。这些模型均基于前文介绍的技术，并通过架构增强实现了大规模扩展，能够处理文本提示等复杂结构化引导模态。

### 5.3.1 Stable Diffusion 3
Stable Diffusion 是一系列最先进的图像生成模型，也是首批大规模应用 latent 扩散模型的架构之一。建议你通过官网（https://stability.ai/news/stable-diffusion-3）亲自体验其生成效果。

Stable Diffusion 3 采用的条件流匹配目标与本笔记介绍的完全一致（见算法 5）3。其论文指出，团队对多种流模型和扩散模型的变体进行了广泛测试，最终发现流匹配的性能最优。训练过程中，Stable Diffusion 3 采用无分类器引导（通过丢弃标签实现），且遵循 5.2 节的 latent 空间训练思路——在预训练自编码器的 latent 空间中训练。值得一提的是，训练高质量自编码器是早期 Stable Diffusion 系列的核心贡献之一。

为增强文本条件的表达能力，Stable Diffusion 3 结合了三种不同类型的文本嵌入：包括 CLIP 嵌入、Google T5-XXL 编码器的预训练序列输出[23]，以及类似[3, 26]的嵌入方式。其中，CLIP 嵌入提供文本的全局粗粒度语义，T5 嵌入提供更细粒度的上下文信息，使模型能够关注文本提示中的特定元素。为适配序列形式的上下文嵌入，研究人员对扩散 Transformer 进行了扩展：除了图像补丁间的自注意力，还引入了图像补丁与文本嵌入序列的跨注意力，将 DiT 原本基于类别的条件能力扩展到序列上下文，这种改进后的 DiT 称为**多模态 DiT（MM-DiT）**（结构见图 15）。Stable Diffusion 3 的最大模型含 80 亿参数，采样时采用欧拉法模拟，步数为 50 步（即需评估网络 50 次），无分类器引导权重设置在 2.0-5.0 之间。

> 3 其论文中采用了不同的噪声条件化约定，但本质上与本笔记的算法一致。

### 5.3.2 Meta Movie Gen Video
接下来介绍 Meta 的视频生成模型 Movie Gen Video（官网：https://ai.meta.com/research/movie-gen/）。与图像不同，视频数据 $x$ 的空间为 $\mathbb{R}^{T \times C \times H \times W}$，其中 $T$ 是新增的时间维度（即帧数）。该模型的设计思路是：将图像生成的现有技术（如自编码器、扩散 Transformer）适配到视频场景，以处理额外的时间维度。

Movie Gen Video 采用条件流匹配目标，使用与算法 5 相同的 CondOT 路径。与 Stable Diffusion 3 类似，它也在冻结的预训练自编码器的 latent 空间中训练——对于视频而言，自编码器的内存压缩作用更为关键，这也是目前大多数视频生成器的视频长度受限的原因。具体来说，研究人员引入了**时间自编码器（TAE）**，将原始视频 $x_t' \in \mathbb{R}^{T' \times 3 \times H \times W}$ 映射到 latent 表示 $x_t \in \mathbb{R}^{T \times C \times H \times W}$，其中压缩比为 $\frac{T'}{T} = \frac{H'}{H} = \frac{W'}{W} = 8$ [21]。为处理长视频，模型采用时间分片（temporal tiling）策略：将长视频分割为多个片段，分别编码后拼接 latent 表示[21]。

模型的核心部分（即 $u_t^\theta(x_t)$）基于 DiT 架构：将 $x_t$ 沿时间和空间维度同时补丁化，图像补丁通过 Transformer 进行两类注意力交互——补丁间的自注意力，以及与语言模型嵌入的跨注意力（与 Stable Diffusion 3 的 MM-DiT 类似）。文本条件方面，Movie Gen Video 结合了三种文本嵌入：UL2 嵌入（用于细粒度文本推理[33]）、ByT5 嵌入（用于关注字符级细节，例如提示中明确要求出现的文本[36]），以及 MetaCLIP 嵌入（在文本-图像共享嵌入空间中训练[13, 21]）。其最大模型含 300 亿参数，更详细的技术细节可参考 Movie Gen 的技术报告[21]。

是否需要我针对某款模型的架构细节（如 MM-DiT 的跨注意力实现、时间自编码器的工作原理）展开更具体的说明？


# A 概率论知识点回顾
本节将简要概述概率论的核心基础概念，部分内容参考自[15]。

## A.1 随机向量
考虑 d 维欧几里得空间中的数据 $x=(x^1, ..., x^d) \in \mathbb{R}^d$，其标准欧几里得内积定义为 $<x, y> = \sum_{i=1}^d x^i y^i$，范数定义为 $\|x\| = \sqrt{<x, x>}$。我们重点关注具有**连续概率密度函数（PDF）** 的随机变量（RVs）$X \in \mathbb{R}^d$——概率密度函数是一个连续函数 $p_X: \mathbb{R}^d \to \mathbb{R}_{≥0}$，事件 $A$ 发生的概率可通过以下积分计算：
$$
\mathbb{P}(X \in A) = \int_A p_X(x) dx \tag{80}
$$
其中 $\int p_X(x) dx = 1$（概率归一化条件）。为简化表述，我们约定：积分区间未明确标注时，默认对整个 d 维空间积分（即 $\int \equiv \int_{\mathbb{R}^d}$）；随机变量 $X_t$ 的概率密度函数 $p_{X_t}$ 简记为 $p_t$；用 $X \sim p$ 或 $X \sim p(X)$ 表示随机变量 $X$ 服从分布 $p$。

生成建模中最常用的概率密度函数之一是**d 维各向同性高斯分布**，其形式为：
$$
\mathcal{N}(x; \mu, \sigma^2 I) = (2\pi \sigma^2)^{-\frac{d}{2}} \exp\left(-\frac{\|x - \mu\|_2^2}{2\sigma^2}\right) \tag{81}
$$
其中 $\mu \in \mathbb{R}^d$ 是分布的均值，$\sigma \in \mathbb{R}_{>0}$ 是标准差。

### 期望与无意识统计学家法则
随机变量的**期望**是在最小二乘意义下与 $X$ 最接近的常数向量，其计算公式为：
$$
\mathbb{E}[X] = \underset{z \in \mathbb{R}^d}{arg\ min} \int \|x - z\|^2 p_X(x) dx = \int x p_X(x) dx
$$

计算随机变量函数的期望时，可借助**无意识统计学家法则（Law of the Unconscious Statistician）**，无需先求解函数的分布，直接通过以下公式计算：
$$
\mathbb{E}[f(X)] = \int f(x) p_X(x) dx \tag{83}
$$
必要时，我们会明确标注期望所针对的随机变量，例如 $\mathbb{E}_X f(X)$。

## A.2 条件密度与条件期望
给定两个随机变量 $X, Y \in \mathbb{R}^d$，它们的联合概率密度函数 $p_{X,Y}(x, y)$ 满足**边际化性质**——对联合密度在一个变量的全空间积分，可得到另一个变量的边际密度：
$$
\int p_{X,Y}(x, y) dy = p_X(x), \quad \int p_{X,Y}(x, y) dx = p_Y(y) \tag{84}
$$
图 16 展示了二维空间中（$d=1$）联合概率密度函数与其边际密度的关系。

### 条件概率密度
当 $p_Y(y) > 0$ 时，给定事件 $Y=y$ 条件下，随机变量 $X$ 的**条件概率密度函数**定义为：
$$
p_{X|Y}(x | y) := \frac{p_{X,Y}(x, y)}{p_Y(y)}
$$
同理可定义 $p_{Y|X}(y | x)$。根据**贝叶斯法则**，条件密度之间满足以下关系：
$$
p_{Y|X}(y | x) = \frac{p_{X|Y}(x | y) p_Y(y)}{p_X(x)}
$$
其中 $p_X(x) > 0$。

### 条件期望
条件期望 $\mathbb{E}[X | Y]$ 是在最小二乘意义下逼近 $X$ 的最优函数 $g_*(Y)$，即：
$$
\begin{aligned}
g_* & := \underset{g: \mathbb{R}^d \to \mathbb{R}^d}{arg\ min} \mathbb{E}\left[\|X - g(Y)\|^2\right] \\
& = \underset{g: \mathbb{R}^d \to \mathbb{R}^d}{arg\ min} \int \|x - g(y)\|^2 p_{X,Y}(x, y) dx dy \\
& = \underset{g: \mathbb{R}^d \to \mathbb{R}^d}{arg\ min} \int \left[\int \|x - g(y)\|^2 p_{X|Y}(x | y) dx\right] p_Y(y) dy
\end{aligned}
$$
对于满足 $p_Y(y) > 0$ 的 $y \in \mathbb{R}^d$，条件期望函数的具体形式为：
$$
\mathbb{E}[X | Y=y] := g_*(y) = \int x p_{X|Y}(x | y) dx \tag{88}
$$
将最优函数 $g_*$ 与随机变量 $Y$ 复合，得到条件期望 $\mathbb{E}[X | Y] := g_*(Y)$——这是一个取值于 $\mathbb{R}^d$ 的随机变量。

需要注意的是，$\mathbb{E}[X | Y=y]$ 与 $\mathbb{E}[X | Y]$ 是两个不同的概念：前者是从 $\mathbb{R}^d$ 到 $\mathbb{R}^d$ 的函数，后者是随机变量。为避免混淆，本节及后续内容均采用上述定义的符号。

### 核心性质
1. **塔性质（Tower Property）**：该性质可简化涉及两个随机变量的条件期望推导，其核心是“条件期望的期望等于原变量的期望”：
$$
\mathbb{E}\left[\mathbb{E}\left[X | Y\right]\right] = \mathbb{E}\left[X\right] \tag{90}
$$
证明：利用条件期望的定义和边际化性质，可验证：
$$
\begin{aligned}
\mathbb{E}[\mathbb{E}[X | Y]] & = \int \left(\int x p_{X|Y}(x | y) dx\right) p_Y(y) dy \\
& \stackrel{(85)}{=} \iint x p_{X,Y}(x, y) dx dy \\
& \stackrel{(84)}{=} \int x p_X(x) dx \\
& = \mathbb{E}[X]
\end{aligned}
$$

2. **联合函数的条件期望**：对于任意两个随机变量 $X, Y$ 及函数 $f(X, Y)$，结合无意识统计学家法则与条件期望的定义，可得到以下恒等式：
$$
\mathbb{E}[f(X, Y) | Y=y] = \int f(x, y) p_{X|Y}(x | y) dx \tag{91}
$$

是否需要我对某个具体概念（如高斯分布的期望计算、贝叶斯法则的应用示例）补充更详细的推导或说明？


# B 福克-普朗克方程（Fokker-Planck equation）的证明
本节将给出福克-普朗克方程（定理 15）的完整自洽证明，连续性方程（定理 12）是该方程的特例。需要说明的是，本节内容并非理解文档其余部分的必要前提，且数学难度较高。若你希望深入了解福克-普朗克方程的推导过程，可继续阅读。

## 证明核心思路
我们首先证明福克-普朗克方程的**必要性**：若随机过程 $X_t \sim p_t$（即 $X_t$ 服从边际概率路径 $p_t$），则 $p_t$ 必然满足福克-普朗克方程。证明的关键技巧是使用**测试函数（test functions）** ——这类函数 $f: \mathbb{R}^d \to \mathbb{R}$ 需满足“无穷可微（光滑）”且“仅在有界域内非零（紧支撑）”。

我们利用以下核心性质：对于任意可积函数 $g_1, g_2: \mathbb{R}^d \to \mathbb{R}$，其逐点相等等价于对所有测试函数 $f$ 的积分相等，即：
$$
g_1(x) = g_2(x) \quad \forall x \in \mathbb{R}^d \quad \Leftrightarrow \quad \int f(x) g_1(x) dx = \int f(x) g_2(x) dx \quad \forall \text{测试函数 } f \tag{92}
$$
换句话说，可通过积分相等间接证明逐点相等。测试函数的优势在于光滑性，允许我们对其求梯度及更高阶导数，且可利用分部积分简化计算。

## 关键积分恒等式
基于散度（div）和拉普拉斯（Δ）算子的定义（见式 (23)），结合分部积分，可推导得到以下两个核心恒等式：
1. 对于函数 $f_1: \mathbb{R}^d \to \mathbb{R}$（标量函数）和 $f_2: \mathbb{R}^d \to \mathbb{R}^d$（向量函数）：
$$
\int \nabla f_1^T(x) f_2(x) dx = -\int f_1(x) \, \text{div}(f_2)(x) dx \tag{94}
$$
2. 对于两个标量函数 $f_1: \mathbb{R}^d \to \mathbb{R}$ 和 $f_2: \mathbb{R}^d \to \mathbb{R}$：
$$
\int f_1(x) \, \Delta f_2(x) dx = \int f_2(x) \, \Delta f_1(x) dx \tag{95}
$$

## 必要性证明（若 $X_t \sim p_t$，则满足福克-普朗克方程）
### 步骤 1：SDE 轨迹的近似展开
首先回顾 SDE 轨迹的随机更新形式（见式 (6)）：
$$
X_{t+h} = X_t + h u_t(X_t) + \sigma_t (W_{t+h} - W_t) + h R_t(h)
$$
其中 $R_t(h)$ 是随机误差项，满足 $h \to 0$ 时 $R_t(h) \to 0$。为简化可读性，暂忽略误差项，近似为：
$$
X_{t+h} \approx X_t + h u_t(X_t) + \sigma_t (W_{t+h} - W_t)
$$
其中 $W_t$ 是布朗运动，满足 $W_{t+h} - W_t \mid X_t \sim \mathcal{N}(0, h I_d)$（增量服从均值为 0、方差为 $h I_d$ 的高斯分布），且 $\mathbb{E}[W_{t+h} - W_t \mid X_t] = 0$（增量条件期望为 0）。

### 步骤 2：测试函数的泰勒展开
对测试函数 $f$ 在 $X_t$ 处进行二阶泰勒展开，得到：
$$
\begin{aligned}
f(X_{t+h}) - f(X_t) &= f\left(X_t + h u_t(X_t) + \sigma_t (W_{t+h} - W_t)\right) - f(X_t) \\
&= \nabla f(X_t)^T \left( h u_t(X_t) + \sigma_t (W_{t+h} - W_t) \right) \\
&\quad + \frac{1}{2} \left( h u_t(X_t) + \sigma_t (W_{t+h} - W_t) \right)^T \nabla^2 f(X_t) \left( h u_t(X_t) + \sigma_t (W_{t+h} - W_t) \right)
\end{aligned}
$$
其中 $\nabla^2 f(X_t)$ 是 $f$ 在 $X_t$ 处的黑塞矩阵（Hessian matrix），且因黑塞矩阵对称，展开后可整理为：
$$
\begin{aligned}
f(X_{t+h}) - f(X_t) &= h \nabla f(X_t)^T u_t(X_t) + \sigma_t \nabla f(X_t)^T (W_{t+h} - W_t) \\
&\quad + \frac{1}{2} h^2 u_t(X_t)^T \nabla^2 f(X_t) u_t(X_t) + h \sigma_t u_t(X_t)^T \nabla^2 f(X_t) (W_{t+h} - W_t) \\
&\quad + \frac{1}{2} \sigma_t^2 (W_{t+h} - W_t)^T \nabla^2 f(X_t) (W_{t+h} - W_t)
\end{aligned}
$$

### 步骤 3：条件期望化简
对等式两边取关于 $X_t$ 的条件期望 $\mathbb{E}[\cdot \mid X_t]$。由于 $W_{t+h} - W_t$ 与 $X_t$ 独立，且其条件期望为 0，含 $W_{t+h} - W_t$ 的线性项会消失，仅保留常数项和二次项：
$$
\begin{aligned}
\mathbb{E}\left[f(X_{t+h}) - f(X_t) \mid X_t\right] &= h \nabla f(X_t)^T u_t(X_t) + \frac{1}{2} h^2 u_t(X_t)^T \nabla^2 f(X_t) u_t(X_t) \\
&\quad + \frac{1}{2} \sigma_t^2 \mathbb{E}_{\epsilon_t \sim \mathcal{N}(0, I_d)} \left[ \epsilon_t^T \nabla^2 f(X_t) \epsilon_t \right]
\end{aligned}
$$
其中 $\epsilon_t = \frac{W_{t+h} - W_t}{\sqrt{h}} \sim \mathcal{N}(0, I_d)$（标准化后的噪声）。利用高斯随机向量的性质 $\mathbb{E}[\epsilon_t^T A \epsilon_t] = \text{trace}(A)$（$A$ 为对称矩阵，$\text{trace}(A)$ 表示矩阵的迹），且拉普拉斯算子 $\Delta f = \text{trace}(\nabla^2 f)$，可进一步化简：
$$
\begin{aligned}
\mathbb{E}\left[f(X_{t+h}) - f(X_t) \mid X_t\right] &= h \nabla f(X_t)^T u_t(X_t) + \frac{1}{2} h^2 u_t(X_t)^T \nabla^2 f(X_t) u_t(X_t) \\
&\quad + \frac{h}{2} \sigma_t^2 \Delta f(X_t)
\end{aligned}
$$

### 步骤 4：时间导数与积分变换
对 $\mathbb{E}[f(X_t)]$ 求时间导数 $\partial_t \mathbb{E}[f(X_t)]$，利用导数的定义（$h \to 0$ 时）：
$$
\partial_t \mathbb{E}[f(X_t)] = \lim_{h \to 0} \frac{1}{h} \mathbb{E}\left[f(X_{t+h}) - f(X_t)\right]
$$
将步骤 3 的条件期望代入，并交换极限与期望的顺序（因测试函数紧支撑，满足控制收敛定理条件），忽略 $h^2$ 项（$h \to 0$ 时趋于 0），得到：
$$
\partial_t \mathbb{E}[f(X_t)] = \mathbb{E}\left[ \nabla f(X_t)^T u_t(X_t) + \frac{1}{2} \sigma_t^2 \Delta f(X_t) \right]
$$
由于 $X_t \sim p_t$，利用无意识统计学家法则，将期望转化为积分：
$$
\partial_t \int f(x) p_t(x) dx = \int \nabla f(x)^T u_t(x) p_t(x) dx + \frac{1}{2} \sigma_t^2 \int f(x) \Delta p_t(x) dx
$$
对等式右边第一项应用积分恒等式 (94)，左边交换导数与积分的顺序（因 $p_t$ 光滑），最终得到：
$$
\int f(x) \partial_t p_t(x) dx = \int f(x) \left( - \text{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \right) dx
$$
根据测试函数的核心性质 (92)，两边积分相等意味着逐点相等，即：
$$
\partial_t p_t(x) = - \text{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \quad \forall x \in \mathbb{R}^d, \, 0 \leq t \leq 1
$$
这正是福克-普朗克方程，必要性得证。

## 充分性证明（若满足福克-普朗克方程，则 $X_t \sim p_t$）
福克-普朗克方程是一类抛物型偏微分方程（PDE）。根据偏微分方程理论，给定固定初始条件（如 $p_0 = p_{\text{init}}$），这类方程存在唯一解。

若 $p_t$ 满足福克-普朗克方程，且 $X_t$ 的真实分布 $q_t$ 也满足该方程（必要性已证明），同时两者初始条件一致（$p_0 = q_0 = p_{\text{init}}$），则由解的唯一性可知 $p_t = q_t$ 对所有 $0 \leq t \leq 1$ 成立，即 $X_t \sim p_t$。充分性得证。

## 特例：连续性方程
当扩散系数 $\sigma_t = 0$ 时，福克-普朗克方程中的拉普拉斯项消失，退化为连续性方程：
$$
\partial_t p_t(x) = - \text{div}(p_t u_t)(x)
$$
这与定理 12 完全一致，说明连续性方程是福克-普朗克方程在无扩散（确定性动力学）情况下的特例。

是否需要我对证明中的某个步骤（如泰勒展开、积分恒等式应用）补充更详细的推导说明？