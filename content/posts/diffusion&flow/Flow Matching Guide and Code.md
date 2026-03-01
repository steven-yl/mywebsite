---
title: "Flow Matching Guide and Code"
subtitle: ""
date: 2026-02-28T10:26:59+08:00
# lastmod: 2026-02-28T10:26:59+08:00
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
featuredImagePreview: "/mywebsite/posts/images/flow-matching-guide-and-code.webp"


---

## 1 Introduction
### 核心定位
该章节作为《Flow Matching Guide and Code》的开篇，核心目标是明确**Flow Matching（FM）** 这一生成建模框架的核心价值、技术本质、发展脉络与应用范围，同时清晰勾勒出论文的整体结构与目标受众，为后续的技术细节铺垫基础。

### 关键内容拆解
#### 1.1 Flow Matching 的定义与核心地位
- **本质属性**：FM 是一种**简单且高效的生成建模框架**，核心是通过学习“速度场（velocity field）”定义“流（flow）”，进而实现从“源分布（source distribution）”到“目标分布（target distribution）”的确定性、时间连续的双射变换。
- **核心目标**：将从源分布 $p$ 采样的样本 $X_0 \sim p$，通过流 $\psi_t$ 变换为目标分布 $q$ 的样本 $X_1 = \psi_1(X_0) \sim q$（如图1a所示），其中流 $\psi_t$ 由求解常微分方程（ODE）得到。
- **行业地位**：已在多个领域实现“当前最优（state-of-the-art）”性能，包括图像生成（Esser et al., 2024）、视频生成（Polyak et al., 2024）、语音生成（Le et al., 2024）、音频生成（Vyas et al., 2023）、蛋白质结构生成（Huguet et al., 2024）和机器人控制（Black et al., 2024）等。

#### 1.2 论文的两大核心目标
1. **学术价值**：提供一份**全面且自包含（comprehensive and self-contained）** 的 FM 参考资料，详细阐述其设计选择、数学基础及研究社区提出的各类扩展。
2. **实用价值**：降低入门门槛，通过配套的 PyTorch 代码库（flow_matching library），帮助新手快速上手并将 FM 应用于自身研究或工程场景。

#### 1.3 FM 的技术起源与发展脉络
FM 的核心思想源于“连续归一化流（Continuous Normalizing Flows, CNF）”（Chen et al., 2018; Grathwohl et al., 2018），其发展历程可概括为“从复杂到简洁”的迭代：
- **早期 CNF 局限**：最初通过最大化训练数据的对数似然（log-likelihood）训练，需要在训练过程中进行 ODE 模拟及微分，计算成本极高。
- **关键优化方向**：后续研究（Rozen et al., 2021; Ben-Hamu et al., 2022）尝试“无模拟训练”，最终演化出现代 FM 算法（Lipman et al., 2022; Liu et al., 2022; Albergo and Vanden-Eijnden, 2022 等）。
- **现代 FM 核心简化**：将生成建模简化为两步流程（如图2所示）：
  1. 设计“概率路径（probability path）” $p_t$，实现从源分布 $p（p_0=p）$ 到目标分布 $q（p_1=q）$ 的平滑插值；
  2. 训练一个神经网络建模的速度场，使其定义的流能够复现该概率路径 $p_t$。

#### 1.4 FM 的泛化能力：从欧氏空间到任意模态
FM 的核心原理具有极强的通用性，已被扩展到多种非传统场景，突破了最初的欧氏空间（$\mathbb{R}^d$）限制：
- **离散状态空间**：Discrete Flow Matching（DFM）将 FM 应用于连续时间马尔可夫链（CTMC），适用于语言建模等离散生成任务（如图1c所示）；
- **黎曼流形**：Riemannian Flow Matching 扩展到黎曼流形（如球面、矩阵李群），成为化学、生物领域（如蛋白质折叠）的最优模型（Yim et al., 2023; Bose et al., 2023）；
- **通用连续时间马尔可夫过程（CTMP）**：Generator Matching（GM）进一步泛化，证明 FM 框架可适配任意模态和任意 CTMP（包括流、扩散过程、跳跃过程等），实现了多种生成模型的统一。

#### 1.5 与 Diffusion Models 的关联
章节明确了 FM 与扩散模型（Diffusion Models）的核心联系与差异：
- **共性**：两者均属于“无模拟训练”的 CTMP 生成模型，核心都是通过学习概率路径实现分布变换；
- **差异**：扩散模型的概率路径通过“前向加噪过程”（由特定 SDE 建模）构建，且通过“分数函数（score function）”参数化生成器；而 FM 提供了更灵活的概率路径设计和生成器参数化方式，且扩散模型可被视为 FM 的一个特例（详见第10章）。

#### 1.6 论文结构预告
章节最后概述了全文的组织逻辑，帮助读者建立阅读框架：
1. 第2章：提供 FM 的“速查指南（cheat-sheet）”，含纯 PyTorch 实现代码；
2. 第3章：深入讲解流模型的数学基础（连续状态空间下最简单的 CTMP）；
3. 第4章：详细介绍 FM 框架在欧氏空间的设计选择与扩展；
4. 第5-7章：分别扩展 FM 到黎曼流形、CTMC（离散状态空间）和离散流匹配；
5. 第8-9章：泛化到通用状态空间和 CTMP，提出 Generator Matching 统一框架；
6. 第10章：深入分析 FM 与扩散模型及其他去噪模型的关联。

### 核心关键词与术语辨析
| 术语 | 英文 | 定义 |
|------|------|------|
| 流 | Flow | 时间依赖的确定性双射变换 $\psi_t: \mathbb{R}^d \to \mathbb{R}^d$，由速度场通过 ODE 定义 |
| 速度场 | Velocity Field | 时间依赖的向量场 $u_t$，规定流在每个位置的瞬时运动方向与速率 |
| 概率路径 | Probability Path | 时间依赖的分布序列 $p_t$，平滑插值源分布 $p$ 和目标分布 $q$ |
| 连续归一化流 | Continuous Normalizing Flows (CNF) | FM 的技术前身，通过 ODE 实现分布变换的生成模型 |
| 连续时间马尔可夫过程 | Continuous-Time Markov Process (CTMP) | 包含流、扩散、跳跃过程等的通用马尔可夫过程框架 |

### 章节核心贡献
1. **定位清晰**：明确 FM 作为“简单、高效、通用”生成建模框架的核心价值，区分其与 CNF、扩散模型的关系；
2. **脉络完整**：梳理 FM 从 CNF 演化而来的技术路线，帮助读者理解其设计动机；
3. **范围明确**：界定论文的覆盖边界（从欧氏空间到流形、从连续到离散），同时指明目标受众（新手与资深研究者）；
4. **结构引导**：通过章节预告降低读者的认知负荷，为后续技术细节的吸收铺垫框架。



## 2 Quick tour and key concepts
该章节作为《Flow Matching Guide and Code》的“入门指南”，核心目标是用直观的数学表达、清晰的步骤拆解和极简的代码实现，帮助读者快速掌握 Flow Matching（FM）的核心逻辑——从“问题定义”到“数学建模”，再到“训练与采样”，全程避开复杂推导，聚焦可理解性与可复现性。

### 核心定位
章节以“欧氏空间（$\mathbb{R}^d$）的基础 FM 框架”为切入点，回答了三个核心问题：
1. FM 的目标是什么？（从源分布生成目标分布的样本）
2. 如何用数学描述 FM 的核心组件？（概率路径、速度场、ODE 流）
3. 如何快速实现一个基础 FM 模型？（PyTorch 代码+关键步骤）

### 关键内容拆解
#### 2.1 核心目标：从“源分布”到“目标分布”的样本生成
FM 的本质是**生成建模**：给定来自目标分布 $q$ 的训练样本（如图像、音频等），构建一个模型，使其能生成全新的、服从 $q$ 的样本。

实现路径被简化为“两步映射”：
1. 从一个**已知且易采样的源分布 $p$**（通常是标准高斯分布 $N(0, I)$）中抽取初始样本 $X_0 \sim p$；
2. 通过一个“平滑变换”，将 $X_0$ 逐步转化为目标样本 $X_1 \sim q$。

这个“平滑变换”的核心是 **概率路径 $p_t$** 和 **速度场 $u_t$**——前者定义了“从 $p$ 到 $q$ 的中间分布序列”，后者定义了“样本如何沿该路径移动”。

#### 2.2 核心组件：数学定义与关联
章节用极简的数学语言定义了 FM 的三大核心组件，以及它们之间的逻辑关系：

##### （1）概率路径 $p_t$：连接源与目标的“桥梁”
- **定义**：一个时间依赖的分布序列 $p_t$（$t \in [0,1]$），满足边界条件：
  - $t=0$ 时，$p_0 = p$（源分布）；
  - $t=1$ 时，$p_1 = q$（目标分布）。
- **直观理解**：$p_t$ 是“源分布逐渐演变为目标分布的过程”，每个 $p_t$ 都是中间状态的分布（例如，$t=0.5$ 时的分布既保留了部分源分布的特征，也包含了部分目标分布的特征）。

##### （2）速度场 $u_t$：样本移动的“导航图”
- **定义**：一个时间依赖的向量场 $u: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$，建模为神经网络，输入是“当前时间 $t$”和“当前样本 $x_t$”，输出是“样本在 $x_t$ 处的瞬时移动方向和速率”。
- **核心作用**：速度场 $u_t$ 是生成“流”的关键——通过求解由 $u_t$ 定义的常微分方程（ODE），可得到样本的移动轨迹，即“流”。

##### （3）流 $\psi_t$：样本变换的“执行器”
- **定义**：由速度场 $u_t$ 诱导的时间依赖变换 $\psi: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$，满足以下 ODE：
  $
  \frac{d}{dt} \psi_t(x) = u_t\left(\psi_t(x)\right), \quad \psi_0(x) = x
  $
  - 初始条件 $\psi_0(x) = x$：$t=0$ 时，样本未发生变换；
  - 核心逻辑：样本 $x$ 沿 ODE 轨迹移动，$t$ 时刻的位置为 $\psi_t(x)$，$t=1$ 时到达目标位置 $\psi_1(x) \sim q$。

##### （4）三者的核心关联
速度场 $u_t$ 生成流 $\psi_t$，流 $\psi_t$ 推动样本沿概率路径 $p_t$ 移动，最终满足：
$
X_t = \psi_t(X_0) \sim p_t \quad (\forall X_0 \sim p_0)
$
即：初始样本 $X_0$ 沿流 $\psi_t$ 移动后，$t$ 时刻的样本 $X_t$ 服从中间分布 $p_t$。

#### 2.3 FM 的核心流程：路径设计→模型训练→样本生成
章节将 FM 框架拆解为“三步蓝图”（对应图 2），每一步都有明确的目标和操作：

##### 步骤 1：设计概率路径 $p_t$（图 2b）
章节选择了最常用的**线性概率路径**（也叫“条件最优传输路径”），其构造逻辑如下：
1. 先定义**条件概率路径 $p_{t|1}(x | x_1)$**：对于每个目标样本 $x_1 \sim q$，定义一个从源分布 $p$ 到 $\delta_{x_1}$（仅在 $x_1$ 处概率为 1 的delta分布）的条件路径：
   $
   p_{t|1}(x | x_1) = \mathcal{N}\left(x \mid t x_1, (1-t)^2 I\right)
   $
   - 直观理解：$t$ 时刻的条件分布是均值为 $t x_1$、方差为 $(1-t)^2 I$ 的高斯分布——$t$ 越大，均值越接近目标样本 $x_1$，方差越小，最终 $t=1$ 时收敛到 $x_1$。

2. 再通过“边缘化”得到**边际概率路径 $p_t$**：聚合所有条件路径，得到全局路径：
   $
   p_t(x) = \int p_{t|1}(x | x_1) q(x_1) dx_1
   $
   - 等价采样方式：为了避免积分，可通过“源样本和目标样本的线性组合”直接采样 $X_t$：
     $
     X_t = t X_1 + (1-t) X_0 \quad (X_0 \sim p, X_1 \sim q)
     $
   - 这是 FM 简化计算的关键：无需显式建模 $p_t$，只需通过线性组合即可生成中间样本 $X_t$。

##### 步骤 2：训练速度场 $u_t^\theta$（图 2c）
训练的核心是“回归任务”：让模型学习的速度场 $u_t^\theta$ 逼近“真实速度场 $u_t$”，后者是使样本沿概率路径 $p_t$ 移动的“理想导航图”。

###### （1）真实速度场 $u_t$ 的简化
直接计算 $u_t$ 复杂，但通过“条件路径”可推导得**条件真实速度场**：
$
u_t(x | x_1) = \frac{x_1 - x}{1-t}
$
- 直观理解：对于中间样本 $x$，其“理想移动速度”与“当前位置到目标 $x_1$ 的距离”成正比，与“剩余时间 $1-t$”成反比——确保 $t=1$ 时刚好到达 $x_1$。

###### （2）FM 损失函数
由于边际速度场 $u_t(x) = \mathbb{E}[u_t(x | X_1) | X_t = x]$，直接优化 $u_t$ 不可行，但章节证明了“边际损失”与“条件损失”的梯度等价（式 2.8），因此采用**条件 Flow Matching 损失（CFM 损失）**：
$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim U[0,1], X_0 \sim p, X_1 \sim q} \left\| u_t^\theta(X_t) - \frac{X_1 - X_t}{1-t} \right\|^2
$
- 简化到极致的形式（代入 $X_t = t X_1 + (1-t) X_0$）：
  $
  \mathcal{L}_{CFM}^{OT,Gauss}(\theta) = \mathbb{E}_{t, X_0, X_1} \left\| u_t^\theta(X_t) - (X_1 - X_0) \right\|^2
  $
- 本质：MSE 回归——让模型输出的速度，逼近“源样本到目标样本的直接差值”（因线性路径的理想速度可简化为 $X_1 - X_0$）。

##### 步骤 3：生成样本（图 2d）
训练完成后，生成新样本的流程极其简单：
1. 从源分布 $p$ 抽取初始样本 $X_0 \sim N(0, I)$；
2. 求解 ODE $\frac{d}{dt} X_t = u_t^\theta(X_t)$（从 $t=0$ 到 $t=1$）；
3. 得到 $X_1 = X_t|_{t=1}$，即为服从目标分布 $q$ 的生成样本。

#### 2.4 极简实现：PyTorch 代码解析
章节提供了一个独立的 PyTorch 实现（Code 1），核心代码仅 50 余行，完美对应“路径设计→训练→采样”三步，关键部分解析如下：

##### （1）模型定义（Flow 类）
- 输入：中间样本 $x_t$（维度 $d$）+ 时间 $t$（维度 1），拼接后输入神经网络；
- 输出：速度预测值（维度 $d$）；
- 网络结构：3 层全连接+ELU 激活，结构简单，聚焦核心逻辑。

##### （2）训练循环（核心步骤）
```python
for _ in range(10000):
    x_1 = Tensor(make_moons(256, noise=0.05)[0])  ## 目标样本（双月数据集）
    x_0 = torch.randn_like(x_1)                   ## 源样本（高斯噪声）
    t = torch.rand(len(x_1), 1)                   ## 随机时间 t~U[0,1]
    x_t = (1 - t) * x_0 + t * x_1                ## 生成中间样本（线性路径）
    dx_t = x_1 - x_0                             ## 真实速度（简化版）
    loss_fn(flow(x_t, t), dx_t).backward()        ## MSE 损失+反向传播
    optimizer.step()
```
- 数据集：用 `make_moons` 生成简单的双月数据集（便于可视化）；
- 核心操作：按线性路径生成 $x_t$，用 $x_1 - x_0$ 作为真实速度标签，训练回归模型。

##### （3）采样过程
```python
x = torch.randn(300, 2)  ## 初始源样本
for i in range(n_steps):
    x = flow.step(x, time_steps[i], time_steps[i+1])  ## 数值求解 ODE
```
- 数值解法：用“中点法”（Midpoint ODE solver）求解 ODE，避免欧拉法的累积误差；
- 可视化：输出不同时间步的样本分布，直观展示“高斯噪声→双月分布”的演变过程。

#### 2.5 核心结论与关键洞察
1. **简化是核心**：FM 避开了传统流模型（CNF）的复杂似然计算，通过“概率路径+回归损失”将生成建模转化为简单的 MSE 训练，无需 ODE 模拟或微分，计算效率极高；
2. **线性路径的优势**：章节选择的线性概率路径不仅易计算，还能保证“样本移动轨迹平滑”，降低 ODE 求解难度；
3. **条件损失的有效性**：通过“边际化技巧”，条件损失与边际损失梯度等价，既简化了计算，又不影响训练效果；
4. **低门槛实现**：基础 FM 模型仅需“全连接网络+MSE 损失+简单 ODE 求解”，无需复杂模块，适合快速上手。

### 章节核心贡献
1. **降维理解**：将 FM 的复杂数学框架拆解为“路径→训练→采样”三步，屏蔽无关细节，聚焦核心逻辑；
2. **可复现性**：提供独立、极简的 PyTorch 代码，读者可直接运行，直观观察 FM 的工作过程；
3. **铺垫基础**：为后续章节（如非欧氏空间 FM、离散 FM、与扩散模型的关联）提供了“基础模板”，后续扩展均可基于此框架修改。



## 3 Flow models
该章节是《Flow Matching Guide and Code》的“数学基础篇”，核心目标是严格定义“流模型（Flow Models）”的数学本质——作为连续时间马尔可夫过程（CTMP）中最简单的确定性模型，流模型是 Flow Matching（FM）框架的核心载体。章节从概率、微分方程、几何变换等基础理论出发，系统推导流模型的核心性质、与速度场的等价关系、概率路径的生成机制，以及似然计算方法，为后续 FM 框架的扩展（如非欧氏空间、离散空间）奠定理论基础。

### 核心定位
流模型是“确定性、时间连续的双射变换”，其核心价值在于：
1. 能将任意源分布 $p$ 转化为目标分布 $q$（只要两者有密度）；
2. 采样效率高（通过数值求解 ODE 实现）；
3. 支持无偏的模型似然估计（区别于扩散模型等随机过程）。

章节围绕“流模型的数学定义→核心性质→实际应用（采样、似然计算）”展开，所有推导均服务于“如何通过流模型实现生成建模”这一核心目标。

### 关键内容拆解
#### 3.1 前置数学基础：概率与随机向量
章节首先回顾了生成建模所需的核心概率概念，为后续推导铺垫：

##### （1）随机向量与概率密度函数（PDF）
- 考虑 $d$ 维欧氏空间 $\mathbb{R}^d$ 中的随机向量 $X$，其 PDF 满足 $p_X(x) \geq 0$ 且 $\int_{\mathbb{R}^d} p_X(x) dx = 1$；
- 事件 $A \subset \mathbb{R}^d$ 的概率为 $\mathbb{P}(X \in A) = \int_A p_X(x) dx$；
- 常用分布：$d$ 维各向同性高斯分布 $\mathcal{N}(x \mid \mu, \sigma^2 I)$，其 PDF 为：
  $
  \mathcal{N}(x \mid \mu, \sigma^2 I) = (2\pi\sigma^2)^{-d/2} \exp\left(-\frac{\|x - \mu\|_2^2}{2\sigma^2}\right)
  $

##### （2）期望与无意识统计学家法则（Law of the Unconscious Statistician, LOTUS）
- 期望定义：$\mathbb{E}[X] = \int x p_X(x) dx$，是“最小二乘意义下最接近 $X$ 的常数向量”；
- LOTUS 法则：对于任意可测函数 $f$，$\mathbb{E}[f(X)] = \int f(x) p_X(x) dx$——无需显式求解 $f(X)$ 的分布，直接通过 $X$ 的 PDF 计算期望，是后续损失函数推导的核心工具。

#### 3.2 条件密度与期望
##### （1）联合密度与边际密度
- 对于两个随机向量 $X, Y$，联合 PDF $p_{X,Y}(x,y)$ 满足边际化性质：
  $
  p_X(x) = \int p_{X,Y}(x,y) dy, \quad p_Y(y) = \int p_{X,Y}(x,y) dx
  $

##### （2）条件密度与贝叶斯法则
- 条件 PDF 定义：$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}$（要求 $p_Y(y) > 0$）；
- 贝叶斯法则：$p_{Y \mid X}(y \mid x) = \frac{p_{X \mid Y}(x \mid y) p_Y(y)}{p_X(x)}$，是后续“边际化技巧”的理论基础。

##### （3）条件期望与全期望性质
- 条件期望 $\mathbb{E}[X \mid Y = y] = \int x p_{X \mid Y}(x \mid y) dx$，是“给定 $Y = y$ 时，最小二乘意义下最接近 $X$ 的函数”；
- 全期望性质（Tower Property）：$\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$——多层期望可简化为单层期望，是后续边际速度场推导的关键工具。

#### 3.3 微分同胚与推前映射（Push-forward Map）
流模型的核心是“双射变换”，章节通过“微分同胚”和“推前映射”严格定义这一性质：


##### 1. 微分同胚（Diffeomorphism）

- **直观理解**：空间的「光滑、可逆、逆也光滑」的变形，不撕裂、不折叠、不粘连，可完美还原。
- **严格定义**：设 $M,\,N$ 为光滑流形（如 $\mathbb{R}^d$），映射 $\phi: M \to N$ 满足：
  - 双射（一一对应 + 满射）
  - 光滑（无穷可微）
  - 逆映射 $\phi^{-1}$ 也光滑
- **FM 中的意义**：FM 的生成过程由一族微分同胚 $\phi_t: \mathbb{R}^d \to \mathbb{R}^d$（$t \in [0,1]$）描述，其中 $\phi_0 = \text{id}$（恒等变换），$\phi_1$ 实现从源分布到目标分布的光滑映射。

##### 2. 推前映射（Push-forward Map）

- **直观理解**：将「空间上的分布」沿着微分同胚变换「搬运」到新空间，保持概率守恒。
- **数学定义**：设 $\phi: X \to Y$ 为可测映射，$p$ 是 $X$ 上的概率分布，则推前分布 $\phi_\sharp p$ 定义为：对任意可测集 $A \subseteq Y$，
  $$(\phi_\sharp p)(A) = p(\phi^{-1}(A)).$$
- **FM 中的核心作用**：源分布 $p_0$ 经 $\phi_t$ 推前得到中间分布 $p_t = (\phi_t)_\sharp p_0$，构成从 $p_0$ 到 $p_1$ 的连续分布路径。

###### 推前映射：分布的变换规则
- 若 $X \sim p_X$，且 $Y = \psi(X)$（$\psi$ 为 $C^1$ 微分同胚），则 $Y$ 的 PDF $p_Y$ 可通过“变量替换”推导得：
  $
  p_Y(y) = p_X\left(\psi^{-1}(y)\right) \left| \det \partial_y \psi^{-1}(y) \right|
  $
  - $\partial_y \psi^{-1}(y)$ 是逆变换的雅可比矩阵；
  - 行列式的绝对值保证了概率密度的非负性和积分守恒（$\int p_Y(y) dy = 1$）；
- 符号表示：用 $\psi_\sharp p_X$ 表示“$\psi$ 对 $p_X$ 的推前映射”，即 $p_Y = \psi_\sharp p_X$。

**详细解释**：
  $$
  p_Y(y) = p_X\left(\psi^{-1}(y)\right) \left| \det \partial_y \psi^{-1}(y) \right|
  $$
- **直观**：推前映射指“把 $X$ 上的分布 $p_X$ 沿映射 $\psi$ 搬到 $Y$ 空间”。样本 $X$ 变成 $Y=\psi(X)$，概率质量随之移动；要得到 $Y$ 的密度 $p_Y(y)$，需找到“谁被映射到 $y$”（即 $x = \psi^{-1}(y)$），再按局部体积伸缩比例修正密度。
- **公式两项的含义**：① $p_X(\psi^{-1}(y))$：$y$ 对应的源点 $x=\psi^{-1}(y)$ 处的 $X$ 的密度，即搬到 $y$ 的那部分概率来自哪、原来多密；② $\left| \det \partial_y \psi^{-1}(y) \right|$：逆映射在 $y$ 处把无穷小体积 $dy$ 拉回 $x$ 空间时，体积的缩放倍数（雅可比行列式绝对值）。若 $dx$ 对应 $dy$ 且 $dy$ 被“压缩”到更小（$|det| < 1$），则同一块概率质量在 $y$ 空间占的体积更小，故 $p_Y(y)$ 更大；反之体积被拉大则 $p_Y$ 更小。
- **变量替换推导**：对任意可测集 $A$，$P(Y \in A) = P(X \in \psi^{-1}(A)) = \int_{\psi^{-1}(A)} p_X(x) dx$。令 $y = \psi(x)$ 换元得 $dx = \left| \det \partial_y \psi^{-1}(y) \right| dy$，故 $\int_{\psi^{-1}(A)} p_X(x) dx = \int_A p_X(\psi^{-1}(y)) \left| \det \partial_y \psi^{-1}(y) \right| dy$，由 $P(Y \in A) = \int_A p_Y(y) dy$ 即得 $p_Y(y)$ 的表达式。
- **为什么用绝对值**：$\det \partial_y \psi^{-1}$ 可能为负（定向改变），但密度必须非负；换元时体积元关系为 $dx = \left| \det \partial_y \psi^{-1}(y) \right| dy$，故取绝对值才能保证 $p_Y \geq 0$ 且 $\int p_Y = 1$。
- **与测度定义一致**：前文推前测度定义为 $(\psi_\sharp p_X)(A) = p_X(\psi^{-1}(A))$；上式给出的 $p_Y$ 正是该推前测度关于 Lebesgue 测度的密度，即 $p_Y = \psi_\sharp p_X$ 的 PDF 形式。

##### 3. 变量替换公式解释（密度变换核心）

- **核心本质**：分布「搬运」时，密度需按局部拉伸/压缩比例修正，该比例由雅可比行列式的绝对值决定。

**推导思路**：推前分布满足「概率守恒」——任意可测集 $A$ 上，$X \sim p_X$ 落在 $A$ 的概率 = $Y = \phi(X)$ 落在 $\phi(A)$ 的概率，即
$$\int_A p_X(x)\,dx = \int_{\phi(A)} p_Y(y)\,dy.$$
对右边做变量替换 $y = \phi(x)$（即 $x = \phi^{-1}(y)$），由多维换元有 $\mathrm{d}y = \left| \det \nabla \phi(x) \right| \mathrm{d}x$，故
$$\int_{\phi(A)} p_Y(y)\,dy = \int_A p_Y(\phi(x)) \left| \det \nabla \phi(x) \right| dx.$$
与左边相等、且 $A$ 任意，故被积函数相等：
$$p_X(x) = p_Y(\phi(x)) \left| \det \nabla \phi(x) \right|.$$
解出 $p_Y$（用 $y$ 表示时令 $x = \phi^{-1}(y)$）即得下面两式。

**一维情形**：设 $y = \phi(x)$，$p_X(x)$ 为 $x$ 的密度，则 $y$ 的密度为
$$p_Y(y) = p_X(\phi^{-1}(y)) \cdot \left| (\phi^{-1})'(y) \right|.$$
推导：概率守恒 $\int_a^b p_X(x)dx = \int_{\phi(a)}^{\phi(b)} p_Y(y)dy$，换元 $y=\phi(x)$ 得 $\mathrm{d}y = \phi'(x)\mathrm{d}x$，故 $\int_a^b p_X dx = \int_a^b p_Y(\phi(x))|\phi'(x)|dx$，即 $p_X(x)=p_Y(\phi(x))|\phi'(x)|$，写回 $y$ 得 $p_Y(y)=p_X(\phi^{-1}(y))|(\phi^{-1})'(y)|$（因 $(\phi^{-1})'(y)=1/\phi'(x)$）。

**高维情形（FM 常用）**：设 $y = \phi(x)$（$\phi$ 为微分同胚），则推前分布的密度为
$$p_Y(y) = p_X(\phi^{-1}(y)) \cdot \left| \det \nabla \phi^{-1}(y) \right|.$$
推导：由上面 $p_X(x) = p_Y(\phi(x)) |\det \nabla \phi(x)|$，用 $x = \phi^{-1}(y)$ 代入得 $p_Y(y) = p_X(\phi^{-1}(y)) / |\det \nabla \phi(\phi^{-1}(y))|$；而链式法则给出 $\nabla \phi^{-1}(y) = [\nabla \phi(x)]^{-1}$，故 $\det \nabla \phi^{-1}(y) = 1/\det \nabla \phi(x)$，即 $|\det \nabla \phi^{-1}(y)| = 1/|\det \nabla \phi(x)|$，因此上式与 $p_Y(y) = p_X(\phi^{-1}(y)) |\det \nabla \phi^{-1}(y)|$ 等价。

**如何计算**（实践中不必对 $\phi^{-1}$ 求导）：
- **链式法则来源**：恒等式 $\phi^{-1}(\phi(x)) = x$ 两边对 $x$ 求导得 $\nabla \phi^{-1}(y)\big|_{y=\phi(x)} \cdot \nabla \phi(x) = I$，故 $\nabla \phi^{-1}(y) = [\nabla \phi(x)]^{-1}$（其中 $x = \phi^{-1}(y)$）。
- **计算步骤**：已知 $y$，欲算 $p_Y(y) = p_X(\phi^{-1}(y)) \cdot |\det \nabla \phi^{-1}(y)|$ 时：
  1. 求 $x = \phi^{-1}(y)$（解方程或用反函数）；
  2. 算 Jacobian $J = \nabla \phi(x)$（只对 $\phi$ 求导，$d\times d$ 矩阵）；
  3. $|\det \nabla \phi^{-1}(y)| = 1/|\det J|$，故 $p_Y(y) = p_X(x) / |\det J|$。
- **要点**：只需算 $\phi$ 的雅可比及其行列式，无需对 $\phi^{-1}$ 求导；若已知的是「从 $x$ 到 $y$ 的映射 $\phi$」，用等价形式 $p_Y(y) = p_X(x) \cdot |\det \nabla \phi(x)|^{-1}$ 更直接（在已知 $x$、$y=\phi(x)$ 时算 $\nabla \phi(x)$ 即可）。

**等价形式（用 $x$ 表示）**：$y = \phi(x)$ 时
$$p_Y(y) = p_X(x) \cdot \left| \det \nabla \phi(x) \right|^{-1}.$$
（即推导中得到的 $p_X(x) = p_Y(\phi(x)) |\det \nabla \phi(x)|$ 的反写。）

#### 3.4 流模型的定义与核心性质
##### （1）流模型的正式定义
流模型是“时间依赖的微分同胚族”，满足：
- 时间连续性：$\psi: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ 是 $C^r([0,1] \times \mathbb{R}^d, \mathbb{R}^d)$ 函数；
- 对每个固定 $t \in [0,1]$，$\psi_t(x) = \psi(t, x)$ 是 $\mathbb{R}^d$ 上的 $C^r$ 微分同胚；
- 生成过程：若 $X_0 \sim p$，则流模型定义的随机过程为 $X_t = \psi_t(X_0)$（$t \in [0,1]$），其边际分布为 $p_t = \psi_{t\sharp} p$（推前映射的结果）。

##### （2）流模型的马尔可夫性
对于任意 $0 \leq t < s \leq 1$，有：
$
X_s = \psi_s(X_0) = \psi_s\left(\psi_t^{-1}(X_t)\right) = \psi_{s \mid t}(X_t)
$
其中 $\psi_{s \mid t} = \psi_s \circ \psi_t^{-1}$（$\psi_s$ 与 $\psi_t^{-1}$ 的复合）。这表明：$X_s$ 的分布仅依赖于 $X_t$，与更早的状态无关——流模型是马尔可夫过程，且是**确定性马尔可夫过程**（区别于扩散模型的随机性）。

##### （3）流与速度场的等价关系
这是章节的核心结论：流 $\psi_t$ 与速度场 $u_t$ 是“一一对应”的，具体通过 ODE 关联：
1. 由速度场生成流：若速度场 $u_t: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ 是 $C^r$ 光滑的（局部 Lipschitz 连续），则通过以下 ODE 可唯一确定流 $\psi_t$：
   $
   \begin{cases}
   \frac{d}{dt} \psi_t(x) = u_t\left(\psi_t(x)\right) \quad \text{（流 ODE）} \\
   \psi_0(x) = x \quad \text{（初始条件）}
   \end{cases}
   $
   - 存在性与唯一性定理（Theorem 1）：若 $u_t$ 是 $C^r$ 光滑的，则上述 ODE 存在唯一解 $\psi_t(x)$，且 $\psi_t$ 是 $C^r$ 微分同胚；
   - 直观理解：速度场 $u_t$ 规定了每个位置 $x$ 在时刻 $t$ 的瞬时移动方向和速率，流 $\psi_t$ 是“沿这些瞬时速度积分得到的轨迹”。

2. 由流提取速度场：若 $\psi_t$ 是 $C^1$ 流，则其对应的速度场可通过“对时间求导”得到：
   $
   u_t(x) = \dot{\psi}_t\left(\psi_t^{-1}(x)\right)
   $
   其中 $\dot{\psi}_t = \frac{d}{dt} \psi_t$——本质是“先通过逆变换找到 $x$ 在 $t=0$ 时刻的初始位置，再计算该初始位置在 $t$ 时刻的瞬时速度”。

#### 3.5 概率路径与连续性方程（Continuity Equation）
流模型的核心目标是“生成从 $p_0 = p$ 到 $p_1 = q$ 的概率路径 $p_t$”，章节通过“连续性方程”建立了速度场、流、概率路径三者的关键联系。

##### （1）概率路径的定义
概率路径是时间依赖的分布序列 $p_t$（$t \in [0,1]$），满足 $p_t = \psi_{t\sharp} p_0$——即 $p_t$ 是源分布 $p_0$ 经流 $\psi_t$ 推前映射的结果。

##### （2）连续性方程：概率守恒的数学表达
若速度场 $u_t$ 生成概率路径 $p_t$（即 $X_t = \psi_t(X_0) \sim p_t$），则 $(u_t, p_t)$ 必须满足连续性方程：
$
\frac{d}{dt} p_t(x) + \text{div}\left(p_t(x) u_t(x)\right) = 0
$
- 散度（div）定义：$\text{div}(v)(x) = \sum_{i=1}^d \partial_{x^i} v^i(x)$，描述向量场在某点的“发散程度”；
- 物理意义：概率是“守恒量”——局部概率密度的变化率（$\frac{d}{dt} p_t(x)$）等于负的“概率通量的散度”（$\text{div}(p_t u_t)$），即“局部概率的增加量 = 流入的概率通量 - 流出的概率通量”（如图 8 所示）。

##### （3）质量守恒定理（Mass Conservation Theorem）
章节通过定理 2 明确了连续性方程与“速度场生成概率路径”的等价关系：
- 若 $p_t$ 是概率路径，$u_t$ 是局部 Lipschitz 可积的向量场，则：
  1. 连续性方程对所有 $t \in [0,1)$ 成立；
  2. $u_t$ 生成 $p_t$（即 $X_t = \psi_t(X_0) \sim p_t$）；
- 两者互为充要条件——这是后续 FM 框架中“通过优化速度场生成目标概率路径”的核心理论依据。

#### 3.6 瞬时变量替换：似然计算的关键  (TODO)
流模型的重要优势是“支持精确似然计算”，章节通过“瞬时变量替换”推导了似然的 ODE 表达式：

##### （1）似然的时间演化方程
对于流模型 $X_t = \psi_t(X_0)$，目标样本 $X_1 = \psi_1(X_0) \sim q$ 的对数似然 $\log p_1(X_1)$ 满足：
$
\frac{d}{dt} \log p_t\left(\psi_t(x)\right) = -\text{div}\left(u_t\right)\left(\psi_t(x)\right)
$

**公式含义**：$x$ 为固定起点（如 $X_0 = x$），$\psi_t(x)$ 为从 $x$ 出发沿速度场 $u_t$ 演化到时刻 $t$ 的位置；$\log p_t(\psi_t(x))$ 即该轨迹在 $t$ 时刻那一点的对数密度。等式表示：沿同一条轨迹，对数密度对时间的导数等于该点速度场散度的负值——散度越大，该点密度随时间减少得越快。

**推导逻辑（三步）**：
1. **推前映射的密度**：由变量替换有 $p_t(\psi_t(x)) = p_0(x) \cdot \left| \det \partial_x \psi_t(x) \right|^{-1}$。
   - **符号**：$p_0$ 为起点分布，$p_t = (\psi_t)_\sharp p_0$ 为推前分布；$x$ 为固定起点，$\psi_t(x)$ 为从 $x$ 出发在 $t$ 时刻的位置；$\partial_x \psi_t(x)$ 为雅可比矩阵。
   - **含义**：左边 $p_t(\psi_t(x))$ 是轨迹上 $t$ 时刻那一点的密度，右边 $p_0(x)$ 是起点密度；等式表示“同一条轨迹上两点的密度通过雅可比行列式的倒数相联系”——概率被 $\psi_t$ 搬运时，密度按局部体积伸缩比例修正。
   - **来源（变量替换）**：设 $y = \psi_t(x)$，推前密度满足 $p_t(y) = p_0(\psi_t^{-1}(y)) \cdot \left| \det \partial_y \psi_t^{-1}(y) \right|$。由 $\partial_y \psi_t^{-1}(y) = (\partial_x \psi_t(x))^{-1}$ 得 $\det \partial_y \psi_t^{-1} = (\det \partial_x \psi_t)^{-1}$，代入并以 $y = \psi_t(x)$ 写回即得上式。
   - **直观**：$\det \partial_x \psi_t(x)$ 表示 $x$ 处无穷小体积被 $\psi_t$ 拉伸的倍数；其绝对值的倒数即密度应乘的因子——体积被拉大则密度变小（概率守恒）。
2. **取对数并对 $t$ 求导**：记 $J_t(x) = \det \partial_x \psi_t(x)$，则 $\log p_t(\psi_t(x)) = \log p_0(x) - \log |J_t(x)|$，对 $t$ 求导得 $\frac{d}{dt}\log p_t(\psi_t(x)) = -\frac{d}{dt}\log |J_t(x)|$；
3. **雅可比与散度**：对 $\dot\psi_t = u_t(\psi_t)$ 有 $\frac{d}{dt}\log |J_t| = \text{div}(u_t)(\psi_t(x))$，代入即得上式；从连续性方程沿轨迹也可推出同一结论。

**核心意义**：沿轨迹从 $t=0$ 到 $t=1$ 积分可得 $\log p_1(\psi_1(x)) - \log p_0(x) = -\int_0^1 \text{div}(u_t)(\psi_t(x))\,dt$。若 $p_0$ 已知（如标准高斯），则 $\log p_1(X_1) = \log p_0(x) - \int_0^1 \text{div}(u_t)(\psi_t(x))\,dt$（其中 $X_1 = \psi_1(x)$，$x = X_0$）。因此无需显式求 $p_1$ 或雅可比行列式，只需沿轨迹对 $\text{div}(u_t)$ 做时间积分即可得到对数似然；若 $u_t$ 由网络给出，散度可用自动微分或估计得到——即把“似然计算”转化为“对速度场散度的积分”。

##### （2）高维场景下的散度估计
当 $d$ 较大时，直接计算 $\text{div}(u_t(x))$（即雅可比矩阵的迹）计算成本极高（复杂度 $O(d^2)$），章节采用 **Hutchinson 迹估计** 实现无偏近似：
$
\text{div}(u_t(x)) = \mathbb{E}_Z \left[ \text{tr}\left(Z^T \partial_x u_t(x) Z\right) \right]
$
其中 $Z \sim \mathcal{N}(0, I)$——通过随机向量 $Z$ 可将散度计算复杂度降至 $O(d)$，且只需一次向量-雅可比乘积（VJP）反向传播即可实现。

##### （3）似然估计的最终形式
将 Hutchinson 估计代入似然演化方程，积分后得到对数似然的无偏估计：
$
\log p_1(\psi_1(x)) = \log p_0(x) - \mathbb{E}_Z \int_0^1 \text{tr}\left(Z^T \partial_x u_t(\psi_t(x)) Z\right) dt
$
- 实际计算时，需通过“反向求解 ODE”实现（从 $t=1$ 到 $t=0$），代码示例见 Code 3。


#### 3.7 流模型的训练：基于似然最大化
传统流模型（如 CNF）的训练目标是“最大化训练数据的对数似然”，即：
$
\mathcal{L}(\theta) = -\mathbb{E}_{Y \sim q} \log p_1^\theta(Y)
$
其中 $p_1^\theta$ 是流模型 $\psi_t^\theta$ 生成的目标分布。

##### 关键局限
训练过程需要“精确求解 ODE 并计算其微分”，导致计算负担极重——这也是后续 FM 框架提出“无模拟训练”的核心动机（无需在训练中求解 ODE）。

### 核心结论与关键洞察
1. **流与速度场的等价性**：流模型的本质是“由速度场定义的 ODE 轨迹”，两者一一对应，这为“通过学习速度场间接控制流的变换”提供了理论基础；
2. **连续性方程的核心作用**：作为“速度场生成概率路径”的充要条件，连续性方程是 FM 框架中“损失函数设计”的底层依据；
3. **似然计算的可行性**：通过 Hutchinson 迹估计，流模型在高维场景下仍能实现无偏似然估计，这是其区别于扩散模型等随机过程的重要优势；
4. **传统流模型的局限**：基于似然的训练依赖 ODE 模拟与微分，计算成本高——这为 FM 框架的“回归式训练”（无需 ODE 模拟）铺垫了必要性。

### 章节核心贡献
1. 严格建立了流模型的数学体系，明确“流→速度场→概率路径”的内在关联，为后续 FM 框架提供理论基石；
2. 解决了高维场景下流模型的似然计算问题，为模型评估提供了可行方案；
3. 揭示了传统流模型的训练局限，凸显了 FM 框架“无模拟训练”的创新价值。


## 4 Flow Matching
Flow Matching（FM，流匹配）是一种可扩展的生成模型训练框架，核心目标是学习一个参数化速度场 $u_{t}^{\theta}$，使其生成的概率路径 $p_{t}$ 能从源分布 $p$（$p_0=p$）平滑插值到目标分布 $q$（$p_1=q$）。该框架通过巧妙的条件化策略和边际化技巧，避免了训练过程中昂贵的ODE模拟，大幅提升了训练效率，同时保持了生成模型的灵活性和性能。

### 4.1 数据基础（Data）
#### 核心定义
- **源分布与目标分布**：设源样本 $X_0 \sim p$（通常是易采样的已知分布，如高斯分布 $N(0,I)$），目标样本 $X_1 \sim q$（通常是未知的数据分布，如图像、音频等）。
- **耦合（Coupling）**：源样本与目标样本的联合分布关系，分为两种核心类型：
  1. 独立耦合：$\pi_{0,1}(X_0,X_1) = p(X_0)q(X_1)$，源和目标样本独立（如从高斯噪声生成图像）；
  2. 依赖耦合：源和目标样本存在依赖关系（如从低分辨率图像生成高分辨率图像、从灰度图生成彩色图）。

#### 关键作用
耦合定义了源与目标的关联方式，直接影响后续概率路径的设计和速度场的学习效果。独立耦合适用于无监督生成任务，依赖耦合则适用于有监督的条件生成任务。

### 4.2 概率路径构建（Building probability paths）
概率路径 $p_t$ 是FM的核心基础，指一系列随时间 $t \in [0,1]$ 变化的分布，需满足边界条件 $p_0=p$ 和 $p_1=q$，即从源分布平滑过渡到目标分布。

#### 构建策略：条件化路径聚合
FM采用“条件路径+边际化”的构建方式，大幅简化路径设计难度：
1. **条件概率路径**：针对每个目标样本 $X_1=x_1$，设计条件概率路径 $p_{t|1}(x|x_1)$，需满足：
   - 初始条件：$p_{0|1}(x|x_1) = \pi_{0|1}(x|x_1)$（$\pi_{0|1}$ 是条件耦合，独立耦合下为 $p(x)$）；
   - 终止条件：$p_{1|1}(x|x_1) = \delta_{x_1}(x)$（$\delta$ 函数表示 $t=1$ 时概率集中于目标样本 $x_1$）。
   - 典型示例：高斯条件路径 $p_{t|1}(x|x_1) = \mathcal{N}(x | tx_1, (1-t)^2I)$，随 $t \to 1$ 逐渐收敛到 $\delta_{x_1}(x)$。

2. **边际概率路径**：通过对所有目标样本的条件路径加权聚合，得到全局概率路径：
   $
   p_t(x) = \int p_{t|1}(x|x_1) q(x_1) dx_1
   $
   加权系数由目标分布 $q(x_1)$ 决定，确保边际路径满足边界条件 $p_0=p$ 和 $p_1=q$。

#### 路径设计的核心要求
- 连续性：$p_t$ 随 $t$ 平滑变化；
- 可计算性：条件路径 $p_{t|1}$ 需易于采样和推导速度场；
- 插值性：严格满足源和目标分布的边界约束。

### 4.3 生成速度场推导（Deriving generating velocity fields）
速度场 $u_t(x)$ 是FM的核心学习对象，其作用是驱动样本沿概率路径 $p_t$ 演化（通过解ODE $\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x))$）。

#### 条件速度场与边际速度场
1. **条件速度场**：对于每个条件路径 $p_{t|1}(x|x_1)$，存在对应的条件速度场 $u_t(x|x_1)$，满足：$u_t(\cdot|x_1)$ 生成 $p_{t|1}(\cdot|x_1)$（即条件路径是条件速度场对应的ODE的解）。
   - 示例：对于高斯条件路径 $p_{t|1}(x|x_1) = \mathcal{N}(x | tx_1, (1-t)^2I)$，条件速度场为 $u_t(x|x_1) = \frac{x_1 - x}{1-t}$。

2. **边际速度场**：通过对条件速度场加权平均，得到生成边际路径 $p_t$ 的边际速度场：
   $
   u_t(x) = \int u_t(x|x_1) p_{1|t}(x_1|x) dx_1
   $
   其中 $p_{1|t}(x_1|x) = \frac{p_{t|1}(x|x_1)q(x_1)}{p_t(x)}$（由贝叶斯法则推导），表示给定当前样本 $x$ 时目标样本 $x_1$ 的后验概率。

{{< admonition tip "具体实现" false >}}

1. **边际速度场**
\[
u_t(x) = \int u_t(x\mid x_1)\; p_{1\mid t}(x_1\mid x)\,dx_1
\]

2. **贝叶斯后验**
\[
p_{1\mid t}(x_1\mid x)
= \frac{p_{t\mid 1}(x\mid x_1)\; q(x_1)}{p_t(x)}
\]

含义：
- \(u_t(x\mid x_1)\)：**条件速度场**（给定目标 \(x_1\)，当前 \(x\) 该往哪走）
- \(p_{t\mid 1}(x\mid x_1)\)：**给定 \(x_1\) 时 \(x\) 的转移密度**（流前向）
- \(q(x_1)\)：**目标分布**（比如真实数据分布）
- \(p_t(x)\)：**当前时间边际分布**
- \(p_{1\mid t}(x_1\mid x)\)：**给定 \(x\)，反推 \(x_1\) 是哪张真实图** → 后验

---

# 核心实现思路（超级关键）
我们要算：
\[
u_t(x) = \mathbb{E}_{x_1 \sim p_{1\mid t}(\cdot\mid x)}\big[\,u_t(x\mid x_1)\,\big]
\]

但有两个麻烦：
1. **不能直接从后验 \(p_{1\mid t}(x_1\mid x)\) 采样**（不知道怎么采样）
2. 但我们**能从真实数据 \(q(x_1)\) 采样**（这就是你的训练集）

所以：
**用从 q 采样的样本，加权，去模拟从 p₁|ₜ 采样的期望。**
这就叫 **重要性采样 / 加权平均**。


积分 **≈ 蒙特卡洛采样平均**
\[
\int (\cdots) p_{1\mid t}(x_1\mid x) dx_1
\approx \frac{1}{K}\sum_{k=1}^K u_t(x\mid x_1^{(k)})
\]
其中
\[
x_1^{(k)} \sim p_{1\mid t}(x_1\mid x)
\]

但 Flow Matching 里**几乎从不直接从后验 \(p_{1\mid t}\) 采样**，而是用：
### **重要采样 / 加权平均**
\[
u_t(x)
\approx \frac{1}{Z}\sum_{x_1 \sim q} u_t(x\mid x_1)\; \underbrace{p_{t\mid 1}(x\mid x_1)}_{\text{权重}}
\]
其中
\[
Z = \sum_{x_1 \sim q} p_{t\mid 1}(x\mid x_1)
\]

---

# 1. 数学推导：把期望换成可计算形式
你要的积分：
\[
u_t(x) = \int u_t(x\mid x_1)\,\color{red}{p_{1\mid t}(x_1\mid x)}\,dx_1
\]

把贝叶斯代入：
\[
\color{red}{p_{1\mid t}(x_1\mid x)}
= \frac{p_{t\mid 1}(x\mid x_1)\,q(x_1)}{p_t(x)}
\]

所以：
\[
u_t(x)
= \int u_t(x\mid x_1)
\cdot \frac{p_{t\mid 1}(x\mid x_1)\,q(x_1)}{p_t(x)}
dx_1
\]

把分母提出来：
\[
u_t(x)
= \frac{1}{p_t(x)}
\int u_t(x\mid x_1)\,p_{t\mid 1}(x\mid x_1)\,\color{red}{q(x_1)}\,dx_1
\]

注意红色部分：
\[
\int (\cdots) \color{red}{q(x_1)} dx_1
= \mathbb{E}_{x_1\sim q}\big[\,\cdots\,\big]
\]

所以：
\[
u_t(x)
= \frac{1}{p_t(x)}\;
\mathbb{E}_{x_1\sim q}\big[\,u_t(x\mid x_1)\,p_{t\mid 1}(x\mid x_1)\,\big]
\]

---

# 2. 分母 \(p_t(x)\) 也能写成期望
\[
p_t(x) = \int p_{t\mid 1}(x\mid x_1)\,q(x_1)\,dx_1
\]

也是对 \(q\) 的期望：
\[
p_t(x) = \mathbb{E}_{x_1\sim q}\big[\,p_{t\mid 1}(x\mid x_1)\,\big]
\]

---

# 3. 合起来：**重要采样公式**
把两个期望塞一起：
\[
u_t(x) =
\frac{\;\mathbb{E}_{x_1\sim q}\big[\,u_t(x\mid x_1)\cdot p_{t\mid 1}(x\mid x_1)\,\big]\;}
{\;\mathbb{E}_{x_1\sim q}\big[\,p_{t\mid 1}(x\mid x_1)\,\big]\;}
\]

这就是**所有 flow matching 论文里真正在代码里用的形式**。

---

# 4. 离散化：变成**加权平均**
期望用**样本平均**近似：
\[
\mathbb{E}[\cdots] \approx \frac{1}{K}\sum_{k=1}^K (\cdots)
\]

代入：
\[
u_t(x)
\approx
\frac{
\sum_{k=1}^K u_t(x\mid x_1^{(k)}) \cdot \underbrace{p_{t\mid 1}(x\mid x_1^{(k)})}_{\text{权重}w_k}
}{
\sum_{k=1}^K w_k
}
\]

其中
\[
x_1^{(k)} \sim q(x_1)
\]

---

# 完整代码

```python
import torch
import torch.nn as nn
```

## 1. 条件速度场网络 \(u_t(x \mid x_1)\)
```python
class ConditionalVelocityField(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim + dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x, x1, t):
        """
        x:  当前样本    [B, D]
        x1: 目标样本    [B, D]
        t:  时间        [B, 1]
        return: u_t(x|x1)  [B, D]
        """
        inp = torch.cat([x, x1, t], dim=-1)
        return self.model(inp)
```

---

## 2. 前向转移密度 \(p_{t\mid 1}(x \mid x_1)\)
Flow Matching 最标准：**线性插值 + 高斯**
\[
x_t = (1-t)x_0 + t x_1 + \sigma_t \varepsilon
\]
所以转移密度是高斯：
\[
p_{t\mid 1}(x\mid x_1)
= \mathcal{N}\big(x;\ \mu_t(x_1),\ \sigma_t^2 I\big)
\]

```python
def batch_mvn_logp(x, mean, std):
    """高斯对数概率"""
    var = std ** 2
    return -0.5 * ((x - mean) ** 2).sum(-1) / var - 0.5 * torch.log(2 * torch.pi * var)

def p_t_given_1(x, x1, t, sigma=0.01):
    """
    p_{t|1}(x | x1)
    x:  [B, D]
    x1: [B, D]
    t:  [B, 1]
    return: 概率密度 [B]
    """
    mu = t * x1 + (1 - t) * x  # 你教程里的流路径
    logp = batch_mvn_logp(x, mu, sigma)
    return logp.exp()
```

---

## 3. 核心：边际速度场 \(u_t(x)\)
严格实现：
\[
u_t(x) = \mathbb{E}_{x_1\sim q}\left[\; u_t(x|x_1) \cdot \frac{p_{t|1}(x|x_1)}{Z}\;\right]
\]
\[
Z = \mathbb{E}_{x_1\sim q}\big[\;p_{t|1}(x|x_1)\;\big]
\]

```python
def marginal_velocity_field(x, t, cond_net, x1_batch, sigma=0.01):
    """
    你要的公式：
    u_t(x) = ∫ u_t(x|x1) * p_{1|t}(x1|x) dx1
    
    输入：
    x:        当前样本        [B, D]
    t:        时间            [B, 1]
    cond_net: 条件速度场
    x1_batch: 来自 q(x1) 的一批真实样本 [N, D]
    """
    B, D = x.shape
    N = x1_batch.shape[0]

    # ---------------------------
    # 1. 扩展成配对形式
    # ---------------------------
    x_tile = x[:, None, :].repeat(1, N, 1)          # [B, N, D]
    t_tile = t[:, None, :].repeat(1, N, 1)          # [B, N, 1]
    x1_tile = x1_batch[None, :, :].repeat(B, 1, 1)  # [B, N, D]

    # ---------------------------
    # 2. 计算所有 u_t(x | x1)
    # ---------------------------
    u_cond = cond_net(
        x_tile.flatten(0, 1),
        x1_tile.flatten(0, 1),
        t_tile.flatten(0, 1)
    ).view(B, N, D)  # [B, N, D]

    # ---------------------------
    # 3. 计算权重 w = p_{t|1}(x | x1)
    # ---------------------------
    w = p_t_given_1(
        x_tile,
        x1_tile,
        t_tile,
        sigma=sigma
    )  # [B, N]

    # ---------------------------
    # 4. 归一化权重 = p_{1|t}(x1|x)
    # ---------------------------
   Z = w.sum(dim=1, keepdim=True) + 1e-8
    w_norm = w / Z  # [B, N]

    # ---------------------------
    # 5. 加权平均 = 积分
    # ---------------------------
    # u_t(x) = sum( u_cond * w_norm )
    u_marginal = (u_cond * w_norm[..., None]).sum(dim=1)  # [B, D]

    return u_marginal
```

---

# 4. 如何使用（训练/推理）
```python
# 超参
dim = 2
sigma = 0.01

# 模型
cond_net = ConditionalVelocityField(dim)

# 批次
B = 32
N = 64  # 用来加权平均的真实样本数量
x = torch.randn(B, dim)    # 当前样本
t = torch.rand(B, 1)       # 时间
x1_batch = torch.randn(N, dim)  # 真实样本 q(x1)

# 计算边际速度场！
u_t = marginal_velocity_field(x, t, cond_net, x1_batch, sigma)

print(u_t.shape)  # torch.Size([32, 2])
```

---

# 代码**逐行对应**
1.
\[
u_t(x) = \int u_t(x\mid x_1)\,p_{1\mid t}(x_1\mid x)\,dx_1
\]
```python
u_marginal = (u_cond * w_norm[..., None]).sum(dim=1)
```

2.
\[
p_{1\mid t}(x_1\mid x) = \frac{p_{t\mid 1}(x\mid x_1)\,q(x_1)}{p_t(x)}
\]
```python
w = p_t_given_1(...)
w_norm = w / w.sum(dim=1, keepdim=True)
```

3. **加权平均 = 对后验期望**
```python
(u_cond * w_norm[..., None]).sum(dim=1)
```
{{< /admonition >}}


#### 关键解读
边际速度场 $u_t(x)$ 可解释为：给定当前样本 $x$（服从 $p_t$），所有可能目标样本对应的条件速度场的加权平均，权重为后验概率 $p_{1|t}(x_1|x)$。这种平均方式确保了 $u_t(x)$ 能准确驱动样本沿边际路径 $p_t$ 演化。

### 4.4 一般条件化与边际化技巧（General conditioning and the Marginalization Trick）
边际化技巧是FM的核心数学基础，其作用是证明：**若条件速度场生成条件概率路径，则通过加权平均得到的边际速度场，必然生成对应的边际概率路径**。

#### 一般化条件化
条件化不仅限于目标样本 $X_1$，可扩展到任意随机变量 $Z$（如源样本 $X_0$、标签 $Y$ 等）：
- 条件概率路径：$p_{t|Z}(x|z)$（生成 $Z=z$ 时的条件路径）；
- 条件速度场：$u_t(x|z)$（生成 $p_{t|Z}(x|z)$）；
- 边际概率路径：$p_t(x) = \int p_{t|Z}(x|z) p_Z(z) dz$；
- 边际速度场：$u_t(x) = \mathbb{E}[u_t(X_t|Z) | X_t=x]$（条件期望形式，更易理解和计算）。

#### 边际化定理（Theorem 3）
##### 前提假设（Assumption 1）
1. $p_{t|Z}(x|z)$ 和 $u_t(x|z)$ 关于 $(t,x)$ 是 $C^1$ 光滑的；
2. $p_Z(z)$ 具有有界支撑（即 $Z$ 仅在有限区域有非零概率）；
3. 对所有 $t \in [0,1)$ 和 $x \in \mathbb{R}^d$，$p_t(x) > 0$。

##### 核心结论
若条件速度场 $u_t(x|z)$ 是条件可积的，且生成条件路径 $p_{t|Z}(x|z)$，则边际速度场 $u_t(x)$ 生成边际路径 $p_t(x)$（对所有 $t \in [0,1)$）。

##### 证明核心逻辑
通过验证边际速度场与边际路径满足连续性方程（Continuity Equation）：
1. 条件速度场与条件路径满足连续性方程（由质量守恒定理）；
2. 对连续性方程两边关于 $Z$ 积分，利用 Leibniz 法则交换积分与微分顺序；
3. 代入边际速度场和边际路径的定义，可证两者满足连续性方程，因此边际速度场生成边际路径。

#### 意义
边际化技巧将复杂的“边际路径生成”问题分解为简单的“条件路径生成”问题，大幅降低了FM的设计难度——只需设计易于处理的条件路径和条件速度场，即可通过边际化得到满足要求的边际路径和速度场。

### 4.5 Flow Matching损失（Flow Matching loss）
FM的训练目标是让参数化速度场 $u_{t}^{\theta}(x)$ 逼近真实边际速度场 $u_t(x)$，核心挑战是真实边际速度场 $u_t(x)$ 难以直接计算（需积分所有目标样本）。FM通过Bregman散度和条件化损失解决这一问题。

#### 核心损失函数
1. **Flow Matching损失（FM损失）**：
   $
   \mathcal{L}_{FM}(\theta) = \mathbb{E}_{t \sim U[0,1], X_t \sim p_t} D(u_t(X_t), u_{t}^{\theta}(X_t))
   $
   其中 $D$ 是Bregman散度（如平方$\ell_2$范数 $D(u,v)=\|u-v\|^2$），表示在概率路径上让参数化速度场逼近真实边际速度场。
   - 问题：真实边际速度场 $u_t(X_t)$ 不可直接计算，因此该损失无法直接优化。

2. **条件Flow Matching损失（CFM损失）**：
   $
   \mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim U[0,1], Z, X_t \sim p_{t|Z}(\cdot|Z)} D(u_t(X_t|Z), u_{t}^{\theta}(X_t))
   $
   直接使用条件速度场 $u_t(X_t|Z)$ 作为监督信号，无需计算边际速度场。

#### 关键定理（Theorem 4）
FM损失与CFM损失的梯度完全等价：
$
\nabla_{\theta} \mathcal{L}_{FM}(\theta) = \nabla_{\theta} \mathcal{L}_{CFM}(\theta)
$
且CFM损失的最小化器恰好是真实边际速度场 $u_t(x)$。

##### 证明核心逻辑
利用Bregman散度的仿射不变性（$\nabla_v D(\mathbb{E}[Y],v) = \mathbb{E}[\nabla_v D(Y,v)]$），将FM损失的梯度转化为条件速度场的期望梯度，最终与CFM损失的梯度等价。

#### 意义
CFM损失是FM可落地的关键：通过条件速度场提供监督信号，避免了边际速度场的直接计算，同时保证了优化目标与FM损失一致。在实际训练中，只需采样 $t$、$Z$（如目标样本 $X_1$）和 $X_t$（从条件路径采样），即可计算损失并更新参数。

### 4.6 基于条件流的条件生成（Solving conditional generation with conditional flows）
本节提供了条件路径和条件速度场的具体设计方法——通过条件流（Conditional Flows）构建，进一步简化FM的工程实现。

#### 条件流定义
条件流 $\psi_t(x|x_1)$ 是满足以下条件的时间依赖映射：
$
\psi_t(x|x_1) = \begin{cases} x & t=0 \\ x_1 & t=1 \end{cases}
$
且 $\psi_t(x|x_1)$ 关于 $(t,x)$ 光滑，关于 $x$ 是微分同胚（diffeomorphism，可逆且导数连续）。

#### 条件路径与条件速度场的推导
1. **条件路径**：由条件流的推前映射（push-forward map）定义：
   $
   p_{t|1}(x|x_1) = [\psi_t(\cdot|x_1)_{\sharp} \pi_{0|1}(\cdot|x_1)](x)
   $
   即通过条件流将初始分布 $\pi_{0|1}(\cdot|x_1)$ 推前得到条件路径，天然满足边界条件 $p_{0|1}(x|x_1)=\pi_{0|1}(x|x_1)$ 和 $p_{1|1}(x|x_1)=\delta_{x_1}(x)$。

2. **条件速度场**：由条件流的时间导数提取（流与速度场的等价性）：
   $
   u_t(x|x_1) = \dot{\psi}_t(\psi_t^{-1}(x|x_1)|x_1)
   $
   其中 $\dot{\psi}_t = \frac{d}{dt}\psi_t$，表示条件流在逆映射点的瞬时速度。

#### 核心优势
- 灵活性：条件流的设计可灵活选择（如线性流、仿射流、 geodesic 流等），适配不同数据类型；
- 可计算性：条件速度场可通过条件流直接推导，无需手动设计；
- 理论保证：条件流的微分同胚性质确保了条件路径的光滑性和可逆性。

#### 训练流程（对应Code 4）
1. 从数据加载器获取源-目标样本对 $(x_0,x_1)$（服从耦合 $\pi_{0,1}$）；
2. 采样时间 $t \sim U[0,1]$；
3. 通过条件流采样得到 $x_t = \psi_t(x_0|x_1)$ 和条件速度场 $dx_t = \dot{\psi}_t(x_0|x_1)$；
4. 计算CFM损失：${cfm}_{loss} = \mathbb{E}[\|u_{t}^{\theta}(x_t) - dx_t\|^2]$ ；
5. 反向传播更新模型参数。

### 4.7 最优传输与线性条件流（Optimal Transport and linear conditional flow）
线性条件流是条件流的一种简单且高效的实例，由最优传输（Optimal Transport, OT）理论推导而来，具有动能最优性。

#### 最优传输背景
动态OT问题的目标是找到动能最小的概率路径和速度场：
$
\min_{p_t,u_t} \int_0^1 \int \|u_t(x)\|^2 p_t(x) dx dt \quad s.t. \ p_0=p, p_1=q, \text{连续性方程}
$
其解为OT位移插值器 $\psi_t^*(x) = t\phi(x) + (1-t)x$（$\phi$ 是OT映射），对应的速度场为常数 $\phi(x) - x$。

#### 线性条件流的推导
为最小化条件速度场的动能上界，通过变分法求解得到线性条件流：
$
\psi_t(x|x_1) = tx_1 + (1-t)x
$
即样本沿源样本 $x$ 到目标样本 $x_1$ 的直线演化。

#### 关键性质
1. 动能最优：线性条件流最小化所有条件流的动能上界；
2. OT等价性：当目标分布 $q = \delta_{x_1}$（单一样本）时，线性条件流就是OT的解析解；
3. 简单易算：对应的条件速度场为 $u_t(x|x_1) = x_1 - x$（代入4.6的条件速度场公式推导），大幅简化训练。

### 4.8 仿射条件流（Affine conditional flows）
仿射条件流是线性条件流的推广，通过引入调度器（scheduler）$(\alpha_t, \sigma_t)$ 增加灵活性，同时保持解析可计算性，是FM中应用最广泛的条件流类型。

#### 定义与调度器约束
仿射条件流的形式为：
$
\psi_t(x|x_1) = \alpha_t x_1 + \sigma_t x
$
其中调度器 $\alpha_t, \sigma_t: [0,1] \to [0,1]$ 需满足：
- 边界条件：$\alpha_0=0, \sigma_0=1$（$t=0$ 时为源样本 $x$）；$\alpha_1=1, \sigma_1=0$（$t=1$ 时为目标样本 $x_1$）；
- 单调性：$\dot{\alpha}_t > 0, -\dot{\sigma}_t > 0$（$\alpha_t$ 递增，$\sigma_t$ 递减）。

#### 核心推导
1. **条件速度场**：由仿射流的时间导数得到：
   $
   u_t(x|x_1) = \dot{\alpha}_t x_1 + \dot{\sigma}_t x
   $
2. **边际速度场**：代入边际速度场公式，得到：
   $
   u_t(x) = \mathbb{E}[\dot{\alpha}_t X_1 + \dot{\sigma}_t X_0 | X_t = x]
   $
   其中 $X_t = \alpha_t X_1 + \sigma_t X_0$（仿射流定义的样本演化）。

#### 关键扩展
##### 4.8.1 速度场参数化
边际速度场可通过多种方式参数化，便于模型学习：
1. 直接参数化：直接学习 $u_t(x)$；
2. 目标预测（$x_{1|t}$）：$u_t(x) = \frac{\dot{\sigma}_t}{\sigma_t}x + \left(\dot{\alpha}_t - \alpha_t \frac{\dot{\sigma}_t}{\sigma_t}\right) \mathbb{E}[X_1 | X_t=x]$；
3. 源预测（$x_{0|t}$）：$u_t(x) = \frac{\dot{\alpha}_t}{\alpha_t}x + \left(\dot{\sigma}_t - \sigma_t \frac{\dot{\alpha}_t}{\alpha_t}\right) \mathbb{E}[X_0 | X_t=x]$。
三种参数化可通过贝叶斯法则和条件期望相互转换（见表1）。

##### 4.8.2 训练后调度器切换
仿射流支持训练后更换调度器（如从方差保持调度器切换到OT调度器），无需重新训练：
通过尺度-时间变换（ST变换）$\psi_r(x_0|x_1) = s_r \psi_{t_r}(x_0|x_1)$，可将基于原调度器 $(\alpha_t, \sigma_t)$ 的速度场 $u_t(x)$ 转换为新调度器 $(\bar{\alpha}_r, \bar{\sigma}_r)$ 的速度场 $\bar{u}_r(x)$，且保证最终生成样本一致（$\bar{\psi}_1(x_0) = \psi_1(x_0)$）。

##### 4.8.3 高斯路径
当源分布为高斯分布 $p = N(0,I)$ 时，仿射条件流对应的条件路径为高斯分布：
$
p_{t|1}(x|x_1) = \mathcal{N}(x | \alpha_t x_1, \sigma_t^2 I)
$
该路径与扩散模型的前向过程等价，且速度场可通过分数函数（score function）参数化：
$
u_t(x) = \frac{\dot{\alpha}_t}{\alpha_t}x - \frac{\dot{\sigma}_t \sigma_t \alpha_t - \dot{\alpha}_t \sigma_t^2}{\alpha_t} \nabla \log p_t(x)
$
其中 $\nabla \log p_t(x)$ 是边际路径的分数函数，与源预测 $x_{0|t}$ 成正比（$\nabla \log p_t(x) = -\frac{1}{\sigma_t}x_{0|t}(x)$）。

### 4.9 数据耦合（Data couplings）
数据耦合定义了源与目标样本的关联方式，直接影响FM的训练效果和适用场景，本节介绍两种实用耦合方式：

#### 4.9.1 配对数据（Paired data）
适用于有监督条件生成任务（如图像修复、超分辨率、去模糊）：
- 耦合定义：$\pi_{0,1}(x_0,x_1) = \pi_{0|1}(x_0|x_1) q(x_1)$，其中 $\pi_{0|1}(x_0|x_1)$ 是从目标样本 $x_1$ 生成源样本 $x_0$ 的变换（如对图像添加掩码、降分辨率、添加模糊）；
- 优势：无需源分布是高斯噪声，可直接利用任务相关的源-目标配对关系，生成效果更贴合任务需求；
- 实现技巧：采样时对 $\pi_{0|1}(x_0|x_1)$ 添加少量噪声，保证源分布的光滑性和多样性。

#### 4.9.2 多样本耦合（Multisample couplings）
适用于无监督生成任务，通过构建低动能的源-目标关联，提升生成样本质量：
- 核心思想：每次采样 $k$ 个源样本和 $k$ 个目标样本，构建最优匹配矩阵 $\pi^k$（最小化运输成本 $c(X_0^{(i)}, X_1^{(j)})$），再从匹配对中采样训练数据；
- 优势：降低源-目标配对的运输成本，使学习到的速度场诱导更直的样本轨迹，减少ODE采样误差；
- 极限性质：当 $k \to \infty$ 时，多样本耦合逼近OT耦合，动能趋近于最优。

### 4.10 条件生成与引导（Conditional generation and guidance）
条件生成的目标是生成满足特定引导信号（如标签、文本描述）的样本，FM支持多种引导策略，适配不同任务需求。

#### 4.10.1 条件模型（Conditional models）
直接学习条件速度场 $u_t(x|y)$，生成条件分布 $q(x_1|y)$：
- 概率路径：$p_{t|Y}(x|y) = \int p_{t|1}(x|x_1) q(x_1|y) dx_1$（基于标签 $y$ 的条件边际路径）；
- 条件速度场：$u_t(x|y) = \mathbb{E}[u_t(x|x_1) | X_t=x, Y=y]$；
- 训练损失：条件CFM损失 $\mathcal{L}_{CFM}(\theta) = \mathbb{E}[D(u_t(X_t|X_1), u_{t}^{\theta}(X_t|Y))]$；
- 适用场景：标签空间小且重复（如分类标签），引导信号与目标样本强相关。

#### 4.10.2 分类器引导与无分类器引导
针对高斯路径设计的引导策略，通过分数函数与分类器结合，提升条件生成的灵活性和效果：

##### 分类器引导（Classifier guidance）
1. 核心原理：利用条件分数函数与无条件分数函数的关系：
   $
   \nabla \log p_{t|Y}(x|y) = \nabla \log p_{Y|t}(y|x) + \nabla \log p_t(x)
   $
   其中 $\nabla \log p_{Y|t}(y|x)$ 是时间依赖分类器的分数函数（预测标签 $y$ 给定 $x$ 和 $t$）。
2. 速度场调整：
   $
   \tilde{u}_t^{\theta,\phi}(x|y) = u_t^{\theta}(x) + b_t w \nabla \log p_{Y|t}^{\phi}(y|x)
   $
   其中 $u_t^{\theta}(x)$ 是无条件速度场，$w$ 是引导强度，$b_t$ 是调度系数。
3. 优势：分类器与生成器独立训练，可灵活调整引导强度；
4. 劣势：需单独训练分类器，存在训练不一致问题。

##### 无分类器引导（Classifier-free guidance, CFG）
1. 核心原理：将分类器分数函数表示为条件分数与无条件分数的差：
   $
   \nabla \log p_{Y|t}(y|x) = \nabla \log p_{t|Y}(x|y) - \nabla \log p_t(x)
   $
   无需单独训练分类器，直接学习统一的条件速度场 $u_t^{\theta}(x|y)$（支持 $y=\emptyset$ 即无条件生成）。
2. 速度场调整：
   $
   \tilde{u}_t^{\theta}(x|y) = (1-w) u_t^{\theta}(x|\emptyset) + w u_t^{\theta}(x|y)
   $
   其中 $w$ 是引导强度，平衡无条件生成的多样性和条件生成的准确性。
3. 训练损失：
   $
   \mathcal{L}_{CFM}(\theta) = \mathbb{E}[D(u_t(X_t|X_1), u_{t}^{\theta}(X_t|(1-\xi)Y + \xi \emptyset))]
   $
   其中 $\xi \sim \text{Bernoulli}(p_{\text{uncond}})$（随机丢弃引导信号，提升模型鲁棒性）。
4. 优势：无需单独训练分类器，生成效果更稳定，是当前大规模条件生成（如文本到图像）的主流策略。

### 核心总结
Flow Matching框架的核心优势在于：
1. **模拟无关的训练**：通过CFM损失和边际化技巧，避免训练过程中解ODE，大幅提升训练效率；
2. **灵活的路径设计**：支持线性流、仿射流、高斯流等多种条件流，适配不同数据类型；
3. **丰富的扩展能力**：可扩展到条件生成、多样本耦合、非欧几里得空间等场景，且与扩散模型等生成模型框架兼容；
4. **高效的生成过程**：训练后通过解ODE生成样本，可通过调整ODE solver（如欧拉法、中点法）平衡速度和精度。

FM的本质是将生成模型的训练转化为“路径设计+速度场回归”，通过数学技巧简化了复杂的概率分布变换问题，成为当前生成模型领域的重要框架之一，尤其在高分辨率图像生成、音频生成、蛋白质结构生成等领域表现突出。

如果你需要进一步了解某一具体模块（如仿射流的调度器设计、无分类器引导的工程实现、非欧几里得空间的FM扩展），可以随时告诉我，我会提供更细致的解读和示例代码说明。





### FM 与其他生成模型的对比

| 对比维度 | Flow Matching（FM） | 扩散模型（Diffusion） | 传统归一化流（NF） |
|----------|---------------------|------------------------|--------------------|
| **核心过程** | 正向确定性流（无反向、无噪声） | 加噪 → 去噪（反向随机过程） | 密度变换（依赖可逆映射） |
| **学习目标** | 光滑向量场（低方差、物理意义明确） | 多尺度噪声（高方差、难拟合） | 对数密度 + 雅可比行列式 |
| **数学基础** | 常微分方程（ODE） | 随机微分方程（SDE） | 变量替换公式 + 雅可比计算 |
| **分布路径** | 显式可控（如线性插值） | 隐式（由噪声调度决定） | 隐式（由映射结构决定） |
| **训练稳定性** | 高（MSE 损失、无病态逆过程） | 低（对超参/步长敏感、易模式崩溃） | 中（雅可比计算复杂、架构受限） |


### FM 关键结论与核心优势

1. **数学统一性**：FM 框架可推广到离散空间（Discrete FM）、黎曼流形（Riemannian FM）等任意状态空间，核心逻辑（拟合向量场）保持一致。
2. **训练高效性**：无需计算雅可比行列式、对数密度，损失为简单 MSE，网络架构无约束（MLP / Transformer / U-Net 均可）。
3. **稳定性根源**：基于 ODE 的确定性流 + 显式分布路径，避免了 SDE 逆过程的病态性和 NF 的雅可比计算瓶颈。
4. **核心逻辑链**：微分同胚（空间变形）→ 推前映射（分布搬运）→ 变量替换公式（密度修正）→ 向量场拟合（FM 核心）→ 稳定生成（无反向/无噪声）。


## 5 Non-Euclidean Flow Matching
**超完整、通俗、逐段、逐公式、不跳步详细解释**
（对应论文 2412.06264 第5节：非欧几里得空间上的流匹配）

---

### 5.1 这一章到底在干嘛？（一句话核心）
前面所有 Flow Matching 都假设：
**数据在平直空间（欧几里得空间）里**，比如 $\mathbb{R}^d$，距离就是普通 L2。

但很多数据**不在平直空间**：
- 球面（方向、角度、旋转、归一化 embedding）
- 流形（低维曲面）
- 非欧空间（距离不是直线）

**第5章就是：
把 Flow Matching 从「平直空间」推广到「任意弯曲空间 / 非欧空间」。**

---

### 5.2 为什么需要 Non-Euclidean FM？
很多数据天然**被约束在某个曲面 M 上**：
- 单位球面 $S^d$（归一化特征、方向向量）
- 旋转群 SO(3)
- 对称正定矩阵空间
- 概率单纯形（分布向量）

如果你强行把它们拉到 $\mathbb{R}^d $ 里做 FM：
- 路径会跑出流形
- 概率没有意义
- 生成结果不合法

所以必须：
**让整个流、速度场、概率路径，全都待在流形 M 上。**

---

### 5.3 预备知识：流形上的基础（极简但够用）
设数据在**黎曼流形 M** 上：
- $x \in M$：流形上的点
- $T_x M$：x 点的**切空间**（像曲面在 x 点的切平面）
- $\exp_x(v)$：**指数映射**
  从 x 出发，沿切空间方向 v 走测地线，到达流形上的点
- $\log_x(y)$：**对数映射**
  y 在流形上到 x 的“最短路径方向”

欧氏空间里：
$
\exp_x(v) = x + v,\quad \log_x(y) = y - x
$
非欧空间里：**不是简单加减**。

---

### 5.4 流形上的概率路径
#### 目标
构造一条**全在 M 上**的概率路径：
$
p_t \quad t\in[0,1],\quad p_0=p,\ p_1=q
$

#### 延续 4.4 的思想：
**先做条件路径，再边缘化**

对每个目标 $x_1$，定义**条件概率路径**：
$
p_{t|1}(\cdot | x_1)
$
满足：
- $t=0$：源分布
- $t=1$：落在 $x_1$

然后**边缘化**得到全局路径：
$
p_t(x) = \mathbb{E}_{x_1\sim q}\big[p_{t|1}(x|x_1)\big]
$

**Marginalization Trick 仍然成立！**
这是整个第5章的支柱。

---

### 5.5 流形上的流与速度场
#### 欧氏空间的流
$
\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x))
$

#### 流形上的流
流形上不能随便加减，必须用**切向量**：
$
\frac{D}{dt} \psi_t(x) = u_t(\psi_t(x))
$

- $\psi_t(x) \in M$：流形上的轨迹
- $u_t(x) \in T_x M$：**切空间里的速度**
- $\frac{D}{dt}$：协变导数（保证不飞出流形）

---

### 5.6 流形上的连续性方程（核心）
欧氏空间：
$
\partial_t p_t + \nabla\cdot(u_t p_t) = 0
$

流形上变成**黎曼连续性方程**：
$
\partial_t p_t + \mathrm{div}_M(u_t p_t) = 0
$

- $\mathrm{div}_M$：**流形上的散度**
- 含义仍然是：**流形上的质量守恒**

---

### 5.7 流形上的 Flow Matching 损失
#### 关键：
**Marginalization Trick 在流形上仍然 100% 成立！**

所以：
- 只要条件速度场 $u_t(x|x_1)$ 合法
- 边缘化后的 $u_t(x)$ 也合法
- 损失函数**结构完全不变**

#### 流形上的 CFM 损失
$
\mathcal{L}_{CFM}(\theta)=
\mathbb{E}_{t,x_1,x_t}
\big\| u_\theta(x_t) - u_t(x_t | x_1) \big\|^2_{x_t}
$

唯一变化：
**范数是流形上的切空间内积**
$
\|v\|^2_x = \langle v,v\rangle_x
$

---

### 5.8 流形上的仿射流（最重要一节）
欧氏仿射流：
$
x_t = \alpha_t x_1 + \sigma_t x_0
$

流形上不能直接加减，
论文给出**流形版仿射流**：

#### 流形上的条件流（核心公式）
$
\psi_t(x_0|x_1)= \exp_{x_1}\!\big(- \sigma_t \exp_{x_1}^{-1}(x_0)
\big)
$

拆成大白话：
1. 从 $x_1$ 看 $x_0$：$v = \log_{x_1}(x_0)$
2. 缩放：$-\sigma_t v$
3. 从 $x_1$ 沿测地线走回去：$\exp_{x_1}(-\sigma_t v)$

满足：
- $t=0$：回到 $x_0$
- $t=1$：落在 $x_1$

#### 对应的条件速度场
$
u_t(x|x_1)=-\dot{\sigma}_t \cdot \log_x(x_1)
$

**太美了：
速度 = 缩放系数 × 流形上“指向 x₁ 的方向”**

---

### 5.9 球面 Flow Matching（最重要例子）
最常用的非欧空间：**单位球面 $S^{d-1}$**
$
\|x\| = 1
$

#### 球面指数映射
$
\exp_x(v)
= \cos(\|v\|)x + \sin(\|v\|)\frac{v}{\|v\|}
$

#### 球面对数映射
$
\log_x(y)
= \arccos(x^\top y) \cdot \frac{y - x^\top y x}{\|y - x^\top y x\|}
$

#### 球面条件速度场
$
u_t(x|x_1)=
-\dot{\sigma}_t \cdot \log_x(x_1)
$

#### 球面 Flow Matching 完整流程
1. $x_0 \sim p$（球面上分布）
2. $x_1 \sim q$（球面上数据）
3. $x_t = \psi_t(x_0|x_1)$（球面上插值）
4. 学习 $u_\theta(x_t)$ 逼近 $u_t(x_t|x_1)$
5. 推理时解**球面 ODE**

全程**不会离开球面**！

---

### 5.10 为什么 Non-Euclidean FM 强大？
- 可以建模**结构化数据**：旋转、方向、特征、概率向量
- 生成结果**天然满足约束**
- 继承 Flow Matching 全部优点：
  - 稳定
  - 不用求解SDE
  - 支持多样本耦合
  - 支持调度器切换
  - 支持条件生成

---

### 5.11 整章终极一句话总结
> **第5章把 Flow Matching 从平直空间推广到任意黎曼流形，
>  核心思想完全不变：条件流 + 边缘化技巧；
>  只把加减换成指数/对数映射，
>  就能生成合法、约束满足、高质量的非欧数据。**
