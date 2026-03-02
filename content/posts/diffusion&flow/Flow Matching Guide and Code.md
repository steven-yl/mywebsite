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

summary: "《Flow Matching Guide and Code》全文技术解读：从流模型数学基础与欧氏空间 FM（概率路径、速度场、条件流匹配、线性/仿射条件流），到黎曼流形、离散 FM 与 Generator Matching 统一框架，并阐明与扩散模型、去噪分数匹配的关系。"
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

#### 3.6 瞬时变量替换：似然计算的关键
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


## 6 Continuous Time Markov Chain Models
本节将连续时间马尔可夫链（CTMC）作为流模型的替代生成模型，适用于生成离散数据（即位于离散（且有限）状态空间中的数据）。CTMC是马尔可夫过程，是后面第7节讨论的离散流匹配（DFM）（Campbell等人，2024；Gat等人，2024）生成模型范式的基础。因此，本节与第3节类似，在第3节中我们将流作为流匹配（FM）生成模型范式的基础进行介绍。

### 6.1 离散状态空间和随机变量
考虑有限版本的\(\mathbb{R}^{d}\)作为我们的状态空间\(S=T^{d}\)，其中\(T=[K]=\{1,2,...,K\}\)，有时也称为词汇表。样本和状态用\(x=(x^{1},...,x^{d}) \in S\)表示，其中\(x^{i} \in T\)是单个坐标或一个标记。我们同样使用状态\(y,z \in S\)。接下来，\(X\)表示取值于状态空间\(S\)的随机变量，其概率由概率质量函数（PMF）\(p_{X}: S \to \mathbb{R}_{≥0}\)控制，满足\(\sum_{x \in S} p_{X}(x)=1\)，事件\(A \subset S\)的概率为：
\[
\mathbb{P}(X \in A)=\sum_{x \in A} p_{X}(x) . (6.1)
\]

记号\(X \sim p_{X}\)或\(X \sim p_{X}(X)\)表示\(X\)服从概率质量函数\(p_{X}\)。离散情况下的δ概率质量函数定义为：
\[
\delta (x,z)=
\begin{cases}
1 & x=z, \\
0 & else.
\end{cases}
\]
有时我们也会在标记上定义δ概率质量函数，例如对于某些\(x^{i},y^{i} \in T\)，有\(\delta(x^{i}, y^{i})\)。

### 6.2 CTMC生成模型
CTMC模型是一族取值于\(S\)的时变随机变量\((X_{t})_{0 ≤t ≤1}\)，构成马尔可夫链，其特征由概率转移核\(p_{t+h | t}\)定义：
\[
p_{t+h|t}(y|x):=\mathbb{P}(X_{t+h}=y|X_{t}=x)=\delta (y,x)+hu_{t}(y,x)+o(h), 
\]
且\(\mathbb{P}(X_{0}=x)=p(x)\)，(6.3)
其中\(p\)表示过程在时间\(t=0\)时的初始分布，\(o(h)\)是满足当\(t \to 0\)时\(o(h)/h \to 0\)的任意函数。

值\(u_{t}(y, x)\)称为速率或速度，表示概率随时间在状态之间转移的速度。所谓“完全特征化”，是指对于任意\(0 ≤t_{1}<\cdots<t_{n} ≤1\)和\(x_{i} \in S\)（\(i \in[n]\)），所有联合概率\(\mathbb{P}(X_{t_{1}}=x_{1},...,X_{t_{n}}=x_{n})\)都通过这种方式定义。

为确保转移概率\(p_{t+h | t}(y | x)\)通过(6.3)定义，速度需要满足以下速率条件：
\[
u_{t}(y, x) \geq 0 \text{ 对所有 } y \neq x, \text{ 且 } \sum_{y} u_{t}(y, x)=0. (6.4)
\]

如果其中一个条件不满足，那么对于任意小的\(h>0\)，转移概率\(p_{t+h | t}(\cdot | x)\)会变为负数或求和结果不等于1。式(6.3)与我们定义流生成模型时的式(3.16)和式(3.19)起着相同的作用。过程\(X_{t}\)的边际概率在时间\(t \in[0,1]\)时用概率质量函数\(p_{t}(x)\)表示。然后，与流的情况类似（式(3.24)），如果存在满足(6.3)且边际为\(p_{t}\)的\(p_{t+h | t}\)，则称\(u_{t}\)生成\(p_{t}\)。(6.5)

<!-- ![CTMC模型通过指定状态间概率的速率（速度）来定义。](https://example.com/figure13) -->
图13 CTMC模型通过指定状态间概率的速率（速度）来定义。

#### 模拟CTMC
要采样\(X_{t}\)，需从\(p\)中采样\(X_{0}\)，并使用（朴素的）欧拉方法逐步采样：
\[
\mathbb{P}(X_{t+h}=y | X_{t})=\delta (y,X_{t})+hu_{t}(y,X_{t}). (6.6)
\]

根据(6.3)，这些步骤会给更新概率带来\(o(h)\)的误差。在实际应用中，这意味着我们需要选择足够小的\(h>0\)，以确保式(6.6)右侧保持为有效的概率质量函数。为确保任意选择的\(h>0\)都能得到有效的概率质量函数，并保持概率中的\(o(h)\)局部误差，可采用以下欧拉方法：
\[
\mathbb{P}(X_{t+h}=y| X_{t})=
\begin{cases}
exp \left[ hu_{t}(X_{t},X_{t})\right] & y=X_{t} \\
\frac{u_{t}(y,X_{t})}{|u_{t}(X_{t},X_{t})|}\left( 1-exp \left[ hu_{t}(X_{t},X_{t})\right] \right) & y\neq X_{t}
\end{cases}.
\]

### 6.3 概率路径和科尔莫戈罗夫方程
与连续情况下的连续性方程类似，CTMC模型\((X_{t})_{0 ≤t ≤1}\)的边际概率\(p_{t}\)由科尔莫戈罗夫方程表征：
\[
\frac{d}{d t} p_{t}(y)=\sum_{x} u_{t}(y, x) p_{t}(x) .
\]

以下经典定理（另见Coddington等人（1956）的定理5.1和5.2）描述了这个线性齐次常微分方程组解的存在唯一性。

**定理12（线性常微分方程解的存在唯一性）**：如果\(u_{t}(y, x)\)在\(C([0,1))\)中（关于时间连续），则对于\(t \in[0,1)\)，存在唯一解\(p_{t}(x)\)满足科尔莫戈罗夫方程(6.8)，且满足初始条件\(p_{0}(x)=p(x)\)。

对于CTMC，解保证在所有时间\(t \in[0,1)\)都存在，不需要额外条件（与定理1中的非线性情况不同）。科尔莫戈罗夫方程与连续性方程(3.25)有着密切的联系。利用速率条件重新排列式(6.8)的右侧：
\[
\begin{aligned}
\sum_{x} u_{t}(y, x) p_{t}(x) & \stackrel{(6.4)}{=} \overbrace{\sum_{x \neq y} u_{t}(y, x) p_{t}(x)}^{入射通量} - \overbrace{\sum_{x \neq y} u_{t}(x, y) p_{t}(y)}^{出射通量} \\
& =-\sum_{x \neq y}\left[j_{t}(x, y)-j_{t}(y, x)\right]
\end{aligned}
\]
其中\(j_{t}(y, x):=u_{t}(y, x) p_{t}(x)\)是概率通量，表示单位时间内从状态\(x\)转移到状态\(y\)的概率。出射通量的超额部分定义为散度，使得科尔莫戈罗夫方程与第3.5节中描述的连续性方程具有相同的结构（Gat等人，2024）。

以下结果是在CTMC框架中构建概率路径和速度的主要工具：

**定理13（离散质量守恒）**：设\(u_{t}(y, x)\)在\(C([0,1))\)中，\(p_{t}(x)\)是时间\(t\)上\(C^{1}([0,1))\)中的概率质量函数。则以下表述等价：
1. \(p_{t}\)、\(u_{t}\)满足\(t \in[0,1)\)时的科尔莫戈罗夫方程(6.8)，且\(u_{t}\)满足速率条件(6.4)。
2. \(u_{t}\)在\(t \in[0,1)\)时按6.5的意义生成\(p_{t}\)。

定理13的证明见附录A.1。

#### 6.3.1 保概率速度
作为离散质量守恒（定理13）的推论，如果速度\(u_{t}(y, x)\)生成概率路径\(p_{t}(x)\)，则：
\[
\tilde{u}_{t}(y, x)=u_{t}(y, x)+v_{t}(y, x) \text{ 生成 } p_{t}(x), (6.9)
\]
只要\(v_{t}(y, x)\)满足速率条件(6.4)并求解无散度速度方程：
\[
\sum_{x} v_{t}(y, x) p_{t}(x)=0 . (6.10)
\]

事实上，\(\tilde{u}_{t}(y, x)\)求解科尔莫戈罗夫方程：
\[
\sum_{x} \tilde{u}_{t}(y, x) p_{t}(x)=\sum_{x} u_{t}(y, x) p_{t}(x)=\dot{p}_{t}(y),
\]
这表明在采样过程中可以添加无散度速度，而不会改变边际概率。这在从离散流匹配模型采样时将是一个有用的性质，后续会详细介绍。


## 7 Discrete Flow Matching
值得注意的是，图2中的流匹配框架可无缝从连续情形扩展到离散情形，进而形成离散流匹配（DFM）框架（Campbell et al., 2024; Gat et al., 2024）。与连续情形类似，其核心步骤为：首先定义一条插值于源概率质量函数（PMF）p和目标概率质量函数q的概率路径\(p_t\)；其次，寻找一个由可学习速度\(u_t^\theta\)定义的连续时间马尔可夫链（CTMC）模型，使其生成该概率路径\(p_t\)；最后，通过最小化布雷格曼散度（Bregman divergence）定义的离散流匹配损失来训练\(u_t^\theta\)。综上，这一过程本质上是求解流匹配问题（4.1）的离散版本。

### 7.1 数据与耦合
我们的目标是将来自源概率质量函数p的样本\(X_0 \sim p\)转换为来自目标概率质量函数q的样本\(X_1 \in q\)，其中\(X_0\)和\(X_1\)是两个取值于状态空间S的随机变量。源样本和目标样本可通过独立耦合\((X_0, X_1) \sim p(X_0) q(X_1)\)相关联，也可通过一般的概率质量函数耦合\(\pi_{0,1}(x_0, x_1)\)建立关联。例如，文本翻译数据中的耦合数据\((x_0, x_1)\)表示同一文档的两种不同语言版本；而在文本生成等应用中，常采用独立配对方式，此时\(p(x_0)\)既可以是状态空间S上的均匀概率（所有状态具有相同概率），也可以通过在词汇表T中添加特殊标记m（即\(T \cup \{m\}\)），并令\(\pi_{0,1}(x_0, x_1) = \delta(x_0, m) q(x_1)\)实现。任何满足\(X_0 \sim \delta(X_0, m)\)的随机变量\(X_0\)均为常数随机变量\(X_0 = (m, ..., m)\)。

### 7.2 离散概率路径
流匹配流程的下一步的是定义一条插值于p和q的概率路径\(p_t\)。沿用4.4节的思路，我们将这些对象基于一个取值于任意空间Z的一般条件随机变量\(Z \sim p_Z\)进行条件化处理。边际概率路径的形式为：
\[p_t(x) = \sum_{z \in \mathcal{Z}} p_{t|Z}(x|z) p_Z(z),\]
其中\(p_{t|Z}(\cdot|z)\)是条件概率质量函数，且边际概率路径需满足边界约束\(p_0 = p\)和\(p_1 = q\)。

### 7.3 边际化技巧（Marginalization Trick）
边际化技巧（见4.4节）可直接推广到离散情形（Campbell et al., 2024; Gat et al., 2024）。假设条件速度场\(u_t(\cdot, \cdot|z)\)以（6.5）式的意义生成\(p_{t|Z}(x|z)\)，则边际速度场为：
\[u_t(y,x) = \sum_z u_t(y,x|z) p_{Z|t}(z|x) = \mathbb{E}\left[ u_t(y,X_t|Z) | X_t = x \right],\]
其定义适用于所有满足\(p_t(x) > 0\)的\(x, y \in S\)，且随机变量\(X_t \sim p_{t|Z}(\cdot|Z)\)。根据贝叶斯法则，有：
\[p_{Z|t}(z|x) = \frac{p_{t|Z}(x|z) p_Z(z)}{p_t(x)}.\]

为证明离散版本的边际化技巧定理（定理3），我们做出如下假设：

**假设3**：对所有\(x \in S\)和\(t \in [0,1)\)，有\(p_{t|Z}(x|z) \in C^1([0,1))\)、\(u_t(y, x|z) \in C([0,1))\)且\(p_t(x) > 0\)。

与连续情形类似，假设\(p_t > 0\)在实际应用中较为温和，因为我们总能通过\((1 - (1 - t)\epsilon) \cdot p_{Z|t} + (1 - t)\epsilon \cdot p_{uni}\)（其中\(p_{uni}\)是状态空间S上的均匀分布，\(\epsilon > 0\)为任意小的常数）来满足该假设。接下来，我们给出并证明这一结果。

**定理14（离散边际化技巧）**：在假设3下，若\(u_t(y, x|z)\)生成\(p_{t|Z}(x|z)\)，则（7.2）式中的边际速度\(u_t(y, x)\)在\(t \in [0,1)\)内生成（7.1）式中的\(p_t(x)\)。

**证明**：该证明的思路与连续情形类似。首先计算：
\[
\begin{aligned}
\frac{d}{dt} p_t(y) 
&= \sum_z \frac{d}{dt} p_{t|Z}(y|z) \, p_Z(z) \\
&\stackrel{(i)}{=} \sum_z \left[ \sum_x u_t(y, x|z) \, p_{t|Z}(x|z) \right] p_Z(z) \\
&= \sum_z \sum_x u_t(y, x|z) \, p_{t|Z}(x|z) \, p_Z(z) \\
&= \sum_x \sum_z u_t(y, x|z) \, p_{t|Z}(x|z) \, p_Z(z) \\
&= \sum_x \sum_z u_t(y, x|z) \, \frac{p_{t|Z}(x|z) p_Z(z)}{p_t(x)} \, p_t(x) \\
&\stackrel{\text{(贝叶斯法则)}}{=} \sum_x \Bigg( \sum_z u_t(y, x|z) \, p_{Z|t}(z|x) \Bigg) p_t(x) \\
&= \sum_x u_t(y, x) \, p_t(x)
\end{aligned}
\]
其中等式（i）由定理13以及\(u_t(y, x|z)\)生成\(p_{t|Z}(y|z)\)这一事实推导得出；等式（ii）通过乘以并除以假设为正的\(p_t(x)\)得到。因此，\(u_t(y, x)\)满足与\(p_t\)相关的科尔莫戈罗夫方程（Kolmogorov Equation）。此外，由于每个\(u_t(y, x|z)\)均满足速率条件（6.4），故\(u_t(y, x)\)也满足该条件。最后，因为\(u_t(y, x|z)\)和\(p_{Z|t}(z|x)\)均属于\(C([0,1))\)，所以\(u_t(y, x) \in C([0,1))\)（特别地，\(p_{Z|t}(z|x) \in C([0,1))\)可由\(t \in [0,1)\)时\(p_t(x) > 0\)的假设推导得出）。根据定理13，由于\(u_t(x, y)\)满足与\(p_t\)相关的科尔莫戈罗夫方程和速率条件，因此它以（6.5）式的意义生成\(p_t\)。

### 7.4 离散流匹配损失
为构建连续时间马尔可夫链（CTMC）生成模型\((X_t)_{0 \leq t \leq 1}\)，我们通过参数θ对速度场\(u_t^\theta(y, x)\)进行参数化（例如使用神经网络）。构建神经网络时需满足速率条件方程（6.4）。训练连续时间马尔可夫链模型的离散流匹配损失定义为：
\[
\mathcal{L}_{DFM}(\theta) = \mathbb{E}_{t, X_t \sim p_t} D_{X_t}\left( u_t(\cdot, X_t), u_t^\theta(\cdot, X_t) \right),
\]
其中\(t \sim U[0,1]\)，且\(u_t(\cdot, x) \in \mathbb{R}^S\)满足速率条件。这意味着\(u_t(\cdot, x) \in \Omega_x\)，其中：
\[
\Omega_x = \left\{ v \in \mathbb{R}^{\mathcal{S}} \mid v(y) \geq 0 \forall y \neq x, \text{ 且 } v(x) = -\sum_{y \neq x} v(y) \right\} \subset \mathbb{R}^{\mathcal{S}},
\]
\(\Omega_x\)是一个凸集，\(D_x(u, v)\)是基于严格凸函数\(\Phi_x: \Omega_x \to \mathbb{R}\)定义的布雷格曼散度。条件离散流匹配损失的形式为：
\[
\mathcal{L}_{CDFM}(\theta) = \mathbb{E}_{t, Z, X_t \sim p_{t|Z}} D_{X_t}\left( u_t(\cdot, X_t|Z), u_t^\theta(\cdot, X_t) \right).
\]
同样，损失（7.4）和（7.6）提供相同的学习梯度。

**定理15**：离散流匹配损失和条件离散流匹配损失的梯度一致：
\[
\nabla_\theta \mathcal{L}_{DFM}(\theta) = \nabla_\theta \mathcal{L}_{CDFM}(\theta).
\]
特别地，条件离散匹配损失的极小值点为边际速度：
\[
u_t^\theta(y, x) = \mathbb{E}\left[ u_t(y, X_t|Z) | X_t = x \right].
\]

**证明**：通过设定\(X = X_t\)、\(Y = (X_t, Z)\)，定义函数\(f: S^2 \to \mathbb{R}^S\)为\((x, z) \mapsto u_t(\cdot, x|z) \in \mathbb{R}^S\)，并对\(t \in [0,1]\)积分，应用命题1即可完成证明。

### 7.5 因子化路径与速度
若按上述方式直接实现离散流匹配（DFM），则需要一个可学习模型\(u_t^\theta(y, x)\)（例如神经网络）为所有可能的状态\(y \in S = T^d\)输出一个速率，这将导致输出维度达到\(K^d\)，对于常见的序列长度d和词汇表大小K而言是不可行的。解决这一问题的一种方法是考虑因子化速度（Campbell et al., 2022）：
\[
u_t(y, x) = \sum_i \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) u_t^i\left( y^i, x \right),
\]
其中\(\bar{i} = (1, ..., i-1, i+1, ..., d)\)表示除i之外的所有索引。因此，上述因子化速度仅在状态x和状态y至多只有一个标记不同时建立连接。使用因子化速度时，我们只需对\(u_t^i(y^i, x)\)进行建模，即可完全定义\(u_t(y, x)\)。相应地，每个\(u_t^i(y^i, x)\)是一个可学习模型，接收\(x \in S\)并输出标量\(u_t^i(y^i, x) \in \mathbb{R}\)，适用于所有\(i \in [d] = \{1, 2, ..., d\}\)和\(y^i \in T\)。因此，模型的输出维度变为易于处理的\(d \cdot K\)。因子化速度\(u_t^i(y, x)\)的速率条件需针对每个维度\(i \in [d]\)满足：
\[
u_t^i\left( y^i, x \right) \geq 0 \text{ 对所有 } y^i \neq x^i, \text{ 且 } \sum_{y^i \in \mathcal{T}} u_t^i\left( y^i, x \right) = 0 \quad \text{ 对所有 } x \in \mathcal{S}.
\]

#### 7.5.1 含因子化速度的连续时间马尔可夫链（CTMC）模拟
使用因子化速度时，可按坐标对连续时间马尔可夫链（CTMC）模型进行采样（Campbell et al., 2024）：
\[
\begin{aligned}
\mathbb{P}\left( X_{t+h} = y | X_t = x \right) & = \delta(y, x) + h \sum_i \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) u_t^i\left( y^i, x \right) + o(h) \\
& = \prod_i \left[ \delta\left( y^i, x^i \right) + h u_t^i\left( y^i, x \right) + o(h) \right],
\end{aligned}
\]
其中第二个等式由\(\delta(y, x) = \prod_i \delta(y^i, x^i)\)以及恒等式\(\prod_i \left[ a^i + h b^i \right] = \prod_i a^i + h \sum_i \left( \prod_{j \neq i} a^j \right) b^i + o(h)\)推导得出。因此，在\(o(h)\)阶近似下，转移核可因子化为坐标独立的转移：
\[
\mathbb{P}\left( X_{t+h}^i = y^i | X_t = x \right) = \delta\left( y^i, x^i \right) + h u^i\left( y^i, x \right) + o(h).
\]
这些转移可通过欧拉方法（6.7）按坐标进行采样。有趣的是，连续流匹配同样具有类似的因子化形式\(u_t(x) = [u_t^1(x), ..., u_t^d(x)] \in \mathbb{R}^d\)，其中\(\dot{X}_t^i(x) = u_t^i(X_t)\)决定坐标i的变化，且可独立采样（连续流匹配中的“采样”本质上是确定性的）。

#### 7.5.2 含因子化速度的概率路径构建
若以特定方式构建概率路径，则其速度将天然具有因子化形式（方程（7.9））。接下来我们将详细说明这一构建过程。为此，我们定义因子化概率路径为以下形式：
\[
q_t(x) = \prod_i q_t^i(x^i).
\]
随后，下述结果表明这些因子化概率路径具有因子化速度。

**命题2**：设\(q_t(x)\)是如（7.12）式所示的因子化概率路径，其中\(u_t^i(y^i, x^i) \in C([0,1))\)生成\(q_t^i(x^i)\)。则\(q_t\)具有如下形式的因子化生成速度：
\[
u_t(y, x) = \sum_i \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) u_t^i\left( y^i, x^i \right).
\]

为进行证明，我们先通过以下式子定义概率质量函数\(q(x)\)的边际分布：
\[
q^i\left( x^i \right) := \sum_{x^{\bar{i}}} q(x)
\]

**证明**：设\(q_t\)是因子化概率路径（7.12），\(u_t^i(y^i, x^i)\)是\(q_t^i(x^i)\)的生成速度。对t求导可得：
\[
\begin{aligned}
\frac{d}{dt} q_t(y) & = \sum_i q_t^{\bar{i}}\left( y^{\bar{i}} \right) \frac{d}{dt} q_t^i\left( y^i \right) \\
& \stackrel{(i)}{=} \sum_i \left[ \sum_{x^{\bar{i}}} \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) q_t^{\bar{i}}\left( x^{\bar{i}} \right) \right] \left[ \sum_{x^i} u_t^i\left( y^i, x^i \right) q_t^i\left( x^i \right) \right] \\
& \stackrel{(ii)}{=} \sum_x \left[ \sum_i \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) u_t^i\left( y^i, x^i \right) \right] q_t(x),
\end{aligned}
\]
其中等式（i）由\(q_t^{\bar{i}}(y^{\bar{i}}) = \sum_{x^{\bar{i}}} \delta(y^{\bar{i}}, x^{\bar{i}}) q_t^{\bar{i}}(x^{\bar{i}})\)和科尔莫戈罗夫方程（6.8）推导得出；等式（ii）通过改变求和顺序，并注意到根据\(q_t\)的定义，有\(q_t^{\bar{i}}(x^{\bar{i}}) q_t^i(x^i) = q_t(x)\)且\(\sum_{x^{\bar{i}}} \sum_{x^i} = \sum_x\)得出。

接下来，我们将展示构建具有因子化速度且插值于任意p和q的路径\(p_t\)的核心工具（Campbell et al., 2024; Gat et al., 2024）。

**定理16（离散因子化边际化技巧）**：考虑通过以下方式构建的边际概率路径：
\[
p_t(x) = \sum_z p_{t|Z}(x|z) p_Z(z), \text{ 其中 } p_{t|Z}(x|z) = \prod_i p_{t|Z}^i(x^i|z),
\]
即条件路径按（7.12）式的意义进行因子化。进一步假设\(u_t^i(y^i, x^i|z) \in C([0,1))\)生成\(C^1([0,1))\)中的\(p_{t|Z}^i(x^i|z)\)，且对所有\(x \in S\)和\(t \in [0,1)\)有\(p_t(x) > 0\)。则边际速度为：
\[
u_t(y, x) = \sum_i \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) u_t^i\left( y^i, x \right)
\]
其中：
\[
u_t^i(y^i, x) = \sum_z u_t^i(y^i, x^i|z) p_{Z|t}(z|x) = \mathbb{E}\left[ u_t^i(y^i, X_t^i|Z) | X_t = x \right]
\]
生成\(p_t(x)\)。

**证明**：根据命题2，因子化条件路径\(p_{t|Z}(x|z)\)具有因子化生成速度\(u_t(y, x|z) = \sum_i \delta(y^{\bar{i}}, x^{\bar{i}}) u_t^i(y^i, x^i|z)\)。因此：
\[
\begin{aligned}
u_t(y, x) & \stackrel{(i)}{=} \sum_z u_t(y, x|z) p_{Z|t}(z|x) \\
& \stackrel{(ii)}{=} \sum_z \left[ \sum_i \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) u_t^i\left( y^i, x^i|z \right) \right] p_{Z|t}(z|x) \\
& \stackrel{(iii)}{=} \sum_i \delta\left( y^{\bar{i}}, x^{\bar{i}} \right) \left[ \sum_z u_t^i\left( y^i, x^i|z \right) p_{Z|t}(z|x) \right].
\end{aligned}
\]
等式（i）由（7.2）式得出；等式（ii）基于\(p_{t|Z}\)具有因子化速度的假设得出；等式（iii）通过改变求和顺序得出。由于\(p_{t|Z}^i(x^i|z) \in C^1([0,1))\)且\(p_t(x) > 0\)，可得\(p_{t|Z}(x|z) \in C^1([0,1))\)；同理，由于\(u_t^i(y^i, x^i|z) \in C([0,1))\)，可得\(u_t(y, x|z) \in C([0,1))\)。因此，定理14表明\(u_t(y, x)\)生成\(p_t(x)\)，符合要求。

通过使用定理16，我们可按以下步骤设计具有因子化速度且插值于源概率质量函数p和目标概率质量函数q的概率路径\(p_t\)：
1. 寻找因子化概率条件路径\(p_{t|Z}(x|z) = \prod_i p_{t|Z}^i(x^i|z)\)，使得边际\(p_t(x)\)满足\(p_0 = p\)和\(p_1 = q\)；
2. 为\(p_{t|Z}^i(x^i|z)\)寻找生成速度\(u_t^i(y^i, x^i|z)\)。这可通过为所有\(y^i \in T\)、固定的\(i \in [d]\)、\(z \in Z\)和\(t \in [0,1)\)求解科尔莫戈罗夫方程实现：
\[
\sum_{x^i} u_t^i\left( y^i, x^i|z \right) p_{t|Z}^i\left( x^i|z \right) = \frac{d}{dt} p_{t|Z}^i\left( y^i|z \right).
\]
需要注意的是，（7.18）式是一个具有\(|T|\)个未知数的欠定线性方程组（未知数数量远少于整个状态空间\(|S|\)）。

#### 7.5.3 因子化速度的条件离散流匹配损失
将边际速度\(u_t^\theta\)表示为因子化速度\(u_t^\theta, i\)的形式，可得到如下条件流匹配损失：
\[
\mathcal{L}_{CDFM}(\theta) = \mathbb{E}_{t, Z, X_t \sim p_{t|Z}} \sum_i D_{X_t}^i\left( u_t^i(\cdot, X_t|Z), u_t^\theta, i(\cdot, X_t) \right),
\]
其中\(t \sim U[0,1]\)，且\(u_t^i(\cdot, x|z)\)、\(u_t^\theta, i(\cdot, x) \in \mathbb{R}^T\)满足速率条件。这意味着\(u_t^i(\cdot, x|z)\)、\(u_t^\theta, i(\cdot, x) \in \Omega_x\)，其中对于\(\alpha \in T\)，我们定义：
\[
\Omega_\alpha = \left\{ v \in \mathbb{R}^{\mathcal{T}} \mid v(\beta) \geq 0 \forall \beta \in \mathcal{T} \setminus \{\alpha\}, \text{ 且 } v(\alpha) = -\sum_{\beta \neq \alpha} v(\beta) \right\} \subset \mathbb{R}^{\mathcal{T}}.
\]
\(\Omega_\alpha\)是一个凸集，\(D_x^i(u, v)\)是基于凸函数\(\Phi_x^i: \Omega_{x^i} \to \mathbb{R}\)定义的布雷格曼散度。与之前类似，我们可通过设定\(X = X_t\)、\(Y = u_t^i(\cdot, X_t, Z) \in \mathbb{R}^\tau\)，令\(D_x^i(u, v)\)为\(\Omega_{x^i} \subset \mathbb{R}^T\)上的布雷格曼散度，并对\(t \in [0,1]\)积分，结合命题1来证明该损失的合理性。

#### 7.5.4 混合路径（Mixture Paths）
现在，我们将实施7.5.2节的内容，以构建实用的概率路径及其对应的条件速度。沿用Gat et al. (2024)的思路，我们基于\(Z = (X_0, X_1)\)进行条件化处理，以适应任意的数据耦合\((X_0, X_1) \sim \pi_{0,1}(X_0, X_1)\)。随后，我们构建因子化条件路径：
\[
p_{t|0,1}(x|x_0, x_1) = \prod_i p_{t|0,1}^i(x^i|x_0, x_1)
\]
作为混合分布：
\[
p_{t|0,1}^i(x^i|x_0, x_1) = \kappa_t \delta\left( x^i, x_1^i \right) + (1 - \kappa_t) \delta\left( x^i, x_0^i \right),
\]
其中\(\kappa: [0,1] \to [0,1]\)是\(C^1([0,1])\)调度器。需要注意的是，满足\(X_t^i \sim p_{t|0,1}^i(\cdot|x_0, x_1)\)的随机变量\(X_t^i\)服从：
\[
X_t^i = \begin{cases} 
x_1^i & \text{ 概率为 } \kappa_t \\
x_0^i & \text{ 概率为 } (1 - \kappa_t)
\end{cases}
\]
即它以依赖于时间t的概率取源状态或目标状态。

若\(\kappa_0 = 0\)且\(\kappa_1 = 1\)，则（7.1）式中的边际\(p_t(x)\)满足边界约束。我们还需要为\(p_{t|0,1}^i(x^i|x_0, x_1)\)寻找生成速度\(u_t^i(y^i, x^i|x_0, x_1)\)，它们是（7.18）式的解。推导过程如下：
\[
\begin{aligned}
\frac{d}{dt} p_{t|Z}^i\left( y^i|z \right) & \stackrel{(7.22)}{=} \dot{\kappa}_t \left[ \delta\left( y^i, x_1^i \right) - \delta\left( y^i, x_0^i \right) \right] \\
& \stackrel{(7.22)}{=} \dot{\kappa}_t \left[ \delta\left( y^i, x_1^i \right) - \frac{p_{t|Z}^i\left( y^i|z \right) - \kappa_t \delta\left( y^i, x_1^i \right)}{1 - \kappa_t} \right] \\
& = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta\left( y^i, x_1^i \right) - p_{t|Z}^i\left( y^i|z \right) \right] \\
& = \sum_{x^i} \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta\left( y^i, x_1^i \right) - \delta\left( y^i, x^i \right) \right] p_{t|Z}^i\left( x^i|z \right),
\end{aligned}
\]
为保持符号简洁，我们交替使用\(z = (x_0, x_1)\)和\(Z = (X_0, X_1)\)。综上，我们找到了生成（7.22）式中路径的条件速度：
\[
u_t^i\left( y^i, x^i|x_0, x_1 \right) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta\left( y^i, x_1^i \right) - \delta\left( y^i, x^i \right) \right].
\]

**代码9**展示了flow_matching库中混合路径的定义方式。

**代码9：离散概率路径**
```python
import torch
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.path_sample import DiscretePathSample

## 创建离散概率路径对象
path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))

## 采样条件路径
## t、X0和X1的批量大小为2
t = torch.tensor([0.25, 0.5])
x_0 = torch.tensor([0, 0])
x_1 = torch.tensor([1, 2])
sample: DiscretePathSample = path.sample(t=t, x_0=x_0, x_1=x_1)
sample.x_0  ## X0为[0, 0]
sample.x_1  ## X1为[1, 2]
## Xt的分布为：
## [以0.75的概率取0，以0.25的概率取1,
##  以0.5的概率取0，以0.5的概率取2]
sample.x_t
sample.t  ## t为[0.25, 0.5]
```

##### 速度后验参数化
与连续情形类似（例如4.8.1节），我们可以通过多种方式对速度\(u_t^i(y^i, x)\)进行参数化。第一种方式是直接对其进行参数化，类似于流中的速度；此处我们采用的另一种方式是基于混合边际速度的下述计算（遵循（7.17）式）：
\[
\begin{aligned}
u_t^i\left( y^i, x \right) & = \sum_{x_0, x_1} \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta\left( y^i, x_1^i \right) - \delta\left( y^i, x^i \right) \right] p_{0,1|t}\left( x_0, x_1|x \right) \\
& = \sum_{x_1^i} \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta\left( y^i, x_1^i \right) - \delta\left( y^i, x^i \right) \right] p_{1|t}^i\left( x_1^i|x \right),
\end{aligned}
\]
其中第二个等式中我们记后验\(p_{0,1|t}\)的边际为：
\[
p_{1|t}^i(x_1^i|x) = \sum_{x_0, x_1^{\bar{i}}} p_{0,1|t}(x_0, x_1|x) = \mathbb{E}\left[ \delta\left( x_1^i, X_1^i \right) | X_t = x \right].
\]
该推导通过可学习后验\(p_{1|t}^{\theta, i}(x_1^i|x)\)表示边际\(u_t^i(y^i, x)\)，这可理解为4.8.1节中\(x_1\)预测的离散版本。接下来，我们将探讨用于学习该后验的损失函数。

##### 混合路径的条件离散流匹配（CDFM）损失
我们提供两种学习\(p_{1|t}^{\theta, i}(x_1^i|x)\)的方案，均通过命题1证明其合理性。第一种方案是通过条件匹配损失学习边际后验（7.26）和（7.27）：
\[
\mathcal{L}_{CM}(\theta) = \mathbb{E}_{t, X_0, X_1, X_t} D_{X_t}\left( \delta\left( \cdot, X_1^i \right), p_{1|t}^{\theta, i}\left( \cdot|X_t \right) \right)
\]
由于\(\delta(\cdot, X_1^i)\)和\(p_{1|t}^{\theta, i}(\cdot|X_t)\)均为概率质量函数，因此我们可将布雷格曼散度设为KL散度\(D(p, q) = \sum_{\alpha \in T} p(\alpha) \log \frac{p(\alpha)}{q(\alpha)}\)（用于比较概率质量函数），得到：
\[
\mathcal{L}_{CM}(\theta) = -\mathbb{E}_{t, X_0, X_1, X_t} \log p_{1|t}^{\theta, i}\left( X_1^i|X_t \right) + const.
\]
另一种方案是沿用7.5.3节的思路，使用（7.19）式中的因子化损失，并通过\(p_{1|t}^{\theta, i}\)对\(u_t^{\theta, i}\)进行参数化。此时，我们可将布雷格曼散度设为广义KL散度（用于比较一般向量，不一定是概率向量）\(u, v \in \mathbb{R}_{≥0}^m\)：
\[
D(u, v) = \sum_j u^j \log \frac{u^j}{v^j} - \sum_j u_j + \sum_j v_j.
\]
选择该散度后，有：
\[
D\left( u_t^i\left( \cdot, x^i|x_0, x_1 \right), u_t^{\theta, i}(\cdot, x) \right) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \left( \delta\left( x_1^i, x^i \right) - 1 \right) \log p_{1|t}^{\theta, i}\left( x_1^i|x \right) + \delta\left( x_1^i, x^i \right) - p_{1|t}^{\theta, i}\left( x^i|x \right) \right]
\]
当基于\(Z = (X_0, X_1)\)进行条件化处理时，上式即实现了损失（7.19）。广义KL损失（7.31）还为目标分布的似然提供了证据下界（ELBO）（Shaul et al., 2024）：
\[
-\log p_1^\theta\left( x_1 \right) \leq \mathbb{E}_{t, X_0, X_t \sim p_{t|0,1}} \sum_i D\left( u_t^i\left( \cdot, X_t^i|X_0, x_1 \right), u_t^{\theta, i}\left( \cdot, X_t \right) \right),
\]
其中\(p_1^\theta\)是模型在时间\(t=1\)时生成的边际。因此，除训练外，广义KL损失还常用于评估。

##### 混合路径采样
基于后验\(p_{1|t}^{\theta, i}\)的参数化方式，可得到如下采样算法。如7.5.1节所述，使用因子化速度可支持按坐标采样（7.11）。根据（7.17）和（7.25）式，有：
\[
\begin{aligned}
\mathbb{P}\left( X_{t+h}^i = y^i | X_t = x \right) & = \delta\left( y^i, x^i \right) + h u^i\left( y^i, x \right) + o(h) \\
& = \sum_{x_1^i} \left[ \delta\left( y^i, x^i \right) + h \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta\left( y^i, x_1^i \right) - \delta\left( y^i, x^i \right) \right] + o(h) \right] p_{1|t}^i\left( x_1^i|x \right).
\end{aligned}
\]
因此，给定\(X_t = x\)，我们可通过对每个\(i \in [d]\)执行以下两个步骤来完成单步采样：（i）从\(p_{1|t}^i(X_1^i|x)\)中采样\(X_1^i\)；（ii）使用速度\(\frac{\dot{\kappa}_t}{1 - \kappa_t} [\delta(y^i, X_1^i) - \delta(y^i, x^i)]\)，根据欧拉步骤（6.7）更新\(X_{t+h}^i\)。直观而言，步骤（ii）决定将\(X_{t+h}^i\)设为\(X_1^i\)还是保持为\(X_t^i\)。

##### 单侧混合路径与概率保持速度
通过添加如6.3.1节所述的散度无关分量（divergence-free component）来扩展采样算法的设计空间通常是有用的。对于因子化路径，散度无关速度\(v_t^i\)需满足（7.18）式，即：
\[
\sum_{x^i} v_t^i\left( y^i, x^i|z \right) p_{t|Z}^i\left( x^i|z \right) = 0.
\]
一般而言，若不学习额外的量（例如\(p_{0|t}^i\)），找到此类概率保持速度可能具有挑战性。然而，在假设源分布为独立同分布（即\(p(x) = \prod_i p(x^i)\)）且耦合为独立耦合\(\pi_{0,1}(x_0, x_1) = p(x_0) q(x_1)\)的情况下，可通过闭式解找到概率保持速度。此时，边际混合路径的形式为：
\[
p_t(x) = \sum_{x_1} p_{t|1}(x|x_1) q(x_1), \text{ 其中 } p_{t|1}(x|x_1) = \prod_i p_{t|1}^i(x^i|x_1),
\]
且\(p_{t|1}^i(x^i|x_1) = [\kappa_t \delta(x^i, x_1^i) + (1 - \kappa_t) p(x^i)]\)。（7.24）式中的条件速度，即：
\[
u_t^i\left( y^i, x^i|x_1 \right) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta\left( y^i, x_1^i \right) - \delta\left( y^i, x^i \right) \right]
\]
同样生成\(p_{t|1}^i(x^i|x_1)\)。为找到散度无关速度，例如，我们可以从该速度中减去\(p_{t|1}^i(y^i, x^i|x_1)\)的反向时间速度\(\bar{u}_t^i\)（Gat et al., 2024），即其满足与\(p_{t|1}^i(y^i, x^i|x_1)\)相关的科尔莫戈罗夫方程，且\(-\bar{u}_t^i\)满足速率条件。此类速度可通过与（7.24）式类似的方式找到：
\[
\bar{u}_t^i\left( y^i, x^i|x_1 \right) = \frac{\dot{\kappa}_t}{\kappa_t} \left[ \delta\left( y^i, x^i \right) - p(x^i) \right].
\]
因此，\(p_{t|1}^i(x^i|x_1)\)条件路径的散度无关速度可通过下式定义：
\[
v_t^i\left( y^i, x^i|x_1 \right) = u_t^i\left( y^i, x^i|x_1 \right) - \bar{u}_t^i\left( y^i, x^i|x_1 \right).
\]
根据6.3.1节，若将散度无关场\(v_t^i(y^i, x^i|x_1)\)添加到速度\(u_t^i(y^i, x^i|x_1)\)中，后者仍会生成相同的概率路径\(p_{t|1}^i(x^i|x_1)\)。因此，定理16表明，由下式定义的边际速度\(u_t^i(y^i, x)\)：
\[
\begin{aligned}
u_t^i\left( y^i, x \right) & = \sum_{x_1} \left[ u_t^i\left( y^i, x^i|x_1 \right) + c_t v_t^i\left( y^i, x^i|x_1 \right) \right] p_{1|t}(x_1|x) \\
& = \sum_{x_1^i} \left[ u_t^i\left( y^i, x^i|x_1^i \right) + c_t v_t^i\left( y^i, x^i|x_1^i \right) \right] p_{1|t}^i\left( x^i|x \right),
\end{aligned}
\]
仍会生成相同的边际路径\(p_t(x)\)，其中第二个等式源于混合路径的\(u_t^i(y^i, x^i|x_1) = u_t^i(y^i, x^i|x_1^i)\)，\(v_t^i(y^i, x^i|x_1)\)同理。综上，给定\(X_t = x\)，广义采样算法的单步包括：（i）从\(p_{1|t}^i(X_1^i|x)\)中采样\(X_1^i\)；（ii）使用速度执行欧拉步骤（6.7）：
\[
u_t^i(y^i, x^i|x_1) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta(y^i, X_1^i) - \delta(y^i, x^i) \right] + c_t \left[ \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta(y^i, x_1^i) - \delta(y^i, x^i) \right] - \frac{\dot{\kappa}_t}{\kappa_t} \left[ \delta(y^i, x^i) - p(x^i) \right] \right],
\]
其中\(c_t > 0\)是时间相关常数。

与代码1中的连续流匹配示例类似，我们在代码11中提供了纯PyTorch实现的离散流匹配独立代码。代码10展示了如何使用flow_matching库训练具有任意数据耦合的离散流。

**代码10：使用混合路径和任意数据耦合训练与采样离散流匹配（DFM）**
```python
import torch
from flow_matching.path import MixtureDiscreteProbPath, DiscretePathSample
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

model = ...  ## 定义可训练的速度模型
optimizer = torch.optim.Adam(model.parameters())

scheduler = PolynomialConvexScheduler(n=1.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)
loss_fn = MixturePathGeneralizedKL(path=path)  ## 广义KL布雷格曼散度

for x_0, x_1 in dataloader:  ## 来自π0,1的样本，形状为[batch_size, *data_dim]
    t = torch.rand(batch_size) * (1.0 - 1e-3)  ## 随机时间t ∼ U[0, 1−10−3]
    sample: DiscretePathSample = path.sample(t=t, x_0=x_0, x_1=x_1)  ## 采样条件路径
    model_output = model(sample.x_t, sample.t)
    
    loss = loss_fn(logits=model_output, x_1=sample.x_1, x_t=sample.x_t, t=sample.t)  ## 条件离散流匹配（CDFM）损失
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class ProbabilityDenoiser(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        logits = self.model(x, t, **extras)
        return torch.nn.functional.softmax(logits.float(), dim=-1)

## 采样X1
probability_denoiser = ProbabilityDenoiser(model=model)
x_0 = torch.randint(size=[batch_size, *data_dim])  ## 指定初始条件
solver = MixtureDiscreteEulerSolver(
    model=probability_denoiser,
    path=path,
    vocabulary_size=vocabulary_size
)

step_size = 1 / 100
x_1 = solver.sample(x_init=x_0, step_size=step_size, time_grid=torch.tensor([0.0, 1.0 - 1e-3]))
```

**代码11：独立离散流匹配代码（flow_matching/examples/standalone_discrete_flow_matching.ipynb）**
```python
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
from sklearn.datasets import make_moons

class DiscreteFlow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 128, v: int = 128):
        super().__init__()
        self.v = v
        self.embed = nn.Embedding(v, h)
        self.net = nn.Sequential(
            nn.Linear(dim * h + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * v))
    
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t[:, None], self.embed(x_t).flatten(1, 2)), -1)).reshape(list(x_t.shape) + [self.v])

batch_size = 256
vocab_size = 128

model = DiscreteFlow(v=vocab_size)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for _ in range(10000):
    x_1 = Tensor(make_moons(batch_size, noise=0.05)[0])
    x_1 = torch.round(torch.clip(x_1 * 35 + 50, min=0.0, max=vocab_size - 1)).long()
    x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, 2))
    
    t = torch.rand(batch_size)
    x_t = torch.where(torch.rand(batch_size, 2) < t[:, None], x_1, x_0)
    
    logits = model(x_t, t)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), x_1.flatten(0, 1)).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()

x_t = torch.randint(low=0, high=vocab_size, size=(200, 2))
t = 0.0
results = [(x_t, t)]
while t < 1.0 - 1e-3:
    p1 = torch.softmax(model(x_t, torch.ones(200) * t), dim=-1)
    h = min(0.1, 1.0 - t)
    one_hot_x_t = nn.functional.one_hot(x_t, vocab_size).float()
    
    u = (p1 - one_hot_x_t) / (1.0 - t)
    x_t = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()
    t += h
    results.append((x_t, t))

fig, axes = plt.subplots(1, len(results), figsize=(15, 2), sharex=True, sharey=True)
for (x_t, t), ax in zip(results, axes):
    ax.scatter(x_t.detach()[:, 0], x_t.detach()[:, 1], s=10)
    ax.set_title(f't={t:.1f}')
plt.tight_layout()
plt.show()
```


## 8 Continuous Time Markov Process Models
在前面的章节中，我们已经在\(\mathbb{R}^{d}\)空间和黎曼流形上构建了流模型（参见第3章和第5章），并为离散数据构建了连续时间马尔可夫链（CTMC）模型（参见第6章）。在本章中，我们希望将这些模型统一并扩展为一种生成模型，使其能够适用于：（1）一般状态空间；（2）一般马尔可夫过程。这种生成模型将使我们能够在第9章中将流匹配（Flow Matching）的原理扩展到适用于多种模态的各类生成模型。

### 8.1 一般状态空间与随机变量
#### 多模态适配
我们的核心目标是不限制所使用的模态。因此，在本章中，设\(S\)为一般状态空间。重要的例子包括\(S=\mathbb{R}^{d}\)（例如图像、向量）、离散\(S\)（例如语言）、黎曼流形\(S\)（例如几何数据），或它们的乘积空间（用于联合生成多种数据模态的多模态模型）。对于所有模态，我们都可以在\(S\)上定义一个度量（或距离函数）\(d: S ×S \to \mathbb{R}_{≥0}\)，即\((x, y) \mapsto d(x, y)\)。例如，对于离散\(S\)，度量定义为：若\(y ≠x\)，则\(d(x, y)=1\)；对所有\(x \in S\)，则\(d(x, x)=0\)。对于\(S=\mathbb{R}^{d}\)，我们使用\(d(x, y)=\|x-y\|\)。我们需要做出一个技术假设：\((S, d)\)是波兰度量空间（Polish metric space），即它是完备的（任何柯西序列都收敛）且可分的（存在可数稠密子集）。机器学习中所有感兴趣的模态都满足这一性质。

#### 一般状态空间上的密度
到目前为止，在本文中我们假设状态空间\(S\)上的概率分布\(p\)由密度函数\(p: S \to \mathbb{R}_{≥0}\)表示。对于一般状态空间，我们采用一般参考测度\(\nu\)，此时密度函数即为拉东-尼科迪姆导数\(\frac{d p}{d \nu}\)。换句话说，概率可以表示为关于\(\nu\)的积分：
\[
\mathbb{P}(A)=\int_{A} p(x) \nu(d x) \quad \text{对所有可测集 } A \subset \mathcal{S}
\]

对于离散\(S\)，\(\nu\)是计数测度（因此积分本质上是求和），而\(p(x)\)就是概率质量函数（PMF）。对于\(S=\mathbb{R}^{d}\)，\(\nu\)是勒贝格测度（因此积分是“常规”积分），而\(p(x)\)就是概率密度函数（PDF）。上述定义将这一概念推广到了任意状态空间。

#### （可选）任意分布的处理
需要注意的是，并非所有概率分布都存在关于参考测度的密度函数。对于不熟悉一般测度论的读者，可将此视为技术说明而忽略，只需关注具有密度函数\(p(x)\)的感兴趣分布即可。但需指出，这并非仅存在于理论中的特殊情况，在机器学习应用中也有实际场景需要考虑：例如，在\(S=\mathbb{R}^{d}\)上，连接两点\(x, y \in \mathbb{R}^{d}\)的直线型概率路径\(p_{t}=\delta_{(1-t) x+t y}\)——该分布无法用密度函数表示；另一个例子是状态空间\(S=C([0,1], \mathbb{R})\)上的概率分布（例如用于轨迹建模），这类分布通常不存在关于常见参考测度的密度函数。为了在数学上处理这类情况，我们将框架扩展到状态空间\(S\)上的一般概率测度\(p\)。为此，我们使用符号\(p(d x)\)，其中“\(d x\)”是表示关于变量\(x\)在测度\(p\)下积分的符号表达式。例如，对于有界可测函数\(f: S \to \mathbb{R}\)，我们将\(f\)在\(p\)下的勒贝格积分或期望值表示为：
\[
\mathbb{E}_{X \sim p}[f(X)]=\int f(x) p(d x)
\]
与前文一致，我们（略微滥用符号）使用相同的记号\(p(x)\)来表示测度\(p(d x)\)的密度函数。

### 8.2 连续时间马尔可夫过程（CTMP）生成模型
同样，我们的核心目标是构建一个适用于任意演化过程的模型——无论该过程是流、扩散过程、连续时间马尔可夫链（CTMC）、它们的组合，还是其他类型。因此，在本节中，我们定义一种通用的演化过程，该过程满足构建生成模型所需的基本正则性假设。对于\(t \in[0,1]\)，设\(X_{t} \in S\)为随机变量。若\((X_{t})_{0 ≤t ≤1}\)满足以下条件，则称其为连续时间马尔可夫过程（CTMP）：
\[
\mathbb{P}[X_{t_{n+1}}\in A|X_{t_{1}},X_{t_{2}},...,X_{t_{n}}]=\mathbb{P}[X_{t_{n+1}}\in A|X_{t_{n}}] \quad (0\leq t_{1}<\cdots <t_{n+1}\leq 1,A\subseteq \mathcal{S})
\]

通俗地说，上述条件意味着该过程具有“无记忆性”：若已知当前状态，则了解过去的状态不会影响对未来的预测。表2概述了几类重要的马尔可夫过程。例如，流形上的流是具有确定性转移的马尔可夫过程，扩散过程是由布朗运动驱动转移的马尔可夫过程，而连续时间马尔可夫链（CTMC）是由速率决定的马尔可夫过程（我们将在8.2.2节详细解释）。每个马尔可夫过程都有一个转移核\((p_{t+h | t})_{0 ≤t<t+h ≤1}\)，该转移核为每个\(x \in S\)分配一个概率分布\(p_{t+h | t}(\cdot | x)\)，使得：
\[
\mathbb{P}\left[X_{t+h} \in A | X_{t}=x\right]=p_{t+h | t}(A | x) \quad \text{对所有 } t, h \geq 0, A \subset \mathcal{S} \text{ 可测}
\]

| 名称 | 流（Flow） | 扩散过程（Diffusion） | 跳跃过程（Jump process） | 连续时间马尔可夫链（Continuous-time Markov chain） |
| --- | --- | --- | --- | --- |
| 状态空间\(S\) | \(S=\mathbb{R}^{d}\) | \(S=\mathbb{R}^{d}\) | 任意\(S\) | 有限\(S\)（\(|S|<\infty\)） |
| 参数 | 速度场：\(u_{t}(x) \in \mathbb{R}^{d}\) | 扩散系数：\(\sigma_{t}(x) \in \mathbb{R}^{d \times d}\)（半正定矩阵） | 跳跃测度\(Q_{t}(d y, x)\) | 速率矩阵：\(u_{t} \in \mathbb{R}^{S \times S}\)，满足\(1^{T} u_{t}=0\)且\(u_{t}(x';x) \geq 0\)（\(x'\neq x\)） |
| 采样方式 | \(X_{t+h}=X_{t}+h u_{t}\left(X_{t}\right)\) | \(X_{t+h}=X_{t}+\sqrt{h} \sigma_{t}\left(X_{t}\right) \epsilon_{t}\)，其中\(\epsilon_{t} \sim \mathcal{N}(0, I)\) | 以概率\(1-h \int Q_{t}(d y, x)\)保持\(X_{t+h}=X_{t}\)；以概率\(h \int Q_{t}(d y, x)\)从\(\frac{Q_{t}(d y, x)}{\int Q_{t}(d y, x)}\)中采样\(X_{t+h}\) | \(X_{t+h} \sim\left(I+h u_{t}\right)\left(\cdot ; X_{t}\right)\) |
| 生成器\(\mathcal{L}_{t}\) | \(\nabla f^{T} u_{t}\) | \(\frac{1}{2} \nabla^{2} f \cdot \sigma_{t}^{2}\) | \(\int [f(y)-f(x)]Q_{t}(dy,x)\) | \(f^{T} u_{t}\) |
| 科尔莫戈罗夫前向方程（KFE，伴随形式） | 连续性方程：\(\partial_{t} p_{t}=-div\left(p_{t} u_{t}\right)\) | 福克-普朗克方程：\(\partial_{t} p_{t}=\frac{1}{2} \nabla^{2} \cdot\left(p_{t} \sigma_{t}^{2}\right)\) | 跳跃连续性方程：\(\partial_{t} p_{t}(x)=\int \left[Q_{t}(x, y) p_{t}(y)-Q_{t}(y, x) p_{t}(x)\right] \nu(d y)\) | 质量守恒：\(\partial_{t} p_{t}=u_{t} p_{t}\) |
| 边际形式 | \(\mathbb{E}_{Z \sim p_{Z \mid t}(\cdot \mid x)}\left[u_{t}(x \mid Z)\right]\) | \(\mathbb{E}_{Z \sim p_{Z \mid t}(\cdot \mid x)}\left[\sigma_{t}^{2}(x \mid Z)\right]\) | \(\mathbb{E}_{Z \sim p_{Z \mid t}(\cdot \mid x)}\left[Q_{t}(y, x \mid Z)\right]\) | \(\mathbb{E}_{Z \sim p_{Z \mid t}(\cdot \mid x)}\left[u_{t}(y ; x \mid Z)\right]\) |
| 生成器匹配（GM）损失示例 | \(\left\|u_{t}(X \mid Z)-u_{t}^{\theta}(X)\right\|^{2}\) | \(\left\|\sigma_{t}^{2}(X \mid Z)-\left(\sigma_{t}^{\theta}\right)^{2}(X)\right\|_{2}^{2}\) | \(\int Q_{t}(y,X|Z)\log \frac{Q_{t}(y,X|Z)}{Q_{t}^{\theta}(y;X)} \nu(d y)\) | \(\sum_{y \neq x} u_{t}(y;X|Z)\log \frac{u_{t}(y;X|Z)}{u_{t}^{\theta}(y;X)}\) |

表2 连续时间马尔可夫过程（CTMP）生成模型的部分示例及其通过生成器匹配（GM）的学习方式。此列表并非详尽无遗，推导过程见第8章。对于扩散过程，我们假设漂移项为零（漂移项相关内容已包含在“流”列中）。科尔莫戈罗夫前向方程（KFE）以伴随形式列出，即假设跳跃核\(Q_{t}(y, x)\)和密度\(p_{t}(x)\)关于参考测度\(\nu\)存在。“半正定矩阵（p.s.d.）”：正定半正定矩阵。

基于马尔可夫假设，一个马尔可夫过程由转移核和\(X_{0}\)的分布唯一确定；反之，任意转移核和初始分布都能定义一个马尔可夫过程。因此，两者之间存在一一对应关系。

我们的下一个目标是定义连续时间马尔可夫过程（CTMP）中与速度场相对应的通用概念。通俗地说，它是转移核在时间\(t\)处的一阶导数：
\[
\mathcal{L}_{t}:=\left.\frac{d}{d h}\right|_{h=0} p_{t+h | t}
\]

我们将这个一阶导数\(\mathcal{L}_{t}\)称为\(p_{t+h | t}\)的生成器（Ethier and Kurtz, 2009; Rüschendorf et al., 2016）。与导数类似，生成器是一阶线性近似，比\(p_{t+h | t}\)更易于参数化。

正如我们将看到的，扩散过程、流以及其他生成模型都可以被视为学习马尔可夫过程生成器的算法（见表2）。这引出了连续时间马尔可夫过程（CTMP）生成模型的一般形式：
\[
\text{CTMP模型（非正式）：} \quad p_{t+h | t}(\cdot | x):=\delta_{x}+h \mathcal{L}_{t}(x)+o(h), \quad X_{0} \sim p.
\]

然而，由于转移核\(p_{t+h | t}\)并非实值函数，方程（8.3）和（8.4）仅为启发式表达，尚未严格定义。因此，本节的首要目标是提供生成器和连续时间马尔可夫过程（CTMP）生成模型的正式定义。

#### 8.2.1 生成器的正式定义
方程（8.3）的第一个问题是：导数通常定义在映射到向量空间的函数上，但\(p_{t+h | t}\)映射到分布。不过，这一问题可以通过使用测试函数来解决。

测试函数是一种“探测”概率分布的工具，它作为理论工具，能将分布视为实值函数处理。具体来说，测试函数集是一族有界可测函数\(f: S \to \mathbb{R}\)，这些函数能完全表征概率分布，即对于状态空间\(S\)上的两个概率分布\(\mu_{1}\)和\(\mu_{2}\)，满足：
\[
\mu_{1}=\mu_{2} \Leftrightarrow \mathbb{E}_{X \sim \mu_{1}}[f(X)]=\mathbb{E}_{X \sim \mu_{2}}[f(X)] \quad \text{对所有 } f\in \mathcal{T}
\]

一般而言，我们会选择尽可能“良好”（或正则）的测试函数集\(\mathcal{T}\)。例如，若\(S=\mathbb{R}^{d}\)，则具有紧支撑的无穷可微函数空间\(\mathcal{T}=C_{c}^{\infty}(\mathbb{R}^{d})\)满足这一性质。对于离散\(S\)，\(\mathcal{T}=\mathbb{R}^{S}\) simply consists of all functions（在这种情况下，函数本质上就是向量）。设\(X_{t} \sim p_{t}\)，我们定义边际作用和转移作用如下：
\[
\left< p_{t}, f\right>:=\int f(x) p_{t}(d x)=\mathbb{E}_{X \sim p_{t}}[f(X)]
\]
\[
\left< p_{t+h | t}, f\right>(x):=\left< p_{t+h | t}(\cdot | x), f\right>=\mathbb{E}\left[f\left(X_{t+h}\right) | X_{t}=x\right]
\]
其中，边际作用将每个测试函数\(f\)映射到标量\(< p_{t}, f> \in \mathbb{R}\)，而转移作用将实值函数\(x \mapsto f(x)\)映射到另一个实值函数\(x \mapsto< p_{t+h | t}, f>(x)\)。塔性质（Tower Property）意味着\(< p_{t},< p_{t+h | t}, f>>=< p_{t+h}, f>\)。需要注意的是，上述表达式仅为“符号化”的点积，但当密度函数\(p_{t}(x)\)存在时，它会成为“真正的”点积，即\(< p_{t}, f>=\int f(x) p_{t}(x) \nu(d x)\)。

要正式定义方程（8.3）中的导数，第二步是为马尔可夫过程赋予某种“光滑性”，我们定义如下：设\(C_{0}(S)\)为在无穷远处消失的连续函数\(f: S \to \mathbb{R}\)构成的空间，即对于所有\(\epsilon>0\)，存在紧集\(K \subset S\)，使得对于所有\(x \in S \setminus K\)，有\(|f(x)|<\epsilon\)。我们在\(C_{0}(S)\)上使用上确界范数\(\|\cdot\|_{\infty}\)。若连续时间马尔可夫过程（CTMP）\(X_{t}\)满足以下两个条件，则称其为费勒过程（Feller process）（Feller, 1955; Rüschendorf et al., 2016）：
1. 强连续性：\(p_{t+h | t}\)的作用在时间上是连续的：
\[
\lim_{h^{\prime} \to h,t^{\prime} \to t}\left\| \left< p_{t^{\prime}+h^{\prime}|t^{\prime}},f\right> -\left< p_{t+h| t},f\right> \right\| _{\infty }=0 \quad \text{对所有 } h,t\geq 0,f\in C_{0}(\mathcal{S})
\]
2. 无穷远不可达性：\(p_{t+h | t}\)的作用保持在无穷远处消失的函数性质：
\[
\left< p_{t+h | t}, f\right> \in C_{0}(\mathcal{S}) \quad \text{对所有 } h, t \geq 0, f \in C_{0}(\mathcal{S})
\]

**假设4**：连续时间马尔可夫过程（CTMP）\((X_{t})_{0 ≤t ≤1}\)是费勒过程（Feller process）。

这一假设对于将\(X_{t}\)用于机器学习模型是合理的：我们定义的概率路径中，生成过程\(X_{t}\)的分布是平滑变化的，且我们的所有数据通常都位于某个有界（紧）集内。

现在，我们回到（8.3），尝试定义\(p_{t+h | t}\)的导数。结合测试函数的视角，我们可以对每个\(x \in S\)求\(< p_{t+h | t}, f>(x)\)的导数，并定义：
\[
\left.\frac{d}{d h}\right|_{h=0}\left< p_{t+h | t}, f\right>(x)=\lim _{h \to 0} \frac{\left< p_{t+h | t}, f\right>(x)-f(x)}{h}:=\left[\mathcal{L}_{t} f\right](x)
\]

我们将这一作用称为生成器\(\mathcal{L}_{t}\)，并将其定义于所有使上述极限在上确界范数\(\|\cdot\|_{\infty}\)下一致存在的测试函数\(f\)上。直观地说，生成器被定义为测试函数上的算子。表2列出了几个生成器的例子，我们将在8.2.2节中推导这些例子。生成器与费勒过程（Feller process）之间存在一一对应关系（Rogers and Williams, 2000; Ethier and Kurtz, 2009; Pazy, 2012）——这与流和向量场之间的对应关系类似（见定理1）。这使得我们后续能够通过神经网络对费勒过程（Feller process）进行参数化。

有了这一定义，（8.4）中的连续时间马尔可夫过程（CTMP）模型就有了严格的形式：
\[
\left< p_{t+h | t}, f\right>=f+h \mathcal{L}_{t} f+o(h) \quad (\text{对所有 } f \in \mathcal{T}) \quad \text{且 } X_{0} \sim p
\]
其中，\(o(h)\)表示误差项\(E(h) \in C_{0}(S)\)，满足\(\lim _{h \to 0} \frac{1}{h}\|E(h)\|_{\infty}=0\)。与流的情况（方程（3.24））和连续时间马尔可夫链（CTMC）的情况（方程（6.5））类似，若存在满足（8.9）的连续时间马尔可夫过程（CTMP）\(X_{t}\)，使得\(X_{t} ~ p_{t}\)，则称\(\mathcal{L}_{t}\)生成\(p_{t}\)：
\[
\mathcal{L}_{t} \text{ 生成 } p_{t} \Leftrightarrow \exists p_{t+h | t} \text{ 满足（8.9），且 } X_{t} \sim p_{t}
\]

换句话说，若一个生成器\(\mathcal{L}_{t}\)满足：（1）以\(p=p_{0}\)初始化；（2）通过\(\mathcal{L}_{t}\)模拟得到的马尔可夫过程的边际分布为\(p_{t}\)，则该生成器\(\mathcal{L}_{t}\)生成概率路径\(p_{t}\)。

#### 8.2.2 连续时间马尔可夫过程（CTMP）模型示例
我们通过几个例子来说明如何计算马尔可夫过程的生成器。本节的结果总结于表2。

##### 流（Flows）
设\(S=\mathbb{R}^{d}\)，\(u:[0,1] ×\mathbb{R}^{d} \to \mathbb{R}^{d}\)（即\((t, x) \mapsto u_{t}(x)\)）是一个时间依赖的速度场，定义了一个流\(\psi_{t}\)（见第3章）。设\(\mathcal{T}=C_{c}^{\infty}(\mathbb{R}^{d})\)为具有紧支撑的无穷可微光滑函数空间。则生成器可计算如下：
\[
\begin{aligned}
\left[\mathcal{L}_{t} f\right](x) & = \lim _{h \to 0} \frac{\mathbb{E}\left[f\left(X_{t+h}\right) | X_{t}=x\right]-f(x)}{h} \\
& \stackrel{(i)}{=} \lim _{h \to 0} \frac{\mathbb{E}\left[f\left(X_{t}+h u_{t}\left(X_{t}\right)+o(h)\right) | X_{t}=x\right]-f(x)}{h} \\
& \stackrel{(ii)}{=} \lim _{h \to 0} \frac{\mathbb{E}\left[f\left(X_{t}\right)+h \nabla f\left(X_{t}\right)^{T} u_{t}\left(X_{t}\right)+o(h) | X_{t}=x\right]-f(x)}{h} \\
& = \lim _{h \to 0} \frac{f(x)+h \nabla f(x)^{T} u_{t}(x)+o(h)-f(x)}{h} \\
& = \nabla f(x)^{T} u_{t}(x),
\end{aligned}
\]
其中，（i）来自流的欧拉近似；（ii）来自\(f\)在\(X_{t}\)附近的一阶泰勒展开。因此，流的生成器为：
\[
\mathcal{L}_{t} f(x)=\nabla f(x)^{T} u_{t}(x)
\]

##### 扩散过程（Diffusion）
设\(S=\mathbb{R}^{d}\)，\(\sigma_{t}: [0, 1] ×\mathbb{R}^{d} \to \mathbb{R}^{d×d}\)（即\((t, x) \mapsto \sigma_{t}(x)\)）是一个时间依赖的函数，映射到对称半正定矩阵\(\sigma_{t}\)，且映射过程连续。扩散系数为\(\sigma_{t}\)的扩散过程由随机微分方程（SDE）\(d X_{t}=\sigma_{t}(X_{t}) d W_{t}\)定义，其中\(W_{t}\)是维纳过程（Wiener process）（Øksendal, 2003）。该过程可通过无穷小采样过程近似：
\[
X_{t+h}=X_{t}+\sqrt{h} \sigma_{t}\left(X_{t}\right) \epsilon_{t}, \quad \epsilon_{t} \sim \mathcal{N}(0, I)
\]

设\(\mathcal{T}=C_{c}^{\infty}(\mathbb{R}^{d})\)，则生成器可计算如下：
\[
\begin{aligned}
\left[\mathcal{L}_{t} f\right](x) & = \lim _{h \to 0} \frac{\mathbb{E}\left[f\left(X_{t}+\sqrt{h} \sigma_{t}\left(X_{t}\right) \epsilon_{t}+o(h)\right) | X_{t}=x\right]-f(x)}{h} \\
& \stackrel{(i)}{=} \lim _{h \to 0} \frac{\mathbb{E}_{\epsilon_{t}}\left[ f(x)+\nabla f(x)^{T} \sqrt{h} \sigma_{t}(x) \epsilon_{t}+\frac{1}{2} h\left[\sigma_{t}(x) \epsilon_{t}\right]^{T} \nabla^{2} f(x)\left[\sigma_{t}(x) \epsilon_{t}\right]+o(h)-f(x)\right]}{h} \\
& = \lim _{h \to 0} \frac{\nabla f(x)^{T} \sqrt{h} \sigma_{t}(x) \mathbb{E}_{\epsilon_{t}}\left[ \epsilon_{t}\right] + \frac{1}{2} h \mathbb{E}_{\epsilon_{t}}\left[ \left[\sigma_{t}(x) \epsilon_{t}\right]^{T} \nabla^{2} f(x)\left[\sigma_{t}(x) \epsilon_{t}\right] \right] + o(h)}{h} \\
& = \frac{1}{2} \mathbb{E}_{\epsilon_{t}}\left[ \epsilon_{t}^{T}\left[\sigma_{t}(x)\right]^{T} \nabla^{2} f(x)\left[\sigma_{t}(x)\right] \epsilon_{t}\right] \\
& \stackrel{(ii)}{=} \frac{1}{2} tr\left(\sigma_{t}(x)^{T} \nabla^{2} f(x) \sigma_{t}(x)\right) \\
& \stackrel{(iii)}{=} \frac{1}{2} tr\left(\sigma_{t}(x) \sigma_{t}(x)^{T} \nabla^{2} f(x)\right) \\
& \stackrel{(iv)}{=} \frac{1}{2} \sigma_{t}^{2}(x) \cdot \nabla^{2} f(x),
\end{aligned}
\]
其中，（i）使用了二阶泰勒展开（由于\(\mathbb{E}[\left\|\sqrt{h} \epsilon_{t}\right\|^{2}] \propto h\)，因此需要二阶展开）；（ii）使用了矩阵迹的性质\(tr(A)=\mathbb{E}_{\epsilon_{t}}[\epsilon_{t}^{T} A \epsilon_{t}]\)（对\(A \in \mathbb{R}^{d ×d}\)）；（iii）使用了迹的循环性质；（iv）利用了\(\sigma_{t}\)的对称性。此外，我们使用\(A \cdot B:=tr(A^{T} B)\)表示矩阵\(A, B \in \mathbb{R}^{d ×d}\)的矩阵内积。因此，扩散过程的生成器为：
\[
\mathcal{L}_{t} f(x)=\frac{1}{2} \sigma_{t}^{2}(x) \cdot \nabla^{2} f(x)
\]

##### 跳跃过程（Jump processes）
接下来，设\(S\)为任意状态空间，考虑跳跃过程。跳跃过程由时间依赖的核\(Q_{t}(d y, x)\)定义，即对于每个\(0 ≤t ≤1\)和每个\(x \in S\)，\(Q_{t}(d y, x)\)是\(S \setminus \{x\}\)上的正测度。跳跃过程的核心思想是：分配给\(S \setminus \{x\}\)的总测度
\[
\lambda _{t}(x)=\int Q_{t}(dy,x)
\]
表示跳跃强度（即无穷小时间内发生跳跃的可能性）。此外，若\(\lambda_{t}(x)>0\)，则可通过归一化\(Q_{t}\)得到跳跃分布的概率核：
\[
J_{t}( d y, x)=\frac{Q_{t}( d y, x)}{\lambda_{t}(x)}
\]

跳跃过程可通过如下无穷小采样过程近似：
\[
X_{t+h}= \begin{cases}X_{t} & \text{ 概率为 } 1-h \lambda_{t}\left(X_{t}\right)+o(h) \\ \sim J_{t}\left( d y, X_{t}\right) & \text{ 概率为 } h \lambda_{t}\left(X_{t}\right)+o(h)\end{cases}
\]

关于跳跃过程的严格处理，可参考（Davis, 1984）。其生成器为：
\[
\begin{aligned}
\mathcal{L}_{t} f(x) & = \lim _{h \to 0} \frac{\mathbb{E}\left[f\left(X_{t+h}\right)-f\left(X_{t}\right) | X_{t}=x\right]}{h} \\
& = \lim _{h \to 0} \frac{\mathbb{E}\left[f\left(X_{t+h}\right)-f\left(X_{t}\right) | X_{t}=x, \text{ 在 } [t, t+h) \text{ 内跳跃}\right] \mathbb{P}\left[ \text{ 在 } [t, t+h) \text{ 内跳跃} | X_{t}=x\right]}{h} \\
& \quad + \lim _{h \to 0} \frac{\mathbb{E}\left[f\left(X_{t+h}\right)-f\left(X_{t}\right) | X_{t}=x, \text{ 在 } [t, t+h) \text{ 内不跳跃}\right] \mathbb{P}\left[ \text{ 在 } [t, t+h) \text{ 内不跳跃} | X_{t}=x\right]}{h} \\
& = \lim _{h \to 0} \frac{\mathbb{E}_{Y \sim J_{t}( d y, x)}[f(Y)-f(x)] h \lambda_{t}(x)}{h} \\
& = \lambda_{t}(x) \mathbb{E}_{Y \sim J_{t}( d y, x)}[f(Y)-f(x)] \\
& = \int (f(y)-f(x)) Q_{t}(dy,x),
\end{aligned}
\]
其中，我们利用了如下事实：若\(X_{t}\)在\([t, t+h]\)内不跳跃，则\(X_{t+h}=X_{t}\)。因此，跳跃过程的生成器为：
\[
\mathcal{L}_{t} f(x)=\int (f(y)-f(x)) Q_{t}(dy,x)=\lambda _{t}(x)\mathbb{E}_{Y\sim J_{t}(dy,x)}[f(Y)-f(x)]
\]

##### 连续时间马尔可夫链（CTMC）
考虑有限离散状态空间\(S\)（即\(|S|<\infty\)）上的连续时间马尔可夫链\(X_{t}\)。实际上，这是离散状态空间上的一种特殊参数化跳跃过程。为了说明这一点，考虑离散状态空间\(S\)上由矩阵\(Q_{t} \in \mathbb{R}_{≥0}^{S ×S}\)定义的标准跳跃核，结合方程（8.36），其生成器为：
\[
\mathcal{L}_{t} f(x)=\sum_{y \in \mathcal{S}}[f(y)-f(x)] Q_{t}(y, x)=\sum_{y \neq x}[f(y)-f(x)] Q_{t}(y, x) \quad \forall x \in \mathcal{S}, f \in \mathbb{R}^{\mathcal{S}}
\]
即\(Q_{t}(x, x)\)的值不影响生成器，且是欠定的。因此，一个自然的约定是通过速率对离散状态空间上的跳跃核进行重新参数化：
\[
u_{t}(y, x)= \begin{cases}Q_{t}(y, x) & \text{ 若 } y \neq x \\ -\sum_{z \neq x} Q_{t}(z, x) & \text{ 若 } y=x\end{cases}
\]

通过这种参数化，我们从第6章中恢复了满足速率条件（6.4）的速率\(u_{t}(y, x)\)。因此，这表明离散空间上的跳跃模型与连续时间马尔可夫链（CTMC）模型（第6章）是一致的。将此代入方程（8.37），可得连续时间马尔可夫链（CTMC）的生成器为：
\[
\mathcal{L}_{t} f(x)=\sum_{y \in \mathcal{S}} f(y) u_{t}(y, x)=f^{T} u_{t}
\]
其中，我们将\(f=(f(x))_{x \in S}\)视为列向量，将\(u_{t} \in \mathbb{R}^{S ×S}\)视为矩阵。因此，生成器函数本质上是左侧的向量乘法。

##### 流形上的流（Flows on manifolds）
接下来，考虑第5章中介绍的黎曼流形\(S=M\)上的流。流\(\psi:[0,1] ×M \to M\)通过向量场\(u:[0,1] ×M \to T M\)由方程（3.19）中的常微分方程（ODE）定义。设\(\psi_{t | s}(x)=\psi_{t}(\psi_{s}^{-1}(x))\)表示从时间\(s\)到时间\(t\)的转移（如（3.17）所示）。则对于光滑函数\(f: M \to \mathbb{R}\)，黎曼流的生成器为：
\[
\mathcal{L}_{t} f(x)=\lim_{h\to 0}\frac {f(\psi _{t+h|t}(x))-f(x)}{h}=\left< \nabla f(x),\frac {d}{dh}\bigg |_{h=0}\psi _{t+h| t}(x)\right> _{g}=\left< \nabla f(x),u_{t}(x)\right> _{g}
\]
其中，\(<\cdot, \cdot>_{g}\)表示定义黎曼度量\(g\)的点积，\(\nabla f\)表示\(f\)关于\(g\)的梯度。实际上，该生成器与函数的李导数（Lie derivative）一致（Jost, 2008），这是微分几何中的一个基本概念。

### 8.3 概率路径与科尔莫戈罗夫方程（Kolmogorov Equation）
对于\(S=\mathbb{R}^{d}\)上的流匹配（Flow Matching），连续性方程（见（3.25））是核心数学方程，它使我们能够构建生成期望概率路径的速度场（见3.5节）。在本节中，我们推导连续时间马尔可夫过程（CTMP）的对应（更通用）方程。设\(X_{t}\)是具有生成器\(\mathcal{L}_{t}\)的连续时间马尔可夫过程（CTMP），且\(X_{t} ~ p_{t}\)，则有：
\[
\frac{d}{dt}\left< p_{t},f\right> =\frac{d}{dh}\bigg | _{h=0}\left< p_{t+h},f\right> =\left. \frac{d}{dh}\bigg | _{h=0}\left< p_{t},\left< p_{t+h|t},f\right> \right> =\bigg < p_{t},\frac{d}{dh}\bigg |_{h=0}\left< p_{t+h|t},f\right> \bigg> =\left< p_{t},\mathcal{L}_{t}f\right>
\]
其中，我们利用了\(< p_{t}, \cdot>\)运算的线性性质来交换导数，并且通过塔性质（Tower Property）可知\(< p_{t},< p_{t+h | t}, f>>=< p_{t+h}, f>\)。这表明：给定马尔可夫过程\(X_{t}\)的生成器\(\mathcal{L}_{t}\)，我们可以通过其无穷小变化恢复其边际概率，即得到科尔莫戈罗夫前向方程（KFE）：
\[
\frac{d}{d t}\left< p_{t}, f\right>=\left< p_{t}, \mathcal{L}_{t} f\right> \quad \text{对所有 } f \in \mathcal{T}
\]

方程（8.40）中的科尔莫戈罗夫前向方程（KFE）描述了测试函数\(f\)的期望演化。这对于处理不存在密度的概率分布是必要的。若密度存在，则可使用更易理解的科尔莫戈罗夫前向方程（KFE）形式，直接描述概率密度的变化。为了呈现这一形式，我们引入伴随生成器\(\mathcal{L}_{t}^{*}\)，它作用于关于参考测度\(\nu\)的概率密度\(p_{t}(x)\)，即\(\mathcal{L}_{t}^{*} p_{t}(x)\)通过以下恒等式隐式定义：
\[
\int p_{t}(x) \mathcal{L}_{t} f(x) \nu(d x)=\int \mathcal{L}_{t}^{*} p_{t}(x) f(x) \nu(d x) \quad \forall f \in \mathcal{T}
\]

此外，我们需要假设\(p_{t}\)关于时间\(t\)是可微的。现在，将（8.41）应用于科尔莫戈罗夫前向方程（KFE）（8.40），可得：
\[
\begin{aligned}
\int \frac{d}{d t} p_{t}(x) f(x) \nu(d x) & =\frac{d}{d t} \int p_{t}(x) f(x) \nu(d x) \\
& =\frac{d}{d t}\left< p_{t}, f\right> \\
& =\left< p_{t}, \mathcal{L}_{t} f\right> \\
& =\int p_{t}(x) \mathcal{L}_{t} f(x) \nu(d x) \\
& =\int \mathcal{L}_{t}^{*} p_{t}(x) f(x) \nu(d x)
\end{aligned}
\]

由于这一等式对所有测试函数\(f\)都成立，结合方程（8.5），我们可以得出结论：这等价于伴随科尔莫戈罗夫前向方程（KFE）：
\[
\frac{d}{d t} p_{t}(x)=\mathcal{L}_{t}^{*} p_{t}(x) \quad \text{对所有 } x \in \mathcal{S}
\]

正如我们将在以下例子中推导的，伴随科尔莫戈罗夫前向方程（KFE）推广了许多用于开发生成模型的著名方程，例如连续性方程或福克-普朗克方程（Song et al., 2021; Lipman et al., 2022）（见表2）。只要概率密度存在，我们就使用伴随科尔莫戈罗夫前向方程（KFE）——以避免使用测试函数，直接处理概率密度。我们将研究结果总结如下：

**定理17（一般质量守恒）**：设\(\mathcal{L}_{t}\)是\((X_{t})_{0 ≤t ≤1}\)的生成器。通俗地说，以下条件等价：
1. \(p_{t}\)和\(\mathcal{L}_{t}\)满足科尔莫戈罗夫前向方程（KFE）（8.40）。
2. \(\frac{d p_{t}}{d \nu}(x)\)和\(\mathcal{L}_{t}\)满足伴随科尔莫戈罗夫前向方程（KFE）（8.47）。
3. \(\mathcal{L}_{t}\)以方程（8.10）的意义生成\(p_{t}\)。

严格来说，当\(\frac{d p_{t}}{d \nu}\)存在且关于时间\(t\)连续可微时，（1）和（2）等价。此外，对于任意状态空间，（3）蕴含（1）。存在一些弱正则性假设可确保（1）蕴含（3）（附录A.3列出了相关假设）。在本文中，我们假设这些假设成立，即假设（3）蕴含（1）。

据我们所知，目前尚无针对抽象一般状态空间的已知结果能确保定理17中条件（3）蕴含（1）。因此，我们在此直接假设这一结论成立。对于机器学习研究人员而言，这一假设在所有感兴趣的状态空间中均成立，因此无需担忧（见附录A.3）。

#### 8.3.1 科尔莫戈罗夫前向方程（KFE）示例
##### 流的伴随科尔莫戈罗夫前向方程（KFE）
设\(S=\mathbb{R}^{d}\)，假设\(p_{t}\)关于勒贝格测度存在密度\(p_{t}(x)\)，且该密度有界且连续可微。则可通过以下方式计算伴随生成器\(\mathcal{L}_{t}^{*}\)：
\[
\begin{aligned}
\left< p_{t}, \mathcal{L}_{t} f\right> & =\mathbb{E}_{x \sim p_{t}}\left[\mathcal{L}_{t} f(x)\right] \\
& =\int \mathcal{L}_{t} f(x) p_{t}(x) d x \\
& \stackrel{(i)}{=} \int \nabla f(x)^{T} u_{t}(x) p_{t}(x) d x \\
& \stackrel{(ii)}{=} \int f(x) \underbrace{\left[-div\left(p_{t} u_{t}\right)(x)\right]}_{=: \mathcal{L}_{t}^{*} p_{t}(x)} d x \\
& =\int f(x) \mathcal{L}_{t}^{*} p_{t}(x) d x
\end{aligned}
\]
其中，（i）来自方程（8.15）；（ii）来自分部积分。上述推导表明，伴随生成器为\(\mathcal{L}_{t}^{*} p_{t}=-div(p_{t} u_{t})(x)\)（因为它满足方程（8.41）中的条件）。利用伴随科尔莫戈罗夫前向方程（KFE），我们恢复了连续性方程（见方程（3.25））：
\[
\frac{d}{d t} p_{t}(x)=-div\left(p_{t} u_{t}\right)(x)
\]
这是我们在3.4节中详细研究的方程。

##### 扩散过程的伴随科尔莫戈罗夫前向方程（KFE）
设\(S=\mathbb{R}^{d}\)，假设\(p_{t}\)关于勒贝格测度存在密度\(p_{t}(x)\)，且该密度有界且连续可微。则可通过以下方式计算伴随生成器\(\mathcal{L}_{t}^{*}\)：
\[
\begin{aligned}
\left< p_{t}, \mathcal{L}_{t} f\right> & =\mathbb{E}_{x \sim p_{t}}\left[\mathcal{L}_{t} f(x)\right] \\
& =\int \mathcal{L}_{t} f(x) p_{t}(x) d x \\
& \stackrel{(i)}{=} \frac{1}{2} \int \sigma_{t}^{2}(x) \cdot \nabla^{2} f(x) p_{t}(x) d x \\
& \stackrel{(ii)}{=} \int f(x) \underbrace{\frac{1}{2} \nabla^{2} \cdot\left(p_{t} \sigma_{t}^{2}\right)(x)}_{=: \mathcal{L}_{t}^{*} p_{t}(x)} d x \\
& =\int f(x) \mathcal{L}_{t}^{*} p_{t}(x) d x
\end{aligned}
\]
其中，（i）来自方程（8.26）；（ii）来自两次分部积分。上述推导表明，伴随生成器为\(\mathcal{L}_{t}^{*} p_{t}=\frac{1}{2} \nabla^{2} \cdot(p_{t} \sigma_{t}^{2})(x)\)（因为它满足方程（8.41）中的条件）。伴随科尔莫戈罗夫前向方程（KFE）进而恢复了著名的福克-普朗克方程（Fokker-Planck Equation）：
\[
\frac{d}{dt}p_{t}(x)=\frac{1}{2}\nabla ^{2}\cdot (p_{t}\sigma _{t}^{2})(x)
\]

##### 跳跃过程的伴随科尔莫戈罗夫前向方程（KFE）
假设\(p_{t}\)关于勒贝格测度存在密度\(p_{t}(x)\)，且该密度有界且连续可微。假设跳跃测度\(Q_{t}(d y, x)\)由核函数\(Q_{t}: S ×S \to \mathbb{R}_{≥0}\)（即\((y, x) \mapsto Q_{t}(y, x)\)）表示，使得：
\[
\int f(y) Q_{t}( d y, x)=\int f(y) Q_{t}(y, x) \nu(d y) \quad \text{对所有可积函数 } f: \mathcal{S} \to \mathbb{R}
\]

则可按以下方式推导伴随生成器：
\[
\begin{aligned}
\left< p_{t}, \mathcal{L}_{t} f\right> & =\iint(f(y)-f(x)) Q_{t}(y, x) \nu(d y) p_{t}(x) \nu(d x) \\
& =\iint f(y) Q_{t}(y, x) p_{t}(x) \nu(d y) \nu(d x)-\iint f(x) Q_{t}(y, x) p_{t}(x) \nu(d y) \nu(d x) \\
& \stackrel{(i)}{=} \iint f(x) Q_{t}(x, y) p_{t}(y) \nu(d y) \nu(d x)-\iint f(x) Q_{t}(y, x) p_{t}(x) \nu(d y) \nu(d x) \\
& =\int f(x) \underbrace{\left[\int Q_{t}(x, y) p_{t}(y)-Q_{t}(y, x) p_{t}(x) \nu(d y)\right]}_{=: \mathcal{L}_{t}^{*} p_{t}(x)} \nu(d x) \\
& =\int f(x) \mathcal{L}_{t}^{*} p_{t}(x) \nu(d x)
\end{aligned}
\]
其中，（i）仅交换了变量\(x\)和\(y\)。上述推导表明，上述定义的\(\mathcal{L}_{t}^{*}\)满足方程（8.41）中的条件，确实描述了跳跃过程的伴随生成器。由此，伴随科尔莫戈罗夫前向方程（KFE）成为跳跃连续性方程（Jump Continuity Equation）：
\[
\frac{d}{d t} p_{t}(x)=\int\left[Q_{t}(x, y) p_{t}(y)-Q_{t}(y, x) p_{t}(x)\right] \nu(d y)=\int \lambda_{t}(y) J_{t}(x, y) p_{t}(y) \nu(d y)-\lambda_{t}(x) p_{t}(x)
\]
其中，我们利用了分解式\(Q_{t}(y, x)=\lambda_{t}(x) J_{t}(y, x)\)，将其分解为跳跃强度\(\lambda_{t}\)和跳跃分布\(J_{t}\)（见方程（8.28））。

##### 连续时间马尔可夫链（CTMC）的伴随科尔莫戈罗夫前向方程（KFE）
对于离散状态空间\(S\)，生成器由方程（8.38）中的\(f^{T} u_{t}\)给出，则有：
\[
\begin{aligned}
\left< p_{t}, \mathcal{L}_{t} f\right> & =\int p_{t}(x) \mathcal{L}_{t} f(x) \nu(d x) \\
& =\sum_{x \in \mathcal{S}} p_{t}(x) \sum_{y \in \mathcal{S}} u_{t}(y, x) f(y) \\
& =\sum_{y \in \mathcal{S}} \underbrace{\left[\sum_{x \in \mathcal{S}} p_{t}(x) u_{t}(y, x)\right]}_{=: \mathcal{L}_{t}^{*} p_{t}(y)} f(y) \\
& =\int \mathcal{L}_{t}^{*} p_{t}(y) f(y) \nu(d y)
\end{aligned}
\]
其中，此处的\(\nu\)仅表示计数测度。因此，伴随科尔莫戈罗夫前向方程（KFE）可简单表示为：
\[
\frac{d}{d t} p_{t}(x)=\sum_{y \in \mathcal{S}} u_{t}(x, y) p_{t}(y)
\]

这与方程（6.8）中推导的连续时间马尔可夫链（CTMC）的科尔莫戈罗夫前向方程（KFE）一致（为了与本节推导保持一致，交换了\(x\)和\(y\)的位置）。

### 8.4 通用表示定理（Universal representation theorem）
生成器使我们能够表征可能的马尔可夫过程空间。具体而言，以下结果不仅能表征一类广泛的连续时间马尔可夫过程（CTMP）生成模型，还能对\(S=\mathbb{R}^{d}\)或离散\(S\)的设计空间进行完整表征。

**定理18（生成器的通用表征）**：在弱正则性假设下，费勒过程（Feller process）\(X_{t}(0 ≤t ≤1)\)的生成器具有以下形式：
1. 离散状态空间（\(|S|<\infty\)）：生成器由速率转移矩阵\(u_{t}\)给出，对应的马尔可夫过程为连续时间马尔可夫链（CTMC）。
2. 欧几里得空间（\(S=\mathbb{R}^{d}\)）：生成器可表示为表2中描述的各分量之和，即：
\[
\mathcal{L}_{t} f(x)=\underbrace{\nabla f(x)^{T} u_{t}(x)}_{\text{流（Flow）}}+\underbrace{\frac{1}{2} \nabla^{2} f(x) \cdot \sigma_{t}^{2}(x)}_{\text{扩散过程（Diffusion）}}+\underbrace{\int[f(y)-f(x)] Q_{t}( d y, x)}_{\text{跳跃过程（Jump）}}
\]
其中，\(u:[0,1] ×\mathbb{R}^{d} \to \mathbb{R}^{d}\)是速度场，\(\sigma:[0,1] ×\mathbb{R}^{d} \to S_{d}^{++}\)是扩散系数（\(S_{d}^{++}\)表示正定半正定矩阵），\(Q_{t}(d y ; x)\)是跳跃测度；\(\nabla^{2} f(x)\)表示\(f\)的黑塞矩阵（Hessian），\(\nabla^{2} f(x) \cdot \sigma_{t}^{2}(x)\)表示弗罗贝尼乌斯内积（Frobenius inner product）。

该证明改编自数学文献中的已知结果（Courrege, 1965; von Waldenfels, 1965），详细证明可参见（Holderrieth et al., 2024）。



## 9 生成器匹配（Generator Matching）
在本章中，我们将介绍生成器匹配（Generator Matching, GM）（Holderrieth et al., 2024）——一种适用于（1）任意数据模态和（2）通用马尔可夫过程的生成建模框架。GM 统一了近年来开发的绝大多数生成模型，包括扩散模型、“离散扩散”模型以及前几节中描述的流匹配（FM）变体。为引入 GM，我们在第 8 章中定义了基于马尔可夫过程生成器构建的连续时间马尔可夫过程（CTMP）生成模型。GM 提出了一种可扩展的生成器训练算法——这也是该方法名称的由来。除了提供统一框架外，GM 还催生了多种新型模型，允许组合不同类别的模型，并能为任意模态（包括跨多种数据模态的模型）构建生成模型。

### 9.1 数据与耦合
与前文一致，我们的目标是将来自分布 \(p\) 的样本 \(X_0 \sim p\) 转换为来自目标分布 \(q\) 的样本 \(X_1 \sim q\)，其中 \(X_0\) 和 \(X_1\) 是两个取值于状态空间 \(S\) 的随机变量。源样本和目标样本可通过独立耦合 \((X_0, X_1) \sim p \otimes q\)（乘积分布）建立关联，也可通过一般的概率质量函数（PMF）耦合 \(\pi_{0,1}\)（即定义在 \(S \times S\) 上、边缘分布为 \(\pi_0 = p\) 和 \(\pi_1 = q\) 的分布）建立关联。与前文唯一的区别在于，此处的 \(S\) 是通用状态空间，而 \(p\) 和 \(q\) 可以是任意概率测度。

### 9.2 通用概率路径
生成器匹配（GM）流程的下一步，与前文一致，是定义一条插值于 \(p\) 和 \(q\) 的概率路径 \(p_t\)。沿用 4.4 节的思路，我们采用条件概率路径 \(p_{t|Z}(dx|z)\)，即一组依赖于潜在状态 \(z \in Z\) 的时变概率测度。给定 \(Z\) 上的分布 \(p_Z\)，我们通过分层采样过程定义对应的边际概率路径 \(p_t(dx)\)：
\[Z \sim p_Z, X_t \sim p_{t|Z}(dx|z) \implies X_t \sim p_t(dx)\]
也就是说，要从 \(p_t\) 中采样，需先从 \(p_Z\) 中采样 \(Z\)，再从 \(p_{t|Z}(dx|z)\) 中采样 \(X_t\)。与前文相同，边际概率路径需满足边界约束 \(p_0 = p\) 和 \(p_1 = q\)。

我们已介绍过两种常见的构造方式（当 \(Z = S\) 且 \(p_Z = q\) 时）：
1. 适用于 \(S = \mathbb{R}^d\) 的仿射条件流（用于连续流匹配；见第 4 章），定义为：
\[Z \sim q, X_0 \sim p, X_t = \sigma_t X_0 + \alpha_t Z \implies X_t \sim p_t(dx)\]
其中 \(\alpha_t, \sigma_t \in \mathbb{R}_{\geq 0}\) 是可微函数，满足 \(\alpha_0 = \sigma_1 = 0\) 和 \(\alpha_1 = \sigma_0 = 1\)。
2. 适用于任意 \(S\) 的混合路径（用于离散状态空间的离散流匹配，见方程（7.22））：
\[Z \sim q, X_0 \sim p, X_t \sim \begin{cases} Z & \text{概率为 } \kappa_t \\ X_0 & \text{概率为 } (1 - \kappa_t) \end{cases} \implies X_t \sim p_t(dx)\]
其中 \(\kappa_t \in \mathbb{R}_{\geq 0}\) 是可微函数，满足 \(\kappa_0 = 0\)、\(\kappa_1 = 1\) 且 \(0 \leq \kappa_t \leq 1\)。

不难看出，仿射条件概率路径和混合概率路径均能插值于 \(p\) 和 \(q\)，即满足 \(p_0 = p\) 和 \(p_1 = q\)。

### 9.3 通过神经网络参数化生成器
给定概率路径 \(p_t\)，我们的目标是构建一个由生成器 \(\mathcal{L}_t\) 指定的连续时间马尔可夫过程（CTMP）模型，使其生成该概率路径（见方程（8.10））。为了通过神经网络实现这一目标，我们首先需要说明如何用带参数 \(\theta\) 的神经网络 \(\mathcal{L}_t^\theta\) 对生成器 \(\mathcal{L}_t\) 进行参数化，具体如下。

设 \(\mathcal{T}\) 为测试函数族（见 8.2.1 节）。生成器 \(\mathcal{L}_t\) 的线性参数化定义如下：对于每个 \(x \in S\)，需满足：
1. 存在凸闭集 \(\Omega_x \subset V_x\)（\(V_x\) 是带有内积 \(\langle \cdot, \cdot \rangle_x\) 的向量空间）；
2. 存在线性算子 \(\mathcal{K}: \mathcal{T} \to C(S; V_x)\)，使得所有待考虑的生成器 \(\mathcal{L}_t\) 均可表示为：
\[
\mathcal{L}_t f(x) = \langle \mathcal{K} f(x), F_t(x) \rangle_x
\]
其中函数 \(F_t\) 满足对所有 \(x \in S\)，\(F_t(x) \in \Omega_x\)。关键在于，算子 \(\mathcal{K}\) 不能依赖于 \(\mathcal{L}_t\)——即仅需学习 \(F_t\)。由此得到：

**参数化生成器**：\(\mathcal{L}_t^\theta f(x) = \langle \mathcal{K} f(x), F_t^\theta(x) \rangle_x\)，其中 \(F_t^\theta\) 是带参数 \(\theta\) 的神经网络，且 \(F_t^\theta\) 将 \(x \in S\) 映射到 \(F_t^\theta(x) \in \Omega_x\)。

下面通过多个示例具体说明这一定义：

#### 流的线性参数化
设 \(S = \mathbb{R}^d\)，\(\Omega_x = \mathbb{R}^d = V_x\)。考虑所有流，其生成器族由下式给出（见方程（8.15））：
\[
\mathcal{L}_t f = \nabla f^T u_t, \quad u_t: \mathbb{R}^d \to \mathbb{R}^d
\]
令 \(\mathcal{K} f = \nabla f\) 且 \(F_t = u_t\)，即可恢复方程（9.3）的形式。这为通过向量场对流生成器进行线性参数化提供了自然方式。

#### 扩散过程的线性参数化
设 \(S = \mathbb{R}^d\)，\(\Omega_x = S_{d}^{++} \subset \mathbb{R}^{d \times d} = V_x\)（其中 \(S_{d}^{++}\) 表示所有半正定矩阵的集合）。扩散过程的生成器由下式给出（见方程（8.26））：
\[
\mathcal{L}_t f = \nabla^2 f \cdot \sigma_t^2, \quad \sigma_t: \mathbb{R}^d \to S_{d}^{++}
\]
令 \(\mathcal{K} f = \nabla^2 f\) 且 \(F_t = \sigma_t^2\)，即可恢复方程（9.3）的形式。这为扩散生成器的线性参数化提供了自然方式。

#### 跳跃过程的线性参数化
设 \(\Omega_x = \{a: S \setminus \{x\} \to \mathbb{R}_{\geq 0} \mid a \text{ 可积}\} \subset L^2(S \setminus \{x\}) = V_x\)，内积定义为 \(\langle a, b \rangle_x = \int_{S \setminus \{x\}} a(x) b(x) \nu(dx)\)。跳跃过程的生成器由下式给出（见方程（8.36））：
\[
\mathcal{L}_t f(x) = \int [f(y) - f(x)] Q_t(y, x) \nu(dy) = \langle \mathcal{K} f(x), Q_t(\cdot; x) \rangle_x
\]
其中我们令 \(\mathcal{K} f(x)\) 为函数 \(y \mapsto f(y) - f(x)\)。令 \(F_t = Q_t\)，即可恢复方程（9.3）的形式——实现对跳跃生成器的线性参数化。需注意，上述参数化仅适用于带有跳跃核 \(Q_t(y, x)\) 的跳跃过程，并非包含所有跳跃测度。

#### 连续时间马尔可夫链（CTMC）的线性参数化
设 \(S\) 为离散空间，\(u_t \in \mathbb{R}^{S \times S}\) 是连续时间马尔可夫链的速率矩阵。与离散流匹配（见方程（7.5））类似，定义：
\[
\Omega_x = \left\{ v \in \mathbb{R}^S \mid v(y) \geq 0 \ \forall y \neq x, \text{ 且 } v(x) = -\sum_{y \neq x} v(y) \right\} \subset V_x = \mathbb{R}^S
\]
由方程（8.38）可知，对于 \(f \in \mathbb{R}^S\)，生成器可表示为：
\[
\mathcal{L}_t f(x) = f^T u_t(\cdot, x) = \langle f, u_t(\cdot, x) \rangle_x
\]
其中 \(V_x = \mathbb{R}^S\)，\(\mathcal{K} f = f\)，\(\langle \cdot, \cdot \rangle_x\) 是标准欧几里得内积。由此可恢复方程（9.3）的形式，为通过速率 \(u_t\) 对连续时间马尔可夫链（CTMC）进行线性参数化提供了自然方式。

#### 流形上的流的线性参数化
设 \(S = M\) 为黎曼流形，考虑 5 节中介绍的流形上的流。由方程（8.39）可知，生成器可表示为：
\[
\mathcal{L}_t f(x) = \langle \nabla f(x), u_t(x) \rangle_g
\]
其中 \(u_t\) 是时变光滑向量场 \(u_t: [0,1] \times M \to TM\)，且对所有 \(x \in M\)，\(u_t(x) \in T_x M\)。令 \(\Omega_x = V_x = T_x M\)，\(\mathcal{K} = \nabla f\)（梯度算子），即可恢复方程（9.3）的形式。这为黎曼流生成器的线性参数化提供了自然方式。

### 9.4 边际生成器与条件生成器
本节将说明如何为边际概率路径寻找生成器。核心流程如下：我们可以先为条件概率路径 \(p_{t|Z}(dx|z)\) 找到生成器（通常可通过解析方法），再利用这些生成器构造边际路径的生成器。具体来说，假设对于每个 \(z \in Z\)，我们找到了（条件）生成器 \(\mathcal{L}_t^z\)，使其生成 \(p_{t|Z}(dx|z)\)——根据定理 17，这等价于满足科尔莫戈罗夫方程（KFE，方程（8.40））：
\[
\frac{d}{dt} \langle p_{t|Z}(\cdot|z), f \rangle = \langle p_{t|Z}(\cdot|z), \mathcal{L}_t^z f \rangle \quad \forall f \in \mathcal{T}
\]
进一步假设我们找到了如下线性参数化（见方程（9.3））：
\[
\mathcal{L}_t^z f(x) = \langle \mathcal{K} f(x), F_t(x|z) \rangle_x \quad z \in Z
\]
其中函数 \(F_t(x|z) \in \Omega_x \subset V_x\)。例如，\(F_t(x|z)\) 可以是连续流匹配中的条件速度场（见 4.3 节）或离散流匹配中的条件速率（见方程（7.2））。这使我们能够得到边际路径生成器的公式：

**定理 19（通用边际化技巧）**：边际概率路径 \((p_t)_{0 \leq t \leq 1}\) 由马尔可夫过程 \(X_t\) 生成，其生成器为：
\[
\mathcal{L}_t f(x) = \mathbb{E}_{Z \sim p_{Z|t}(\cdot|x)} \left[ \mathcal{L}_t^Z f(x) \right]
\]
其中 \(p_{Z|t}(dz|x)\) 是后验分布（即给定 \(x\) 时 \(z\) 的条件分布）。生成器 \(\mathcal{L}_t\) 的线性参数化为：
\[
F_t(x) = \mathbb{E}_{Z \sim p_{Z|t}(\cdot|x)} \left[ F_t(x|Z) \right]
\]

上述定理为我们提供了训练目标：用神经网络近似方程（9.13）中的 \(\mathcal{L}_t\)。前几章中的边际化技巧（定理 3、定理 10、定理 14）都是该定理的特例。下面给出证明，并展示几个新的实例化场景。

**证明**：要证明 \(\mathcal{L}_t\) 生成 \(p_t\)，需根据定理 17 证明其满足科尔莫戈罗夫方程（KFE）。设 \(p_{t+h|t}(\cdot|x, z)\) 是 \(\mathcal{L}_t^z\) 的转移核，则：
\[
\begin{aligned}
\frac{d}{dt} \langle p_t, f \rangle &= \lim_{h \to 0} \frac{1}{h} \left( \langle p_{t+h}, f \rangle - \langle p_t, f \rangle \right) \\
&= \lim_{h \to 0} \frac{1}{h} \left( \mathbb{E}_{Z \sim p_Z, X' \sim p_{t+h|t}(\cdot|Z)} [f(X')] - \mathbb{E}_{Z \sim p_Z, X \sim p_{t|Z}(\cdot|Z)} [f(X)] \right) \\
&= \lim_{h \to 0} \frac{1}{h} \left( \mathbb{E}_{Z \sim p_Z, X \sim p_{t|Z}(\cdot|Z), X' \sim p_{t+h|t}(\cdot|X, Z)} [f(X') - f(X)] \right) \\
&= \mathbb{E}_{X \sim p_t} \left( \mathbb{E}_{Z \sim p_{Z|t}(\cdot|X)} \left( \lim_{h \to 0} \frac{1}{h} \left( \mathbb{E}_{X' \sim p_{t+h|t}(\cdot|X, Z)} [f(X') - f(X)] \right) \right) \right) \\
&= \mathbb{E}_{X \sim p_t} \left( \mathbb{E}_{Z \sim p_{Z|t}(\cdot|X)} [\mathcal{L}_t^Z f(X)] \right) \\
&= \langle p_t, \mathcal{L}_t f \rangle
\end{aligned}
\]

关于 \(F_t\) 形式的证明如下：
\[
\begin{aligned}
\mathbb{E}_{Z \sim p_{Z|t}(\cdot|X)} [\mathcal{L}_t^z f(x)] &\stackrel{(9.12)}{=} \mathbb{E}_{Z \sim p_{Z|t}(\cdot|X)} \left( \langle \mathcal{K} f(x), F_t(x|z) \rangle_x \right) \\
&= \langle \mathcal{K} f(x), \mathbb{E}_{Z \sim p_{Z|t}(\cdot|X)} [F_t(x|z)] \rangle_x \\
&= \langle \mathcal{K} f(x), F_t(x) \rangle_x
\end{aligned}
\]
其中我们利用内积的线性性质，将其与期望交换顺序。这表明 \(F_t\) 是边际生成器的线性参数化（见方程（9.3））。

#### 示例 1：跳跃过程
设 \(S\) 为任意状态空间，\(Q_t(y, x|z)\) 是条件跳跃核（\(y, x \in S\)，\(z \in Z\)），生成条件概率路径 \(p_{t|Z}(dx|z)\)。利用跳跃核的线性参数化（见方程（9.7）），边际跳跃核：
\[
Q_t(y, x) = \mathbb{E}_{Z \sim p_{Z|t}(\cdot|x)} [Q_t(y, x|z)]
\]
生成边际概率 \(p_t(dx)\)。

#### 示例 2：边际扩散系数
设 \(S = \mathbb{R}^d\)，\(\sigma_t^2(x|z)\) 是扩散系数，生成条件概率路径 \(p_{t|Z}(dx|z)\)。利用扩散系数的线性参数化（见方程（9.6）），边际扩散系数：
\[
\sigma_t^2(x) = \mathbb{E}_{Z \sim p_{Z|t}(\cdot|x)} [\sigma_t^2(x|Z)]
\]
生成边际概率路径 \(p_t(dx)\)。

### 9.5 生成器匹配损失
接下来，我们将开发用于训练连续时间马尔可夫过程（CTMP）模型的训练目标。假设我们有神经网络 \(F_t^\theta\)，其通过方程（9.4）给出生成器参数化 \(\mathcal{L}_t^\theta\)。根据定理 19 推导，我们的目标是近似方程（9.14）给出的真实边际线性参数化 \(F_t\)。与前文一致，假设对于每个 \(x \in S\)，我们有通过下式定义的布雷格曼散度 \(D_x: \Omega_x \times \Omega_x \to \mathbb{R}\)：
\[
D_x(a, b) = \Phi_x(a) - \left[ \Phi_x(b) + \langle a - b, \nabla \Phi_x(b) \rangle \right], \quad a, b \in \Omega_x
\]
其中 \(\Phi_x: \Omega_x \to \mathbb{R}\) 是严格凸函数（见图 10）。用于训练连续时间马尔可夫过程（CTMP）模型的生成器匹配损失定义为：
\[
\mathcal{L}_{GM}(\theta) = \mathbb{E}_{t, X_t \sim p_t} D_{X_t} \left( F_t(X_t), F_t^\theta(X_t) \right)
\]
其中 \(t \sim U[0,1]\)。

遗憾的是，上述训练目标难以直接计算——因为我们既不知道边际生成器 \(\mathcal{L}_t\)，也不知道其参数化 \(F_t\)（仅知道方程（9.14）给出的难以计算的公式）。因此，我们引入条件生成器匹配损失作为可计算的替代方案，形式如下：
\[
\mathcal{L}_{CGM}(\theta) = \mathbb{E}_{t, Z, X_t \sim p_{t|Z}} D_{X_t} \left( F_t(X_t|Z), F_t^\theta(X_t) \right)
\]

该目标具有可计算性，因为在许多情况下我们可以通过解析方法推导 \(F_t(x|z)\)（见 9.6 节）。如下定理所示，损失（9.16）和（9.17）提供相同的学习梯度：

**定理 20**：生成器匹配损失和条件生成器匹配损失的梯度一致：
\[
\nabla_\theta \mathcal{L}_{GM}(\theta) = \nabla_\theta \mathcal{L}_{CGM}(\theta)
\]
特别地，条件生成器匹配损失的极小值点是边际生成器的线性参数化（方程（9.14））：
\[
F_t^\theta(x) = \mathbb{E}_{Z \sim p_{Z|t}(\cdot|x)} [F_t(x|Z)]
\]
此外，要满足上述性质，\(D_x\) 必须是布雷格曼散度。

上述定理将前几节推导的定理 4、定理 11 和定理 15 推广到了通用连续时间马尔可夫过程（CTMP）模型。它允许我们通过最小化条件生成器匹配损失，以可扩展的方式轻松训练任何由神经网络 \(F_t^\theta\) 参数化的连续时间马尔可夫过程（CTMP）模型。此外，它还全面表征了损失函数的空间。定理 20 的证明与定理 4 相同，只需将 \(u_t\) 替换为 \(F_t\)。关于 \(D\) 必须是布雷格曼散度的必要性证明，可参考（Holderrieth et al., 2024）。

#### 示例：训练扩散系数
我们将说明定理 20 如何用于训练随机微分方程（SDE）的扩散系数。设 \(S = \mathbb{R}^d\)，\(\sigma_t^2(x|z)\) 是生成条件概率路径 \(p_{t|Z}(dx|z)\) 的扩散系数。我们可以用神经网络 \((\sigma_t^2)^\theta(x) \in \mathbb{R}^{d \times d}\) 对扩散系数进行参数化。此时，条件生成器匹配损失为：
\[
\mathcal{L}_{CGM}(\theta) = \mathbb{E}_{t, Z, X_t \sim p_{t|Z}} \left\| \sigma_t^2(X_t|Z) - (\sigma_t^2)^\theta(X_t) \right\|^2
\]
其中我们使用均方误差作为布雷格曼散度（也可选择其他形式）。（Holderrieth et al., 2024）中给出了此类模型的训练示例。

### 9.6 寻找作为科尔莫戈罗夫方程（KFE）解的条件生成器
为了通过条件生成器匹配损失实现可扩展训练（见定理 20），我们需要找到满足科尔莫戈罗夫方程（KFE）的条件生成器 \(\mathcal{L}_t^z\)：
\[
\frac{d}{dt} \langle p_{t|Z}(\cdot|z), f \rangle = \langle p_{t|Z}(\cdot|z), \mathcal{L}_t^z f \rangle \quad \forall f \in \mathcal{T}, z \in Z
\]

如果 \(p_{t|Z}(dx|z)\) 关于参考测度 \(\nu\) 存在密度 \(p_{t|Z}(x|z)\)，则可等价求解伴随科尔莫戈罗夫方程（KFE）：
\[
\frac{d}{dt} p_{t|Z}(x|z) = \left[ (\mathcal{L}_t^z)^* p_{t|Z}(\cdot|z) \right](x) \quad \forall x \in S, z \in Z
\]

一般而言，方程（9.20）和（9.21）是难以解析求解的方程，目前尚无适用于任意生成器的通用求解公式。因此，我们通过两个示例说明如何求解，以作参考。

此处我们以跳跃模型为例（适用于任意状态空间）进行说明。如 8.2.2 节所述，跳跃模型由跳跃测度 \(Q_t\) 定义，可分解为：
\[
Q_t(dy, x) = \lambda_t(x) J_t(dy, x) \quad \forall x \in S
\]
\[
\lambda_t(x) \geq 0 \quad \forall x \in S
\]
\[
\int J_t(dy, x) = 1 \quad \forall x \in S
\]
其中 \(\lambda_t(x)\) 表示跳跃强度，\(J_t\) 表示指定跳跃分布的概率核。为简化符号，我们省略了对 \(z \in Z\) 的依赖（为避免与边际概率路径混淆，保留了 \(p_{t|Z}(dx|z)\) 中对 \(z\) 的依赖）。

#### 凸混合的跳跃模型
考虑由下式给出的混合概率路径（见方程（7.22））：
\[
p_{t|Z}(dx|z) = \kappa_t \delta_z(dx) + (1 - \kappa_t) p(dx), \quad z \in S
\]

利用跳跃过程生成器的形式（见方程（8.36）），科尔莫戈罗夫方程（KFE）可表示为：
\[
\frac{d}{dt} \langle p_{t|Z}(dx|z), f \rangle = \mathbb{E}_{X \sim p_{t|Z}(\cdot|z)} \left[ \lambda_t(X) \mathbb{E}_{Y \sim J_t(dy, X)} [f(Y) - f(X)] \right] \quad \forall f \in \mathcal{T}, x \in S
\]
其中 \(\lambda_t, J_t\) 满足方程（9.23）和（9.24）中的约束。我们断言，对于如下跳跃模型，上述方程成立：
\[
Q_t(dy, x) = \lambda_t(x) J_t(dy, x), \quad \lambda_t(x) = \frac{\dot{\kappa}_t}{1 - \kappa_t}, \quad J_t(dy, x) = \delta_z(dy)
\]
即跳跃强度由 \(\lambda_t\) 给出，且一旦决定跳跃，将直接跳至 \(z \in S\)。为验证这一点，我们证明上述跳跃过程满足科尔莫戈罗夫方程（KFE）：
\[
\begin{aligned}
&\mathbb{E}_{X \sim p_{t|Z}(\cdot|z)} \left[ \lambda_t(X) \mathbb{E}_{Y \sim J_t(\cdot, X)} [f(Y) - f(X)] \right] \\
&= \frac{\dot{\kappa}_t}{1 - \kappa_t} \mathbb{E}_{X \sim p_{t|Z}(\cdot|z)} [f(z) - f(X)] \\
&= \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ f(z) - \mathbb{E}_{X \sim p_{t|Z}(\cdot|z)} [f(X)] \right] \\
&= \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ f(z) - \left( \kappa_t f(z) + (1 - \kappa_t) \mathbb{E}_{X \sim p} [f(X)] \right) \right] \\
&= \dot{\kappa}_t f(z) - \dot{\kappa}_t \mathbb{E}_{X \sim p} [f(X)] \\
&= \frac{d}{dt} \left[ \kappa_t f(z) + (1 - \kappa_t) \mathbb{E}_{X \sim p} [f(X)] \right] \\
&= \frac{d}{dt} \langle p_{t|Z}(\cdot|z), f \rangle
\end{aligned}
\]

因此，上述过程满足跳跃科尔莫戈罗夫方程（KFE）（方程（9.26）），即我们已构建了一个跳跃模型。我们在 7.24 节中已见过该模型在离散状态空间中的特例；此处我们证明，也可在欧几里得空间 \(\mathbb{R}^d\) 等其他空间中构建类似的跳跃模型。

#### 具有密度的任意路径的跳跃模型
假设概率 \(p_{t|Z}(dx|z)\) 关于 \(S\) 上的参考测度 \(\nu\) 存在密度 \(p_{t|Z}(x|z)\)，且该密度关于 \(t\) 可微（注意，方程（9.25）中的混合路径在 \(S = \mathbb{R}^d\) 时不满足这一条件）。进一步假设跳跃核 \(J_t(y, x)\) 存在密度。此时，伴随科尔莫戈罗夫方程（KFE）变为跳跃连续性方程（方程（8.66））：
\[
\frac{d}{dt} p_{t|Z}(x|z) = \int \lambda_t(y) J_t(x, y) p_{t|Z}(y|z) dy - p_{t|Z}(x|z) \lambda_t(x)
\]
\[
\Leftrightarrow p_{t|Z}(x|z) \left[ \frac{d}{dt} \log p_{t|Z}(x|z) + \lambda_t(x) \right] = \int \lambda_t(y) J_t(x, y) p_{t|Z}(y|z) dy
\]

令 \(J_t(x, y) = J_t(x)\)（“目标状态无关”），并使用 \(\partial_t = \frac{d}{dt}\)，则上式等价于：
\[
p_{t|Z}(x|z) \left[ \partial_t \log p_{t|Z}(x|z) + \lambda_t(x) \right] = J_t(x) \int \lambda_t(y) p_{t|Z}(y|z) \nu(dy)
\]
\[
\Leftrightarrow \frac{p_{t|Z}(x|z) \left[ \partial_t \log p_{t|Z}(x|z) + \lambda_t(x) \right]}{\int \lambda_t(y) p_{t|Z}(y|z) \nu(dy)} = J_t(x)
\]

为定义有效的跳跃过程，需满足 \(\lambda_t(x) \geq 0\) 和 \(J_t(x) \geq 0\)。因此：
\[
\lambda_t(x) \geq 0, J_t(x) \geq 0 \Leftrightarrow \lambda_t(x) \geq \left[ -\partial_t \log p_{t|Z}(x|z) \right]_+
\]
其中 \([x]_+ = \max(x, 0)\) 表示ReLU操作。进一步，需确保 \(J_t\) 定义有效的跳跃分布（即积分等于 1），这一点可验证如下：
\[
\begin{aligned}
1 &= \int J_t(x) dx \\
\Leftrightarrow \int \lambda_t(x) p_{t|Z}(x|z) \nu(dx) &= \int p_{t|Z}(x|z) \left[ \partial_t \log p_{t|Z}(x|z) + \lambda_t(x) \right] \nu(dx) \\
\Leftrightarrow 0 &= \int \partial_t p_{t|Z}(x|z) \nu(dx) \\
\Leftrightarrow 0 &= \partial_t \int p_{t|Z}(x|z) \nu(dx) \\
\Leftrightarrow 0 &= 0
\end{aligned}
\]
即 \(J_t\) 的积分确实等于 1。选择最小的 \(\lambda_t(x)\)，可得如下定义的跳跃模型是跳跃连续性方程的解，因此生成条件概率路径 \(p_{t|Z}(x|z)\)：
\[
\lambda_t(x) = \left[ -\partial_t \log p_{t|Z}(x|z) \right]_+
\]
\[
J_t(x) = \frac{p_{t|Z}(x|z) \left[ \partial_t \log p_{t|Z}(x|z) \right]_+}{\int p_{t|Z}(y|z) \left[ \partial_t \log p_{t|Z}(y|z) \right]_+ \nu(dy)} = \frac{\left[ \partial_t p_{t|Z}(x|z) \right]_+}{\int \left[ \partial_t p_{t|Z}(y|z) \right]_+ \nu(dy)}
\]

初看之下，跳跃分布与位置无关似乎并不理想。然而，若将该模型扩展到多维空间，跳跃分布将依赖于位置，从而形成强大的生成模型（Holderrieth et al., 2024）。

### 9.7 组合模型
本节将说明生成器匹配（GM）如何允许以不同方式组合同一状态空间 \(S\) 上的生成模型。核心原理很简单：生成器是线性算子，而科尔莫戈罗夫方程（KFE）\(\partial_t \langle p_t, f \rangle = \langle p_t, \mathcal{L}_t f \rangle\) 是线性方程——因此，我们可以像在线性代数中组合矩阵方程的解一样组合该方程的解。具体来说，设 \(\mathcal{L}_t\) 和 \(\mathcal{L}_t'\) 是两个生成器，分别对应两个求解概率路径 \(p_t\) 的科尔莫戈罗夫方程（KFE）的马尔可夫过程。则对于 \(\alpha_t^1, \alpha_t^2 \in \mathbb{R}\) 且 \(\alpha_t^1 + \alpha_t^2 = 1\)，有：
\[
\begin{aligned}
\langle p_t, (\alpha_t^1 \mathcal{L}_t + \alpha_t^2 \mathcal{L}_t') f \rangle &= \alpha_t^1 \langle p_t, \mathcal{L}_t f \rangle + \alpha_t^2 \langle p_t, \mathcal{L}_t' f \rangle \\
&= \alpha_t^1 \partial_t \langle p_t, f \rangle + \alpha_t^2 \partial_t \langle p_t, f \rangle \\
&= (\alpha_t^1 + \alpha_t^2) \partial_t \langle p_t, f \rangle \\
&= \partial_t \langle p_t, f \rangle
\end{aligned}
\]
即 \(\alpha_t^1 \mathcal{L}_t + \alpha_t^2 \mathcal{L}_t'\) 仍是科尔莫戈罗夫方程（KFE）的解。一个需要注意的细节是，\(\alpha_t^1, \alpha_t^2\) 的正负性，以及 \(\mathcal{L}_t\) 和 \(\mathcal{L}_t'\) 对应前向时间还是后向时间的马尔可夫过程。由此得到：

**命题 3（组合模型）**：设 \(p_t\) 是边际概率路径，则以下生成器均求解 \(p_t\) 的科尔莫戈罗夫方程（KFE），因此定义了以 \(p_t\) 为边际的生成模型：
1. 马尔可夫叠加：\(\alpha_t^1 \mathcal{L}_t + \alpha_t^2 \mathcal{L}_t'\)，其中 \(\mathcal{L}_t, \mathcal{L}_t'\) 是两个求解 \(p_t\) 的科尔莫戈罗夫方程（KFE）的马尔可夫过程生成器，且 \(\alpha_t^1, \alpha_t^2 \geq 0\) 满足 \(\alpha_t^1 + \alpha_t^2 = 1\)。
2. 无散度分量：\(\mathcal{L}_t + \beta_t \mathcal{L}_t^{div}\)，其中 \(\mathcal{L}_t^{div}\) 是满足对所有 \(f \in \mathcal{T}\) 有 \(\langle p_t, \mathcal{L}_t^{div} f \rangle = 0\) 的生成器，且 \(\beta_t \geq 0\)。此类 \(\mathcal{L}_t^{div}\) 称为无散度生成器。
3. 预测-校正：\(\alpha_t^1 \mathcal{L}_t + \alpha_t^2 \overline{\mathcal{L}}_t\)，其中 \(\mathcal{L}_t\) 是前向时间内求解 \(p_t\) 的科尔莫戈罗夫方程（KFE）的生成器，\(\overline{\mathcal{L}}_t\) 是后向时间内求解 \(p_t\) 的科尔莫戈罗夫方程（KFE）的生成器，且 \(\alpha_t^1, \alpha_t^2 \geq 0\) 满足 \(\alpha_t^1 - \alpha_t^2 = 1\)。

下面通过马尔可夫叠加和无散度分量的示例说明命题 3。预测-校正方案的强大之处可参考（Gat et al., 2024）。

#### 示例 1：马尔可夫叠加——组合跳跃模型和流模型
马尔可夫叠加可用于组合不同类别的生成模型。这些模型可以是两个单独训练的网络，也可以是在一个网络中同时训练的两个生成器匹配（GM）模型（Holderrieth et al., 2024）。此处我们以组合 \(S = \mathbb{R}^d\) 上的跳跃模型和流模型为例进行说明。假设我们有两个模型，每个模型都生成概率路径 \(p_t\)：（1）流模型 \(u_t\)；（2）带有跳跃强度 \(\lambda_t\) 和跳跃分布 \(J_t\) 的跳跃模型。根据命题 3，对于 \(\alpha_t^1, \alpha_t^2 \geq 0\) 且 \(\alpha_t^1 + \alpha_t^2 = 1\)，以下生成器定义了生成 \(p_t\) 的有效生成器匹配（GM）模型：
\[
\begin{aligned}
\mathcal{L}_t f(x) &= \alpha_t^1 \mathcal{L}_t^{jump} f(x) + \alpha_t^2 \mathcal{L}_t^{flow} f(x) \\
&= \left( \alpha_t^1 \lambda_t(x) \right) \mathbb{E}_{Y \sim J_t(\cdot, x)} [f(Y) - f(x)] + \nabla f^T(x) \left( \alpha_t^2 u_t(x) \right)
\end{aligned}
\]
其中我们使用了方程（8.17）和（8.36）。实际上，上述生成器描述了分段确定性马尔可夫过程——一种组合常微分方程（ODE）和跳跃模型的模型（Davis, 1984）。如上述方程所示，需将跳跃强度按 \(\alpha_t^1\) 缩放，将向量场按 \(\alpha_t^2\) 缩放。由此得到的生成器匹配（GM）模型的采样过程如下：
\[
X_0 \sim p_0 = p
\]
\[
X_{t+h} = \begin{cases} \sim J_t(dy, X_t) & \text{概率为 } h \alpha_t^1 \lambda_t(X_t) \\ X_t + h \alpha_t^2 u_t(X_t) & \text{概率为 } 1 - h \alpha_t^1 \lambda_t(X_t) \end{cases}
\]

（Holderrieth et al., 2024）中给出了多个跳跃模型和流模型的马尔可夫叠加示例，并证明其能带来性能提升。

#### 示例 2：无散度分量——马尔可夫链蒙特卡洛（MCMC）算法
要找到无散度分量，可利用现有的马尔可夫链蒙特卡洛（MCMC）算法——所有这些算法都提供了寻找无散度分量的通用方案。此处我们通过两个著名示例进行说明。假设给定带有密度 \(p_t(x)\) 的通用概率路径 \(p_t\)。则生成器 \(\mathcal{L}_t^{div}\) 为无散度的等价条件是其伴随算子将 \(p_t\) 映射为零：
\[
\langle p_t, \mathcal{L}_t^{div} f \rangle = 0 \quad \forall f \in \mathcal{T} \Leftrightarrow [\mathcal{L}_t^{div}]^* p_t(x) = 0 \quad \forall x \in S
\]

##### 示例 2.1：朗之万动力学（Langevin Dynamics）
设 \(S = \mathbb{R}^d\)，朗之万动力学对应于带有速度场 \(\frac{1}{2} \beta_t^2 \nabla \log p_t(x)\) 和扩散系数 \(\beta_t\) 的随机微分方程（SDE），即动力学由下式给出：
\[
dX_t = \frac{1}{2} \beta_t^2 \nabla \log p_t(x) dt + \beta_t dW_t
\]

该随机微分方程（SDE）的伴随生成器为：
\[
[\mathcal{L}_t^{div}]^* p_t \stackrel{(i)}{=} -div\left( p_t \frac{1}{2} \beta_t^2 \nabla \log p_t \right)(x) + \frac{1}{2} \beta_t^2 \Delta p_t(x)
\]
\[
\stackrel{(ii)}{=} -\frac{1}{2} div\left( \beta_t^2 \nabla p_t \right)(x) + \frac{1}{2} \beta_t^2 \Delta p_t(x)
\]
\[
\stackrel{(iii)}{=} -\frac{1}{2} \beta_t^2 \Delta p_t(x) + \frac{1}{2} \beta_t^2 \Delta p_t(x) = 0
\]
其中（i）由 8.3.1 节推导的流和扩散的伴随算子形式得出；（ii）因 \(\nabla \log p_t = \nabla p_t / p_t\) 得出；（iii）由散度-拉普拉斯恒等式 \(div \nabla = \Delta\) 得出。上述结果表明，朗之万动力学的生成器满足方程（9.37），因此在命题 3 的意义下是无散度的。这一事实在统计物理和马尔可夫链蒙特卡洛（MCMC）中被广泛应用（Roberts and Tweedie, 1996）。命题 3 表明，我们可以将这些动力学（对于任意 \(\beta_t \geq 0\)）添加到任何生成模型中。在第 10 节中，我们将利用这一点推导扩散模型的随机采样。

##### 示例 2.2：梅特罗波利斯-黑斯廷斯算法（Metropolis-Hastings Algorithm）
设 \(S\) 为通用状态空间。梅特罗波利斯-黑斯廷斯算法（Hastings, 1970）描述了满足细致平衡条件的跳跃核 \(Q_t(y, x)\) 的构造：
\[
Q_t(y, x) p_t(x) = Q_t(x, y) p_t(y) \quad \forall x, y \in S
\]
\[
\Rightarrow [\mathcal{L}_t^{div}]^* p_t(x) \stackrel{(i)}{=} \int Q_t(y, x) p_t(x) - Q_t(x, y) p_t(y) = 0
\]
其中（i）使用了方程（8.66）。这表明方程（9.37）得到满足，\(Q_t\) 是无散度的。命题 3 表明，可将此类梅特罗波利斯方案任意添加到任何遵循概率路径 \(p_t\) 的生成器匹配（GM）模型中。

### 9.8 多模态模型
最后，我们简要说明生成器匹配（GM）如何支持联合构建多种数据模态的生成模型。例如，生成图像及其对应的文本描述的模型。两种模态可表示为两个状态空间 \(S_1, S_2\)（例如，\(S_1\) 对应图像，\(S_2\) 对应文本），而多模态模型则是定义在乘积空间 \(S = S_1 \times S_2\) 上的生成模型。由于 \(S\) 本身也是一个状态空间，且生成器匹配（GM）适用于任意状态空间，我们可以像构建其他生成器匹配（GM）模型一样构建多模态模型。

然而，有一种特定的概率路径构造方式允许我们重用为单个模态构建的生成器匹配（GM）模型。例如，我们可以通过组合离散流匹配（DFM）模型和连续流匹配（FM）模型来构建联合文本-图像模型。这种特定构造依赖于因子化条件概率路径。我们已在 7.5.2 节中见过离散流匹配（DFM）中的简单案例——因子化概率路径会产生因子化速度。这一结论适用于更通用的任意模态。尽管该构造简单直观，但要以完全通用的形式表达则较为繁琐。有关严格处理，可参考（Holderrieth et al., 2024）。（Campbell et al., 2024）中也实现了该构造的一个特定实例，用于多模态蛋白质生成。这表明生成器匹配（GM）能够以原则性和严谨的方式支持多模态模型的构建。


## 10 与扩散模型及其他去噪模型的关系
本章将生成器匹配（GM）与当前广泛使用的去噪类生成模型建立联系，包括**扩散模型**、**去噪分数匹配**、**条件去噪概率模型**等。我们会说明：所有这些模型都可以被**生成器匹配**统一解释，并且可以直接从第9章的框架中推导出来。

### 10.1 连续时间扩散模型
我们首先回顾连续时间扩散模型的标准形式，然后证明它是生成器匹配的特例。

设状态空间 \(S=\mathbb{R}^d\)。正向扩散过程（噪声过程）为：
\[
dX_t = b_t(X_t)dt + \sigma_t dW_t
\]
其中：
- \(b_t\) 为漂移项
- \(\sigma_t\) 为标量噪声强度
- \(W_t\) 为标准布朗运动

对应的**生成器**为：
\[
\mathcal{L}_t f(x) = \nabla f(x)^\top b_t(x) + \frac12 \sigma_t^2 \Delta f(x)
\]
其中 \(\Delta = \mathrm{tr}(\nabla^2)\) 为拉普拉斯算子。

对应的**科尔莫戈罗夫前向方程**（福克-普朗克方程）为：
\[
\partial_t p_t(x) = -\nabla\cdot\left(b_t(x)p_t(x)\right) + \frac12\sigma_t^2\Delta p_t(x)
\]

在标准扩散模型中，通常选择：
\[
b_t(x) = 0,\qquad \sigma_t > 0
\]
即**无漂移、纯扩散**。此时生成器简化为：
\[
\mathcal{L}_t f(x) = \frac12 \sigma_t^2 \Delta f(x)
\]
福克-普朗克方程简化为：
\[
\partial_t p_t = \frac12 \sigma_t^2 \Delta p_t
\]

#### 10.1.1 反向过程
连续时间扩散模型的核心是**反向过程**：
\[
dX_t = \left( b_{1-t}(X_t) - \sigma_{1-t}^2 \nabla\log p_{1-t}(X_t) \right)dt + \sigma_{1-t} d\widetilde{W}_t
\]
其中 \(\widetilde{W}_t\) 是反向时间的布朗运动。

在无漂移情况下 \(b_t=0\)，反向过程为：
\[
dX_t = -\sigma_{1-t}^2 \nabla\log p_{1-t}(X_t) dt + \sigma_{1-t} d\widetilde{W}_t
\]
这里的 \(-\sigma_t^2\nabla\log p_t\) 被称为**分数函数**（score function）。

#### 10.1.2 扩散模型是生成器匹配的特例
我们现在证明：**标准连续时间扩散模型等价于一种特殊的生成器匹配**。

在生成器匹配中，我们学习生成器：
\[
\mathcal{L}_t^\theta f(x) = \nabla f(x)^\top u_t^\theta(x) + \frac12 (\sigma_t^2)^\theta(x) \Delta f(x)
\]
它同时学习**流（漂移）**和**扩散**两部分。

在标准扩散模型中：
- 扩散部分固定为 \(\sigma_t^2\)，不学习
- 只学习漂移项 \(u_t^\theta(x) = -\sigma_t^2\nabla\log p_t(x)\)

这对应生成器匹配中**只学习流分量、固定扩散分量**的特殊情况。

因此：
> 所有连续时间扩散模型 = 只学习漂移项、固定扩散项的生成器匹配。

### 10.2 与去噪分数匹配的联系
去噪分数匹配（DSM）是训练扩散模型的经典方法。我们证明它也包含在生成器匹配框架中。

#### 10.2.1 去噪分数匹配回顾
给定条件分布 \(p_{t|Z}(x|z)\)（通常是带噪声的目标分布），分数匹配目标为：
\[
\mathcal{L}_{\text{DSM}}(\theta)
= \mathbb{E}_{t,Z,X_t\sim p_{t|Z}}
\big\| \nabla\log p_{t|Z}(X_t|Z) - \nabla\log p_t^\theta(X_t) \big\|^2
\]
模型学习近似条件分数 \(\nabla\log p_{t|Z}(x|z)\)。

#### 10.2.2 从生成器匹配直接推导出分数匹配
在生成器匹配中，对于**纯扩散过程**：
\[
\mathcal{L}_t^z f(x) = \frac12 \sigma_t^2 \Delta f(x)
\]
对应的**边际生成器**为：
\[
\mathcal{L}_t f(x) = \mathbb{E}_{Z|t,x}[\mathcal{L}_t^Z f(x)]
= \frac12 \sigma_t^2 \Delta f(x)
\]

但我们可以在生成器中**额外加入一个流分量**，使其仍然生成同一条概率路径 \(p_t\)。根据第9.7节的无散度分量理论，朗之万动力学是无散度的，因此可以叠加：
\[
\widetilde{\mathcal{L}}_t f(x)
= \frac12 \sigma_t^2 \Delta f(x)
+ \nabla f(x)^\top \left( \sigma_t^2 \nabla\log p_t(x) \right)
\]

这个新生成器**仍然生成相同的 \(p_t\)**。
它对应的反向过程正是标准扩散的反向SDE：
\[
dX_t = -\sigma_t^2\nabla\log p_t(X_t)dt + \sigma_t dW_t
\]

在生成器匹配中，我们直接拟合**条件速度**：
\[
u_t(x|z) = \sigma_t^2 \nabla\log p_{t|Z}(x|z)
\]
对应的损失为：
\[
\mathcal{L}_{\text{CGM}}(\theta)
= \mathbb{E}\big\| \sigma_t^2\nabla\log p_{t|Z}(X_t|Z) - u_t^\theta(X_t) \big\|^2
\]

这**完全等价于去噪分数匹配损失**（只差一个常数 \(\sigma_t^4\)）。

因此：
> 去噪分数匹配 = 生成器匹配在纯扩散过程下的特例。

### 10.3 与条件去噪概率模型的联系
条件去噪概率模型（CDPM）直接学习**去噪分布** \(p_{t|t+h}(x_t|x_{t+h})\) 或 \(p_{\text{target}|t}(z|x_t)\)。我们证明它同样是生成器匹配的特例。

#### 10.3.1 条件去噪概率模型
这类模型的典型训练目标是：
\[
\mathcal{L}_{\text{CDPM}}(\theta)
= \mathbb{E}_{t,Z,X_t\sim p_{t|Z}}
-\log p^\theta(Z|X_t,t)
\]
即用 \(X_t\) 去预测原始干净样本 \(Z\)。

在生成器匹配中：
- 对于仿射概率路径，条件速度为 \(u_t(x|z) = \dot{\alpha}_t z + \dot{\sigma}_t x\)
- 对于离散混合路径，条件速率由 \(\dot{\kappa}_t\) 决定

这些条件信号都可以写成**从 \(X_t\) 预测 \(Z\)** 的形式。因此：
\[
\text{条件去噪概率模型}
\quad\Longleftrightarrow\quad
\text{生成器匹配的条件形式}
\]

#### 10.3.2 离散扩散模型
离散状态空间上的扩散模型（如用于文本、分子的离散扩散）等价于：
- 连续时间马尔可夫链（CTMC）
- 由速率矩阵 \(u_t(y,x)\) 定义

而第9章已经证明：
> 所有离散扩散模型 = 离散状态空间上的生成器匹配。

这统一了**连续扩散**与**离散扩散**。

### 10.4 与流匹配的联系
流匹配（Flow Matching）是生成器匹配在**纯确定性流**下的特例：
- 生成器只有流分量：\(\mathcal{L}_t f = \nabla f^\top u_t\)
- 无扩散、无跳跃
- 损失为 \(\mathbb{E}\|u_t(x|z)-u_t^\theta(x)\|^2\)

因此：
> 流匹配 = 无扩散、无跳跃的生成器匹配。

### 10.5 统一视图
本章所有结论可以总结为一张统一关系图：

- **生成器匹配（GM）**：最通用框架，学习任意马尔可夫过程生成器
  - 包含**流分量**：对应流匹配
  - 包含**扩散分量**：对应扩散模型、分数匹配
  - 包含**跳跃分量**：对应离散扩散、CTMC

- **扩散模型**：GM的子集，固定扩散、只学漂移
- **去噪分数匹配**：GM在纯扩散下的特例
- **条件去噪概率模型**：GM的条件预测形式
- **流匹配**：GM在纯流、无噪声下的特例
- **离散扩散**：GM在离散状态空间下的特例

这意味着：
> 所有主流去噪类生成模型，都是生成器匹配的**特殊情况**。

### 10.6 实际意义
这种统一带来了直接的实践价值：

1. **混合模型**：可以在一个模型里同时学习流、扩散、跳跃，自动获得更好的采样与训练稳定性。
2. **统一训练**：所有模型都可以用同一个条件生成器匹配损失训练。
3. **任意模态**：同一套理论适用于连续数据、离散数据、流形、多模态数据。
4. **设计自由**：可以按需设计概率路径、生成器结构、损失函数。

换句话说，生成器匹配提供了一个**完整、统一、可扩展**的生成模型设计与训练范式。


## 附录（Appendix）
### A 补充证明（Additional proofs）
#### A.1 离散质量守恒（Discrete Mass Conservation）
**引理1（满足速率条件的科尔莫戈罗夫方程的概率质量函数解）**：考虑科尔莫戈罗夫方程（6.8）的解\(f_t(x)\)，其初始条件为\(f_0(x)=p(x)\)（其中\(p\)是概率质量函数（PMF）），且\(u_t(y,x)\)关于时间\(t\)是连续函数（\(C([0,1])\)）并满足速率条件（6.4）。则对于所有\(t \in [0,1]\)，\(f_t(x)\)均为概率质量函数（PMF）。

**证明**：设\(f_t(x)\)（\(t \in [0,1]\)）是科尔莫戈罗夫方程的解，定理12已保证其存在性和唯一性。\(f_t(x)\)为概率质量函数（PMF）的充要条件是满足：
\[f_t(x) \geq 0 \quad \text{且} \quad \sum _{x} f_t(x) = 1 \tag{A.1}\]

后一个条件可通过对科尔莫戈罗夫方程两侧求和证明：
\[
\frac{d}{dt} \sum _{x} f_t(x) = \sum _{x} \sum _{z} u_t(x, z) p_t(z) = 0
\]
其中第二个等式源于速率条件中\(\sum _{y} u_t(y, x) = 0\)。由于\(\sum _{x} f_0(x) = \sum _{x} p(x) = 1\)，因此对于所有\(t \in [0,1]\)，恒有\(\sum _{x} f_t(x) \equiv 1\)。

为证明对所有\(x \in S\)，\(f_t(x) \geq 0\)，我们利用动力系统的凸不变集相关结论。具体而言，Prüss等人（2010）的定理7.3.4指出：只要初始条件\(f_0 = p\)满足该约束（显然成立），且当\(w(z)\)位于约束边界（即\(w\)是概率质量函数（PMF）且对某些\(z \in S\)有\(w(z)=0\)）时，约束外法向量的内积非负（即\(\sum _{x,y} u_t(y, x) w(x) \delta(y, z) \geq 0\)），则解\(f_t(x)\)对所有\(t \in [0,1]\)和\(x \in S\)均满足\(f_t(x) \geq 0\)。验证该条件如下：
\[
\sum _{x,y} u_t(y, x) w(x) \delta(y, z) = \sum _{x} u_t(z, x) w(x) = \sum _{x \neq z} u_t(z, x) w(x) \geq 0
\]
其中第二个等式利用了\(w(z)=0\)，最后一个不等式源于速率条件（6.4）中\(z \neq x\)时\(u_t(z, x) \geq 0\)且对所有\(y\)有\(w(y) \geq 0\)。

**定理13（离散质量守恒）**：设\(u_t(y,x)\)关于时间\(t\)是连续函数（\(C([0,1))\)），且\(p_t(x)\)是关于时间\(t\)的一阶连续可微概率质量函数（PMF）（\(C^1([0,1))\)）。则以下条件等价：
1. \(p_t\)与\(u_t\)满足\(t \in [0,1)\)时的科尔莫戈罗夫方程（6.8），且\(u_t\)满足速率条件（6.4）；
2. 按（6.5）的定义，\(u_t\)生成\(t \in [0,1)\)时的\(p_t\)。

**证明**：首先假设条件2成立。此时，概率转移核\(p_{t+h|t}(y|x)\)满足（6.3）：
\[p_{t+h|t}(y|x) = \delta(y, x) + h u_t(y, x) + o(h) \tag{A.2}\]

根据全概率公式，边际分布\(p_{t+h}(y)\)可表示为：
\[p_{t+h}(y) = \sum _{x} p_{t+h|t}(y|x) p_t(x) \tag{A.3}\]

将（A.2）代入（A.3）并整理得：
\[
\frac{p_{t+h}(y) - p_t(y)}{h} = \sum _{x} u_t(y, x) p_t(x) + o(1)
\]
其中\(o(1) = o(h)/h\)，且当\(h \to 0\)时\(o(1) \to 0\)（符合\(o(h)\)的定义）。对\(h \to 0\)取极限，可得\((p_t, u_t)\)满足科尔莫戈罗夫方程（6.8）。

接下来证明\(u_t\)满足速率条件（6.4）：若存在\(y \neq x\)使得\(u_t(y, x) < 0\)，则由（A.2）可知，对于足够小的\(h > 0\)，\(p_{t+h|t}(y|x) < 0\)，这与\(p_{t+h|t}\)是概率核矛盾；若\(\sum _{y} u_t(y, x) = c \neq 0\)，则由（A.2）可知\(1 = \sum _{y} p_{t+h|t}(y|x) = 1 + h c + o(h)\)，对于足够小的\(h > 0\)，这同样矛盾。

反之，假设条件1成立，即\((u_t, p_t)\)满足初始条件为\(p_0 = p\)的科尔莫戈罗夫方程（6.8）。根据定理12，设\(p_{s|t}(y|x)\)是以下科尔莫戈罗夫方程的唯一解：
\[
\frac{d}{ds} p_{s|t}(y|x) = \sum _{z} u_t(y, z) p_{s|t}(z|x) \tag{A.4}
\]
其初始条件为\(p_{t|t}(y, x) = \delta(y, x)\)，其中\(0 \leq t \leq s < 1\)，且\(t\)和\(y\)为常数。由引理1可知，\(p_{s|t}(\cdot|x)\)是概率质量函数（PMF）。

\(\sum _{x} p_{s|t}(y|x) p(x)\)同样满足科尔莫戈罗夫方程，因为：
\[
\frac{d}{dt} \sum _{x} p_{s|t}(y|x) p(x) = \sum _{x} \left[ \sum _{z} u_t(y, z) p_{s|t}(z|x) \right] p(x) = \sum _{z} u_t(y, z) \left[ \sum _{x} p_{s|t}(z|x) p(x) \right] \tag{A.5}
\]
其初始条件为\(\sum _{x} p_{t|t}(y|x) p(x) = p(y)\)。由于科尔莫戈罗夫方程的解具有唯一性（定理12），因此\(\sum _{x} p_{s|t}(y|x) p(x) = p_s(y)\)，符合要求。

最后，对于\(0 \leq t \leq r \leq s < 1\)，转移核的半群性质\(\sum _{z} p_{s|r}(y|z) p_{r|t}(z|x) = p_{s|t}(y|x)\)，可通过将\(p_{r|t}\)作为时间\(r\)处的初始条件，重复（A.5）中的推导过程证明。综上，我们找到了生成\(p_t\)的转移核\(p_{t+h|t}\)，条件2成立。

#### A.2 流形边际化技巧（Manifold Marginalization Trick）
**定理10（流形边际化技巧）**：在假设2下，若\(u_t(x|x_1)\)是条件可积的且生成条件概率路径\(p_t(\cdot|x_1)\)，则边际速度场\(u_t(\cdot)\)生成边际概率路径\(p_t(\cdot)\)。

**证明**：为证明\(u_t(\cdot)\)生成\(p_t(\cdot)\)，我们需验证它们满足质量守恒定理的条件。首先，验证\(u_t(x)\)与\(p_t(x)\)满足连续性方程（5.1）：
\[
\frac{d}{dt} p_t(x) \stackrel{(i)}{=} \int_{\mathcal{M}} \frac{d}{dt} p_{t|1}(x|x_1) q(x_1) dvol_{x_1} \tag{A.6}
\]
\[
\stackrel{(ii)}{=} -\int_{\mathcal{M}} div_g\left[ u_t(x|x_1) p_{t|1}(x|x_1) \right] q(x_1) dvol_{x_1} \tag{A.7}
\]
\[
\stackrel{(i)}{=} -div_g \int_{\mathcal{M}} u_t(x|x_1) p_{t|1}(x|x_1) q(x_1) dvol_{x_1} \tag{A.8}
\]
\[
\stackrel{(iii)}{=} -div_g\left[ u_t(x) p_t(x) \right] \tag{A.9}
\]
其中：
- （i）通过莱布尼茨法则交换微分（\(\frac{d}{dt}\)和\(div_g\)）与积分运算，且\(p_{t|1}(x|x_1)\)和\(u_t(x|x_1)\)关于\(t\)是一阶连续可微的（\(C^1\)），同时\(q\)具有有界支撑或流形\(\mathcal{M}\)是紧的，确保积分可交换；
- （ii）利用\(u_t(\cdot|x_1)\)生成\(p_{t|1}(\cdot|x_1)\)这一事实及定理9；
- （iii）通过乘以并除以\(p_t(x)\)（由假设知其严格为正），结合（5.10）中\(u_t\)的定义推导得出。

最后，采用与定理3证明相同的论证，可证明\(u_t\)是可积且局部利普希茨连续的。

#### A.3 科尔莫戈罗夫方程的正则性假设（Regularity assumptions for KFE）
需说明的是，假设5在相对较弱的条件下成立，数学文献中存在大量关于不同场景下科尔莫戈罗夫方程（KFE）解\(p_t\)唯一性的研究。然而，据我们所知，目前尚无针对一般状态空间和马尔可夫过程的正则性假设相关已知结果，因此在此直接将其作为假设陈述。对于机器学习实践者而言，该假设在所有感兴趣的状态空间中均成立。为佐证这一点，我们列举数学文献中关于特定空间和马尔可夫过程类别的唯一性结果及对应的正则性假设：
1. \(\mathbb{R}^d\)和流形上的流：（Villani等人，2009，质量守恒公式，第15页）、（DiPerna和Lions，1989）、（Ambrosio，2004）；
2. \(\mathbb{R}^d\)和流形上的扩散过程：（Villani等人，2009，扩散定理，第16页）；
3. \(\mathbb{R}^d\)上的一般伊藤随机微分方程（Ito-SDEs）：（Figalli，2008，定理1.3和1.4）、（Kurtz，2011，推论1.3）、（Bogachev等人，2022）；
4. 离散状态空间：此处科尔莫戈罗夫方程（KFE）为线性常微分方程（ODE），在系数连续的假设下，其解具有唯一性（见定理13）。
