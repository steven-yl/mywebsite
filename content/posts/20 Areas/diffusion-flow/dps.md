---
title: "扩散后验采样（DPS）"
subtitle: ""
date: 2026-05-26T10:00:00+08:00
draft: false
authors: [Steven]
description: "扩散后验采样（DPS）"
summary: "扩散后验采样（DPS）"
tags: [diffusion/flow DPS]
categories: [diffusion/flow, DPS]
series: [diffusion/flow系列]
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---


### 一、文档定位与目标

本文档旨在对**扩散后验采样（Diffusion Posterior Sampling, DPS）** 进行系统性的技术解读。DPS 是 Chung 等人于 2023 年在 ICLR 上提出的一种利用预训练扩散模型作为生成先验来解决一般噪声（非线性）逆问题的方法。文档面向具备扩散模型基础知识的算法工程师与研究人员，内容覆盖 DPS 的数学原理、算法流程、代码实现、变体演进与局限性分析，兼顾理论深度与工程实用性。

### 二、整体知识结构

DPS 的核心思想可概括为：**在扩散模型的反向生成过程中，利用 Tweedie 公式从当前噪声样本估计出干净图像的期望，再通过该估计计算似然梯度，将其注入反向扩散步骤，从而引导生成过程符合观测约束**。

```mermaid
flowchart TD
    A[观测 y] --> B[DPS 采样循环]
    B --> C[t 从 T 到 1]
    C --> D[步骤1: 通过 Tweedie 公式\n从 x_t 估计 x_0_hat]
    D --> E[步骤2: 计算似然梯度\n∇_x_t log p(y|x_t) ≈ -∇_x_t ||A(x_0_hat) - y||^2]
    E --> F[步骤3: 更新采样步\nx_{t-1} = μ_t(x_t) + λ·∇_x_t log p(y|x_t) + g_t·ε]
    F --> G[t 循环继续]
    G --> H[输出后验样本 x_0]
```

本文档的各章节围绕上图中的知识架构展开：

- **第 1 章 背景与动机**：介绍逆问题的基础概念、扩散模型的关键原理，以及 DPS 要解决的核心技术挑战。
- **第 2 章 DPS 核心算法**：详细推导 Tweedie 公式、似然近似方法、后验采样迭代更新方程，并给出完整的算法流程图。
- **第 3 章 代码实现**：基于官方开源代码库，剖析 `dps_sampler.py`、`measurements.py`、`dps_utils.py` 等核心模块的函数构成与调用关系。
- **第 4 章 变体与演进**：梳理 DPS 的后续改进工作，包括 DPS-CM、CA-DPS、C-DPS、MAP-DPS、CL-DPS、DiffStateGrad、FlowDPS 等，分析各自的创新点与适用场景。
- **第 5 章 评估与对比**：汇总基准测试数据集、评价指标和主要方法的性能对比。
- **第 6 章 局限性分析**：总结 DPS 在计算效率、理论近似误差、多样性等方面存在的局限性。
- **附录**：提供术语表和公式速查表。

### 三、模块划分与关联说明

| 模块 | 功能 | 与 DPS 的关系 |
|------|------|--------------|
| 扩散模型（无条件） | 提供先验分布 \(p(x)\) | DPS 直接调用预训练好的无条件扩散模型作为先验 |
| 前向测量算子 \(A\) | 定义退化过程 \(y = A(x) + n\) | 用户根据实际问题提供，DPS 在运行时调用 |
| Tweedie 公式 | 从 \(x_t\) 估计 \(x_0\) | DPS 的核心估计器，连接扩散模型与测量约束 |
| 似然近似 | 近似计算 \(\nabla_{x_t} \log p(y\|x_t)\) | DPS 的创新贡献，解决扩散模型中似然的 intractability |
| 后验采样迭代 | 执行带引导的反向扩散步 | DPS 的整体采样框架 |

### 四、优缺点与适用场景对比

| 特性 | DPS | 投影式方法（如 MCG） | 谱域方法（如 DDRM） | 条件扩散模型 |
|------|-----|---------------------|---------------------|-------------|
| **是否支持噪声测量** | 是 | 否（噪声会被放大） | 有限支持 | 需要训练 |
| **是否支持非线性测量** | 是 | 否 | 否（需要 SVD） | 可训练但非通用 |
| **训练需求** | 无（使用预训练模型） | 无 | 无 | 需要大量配对数据 |
| **计算复杂度** | 高（需反向传播通过 U-Net） | 较低 | 较低 | 中等（推理时低） |
| **通用性** | 极高（任意前向模型） | 有限（线性/无噪） | 有限（需要 SVD） | 任务专属 |
| **多样性** | 高 | 中 | 中 | 任务相关 |

**选择建议**：
- **首选 DPS 的场景**：测量包含噪声、前向模型非线性或难以 SVD 分解、希望复用预训练扩散模型且无需训练。
- **替代方案场景**：线性无噪逆问题且需要极快推理时可选 DDRM；需要确定性重建而非采样多样性时考虑 PnP 方法；有充足配对数据且追求快速推理时训练条件扩散模型。

## 1 背景与动机

### 1.1 逆问题及其挑战

#### 1.1.1 问题形式化

逆问题的目标是从观测测量 \(y\) 中恢复原始信号 \(x\)，观测过程由前向测量算子 \(\mathcal{A}\) 和探测器噪声 \(n\) 共同决定：

\[
y = \mathcal{A}(x) + n, \quad n \sim \mathcal{N}(0, \sigma_y^2 I)
\]

其中 \(\mathcal{A}(\cdot): \mathbb{R}^n \to \mathbb{R}^m\)。由于逆问题通常是病态的（ill-posed），仅凭观测 \(y\) 不足以唯一确定 \(x\)，必须引入关于 \(x\) 的先验知识 \(p(x)\)。从后验分布 \(p(x|y) \propto p(y|x) \cdot p(x)\) 进行采样是理论上最合理的求解方式。

#### 1.1.2 现有方法的局限

在 DPS 提出之前，利用扩散模型求解逆问题的主要方法可分为两类：

**（1）投影式方法**：如 Song 等人（2021b）和 Chung 等人（2022b）的工作，将生成样本投影到测量子空间以保证测量一致性。该方法在以下情况下严重失效：测量中存在噪声（噪声在生成过程中被放大）、测量过程是非线性的。

**（2）谱域方法**：如 Kawar 等人（2021, 2022）的工作，在谱域运行扩散过程，通过 SVD 将测量域噪声映射到谱域。该方法受限于 SVD 的计算成本，当正向模型复杂时计算难以承受。例如，Kawar 2022 仅考虑了可分离高斯核的去模糊问题。

DPS 的核心动机正是在于设计一种**无需 SVD、无需严格投影、能够处理噪声和非线性测量的通用方法**。

### 1.2 扩散模型预备知识

#### 1.2.1 前向扩散过程

扩散模型定义了一个逐步向数据添加噪声的过程。给定数据分布 \(p(x_0)\)，扩散前向过程可表示为：

\[
x_t = \alpha_t x_0 + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

其中 \(\alpha_t\) 和 \(\sigma_t\) 是预先定义的信噪比参数，满足 \(x_t\) 的边缘分布：\(p(x_t|x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)\)。

在实际实现中，常用的是方差保持（VP）型扩散模型，其中 \(\alpha_t\) 随 t 增大而减小，\(\sigma_t\) 随之增大。

#### 1.2.2 反向扩散与分数匹配

反向过程的目的是从噪声 \(x_T \sim \mathcal{N}(0, \sigma_T^2 I)\) 逐步去噪生成 \(x_0\)，这依赖于对数密度函数的梯度 \(\nabla_{x_t} \log p_t(x_t)\)，称为 **Stein 分数**（score）。

扩散模型的训练目标正是使用分数匹配（score matching）技术，训练一个神经网络 \(s_\theta(x, t)\) 来近似真实的分数函数 \(\nabla_{x_t} \log p_t(x_t)\)。

#### 1.2.3 条件扩散与后验采样

若要从后验分布 \(p(x|y)\) 中采样，需要获取条件分数 \(\nabla_{x_t} \log p(x_t|y)\)。根据贝叶斯公式：

\[
\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t)
\]

第一项由预训练的无条件扩散模型 \(s_\theta(x_t, t)\) 提供。第二项似然梯度的计算却异常困难，因为 \(p(y|x_t)\) 涉及对所有可能 \(x_0\) 的积分：

\[
p(y|x_t) = \int p(y|x_0) p(x_0|x_t) dx_0
\]

这正是扩散模型求解逆问题的核心障碍，也是 DPS 所要解决的关键问题。

### 1.3 DPS 的核心思想与创新

DPS 的关键洞察在于：通过 **Tweedie 公式**从 \(x_t\) 估计出 \(x_0\) 的条件期望 \(\hat{x}_0 = \mathbb{E}[x_0|x_t]\)，然后用该估计来近似似然梯度：

\[
\nabla_{x_t} \log p(y|x_t) \approx -\nabla_{x_t} \|\mathcal{A}(\hat{x}_0) - y\|^2
\]

这一近似使得 DPS 能够：处理**任意**前向模型（包括非线性和带噪声的）、无需对前向模型进行 SVD 分解、在反向扩散过程中**自然集成**测量约束。

## 2 DPS 核心算法

### 2.1 Tweedie 公式与干净图像估计

#### 2.1.1 Tweedie 公式的数学形式

Tweedie 公式是 DPS 算法的理论基础。对于高斯扩散过程，给定噪声观测 \(x_t\)，原始数据 \(x_0\) 的条件期望可表示为：

\[
\mathbb{E}[x_0|x_t] = \frac{x_t + \sigma_t^2 \nabla_{x_t} \log p_t(x_t)}{\alpha_t}
\]

此公式建立了**分数函数**与**去噪估计**之间的直接关联：分数指向高密度区域的方向，\(\sigma_t^2\) 对修正幅度进行缩放（噪声越大，越依赖先验；噪声越小，越依赖观测）。

#### 2.1.2 \(\hat{x}_0\) 的估计

在 DPS 中，我们使用预训练的分数网络 \(s_\theta(x_t, t)\) 来近似 \(\nabla_{x_t} \log p_t(x_t)\)，从而得到：

\[
\hat{x}_0 := \mathbb{E}[x_0|x_t] = \frac{x_t + \sigma_t^2 s_\theta(x_t, t)}{\alpha_t}
\]

**直观理解**：在当前反向扩散步的 \(x_t\) 处，Tweedie 公式给出了一步“跳跃”到干净图像的预测。这个预测越准确，后续的测量约束计算就越可靠。

### 2.2 似然梯度近似

#### 2.2.1 线性测量情形

对于线性前向模型 \(A\)，若测量噪声为高斯分布，后验分布可解析求解：

\[
p(y|x_t) = \mathcal{N}(y; \bar{\mu}_t, \bar{\Sigma}_t)
\]
\[
\bar{\mu}_t = \frac{A x_t}{\alpha_t}, \quad \bar{\Sigma}_t = \frac{\sigma_t^2}{\alpha_t^2} A A^\top + \sigma_y^2 I
\]
\[
\nabla_{x_t} \log p(y|x_t) = \frac{1}{\alpha_t} A^\top \bar{\Sigma}_t^{-1} (y - \bar{\mu}_t)
\]

但精确求解涉及矩阵求逆，计算开销极大。DPS 采用更高效的近似：

\[
\nabla_{x_t} \log p(y|x_t) \approx -\frac{1}{\sigma_y^2} \cdot \nabla_{x_t} \|A\hat{x}_0 - y\|^2
\]

#### 2.2.2 通用情形的近似推导

对于任意（可能非线性）的前向模型 \(\mathcal{A}\)，DPS 做出以下简化假设：

1. 使用 Tweedie 估计 \(\hat{x}_0\) 代替真实的 \(x_0\) 来“冻结”似然条件；
2. 近似 \(p(y|x_t) \approx p(y|\hat{x}_0)\)，其中 \(\hat{x}_0\) 是 \(x_t\) 的函数；
3. 假设似然函数为高斯形式 \(p(y|x) \propto \exp\left(-\frac{1}{2\sigma_y^2}\|\mathcal{A}(x) - y\|^2\right)\)。

由此得到近似：

\[
\nabla_{x_t} \log p(y|x_t) \approx -\frac{1}{\sigma_y^2} \nabla_{x_t} \|\mathcal{A}(\hat{x}_0) - y\|^2
\]

值得强调的是，这一近似对 DPS 的成功至关重要，也是 DPS 能够通用性地处理各种逆问题的关键。后续的 DPS-CM 等方法正是发现早期阶段直接使用上述近似可能引入高频误差，从而提出了改进方案。

#### 2.2.3 计算实现细节

在代码层面，上述近似通过**自动微分**实现。具体流程为：

- **前向**: \(\hat{x}_0 \leftarrow (x_t + \sigma_t^2 s_\theta(x_t, t)) / \alpha_t\)
- **前向**: \(pred \leftarrow \mathcal{A}(\hat{x}_0)\)
- **损失**: \(loss \leftarrow \|pred - y\|^2\)（可选加权或归一化）
- **反向**: \(\nabla_{x_t} loss \leftarrow \text{grad}(loss, x_t)\)

### 2.3 后验采样更新方程

#### 2.3.1 DDPM 型反向过程

在 DDPM 采样框架下，不含条件引导的反向扩散更新为：

\[
x_{t-1} = \mu_t(x_t) + g_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

其中 \(\mu_t(x_t)\) 是给定 \(x_t\) 时对 \(x_{t-1}\) 的条件期望，\(g_t\) 是预设的噪声尺度。

#### 2.3.2 DPS 的条件采样更新

DPS 将似然梯度以一定强度注入反向扩散步骤：

\[
x_{t-1} = \mu_t(x_t) + \underbrace{\lambda_t \nabla_{x_t} \log p(y|x_t)}_{\text{测量引导项}} + g_t \epsilon
\]

其中 \(\lambda_t\) 是控制测量引导强度的步长参数。

**步长设计**：\(\lambda_t\) 的选择对 DPS 的性能有显著影响。论文中的典型设置是固定值 \(\lambda \in [0.1, 1.0]\)，或按时间步衰减。需要注意的是，\(\lambda\) 太大可能导致样本偏离数据流形，太小则测量约束不足。

#### 2.3.3 完整 DPS 算法

```
算法：扩散后验采样（DPS）

输入：
  - y: 观测测量值
  - s_θ(x, t): 预训练的分数网络（无条件）
  - A: 前向测量算子
  - T: 总扩散步数
  - λ: 引导步长
  - σ_y: 测量噪声标准差

输出：
  - x_0: 从后验分布 p(x|y) 中采样的样本

1. 初始化 x_T ~ N(0, σ_T^2 I)
2. For t = T, T-1, ..., 1:
   a.  计算 α_t, σ_t（根据预设的扩散噪声调度）
   b.  通过 Tweedie 公式估计干净图像：
        x_0_hat ← (x_t + σ_t^2 * s_θ(x_t, t)) / α_t
   c.  近似似然梯度：
        grad_ll ← ∇_{x_t} ||A(x_0_hat) - y||^2
   d.  执行带引导的反向扩散更新：
        x_{t-1} ← μ_t(x_t) + λ · grad_ll + g_t · ε
3. 返回 x_0
```

### 2.4 算法流程图

```mermaid
flowchart TD
    S([开始]) --> I[输入观测y\n预训练分数网络 s_θ\n前向模型 A]
    I --> N[初始化 x_T ~ N(0, σ_T^2 I)]
    N --> L{已采样所有t?}
    L -- 否 --> F[t = T to 1]
    F --> T1[通过Tweedie公式\n估计 x_0_hat]
    T1 --> T2[计算前向预测\nA(x_0_hat)]
    T2 --> T3[计算似然损失\nloss = ||A(x_0_hat) - y||^2]
    T3 --> T4[反向传播\n计算 ∇_x_t loss]
    T4 --> T5[反向扩散更新\nx_{t-1} = μ_t + λ·∇_x_t loss + g_t·ε]
    T5 --> L
    L -- 是 --> O([输出后验样本 x_0])
```

## 3 代码实现

### 3.1 代码库结构

DPS 的官方代码库位于 `https://github.com/DPS2022/diffusion-posterior-sampling`，核心目录结构如下：

```
diffusion-posterior-sampling/
├── dps_sampler.py          # DPS 主采样器
├── measurements.py         # 前向测量算子定义
├── dps_utils.py           # 辅助工具函数
├── models/                # 预训练扩散模型加载
├── configs/               # 任务配置文件
└── scripts/               # 运行脚本
```

### 3.2 `dps_sampler.py` 核心类与函数

本模块实现 DPS 的核心采样逻辑。

#### 3.2.1 类定义：`DPS`

| 方法 | 参数 | 功能 |
|------|------|------|
| `__init__()` | `model, measurement, y, sigma_y, eta, lambda_` | 初始化采样器 |
| `sample()` | `x_init` | 执行完整采样过程 |
| `denoise_step()` | `x_t, t` | 执行单步反向去噪 |
| `get_clean_estimate()` | `x_t, t` | 通过 Tweedie 估计 x_0_hat |
| `compute_grad_ll()` | `x_0_hat, t` | 计算似然梯度 |
| `update_step()` | `x_t, t, grad_ll` | 执行条件采样更新 |

#### 3.2.2 `sample()` 流程说明

```python
def sample(self, x_init=None):
    """
    DPS主采样循环
    
    参数:
        x_init: 初始噪声样本，若为None则从标准正态采样
    
    返回:
        x_0: 后验样本
    """
    # 1. 初始化 x_T
    x = x_init if x_init is not None else torch.randn_like(...)
    
    # 2. 时间步循环
    for i, t in enumerate(self.timesteps):
        # 2.1 估计干净图像
        x_0_hat = self.get_clean_estimate(x, t)
        
        # 2.2 计算似然梯度
        grad_ll = self.compute_grad_ll(x_0_hat, t)
        
        # 2.3 更新采样步
        x = self.update_step(x, t, grad_ll)
    
    return x
```

#### 3.2.3 `get_clean_estimate()` 核心实现

```python
def get_clean_estimate(self, x_t, t):
    """
    通过 Tweedie 公式估计 x_0_hat
    
    x_0_hat = (x_t + σ_t^2 * s_θ(x_t, t)) / α_t
    """
    alpha_t = self.alpha[t]
    sigma_t = self.sigma[t]
    
    # 获取分数网络预测（即噪声预测或直接分数输出）
    score = self.model(x_t, t)
    
    # Tweedie 公式
    x_0_hat = (x_t + sigma_t**2 * score) / alpha_t
    
    return x_0_hat
```

#### 3.2.4 `compute_grad_ll()` 核心实现

```python
def compute_grad_ll(self, x_0_hat, t):
    """
    通过自动微分计算似然梯度
    
    ∇_x_t log p(y|x_t) ≈ -∇_x_t ||A(x_0_hat) - y||^2 / σ_y^2
    """
    # 确保 x_0_hat 需要梯度
    x_0_hat.requires_grad_(True)
    
    # 前向：计算测量预测
    Ax = self.measurement.A(x_0_hat)
    
    # 计算损失（通常为 MSE）
    loss = torch.norm(Ax - self.y) ** 2
    
    # 反向传播计算梯度
    grad = torch.autograd.grad(loss, x_0_hat)[0]
    
    # 在 DPS 中，梯度的符号和缩放可能调整
    return -grad / self.sigma_y**2
```

#### 3.2.5 `update_step()` 核心实现

```python
def update_step(self, x_t, t, grad_ll):
    """
    DDPM 反向扩散更新 + 测量引导
    
    x_{t-1} = μ_t(x_t) + λ·∇_x_t log p(y|x_t) + g_t·ε
    """
    # 获取当前步的扩散参数
    alpha_t = self.alpha[t]
    sigma_t = self.sigma[t]
    alpha_prev = self.alpha[t-1] if t > 0 else 1.0
    
    # 计算 μ_t(x_t)
    # 对于 DDPM: μ_t = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t))
    # 其中 ε_θ 是预测噪声，β_t = 1 - α_t/α_{t-1}
    eps = self.model(x_t, t)
    mu = (x_t - (beta_t / (1 - alpha_cumprod[t]).sqrt()) * eps) / alpha_t.sqrt()
    
    # 计算噪声尺度 g_t
    g_t = sigma_t
    
    # 添加引导项并采样
    x_prev = mu + self.lambda_ * grad_ll + g_t * torch.randn_like(x_t)
    
    return x_prev
```

### 3.3 `measurements.py` 前向测量算子

本模块定义了各种逆问题的前向测量算子。

#### 3.3.1 线性测量类：`LinearOperator`

| 方法 | 参数 | 功能 |
|------|------|------|
| `A(x)` | `x` | 前向测量：y = A(x) |
| `A_adjoint(y)` | `y` | 伴随算子（若需要） |

#### 3.3.2 非线性测量类：`NonlinearOperator`

| 方法 | 参数 | 功能 |
|------|------|------|
| `A(x)` | `x` | 非线性前向测量 |
| `A_adjoint(y)` | `y` | 伴随算子（可选） |

#### 3.3.3 支持的逆问题类型

| 类型 | 前向算子 | 描述 |
|------|----------|------|
| 超分辨率 | \(A(x) = (x \downarrow k) + noise\) | 下采样加噪声 |
| 去模糊 | \(A(x) = k * x + noise\) | 卷积模糊核加噪声 |
| 图像修复 | \(A(x) = M \odot x + noise\) | 二进制掩码加噪声 |
| 压缩感知 | \(A(x) = \Phi x + noise\) | 随机测量矩阵 |
| 相位恢复 | \(A(x) = \|\mathcal{F}(x)\|^2\) | 傅里叶幅度测量（非线性） |
| 非均匀去模糊 | \(A(x) = K(x) + noise\) | 空间变化模糊核 |

非线性前向算子的核心用例是**傅里叶相位恢复**：输入图像仅保留傅里叶变换的幅度信息，丢失相位。DPS 可以借助 \(\hat{x}_0\) 在每一步计算幅度损失，引导扩散过程恢复到正确的相位。

### 3.4 完整可运行示例

```python
import torch
from dps_sampler import DPS
from measurements import SuperResolution, GaussianBlur, Inpainting

# 1. 定义逆问题
task = "super_resolution"  # 可选: "super_resolution", "deblur", "inpainting"
if task == "super_resolution":
    measurement = SuperResolution(scale_factor=4, noise_sigma=0.05)
elif task == "deblur":
    measurement = GaussianBlur(kernel_size=9, sigma=3.0, noise_sigma=0.01)
elif task == "inpainting":
    mask = torch.bernoulli(torch.full((1, 3, 256, 256), 0.3))
    measurement = Inpainting(mask=mask, noise_sigma=0.05)

# 2. 加载预训练的扩散模型
# 以 DDPM 类型为例
model = load_pretrained_diffusion("ffhq_256")  # 需用户实现

# 3. 生成观测 y（在真实应用中 y 已知，此处模拟）
x_true = load_ground_truth_image()
y = measurement.A(x_true)

# 4. 初始化 DPS 采样器
sampler = DPS(
    model=model,          # 预训练扩散模型
    measurement=measurement,
    y=y,                  # 观测测量
    sigma_y=0.05,         # 测量噪声标准差
    eta=1.0,              # DDIM 参数（η=1 为 DDPM）
    lambda_=0.5,          # 引导步长
    timesteps=1000        # 总采样步数
)

# 5. 执行后验采样
x_recon = sampler.sample()

# 6. 评估重建质量
psnr = compute_psnr(x_recon, x_true)
ssim = compute_ssim(x_recon, x_true)
print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
```

## 4 变体与演进

### 4.1 DPS-CM（Crafted Measurement）

**提出时间与出处**：2024-2025 年，Shijie Zhou 等人

**核心创新**：DPS-CM 发现，DPS 在早期阶段直接使用噪声测量 \(y\) 来形成后验估计，可能在早期引入高频信息，导致累积误差。为解决这一问题，DPS-CM 引入了一个**经过反去噪过程构建的测量**（crafted measurement），而非使用扩散前向过程构建的测量来形成后验估计。

**关键改进**：

- 在早期采样步中，使用"精心构建"的噪声测量替代原始观测，更好地与扩散先验对齐
- 有效缓解了因累积后验估计误差导致的先验与测量之间不对齐问题

**性能提升**：在泊松噪声去模糊、非线性去模糊、超分辨率和图像修复等任务上均有改进。

### 4.2 CA-DPS（Covariance-Aware DPS）

**提出时间与出处**：2024 年，Yismaw 等人

**核心创新**：DPS 忽略了对数似然的协方差信息，CA-DPS 在似然函数中加入了协方差修正项，同时避免反向传播通过扩散模型。该方法在假设高斯先验的前提下，能够同时实现协方差修正和计算效率。

**关键改进**：

- 协方差修正：使用精确的协方差矩阵替代 DPS 中的标量缩放近似
- 计算效率：引入矩阵分解和求逆的高效实现方案

**性能提升**：在某些线性逆问题上实现更好的收敛性，在自然图像数据集上性能优于 DPS。

### 4.3 C-DPS（Coupled Data and Measurement Space DPS）

**核心创新**：C-DPS 将数据空间和测量空间耦合在一起扩展扩散模型，无需约束调整或似然近似。通过在联合空间中进行扩散采样，自然地将测量信息融入生成过程。

**优势**：消除了 DPS 中关键的超参数 \(\lambda_t\) 调整需求，提供了更稳定、更自动化的条件采样框架。

### 4.4 MAP-DPS（Maximizing a Posterior DPS）

**提出时间与出处**：2025 年，Tongda Xu 等人

**核心理论洞察**：Xu 等人对 DPS 的工作机制进行了重要重新解读。通过实验验证：

1. DPS 的条件分数估计与正确训练的条件扩散模型的分数存在显著差异
2. DPS 的条件分数估计的均值显著偏离零，无法构成有效的分数估计
3. DPS 生成高质量样本但多样性显著降低

基于这些发现，Xu 等人提出：**DPS 实际上并非执行后验采样，而是更接近于最大化后验（MAP）估计**。这一重新解读对整个 DPS 研究界产生了深远影响。

**提出的改进**：

- **明确最大化后验**：通过多步梯度上升和投影来显式最大化后验
- **轻量级条件分数估计**：使用仅 100 张图像训练 8 GPU 小时的条件分数估计器
- **源代码如下**：`https://github.com/tongdaxu/Rethinking-Diffusion-Posterior-Sampling`

**性能提升**：两项改进均显著提升了 DPS 的性能。

### 4.5 CL-DPS（Contrastive Learning DPS）

**核心创新**：CL-DPS 解决的是**盲逆问题**（测量算子未知）场景。通过对比学习训练一个辅助的深度神经网络作为似然估计器，无需事先知道测量算子即可估计 \(p(y|x)\)。

**关键技术**：

- 使用改进版 MoCo 训练辅助网络
- 引入重叠分块（overlapped patch-wise）推理方法以提高似然估计精度

**应用场景**：非盲 DPS 无法处理的盲逆问题，如旋转去模糊等未知测量算子的非线性逆问题。

### 4.6 DiffStateGrad（Diffusion State-Guided Projected Gradient）

**核心创新**：DiffStateGrad 将测量梯度投影到扩散状态空间，不同于以往投影到数据流形 \(\mathcal{M}_0\) 的做法，而是投影到噪声流形 \(\mathcal{M}_t\) 上。这样做的好处是保持采样状态在正确的噪声水平流形上，避免了状态偏离期望分布的问题。

### 4.7 FlowDPS

**核心创新**：FlowDPS 将 DPS 思想扩展到基于流的扩散模型（如 Stable Diffusion 3.0），验证了 DPS 框架在流模型上的通用性。

### 4.8 变体对比总结

| 变体 | 核心改进 | 适用场景 | 代码仓库 |
|------|----------|----------|----------|
| **DPS** | 基线 | 通用逆问题 | [DPS2022/diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling) |
| **DPS-CM** | 精心构建测量 | 带噪声逆问题 | [sjz5202/DPS-CM](https://github.com/sjz5202/DPS-CM) |
| **CA-DPS** | 协方差修正 | 线性逆问题 | [CSIPlab/CoDPS](https://github.com/CSIPlab/CoDPS) |
| **MAP-DPS** | 重新解读 + 多步梯度 | 高分辨率任务 | [tongdaxu/Rethinking-Diffusion-Posterior-Sampling](https://github.com/tongdaxu/Rethinking-Diffusion-Posterior-Sampling) |
| **CL-DPS** | 对比学习估计 | 盲逆问题 | [cldps/cldps](https://github.com/cldps/cldps) |

## 5 评估与对比

### 5.1 基准数据集

DPS 及其变体通常在以下数据集上评估：

- **FFHQ**（Flickr-Faces-HQ）：人脸图像，分辨率 256×256/512×512
- **ImageNet**：自然图像，常用于线性及非线性逆问题测试
- **CelebA**：人脸属性数据集
- **LSUN-Bedroom/Church**：室内/室外场景

### 5.2 评价指标

| 指标 | 全称 | 描述 | 范围 |
|------|------|------|------|
| **PSNR** | Peak Signal-to-Noise Ratio | 峰值信噪比，衡量像素级重建精度 | 越高越好（dB） |
| **SSIM** | Structural Similarity Index Measure | 结构相似性，衡量感知质量 | [0,1]，越高越好 |
| **LPIPS** | Learned Perceptual Image Patch Similarity | 基于深度特征的感知相似度 | 越低越好 |
| **FID** | Fréchet Inception Distance | 生成质量多样性评估 | 越低越好 |

为综合评估采样质量，部分工作采用组合评分：\(\text{PSNR}/40 + \text{SSIM}\)。

### 5.3 主要方法性能对比

| 方法 | 是否需要训练 | 噪声处理 | 非线性处理 | 内存/计算效率 | 典型性能（PSNR） |
|------|-------------|----------|------------|--------------|-----------------|
| DPS | 否 | ✓ | ✓ | 高（NFE ~1000） | 中高 |
| DDRM | 否 | 有限 | 否 | 中 | 中 |
| GDM | 否 | 否 | 否 | 低 | 低 |
| MCG | 否 | 否 | 否 | 高 | 低 |
| DPS-MO | 否 | ✓ | 有限 | 高（NFE ~100） | 更高（新 SOTA） |

**注释**：
- NFE（Number of Function Evaluations，函数评估次数）。标准 DPS 需 1000 步采样，计算成本较高。
- DPS-MO 通过测量优化，在 FFHQ 256 数据集上达到 28.71 dB PSNR，仅需 100 步 NFE，而 DPS 等现有方法需要 1000–4000 步才能达到同等性能。

## 6 局限性分析

### 6.1 计算效率问题

DPS 最显著的局限性在于计算开销。每个反向扩散步都需要：

1. 通过扩散模型的前向传播来获取分数
2. 通过 Tweedie 公式计算 \(\hat{x}_0\)
3. 通过测量算子的前向传播
4. 损失计算后通过自动微分获取梯度

这一系列操作使得 DPS 的计算成本显著高于 DDRM 或 GDM 等方法。采样步数通常需要 1000 步，每步包含一次 U-Net 推理和一次梯度计算。DPS-CM 等方法尝试缓解这一问题，但 DPS 的高计算成本仍是其广泛应用的主要障碍之一。

### 6.2 理论近似误差

DPS 的核心近似 \(\nabla_{x_t} \log p(y|x_t) \approx -\nabla_{x_t} \|A(\hat{x}_0) - y\|^2\) 在理论上存在以下问题：

- **期望估计的偏差**：使用 \(\hat{x}_0 = \mathbb{E}[x_0|x_t]\) 代替真实的 \(x_0\)，对于多峰分布而言，期望可能位于密度较低的区域，导致严重偏差
- **模态遗漏**：当后验分布存在多个模态时，期望估计可能落在模态之间，无法代表任何真实样本
- **MAP 性质**：MAP-DPS 的研究表明，DPS 实际上可能更接近于 MAP 估计而非真正的后验采样，这意味着采样结果的多样性可能被显著低估

### 6.3 多样性与模式坍塌

由于 DPS 实质上更接近 MAP 优化，其采样多样性存在固有限制。当测量约束强时，DPS 往往产生确定性输出而非多样化采样结果。这对于需要不确定性量化的应用场景（如医疗影像重建）尤为不利。

## 附录

### A. 术语表

| 术语 | 英文 | 释义 |
|------|------|------|
| **扩散模型** | Diffusion Model | 通过逐步添加噪声并学习逆向去噪过程来生成数据的生成模型 |
| **Stein 分数** | Stein Score | 概率密度对数的梯度 \(\nabla_x \log p(x)\) |
| **分数匹配** | Score Matching | 训练模型近似真实分数的技术 |
| **Tweedie 公式** | Tweedie‘s Formula | 通过噪声观测估计原始数据条件期望的统计公式 |
| **逆问题** | Inverse Problem | 从观测数据中反推原始信号的病态问题 |
| **前向模型** | Forward Model | 定义测量过程的算子 \(\mathcal{A}\) |
| **病态问题** | Ill-posed Problem | 解不唯一或不稳定的数学问题 |
| **非线性逆问题** | Nonlinear Inverse Problem | 前向模型为非线性的逆问题 |
| **盲逆问题** | Blind Inverse Problem | 测量算子未知的逆问题 |

### B. 重要数学公式速查

| 公式 | 含义 |
|------|------|
| \(y = \mathcal{A}(x) + n\) | 逆问题的观测模型 |
| \(p(x|y) \propto p(y|x) \cdot p(x)\) | 贝叶斯公式 |
| \(x_t = \alpha_t x_0 + \sigma_t \epsilon\) | 扩散前向过程 |
| \(\mathbb{E}[x_0|x_t] = (x_t + \sigma_t^2 \nabla_{x_t} \log p(x_t)) / \alpha_t\) | Tweedie 公式 |
| \(\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y|x_t)\) | 条件分数分解 |
| \(\hat{x}_0 = (x_t + \sigma_t^2 s_\theta(x_t, t)) / \alpha_t\) | DPS 中的干净图像估计 |
| \(\nabla_{x_t} \log p(y|x_t) \approx -\nabla_{x_t} \|A(\hat{x}_0) - y\|^2\) | DPS 似然梯度近似 |
| \(x_{t-1} = \mu_t(x_t) + \lambda \nabla_{x_t} \log p(y|x_t) + g_t \epsilon\) | DPS 采样更新 |

### C. 资源参考

- **原始 DPS 论文**：Chung et al., “Diffusion Posterior Sampling for General Noisy Inverse Problems”, ICLR 2023
- **官方代码**：https://github.com/DPS2022/diffusion-posterior-sampling
- **DPS-CM 代码**：https://github.com/sjz5202/DPS-CM
- **MAP-DPS 代码**：https://github.com/tongdaxu/Rethinking-Diffusion-Posterior-Sampling
- **CA-DPS 代码**：https://github.com/CSIPlab/CoDPS
- **CL-DPS 代码**：https://github.com/cldps/cldps