---
title: "扩散模型中的噪声调度（Noise Schedule）—— 完整理论笔记"
subtitle: ""
date: 2026-03-27T00:00:00+08:00
draft: false
authors: [Steven]
description: "系统整理扩散模型中 VP 与 VE 两类噪声调度，包含公式、动机、数学性质与可运行代码示例。"
summary: "从线性、余弦到 EDM、SNR-based 调度，结合采样器与时间嵌入给出完整工程实践指南。"

tags: [diffusion/flow, noise schedule, DDPM, EDM]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 11
series_weight: 11

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""

markmap:
    initialExpandLevel: 30

---


## 一、调度的作用

噪声调度定义了前向扩散过程中，在每个时间步（或连续时间）对数据添加的噪声量。它决定了：
- 训练时网络所见的噪声强度分布；
- 采样时逆向过程的路径形状；
- 最终生成质量与采样效率。

一个好的调度应使网络在训练时**所有噪声水平上都有足够的学习信号**，并且采样时能**高效地逼近数据分布**。

---

## 二、两种核心参数化

扩散模型的前向过程有两种等价的主流参数化方式，它们通过数学变换可以互相转换。

### 1. 方差保持（Variance Preserving，VP）
- **加噪公式**：$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$
- **性质**：$\mathbb{E}[\|x_t\|^2] = \mathbb{E}[\|x_0\|^2]$（数据方差保持恒定）
- **典型调度**：DDPM 线性、余弦

### 2. 方差爆炸（Variance Exploding，VE）
- **加噪公式**：$x_t = x_0 + \sigma_t\,\varepsilon$
- **性质**：$x_t$ 的方差随 $t$ 单调增长至无穷（若不归一化）
- **典型调度**：NCSN 几何序列、EDM 连续调度

- **转换关系**：$\sigma_t = \sqrt{1/\bar\alpha_t - 1}$，两者完全等价，只是符号习惯不同。
- 代码示例：
  ```python
  import numpy as np
  import torch
  import matplotlib.pyplot as plt

  def alpha_bar_to_sigma(alpha_bar):
      """VP 参数 ᾱ 转换为 VE 参数 σ"""
      return np.sqrt(1/alpha_bar - 1)

  def sigma_to_alpha_bar(sigma):
      """VE 参数 σ 转换为 VP 参数 ᾱ"""
      return 1 / (1 + sigma**2)
  ``` 

---

## 三、VP 类调度详解

这类调度以离散时间步 $t=1..T$ 给出噪声方差 $\beta_t$，再计算累积信号保持率 $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$。  
**核心思想**：控制信号与噪声的混合比例，使得 $x_t$ 的方差保持不变（$\mathbb{E}[\|x_t\|^2] = \mathbb{E}[\|x_0\|^2]$）。

### 3.1 线性调度（DDPM 原始）(Ho et al., 2020) <!-- markmap: foldAll -->

#### **定义与公式**
- DDPM 中，$\beta_t$ 从 $\beta_{\min}$ 到 $\beta_{\max}$ 线性增长：
  $$
  \beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min}), \quad t=1,\dots,T
  $$
  典型参数：$T=1000,\ \beta_{\min}=10^{-4},\ \beta_{\max}=0.02$。  
  由此计算 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod_{s=1}^t \alpha_s$。

#### **设计动机**
线性调度是扩散模型最早的调度形式，其直观性在于噪声方差的增长速率是恒定的。早期阶段 $\beta$ 很小，图像变化缓慢，符合“逐步破坏数据”的直觉；后期 $\beta$ 较大，使数据快速变成纯噪声。

#### **数学特性**
- 信噪比 $\text{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$ 在早期下降缓慢，中期加速下降，末期趋于 0。
- 由于 $\beta_t$ 线性增长，累积乘积 $\bar\alpha_t$ 在后期呈指数级衰减。

#### **代码示例**
```python
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """生成线性 β 序列"""
    betas = torch.linspace(beta_start, beta_end, T)
    return betas

def compute_alpha_bar(betas):
    """从 β 计算 ᾱ"""
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar

T = 1000
betas = linear_beta_schedule(T)
alpha_bar = compute_alpha_bar(betas)
sigmas = alpha_bar_to_sigma(alpha_bar.numpy())

# 可视化
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas.numpy())
plt.title(r'$\beta_t$ (linear)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar.numpy())
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas)
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 3.2 余弦调度（Improved DDPM）(Nichol & Dhariwal, 2021)  <!-- markmap: foldAll -->

#### **定义与公式**
- 改进的 DDPM 不再直接定义 $\beta_t$，而是定义 $\bar\alpha_t$ 的余弦形式：
  $$
  \bar\alpha_t = \frac{\cos^2\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)}{\cos^2\left(\frac{s}{1+s}\cdot\frac{\pi}{2}\right)}, \quad s=0.008
  $$
  然后通过 $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$ 反推 $\beta_t$。其中 $s$ 是一个很小的偏移量，防止 $t=0$ 时 $\bar\alpha_0$ 恰好为 1 导致数值问题。

#### **设计动机**  
线性调度在中间阶段信噪比下降过快，使得网络难以学习中等噪声水平的去噪。余弦调度通过让 $\bar\alpha_t$ 在中间阶段缓慢下降，保持了较高的信噪比，使网络有更多机会学习有意义的去噪映射。

#### **数学特性**  
- 在 $t=0$ 附近 $\bar\alpha_0\approx 1$，在 $t=T$ 时 $\bar\alpha_T\approx 0$。
- 信噪比在对数域大致呈线性下降，但中间段比线性调度更平缓。
- $\beta_t$ 在两端较小，中间稍大，呈现先增后减的形状（非单调）。

#### **代码示例**
```python
def cosine_alpha_bar(t, s=0.008):
    """连续 t∈[0,1] 对应的 ᾱ(t)"""
    return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2

def cosine_beta_schedule(T, s=0.008, max_beta=0.999):
    """生成余弦 β 序列"""
    t = np.linspace(0, 1, T+1)  # 包含 0 和 1
    alpha_bar_t = cosine_alpha_bar(t, s)
    # 计算 β_t = 1 - ᾱ_t / ᾱ_{t-1}
    betas = 1 - alpha_bar_t[1:] / alpha_bar_t[:-1]
    betas = np.clip(betas, 0, max_beta)
    return torch.tensor(betas, dtype=torch.float32)

betas_cos = cosine_beta_schedule(T)
alpha_bar_cos = compute_alpha_bar(betas_cos)
sigmas_cos = alpha_bar_to_sigma(alpha_bar_cos.numpy())

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_cos.numpy())
plt.title(r'$\beta_t$ (cosine)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_cos.numpy())
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas_cos)
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 3.3 平方根调度（常见变种） <!-- markmap: foldAll -->

#### **定义与公式**  
- 直接定义 $\bar\alpha_t = 1 - (t/T)^p$，其中 $p>0$ 是幂指数。  
  当 $p=1$ 时，$\bar\alpha_t$ 线性衰减，对应信噪比 $\text{SNR}(t) = (1 - t/T) / (t/T)$，即与线性调度类似但参数不同。  
  当 $p>1$ 时，早期 $\bar\alpha_t$ 接近 1 的时间更长，信噪比保持更高；当 $p<1$ 时，早期衰减更快。

#### **设计动机**  
通过调整幂指数 $p$ 可以精细控制信息保留的速度。例如，对于需要长期依赖的数据（如文本、时间序列），使用 $p>1$ 可使早期信息保留更久，帮助网络学习全局结构。

#### **数学特性**  
- 信噪比 $\text{SNR}(t) = \frac{1 - (t/T)^p}{(t/T)^p}$，在 $t$ 较小时 SNR 极高，随 $t$ 增大按幂律衰减。
- $\beta_t$ 可通过差分得到：$\beta_t = 1 - \frac{1 - ((t+1)/T)^p}{1 - (t/T)^p}$，但通常不显式计算 $\beta_t$，而是直接使用 $\bar\alpha_t$。

#### **代码示例**
```python
def sqrt_alpha_bar(T, p=2):
    """生成平方根 ᾱ 序列，p>0"""
    t = torch.linspace(0, 1, T+1)[1:]  # 从 1/T 到 1
    alpha_bar = 1 - t**p
    return alpha_bar

alpha_bar_sqrt = sqrt_alpha_bar(T, p=2)
betas_sqrt = 1 - alpha_bar_sqrt / torch.cat((torch.tensor([1.0]), alpha_bar_sqrt[:-1]))
sigmas_sqrt = alpha_bar_to_sigma(alpha_bar_sqrt.numpy())

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_sqrt.numpy())
plt.title(r'$\beta_t$ (sqrt, p=2)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_sqrt.numpy())
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas_sqrt)
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 3.4 Sigmoid 调度（GeoDiff, Xu et al., 2022） <!-- markmap: foldAll -->

#### **定义与公式**  
- 利用 sigmoid 函数构造 $\beta_t$：
  $$
  \beta_t = \sigma\left(-6 + \frac{12t}{N-1}\right) \cdot (\beta_{\text{end}} - \beta_{\text{start}}) + \beta_{\text{start}}
  $$
  其中 $\sigma(x) = 1/(1+e^{-x})$ 是标准 sigmoid 函数。通常 $\beta_{\text{start}}=0.0001,\ \beta_{\text{end}}=0.02$，但也可按需调整。

#### **设计动机**  
sigmoid 函数在中间区域变化快，两端变化慢，形成 S 形曲线。这种形状可以使噪声在中间阶段快速增加，而在两端缓慢变化，适合某些需要精细控制早期和后期噪声的任务，如分子构象生成或超分辨率。

#### **数学特性**  
- $\beta_t$ 从 $\beta_{\text{start}}$ 开始，缓慢上升，在 $t\approx N/2$ 附近迅速增长，最后缓慢趋近 $\beta_{\text{end}}$。
- 对应的 $\bar\alpha_t$ 在中段快速下降，两端平缓，信噪比在两端保持较好。

#### **代码示例**
```python
def sigmoid_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """生成 S 形 β 序列"""
    t = torch.linspace(0, 1, T)
    # 将 t 映射到 [-6, 6] 区间
    x = -6 + 12 * t
    sig = torch.sigmoid(x)
    betas = beta_start + (beta_end - beta_start) * sig
    return betas

betas_sigmoid = sigmoid_beta_schedule(T)
alpha_bar_sigmoid = compute_alpha_bar(betas_sigmoid)
sigmas_sigmoid = alpha_bar_to_sigma(alpha_bar_sigmoid.numpy())

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_sigmoid.numpy())
plt.title(r'$\beta_t$ (sigmoid)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_sigmoid.numpy())
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas_sigmoid)
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 3.5 多项式调度（常见变种） <!-- markmap: foldAll -->

#### **定义与公式**  
- 定义 $\bar\alpha_t = 1 - t/T$ 的 $k$ 次方：$\bar\alpha_t = (1 - t/T)^k$，或更一般地 $\bar\alpha_t = (1 - (t/T)^p)^q$。  
  当 $k=1$ 时即为线性衰减；$k>1$ 时早期衰减慢；$k<1$ 时早期衰减快。

#### **设计动机**  
多项式调度提供了比幂律更灵活的曲线族，可用于模拟不同数据分布下的信息衰减速率。

#### **数学特性**  
- 信噪比 $\text{SNR}(t) = \frac{(1 - t/T)^k}{1 - (1 - t/T)^k}$，在 $t=0$ 时无穷大，$t=T$ 时为 0。
- 通过调整指数可以控制曲率。

#### **代码示例**
```python
def polynomial_alpha_bar(T, k=1):
    """生成多项式 ᾱ 序列，ᾱ_t = (1 - t/T)^k"""
    t = torch.linspace(0, 1, T+1)[1:]  # 从 1/T 到 1
    alpha_bar = (1 - t) ** k
    return alpha_bar

k = 2  # 早期衰减慢
alpha_bar_poly = polynomial_alpha_bar(T, k)
betas_poly = 1 - alpha_bar_poly / torch.cat((torch.tensor([1.0]), alpha_bar_poly[:-1]))
sigmas_poly = alpha_bar_to_sigma(alpha_bar_poly.numpy())

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_poly.numpy())
plt.title(r'$\beta_t$ (poly, k=2)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_poly.numpy())
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas_poly)
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 3.6 学习调度（可学习参数） <!-- markmap: foldAll -->

#### **定义与公式**  
- 将 $\beta_t$ 作为可学习参数，在训练过程中通过梯度下降优化。通常需要添加正则化项（如平滑性约束）防止过拟合。

#### **设计动机**  
摆脱人工设计的调度，让模型自动找到最优的噪声增长曲线，可能提高最终生成质量或采样效率。

#### **数学特性**  
- $\beta_t$ 不再是固定值，而是随训练更新。
- 可以结合信噪比约束或单调性约束保证物理合理性。

#### **代码示例**
```python
import torch.nn as nn
import torch.optim as optim

class LearnableSchedule(nn.Module):
    def __init__(self, T, init_betas=None):
        super().__init__()
        if init_betas is None:
            init_betas = linear_beta_schedule(T)
        self.betas = nn.Parameter(init_betas.clone())
        # 可选：添加 softmax 或 sigmoid 约束，保证值在 (0,1)
    def forward(self):
        # 约束 betas 在 (0,1) 且单调递增
        betas = torch.sigmoid(self.betas)  # 映射到 (0,1)
        # 强制单调性（可选）
        betas = torch.cummax(betas, dim=0)[0]
        return betas

# 模拟训练步骤（仅演示）
schedule = LearnableSchedule(T)
optimizer = optim.Adam(schedule.parameters(), lr=1e-3)
# 假设有一个损失函数 loss = some_function(schedule())
for step in range(100):
    optimizer.zero_grad()
    betas = schedule()  # 返回约束后的 β
    # 计算损失（例如基于生成质量的奖励）
    # loss = ... 
    # loss.backward()
    # optimizer.step()
```

---

## 四、VE / EDM 类调度详解

这类调度直接以噪声强度 $\sigma_t$ 为核心，常常在**连续时间框架**下设计，便于与高效采样器（如 DDIM、DPM-Solver）结合。加噪公式为 $x = x_0 + \sigma_t\,\varepsilon$，其中 $\sigma_t$ 从 $\sigma_{\min}$ 单调递增到 $\sigma_{\max}$。

### 4.1 NCSN 指数调度（VE）(Song & Ermon, 2019)  <!-- markmap: foldAll -->

#### **定义与公式**  
- NCSN（Noise Conditional Score Network）使用几何级数定义 $\sigma_t$：
  $$
  \sigma_t = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{\frac{t-1}{T-1}},\quad t=1,\dots,T
  $$
  典型参数：$\sigma_{\min}=0.01,\ \sigma_{\max}=50,\ T=1000$。  
  也可写作 $\sigma_t = \sigma_{\min} \cdot r^{t-1}$，其中 $r = (\sigma_{\max}/\sigma_{\min})^{1/(T-1)}$。

#### **设计动机**  
在 $\log\sigma$ 轴上均匀分布，使得网络在训练时看到各噪声水平的样本数均衡，从而更好地学习得分函数（score function）。因为得分函数在不同噪声水平下具有不同的尺度，均匀覆盖有助于网络泛化。

#### **数学特性**  
- $\log\sigma_t$ 随 $t$ 线性增长。
- 信噪比 $\text{SNR}(t) = 1/\sigma_t^2$ 在 $\log$ 轴上线性递减。
- 加噪样本 $x_t = x_0 + \sigma_t\varepsilon$，方差随 $t$ 指数增长。

#### **代码示例**
```python
def ncsn_sigma_schedule(T, sigma_min=0.01, sigma_max=50):
    """NCSN 指数 σ 序列"""
    t = torch.linspace(0, 1, T)
    sigmas = sigma_min * (sigma_max / sigma_min) ** t
    return sigmas

sigmas_ncsn = ncsn_sigma_schedule(T)
alpha_bar_ncsn = sigma_to_alpha_bar(sigmas_ncsn.numpy())
betas_ncsn = 1 - alpha_bar_ncsn / np.concatenate(([1.0], alpha_bar_ncsn[:-1]))

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_ncsn)
plt.title(r'$\beta_t$ (NCSN)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_ncsn)
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas_ncsn.numpy())
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 4.2 EDM 调度（Elucidating Diffusion Models）(Karras et al., 2022)  <!-- markmap: foldAll -->

#### **定义与公式**  
- EDM 将噪声水平视为连续时间 $t$ 的函数 $\sigma(t)$，并统一了训练与采样框架。常见形式：
  - **线性**：$\sigma(t) = t$，$t\in[0,1]$ 或 $t\in[\sigma_{\min},\sigma_{\max}]$。
  - **幂次**：$\sigma(t) = t^p$，$p>0$，例如 $p=1/2$ 或 $p=2$。

- 训练时，从 $[\log\sigma_{\min},\log\sigma_{\max}]$ 上均匀采样 $\log\sigma$，即 $\sigma = \exp(\log\sigma_{\min} + u(\log\sigma_{\max}-\log\sigma_{\min})),\ u\sim U(0,1)$。  
  加噪：$x = x_0 + \sigma\varepsilon$，网络输入为 $(x, \log\sigma)$，输出预测的噪声 $\varepsilon_\theta(x,\sigma)$。

- 采样时，定义离散化路径 $\{\sigma_i\}_{i=0}^{N}$，通常取 $\sigma_0 = \sigma_{\max} > \sigma_1 > \dots > \sigma_N = 0$，然后使用 ODE 求解器（如 Heun 二阶方法）从 $\sigma_{\max}$ 逐步积分到 0。

#### **设计动机**  
将调度与采样器统一设计，使得训练时的噪声分布与采样时的离散化路径相匹配，从而在少步数下获得高质量生成。EDM 特别强调使用二阶或高阶 ODE 求解器，并针对这些求解器优化了调度参数（如 $\sigma_{\min},\sigma_{\max}$ 的选择）。

#### **数学特性**  
- 信噪比 $\text{SNR}(t) = 1/\sigma(t)^2$，与 $\sigma(t)$ 的选择直接相关。
- 训练时对数均匀采样 $\sigma$，使得网络在 $\log\sigma$ 空间均匀分布，避免某些噪声水平被忽视。
- 采样时使用 Heun 等二阶方法，误差项与 $\sigma$ 的步长相关，因此调度通常设计为在 $\log\sigma$ 轴上均匀离散化，以平衡步长。

#### **代码示例**
```python
def edm_sigma_schedule(T, sigma_min=0.002, sigma_max=80, rho=7):
    """EDM 风格 σ 序列（在 logσ 轴上均匀分布）"""
    # 离散化步长在 log σ 上均匀
    log_sigmas = np.linspace(np.log(sigma_min), np.log(sigma_max), T)
    sigmas = np.exp(log_sigmas)
    return torch.tensor(sigmas, dtype=torch.float32)

sigmas_edm = edm_sigma_schedule(T)
alpha_bar_edm = sigma_to_alpha_bar(sigmas_edm.numpy())
betas_edm = 1 - alpha_bar_edm / np.concatenate(([1.0], alpha_bar_edm[:-1]))

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_edm)
plt.title(r'$\beta_t$ (EDM)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_edm)
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas_edm.numpy())
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 4.3 基于信噪比的线性调度（常见于连续时间扩散）  <!-- markmap: foldAll -->

#### **定义与公式**  
- 直接定义 $\log\text{SNR}(t)$ 随时间线性递减：
  $$
  \log\text{SNR}(t) = \log\text{SNR}_{\min} + t \cdot (\log\text{SNR}_{\max} - \log\text{SNR}_{\min}),\quad t\in[0,1]
  $$
  其中 $\text{SNR}_{\max}$ 对应 $t=0$ 时的信噪比（通常很大），$\text{SNR}_{\min}$ 对应 $t=1$ 时的信噪比（通常很小）。  

- 然后可以转换为 VP 或 VE 参数化：
  - VP：$\bar\alpha_t = \frac{\text{SNR}(t)}{1+\text{SNR}(t)},\quad \sigma_t = \sqrt{1/\bar\alpha_t - 1}$
  - VE：$\sigma_t = 1/\sqrt{\text{SNR}(t)}$

#### **设计动机**  
信噪比是衡量信号与噪声相对强度的直接指标，控制其变化率能够更直观地反映信息保留过程。线性衰减在 $\log\text{SNR}$ 上均匀，相当于在“感知”尺度上均匀变化，适合许多数据分布。

#### **数学特性**  
- 在 VP 下，$\bar\alpha_t = \frac{1}{1+\exp(-a t - b)}$，即 sigmoid 形式。
- 在 VE 下，$\sigma_t = \exp(-a t/2 - b/2)$，即指数衰减。

#### **代码示例**
```python
def snr_linear_schedule(T, snr_min=1e-4, snr_max=1e4):
    """基于信噪比的线性调度，返回 ᾱ 和 σ"""
    t = torch.linspace(0, 1, T)
    log_snr = torch.log(torch.tensor(snr_min)) + t * (torch.log(torch.tensor(snr_max)) - torch.log(torch.tensor(snr_min)))
    snr = torch.exp(log_snr)
    alpha_bar = snr / (1 + snr)
    sigma = 1 / torch.sqrt(snr)
    return alpha_bar, sigma

alpha_bar_snr, sigma_snr = snr_linear_schedule(T)
betas_snr = 1 - alpha_bar_snr / torch.cat((torch.tensor([1.0]), alpha_bar_snr[:-1]))

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_snr.numpy())
plt.title(r'$\beta_t$ (SNR-based)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_snr.numpy())
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigma_snr.numpy())
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

### 4.4 Log-Linear 调度（与 NCSN 指数调度等价）  <!-- markmap: foldAll -->

#### **定义与公式**  
- 定义 $\log\sigma_t$ 随 $t$ 线性增长：
  $$
  \log\sigma_t = \log\sigma_{\min} + \frac{t-1}{T-1}(\log\sigma_{\max} - \log\sigma_{\min})
  $$
  即 $\sigma_t = \sigma_{\min} \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{\frac{t-1}{T-1}}$，这实际上与 NCSN 指数调度完全相同。但在连续时间版本中，$\sigma(t) = \exp(a t + b)$，与 SNR 线性调度等价。

#### **设计动机**  
在 $\log\sigma$ 轴上均匀分布，使网络在噪声强度的对数尺度上学习，这在很多工作中被证明有利于训练。此外，Log-Linear 调度实现简单，仅需两个端点。

#### **数学特性**  
- 与 NCSN 指数调度完全一致，可视为其特例。
- 信噪比 $\text{SNR}(t) = 1/\sigma_t^2$ 在 $\log$ 轴上线性递减。

#### **代码示例**（与 NCSN 相同，可重用）
```python
# 与 NCSN 指数调度代码相同，故略
```

### 4.5 分段调度（自定义）  <!-- markmap: foldAll -->

#### **定义与公式**  
- 将调度分为若干段，每段采用不同的函数形式。例如：
  - 早期（$t\in[0,0.3]$）：线性增长 $\sigma(t)$，保持低噪声水平。
  - 中期（$t\in[0.3,0.7]$）：指数增长，快速提高噪声。
  - 后期（$t\in[0.7,1]$）：饱和，$\sigma(t)$ 趋近 $\sigma_{\max}$。

- 具体形式可根据任务设计，如 $\sigma(t) = \begin{cases} a t & t < t_1 \\ b e^{c t} & t_1 \le t < t_2 \\ \sigma_{\max} & t \ge t_2 \end{cases}$。

#### **设计动机**  
为了兼顾不同阶段的特性，例如早期需要精细保留结构，中期需要快速扩散，后期需要稳定到纯噪声。通过分段可以灵活定制，优化特定任务（如图像修复、编辑）的生成路径。

#### **数学特性**  
- 每段可以是线性、指数、多项式等。
- 在连接点处需保证连续性（通常也要求一阶可导，以避免采样器不稳定）。

#### **代码示例**
```python
def piecewise_sigma_schedule(T, t1=0.3, t2=0.7, sigma_max=10, sigma_min=0.01):
    """分段调度：线性-指数-常数"""
    t = np.linspace(0, 1, T)
    sigma = np.zeros_like(t)
    # 第一段：线性
    idx1 = t < t1
    sigma[idx1] = sigma_min + (sigma_max - sigma_min) * (t[idx1] / t1)
    # 第二段：指数
    idx2 = (t >= t1) & (t < t2)
    # 确保连接点连续
    sigma1_end = sigma_min + (sigma_max - sigma_min) * (t1 / t1)  # 即 sigma_max? 这里修正
    # 更合理的：第一段线性到某个中间值，第二段指数到 sigma_max
    sigma_mid = 1.0  # 自定义中间值
    sigma[idx2] = sigma_mid * np.exp((t[idx2] - t1) / (t2 - t1) * np.log(sigma_max / sigma_mid))
    # 第三段：常数
    idx3 = t >= t2
    sigma[idx3] = sigma_max
    return torch.tensor(sigma, dtype=torch.float32)

sigmas_piece = piecewise_sigma_schedule(T)
alpha_bar_piece = sigma_to_alpha_bar(sigmas_piece.numpy())
betas_piece = 1 - alpha_bar_piece / np.concatenate(([1.0], alpha_bar_piece[:-1]))

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(betas_piece)
plt.title(r'$\beta_t$ (piecewise)')
plt.subplot(1, 3, 2)
plt.plot(alpha_bar_piece)
plt.title(r'$\bar\alpha_t$')
plt.subplot(1, 3, 3)
plt.plot(sigmas_piece.numpy())
plt.title(r'$\sigma_t$')
plt.tight_layout()
plt.show()
```

---

## 五、连续时间与自适应调度

### 5.1 连续时间调度  <!-- markmap: foldAll -->

#### **定义**  
- 将时间 $t$ 视为连续变量（如 $t\in[0,1]$），用函数 $\sigma(t)$ 或 $\bar\alpha(t)$ 定义噪声水平。训练时在 $[0,1]$ 上随机采样 $t$，加噪对应 $\sigma(t)$；采样时可根据需要选择任意步数离散化该函数。

#### **设计动机**  
摆脱离散时间步的限制，使模型能够以任意精度采样，并与 ODE/SDE 理论无缝结合。连续时间框架也便于推导高阶求解器，如 DPM-Solver。

#### **数学特性**  
- 前向过程可写为 SDE：$dx = f(x,t)dt + g(t)dw$，其中 $f,g$ 与 $\sigma(t)$ 或 $\bar\alpha(t)$ 相关。
- 逆向过程对应 ODE：$\frac{dx}{dt} = \text{(score function)}$，可使用数值积分求解。

#### **代码示例**
```python
def continuous_sigma_func(t, sigma_min=0.01, sigma_max=80, rho=7):
    """EDM 风格的连续 σ(t)，t∈[0,1]"""
    # 使用幂次调度 σ = sigma_min + (sigma_max - sigma_min) * t^rho
    return sigma_min + (sigma_max - sigma_min) * (t ** rho)

def sample_continuous_time(batchsize, device='cpu'):
    """训练时采样连续时间 t 和对应的 σ"""
    t = torch.rand(batchsize, device=device)  # 均匀采样
    sigma = continuous_sigma_func(t)
    return t, sigma

# 使用示例
t, sigma = sample_continuous_time(4)
print(f"t: {t}\nsigma: {sigma}")
```

### 5.2 自适应/数据依赖调度  <!-- markmap: foldAll -->

#### **定义**  
- 调度参数（如 $\beta_t$ 或 $\sigma_t$）根据数据分布或训练状态动态调整。例如：
  - 根据当前噪声水平下网络损失的大小，动态调整该水平的采样概率（重要性采样）。
  - 根据数据集的方差或图像复杂度，调整 $\sigma_{\max}$ 或 $\sigma_{\min}$。
  - 在训练过程中逐渐增加噪声范围（调度预热）。

#### **设计动机**
固定的调度可能无法适应所有数据或训练阶段。通过自适应调整，可以更有效地利用训练信号，提高收敛速度或最终质量。

#### **数学特性**  
- 调度不再是预先固定的，而是随训练动态变化。
- 需要额外的机制（如损失监测、数据统计）来更新调度参数。

#### **代码示例**
```python
class AdaptiveSchedule:
    def __init__(self, initial_sigmas, update_interval=100):
        self.sigmas = initial_sigmas.clone()
        self.update_interval = update_interval
        self.step = 0

    def sample_batch(self, x0):
        # 均匀采样 σ
        idx = torch.randint(len(self.sigmas), (x0.shape[0],))
        return self.sigmas[idx]

    def update(self, loss_by_sigma):
        """根据每个噪声水平的损失调整采样概率"""
        # 简化示例：提高损失高的 σ 的采样概率
        prob = torch.softmax(loss_by_sigma, dim=0)
        # 重新采样 σ 序列（实际中可能需要重采样或调整权重）
        pass
```

---

## 六、信噪比（SNR）与调度设计的内在联系  <!-- markmap: foldAll -->

SNR 是衡量信号与噪声相对强度的核心指标。在扩散模型中，SNR 随 $t$ 单调递减。调度设计本质上是在控制 SNR 的衰减曲线。

- **VP 参数化下**：$\text{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t}$
- **VE 参数化下**：$\text{SNR}(t) = \frac{1}{\sigma_t^2}$

**重要推论**：
- SNR 决定了训练时网络学习去噪的难度。SNR 过高时，网络几乎看不到噪声，梯度较小；SNR 过低时，信号完全被淹没，网络只能预测纯噪声。
- 一个好的调度应使 SNR 在中间区域保持适中，使网络有机会学习中等强度的去噪任务。
- 余弦调度、EDM 调度等正是通过在 SNR 对数域线性变化，或在中段平缓下降，来平衡各阶段的学习难度。

#### **代码示例**
```python
def compute_snr(alpha_bar):
    return alpha_bar / (1 - alpha_bar)

def plot_snr_curves():
    T = 1000
    # 线性调度
    betas_lin = linear_beta_schedule(T)
    alpha_bar_lin = compute_alpha_bar(betas_lin)
    snr_lin = compute_snr(alpha_bar_lin.numpy())
    # 余弦调度
    betas_cos = cosine_beta_schedule(T)
    alpha_bar_cos = compute_alpha_bar(betas_cos)
    snr_cos = compute_snr(alpha_bar_cos.numpy())
    # EDM 调度（VE 参数）
    sigmas_edm = edm_sigma_schedule(T)
    snr_edm = 1 / (sigmas_edm.numpy() ** 2)

    plt.figure(figsize=(8, 4))
    plt.semilogy(snr_lin, label='linear')
    plt.semilogy(snr_cos, label='cosine')
    plt.semilogy(snr_edm, label='EDM')
    plt.xlabel('t')
    plt.ylabel('SNR')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

## 七、调度与采样器的关系  <!-- markmap: foldAll -->

采样器（如 DDPM、DDIM、DPM-Solver）依赖调度提供的噪声序列（$\sigma_t$ 或 $\bar\alpha_t$）来执行逆向步骤。

- **DDPM**：使用完整的 $\beta_t$ 或 $\sigma_t$ 序列，每步添加随机噪声（$\eta>0$）。采样质量高但步数多。
- **DDIM**：利用确定性去噪，允许跳过中间步，可在调度上任意子采样。采样效率高，但质量受调度连续性影响。
- **DPM-Solver**：基于 ODE 的高阶求解器，假设 SNR 或 $\sigma(t)$ 满足某些平滑性。与连续时间调度（如 EDM 的 $\sigma(t)=t$）配合最佳。
- **Heun 等二阶方法**：在 EDM 框架下，配合幂次调度可实现 20–50 步高质量生成。

调度与采样器的匹配：例如，余弦调度配合 DDIM 能显著减少步数；EDM 调度本身设计时已考虑二阶 ODE 求解器。

#### **代码示例（DDIM 子采样）**
```python
def ddim_subsample(sigmas, steps):
    """从递增的 σ 序列中取出 steps 个递减值用于 DDIM"""
    N = len(sigmas)
    indices = list((N * (1 - np.arange(0, steps)/steps)).round().astype(np.int64) - 1)
    indices = indices + [0]  # 包含最小 σ
    return sigmas[indices[::-1]]  # 返回递减顺序

sigmas_full = edm_sigma_schedule(1000)
sigmas_ddim = ddim_subsample(sigmas_full, 20)
print(f"DDIM 使用 {len(sigmas_ddim)-1} 步，σ 范围 {sigmas_ddim[0]:.2f} -> {sigmas_ddim[-1]:.2f}")
```

---

## 八、调度与网络输入（时间嵌入）  <!-- markmap: foldAll -->

网络通常需要接受噪声水平作为条件输入。常见做法有：

- **原始时间步 $t$**：嵌入为位置编码（如正弦/余弦），用于 VP 调度。
- **$\log\sigma$**：直接使用噪声水平的对数，适合 VE 调度。EDM 推荐使用 $\log\sigma$ 作为输入，因为它在训练中对数均匀采样，使网络更易学习。
- **SNR 或 $\log\text{SNR}$**：在某些工作中直接嵌入信噪比，方便与理论分析对应。

选择哪种嵌入形式会影响网络对不同噪声水平的泛化能力。

#### **代码示例（时间嵌入）**
```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, sigma):
        # sigma 形状 (B,) 或 (B,1)
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        # 使用 logσ 作为输入
        log_sigma = torch.log(sigma)
        # 正弦位置编码
        freqs = torch.exp(torch.linspace(0, np.log(1000), self.dim // 2, device=sigma.device))
        emb = log_sigma * freqs
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
```

---

## 九、训练中的噪声分布采样策略  <!-- markmap: foldAll -->

训练时，我们需要从调度中采样 $\sigma_t$（或 $t$）来构造加噪样本。采样策略影响网络在各个噪声水平上的学习均衡性。

- **均匀采样 $t$**：经典做法，但若 $\sigma_t$ 或 SNR 变化剧烈，会导致某些水平样本过少。
- **对数均匀采样 $\sigma$**：在 VE 调度中常用，使网络在 $\log\sigma$ 空间均匀分布，避免极端值主导。
- **基于 SNR 的均匀采样**：在 SNR 对数域均匀采样，与 VP 结合时相当于让网络在信噪比空间均匀分布。

#### **代码示例（对数均匀采样 σ）**
```python
def sample_sigma_log_uniform(batchsize, sigma_min, sigma_max, device='cpu'):
    """从 [sigma_min, sigma_max] 对数均匀采样"""
    log_sigma_min = np.log(sigma_min)
    log_sigma_max = np.log(sigma_max)
    log_sigma = torch.rand(batchsize, device=device) * (log_sigma_max - log_sigma_min) + log_sigma_min
    return torch.exp(log_sigma)

sigma_samples = sample_sigma_log_uniform(1000, 0.01, 50)
plt.hist(sigma_samples.numpy(), bins=50, log=True)
plt.xlabel('σ')
plt.title('对数均匀采样结果')
plt.show()
```

---

## 十、条件生成中调度对 CFG 的影响  <!-- markmap: foldAll -->

Classifier-Free Guidance (CFG) 通过在采样时混合条件与无条件预测来增强条件控制。调度影响 CFG 的强度选择：
- 当 SNR 较高（$\sigma$ 小）时，条件信号容易被淹没，CFG 可能需要更高系数；
- 在噪声水平较低的区域，CFG 可能导致过饱和或 artifacts，因此常对调度末段进行截断或特殊处理。

#### **代码示例（CFG 权重衰减）**
```python
def cfg_scale_schedule(step, total_steps, cfg_start=1.0, cfg_end=4.0):
    """线性增加 CFG 权重，从 cfg_start 到 cfg_end"""
    return cfg_start + (cfg_end - cfg_start) * (step / total_steps)

# 在采样循环中使用
cfg_scales = [cfg_scale_schedule(i, 20) for i in range(20)]
plt.plot(cfg_scales)
plt.xlabel('采样步')
plt.ylabel('CFG scale')
plt.show()
```

---

## 十一、动态调度与训练中调整  <!-- markmap: foldAll -->

动态调度是指在训练过程中改变调度参数，以优化收敛或最终质量。例如：
- **调度预热**：早期使用较温和的调度，随训练进行逐渐增加噪声范围。
- **损失自适应**：根据网络在特定噪声水平上的损失，动态调整该水平的采样概率。
- **多阶段调度**：先训练低噪声区域，再逐渐引入高噪声区域，类似于课程学习。

#### **代码示例（调度预热）**
```python
def warmup_schedule(epoch, total_epochs, sigma_max_final=50, sigma_max_initial=10):
    """线性增加 σ_max"""
    t = epoch / total_epochs
    sigma_max = sigma_max_initial + t * (sigma_max_final - sigma_max_initial)
    # 重新生成调度
    sigmas = edm_sigma_schedule(1000, sigma_min=0.01, sigma_max=sigma_max)
    return sigmas

# 模拟训练
for epoch in range(100):
    sigmas = warmup_schedule(epoch, 100)
    # 使用 sigmas 进行训练...
```

---

## 十二、v-parameterization 中的调度  <!-- markmap: foldAll -->

v-parameterization（如 ProlificDreamer 中使用）将网络输出改为预测速度 $v = \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x_0$。此时调度公式需相应调整，但核心仍是 $\bar\alpha_t$ 序列。该参数化常用于高分辨率生成或扩散蒸馏。

#### **代码示例**
```python
def v_parameterization_loss(x0, eps, alpha_bar):
    """v-parameterization 的损失"""
    sqrt_alpha = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)
    v_target = sqrt_alpha * eps - sqrt_one_minus_alpha * x0
    # 网络输出 v_pred，损失为 MSE(v_target, v_pred)
    # loss = F.mse_loss(v_pred, v_target)
    pass
```

---

## 十三、最新趋势与前沿  <!-- markmap: foldAll -->

- **Rectified Flow**：直接学习 ODE 的直线路径，调度与流场结合，可实现一步生成。
- **Consistency Models**：通过蒸馏训练，将任意噪声直接映射到数据，调度的角色减弱但仍在训练中使用。
- **Flow Matching**：与扩散类似，但使用更灵活的概率路径，调度设计可视为选择路径的时间参数化。
- **Latent 扩散中的调度优化**：Stable Diffusion 3 等采用自定义的连续时间调度，结合多阶段采样策略，在潜空间实现高效生成。

#### **代码示例（Rectified Flow 简单调度）**
```python
def rectified_flow_schedule(t, sigma_max=1.0):
    """Rectified Flow 常用调度：线性插值"""
    return t  # 直接返回时间本身，对应 x_t = (1-t)*x0 + t*eps
```

---

## 十四、调度选择指南（详细版）

| 调度类型 | 优点 | 缺点 | 适合场景 / 理由 |
|---------|------|------|----------------|
| **线性调度 (DDPM)** | 1. 实现简单，与原始 DDPM 完全一致。<br>2. 复现经典论文时无需调整。<br>3. 社区支持广泛，预训练权重多。 | 1. 中间阶段信噪比下降过快，网络难以学习中等噪声水平的去噪。<br>2. 采样时通常需要 1000 步才能获得高质量结果，效率低。 | 1. 复现 DDPM 论文或微调已有 DDPM 模型。<br>2. 小规模实验，对采样速度无严格要求。 |
| **余弦调度 (Improved DDPM)** | 1. 中间阶段信噪比更高，训练更稳定。<br>2. 采样时仅需 100–200 步即可达到高质量。<br>3. 兼容所有 VP 参数化模型。 | 1. 需要计算余弦并反推 β，实现略复杂。<br>2. 对于极低噪声区域，可能仍存在数值不稳定性（需截断）。 | 1. 大多数 VP 条件生成任务（如文本到图像、类别条件）。<br>2. 希望提升采样效率且保持预训练权重兼容性。 |
| **NCSN 指数调度 (VE)** | 1. 在 log σ 轴上均匀分布，训练时各噪声水平样本均衡。<br>2. 天然适合学习得分函数，与 SDE 理论契合。 | 1. 直接采样时需较多步数（通常 1000 步）。<br>2. 若不加处理，σ 范围可能过大导致数值不稳定。 | 1. 学习得分函数（如 NCSN 系列）。<br>2. 作为 VE 基线与 SDE 求解器结合使用。 |
| **EDM 调度** | 1. 训练与采样统一使用 σ 序列，代码简洁。<br>2. 配合二阶/高阶 ODE 求解器，可在 20–50 步内达到 SOTA 质量。<br>3. 设计时考虑了采样器的数值稳定性。 | 1. 需要配合专门的采样器（如 Heun、DPM-Solver）才能发挥最佳效果。<br>2. 对于初学者，参数调优（如 σ_min、σ_max 选择）较敏感。 | 1. 从头训练高分辨率图像（如 256×256 以上）。<br>2. 追求极致采样速度与质量平衡（如实时生成、迭代蒸馏）。 |
| **基于 SNR 的线性调度** | 1. 直接控制信噪比变化率，理论分析方便。<br>2. 在 SNR 对数域均匀采样，训练时网络能均衡学习不同信噪比水平。 | 1. 需要先确定 SNR_min 和 SNR_max，边界选择影响效果。<br>2. 与 VP/VE 的转换可能引入额外计算。 | 1. 语言模型扩散、离散数据扩散。<br>2. 连续时间扩散理论研究。 |
| **Log‑Linear 调度** | 1. 实现简单，仅需两个端点。<br>2. 在 log σ 上均匀，适合与对数嵌入的网络配合。 | 1. 缺乏中间区域的精细控制。<br>2. 对某些数据分布可能不是最优。 | 1. 快速原型验证、小数据集。<br>2. 与 Scaled 修饰器配合用于 U‑Net 示例。 |
| **Sigmoid 调度** | 1. 噪声变化呈 S 形，中间快两端慢，适合某些几何数据（如分子构象）。<br>2. 可调节曲线形状（通过线性映射的范围）。 | 1. 对一般图像数据可能不具优势。<br>2. 需要根据任务调整参数。 | 1. 分子构象生成（GeoDiff）。<br>2. 超分辨率等需要精细控制噪声增长的任务。 |
| **分段调度** | 1. 可灵活定制不同阶段的噪声增长速率。<br>2. 能针对特定任务（如修复、编辑）优化。 | 1. 设计复杂，需要多次实验调参。<br>2. 不易与现有预训练权重兼容。 | 1. 特定下游任务（如扩散修复、图像编辑）。<br>2. 研究探索性项目。 |

> **选型建议**：若不确定，可先从**余弦调度**（VP 类）或**EDM 调度**（VE 类）开始，这两者在多数任务中表现稳健。

---

## 十五、总结与趋势

- 噪声调度是扩散模型的“时间轴”，决定了训练与采样的行为。
- 两种主流参数化 VP 与 VE 通过 $\sigma = \sqrt{1/\bar\alpha - 1}$ 等价，选择取决于实现便利性和社区习惯。
- 经典调度（线性、余弦）简单有效，适合入门和复现；现代调度（EDM、SNR-based）更注重统一性与采样效率。
- 调度与采样器、时间嵌入、训练采样策略、条件生成等因素紧密耦合，需综合考虑。
- 未来方向：更精细的自适应调度、与架构联合优化、以及扩展到离散数据的统一调度框架。同时，扩散蒸馏和流匹配等新范式正在重新定义调度的作用。

---

## 十六、**参考文献**  <!-- markmap: foldAll -->
1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.  
2. Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.  
3. Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *NeurIPS*.  
4. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR*.  
5. Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. *NeurIPS*.  
6. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR*.  
7. Xu, M., Yu, L., Song, Y., Shi, C., Ermon, S., & Tang, J. (2022). GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation. *ICLR*.  
8. Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICML*.  
9. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency Models. *ICML*.  
10. Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR*.  
11. Sauer, A., Karras, T., & Laine, S. (2023). Leveraging Diffusion Models for High-Fidelity Image Generation. *CVPR* (for EDM extensions).

---

## 十七、**附录：常用参数对照表**

| 名称 | 符号 | 定义 | 与调度关系 |
|------|------|------|-----------|
| 噪声方差（VP） | $\beta_t$ | 前向噪声方差 | 基础序列 |
| 信号保持率 | $\bar\alpha_t$ | $\prod_{s=1}^t (1-\beta_s)$ | 控制信号强度 |
| 噪声水平（VE） | $\sigma_t$ | $\sqrt{1/\bar\alpha_t - 1}$ | 直接控制加噪 |
| 信噪比（SNR） | $\text{SNR}_t$ | $\bar\alpha_t/(1-\bar\alpha_t) = 1/\sigma_t^2$ | 核心设计指标 |
| 时间步（离散） | $t$ | $1,\dots,T$ | 调度索引 |
| 连续时间 | $s$ 或 $t$ | $0$ 到 $1$ | 连续调度函数参数 |


## 十八、图解
<div style="position: relative;">
  <iframe
    src="/mywebsite/posts/Schedule.html"
    id="my-iframe"
    style="width:100%;height:70vh;border:0;">
  </iframe>
  <button onclick="toggleFullscreen()" style="
    position: absolute;
    bottom: 8px;
    right: 8px;
    padding: 3px 8px;
    font-size: 12px;
    line-height: 1.3;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 3px;
    cursor: pointer;
  ">⛶ 全屏</button>
</div>

<script>
function toggleFullscreen() {
  const iframe = document.getElementById('my-iframe');
  if (!document.fullscreenElement) {
    iframe.requestFullscreen?.() || 
    iframe.webkitRequestFullscreen?.() || 
    iframe.msRequestFullscreen?.();
  } else {
    document.exitFullscreen?.() || 
    document.webkitExitFullscreen?.() || 
    document.msExitFullscreen?.();
  }
}
</script>