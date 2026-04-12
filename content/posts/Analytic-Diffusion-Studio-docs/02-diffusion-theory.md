---
title: "Analytic Diffusion Studio — 扩散模型理论基础"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 扩散模型理论基础"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 扩散模型理论基础"
---

## 2.1 前向过程（加噪）

扩散模型的前向过程将干净图像 $x_0$ 逐步加噪，得到噪声图像 $x_t$：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中：
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s)$ 是累积信噪比
- $\beta_t$ 是噪声调度表（本项目使用线性调度：$\beta_1 = 0.0001$，$\beta_T = 0.02$）
- 当 $t$ 增大时，$\bar{\alpha}_t \to 0$，图像趋向纯噪声

**直觉理解**：前向过程是一个确定性的"加噪配方"。给定 $x_0$ 和时间步 $t$，可以一步跳到 $x_t$，无需逐步模拟。

## 2.2 反向过程（去噪）

反向过程的目标是从噪声 $x_T \sim \mathcal{N}(0, I)$ 逐步恢复干净图像。核心问题是估计去噪函数：

$$\hat{x}_0 = D(x_t, t)$$

传统方法训练神经网络来学习 $D$；本项目的方法直接用数据统计量构造 $D$。

## 2.3 DDIM 采样

本项目使用 DDIM (Denoising Diffusion Implicit Models) 调度器进行采样。DDIM 的更新规则为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \hat{\epsilon}_t$$

其中预测噪声 $\hat{\epsilon}_t$ 由预测的 $\hat{x}_0$ 反推：

$$\hat{\epsilon}_t = \frac{x_t - \sqrt{\bar{\alpha}_t} \cdot \hat{x}_0}{\sqrt{1 - \bar{\alpha}_t}}$$

在代码中，这对应 `BaseDenoiser.compute_noise_from_x0()` 方法：

```python
def compute_noise_from_x0(self, x_t, pred_x0, timestep):
    alpha_prod = self.scheduler.alphas_cumprod[t]
    beta_prod = 1 - alpha_prod
    sqrt_alpha = torch.sqrt(alpha_prod)
    sqrt_beta = torch.sqrt(beta_prod + 1e-8)
    return (x_t - sqrt_alpha * pred_x0) / sqrt_beta
```

**为什么用 DDIM 而不是 DDPM？** DDIM 是确定性采样（无额外随机噪声），这使得不同方法在相同种子下可以公平对比。

## 2.4 贝叶斯最优去噪器

给定噪声观测 $x_t$，贝叶斯最优估计为后验均值：

$$D^*(x_t, t) = \mathbb{E}[x_0 | x_t] = \frac{\int x_0 \cdot p(x_t | x_0) \cdot p(x_0) \, dx_0}{\int p(x_t | x_0) \cdot p(x_0) \, dx_0}$$

由于 $p(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$，展开后得到：

$$D^*(x_t, t) = \frac{\sum_i x_0^{(i)} \cdot \exp\left(-\frac{\|x_t - \sqrt{\bar{\alpha}_t} x_0^{(i)}\|^2}{2(1-\bar{\alpha}_t)}\right)}{\sum_i \exp\left(-\frac{\|x_t - \sqrt{\bar{\alpha}_t} x_0^{(i)}\|^2}{2(1-\bar{\alpha}_t)}\right)}$$

这就是 `OptimalDenoiser` 实现的公式——对数据集中所有图像做 softmax 加权平均。

## 2.5 Wiener 滤波（线性最优）

如果假设数据分布为高斯 $x_0 \sim \mathcal{N}(\mu, \Sigma)$，则贝叶斯最优去噪器退化为线性形式：

$$D_{\text{Wiener}}(x_t, t) = L_t \cdot x_t + H_t \cdot \mu$$

其中：
- $L_t = \frac{\bar{\alpha}_t \Sigma}{(1-\bar{\alpha}_t)I + \bar{\alpha}_t \Sigma} \cdot \frac{1}{\sqrt{\bar{\alpha}_t}}$
- $H_t = I - L_t \sqrt{\bar{\alpha}_t}$

通过对协方差矩阵 $\Sigma$ 做 SVD 分解 $\Sigma = U \Lambda V^H$，可以高效计算收缩因子：

$$\text{shrink}_i = \frac{\bar{\alpha}_t \lambda_i}{(1-\bar{\alpha}_t) + \bar{\alpha}_t \lambda_i}$$

## 2.6 局部性（Locality）

PCA Locality 方法的核心发现：扩散模型的去噪操作具有**空间局部性**——预测某个像素的值主要依赖于其邻域像素，而非全图。

这种局部性可以从数据协方差矩阵的结构中自然涌现。具体来说，Wiener 滤波矩阵 $L_t L_t^T$ 的非对角元素在远离对角线时迅速衰减，形成类似"局部感受野"的模式。

PCA Locality 方法将这种局部性显式地编码为二值掩码，用于修改最优去噪器中的距离度量，使其只关注局部像素。

## 2.7 噪声调度表

本项目使用线性噪声调度：

```python
# 在 BaseDenoiser.__init__ 中通过 DDIMScheduler 设置
DDIMScheduler(
    beta_start=0.0001,   # β₁
    beta_end=0.02,       # β_T
    beta_schedule="linear",
    prediction_type="epsilon",  # 预测噪声 ε
)
```

时间步从 999（高噪声）到 0（低噪声），采样时使用 `num_inference_steps`（默认 10）个等间距时间步。
