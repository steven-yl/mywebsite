---
title: "Analytic Diffusion Studio — Wiener 滤波去噪器"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — Wiener 滤波去噪器"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — Wiener 滤波去噪器"
---

# 06 — Wiener 滤波去噪器

文件：`src/local_diffusion/models/wiener.py`

## 6.1 概述

Wiener 滤波器是经典信号处理中的线性最优滤波器。在扩散模型的语境下，它假设数据分布为高斯分布 $x_0 \sim \mathcal{N}(\mu, \Sigma)$，在此假设下推导出闭式的线性去噪公式。

**解决的问题**：给定噪声图像 $x_t$，如何在均方误差意义下最优地恢复 $x_0$（限制为线性估计器）。

**适用场景**：
- 快速基线对比
- 低分辨率数据集（协方差矩阵可放入内存）
- 理解扩散去噪的线性近似

## 6.2 数学推导

### 前提

假设 $x_0 \sim \mathcal{N}(\mu, \Sigma)$，前向过程给出：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

### 线性最优估计

在高斯假设下，后验 $p(x_0 | x_t)$ 也是高斯的，其均值（即 MMSE 估计）为：

$$\hat{x}_0 = L_t \cdot x_t + H_t \cdot \mu$$

其中：

$$L_t = \frac{\bar{\alpha}_t \Sigma}{(1-\bar{\alpha}_t)I + \bar{\alpha}_t \Sigma} \cdot \frac{1}{\sqrt{\bar{\alpha}_t}}$$

$$H_t = I - \sqrt{\bar{\alpha}_t} \cdot L_t$$

### SVD 加速

对协方差矩阵做 SVD：$\Sigma = U \Lambda V^H$

收缩因子（对角矩阵）：

$$\text{shrink}_i = \frac{\bar{\alpha}_t \lambda_i}{(1-\bar{\alpha}_t) + \bar{\alpha}_t \lambda_i}$$

则 $L_t L_t^T = U \cdot \text{diag}(\text{shrink}) \cdot V^H$。

## 6.3 类定义

```python
@register_model("wiener")
class DenoisingWiener(BaseDenoiser):
    def __init__(self, dataset, device, num_steps, *, params=None, **kwargs):
        # 继承 BaseDenoiser
        # 设置 wiener_path（默认 data/models/wiener/{dataset}_{resolution}）
```

### 构造函数参数

| 参数 | 说明 |
|------|------|
| `dataset` | DatasetBundle 实例 |
| `device` | 计算设备 |
| `num_steps` | DDIM 采样步数 |
| `params.wiener_path` | Wiener 滤波器存储路径（可选） |

## 6.4 train() 方法

```python
def train(self, dataset: DatasetBundle):
    try:
        U, LA, Vh, mean = load_wiener_filter(self.wiener_path, device=self.device)
    except FileNotFoundError:
        # 从数据集计算协方差矩阵
        S, mean = compute_wiener_filter(
            dataloader=dataset.dataloader,
            device=self.device,
            resolution=self.resolution,
            n_channels=self.n_channels,
        )
        # SVD 分解
        U, LA, Vh = torch.linalg.svd(S)
        save_wiener_filter(U, LA, Vh, mean, self.wiener_path)

    self.register_buffer("U", U)
    self.register_buffer("LA", LA)    # 特征值（奇异值）
    self.register_buffer("Vh", Vh)
    self.register_buffer("mean", mean)
```

流程：
1. 尝试从磁盘加载已有的 SVD 分解结果
2. 若不存在，从数据集计算协方差矩阵 $\Sigma$ 和均值 $\mu$
3. 对 $\Sigma$ 做 SVD 分解
4. 保存到磁盘（下次直接加载）
5. 注册为 PyTorch buffer（随模型移动到 GPU）

## 6.5 _get_Lt_Ht() 方法

```python
def _get_Lt_Ht(self, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t

    # 计算收缩因子
    shrink_factors = alpha_prod_t * self.LA / (beta_prod_t + alpha_prod_t * self.LA)
    LAshrink = torch.diag(shrink_factors)
    LLt = self.U @ LAshrink @ self.Vh    # 收缩后的滤波矩阵

    I = torch.eye(LLt.shape[0], device=LLt.device)
    Ht = I - LLt                          # 均值项系数
    Lt = LLt / torch.sqrt(alpha_prod_t)   # 输入项系数（除以 √ᾱ_t）
    return Lt, Ht
```

返回两个矩阵：
- `Lt`：作用于 $x_t$ 的线性变换
- `Ht`：作用于均值 $\mu$ 的线性变换

## 6.6 denoise() 方法

```python
@torch.no_grad()
def denoise(self, latents, timestep, *, generator=None, **kwargs):
    timestep_index = int(timestep.item())
    Lt, Ht = self._get_Lt_Ht(timestep_index)

    latents_flat = latents.flatten(start_dim=1)  # [B, n]

    # L_t @ x_t
    lx0_flat = (Lt @ latents_flat.T).T           # [B, n]

    # H_t @ μ
    mean_term_flat = (Ht @ self.mean.unsqueeze(-1)).squeeze(-1)  # [n]

    # x̂₀ = L_t x_t + H_t μ
    total_x0 = (lx0_flat + mean_term_flat.unsqueeze(0)).view_as(latents)
    return total_x0
```

计算步骤：
1. 将图像展平为向量 `[B, n_pixels]`
2. 矩阵乘法 $L_t \cdot x_t$
3. 矩阵乘法 $H_t \cdot \mu$
4. 相加得到预测 $\hat{x}_0$
5. 恢复为图像形状

**复杂度**：$O(B \cdot n^2)$，其中 $n = C \times H \times W$ 是像素总数。对于 64×64 RGB 图像，$n = 12288$，矩阵大小约 150M 个元素。

## 6.7 配置示例

```yaml
# configs/wiener/cifar10.yaml
model:
  name: wiener
  # Wiener 模型无额外超参数（params 为空）
```

Wiener 模型不需要额外超参数，所有行为由数据集统计量决定。

## 6.8 局限性

- 高斯假设过于简化，真实图像分布远非高斯
- 生成的图像趋向模糊（因为是所有可能 $x_0$ 的加权平均）
- 协方差矩阵大小为 $n \times n$，高分辨率时内存不可行
- 作为线性估计器，无法捕捉非线性结构
