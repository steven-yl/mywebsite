---
title: "Analytic Diffusion Studio — 平滑最优去噪器"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 平滑最优去噪器"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 8
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 平滑最优去噪器"
---

# 08 — 平滑最优去噪器 (SCFDM)

文件：`src/local_diffusion/models/scfdm.py`

论文：[Score-based Generative Models with Closed-Form Denoisers](https://arxiv.org/abs/2310.12395)

## 8.1 概述

SCFDM (Smoothed Closed-Form Diffusion Model) 是对 Optimal 去噪器的平滑改进。核心思想是：对输入 $x_t$ 添加多组小幅高斯扰动，分别用 Optimal 去噪器处理，然后取平均。

**解决的问题**：Optimal 去噪器的输出可能不够平滑（因为 softmax 权重对输入敏感），通过蒙特卡洛平均来平滑输出。

**与 Optimal 的关系**：SCFDM 继承自 `OptimalDenoiser`，复用其 FAISS 索引和 softmax 加权逻辑，仅在 `denoise()` 中添加扰动-平均步骤。

## 8.2 数学公式

$$D_{\text{SCFDM}}(x_t, t) = \frac{1}{M} \sum_{j=1}^{M} D^*(x_t + \sigma_s \epsilon_j, t), \quad \epsilon_j \sim \mathcal{N}(0, I)$$

其中：
- $D^*$ 是 Optimal 去噪器
- $M$ 是噪声采样数（`num_noise`）
- $\sigma_s$ 是平滑标准差（`smoothing_std`）

## 8.3 类定义

```python
@register_model("scfdm")
class SmoothedCFDM(OptimalDenoiser):
    """继承自 OptimalDenoiser，添加高斯平滑。"""

    def __init__(self, dataset, device, num_steps, *, params=None, **kwargs):
        super().__init__(dataset=dataset, device=device, num_steps=num_steps,
                         params=params, **kwargs)
        self.num_noise = int(params.get("num_noise", 1))
        self.smoothing_std = float(params.get("smoothing_std", 0.0))
```

### 构造函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `params.num_noise` | `1` | 高斯扰动采样数 M |
| `params.smoothing_std` | `0.0` | 扰动标准差 σ_s |
| `params.temperature` | `1.0` | 继承自 Optimal |
| `params.num_neighbors` | `2000` | 继承自 Optimal |

当 `num_noise=1, smoothing_std=0.0` 时，SCFDM 退化为 Optimal。

### 参数校验

```python
if self.num_noise <= 0:
    raise ValueError("num_noise must be a positive integer")
if self.smoothing_std < 0:
    raise ValueError("smoothing_std must be non-negative")
```

## 8.4 train() 方法

直接继承自 `OptimalDenoiser.train()`，构建或加载 FAISS 索引。无额外逻辑。

## 8.5 denoise() 方法

```python
@torch.no_grad()
def denoise(self, latents, timestep, *, generator=None, **kwargs):
    # 1. 生成 M 组高斯噪声
    batch_shape = (self.num_noise, *latents.shape)  # [M, B, C, H, W]
    noise = torch.randn(batch_shape, generator=generator,
                        device=latents.device, dtype=latents.dtype)

    # 2. 对 x_t 添加扰动
    smoothed_latents = latents.unsqueeze(0) + self.smoothing_std * noise  # [M, B, C, H, W]

    # 3. 展平为单个大批次
    flat_latents = smoothed_latents.reshape(-1, *latents.shape[1:])  # [M*B, C, H, W]

    # 4. 调用父类 Optimal 的 denoise
    pred_x0 = super().denoise(flat_latents, timestep, generator=generator, **kwargs)

    # 5. 恢复形状并取平均
    return pred_x0.reshape(self.num_noise, *latents.shape).mean(dim=0)  # [B, C, H, W]
```

### 步骤详解

1. 生成 `[M, B, C, H, W]` 形状的噪声
2. 广播加法：`latents.unsqueeze(0)` 形状为 `[1, B, C, H, W]`，加上噪声后得到 M 个扰动版本
3. 将 M 个扰动版本合并为一个大批次 `[M*B, C, H, W]`，一次性送入 Optimal 去噪器
4. 将结果恢复为 `[M, B, C, H, W]`，沿第 0 维取平均

**计算量**：是 Optimal 的 M 倍（FAISS 搜索量增加 M 倍）。

## 8.6 配置示例

```yaml
# configs/scfdm/celeba_hq.yaml
model:
  name: scfdm
  params:
    temperature: 1.0
    num_neighbors: 200
    num_noise: 10          # 10 组扰动
    smoothing_std: 0.1     # 扰动标准差
```

## 8.7 超参数影响

| 参数 | 增大效果 | 减小效果 |
|------|---------|---------|
| `num_noise` | 更平滑，计算量线性增加 | 更快，但平滑效果弱 |
| `smoothing_std` | 更强的平滑，可能过度模糊 | 趋向原始 Optimal |

## 8.8 继承关系

```
BaseDenoiser
    └── OptimalDenoiser
            └── SmoothedCFDM
```

SCFDM 完全复用 Optimal 的：
- `__init__` 中的 FAISS 配置
- `train()` 中的索引构建/加载
- `denoise()` 中的 softmax 加权逻辑

仅重写 `denoise()` 添加扰动-平均包装。
