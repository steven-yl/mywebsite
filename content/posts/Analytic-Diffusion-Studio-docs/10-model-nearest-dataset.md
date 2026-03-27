---
title: "Analytic Diffusion Studio — 最近邻基线"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 最近邻基线"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 最近邻基线"
---

# 10 — 最近邻基线 (Nearest Dataset)

文件：`src/local_diffusion/models/nearest_dataset.py`

## 10.1 概述

Nearest Dataset 是最简单的去噪基线：对于每个噪声图像 $x_t$，在数据集中找到欧氏距离最近的图像作为 $\hat{x}_0$。

**解决的问题**：提供一个最低复杂度的参考基线，用于衡量其他方法的改进幅度。

**直觉**：如果 Optimal 去噪器的温度 $\tau \to 0$，softmax 退化为 argmax，就得到最近邻。

## 10.2 数学公式

$$D_{\text{NN}}(x_t, t) = x_0^{(i^*)}, \quad i^* = \arg\min_i \|x_t - x_0^{(i)}\|_2$$

注意：这里直接用 $x_t$ 与 $x_0^{(i)}$ 比较，没有做 $\sqrt{\bar{\alpha}_t}$ 缩放（与 Optimal 不同）。

## 10.3 类定义

```python
@register_model("nearest_dataset")
class NearestDatasetDenoiser(BaseDenoiser):
    def __init__(self, dataset, device, num_steps, *, params=None, **kwargs):
        params = params or {}
        super().__init__(
            resolution=dataset.resolution,
            device=device,
            num_steps=num_steps,
            in_channels=dataset.in_channels,
            dataset_name=dataset.name,
            **kwargs,
        )
        self.dataset = dataset
```

无额外超参数。

## 10.4 train() 方法

```python
def train(self, dataset: DatasetBundle):
    images = []
    for batch in dataset.dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        images.append(batch)

    dataset_tensor = torch.cat(images, dim=0).contiguous().to(self.device)
    self.register_buffer("dataset_images", dataset_tensor)
    self.to(self.device)
    return self
```

将整个数据集加载到 GPU 内存，存储为 `[N, C, H, W]` 张量。

**内存需求**：
- MNIST (60k × 1 × 28 × 28)：约 150 MB
- CIFAR-10 (50k × 3 × 32 × 32)：约 600 MB
- CelebA-HQ 64×64 (30k × 3 × 64 × 64)：约 1.4 GB

## 10.5 denoise() 方法

```python
@torch.no_grad()
def denoise(self, latents, timestep, *, generator=None, **_):
    # 1. 展平
    latents_flat = latents.flatten(start_dim=1)  # [B, n]

    # 2. 计算所有距离
    distances = torch.cdist(latents_flat,
                            self.dataset_images.flatten(start_dim=1))  # [B, N]

    # 3. 找最近邻
    min_dist, indices = torch.min(distances, dim=1)  # [B]

    # 4. 返回最近邻图像
    pred_x0 = self.dataset_images[indices]  # [B, C, H, W]
    return pred_x0
```

### 步骤详解

1. 将查询和数据集图像都展平为向量
2. `torch.cdist` 计算成对 L2 距离矩阵 `[B, N]`
3. 沿数据集维度取最小值，得到最近邻索引
4. 用索引取出对应的数据集图像

**复杂度**：$O(B \cdot N \cdot n)$，其中 $N$ 是数据集大小，$n$ 是像素数。

## 10.6 配置示例

```yaml
# configs/nearest_dataset/mnist.yaml
model:
  name: nearest_dataset
  params: {}  # 无超参数
```

## 10.7 局限性

- 只能"复制"训练集中的图像，无法生成新图像
- 不考虑时间步 $t$（距离计算不依赖 $\bar{\alpha}_t$）
- 需要将整个数据集放入 GPU 内存
- `torch.cdist` 对大数据集可能内存不足（需要 `[B, N]` 距离矩阵）
- 生成多样性完全取决于初始噪声到不同数据集图像的距离

## 10.8 与 Optimal 的对比

| 特性 | Nearest | Optimal |
|------|---------|---------|
| 距离缩放 | 无 | $x_t / \sqrt{\bar{\alpha}_t}$ |
| 选择方式 | argmax（硬选择） | softmax（软加权） |
| 输出 | 单张数据集图像 | 多张图像的加权平均 |
| 时间步感知 | 否 | 是 |
| 近邻搜索 | 暴力搜索 | FAISS 加速 |
