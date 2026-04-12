---
title: "Analytic Diffusion Studio — PCA Locality 去噪器"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — PCA Locality 去噪器"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 9
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — PCA Locality 去噪器"
---

# 09 — PCA Locality 去噪器

文件：`src/local_diffusion/models/pca_locality.py`

论文：[Locality in Image Diffusion Models Emerges from Data Statistics](https://arxiv.org/abs/2509.09672)

## 9.1 概述

PCA Locality 是本项目的核心创新方法。它发现扩散模型的去噪操作具有空间局部性——这种局部性不是人为设计的（如卷积核），而是从数据的协方差结构中自然涌现的。

**核心发现**：Wiener 滤波矩阵 $L_t L_t^T$ 的结构揭示了像素间的去噪依赖关系。远离的像素对之间的依赖权重接近零，形成类似"感受野"的局部模式。

**方法**：将这种局部性显式编码为二值掩码，用于修改 Optimal 去噪器中的距离度量，使每个像素只关注其局部邻域。

## 9.2 数学公式

### 标准 Optimal 去噪器的距离

$$d_i(x_t) = \|x_t - \sqrt{\bar{\alpha}_t} x_0^{(i)}\|^2 = \sum_n (x_t^{(n)} - \sqrt{\bar{\alpha}_t} x_0^{(i,n)})^2$$

这是全局 L2 距离，所有像素等权参与。

### PCA Locality 的局部距离

$$d_i^{\text{local}}(x_t) = \sum_n M_{nm} \cdot (x_t^{(n)} - \sqrt{\bar{\alpha}_t} x_0^{(i,n)})^2$$

其中 $M$ 是从 Wiener 滤波矩阵导出的二值掩码。掩码 $M_{nm}$ 表示像素 $n$ 的去噪是否依赖于像素 $m$。

### 掩码构造

1. 计算 Wiener 收缩矩阵：$L_t L_t^T = U \cdot \text{diag}(\text{shrink}) \cdot V^H$
2. 行归一化：$M_{nm} = \frac{(L_t L_t^T)_{nm}}{(L_t L_t^T)_{nn}}$
3. 二值化：$M_{nm} = \mathbb{1}[|M_{nm}| \geq \theta \cdot \max|M|]$

其中 $\theta$ 是 `mask_threshold` 参数。

### 最终去噪公式

$$D_{\text{PCA}}(x_t, t) = \frac{\sum_i x_0^{(i)} \cdot \exp\left(-\frac{d_i^{\text{local}}(x_t)}{2(1-\bar{\alpha}_t)\tau}\right)}{\sum_i \exp\left(-\frac{d_i^{\text{local}}(x_t)}{2(1-\bar{\alpha}_t)\tau}\right)}$$

注意：这里的 softmax 是**逐像素**的——每个像素 $n$ 有自己的权重分布（因为掩码 $M$ 的每一行不同）。

## 9.3 WeightedStreamingSoftmax 类

由于数据集可能很大，无法一次性加载所有图像计算 softmax。本实现使用流式算法，逐批处理数据集。

```python
class WeightedStreamingSoftmax:
    """
    流式加权 softmax 平均。
    参见论文 Appendix C3 中的 WSSM 算法。
    """

    def __init__(self, *, device=None, dtype=torch.float32, eps=1e-8):
        self.sum_weighted = None   # 加权和 [B, n]
        self.sum_weights = None    # 权重和 [B, n]（注意：逐像素）
```

### add() 方法

```python
def add(self, x0b: torch.Tensor, logits: torch.Tensor):
    """
    添加一批数据集图像的贡献。

    参数：
        x0b: 数据集图像批次 [k, n]（k 是批大小，n 是像素数）
        logits: 对数权重 [B, k, n]（B 是查询批大小）
    """
    b, k, n = logits.shape

    # 数值稳定的 softmax（减去最大值）
    logits_max, _ = logits.max(dim=1, keepdim=True)
    logits_exp = torch.exp(logits - logits_max)
    weights = logits_exp / logits_exp.sum(dim=1, keepdim=True)  # [B, k, n]

    # 加权和：einsum("bkn,kn->bn")
    weighted_sum = torch.einsum("bkn,kn->bn", weights, x0b)    # [B, n]
    weight_sum = weights.sum(dim=1)                              # [B, n]

    # 累加
    if self.sum_weighted is None:
        self.sum_weighted = weighted_sum
        self.sum_weights = weight_sum
    else:
        self.sum_weighted += weighted_sum
        self.sum_weights += weight_sum
```

**关键点**：
- `logits` 的形状是 `[B, k, n]`，第三维 `n` 表示每个像素有独立的权重
- `einsum("bkn,kn->bn")` 对每个查询样本 b，将 k 个数据集图像按像素级权重加权求和
- 流式累加 `sum_weighted` 和 `sum_weights`

### get_average() 方法

```python
def get_average(self):
    if self.sum_weighted is None:
        return None
    return self.sum_weighted / (self.sum_weights + self.eps)
```

返回最终的加权平均结果 `[B, n]`。

**注意**：这个流式 softmax 是近似的——它在每个批次内做局部 softmax 归一化，然后累加。严格来说，全局 softmax 需要知道所有 logits 的最大值。但在实践中，这种近似效果良好。

## 9.4 类定义

```python
@register_model("pca_locality")
class PCALocalityDenoiser(BaseDenoiser):
    def __init__(self, dataset, device, num_steps, *, params=None, **kwargs):
```

### 构造函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `params.temperature` | `1.0` | softmax 温度 τ |
| `params.mask_threshold` | `0.02` | 掩码二值化阈值 θ |
| `params.wiener_path` | `None` | Wiener SVD 存储路径 |

默认 Wiener 路径：`data/models/wiener/{dataset}_{resolution}`（与 Wiener 模型共享）。

## 9.5 train() 方法

```python
def train(self, dataset: DatasetBundle):
    try:
        U, LA, Vh, mean = load_wiener_filter(self.wiener_path, device=self.device)
    except FileNotFoundError:
        S, mean = compute_wiener_filter(
            dataloader=dataset.dataloader, device=self.device,
            resolution=self.resolution, n_channels=self.n_channels,
        )
        U, LA, Vh = torch.linalg.svd(S)
        save_wiener_filter(U, LA, Vh, mean, self.wiener_path)

    self.register_buffer("U", U.to(self.device))
    self.register_buffer("LA", LA.to(self.device))
    self.register_buffer("Vh", Vh.to(self.device))
    self.register_buffer("mean", mean.to(self.device))
    self.dataset = dataset  # 保留数据集引用（流式遍历用）
```

与 Wiener 模型的 `train()` 几乎相同，但额外保留了 `self.dataset` 引用，因为 `denoise()` 需要流式遍历数据集。

## 9.6 _projection_mask() 方法

```python
def _projection_mask(self, timestep_index):
    alpha_prod_t = self.scheduler.alphas_cumprod[timestep_index]
    beta_prod_t = 1 - alpha_prod_t

    # 1. 计算收缩因子
    shrink_factors = alpha_prod_t * self.LA / (beta_prod_t + alpha_prod_t * self.LA)
    LAshrink = torch.diag(shrink_factors)

    # 2. 构造 LLᵀ 矩阵
    LLt = self.U @ LAshrink @ self.Vh  # [n, n]

    # 3. 行归一化
    denom = torch.diagonal(LLt).unsqueeze(1)
    denom[denom.abs() < self.eps] = 1.0
    mask = LLt / denom

    # 4. 二值化
    if self.mask_threshold > 0:
        threshold = mask.abs().max() * self.mask_threshold
        mask = torch.where(mask.abs() >= threshold,
                          torch.ones_like(mask), torch.zeros_like(mask))

    return mask, alpha_prod_t, beta_prod_t
```

### 步骤详解

1. **收缩因子**：与 Wiener 相同，$\text{shrink}_i = \frac{\bar{\alpha}_t \lambda_i}{(1-\bar{\alpha}_t) + \bar{\alpha}_t \lambda_i}$
2. **LLᵀ 矩阵**：$n \times n$ 矩阵，$(i,j)$ 元素表示像素 $i$ 对像素 $j$ 的去噪依赖强度
3. **行归一化**：使对角线元素为 1（自身依赖归一化）
4. **二值化**：绝对值低于 $\theta \cdot \max|M|$ 的元素置零，其余置一

**掩码的物理意义**：掩码的第 $n$ 行表示像素 $n$ 的"感受野"——哪些像素参与了像素 $n$ 的去噪。

## 9.7 denoise() 方法

```python
@torch.no_grad()
def denoise(self, latents, timestep, *, generator=None, **kwargs):
    t_idx = int(timestep.item())
    mask, alpha_prod_t, beta_prod_t = self._projection_mask(t_idx)
    sqrt_alpha = torch.sqrt(alpha_prod_t)

    xt = latents.flatten(start_dim=1)  # [B, n]
    first_moment = WeightedStreamingSoftmax(device=latents.device, dtype=latents.dtype)

    # 流式遍历数据集
    for x0_batch in tqdm(self.dataset.dataloader, desc="PCA locality", leave=False):
        images = x0_batch[0] if isinstance(x0_batch, (tuple, list)) else x0_batch
        x0b = images.to(latents.device).flatten(start_dim=1)  # [k, n]

        # 1. 逐像素平方差
        delta = (xt.unsqueeze(1) - sqrt_alpha * x0b.unsqueeze(0)) ** 2  # [B, k, n]

        # 2. 应用掩码（矩阵乘法）
        ds_chunk = torch.einsum("bkn,nm->bkm", delta, mask)  # [B, k, n]

        # 3. 计算 logits
        logits = -ds_chunk / (2 * beta_prod_t * self.temperature)  # [B, k, n]

        # 4. 流式累加
        first_moment.add(x0b, logits)

    # 5. 获取最终平均
    x0_mean = first_moment.get_average()  # [B, n]
    pred_x0 = x0_mean.view_as(latents)
    return pred_x0
```

### 步骤详解

1. **逐像素平方差** `delta[b, k, n]`：查询 $b$ 与数据集图像 $k$ 在像素 $n$ 上的平方差
   - `xt.unsqueeze(1)`：`[B, 1, n]`
   - `x0b.unsqueeze(0)`：`[1, k, n]`
   - 广播后得到 `[B, k, n]`

2. **掩码投影** `ds_chunk[b, k, m]`：对每个像素 $m$，将其感受野内的平方差加权求和
   - `einsum("bkn,nm->bkm")`：对 $n$ 维求和，$M_{nm}$ 作为权重
   - 结果：像素 $m$ 的局部距离

3. **logits**：$-\frac{d^{\text{local}}}{2(1-\bar{\alpha}_t)\tau}$

4. **流式累加**：通过 `WeightedStreamingSoftmax` 逐批累加

5. **最终输出**：加权平均的结果

### 计算复杂度

每个数据集批次：
- `delta` 计算：$O(B \cdot k \cdot n)$
- `einsum` 掩码投影：$O(B \cdot k \cdot n^2)$（瓶颈）
- 总计：$O(B \cdot N \cdot n^2)$，其中 $N$ 是数据集大小

## 9.8 配置示例

```yaml
# configs/pca_locality/celeba_hq.yaml
dataset:
  resolution: 64  # 降低分辨率以控制 n² 复杂度

model:
  name: pca_locality
  params:
    temperature: 1.0
    mask_threshold: 0.02   # 2% 阈值
```

### 不同数据集的 mask_threshold

| 数据集 | mask_threshold | 说明 |
|--------|---------------|------|
| MNIST | 0.005 | 更低阈值（图像结构简单，需要更大感受野） |
| Fashion-MNIST | 0.005 | 同上 |
| CIFAR-10 | 0.05 | 中等阈值 |
| CelebA-HQ | 0.02 | 较低阈值（人脸需要较大感受野） |
| AFHQ | 0.02 | 同 CelebA-HQ |

## 9.9 与其他方法的关系

```
Wiener 滤波器
    │
    │ 提取 LLᵀ 矩阵 → 构造局部性掩码
    │
    ▼
PCA Locality = 局部性掩码 + Optimal 去噪器的 softmax 加权
    │
    │ 如果掩码 = 全 1 矩阵（无局部性）
    ▼
Optimal 去噪器（全局距离）
```

PCA Locality 可以看作 Wiener（线性、全局）和 Optimal（非线性、全局）的结合：
- 从 Wiener 借用局部性结构
- 从 Optimal 借用非线性 softmax 加权

## 9.10 局限性

- 每个去噪步都需要遍历整个数据集，推理速度慢
- 掩码投影的 $O(n^2)$ 复杂度限制了分辨率（通常降至 64×64）
- 流式 softmax 是近似的，可能引入误差
- 掩码阈值需要针对不同数据集调优
