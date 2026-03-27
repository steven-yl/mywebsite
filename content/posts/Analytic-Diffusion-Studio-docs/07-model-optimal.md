---
title: "Analytic Diffusion Studio — 最优贝叶斯去噪器"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 最优贝叶斯去噪器"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 最优贝叶斯去噪器"
---

# 07 — 最优贝叶斯去噪器 (Optimal)

文件：`src/local_diffusion/models/optimal.py`

## 7.1 概述

Optimal 去噪器实现了贝叶斯最优估计——后验均值 $\mathbb{E}[x_0 | x_t]$。它不做任何分布假设，直接对数据集中的所有图像做 softmax 加权平均。

**解决的问题**：在给定有限数据集的条件下，计算理论上最优的去噪估计。

**核心思想**：将数据集视为经验分布 $p(x_0) = \frac{1}{N}\sum_i \delta(x_0 - x_0^{(i)})$，则后验均值变为 softmax 加权平均。

## 7.2 数学公式

贝叶斯最优去噪器：

$$D^*(x_t, t) = \frac{\sum_{i=1}^{N} x_0^{(i)} \cdot w_i(x_t, t)}{\sum_{i=1}^{N} w_i(x_t, t)}$$

其中权重为：

$$w_i(x_t, t) = \exp\left(-\frac{\bar{\alpha}_t \|x_t / \sqrt{\bar{\alpha}_t} - x_0^{(i)}\|^2}{2(1-\bar{\alpha}_t) \cdot \tau}\right)$$

- $\tau$ 是温度参数（`temperature`），$\tau = 1$ 时为标准贝叶斯最优
- 分子中 $x_t / \sqrt{\bar{\alpha}_t}$ 是对 $x_t$ 的缩放，使其与 $x_0$ 在同一尺度

## 7.3 FAISS 加速

直接遍历所有 N 个数据点计算权重代价太高。本实现使用 FAISS 库进行近似最近邻搜索，只对 top-k 个近邻计算权重。

### FAISS 索引构建

```python
if num_images > 1_000_000:
    # 大数据集：使用 IVF（倒排文件）索引
    nlist = min(4096, num_images // 39)
    quantizer = faiss.IndexFlatL2(self.dim)
    index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
    index.train(train_data)  # 需要训练聚类中心
else:
    # 小数据集：精确搜索
    index = faiss.IndexFlatL2(self.dim)

index.add(dataset_images.numpy().astype(np.float32))
```

- `IndexFlatL2`：暴力 L2 距离搜索，精确但慢
- `IndexIVFFlat`：基于倒排文件的近似搜索，先找到最近的聚类，再在聚类内搜索

### macOS 兼容性

```python
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    faiss.omp_set_num_threads(1)  # 避免 fork 安全问题
```

## 7.4 类定义

```python
@register_model("optimal")
class OptimalDenoiser(BaseDenoiser):
    def __init__(self, dataset, device, num_steps, *, params=None, **kwargs):
```

### 构造函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `params.index_path` | `None` | FAISS 索引存储路径 |
| `params.temperature` | `1.0` | softmax 温度参数 |
| `params.num_neighbors` | `2000` | 近邻搜索数量 k |

默认索引路径：`data/models/optimal/{dataset_name}_{resolution}`

## 7.5 train() 方法

```python
def train(self, dataset: DatasetBundle):
    try:
        self.faiss_index, self.dataset_images, self.dim = load_optimal_index(self.index_path)
    except FileNotFoundError:
        # 1. 遍历数据集，展平为 [N, n_pixels]
        # 2. 构建 FAISS 索引
        # 3. 保存索引和数据到磁盘
```

保存的文件：
- `index.index`：FAISS 索引文件
- `data.pt`：包含 `dataset_images` 张量和 `dim` 维度信息

## 7.6 denoise() 方法

```python
@torch.no_grad()
def denoise(self, latents, timestep, *, generator=None, **kwargs):
    # 1. 获取调度器参数
    alpha_prod_t = self.scheduler.alphas_cumprod[timestep_index]
    beta_prod_t = 1 - alpha_prod_t

    # 2. 缩放查询向量：x_scaled = x_t / √ᾱ_t
    latents_scaled = latents / torch.sqrt(alpha_prod_t)
    latents_flat = latents_scaled.flatten(start_dim=1)

    # 3. FAISS 近邻搜索
    query_vectors = latents_flat.cpu().numpy().astype(np.float32)
    k = min(self.num_neighbors, self.dataset_images.shape[0])
    distances_np, indices_np = self.faiss_index.search(query_vectors, k)

    # 4. 缩放距离（补偿查询缩放）
    scaled_distances = distances * alpha_prod_t

    # 5. 获取近邻图像
    neighbor_images = self.dataset_images[indices].to(self.device)

    # 6. 计算 softmax 权重
    logits = -scaled_distances / (2 * beta_prod_t * self.temperature)
    weights = torch.softmax(logits, dim=1)

    # 7. 加权平均
    pred_x0_flat = torch.bmm(weights.unsqueeze(1), neighbor_images).squeeze(1)
    pred_x0 = pred_x0_flat.view_as(latents)
    return pred_x0
```

### 步骤详解

1. **缩放查询**：将 $x_t$ 除以 $\sqrt{\bar{\alpha}_t}$，使查询向量与数据集图像在同一尺度。这等价于在原始空间中比较 $x_t$ 与 $\sqrt{\bar{\alpha}_t} x_0^{(i)}$。

2. **FAISS 搜索**：在缩放后的空间中找到 k 个最近邻。FAISS 返回 L2 距离和索引。

3. **距离修正**：FAISS 返回的距离是缩放空间中的，乘以 $\bar{\alpha}_t$ 恢复到原始空间。

4. **softmax 权重**：$\text{logit}_i = -\frac{\bar{\alpha}_t d_i^2}{2(1-\bar{\alpha}_t)\tau}$，然后 softmax 归一化。

5. **加权平均**：`torch.bmm` 实现批量矩阵乘法 `[B, 1, k] × [B, k, n] → [B, 1, n]`。

## 7.7 辅助函数

### save_optimal_index()

```python
def save_optimal_index(faiss_index, dataset_images, save_path, dim):
    faiss.write_index(faiss_index, str(save_path / "index.index"))
    torch.save({"data": dataset_images.cpu(), "dim": dim}, save_path / "data.pt")
```

### load_optimal_index()

```python
def load_optimal_index(load_path):
    faiss_index = faiss.read_index(str(load_path / "index.index"))
    saved_data = torch.load(load_path / "data.pt", weights_only=True)
    return faiss_index, saved_data["data"], saved_data["dim"]
```

## 7.8 配置示例

```yaml
# configs/optimal/cifar10.yaml
model:
  name: optimal
  params:
    temperature: 1.0
    num_neighbors: 200
```

## 7.9 温度参数的影响

| 温度 τ | 效果 |
|--------|------|
| τ → 0 | 退化为最近邻（硬选择） |
| τ = 1 | 标准贝叶斯最优 |
| τ → ∞ | 所有权重趋于均匀（输出趋向全局均值） |

## 7.10 局限性

- 需要将整个数据集的展平图像存储在内存中
- FAISS 搜索需要 CPU-GPU 数据传输
- 生成图像是数据集图像的加权平均，可能模糊
- k 值选择影响精度和速度的权衡
