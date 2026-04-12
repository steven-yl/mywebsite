---
title: "smalldiffusion 数据模块：data.py"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 数据模块：data.py"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 3
hiddenFromHomePage: false
hiddenFromSearch: false
---

> 本文件提供数据集工具函数和三个 2D 玩具数据集，用于快速验证扩散模型的正确性。

## 3.1 模块结构

```
data.py
├── MappedDataset          # 数据集映射包装器
├── img_train_transform    # 图像训练预处理
├── img_normalize          # 图像反归一化
├── Swissroll              # 瑞士卷数据集
├── DatasaurusDozen        # Datasaurus 数据集
├── interpolate_polyline() # 多段线插值辅助函数
└── TreeDataset            # 树形条件数据集
```

---

## 3.2 MappedDataset

### 是什么

一个通用的数据集包装器，对原始数据集的每个元素应用一个映射函数。

```python
class MappedDataset(Dataset):
    def __init__(self, dataset, fn):
        self.dataset = dataset
        self.fn = fn
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.fn(self.dataset[i])
```

### 为什么需要

PyTorch 的标准图像数据集（如 `FashionMNIST`, `CIFAR10`）返回 `(image, label)` 元组。但无条件扩散模型的训练只需要图像数据，不需要标签。`MappedDataset` 提供了一种简洁的方式来丢弃标签或做其他变换。

### 使用示例

```python
from torchvision.datasets import FashionMNIST
from smalldiffusion import MappedDataset, img_train_transform

# 丢弃标签，只保留图像
dataset = MappedDataset(
    FashionMNIST('datasets', train=True, download=True, transform=img_train_transform),
    lambda x: x[0]  # x 是 (image, label)，只取 image
)
# dataset[0] 现在直接返回图像张量，而非 (image, label) 元组
```

---

## 3.3 img_train_transform

### 是什么

用于图像扩散模型训练的标准预处理管道。

```python
img_train_transform = tf.Compose([
    tf.RandomHorizontalFlip(),              # 随机水平翻转（数据增强）
    tf.ToTensor(),                          # PIL Image → Tensor, 值域 [0, 1]
    tf.Lambda(lambda t: (t * 2) - 1)        # 归一化到 [-1, 1]
])
```

### 为什么归一化到 [-1, 1]

扩散模型假设数据分布的均值接近 0。将像素值从 `[0, 1]` 映射到 `[-1, 1]` 使数据中心化，有助于训练稳定性。

---

## 3.4 img_normalize

### 是什么

将模型输出从 `[-1, 1]` 反归一化回 `[0, 1]` 用于可视化。

```python
img_normalize = lambda x: ((x + 1)/2).clamp(0, 1)
```

### 使用示例

```python
from torchvision.utils import save_image, make_grid
from smalldiffusion import img_normalize

# 采样后反归一化并保存
*xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6, batchsize=64)
save_image(img_normalize(make_grid(x0)), 'samples.png')
```

---

## 3.5 Swissroll 数据集

### 是什么

经典的 2D 瑞士卷数据集，点沿螺旋线分布。

```python
class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N, center=(0,0), scale=1.0):
        t = tmin + torch.linspace(0, 1, N) * tmax
        center = torch.tensor(center).unsqueeze(0)
        self.vals = center + scale * torch.stack([
            t * torch.cos(t) / tmax,
            t * torch.sin(t) / tmax
        ]).T

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `tmin` | `float` | 螺旋起始角度（弧度） |
| `tmax` | `float` | 螺旋终止角度（弧度） |
| `N` | `int` | 数据点数量 |
| `center` | `tuple` | 螺旋中心坐标，默认 (0, 0) |
| `scale` | `float` | 缩放因子，默认 1.0 |

### 数学原理

参数方程：

$$x(t) = \frac{t \cos(t)}{t_{\max}}, \quad y(t) = \frac{t \sin(t)}{t_{\max}}$$

其中 $t$ 在 $[t_{\min}, t_{\min} + t_{\max}]$ 上均匀采样。除以 $t_{\max}$ 使数据归一化到合理范围。

### 使用示例

```python
import numpy as np
from torch.utils.data import DataLoader
from smalldiffusion import Swissroll

dataset = Swissroll(np.pi/2, 5*np.pi, 100)
print(f"数据点数: {len(dataset)}")       # 100
print(f"数据维度: {dataset[0].shape}")   # torch.Size([2])

loader = DataLoader(dataset, batch_size=2048)
```

---

## 3.6 DatasaurusDozen 数据集

### 是什么

加载 [Datasaurus Dozen](https://www.research.autodesk.com/publications/same-stats-different-graphs/) 数据集中的指定子集。这是一组统计特征相同但形状完全不同的 2D 数据集。

```python
class DatasaurusDozen(Dataset):
    def __init__(self, csv_file, dataset, enlarge_factor=15,
                 delimiter='\t', scale=50, offset=50):
        self.enlarge_factor = enlarge_factor
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    point = torch.tensor(list(map(float, rest)))
                    self.points.append((point - offset) / scale)

    def __len__(self):
        return len(self.points) * self.enlarge_factor

    def __getitem__(self, i):
        return self.points[i % len(self.points)]
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `csv_file` | `str` | TSV 文件路径 |
| `dataset` | `str` | 子数据集名称（如 `"dino"`, `"star"` 等） |
| `enlarge_factor` | `int` | 数据重复倍数，默认 15 |
| `scale` | `float` | 缩放因子，默认 50 |
| `offset` | `float` | 偏移量，默认 50 |

### 设计细节

- `enlarge_factor`：原始数据点较少（约 142 个），通过重复扩大数据集，使 DataLoader 的 batch 采样更有效
- `(point - offset) / scale`：将数据中心化并缩放到 [-1, 1] 附近
- `__getitem__` 使用取模运算实现循环访问

### 使用示例

```python
from smalldiffusion import DatasaurusDozen

dataset = DatasaurusDozen('datasets/DatasaurusDozen.tsv', 'dino')
print(f"数据点数: {len(dataset)}")       # 142 * 15 = 2130
print(f"数据维度: {dataset[0].shape}")   # torch.Size([2])
```

---

## 3.7 interpolate_polyline 辅助函数

### 是什么

沿多段线（polyline）均匀采样点的工具函数，被 `TreeDataset` 使用。

```python
def interpolate_polyline(points, num_samples):
    points = np.array(points)
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)  # 相邻点距离
    cumdist = np.concatenate(([0], np.cumsum(dists)))          # 累积弧长
    total_length = cumdist[-1]
    sample_dists = np.linspace(0, total_length, num_samples)   # 均匀弧长采样点
    samples = []
    for d in sample_dists:
        seg = np.searchsorted(cumdist, d, side='right') - 1
        seg = min(seg, len(dists) - 1)
        t = (d - cumdist[seg]) / dists[seg] if dists[seg] > 0 else 0
        sample = (1 - t) * points[seg] + t * points[seg + 1]
        samples.append(sample)
    return np.array(samples)
```

### 工作原理

1. 计算相邻点之间的欧氏距离
2. 计算累积弧长
3. 在总弧长上均匀取 `num_samples` 个位置
4. 对每个位置，找到所在线段并线性插值

---

## 3.8 TreeDataset

### 是什么

一个带标签的 2D 条件数据集，数据点沿树形结构分布。每个叶节点对应一个类别，用于条件扩散模型的训练和 Classifier-Free Guidance 的演示。

```python
class TreeDataset(Dataset):
    def __init__(self, branching_factor=4, depth=3, num_samples_per_path=30):
        self.data = []
        self.total_leaves = branching_factor ** depth
        for i in range(self.total_leaves):
            path_points = [np.array([0.0, 0.0])]  # 根节点
            for l in range(1, depth + 1):
                group_size = branching_factor ** (depth - l)
                A_l = i // group_size
                avg_index = A_l * group_size + (group_size - 1) / 2.0
                theta = avg_index * (2 * np.pi / self.total_leaves)
                r = l / depth
                p = np.array([r * np.cos(theta), r * np.sin(theta)])
                path_points.append(p)
            samples = interpolate_polyline(path_points, num_samples_per_path)
            for sample in samples:
                self.data.append((torch.tensor(sample, dtype=torch.float32), i))
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `branching_factor` | `int` | 4 | 每个节点的分支数 |
| `depth` | `int` | 3 | 树的深度 |
| `num_samples_per_path` | `int` | 30 | 每条路径上的采样点数 |

### 数据结构

- 总叶节点数 = `branching_factor ** depth`（默认 64）
- 每个叶节点有一条从根 (0,0) 到单位圆上某点的路径
- 路径经过 `depth` 个中间节点，每个节点的角度由其子树的平均叶节点位置决定
- 每条路径上均匀采样 `num_samples_per_path` 个点
- 总数据量 = `total_leaves * num_samples_per_path`

### 返回格式

与其他数据集不同，`TreeDataset.__getitem__` 返回 `(coordinate, label)` 元组：
- `coordinate`: `torch.FloatTensor`，形状 `[2]`
- `label`: `int`，叶节点索引（类别标签）

这种格式直接支持条件训练（`training_loop` 的 `conditional=True`）。

### 使用示例

```python
from torch.utils.data import DataLoader
from smalldiffusion import TreeDataset, ConditionalMLP, ScheduleLogLinear, training_loop

dataset = TreeDataset(branching_factor=4, depth=3)
print(f"总叶节点: {dataset.total_leaves}")  # 64
print(f"总数据点: {len(dataset)}")           # 64 * 30 = 1920

loader = DataLoader(dataset, batch_size=512, shuffle=True)
batch, labels = next(iter(loader))
print(f"数据形状: {batch.shape}")    # [512, 2]
print(f"标签形状: {labels.shape}")   # [512]

# 条件训练
model = ConditionalMLP(dim=2, num_classes=dataset.total_leaves)
schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10)
trainer = training_loop(loader, model, schedule, epochs=100, conditional=True)
losses = [ns.loss.item() for ns in trainer]
```
