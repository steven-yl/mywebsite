---
title: "IdealDenoiser 技术文档"
date: 2026-04-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "IdealDenoiser 技术文档"
tags: [Deep Learning]
categories: [Deep Learning]
series: [Deep Learning系列]
weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
---

## 1. 概述

`IdealDenoiser` 是一个非参数的、基于数据集的去噪模块，继承自 `torch.nn.Module` 与 `ModelMixin`。其核心思想是：给定一个带噪声的观测 `x` 和噪声水平 `σ`，利用整个训练数据集中的干净样本，通过高斯核加权平均来估计真实信号，最终输出噪声预测值 `(x - 估计的干净信号) / σ`。该模块通常用于扩散模型的基准测试、理论分析或小规模数据实验，因其直接依赖全部训练样本，故能提供近似最优的 MMSE（最小均方误差）去噪性能。

## 2. 数学原理

### 2.1 问题设定

假设干净数据$x_0$服从未知分布$p_{\text{data}}$。观测到带噪样本：
$$
x = x_0 + \sigma \,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
其中$\sigma$为已知噪声标准差。MMSE 估计器为后验条件期望：
$$
\hat{x}_0 = \mathbb{E}[x_0 \mid x] = \int x_0 \, p(x_0 \mid x) \, dx_0
$$

### 2.2 经验贝叶斯近似

由于$p_{\text{data}}$未知，我们用有限数据集$\{x_0^{(i)}\}_{i=1}^{N}$代替。根据贝叶斯公式：
$$
p(x_0^{(i)} \mid x) \propto p(x \mid x_0^{(i)}) \, p(x_0^{(i)})
$$
假设$p(x_0^{(i)}) = 1/N$（均匀先验），且似然为高斯：
$$
p(x \mid x_0^{(i)}) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp\left(-\frac{\|x - x_0^{(i)}\|^2}{2\sigma^2}\right)
$$
则后验概率正比于$\exp\left(-\frac{\|x - x_0^{(i)}\|^2}{2\sigma^2}\right)$。归一化后得到权重：
$$
w_i = \text{softmax}\left( -\frac{\|x - x_0^{(i)}\|^2}{2\sigma^2} \right)
$$
因此 MMSE 估计为：
$$
\hat{x}_0 = \sum_{i=1}^{N} w_i \, x_0^{(i)}
$$

### 2.3 输出含义

在扩散模型中，去噪器通常预测噪声$\epsilon_\theta(x, \sigma)$，并满足：
$$
x = \hat{x}_0 + \sigma \,\epsilon_\theta(x, \sigma)
$$
移项可得：
$$
\epsilon_\theta(x, \sigma) = \frac{x - \hat{x}_0}{\sigma}
$$
这正是 `IdealDenoiser` 的返回值。因此，该模块提供了一个理论上近乎最优的噪声预测器（给定经验分布）。

## 3. 代码实现详解

### 3.1 `sq_norm` 辅助函数
```python
def sq_norm(M, k):
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1, k)
```
- **输入**：`M` 形状 `(b, dim)`，`k` 为目标重复次数。
- **输出**：形状 `(b, k)`，每行均为对应样本的平方范数$\|M_i\|^2$。
- **用途**：为计算成对欧氏距离提供平方范数项。

### 3.2 `IdealDenoiser` 类

#### `__init__(self, dataset)`
```python
self.data = torch.stack([dataset[i] for i in range(len(dataset))])
self.input_dims = self.data.shape[1:]
```
- 将整个数据集加载为一个张量 `self.data`，形状 `(N, C, H, W, ...)`。
- `input_dims` 保存每个样本的空间/通道维度，用于后续形状还原。

#### `forward(self, x, sigma, cond=None)`
参数：
- `x`：带噪输入，形状 `(batch_size, *input_dims)`。
- `sigma`：噪声水平，可以是标量或形状 `(batch_size,)` 的张量。
- `cond`：未使用（保留以兼容接口）。

**执行流程：**

1. **设备对齐**  
   `data = self.data.to(x)` 将数据集移至 `x` 所在设备。

2. **展平**  
   `x_flat = x.flatten(start_dim=1)` → `(xb, xr)`  
   `d_flat = data.flatten(start_dim=1)` → `(db, dr)`  
   其中 `xb` 为 batch 大小，`db` 为数据集大小（样本数），`xr == dr` 为展平后的特征维度。

3. **平方距离矩阵计算**  
   利用恒等式$\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2 a\cdot b$：
   ```python
   sq_diffs = sq_norm(x_flat, db).T + sq_norm(d_flat, xb) - 2 * d_flat @ x_flat.T
   ```
   - `sq_norm(x_flat, db)` 形状 `(xb, db)`，其中第 `i` 行元素均为$\|x_i\|^2$。转置后为 `(db, xb)`。
   - `sq_norm(d_flat, xb)` 形状 `(db, xb)`，每行均为对应数据样本的平方范数。
   - 矩阵乘积 `d_flat @ x_flat.T` 形状 `(db, xb)`，元素为 `d_j · x_i`。
   - 最终 `sq_diffs` 形状 `(db, xb)`，其中 `sq_diffs[j, i] = ||x_i - d_j||^2`。

4. **权重计算**  
   ```python
   weights = torch.softmax(-sq_diffs / (2 * sigma.squeeze()**2), dim=0)
   ```
   - `sigma.squeeze()` 确保标量或形状 `(xb,)` 都能正确处理广播。  
     若 `sigma` 是标量，则除以同一个值；若为批向量，则对每个 batch 样本使用对应的方差。
   - 除以 `2σ²`，再经过 `dim=0` 上的 softmax，得到形状 `(db, xb)` 的权重矩阵，每列（对应一个 batch 样本）归一化和为 1。

5. **加权平均**  
   ```python
   eps = torch.einsum('ij,i...->j...', weights, data)
   ```
   - `weights` 形状 `(db, xb)`，`data` 形状 `(db, *input_dims)`。
   - Einstein 求和：对每个 batch 样本 `j`，计算$\sum_i \text{weights}[i, j] \cdot \text{data}[i]$，输出形状 `(xb, *input_dims)`。  
     该结果即为$\hat{x}_0$（估计的干净信号）。

6. **返回噪声预测**  
   ```python
   return (x - eps) / sigma
   ```
   - 形状与 `x` 相同，对应噪声$\epsilon$的估计。

## 4. 特性与限制

### 优势
- **理论最优**：当数据集足以覆盖真实分布且样本量极大时，该去噪器趋近 MMSE 估计器。
- **无训练参数**：即插即用，适合作为基线或验证工具。
- **精确计算**：对任意噪声水平 `σ` 均能给出确定性的去噪结果。

### 局限
- **存储开销**：需在内存中保存整个数据集，对于大规模数据集（如 ImageNet）不可行。
- **计算复杂度**：每 forward 需计算$O(N \cdot B \cdot D)$的矩阵乘积，其中$N$为数据集大小，$B$为 batch 大小，$D$为特征维度；随$N$线性增长。
- **分布假设**：隐含假设数据集样本为独立同分布，且先验均匀；若数据集有偏差，输出也会偏差。

## 5. 使用示例

```python
from torch.utils.data import TensorDataset

# 创建一个小型数据集
clean_data = torch.randn(1000, 3, 32, 32)  # 1000 张 32x32 彩色图
dataset = TensorDataset(clean_data)

# 实例化理想去噪器
denoiser = IdealDenoiser(dataset)

# 模拟带噪输入
x = clean_data[0:4] + 0.1 * torch.randn(4, 3, 32, 32)
sigma = torch.tensor(0.1)

# 预测噪声
noise_pred = denoiser(x, sigma)

# 恢复干净信号
clean_est = x - sigma * noise_pred
```

## 6. 引用与注意事项

- 该实现常用于扩散模型的理论分析，例如证明“最优去噪器可由数据集的后验均值给出”。
- 实际应用中，可配合子采样或随机近似（如随机选取子集计算距离）来降低开销。
- 与 `diffusers` 中的 `ModelMixin` 结合使用时，需确保兼容性（如 `from_pretrained` 等方法未被重写，但本模块无参数，影响不大）。

---