---
title: "第二章：归一化技术（TorchCode）"
date: 2026-04-01T10:00:00+08:00
draft: false
authors: [Steven]
description: "LayerNorm、BatchNorm、RMSNorm 的数学定义、训练/推理差异与实现要点。"
summary: "TorchCode 文档第二章：归一化技术全解。"

tags: [PyTorch, TorchCode]
categories: [PyTorch]
series: [TorchCode 系列]
weight: 3
series_weight: 3
hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

# 第二章：归一化技术

归一化是深度学习训练稳定性的关键。本章详解三种主流归一化方法：LayerNorm、BatchNorm、RMSNorm。

---

## 2.1 Layer Normalization

### 是什么
LayerNorm 对每个样本的特征维度进行归一化，使其均值为 0、方差为 1，然后通过可学习的缩放（γ）和偏移（β）参数恢复表达能力。

### 数学定义

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu$ 和 $\sigma^2$ 在最后一个维度上计算：

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$

注意：方差使用无偏估计（`unbiased=False`，即除以 $d$ 而非 $d-1$）。

### 为什么需要
- 解决深层网络中的**内部协变量偏移**（Internal Covariate Shift）：每层输入的分布随训练变化，导致后续层需要不断适应
- 加速收敛，允许使用更大的学习率
- 对 batch size 不敏感（与 BatchNorm 不同）

### 参数说明
- `gamma`（γ）：缩放参数，形状与最后一维相同，初始化为 1
- `beta`（β）：偏移参数，形状与最后一维相同，初始化为 0
- `eps`：防止除零的小常数，通常为 1e-5

### 代码示例

```python
import torch

def my_layer_norm(x: torch.Tensor, gamma: torch.Tensor,
                  beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta

# 测试
x = torch.randn(2, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = my_layer_norm(x, gamma, beta)

print("均值:", out.mean(dim=-1))   # 接近 0
print("标准差:", out.std(dim=-1))  # 接近 1

# 与 PyTorch 对比
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
print("匹配:", torch.allclose(out, ref, atol=1e-4))
```

### 适用场景
- Transformer 的标准归一化方法（Pre-Norm 或 Post-Norm）
- RNN/LSTM 中也可使用
- 不依赖 batch 统计量，适合小 batch 或变长序列

---

## 2.2 Batch Normalization

### 是什么
BatchNorm 对每个特征在 batch 维度上进行归一化。训练时使用当前 batch 的统计量，推理时使用训练过程中累积的 running statistics。

### 数学定义

$$\text{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

其中 $\mu_B$ 和 $\sigma_B^2$ 在 batch 维度（dim=0）上计算。

### 训练 vs 推理

| 阶段 | 均值/方差来源 | running stats 操作 |
|------|-------------|-------------------|
| 训练 | 当前 batch 统计量 | 指数移动平均更新 |
| 推理 | running_mean / running_var | 只读，不更新 |

Running statistics 更新规则：
```
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var  = (1 - momentum) * running_var  + momentum * batch_var
```

### 为什么需要
- 2015 年由 Ioffe & Szegedy 提出，极大加速了 CNN 训练
- 允许更高的学习率
- 有轻微的正则化效果（因为 batch 统计量引入了噪声）

### 与 LayerNorm 的关键区别
- BatchNorm 在 batch 维度归一化，LayerNorm 在特征维度归一化
- BatchNorm 依赖 batch size（小 batch 时统计量不稳定）
- BatchNorm 需要维护 running statistics（推理时使用）
- BatchNorm 在 CNN 中效果好，但在 Transformer/NLP 中 LayerNorm 更优

### 代码示例

```python
import torch

def my_batch_norm(x, gamma, beta, running_mean, running_var,
                  eps=1e-5, momentum=0.1, training=True):
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        # 原地更新 running statistics（detach 防止梯度流入）
        running_mean.mul_(1 - momentum).add_(batch_mean.detach() * momentum)
        running_var.mul_(1 - momentum).add_(batch_var.detach() * momentum)
    else:
        batch_mean = running_mean
        batch_var = running_var

    return gamma * (x - batch_mean) / torch.sqrt(batch_var + eps) + beta

# 测试
x = torch.randn(8, 4)  # 8 个样本，4 个特征
gamma = torch.ones(4)
beta = torch.zeros(4)
running_mean = torch.zeros(4)
running_var = torch.ones(4)

# 训练模式
out = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
print("列均值:", out.mean(dim=0))  # 接近 0
print("running_mean 已更新:", running_mean)

# 推理模式
out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
print("推理输出形状:", out_eval.shape)
```

### 关键细节
- `running_mean` 和 `running_var` 是 buffer（不需要梯度），不是可学习参数
- 方差使用 `unbiased=False`（除以 N 而非 N-1）
- `momentum=0.1` 是 PyTorch 默认值，表示新 batch 统计量的权重

### 适用场景
- CNN（ResNet、VGG 等）的标准配置
- 不适合 batch size 很小的场景
- 不适合序列长度变化的 NLP 任务

---

## 2.3 RMSNorm（Root Mean Square Normalization）

### 是什么
RMSNorm 是 LayerNorm 的简化版本，去掉了均值减除步骤，只用均方根（RMS）进行缩放。

### 数学定义

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot w$$

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

### 与 LayerNorm 的区别
- 不减均值（no mean subtraction）
- 不加偏移（no beta/shift parameter）
- 只有一个可学习参数 `weight`（对应 LayerNorm 的 γ）
- 计算量更少（省去了均值计算和减法）

### 为什么有效
论文 [Zhang & Sennrich, 2019] 发现，LayerNorm 的成功主要归功于缩放不变性（re-scaling invariance），而非重新中心化（re-centering）。因此去掉均值减除不影响效果，还能提升效率。

### 代码示例

```python
import torch

def rms_norm(x: torch.Tensor, weight: torch.Tensor,
             eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight

# 测试
x = torch.randn(2, 8)
w = torch.ones(8)
out = rms_norm(x, w)
print("输出形状:", out.shape)
print("RMS of output:", out.pow(2).mean(dim=-1).sqrt())  # 接近 1
```

### 适用场景
- LLaMA、LLaMA 2、Gemma、Mistral 等现代 LLM
- 任何追求计算效率的 Transformer 变体
- 通常 eps=1e-6（比 LayerNorm 的 1e-5 更小）

---

## 2.4 三种归一化方法总结对比

```
输入张量 x: (Batch, Sequence, Features)

LayerNorm:  对每个 (batch, seq) 位置的 Features 维度归一化
            ┌─────────────────┐
            │ ████████████████ │ ← 这一行归一化
            │ ████████████████ │
            └─────────────────┘

BatchNorm:  对每个 Feature 在 Batch 维度上归一化
            ┌─────────────────┐
            │ █               │
            │ █               │ ← 这一列归一化
            │ █               │
            └─────────────────┘

RMSNorm:   同 LayerNorm 的维度，但不减均值
            ┌─────────────────┐
            │ ████████████████ │ ← 只除以 RMS，不减均值
            │ ████████████████ │
            └─────────────────┘
```

| 特性 | LayerNorm | BatchNorm | RMSNorm |
|------|-----------|-----------|---------|
| 归一化维度 | 特征维 | 批次维 | 特征维 |
| 减均值 | ✅ | ✅ | ❌ |
| 可学习参数 | γ, β | γ, β | weight |
| Running stats | ❌ | ✅ | ❌ |
| 依赖 batch size | ❌ | ✅ | ❌ |
| 计算开销 | 中 | 中 | 低 |
| 典型用途 | Transformer | CNN | 现代 LLM |
