---
title: "第五章：训练与优化（TorchCode）"
date: 2026-04-01T10:00:00+08:00
draft: false
authors: [Steven]
description: "Kaiming 初始化、Adam、余弦学习率与 Warmup、梯度裁剪与累积、线性回归等训练工程。"
summary: "TorchCode 文档第五章：训练与优化实践。"

tags: [PyTorch, TorchCode]
categories: [PyTorch]
series: [TorchCode 系列]
weight: 6
series_weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

# 第五章：训练与优化

本章涵盖模型训练全流程中的关键技术：权重初始化、优化器、学习率调度、梯度管理和线性回归。

---

## 5.1 Kaiming Initialization（He 初始化）

### 是什么
Kaiming 初始化是针对 ReLU 激活函数设计的权重初始化方法，确保前向传播和反向传播中信号方差保持稳定。

### 数学定义

$$W \sim \mathcal{N}(0, \text{std}^2), \quad \text{std} = \sqrt{\frac{2}{\text{fan\_in}}}$$

其中 `fan_in = weight.shape[1]`（输入特征数）。

### 为什么需要
- 初始化过小：信号逐层衰减，深层梯度消失
- 初始化过大：信号逐层放大，梯度爆炸
- Kaiming 初始化考虑了 ReLU 会将一半激活置零的特性（因此分子是 2 而非 1）

### 与 Xavier 初始化的区别
- Xavier：$\text{std} = \sqrt{1/\text{fan\_in}}$，适合 Sigmoid/Tanh
- Kaiming：$\text{std} = \sqrt{2/\text{fan\_in}}$，适合 ReLU（因为 ReLU 丢弃了一半信号）

### 代码示例

```python
import torch
import math

def kaiming_init(weight: torch.Tensor) -> torch.Tensor:
    fan_in = weight.shape[1]
    std = math.sqrt(2.0 / fan_in)
    weight.data = torch.randn_like(weight) * std
    return weight

# 测试
w = torch.empty(256, 512)
kaiming_init(w)
print(f"均值: {w.mean():.4f} (期望 ~0)")
print(f"标准差: {w.std():.4f} (期望 {math.sqrt(2/512):.4f})")
```

---

## 5.2 Adam Optimizer

### 是什么
Adam（Adaptive Moment Estimation）是最广泛使用的优化器，结合了动量（Momentum）和自适应学习率（RMSProp）的优点。

### 算法（每个参数）

```
m = β₁ * m + (1 - β₁) * grad          # 一阶矩（梯度均值）
v = β₂ * v + (1 - β₂) * grad²         # 二阶矩（梯度方差）
m̂ = m / (1 - β₁ᵗ)                     # 偏差修正
v̂ = v / (1 - β₂ᵗ)                     # 偏差修正
p -= lr * m̂ / (√v̂ + ε)                # 参数更新
```

### 为什么需要偏差修正
m 和 v 初始化为 0，训练初期它们偏向 0。偏差修正 $\hat{m} = m / (1 - \beta_1^t)$ 补偿了这个偏差，使得初期的估计更准确。

### 超参数
- `lr = 1e-3`：学习率（LLM 训练通常 1e-4 ~ 3e-4）
- `β₁ = 0.9`：一阶矩衰减率（控制动量）
- `β₂ = 0.999`：二阶矩衰减率（控制自适应学习率）
- `ε = 1e-8`：防止除零

### 代码示例

```python
import torch

class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        # 为每个参数初始化一阶和二阶矩
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

# 测试
w = torch.randn(4, 3, requires_grad=True)
opt = MyAdam([w], lr=0.01)
for i in range(5):
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"Step {i}: loss={loss.item():.4f}")
```

---

## 5.3 Cosine Learning Rate Schedule with Warmup

### 是什么
先线性预热（warmup），再按余弦曲线衰减学习率。这是 LLM 训练的标准学习率调度策略。

### 公式

```
step < warmup:
    lr = max_lr × step / warmup_steps

step >= warmup:
    progress = (step - warmup) / (total - warmup)
    lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))
```

### 为什么需要 Warmup
训练初期，Adam 的二阶矩估计不准确（偏差修正也不够），大学习率可能导致不稳定。线性预热让模型在小学习率下"热身"。

### 为什么用余弦衰减
- 比线性衰减更平滑
- 训练后期学习率缓慢下降，有助于精细调整
- 实践中效果优于阶梯衰减

### 代码示例

```python
import math

def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# 测试
lrs = [cosine_lr_schedule(i, 100, 10, 0.001) for i in range(101)]
print(f"Step 0:   {lrs[0]:.6f}")    # 0.000000（从 0 开始）
print(f"Step 10:  {lrs[10]:.6f}")   # 0.001000（warmup 结束，达到 max_lr）
print(f"Step 55:  {lrs[55]:.6f}")   # ~0.000500（中间值）
print(f"Step 100: {lrs[100]:.6f}")  # 0.000000（衰减到 min_lr）
```

### 学习率曲线示意

```
lr
│  ╱╲
│ ╱  ╲
│╱    ╲
│      ╲
│       ╲
│        ╲___
└──────────────── step
 warmup  cosine decay
```

---

## 5.4 Gradient Norm Clipping（梯度范数裁剪）

### 是什么
当梯度的总范数超过阈值时，等比例缩小所有梯度，防止梯度爆炸。

### 算法

```
1. 计算总范数: total_norm = sqrt(Σ ||p.grad||²)
2. 如果 total_norm > max_norm:
     scale = max_norm / total_norm
     对所有参数: p.grad *= scale
3. 返回原始 total_norm
```

### 为什么需要
- RNN/LSTM 中梯度爆炸是常见问题
- Transformer 训练中也会偶发梯度尖峰
- 裁剪保持梯度方向不变，只缩小幅度

### 代码示例

```python
import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    parameters = list(parameters)
    total_norm_sq = sum(
        p.grad.data.norm() ** 2 for p in parameters if p.grad is not None
    )
    total_norm = total_norm_sq.sqrt().item()

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)

    return total_norm

# 测试
p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
print(f"裁剪前范数: {p.grad.norm().item():.2f}")
orig = clip_grad_norm([p], max_norm=1.0)
print(f"裁剪后范数: {p.grad.norm().item():.2f}")
print(f"原始范数: {orig:.2f}")
```

---

## 5.5 Gradient Accumulation（梯度累积）

### 是什么
将一个大 batch 拆分为多个 micro-batch，逐个前向+反向，累积梯度后统一更新参数。效果等价于大 batch 训练，但内存只需容纳一个 micro-batch。

### 算法

```
optimizer.zero_grad()
for (x, y) in micro_batches:
    loss = loss_fn(model(x), y) / len(micro_batches)  # 缩放 loss
    loss.backward()  # 梯度累积（不清零）
optimizer.step()
```

### 为什么要除以 micro-batch 数量
`loss.backward()` 会将梯度累加到 `.grad` 上。如果不缩放，累积 N 个 micro-batch 的梯度等于 N 倍的单 batch 梯度。除以 N 使其等价于在完整大 batch 上计算的梯度。

### 代码示例

```python
import torch
import torch.nn as nn

def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(micro_batches)

    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / n
        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    return total_loss

# 测试
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
micro_batches = [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)]
loss = accumulated_step(model, opt, nn.MSELoss(), micro_batches)
print(f"累积 loss: {loss:.4f}")
```

### 适用场景
- GPU 内存不足以容纳大 batch
- 分布式训练中配合数据并行
- LLM 训练中常用（effective batch size = micro_batch × accumulation_steps × num_gpus）

---

## 5.6 Linear Regression（线性回归）

### 是什么
线性回归是最基础的机器学习模型，寻找 $\hat{y} = Xw + b$ 使得预测值与真实值的均方误差最小。本练习要求用三种方法实现。

### 方法一：闭式解（Normal Equation）

$$\theta = (X_{aug}^T X_{aug})^{-1} X_{aug}^T y$$

其中 $X_{aug}$ 是在 X 后拼接一列 1（用于 bias）。

```python
def closed_form(X, y):
    X_aug = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    theta = torch.linalg.lstsq(X_aug, y).solution
    return theta[:-1], theta[-1]
```

### 方法二：手动梯度下降

```python
def gradient_descent(X, y, lr=0.01, steps=1000):
    N, D = X.shape
    w = torch.zeros(D)
    b = torch.tensor(0.0)
    for _ in range(steps):
        pred = X @ w + b
        error = pred - y
        w -= lr * (2.0 / N) * (X.T @ error)
        b -= lr * (2.0 / N) * error.sum()
    return w, b
```

### 方法三：PyTorch autograd

```python
def nn_linear(X, y, lr=0.01, steps=1000):
    model = torch.nn.Linear(X.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(steps):
        pred = model(X).squeeze(-1)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model.weight.data.squeeze(0), model.bias.data.squeeze(0)
```

### 三种方法对比

| 方法 | 优点 | 缺点 |
|------|------|------|
| 闭式解 | 精确解，一步到位 | 需要矩阵求逆，大数据集不适用 |
| 手动梯度下降 | 理解底层原理 | 需要手动推导梯度 |
| autograd | 通用，可扩展到任意模型 | 需要迭代，有超参数 |
