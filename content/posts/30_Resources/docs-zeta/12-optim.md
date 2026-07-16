---
title: "第 11 章：优化器（zeta.optim）"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "第 11 章：优化器（zeta.optim）。"
summary: "第 11 章：优化器（zeta.optim）。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 12
series_weight: 12
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 11 章：优化器（zeta.optim）

## 1. 公开优化器清单

| 优化器 | 文件 | 类型 |
|--------|------|------|
| `DecoupledLionW` | `decoupled_lion.py` | 解耦权重衰减 Lion |
| `DecoupledLionW8Bit` | `lion8b.py` | 8-bit Lion |
| `SophiaG` | `decoupled_sophia.py` | 二阶曲率估计 |
| `StableAdamWUnfused` | `stable_adam.py` | 稳定 AdamW |
| `ScaledAdam` | `batched_optimizer.py` | 缩放 Adam |
| `Muon` | `muon.py` | 正交化更新 |
| `GradientAscent` | `gradient_ascent.py` | 梯度上升（对抗） |
| `GradientEquilibrum` | `gradient_equillibrum.py` | 梯度均衡 |
| `BatchedOptimizer` | `batched_optimizer.py` | 批量参数优化基类 |
| `Eden` / `Eve` | `batched_optimizer.py` | 实验性优化器 |
| `decoupled_optimizer` | `decoupled_optimizer.py` | 解耦优化器工厂 |

**未导出**：`FastAdaptiveOptimizer`（`all_new_optimizer.py`）、`parallel_gradient_descent`。

---

## 2. Lion / DecoupledLionW

### 2.1 更新规则

Lion（EvoLved Sign Momentum）：

$$c_{t+1} = \beta_1 m_t + (1-\beta_1)\nabla f(x_t)$$
$$x_{t+1} = x_t - \eta \cdot \text{sign}(c_{t+1})$$
$$m_{t+1} = \beta_2 m_t + (1-\beta_2)\nabla f(x_t)$$

**特点**：只用 sign，内存与计算低于 Adam；解耦权重衰减（Decoupled WD）与 AdamW 一致。

### 2.2 API

```python
import torch
from zeta.optim import DecoupledLionW

model = torch.nn.Linear(64, 64)
opt = DecoupledLionW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

**论文**：[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)

---

## 3. Sophia-G

### 3.1 原理

结合梯度与 **Hessian 对角估计**（via Hutchinson）：

$$x_{t+1} = x_t - \eta \cdot \frac{\nabla f}{\max(\hat{h}, \epsilon)}$$

其中 $\hat{h}$ 为对角 Hessian 指数移动平均。

**优势**：大 batch 预训练可比 AdamW 快约 2×。  
**论文**：[Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)

```python
from zeta.optim import SophiaG

# opt = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99))
```

---

## 4. Muon

### 4.1 原理

对 2D 参数（权重矩阵）的更新做 **正交化**（Newton-Schulz 迭代），使更新矩阵接近正交：

$$\Delta W \leftarrow \text{orthogonalize}(\nabla_W \mathcal{L})$$

**用途**：隐藏层大矩阵训练稳定性；与 Adam 配合（embedding/head 用 Adam，矩阵层用 Muon）。

**参考**：[Keller Jordan Muon](https://github.com/KellerJordan/modded-nanogpt)（modded-nanogpt 社区）

```python
from zeta.optim import Muon

# matrix_params = [p for p in model.parameters() if p.ndim == 2]
# opt = Muon(matrix_params, lr=0.02)
```

---

## 5. ScaledAdam / BatchedOptimizer

### 5.1 `BatchedOptimizer`

将参数分批处理，支持 per-parameter 学习率与状态，适合 **参数规模差异大** 的模型（如 Embedding vs Linear）。

### 5.2 `ScaledAdam`

按参数张量尺度自适应缩放步长。

### 5.3 `Eden` / `Eve`

实验性变体，详见 `batched_optimizer.py` 源码。

---

## 6. StableAdamWUnfused

非融合版 AdamW，避免某些 fused kernel 的数值问题；适合调试与特殊 dtype。

标准 AdamW：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$
$$x_{t+1} = x_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda x_t\right)$$

---

## 7. 特殊用途优化器

### 7.1 `GradientAscent`

梯度 **上升** 而非下降，用于对抗训练、GAN 判别器最大化。

### 7.2 `GradientEquilibrum`

平衡多任务或多模态梯度范数，缓解某一模态主导。

### 7.3 `decoupled_optimizer`

工厂函数，按配置构建解耦 WD 优化器。

---

## 8. 选型对比

| 优化器 | 内存 | 大 batch | 二阶信息 | 典型场景 |
|--------|------|----------|----------|----------|
| AdamW | 中 | 好 | 对角 | 默认基线 |
| Lion | **低** | 好 | 无 | 微调、省显存 |
| SophiaG | 中 | **很好** | 对角 Hessian | 预训练 |
| Muon | 中 | 好 | 正交约束 | 大矩阵层 |
| ScaledAdam | 中 | 好 | 无 | 多尺度参数 |

---

## 9. 与 training 集成

```python
from zeta.training import Trainer, train
from zeta.optim import DecoupledLionW

# Trainer 内部可注入自定义 optimizer
```

详见 [14-training.md](./14-training.md)。

---

上一章：[11-ops.md](./11-ops.md) | 下一章：[13-rl.md](./13-rl.md)
