---
title: "smalldiffusion 技术文档索引"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 技术文档索引"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
---


> smalldiffusion 是一个轻量级扩散模型库，用不到 100 行核心代码实现了扩散模型的训练与采样。
> 本文档对项目进行全面技术解读，从整体架构到每个函数的实现细节。

## 文档结构

| 文件 | 内容 |
|------|------|
| [01_overview.md](01_overview.md) | 项目总览：架构设计、模块关系、扩散模型数学基础 |
| [02_diffusion.md](02_diffusion.md) | 核心模块：噪声调度、训练循环、采样算法 (`diffusion.py`) |
| [03_data.md](03_data.md) | 数据模块：数据集工具、玩具数据集 (`data.py`) |
| [04_model_base.md](04_model_base.md) | 模型基础：ModelMixin、预测模式修饰器、注意力机制、嵌入层 (`model.py`) |
| [05_model_dit.md](05_model_dit.md) | Diffusion Transformer 模型 (`model_dit.py`) |
| [06_model_unet.md](06_model_unet.md) | U-Net 模型 (`model_unet.py`) |
| [07_examples.md](07_examples.md) | 实战示例：从玩具模型到 Stable Diffusion |

## 模块依赖关系

```
data.py ──────────┐
                   ├──> diffusion.py (training_loop, samples)
model.py ─────────┤
  ├─ model_dit.py ┤
  └─ model_unet.py┘
```

## 快速开始

```python
from torch.utils.data import DataLoader
from smalldiffusion import Swissroll, TimeInputMLP, ScheduleLogLinear, training_loop, samples
import numpy as np

dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
loader   = DataLoader(dataset, batch_size=2048)
model    = TimeInputMLP(hidden_dims=(16,128,128,128,128,16))
schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
trainer  = training_loop(loader, model, schedule, epochs=15000)
losses   = [ns.loss.item() for ns in trainer]
*xt, x0  = samples(model, schedule.sample_sigmas(20), gam=2)
```
