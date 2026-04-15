---
title: "网络模块"
date: 2026-04-14T10:00:00+08:00
draft: false
authors: [Steven]
description: "常用神经网络模块详解"
summary: "常用神经网络模块详解"
tags: [Deep Learning]
categories: [Deep Learning]
series: [Deep Learning系列]
weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
---

## 常用神经网络模块

### 0. 基础组件
- nn.Module
- nn.Conv2d
- nn.Linear
- nn.BatchNorm2d
- nn.ReLU
- nn.Sigmoid
- nn.Tanh
- nn.Softmax
- nn.Dropout
- nn.Sequential
- nn.ModuleList
- nn.ModuleDict
- nn.Parameter
- nn.ParameterList
- nn.ParameterDict
- nn.Embedding
- nn.EmbeddingBag

### 1. SinusoidalPosEmb

```python 1. SinusoidalPosEmb
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
```
