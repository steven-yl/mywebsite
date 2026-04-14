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

### 1. SinusoidalTimeEmbedding

```python 1. SinusoidalTimeEmbedding
class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding module from the DDPM paper, followed by
    an MLP to make it more expressive.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is shape [B]
        device = t.device
        half_dim = self.embed_dim // 2
        
        # Standard sinusoidal formula
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        
        return embeddings
```
