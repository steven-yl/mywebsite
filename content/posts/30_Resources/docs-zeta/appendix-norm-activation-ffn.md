---
title: "附录 A：归一化、激活与前馈网络"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "本章补充 nn/modules 中 归一化、激活、FFN 类模块的算法原理，与 16-nn-modules-catalog.md 配合使用。"
summary: "本章补充 nn/modules 中 归一化、激活、FFN 类模块的算法原理，与 16-nn-modules-catalog.md 配合使用。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 99
series_weight: 99
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 附录 A：归一化、激活与前馈网络

本章补充 `nn/modules` 中 **归一化、激活、FFN** 类模块的算法原理，与 [16-nn-modules-catalog.md](./16-nn-modules-catalog.md) 配合使用。

---

## 1. 归一化层

### 1.1 LayerNorm

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

**文件**：`layernorm.py` — `LayerNorm`, `l2norm`

### 1.2 RMSNorm

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$$

去掉均值中心化，LLaMA 默认。**文件**：`rms_norm.py`, `simple_rmsnorm.py`, `triton_rmsnorm.py`

**论文**：[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

### 1.3 其他归一化

| 类 | 文件 | 特点 |
|----|------|------|
| `ScaleNorm` | `scale_norm.py` | 可学习标量缩放 |
| `SubLN` | `subln.py` | 子层归一化（DeepNet） |
| `AdaptiveLayerNorm` | `adaptive_layernorm.py` | 条件 $\gamma(c), \beta(c)$ |
| `AdaptiveRMSNorm` | `adaptive_rmsnorm.py` | 条件 RMSNorm |
| `MMLayerNorm` | `mm_layernorm.py` | 多模态分别归一化 |
| `VLayerNorm` | `v_layernorm.py` | 视觉特征 |
| `ChanLayerNorm` | `chan_layer_norm.py` | 通道维（NCHW） |
| `QuantizedLN` | `quantized_layernorm.py` | 量化友好 LN |
| `FractoralNorm` | `fractoral_norm.py` | 分形归一化 |
| `NormalizationFractral` | `norm_fractorals.py` | 分形归一化变体 |
| `PostNorm` | `norm_utils.py` | Post-LN 包装 |
| `PreNorm` | `prenorm.py` | Pre-LN 包装 |
| `add_norm` | `add_norm.py` | 加法+归一化融合 |

---

## 2. 激活函数

### 2.1 GELU 族（`_activations.py`）

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)]))$$

| 类 | 近似/特点 |
|----|-----------|
| `GELUActivation` | 精确 erf 形式 |
| `FastGELUActivation` | tanh 近似 |
| `QuickGELUActivation` | $x \cdot \sigma(1.702 x)$ |
| `NewGELUActivation` | BERT/GPT 使用 |
| `PytorchGELUTanh` | C 实现 tanh 近似 |
| `AccurateGELUActivation` | 高精度快速版 |
| `ClippedGELUActivation` | 裁剪范围，利于量化 |

### 2.2 门控激活

**SwiGLU**（`swiglu.py`）：

$$\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV), \quad \text{Swish}(x) = x \cdot \sigma(x)$$

**GLU**（`glu.py`）：$\text{GLU}(x) = (xW + b) \odot \sigma(xV + c)$

**ReluSquared**（`relu_squared.py`）：$\text{ReLU}^2(x)$ — [Primer](https://arxiv.org/abs/2109.08668)

**Snake**（`snake_act.py`）：$x + \frac{1}{a}\sin^2(ax)$ — 可学习频率

**Mish / Laplace / LeakyRELU**：见 catalog 对应条目

---

## 3. 前馈网络（FFN）

### 3.1 `FeedForward`

**文件**：`feedforward.py`

标准 Transformer FFN，可选 GLU：

$$\text{FFN}(x) = W_2 \cdot \text{Act}(W_1 x + b_1) + b_2$$

```python
import torch
from zeta.nn import FeedForward

ff = FeedForward(256, 512, glu=True, post_act_ln=True, dropout=0.1)
x = torch.randn(4, 128, 256)
out = ff(x)
```

### 3.2 变体

| 类 | 文件 | 说明 |
|----|------|------|
| `FeedForwardNetwork` | `feedforward_network.py` | 多层 FFN |
| `SimpleFeedForward` | `simple_feedforward.py` | 简化版 |
| `LogFF` | `log_ff.py` | 对数域 FFN |
| `Conv2DFeedforward` | `conv_mlp.py` | 2D 卷积 FFN |
| `MLP` / `CustomMLP` | `mlp.py`, `flexible_mlp.py` | 通用 MLP |
| `MLPMixer` | `mlp_mixer.py` | Token/Channel MLP 交替 |
| `BlockMLP` / `BlockButterflyLinear` | `block_butterfly_mlp.py` | 蝴蝶因子分解 |
| `MonarchMLP` | `monarch_mlp.py` | Monarch 矩阵 |
| `FusedDenseGELUDense` | `fused_gelu_dense.py` | 融合两线性+GELU |

### 3.3 融合算子

| 模块 | 作用 |
|------|------|
| `FusedDropoutLayerNorm` | Dropout+LayerNorm 单 kernel |
| `fused_dropout_add` | 融合 dropout 与残差 |
| `FusedProjSoftmax` | 投影+softmax 融合 |

---

## 4. 残差与结构块

| 类 | 文件 | 公式 |
|----|------|------|
| `Residual` | `residual.py` | $y = x + f(x)$ |
| `SkipConnection` | `skipconnection.py` | 跳跃连接 |
| `GatedResidualBlock` | `gated_residual_block.py` | $y = x + g \odot f(x)$ |
| `HighwayLayer` | `highway_layer.py` | $y = t \odot H(x) + (1-t) \odot x$ |
| `LayerScale` | `layer_scale.py` | $y = x + \alpha \odot f(x)$（CaiT） |
| `StochDepth` / `StochasticSkipBlocK` | `stoch_depth.py`, `stochastic_depth.py` | 随机深度 |
| `RecursiveBlock` | `recursive_block.py` | 权重共享递归 |
| `DualPathBlock` | `dual_path_block.py` | 双路径 |
| `TripleSkipBlock` | `triple_skip.py` | 三跳连接 |
| `MultiScaleBlock` | `multi_scale_block.py` | 多尺度 |
| `FeedbackBlock` | `feedback_block.py` | 反馈连接 |
| `DynamicRoutingBlock` | `dynamic_routing_block.py` | CapsNet 路由 |

---

## 5. 可运行示例：Pre-Norm Transformer FFN 块

```python
import torch
from zeta.nn import RMSNorm, SwiGLUStacked, Residual

class FFBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ff = SwiGLUStacked(dim, dim * 4)
        self.res = Residual()

    def forward(self, x):
        return self.res(x, lambda t: self.ff(self.norm(t)))

block = FFBlock(256)
x = torch.randn(2, 32, 256)
print(block(x).shape)
```

---

返回：[README.md](./README.md)
