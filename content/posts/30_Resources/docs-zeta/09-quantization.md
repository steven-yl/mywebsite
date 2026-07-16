---
title: "第 8 章：量化（zeta.nn.quant）"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "第 8 章：量化（zeta.nn.quant）。"
summary: "第 8 章：量化（zeta.nn.quant）。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 9
series_weight: 9
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 8 章：量化（zeta.nn.quant）

## 1. 模块清单

| 文件 | 公开符号 | 作用 |
|------|----------|------|
| `bitlinear.py` | `BitLinear`, `absmax_quantize` | 1-bit 权重 + 8-bit 激活 |
| `half_bit_linear.py` | `HalfBitLinear` | 半 bit 线性层 |
| `qlora.py` | `QloraLinear`, `NF4Tensor`, NF4 MLP 变体 | QLoRA 4-bit 量化 |
| `niva.py` | `niva` | 动态量化工具函数 |
| `absmax.py` | `absmax_quantize` | 绝对最大值量化 |
| `ste.py` | `STE`, `STEFunc` | 直通估计器 |
| `quick.py` | `QUIK` | QUIK 量化方案 |
| `lfq.py` | `LFQ` | 有限标量量化 |
| `qmoe.py` | `QMOEQuantizer`, `batch_gptq` *(未导出)* | MoE GPTQ |
| `residual_vq.py` | `ResidualVectorQuantizer` *(未导出)* | 残差向量量化 |

---

## 2. BitLinear

### 2.1 原理（BitNet 风格）

**权重二值化**：

$$\bar{w} = \frac{1}{n}\sum w_i, \quad w_q = \text{sign}(w - \bar{w}) \cdot \beta$$

$$\beta = \frac{\|w_q\|_1}{n}$$

**激活 absmax 量化**（8-bit）：

$$s = \frac{2^{b-1}-1}{\max|x|}, \quad x_q = \text{round}(s \cdot x)$$

**前向**：

$$y = \frac{1}{\beta} \cdot (x_q \cdot w_q^\top)$$

### 2.2 API

| 方法 | 作用 |
|------|------|
| `BitLinear.__init__(in_features, out_features, groups=1)` | 初始化 |
| `reset_parameters()` | Kaiming 初始化 |
| `forward(input)` | 量化前向 |

```python
import torch
from zeta.nn.quant import BitLinear

linear = BitLinear(512, 512)
x = torch.randn(4, 512)
y = linear(x)
print(y.shape)
```

**论文**：[BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)  
**开源**：[microsoft/unilm/BitNet](https://github.com/microsoft/unilm/tree/master/bitnet)

---

## 3. QLoRA

### 3.1 原理

将预训练权重 $W$ 量化为 4-bit（NF4），训练低秩适配器 $BA$：

$$y = \text{dequant}(W_{4\text{bit}}) x + BAx$$

NF4（NormalFloat4）：对正态分布权重优化的 4-bit 非均匀量化级别。

### 3.2 `QloraLinear`

| 组件 | 作用 |
|------|------|
| 4-bit 权重存储 | 压缩显存 |
| LoRA 分支 | 可训练适配 |
| `forward` | 反量化 + LoRA 输出 |

**论文**：[QLoRA](https://arxiv.org/abs/2305.14314)

```python
import torch
from zeta.nn.quant import QloraLinear

layer = QloraLinear(4096, 4096, r=16)
x = torch.randn(1, 128, 4096)
y = layer(x)
```

---

## 4. Niva 动态量化

### 4.1 `niva` 函数

**是什么**：对已有模型的指定层类型（`nn.Linear`, `nn.Conv2d`）做 **动态量化**（推理时量化激活）。

| 参数 | 作用 |
|------|------|
| `model` | 待量化模型 |
| `model_path` | 预训练权重路径 |
| `output_path` | 输出路径 |
| `quant_type` | `"dynamic"` |
| `quantize_layers` | 目标层类型列表 |
| `dtype` | `torch.qint8` 等 |

```python
import torch
from torch import nn
from zeta.nn.quant import niva

# model = YourModel()
# niva(model, "weights.pt", "quantized.pt",
#      quant_type="dynamic", quantize_layers=[nn.Linear], dtype=torch.qint8)
```

**适用**：部署时减小模型体积与延迟；训练时通常不用。

---

## 5. 直通估计器（STE）

### 5.1 原理

量化不可微，前向用量化值，反向梯度直通：

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial Q(x)}$$

### 5.2 `STE` / `STEFunc`

**文件**：`ste.py`

用于 BitLinear 等自定义量化层的梯度估计。

---

## 6. 其他量化方案

| 方案 | 文件 | 说明 |
|------|------|------|
| `QUIK` | `quick.py` | INT4/INT8 混合量化 |
| `LFQ` | `lfq.py` | 有限标量量化（FSQ 类） |
| `HalfBitLinear` | `half_bit_linear.py` | 1.5-bit 权重 |
| `QMOEQuantizer` | `qmoe.py` | MoE 专家 GPTQ |
| `ResidualVectorQuantizer` | `residual_vq.py` | 残差 VQ（生成模型常用） |

---

## 7. 量化选型

| 方法 | 位宽 | 训练 | 推理加速 | 典型场景 |
|------|------|------|----------|----------|
| BitLinear | W1 A8 | 可（STE） | **高** | 边缘 LLM |
| QLoRA | W4 + LoRA | **微调** | 中 | 单卡微调大模型 |
| Niva 动态 | W8 A8 | 否 | 中 | 快速部署 |
| GPTQ/QMOE | W4 | 否 | 高 | 批量权重量化 |

---

## 8. `hyper_optimize` 装饰器

**文件**：`nn/modules/pyro.py`

统一启用多种优化：

```python
import torch
from zeta.nn import hyper_optimize

@hyper_optimize(torch_compile=True, quantize=True, mixed_precision=True)
def model(x):
    return x @ x.T

out = model(torch.randn(32, 32))
```

| 选项 | 作用 |
|------|------|
| `torch_compile` | `torch.compile` 图优化 |
| `quantize` | 自动量化 |
| `mixed_precision` | AMP |
| `torch_fx` / `torch_script` | 图捕获 |

---

## 9. 参考文献

| 主题 | 链接 |
|------|------|
| BitNet | [2310.11453](https://arxiv.org/abs/2310.11453) |
| QLoRA | [2305.14314](https://arxiv.org/abs/2305.14314) |
| GPTQ | [2210.17323](https://arxiv.org/abs/2210.17323) |
| FSQ/LFQ | [2310.05737](https://arxiv.org/abs/2310.05737) |
| PyTorch Quantization | [PyTorch Docs](https://pytorch.org/docs/stable/quantization.html) |

---

上一章：[08-multimodal.md](./08-multimodal.md) | 下一章：[10-models.md](./10-models.md)
