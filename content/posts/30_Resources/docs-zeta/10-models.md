---
title: "第 9 章：完整模型（zeta.models）"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "第 9 章：完整模型（zeta.models）。"
summary: "第 9 章：完整模型（zeta.models）。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 10
series_weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 9 章：完整模型（zeta.models）

## 1. 公开模型清单

| 模型 | 文件 | 导出 | 类型 |
|------|------|------|------|
| `BaseModel` | `base.py` | ✓ | 抽象基类 |
| `ViT` | `vit.py` | ✓ | 视觉分类 |
| `MaxVit` | `max_vit.py` | ✓ | 混合 Conv+Attention |
| `MegaVit` | `mega_vit.py` | ✓ | 大规模 ViT 变体 |
| `NaViT` | `navit.py` | ✓ | 原生分辨率 ViT |
| `PalmE` | `palme.py` | ✓ | 机器人/多模态 VLM |
| `GPT4` | `gpt4.py` | ✓ | 文本 LM |
| `GPT4MultiModal` | `gpt4.py` | ✓ | 多模态 GPT |
| `LLama2` | `llama.py` | ✓ | LLaMA 风格因果 LM |
| `Andromeda` | `andromeda.py` | ✓ | 实验性架构 |
| `BEiT3` | `BEiT3.py` | ✗ | 多模态 BEiT |
| `LongNet` | `LongNet.py` | ✗ | 超长序列 |
| `Kosmos` | `kosmos.py` | ✗ | 多模态 Kosmos |
| `MultiModalMamba` | `mm_mamba.py` | ✗ | Mamba 多模态 |

---

## 2. `BaseModel`

**文件**：`base.py`

| 方法 | 作用 |
|------|------|
| `forward` | 抽象前向 |
| `save` / `load` | 权重序列化（若实现） |

所有 models 的公共接口约定。

---

## 3. 视觉模型

### 3.1 `ViT`

**结构**：

$$\text{ViT}(x) = \text{Head}(\text{Transformer}(\text{PatchEmbed}(x)))$$

| 参数 | 典型值 |
|------|--------|
| `image_size` | 224 |
| `patch_size` | 16 |
| `dim` | 768 |
| `depth` | 12 |
| `heads` | 12 |
| `num_classes` | 1000 |

```python
import torch
from zeta.models import ViT

model = ViT(image_size=224, patch_size=16, num_classes=1000)
x = torch.randn(2, 3, 224, 224)
logits = model(x)
```

**论文**：[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

### 3.2 `MaxVit`

**特点**：MBConv 局部 + 块级全局注意力（MaxViT 风格）。

**论文**：[MaxViT](https://arxiv.org/abs/2204.01697)

### 3.3 `MegaVit`

大规模 ViT 变体，含更多尺度与深度配置。

### 3.4 `NaViT`（Native Resolution ViT）

**核心思想**：不强制固定分辨率，按 patch 打包变长图像序列（类似 NaViT 论文）。

**优势**：无需 letterbox；批次内不同分辨率高效训练。

**论文**：[Patch n' Pack](https://arxiv.org/abs/2307.06389)

---

## 4. 语言模型

### 4.1 `GPT4` / `GPT4MultiModal`

**文件**：`gpt4.py`

| 类 | 特点 |
|----|------|
| `GPT4` | 因果 Transformer LM |
| `GPT4MultiModal` | 扩展多模态输入 |

基于 `structs.Transformer` + `Decoder`，集成 RoPE、Flash Attention、交叉注意力等选项。

### 4.2 `LLama2`

**文件**：`llama.py`

LLaMA 2 风格配置：RMSNorm、SwiGLU FFN、RoPE、GQA。

**参考**：[Llama 2](https://arxiv.org/abs/2307.09288)

### 4.3 `Andromeda`

实验性架构，可能组合 SSM、MoE 等新组件。

---

## 5. 多模态模型

### 5.1 `PalmE`

**文件**：`palme.py`

**架构**：

```
Image → ViTransformerWrapper(Encoder) → context embeddings
Text  → Transformer(Decoder, cross_attend=True) → logits
```

**论文**：[PaLM-E](https://arxiv.org/abs/2303.07854) — 嵌入式多模态语言模型，面向机器人规划。

```python
import torch
from zeta.models import PalmE

model = PalmE()
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 128))
out = model(img, text)
```

### 5.2 `Kosmos`（内部）

**文件**：`kosmos.py`

Kosmos 风格多模态，含 `KosmosTokenizer`（注意：无独立 tokenizers 包）。

### 5.3 `BEiT3`（内部）

统一多模态预训练架构。

### 5.4 `MultiModalMamba`（内部）

Mamba 骨干 + 多模态融合。

---

## 6. 长序列模型

### 6.1 `LongNet`（内部）

**文件**：`LongNet.py`

膨胀注意力实现超长上下文（10M+ tokens 理论）。

**论文**：[LongNet](https://arxiv.org/abs/2307.02486)

---

## 7. 模型对比

| 模型 | 模态 | 复杂度 | 适用场景 |
|------|------|--------|----------|
| ViT | 图像 | $O(N^2)$ | 分类、特征提取 |
| NaViT | 图像 | 变长 | 多分辨率训练 |
| LLama2 | 文本 | $O(L^2)$ | 因果 LM |
| GPT4MultiModal | 图文 | $O(L^2)$ | 通用 VLM 原型 |
| PalmE | 图文 | $O(L^2)$ | 具身智能 |
| LongNet | 文本 | 亚二次 | 极长文档 |

---

## 8. 扩展新模型

推荐模式：

```python
from zeta.models.base import BaseModel
from zeta.structs import Transformer, Decoder

class MyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.lm = Transformer(
            num_tokens=32000,
            max_seq_len=4096,
            attn_layers=Decoder(dim=1024, depth=24, heads=16),
        )

    def forward(self, x):
        return self.lm(x)
```

---

上一章：[09-quantization.md](./09-quantization.md) | 下一章：[11-ops.md](./11-ops.md)
