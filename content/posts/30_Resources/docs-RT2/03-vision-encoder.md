---
title: "3. 视觉编码器：ViTransformerWrapper 与 ViT"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "RT-2 图像编码路径：像素到 Patch 嵌入、ViT 与 Transformer Encoder 完整算法与实现。"
summary: "RT-2 视觉编码器 ViTransformerWrapper 与 ViT 详解。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 4
series_weight: 4
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 3. 视觉编码器：ViTransformerWrapper 与 ViT

本章详解 RT-2 中图像编码路径：从像素到 Patch 嵌入，再到 Transformer Encoder 输出的完整算法与实现。

---

## 3.1 为什么需要视觉编码器

机器人策略的输入是 **高维像素矩阵** $(H \times W \times 3)$，而语言模型处理的是 **离散 Token 序列**。视觉编码器的作用是：

1. **降维**：$256 \times 256 \times 3 = 196{,}608$ 维 → $64 \times 512 = 32{,}768$ 维
2. **提取语义特征**：从低级纹理到物体/场景级表示
3. **统一表示空间**：输出向量与 Decoder 隐藏维度对齐，供 Cross-Attention 使用

---

## 3.2 Vision Transformer (ViT) 原理

### 3.2.1 Patch Embedding

将图像切分为不重叠的 Patch 并线性投影（[Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)）：

$$
\mathbf{x}_p = \text{Flatten}(\mathbf{I}_{p}) \in \mathbb{R}^{P^2 \cdot C}, \quad
\mathbf{z}_p = \mathbf{E} \mathbf{x}_p + \mathbf{e}_{pos}^p
$$

其中：
- $P$ = `patch_size` = 32
- $C$ = 3（RGB）
- $\mathbf{E} \in \mathbb{R}^{d \times P^2 C}$ 为可学习投影
- $\mathbf{e}_{pos}^p$ 为位置编码

本仓库默认：$d=512$，Patch 数 $N = (256/32)^2 = 64$。

### 3.2.2 einops 重排

```python
x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
```

| 输入形状 | 输出形状 |
|----------|----------|
| `(B, 3, 256, 256)` | `(B, 64, 3072)` |

其中 $3072 = 32 \times 32 \times 3$。

### 3.2.3 Transformer Encoder 层

每层包含：

$$
\mathbf{z}' = \text{LayerNorm}(\mathbf{z} + \text{MultiHeadAttention}(\mathbf{z}))
$$
$$
\mathbf{z}'' = \text{LayerNorm}(\mathbf{z}' + \text{FFN}(\mathbf{z}'))
$$

**Encoder 使用双向 Self-Attention**（无因果掩码），每个 Patch 可 attend 到所有 Patch。

---

## 3.3 ViTransformerWrapper 完整 API

**来源**：`zetascale` → `zeta.structs.transformer.ViTransformerWrapper`

### 3.3.1 `__init__`

```python
def __init__(
    self,
    *,
    image_size,      # 256
    patch_size,      # 32
    attn_layers,     # Encoder 实例
    channels=3,
    num_classes=None,
    post_emb_norm=False,
    emb_dropout=0.0,
):
```

| 参数 | 说明 |
|------|------|
| `image_size` | 图像边长，必须被 `patch_size` 整除 |
| `patch_size` | Patch 边长 |
| `attn_layers` | **必须是 `Encoder` 实例** |
| `channels` | 输入通道数，默认 3 |
| `num_classes` | 若设置，forward 返回分类 logits；RT2 不设置 |
| `post_emb_norm` | Patch 嵌入后额外 LayerNorm |
| `emb_dropout` | 嵌入 Dropout 率 |

**内部子模块**：

| 子模块 | 类型 | 作用 |
|--------|------|------|
| `pos_embedding` | `nn.Parameter(1, num_patches, dim)` | 可学习绝对位置编码 |
| `patch_to_embedding` | `Sequential(LayerNorm, Linear, LayerNorm)` | Patch → 隐藏维度 |
| `attn_layers` | `Encoder` | Transformer 层堆叠 |
| `mlp_head` | `Linear` 或 `Identity` | 分类头（RT2 中为 Identity） |

### 3.3.2 `forward`

```python
def forward(self, img, return_embeddings=False):
```

#### 算法步骤

```
Step 1: Patch 切分
    img (B,C,H,W) → x (B, num_patches, patch_dim)

Step 2: Patch Embedding
    x = patch_to_embedding(x)  → (B, num_patches, dim)

Step 3: 加位置编码
    x = x + pos_embedding[:, :n]

Step 4: 可选 post_emb_norm + dropout
    x = dropout(post_emb_norm(x))

Step 5: Transformer Encoder
    x = attn_layers(x)  → (B, num_patches, dim)

Step 6: 返回
    if return_embeddings or no mlp_head:
        return x  # RT2 走此分支
    else:
        return mlp_head(x.mean(dim=-2))  # 分类
```

#### RT2 调用方式

```python
encoded = self.encoder(img, return_embeddings=True)
# encoded.shape = (B, 64, 512)
```

**关键**：必须 `return_embeddings=True`，否则当 `num_classes=None` 时虽也返回嵌入，但显式设置是最佳实践；若误设 `num_classes` 且不传 `return_embeddings`，会对 patch 序列做 mean pooling 后分类。

---

## 3.4 Encoder 类

**来源**：`zeta.structs.transformer.Encoder`

```python
class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)
```

| 属性 | 值 | 含义 |
|------|-----|------|
| `causal` | `False` | 双向注意力，每个 token 可见全体 |
| `dim` | 512 | 隐藏维度 |
| `depth` | 6 | 层数 |
| `heads` | 8 | 注意力头数 |

### AttentionLayers.forward（Encoder 模式）

```python
def forward(self, x, context=None, mask=None, ...):
    assert not (self.cross_attend ^ exists(context))
    # Encoder: cross_attend=False, context=None
    for layer_type, (norm, block, residual_fn) in ...:
        if layer_type == "a":  # Self-Attention
            out, inter = block(x, mask=mask, ...)
        elif layer_type == "f":  # FeedForward
            out = block(x)
        x = residual_fn(out, x)
    return x
```

默认层模式 `("a", "f")` × depth = 交替 **Attention + FeedForward**。

---

## 3.4.1 Self-Attention 数学细节

对第 $l$ 层：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right) V_i
$$

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

FeedForward（FFN）：

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

通常 $W_1$ 扩展至 $4d$，$W_2$ 投影回 $d$。

---

## 3.5 与论文 ViT 的对比

| 特性 | 论文 ViT-22B | 本仓库 ViT |
|------|-------------|------------|
| 参数量 | ~22B | ~10M 量级 |
| Patch 大小 | 14 (ViT-G/14) | 32 |
| 层数 | 数十层 | 6 |
| 预训练 | WebLI 10B 图像 | 无 |
| 位置编码 | 多种变体 | 可学习绝对位置 |

---

## 3.6 可运行示例：独立 ViT 编码

```python
import torch
from zeta.structs import Encoder, ViTransformerWrapper

# 构建与 RT2 相同的编码器
encoder = ViTransformerWrapper(
    image_size=256,
    patch_size=32,
    attn_layers=Encoder(dim=512, depth=6, heads=8),
)

img = torch.randn(2, 3, 256, 256)
embeddings = encoder(img, return_embeddings=True)

print(embeddings.shape)  # torch.Size([2, 64, 512])

# 可视化 patch 数量
num_patches = (256 // 32) ** 2
print(f"Patches: {num_patches}, dim: {embeddings.shape[-1]}")
```

---

## 3.7 结构示意图

```
Input Image (256×256×3)
        │
        ▼
┌───────────────────┐
│  Patch Split      │  8×8 grid → 64 patches
│  each 32×32×3     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Linear Project   │  3072 → 512
│  + Pos Embed      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Encoder Layer 1  │  Self-Attn (bidirectional) + FFN
│  Encoder Layer 2  │
│  ...              │
│  Encoder Layer 6  │
└───────────────────┘
        │
        ▼
Output: (B, 64, 512) → 传入 Decoder Cross-Attention
```

---

## 3.8 参考文献

| 文献 | 链接 |
|------|------|
| An Image is Worth 16x16 Words (ViT) | https://arxiv.org/abs/2010.11929 |
| Scaling Vision Transformers | https://arxiv.org/abs/2106.04560 |
| Zeta ViTransformerWrapper 源码 | https://github.com/kyegomez/zeta/blob/master/zeta/structs/transformer.py |
| OpenCLIP ViT | https://github.com/mlfoundations/open_clip |

---

## 3.9 下一章

图像嵌入如何被 Decoder 消费 → [04-decoder-autoregression.md](./04-decoder-autoregression.md)
