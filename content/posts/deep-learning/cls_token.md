---
title: "CLS Token 原理和应用"
date: 2026-04-14T10:00:00+08:00
draft: false
authors: [Steven]
description: "CLS Token 原理和应用"
summary: "CLS Token 原理和应用"
tags: [Deep Learning]
categories: [Deep Learning]
series: [Deep Learning系列]
weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
---

## 1. 概述

`cls_token` 是在 Transformer 架构中引入的一个**可学习的特殊向量**，通常被放置在输入序列的开头位置。经过多层 Transformer 编码后，该 token 对应的输出向量被设计为**整个输入序列的全局表示**，广泛用于分类任务（如文本分类、图像分类）以及预训练模型的聚合表示。

- **提出背景**：BERT（Devlin et al., 2019）首次在 NLP 中使用 `[CLS]` token；Vision Transformer（ViT，Dosovitskiy et al., 2020）将其引入计算机视觉。
- **核心思想**：利用 Transformer 的自注意力机制，让 `cls_token` 与序列中所有其他 token 进行信息交互，自动学习如何聚合并压缩全局信息。

---

## 2. 原理详解

### 2.1 输入构造

假设原始输入序列包含 \(N\) 个 token（例如，文本中的单词或图像中的 patch），每个 token 的特征维度为 \(D\)。原始 token 序列为：

\[
X_{\text{orig}} \in \mathbb{R}^{N \times D}
\]

定义一个可学习的向量 \( \text{cls} \in \mathbb{R}^{1 \times D} \)，将其拼接到序列最前面：

\[
X = [\text{cls}; \; X_{\text{orig}}] \in \mathbb{R}^{(N+1) \times D}
\]

同时，为了保留位置信息，需要为 `cls_token` 分配一个位置编码（可以是固定的正弦编码或可学习的嵌入），并与输入相加：

\[
X_{\text{input}} = X + P
\]

其中 \(P \in \mathbb{R}^{(N+1) \times D}\) 是位置编码矩阵。

### 2.2 通过 Transformer 编码

将 \(X_{\text{input}}\) 输入到 \(L\) 层 Transformer 编码器（每层包含多头自注意力和前馈网络，并带有残差连接和层归一化）。在每一层中，`cls_token` 作为一个普通的 token 参与自注意力计算：

- **Query, Key, Value** 均来自所有 token（包括 `cls_token`）。
- `cls_token` 可以“关注”到序列中的任何其他 token，并更新自己的表示。
- 其他 token 也会关注 `cls_token`，从而将全局信息传播到局部表示。

经过 \(L\) 层后，得到输出序列 \(Z \in \mathbb{R}^{(N+1) \times D}\)。其中第一个向量 \(Z_0\)（对应 `cls_token` 的位置）即为**全局聚合表示**。

### 2.3 为什么 `cls_token` 能聚合全局信息？

- **无偏初始化**：`cls_token` 没有与任何特定输入位置绑定，其初始值是随机的，必须通过注意力机制从其他 token 中“收集”信息来形成有意义的表示。
- **自注意力的全连接性**：在每一层，`cls_token` 都能与序列中所有 token 交换信息，且梯度可以流过所有连接，因此它能够学习到一种加权策略：对任务重要的 token 赋予更高的注意力权重，不重要的 token 忽略或压低。
- **可学习的聚合函数**：相比于固定的全局平均池化（每个 token 权重相等），`cls_token` 实现了**自适应加权求和**，且这个权重是通过任务损失函数端到端学习得到的。

### 2.4 与位置编码的关系

- `cls_token` 位于序列的第一个位置，因此它的位置编码通常是固定的（如 BERT 中位置 0 的 embedding）或可学习的（如 ViT 中单独学习 `cls` 的位置编码）。
- 这种位置编码告知模型：`cls_token` 是一个特殊的“哨兵”位置，不携带任何局部语义。

---

## 3. 典型使用场景

| 模型 | 输入类型 | 用途 |
|------|----------|------|
| BERT | 文本 token 序列 | 分类（情感分析、意图识别）、下一句预测 |
| ViT  | 图像 patch 序列 | 图像分类 |
| DETR | 图像特征 + 物体 query | 目标检测（`cls_token` 变体为 object queries） |
| 多模态模型 | 文本 + 图像 token | 融合表示（如 CLIP 的 class embedding） |

---

## 4. 代码示例（PyTorch）

### 4.1 在 Vision Transformer 中实现分类

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., dropout=0.1):
        super().__init__()
        # 计算 patch 数量
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Patch embedding: 将图像块映射到 D 维
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS token (可学习)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer 编码器 (可以使用 nn.TransformerEncoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        # 初始化 cls_token 和 pos_embed
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # 其他权重初始化略...
    
    def forward(self, x):
        B = x.shape[0]
        # 1. Patch embedding
        x = self.patch_embed(x)                     # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)            # [B, num_patches, embed_dim]
        
        # 2. 拼接 cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, num_patches+1, embed_dim]
        
        # 3. 加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Transformer 编码
        x = self.blocks(x)                           # [B, N+1, embed_dim]
        
        # 5. 取出 cls_token 的输出
        cls_out = x[:, 0, :]                         # [B, embed_dim]
        cls_out = self.norm(cls_out)
        
        # 6. 分类
        logits = self.head(cls_out)                  # [B, num_classes]
        return logits
```

### 4.2 在 BERT 风格文本分类中使用

```python
from transformers import BertModel, BertConfig
import torch.nn as nn

class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        # BERT 内部会自动在 input_ids 前插入 [CLS] token
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 对应向量
        logits = self.classifier(cls_output)
        return logits
```

---

## 5. 最佳实践与建议

### 5.1 训练细节

- **初始化**：`cls_token` 通常使用均值为 0、标准差为 0.02 的截断正态分布初始化（ViT）或零初始化（BERT 源码中 `[CLS]` embedding 也是随机初始化）。
- **学习率**：`cls_token` 应与模型其他参数使用相同的学习率，无需特殊调整。
- **位置编码**：建议为 `cls_token` 单独分配一个位置编码（无论是可学习的还是固定的），否则模型可能难以区分它与其他 token。

### 5.2 何时使用 `cls_token` vs 其他池化方式？

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **`cls_token`** | 自适应、可学习、性能通常最优 | 增加参数量（~0.1% 可忽略） | 大部分分类任务（特别是预训练-微调范式） |
| **平均池化** | 简单、无额外参数、对噪声鲁棒 | 忽略 token 重要性差异 | 序列长度变化大、小数据集 |
| **最大池化** | 捕捉最显著特征 | 丢失分布信息 | 强局部信号任务 |
| **注意力池化** | 灵活加权 | 需要额外参数 | 细粒度分类 |

实验表明，在大规模预训练模型（BERT、ViT）中，`cls_token` 的效果通常优于或等于全局平均池化，因此成为默认选择。

### 5.3 常见陷阱

- **忘记位置编码**：若 `cls_token` 没有位置编码，模型可能无法区分它与其他 token 的顺序关系。
- **错误使用掩码**：在解码器或生成任务中使用 `cls_token` 时，若因果掩码（causal mask）屏蔽了 `cls_token` 对未来的关注，会损害其聚合能力。通常 `cls_token` 仅用于编码器。
- **多 `cls_token`**：某些变体（如 SETR 用于分割）使用多个 learnable query 而不是单个 `cls_token`，此时应视为一组全局表示。

---

## 6. 进阶理解

### 6.1 `cls_token` 的注意力可视化

研究表明，在训练好的 BERT 或 ViT 中，`cls_token` 对输入序列不同位置的注意力权重呈现**与任务相关的模式**：

- 在文本分类中，`[CLS]` 会更关注句首、句尾以及情感词。
- 在图像分类中，`cls_token` 的注意力图通常高亮物体的主体区域，而忽略背景。

这证明了 `cls_token` 成功学习了内容相关的聚合策略。

### 6.2 与对比学习中的全局表示的关系

在 SimCLR、MoCo 等对比学习框架中，图像经过编码器后的全局特征通常来自**全局平均池化**或**一个额外的投影头**。`cls_token` 可以视为一种“可学习的池化”，在某些自监督方法中（如 DINO）被证明非常有效。

### 6.3 替代方案：无 `cls_token` 的 Transformer

- **仅使用平均池化**：如 GPT 系列在分类时取最后一个 token 的输出（但这不是全局聚合，而是自回归因果表示）。
- **所有 token 输出拼接**：用于序列标注任务（如 NER）。
- **Perceiver 架构**：使用少量 latent tokens 与输入序列交叉注意力，可视为 `cls_token` 的泛化。

---

## 7. 总结

| 特性 | 描述 |
|------|------|
| **本质** | 一个可学习的向量，插入输入序列头部 |
| **训练方式** | 与模型其他参数一起通过反向传播更新 |
| **输出用途** | 作为整个序列的全局表示，输入分类头 |
| **优势** | 自适应聚合、利用自注意力机制、与预训练兼容 |
| **局限性** | 不适用于所有任务（如序列标注），需要位置编码配合 |

`cls_token` 是 Transformer 架构中一个简单而强大的设计模式，它将注意力机制的灵活性引入到全局表示学习中，成为现代深度学习模型处理分类任务的标准组件。

---

## 8. 参考文献

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
2. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
3. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
4. Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*.

