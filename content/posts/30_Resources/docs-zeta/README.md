---
title: "Zeta 技术文档索引"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "Zetascale（pip install zetascale）是一个模块化 PyTorch 深度学习框架，提供 Transformer、SSM、MoE、量化、多模态、强化学习等可复用构建块。本文档集从总体架构到局部 API 做完整技术解读。"
summary: "Zetascale（pip install zetascale）是一个模块化 PyTorch 深度学习框架，提供 Transformer、SSM、MoE、量化…"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# Zeta 技术文档索引

> **Zetascale**（`pip install zetascale`）是一个模块化 PyTorch 深度学习框架，提供 Transformer、SSM、MoE、量化、多模态、强化学习等可复用构建块。本文档集从总体架构到局部 API 做完整技术解读。

**版本**：2.8.8 | **Python**：≥3.10 | **核心依赖**：PyTorch、einops、transformers、accelerate

---

## 文档结构

| 章节 | 文件 | 内容 |
|------|------|------|
| **0. 总览** | [01-overview.md](./01-overview.md) | 整体架构、模块关系、优缺点、适用场景对比 |
| **1. 架构骨架** | [02-structs.md](./02-structs.md) | `structs`：Transformer、Encoder/Decoder、自回归封装 |
| **2. 注意力机制** | [03-attention.md](./03-attention.md) | 22 种注意力变体：MHA、MQA、Flash、Linear、Sparse 等 |
| **3. 位置编码与偏置** | [04-embeddings-biases-masks.md](./04-embeddings-biases-masks.md) | Embeddings、RoPE、ALiBi、Masks |
| **4. 状态空间模型** | [05-ssm-mamba.md](./05-ssm-mamba.md) | SSM、Mamba、H3、PScan、UMamba |
| **5. 混合专家 MoE** | [06-moe.md](./06-moe.md) | MoE 路由、Top-K Gating、稀疏 MoE |
| **6. 视觉与卷积** | [07-vision-conv.md](./07-vision-conv.md) | ViT、U-Net、ResNet、Conv 模块 |
| **7. 多模态** | [08-multimodal.md](./08-multimodal.md) | 跨模态注意力、Q-Former、FiLM、融合 |
| **8. 量化** | [09-quantization.md](./09-quantization.md) | BitLinear、QLoRA、Niva、LFQ |
| **9. 完整模型** | [10-models.md](./10-models.md) | ViT、GPT4、PalmE、NaViT 等 |
| **10. 底层算子** | [11-ops.md](./11-ops.md) | Softmax 变体、矩阵根、分布式、多模态 reshape |
| **11. 优化器** | [12-optim.md](./12-optim.md) | Lion、Sophia、Muon、ScaledAdam |
| **12. 强化学习** | [13-rl.md](./13-rl.md) | DPO、PPO、Actor-Critic、奖励模型 |
| **13. 训练基础设施** | [14-training.md](./14-training.md) | Trainer、FSDP、数据加载、调度器 |
| **14. 工具函数** | [15-utils.md](./15-utils.md) | 采样、内存分析、类型检查、I/O |
| **15. NN 模块完整目录** | [16-nn-modules-catalog.md](./16-nn-modules-catalog.md) | 216 个 `nn/modules` 文件 API 全览 |
| **16. 内部模块附录** | [17-internal-modules.md](./17-internal-modules.md) | 未公开导出但存在的模块 |
| **附录 A** | [appendix-norm-activation-ffn.md](./appendix-norm-activation-ffn.md) | 归一化、激活、FFN 算法补充 |

---

## 快速导航：按使用场景

| 场景 | 推荐入口 |
|------|----------|
| 构建语言模型 | [02-structs](./02-structs.md) → [03-attention](./03-attention.md) → [04-embeddings](./04-embeddings-biases-masks.md) |
| 长序列 / 线性复杂度 | [05-ssm-mamba](./05-ssm-mamba.md)、[03-attention#linear](./03-attention.md) |
| 视觉任务 | [07-vision-conv](./07-vision-conv.md)、[10-models#vit](./10-models.md) |
| 多模态 VLM | [08-multimodal](./08-multimodal.md)、[10-models#palme](./10-models.md) |
| 模型压缩 / 部署 | [09-quantization](./09-quantization.md)、[11-ops](./11-ops.md) |
| RLHF / 对齐 | [13-rl#dpo](./13-rl.md) |
| 稀疏 / 大模型扩展 | [06-moe](./06-moe.md) |

---

## 包结构总览

```
zeta/
├── models/          # 端到端模型（ViT, GPT4, PalmE, NaViT …）
├── nn/
│   ├── attention/   # 注意力机制（24 文件）
│   ├── biases/      # 位置偏置（ALiBi, Relative, Dynamic）
│   ├── embeddings/  # 嵌入与位置编码（21 文件）
│   ├── masks/       # 注意力掩码
│   ├── modules/     # 核心 NN 积木（217 文件）
│   ├── quant/       # 量化层
│   └── modules/xmoe/# 分布式 MoE 路由
├── ops/             # 底层数学算子（17 文件）
├── optim/           # 优化器（12 文件）
├── rl/              # 强化学习（12 文件）
├── structs/         # 高层架构骨架（12 文件）
├── training/        # 训练循环与分布式（9 文件）
└── utils/           # 通用工具（14 文件）
```

**根包导入**：`from zeta import *` 会聚合 `models`、`nn`、`ops`、`optim`、`quant`、`rl`、`training`、`utils` 的全部公开 API。

---

## 与现有文档的关系

- **API 速查**：`docs/zeta/` 下有按模块拆分的 MkDocs 页面（如 `docs/zeta/nn/modules/mamba.md`）
- **本文档集**：在忠实原意基础上做**提炼、重组、补全**算法原理与概念解释，适合系统学习与进阶
- **在线文档**：[zeta.apac.ai](https://zeta.apac.ai) / [ReadTheDocs](https://zeta.readthedocs.io)

---

## 参考资源

| 类型 | 链接 |
|------|------|
| GitHub | https://github.com/kyegomez/zeta |
| PyPI | https://pypi.org/project/zetascale |
| 论文合集 | 各章节内按算法附论文/博客/开源实现链接 |
