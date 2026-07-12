---
title: "RT-2 技术文档索引"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "RT-2 开源实现技术文档索引，涵盖 VLA 理论、代码实现、视觉编码、解码器、动作 token 化、训练数据、评估与 API。"
summary: "RT-2 技术文档索引与快速导航。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# RT-2 技术文档索引

> 本文档集对 [RT-2 (Robotic Transformer 2)](https://github.com/kyegomez/RT-2) 开源实现进行**从总体到局部**的完整技术解读，涵盖论文原理、本仓库代码、依赖组件、训练数据、评估体系与使用方式。

---

## 文档结构

| 章节 | 文件 | 内容概要 |
|------|------|----------|
| **0. 总览** | [00-overview.md](./00-overview.md) | 整体架构、知识图谱、模块关联、优缺点与适用场景对比 |
| **1. VLA 理论** | [01-vla-theory.md](./01-vla-theory.md) | Vision-Language-Action 模型原理、RT-2 论文算法、数学公式与参考文献 |
| **2. 代码实现** | [02-implementation.md](./02-implementation.md) | 仓库结构、`RT2` 类完整 API、调用链、与论文实现的对应关系 |
| **3. 视觉编码器** | [03-vision-encoder.md](./03-vision-encoder.md) | `ViTransformerWrapper` / ViT / `Encoder` 原理与实现 |
| **4. 语言解码器** | [04-decoder-autoregression.md](./04-decoder-autoregression.md) | `Transformer` / `Decoder` / `AutoregressiveWrapper` 原理与实现 |
| **5. 动作表示** | [05-action-tokenization.md](./05-action-tokenization.md) | 动作离散化、Token 映射、Co-Fine-Tuning、输出约束 |
| **6. 训练与数据** | [06-training-datasets.md](./06-training-datasets.md) | 训练策略、超参数、数据集混合比例、模型变体 |
| **7. 评估体系** | [07-evaluation.md](./07-evaluation.md) | 基准测试、泛化场景、涌现能力、消融实验 |
| **8. 使用指南** | [08-usage-api.md](./08-usage-api.md) | 安装、可运行示例、测试用例、API 参考 |

---

## 快速导航

### 按角色

- **研究者 / 算法工程师** → 从 [00-overview](./00-overview.md) 开始，深入 [01-vla-theory](./01-vla-theory.md) 与 [05-action-tokenization](./05-action-tokenization.md)
- **开发者 / 集成者** → 直接阅读 [02-implementation](./02-implementation.md) 与 [08-usage-api](./08-usage-api.md)
- **复现论文实验** → 阅读 [06-training-datasets](./06-training-datasets.md) 与 [07-evaluation](./07-evaluation.md)

### 按概念

| 概念 | 所在章节 |
|------|----------|
| Vision-Language-Action (VLA) | [01-vla-theory](./01-vla-theory.md) |
| Co-Fine-Tuning | [05-action-tokenization](./05-action-tokenization.md), [06-training-datasets](./06-training-datasets.md) |
| 动作离散化 (256 bins) | [05-action-tokenization](./05-action-tokenization.md) |
| ViT 视觉编码 | [03-vision-encoder](./03-vision-encoder.md) |
| Cross-Attention 多模态融合 | [04-decoder-autoregression](./04-decoder-autoregression.md) |
| 自回归生成 | [04-decoder-autoregression](./04-decoder-autoregression.md) |
| Flash Attention | [04-decoder-autoregression](./04-decoder-autoregression.md) |

---

## 项目仓库结构

```
RT-2/
├── rt2/
│   ├── __init__.py      # 导出 RT2 类
│   └── model.py         # RT2 核心模型实现
├── example.py           # 最小可运行示例
├── tests/test.py        # 单元测试
├── requirements.txt     # Python 依赖
├── pyproject.toml       # Poetry 打包配置
└── docs/                # 本技术文档集
```

---

## 核心参考文献

| 资源 | 链接 |
|------|------|
| RT-2 论文 (CoRL 2023) | [PDF](https://robotics-transformer2.github.io/assets/rt2.pdf) / [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) |
| RT-2 项目主页 | https://robotics-transformer2.github.io |
| RT-1 论文 | [Brohan et al., 2022](https://arxiv.org/abs/2212.06817) |
| PaLM-E 论文 | [Driess et al., 2023](https://arxiv.org/abs/2303.03378) |
| PaLI-X 论文 | [Chen et al., 2023](https://arxiv.org/abs/2305.18517) |
| Zeta 依赖库 | [kyegomez/zeta](https://github.com/kyegomez/zeta) |

---

## 本实现与论文 RT-2 的关系

| 维度 | 论文 RT-2 | 本仓库实现 |
|------|-----------|------------|
| 骨干网络 | PaLI-X (5B/55B) 或 PaLM-E (12B) | 轻量级 ViT + Transformer Decoder（zetascale） |
| 参数量 | 数十亿 | 默认约 ~50M 量级（可配置） |
| 动作 Token 化 | 完整实现 | 架构支持，需自行实现编解码 |
| Co-Fine-Tuning | Web + Robot 混合训练 | 未内置训练脚本 |
| 推理部署 | 多 TPU 云端服务 | 本地 PyTorch 前向传播 |

本仓库提供 **RT-2 架构思想的简化 PyTorch 实现**，便于理解 VLA 范式与快速原型验证；生产级部署需参考论文中的完整训练与推理管线。
