---
title: "OpenVLA 技术文档"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "OpenVLA 开源 VLA 模型技术文档索引，涵盖架构、VLM 基座、动作 token 化、RLDS 数据、训练微调、推理部署与评估。"
summary: "OpenVLA 技术文档索引、核心数据流与模块导航。"
tags: [openvla, robots]
categories: [docs OpenVLA]
series: [openvla-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# OpenVLA 技术文档

> 本文档基于 [OpenVLA](https://github.com/openvla/openvla) 源码与论文 [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) 编写，面向希望深入理解、训练、微调与部署 Vision-Language-Action (VLA) 模型的开发者。

---

## 文档索引

| 章节 | 文件 | 内容概要 |
|------|------|----------|
| **0. 总览** | [01-architecture-overview.md](./01-architecture-overview.md) | 整体架构、知识结构、模块关联、方案对比与适用场景 |
| **1. 视觉-语言模型 (VLM)** | [02-vision-language-model.md](./02-vision-language-model.md) | Prismatic VLM 架构、Vision/LLM Backbone、Projector、多模态融合 |
| **2. 动作离散化与预测** | [03-action-tokenization-and-prediction.md](./03-action-tokenization-and-prediction.md) | ActionTokenizer 原理、归一化/反归一化、自回归动作生成 |
| **3. 数据管道** | [04-data-pipeline-rlds.md](./04-data-pipeline-rlds.md) | RLDS 格式、Open X-Embodiment、数据混合、预处理与增强 |
| **4. 训练与微调** | [05-training-and-fine-tuning.md](./05-training-and-fine-tuning.md) | FSDP 全量训练、LoRA 微调、损失函数、训练阶段 |
| **5. 推理与部署** | [06-inference-and-deployment.md](./06-inference-and-deployment.md) | HuggingFace 集成、predict_action、REST API 部署 |
| **6. 评估** | [07-evaluation.md](./07-evaluation.md) | BridgeData V2、LIBERO 仿真基准、评估流程 |
| **7. 配置与模块参考** | [08-configuration-and-modules.md](./08-configuration-and-modules.md) | 目录结构、配置系统、模块边界、扩展指南 |

---

## 项目一句话总结

**OpenVLA** 将在大规模机器人演示数据上预训练的 **Prismatic VLM**（视觉-语言模型）扩展为 **VLA**（视觉-语言-动作模型）：将连续机器人动作离散化为 LLM 词表末尾的 token，以 **next-token prediction**（下一 token 预测）的方式自回归生成动作，实现通用机器人操作策略。

---

## 核心数据流（端到端）

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  RGB 图像    │───▶│ Vision Backbone   │───▶│ Projector (MLP) │
│  + 语言指令  │    │ (DINOv2+SigLIP)  │    │ 视觉→LLM 维度   │
└─────────────┘    └──────────────────┘    └────────┬────────┘
                                                    │
┌─────────────┐    ┌──────────────────┐             ▼
│ 连续动作     │◀───│ ActionTokenizer  │◀─── LLM (Llama-2 7B)
│ (7-DoF)     │    │ 反离散化+反归一化  │     自回归生成 action tokens
└─────────────┘    └──────────────────┘
```

---

## 仓库顶层结构

```
openvla/
├── prismatic/          # 核心 Python 包：模型、数据、训练
├── vla-scripts/        # VLA 训练/微调/部署/权重转换脚本
├── experiments/        # 真机与仿真评估代码
├── scripts/            # 原 Prismatic VLM 训练脚本（遗留）
├── docs/               # 本技术文档
├── pyproject.toml      # 依赖与项目配置
└── README.md           # 快速上手指南
```

---

## 预训练模型

| 模型 | HuggingFace | 基础 VLM | 训练数据 |
|------|-------------|----------|----------|
| `openvla-7b` | [openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b) | DINO-SigLIP 224px + Llama-2 7B | OXE Magic Soup++ (~970K 轨迹) |
| `openvla-v01-7b` | [openvla/openvla-7b-v01](https://huggingface.co/openvla/openvla-7b-v01) | SigLIP 224px + Vicuña v1.5 7B | OXE Magic Soup (Octo 混合) |

---

## 关键依赖

| 包 | 版本 | 用途 |
|----|------|------|
| PyTorch | 2.2.0 | 深度学习框架 |
| transformers | 4.40.1 | LLM 与 HF AutoClasses |
| timm | 0.9.10 | Vision Transformer |
| flash-attn | 2.5.5 | 高效注意力（训练/推理可选） |
| tensorflow / tfds | 2.15.0 / 4.9.3 | RLDS 数据加载 |
| dlimp | git | RLDS 数据变换管道 |
| peft | 0.11.1 | LoRA 微调 |
| draccus | 0.8.0 | 配置 dataclass 管理 |

---

## 相关论文与资源

- **OpenVLA 论文**: [Kim et al., 2024](https://arxiv.org/abs/2406.09246)
- **Prismatic VLMs**: [Karamcheti et al., 2024](https://arxiv.org/abs/2402.07817) — [GitHub](https://github.com/TRI-ML/prismatic-vlms)
- **Open X-Embodiment**: [Padalkar et al., 2023](https://arxiv.org/abs/2310.08864) — [Website](https://robotics-transformer-x.github.io/)
- **LLaVA 多模态融合**: [Liu et al., 2023](https://arxiv.org/abs/2304.08485)
- **RT-1 / Octo**: [Brohan et al., 2022](https://arxiv.org/abs/2212.06817) / [Octo Model](https://github.com/octo-models/octo)
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **FSDP**: [Zhao et al., 2023](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---

## 阅读建议

1. **快速上手推理**：读 [06-inference-and-deployment.md](./06-inference-and-deployment.md)
2. **LoRA 微调新任务**：读 [05-training-and-fine-tuning.md](./05-training-and-fine-tuning.md) + [04-data-pipeline-rlds.md](./04-data-pipeline-rlds.md)
3. **从头预训练 VLA**：读 [01-architecture-overview.md](./01-architecture-overview.md) → [05-training-and-fine-tuning.md](./05-training-and-fine-tuning.md)
4. **理解算法原理**：按 02 → 03 → 04 顺序阅读
