---
title: "TorchCode 技术文档索引"
date: 2026-04-01T10:00:00+08:00
draft: false
authors: [Steven]
description: "TorchCode 项目 40 个 PyTorch 算子练习的技术文档索引：章节导航与难度分布。"
summary: "从零实现算子的练习项目配套文档索引，链向总览与各章详解。"

tags: [PyTorch, TorchCode]
categories: [PyTorch]
series: [TorchCode 系列]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

# TorchCode 技术文档

> 本文档是对 TorchCode 项目中 40 个 PyTorch 算子/算法练习的完整技术解读。从基础激活函数到前沿 RLHF 损失函数，覆盖深度学习工程师需要掌握的核心实现细节。

## 文档索引

| 章节 | 文件 | 内容概要 |
|------|------|----------|
| 总览 | [00_overview.md](00_overview.md) | 项目整体架构、知识图谱、学习路径、各模块关联与对比 |
| 第一章 | [01_activations_and_fundamentals.md](01_activations_and_fundamentals.md) | 激活函数（ReLU、GELU、Softmax）、线性层、嵌入层、Dropout、交叉熵损失 |
| 第二章 | [02_normalization.md](02_normalization.md) | LayerNorm、BatchNorm、RMSNorm — 归一化技术全解 |
| 第三章 | [03_attention_mechanisms.md](03_attention_mechanisms.md) | Scaled Dot-Product Attention、Multi-Head Attention、Causal Attention、Cross-Attention、GQA、Sliding Window、Linear Attention、KV Cache、RoPE、Flash Attention |
| 第四章 | [04_architectures.md](04_architectures.md) | GPT-2 Block、SwiGLU MLP、ViT Patch Embedding、LoRA、Mixture of Experts、Conv2d |
| 第五章 | [05_training_optimization.md](05_training_optimization.md) | Adam 优化器、Cosine LR、梯度裁剪、梯度累积、Kaiming 初始化、线性回归 |
| 第六章 | [06_inference_decoding.md](06_inference_decoding.md) | Top-k/Top-p 采样、Beam Search、Speculative Decoding |
| 第七章 | [07_advanced_topics.md](07_advanced_topics.md) | BPE 分词器、INT8 量化、DPO Loss、GRPO Loss、PPO Loss |

## 难度分布

- 🟢 Easy（9 题）：ReLU、Softmax、Cross-Entropy、Dropout、Embedding、GELU、Kaiming Init、Gradient Clipping、Gradient Accumulation
- 🟡 Medium（14 题）：Linear、LayerNorm、BatchNorm、RMSNorm、SwiGLU MLP、Conv2d、Cross-Attention、LoRA、ViT Patch、Adam、Cosine LR、Top-k Sampling、Beam Search、Linear Regression
- 🔴 Hard（17 题）：Attention、MHA、Causal Attention、GQA、Sliding Window、Linear Attention、GPT-2 Block、KV Cache、RoPE、Flash Attention、MoE、Speculative Decoding、BPE、INT8 Quantization、DPO Loss、GRPO Loss、PPO Loss
