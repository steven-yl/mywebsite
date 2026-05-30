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
---

> 本文档是对 TorchCode 项目中 40 个 PyTorch 算子/算法练习的完整技术解读。从基础激活函数到前沿 RLHF 损失函数，覆盖深度学习工程师需要掌握的核心实现细节。

## 文档索引

| 章节 | 笔记 | 内容概要 |
|------|------|----------|
| 总览 | [[00-总览：TorchCode 知识架构与学习路径]] | 项目整体架构、知识图谱、学习路径 |
| 第一章 | [[01-第一章：激活函数与基础组件（TorchCode）]] | ReLU、GELU、Softmax、Linear、Embedding、Dropout |
| 第二章 | [[02-第二章：归一化技术（TorchCode）]] | LayerNorm、BatchNorm、RMSNorm |
| 第三章 | [[03-第三章：注意力机制（TorchCode）]] | MHA、GQA、RoPE、Flash Attention、KV Cache |
| 第四章 | [[04-第四章：架构与模型组件（TorchCode）]] | GPT-2 Block、SwiGLU、ViT、LoRA、MoE |
| 第五章 | [[05-第五章：训练与优化（TorchCode）]] | Adam、Cosine LR、梯度裁剪、Kaiming 初始化 |
| 第六章 | [[06-第六章：推理与解码策略（TorchCode）]] | Top-k/Top-p、Beam Search、Speculative Decoding |
| 第七章 | [[07-第七章：高级主题（TorchCode）]] | BPE、INT8 量化、DPO/GRPO/PPO Loss |

## 难度分布

- 🟢 Easy（9 题）：ReLU、Softmax、Cross-Entropy、Dropout、Embedding、GELU、Kaiming Init、Gradient Clipping、Gradient Accumulation
- 🟡 Medium（14 题）：Linear、LayerNorm、BatchNorm、RMSNorm、SwiGLU MLP、Conv2d、Cross-Attention、LoRA、ViT Patch、Adam、Cosine LR、Top-k Sampling、Beam Search、Linear Regression
- 🔴 Hard（17 题）：Attention、MHA、Causal Attention、GQA、Sliding Window、Linear Attention、GPT-2 Block、KV Cache、RoPE、Flash Attention、MoE、Speculative Decoding、BPE、INT8 Quantization、DPO Loss、GRPO Loss、PPO Loss

## 相关笔记

- [[00-总览：TorchCode 知识架构与学习路径]]
- [[01-第一章：激活函数与基础组件（TorchCode）]]
