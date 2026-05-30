---
title: "smalldiffusion 技术文档索引"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 技术文档索引"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
---

> smalldiffusion 是一个轻量级扩散模型库，用不到 100 行核心代码实现了扩散模型的训练与采样。
> 本文档对项目进行全面技术解读，从整体架构到每个函数的实现细节。

## 文档结构

| 笔记 | 内容 |
|------|------|
| [[01-smalldiffusion 项目总览]] | 项目总览：架构设计、模块关系、扩散模型数学基础 |
| [[02-smalldiffusion 核心模块：diffusion.py]] | 核心模块：噪声调度、训练循环、采样算法 |
| [[03-smalldiffusion 数据模块：data.py]] | 数据模块：数据集工具、玩具数据集 |
| [[04-smalldiffusion 模型基础：model.py]] | 模型基础：ModelMixin、预测模式修饰器、注意力机制 |
| [[05-smalldiffusion 模型：model_dit.py]] | Diffusion Transformer 模型 |
| [[06-smalldiffusion 模型：model_unet.py]] | U-Net 模型 |
| [[07-smalldiffusion 实战示例]] | 实战示例：从玩具模型到 Stable Diffusion |

## 模块依赖关系

```
data.py ──────────┐
                   ├──> diffusion.py (training_loop, samples)
model.py ─────────┤
  ├─ model_dit.py ┤
  └─ model_unet.py┘
```

## 相关笔记

- [[01-smalldiffusion 项目总览]]
- [[02-smalldiffusion 核心模块：diffusion.py]]
