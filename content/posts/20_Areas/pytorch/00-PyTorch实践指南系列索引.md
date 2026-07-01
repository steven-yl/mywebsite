---
title: "PyTorch实践指南系列索引"
date: 2026-05-30T10:00:00+08:00
draft: false
authors: [Steven]
description: "PyTorch实践指南系列索引：Obsidian 双链导航。"
summary: "本目录全部笔记的 wikilink 索引，便于图谱与反向链接浏览。"
tags: [PyTorch, 索引]
categories: [PyTorch]
series: [PyTorch实践指南]
weight: 0
series_weight: 0
hiddenFromHomePage: false
seriesNavigation: false
hiddenFromSearch: false
---

> 本目录 (`content/posts/pytorch`) 下的笔记。Obsidian 中使用 `[[笔记名]]` 建立双链。

## 全部笔记

| 笔记 | 说明 |
|------|------|
| [[PyTorch dtype 与类型体系]] | 系统说明 PyTorch 张量的 dtype：分类、默认推断、转换、类型提升、应用场景及与 NumPy 的对应关系。 |
| [[PyTorch lr曲线]] | lr曲线图 |
| [[PyTorch 分布式训练与操作工具技术文档]] | 从进程组初始化、DDP 封装、数据分片、集体通信到 Lightning 封装，全面讲解如何在单机多卡与多机多卡场景下正确使用 PyTorch 分布式训练。 |
| [[PyTorch 激活函数]] | 本文汇总 Sigmoid、Tanh、ReLU、GELU、Swish 等激活函数，并提供分组图与总览图。 |
| [[PyTorch Dataset 体系技术文档]] | 覆盖 map-style/IterableDataset、全部内置 Dataset 扩展、图数据与 HF datasets、典型项目扩展模式、paddin... |
| [[Pytorch 权重初始化方法]] | 全面对比深度学习权重初始化方法的原理、公式推导、优缺点与适用场景，附 PyTorch 代码示例和 Transformer 架构初始化最佳实践。 |
| [[PyTorch DataLoader 技术解读]] | 从索引流、取样本、成 batch 三条线讲清 DataLoader 职责，涵盖 Sampler、collate_fn、num_workers、pin_me... |
| [[PyTorch 模型训练技术文档：求解器、参数配置与训练循环]] | 从总览到各章节：Optimizer/SGD/Adam/AdamW 全解读、LRScheduler 族、param_groups、梯度累积与裁剪、损失选型及... |
| [[Kaiming（He）初始化：方差推导与 ReLU 网络]] | 用前向方差分析解释为何 ReLU 网络宜用方差 2/fan_in 的权重初始化，并对比 Xavier、给出 PyTorch 中的对应实现。 |
| [[PyTorch 优化器原理]] | 解析PyTorch 优化器原理、更新公式与偏差校正推导，并给出超参数建议与优缺点总结。 |
| [[PyTorch 概率分 (`torch.distributions`)]] | PyTorch 概率分布库 (`torch.distributions`) 完全指南 |
| [[PyTorch 随机种子（Seed）全面解释]] | 系统梳理随机种子的概念与 PyTorch 实践，包括 DataLoader worker、cuDNN、不确定性算子和常见陷阱，帮助你搭建可复现实验环境。 |
| [[Pytorch Batch Size 与学习率缩放规则]] | 详解分布式训练中 batch size 扩大时学习率的线性缩放、平方根缩放及线性+长 warmup 的推导依据与使用建议。 |

## 相关笔记

- [[PyTorch dtype 与类型体系]]
- [[PyTorch lr曲线]]
- [[PyTorch 分布式训练与操作工具技术文档]]
