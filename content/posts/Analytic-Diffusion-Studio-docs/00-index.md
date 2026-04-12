---
title: "Analytic Diffusion Studio — 技术文档索引"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 技术文档索引"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 技术文档索引"
---

> 本文档对 Analytic Diffusion Studio 项目进行全面、深入的技术解读。文档按"总体→局部"组织，先给出整体架构总览，再按模块逐章展开。

## 文档目录

| 文件 | 内容 |
|------|------|
| [01-overview.md](01-overview.md) | 项目总览：背景知识、整体架构、模块关系、方法对比 |
| [02-diffusion-theory.md](02-diffusion-theory.md) | 扩散模型理论基础：前向/反向过程、DDIM 调度器、去噪公式 |
| [03-configuration.md](03-configuration.md) | 配置系统：OmegaConf 层级合并、数据类定义、路径解析、CLI 覆盖 |
| [04-data.md](04-data.md) | 数据模块：注册机制、数据集工厂、预处理管线、支持的数据集 |
| [05-models-base.md](05-models-base.md) | 模型基类：BaseDenoiser 接口、DDIM 采样循环、轨迹记录 |
| [06-model-wiener.md](06-model-wiener.md) | Wiener 滤波去噪器：协方差矩阵、SVD 分解、线性去噪公式 |
| [07-model-optimal.md](07-model-optimal.md) | 最优贝叶斯去噪器：FAISS 索引、softmax 加权平均、温度参数 |
| [08-model-scfdm.md](08-model-scfdm.md) | 平滑最优去噪器 (SCFDM)：高斯扰动平均、继承关系 |
| [09-model-pca-locality.md](09-model-pca-locality.md) | PCA Locality 去噪器：局部性掩码、流式 softmax、核心创新 |
| [10-model-nearest-dataset.md](10-model-nearest-dataset.md) | 最近邻基线：欧氏距离检索 |
| [11-model-baseline-unet.md](11-model-baseline-unet.md) | 基线 UNet：神经网络架构、权重加载、对比评估 |
| [12-metrics-and-evaluation.md](12-metrics-and-evaluation.md) | 评估指标与实验流程：R²、MSE、WandB 集成、输出结构 |
| [13-utilities.md](13-utilities.md) | 工具模块：Wiener 滤波计算/存储、UNet 网络组件 |
