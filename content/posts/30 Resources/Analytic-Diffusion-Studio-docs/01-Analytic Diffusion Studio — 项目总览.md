---
title: "Analytic Diffusion Studio — 项目总览"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 项目总览"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 项目总览"
---

## 1.1 项目背景

**Analytic Diffusion Studio** 是一个面向"无训练 (training-free) 解析扩散模型"的统一实验框架。传统扩散模型（如 DDPM、DDIM）需要训练一个神经网络（通常是 UNet）来学习去噪函数；而本项目实现的方法直接利用数据集的统计量（均值、协方差、最近邻等）构造解析形式的去噪器，无需任何梯度优化。

核心论文：
- **PCA Locality**: [Locality in Image Diffusion Models Emerges from Data Statistics](https://arxiv.org/abs/2509.09672)
- **SCFDM**: [Score-based Generative Models with Closed-Form Denoisers](https://arxiv.org/abs/2310.12395)

## 1.2 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    generate.py (入口)                     │
│  解析 CLI → 加载配置 → 构建数据集 → 创建模型 → 采样 → 评估  │
└────────┬──────────┬──────────┬──────────┬───────────────┘
         │          │          │          │
         ▼          ▼          ▼          ▼
   configuration   data/     models/    metrics
   (配置系统)    (数据加载)  (去噪模型)  (评估指标)
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              BaseDenoiser   utils/      baseline_unet
              (抽象基类)   (Wiener/UNet)  (对比基线)
                    │
        ┌───────┬───┴────┬──────────┬──────────┐
        ▼       ▼        ▼          ▼          ▼
     Wiener  Optimal   SCFDM   PCA Locality  Nearest
```

## 1.3 模块职责

| 模块 | 路径 | 职责 |
|------|------|------|
| 配置系统 | `configuration.py` | YAML 配置加载、合并、路径解析、运行目录创建 |
| 数据模块 | `data/` | 数据集注册、加载、预处理（归一化到 [-1,1]）、后处理 |
| 模型注册 | `models/__init__.py` | 全局模型注册表，`@register_model` 装饰器 |
| 基类 | `models/base.py` | `BaseDenoiser` 抽象接口、DDIM 采样循环 |
| 去噪模型 | `models/*.py` | 5 种去噪方法的具体实现 |
| 工具 | `utils/` | Wiener 滤波计算/存储、UNet 网络定义 |
| 评估 | `metrics.py` | R²、MSE、L2 距离计算 |
| 入口 | `generate.py` | 实验编排：采样、评估、日志、WandB |

## 1.4 五种去噪方法对比

| 方法 | 类名 | 核心思想 | 预计算 | 推理复杂度 | 适用场景 |
|------|------|---------|--------|-----------|---------|
| **Wiener** | `DenoisingWiener` | 线性最优滤波（基于协方差矩阵） | 协方差 SVD | O(n²) 矩阵乘法 | 快速基线，低分辨率 |
| **Optimal** | `OptimalDenoiser` | 贝叶斯最优估计（softmax 加权平均） | FAISS 索引 | O(N·k) 近邻搜索 | 理论上界参考 |
| **SCFDM** | `SmoothedCFDM` | 对 Optimal 做高斯平滑 | FAISS 索引 | O(M·N·k) | 平滑生成效果 |
| **PCA Locality** | `PCALocalityDenoiser` | 局部性掩码 + 流式 softmax | 协方差 SVD | O(N·n²) 流式 | 核心创新方法 |
| **Nearest** | `NearestDatasetDenoiser` | 最近邻检索 | 数据集加载到内存 | O(N·n) 距离计算 | 最简单基线 |

其中 N = 数据集大小，n = 像素维度，k = 近邻数，M = 噪声采样数。

## 1.5 方法优缺点

**Wiener 滤波器**
- ✅ 推理极快（单次矩阵乘法）
- ✅ 有闭式解，理论清晰
- ❌ 假设数据为高斯分布，生成图像模糊

**Optimal 去噪器**
- ✅ 理论上是贝叶斯最优估计
- ❌ 需要遍历大量近邻，计算量大
- ❌ 生成图像趋向数据集均值

**SCFDM**
- ✅ 通过高斯平滑改善 Optimal 的锐度
- ❌ 计算量是 Optimal 的 M 倍

**PCA Locality**
- ✅ 核心创新：发现扩散模型的局部性来自数据统计
- ✅ 通过掩码实现像素级局部去噪
- ❌ 需要流式遍历整个数据集，推理较慢

**Nearest Dataset**
- ✅ 实现最简单，便于理解
- ❌ 只能"复制"训练集图像，无法生成新图像

## 1.6 执行流程

一次完整实验的执行流程：

```
1. parse_args()          → 解析 --config 和 CLI 覆盖参数
2. load_config()         → 加载 YAML，合并 defaults，应用覆盖
3. ensure_run_directory()→ 创建输出目录结构
4. set_random_seeds()    → 设置全局随机种子（可复现）
5. build_dataset()       → 通过注册表构建数据集 + DataLoader
6. create_model()        → 通过注册表实例化去噪模型
7. model.train()         → 预计算（加载数据/计算协方差/构建索引）
8. model.sample()        → DDIM 采样循环，逐步去噪
9. evaluate_main_model() → 保存图像、网格、轨迹
10. evaluate_comparison()→ 与 UNet 基线对比（可选）
11. wandb.finish()       → 关闭 WandB 日志
```

## 1.7 技术栈

| 依赖 | 用途 |
|------|------|
| PyTorch | 张量计算、模型定义 |
| torchvision | 数据集加载、图像变换 |
| diffusers (HuggingFace) | DDIMScheduler 调度器 |
| OmegaConf | 配置文件管理 |
| FAISS | 高效近邻搜索（Optimal/SCFDM） |
| WandB | 实验追踪与可视化 |
| NumPy | 数值计算辅助 |
| Pillow | 图像 I/O |
| tqdm | 进度条 |
