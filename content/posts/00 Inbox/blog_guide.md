---
title: "blog guide"
date: 2026-03-12T00:00:00+08:00
draft: false
authors: [Steven]
description: "整合全站博文的 title、tags、categories、series，便于查阅与保持 frontmatter 一致；编写规范见 AGENTS.md。"

tags: [tools]
categories: [tools]
series: [tools系列]
weight: 0
hiddenFromHomePage: true
hiddenFromSearch: false
---

本文档整合 `content/posts` 下各篇博文的 **title**、**tags**、**categories**、**series**，便于查阅与保持 frontmatter 一致。新文或改文时，**tags/categories/series 优先从下方 [§1 全站去重列表](#1-全站标签--分类--系列汇总) 中复用**；完整约定见仓库根目录 [AGENTS.md](AGENTS.md) 中的「博客编写规范」。

<!--more-->

---
GUIDE_VERSION: 2026-04-03T19:47:14+08:00

## 1 全站标签 / 分类 / 系列汇总

**tags（去重）：**
- AI
- ALiBi
- Activation Function
- Adam
- Analytic Diffusion Studio
- Bayes
- CS492D
- Consistency Model
- DDPM
- DataLoader
- Dataset
- Deep Learning
- DiffusionDriveV2
- DiscreteFL
- EDM
- Kaiming
- LR
- Loss Functions
- Normalization
- OpenClaw
- PyTorch
- RL
- Riemannian
- RoPE
- Sampler
- Tensor
- TorchCode
- Transformer
- Xavier
- cursor
- diffusion/flow
- distributions
- draft
- drifting-model
- dtype
- flow matching
- how-to-doit
- initialization
- noise schedule
- optimization
- papers
- prompt
- random-seed
- reproducibility
- robots
- smalldiffusion
- todo
- tools
- tutorial
- 代码解读
- 优化器
- 位置编码
- 决策
- 分布式训练
- 多模态
- 带约束扩散
- 思维模型
- 扩散桥
- 扩散模型
- 方法论
- 最优传输
- 概率论
- 深度学习
- 混合精度
- 环面
- 生成式AI
- 统计
- 薛定谔桥
- 认知科学
- 训练
- 问题解决

**categories（去重）：**
- Analytic Diffusion Studio
- CS492D
- Deep Learning
- OpenClaw
- PyTorch
- RL
- diffusion/flow
- distributions
- draft
- how-to-doit
- papers
- robots
- smalldiffusion
- todo
- tools
- 思维方法
- 概率论

**series（去重）：**
- Analytic Diffusion Studio系列
- CS492D系列
- Deep Learning系列
- DoIt
- PyTorch实践指南
- RL系列
- TorchCode 系列
- diffusion/flow-tutorial
- diffusion/flow系列
- draft系列
- robots系列
- smalldiffusion系列
- todo系列
- tools系列
- 思维工具系列
- 概率论系列

---

## 2 diffusion-flow（扩散/流模型）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `diffusion-flow/Bayes：先验与后验.md` | Bayes：先验与后验 | diffusion/flow, Bayes, 统计 | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Consistency Model.md` | Consistency Model | diffusion/flow, Consistency Model, todo | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/DDPM从条件贝叶斯到反向过程.md` | DDPM从条件贝叶斯到反向过程 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/DDPM原理详解.md` | DDPM原理详解 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Diffusion-Flow-Formula.md` | Diffusion-Flow-Formula | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/ELBO.md` | ELBO | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Flow-Matching-Formula.md` | Flow-Matching-Formula | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/MeanFlow.md` | MeanFlow | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/SDE-ODE-离散-连续的转换.md` | SDE-ODE-离散-连续的转换 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/扩散桥原理.md` | 扩散桥原理 | 扩散模型, 薛定谔桥, 扩散桥, 带约束扩散 | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/扩散桥泛谈：从薛定谔的思想实验到生成式模型.md` | 扩散桥泛谈：从薛定谔的思想实验到生成式模型 | 扩散模型, 薛定谔桥, 最优传输, 生成式AI, 多模态, 概率论 | diffusion/flow, papers | diffusion/flow系列 |
| `diffusion-flow/扩散模型中的噪声调度（Noise Schedule）—— 完整理论笔记.md` | 扩散模型中的噪声调度（Noise Schedule）—— 完整理论笔记 | diffusion/flow, noise schedule, DDPM, EDM | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/扩散模型反向去噪公式.md` | 扩散模型反向去噪公式 | diffusion/flow, flow matching, Consistency Model, Sampler | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/扩散模型泛谈(深度研究报告).md` | 扩散模型泛谈(深度研究报告) | diffusion/flow | diffusion/flow, papers | diffusion/flow系列 |
| `diffusion-flow/朗之万动力学中的噪声系数与稳态分布.md` | 朗之万动力学中的噪声系数与稳态分布 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow-tutorial/An Introduction to Flow Matching and Diffusion Models.md` | An Introduction to Flow Matching and Diffusion Models | diffusion/flow, tutorial | diffusion/flow | diffusion/flow-tutorial |
| `diffusion-flow-tutorial/Flow Matching Guide and Code 第5章解读：FlatTorus Riemannian Flow Matching 训练逻辑技术文档.md` | Flow Matching Guide and Code 第5章解读：FlatTorus Riemannian Flow Matching 训练逻辑技术文档 | diffusion/flow, flow matching, Riemannian, 环面, 代码解读 | diffusion/flow | diffusion/flow-tutorial |
| `diffusion-flow-tutorial/Flow Matching Guide and Code 第5章解读：Non-Euclidean Flow Matching.md` | Flow Matching Guide and Code 第5章解读：Non-Euclidean Flow Matching | diffusion/flow, tutorial, Riemannian, flow matching | diffusion/flow | diffusion/flow-tutorial |
| `diffusion-flow-tutorial/Flow Matching Guide and Code 第5章解读：指数映射-对数映射-测地线条件流.md` | Flow Matching Guide and Code 第5章解读：指数映射-对数映射-测地线条件流 | Riemannian | diffusion/flow | diffusion/flow-tutorial |
| `diffusion-flow-tutorial/Flow Matching Guide and Code(项目解析).md` | Flow Matching Guide and Code(项目解析) | diffusion/flow, tutorial | diffusion/flow | diffusion/flow-tutorial |
| `diffusion-flow-tutorial/Flow Matching Guide and Code- Discrete Flow Matching.md` | Flow Matching Guide and Code: Discrete Flow Matching | DiscreteFL, diffusion/flow, tutorial | diffusion/flow | diffusion/flow-tutorial |
| `diffusion-flow-tutorial/Flow Matching Guide and Code.md` | Flow Matching Guide and Code | diffusion/flow, tutorial | diffusion/flow | diffusion/flow-tutorial |
| `diffusion-flow-tutorial/The Principles of Diffusion Models.md` | The Principles of Diffusion Models | diffusion/flow, tutorial, todo | diffusion/flow | diffusion/flow-tutorial |

---

## 3 papers（论文解读）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `papers/DiffusionDriveV2 代码结构图.md` | DiffusionDriveV2 代码结构图 | diffusion/flow, papers, DiffusionDriveV2, todo | diffusion/flow, papers | diffusion/flow系列 |
| `papers/DiffusionDriveV2 网络结构图.md` | DiffusionDriveV2 网络结构图 | diffusion/flow, papers, DiffusionDriveV2, todo | diffusion/flow, papers | diffusion/flow系列 |
| `papers/Generative Modeling via Drifting.md` | Generative Modeling via Drifting | diffusion/flow, drifting-model, papers | diffusion/flow, papers | diffusion/flow系列 |
| `papers/Minimal-Drifting-Models.md` | Minimal-Drifting-Models | diffusion/flow, drifting-model, papers | diffusion/flow, papers, todo | diffusion/flow系列 |
| `papers/OmniXtreme算法原理详细解析.md` | OmniXtreme算法原理详细解析 | robots, papers | robots, papers | robots系列 |

---

## 4 robots（机器人）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `robots/导纳与阻抗-力位混合控制.md` | 导纳与阻抗-力位混合控制 | robots, tutorial | robots | robots系列 |
| `robots/机器人动力学参数辨识流程.md` | 机器人动力学参数辨识流程 | robots, tutorial | robots | robots系列 |
| `robots/机器人柔顺控制与动力学补偿关键技术综述.md` | 机器人柔顺控制与动力学补偿关键技术综述 | robots, tutorial | robots | robots系列 |
| `robots/机器人柔顺控制技术笔记.md` | 机器人柔顺控制技术笔记 | robots, tutorial | robots | robots系列 |
| `robots/机械臂控制方法.md` | 机械臂控制方法 | robots, tutorial | robots | robots系列 |

---

## 5 tools（工具）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `tools/Cursor 插件使用指南.md` | Cursor 插件使用指南 | cursor, tools | tools | tools系列 |
| `tools/Cursor 插件市场介绍.md` | Cursor 插件市场介绍 | cursor, tools | tools | tools系列 |
| `tools/Cursor 的使用技巧.md` | Cursor 的使用技巧 | cursor, tools | tools | tools系列 |
| `tools/OpenClaw 安装文件与目录结构总览.md` | OpenClaw 安装文件与目录结构总览 | OpenClaw, tools | OpenClaw, tools | （空） |
| `tools/Prompt 汇总与记录.md` | Prompt 汇总与记录 | prompt, tools, AI | tools | tools系列 |
| `tools/antigravity-awesome-skills.md` | antigravity-awesome-skills | cursor, tools | tools | tools系列 |

---

## 6 how-to-doit（DoIt 主题使用）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `how-to-doit/exampleSite使用指南.md` | exampleSite使用指南 | how-to-doit | how-to-doit | DoIt |
| `how-to-doit/如何在 DoIt 中编写文章.md` | 如何在 DoIt 中编写文章 | how-to-doit | how-to-doit | DoIt |

---

## 7 pytorch / others

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `pytorch/Kaiming（He）初始化：方差推导与 ReLU 网络.md` | Kaiming（He）初始化：方差推导与 ReLU 网络 | PyTorch, Deep Learning, Kaiming, Xavier | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch DataLoader 技术解读.md` | PyTorch DataLoader 技术解读 | PyTorch, DataLoader, Deep Learning | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch Dataset 体系技术文档.md` | PyTorch Dataset 体系技术文档 | PyTorch, Dataset, Deep Learning | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch Tensor 工具函数技术文档：创建、计算、拼接与索引.md` | PyTorch Tensor 工具函数技术文档：创建、计算、拼接与索引 | PyTorch, Deep Learning | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch dtype 与类型体系.md` | PyTorch dtype 与类型体系 | PyTorch, dtype, Tensor, 混合精度 | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch lr曲线.md` | PyTorch lr曲线 | PyTorch, Deep Learning, LR | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch 优化器原理.md` | PyTorch 优化器原理 | PyTorch, Adam, optimization | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch 分布式训练与操作工具技术文档.md` | PyTorch 分布式训练与操作工具技术文档 | PyTorch, 分布式训练, Deep Learning | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch 概率分 (`torch.distributions`).md` | PyTorch 概率分 (`torch.distributions`) | PyTorch, distributions | PyTorch, distributions | PyTorch实践指南 |
| `pytorch/PyTorch 模型训练技术文档：求解器、参数配置与训练循环.md` | PyTorch 模型训练技术文档：求解器、参数配置与训练循环 | PyTorch, Deep Learning, 优化器, 训练 | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch 激活函数.md` | PyTorch 激活函数 | PyTorch, Deep Learning, Activation Function | PyTorch | PyTorch实践指南 |
| `pytorch/PyTorch 随机种子（Seed）全面解释.md` | PyTorch 随机种子（Seed）全面解释 | PyTorch, random-seed, reproducibility | PyTorch | PyTorch实践指南 |
| `pytorch/Pytorch Batch Size 与学习率缩放规则.md` | Pytorch Batch Size 与学习率缩放规则 | PyTorch, 优化器, 训练 | PyTorch | PyTorch实践指南 |
| `pytorch/Pytorch 权重初始化方法.md` | Pytorch 权重初始化方法 | PyTorch, Deep Learning, initialization, Xavier, Kaiming | PyTorch | PyTorch实践指南 |
| `torchcode-docs/00-总览：TorchCode 知识架构与学习路径.md` | 总览：TorchCode 知识架构与学习路径 | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/01-第一章：激活函数与基础组件（TorchCode）.md` | 第一章：激活函数与基础组件（TorchCode） | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/02-第二章：归一化技术（TorchCode）.md` | 第二章：归一化技术（TorchCode） | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/03-第三章：注意力机制（TorchCode）.md` | 第三章：注意力机制（TorchCode） | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/04-第四章：架构与模型组件（TorchCode）.md` | 第四章：架构与模型组件（TorchCode） | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/05-第五章：训练与优化（TorchCode）.md` | 第五章：训练与优化（TorchCode） | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/06-第六章：推理与解码策略（TorchCode）.md` | 第六章：推理与解码策略（TorchCode） | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/07-第七章：高级主题（TorchCode）.md` | 第七章：高级主题（TorchCode） | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `torchcode-docs/TorchCode 技术文档索引.md` | TorchCode 技术文档索引 | PyTorch, TorchCode | PyTorch | TorchCode 系列 |
| `others/幼儿园AI教育平台调研.md` | 幼儿园AI教育平台调研 |  |  |  |
| `deep-learning/CLS Token 原理和应用.md` | CLS Token 原理和应用 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/Flash Attention 原理公式与推导.md` | Flash Attention 原理公式与推导 | （空） | （空） | （空） |
| `deep-learning/IdealDenoiser 技术文档.md` | IdealDenoiser 技术文档 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/KL 散度与离散流匹配中的广义 KL 损失.md` | KL 散度与离散流匹配中的广义 KL 损失 | Deep Learning, flow matching | Deep Learning | Deep Learning系列 |
| `deep-learning/Loss Functions：系统化整理.md` | Loss Functions：系统化整理 | Deep Learning, Loss Functions | Deep Learning | Deep Learning系列 |
| `deep-learning/Speculative Decoding 原理与实现.md` | Speculative Decoding 原理与实现 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/技术要点.md` | 技术要点 |  | Deep Learning | Deep Learning系列 |
| `deep-learning/深度学习中的位置编码详解：从 Sinusoidal 到 RoPE 与 ALiBi.md` | 深度学习中的位置编码详解：从 Sinusoidal 到 RoPE 与 ALiBi | 位置编码, Transformer, RoPE, ALiBi, 深度学习, todo | Deep Learning | Deep Learning系列 |
| `deep-learning/深度学习中的常见归一化方法.md` | 深度学习中的常见归一化方法 | Deep Learning, Normalization | Deep Learning | Deep Learning系列 |
| `deep-learning/深度学习组件设计完整技术文档.md` | 深度学习组件设计完整技术文档 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/特征升维与降维.md` | 特征升维与降维 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/神经网络模型常用处理方式.md` | 神经网络模型常用处理方式 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/线性回归的最小二乘闭式解的推导.md` | 线性回归的最小二乘闭式解的推导 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/网络条件化方法.md` | 网络条件化方法 | Deep Learning | Deep Learning | Deep Learning系列 |
| `deep-learning/网络模块.md` | 网络模块 | Deep Learning | Deep Learning | Deep Learning系列 |
| `math/概率论常用公式技术文档.md` | 概率论常用公式技术文档 | 概率论 | 概率论 | 概率论系列 |
| `math/矩阵分解与奇异值分解（SVD）.md` | 矩阵分解与奇异值分解（SVD） | Deep Learning | Deep Learning | Deep Learning系列 |
| `smalldiffusion-docs/00-smalldiffusion 技术文档索引.md` | smalldiffusion 技术文档索引 | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `smalldiffusion-docs/01-smalldiffusion 项目总览.md` | smalldiffusion 项目总览 | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `smalldiffusion-docs/02-smalldiffusion 核心模块：diffusion.py.md` | smalldiffusion 核心模块：diffusion.py | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `smalldiffusion-docs/03-smalldiffusion 数据模块：data.py.md` | smalldiffusion 数据模块：data.py | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `smalldiffusion-docs/04-smalldiffusion 模型基础：model.py.md` | smalldiffusion 模型基础：model.py | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `smalldiffusion-docs/05-smalldiffusion 模型：model_dit.py.md` | smalldiffusion 模型：model_dit.py | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `smalldiffusion-docs/06-smalldiffusion 模型：model_unet.py.md` | smalldiffusion 模型：model_unet.py | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `smalldiffusion-docs/07-smalldiffusion 实战示例.md` | smalldiffusion 实战示例 | diffusion/flow, smalldiffusion | diffusion/flow, smalldiffusion | smalldiffusion系列 |
| `Analytic-Diffusion-Studio-docs/00-Analytic Diffusion Studio — 技术文档索引.md` | Analytic Diffusion Studio — 技术文档索引 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/01-Analytic Diffusion Studio — 项目总览.md` | Analytic Diffusion Studio — 项目总览 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/02-Analytic Diffusion Studio — 扩散模型理论基础.md` | Analytic Diffusion Studio — 扩散模型理论基础 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/03-Analytic Diffusion Studio — 配置系统.md` | Analytic Diffusion Studio — 配置系统 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/04-Analytic Diffusion Studio — 数据模块.md` | Analytic Diffusion Studio — 数据模块 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/05-Analytic Diffusion Studio — 模型基类与采样循环.md` | Analytic Diffusion Studio — 模型基类与采样循环 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/06-Analytic Diffusion Studio — Wiener 滤波去噪器.md` | Analytic Diffusion Studio — Wiener 滤波去噪器 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/07-Analytic Diffusion Studio — 最优贝叶斯去噪器.md` | Analytic Diffusion Studio — 最优贝叶斯去噪器 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/08-Analytic Diffusion Studio — 平滑最优去噪器.md` | Analytic Diffusion Studio — 平滑最优去噪器 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/09-Analytic Diffusion Studio — PCA Locality 去噪器.md` | Analytic Diffusion Studio — PCA Locality 去噪器 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/10-Analytic Diffusion Studio — 最近邻基线.md` | Analytic Diffusion Studio — 最近邻基线 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/11-Analytic Diffusion Studio — 基线 UNet 模型.md` | Analytic Diffusion Studio — 基线 UNet 模型 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/12-Analytic Diffusion Studio — 评估指标与实验流程.md` | Analytic Diffusion Studio — 评估指标与实验流程 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `Analytic-Diffusion-Studio-docs/13-Analytic Diffusion Studio — 工具模块.md` | Analytic Diffusion Studio — 工具模块 | diffusion/flow, Analytic Diffusion Studio | diffusion/flow, Analytic Diffusion Studio | Analytic Diffusion Studio系列 |
| `CS492D/1.扩散模型及其应用课程介绍.md` | 1.扩散模型及其应用课程介绍 | CS492D | CS492D | CS492D系列 |
| `CS492D/10.基于分数蒸馏的3D生成与编辑方法.md` | 10.基于分数蒸馏的3D生成与编辑方法 | CS492D | CS492D | CS492D系列 |
| `CS492D/11.扩散同步-SyncDiffusion.md` | 11.扩散同步-SyncDiffusion | CS492D | CS492D | CS492D系列 |
| `CS492D/12.扩散模型在逆问题中的应用.md` | 12.扩散模型在逆问题中的应用 | CS492D | CS492D | CS492D系列 |
| `CS492D/13.扩散模型在逆问题中的应用.md` | 13.扩散模型在逆问题中的应用 | CS492D | CS492D | CS492D系列 |
| `CS492D/14.扩散模型中的概率流ODE与DPM-Solver.md` | 14.扩散模型中的概率流ODE与DPM-Solver | CS492D | CS492D | CS492D系列 |
| `CS492D/15.流匹配-Flow Matching(1).md` | 15.流匹配-Flow Matching(1) | CS492D | CS492D | CS492D系列 |
| `CS492D/16.流匹配-Flow Matching(2).md` | 16.流匹配-Flow Matching(2) | CS492D | CS492D | CS492D系列 |
| `CS492D/17.扩散模型及其应用-课程总结.md` | 17.扩散模型及其应用-课程总结 | CS492D | CS492D | CS492D系列 |
| `CS492D/2.生成模型导论——GAN与VAE.md` | 2.生成模型导论——GAN与VAE | CS492D | CS492D | CS492D系列 |
| `CS492D/3.去噪扩散概率模型-DDPM1.md` | 3.去噪扩散概率模型-DDPM1 | CS492D | CS492D | CS492D系列 |
| `CS492D/4.去噪扩散概率模型-DDPM2.md` | 4.去噪扩散概率模型-DDPM2 | CS492D | CS492D | CS492D系列 |
| `CS492D/5.去噪扩散隐式模型-DDIM.md` | 5.去噪扩散隐式模型-DDIM | CS492D | CS492D | CS492D系列 |
| `CS492D/6.去噪扩散隐式模型-DDIM与无分类器引导-CFG.md` | 6.去噪扩散隐式模型-DDIM与无分类器引导-CFG | CS492D | CS492D | CS492D系列 |
| `CS492D/7.扩散模型中的条件生成与微调方法.md` | 7.扩散模型中的条件生成与微调方法 | CS492D | CS492D | CS492D系列 |
| `CS492D/8.扩散模型的零样本应用.md` | 8.扩散模型的零样本应用 | CS492D | CS492D | CS492D系列 |
| `CS492D/9.DDIM反演与分数蒸馏.md` | 9.DDIM反演与分数蒸馏 | CS492D | CS492D | CS492D系列 |
| `CS492D/Formula.md` | Formula | CS492D | CS492D | CS492D系列 |
| `rl/GAE（Generalized Advantage Estimation）.md` | GAE（Generalized Advantage Estimation） | RL | RL | RL系列 |
| `think/9大思维模型与方法论：从理论到实践的完整指南.md` | 9大思维模型与方法论：从理论到实践的完整指南 | 思维模型, 方法论, 决策, 问题解决, 认知科学 | 思维方法 | 思维工具系列 |

---

## 8 小结与说明

- 本索引**未包含**无 title 的文件（如 `others/prompt.md`）及 `draft.md` / `todo.md`；draft 文章仍列出便于统一维护。
- 新增文章后请同步更新本文档（含 §1 去重列表与对应分类表），并优先复用已有 tags/categories/series。

**延伸阅读**

- 仓库根目录 [AGENTS.md](AGENTS.md)：博客编写规范、Frontmatter 约定、目录与分类选择。
- 主题文档：`themes/DoIt/exampleSite/content/posts/` 下 built-in-shortcodes、content、extended-shortcodes 等。
