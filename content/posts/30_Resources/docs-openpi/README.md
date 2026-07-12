---
title: "openpi 技术文档索引"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "openpi 机器人 VLA 库的技术文档索引，涵盖架构、模型、训练、推理与客户端运行时。"
summary: "openpi 技术文档索引、术语表与代码地图。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# openpi 技术文档

本套文档对 [docs openpi](https://github.com/Physical-Intelligence/openpi)（Physical Intelligence 团队开源的机器人视觉-语言-动作模型库）进行系统化的技术解读。文档遵循「先总体、后局部」的组织方式：先给出整体架构与知识结构，再按逻辑分章节深入到每个模块的类、函数、算法与数据流。

> 阅读建议：如果你是第一次接触本项目，请从「01 架构总览」开始，建立全局认知后再按需跳读具体章节。每章都尽量做到自包含，并在关键处给出可运行的代码片段、公式与示意图。

---

## 文档索引

| 章节 | 文件 | 主题 | 适合谁读 |
| --- | --- | --- | --- |
| 00 | 本文件 `README.md` | 文档索引、全局导航、术语表 | 所有人 |
| 01 | [`01-architecture-overview.md`](01-architecture-overview.md) | 架构总览：三类模型、五大子系统、数据流、设计取舍 | 所有人（必读） |
| 02 | [`02-models-flow-matching.md`](02-models-flow-matching.md) | π₀ / π₀.₅ 流匹配（Flow Matching）模型原理与 JAX 实现 | 模型/算法 |
| 03 | [`03-models-pi0-fast.md`](03-models-pi0-fast.md) | π₀-FAST 自回归模型与 FAST 动作分词器 | 模型/算法 |
| 04 | [`04-backbone-tokenizers.md`](04-backbone-tokenizers.md) | 骨干网络：Gemma 混合专家、SigLIP/ViT、LoRA、各类 Tokenizer | 模型/算法 |
| 05 | [`05-data-pipeline.md`](05-data-pipeline.md) | 数据管线：Transform 体系、归一化、DataLoader、RLDS/LeRobot | 数据/训练 |
| 06 | [`06-training-system.md`](06-training-system.md) | 训练系统：配置注册、JAX 训练循环、PyTorch DDP、FSDP、EMA、检查点 | 训练 |
| 07 | [`07-inference-policy-serving.md`](07-inference-policy-serving.md) | 推理与服务：Policy 封装、WebSocket 服务端、归一化往返 | 部署/服务 |
| 08 | [`08-client-runtime.md`](08-client-runtime.md) | 客户端运行时：openpi-client 包、Runtime 循环、动作分块、msgpack 协议 | 机器人集成 |
| 09 | [`09-pytorch-implementation.md`](09-pytorch-implementation.md) | PyTorch 实现细节：PI0Pytorch、PaliGemmaWithExpertModel、transformers patch | PyTorch 用户 |
| 10 | [`10-network-architecture-diagrams.md`](10-network-architecture-diagrams.md) | 模型网络结构框架图：逐层张量形状、维度标注、三类模型对照 | 模型/算法 |
| — | [`gemma_pytorch_detailed.md`](gemma_pytorch_detailed.md) | PaliGemmaWithExpert 逐层 PyTorch 实现详解（09 章补充） | PyTorch 用户 |
| 11 | [`11-references-and-resources.md`](11-references-and-resources.md) | 论文、博客、开源项目与数学符号参考 | 延伸阅读 |
| 12 | [`12-examples-and-scripts.md`](12-examples-and-scripts.md) | examples/、scripts/ 入口与扩展新平台指南 | 工程落地 |

### 配套用户文档（docs/ 根目录）

| 文件 | 主题 |
| --- | --- |
| [`../remote_inference.md`](../remote_inference.md) | WebSocket 远程推理快速上手 |
| [`../norm_stats.md`](../norm_stats.md) | 归一化统计计算与重载 |
| [`../docker.md`](../docker.md) | Docker 安装与部署 |

---

## 全局术语表（速查）

| 术语 | 含义 |
| --- | --- |
| **VLA** | Vision-Language-Action 模型，输入图像+语言+本体状态，输出机器人动作序列 |
| **π₀ (pi0)** | 基于流匹配（flow matching）的连续动作 VLA |
| **π₀.₅ (pi05)** | π₀ 的升级版，状态进入离散语言 token，时间步用 AdaRMS 注入，泛化更好 |
| **π₀-FAST** | 自回归 VLA，用 FAST 分词器把动作离散成 token，再用 LLM 自回归生成 |
| **PaliGemma** | Google 的视觉-语言模型（SigLIP 视觉塔 + Gemma-2B 语言模型），作为主干 |
| **Action Expert** | 一个独立的 Gemma-300M 专家，与 PaliGemma 共享注意力但独立权重，负责动作 |
| **Flow Matching** | 学习一个速度场，把噪声沿 ODE 积分回数据分布的生成范式 |
| **AdaRMS** | Adaptive RMSNorm，用条件向量（这里是时间步）调制归一化的 scale/shift/gate |
| **Action Horizon** | 单次推理输出的动作步数（动作块/chunk 的长度） |
| **Action Dim** | 动作向量维度（如 ALOHA 14 维、DROID 8 维，模型内部统一 pad 到 32） |
| **Observation** | 模型输入的结构化容器：images / image_masks / state / tokenized_prompt 等 |
| **Transform** | 数据变换函数，实现 `DataTransformFn` 协议，可组合成管线 |
| **Norm Stats** | 归一化统计量（mean/std/q01/q99），保证训练与推理一致 |
| **LeRobot** | HuggingFace 的机器人数据集格式与加载库，本项目默认数据来源 |
| **RLDS** | Reinforcement Learning Datasets（TFDS）格式，用于超大数据集（DROID） |
| **FSDP** | Fully-Sharded Data Parallelism，把参数/梯度分片到多卡以省显存（仅 JAX） |
| **EMA** | Exponential Moving Average，对权重做指数滑动平均（仅 JAX 训练） |
| **NNX** | Flax 的新式神经网络 API；本项目 JAX 模型基于 NNX，部分用 linen bridge |

---

## 代码地图（关键路径）

```
src/openpi/
├── models/                 # JAX 模型与骨干（见 02/03/04 章）
│   ├── model.py            # Observation/BaseModel/BaseModelConfig 抽象
│   ├── pi0.py              # π₀ / π₀.₅ 流匹配模型
│   ├── pi0_config.py       # Pi0Config
│   ├── pi0_fast.py         # π₀-FAST 自回归模型
│   ├── gemma.py            # Gemma 混合专家 Transformer（pi0 用）
│   ├── gemma_fast.py       # Gemma（FAST 用，带 KV 缓存）
│   ├── siglip.py           # SigLIP/ViT 视觉塔
│   ├── lora.py             # LoRA 实现
│   └── tokenizer.py        # Paligemma/FAST/Binning/FSQ 分词器
├── models_pytorch/         # PyTorch 模型（见 09 章）
│   ├── pi0_pytorch.py      # PI0Pytorch
│   ├── gemma_pytorch.py    # PaliGemmaWithExpertModel
│   └── preprocessing_pytorch.py
├── training/               # 训练系统（见 06 章）
│   ├── config.py           # TrainConfig / DataConfig / 配置注册表
│   ├── data_loader.py      # 统一数据加载器
│   ├── optimizer.py        # 优化器与学习率调度
│   ├── sharding.py         # FSDP 分片
│   ├── checkpoints.py      # Orbax 检查点
│   ├── weight_loaders.py   # 预训练权重加载
│   └── droid_rlds_dataset.py
├── policies/               # 策略与机器人适配（见 05/07 章）
│   ├── policy.py           # Policy / PolicyRecorder
│   ├── policy_config.py    # create_trained_policy 工厂
│   ├── aloha_policy.py     # ALOHA 输入/输出变换
│   ├── libero_policy.py    # LIBERO 输入/输出变换
│   └── droid_policy.py     # DROID 输入/输出变换
├── serving/
│   └── websocket_policy_server.py  # WebSocket 服务端（见 07 章）
├── shared/                 # 通用工具
│   ├── normalize.py        # 归一化统计
│   ├── download.py         # 资产下载缓存
│   ├── image_tools.py      # 图像 resize/pad
│   └── array_typing.py     # 运行时类型检查
└── transforms.py           # Transform 体系核心（见 05 章）

packages/openpi-client/     # 轻量客户端包（见 08 章）
└── src/openpi_client/
    ├── websocket_client_policy.py
    ├── action_chunk_broker.py
    ├── msgpack_numpy.py
    └── runtime/            # Runtime / Agent / Environment / Subscriber

scripts/
├── train.py                # JAX 训练入口
├── train_pytorch.py        # PyTorch 训练入口
├── serve_policy.py         # 策略服务入口
└── compute_norm_stats.py   # 计算归一化统计
```
