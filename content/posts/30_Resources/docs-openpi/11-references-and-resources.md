---
title: "11 参考文献与外部资源"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "汇总 openpi 涉及的核心论文、官方博客、开源项目与文档链接，按主题分类便于延伸阅读。"
summary: "openpi 参考文献与外部资源索引。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 11
series_weight: 11

> 本章汇总 openpi 涉及的核心论文、官方博客、开源项目与文档链接，按主题分类，便于延伸阅读与算法溯源。

---

## 11.1 Physical Intelligence 官方资源

| 资源 | 链接 | 与本项目关系 |
| --- | --- | --- |
| **π₀ 博客** | [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/blog/pi0) | 流匹配 VLA 的总体介绍；openpi 中 `Pi0` 的实现基础 |
| **π₀.₅ 博客** | [π₀.₅ and Knowledge Insulation](https://www.physicalintelligence.company/blog/pi05) | 离散状态输入、AdaRMS 时间注入、知识隔离训练 |
| **FAST 论文/研究页** | [FAST: Efficient Action Tokenization for VLA Models](https://www.physicalintelligence.company/research/fast) | π₀-FAST 的动作离散化方案 |
| **Knowledge Insulation** | [Knowledge Insulation](https://www.physicalintelligence.company/research/knowledge_insulation) | π₀.₅ 预训练方法论 |
| **openpi GitHub** | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 本仓库源码 |
| **预训练检查点** | `gs://openpi-assets/checkpoints/` | 通过 `openpi.shared.download.maybe_download` 自动缓存 |

---

## 11.2 流匹配与扩散生成模型

### 核心概念

流匹配（Flow Matching）学习从噪声分布到数据分布的**速度场** $v_\theta(x_t, t)$，通过 ODE 积分生成样本。openpi 采用**直线插值（Rectified Flow）**：

$$x_t = t \cdot \epsilon + (1-t) \cdot a, \quad u_t = \epsilon - a, \quad \mathcal{L} = \mathbb{E}\|v_\theta(x_t,t) - u_t\|^2$$

代码中 $t=1$ 为纯噪声、$t=0$ 为目标动作（与 π₀ 论文符号相反，见 `pi0.py` 注释）。

### 参考文献

| 论文 | 链接 | 要点 |
| --- | --- | --- |
| **Flow Matching for Generative Modeling** (Lipman et al., 2023) | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | 流匹配基础框架 |
| **Rectified Flow** (Liu et al., 2023) | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) | 直线流路径，openpi 采用的插值形式 |
| **Conditional Flow Matching** (Tong et al., 2024) | [arXiv:2302.00482](https://arxiv.org/abs/2302.00482) | 条件生成（以观测为条件的动作生成） |
| **Score-Based Generative Modeling** (Song et al., 2021) | [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) | 扩散/分数模型背景（对比理解） |

### 相关开源项目

- [torchcfm](https://github.com/atong01/conditional-flow-matching) — PyTorch 条件流匹配参考实现
- [flow_matching](https://github.com/facebookresearch/flow_matching) — Meta 流匹配库

---

## 11.3 视觉-语言-动作（VLA）与机器人策略

| 论文/项目 | 链接 | 与 openpi 对比 |
| --- | --- | --- |
| **RT-2** (Brohan et al., 2023) | [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) | 自回归 VLA；openpi 的 `BinningTokenizer` 参考 RT-2 分桶 |
| **OpenVLA** (Kim et al., 2024) | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) | 开源 VLA；LeRobot 生态 |
| **Octo** (Team et al., 2024) | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | Transformer 策略；动作 chunk 概念类似 |
| **Diffusion Policy** (Chi et al., 2023) | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) | 扩散动作生成；π₀ 用流匹配替代 |
| **ACT** (Zhao et al., 2023) | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | 动作 chunk + CVAE；ALOHA 常用基线 |

---

## 11.4 骨干网络

### PaliGemma / Gemma

| 资源 | 链接 | 在 openpi 中的用途 |
| --- | --- | --- |
| **PaliGemma** | [HuggingFace: google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) | 视觉-语言主干（SigLIP + Gemma-2B） |
| **Gemma** (Google, 2024) | [arXiv:2403.08295](https://arxiv.org/abs/2403.08295) | 语言模型与动作专家架构 |
| **big_vision PaliGemma** | [github.com/google-research/big_vision](https://github.com/google-research/big_vision) | `gemma.py`、`siglip.py` 代码来源 |

### SigLIP / ViT

| 资源 | 链接 | 在 openpi 中的用途 |
| --- | --- | --- |
| **SigLIP** (Zhai et al., 2023) | [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | 视觉编码器 |
| **ViT** (Dosovitskiy et al., 2021) | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | SigLIP 视觉塔基础 |
| **MoCo v3 位置编码** | [arXiv:2104.02057](https://arxiv.org/abs/2104.02057) | `posemb_sincos_2d` 参考 |

### 注意力与归一化

| 概念 | 参考 | openpi 实现 |
| --- | --- | --- |
| **RoPE** | [Su et al., 2021](https://arxiv.org/abs/2104.09864) | `gemma._apply_rope` |
| **MQA/GQA** | [Shazeer, 2019](https://arxiv.org/abs/1911.02150) | `num_kv_heads=1` |
| **AdaRMS / FiLM** | [Perez et al., 2018](https://arxiv.org/abs/1709.07871) | π₀.₅ 时间条件注入 |
| **GeGLU** | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) | `FeedForward` 门控 MLP |
| **LoRA** | [Hu et al., 2022](https://arxiv.org/abs/2106.09685) | `lora.py` 低秩微调 |

---

## 11.5 FAST 动作分词器

| 资源 | 链接 | 说明 |
| --- | --- | --- |
| **FAST 处理器** | [HuggingFace: physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) | `FASTTokenizer` 加载的预训练分词器 |
| **FSQ** (Mentzer et al., 2024) | [arXiv:2309.15505](https://arxiv.org/abs/2309.15505) | 有限标量量化；`FSQTokenizer` 基线 |
| **LFQ** | [Yu et al., 2024](https://arxiv.org/abs/2310.05737) | 查找自由量化变体 |

FAST 将连续动作轨迹 $a \in \mathbb{R}^{H \times D}$ 编码为离散 token 序列，再映射到 PaliGemma 词表末尾：

$$\text{pg\_token} = \text{vocab\_size} - 1 - 128 - \text{fast\_token}$$

解码时 `ExtractFASTActions` 反向执行此映射。

---

## 11.6 数据格式与训练基础设施

| 项目 | 链接 | 在 openpi 中的角色 |
| --- | --- | --- |
| **LeRobot** | [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) | 默认数据集格式与加载 |
| **RLDS / TFDS** | [github.com/google-research/rlds](https://github.com/google-research/rlds) | DROID 大规模训练 |
| **DROID 数据集** | [droid-dataset.github.io](https://droid-dataset.github.io/) | π₀-FAST-DROID、π₀.₅-DROID 训练数据 |
| **LIBERO** | [libero-project.github.io](https://libero-project.github.io/) | 仿真 benchmark 微调示例 |
| **ALOHA** | [tonyzhaozh.github.io/aloha](https://tonyzhaozh.github.io/aloha/) | 双臂遥操作平台 |
| **Orbax Checkpoint** | [github.com/google/orbax](https://github.com/google/orbax) | JAX 检查点管理 |
| **Flax NNX** | [flax.readthedocs.io/en/latest/nnx](https://flax.readthedocs.io/en/latest/nnx/index.html) | JAX 模型 API |
| **Optax** | [github.com/google-deepmind/optax](https://github.com/google-deepmind/optax) | JAX 优化器 |
| **tyro** | [github.com/brentyi/tyro](https://github.com/brentyi/tyro) | CLI 配置解析 |

---

## 11.7 分布式训练

| 主题 | 参考 | openpi 实现 |
| --- | --- | --- |
| **FSDP** | [Meta FSDP 博客](https://engineering.fb.com/2021/07/15/open-source/fsdp/) | `training/sharding.py`（仅 JAX） |
| **PyTorch DDP** | [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | `scripts/train_pytorch.py` |
| **JAX sharding** | [JAX sharding guide](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) | `fsdp_sharding`、`activation_sharding_constraint` |

---

## 11.8 推理与服务

| 主题 | 参考 | openpi 实现 |
| --- | --- | --- |
| **WebSocket 协议** | [RFC 6455](https://datatracker.ietf.org/doc/html/rfc6455) | `websocket_policy_server.py` |
| **msgpack** | [msgpack.org](https://msgpack.org/) | 观测/动作序列化 |
| **msgpack-numpy** | [github.com/lebedov/msgpack-numpy](https://github.com/lebedov/msgpack-numpy) | `msgpack_numpy.py`（安全子集，不回退 pickle） |

用户向文档：
- [远程推理](../remote_inference.md)
- [归一化统计](../norm_stats.md)
- [Docker 部署](../docker.md)

---

## 11.9 数学符号速查（全书统一）

| 符号 | 含义 |
| --- | --- |
| $a$ | 真实动作轨迹，shape `[action_horizon, action_dim]` |
| $\epsilon$ | 标准高斯噪声 |
| $t$ | 流匹配时间步，$t \in [0,1]$；代码中 $t=1$ 为噪声端 |
| $x_t$ | 插值状态 $x_t = t\epsilon + (1-t)a$ |
| $u_t$ | 目标速度 $u_t = \epsilon - a$ |
| $v_\theta$ | 模型预测速度场 |
| $H, D$ | 动作 horizon 与维度（代码中 `action_horizon`, `action_dim`） |
| $B$ | batch size |

---

## 11.10 推荐阅读路径（按目标）

**理解 π₀ 流匹配算法**
1. [02 流匹配模型](02-models-flow-matching.md) → Flow Matching 论文 → π₀ 博客

**理解 π₀-FAST**
1. [03 π₀-FAST](03-models-pi0-fast.md) → FAST 研究页 → RT-2 论文（对比）

**微调 LIBERO**
1. [05 数据管线](05-data-pipeline.md) → [06 训练系统](06-training-system.md) → `examples/libero/README.md`

**部署到真机**
1. [07 推理服务](07-inference-policy-serving.md) → [08 客户端](08-client-runtime.md) → [remote_inference.md](../remote_inference.md)

**PyTorch 迁移**
1. [09 PyTorch 实现](09-pytorch-implementation.md) → [gemma_pytorch_detailed.md](gemma_pytorch_detailed.md) → `examples/convert_jax_model_to_pytorch.py`
