---
title: "1. VLA 理论与 RT-2 算法原理"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "从 VLA 范式出发，解读 RT-2 论文核心算法、数学公式与训练推理流程。"
summary: "RT-2 VLA 理论与算法原理详解。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 2
series_weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 1. VLA 理论与 RT-2 算法原理

本章从 Vision-Language-Action (VLA) 范式出发，完整解读 RT-2 论文中的核心算法、数学公式与训练推理流程。

---

## 1.1 问题定义

### 1.1.1 传统机器人学习的困境

机器人策略学习通常面临 **数据稀缺** 与 **泛化不足** 的矛盾：

- 收集百万级机器人轨迹成本极高
- 纯机器人数据训练的模型难以理解 "把苹果放到**相同颜色**的杯子里" 这类需要语义推理的指令
- 将 VLM 仅用于**高层规划**（输出子任务序列）时，低层控制器无法受益于 Web 预训练知识

### 1.1.2 VLA 的核心命题

> **能否将大规模预训练 VLM 直接整合到低层机器人控制中，以提升泛化并启用涌现的语义推理？**

RT-2 的回答是：**可以**——通过将动作表示为文本 Token，与语言 Token 共享输出空间，对 VLM 进行 Co-Fine-Tuning。

---

## 1.2 Vision-Language-Action (VLA) 范式

### 1.2.1 定义

**VLA 模型**是一类将视觉、语言、动作统一在同一 Transformer 输出空间的模型：

$$
f_\theta: (\mathbf{I}, \mathbf{T}) \rightarrow \mathbf{a}
$$

其中：
- $\mathbf{I}$：机器人相机图像
- $\mathbf{T}$：自然语言任务指令（Token 序列）
- $\mathbf{a}$：机器人动作（同样表示为 Token 序列）

### 1.2.2 与相关范式的区别

| 范式 | 输入 | 输出 | 权重共享 |
|------|------|------|----------|
| VLM | 图像 + 文本 | 文本 | - |
| VLM + Planner | 图像 + 文本 | 文本计划 | 规划与执行分离 |
| **VLA (RT-2)** | 图像 + 文本 | **文本或动作 Token** | **完全共享** |

关键优势：**无需新增动作专用层**，预训练 VLM 的计算投资可直接复用。

---

## 1.3 RT-2 整体流程

### 1.3.1 训练阶段

```
┌─────────────────────────────────────────────────────────────┐
│                    Co-Fine-Tuning 数据混合                    │
├──────────────────────────┬──────────────────────────────────┤
│   Web 规模 VLM 数据       │   机器人演示数据                    │
│   - VQA                    │   - 图像 + 指令 + 动作序列         │
│   - 图像描述               │   - 动作 → Token 字符串            │
│   - 视觉推理               │   - VQA 格式包装                   │
└──────────────────────────┴──────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  VLM Backbone   │
                    │  (PaLI-X/PaLM-E)│
                    └─────────────────┘
                              │
                              ▼
                    最小化交叉熵损失
                    L = -Σ log P(token_t | token_{<t}, I)
```

### 1.3.2 推理阶段（闭环控制）

```
观测图像 I_t  +  指令 T
        │
        ▼
   VLA 模型自回归生成动作 Token: "1 128 91 241 5 101 127"
        │
        ▼
   De-Tokenize → 连续动作向量
        │
        ▼
   机器人执行 → 新观测 I_{t+1} → 循环
```

---

## 1.4 预训练 VLM 骨干

RT-2 基于两种 VLM 实例化：

### 1.4.1 RT-2-PaLI-X

| 组件 | 规格 |
|------|------|
| 视觉编码器 | ViT-22B (或 ViT-G/14 for 3B) |
| 语言骨干 | UL2 Encoder-Decoder (~32B, 50层) |
| 融合方式 | 图像 Token 经投影层送入 Enc-Dec |
| Tokenizer | 整数 0-1000 各有独立 Token |

**架构**：Encoder-Decoder，适合 seq2seq 任务。

### 1.4.2 RT-2-PaLM-E

| 组件 | 规格 |
|------|------|
| 视觉编码器 | ViT-4B (ViT-22B-e) |
| 语言骨干 | PaLM-12B (Decoder-Only) |
| 融合方式 | 图像嵌入与文本 Token **拼接** 输入 Decoder |
| Tokenizer | 覆盖 256 个最少使用 Token 作为动作词表 |

**架构**：Decoder-Only，支持 Chain-of-Thought 推理。

### 1.4.3 与本仓库实现的对应

本仓库 `RT2` 类采用 **ViT Encoder + Cross-Attention Decoder** 结构，在概念上介于 PaLI-X（Enc-Dec + Cross-Attn）与简化 ViT-LM 之间，但规模远小于论文版本。

---

## 1.5 动作空间与离散化

### 1.5.1 动作向量定义（继承 RT-1）

RT-2 沿用 RT-1 的 **7-DoF 移动操作** 动作空间：

$$
\mathbf{a} = (\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g, \tau)
$$

| 维度 | 含义 | 类型 |
|------|------|------|
| $\Delta x, \Delta y, \Delta z$ | 末端执行器位置增量 | 连续 |
| $\Delta r_x, \Delta r_y, \Delta r_z$ | 末端执行器旋转增量 | 连续 |
| $g$ | 夹爪开合程度 | 连续 |
| $\tau$ | 终止标志 (terminate) | 离散 {0, 1} |

### 1.5.2 均匀离散化

每个连续维度均匀划分为 **256 个 bin**：

$$
b_i = \left\lfloor \frac{a_i - a_i^{\min}}{a_i^{\max} - a_i^{\min}} \times 255 \right\rfloor, \quad b_i \in \{0, 1, \ldots, 255\}
$$

逆映射（de-tokenize）：

$$
\hat{a}_i = a_i^{\min} + \frac{b_i}{255} (a_i^{\max} - a_i^{\min})
$$

### 1.5.3 动作字符串格式

8 个整数 bin 索引拼接为空格分隔的字符串：

```
"terminate Δpos_x Δpos_y Δpos_z Δrot_x Δrot_y Δrot_z gripper"
```

示例：`"1 128 91 241 5 101 127"` 表示终止=1，位置/旋转/夹爪各维 bin 索引。

### 1.5.4 Language-Table 变体

在 Language-Table 仿真中，动作格式简化为 2D：

```
"X Y"   # X, Y ∈ {-10, -9, ..., +9, +10}
```

表示末端执行器 2D 笛卡尔位移 setpoint。

---

## 1.6 Token 映射策略

### 1.6.1 PaLI-X：直接整数 Token

PaLI-X 的 SentencePiece 词表中，整数 0-1000 各有唯一 Token，因此：

$$
\text{Token}(b_i) = \text{SPM\_Token}(b_i), \quad b_i \in [0, 255]
$$

### 1.6.2 PaLM-E：覆盖低频 Token

PaLM-E 无便捷数字 Token，选取 **256 个最少使用的 Token** 覆盖动作 bin：

$$
\mathcal{V}_{\text{action}} = \{t_1, t_2, \ldots, t_{256}\} \subset \mathcal{V}_{\text{least\_freq}}
$$

这属于 **Symbol Tuning**（[Wei et al., 2023](https://arxiv.org/abs/2305.08298)）的一种应用。

---

## 1.7 训练目标

### 1.7.1 自回归语言建模损失

给定多模态输入，优化标准交叉熵：

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, \mathbf{I})
$$

其中 $y_t$ 可以是自然语言 Token 或动作 Token。

### 1.7.2 Co-Fine-Tuning

**关键发现**：仅在机器人数据上微调会导致灾难性遗忘；**混合 Web 数据与机器人数据** 训练效果更好。

每个 batch 中通过 **提高机器人数据采样权重** 平衡两类数据：

$$
P(\text{batch}) = w_{\text{robot}} \cdot P_{\text{robot}} + (1 - w_{\text{robot}}) \cdot P_{\text{web}}
$$

论文中 RT-2-PaLI-X 混合比：Robot 50% / Web 50%；RT-2-PaLM-E：Robot 66% / Web 34%。

### 1.7.3 输出约束 (Output Constraint)

推理时，当 prompt 为机器人动作任务，**仅采样有效动作 Token**（256 个 bin 对应 Token），避免生成无效自然语言 Token。

---

## 1.8 Chain-of-Thought (CoT) 推理

RT-2-PaLM-E 支持 CoT：在动作 Token 前生成 **Plan** 步骤：

```
User: Pick up the object that is different from all other objects
Plan: Pick rxbar chocolate
Action: 132 114 128 5 25 156
```

训练时在数据中加入 `"Plan: ..."` 前缀，使模型先推理再执行。

---

## 1.9 涌现能力 (Emergent Capabilities)

Web 预训练知识迁移到机器人控制，产生三类涌现能力：

| 类别 | 示例指令 | 所需能力 |
|------|----------|----------|
| **符号理解** | "move apple to 3" | 数字/图标语义 |
| **推理** | "move apple to cup with same color" | 视觉推理、数学、多语言 |
| **人物识别** | "move coke can to person with glasses" | 人体/属性识别 |

这些指令**从未出现在机器人训练数据**中，能力来自 VLM 预训练。

---

## 1.10 实时推理

| 模型 | 参数量 | 推理频率 | 部署方式 |
|------|--------|----------|----------|
| RT-2-PaLI-X-55B | 55B | 1-3 Hz | 多 TPU 云端服务 |
| RT-2-PaLI-X-5B | 5B | ~5 Hz | 云端 |
| RT-2-PaLI-3B | 3B | ~5 Hz | 本地/仿真 |

论文通过 **网络查询云端 TPU 服务** 实现闭环控制，是迄今最大规模直接用于机器人控制的模型。

---

## 1.11 核心实验结论

### 1.11.1 泛化性能（Unseen Average）

| 模型 | Seen | Unseen Avg |
|------|------|------------|
| R3M | 45% | 12% |
| VC-1 | 63% | 10% |
| RT-1 | 92% | 32% |
| MOO | 75% | 35% |
| **RT-2-PaLI-X-55B** | 91% | **62%** |
| **RT-2-PaLM-E-12B** | 93% | **62%** |

RT-2 在未见物体/背景/环境上约为 RT-1 的 **2 倍** 成功率。

### 1.11.2 涌现评估

| 模型 | Symbol | Reasoning | Person | Avg |
|------|--------|-----------|--------|-----|
| RT-1 | 16% | 16% | 20% | 17% |
| RT-2-PaLI-X-55B | 82% | 46% | 53% | 60% |
| RT-2-PaLM-E-12B | 36% | 43% | 43% | 40% |

### 1.11.3 训练策略消融

| 策略 | Unseen Avg |
|------|------------|
| 5B from scratch | 9% |
| 5B fine-tuning (robot only) | 42% |
| 5B co-fine-tuning | 44% |
| 55B co-fine-tuning | **63%** |

**结论**：VLM 预训练 + Co-Fine-Tuning + 大参数量 三者缺一不可。

---

## 1.12 数学附录：Transformer 注意力（本仓库依赖）

本仓库使用的 zetascale 库实现了标准 Scaled Dot-Product Attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

- **Encoder**（ViT）：$M=0$（双向，无因果掩码）
- **Decoder**（语言）：$M$ 为因果掩码（上三角 $-\infty$）
- **Cross-Attention**：$Q$ 来自解码器，$K,V$ 来自编码器图像嵌入

**Flash Attention** 通过分块计算降低显存：

$$
O = \text{softmax}(QK^\top / \sqrt{d}) V \approx \text{BlockSoftmax}(Q, K, V)
$$

详见 [04-decoder-autoregression.md](./04-decoder-autoregression.md)。

---

## 1.13 参考文献与资源

### 论文

| 文献 | 链接 |
|------|------|
| RT-2 (CoRL 2023) | [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) |
| RT-1 | [arXiv:2212.06817](https://arxiv.org/abs/2212.06817) |
| PaLM-E | [arXiv:2303.03378](https://arxiv.org/abs/2303.03378) |
| PaLI-X | [arXiv:2305.18517](https://arxiv.org/abs/2305.18517) |
| ViT | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| Co-Fine-Tuning / Symbol Tuning | [arXiv:2305.08298](https://arxiv.org/abs/2305.08298) |
| Flash Attention | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |

### 博客与解读

| 资源 | 链接 |
|------|------|
| Google DeepMind 官方博客 | https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/ |
| 项目主页与 Demo | https://robotics-transformer2.github.io |
| Two Minute Papers 视频 | https://www.youtube.com/watch?v=Uq__fZ7b9Xg |

### 开源项目

| 项目 | 链接 |
|------|------|
| 本仓库 | https://github.com/kyegomez/RT-2 |
| Zeta 组件库 | https://github.com/kyegomez/zeta |
| OpenVLA | https://github.com/openvla/openvla |
| Octo | https://github.com/octo-models/octo |
