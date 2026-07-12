---
title: "7. 评估体系与实验结果"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "RT-2 评估协议、基准测试、泛化场景、涌现能力测试与消融实验。"
summary: "RT-2 评估体系与论文实验结果解读。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 8
series_weight: 8
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 7. 评估体系与实验结果

本章详解 RT-2 论文中的评估协议、基准测试、泛化场景、涌现能力测试与消融实验，并说明如何在本仓库上构建类似测试。

---

## 7.1 评估目标

RT-2 论文通过约 **6,000 次真实机器人评估** 回答四个问题：

1. 在已见任务与未见泛化场景上表现如何？
2. 能否观察到 Web 预训练带来的**涌现能力**？
3. 参数量与训练策略如何影响泛化？
4. 是否支持 Chain-of-Thought 推理？

---

## 7.2 机器人评估套件 (Robot Evaluation Suite)

| 评估维度 | 描述 | 方法 | 期望 |
|----------|------|------|------|
| **Seen Tasks** | 训练分布内任务 | 200+ 训练指令变体 | SOTA 成功率 |
| **Unseen Objects** | 新物体实例 | 训练中未见的物体 | 高泛化 |
| **Unseen Backgrounds** | 新背景 | 不同桌面/环境纹理 | 高泛化 |
| **Unseen Environments** | 新场景 | 如办公室桌面 vs 厨房 | 高泛化 |
| **Robustness** | 输入扰动 | 小幅图像/指令扰动 | 输出稳定 |
| **Efficiency** | 任务耗时 | 完成时间 | 可接受延迟 |
| **Scalability** | 任务复杂度 | 递增难度 | 性能缓降 |

### 7.2.1 硬件平台

- **7-DoF 移动操作机器人**（与 RT-1 相同）
- 动作空间：§5.2 定义的 8 维离散化动作
- 相机： onboard RGB

---

## 7.3 基线方法 (Baselines)

所有基线使用**相同机器人数据**：

| 基线 | 方法 | 挑战点 |
|------|------|--------|
| **RT-1** | 35M Transformer，无 VLM 预训练 | 验证 VLM 预训练是否必要 |
| **VC-1** | 视觉基础模型 ViT-L + USE 语言嵌入 | 对比机器人专用视觉预训练 |
| **R3M** | Ego4D 人类活动预训练视觉表征 | 对比视频预训练 |
| **MOO** | VLM 标记目标像素 + RT-1 策略 | 对比结构化 VLM 用法 |

### 7.3.1 基线关键区别

- **RT-1**：纯机器人数据，无 Web 知识
- **VC-1 / R3M**：仅视觉预训练，语言单独嵌入
- **MOO**：VLM 不参与端到端动作生成
- **RT-2**：VLM 端到端输出动作 Token，权重完全共享

---

## 7.4 泛化评估场景

### 7.4.1 四类 Unseen 测试

| 类别 | Easy | Hard |
|------|------|------|
| **Unseen Objects** | 常见新物体 | 难抓取/独特玩具 |
| **Unseen Backgrounds** | 简单新背景 | 复杂背景+新物体 |
| **Unseen Environments** | 厨房水槽变体 | 办公室桌面+显示器 |

### 7.4.2 Seen Tasks 组成

| 技能 | 指令数 |
|------|--------|
| Pick objects | 36 |
| Knock objects | 35 |
| Place upright | 35 |
| Move objects | 48 |
| Open/close drawers | 18 |
| Pick from / place into drawers | 36 |
| **合计** | **200+** |

Unseen 类别另有 **280+** 以 pick/place 为主的指令（详见论文 Appendix F.3）。

---

## 7.5 定量结果：整体泛化

### 7.5.1 主结果表

| 模型 | Seen Tasks | Unseen Objects | Unseen Backgrounds | Unseen Environments | **Unseen Avg** |
|------|------------|----------------|--------------------|-----------------------|----------------|
| R3M | 45% | 14% | 9% | 0% | 12% |
| VC-1 | 63% | 10% | 3% | 0% | 10% |
| RT-1 | 92% | 43% | 9% | 26% | 32% |
| MOO | 75% | 48% | 41% | 19% | 35% |
| **RT-2-PaLI-X-55B** | 91% | 62% | 48% | 63% | **62%** |
| **RT-2-PaLM-E-12B** | 93% | 76% | 71% | 36% | **62%** |

**关键发现**：
- Seen Tasks：RT-2 与 RT-1 相当（~92%）
- Unseen Avg：RT-2 ≈ **2× RT-1**（62% vs 32%）
- PaLM-E 在难泛化（新物体）更强；PaLI-X 在新环境更强

---

## 7.6 涌现能力评估 (Emergent Evaluation)

### 7.6.1 三类涌现技能

| 类别 | 示例指令 | 测试能力 |
|------|----------|----------|
| **Symbol Understanding** | "move apple to 3", "push coke on heart icon" | 数字/符号语义 |
| **Reasoning** | "move apple to cup with same color", "move X near sum of 2+1", 多语言 | 视觉推理、数学、语言 |
| **Person Recognition** | "move coke to person with glasses" | 人体属性识别 |

**重要**：这些指令**从未出现在机器人训练数据**中。

### 7.6.2 涌现评估结果

| 模型 | Symbol | Reasoning | Person | **Avg** |
|------|--------|-----------|--------|---------|
| VC-1 | 11% | 10% | 13% | 11% |
| RT-1 | 16% | 16% | 20% | 17% |
| **RT-2-PaLI-X-55B** | 82% | 46% | 53% | **60%** |
| **RT-2-PaLM-E-12B** | 36% | 43% | 43% | **40%** |

RT-2-PaLI-X 在符号理解上显著领先；RT-2-PaLM-E 在推理类更均衡。

---

## 7.7 规模与训练消融

| 模型规模 | 训练策略 | Unseen Objects | Unseen Backgrounds | Unseen Environments | **Avg** |
|----------|----------|----------------|--------------------|-----------------------|---------|
| RT-2-PaLI-X 5B | from scratch | 0% | 0% | 0% | 9% |
| RT-2-PaLI-X 5B | fine-tuning | 24% | 50% | 23% | 42% |
| RT-2-PaLI-X 5B | co-fine-tuning | 60% | 29% | 24% | 44% |
| RT-2-PaLI-X 55B | fine-tuning | 60% | 38% | 19% | 52% |
| RT-2-PaLI-X 55B | co-fine-tuning | 70% | 48% | 35% | **63%** |

---

## 7.8 Language-Table 仿真基准

| 模型 | 成功率 |
|------|--------|
| BC-Zero | 72 ± 3% |
| RT-1 | 74 ± 13% |
| LAVA | 77 ± 4% |
| **RT-2-PaLI-3B** | **90 ± 10%** |

开源环境：[google-research/language-table](https://github.com/google-research/language-table)

---

## 7.9 Chain-of-Thought 评估

RT-2-PaLM-E 支持 Plan → Action 两阶段输出：

| User 指令 | Plan | Action |
|-----------|------|--------|
| Pick up the object that is different | Pick rxbar chocolate | (tokens) |
| Move the green objects together | Move green can near green rice chip bag | (tokens) |
| I am sleepy, bring me a drink | Pick redbull can | (tokens) |

**评估方式**：定性案例 + 成功率对比（有/无 CoT prompt）。

---

## 7.10 失败案例分析

**典型失败**（Language-Table 真实部署）：
- 模型能正确理解语言并移动到目标物体
- 但无法控制**未见过的物体动力学**（如非训练分布的块体物理）

**启示**：VLA 迁移语义/视觉知识，但**物理动力学**仍受机器人数据分布限制。

---

## 7.11 本仓库测试体系

### 7.11.1 单元测试 (`tests/test.py`)

本仓库提供 **模型正确性** 测试，非机器人任务成功率测试：

| 测试 | 验证 |
|------|------|
| 形状测试 | 各配置下输出 tensor 形状 |
| 异常测试 | 缺少参数、错误 encoder 模式 |
| 参数扰动 | 运行时修改 dim/depth/heads |

```bash
pytest tests/test.py -v
```

### 7.11.2 建议的集成评估流程

```python
# 1. 加载模型与 ActionTokenizer
# 2. 对 hold-out 验证集计算 token 预测准确率
# 3. 反量化动作，计算与 GT 的 L2 误差
# 4. （可选）接入仿真环境 roll-out success rate
```

---

## 7.12 评估指令示例 (Appendix F.3 摘要)

### Symbol Understanding
- "move apple to 3"
- "push coke can on top of heart"

### Reasoning
- "move the apple to cup with same color"
- "move banana near the sum of two plus one"
- "mueve la manzana al vaso verde" (西班牙语)

### Person Recognition
- "move the coke can to the person with glasses"

---

## 7.13 评估方法学要点

| 要点 | 说明 |
|------|------|
| A/B Testing | 涌现评估中四模型在相同条件下顺序测试 |
| 重复次数 | 每条指令 1–5 次（涌现类固定 5 次） |
| 成功率定义 | 任务完成判定（与 RT-1 协议一致） |
| 统计 | 报告均值 ± 标准差（Language-Table） |

---

## 7.14 参考文献

| 资源 | 链接 |
|------|------|
| RT-2 论文 §4 | https://arxiv.org/abs/2307.15818 |
| RT-1 评估协议 | https://arxiv.org/abs/2212.06817 |
| VC-1 | https://arxiv.org/abs/2306.14824 |
| R3M | https://arxiv.org/abs/2202.09605 |
| MOO | https://arxiv.org/abs/2303.00987 |
| 项目 Demo | https://robotics-transformer2.github.io |

---

## 7.15 相关章节

- 训练数据 → [06-training-datasets.md](./06-training-datasets.md)
- 动作 Token → [05-action-tokenization.md](./05-action-tokenization.md)
