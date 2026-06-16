---
title: Q-Guided Flow (QGF) 技术文档索引
subtitle: ""
date: 2026-06-17T10:26:59+08:00
# lastmod: 2026-06-17T10:26:59+08:00
draft: false
authors: [Steven]
description: ""
tags: [diffusion/flow, qgf]
categories: [qgf]
series: [qgf-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# Q-Guided Flow (QGF) 技术文档索引

本目录包含 [Q-Guided Flow](https://arxiv.org/pdf/2606.11087) 代码库的算法原理与实现解读，面向「先建立全局图景、再深入模块」的阅读顺序。

## 文档地图

| 序号 | 文件 | 内容 |
|------|------|------|
| 0 | [README.md](./README.md) | 本索引与整体架构总览 |
| 1 | [01-algorithm-overview.md](./01-algorithm-overview.md) | 问题设定、方法分类、训练/推理流程、方法对比 |
| 2 | [02-flow-matching-bc.md](./02-flow-matching-bc.md) | 流匹配（Flow Matching）数学基础与 BC 策略训练 |
| 3 | [03-iql-critic-value.md](./03-iql-critic-value.md) | IQL 价值函数与 Q 网络：损失、期望分位回归、n-step |
| 4 | [04-qgf-core.md](./04-qgf-core.md) | **QGF 核心**：`QGFAgent` 全函数解读、训练与推理 |
| 5 | [05-inference-guidance-variants.md](./05-inference-guidance-variants.md) | 测试时引导变体：QFQL、Jacobian、Best-of-N |
| 6 | [06-test-time-baselines.md](./06-test-time-baselines.md) | GradStep、RobustQ、CFGRL 等测试时基线 |
| 7 | [07-train-time-methods.md](./07-train-time-methods.md) | FQL、EDP、QAM、IQL、SAC 等训练时方法 |
| 8 | [08-networks-and-modules.md](./08-networks-and-modules.md) | 网络结构、公共工具、扩散模块 |
| 9 | [09-data-training-evaluation.md](./09-data-training-evaluation.md) | 数据集、训练主循环、评估与实验脚本 |

## 整体架构（一图速览）

```
离线数据集 D = {(s, a, r, s')}
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Flow Matching BC                  IQL Critic + Value
  (ActorFlowField v_θ)              (Q_φ, V_ψ, target Q)
  模仿数据分布 π_BC                    仅 TD / 期望分位，不更新 actor
        │                              │
        └────────── 解耦训练 ──────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
     训练时方法                  测试时方法 (QGF)
  FQL / EDP / QAM / CFGRL      推理: v = v_BC + w·∇Q
  actor 与 critic 联合优化      无需 actor RL 梯度
```

## 核心思想（30 秒版）

1. **训练**：用行为克隆（BC）训练流匹配策略 $v_\theta(s, a_t, t)$；用 IQL 单独训练 $Q_\phi(s,a)$ 与 $V_\psi(s)$。二者**不解耦 actor-critic 联合优化**。
2. **推理**：在 Euler 去噪的每一步，用 Q 函数梯度引导速度场：
   $$
   v_{\text{guided}} = v_{\text{BC}} + w \cdot \nabla_{a_t} Q\bigl(s, \hat{a}(a_t)\bigr)
   $$
   其中 $\hat{a}$ 是「干净动作」的近似（QGF 用单步 Euler：$\hat{a} = \mathrm{clip}(a_t + (1-t)v_{\text{BC}}, -1, 1)$）。
3. **优势**：避免 BPTT 穿过去噪链；避免在 OOD 噪声动作上直接求 Q 梯度（QFQL 的问题）；方差低于完整 Jacobian 链式法则（可选 QGF-Jacobian）。

## 代码入口速查

| 用途 | 路径 |
|------|------|
| 主训练/评估 | `main.py` |
| QGF 实现 | `agents/qgf.py` |
| 公共 batch/Q 聚合 | `agents/common.py` |
| 流场网络 | `utils/networks.py` → `ActorFlowField` |
| 序列采样 / n-step | `utils/datasets.py` → `sample_sequence` |
| 测试时多权重评估 | `utils/evaluation.py` → `eval_with_test_time_guidance` |
| BC+IQL 共享底座训练 | `scripts/bc_iql_train.py` |
| QGF 测试时评估 | `scripts/exp_qgf_test_time_eval.py` |

## 推荐阅读路径

- **只想理解 QGF**：README → 01 → 02 → 03 → 04 → 05
- **对比论文基线**：01 → 06 → 07
- **复现实验 / 改代码**：04 → 09 → 08

## 符号约定

| 符号 | 含义 |
|------|------|
| $s$ | 观测（observation） |
| $a, a_t$ | 动作；$a_t$ 为去噪中间态 |
| $t \in [0,1]$ | 流时间；$t=0$ 为噪声，$t=1$ 为数据 |
| $v_\theta$ | 流速度场（BC 策略网络输出） |
| $Q_\phi, V_\psi$ | IQL critic / value |
| $H$ | `horizon_length`（动作块长度 / n-step 步数） |
| $w$ | `guidance_weight` 测试时引导强度 |
| $\gamma$ | `discount` 折扣因子 |
