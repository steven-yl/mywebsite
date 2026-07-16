---
title: "LingBot-VLA 2.0 技术文档索引"
subtitle: ""
date: 2026-07-13T14:00:00+08:00
draft: false
authors: [Steven]
description: "LingBot-VLA 2.0（Vision-Language-Action）是一套面向真实机器人部署的视觉-语言-动作基础模型。本文档体系从总体架构到各子模块算法原理、实现细节与使用方式，提供完整技术解读。"
summary: "LingBot-VLA 2.0（Vision-Language-Action）是一套面向真实机器人部署的视觉-语言-动作基础模型。本文档体系从总体架构到各子模…"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA2, robots]
series: [lingbotVLA2-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# LingBot-VLA 2.0 技术文档索引

> **LingBot-VLA 2.0**（Vision-Language-Action）是一套面向真实机器人部署的视觉-语言-动作基础模型。本文档体系从总体架构到各子模块算法原理、实现细节与使用方式，提供完整技术解读。
>
> 论文：[From Foundation to Application: Improving VLA Models in Practice](https://arxiv.org/pdf/2607.06403)（arXiv:2607.06403）

---

## 文档导航

| 章节 | 文件 | 内容概要 |
|------|------|----------|
| **0. 总览** | [00-overview.md](./00-overview.md) | 整体架构、知识结构、模块关联、优缺点与适用场景对比 |
| **1. 模型架构** | [01-model-architecture.md](./01-model-architecture.md) | Qwen3-VL 骨干、MoE Action Expert、统一动作空间、注意力机制 |
| **2. Flow Matching** | [02-flow-matching.md](./02-flow-matching.md) | 连续流匹配算法原理、训练/推理公式、与 Diffusion 对比 |
| **3. Dual-Query 蒸馏** | [03-dual-query-distillation.md](./03-dual-query-distillation.md) | LingBot-Depth、DINO-Video 教师模型、几何/时序先验注入 |
| **4. 数据流水线** | [04-data-pipeline.md](./04-data-pipeline.md) | LeRobot 格式、Robot Config、归一化、多数据集拼接 |
| **5. 训练系统** | [05-training-system.md](./05-training-system.md) | 训练入口、分布式 FSDP2、MoE 负载均衡、优化器、检查点 |
| **6. 推理与部署** | [06-inference-deployment.md](./06-inference-deployment.md) | 实时策略服务、WebSocket、开环评估、RoboTwin 仿真 |
| **7. 配置参考** | [07-configuration.md](./07-configuration.md) | YAML 配置详解、真机 vs 仿真差异、参数速查 |
| **8. 参考文献** | [08-references.md](./08-references.md) | 论文、博客、开源项目链接汇总 |

### 已有补充文档

| 文件 | 说明 |
|------|------|
| [config/lingbotvla_config_doc.md](./config/lingbotvla_config_doc.md) | 配置参数完整字段表（dataclass 级别） |
| [../configs/vla/Training_Config.md](../configs/vla/Training_Config.md) | 训练配置实操指南 |
| [../lingbotvla/data/vla_data/README.md](../lingbotvla/data/vla_data/README.md) | 自定义数据集构建指南 |

---

## 快速定位

### 我想了解...

| 目标 | 阅读章节 |
|------|----------|
| 项目整体是什么、各模块如何协作 | [00-overview.md](./00-overview.md) |
| 55 维动作空间如何定义 | [01-model-architecture.md#统一动作表示](./01-model-architecture.md) |
| 为什么用 Flow Matching 而非 Diffusion | [02-flow-matching.md](./02-flow-matching.md) |
| 深度/视频蒸馏如何工作 | [03-dual-query-distillation.md](./03-dual-query-distillation.md) |
| 如何准备自己的 LeRobot 数据 | [04-data-pipeline.md](./04-data-pipeline.md) |
| 如何启动 post-training | [05-training-system.md](./05-training-system.md) |
| 如何部署到真机 | [06-inference-deployment.md](./06-inference-deployment.md) |
| 真机与仿真配置有何不同 | [07-configuration.md#真机-vs-仿真](./07-configuration.md) |

### 核心代码入口

| 功能 | 路径 |
|------|------|
| 训练入口 | `train.sh` → `tasks/vla/train_lingbotvla.py` |
| VLA 2.0 模型 | `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` |
| 数据加载 | `lingbotvla/data/vla_data/` |
| 真机部署 | `deploy/lingbot_vla_v2_policy.py` |
| 开环评估 | `scripts/open_loop_eval.py` |
| RoboTwin 评估 | `experiment/robotwin/start_robotwin_infer_and_eval.sh` |

---

## 版本说明

| 版本 | 骨干 | 特点 |
|------|------|------|
| LingBot-VLA 1.0 | Qwen2.5-VL | Flow Matching + Action Expert，无原生深度/视频蒸馏 |
| **LingBot-VLA 2.0** | Qwen3-VL-4B | MoE Action Expert + Dual-Query 蒸馏 + 55 维统一动作空间 |

本仓库默认面向 **LingBot-VLA 2.0**（`config_key: LingbotVLAV2Config`）。
