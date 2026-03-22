---
title: "Title / Tags / Categories / Series"
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

本文档整合 `content/posts` 下各篇博文的 **title**、**tags**、**categories**、**series**，便于查阅与保持 frontmatter 一致。新文或改文时，**tags/categories/series 优先从下方 [§1 全站去重列表](#1-全站标签--分类--系列汇总) 中复用**；完整约定见仓库根目录 [AGENTS.md](../../AGENTS.md) 中的「博客编写规范」。

<!--more-->

---
GUIDE_VERSION: 2026-03-22T17:59:50+08:00

## 1 全站标签 / 分类 / 系列汇总

**tags（去重）：**
- AI
- Consistency Model
- cursor
- DataLoader
- Dataset
- Deep Learning
- diffusion/flow
- doing
- DoIt
- drifting-model
- flow matching
- how-to-doit
- OpenClaw
- papers
- prompt
- PyTorch
- Riemannian
- robots
- Sampler
- tools
- torch.utils.data
- tutorial
- 优化器
- 训练

**categories（去重）：**
- diffusion/flow
- how-to-doit
- OpenClaw
- papers
- PyTorch
- robots
- tools
- 深度学习

**series（去重）：**
- diffusion/flow系列
- DoIt
- robots系列
- tools系列
 - PyTorch 实践指南

---

## 2 diffusion-flow（扩散/流模型）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `diffusion-flow/An-Introduction-to-Flow-Matching-and-Diffusion-Models.md` | An Introduction to Flow Matching and Diffusion Models | diffusion/flow, tutorial | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Consistency-Model.md` | Consistency Model | diffusion/flow, Consistency Model | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/DDPM从条件贝叶斯到反向过程.md` | DDPM从条件贝叶斯到反向过程 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/DDPM前向过程.md` | DDPM前向过程 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/DDPM反向过程.md` | DDPM反向过程 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/ELBO.md` | ELBO | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Flow-Matching-Guide-and-Code.md` | Flow Matching Guide and Code | diffusion/flow, tutorial | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Flow-Matching-Guide-and-Code(项目解析).md` | Flow Matching Guide and Code(项目解析) | diffusion/flow, tutorial | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Flow-Matching-Guide-and-Code-5-geodesic-conditional-flow.md` | Flow Matching Guide and Code 第5章解读：指数映射-对数映射-测地线条件流 | Riemannian | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Flow-Matching-Guide-and-Code-5-Non-Euclidean-Flow-Matching.md` | Flow Matching Guide and Code 第5章解读：Non-Euclidean Flow Matching | diffusion/flow, tutorial, Riemannian, flow matching | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/Flow-Matching-Formula-derivation.md` | Flow-Matching-Formula-derivation | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/meanflow.md` | MeanFlow | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/SDE-ODE-离散&连续的转换.md` | SDE-ODE-离散-连续的转换 | diffusion/flow | diffusion/flow | diffusion/flow系列 |
| `diffusion-flow/The-Principles-of-Diffusion-Models.md` | The Principles of Diffusion Models | diffusion/flow, tutorial, doing | diffusion/flow | diffusion/flow系列 |

---

## 3 papers（论文解读）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `papers/Generative-Modeling-via-Drifting.md` | Generative Modeling via Drifting | diffusion/flow, drifting-model, papers | diffusion/flow, papers | diffusion/flow系列 |
| `papers/Minimal-Drifting-Models-算法与实现说明.md` | Minimal-Drifting-Models | diffusion/flow, drifting-model, papers | diffusion/flow, papers | diffusion/flow系列 |
| `papers/OmniXtreme.md` | OmniXtreme算法原理详细解析 | robots, papers | robots, papers | robots系列 |

---

## 4 robots（机器人）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `robots/arm_control.md` | 机械臂柔顺控制综合笔记：模式对照、阻抗/导纳、力感知、动力学估计与拖动示教 | robots, tutorial | robots | robots系列 |

---

## 5 tools（工具）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `tools/antigravity-awesome-skills.md` | antigravity-awesome-skills | cursor, tools | tools | tools系列 |
| `tools/cursor-plugins.md` | Cursor 插件市场介绍 | cursor, tools | tools | tools系列 |
| `tools/cursor-usage-tips.md` | Cursor 的使用技巧 | cursor, tools | tools | tools系列 |
| `tools/installed-plugins-guide.md` | Cursor 插件使用指南 | cursor, tools | tools | tools系列 |
| `tools/openclaw.md` | OpenClaw 安装文件与目录结构总览 | OpenClaw, tools | OpenClaw, tools | — |
| `tools/prompt_guide.md` | Prompt 汇总与记录 | prompt, tools, AI | tools | tools系列 |

---

## 6 how-to-doit（DoIt 主题使用）

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `how-to-doit/exampleSite-usage-guide.md` | exampleSite使用指南 | how-to-doit | how-to-doit | DoIt |
| `how-to-doit/how-to-DoIt.md` | 如何在 DoIt 中编写文章 | how-to-doit | how-to-doit | DoIt |

---

## 7 pytorch / others

| 路径 | title | tags | categories | series |
|------|--------|------|------------|--------|
| `pytorch/pytorch_dataset_guide.md` | PyTorch Dataset 体系技术文档 | PyTorch, Dataset, Deep Learning | PyTorch | PyTorch 实践指南 |
| `pytorch/dataloader_guide.md` | PyTorch DataLoader 技术解读 | PyTorch, DataLoader, Deep Learning | PyTorch | PyTorch 实践指南 |
| `pytorch/distributed_training_guide.md` | PyTorch 分布式训练与操作工具技术文档 | PyTorch, 分布式训练, Deep Learning | PyTorch | PyTorch 实践指南 |
| `pytorch/training_solver_guide.md` | PyTorch 模型训练技术文档：求解器、参数配置与训练循环 | PyTorch, Deep Learning, 优化器, 训练 | PyTorch | PyTorch 实践指南 |
| `pytorch/tensor_ops_guide.md` | PyTorch 张量工具函数技术文档：创建、计算、拼接与索引 | PyTorch, Deep Learning | PyTorch | PyTorch 实践指南 |
| `others/edu-ai.md` | 幼儿园AI教育平台调研 | （空） | （空） | （空） |

---

## 8 小结与说明

- 本索引**未包含** `draft.md`；表中 `others/edu-ai.md`、`papers/OmniXtreme.md` 为 draft，仍列出便于统一维护。
- 新增文章后请同步更新本文档（含 §1 去重列表与对应分类表），并优先复用已有 tags/categories/series。

**延伸阅读**

- 仓库根目录 [AGENTS.md](../../AGENTS.md)：博客编写规范、Frontmatter 约定、目录与分类选择。
- 主题文档：`themes/DoIt/exampleSite/content/posts/` 下 built-in-shortcodes、content、extended-shortcodes 等。

