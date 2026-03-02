---
title: "Cursor 插件使用指南"
subtitle: "Superpowers、Continual Learning、Hugging Face Skills 的用法说明"
date: 2026-03-02T12:00:00+08:00
draft: false
authors: [Steven]
description: "针对Cursor 插件（Superpowers、Continual Learning、Hugging Face Skills）撰写使用指南：各插件的用途、何时触发、技能列表与典型用法，便于按场景正确使用。"

tags: [cursor, tools]
categories: [tools]
series: [tools系列]
weight: 101
series_weight: 101

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

本文档面向**当前工作区已安装**的 Cursor 插件，逐项说明其用途、触发方式与技能列表，方便按场景使用。更多插件介绍与安装方式见 [Cursor 插件市场介绍](cursor-plugins.md)。

<!--more-->

---

## 目录

1. [概述](#1-概述)：已安装插件列表与文档约定  
2. [Superpowers](#2-superpowers)：TDD、调试与协作技能库  
3. [Continual Learning](#3-continual-learning)：从对话中学习并维护 AGENTS.md  
4. [Hugging Face Skills](#4-hugging-face-skills)：Hub、训练、评估与论文发布  
5. [小结与延伸阅读](#5-小结与延伸阅读)

---

## 1 概述

当前环境已安装的 Cursor 插件如下，均以 **Skills（技能）** 为主，部分带 **Hooks** 或 **MCP**。

| 插件 | 用途简述 | 主要组件 |
|------|----------|----------|
| **Superpowers** | 写代码的流程套路：TDD、调试、协作 | Skills、Commands、Agents、Hooks |
| **Continual Learning** | 从对话中提炼偏好与项目事实，维护 AGENTS.md | Skill、Hooks |
| **Hugging Face Skills** | Hugging Face Hub 上的数据集、训练、评估、论文等 | Skills、MCP |

**文档约定**：以下各节按「是什么 → 何时用 / 怎么触发 → 技能列表与用法 → 注意点」组织；技能名以各插件 SKILL.md 中的 `name` 或描述为准。

---

## 2 Superpowers

### 2.1 是什么

**Superpowers** 是一套「怎么用 AI 写代码」的**流程技能库**：不绑定具体业务产品，而是规定 TDD、系统化调试、代码评审、写计划、分支收尾等**固定步骤**，让 AI 在动手前先选对流程、再执行。

- **官方**：[GitHub - obra/superpowers](https://github.com/obra/superpowers)
- **版本**：以插件 manifest 为准（如 4.x）。

### 2.2 何时用 / 怎么触发

- **自动**：插件通过 **using-superpowers** 技能规定：只要任务有约 1% 可能适用某条技能，AI 就应先调用该技能再回答或改代码，因此多数「加功能、修 bug、做评审、写计划」场景会自动走对应流程。
- **手动**：在 Chat 中可输入 **`/技能名`** 显式触发（具体以 Cursor Settings → Rules/Skills 中展示的为准）。

### 2.3 技能列表与典型用法

| 技能 | 用途 | 典型场景 |
|------|------|----------|
| **using-superpowers** | 对话开始时如何查找、何时必须用技能 | 每次对话的流程入口，通常由系统应用 |
| **brainstorming** | 做创意/功能/设计前先澄清意图与方案 | 「我们来做 XX 功能」→ 先头脑风暴再实现 |
| **writing-plans** | 有需求或规格时先写实施计划 | 有明确需求文档或 spec 时先写计划 |
| **executing-plans** | 按写好的计划执行，带检查点 | 计划写好后在单独会话中执行 |
| **test-driven-development** | 写功能/修 bug 前先写测试（TDD） | 实现或修复前先写/补测试 |
| **systematic-debugging** | 遇到 bug 先系统化定位再改代码 | 报错、异常行为时先按调试流程排查 |
| **requesting-code-review** | 完成功能后主动请求代码评审 | 功能做完、准备合并前请求 review |
| **receiving-code-review** | 收到评审意见时如何理解与落实 | 收到 review 反馈后按流程处理 |
| **verification-before-completion** | 声称「做完」前必须先跑验证、看结果 | 完成前跑测试/构建，用结果证明 |
| **finishing-a-development-branch** | 开发分支收尾：合并、PR、清理 | 分支开发结束，决定合并/PR/清理 |
| **using-git-worktrees** | 用 git worktree 做隔离开发 | 需要与当前工作区隔离的新功能/实验 |
| **dispatching-parallel-agents** | 多个独立任务时派给并行 Agent | 2+ 个无依赖任务可并行处理 |
| **subagent-driven-development** | 用子 Agent 驱动开发的模式 | 按插件约定的子 Agent 流程开发 |
| **writing-skills** | 如何编写、校验新技能 | 创建或修改 SKILL.md 时使用 |

**技能优先级**：若多条技能都可能适用，先**流程类**（如 brainstorming、systematic-debugging），再**实现类**（如具体写代码、调 API）。

### 2.4 注意点

- 部分技能为**刚性**（如 TDD、调试）：须严格按步骤执行；部分为**弹性**（如某些模式）：可依情境调整，以技能内说明为准。
- 用户只需说「要什么」；「怎么做」由对应技能决定，无需记忆具体步骤。

---

## 3 Continual Learning

### 3.1 是什么

**Continual Learning** 从**对话记录（transcript）的变更**中增量提取「可复用的用户偏好」和「持久的工作区事实」，并只以**简明要点**更新项目根目录的 **AGENTS.md**，实现「越用越贴你习惯」的记忆层。

- **官方**：[Cursor 插件仓库](https://github.com/cursor/plugins)（Continual Learning 插件）
- **不存储**：密钥、一次性任务说明、临时细节（分支名、commit、临时报错等）。

### 3.2 何时用 / 怎么触发

- **自动**：通过 **Hooks** 在设定的事件（如对话结束、会话变更）触发，对新增或更新的 transcript 做增量处理。
- **手动**：当用户说「从历史对话里总结一下偏好」「维护一下 AGENTS.md 记忆」「做一次 continual learning」等时，应使用 **continual-learning** 技能，按技能内流程读取 transcript、索引与 AGENTS.md，再写回 AGENTS.md。

### 3.3 技能：continual-learning

- **作用**：从 transcript 增量中抽取「重复出现的用户纠正/偏好」和「稳定、可操作的工作区事实」，只更新 AGENTS.md 的 `## Learned User Preferences` 与 `## Learned Workspace Facts`，且仅保留**纯要点**（无证据标签、无过程说明）。
- **输入**：  
  - transcript 根目录：`~/.cursor/projects/<workspace-slug>/agent-transcripts/`  
  - 现有记忆文件：`AGENTS.md`  
  - 增量索引：`.cursor/hooks/state/continual-learning-index.json`
- **纳入条件**：可操作、跨会话稳定、在多轮对话中重复或用户明确说成通用规则、非敏感。
- **输出约定**：AGENTS.md 只保留上述两个小节，条目为简明 bullet，不写推理过程或元数据。

### 3.4 注意点

- 首次使用或索引丢失时会处理更多 transcript；之后仅处理**新增或 mtime 更新的文件**，避免全量重扫。
- 若需「从某次聊天里学一条偏好」，可在对话中明确说「把这条记到 AGENTS.md」或触发一次 continual learning。

---

## 4 Hugging Face Skills

### 4.1 是什么

**Hugging Face Skills** 提供与 Hugging Face Hub 和 ML 工作流相关的**技能**（及配套 MCP）：数据集、训练、评估、论文发布、Gradio 演示、Jobs 与 Trackio 等，在 Cursor 里用自然语言或脚本完成 Hub 操作与云端任务。

- **官方**：[GitHub - huggingface/skills](https://github.com/huggingface/skills)
- **组件**：多条 Skills + 可选 MCP 服务（见 Cursor Settings → MCP）。

### 4.2 何时用 / 怎么触发

- 当你的问题或任务涉及 **Hugging Face Hub**（模型、数据集、Spaces、论文）、**云端训练/推理/任务**、**评估与模型卡**、**Gradio 应用** 时，AI 会按描述匹配对应技能并调用。
- 部分能力依赖 **MCP**：需在 Cursor 中启用对应 Hugging Face MCP 并配置认证（如 Token），才能完成上传、运行 Job 等操作。

### 4.3 技能列表与典型用法

| 技能 | 用途 | 典型场景 |
|------|------|----------|
| **hugging-face-cli** | 用 `hf` CLI 做 Hub 操作 | 下载/上传模型/数据集/Space、建仓库、管缓存、在 HF 上跑任务 |
| **hugging-face-datasets** | 在 Hub 上创建与管理数据集 | 初始化数据集仓库、配置与 system prompt、流式写入、SQL 查询与转换 |
| **hugging-face-model-trainer** | 在 HF Jobs 上训练/微调语言模型 | TRL（SFT/DPO/GRPO/reward 等）、GGUF 转换、选硬件与成本、Trackio、认证与模型持久化 |
| **hugging-face-jobs** | 在 HF 上跑任意计算任务 | UV 脚本、Docker Job、GPU/CPU 选型、成本估算、密钥与超时、结果持久化 |
| **hugging-face-evaluation** | 管理模型卡中的评估结果 | 从 README 抽评估表、从 Artificial Analysis 等导入分数、用 vLLM/lighteval 跑评估、model-index 格式 |
| **hugging-face-paper-publisher** | 在 Hub 上发布与管理论文 | 建论文页、关联模型/数据集、认领作者、生成 Markdown 文章 |
| **hugging-face-trackio** | 训练实验追踪与可视化 | 训练中打点（Python API）、用 CLI 查询/分析指标、实时看板、与 HF Space 同步、JSON 输出 |
| **huggingface-gradio** | 用 Gradio 做 Web 演示 | 创建/编辑 Gradio 应用、组件、事件、布局、聊天界面 |
| **hugging-face-tool-builder** | 用 HF API 做可复用脚本/工具 | 需要多次或组合调用 HF API、做数据拉取/增强/处理脚本时 |

（若安装了 HF MCP，可能还有 **hf-mcp** 等与 MCP 配套的技能，以 Cursor 中实际列出的为准。）

### 4.4 注意点

- **认证**：上传、创建仓库、跑 Jobs 等需在 Hub 或 MCP 配置中设置 Token，不要将 Token 写进文档或代码。
- **成本**：Jobs、训练等涉及云端算力，技能内会涉及硬件与成本估算，执行前确认配额与计费。

---

## 5 小结与延伸阅读

**小结**

- **Superpowers**：通过「技能优先」规则与多条流程技能（TDD、调试、评审、计划、分支收尾等），让 AI 先选流程再执行，适合日常开发与协作。
- **Continual Learning**：从 transcript 增量更新 AGENTS.md，只保留用户偏好与工作区事实的要点，适合维护长期记忆与偏好。
- **Hugging Face Skills**：覆盖 Hub 上的数据集、训练、评估、论文、Gradio、Jobs、Trackio 等，配合 MCP 可在 Cursor 内完成端到端 ML 工作流。

**延伸阅读**

- [Cursor 插件市场介绍](cursor-plugins.md) — 插件体系、分类与安装  
- [Cursor 使用技巧](cursor-usage-tips.md) — Rules、Skills、MCP 等通用用法  
- [Plugins \| Cursor Docs](https://cursor.com/docs/plugins) — 官方插件文档  
- [Superpowers - GitHub](https://github.com/obra/superpowers)  
- [Hugging Face Skills - GitHub](https://github.com/huggingface/skills)

*文档根据当前已安装插件的 manifest 与 SKILL.md 整理，技能名与触发方式以实际 Cursor 环境为准。*
