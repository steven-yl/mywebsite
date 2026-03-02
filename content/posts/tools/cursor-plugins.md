---
title: "Cursor 插件市场介绍"
subtitle: "Marketplace 插件分类与使用指南"
date: 2026-03-02T10:00:00+08:00
draft: false
authors: [Steven]
description: "介绍 Cursor Marketplace 中的插件体系：插件组件（Rules、Skills、Agents、MCP、Hooks）、按分类的插件列表（精选、基础设施、数据与分析、效率、Agent 编排等）、安装与管理方式，便于按场景选型与使用。"

tags: [cursor, tools, marketplace, plugins]
categories: [tools]
series: [tools系列]
weight: 102
series_weight: 102

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

本文介绍 [Cursor Marketplace](https://cursor.com/marketplace) 中的插件：先说明**插件包含什么**以及**如何安装与管理**，再按官方分类列出各插件的用途，便于按场景选型。

<!--more-->

---

## 目录

1. [插件是什么](#1-插件是什么)：组件说明、安装与安全  
2. [精选插件（Featured）](#2-精选插件featured)  
3. [基础设施（Infrastructure）](#3-基础设施infrastructure)  
4. [数据与分析（Data & Analytics）](#4-数据与分析data--analytics)  
5. [效率与协作（Productivity）](#5-效率与协作productivity)  
6. [支付与区块链](#6-支付与区块链)  
7. [Agent 编排与工具](#7-agent-编排与工具)  
8. [其他插件](#8-其他插件)  
9. [使用与管理](#9-使用与管理)：安装、技能、MCP、发布  
10. [小结与延伸阅读](#10-小结与延伸阅读)

---

## 1 插件是什么

插件把 **规则、技能、Agent、命令、MCP 服务、自动化钩子** 打包成可安装的扩展，在 Cursor IDE / CLI / Cloud 中统一使用。

> 本文档介绍 [Cursor Marketplace](https://cursor.com/marketplace) 中的插件，便于按场景选型与使用。  
> 官方文档：[Plugins | Cursor Docs](https://cursor.com/docs/plugins)

### 1.1 组件说明

| 组件 | 说明 |
|------|------|
| **Rules** | 持久化 AI 指引与编码规范（`.mdc` 文件） |
| **Skills** | 面向复杂任务的专项能力（如某云/某产品的操作流程） |
| **Agents** | 自定义 Agent 配置与提示词 |
| **Commands** | 可由 Agent 执行的命令文件 |
| **MCP Servers** | Model Context Protocol 集成（外部工具/API 对接） |
| **Hooks** | 由事件触发的自动化脚本 |

### 1.2 安装与安全

- **安装方式**：在 Cursor 内打开 Marketplace 面板，搜索并安装；可针对**当前项目**或**用户全局**安装。
- **安全**：上架前经人工审核，须开源，更新也会再次审核。详见 [Marketplace Security](https://cursor.com/docs/plugins/security.md)。

---

## 2 精选插件（Featured）

这里列的是 Cursor 官方重点推荐的插件，对应**团队沟通、设计、前端部署、任务管理、支付、笔记**等常见场景；装好后可以在 Cursor 里用自然语言操作这些产品，不用反复切窗口。

| 插件 | 是什么 | 简介 | 通俗说 | 官方链接 |
|------|--------|------|--------|----------|
| **Slack** | 团队即时通讯工具（类似企业微信/钉钉） | Slack MCP 服务：搜索频道、发送消息、执行常用 Slack 操作。 | 在 Cursor 里说「发到 #前端 频道」就能发 Slack，不用切过去。 | [slack.com](https://slack.com) |
| **Figma** | 在线 UI 设计工具，做界面原型和设计稿 | Figma MCP + 技能，支持设计到代码的常见工作流。 | 根据设计稿写组件、对标注，和 Figma 文件联动，少对一遍像素。 | [figma.com](https://www.figma.com) |
| **Vercel** | 前端/Next.js 的托管与部署平台，一键上线 | Vercel 开发工具包：React/Next.js 最佳实践与一键部署。 | 按 Vercel 规范写 React/Next，写完一句话部署上线，不用自己配服务器。 | [vercel.com](https://vercel.com) |
| **Linear** | 面向研发的项目与 Bug 管理工具（类似 Jira 但更轻） | 在 Linear 中管理 issue、项目、文档，由 AI 助手执行操作。 | 用说话建任务、改状态、看进度，不用点进 Linear 网页。 | [linear.app](https://linear.app) |
| **Stripe** | 国际常用的在线支付接入服务（接信用卡等） | Stripe 集成：最佳实践、API/SDK 升级指引、Stripe MCP 服务。 | 接支付、查文档、升级 SDK 时让 AI 按 Stripe 规范来，少踩坑。 | [stripe.com](https://stripe.com) |
| **Notion** | 笔记/文档/待办工具，可当知识库用 | Notion 技能 + Notion MCP，在 Cursor 中管理任务与文档。 | 在 Cursor 里查 Notion 页面、记待办、同步文档，和写代码放一起。 | [notion.so](https://www.notion.so) |

---

## 3 基础设施（Infrastructure）

这类插件对应**数据库、登录认证、云部署、错误监控、浏览器测试、边缘计算**等「把应用跑起来」会用到的服务；在 Cursor 里用对话就能查库表、配登录、看报错、跑真机测试。

| 插件 | 是什么 | 简介 | 通俗说 | 官方链接 |
|------|--------|------|--------|----------|
| **PlanetScale** | 托管 MySQL 兼容数据库（分支、在线改表） | 已认证的托管 MCP，访问 PlanetScale 组织、数据库、分支、Schema、Insights。 | 在对话里查库表结构、分支、数据洞察，不用开 PlanetScale 控制台。 | [planetscale.com](https://planetscale.com) |
| **Clerk** | 用户登录/注册/权限 SaaS，接在网站或 App 里 | 身份认证工具包：配置指南、MCP、组织/Webhook/测试等专项技能。 | 做登录注册、权限、Webhook 时按 Clerk 官方套路来，少踩坑。 | [clerk.com](https://clerk.com) |
| **Browserbase Functions** | 云端无头浏览器，跑自动化脚本（如定时填表、点按钮） | 在 Browserbase 上部署无头浏览器自动化：定时或 Webhook 触发的云端任务。 | 把「自动打开网页、点按钮、填表」放到云端定时跑，不用自己开浏览器。 | [browserbase.com](https://browserbase.com) |
| **Neon Postgres** | 托管 PostgreSQL 数据库（按用付费、分支） | 管理 Neon 项目与数据库：neon-postgres 技能 + Neon MCP。 | 管 Neon 上的 Postgres 库、查库表，一句话搞定。 | [neon.tech](https://neon.tech) |
| **AWS** | 亚马逊云，提供服务器、存储、数据库等 | 部署到 AWS：架构建议、成本估算、IaC 部署。 | 想往 AWS 上部署时，让 AI 给架构建议、算费用、生成 IaC 代码。 | [aws.amazon.com](https://aws.amazon.com) |
| **Sentry** | 错误与性能监控，线上报错会推到这里 | 调试集成：MCP、命令与技能，便于排查错误与性能。 | 报错和性能问题直接连到 Sentry，在 Cursor 里看堆栈、复现步骤。 | [sentry.io](https://sentry.io) |
| **Browserstack** | 真机/真浏览器云测试（各种手机、浏览器型号） | 在真机/真浏览器上测试网站与 App，用自然语言运行/调试测试、管理用例。 | 说「在 iPhone 14 上跑一下」就能在真机上跑测试，看截图和日志。 | [browserstack.com](https://www.browserstack.com) |
| **Cloudflare** | CDN + 边缘计算（Workers）、DDoS 防护等 | Cloudflare 开发生态：Workers、Durable Objects、Agents SDK、MCP、Wrangler、性能工具。 | 写 Worker、配 CDN、用 Wrangler，按 Cloudflare 文档来，一句话查配置。 | [cloudflare.com](https://www.cloudflare.com) |
| **Convex** | 实时后端即服务（数据库 + 函数，TypeScript） | 官方 Convex 插件：TypeScript 响应式后端，含规则、技能、MCP 与自动化钩子。 | 用 TypeScript 写实时后端，建表、写函数、部署，AI 按 Convex 规范帮你。 | [convex.dev](https://convex.dev) |

---

## 4 数据与分析（Data & Analytics）

这类插件对应**机器学习平台、数据分析、数据仓库、BI、ORM**等：训模型、看埋点、跑 SQL、写 schema 时，在 Cursor 里说一句就能操作，不用记一堆控制台入口。

| 插件 | 是什么 | 简介 | 通俗说 | 官方链接 |
|------|--------|------|--------|----------|
| **Hugging Face** | AI 模型/数据集托管平台（开源模型、数据集、Spaces 演示） | AI/ML 技能：数据集创建、模型训练与评估、论文发布等 Hub 工作流。 | 传数据、训模型、跑评估、发论文到 Hub，在 Cursor 里说一句就办。 | [huggingface.co](https://huggingface.co) |
| **PostHog** | 产品分析 + 功能开关 + A/B 实验（开源可自建） | 在 Cursor 中访问 PostHog 分析、功能开关、实验与错误追踪。 | 看埋点、做 A/B 实验、查错误，不用切到 PostHog 后台。 | [posthog.com](https://posthog.com) |
| **ClickHouse** | 列式 OLAP 数据库，适合大数据量分析查询 | ClickHouse 技能（最佳实践）、规则与 MCP。 | 写 ClickHouse 查询、建表、优化时，按最佳实践来，少写错。 | [clickhouse.com](https://clickhouse.com) |
| **Databricks Skills** | 大数据与 ML 平台（Spark、数据湖、模型服务） | Databricks：CLI、Apps、Unity Catalog、Model Serving、Asset Bundles 等。 | 管集群、数据目录、模型服务、资源包，用自然语言操作 Databricks。 | [databricks.com](https://www.databricks.com) |
| **Snowflake** | 云数据仓库，存大量结构化数据并做分析 | Snowflake：1 个技能 + 1 个 MCP 服务。 | 查 Snowflake 数据、跑 SQL、管仓库，在对话里搞定。 | [snowflake.com](https://www.snowflake.com) |
| **Supabase** | 开源 Firebase 替代：数据库 + 认证 + 存储 + 实时 | 访问 Supabase 项目：管理表、拉取配置、查询数据。 | 管 Supabase 表、查配置、查数据，像在控制台一样但用说话完成。 | [supabase.com](https://supabase.com) |
| **Hex** | 协作式数据分析 Notebook（SQL + Python + 图表） | Hex MCP：连接 Cursor 与 Hex 工作区，做数据分析与 Notebook 协作。 | 和 Hex 里的分析、Notebook 打通，在 Cursor 里引用或同步结果。 | [hex.tech](https://hex.tech) |
| **Amplitude** | 产品行为分析（用户行为、留存、漏斗等） | Amplitude 可复用分析技能：图表分析、看板回顾、反馈归纳等。 | 看图表、复盘看板、总结用户反馈，用现成分析技能，不用手写 SQL。 | [amplitude.com](https://amplitude.com) |
| **Prisma** | Node/TS 的 ORM，用代码定义表结构并迁移 | 官方 Prisma 插件：MCP、规则、技能与自动化，用于数据库开发。 | 写 schema、迁移、查库，AI 按 Prisma 用法来，少写错、少查文档。 | [prisma.io](https://www.prisma.io) |

---

## 5 效率与协作（Productivity）

这类插件对应**公司内搜索、项目管理、需求与文档**：在 Cursor 里就能搜文档/Slack/邮件、建任务、写 Jira/Confluence，不用记各个系统入口。

| 插件 | 是什么 | 简介 | 通俗说 | 官方链接 |
|------|--------|------|--------|----------|
| **Glean** | 企业内统一搜索（文档、Slack、邮件、代码库、人） | 官方 Glean 插件：搜索文档/Slack/邮件、跨仓库查代码、找专家与干系人。 | 一句话搜公司文档、Slack、邮件和代码，还能找到「这事该问谁」。 | [glean.com](https://glean.com) |
| **Monday.com** | 可视化项目管理/看板（任务、时间线、自动化） | 将 AI 助手接入 monday.com，管理项目、任务与日常工作。 | 用说话建任务、改状态、看看板，和 monday.com 同步，不用来回点。 | [monday.com](https://monday.com) |
| **Atlassian** | Jira（需求/Bug）+ Confluence（文档）等一整套 | Jira、Confluence、需求梳理、待办、状态报告等 MCP 与技能。 | 建 Jira issue、写 Confluence、排需求、出状态报告，都在对话里完成。 | [atlassian.com](https://www.atlassian.com) |

---

## 6 支付与区块链

这类插件让 AI 能操作**链上钱包**（如 Solana 上的 Phantom）：换币、签名、查地址等，并参考官方文档少出错；适合做 Web3/区块链相关开发时在 Cursor 里联动。

| 插件 | 是什么 | 简介 | 通俗说 | 官方链接 |
|------|--------|------|--------|----------|
| **Phantom Connect** | Solana 等链上的钱包（浏览器插件/App），存币、签名、Swap | 为 Agent 提供钱包能力：在 Phantom 支持链上 Swap、签名、管理地址，并引用 Phantom 文档。 | 让 AI 能替你做链上操作：换币、签名、查地址，并参考 Phantom 文档少出错。 | [phantom.app](https://phantom.app) |

---

## 7 Agent 编排与工具（Agent Orchestration）

这类插件不绑定某个业务产品，而是**增强 Cursor 本身**：管好 MCP/技能/密钥、追踪 LLM 调用、上网搜与调研、用现成 TDD/调试套路、一键建插件、按版本拉文档等，让 AI 更稳、更可控、更「会查资料」。

| 插件 | 是什么 | 简介 | 通俗说 | 官方链接 |
|------|--------|------|--------|----------|
| **Runlayer** | 管理 Cursor 里 MCP/技能/Agent 的运行与权限 | 更安全地运行 MCP、技能与 Agent：发现运行内容、实时策略、秘密保护、审计日志。 | 看清哪些 MCP/技能在跑、谁在用，统一管权限和密钥，留审计记录。 | [runlayer.com](https://runlayer.com) |
| **Langfuse** | LLM 可观测平台：追踪调用、管提示词、评估效果 | 与 Langfuse 协作的技能：追踪、提示管理、评估等 LLM 工程能力。 | 追踪每次 LLM 调用、管提示词、看效果，做 LLM 应用时的观测台。 | [langfuse.com](https://langfuse.com) |
| **Parallel** | 命令行工具：网页搜索、抓取内容、深度调研、数据补全 | 由 parallel-cli 驱动的网页搜索、内容抽取、深度调研与数据增强。 | 让 AI 能上网搜、扒页面、做深度调研、补全数据，不只会聊代码。 | [parallel.cli](https://parallel.cli) |
| **Superpowers** | 一套「怎么用 AI 写代码」的套路（TDD、调试、协作） | 核心技能库：TDD、调试、协作模式与常用技巧。 | 用 TDD、调试清单、协作话术等现成套路，少想一步是一步。 | [Cursor Marketplace](https://cursor.com/marketplace) |
| **Compound Engineering** | 预置多类 Agent：代码审查、调研、设计、自动化流程 | 多 Agent 开发工具：29 个 Agent、22 个命令、19 个技能、1 个 MCP。 | 一堆现成 Agent：代码审查、调研、画图、自动化流程，按需调，不用从零配。 | [compound.dev](https://compound.dev) |
| **Create Plugin** | 帮你生成 Cursor 插件目录和 manifest 的脚手架 | 脚手架与校验：新建插件目录、生成 manifest、提交前质量检查。 | 想自己写 Cursor 插件时，一键生成目录和 manifest，并做提交前检查。 | [Cursor 建插件文档](https://cursor.com/docs/plugins/building) |
| **Context7** | 按技术栈版本拉取最新文档/示例进上下文 | Upstash Context7 MCP：按版本拉取最新文档与示例到 LLM 上下文中。 | 缺文档时按你当前版本拉最新文档和示例进上下文，避免 AI 瞎编。 | [upstash.com](https://upstash.com) |

---

## 8 其他插件（All Plugins）

这里是不按业务分类的**通用增强**：让 Cursor 从你的使用习惯里学习并更新项目说明，或复用 Cursor 团队自己的开发流程（CI、PR、冲突、冒烟测试等）。

| 插件 | 是什么 | 简介 | 通俗说 | 官方链接 |
|------|--------|------|--------|----------|
| **Continual Learning** | 从你的对话与修改中总结偏好和项目事实 | 从对话变更中学习持久化偏好与工作区事实，用简明要点维护 AGENTS.md。 | AI 会从你平时的改法和偏好里总结，自动更新 AGENTS.md，越用越贴你习惯。 | [Cursor Marketplace](https://cursor.com/marketplace) |
| **Cursor Team Kit** | Cursor 官方团队用的 CI/审查/发布流程 | Cursor 内部工作流：CI、代码审查、发布；覆盖 CI 监控与修复、PR、冲突、冒烟、编译、清理与总结。 | 像 Cursor 团队一样：看 CI、审 PR、解冲突、跑冒烟、清代码、写总结，适合想统一流程的团队。 | [Cursor Marketplace](https://cursor.com/marketplace) |

---

## 9 使用与管理

### 9.1 安装

在 Cursor 中打开 **Marketplace** 面板，搜索插件名，选择「安装」；可选**项目级**（仅当前仓库）或**用户级**（所有项目可用）。

### 9.2 技能与规则

在 **Cursor Settings → Rules** 中可对插件带来的技能/规则进行管理：

- 技能可设为「始终应用 / Agent 决定 / 手动」；在聊天中输入 **`/skill-name`** 可手动触发对应技能。
- 规则同样可配置生效方式（Always、Agent Decides、Manual 等），与 [Cursor 使用技巧](cursor-usage-tips.md) 中的 Rules 行为一致。

### 9.3 MCP

在 **Cursor Settings → Features → Model Context Protocol** 中可单独**开关**每个 MCP 服务；关闭后该服务不会加载，也不会在对话中暴露给 AI。部分插件需在 MCP 配置中填写 Token 或环境变量才能正常使用。

### 9.4 发布自有插件

自建插件需包含 **`.cursor-plugin/plugin.json`** 清单文件，并到 [cursor.com/marketplace/publish](https://cursor.com/marketplace/publish) 提交审核。详见 [Building Plugins](https://cursor.com/docs/plugins/building.md)。

---

## 10 小结与延伸阅读

**小结**

- **插件**：打包 Rules、Skills、Agents、Commands、MCP、Hooks，在 Marketplace 安装，可项目级或用户级。
- **分类**：精选、基础设施、数据与分析、效率、支付、Agent 编排等，按场景选装即可。
- **管理**：安装后通过 Settings 管理技能/规则与 MCP 开关；发布需提交审核并符合安全与开源要求。

**延伸阅读**

- [Cursor Marketplace](https://cursor.com/marketplace) — 浏览与安装插件  
- [Plugins 文档](https://cursor.com/docs/plugins) — 概念、安装、管理  
- [Building Plugins](https://cursor.com/docs/plugins/building.md) — 开发与提交指南  
- [Marketplace Security](https://cursor.com/docs/plugins/security.md) — 安全与审核说明  

*文档根据 Cursor 官方市场与文档整理，分类与描述以官网为准。*
