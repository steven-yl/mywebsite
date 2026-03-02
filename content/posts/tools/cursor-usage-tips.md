---
title: "Cursor 的使用技巧"
subtitle: "提升 AI 编程效率的实用指南"
date: 2026-03-02T10:00:00+08:00
draft: false
authors: [Steven]
description: "介绍 Cursor 的核心功能与使用技巧：Chat/Composer、Rules（规则）、Skills（技能）、Hooks（钩子）、MCP 的配置与用法，以及高级技巧，帮助在日常编码中更高效地使用 AI 辅助开发。"

tags: [cursor, tools]
categories: [tools]
series: [tools系列]
weight: 100
series_weight: 100

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

本文整理 Cursor 的常用功能与使用技巧，方便在写代码、读项目、改 bug 时更顺手地借助 AI。先掌握**和 AI 怎么交互**，再按需配置**规则、技能、钩子、MCP**，即可持续提效。

<!--more-->

---

## 目录

1. [核心交互方式](#1-核心交互方式)：Chat、Composer、行内编辑（Cmd+K）与 @ 引用  
2. [按场景怎么选](#2-按场景怎么选)：读代码、加功能、重构、查定义等  
3. [Commands（/ 命令）](#3-commands-命令)：聊天框输入 / 触发的自定义工作流  
4. [Rules（规则）](#4-rules规则)：项目规则、AGENTS.md、User/Team Rules  
5. [Skills（技能）](#5-skills技能)：可复用工作流与 SKILL.md  
6. [Hooks（钩子）](#6-hooks钩子)：在关键节点执行脚本  
7. [MCP](#7-mcp)：连接外部工具与数据源  
8. [进阶技巧](#8-进阶技巧)：上下文、提示词、大仓库、终端与 Tab  
9. [小结](#9-小结)

---

## 1 核心交互方式

和 Cursor 里的 AI 打交道主要有三种方式：**Chat**（对话）、**Composer**（多文件编辑）、**行内编辑**（Cmd/Ctrl+K）。配合 **@ 引用** 控制上下文，效果更好。

### 1.1 Chat（对话）

- **快捷键**：`Cmd/Ctrl + L` 打开 Chat。
- **用法**：提问、让 AI 解释代码、给建议。在编辑器中**先选中一段代码**再打开 Chat，这段代码会自动作为上下文。
- **@ 引用**：
  - `@文件名`：把该文件内容纳入上下文；
  - `@文件夹/`：引用整个目录；
  - `@Codebase`：让 AI 在代码库中搜索相关位置。

**建议**：问题尽量具体（如「这段函数在哪些地方被调用？」「如何在不破坏现有 API 的前提下加一个可选参数？」），并配合 @ 引用。

### 1.2 Composer（多文件编辑）

- **快捷键**：`Cmd/Ctrl + I` 打开 Composer。
- **用法**：描述跨文件、多步骤的需求，AI 会规划步骤并直接改多个文件，适合重构、加功能、修一串相关 bug。可勾选 **Background** 在后台跑，不阻塞当前编辑。

**建议**：把任务拆成「先做什么、再做什么」写清楚，或明确「只改 XX 目录下的文件」，能减少无关修改。

### 1.3 行内编辑（Cmd/Ctrl + K）

- **用法**：在代码中**选中一段**，按 `Cmd/Ctrl + K`，输入「改成 XXX」「加类型注解」「换成 async」等指令；AI 只改选中区域，不动其它文件。
- **适用**：局部微调、重命名、格式统一；比先开 Chat 再复制结果更快，适合高频小改。

---

## 2 按场景怎么选

| 场景 | 建议用法 |
|------|----------|
| 读不懂某段代码 | 选中代码 + Chat，问「这段在做什么？和 XX 的关系是？」 |
| 加一个小功能 | Chat 或 Composer，说清「在哪个文件/模块、要什么行为」 |
| 大范围重构 | Composer，分步描述（先改接口、再改实现、最后改调用方） |
| 统一风格/规范 | 在 `.cursor/rules` 或 `AGENTS.md` 里写好，再让 AI 按规则改 |
| 查调用关系/定义 | Chat + `@Codebase`，问「哪里调用了 XXX？」或「XXX 的定义在哪？」 |

---

## 3 Commands（/ 命令）

在 **Chat 输入框里输入 `/`**，会列出当前可用的**自定义命令**。每个命令对应一份 **Markdown 文件**，内容会作为「固定提示 + 步骤」一并发给 AI，用来做代码审查、写 PR、跑测试、安全审计、新人上手等**可复用工作流**。

### 3.1 作用概览

| 作用 | 说明 |
|------|------|
| **一键触发流程** | 输入 `/review`、`/pr` 等，不用每次手打一长段提示词 |
| **团队统一规范** | 团队命令由管理员在 Dashboard 配置，所有人输入 `/` 即可用同一套流程 |
| **带参数使用** | `/` 后面的文字会一起发给 AI，例如：`/commit and /pr these changes to address DX-523` |

### 3.2 存放位置与优先级

| 类型 | 位置 | 谁可用 |
|------|------|--------|
| **Team 命令** | Cursor Dashboard（Team Content → Commands） | 该团队所有成员，由管理员创建/修改 |
| **全局命令** | `~/.cursor/commands/` | 当前用户所有项目 |
| **项目命令** | 项目根目录 `.cursor/commands/` | 仅当前项目，可随 git 共享 |

在输入框输入 `/` 时，上述三个来源的命令会一起出现，按名称选用即可。

### 3.3 如何创建

1. 在项目根建目录 **`.cursor/commands/`**（或用户目录下 `~/.cursor/commands/`）。
2. 新建 **`.md` 文件**，文件名即命令名（如 `review-code.md` → 输入 `/review-code`）。
3. 文件内容为**纯 Markdown**：写清楚「这个命令要 AI 做什么、按什么步骤做」，例如检查清单、步骤列表、PR 模板等。

示例目录结构：

```text
.cursor/commands/
  code-review-checklist.md
  create-pr.md
  run-all-tests-and-fix.md
  security-audit.md
  onboard-new-developer.md
```

### 3.4 和 Rules / Skills 的区别

- **Rules**：系统级「约束与规范」，在每次请求时按条件注入，影响 AI 的**风格与边界**。
- **Skills**：由 AI 根据对话**自动判断**是否启用的「任务流程」，存在固定目录，有 name/description。
- **Commands**：**用户主动**在聊天框输入 `/xxx` 触发，内容作为**本次请求的提示**，适合「我明确要跑这套流程」的场景（如每次 PR 前执行同一套 review 步骤）。

---

## 4 Rules（规则）

规则在每次请求时作为**系统级上下文**注入，让 AI 在写代码、解释、改文件时有一致的行为约束。规则来源按优先级合并：**Team Rules → Project Rules → User Rules**。

### 4.1 规则类型与生效方式

| 类型 | 说明 | 配置方式 |
|------|------|----------|
| Always Apply | 每次对话都注入 | `alwaysApply: true` |
| Apply Intelligently | 由 AI 根据 `description` 判断是否相关 | `alwaysApply: false` 且不设 globs |
| Apply to Specific Files | 仅当打开/引用的文件匹配路径时生效 | 在 frontmatter 中设置 `globs` |
| Apply Manually | 仅在对话中 @ 提及时生效 | 如 `@my-rule`、`@react-patterns.mdc` |

### 4.2 项目规则：`.cursor/rules`

- **位置**：项目根目录 `.cursor/rules/`，建议纳入版本控制，团队共享。
- **格式**：支持 `.md`（纯 Markdown）和 `.mdc`（带 YAML frontmatter，可精确控制生效条件）。

目录示例：

```text
.cursor/rules/
  react-patterns.mdc
  api-guidelines.md
  frontend/
    components.md
```

.mdc 示例（frontmatter + 正文）：

```markdown
---
description: "前端组件与 API 校验规范"
globs: ["**/*.tsx", "src/components/**"]
alwaysApply: false
---

- 组件命名遵循 PascalCase
- 动画用 Framer Motion，样式用 Tailwind
- API 目录下用 zod 定义 schema 并导出类型
```

Frontmatter 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `description` | string | 规则用途简述，供「Apply Intelligently」时被 AI 选用 |
| `globs` | string 或数组 | 文件匹配（minimatch），如 `**/*.ts`、`src/api/**`；多条用 YAML 列表 |
| `alwaysApply` | boolean | 为 `true` 时每条对话都会带上该规则 |

**Globs 注意**：扩展名精确匹配（`*.js` 不匹配 `.jsx`）；`src/*` 与 `src/**` 含义不同。

### 4.3 在规则中引用文件

在规则正文里写 **`@文件名`**（如 `@service-template.ts`），该文件会被一并纳入上下文；适合放模板、示例路径，避免在规则里大段贴代码。

### 4.4 AGENTS.md（轻量替代）

- 在项目根或任意子目录放 **`AGENTS.md`**，纯 Markdown，无 frontmatter。
- 子目录的 `AGENTS.md` 会与祖先目录的合并，**越具体路径的说明优先级越高**。
- 适合「简单、可读、少配置」的项目说明。

```text
project/
  AGENTS.md
  frontend/AGENTS.md
  backend/AGENTS.md
```

### 4.5 User Rules 与 Team Rules

- **User Rules**：在 **Cursor Settings → Rules** 中配置，全局生效，仅用于 **Agent（Chat）**，不作用于 Cmd+K 行内编辑。
- **Team Rules**：Team/Enterprise 在 Dashboard 中管理，可设为「强制」；格式为纯文本，无 globs/alwaysApply。

### 4.6 使用建议

- 单条规则控制在 **500 行以内**，详细内容用「引用文件」或链接代替。
- 规则写清楚、可执行，并带**具体示例**或 `@` 引用；聚焦**本项目特有**的约定和流程，不必塞入整本风格指南。

---

## 5 Skills（技能）

Skills 是「任务说明 + 步骤」的 Markdown 文档，放在固定目录。AI 根据 **description** 判断当前对话是否适用，若适用则按技能中的步骤执行，适合 PR 审查、提交信息生成、数据库查询等**可复用工作流**。

### 5.1 存放位置与作用域

| 类型 | 路径 | 作用域 |
|------|------|--------|
| 个人技能 | `~/.cursor/skills/<skill-name>/` | 所有项目可用 |
| 项目技能 | `.cursor/skills/<skill-name>/` | 仅当前仓库，可随 git 共享 |

不要使用 `~/.cursor/skills-cursor/`，该目录为 Cursor 内置技能保留。

### 5.2 目录结构

每个技能是一个**目录**，内含必选的 `SKILL.md`，以及可选的参考、示例、脚本：

```text
skill-name/
  SKILL.md
  reference.md
  examples.md
  scripts/
    validate.py
    helper.sh
```

### 5.3 SKILL.md 格式

**Frontmatter（必填）**：

```markdown
---
name: your-skill-name
description: 用第三人称写清「做什么」以及「何时使用」。例如：对 PR 按团队标准做代码审查。在用户请求 code review、查看 PR 或代码变更时使用。
---
```

- **name**：小写字母、数字、连字符，最多 64 字符。
- **description**：最多 1024 字符；需同时包含 **WHAT**（能力）和 **WHEN**（触发场景），便于 AI 自动选用。

正文用 Markdown 写步骤、检查项、模板、示例；建议主文件 **不超过 500 行**，细节放到 `reference.md` 等，在 SKILL.md 里用链接引用。

### 5.4 编写原则

- **简洁**：只写 AI 不具备或易错的上下文。
- **渐进披露**：核心流程在 SKILL.md，细节进 reference/examples。
- **输出格式**：需要固定格式时给**模板**或**示例**。
- **脚本**：对易出错、需一致的操作，提供现成脚本路径与用法，让 Agent 执行而非现场生成大段代码。

### 5.5 在 Cursor 中的使用

在 **Cursor Settings → Rules** 中启用 **Agent Skills**。技能以「由 Agent 决定是否采用」的方式注入，不能设为「始终应用」或「仅手动 @」；对话场景匹配 description 时，Agent 会优先按该技能的步骤执行。

---

## 6 Hooks（钩子）

Hooks 在 Agent 执行流程的**关键节点**运行外部脚本，通过 **stdin/stdout 传 JSON** 与 Cursor 通信，可用于：会话开始时注入上下文、执行前校验/拦截危险操作、编辑后跑格式化或审计等。

### 6.1 配置文件位置

| 作用域 | 路径 |
|--------|------|
| 项目 | `.cursor/hooks.json` |
| 用户全局 | `~/.cursor/hooks.json` |

多处配置会合并；**优先级高的先执行**；任一 hook 可通过 **退出码 2** 阻止当前操作。

### 6.2 Hook 类型

| Hook 名称 | 时机 |
|-----------|------|
| `sessionStart` / `sessionEnd` | 会话开始 / 结束 |
| `beforeSubmitPrompt` | 用户提交 prompt 之前 |
| `beforeReadFile` / `afterFileEdit` | 读文件前 / 文件被编辑后 |
| `beforeShellExecution` / `afterShellExecution` | 执行 Shell 命令前 / 后 |
| `beforeMCPExecution` / `afterMCPExecution` | 执行 MCP 工具前 / 后 |
| `preToolUse` / `postToolUse` / `postToolUseFailure` | 任意工具调用前 / 成功后 / 失败后 |
| `subagentStart` / `subagentStop` | 子 Agent 启动 / 结束 |
| `stop` | Agent 即将结束 |
| `preCompact` | 上下文即将被压缩前 |

Tab 补全相关：`beforeTabFileRead`、`afterTabFileEdit`。

### 6.3 配置示例

```json
{
  "version": 1,
  "hooks": {
    "preToolUse": [
      {
        "command": "./scripts/validate-shell.sh",
        "matcher": "Shell"
      }
    ],
    "postToolUse": [
      { "command": "./scripts/audit.sh" }
    ],
    "beforeSubmitPrompt": [
      { "command": "./scripts/check-prompt.js" }
    ]
  }
}
```

- **command**：要执行的命令。
- **matcher**：可选，正则匹配「工具名」，仅匹配到的工具才触发该 hook。

### 6.4 脚本行为约定

- **输入**：Cursor 通过 stdin 传入 JSON。
- **输出**：脚本向 stdout 输出 JSON。
- **退出码**：**0** 成功；**2** **阻止当前操作**；其他视为失败，默认不阻止（fail-open）。

典型用途：禁止含 `rm -rf` 的 Shell、写库前检查 PII、编辑后自动运行 formatter。

### 6.5 与 Claude Code 的兼容

在 **Cursor Settings → Features** 中开启 **Third-party skills** 后，可加载 Claude Code 的 hook 配置（如 `.claude/settings.json`）；Claude 的 hook 名会映射到 Cursor。若需完整能力（如 `subagentStart`、团队级下发），建议用 Cursor 原生 `.cursor/hooks.json`。

---

## 7 MCP

MCP（Model Context Protocol）让 Cursor 连接**外部工具与数据源**，以 **Tools / Prompts / Resources** 等形式暴露给 AI，从而在对话中直接查文档、操作浏览器、访问数据库等。

### 7.1 基本概念

- **MCP Server**：独立进程或远程服务，通过标准协议与 Cursor 通信。
- **能力**：Tools（可调用函数）、Prompts（模板化提示）、Resources（可读结构化数据）、Roots（URI/文件系统边界）等。
- Cursor 根据用户意图与工具描述，决定是否调用某个 MCP 工具。

### 7.2 传输方式

| 传输类型 | 运行方式 | 典型用途 | 认证 |
|----------|----------|----------|------|
| stdio | 本地命令 | 本地脚本、CLI 工具 | 环境变量/参数 |
| SSE | 本地或远程 HTTP | 远程 API、OAuth 服务 | 常为 OAuth |
| Streamable HTTP | 本地或远程 HTTP | 多用户场景 | 常为 OAuth |

### 7.3 配置（mcp.json）

在 **Cursor Settings → MCP** 中可添加/编辑，或直接维护 **mcp.json**。支持 **config 插值**：`${env:VAR}`、`${workspaceFolder}` 等。

**stdio（Node）**：

```json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "some-mcp-server"],
      "env": { "API_KEY": "${env:API_KEY}" }
    }
  }
}
```

**stdio（Python）**：

```json
{
  "mcpServers": {
    "py-server": {
      "command": "python",
      "args": ["mcp-server.py"],
      "env": { "API_KEY": "value" }
    }
  }
}
```

**远程 SSE/HTTP**：

```json
{
  "mcpServers": {
    "remote": {
      "url": "https://api.example.com/mcp",
      "headers": { "Authorization": "Bearer ${env:TOKEN}" }
    }
  }
}
```

### 7.4 使用注意

- **先看工具描述**：确认 name/description/parameters 与副作用，避免误操作。
- **浏览器类 MCP（如 cursor-ide-browser）**：先 `browser_navigate`，再 `browser_lock`，再点击/输入，最后 `browser_unlock`；交互前用 `browser_tabs`、`browser_snapshot` 获取页面结构与元素引用；等待加载用短间隔重试 + snapshot。
- **安全**：只添加可信服务器，敏感信息用环境变量。

### 7.5 常用 MCP 示例

- **cursor-ide-browser**：IDE 内控制浏览器，前端调试、E2E、截图与性能分析。
- **GitHub / GitLab**：仓库、Issue、PR。
- **数据库类**（Prisma、MongoDB、DuckDB）：查 schema、跑查询、迁移。
- **文档类**（AWS Docs、MS Learn、Astro docs）：查最新文档与示例。

---

## 8 进阶技巧

### 8.1 上下文与 @ 引用

- **@ 的优先级**：先 @ 关键文件（如入口、类型定义），再 @ 相关目录，最后用「只改 XX」约束范围。
- **避免上下文爆炸**：大文件用「@文件名 + 行号或函数名」或选中片段再问，不要整文件塞进对话。
- **多轮对话**：复杂任务拆成多轮，每轮明确「基于上一步的结果，接下来做 YYY」。

### 8.2 提示词

- **角色 + 约束**：如「你是一个熟悉 React 的开发者，只改 `components/` 下的文件，保持现有 TypeScript 严格模式。」
- **输出格式**：需要清单、步骤、对比时，直接说「请用列表」「先给步骤再给代码」「用表格对比 A 和 B」。
- **反例**：说「不要做 XXX」「不要改 YYY 文件」「保持现有 API 不变」，比只说「要做什么」更稳。

### 8.3 大仓库与长对话

- **大仓库**：多用「@文件夹」限定范围，或「先搜再问」（如「在 `src/utils` 里找解析 URL 的函数」）。
- **长对话**：上下文会截断；重要结论可显式总结，或写进 `.cursor/rules` / 注释，下一轮 @ 引用。
- **Background Composer**：大任务用后台跑，自己继续写代码，完成后再看 diff 和说明。

### 8.4 Agent 与子任务

在 Composer 中描述「先探索代码库再改」时，AI 可能启动 **explore** 等子 Agent。适合不熟悉的大仓库、需要「先找全调用点再统一改」的重构、跨模块 bug 排查。提示里可写清「先列出所有涉及 XX 的文件，再给出修改方案」。

### 8.5 规则细化

`.cursor/rules/` 下可放多条规则，按文件名、路径、语言匹配（如 `*.ts`、`src/api/`）；不同目录可约定不同规范。在规则里写「修改前先 grep/搜索 XXX」「禁止直接改 YYY 文件」等，可进一步收窄 AI 行为。

### 8.6 终端与 Tab 补全

- **终端**：Chat 里可让 AI 生成并运行命令，结果回到对话；适合写脚本、跑测试、排查构建错误。涉及敏感或破坏性操作时，先看清生成命令再执行。
- **Tab 补全**：在设置里可调节补全的激进程度；补全不理想时，选中已补全内容再 **Cmd/Ctrl + K** 用自然语言微调。

---

## 9 小结

- **交互**：Chat（对话）、Composer（多文件）、Cmd+K（行内编辑），配合 @ 引用。
- **Commands**：在 Chat 输入 `/` 触发的自定义命令，Markdown 文件定义流程，可团队/全局/项目级。
- **Rules**：.cursor/rules、.mdc、AGENTS.md 与 User/Team Rules，控制生效方式与作用范围。
- **Skills**：~/.cursor/skills 或 .cursor/skills，由 description 触发，适合审查、提交信息、专项工作流。
- **Hooks**：hooks.json，在关键节点执行脚本，可校验、拦截或后处理。
- **MCP**：stdio/SSE/HTTP 连接外部能力，使用前看清工具说明与调用顺序。
- **进阶**：精准 @ 上下文、角色与反例提示词、大仓库与 Background Composer、规则细化、终端与 Tab。

按需组合以上方式，把 Cursor 当成「随时可问、可改代码」的 AI 助手，提高日常开发效率。
