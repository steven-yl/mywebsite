---
title: "OpenClaw 安装文件与目录结构总览"
date: 2024-06-09T00:00:00+08:00
draft: false
categories: ["OpenClaw", "tools"]
tags: ["OpenClaw", "tools"]
description: "本文详细梳理 OpenClaw 的安装目录、配置与核心文件，便于检查安装状态与进行后续个性化配置。"
---

## OpenClaw 安装文件汇总

### 一、核心安装文件（npm全局安装）


| 文件名/目录         | 位置                                                                | 解释        |
| -------------- | ----------------------------------------------------------------- | --------- |
| `openclaw`     | `~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/bin/`    | 主CLI可执行文件 |
| `package.json` | `~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/`        | npm包信息与依赖 |
| `docs/`        | `~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/docs/`   | 官方文档目录    |
| `skills/`      | `~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/skills/` | 内置技能库目录   |
| `src/`         | `~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/src/`    | 源代码目录     |


### 二、用户配置目录（首次运行后创建）


| 文件名/目录           | 位置                                  | 解释              |
| ---------------- | ----------------------------------- | --------------- |
| `config/`        | `~/.openclaw/config/`               | 配置文件目录          |
| `models.json`    | `~/.openclaw/config/models.json`    | 模型配置（API密钥、端点等） |
| `providers.json` | `~/.openclaw/config/providers.json` | 服务提供商配置         |
| `channels.json`  | `~/.openclaw/config/channels.json`  | 消息频道配置          |
| `workspace/`     | `~/.openclaw/workspace/`            | 用户工作空间根目录       |
| `logs/`          | `~/.openclaw/logs/`                 | 系统日志目录          |
| `data/`          | `~/.openclaw/data/`                 | 应用数据存储目录        |


### 三、工作空间文件（助手个性化）


| 文件名             | 位置                                           | 解释                |
| --------------- | -------------------------------------------- | ----------------- |
| `SOUL.md`       | `~/.openclaw/workspace/SOUL.md`              | 助手"灵魂"文件（个性、行为准则） |
| `USER.md`       | `~/.openclaw/workspace/USER.md`              | 用户信息文件            |
| `IDENTITY.md`   | `~/.openclaw/workspace/IDENTITY.md`          | 助手身份定义（名称、表情等）    |
| `TOOLS.md`      | `~/.openclaw/workspace/TOOLS.md`             | 本地工具配置笔记          |
| `MEMORY.md`     | `~/.openclaw/workspace/MEMORY.md`            | 助手长期记忆（仅主会话加载）    |
| `AGENTS.md`     | `~/.openclaw/workspace/AGENTS.md`            | 工作空间使用指南          |
| `HEARTBEAT.md`  | `~/.openclaw/workspace/HEARTBEAT.md`         | 心跳任务清单            |
| `BOOTSTRAP.md`  | `~/.openclaw/workspace/BOOTSTRAP.md`         | 首次启动引导文件（完成后删除）   |
| `memory/`       | `~/.openclaw/workspace/memory/`              | 每日记忆文件目录          |
| `YYYY-MM-DD.md` | `~/.openclaw/workspace/memory/YYYY-MM-DD.md` | 每日记忆文件（按日期）       |


### 四、环境与运行时文件


| 文件名           | 位置                                                 | 解释                                  |
| ------------- | -------------------------------------------------- | ----------------------------------- |
| `.openclawrc` | `~/.openclawrc` 或 `~/.config/openclaw/config.json` | 环境配置文件                              |
| 环境变量          | 系统环境                                               | OPENCLAW_WORKSPACE, OPENCLAW_MODEL等 |
| PID文件         | 系统临时目录                                             | 网关守护进程的进程ID文件                       |
| 会话缓存          | 系统临时目录                                             | 运行时会话状态缓存                           |


### 五、技能文件（示例）


| 技能名          | 位置                                      | 解释                |
| ------------ | --------------------------------------- | ----------------- |
| `1password/` | `~/.nvm/.../openclaw/skills/1password/` | 1Password CLI集成技能 |
| `weather/`   | `~/.nvm/.../openclaw/skills/weather/`   | 天气查询技能            |
| `obsidian/`  | `~/.nvm/.../openclaw/skills/obsidian/`  | Obsidian笔记技能      |
| `SKILL.md`   | 各技能目录下的`SKILL.md`                       | 技能使用说明文档          |


### 六、当前你的安装状态


| 项目         | 状态                 | 说明                |
| ---------- | ------------------ | ----------------- |
| **安装版本**   | 2026.3.2 (85377a2) | 最新稳定版             |
| **Node版本** | v22.22.1           | 通过nvm管理           |
| **工作空间**   | 已初始化               | 包含所有模板文件          |
| **模型配置**   | 已配置                | DeepSeek Chat API |
| **技能加载**   | 正常                 | 内置技能可用            |
| **API问题**  | Amazon Bedrock缺失   | 不影响核心功能           |


### 七、重要路径总结

1. **CLI命令**: `~/.nvm/versions/node/v22.22.1/bin/openclaw`
2. **配置目录**: `~/.openclaw/config/`
3. **工作空间**: `~/.openclaw/workspace/`
4. **技能目录**: `~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/skills/`
5. **日志目录**: `~/.openclaw/logs/`

## OpenClaw 技能（Skills）技术说明列表

### 技能概览

OpenClaw 技能是模块化的功能扩展，每个技能提供特定领域的工具和自动化能力。

### 技能安装方法

技能来源分为三类：**随 OpenClaw 内置**、**通过 ClawHub 安装**、**额外包或手动安装**。下表说明对应关系与安装方式。


| 来源                   | 说明                                                       | 安装方式                                     |
| -------------------- | -------------------------------------------------------- | ---------------------------------------- |
| **openclaw-bundled** | 随 `npm install -g openclaw` 一起安装，位于 `openclaw/skills/` 下 | 升级 OpenClaw 即更新：`npm update -g openclaw` |
| **openclaw-extra**   | 飞书等扩展技能包，需单独安装                                           | 使用 npm 安装对应包，或从 ClawHub 安装               |
| **ClawHub 市场**       | 社区/官方在 [clawhub.com](https://clawhub.com) 发布的技能          | 使用 `clawhub` CLI 搜索、安装、更新                |


#### 1. 使用 ClawHub 安装与更新技能

ClawHub 是 OpenClaw 的技能市场，用法与 npm 类似。需先确保已安装 `clawhub` CLI（通常随 OpenClaw 或单独安装）。

```bash
## 搜索技能（按关键词）
clawhub search weather
clawhub search feishu

## 安装指定技能到 OpenClaw 技能目录
clawhub install <skill-name>

## 更新已安装技能到最新版本
clawhub update <skill-name>

## 更新所有已通过 ClawHub 安装的技能
clawhub update --all

## 列出已安装技能（或使用 openclaw 自带的列表）
openclaw skills list
```

安装后技能会出现在 OpenClaw 的加载路径中（一般为全局 `openclaw/skills/` 或用户覆盖目录），下次启动会话即可使用。

#### 2. 安装 openclaw-extra（如飞书套件）

飞书文档、云盘、权限、知识库、任务、加急等技能属于 **openclaw-extra**，需单独安装对应 npm 包（包名以 `openclaw-extra-` 或项目约定为准）。示例：

```bash
## 若存在飞书扩展包（包名以官方仓库为准）
npm install -g openclaw-extra-feishu

## 安装后，技能目录通常位于该包下的 skills/ 或由 OpenClaw 配置指向
```

具体包名与安装路径以 [OpenClaw 官方文档](https://github.com/openclaw/openclaw) 或 ClawHub 说明为准。安装完成后用 `openclaw skills list` 确认是否出现 `feishu-doc`、`feishu-drive` 等。

#### 3. 手动安装或链接技能目录

若你本地有技能目录（例如自己用 `skill-creator` 创建的，或从别处克隆的仓库），可将其放到 OpenClaw 能扫描到的位置：

```bash
## 方式 A：复制到 OpenClaw 内置技能目录（需写权限）
OPENCLAW_SKILLS="$HOME/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/skills"
cp -r /path/to/my-skill "$OPENCLAW_SKILLS/"

## 方式 B：若 OpenClaw 支持自定义技能路径（通过环境变量或配置）
export OPENCLAW_SKILLS_EXTRA="/path/to/my-skills"
## 将 my-skill 放在 /path/to/my-skills/my-skill/ 下，内含 SKILL.md
```

每个技能目录内需包含 `SKILL.md`，OpenClaw 通过该文件识别技能与使用说明。

#### 4. 安装后验证

```bash
## 列出当前可用的技能（含内置与已安装）
openclaw skills list

## 查看某技能的说明文档
cat "$(npm root -g)/openclaw/skills/weather/SKILL.md"
## 或按你的 Node 路径：~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/skills/<skill>/SKILL.md
```

许多技能依赖系统已安装的 CLI（如 `op`、`ffmpeg`、`gh`）。若列表显示「已加载」但使用时报错，请检查对应 CLI 是否在 `PATH` 中：

```bash
which op ffmpeg gh gog   # 按需替换为技能所需命令
```

#### 5. 内置技能没有安装全时怎么办

「内置技能」指来源为 **openclaw-bundled** 的技能。若列表里很多显示为 **missing**，通常有两种情况，按下面顺序处理即可。

**情况一：技能目录本身不存在（npm 包里没有或未同步）**

先保证 OpenClaw 和技能目录完整：

```bash
## 1. 升级到最新版，确保内置技能目录完整
npm update -g openclaw

## 2. 用 ClawHub 把缺失的 bundled 技能补装到本地（技能名与汇总表一致）
clawhub install apple-notes
clawhub install apple-reminders
clawhub install github
## … 按需对每个 missing 的技能执行 clawhub install <技能名>

## 3. 或一次性搜索并安装（按 ClawHub 返回的包名安装）
clawhub search openclaw
clawhub install <包名>
```

**情况二：技能目录已有，但显示为 missing（多因依赖 CLI 未装）**

很多 bundled 技能依赖系统里的对应 CLI，未安装时会被判为不可用。安装对应依赖后即可变为 ready：

```bash
## 示例：常见技能与所需 CLI（macOS 可用 brew 安装）
brew install 1password-cli          # 1password
brew install ffmpeg                 # video-frames
brew install gh                     # github, gh-issues
# memo、remindctl、things、grizzly 等需在 App Store 或各自官网安装
# 具体每个技能依赖见该技能目录下的 SKILL.md 或文档
```

**推荐流程小结**

1. 运行 `openclaw skills list`（或当前环境下的等价命令）看哪些是 missing。
2. 若缺失的是「技能目录」：用 `npm update -g openclaw` + `clawhub install <技能名>` 补全。
3. 若缺失的是「依赖 CLI」：按该技能的 SKILL.md 或上表安装对应命令行工具，再查一次列表确认变为 ready。

#### 6. 常见问题


| 现象                  | 可能原因                                  | 处理方式                                                                                 |
| ------------------- | ------------------------------------- | ------------------------------------------------------------------------------------ |
| 技能汇总表里显示「missing」   | 未安装该技能或对应 CLI                         | 用 `clawhub install <name>` 安装技能；用 brew/system 安装缺失 CLI                               |
| `clawhub` 命令不存在     | 未安装 ClawHub CLI                       | `npm install -g clawhub` 或按官方文档安装                                                    |
| 飞书类技能不出现            | 未安装 openclaw-extra 飞书包                | 安装对应 npm 包并在 OpenClaw 配置中启用/指向其 skills 目录                                            |
| 安装后仍不加载             | 技能目录无 `SKILL.md` 或路径未在 OpenClaw 扫描范围内 | 检查目录结构并确认 `OPENCLAW_SKILLS` / 配置中的技能路径                                               |
| 升级 OpenClaw 后部分技能消失 | 依赖内置目录，被覆盖或重置                         | 用 `clawhub update --all` 或重新 `clawhub install`；自定义技能放在 `OPENCLAW_SKILLS_EXTRA` 等独立目录 |


---

### 技能汇总表（57 项）


| 技能                   | 说明                                                            | 来源               | 使用方法说明                                                                                                                                          |
| -------------------- | ------------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| feishu-doc 飞书文档      | 飞书文档读写与评论管理                                                   | openclaw-extra   | 用户提到飞书文档、云文档、docx 链接或文档评论时使用                                                                                                                    |
| feishu-drive 飞书云盘    | 飞书云盘文件管理                                                      | openclaw-extra   | 用户提到云空间、文件夹、云盘时使用                                                                                                                               |
| feishu-perm 飞书权限     | 飞书文档与文件权限管理                                                   | openclaw-extra   | 用户提到分享、权限、协作者时使用                                                                                                                                |
| feishu-wiki 飞书知识库    | 飞书知识库导航                                                       | openclaw-extra   | 用户提到知识库、wiki 或 wiki 链接时使用                                                                                                                       |
| feishu-task 飞书任务     | 飞书任务、任务列表、子任务、评论与附件管理                                         | openclaw-extra   | 用户提到任务、任务列表、子任务、任务评论/附件/链接时使用                                                                                                                   |
| feishu-urgent 飞书加急   | 飞书加急消息（buzz）通知                                                | openclaw-extra   | 用户提到消息加急、buzz、提醒或升级时使用                                                                                                                          |
| 1password            | 1Password CLI (op) 的安装与使用                                     | openclaw-bundled | 安装 CLI、桌面集成、登录（单/多账号）、通过 op 读取/注入/运行密钥时使用                                                                                                       |
| apple-notes 苹果备忘录    | 通过 memo CLI 管理 Apple 备忘录（创建、查看、编辑、删除、搜索、移动、导出）                | openclaw-bundled | 用户要求添加笔记、列出/搜索笔记或管理笔记文件夹时使用                                                                                                                     |
| apple-reminders 苹果提醒 | 通过 remindctl 管理 Apple 提醒（列表、添加、编辑、完成、删除），支持列表、日期筛选、JSON/纯文本输出 | openclaw-bundled | 用户要求管理提醒、待办或提醒列表时使用                                                                                                                             |
| bear-notes Bear 笔记   | 通过 grizzly CLI 创建、搜索、管理 Bear 笔记                               | openclaw-bundled | 用户提到 Bear 或 Bear 笔记时使用                                                                                                                          |
| blogwatcher 博客监控     | 使用 blogwatcher CLI 监控博客与 RSS/Atom 源更新                         | openclaw-bundled | 用户要求监控博客、RSS 或订阅更新时使用                                                                                                                           |
| blucli               | BluOS CLI (blu)：发现、播放、分组、音量                                   | openclaw-bundled | 用户提到 BluOS、Blu 音箱或多房间音频时使用                                                                                                                      |
| bluebubbles          | 通过 BlueBubbles 发送或管理 iMessage                                 | openclaw-bundled | 需要发 iMessage 时使用；通过通用消息工具，channel=bluebubbles                                                                                                   |
| camsnap              | 从 RTSP/ONVIF 摄像头抓取帧或片段                                        | openclaw-bundled | 用户提到摄像头、监控、RTSP、ONVIF 或抓帧时使用                                                                                                                    |
| clawhub              | 使用 ClawHub CLI 在 clawhub.com 搜索、安装、更新、发布 Agent 技能             | openclaw-bundled | 需要按需拉取新技能、同步到最新/指定版本或发布技能包时使用                                                                                                                   |
| coding-agent 编程代理    | 通过后台进程将编程任务委托给 Codex、Claude Code 或 Pi                         | openclaw-bundled | 开发新功能/应用、审查 PR、重构大代码库、需文件探索的迭代编码时使用；不用于简单单行修复、仅读代码、~/clawd 工作区；需支持 pty:true 的 bash                                                              |
| discord              | 通过消息工具 (channel=discord) 进行 Discord 操作                        | openclaw-bundled | 用户提到 Discord、频道或机器人时使用                                                                                                                          |
| eightctl             | 控制 Eight Sleep 床垫：状态、温度、闹钟、日程                                 | openclaw-bundled | 用户提到 Eight Sleep、床垫温度或睡眠设备时使用                                                                                                                   |
| gemini               | Gemini CLI：一次性问答、摘要与生成                                        | openclaw-bundled | 用户要求用 Gemini 做问答、总结或生成时使用                                                                                                                       |
| gh-issues            | 拉取 GitHub issues、派生子代理实现修复并开 PR、监控并处理 PR 评论                   | openclaw-bundled | 使用 /gh-issues [owner/repo] 及 --label、--limit、--milestone、--assignee、--fork、--watch、--reviews-only、--cron、--dry-run、--model、--notify-channel 等参数 |
| gifgrep              | 用 CLI/TUI 搜索 GIF 源、下载结果、提取静帧/表                                | openclaw-bundled | 用户要求搜索 GIF、下载动图或提取帧时使用                                                                                                                          |
| github               | 通过 gh CLI：issues、PR、CI、代码审查、API 查询                            | openclaw-bundled | 查 PR/CI、创建/评论 issue、筛选 PR/issue、查看 run 日志时使用；不用于复杂 Web 操作、跨多仓库批量、gh 未登录时                                                                        |
| gog Google 工作区       | Gmail、日历、Drive、通讯录、Sheets、Docs 的 CLI                          | openclaw-bundled | 用户提到 Gmail、日历、Drive、文档、表格时使用                                                                                                                    |
| goplaces             | 通过 goplaces CLI 调用 Google Places API (New)：文本搜索、地点详情、解析、评论    | openclaw-bundled | 需要人类可读的地点查询或脚本用 JSON 时使用                                                                                                                        |
| healthcheck 健康检查     | 主机安全加固与风险偏好配置                                                 | openclaw-bundled | 用户询问安全审计、防火墙/SSH/更新加固、风险态势、暴露审查、OpenClaw 定时任务或版本状态时使用                                                                                           |
| himalaya             | 通过 IMAP/SMTP 管理邮件的 CLI：列出、读、写、回复、转发、搜索、整理，支持多账号与 MML          | openclaw-bundled | 用户要求用终端管理邮箱、查信、发信时使用                                                                                                                            |
| imsg                 | iMessage/SMS CLI：列出会话、历史，通过 Messages.app 发消息                  | openclaw-bundled | 用户要求发 iMessage、查聊天记录时使用（非 BlueBubbles 场景）                                                                                                       |
| mcporter             | 使用 mcporter CLI 列出、配置、鉴权并直接调用 MCP 服务器/工具（HTTP 或 stdio）        | openclaw-bundled | 需要调用 MCP、编辑 MCP 配置或生成 CLI/类型时使用                                                                                                                 |
| model-usage 模型用量     | 用 CodexBar CLI 本地成本数据汇总 Codex/Claude 按模型用量                    | openclaw-bundled | 用户问 codexbar 模型级用量/成本或需要可脚本化的按模型汇总时使用                                                                                                           |
| nano-banana-pro      | 通过 Gemini 3 Pro Image 生成或编辑图片                                 | openclaw-bundled | 用户要求用 Nano Banana Pro 做图或改图时使用                                                                                                                  |
| nano-pdf             | 使用 nano-pdf CLI 用自然语言指令编辑 PDF                                 | openclaw-bundled | 用户要求用自然语言修改 PDF 时使用                                                                                                                             |
| notion               | Notion API：创建和管理页面、数据库、块                                      | openclaw-bundled | 用户提到 Notion、页面、数据库或块时使用                                                                                                                         |
| obsidian             | 通过 obsidian-cli 操作 Obsidian 仓库（纯 Markdown）并做自动化               | openclaw-bundled | 用户提到 Obsidian、 vault、笔记库时使用                                                                                                                     |
| openai-image-gen     | 通过 OpenAI Images API 批量生成图片；随机提示采样 + index.html 图库            | openclaw-bundled | 用户要求批量出图或建图库时使用                                                                                                                                 |
| openai-whisper       | 使用 Whisper CLI 本地语音转文字（无需 API 密钥）                             | openclaw-bundled | 用户要求本地转写音频时使用                                                                                                                                   |
| openai-whisper-api   | 通过 OpenAI 音频转录 API (Whisper) 转写音频                             | openclaw-bundled | 用户要求用云端 Whisper 转写时使用                                                                                                                           |
| openhue              | 通过 OpenHue CLI 控制飞利浦 Hue 灯与场景                                 | openclaw-bundled | 用户提到 Hue、智能灯、场景时使用                                                                                                                              |
| oracle               | 使用 oracle CLI 的最佳实践（提示与文件打包、引擎、会话、文件附件）                       | openclaw-bundled | 复杂提示、多文件打包、会话或附件模式时使用                                                                                                                           |
| ordercli             | 仅 Foodora 的 CLI：历史订单与当前订单状态（Deliveroo 开发中）                    | openclaw-bundled | 用户问 Foodora 订单、外卖状态时使用                                                                                                                          |
| peekaboo             | 使用 Peekaboo CLI 截取并自动化 macOS 界面                               | openclaw-bundled | 用户要求截屏、UI 自动化、macOS 界面操作时使用                                                                                                                     |
| sag                  | ElevenLabs 文本转语音，mac 风格 say 体验                                | openclaw-bundled | 用户要求 TTS、朗读或 ElevenLabs 时使用                                                                                                                     |
| session-logs 会话日志    | 使用 jq 搜索与分析自己的会话日志（较早/父级对话）                                   | openclaw-bundled | 用户要求查历史会话、分析对话时使用                                                                                                                               |
| sherpa-onnx-tts      | 通过 sherpa-onnx 本地文本转语音（离线、无云）                                 | openclaw-bundled | 用户要求离线 TTS、不联网朗读时使用                                                                                                                             |
| skill-creator 技能创建器  | 创建或更新 AgentSkills，含脚本、引用与资源的设计与打包                             | openclaw-bundled | 用户要设计、结构化或打包新技能时使用                                                                                                                              |
| slack                | 通过 slack 工具从 OpenClaw 控制 Slack                                | openclaw-bundled | 用户提到 Slack、反应、置顶频道/DM 时使用                                                                                                                       |
| songsee              | 使用 songsee CLI 从音频生成频谱图与特征面板可视化                               | openclaw-bundled | 用户要求音频可视化、频谱图时使用                                                                                                                                |
| sonoscli             | 控制 Sonos 音箱：发现/状态/播放/音量/分组                                    | openclaw-bundled | 用户提到 Sonos、多房间播放时使用                                                                                                                             |
| spotify-player       | 通过 spogo（优先）或 spotify_player 在终端控制 Spotify 播放与搜索              | openclaw-bundled | 用户要求用终端控制 Spotify 时使用                                                                                                                           |
| summarize            | 对 URL、播客、本地文件做摘要或提取文本/转录                                      | openclaw-bundled | 用户要求「转写这个 YouTube/视频」或摘要链接/播客时使用                                                                                                                |
| things-mac           | 通过 things CLI 管理 Things 3（URL 添加/更新项目与待办；本地读/搜/列）             | openclaw-bundled | 用户要求往 Things 加任务、列收件箱/今天/即将到来、搜索任务或查看项目/区域/标签时使用                                                                                                |
| tmux                 | 通过发送按键并抓取 pane 输出远程控制 tmux 会话                                 | openclaw-bundled | 用户要求控制 tmux、在 pane 里跑交互式 CLI 时使用                                                                                                                |
| trello               | 通过 Trello REST API 管理看板、列表、卡片                                 | openclaw-bundled | 用户提到 Trello、看板、卡片时使用                                                                                                                            |
| video-frames         | 使用 ffmpeg 从视频中提取帧或短视频片段                                       | openclaw-bundled | 用户要求从视频提帧、剪片段时使用                                                                                                                                |
| voice-call 语音通话      | 通过 OpenClaw voice-call 插件发起语音通话                               | openclaw-bundled | 用户要求发起语音通话时使用                                                                                                                                   |
| wacli                | 通过 wacli CLI 发 WhatsApp 或搜索/同步 WhatsApp 历史（非普通用户聊天）           | openclaw-bundled | 用户要求发 WhatsApp、查历史或同步时使用                                                                                                                        |
| weather 天气           | 通过 wttr.in 或 Open-Meteo 获取当前天气与预报                             | openclaw-bundled | 用户问某地天气、温度、预报时使用；不用于历史天气、预警或详细气象分析；无需 API 密钥                                                                                                    |
| xurl X/Twitter       | 对 X (Twitter) API 做认证请求的 CLI：发推、回复、引用、搜索、读推、关注、DM、上传媒体等       | openclaw-bundled | 用户提到发推、搜推、X API、Twitter 时使用                                                                                                                     |


---

### 详细技能说明

#### 1. **1password**

**作用**: 安全地管理和注入密码、密钥等敏感信息
**使用方法**:

```bash
## 安装 op CLI
brew install --cask 1password-cli

## 登录 1Password
op account add

## 读取密码
op item get "GitHub" --fields password
```

**配置**: 需要 1Password 账户，桌面应用集成或服务账户令牌

#### 2. **clawhub**

**作用**: OpenClaw 技能市场，类似 npm 但针对技能
**使用方法**:

```bash
## 搜索技能
clawhub search weather

## 安装技能
clawhub install weather

## 更新技能
clawhub update weather
```

**配置**: 需要 clawhub.com 访问，可能需登录

#### 3. **gifgrep**

**作用**: 从多个 GIF 提供商搜索、下载和处理 GIF
**使用方法**:

```bash
## 搜索 GIF
gifgrep search "cat dancing"

## 下载结果
gifgrep download --id 12345

## 提取帧
gifgrep frames --input cat.gif --output frames/
```

**配置**: 可能需要 API 密钥（Tenor、Giphy 等）

#### 4. **gog**

**作用**: Google Workspace 全套工具 CLI 集成
**使用方法**:

```bash
## 查看 Gmail
gog mail list --unread

## 日历事件
gog calendar events --today

## Drive 文件
gog drive list --folder root
```

**配置**: 需要 Google Cloud 项目，OAuth 2.0 凭据

#### 5. **healthcheck**

**作用**: 系统安全硬化和风险配置检查
**使用方法**:

```bash
## 运行安全检查
healthcheck audit

## 检查防火墙
healthcheck firewall

## 查看风险容忍度
healthcheck risk
```

**配置**: 需要系统权限，可配置 cron 定期检查

#### 6. **mcporter**

**作用**: 模型上下文协议（MCP）服务器管理
**使用方法**:

```bash
## 列出 MCP 服务器
mcporter list

## 调用服务器工具
mcporter call --server filesystem --tool read_file --args '{"path":"/tmp/test.txt"}'

## 生成类型定义
mcporter types --server filesystem --output ./types.ts
```

**配置**: 需要 MCP 服务器配置（HTTP 或 stdio）

#### 7. **obsidian**

**作用**: Obsidian 笔记库的 CLI 自动化
**使用方法**:

```bash
## 列出笔记
obsidian list --vault personal

## 创建笔记
obsidian create --title "会议记录" --vault work

## 搜索内容
obsidian search "TODO" --vault personal
```

**配置**: 需要 Obsidian 安装，指定库路径

#### 8. **openhue**

**作用**: Philips Hue 智能灯光系统控制
**使用方法**:

```bash
## 发现 Hue Bridge
openhue discover

## 列出灯光
openhue lights list

## 控制灯光
openhue lights set --id 1 --on --brightness 80 --color "#FF5500"

## 创建场景
openhue scenes create --name "阅读模式" --lights 1,2,3 --brightness 60
```

**配置**: 需要 Hue Bridge IP 和 API 用户名

#### 9. **oracle**

**作用**: 高级提示工程和文件处理工具
**使用方法**:

```bash
## 复杂提示处理
oracle prompt --file requirements.txt --engine gpt-4

## 文件打包
oracle bundle --input ./src --output bundle.json

## 会话管理
oracle session --name project-analysis --attach ./docs/
```

**配置**: 无特殊依赖，支持多种 LLM 引擎

#### 10. **ordercli**

**作用**: 外卖平台订单管理（目前支持 Foodora）
**使用方法**:

```bash
## 查看历史订单
ordercli history --limit 10

## 当前订单状态
ordercli status

## 订单详情
ordercli detail --order-id 123456
```

**配置**: 需要外卖平台账户登录（Foodora）

#### 11. **skill-creator**

**作用**: 创建、开发和发布 OpenClaw 技能
**使用方法**:

```bash
## 创建新技能
skill-creator init --name my-skill --description "我的自定义技能"

## 添加工具
skill-creator add-tool --name fetch-data --description "获取数据"

## 打包发布
skill-creator pack --output ./dist/
```

**配置**: 需要 Node.js，了解技能结构

#### 12. **video-frames**

**作用**: 从视频文件中提取帧或短片段
**使用方法**:

```bash
## 提取关键帧
video-frames extract --input video.mp4 --output frames/ --method keyframes

## 创建缩略图
video-frames thumbnail --input video.mp4 --time 00:01:30

## 提取片段
video-frames clip --input video.mp4 --start 00:00:10 --duration 5
```

**配置**: 需要 ffmpeg 安装

#### 13. **wacli**

**作用**: WhatsApp 消息自动化（非普通聊天）
**使用方法**:

```bash
## 发送消息
wacli send --to "+1234567890" --message "Hello"

## 搜索历史
wacli search --query "重要会议" --limit 20

## 同步联系人
wacli contacts sync
```

**配置**: 需要 WhatsApp Web 连接，QR 码登录

#### 14. **weather**

**作用**: 天气查询和预报
**使用方法**:

```bash
## 当前天气
weather current --location "Shanghai"

## 天气预报
weather forecast --location "Beijing" --days 3

## 详细报告
weather detail --location "New York" --units metric
```

**配置**: 使用 wttr.in 或 Open-Meteo，无需 API 密钥

---

### 技能使用通用模式

#### 1. **技能激活**

大多数技能在相关上下文中自动激活，或通过读取技能目录中的 `SKILL.md` 文件。

#### 2. **配置检查**

```bash
## 检查技能是否可用
openclaw skills list

## 查看特定技能文档
cat ~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/skills/weather/SKILL.md
```

#### 3. **权限要求**

- 🔓 **无特殊权限**: weather, oracle, gifgrep
- 🔐 **需要配置**: 1password, gog, openhue, wacli
- ⚠️ **需要系统权限**: healthcheck, video-frames (ffmpeg)
- 🌐 **需要网络**: 所有涉及外部服务的技能

#### 4. **错误处理**

```bash
## 常见错误
## 1. 缺少依赖: 安装所需 CLI 工具
## 2. 配置缺失: 检查 ~/.openclaw/config/
## 3. 权限不足: 检查文件权限或使用 sudo
## 4. 网络问题: 检查连接和 API 密钥
```

#### 5. **技能开发**

```bash
## 基于现有技能学习
cp -r ~/.nvm/versions/node/v22.22.1/lib/node_modules/openclaw/skills/weather/ ./my-skill/

## 修改并测试
cd my-skill
## 编辑 SKILL.md 和工具文件
```

---

### 推荐技能组合

#### 基础日常使用

1. **weather** - 天气查询
2. **obsidian** - 笔记管理
3. **1password** - 密码管理

#### 智能家居自动化

1. **openhue** - 灯光控制
2. **weather** + **openhue** - 根据天气自动调整灯光

#### 工作效率套件

1. **gog** - Google Workspace
2. **obsidian** - 知识管理
3. **oracle** - 复杂任务处理

#### 开发者工具集

1. **skill-creator** - 技能开发
2. **mcporter** - MCP 集成
3. **video-frames** - 媒体处理

---

### 故障排除

#### 技能不工作？

1. **检查依赖**: `which op`, `which ffmpeg` 等
2. **检查配置**: `ls -la ~/.openclaw/config/`
3. **查看日志**: `tail -f ~/.openclaw/logs/skills.log`
4. **更新技能**: `clawhub update [skill-name]`

#### 需要新技能？

1. **搜索市场**: `clawhub search [keyword]`
2. **自行开发**: 使用 skill-creator
3. **请求功能**: [https://github.com/openclaw/openclaw/issues](https://github.com/openclaw/openclaw/issues)

---

### 安全注意事项


| 技能            | 风险等级 | 建议               |
| ------------- | ---- | ---------------- |
| **1password** | 高    | 保护主密码，定期轮换令牌     |
| **gog**       | 中    | 使用最小权限 OAuth 范围  |
| **wacli**     | 中    | 使用备用 WhatsApp 账号 |
| **openhue**   | 低    | 限制本地网络访问         |
| **weather**   | 低    | 无敏感数据            |


