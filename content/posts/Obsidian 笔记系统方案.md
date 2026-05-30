



## 🗂️ 一、文件夹结构（推荐）

```text
MyNotes/                     # 你的 Obsidian 仓库根目录
├── 00 Inbox/                # 快速捕获，未加工的零散内容
│   ├── 草稿/                # 写作草稿或临时想法
│   └── 待分类/              # 定期清理，归入下方文件夹
├── 10 Projects/             # 有明确截止日期的任务
│   ├── 写年终报告/
│   ├── 学习 Obsidian 插件开发/
│   └── 搭建个人博客/
├── 20 Areas/                # 长期负责的领域（无截止，持续维护）
│   ├── 健康/
│   ├── 财务管理/
│   ├── AI 工具研究/
│   └── 职业发展/
├── 30 Resources/            # 兴趣/知识素材库（按主题）
│   ├── 读书笔记/
│   ├── 技术文档/
│   ├── 会议记录/
│   └── 灵感片段/
├── 40 Archives/             # 已完成的 Projects / 不再活跃的 Areas
├── 99 Templates/            # 笔记模板（见下文）
├── _scripts/                # Dataview 查询脚本、自定义 CSS 等
└── Attachments/             # 图片、PDF 等附件
```

### 设计思路
- **数字前缀**：控制文件夹排序，`00` 为入口，`10-30` 为活跃区，`40` 为归档区。
- **`00 Inbox`**：快速记录，配合快捷键 `Ctrl/Cmd + N` 创建新笔记，放入 Inbox，每天/每周整理一次。
- **Projects vs Areas**：Projects 有终点，Areas 持续维护（如“健康”）。Projects 完成后移到 `40 Archives`。
- **`30 Resources`**：存放知识卡片，配合 Zettelkasten 风格的 `[[]]` 链接。
- **`_scripts` 和 `99 Templates`**：放工具类文件，不干扰笔记浏览。

---

## 🔌 二、插件推荐列表（分级别）

### ✅ 核心插件（Obsidian 自带，务必开启）
| 插件 | 作用 |
$$
|------|------|
$$
| 模板 | 插入预定义模板（配合 `99 Templates` 文件夹） |
| 标签面板 | 查看所有标签，辅助分类 |
| 大纲 | 快速跳转标题 |
| 星标 | 常用笔记或文件夹置顶 |
$$
| 搜索 | 全局搜索 + 搜索语法 |
$$
### ⭐ 必装社区插件（提升体验）
| 插件 | 功能 | 备注 |
$$
|------|------|------|
$$
$$
| **Dataview** | 自动查询生成列表/表格 | 核心工具（下面有示例） |
$$
$$
| **Templater** | 动态模板（可执行 JS） | 替代自带模板，更强大 |
$$
$$
| **Calendar** | 在侧边栏显示日历，点击创建日记 | 配合日记模板 |
$$
$$
| **Excalidraw** | 绘图工具 | 适合流程图、概念图 |
$$
$$
| **QuickAdd** | 快速捕获并执行动作（如带标签的新笔记） | 极大提升输入速度 |
$$
### 🚀 AI 集成插件（按需选择）
| 场景 | 推荐插件 | 配置要点 |
$$
|------|----------|----------|
$$
$$
| 日常写作辅助 | **Copilot** | 自定义模型指向 DeepSeek API（便宜） |
$$
| 批量处理笔记 | **Claudian** + MCP | 需配置 DeepSeek 兼容 API 和 MCP 服务器 |
$$
| 语义搜索/问答 | **Smart Connections** | 本地嵌入模型，不依赖外部 API |
$$
$$
| 代码或技术笔记 | **Cursor 直接打开仓库** | 利用 Cursor 的 Composer 修改笔记 |
$$
---

## 📝 三、简单 Dataview 查询示例

将下面的代码放到一个普通笔记中，切换到“阅读视图”即可看到动态结果。

### 1️⃣ 所有未完成的项目（带截止日期）
```dataview
TABLE due as "截止日期", file.folder as "位置"
FROM "10 Projects"
WHERE due AND !completed
SORT due ASC
```
$$
*前提：你需要在笔记 frontmatter 中添加 `due:` 和 `completed:` 字段。*
$$
### 2️⃣ 最近一周的收件箱内容
```dataview
LIST file.link
FROM "00 Inbox"
WHERE file.cday >= date(today) - dur(7 days)
SORT file.cday DESC
```

### 3️⃣ 未被链接的孤立笔记（容易遗忘）
```dataview
LIST
FROM -"99 Templates" AND -"_scripts"
WHERE length(file.inlinks) = 0
SORT file.mtime DESC
LIMIT 20
```

---

## 📄 四、模板示例（用于 Templater）

### 1️⃣ 日常笔记模板（`99 Templates/Daily.md`）
```markdown
---
date: <% tp.date.now("YYYY-MM-DD") %>
weekday: <% tp.date.now("dddd", 0, tp.file.title, "YYYY-MM-DD") %>
tags: [daily, review]
---

# <% tp.file.title %> 的日常

## 🎯 今日要事
1. [ ] 
2. [ ] 
3. [ ] 

## 📝 笔记记录
- 

## 💡 灵光一闪
- 

## ✅ 今日回顾
- 
```

### 2️⃣ 项目笔记模板（`99 Templates/Project.md`）
```markdown
---
type: project
status: 进行中
due: <% tp.date.now("YYYY-MM-DD", 7) %>
tags: [project]
---

# <% tp.file.title %>

## 🎯 目标
- 

## 📅 里程碑
- [ ] 

## 🔗 相关资料
- 

## 📝 日志
### <% tp.date.now("YYYY-MM-DD") %>
- 
```

使用方式：安装 Templater 后，在命令面板执行 `Templater: Insert template` 并选择对应模板。

---

## 🧠 五、完整起步建议（一周行动清单）
$$
1. **第1天**：按上面的文件夹结构建好目录，开启核心插件，安装 Dataview 和 Templater。
$$
$$
2. **第2-3天**：只做一件事——看到任何东西都记到 `00 Inbox`，不整理。
$$
$$
3. **第4天**：每天结束时花10分钟，将 Inbox 笔记移入 `10/20/30` 对应的文件夹，并加上 `[[]]` 链接。
$$
$$
4. **第5天**：设置一个模板（比如 Daily），开始用 `Ctrl/Cmd + N` 快速创建笔记。
$$
$$
5. **第6-7天**：尝试写一个 Dataview 查询（比如上面的孤岛笔记查询），看看哪些笔记是孤岛，补充链接。
$$
$$
当这个基础流程跑顺后，你可以再逐步引入 AI 插件。如果需要的话，我可以帮你写一个 **QuickAdd 脚本** 实现“一键捕获 + 自动打标签”，或者根据你的笔记量设计更精细的 Dataview 看板。
$$