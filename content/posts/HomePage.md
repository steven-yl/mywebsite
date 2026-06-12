---
cssclass: dashboard
draft: true
hiddenFromHomePage: true
hiddenFromSearch: true
---

> [!note]+ 概览
>
> ```dataview
> TABLE WITHOUT ID
>   分区,
>   length(rows) AS 数量
> FROM ""
> WHERE file.ext = "md"
>   AND (contains(file.folder, "00 Inbox") OR contains(file.folder, "10 Projects") OR contains(file.folder, "20 Areas") OR contains(file.folder, "30 Resources"))
>   AND !contains(file.path, "Excalidraw")
>   AND !contains(file.name, "00-")
>   AND file.name != "HomePage"
>   AND file.name != "模板"
> GROUP BY choice(
>   contains(file.folder, "00 Inbox"), "📥 Inbox",
>   choice(contains(file.folder, "10 Projects"), "🚀 Projects",
>   choice(contains(file.folder, "20 Areas"), "🧩 Areas", "📚 Resources"))
> ) AS 分区
> SORT 分区 ASC
> ```

<div class="dash-columns dash-tri">

> [!todo]+ Inbox
>
> ```dataview
> LIST
> FROM ""
> WHERE file.ext = "md"
>   AND contains(file.folder, "00 Inbox")
>   AND file.name != "HomePage"
> SORT file.mtime DESC
> LIMIT 5
> ```

> [!warning]+ Todo
>
> ```dataview
> LIST
> FROM ""
> WHERE file.ext = "md"
>   AND contains(file.tags, "todo")
>   AND !contains(file.name, "00-")
>   AND file.name != "HomePage"
> SORT file.mtime DESC
> LIMIT 5
> ```

> [!failure]+ Draft
>
> ```dataview
> LIST
> FROM ""
> WHERE file.ext = "md"
>   AND contains(file.tags, "draft")
>   AND file.name != "HomePage"
> SORT file.mtime DESC
> LIMIT 5
> ```

</div>

<div class="dash-columns">

> [!example]+ 最近编辑
>
> ```dataview
> TABLE WITHOUT ID
>   choice(contains(file.folder, "20 Areas"), "🧩", choice(contains(file.folder, "30 Resources"), "📚", choice(contains(file.folder, "00 Inbox"), "📥", "🚀"))) AS " ",
>   link(file.name) AS 笔记,
>   dateformat(file.mtime, "MM-dd") AS 日期
> FROM ""
> WHERE file.ext = "md"
>   AND (contains(file.folder, "00 Inbox") OR contains(file.folder, "10 Projects") OR contains(file.folder, "20 Areas") OR contains(file.folder, "30 Resources"))
>   AND !contains(file.folder, "40 Archives")
>   AND !contains(file.folder, "50 Templates")
>   AND !contains(file.path, "Excalidraw")
>   AND !contains(file.name, "00-")
>   AND file.name != "HomePage"
>   AND file.name != "模板"
>   AND file.name != "Obsidian 笔记系统方案"
> SORT file.mtime DESC
> LIMIT 6
> ```

> [!question]+ 孤立笔记
>
> ```dataview
> LIST
> FROM ""
> WHERE file.ext = "md"
>   AND (contains(file.folder, "00 Inbox") OR contains(file.folder, "10 Projects") OR contains(file.folder, "20 Areas") OR contains(file.folder, "30 Resources"))
>   AND length(file.inlinks) = 0
>   AND length(file.outlinks) = 0
>   AND file.name != "HomePage"
>   AND !contains(file.name, "00-")
>   AND file.name != "模板"
> SORT file.mtime DESC
> LIMIT 6
> ```

</div>

> [!abstract]+ 导航
>
> | 分区 | 入口 |
> |:--|:--|
> | Inbox | [[00 Inbox/待读链接\|待读链接]] · [[00 Inbox/待读链接整理\|待读链接整理]] · [[00 Inbox/question\|question]] |
> | Projects | [[10 Projects/how-to-doit/00-DoIt系列索引\|DoIt 博客]] · [[10 Projects/how-to-doit/如何在 DoIt 中编写文章\|写文章]] |
> | Templates | [[50 Templates/模板\|笔记模板]] |
> | Archives | [[40 Archives/others/00-others系列索引\|归档]] |
> | 参考 | [[Obsidian 笔记系统方案\|Obsidian 方案]] |
>
> **Areas** · [[20 Areas/diffusion-flow/00-diffusion-flow系列索引\|diffusion]] · [[20 Areas/pytorch/00-PyTorch实践指南系列索引\|PyTorch]] · [[20 Areas/deep-learning/00-深度学习系列索引\|深度学习]] · [[20 Areas/rl/00-RL系列索引\|RL]] · [[20 Areas/robots/00-robots系列索引\|robots]] · [[20 Areas/math/00-math系列索引\|math]] · [[20 Areas/think/00-思维工具系列索引\|think]]
>
> **Resources** · [[30 Resources/CS492D/00-CS492D系列索引\|CS492D]] · [[30 Resources/ai-tools/00-tools系列索引\|tools]] · [[30 Resources/papers/00-papers系列索引\|papers]] · [[30 Resources/diffusion-flow-tutorial/00-diffusion-flow-tutorial系列索引\|tutorial]] · [[30 Resources/torchcode-docs/00-TorchCode 技术文档索引\|TorchCode]] · [[30 Resources/smalldiffusion-docs/00-smalldiffusion 技术文档索引\|smalldiffusion]] · [[30 Resources/Analytic-Diffusion-Studio-docs/00-Analytic Diffusion Studio — 技术文档索引\|ADS]]

> [!info|no-icon]-
> [diffusion/flow](obsidian://search?q=tag:%23diffusion%2Fflow) · [PyTorch](obsidian://search?q=tag:%23PyTorch) · [tools](obsidian://search?q=tag:%23tools) · [todo](obsidian://search?q=tag:%23todo) · [draft](obsidian://search?q=tag:%23draft)
