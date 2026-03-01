---
title: "如何在 DoIt 中编写文章"
subtitle: "Front Matter 填写说明与 content 目录规范"
date: 2026-02-28T10:26:59+08:00
# lastmod: 2026-02-28T10:26:59+08:00
draft: false
authors: [Steven]
description: "指导在 content/posts 中新增与编写文档，说明 front matter 各字段含义、编写规范，以及与 content 下各目录的兼容关系。"

tags: [how-to-doit]
categories: [how-to-doit]
series: [DoIt]
weight: 1000
series_weight: 1000

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: "/mywebsite/posts/images/how-to-doit.webp"

---

本文说明在本站（Hugo + DoIt 主题）中如何**新增文章**、如何**填写 front matter**，以及如何与 `content/` 下的**分类、系列、标签、作者**等目录配合使用。

<!--more-->

---

## 1 在 posts 中新增一篇文章

### 1.1 文件位置与命名

- **路径**：所有博文放在 `content/posts/` 下。
- **文件名**：建议使用英文或拼音，避免空格（可用 `-` 连接），例如：
  - `how-to-DoIt.md`
  - `An Introduction to Flow Matching and Diffusion Models.md`（含空格也可，但 URL 会转义）
- **格式**：Markdown（`.md`），文件开头必须是 **YAML front matter**（用 `---` 包裹）。

### 1.2 最小可用 front matter 示例

```yaml
---
title: "文章标题"
date: 2026-02-28
draft: false
---
```

其余字段会使用 `config/_default/params.toml` 中 `[page]` 的默认值。

---

## 2 Front Matter 字段说明与填写规范

下面按「常用 → 可选」顺序说明各字段含义、可选值及与 `content/` 目录的对应关系。

### 2.1 标题与描述

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `title` | 字符串 | 文章标题，必填，用于列表与正文页 | `"An Introduction to Flow Matching"` |
| `subtitle` | 字符串 | 副标题，可选，显示在正文标题下方 | `"subtitle"` 或留空 |
| `description` | 字符串 | 摘要/描述，用于 SEO 与摘要展示；不填时部分场景用正文摘要 | `"本文介绍..."` |

**规范建议**：`title` 简洁明确；`description` 可写一两句概括，便于搜索与列表展示。

### 2.2 时间

| 字段 | 类型 | 说明 | 与站点配置关系 |
|------|------|------|----------------|
| `date` | 日期 | 发布时间，建议必填 | 格式由 `params.dateFormat` 控制（如 `2006-01-02`） |
| `lastmod` | 日期 | 最后修改时间，可选 | **不写时**：若 `config/_default/hugo.toml` 中 `enableGitInfo = true`，会使用该文件的 **Git 最后提交时间**；否则不显示“最后修改” |

**规范建议**：新文章写上当天的 `date`；若希望“最后修改”来自 Git，可不写 `lastmod` 并保持 `enableGitInfo = true`。

### 2.3 发布与作者

| 字段 | 类型 | 说明 |
|------|------|------|
| `draft` | 布尔 | `true` 时文章不会在正式构建中发布；本地 `hugo server` 默认也不显示，需加 `-D` 才显示 |
| `authors` | 列表 | 作者标识列表，与 `content/authors/` 及（若有）`data/authors` 配合使用，如 `["steven"]` |
| （无 authors 时） | — | 使用站点默认作者：`config/_default/params.toml` 中 `[author]` 的 `name`、`link`、`avatar` 等 |

**与 content 的兼容**：`content/authors/_index.zh-cn.md`（及 `_index.en.md`）是“作者”分类页；具体作者信息可在 `data/authors/` 中配置，供 `authors` 引用。

### 2.4 分类、系列与标签（与 content 目录的对应）

这三类都会在文章页和列表页显示，且**必须与 content 下已有目录/词条对应**，否则链接会 404 或显示异常。

| 字段 | 类型 | 对应 content 结构 | 说明 |
|------|------|-------------------|------|
| `categories` | 列表 | `content/categories/<分类名>/` | 每个元素对应一个分类目录名，如 `["documentation"]` → `content/categories/documentation/`，需存在 `_index.zh-cn.md` 或 `_index.en.md` |
| `series` | 列表 | `content/series/<系列名>/` | 如 `[getting-start]` → `content/series/getting-start/`，用于“系列”导航与列表 |
| `tags` | 列表 | `content/tags/<标签名>/` | 如 `["Flow Matching", "Diffusion"]` → 会生成/使用 `content/tags/` 下对应词条（通常由 Hugo 自动按 taxonomy 生成页面，若使用 `tags/content/` 这种结构，则需存在对应 `_index`） |

**规范建议**：

- **先有目录再引用**：新增分类或系列时，先在 `content/categories/` 或 `content/series/` 下新建目录并添加 `_index.zh-cn.md`（及多语言时的 `_index.en.md`），再在文章里写 `categories` / `series`。
- **命名一致**：front matter 里的 `categories`、`series` 取值要与目录名一致（如 `getting-start` 对应 `content/series/getting-start/`）。
- **标签**：tags 一般会通过 Hugo taxonomy 自动建页，若你站有 `content/tags/xxx/_index.zh-cn.md`，则标签名需与 `xxx` 对应。

**示例**（与当前站点一致）：

```yaml
tags: ["Flow Matching", "Diffusion"]
categories: ["documentation"]
series: [getting-start]
```

即：该文属于分类「文档」、系列「开始使用 DoIt」，并带有两个标签。

### 2.5 首页与搜索可见性

| 字段 | 类型 | 说明 |
|------|------|------|
| `hiddenFromHomePage` | 布尔 | `true` 时文章**不会出现在首页**文章列表中；默认 `false`（由 `params.page.hiddenFromHomePage` 控制） |
| `hiddenFromSearch` | 布尔 | `true` 时文章**不会出现在站内搜索结果**中（如 PageFind）；默认 `false` |

**规范建议**：正式发布的文章通常两者都设为 `false`；测试文或暂不对外展示的可设为 `true`。

### 2.6 头图

| 字段 | 类型 | 说明 |
|------|------|------|
| `featuredImage` | 字符串 | 正文头图 URL 或站点内路径（如 `/images/xxx.webp`） |
| `featuredImagePreview` | 字符串 | 列表/摘要卡中的预览图；不填时主题会回退使用 `featuredImage` |

**规范建议**：若暂无头图可留空 `""`；图片建议放在 `static/images/` 下，路径如 `/images/xxx.webp`。

### 2.7 目录（TOC）与代码块

| 字段 | 类型 | 说明 |
|------|------|------|
| `toc.enable` | 布尔 | 是否显示文章目录（侧边/静态） |
| `toc.auto` | 布尔 | 是否自动展开/折叠侧边目录；`false` 表示“始终展开”式 |
| `toc.keepStatic` | 布尔 | 是否在正文前保留静态目录块（可单独在 front matter 覆盖） |
| `code.maxShownLines` | 数字 | 代码块默认展开行数，超过可折叠 |
| `code.lineNos` | 布尔 | 是否显示行号 |
| `code.copy` | 布尔 | 是否显示“复制”按钮 |
| `code.wrap` | 布尔 | 是否换行（长行） |
| `code.header` | 布尔 | 是否显示代码块标题 |

未在 front matter 中写的项会使用 `config/_default/params.toml` 中 `[page.toc]` 与 `[page.code]` 的默认值。

**示例**（长文、代码多时常用）：

```yaml
toc:
  enable: true
  auto: false

code:
  maxShownLines: 100
  lineNos: true
  copy: true
```

---

## 3 与 content 下各目录的兼容关系小结

| content 目录 | 作用 | 与 posts 的配合方式 |
|--------------|------|----------------------|
| `content/posts/` | 所有博文 | 在此新增 `.md`，写好 front matter 与正文 |
| `content/categories/<名称>/` | 分类页 | 在文章里写 `categories: ["名称"]`，且该目录需存在并含 `_index` |
| `content/series/<名称>/` | 系列页与系列导航 | 在文章里写 `series: [名称]`，且该目录需存在并含 `_index` |
| `content/tags/` | 标签页 | 在文章里写 `tags: ["标签A", "标签B"]`，与 taxonomy 或既有 `tags/xxx/` 对应 |
| `content/authors/` | 作者列表页 | 使用 `authors: ["id"]` 时，可与 `data/authors` 或作者页结构配合 |
| `content/about/` | 关于页 | 独立页面，不依赖 posts front matter |
| `content/showcase/` | 展示页 | 独立页面 |
| `content/offline/` | 离线页 | 独立页面 |

**多语言**：若启用多语言，`_index` 会有 `_index.zh-cn.md`、`_index.en.md` 等；posts 的 front matter 不直接写语言，由文件名或 `config` 的 `defaultContentLanguage` 等决定。

---

## 4 推荐的新文章模板（复制即用）

```yaml
---
title: "你的文章标题"
subtitle: ""                    # 可选
date: 2026-02-28                # 建议改成当天
# lastmod: 2026-02-28           # 可选；不写则用 Git 最后提交时间（需 enableGitInfo = true）
draft: false
# authors: [steven]             # 可选；不写则用站点默认作者
description: "一句话描述本文"

tags: ["标签1", "标签2"]
categories: ["documentation"]   # 需存在 content/categories/documentation/
series: [getting-start]         # 需存在 content/series/getting-start/

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""

toc:
  enable: true
  auto: false

code:
  maxShownLines: 100
  lineNos: true
  copy: true
---
```

正文从第二个 `---` 之后开始写，使用标准 Markdown；如需“摘要与全文分隔”，在摘要结束处插入 `<!--more-->`。

---

## 5 编写规范（约定）

1. **必填**：`title`、`date`、`draft`（建议显式写 `false` 再发布）。
2. **分类/系列**：只填写已在 `content/categories/`、`content/series/` 下创建过的名称。
3. **标签**：与站点现有标签或 `content/tags/` 结构一致，避免临时造词导致大量空标签页。
4. **描述**：正式文章建议写 `description`，便于 SEO 和列表展示。
5. **头图**：可选；若用，路径统一用站点内路径（如 `/images/...`）。
6. **lastmod**：若希望由 Git 自动反映修改时间，可不写 `lastmod` 并保持 `enableGitInfo = true`。

按上述方式在 `content/posts/` 下新增或修改文章，即可与 DoIt 主题及本站 `content/` 目录结构兼容、正常显示在首页、分类、系列与搜索中。
