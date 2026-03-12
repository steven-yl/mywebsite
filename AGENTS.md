## Learned User Preferences

<!-- markdown语法，请参考如下路径下的文档内容规范：
- themes/DoIt/exampleSite/content/posts/theme-documentation-built-in-shortcodes/index.zh-cn.md
- themes/DoIt/exampleSite/content/posts/theme-documentation-content/index.zh-cn.md
- themes/DoIt/exampleSite/content/posts/theme-documentation-extended-shortcodes/index.zh-cn.md -->
---

## Learned Workspace Facts



---

## 博客状态更新

当收到"更新博客状态"输入时，根据【git commit的时间】比【guide版本号】时间判断新增内容，更新content/posts/blog_guide.md内容, 同时更新guide版本号(日期)、新增的标签 / 分类 / 系列。


---

## 博客编写规范

结合当前 `content/posts` 下已发表文章，总结本博客的推荐编写规范，供新文章创建与旧文维护时参考。

### Frontmatter 约定

- **必填字段**：
  - `title`：简洁明确，能在列表中一眼看出主题。
  - `date`：使用本地时间，含时区，例如 `2026-03-12T00:00:00+08:00`。
  - `draft`：新建时可设为 `true`，发布前改为 `false`。
  - `authors`：统一使用 `[Steven]`。
  - `description`：1～2 句自然语言总结全文核心内容，便于列表与 SEO。
  - `summary`：1～2 句自然语言总结全文核心内容，便于列表与 SEO。
- **推荐字段**：
  - `tags`：细颗粒度关键词（技术名词、工具名、主题），从[第 1 节的去重列表](#1-全站标签--分类--系列汇总)中优先复用，避免同义词泛滥。
  - `categories`：较粗粒度的栏目，如 `diffusion/flow`、`papers`、`tools`、`PyTorch`、`how-to-doit` 等，可多选但不宜过多。
  - `series`：属于系列文章时必填，如 `diffusion/flow系列`、`tools系列`、`DoIt`、`robots系列`。
  - `weight` / `series_weight`：用于控制同一目录或系列内的排序（数值越小越靠前），长系列建议按章节顺序递增。
  - `featuredImage` / `featuredImagePreview`：如果有同名的配图，增加内容，路径建议为 `/mywebsite/posts/images/<slug>.webp`，文件名与 slug 对应。

### 目录与分类选择

- **目录归属**：
  - 扩散/流模型相关文章：`content/posts/diffusion-flow/`。
  - 论文解读类：`content/posts/papers/`。
  - 工具、环境、插件、Prompt 等：`content/posts/tools/`。
  - PyTorch 技术文档、深度学习实战：`content/posts/pytorch/`。
  - 其他不成体系的杂记：`content/posts/others/`。
- **tags / categories / series 使用建议**：
  - 新文章优先从本文件第 1 节中的去重列表中挑选已有值，必要时再新增。
  - 同一技术栈保持统一命名，例如 `diffusion/flow`、`PyTorch`、`tools`，避免出现 `Diffusion` / `diffusion` 等大小写混用。
  - 若某篇文章属于长系列（如 Flow Matching 解读、DDPM 系列），务必设置 `series`，方便 Hugo 按系列聚合展示。

### 正文结构与写作风格
- **语言与术语**：
  - 中文为主，遇到专有名词（如算法名、函数名、类名）保持英文原文，并在首次出现时简单解释。
  - 同一技术在不同文章中使用同一译法和英文写法（如「扩散模型 / Diffusion Models」）。
- **代码与公式**：
  - 代码块注明语言（如 ```python、```bash、```markdown），便于高亮与复制。
  - 若有数学公式，使用 KaTeX 格式：行内 `$...$`，块级 `$$...$$`。
  - 示例代码应尽量可运行，若为伪代码要在文本中注明。

### 链接与交叉引用
- 当新文章引用旧文中的推导、代码或概念时，优先使用站内链接，并在附近简要复述关键信息，避免读者必须来回跳转。
- 若某篇文章可视为另一个更长文档的「轻量版」，可在开头说明阅读路径（如「本篇是 `Flow-Matching-Guide-and-Code.md` 的项目解析向导」）。

### 实时更新关联文档
- 新增新文章后，同步更新blog_guide.md文档内容

以上规范会随着仓库演进适当更新，新增文章时若有偏离，提示如何在本文件中补充规范或注释原因，再在具体文章中实施。