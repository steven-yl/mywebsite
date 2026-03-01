---
title: "exampleSite使用指南"
date: 2026-02-28
draft: false
authors: [Steven]
description: "themes/DoIt/exampleSite 下各文件/目录的作用及如何在自己的站点中使用。"
tags: [how-to-doit]
categories: [how-to-doit]
series: [DoIt]
weight: 1001
series_weight: 1001


hiddenFromHomePage: false
hiddenFromSearch: false

---

`themes/DoIt/exampleSite` 是 DoIt 主题的**示例站点**，用一套完整的 config、content、static、data 演示主题功能。**不要用它覆盖你的项目**，只按需参考或拷贝需要的部分。


---

## 一、根目录

**schema.json**  
JSON Schema，描述站点配置结构，供编辑器校验/补全。Hugo 构建不读它，一般不需要拷贝到自己的站。

---

## 二、config/_default/

站点配置。你的站在项目根已有 `config/_default/`，这里**只作参考**，需要哪项就抄哪项，不要整份覆盖。

| 文件名 | 作用 |
|--------|------|
| hugo.toml | 主配置：baseURL、标题、默认语言、主题、Git 等 |
| hugo.zh-cn.toml / hugo.en.toml | 按语言的 Hugo 配置 |
| params.toml | 主题参数：外观、首页、目录、代码块、评论、搜索等 |
| params.zh-cn.toml / params.en.toml | 按语言的主题参数 |
| menu.zh-cn.toml / menu.en.toml | 顶部导航菜单（文章、标签、分类、系列、关于等） |
| markup.toml | Markdown 扩展：高亮、目录、锚点等 |
| taxonomies.toml | 分类法：categories、tags、series |
| permalinks.toml | 文章/页面 URL 规则 |
| outputFormats.toml / outputs.toml | 输出格式与各 section 输出配置 |
| sitemap.toml | 站点地图 |
| pagination.toml | 分页 |
| mediaTypes.toml | 媒体类型 |
| privacy.toml | 隐私相关（第三方脚本等） |

**用法**：缺某项时到 exampleSite 里找到对应文件，把相关配置抄到你项目根目录的 `config/_default/` 下同名或已有文件中。

---

## 三、content/

示例页面与文章。你的内容应写在项目根 `content/`，这里只当**模板或参考**。

### 单页

| 路径 | 作用 | 用法 |
|------|------|------|
| content/about/index.zh-cn.md、index.en.md | 关于页 | 没有关于页时可拷 about 整目录到项目 content/about/，再改文案 |
| content/showcase/index.*.md | 作品/展示页 | 需要时拷到项目 content/showcase/ 并改内容 |
| content/offline/index.en.md | PWA 离线说明页 | 开 PWA 时可拷到 content/offline/，需要则补中文版 |

### 分类 / 标签 / 系列 / 作者

| 路径 | 作用 | 用法 |
|------|------|------|
| content/categories/documentation/_index.*.md | 分类「文档」的标题与说明 | 新建分类时照此结构建 content/categories/&lt;名称&gt;/_index.zh-cn.md |
| content/tags/configuration、content、installation | 各标签页 | 新建标签时在 content/tags/&lt;标签名&gt;/ 下放 _index.zh-cn.md |
| content/series/_index.*.md | 系列总页 | 参考 |
| content/series/getting-start、how-to-doit、test-series | 各系列说明 | 新建系列时拷一个 _index 到 content/series/&lt;系列名&gt;/ 并改 title |
| content/authors/_index.*.md | 作者列表页 | 用多作者时可拷到 content/authors/ 并改文案 |

### 文章 content/posts/

- **theme-documentation-***、**basic-markdown-syntax**、**emoji-support**、**pwa-support** 等：主题文档与功能示例，需要时打开对应 md 看 front matter 和写法，不要整份拷到你的 posts。
- **posts/tests/** 下各目录：Bilibili、音乐、Mermaid、KaTeX 等 shortcode 测试，仅演示用；想用某 shortcode 时去对应文章里看用法。

---

## 四、static/

构建时原样复制到站点根目录，主要是**站点图标与 PWA**。

| 文件 | 作用 |
|------|------|
| favicon.ico | 浏览器标签页图标 |
| favicon-16x16.png、favicon-32x32.png | 多尺寸 favicon |
| apple-touch-icon.png | iOS 主屏图标 |
| android-chrome-192x192.png、512x512.png | Android / PWA 图标 |
| safari-pinned-tab.svg | Safari 固定标签图标 |
| mstile-150x150.png | Windows 磁贴 |
| site.webmanifest | PWA manifest（名称、图标、主题色等） |
| browserconfig.xml | IE/旧 Edge 磁贴配置 |
| _redirects | 部署重定向（如 Netlify/Cloudflare） |

**用法**：把 exampleSite 的 static/ 拷到项目根 static/，再**替换**成自己的 favicon、各尺寸图标，并修改 site.webmanifest 里的 name、short_name、description 等。

---

## 五、data/

| 路径 | 作用 | 用法 |
|------|------|------|
| data/authors/*.toml | 作者 id → 名称、链接、头像、邮箱 | 文章里写 authors: ["PCloud"] 时会读这里。用多作者时拷到项目 data/authors/，复制一份 toml 改成自己的 id（文件名即 id），填 name、link、avatar、email |

---

## 六、assets/

| 路径 | 作用 | 用法 |
|------|------|------|
| assets/css/_custom.scss | 自定义样式（主题会 @import） | 拷到项目 assets/css/_custom.scss，在里面写自己的 CSS |
| assets/css/_override.scss | 覆盖主题变量（颜色、字体等） | 要改主题变量时拷到项目 assets/css/ 并修改变量 |
| assets/images/* | 示例站头像、截图 | 仅参考，自己的图放项目 assets/images/ 或 static/images/ |
| assets/music/*.mp3 | 示例站音乐 | 仅演示，自己的音乐放项目并改 shortcode 路径 |

---

## 七、快速对照：我想… 该怎么做

| 需求 | 做法 |
|------|------|
| 用和示例站一样的图标 / PWA | 拷贝 static/ 到项目根，再替换图标和 site.webmanifest 中的名称、描述 |
| 自定义样式 | 拷贝 assets/css/_custom.scss（及按需 _override.scss）到项目 assets/css/ 再改 |
| 用多作者 | 拷贝 data/authors/*.toml 到项目 data/authors/，按 id 新建/修改 toml |
| 菜单、参数、多语言 | 在 config/_default/ 里找到对应文件，把需要的片段抄到自己项目的 config，不整份覆盖 |
| 关于页、展示页、离线页 | 从 content/about、showcase、offline 拷到项目 content 下并改内容 |
| 新分类 / 标签 / 系列 | 参考 content/categories、tags、series 的 _index 结构，在项目里建同名目录和 _index.zh-cn.md |
| 学写文章或 shortcode | 打开 content/posts/ 里对应示例或 tests 文章，看 front matter 和写法，按需复制单篇当模板 |

原则：**exampleSite 是参考与模板库，按“拷贝到项目根对应目录 + 按需修改”使用，不要用 exampleSite 整体替换你现有的 config 或 content。**
