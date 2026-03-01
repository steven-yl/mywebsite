# [mywebsite](https://steven-yl.github.io/mywebsite/)

基于 Hugo + DoIt 主题的个人站点。

---

## Hugo 常用指令（带说明）

### 本地预览

| 指令 | 说明 |
|------|------|
| `hugo serve` | 启动开发服务器，默认 `http://localhost:1313`，修改内容会热重载。 |
| `hugo serve -e production` | 以 **production** 环境运行，与正式构建一致（会应用 production 的 config、不包含 draft 等）。 |
| `hugo serve --disableFastRender` | 关闭“快速渲染”：每次改动后完整重建，避免模板/资源缓存导致页面不更新。 |
| `hugo serve --cleanDestinationDir` | 每次启动前**清空** `public/`（或 `--destination` 指定目录），避免旧文件残留。 |
| `hugo serve -D` | 同时渲染 **draft** 文章（`draft: true` 的也会显示）。 |
| `hugo serve -b https://example.com` | 指定 **baseURL**，影响 canonical、绝对链接等。 |

**常用组合**（开发时看到最新效果、避免缓存问题）：

```bash
hugo serve --disableFastRender --cleanDestinationDir
```

**生产环境预览**（与部署结果一致）：

```bash
hugo serve -e production --cleanDestinationDir
```

---

### 构建与部署

| 指令 | 说明 |
|------|------|
| `hugo` | 构建站点到默认目录 `public/`，不包含 draft、过期与未来日期的文章。 |
| `hugo -e production` | 以 production 环境构建，一般部署时用。 |
| `hugo -D` | 构建时**包含 draft** 文章。 |
| `hugo --cleanDestinationDir` | 构建前先清空输出目录。 |
| `hugo --minify` | 对 HTML/CSS/JS 做压缩，减小体积。 |
| `hugo -d ./dist` | 指定输出目录（此处为 `./dist`）。 |

**部署前常用**：

```bash
hugo -e production --cleanDestinationDir --minify
```

---

### 内容与站点管理

| 指令 | 说明 |
|------|------|
| `hugo new posts/文章名.md` | 在 `content/posts/` 下用 **archetype** 新建一篇文章。 |
| `hugo new about.md` | 在 `content/` 根下新建 `about.md`（无子目录时）。 |
| `hugo list drafts` | 列出所有 **draft** 文章。 |
| `hugo list future` | 列出 **publishDate** 在未来的文章。 |
| `hugo list expired` | 列出已 **expiryDate** 过期的文章。 |
| `hugo list all` | 列出所有页面（含 draft、未来、过期），可配合 `-D` 等理解渲染范围。 |

---

### 环境与配置

| 指令 | 说明 |
|------|------|
| `hugo config` | 输出最终合并后的配置（含环境变量、多环境等）。 |
| `hugo version` | 输出 Hugo 版本。 |

---

### 多语言站点（本项目为 zh-cn 等）

| 指令 | 说明 |
|------|------|
| `hugo serve --bind 0.0.0.0` | 监听所有网卡，局域网内其他设备可访问。 |
| `hugo --gc` | 构建后执行 **garbage collect**，删除未使用的缓存文件。 |