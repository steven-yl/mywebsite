---
title: "Analytic Diffusion Studio — 配置系统"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 配置系统"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 3
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 配置系统"
---

文件：`src/local_diffusion/configuration.py`

配置系统基于 OmegaConf 实现，支持 YAML 继承、CLI 覆盖、结构化类型校验。

## 3.1 配置数据类

配置系统定义了一组嵌套的 `dataclass`，作为配置的结构化 schema：

### ExperimentConfig
实验元数据。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"default"` | 实验分组名（决定输出目录层级） |
| `run_name` | str | `None` | 运行名称（输出子目录名） |
| `seed` | int | `42` | 全局随机种子 |
| `tags` | List[str] | `[]` | 实验标签（用于 WandB 过滤） |
| `append_timestamp` | bool | `True` | 是否在运行名后追加时间戳 |
| `device` | str | 自动检测 | 计算设备：`cuda` / `cpu` / `mps` |

### PathsConfig
目录布局。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root` | str | `"data"` | 数据根目录 |
| `datasets` | str | `None` → `data/datasets` | 数据集存储路径 |
| `models` | str | `None` → `data/models` | 模型/索引存储路径 |
| `runs` | str | `None` → `data/runs` | 实验输出路径 |
| `wandb` | str | `"wandb"` | WandB 日志路径 |

### DatasetConfig
数据集加载参数。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"mnist"` | 数据集名称（注册表键） |
| `split` | str | `"train"` | 数据集划分 |
| `download` | bool | `True` | 是否自动下载 |
| `batch_size` | int | `512` | DataLoader 批大小 |
| `num_workers` | int | `4` | DataLoader 工作进程数 |
| `subset_size` | int | `None` | 子集大小（`None` 表示全量） |
| `root` | str | `None` | 数据集根目录（默认继承 paths.datasets） |
| `resolution` | int | `None` | 图像分辨率覆盖 |

### ModelConfig
模型选择与超参数。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | `"nearest_dataset"` | 模型名称（注册表键） |
| `params` | Dict | `{}` | 模型特定超参数 |

### SamplingConfig
采样参数。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_samples` | int | `16` | 生成样本总数 |
| `batch_size` | int | `8` | 每批生成数量 |
| `num_inference_steps` | int | `10` | DDIM 推理步数 |

### MetricsConfig
评估与输出配置。

| 字段 | 类型 | 说明 |
|------|------|------|
| `baseline_path` | str | UNet 基线权重路径（`None` 则跳过对比） |
| `output` | OutputConfig | 输出开关（见下） |
| `wandb` | WandbConfig | WandB 配置（见下） |

### OutputConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `code_snapshot` | `True` | 保存代码快照（tar.gz） |
| `save_metrics` | `True` | 保存 metrics.json |
| `save_final_images` | `True` | 保存最终生成图像 |
| `save_image_grid` | `True` | 保存图像网格 |
| `save_intermediate_images` | `True` | 保存中间去噪步骤图像 |

### WandbConfig

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `True` | 是否启用 WandB |
| `project` | `"local-diffusion"` | WandB 项目名 |
| `entity` | `None` | WandB 团队/用户 |
| `mode` | `"online"` | `online` / `offline` / `disabled` |
| `tags` | `None` | 标签（`None` 时继承 experiment.tags） |
| `job_type` | `"generation"` | WandB 任务类型 |

### RunPaths
运行时路径集合（由 `ensure_run_directory()` 创建）。

| 字段 | 说明 |
|------|------|
| `run_dir` | 运行根目录 |
| `artifacts` | 产物目录 |
| `images` | 最终图像目录 |
| `tensors` | 张量存储目录 |
| `intermediate_images` | 中间步骤图像目录 |
| `logs` | 日志目录 |
| `config` | 保存的配置文件路径 |

## 3.2 配置加载流程

`load_config(config_path, overrides)` 的执行步骤：

```
1. _resolve_config_path()
   ├── 绝对路径 → 直接使用
   ├── 相对路径 → 先尝试 configs/ 前缀
   └── 否则原样使用

2. OmegaConf.load(config_path)
   └── 解析 YAML 文件

3. 处理 defaults 列表
   ├── 遍历 defaults 中的路径
   ├── 解析每个默认配置文件
   └── OmegaConf.merge(*defaults, main_config)

4. OmegaConf.merge(structured_base, merged)
   └── 与 Config dataclass 合并（提供类型校验和默认值）

5. 应用 CLI overrides
   └── OmegaConf.from_dotlist(overrides)

6. _resolve_default_paths()
   └── 将相对路径解析为绝对路径

7. _resolve_metrics_defaults()
   └── wandb.tags 为空时继承 experiment.tags

8. OmegaConf.set_readonly(True)
   └── 冻结配置，防止意外修改
```

## 3.3 配置继承示例

`configs/pca_locality/celeba_hq.yaml`：

```yaml
defaults:
  - /defaults.yaml          # 继承基础配置

experiment:
  run_name: pca_locality_celeba_hq  # 覆盖运行名
  tags: [baseline, pca_locality, celeba_hq]

dataset:
  name: celeba_hq
  resolution: 64            # 覆盖分辨率（降低计算量）

model:
  name: pca_locality
  params:
    temperature: 1.0
    mask_threshold: 0.02
```

合并优先级：`defaults.yaml` < 具体配置文件 < CLI overrides

## 3.4 CLI 覆盖

通过点号分隔的路径覆盖任意配置值：

```bash
uv run generate.py --config pca_locality/celeba_hq.yaml \
    sampling.num_samples=16 \
    model.params.temperature=0.5 \
    experiment.device=cpu \
    metrics.wandb.enabled=false
```

## 3.5 运行目录创建

`ensure_run_directory(cfg)` 创建如下目录结构：

```
data/runs/{experiment.name}/{run_name}_{timestamp}/
├── config.yaml
├── artifacts/
│   ├── images/
│   ├── tensors/
│   └── intermediate_images/
└── logs/
```

## 3.6 代码快照

`snapshot_codebase()` 函数：
1. 调用 `git ls-files` 获取所有 git 跟踪的文件
2. 打包为 `code_snapshot.tar.gz`
3. 存储在运行目录下，确保实验可复现

## 3.7 辅助函数

| 函数 | 说明 |
|------|------|
| `save_config(cfg, path)` | 将 OmegaConf 配置保存为 YAML |
| `config_to_dict(cfg)` | 将配置转为普通 Python dict |
| `get_git_tracked_paths(root)` | 获取 git 跟踪的文件列表 |
