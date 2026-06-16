---
title: 04 — 数据管道与变换
subtitle: ""
date: 2026-06-17T10:26:59+08:00
# lastmod: 2026-06-17T10:26:59+08:00
draft: false
authors: [Steven]
description: ""
tags: [openpi]
categories: [openpi]
series: [openpi-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 4 章：数据管道与变换

源码：`src/openpi/transforms.py`、`training/data_loader.py`、`shared/normalize.py`、`shared/image_tools.py`。

## 4.1 设计目标

机器人数据集键名各异（LeRobot、RLDS、自定义 repack），而模型只接受统一的 `Observation` 字典。变换层在**数据加载器与模型之间**做可组合的键映射、数值缩放、图像尺寸与分词，使同一 `Pi0` 类可服务 ALOHA、LIBERO、DROID 等平台。

## 4.2 核心协议与组合

### `DataTransformFn`（Protocol）

```python
def __call__(self, data: DataDict) -> DataDict: ...
```

- 输入/输出均为**未 batch** 的嵌套 dict，叶子一般为 `numpy.ndarray`。
- 可在原 dict 上就地修改，也可返回新结构。

### `Group`

| 字段 | 含义 |
|------|------|
| `inputs` | 模型输入侧变换，按顺序执行 |
| `outputs` | 模型输出侧变换 |

`push(inputs=..., outputs=...)`：新 `inputs` 追加到末尾；新 `outputs` **插入到开头**（保证与输入逆序对称）。

### `CompositeTransform` / `compose`

顺序执行多个 `DataTransformFn`，`Policy` 用 `compose` 打包输入/输出链。

## 4.3 变换类完整说明

### `RepackTransform(structure)`

用嵌套 dict 描述**新键 → 扁平旧路径**（`flatten_dict` 用 `/` 分隔）。用于把 LeRobot 字段映射到 `image`/`state`/`actions` 等标准键。

### `InjectDefaultPrompt(prompt)`

若 batch 无 `prompt` 且配置了默认字符串，则注入。

### `Normalize(norm_stats, use_quantiles=False, strict=False)`

对 `data` 与 `norm_stats` 结构对齐的叶子做归一化：

- **Z-score**：\((x - \mu) / (\sigma + 10^{-6})\)
- **分位数**（`use_quantiles=True`）：映射到 \([-1,1]\)  
  \((x - q_{01}) / (q_{99} - q_{01} + \epsilon) \cdot 2 - 1\)

`strict=True` 时缺少统计键会报错。

### `Unnormalize(norm_stats, use_quantiles=False)`

`Normalize` 的逆变换；`strict=True` 要求 stats 键全覆盖。

### `ResizeImages(height, width)`

对 `data["image"]` 各相机调用 `openpi_client.image_tools.resize_with_pad`（保持纵横比 pad）。

### `SubsampleActions(stride)`

`actions = actions[::stride]`，用于降低动作序列频率。

### `DeltaActions(mask)` / `AbsoluteActions(mask)`

对 `mask` 为 True 的维度做**相对当前 state 的增量**或还原为绝对值：

```python
# DeltaActions 概念
actions[..., mask] -= state[..., mask]
```

训练常用 delta，推理后 `AbsoluteActions` 加回 state。

### `TokenizePrompt(tokenizer, discrete_state_input=False)`

调用 `PaligemmaTokenizer.tokenize(prompt, state)`，写入：

- `tokenized_prompt`
- `tokenized_prompt_mask`

π₀.₅ 设 `discrete_state_input=True` 将离散 state 写入文本。

### `TokenizeFASTInputs(tokenizer)`

调用 `FASTTokenizer.tokenize`，额外写入 `token_ar_mask`、`token_loss_mask`。

### `ExtractFASTActions(tokenizer, action_horizon, action_dim)`

将 `sample_actions` 返回的 token id 解码为 `actions` ndarray。

### `PromptFromLeRobotTask(task_key)`

从 LeRobot 样本的 task 字段生成 `prompt` 字符串。

### `PadStatesAndActions(model_action_dim)`

将 `state` 与 `actions` 最后一维 pad 到 `model_action_dim`（与 checkpoint 对齐）。

## 4.4 工具函数

| 函数 | 作用 |
|------|------|
| `flatten_dict` / `unflatten_dict` | 嵌套 dict ↔ 扁平 `/` 路径 |
| `transform_dict(patterns, tree)` | 按 glob 模式重命名/删除键 |
| `apply_tree(data, template, fn, strict)` | 对 data 与 template 同结构叶子应用 `fn` |
| `pad_to_dim(x, target_dim, axis, value)` | 末维或指定轴 padding |
| `make_bool_mask(*dims)` | 如 `make_bool_mask(6, -1, 6, -1)` 生成 gripper 等维度的 bool 掩码 |
| `_assert_quantile_stats` | 校验 `q01`/`q99` 存在 |

## 4.5 归一化统计（`shared/normalize.py`）

### `NormStats`

- `mean`, `std`：Z-score
- `q01`, `q99`：分位数归一化（可选）

### `RunningStats`

在线累积 mean/std 或分位数，供 `scripts/compute_norm_stats.py` 扫描数据集。

### `save` / `load` / `serialize_json` / `deserialize_json`

检查点 `assets/<asset_id>/norm_stats.json` 持久化；推理时 `create_trained_policy` 自动加载。

## 4.6 数据加载（`training/data_loader.py`）

### 协议

- `Dataset`：`__getitem__(index) -> sample`
- `IterableDataset`：`__iter__`
- `DataLoader`：`data_config()` + 迭代 batch

### 包装类

| 类 | 作用 |
|----|------|
| `TransformedDataset` | 对 map-style 数据集应用 `compose(transforms)` |
| `IterableTransformedDataset` | 流式 RLDS 同理 |
| `FakeDataset` | `model.fake_obs/act` 合成数据，测训练循环 |

### 工厂函数

| 函数 | 说明 |
|------|------|
| `create_torch_dataset` | LeRobot `LeRobotDataset` + repo_id |
| `create_rlds_dataset` | `DroidRldsDataset`（TensorFlow RLDS） |
| `transform_dataset` | 拼接 `repack` + `data_transforms` + `Normalize` + `model_transforms` |
| `transform_iterable_dataset` | RLDS 版 |
| `create_data_loader` | 根据 `DataConfig.rlds_data_dir` 选 torch 或 rlds |
| `create_torch_data_loader` | `TorchDataLoader`：多 worker、`_collate_fn` stack numpy |
| `create_rlds_data_loader` | `RLDSDataLoader`：JAX 数组 batch |

### `TorchDataLoader` 要点

- `local_batch_size = batch_size // jax.process_count()`
- `num_workers` 默认 2；`_worker_init_fn` 限制 JAX 预分配
- `sharding` 可选，将 batch 转为 `jax.Array` 分片

### `DataLoaderImpl`

统一 `__iter__` 返回 `(Observation, Actions)` 或带 metadata 的 batch，供 `train.py` 使用。

## 4.7 平台策略映射（`policies/*_policy.py`）

各 `*Inputs` / `*Outputs` 实现 `DataTransformFn`，在 `DataConfigFactory` 中注册。

| 模块 | Inputs 要点 | Outputs 要点 |
|------|-------------|--------------|
| `aloha_policy` | 关节翻转、gripper 角/线转换、`adapt_to_pi` | 逆变换动作 |
| `libero_policy` | 双相机 + state 8 维 → 标准 image 键 | 截取有效动作维 |
| `droid_policy` | 三路相机、关节 state | 7D 动作等 |

辅助：`make_*_example()` 生成文档/测试用假观测。

## 4.8 训练 batch 形状

经 collate 后典型 JAX batch：

```text
image: {cam: float32[B, 224, 224, 3]}  # 后续 Observation.from_dict 映射到 [-1,1]
state: float32[B, action_dim]
actions: float32[B, action_horizon, action_dim]
tokenized_prompt: int32[B, max_token_len]
...
```

## 4.9 典型用法

```python
# 仅演示变换链组装（无需 checkpoint）
from openpi.transforms import compose, Normalize, ResizeImages, PadStatesAndActions
from openpi.shared import normalize as norm

stats = norm.load("/path/to/assets/norm_stats")
pipeline = compose([
    ResizeImages(224, 224),
    Normalize(stats, use_quantiles=True),
    PadStatesAndActions(model_action_dim=32),
])
sample = pipeline(raw_sample)
```

## 4.10 章节边界

- 训练如何消费 DataLoader → [05-training-system.md](./05-training-system.md)
- Tokenizer 细节 → [07-backbone-tokenizers.md](./07-backbone-tokenizers.md)
- 运维向 norm 说明 → [../norm_stats.md](../norm_stats.md)
