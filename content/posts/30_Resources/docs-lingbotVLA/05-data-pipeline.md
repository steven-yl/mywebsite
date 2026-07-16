---
title: "5. 数据流水线"
subtitle: ""
date: 2026-07-13T12:00:00+08:00
draft: false
authors: [Steven]
description: "5. 数据流水线。"
summary: "5. 数据流水线。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA, robots]
series: [lingbotVLA-docs]
weight: 5
series_weight: 5
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 5. 数据流水线

---

## 5.1 总体流程

```
LeRobot v3.0 原始数据
  → LeRobotDataset (base_dataset.py)
  → FeatureTransform.apply (utils.py + transform.py)
  → VLADataset.__getitem__
  → VLADataCollatorWithPacking
  → MakeMicroBatchCollator (梯度累积)
  → train_lingbotvla.py
```

---

## 5.2 LeRobot 数据集

### `LeRobotDataset`（`base_dataset.py`）

继承 `lerobot.datasets.lerobot_dataset.LeRobotDataset`：

| 扩展 | 说明 |
|------|------|
| 跳过 HF video column | 视频帧走 `_query_videos` |
| `Resize(image_size)` | 默认 224×224 |
| `task` 字符串 | 由 task index 映射 |

### `VLADataset`

**构造参数：**

| 参数 | 来源 |
|------|------|
| `repo_id` / `train_path` | `data.train_path` |
| `data_name` | `configs/robot_configs/{data_name}.yaml` |
| `data_config` | `DataArguments` |
| `chunk_size` | `train.chunk_size` |
| `use_depth_align` | `align_params != {}` |

**Delta timestamps：** 对每个 action key，按 dataset fps 加载未来 `chunk_size` 帧：

```python
delta_timestamps = {
    action_key: [i/fps for i in range(chunk_size)]
    for action_key in action_keys
}
```

**`__getitem__`：** 失败时最多重试 200 次（随机 index 回退）。

**输出字段：**

| 键 | 形状 | 说明 |
|----|------|------|
| `images` | `(N_cam, C, H, W)` | 归一化图像 |
| `img_masks` | `(N_cam,)` | 相机是否有效 |
| `state` | `(max_state_dim,)` | padding 后状态 |
| `lang_tokens` | `(tokenizer_max_length,)` | 任务文本 token |
| `lang_masks` | `(tokenizer_max_length,)` | 有效 token mask |
| `actions` | `(chunk_size, max_action_dim)` | 未来动作块 |
| `action_is_pad` | `(chunk_size,)` | 轨迹末尾 pad |
| `joint_mask` | `(max_action_dim,)` | 哪些维度是真实关节 |
| `pil_images` | 可选 | depth 教师用 RGB |

---

## 5.3 双层配置：VLA YAML + Robot Config

### VLA 训练配置 (`configs/vla/*.yaml`)

声明**统一特征空间**：

```yaml
data:
  data_name: robotwin
  joints:
    - arm.position: 14
    - effector.position: 2
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  norm_type: bounds_99
  norm_stats_file: assets/norm_stats/robotwin_50.json
```

### Robot Config (`configs/robot_configs/robotwin.yaml`)

声明**原始 LeRobot key → 统一 key** 映射：

```yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state: {start: 0, end: 6}
        - observation.state: {start: 7, end: 13}
actions:
  - action.arm.position:
      origin_keys: [...]
      subtract_state: False   # True → 动作变 delta
images:
  - observation.images.camera_top:
      origin_keys: observation.images.cam_high
```

**一致性检查：** `check_robot_config()` 验证 robot config 的 joint/camera 名 ⊆ VLA config。

---

## 5.4 `FeatureTransform`

**文件：** `data/vla_data/utils.py`

### `FeatureInfo`

解析 `data.joints` / `data.cameras` 为结构化列表。

### `get_feature_mapping(robot_config, feature_info)`

构建 forward / reverse 键映射表。

### `apply(raw_sample)` 流水线

1. 设置 `action_is_pad`
2. **`convert_features`** — slice + concat 原始 key
3. **Delta 动作** — 若 `subtract_state: True`，action -= state
4. **`Normalizer.normalize`** — 按 JSON 统计量
5. **`pad_and_concat`** — 各 joint 类型 pad 到 `max_*_dim`
6. **`prepare_images/state/action/language/joint_pad`**

### `unapply(sample)` — 推理逆变换

逆 pad → unnormalize → 加回 state → reverse key mapping。

---

## 5.5 归一化

### 运行时 `Normalizer`（`transform.py`）

| `norm_type` | 公式 |
|-------------|------|
| `meanstd` | $(x - \mu) / \sigma$ |
| `bounds_99` | 线性映射 $[q_{01}, q_{99}] → [-1, 1]$ |
| `minmax` | $[min, max] → [-1, 1]$ |
| `identity` | 不变 |

### 离线 `RunningStats`（`utils/normalize.py`）

在线累积 mean/std/min/max/分位数（5000-bin 直方图）。

### `scripts/compute_norm.py`

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm.py ./configs/vla/real_load20000h.yaml \
    --data.data_name robotwin \
    --data.train_path /path/to/dataset \
    --data.norm_stats_file assets/norm_stats/custom.json
```

- 使用 `VLADataset(..., do_nomalize=False)` 获取原始转换后特征
- 对 `subtract_state: True` 的 action，按 chunk reshape 统计

**场景选择：**

| 环境 | 推荐 norm_type |
|------|----------------|
| 仿真 RoboTwin | `bounds_99` |
| 真实机器人 | `meanstd` |

---

## 5.6 图像与语言准备

### `prepare_images`（`transform.py`）

- Qwen：processor 输出 pixel values
- 其他：resize_with_pad + scale to [-1,1]

### `prepare_language`

PaliGemma 风格：`<bos>{task}\n` tokenize + pad/truncate 到 `tokenizer_max_length`。

---

## 5.7 Collator 与 DataLoader

### `VLADataCollatorWithPacking`（`data_transform.py`）

**state_features**（batch 维 concat）：  
`state, images, img_masks, lang_tokens, lang_masks, action_is_pad, actions, joint_mask, label, fast_mask`

其余 key：`default_collate`

### `MakeMicroBatchCollator`（`data_collator.py`）

将 batch 拆为 `num_micro_batch` 份供梯度累积：

```
num_micro_batch = global_batch_size / (micro_batch_size × dp_size)
```

### `build_dataloader`（`data_loader.py`）

VLA 路径（`rmpad=false`）：

```
StatefulDistributedSampler
  → DistributedDataloader
  → MakeMicroBatchCollator(VLADataCollatorWithPacking)
```

> **注意：** Qwen2-VL 训练必须 `rmpad: false`，否则报错。

---

## 5.8 其他 data 模块（非 VLA 主路径）

| 文件 | 用途 |
|------|------|
| `dataset.py` | DummyDataset, MappingDataset（文本 LM） |
| `dynamic_batching.py` | 动态 token batch（文本） |
| `batching_strategy.py` | TextBatchingStrategy |
| `chat_template.py` | 文本 chat 模板 |
| `constants.py` | IGNORE_INDEX, 多模态 token id |

---

## 5.9 自定义数据集 Checklist

1. 转换为 [LeRobot v3.0](https://github.com/huggingface/lerobot)
2. 编写 `configs/robot_configs/<name>.yaml`
3. VLA yaml 设置 `data_name`, `joints`, `cameras`
4. 运行 `compute_norm.py`
5. `bash train.sh tasks/vla/train_lingbotvla.py ...`

多任务合并（同机器人配置）：

```bash
python scripts/merge_lerobot_v21.py --sources task1,task2,... --output merged/
# 再转 v3.0
```

---

## 5.10 示例：单样本 dict

```python
# 训练 batch 中一个 micro-batch 的 key（概念示例）
batch = {
    "images": torch.Size([32, 3, 3, 224, 224]),      # B, N_cam, C, H, W
    "img_masks": torch.Size([32, 3]),
    "state": torch.Size([32, 75]),
    "lang_tokens": torch.Size([32, 72]),
    "lang_masks": torch.Size([32, 72]),
    "actions": torch.Size([32, 50, 75]),
    "action_is_pad": torch.Size([32, 50]),
    "joint_mask": torch.Size([32, 75]),
}
```

---

## 5.11 相关链接

- [LeRobot v3.0 转换脚本](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py)
- [项目 Custom Data Guide](../../lingbotvla/data/vla_data/README.md)
- [GM-100 数据集](https://huggingface.co/datasets/robbyant/gm100)
