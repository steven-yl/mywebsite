---
title: "10 CLI 命令参考"
subtitle: ""
date: 2026-07-10T17:44:00+08:00
draft: false
authors: [Steven]
description: "18 个 lerobot 控制台命令入口、参数与典型用法。"
summary: "LeRobot CLI 命令完整参考。"
tags: [lerobot, robots]
categories: [docs lerobot, robots]
series: [lerobot-docs]
weight: 10
series_weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 10 — CLI 命令参考

## 1. 概览

LeRobot 在 `pyproject.toml [project.scripts]` 注册 **18** 个控制台命令，入口均在 `src/lerobot/scripts/`。

安装后可用：

```bash
uv sync --locked --extra core_scripts  # 录制/回放等
lerobot-info                           # 验证安装
```

---

## 2. 命令总表

| 命令 | 配置类 / 解析 | 主要用途 |
|------|---------------|----------|
| `lerobot-info` | 无 | 版本、依赖、设备信息 |
| `lerobot-train` | `TrainPipelineConfig` | 策略/奖励模型训练 |
| `lerobot-train-tokenizer` | `TokenizerTrainingConfig` | FAST action tokenizer |
| `lerobot-eval` | `EvalPipelineConfig` | 仿真 benchmark 评估 |
| `lerobot-rollout` | `RolloutConfig` | 真机策略部署 |
| `lerobot-record` | `RecordConfig` | 遥操作数据采集 |
| `lerobot-replay` | `ReplayConfig` | 数据集动作回放 |
| `lerobot-teleoperate` | `TeleoperateConfig` | 纯遥操作测试 |
| `lerobot-calibrate` | `CalibrateConfig` | 电机/设备标定 |
| `lerobot-setup-motors` | `SetupConfig` | 配置 motor ID |
| `lerobot-setup-can` | `CANSetupConfig` | CAN 总线 setup |
| `lerobot-find-port` | argparse | 列出串口 |
| `lerobot-find-cameras` | argparse | 发现相机 |
| `lerobot-find-joint-limits` | `FindJointLimitsConfig` | 探测关节限位 |
| `lerobot-dataset-viz` | argparse | Rerun 可视化数据集 |
| `lerobot-imgtransform-viz` | `DatasetConfig`+transforms | 可视化图像增强 |
| `lerobot-edit-dataset` | `EditDatasetConfig` | 数据集 CRUD/合并 |
| `lerobot-annotate` | `AnnotationPipelineConfig` | VLM 自动语言标注 |

### 未注册为 script 的模块（`python -m`）

| 模块 | 用途 |
|------|------|
| `lerobot.scripts.convert_dataset_v21_to_v30` | v2.1→v3 迁移 |
| `lerobot.scripts.augment_dataset_quantile_stats` | 补充分位数 stats |

---

## 3. 数据与训练

### lerobot-record

**作用**：teleop → robot → LeRobotDataset

**关键参数**：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --dataset.repo_id=user/my_dataset \
  --dataset.num_episodes=50 \
  --dataset.single_task="describe task" \
  --dataset.fps=30 \
  --dataset.push_to_hub=true
```

**主流程**：connect → `record_loop` @ fps → `save_episode` → optional Hub push

**依赖 extra**：`core_scripts`, 对应硬件 extra

---

### lerobot-replay

**作用**：读取 dataset action → robot

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --dataset.repo_id=user/my_dataset \
  --dataset.episode=0
```

---

### lerobot-edit-dataset

**作用**：多态数据集操作

```bash
# 合并
lerobot-edit-dataset \
  --operation.type=merge \
  --operation.repo_ids='["user/ds1","user/ds2"]' \
  --operation.output_repo_id=user/merged

# 重编码视频
lerobot-edit-dataset \
  --operation.type=reencode_videos \
  --repo_id=user/my_dataset
```

**operation.type**：`delete_episodes`, `split`, `merge`, `remove_feature`, `modify_tasks`, `convert_image_to_video`, `recompute_stats`, `reencode_videos`, `info`

---

### lerobot-annotate

**作用**：OpenAI 兼容 VLM 服务器为帧/episode 生成语言标注

**extra**：`lerobot[annotations]`

---

### lerobot-train

见 [06-training-evaluation.md](./06-training-evaluation.md)

---

### lerobot-eval

见 [06](./06-training-evaluation.md) / [08-environments.md](./08-environments.md)

---

## 4. 部署与控制

### lerobot-rollout

见 [09-rollout-inference.md](./09-rollout-inference.md)

---

### lerobot-teleoperate

与 record 相同硬件 wiring，**不写 dataset**：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --teleop.type=so101_leader \
  ...
```

---

## 5. 硬件 setup

### lerobot-calibrate

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1
```

### lerobot-setup-motors

配置 Feetech/Dynamixel ID 与波特率。

### lerobot-setup-can

Damiao/Robstride CAN 接口配置。

### lerobot-find-port / find-cameras / find-joint-limits

诊断工具，无 draccus 大配置。

---

## 6. 可视化

### lerobot-dataset-viz

```bash
lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0
```

**extra**：`dataset_viz`（dataset + rerun）

### lerobot-imgtransform-viz

预览 `transforms/` 中图像增强对 dataset 的影响。

---

## 7. 信息与诊断

### lerobot-info

```bash
lerobot-info
```

输出 Python、torch、lerobot 版本、CUDA 可用性等（最小依赖即可运行）。

---

## 8. CLI 通用模式

### 8.1 类型选择

```bash
--{field}.type={registered_name}
```

例：`--policy.type=act`, `--env.type=libero`, `--strategy.type=dagger`

### 8.2 预训练路径

```bash
--policy.path=lerobot/act_aloha
```

自动加载 config.json 并合并后续 CLI override。

### 8.3 插件发现

```bash
--env.discover_packages_path=my_plugin_pkg
```

### 8.4 YAML 配置

```bash
lerobot-train --config_path=train_config.yaml
```

---

## 9. Extras 与命令对照

| 命令 | 常见 extras |
|------|-------------|
| record/replay/teleop | `core_scripts`, 硬件 extra |
| train | `training`, policy extra |
| eval | `evaluation`, env extra, policy extra |
| rollout | `core_scripts`, policy extra |
| dataset-viz | `dataset_viz` |
| annotate | `annotations` |
| async 部署 | `async`, policy extra |

---

## 10. 示例：从零到部署命令链

```bash
# 1. 环境
uv sync --locked --extra core_scripts --extra feetech --extra training

# 2. 录数据
lerobot-record --robot.type=so101_follower ... --dataset.repo_id=user/demo

# 3. 训练
lerobot-train --policy.type=act --dataset.repo_id=user/demo --steps=50000

# 4. 仿真验证（若有 sim）
lerobot-eval --policy.path=outputs/.../pretrained_model --env.type=pusht --eval.n_episodes=10

# 5. 真机
lerobot-rollout --robot.type=so101_follower --policy.path=outputs/.../pretrained_model --strategy.type=base
```

---

## 下一章

- 奖励模型、RL、工具 → [11-advanced-modules.md](./11-advanced-modules.md)
