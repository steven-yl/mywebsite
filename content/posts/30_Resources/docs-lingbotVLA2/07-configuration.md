---
title: "7. 配置参考"
subtitle: ""
date: 2026-07-13T14:00:00+08:00
draft: false
authors: [Steven]
description: "本章汇总 LingBot-VLA 2.0 的 YAML 配置体系、真机与仿真差异、以及关键参数速查。"
summary: "本章汇总 LingBot-VLA 2.0 的 YAML 配置体系、真机与仿真差异、以及关键参数速查。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA2, robots]
series: [lingbotVLA2-docs]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 7. 配置参考

本章汇总 LingBot-VLA 2.0 的 YAML 配置体系、真机与仿真差异、以及关键参数速查。

> 完整字段表见 [config/lingbotvla_config_doc.md](./config/lingbotvla_config_doc.md)  
> 实操指南见 [configs/vla/Training_Config.md](../configs/vla/Training_Config.md)

---

## 7.1 配置文件层次

```
configs/
├── vla/
│   ├── robotwin/robotwin.yaml      # RoboTwin 50 任务 post-training
│   ├── real_robot/real_robot.yaml  # 真机微调模板
│   └── norm_compute/post_data.yaml # 归一化统计计算
└── robot_configs/
    ├── robotwin.yaml               # 仿真特征映射
    └── agilex_cobot_magic.yaml     # 真机示例
```

命令行可覆盖任意 YAML 字段：

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml \
  --train.lr 1e-4 \
  --train.max_steps 20000
```

---

## 7.2 真机 vs 仿真

| 配置项 | 真机 (`real_robot.yaml`) | 仿真 (`robotwin.yaml`) |
|--------|--------------------------|------------------------|
| `data.norm_type` | 各关节 `meanstd` | 各关节 `bounds_99_woclip` |
| `train.loss_type` | `fm`（MSE） | `L1_fm`（L1） |
| `action.arm.position` subtract_state | **True**（推荐） | False |
| `data.prompt_type` | 按任务 | `global` |
| 相机数量 | 常 3（top + 双腕） | 同左 |
| 关节子集 | 可能含 waist/head/base/hand | 通常 arm + effector |

---

## 7.3 Model 配置

```yaml
model:
  model_path: /path/to/lingbot-vla-v2-6b/hf_ckpt
  tokenizer_path: /path/to/Qwen3-VL-4B-Instruct
  post_training: true
  adanorm_time: true
  config_key: LingbotVLAV2Config    # 必须
  moe_implementation: fused         # fused MoE 内核
```

| 参数 | 说明 |
|------|------|
| `config_key` | `LingbotVLAV2Config`（v2）或 `LingbotVLAConfig`（v1） |
| `moe_implementation` | `fused` / `eager` / `None` |
| `freeze_vision_encoder` | 冻结 ViT（在 `train` 段也可配置） |
| `vit_attn_implementation` | 默认 `flash_attention_2` |

---

## 7.4 Data 配置

```yaml
data:
  datasets_type: vla
  data_name: multi
  train_path: assets/training_data/robotwin.txt
  robot_config_root: ./configs/robot_configs
  joints:
    - arm.position: 14
    - end.position: 14
    - effector.position: 2
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  norm_type:
    - arm.position: bounds_99_woclip
  norm_stats_file: assets/norm_stats/robotwin.json
  prompt_type: global
  use_future_image: true
  num_workers: 8
  img_size: 256
```

---

## 7.5 Train 核心配置

### 7.5.1 动作与模型结构

```yaml
train:
  action_dim: 55
  max_action_dim: 55
  max_state_dim: 55
  chunk_size: 50              # 等价 n_action_steps
  num_steps: 10               # 推理去噪步数
  tokenizer_max_length: 72
  attention_implementation: flex_cached
  vlm_causal: true
```

### 7.5.2 MoE

```yaml
train:
  use_moe: true
  token_moe_layers: [0,1,...,35]
  token_num_experts: 32
  token_top_k: 4
  token_moe_intermediate_size: 512
  token_shared_intermediate_size: 704
  bias_update_speed: 0.00025
  router_activation: sigmoid
  routed_scaling_factor: 4.0
  use_shared_expert_gate: false
  use_moe_expert_lr: true
  router_z_loss_coeff: 1e-4
  sequence_wise_loss_coeff: 1e-3
  sequence_wise_mode: per_sequence
```

### 7.5.3 分布式与精度

```yaml
train:
  data_parallel_mode: fsdp2
  module_fsdp_enable: true
  vlm_fsdp: true
  enable_gradient_checkpointing: true
  enable_fp32: true
  use_compile: true
  precompute_grid_thw: true
  rmpad: false
```

### 7.5.4 优化与批大小

```yaml
train:
  optimizer: muon             # 或 adamw
  lr: 1.0e-4
  lr_decay_style: constant
  micro_batch_size: 1
  global_batch_size: 3
  gradient_accumulation_steps: 1
  max_steps: 30000
  num_train_epochs: 29000
  max_grad_norm: 1.0
```

### 7.5.5 检查点

```yaml
train:
  output_dir: /path/to/save
  ckpt_manager: dcp
  save_steps: 30000
  enable_resume: true
```

---

## 7.6 align_params（原生深度蒸馏）

完整示例见 `Training_Config.md`。核心字段：

```yaml
train:
  align_params:
    mode: query
    num_task_tokens: 8
    depth_loss_weight: 0.004
    future_depth_loss_weight: 0.004
    use_future_video: true
    visual_steps: 5000
    depth:
      model_type: MoRGBD
      moge_path: /path/to/moge2-vitb-normal.pt
      morgbd_path: /path/to/morgbd_v2_mixdata.pt
      use_future_depth: true
      block_future_depth_to_action: true
    video:
      ckpt_path: /path/to/teacher_step_10000.pth
      config_path: /path/to/config.yaml
      future_video_loss_weight: 0.004
      use_patch_loss: true
      use_current_patch_loss: true
      block_suffix_to_future_video: true
```

**禁用蒸馏**：删除或注释整个 `align_params` 块。

---

## 7.7 参数解析源码

| 文件 | 内容 |
|------|------|
| `lingbotvla/utils/arguments.py` | `ModelArguments`, `DataArguments`, `TrainingArguments` dataclass |
| `tasks/vla/train_lingbotvla.py` | `MyTrainingArguments` 扩展字段 |
| `lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py` | 模型 Config 类 |

---

## 7.8 快速配置模板

### 最小 Post-training（无蒸馏）

```yaml
model:
  model_path: path/to/ckpt
  tokenizer_path: path/to/Qwen3-VL-4B-Instruct
  config_key: LingbotVLAV2Config
  post_training: true

data:
  datasets_type: vla
  data_name: my_robot
  train_path: path/to/lerobot
  robot_config_root: configs/robot_configs

train:
  output_dir: output/
  action_dim: 55
  max_action_dim: 55
  max_state_dim: 55
  max_steps: 10000
  micro_batch_size: 1
  data_parallel_mode: fsdp2
  enable_gradient_checkpointing: true
```

### 梯度累积（小显存）

```yaml
train:
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  # global_batch_size 自动 = 1 * num_gpus * 4
```

---

## 7.9 环境变量

| 变量 | 用途 |
|------|------|
| `QWEN3VL_PATH` | 部署时 Qwen3-VL 路径 |
| `QWEN3_PATH` | 开环评估 |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU |
| `WANDB_API_KEY` | WandB 日志 |

---

## 7.10 章节关系

| 主题 | 章节 |
|------|------|
| 数据字段含义 | [04-data-pipeline.md](./04-data-pipeline.md) |
| MoE 原理 | [01-model-architecture.md](./01-model-architecture.md) |
| 蒸馏参数含义 | [03-dual-query-distillation.md](./03-dual-query-distillation.md) |
| 训练启动 | [05-training-system.md](./05-training-system.md) |
