---
title: "8. 配置参考"
subtitle: ""
date: 2026-07-13T12:00:00+08:00
draft: false
authors: [Steven]
description: "本文整合 configs/vla/、configs/robotconfigs/ 与 utils/arguments.py 中的参数说明。"
summary: "本文整合 configs/vla/、configs/robotconfigs/ 与 utils/arguments.py 中的参数说明。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA, robots]
series: [lingbotVLA-docs]
weight: 8
series_weight: 8
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 8. 配置参考

本文整合 `configs/vla/`、`configs/robot_configs/` 与 `utils/arguments.py` 中的参数说明。

完整训练指南另见：[configs/vla/Training_Config.md](../../configs/vla/Training_Config.md)

---

## 8.1 配置文件一览

| 文件 | 场景 |
|------|------|
| `robotwin_load20000h.yaml` | RoboTwin 仿真，无 depth |
| `robotwin_load20000h_depth.yaml` | RoboTwin + depth |
| `real_load20000h.yaml` | 真实机器人模板 |
| `real_load20000h_depth.yaml` | 真实机器人 + depth |
| `robot_configs/robotwin.yaml` | RoboTwin 特征映射 |

---

## 8.2 ModelArguments (`model.*`)

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_path` | str | 预训练权重 HF 路径或本地目录 |
| `config_path` | str | 可选，覆盖 config.json |
| `tokenizer_path` | str | Qwen2.5-VL tokenizer/processor |
| `vocab_size` | int | 自动推断（Qwen: 151936） |
| `use_lm_head` | bool | 是否保留 VLM lm_head |
| `post_training` | bool | 严格权重匹配模式 |
| `force_use_huggingface` | bool | 强制 HF 加载 |

---

## 8.3 DataArguments (`data.*`)

| 参数 | 类型 | 说明 |
|------|------|------|
| `datasets_type` | str | 必须为 `vla` |
| `train_path` | str | LeRobot v3.0 数据集路径 |
| `data_name` | str | robot config 文件名（无 .yaml） |
| `robot_config_root` | str | 默认 `configs/robot_configs` |
| `joints` | List | 如 `["arm.position: 14", "effector.position: 2"]` |
| `cameras` | List[str] | 统一相机名 |
| `norm_type` | str | `meanstd` / `bounds_99` / `minmax` / `identity` |
| `norm_stats_file` | str | 预计算 JSON |
| `num_workers` | int | DataLoader workers |
| `drop_last`, `pin_memory`, `prefetch_factor` | — | 标准 DataLoader |

---

## 8.4 TrainingArguments (`train.*`) — 核心

### 动作与模型

| 参数 | 默认 | 说明 |
|------|------|------|
| `chunk_size` | 50 | 动作预测 horizon → `n_action_steps` |
| `max_action_dim` | 75 | action padding 维 |
| `max_state_dim` | 75 | state padding 维 |
| `tokenizer_max_length` | 72 | 语言 token 上限 |
| `num_steps` | 10 | 推理 FM 步数 |
| `loss_type` | `fm` | `fm`=MSE, `L1_fm`=L1 |
| `attention_implementation` | flex | `flex` / `eager` |

### 冻结与精度

| 参数 | 说明 |
|------|------|
| `freeze_vision_encoder` | 冻结 Qwen ViT |
| `train_expert_only` | 冻结整个 VLM，只训 expert+head |
| `train_state_proj` | 必须为 true |
| `enable_fp32` | action expert 用 fp32 |
| `enable_mixed_precision` | 全局混合精度 |
| `use_compile` | torch.compile |

### 批大小

$$
\text{global\_batch} = \text{micro\_batch\_size} \times \text{num\_gpus} \times \text{gradient\_accumulation\_steps}
$$

| 参数 | 说明 |
|------|------|
| `micro_batch_size` | 每 GPU 每步样本数 |
| `gradient_accumulation_steps` | 梯度累积 |
| `global_batch_size` | 可选显式指定（需与上式一致） |

### 训练时长

| 参数 | 说明 |
|------|------|
| `max_steps` | 最大更新步数 |
| `num_train_epochs` | epoch 上限；与 max_steps 取先达到者 |
| `save_steps` | checkpoint 间隔 |
| `save_epochs` | 0=禁用 epoch 保存 |

### 分布式

| 参数 | 默认 | 说明 |
|------|------|------|
| `data_parallel_mode` | ddp | `fsdp2` 推荐 |
| `enable_full_shard` | false | ZeRO-3 风格全分片 |
| `module_fsdp_enable` | true | 按模块 wrap |
| `ulysses_parallel_size` | 1 | Ulysses SP |
| `tensor_parallel_size` | 1 | TP |
| `expert_parallel_size` | 1 | MoE EP |
| `ckpt_manager` | dcp | `dcp` / `bytecheckpoint` |
| `enable_resume` | false | 自动 resume |
| `rmpad` | false | VLA 必须 false |

### 高级 VLA 选项

| 参数 | 说明 |
|------|------|
| `adanorm_time` | Expert AdaRMSNorm 时间条件 |
| `separate_time_proj` | 时间 MLP 与 action 分离 |
| `norm_qkv` | Attention 前 Q/K LayerNorm |
| `align_params` | 深度对齐 dict（空=关闭） |
| `vlm_causal` | prefix 是否 causal mask |

---

## 8.5 align_params 结构

```yaml
align_params:
  mode: 'query'                    # direct | query
  num_task_tokens: 8
  use_image_tokens: True
  use_task_tokens: False
  use_text_tokens: False
  use_contrastive: True
  contrastive_loss_weight: 0.3
  depth_loss_weight: 0.004
  llm:
    dim_out: 2048
    image_token_size: 8
    image_input_size: 224
  depth:
    model_type: MoRGBD
    moge_path: "Ruicheng/moge-2-vitb-normal"
    morgbd_path: "robbyant/lingbot-depth-pretrain-vitl-14"
    num_layers: 1
    num_heads: 4
    dim_head: 32
    ff_mult: 1
    num_backbone_tokens: 256
    token_size: 16
    dim_out: 1024
    input_size: 224
```

---

## 8.6 场景对比表

| 配置项 | RoboTwin 仿真 | 真实机器人 |
|--------|---------------|------------|
| `norm_type` | `bounds_99` | `meanstd` |
| `loss_type` | `L1_fm` | `fm` (MSE) |
| `lr` | 1e-4 | 5e-5 |
| `max_steps` | 20000 | 40000 |
| `subtract_state` (robot yaml) | False | arm True 推荐 |
| depth | 可选 | 推荐 w/ depth |

---

## 8.7 Robot Config 模板

```yaml
# configs/robot_configs/my_robot.yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state: {start: 0, end: 7}

actions:
  - action.arm.position:
      origin_keys:
        - action: {start: 0, end: 7}
      subtract_state: True    # 真实机推荐 delta joint

images:
  - observation.images.camera_top:
      origin_keys: observation.images.top_cam
  - observation.images.camera_wrist_left
```

对应 VLA yaml：

```yaml
data:
  data_name: my_robot
  joints:
    - arm.position: 14
  cameras:
    - camera_top
    - camera_wrist_left
```

---

## 8.8 CLI 覆盖示例

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin_load20000h.yaml \
    --data.train_path /data/robotwin \
    --data.norm_stats_file assets/norm_stats/robotwin_50.json \
    --train.output_dir output/exp1 \
    --train.micro_batch_size 8 \
    --train.gradient_accumulation_steps 4 \
    --train.max_steps 10000 \
    --train.lr 5e-5
```

训练开始时会保存完整参数到 `{output_dir}/lingbotvla_cli.yaml`。

---

## 8.9 预训练模型与数据

| 资源 | HuggingFace |
|------|-------------|
| LingBot-VLA-4B | [robbyant/lingbot-vla-4b](https://huggingface.co/robbyant/lingbot-vla-4b) |
| LingBot-VLA-4B-Depth | [robbyant/lingbot-vla-4b-depth](https://huggingface.co/robbyant/lingbot-vla-4b-depth) |
| Posttrain-Robotwin | [robbyant/lingbot-vla-4b-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-vla-4b-posttrain-robotwin) |
| Qwen2.5-VL-3B | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| GM-100 数据 | [robbyant/gm100](https://huggingface.co/datasets/robbyant/gm100) |

下载：

```bash
python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-4b --local_dir lingbot-vla-4b
```

---

## 8.10 依赖版本

| 组件 | 版本 |
|------|------|
| Python | 3.12.3 |
| PyTorch | 2.8.0 |
| CUDA | 12.8 |
| LeRobot | v3.0 |

安装：`bash install.sh`

---

## 8.11 引用

```bibtex
@article{wu2026pragmatic,
  title={A Pragmatic VLA Foundation Model},
  author={Wei Wu and Fan Lu and others},
  journal={arXiv preprint arXiv:2601.18692v1},
  year={2026}
}
```
