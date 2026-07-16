---
title: "第十四章：工具链与数据管线"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第十四章：工具链与数据管线。"
summary: "第十四章：工具链与数据管线。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 14
series_weight: 14
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第十四章：工具链与数据管线

## 14.1 概述

`tools/` 目录提供数据转换、扩散先验训练、日志可视化等辅助工具，不参与主训练循环（`run.py`），但在完整工作流中不可或缺。

```
tools/
├── gmr_to_mimickit/       # GMR 动作 → MimicKit .pkl
├── smpl_to_mimickit/      # AMASS SMPL → MimicKit .pkl
├── diffusion_model/       # TinyMDM 先验训练（SMP）
├── plot_log/              # 训练曲线绘制
└── util/                  # 可视化辅助
```

---

## 14.2 动作数据转换

### 14.2.1 GMR → MimicKit

[GMR (General Motion Retargeting)](https://github.com/YanjieZe/GMR) 将人体动作重定向到机器人。

```bash
# 详见 tools/gmr_to_mimickit/README.md
python tools/gmr_to_mimickit/gmr_to_mimickit.py --input <gmr_file> --output <output.pkl>
```

输出格式符合 [03-animation-module](03-animation-module.md) 定义的 `.pkl` 结构。

### 14.2.2 AMASS SMPL → MimicKit

[AMASS](https://amass.is.tue.mpg.de) 提供大规模 SMPL 人体动作。

```bash
# 详见 tools/smpl_to_mimickit/README.md
python tools/smpl_to_mimickit/smpl_to_mimickit.py --input <amass.npz> --output <output.pkl> --xml <smpl.xml>
```

### 14.2.3 为什么需要转换

MimicKit 要求动作帧向量与**目标角色运动学树**的关节顺序一致（深度优先遍历）。外部数据必须经过重定向/重排列。

---

## 14.3 扩散先验训练（SMP）

### 14.3.1 入口

```bash
python tools/diffusion_model/train_tinymdm.py \
  --cfg_path tools/diffusion_model/config/tinymdm_multi_clip.yaml \
  --out_dir output/smp_prior
```

### 14.3.2 配置文件

| 文件 | 用途 |
|------|------|
| `tinymdm_multi_clip.yaml` | 多片段数据集先验 |
| `tinymdm_single_clip.yaml` | 单片段先验 |

关键字段：

```yaml
motion_file: "data/datasets/dataset_humanoid_locomotion.yaml"
env_config: "data/envs/amp_humanoid_env.yaml"  # 定义 disc_obs 维度
T: 50                    # 扩散步数
arch_name: "DiT"         # DiT / CondDiT
estimate_mode: "epsilon" # 预测类型
noise_schedule_mode: "squaredcos_cap_v2"
control_freq: 30
```

### 14.3.3 `motion_prior_dataset.py`

- 从 `MotionLib` 加载动作
- 构建 `disc_obs` 格式训练样本
- 与 RL 环境使用相同的观测构造函数，保证分布一致

### 14.3.4 输出

```
output/smp_prior/
├── model.pt                 # 先验权重
├── diffusion_config.yaml    # 推理配置（供 SMPAgent 加载）
└── log.txt                  # 训练日志
```

### 14.3.5 训练后使用

在 `smp_task_humanoid_agent.yaml` 中设置：

```yaml
smp_prior_cfg: "output/smp_prior/diffusion_config.yaml"
smp_prior_model: "output/smp_prior/model.pt"
```

---

## 14.4 日志可视化

### 14.4.1 TensorBoard

```bash
tensorboard --logdir=output/ --port=6006 --samples_per_plugin scalars=999999
```

### 14.4.2 plot_log.py

```bash
python tools/plot_log/plot_log.py --log_file output/log.txt
```

解析 `log.txt` 中的 `Train_Return`、`Test_Return`、损失等指标绘制曲线。

### 14.4.3 日志集合（Logger collections）

| Collection | 内容 |
|------------|------|
| `0_Main` | Test_Return, Train_Return |
| `1_Info` | Samples, Wall_Time, Iteration |
| `2_Env` | 跟踪误差等环境诊断 |

---

## 14.5 可视化工具

### 14.5.1 `char_vis_util.py`

角色可视化辅助函数，用于调试关节/身体位置。

### 14.5.2 `plot_util.py`

通用绘图工具。

### 14.5.3 `view_motion` 环境

无需训练即可预览动作：

```bash
python mimickit/run.py --mode test \
  --arg_file args/view_motion_humanoid_args.txt --visualize true
```

---

## 14.6 数据集配置

### 14.6.1 内置数据集

| 文件 | 内容 |
|------|------|
| `dataset_humanoid_locomotion.yaml` | 人形行走/跑步 |
| `dataset_humanoid_sword_shield.yaml` | 剑盾战斗 |
| `dataset_humanoid_sword_shield_locomotion.yaml` | 混合 |
| `dataset_go2_locomotion.yaml` | Go2 四足 |

### 14.6.2 自定义数据集

```yaml
motions:
  - file: "data/motions/my_character/walk.pkl"
    weight: 2.0
  - file: "data/motions/my_character/run.pkl"
    weight: 1.0
```

在 env YAML 中设置 `motion_file: "data/datasets/my_dataset.yaml"`。

---

## 14.7 预训练模型

`data/models/` 提供预训练检查点（需从 SharePoint 下载）：

| 模型 | 算法 | 技能 |
|------|------|------|
| `deepmimic_humanoid_spinkick_model.pt` | PPO | Spin kick |
| `amp_humanoid_spinkick_model.pt` | AMP | Spin kick |
| `amp_location_humanoid_model.pt` | AMP | 行走至目标 |
| `ase_humanoid_sword_shield_model.pt` | ASE | 剑盾 |
| `lcp_g1_walk_model.pt` | LCP | G1 行走 |
| `smp_location_humanoid_model.pt` | SMP | 任务导航 |

对应训练日志在 `data/logs/`。

---

## 14.8 完整工作流示例

### 14.8.1 从零训练 DeepMimic

```
1. 下载 data/ 资产包
2. 准备/转换 .pkl 动作
3. 配置 env YAML（motion_file, reward 权重）
4. python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt
5. tensorboard / plot_log 监控
6. --mode test 验证
```

### 14.8.2 SMP 任务策略（无动作数据）

```
1. 使用预训练 LaFAN1 先验（或自训 TinyMDM）
2. 配置 smp_task_humanoid_agent.yaml（enable_gsi: true）
3. python mimickit/run.py --arg_file args/smp_location_humanoid_args.txt
4. 调 smp_reward_weight / sds_loss_scale
```

### 14.8.3 新机器人（G1）训练

```
1. 确认 URDF 在 data/assets/
2. URDF 关节限位正确（limited=true, 非零 bounds）
3. GMR 重定向动作 → .pkl
4. 选择算法（DeepMimic/ADD/LCP）
5. 复制并修改 g1 env/agent YAML
```

---

## 14.9 外部工具链接

| 工具 | 用途 | 链接 |
|------|------|------|
| GMR | 动作重定向 | https://github.com/YanjieZe/GMR |
| AMASS | 人体动作数据 | https://amass.is.tue.mpg.de |
| LaFAN1 | 动作数据集 | https://github.com/ubisoft/ubisoft-laforge-animation-dataset |
| HuggingFace Diffusers | 扩散调度器 | https://github.com/huggingface/diffusers |
| TensorBoard | 训练监控 | https://www.tensorflow.org/tensorboard |

---

[← SMP](13-algorithm-smp.md) | [返回索引](TECHNICAL_INDEX.md)
