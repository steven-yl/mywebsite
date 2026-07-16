---
title: "第九章：AMP 算法"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第九章：AMP 算法。"
summary: "第九章：AMP 算法。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 9
series_weight: 9
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第九章：AMP 算法

## 9.1 算法概述

**Adversarial Motion Priors (AMP)**（Peng et al., TOG 2021）用对抗判别器替代逐帧跟踪奖励：判别器区分**仿真生成运动**与**参考数据集运动**，策略获得"骗过判别器"的奖励，从而学习数据集风格。

- 项目页：https://xbpeng.github.io/projects/AMP/
- 论文：https://doi.org/10.1145/3450626.3459670
- 开源参考：[AMP_for_hardware](https://github.com/escontra/AMP_for_hardware)

### 9.1.1 为什么需要 AMP

DeepMimic 需精确参考轨迹，难以泛化到**多样风格**或**任务+风格**组合。AMP 只需无标签动作数据集，策略可自由执行任务同时保持运动风格。

### 9.1.2 适用场景

- 大规模 locomotion 数据集（LaFAN1、自定义 YAML 数据集）
- 纯模仿（`task_reward_weight=0`）或任务+模仿混合

---

## 9.2 数学原理

### 9.2.1 判别器

输入：$d \in \mathbb{R}^{N_{steps} \cdot D_{disc}}$（多步运动特征展平）

$$
D_\psi(d) \in \mathbb{R}
$$

**训练目标（LSGAN 风格 BCE）：**

$$
L_D = \frac{1}{2}\mathbb{E}_{d \sim \pi}[\text{BCE}(D(d), 0)] + \frac{1}{2}\mathbb{E}_{d \sim demo}[\text{BCE}(D(d), 1)]
$$

**梯度惩罚（WGAN-GP 风格）：**

$$
L_{GP} = \mathbb{E}\left[\|\nabla_d D(d)\|^2\right]
$$

对 agent 和 demo 样本分别计算。

**Logit 正则：**

$$
L_{reg} = \|\psi_{last}\|^2
$$

### 9.2.2 策略奖励

$$
p = \sigma(D(d_{agent})), \quad r_{disc} = -\log(1 - p + \epsilon) \cdot s_{disc}
$$

其中$s_{disc}$为 `disc_reward_scale`。与 GAIL 奖励形式一致。

### 9.2.3 总奖励

$$
r = w_{task} \cdot r_{task} + w_{disc} \cdot r_{disc}
$$

纯模仿时$w_{task} = 0$；任务训练时两者加权（如 location 任务）。

### 9.2.4 与 GAIL 的关系

AMP 可视为 GAIL 在运动控制领域的特化：

- 判别器输入为**物理特征**（关节旋转、速度等）而非原始状态
- 使用 PPO 而非 TRPO 优化策略
- 参考：[GAIL (Ho & Ermon, 2016)](https://arxiv.org/abs/1606.03476)

---

## 9.3 代码实现：`AMPAgent`

继承 `PPOAgent`，扩展判别器训练。

### 9.3.1 超参数

| 参数 | 说明 |
|------|------|
| `disc_epochs` | 判别器更新 epoch |
| `disc_batch_size` | 批大小 |
| `disc_replay_samples` | replay buffer 采样数 |
| `disc_logit_reg` | logit 正则权重 |
| `disc_grad_penalty` | 梯度惩罚权重 |
| `disc_reward_scale` | 奖励缩放 |
| `disc_buffer_size` | agent 观测 replay 容量 |
| `task_reward_weight` | 任务奖励权重 |
| `disc_reward_weight` | 判别器奖励权重 |

### 9.3.2 方法清单

| 方法 | 说明 |
|------|------|
| `_build_model` | `AMPModel`（+ disc_net） |
| `_build_optimizer` | + `disc_optimizer` |
| `_build_exp_buffer` | + `_disc_buffer` replay |
| `_build_normalizers` | + `_disc_obs_norm` |
| `_record_data_post_step` | 记录 `disc_obs` |
| `_build_train_data` | demo 采样 + replay + 奖励计算 |
| `_record_disc_demo_data` | 从 env 获取 demo 观测 |
| `_store_disc_replay_data` | 存入 replay |
| `_compute_rewards` | 混合 task + disc 奖励 |
| `_update_model` | PPO + 判别器更新 |
| `_update_disc` | 判别器训练循环 |
| `_compute_disc_loss` | BCE + GP + reg |
| `_disc_loss_neg` / `_disc_loss_pos` | BCE 损失 |
| `_compute_disc_acc` | 分类准确率 |
| `_calc_disc_rewards` | 计算 disc 奖励 |

### 9.3.3 训练数据流

```
rollout → 记录 disc_obs
    → fetch_disc_obs_demo(n) 从 MotionLib 采样
    → store replay buffer
    → calc_disc_rewards → 覆盖 reward
    → PPO build_train_data (TD(λ))
    → update critic/actor/disc
```

---

## 9.4 环境：`AMPEnv`

- `_update_reward()` 为空（奖励在 Agent 端计算）
- `info["disc_obs"]` 每步输出
- 详见 [05-environments](05-environments.md) 5.6 节

---

## 9.5 配置示例

**纯模仿：**

```yaml
# amp_humanoid_agent.yaml
task_reward_weight: 0.0
disc_reward_weight: 1.0
```

**任务+模仿：**

```yaml
# amp_task_humanoid_agent.yaml
task_reward_weight: 0.5
disc_reward_weight: 0.5
```

---

## 9.6 训练命令

```bash
# 纯模仿
python mimickit/run.py --arg_file args/amp_humanoid_args.txt --out_dir output/

# 任务+模仿（location）
python mimickit/run.py --arg_file args/amp_location_humanoid_args.txt --out_dir output/
```

---

## 9.7 调参建议

| 参数 | 建议 |
|------|------|
| `disc_reward_scale` | 控制模仿信号强度，典型 2.0 |
| `disc_grad_penalty` | 稳定训练，典型 5.0 |
| `disc_reward_weight` vs `task_reward_weight` | 平衡风格与任务完成 |
| `num_disc_obs_steps` | 更多步捕获时序特征，但维数增大 |

---

## 9.8 引用

```bibtex
@article{2021-TOG-AMP,
  author = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
  title = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
  journal = {ACM Trans. Graph.},
  year = {2021},
  doi = {10.1145/3450626.3459670}
}
```

---

[← AWR](08-algorithm-awr.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：ASE →](10-algorithm-ase.md)
