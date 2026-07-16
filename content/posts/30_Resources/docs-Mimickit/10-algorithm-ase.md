---
title: "第十章：ASE 算法"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第十章：ASE 算法。"
summary: "第十章：ASE 算法。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 10
series_weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第十章：ASE 算法

## 10.1 算法概述

**Adversarial Skill Embeddings (ASE)**（Peng et al., TOG 2022）在 AMP 基础上引入**可复用潜变量**$z$：策略$\pi(a|s,z)$条件于技能嵌入，编码器从运动特征预测$z$，实现大规模可复用技能库。

- 项目页：https://xbpeng.github.io/projects/ASE/
- 论文：ACM TOG 2022
- 开源参考：[ASE (GitHub)](https://github.com/nv-tlabs/ASE)

### 10.1.1 解决什么问题

AMP 策略不含显式技能参数，难以**切换/组合**技能。ASE 将技能编码为单位球面向量$z \in \mathbb{R}^{d}, \|z\|=1$，可：

- 训练时随机采样 z 覆盖技能空间
- 下游任务通过指定 z 复用技能
- 通过多样性损失确保不同 z 产生不同动作

### 10.1.2 适用场景

- 多技能数据集（剑盾战斗、locomotion 变体）
- 需要技能嵌入供下游任务使用

---

## 10.2 数学原理

### 10.2.1 潜变量采样

$$
z \sim \mathcal{N}(0, I), \quad z \leftarrow z / \|z\|
$$

每$\Delta t \in [T_{min}, T_{max}]$秒重新采样（`latent_time_min/max`）。

### 10.2.2 条件策略

$$
\pi_\theta(a | s, z), \quad V_\phi(s, z)
$$

Actor/Critic 输入拼接$[s, z]$。

### 10.2.3 编码器

$$
\hat{z} = g_\xi(d_{disc})
$$

**编码器损失（对比式）：**

$$
L_{enc} = -\mathbb{E}\left[\sum_k z_k \cdot \hat{z}_k\right] = \mathbb{E}[\text{err}]
$$

其中$err = -\sum_k z_k \hat{z}_k$（负点积，最大化对齐）。

**编码器奖励：**

$$
r_{enc} = \max(0, -err)
$$

### 10.2.4 总奖励

$$
r = w_{task} r_{task} + w_{disc} r_{disc} + w_{enc} r_{enc}
$$

### 10.2.5 多样性损失

采样新$z'$，计算：

$$
\Delta_a = \|\mu(s,z) - \mu(s,z')\|^2, \quad \Delta_z = 0.5 - 0.5 \cdot z \cdot z'
$$

$$
L_{div} = (\rho_{target} - \Delta_a / (\Delta_z + \epsilon))^2
$$

鼓励：**z 差异大时动作差异也应大**（`diversity_tar` =$\rho_{target}$）。

### 10.2.6 与 VAE/InfoGAN 的对比

ASE 不使用 ELBO，而是通过对抗奖励 + 对比编码器 + 多样性约束学习潜空间。参考：

- [DIAYN (Eysenbach et al., 2018)](https://arxiv.org/abs/1802.06059) — 无监督技能发现
- [PaCo (2021)](https://arxiv.org/abs/2108.06026) — 参数化动作空间

---

## 10.3 代码实现：`ASEAgent`

继承 `AMPAgent`。

### 10.3.1 超参数

| 参数 | 说明 |
|------|------|
| `latent_time_min/max` | z 重采样间隔范围（秒） |
| `diversity_weight` | 多样性损失权重 |
| `diversity_tar` | 目标多样性比率 |
| `enc_epochs` / `enc_batch_size` | 编码器训练 |
| `enc_reward_weight` | 编码器奖励权重 |

### 10.3.2 方法清单

| 方法 | 说明 |
|------|------|
| `_build_model` | `ASEModel`（+ enc_net） |
| `_build_latent_buf` | 每环境维护 z 和重采样时间 |
| `_reset_latents(env_ids)` | 重采样 z |
| `_update_latents()` | 检查时间是否到期 |
| `_decide_action` | actor 输入含 z |
| `_record_data_pre_step` | 记录 latents |
| `_build_train_data` | critic 输入含 z |
| `_compute_rewards` | + enc_reward |
| `_calc_enc_rewards` | 编码器奖励 |
| `_update_enc` | 编码器训练 |
| `_compute_enc_loss` | 编码器损失 |
| `_calc_enc_error` | 负点积误差 |
| `_compute_actor_loss` | + diversity_loss |
| `_compute_diversity_loss` | 多样性约束 |
| `_compute_critic_loss` | 输入含 z |
| `_sample_latents(n)` | 归一化高斯采样 |

---

## 10.4 环境：`ASEEnv`

继承 `AMPEnv`，增加 `default_reset_prob`：以一定概率从默认姿态重置。

---

## 10.5 训练命令

```bash
python mimickit/run.py --arg_file args/ase_humanoid_args.txt \
  --env_config data/envs/ase_humanoid_sword_shield_env.yaml \
  --out_dir output/
```

---

## 10.6 调参建议

| 参数 | 建议 |
|------|------|
| `diversity_weight` | 过小技能退化，过大动作不稳定 |
| `enc_reward_weight` | 控制 z 与运动的绑定强度 |
| `latent_dim`（model 配置） | 技能复杂度上界 |
| `latent_time_min/max` | 控制每个 z 持续时间 |

---

## 10.7 引用

```bibtex
@article{2022-TOG-ASE,
  author = {Peng, Xue Bin and Guo, Yunrong and Halper, Lina and Levine, Sergey and Fidler, Sanja},
  title = {ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters},
  journal = {ACM Trans. Graph.},
  year = {2022}
}
```

---

[← AMP](09-algorithm-amp.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：ADD →](11-algorithm-add.md)
