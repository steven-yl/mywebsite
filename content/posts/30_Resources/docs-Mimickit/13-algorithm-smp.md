---
title: "第十三章：SMP 算法"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第十三章：SMP 算法。"
summary: "第十三章：SMP 算法。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 13
series_weight: 13
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第十三章：SMP 算法

## 13.1 算法概述

**Score-Matching Motion Priors (SMP)**（Mu et al., SIGGRAPH 2026）用**预训练扩散模型**作为运动先验：策略获得 Score Distillation Sampling (SDS) 风格奖励，无需对抗判别器。支持 **GSI (Generative State Initialization)** 从先验采样初始状态。

- 项目页：https://xbpeng.github.io/projects/SMP/
- 两阶段流程：① 训练 TinyMDM 扩散先验 ② RL 策略优化

### 13.1.1 与 AMP/SMP 对比

| 方面 | AMP | SMP |
|------|-----|-----|
| 先验形式 | 在线判别器 | 离线扩散模型 |
| 先验训练 | 与策略联合 | **独立预训练** |
| 策略训练时需动作数据 | 是 | **否**（配合 GSI） |
| 奖励计算 | 判别器 logit | SDS 去噪误差 |
| 先验复用 | 不可分离 | **可复用**于多任务 |

### 13.1.2 适用场景

- 纯任务控制（location/steering/dodgeball）无需参考动作
- 需要可复用运动先验的多任务训练
- 单片段精确模仿（先验 + 策略联合）

---

## 13.2 数学原理

### 13.2.1 扩散模型（DDPM）

前向加噪：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)
$$

反向去噪网络$\epsilon_\theta(x_t, t)$预测噪声。

训练目标：

$$
L_{diff} = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

参考：[DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)，[Diffusers 库](https://github.com/huggingface/diffusers)

### 13.2.2 SDS 损失（ESM_SDS_loss）

对策略产生的运动特征$x$（`disc_obs`），在多个时间步$t \in \{t_1, t_2, ...\}$：

1. 加噪：$x_t = \sqrt{\bar{\alpha}_t} x + \sqrt{1-\bar{\alpha}_t} \epsilon$
2. 去噪预测：$\hat{x}_0 = f_\theta(x_t, t)$
3. 计算预测噪声：$\hat{\epsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}$
4. SDS 误差：$\|\hat{\epsilon} - \epsilon\|^2$

默认时间步：`[22, 15, 8]`（`diffusion_steps`）。

### 13.2.3 SMP 奖励

$$
\bar{L}_{SDS} = \text{mean}_t(\text{normalize}(L_{SDS,t}))
$$

$$
r_{SMP} = \exp(-\bar{L}_{SDS} \cdot s_{sds}) \cdot s_{smp}
$$

其中$s_{sds}$= `sds_loss_scale`，$s_{smp}$= `smp_reward_scale`。

### 13.2.4 总奖励

$$
r = w_{task} \cdot r_{task} + w_{smp} \cdot r_{SMP}
$$

### 13.2.5 GSI（Generative State Initialization）

每 `gsi_iters` 迭代，从冻结先验采样运动特征序列作为**初始状态池**：

$$
x_{init} \sim p_\theta(x)
$$

策略训练时从池中采样重置状态，无需参考动作数据。

---

## 13.3 TinyMDM 扩散先验

### 13.3.1 架构

| 组件 | 说明 |
|------|------|
| `TinyStableMotionDiTModel` | DiT 架构去噪网络 |
| `CondTinyStableMotionDiTModel` | 条件 DiT（多片段类别） |
| `DDPMScheduler` / `DDIMScheduler` | HuggingFace diffusers |
| `EMA` | 指数移动平均模型 |
| `ClassifierFreeSampleModel` | CFG 采样 |

### 13.3.2 `TinyMDMModel` 关键方法

| 方法 | 说明 |
|------|------|
| `forward(x, timesteps)` | 训练去噪 |
| `sample(shape, batch_size, sampler, num_inference_steps)` | DDPM 采样 |
| `sample_ema(...)` | EMA 模型采样 |
| `ESM_SDS_loss(norm_x_obs, t_lst)` | SDS 奖励计算 |
| `normalize(samples)` / `unnormalize(...)` | 观测归一化 |
| `update_normalizer(samples)` | 更新归一化统计 |

### 13.3.3 输入维度

```
input_dim = num_disc_obs_steps × per_step_features
input_channel = input_dim / num_disc_obs_steps
```

reshape 为 `[batch, num_steps, input_channel]` 送入 DiT。

---

## 13.4 代码实现：`SMPAgent`

继承 `PPOAgent`（**不**继承 AMP）。

### 13.4.1 超参数

| 参数 | 说明 |
|------|------|
| `smp_prior_cfg` | 先验配置文件路径 |
| `smp_prior_model` | 先验权重 `.pt` |
| `sds_loss_scale` | SDS 指数衰减尺度 |
| `smp_reward_scale` | 奖励缩放 |
| `diffusion_steps` | SDS 时间步列表 |
| `task_reward_weight` | 任务奖励权重 |
| `smp_reward_weight` | SMP 奖励权重 |
| `enable_gsi` | 是否启用 GSI |
| `gsi_iters` | GSI 更新间隔 |
| `gsi_buffer_size` | 初始状态池大小 |
| `gsi_regen_num_motions` | 每次再生成数量 |

### 13.4.2 方法清单

| 方法 | 说明 |
|------|------|
| `_build_model` | `SMPModel` + 加载先验 |
| `_build_prior_model` | 加载 `TinyMDMModel` 并冻结 |
| `_check_prior_env_config` | 验证先验与 env 配置一致 |
| `_build_normalizers` | + `_sds_normalizer` |
| `_train_iter` | + SDS 归一化更新 + GSI 更新 |
| `_record_data_post_step` | 记录 `disc_obs` |
| `_compute_rewards` | task + SMP 奖励 |
| `_calc_smp_rewards` | 批量 SDS 计算 |
| `_init_gsi_buffer` | 初始化 GSI 池 |
| `_update_gsi_buffer` | 周期性再生成 |
| `_generate_init_states` | 从先验采样 |

---

## 13.5 两阶段训练流程

### 13.5.1 阶段一：训练扩散先验

```bash
# 多片段数据集
python tools/diffusion_model/train_tinymdm.py \
  --cfg_path tools/diffusion_model/config/tinymdm_multi_clip.yaml \
  --out_dir output/smp_prior

# 单片段
python tools/diffusion_model/train_tinymdm.py \
  --cfg_path tools/diffusion_model/config/tinymdm_single_clip.yaml \
  --out_dir output/smp_prior
```

输出：`model.pt` + `diffusion_config.yaml`

### 13.5.2 阶段二：训练任务策略

```bash
# 使用预训练 LaFAN1 先验 + GSI（无需动作数据）
python mimickit/run.py --arg_file args/smp_location_humanoid_args.txt --out_dir output/

# 单片段模仿（GSI 关闭，需 motion_file）
python mimickit/run.py --arg_file args/smp_humanoid_args.txt --out_dir output/
```

### 13.5.3 Agent 配置

```yaml
smp_prior_cfg: "data/models/smp_prior/diffusion_config.yaml"
smp_prior_model: "data/models/smp_prior/model.pt"
enable_gsi: true
smp_reward_weight: 1.0
task_reward_weight: 0.5
sds_loss_scale: 0.5
diffusion_steps: [22, 15, 8]
```

---

## 13.6 调参优先级

经验优先级（来自 README_SMP）：

```
smp_reward_weight > sds_loss_scale >= diffusion_steps
```

| 参数 | 影响 |
|------|------|
| `smp_reward_weight` | 先验 vs 任务平衡 |
| `sds_loss_scale` | 奖励对 SDS 误差敏感度 |
| `diffusion_steps` | 多时间步平均的稳定性 |

---

## 13.7 参考资料

| 资源 | 链接 |
|------|------|
| DDPM | https://arxiv.org/abs/2006.11239 |
| Score SDE | https://arxiv.org/abs/2011.13456 |
| DreamFusion SDS | https://arxiv.org/abs/2209.14910 |
| LaFAN1 数据集 | https://github.com/ubisoft/ubisoft-laforge-animation-dataset |
| HuggingFace Diffusers | https://github.com/huggingface/diffusers |

---

## 13.8 引用

```bibtex
@article{mu2026smp,
  title = {SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control},
  author = {Mu, Yuxuan and Zhang, Ziyu and others},
  journal = {ACM Transactions on Graphics (SIGGRAPH 2026)},
  year = {2026}
}
```

---

[← LCP](12-algorithm-lcp.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：工具与数据 →](14-tools-and-data.md)
