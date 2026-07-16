---
title: "第七章：DeepMimic + PPO 算法"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第七章：DeepMimic + PPO 算法。"
summary: "第七章：DeepMimic + PPO 算法。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第七章：DeepMimic + PPO 算法

## 7.1 算法概述

**DeepMimic**（Peng et al., TOG 2018）将运动模仿建模为：最大化仿真角色与参考动作之间的相似度奖励。**PPO**（Proximal Policy Optimization）作为策略优化器，提供稳定的 on-policy 梯度更新。

- 项目页：https://xbpeng.github.io/projects/DeepMimic/
- 论文：https://doi.org/10.1145/3197517.3201311
- 开源参考：[DeepMimic (GitHub)](https://github.com/xbpeng/DeepMimic)

### 7.1.1 适用场景

- 精确复现**单个**或**少量**参考动作（spin kick、翻滚等）
- 需要明确逐帧对齐的模仿任务
- 作为其他算法（AMP 等）的**基线对比**

---

## 7.2 数学原理

### 7.2.1 奖励（见 [05-environments](05-environments.md) 5.5.4）

$$
r_t = \sum_i w_i \exp(-\lambda_i e_i)
$$

其中$e_i$为姿态/速度/根/关键点误差。

### 7.2.2 TD(λ) 价值目标

$$
V_t^{target} = R_t^{(\lambda)}, \quad R_t^{(\lambda)} = r_t + \gamma[(1-\lambda)V_{t+1} + \lambda R_{t+1}^{(\lambda)}]
$$

### 7.2.3 优势函数

$$
A_t = V_t^{target} - V_\phi(s_t)
$$

训练时对随机采样的动作批次做标准化并裁剪：

$$
\hat{A}_t = \text{clip}\left(\frac{A_t - \bar{A}}{\sigma_A + \epsilon}, -c, c\right)
$$

### 7.2.4 PPO Clipped Surrogate Loss

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

$$
L^{CLIP}(\theta) = -\mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

默认$\epsilon = 0.2$（`ppo_clip_ratio`）。

### 7.2.5 Critic Loss

$$
L^{VF} = \mathbb{E}\left[(V_\phi(s_t) - V_t^{target})^2\right]
$$

### 7.2.6 完整 Actor Loss

$$
L^{actor} = L^{CLIP} + w_{bound} L_{bound} - w_{ent} H(\pi) + w_{reg} L_{reg}
$$

---

## 7.3 代码实现：`PPOAgent`

### 7.3.1 超参数（`_load_params`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `actor_epochs` | 5 | Actor 每轮更新 epoch 数 |
| `actor_batch_size` | 4 | 批大小（× num_envs） |
| `critic_epochs` | 2 | Critic epoch 数 |
| `critic_batch_size` | 2 | Critic 批大小 |
| `td_lambda` | 0.95 | TD(λ) 参数 |
| `ppo_clip_ratio` | 0.2 | PPO 裁剪范围 |
| `norm_adv_clip` | 4.0 | 优势裁剪 |
| `action_bound_weight` | 10.0 | 动作边界惩罚 |
| `action_entropy_weight` | 0.0 | 熵奖励 |
| `exp_prob_beg/end` | 1.0 | 探索概率退火 |

### 7.3.2 方法清单

| 方法 | 说明 |
|------|------|
| `_build_model` | 创建 `PPOModel` |
| `_build_optimizer` | Actor/Critic 两个 `MPOptimizer` |
| `_decide_action` | 训练时以 `exp_prob` 在 sample/mode 间切换 |
| `_record_data_pre_step` | 额外记录 `a_logp`, `rand_action_mask` |
| `_build_train_data` | 计算 TD(λ) 回报与归一化优势 |
| `_update_model` | 更新 Critic 再更新 Actor |
| `_update_critic` | MSE 价值损失 |
| `_update_actor` | PPO clipped loss |
| `_compute_critic_loss` | MSE |
| `_compute_actor_loss` | Clipped surrogate + 可选正则 |
| `_get_exp_prob` | 探索概率线性退火 |

### 7.3.3 动作决策

```python
# 训练模式：随机探索
norm_a_rand = dist.sample()
norm_a_mode = dist.mode
norm_a = where(bernoulli(exp_prob), norm_a_rand, norm_a_mode)

# 测试模式：确定性
norm_a = dist.mode
```

仅 `rand_action_mask == 1` 的样本参与 Actor 损失计算（PPO 需要 on-policy 比率）。

---

## 7.4 环境配对

| 配置 | 说明 |
|------|------|
| `env_name: deepmimic` | 使用 `DeepMimicEnv` |
| `agent_name: PPO` | 使用 `PPOAgent` |
| `motion_file` | 单片段 `.pkl` 或数据集 `.yaml` |

---

## 7.5 训练与测试命令

```bash
# 训练
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --visualize false --out_dir output/

# 测试
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --mode test --num_envs 4 --visualize true \
  --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```

---

## 7.6 调参建议

| 参数 | 影响 |
|------|------|
| `reward_*_scale` | 误差敏感度，过大导致奖励稀疏 |
| `reward_*_w` | 各分量相对重要性 |
| `pose_termination_dist` | 过严导致频繁 FAIL |
| `action_std` | 探索噪声，过大动作抖动 |
| `steps_per_iter` | 每轮样本量，影响 PPO 稳定性 |

---

## 7.7 引用

```bibtex
@article{2018-TOG-deepMimic,
  author = {Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and van de Panne, Michiel},
  title = {DeepMimic: Example-guided Deep Reinforcement Learning of Physics-based Character Skills},
  journal = {ACM Trans. Graph.},
  year = {2018},
  doi = {10.1145/3197517.3201311}
}
```

---

[← 学习核心](06-learning-core.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：AWR →](08-algorithm-awr.md)
