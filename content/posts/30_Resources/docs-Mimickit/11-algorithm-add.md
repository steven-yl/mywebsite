---
title: "第十一章：ADD 算法"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第十一章：ADD 算法。"
summary: "第十一章：ADD 算法。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 11
series_weight: 11
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第十一章：ADD 算法

## 11.1 算法概述

**Adversarial Differential Discriminator (ADD)**（Zhang et al., SIGGRAPH Asia 2025）改进 AMP：判别器输入**观测差分**$\Delta = d_{demo} - d_{agent}$而非绝对观测，正类为**零差分**（完美匹配），负类为非零差分。

- 项目页：https://xbpeng.github.io/projects/ADD/
- 论文：SIGGRAPH Asia 2025

### 11.1.1 为什么需要 ADD

AMP 判别器直接分类绝对运动特征，对**状态分布偏移**敏感。ADD 关注"与参考的差距"，更聚焦于**模仿误差**，训练更稳定。

### 11.1.2 与 AMP 对比

| 方面 | AMP | ADD |
|------|-----|-----|
| 判别器输入 | 绝对 `disc_obs` | `disc_obs_demo - disc_obs` |
| 正类 | demo 观测 | 零差分向量 |
| 负类 | agent 观测 | 非零差分 |
| 归一化器 | `Normalizer` | `DiffNormalizer` |
| demo 采样时机 | Agent 端统一采样 | Env 每步输出配对观测 |

---

## 11.2 数学原理

### 11.2.1 差分定义

$$
\Delta_t = d_{demo,t} - d_{agent,t}
$$

其中$d$为 `disc_obs`（多步运动特征展平）。

### 11.2.2 判别器训练

**正类（完美匹配）：**

$$
\Delta^+ = \mathbf{0}
$$

$$
L_{pos} = \text{BCE}(D(\Delta^+), 1)
$$

**负类（非零差分）：**

$$
L_{neg} = \text{BCE}(D(\text{normalize}(\Delta)), 0)
$$

$$
L_D = \frac{1}{2}(L_{pos} + L_{neg}) + \lambda_{GP} L_{GP} + \lambda_{reg}\|\psi\|^2
$$

### 11.2.3 策略奖励

对 agent 当前状态，计算$\Delta = d_{demo} - d_{agent}$（env 提供配对 demo），归一化后：

$$
r_{disc} = -\log(1 - \sigma(D(\text{normalize}(\Delta))) + \epsilon) \cdot s_{disc}
$$

与 AMP 奖励形式相同，但判别器输入为差分。

### 11.2.4 直觉

零差分 = "与参考完全一致" → 高奖励；大差分 = "偏离参考" → 低奖励。判别器学习差分空间的"自然性"流形。

---

## 11.3 代码实现：`ADDAgent`

继承 `AMPAgent`，重写判别器相关逻辑。

### 11.3.1 方法清单

| 方法 | 与 AMP 的差异 |
|------|--------------|
| `_build_model` | `ADDModel`（结构同 AMP） |
| `_build_normalizers` | 使用 `DiffNormalizer` |
| `_build_pos_diff` | 创建零向量正样本 |
| `_record_data_post_step` | 同时记录 `disc_obs` 和 `disc_obs_demo` |
| `_record_disc_demo_data` | **空操作**（demo 已在 rollout 记录） |
| `_store_disc_replay_data` | replay 同时存 obs 和 demo |
| `_compute_rewards` | 基于 `obs_diff = demo - agent` |
| `_compute_disc_loss` | 正类零差分 + 负类归一化差分 |

### 11.3.2 判别器损失逻辑

```python
# 正类：零差分
pos_diff = zeros(disc_obs_shape)
disc_pos_logit = model.eval_disc(pos_diff)
disc_loss_pos = BCE(disc_pos_logit, ones)

# 负类：demo - agent 差分
diff_obs = tar_disc_obs - disc_obs
norm_diff = disc_obs_norm.normalize(diff_obs)
disc_neg_logit = model.eval_disc(norm_diff)
disc_loss_neg = BCE(disc_neg_logit, zeros)
```

---

## 11.4 环境：`ADDEnv`

继承 `AMPEnv`，每步输出：

```python
info["disc_obs"]       # agent 状态
info["disc_obs_demo"]  # 同时刻参考 demo
```

`_fetch_disc_demo_data` 从 MotionLib 获取与 agent 同时刻的参考观测。

---

## 11.5 训练命令

```bash
# Humanoid
python mimickit/run.py --arg_file args/add_humanoid_args.txt --out_dir output/

# G1
python mimickit/run.py --arg_file args/add_g1_args.txt --out_dir output/
```

支持角色：humanoid, g1, go2, smpl, pi_plus。

---

## 11.6 调参建议

与 AMP 类似，额外注意：

- `DiffNormalizer` 需足够 `normalizer_samples` 以稳定差分统计
- 配对 demo 必须与 agent 同时刻对齐（env 保证）

---

## 11.7 引用

```bibtex
@inproceedings{zhang2025ADD,
  author = {Zhang, Ziyu and Bashkirov, Sergey and Yang, Dun and Shi, Yi and Taylor, Michael and Peng, Xue Bin},
  title = {Physics-Based Motion Imitation with Adversarial Differential Discriminators},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  year = {2025}
}
```

---

[← ASE](10-algorithm-ase.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：LCP →](12-algorithm-lcp.md)
