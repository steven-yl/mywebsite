---
title: "第八章：AWR 算法"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第八章：AWR 算法。"
summary: "第八章：AWR 算法。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 8
series_weight: 8
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第八章：AWR 算法

## 8.1 算法概述

**Advantage-Weighted Regression (AWR)**（Peng et al., 2019）是一种简单可扩展的**离策略** RL 方法：用优势函数对动作进行指数加权，将策略改进转化为加权最大似然回归。

- 项目页：https://xbpeng.github.io/projects/AWR/
- 论文：https://arxiv.org/abs/1910.00177
- 开源参考：[AWR (GitHub)](https://github.com/xbpeng/AWR)

### 8.1.1 与 PPO 对比

| 方面 | PPO | AWR |
|------|-----|-----|
| 策略更新 | Clipped surrogate ratio | 优势加权 log_prob |
| On/Off-policy | On-policy（需 importance ratio） | 允许离策略重用样本 |
| 实现复杂度 | 需存储 old_log_prob | 更简单 |

### 8.1.2 适用场景

- 与 DeepMimic 相同环境，对比 PPO 样本效率
- 研究离策略模仿学习

---

## 8.2 数学原理

### 8.2.1 优势加权

$$
w_t = \exp\left(\frac{\hat{A}_t}{T}\right), \quad w_t \leftarrow \min(w_t, w_{max})
$$

其中$T$为 `awr_temp` 温度，$w_{max}$为 `a_weight_clip`。

温度越高，权重分布越平坦（更多样本参与）；温度越低，仅高优势样本主导。

### 8.2.2 Actor Loss

$$
L^{AWR} = -\mathbb{E}\left[w_t \cdot \log \pi_\theta(a_t | s_t)\right]
$$

等价于最大化加权对数似然，高优势动作被赋予更大权重。

### 8.2.3 Critic Loss

与 PPO 相同：TD(λ) 目标 + MSE。

$$
L^{VF} = \mathbb{E}\left[(V_\phi(s_t) - R_t^{(\lambda)})^2\right]
$$

### 8.2.4 与策略迭代的关系

AWR 可视为对约束策略搜索的近似：在信任域内，最优策略正比于$\exp(A/T) \pi_{old}$。参考：

- [AWAC (Nair et al., 2020)](https://arxiv.org/abs/1910.00177)
- [RWR (Peters & Schaal, 2007)](https://authors.library.caltech.edu/records/1986)

---

## 8.3 代码实现：`AWRAgent`

### 8.3.1 超参数

| 参数 | 说明 |
|------|------|
| `awr_temp` | 优势温度$T$|
| `a_weight_clip` | 权重上界 |
| `td_lambda` | TD(λ) |
| `actor_epochs` / `critic_epochs` | 更新轮数 |

### 8.3.2 方法清单

| 方法 | 与 PPO 的差异 |
|------|--------------|
| `_build_model` | 使用 `AWRModel`（结构同 PPO） |
| `_decide_action` | **不记录** `a_logp` |
| `_build_train_data` | 计算 `a_weight` 而非 `adv` |
| `_compute_actor_loss` | 加权 `-mean(w * log_prob)` |

### 8.3.3 核心损失代码逻辑

```python
a_weight = torch.exp(norm_adv / self._awr_temp)
a_weight = torch.clamp_max(a_weight, self._a_weight_clip)

a_dist = self._model.eval_actor(norm_obs)
a_logp = a_dist.log_prob(norm_a)
actor_loss = -torch.mean(a_weight * a_logp)
```

---

## 8.4 环境配对

```yaml
# env: deepmimic（同 PPO）
# agent: deepmimic_humanoid_awr_agent.yaml
agent_name: "AWR"
```

---

## 8.5 训练命令

```bash
python mimickit/run.py --arg_file args/deepmimic_humanoid_awr_args.txt \
  --visualize false --out_dir output/
```

---

## 8.6 调参建议

| 参数 | 建议 |
|------|------|
| `awr_temp` | 从 1.0 开始；过小权重退化，过大等价均匀回归 |
| `a_weight_clip` | 防止极端优势导致梯度爆炸，典型 20–100 |
| `exp_prob` | 保持足够探索以覆盖动作空间 |

---

## 8.7 引用

```bibtex
@article{AWRPeng19,
  author = {Xue Bin Peng and Aviral Kumar and Grace Zhang and Sergey Levine},
  title = {Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning},
  journal = {CoRR},
  volume = {abs/1910.00177},
  year = {2019}
}
```

---

[← DeepMimic+PPO](07-algorithm-deepmimic-ppo.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：AMP →](09-algorithm-amp.md)
