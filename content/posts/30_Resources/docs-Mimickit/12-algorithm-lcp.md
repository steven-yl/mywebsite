---
title: "第十二章：LCP 算法"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第十二章：LCP 算法。"
summary: "第十二章：LCP 算法。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 12
series_weight: 12
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第十二章：LCP 算法

## 12.1 算法概述

**Lipschitz-Constrained Policies (LCP)**（Chen et al., IROS 2025）在 PPO Actor 损失上增加**平滑约束**：惩罚策略对观测的梯度范数，使策略满足近似 Lipschitz 条件，减少动作抖动。

- 项目页：https://xbpeng.github.io/projects/LCP/
- 论文：IROS 2025

### 12.1.1 解决什么问题

物理仿真策略常对观测微小变化产生剧烈动作响应，导致：

- 仿真中动作抖动、不自然
- 部署到真实机器人时放大噪声

LCP 通过约束$\|\nabla_s \log \pi(a|s)\|$获得平滑策略。

### 12.1.2 设计特点

`LCPAgent` 是 **Wrapper 类**：默认继承 `PPOAgent`，仅重写 `_compute_actor_loss`。理论上可包装其他 Agent。

---

## 12.2 数学原理

### 12.2.1 Lipschitz 约束

策略$\pi_\theta(a|s)$的 Lipschitz 常数$K$满足：

$$
\|\pi(a|s_1) - \pi(a|s_2)\| \leq K \|s_1 - s_2\|
$$

对 log 概率梯度约束：

$$
L_{LCP} = \mathbb{E}\left[\left\|\nabla_s \log \pi_\theta(a|s)\right\|^2\right]
$$

### 12.2.2 完整 Actor Loss

$$
L^{actor} = L^{PPO} + w_{LCP} \cdot L_{LCP}
$$

其中$w_{LCP}$为 `lcp_weight`。

### 12.2.3 与谱归一化/梯度惩罚的关系

- **谱归一化**：约束网络层 Lipschitz 常数（架构级）
- **LCP**：直接约束策略输出对输入的敏感度（目标级）
- 参考：[Czarnecki et al., 2017 — Sobolev Training](https://arxiv.org/abs/1706.04859)

---

## 12.3 代码实现

### 12.3.1 `LCPAgent`

| 方法 | 说明 |
|------|------|
| `_load_params` | 加载 `lcp_weight` |
| `_build_model` | `LCPModel`（同 PPO） |
| `_compute_actor_loss` | 调用 super + 加 LCP 项 |
| `_compute_lcp_loss` | 计算梯度范数损失 |

### 12.3.2 LCP 损失实现

```python
def _compute_lcp_loss(self, norm_obs, norm_a):
    norm_obs.requires_grad_(True)
    a_dist = self._model.eval_actor(norm_obs)
    a_logp = a_dist.log_prob(norm_a)

    a_logp_grad = torch.autograd.grad(
        a_logp, norm_obs,
        grad_outputs=torch.ones_like(a_logp),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    a_logp_grad_norm = torch.sum(torch.square(a_logp_grad), dim=-1)
    return torch.mean(a_logp_grad_norm)
```

需要 `create_graph=True` 以便 LCP 损失参与 Actor 反向传播。

---

## 12.4 环境配对

LCP 可与任意 PPO 环境组合，默认配置为 G1 行走：

```yaml
# lcp_g1_agent.yaml
agent_name: "LCP"
lcp_weight: <需调参>
```

```yaml
# deepmimic_g1_env.yaml
env_name: "deepmimic"
```

---

## 12.5 训练命令

```bash
python mimickit/run.py --arg_file args/deepmimic_g1_ppo_args.txt \
  --agent_config data/agents/lcp_g1_agent.yaml \
  --out_dir output/
```

---

## 12.6 调参建议

| 参数 | 建议 |
|------|------|
| `lcp_weight` | **最关键参数**；过大动作迟钝，过小无平滑效果 |
| 起始值 | 从 1e-4 ~ 1e-2 搜索 |
| 不同任务/角色 | 需重新调参 |

日志中关注 `lcp_loss` 与 `actor_loss` 的量级比例。

---

## 12.7 引用

```bibtex
@article{chen2025lcp,
  title = {Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies},
  author = {Zixuan Chen and Xialin He and Yen-Jen Wang and others},
  journal = {IROS},
  year = {2025}
}
```

---

[← ADD](11-algorithm-add.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：SMP →](13-algorithm-smp.md)
