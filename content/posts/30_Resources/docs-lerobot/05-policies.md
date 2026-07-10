---
title: "05 策略模型"
subtitle: ""
date: 2026-07-10T17:44:00+08:00
draft: false
authors: [Steven]
description: "16 种策略架构、训练/推理 API、选型建议与 PreTrainedPolicy 接口。"
summary: "LeRobot 策略模型体系与训练推理接口。"
tags: [lerobot, robots]
categories: [docs lerobot, robots]
series: [lerobot-docs]
weight: 5
series_weight: 5
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 05 — 策略模型（Policies）

> **算法公式、推导与论文链接** → 见专章 [13-algorithms-and-mathematics.md](./13-algorithms-and-mathematics.md)  
> **PreTrainedPolicy 全部方法** → [12-core-api-reference.md §1](./12-core-api-reference.md)

---

## 1. 模块边界

```
policies/
├── pretrained.py          # PreTrainedPolicy
├── factory.py             # make_policy, make_pre_post_processors, get_policy_class
├── act/ diffusion/ tdmpc/ vqbet/ multi_task_dit/ gaussian_actor/
├── smolvla/ pi0/ pi05/ pi0_fast/ groot/ xvla/ wall_x/ eo1/ molmoact2/ vla_jepa/
└── rtc/                   # Real-Time Chunking（推理基础设施，非 policy.type）
```

每个策略目录三件套：

| 文件 | 内容 |
|------|------|
| `configuration_*.py` | `@PreTrainedConfig.register_subclass` |
| `modeling_*.py` | `*Policy(PreTrainedPolicy)`：`forward`, `select_action`, … |
| `processor_*.py` | 可选：专用 normalize/rename 构建 |

---

## 2. PreTrainedPolicy 契约

### 2.1 训练 API

```python
loss, info_dict = policy.forward(batch)  # batch 已经过 preprocessor
loss.backward()
```

`forward` 返回标量 loss；`info_dict` 供 W&B 记录（各策略不同 key）。

### 2.2 推理 API

| 方法 | 何时调用 | 输出 |
|------|----------|------|
| `select_action(batch)` | 每控制步 / Sync 引擎 | `(B, action_dim)` |
| `predict_action_chunk(batch)` | RTC / chunking | `(B, H, action_dim)` |
| `reset()` | episode 开始 | — |

### 2.3 持久化

`save_pretrained` 保存 `config.json` + `model.safetensors`；processor 单独目录。

---

## 3. 工厂（factory.py）

| 函数 | 作用 |
|------|------|
| `get_policy_class(name)` | 懒加载 16 类 |
| `make_policy_config(type, **kwargs)` | 构造 Config |
| `make_policy(cfg, ds_meta?, env_cfg?, rename_map?)` | 推断 features + 加载权重 |
| `make_pre_post_processors(...)` | 策略 pre/post pipeline |

**16 种 `name`**：`tdmpc`, `diffusion`, `act`, `multi_task_dit`, `vqbet`, `pi0`, `pi0_fast`, `pi05`, `gaussian_actor`, `smolvla`, `groot`, `xvla`, `wall_x`, `eo1`, `molmoact2`, `vla_jepa`

未知 name → `PreTrainedConfig.get_choice_class` 插件 fallback。

---

## 4. 策略详解（16 种）

### 4.1 模仿学习

#### ACT (`act`)

| 项 | 内容 |
|----|------|
| **算法** | Action Chunking Transformer + 可选 VAE |
| **损失** | $\mathcal{L}_{\text{L1}} + \lambda \mathcal{L}_{\text{KL}}$ — 见 [§1.1](./13-algorithms-and-mathematics.md#11-actaction-chunking-transformer) |
| **架构** | ResNet → Transformer enc/dec → chunk |
| **推理** | action queue 或 temporal ensembling |
| **方法** | `get_optim_params()` 分组 LR；`select_action()` 管理 queue |
| **论文** | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) |
| **适用** | 中数据量、双臂/单臂 baseline |

#### Diffusion (`diffusion`)

| 项 | 内容 |
|----|------|
| **算法** | DDPM/DDIM on action trajectory |
| **损失** | $\|\epsilon_\theta - \epsilon\|^2$ — [§2.1](./13-algorithms-and-mathematics.md#21-diffusion-policy) |
| **架构** | ResNet+SpatialSoftmax → 1D U-Net |
| **推理** | 迭代 denoise + obs/action queue |
| **论文** | [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) |

#### VQ-BeT (`vqbet`)

| 项 | 内容 |
|----|------|
| **算法** | RVQ-VAE + GPT behavior transformer |
| **损失** | Focal(codes) + $\lambda$ L1(offset) — [§4.1](./13-algorithms-and-mathematics.md#41-vq-bet) |
| **训练** | 两阶段：先 RVQ，再 BeT |
| **论文** | [arXiv:2403.03181](https://arxiv.org/abs/2403.03181) |

#### Multi-Task DiT (`multi_task_dit`)

| 项 | 内容 |
|----|------|
| **算法** | CLIP 条件 DiT；diffusion 或 flow |
| **论文** | [arXiv:2507.05331](https://arxiv.org/abs/2507.05331) |
| **适用** | 多任务 + 语言 |

---

### 4.2 强化学习

#### TDMPC (`tdmpc`)

| 项 | 内容 |
|----|------|
| **算法** | Latent TD-MPC + FOWM expectile V |
| **损失** | 5 项加权 — [§3.1](./13-algorithms-and-mathematics.md#31-td-mpc--fowm) |
| **推理** | MPC (CEM/MPPI) 或 direct π |
| **论文** | [TD-MPC](https://arxiv.org/abs/2203.04955), [FOWM](https://arxiv.org/abs/2310.16029) |

#### Gaussian Actor (`gaussian_actor`)

| 项 | 内容 |
|----|------|
| **算法** | Tanh-squashed Gaussian（SAC actor） |
| **forward** | 返回 log_prob，**非 BC loss** |
| **用途** | `rl/` + HIL-SERL |
| **论文** | [SAC](https://arxiv.org/abs/1801.01290) |

---

### 4.3 VLA 与大型多模态

#### PI0 (`pi0`) / PI05 (`pi05`) / PI0-FAST (`pi0_fast`)

| 变体 | 动作表示 | 损失 |
|------|----------|------|
| pi0 / pi05 | 连续 flow matching | $\|v - (\epsilon-a)\|^2$ |
| pi0_fast | FAST 离散 token | Cross-entropy |

- **架构**：PaliGemma + Gemma action expert（fast 为 AR token）
- **论文**：[arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- **上游**：[OpenPI](https://github.com/Physical-Intelligence/openpi)
- **部署**：推荐 `--inference.type=rtc`

#### SmolVLA (`smolvla`)

- SmolVLM2-500M + action expert；flow 同 π₀
- 论文：[arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- extra：`lerobot[smolvla]`

#### GR00T (`groot`)

- Eagle 2.5-VL + CrossAttentionDiT
- Flow 插值方向与 π₀ **不同** — 见 [§2.3](./13-algorithms-and-mathematics.md#23-gr00t-flow-matching插值方向相反)
- 论文：[arXiv:2503.14734](https://arxiv.org/abs/2503.14734)
- 上游：[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)

#### XVLA (`xvla`)

- Florence-2 + SoftPromptedTransformer
- 损失随 `action_mode`（ee6d / joint）变化 — [§5.2](./13-algorithms-and-mathematics.md#52-xvla-动作空间损失action_hub)

#### WALL-X (`wall_x`)

- Qwen2.5-VL MoE；$\mathcal{L} = \mathcal{L}_{\text{CE}}^{\text{lang}} + \lambda \mathcal{L}_{\text{flow}}$
- 上游：https://github.com/x2-robot/wall-x

#### EO-1 (`eo1`)

- Qwen2.5-VL + state/action placeholder tokens + flow

#### MolmoAct2 (`molmoact2`)

- 连续 flow + 离散 token；`training_mode`: flow_only / discrete_only / both
- 博客：https://allenai.org/blog/molmoact2

#### VLA-JEPA (`vla_jepa`)

- Qwen3-VL + DiT action + V-JEPA world model
- $\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{wm}}$

---

## 5. RTC（非 policy.type）

**路径**：`policies/rtc/`

| 组件 | 文件 | 作用 |
|------|------|------|
| `ActionQueue` | `action_queue.py` | 线程安全 chunk 队列 |
| `RTCProcessor` | `rtc_processor.py` | prefix re-anchor |
| `LatencyTracker` | `latency_tracker.py` | 延迟统计 |
| `ActionInterpolator` | `action_interpolator.py` | 平滑执行 |

算法说明：[13 §7 RTC](./13-algorithms-and-mathematics.md#7-rtc-推理算法)

---

## 6. 选型对比

| 策略 | 参数量 | 语言 | 数据量 | 推理引擎 | 真机 |
|------|--------|------|--------|----------|------|
| ACT | 小 | ✗ | 50–200 demos | sync | ✓ 常用 |
| Diffusion | 中 | ✗ | 100+ | sync | ✓ |
| PI0/SmolVLA | 大 | ✓ | 1k+ 或多任务 | **rtc** | ✓ SOTA 方向 |
| TDMPC | 中 | ✗ | RL rollouts | sync | 仿真多 |
| VQ-BeT | 中 | ✗ | 中长 horizon | sync | ✓ |

---

## 7. PEFT 微调

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --peft.method_type=LORA \
  --peft.r=8 \
  --dataset.repo_id=user/task
```

`wrap_with_peft()` → 冻结 base，训练 adapter。

---

## 8. 示例

### 8.1 构造并 forward

```python
import torch
from lerobot.policies.factory import make_policy_config, get_policy_class

cfg = make_policy_config(
    "act",
    input_features={
        "observation.state": {"type": "STATE", "shape": (6,)},
        "observation.images.top": {"type": "VISUAL", "shape": (3, 96, 96)},
    },
    output_features={"action": {"type": "ACTION", "shape": (6,)}},
)
policy = get_policy_class("act")(cfg)
batch = {
    "observation.state": torch.randn(2, 6),
    "observation.images.top": torch.randn(2, 3, 96, 96),
    "action": torch.randn(2, 6),
}
loss, logs = policy.forward(batch)
print(loss.item())
```

### 8.2 训练命令

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --steps=100000 \
  --batch_size=64
```

---

## 相关章节

- [13 算法与数学](./13-algorithms-and-mathematics.md)
- [06 训练](./06-training-evaluation.md)
- [04 Processor](./04-processor-pipeline.md)
- [09 部署](./09-rollout-inference.md)
