---
title: "06 训练与评估"
subtitle: ""
date: 2026-07-10T17:44:00+08:00
draft: false
authors: [Steven]
description: "lerobot-train/eval 流程、优化器、Accelerate 分布式与评估循环。"
summary: "LeRobot 训练与评估流程、优化器与分布式配置。"
tags: [lerobot, robots]
categories: [docs lerobot, robots]
series: [lerobot-docs]
weight: 6
series_weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 06 — 训练与评估

## 1. 模块边界

```
scripts/lerobot_train.py    # 训练主入口
scripts/lerobot_eval.py     # 仿真评估
common/train_utils.py       # checkpoint
common/wandb_utils.py       # W&B
optim/
├── factory.py              # make_optimizer_and_scheduler
├── optimizers.py           # Adam/AdamW/SGD/...
└── schedulers.py           # cosine, diffuser, vqbet
datasets/factory.py         # make_train_eval_datasets
policies/factory.py         # make_policy, make_pre_post_processors
envs/factory.py             # make_env（训练中 periodic eval）
```

---

## 2. lerobot-train 流程

### 2.1 配置

顶层：`TrainPipelineConfig`（`configs/train.py`）

关键字段：

| 字段 | 说明 |
|------|------|
| `steps` | 总优化步数 |
| `batch_size` | 全局 batch（Accelerate 分片） |
| `save_freq` | checkpoint 间隔 |
| `log_freq` | 日志间隔 |
| `eval_freq` |  held-out batch loss |
| `env_eval_freq` | 仿真 rollout 评估间隔 |
| `use_policy_training_preset` | 使用 policy 内置 optimizer 配置 |
| `seed` | 随机种子 |

### 2.2 主流程（伪代码）

```python
cfg.validate()
train_ds, eval_ds = make_train_eval_datasets(cfg)
policy = make_policy(cfg.policy, ds_meta=train_ds.meta)
policy = policy.wrap_with_peft(cfg.peft)  # 可选
preprocessor, postprocessor = make_pre_post_processors(cfg.policy, dataset_stats=...)

optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
accelerator = Accelerator(...)

for step in range(cfg.steps):
    batch = next(dataloader)
    batch = preprocessor(batch)
    loss, logs = policy.forward(batch)
    accelerator.backward(loss)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()

    if step % cfg.save_freq == 0:
        save_checkpoint(policy, preprocessor, postprocessor, ...)
    if step % cfg.env_eval_freq == 0 and cfg.env:
        eval_policy_all(...)  # 仿真
```

### 2.3 DataLoader

- **Sampler**：`EpisodeAwareSampler` — 不非法跨 episode
- **Collate**：`lerobot_collate_fn`（`utils/collate.py`）— stack tensor，保留字符串 task
- **Workers**：解码视频时 `num_workers>0` 需注意 decoder 进程安全

### 2.4 奖励模型训练

`TrainPipelineConfig.reward_model` 替代 `policy` 时，走 `rewards/factory.py` 构建 reward model，损失由对应 `forward` 定义（SARM、Robometer 等）。

---

## 3. 优化器系统（`optim/`）

### 3.1 OptimizerConfig 注册类型

| type | 类 | 特点 |
|------|-----|------|
| `adam` | Adam | 默认 |
| `adamw` | AdamW | 权重衰减解耦 |
| `sgd` | SGD | |
| `xvla-adamw` | XVLA 专用 | 分组 LR |
| `multi_adam` | MultiAdam | 多组件多 optimizer |

### 3.2 LRSchedulerConfig

| type | 适用 |
|------|------|
| `cosine_decay_with_warmup` | 通用 |
| `diffuser` | Diffusion policy |
| `vqbet` | VQ-BeT 两阶段 |

### 3.3 `make_optimizer_and_scheduler(cfg, policy)`

- `use_policy_training_preset=True` → 调用 `policy.get_optim_params()` 与 `cfg.policy.optimizer`
- 否则使用 `cfg.optimizer` 全局配置

---

## 4. Accelerate 集成

- 多 GPU / 混合精度由 `Accelerator` 管理
- `policy.use_amp` 控制 autocast
- FSDP 等高级并行见 `docs/source/multi_gpu_training.mdx`

---

## 5. Checkpoint 内容

典型输出目录：

```
outputs/train/run_name/
├── checkpoints/
│   └── last/
│       ├── pretrained_model/
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   ├── policy_preprocessor/
│       │   └── policy_postprocessor/
│       └── training_state.pth   # optimizer, step, rng
```

`push_to_hub` 上传 `pretrained_model` + training config YAML。

---

## 6. lerobot-eval 流程

### 6.1 配置

`EvalPipelineConfig`（`configs/eval.py`）：

- **`--policy.path`**（必需）：Hub 或本地 checkpoint
- **`--env.type`**：仿真环境
- **`--eval.n_episodes`**：评估 episode 数
- **`--eval.batch_size`**：向量化环境数

### 6.2 主流程

```python
envs = make_env(cfg.env, n_envs=cfg.eval.batch_size)
policy = make_policy(cfg.policy, env_cfg=cfg.env)
pre, post = make_pre_post_processors(...)

metrics = eval_policy_all(envs, policy, pre, post, n_episodes=...)
# success rate, cum reward, ...
```

### 6.3 可选录制

评估轨迹可写入新 `LeRobotDataset` 供分析。

### 6.4 与 train 内 eval 区别

| | train 内 env_eval | lerobot-eval |
|--|-------------------|--------------|
| 触发 | `env_eval_freq` | 独立 CLI |
| 用途 | 训练曲线监控 | 基准测试、论文数字 |
| 配置 | `TrainPipelineConfig.env` | `EvalPipelineConfig` |

---

## 7. lerobot-train-tokenizer

为 **PI0-FAST** 等训练 FAST action tokenizer：

- 从 dataset 采样 action chunks
- 学习离散化 codebook
- 输出 tokenizer 权重供 policy 使用

---

## 8. 样本加权（RA-BC）

`utils/sample_weighting.py` + reward model 权重（SARM/Robometer TOPReward）：

- `policy.forward(batch, reduction="none")` 支持 per-sample loss
- 按 annotation 权重重加权 BC loss

---

## 9. 示例

### 9.1 最小训练命令

```bash
uv sync --locked --extra training --extra pusht

lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/pusht \
  --env.type=pusht \
  --steps=10000 \
  --batch_size=64 \
  --save_freq=5000 \
  --wandb.enable=false
```

### 9.2 评估命令

```bash
uv sync --locked --extra evaluation --extra libero --extra pi

lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

### 9.3 编程式 eval 钩子（概念）

```python
from lerobot.datasets.factory import make_dataset
from lerobot.configs.default import DatasetConfig
from dataclasses import dataclass

@dataclass
class C:
    dataset: DatasetConfig

cfg = C(dataset=DatasetConfig(repo_id="lerobot/pusht"))
ds = make_dataset(cfg)
print("train frames:", len(ds))
```

---

## 10. 监控与调试

| 工具 | 用途 |
|------|------|
| W&B (`WandBConfig`) | loss、lr、eval 指标 |
| `eval_freq` | held-out parquet batch loss |
| `env_eval_freq` | 仿真 success rate |
| Rerun | 数据集/轨迹可视化（非 train 内置） |

---

## 下一章

- 真机录制 → [07-hardware-layer.md](./07-hardware-layer.md)
- 真机部署 → [09-rollout-inference.md](./09-rollout-inference.md)
