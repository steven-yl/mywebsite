---
title: "11 扩展模块"
subtitle: ""
date: 2026-07-10T17:44:00+08:00
draft: false
authors: [Steven]
description: "奖励模型、RL、标注工具与 utils 等扩展模块。"
summary: "LeRobot 奖励、RL、标注与工具扩展模块。"
tags: [lerobot, robots]
categories: [docs lerobot, robots]
series: [lerobot-docs]
weight: 11
series_weight: 11
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 11 — 扩展模块（Rewards / RL / Annotations / Utils）

## 1. 模块概览

| 模块 | 路径 | 作用 |
|------|------|------|
| 奖励模型 | `rewards/` | 学习进度估计、RA-BC 加权、HIL reward |
| 强化学习 | `rl/` | SAC 在线 RL actor/learner |
| 标注管道 | `annotations/` | VLM  steerable 标注 |
| 异步传输 | `async_inference/` + `transport/` | gRPC 远程推理 |
| 优化器 | `optim/` | 见 [06](./06-training-evaluation.md) |
| 图像变换 | `transforms/` | 训练时增强配置 |
| 工具 | `utils/` | 跨模块 helper |

---

## 2. 奖励模型（Rewards）

### 2.1 基类

**文件**：`rewards/pretrained.py`

类似 `PreTrainedPolicy`：

- `RewardModelConfig`（`configs/rewards.py`）ChoiceRegistry 基类
- `make_reward_model()` — `rewards/factory.py`

### 2.2 四种类型

| type | 类 | 用途 | 损失（概要） |
|------|-----|------|--------------|
| `reward_classifier` | `RewardClassifier` | 成功/阶段二分类 | BCE / CE |
| `sarm` | `SARM` | 阶段 index + 段内 τ | $\mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{MSE}(\tau)}$ |
| `robometer` | `Robometer` | 操作质量评分 | 回归 / ranking |
| `topreward` | `TOPReward` | 轨迹 preference | ranking loss |

每种含：`configuration_*.py`, `modeling_*.py`, `processor_*.py`, 可选 `compute_rabc_weights.py`

**公式与 RA-BC 推导** → [13-algorithms §6](./13-algorithms-and-mathematics.md#6-奖励模型与-ra-bc)

### 2.3 RA-BC 样本加权

SARM/Robometer/TOPReward 可计算 per-frame 权重：

```
weights = compute_rabc_weights(dataset, reward_model)
train: loss = (policy.forward(batch, reduction="none") * weights).mean()
```

用于强调「高学习价值」片段，改善 BC 效率。

### 2.4 训练入口

`lerobot-train --reward_model.type=sarm ...`（与 `--policy` 互斥或分阶段）

**extras**：`lerobot[sarm]`, `lerobot[robometer]`, `lerobot[topreward]`

---

## 3. 强化学习（RL）

### 3.1 模块结构

```
rl/
├── algorithms/
│   └── sac/              # SAC 配置与算法
├── actor.py              # 环境交互采集
├── learner.py            # 参数更新
└── ...
```

### 3.2 SAC

- **策略**：`gaussian_actor` policy type
- **配置**：`RLAlgorithmConfig.register_subclass("sac")`
- **环境**：常与 `gym_manipulator` + HIL-SERL 联用

### 3.3 与 IL 对比

| | 模仿学习 | RL (SAC/HIL-SERL) |
|--|----------|-------------------|
| 数据 | 固定 dataset | 在线 rollout buffer |
| 损失 | BC / diffusion | TD + entropy |
| 人类 | 仅演示 | 干预、gamepad |

**extra**：`lerobot[hilserl]`

---

## 4. 标注管道（Annotations）

### 4.1 模块

```
annotations/
└── steerable_pipeline/   # VLM 引导的 steerable 标注
```

### 4.2 lerobot-annotate

- 后端：OpenAI 兼容 API（vLLM / transformers serve /  hosted）
- 写入 dataset language 列（parquet）
- 分布式：HF Jobs + `examples/annotations/run_hf_job.py`

**extra**：`lerobot[annotations]`（不含 vllm 硬依赖）

### 4.3 与 TrainingRecipe 配合

`configs/recipe.py` — YAML 对话模板，VLA 训练时 `RenderMessagesStep` 渲染多轮 language。

---

## 5. Async Inference 详解

见 [09-rollout-inference.md](./09-rollout-inference.md) §9。

**proto 定义**：`transport/services.proto` → `services_pb2.py`

典型拓扑：

```
[Robot + Client] --gRPC--> [GPU Server + PolicyServer]
```

---

## 6. Transforms（图像增强）

**路径**：`transforms/transforms.py`

- 与 `DatasetConfig.image_transforms` 关联
- 在 `LeRobotDataset.__getitem__` 解码后应用
- `lerobot-imgtransform-viz` 预览效果

常见：ColorJitter、RandomCrop、Resize — 配置为 draccus dataclass。

---

## 7. Utils 工具库

**路径**：`utils/`

| 模块 | 关键内容 |
|------|----------|
| `constants.py` | `ACTION`, `OBS_STATE`, checkpoint 默认名 |
| `hub.py` | `HubMixin` 基类 |
| `device_utils.py` | `auto_select_torch_device`, AMP 检测 |
| `logging_utils.py` | 训练 metrics 日志 |
| `collate.py` | `lerobot_collate_fn` |
| `feature_utils.py` | dataset ↔ policy feature 转换 |
| `import_utils.py` | 插件注册、`register_third_party_plugins` |
| `random_utils.py` | seed |
| `rotation.py` | 旋转矩阵/quaternion |
| `action_interpolator.py` | rollout 动作插值 |
| `visualization_utils.py` | Rerun 辅助 |
| `bimanual.py` | `BimanualMixin` 双臂 prefix |
| `errors.py` | 兼容版本异常 |
| `io_utils.py` | 通用 IO |
| `process.py` | 子进程 helper |
| `pedal.py` | DAgger 踏板输入 |
| `keyboard_input.py` | 键盘 listener |

### 7.1 HubMixin

统一：

- `save_pretrained(local_dir)`
- `from_pretrained(hub_id_or_path)`
- `push_to_hub(repo_id)`

用于 Config、Policy、ProcessorPipeline。

### 7.2 register_third_party_plugins

在 record/replay/rollout 启动时扫描 entry points，加载外部 robot/policy/env 注册。

---

## 8. Common 模块

```
common/
├── train_utils.py      # save_checkpoint, load_training_state
├── control_utils.py    # record_loop 共享逻辑
└── wandb_utils.py      # W&B init/log
```

`control_utils.record_loop` 被 `lerobot-record` 与部分 rollout 策略复用。

---

## 9. Model 子模块

**路径**：`model/` — 跨 policy 共享的小模块（非顶层 policy 实现）。

具体工具函数供个别 VLA/port 使用；新策略优先放在 `policies/{name}/` 自包含。

---

## 10. Templates

**路径**：`templates/`

- `lerobot_modelcard_template.md` — Hub model card
- `lerobot_rewardmodel_modelcard_template.md`

`PreTrainedPolicy.generate_model_card()` 填充。

---

## 11. 测试与 CI 相关

| 路径 | 说明 |
|------|------|
| `tests/` | pytest；fixtures 在 `tests/fixtures/` |
| `tests/utils.py` | 硬件 skip 装饰器 |
| `Makefile` | E2E：`make test-end-to-end` |

开发者：

```bash
uv sync --locked --extra test --extra dev
uv run pytest tests -svv --maxfail=10
pre-commit run --all-files
```

---

## 12. Docker

```
docker/
├── Dockerfile.user      # 用户镜像
└── Dockerfile.internal  # CI 镜像
```

benchmark 专用镜像：如 `Dockerfile.benchmark.robomme` 解决 numpy 冲突。

---

## 13. 示例

### 13.1 设备自动选择

```python
from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available

device = auto_select_torch_device()
print(device, "amp:", is_amp_available(device))
```

### 13.2 插件注册扫描

```python
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
# 之后 CLI 可解析第三方 @register_subclass 类型
```

### 13.3 SARM 训练（命令骨架）

```bash
uv sync --locked --extra training --extra sarm

lerobot-train \
  --reward_model.type=sarm \
  --dataset.repo_id=user/long_horizon_demo \
  --steps=30000
```

---

## 14. 文档维护说明

| 官方用户文档 | 本技术文档 |
|--------------|------------|
| `docs/source/*.mdx` | `docs/technical/*.md` |
| 安装、硬件图文教程 | 架构、API、调用链 |
| Hugging Face 站点构建 | 仓库内 Markdown 速查 |

当 codebase 升级（如 `CODEBASE_VERSION`、新 policy type）时，应同步更新：

1. [README 索引](./README.md) 注册表
2. [05-policies](./05-policies.md) / [02-config](./02-core-types-and-config.md)
3. [10-cli-reference](./10-cli-reference.md)

---

## 返回索引

[← 文档索引](./README.md)
