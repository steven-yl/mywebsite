# 01 — 架构总览

## 1. 项目定位

**LeRobot** 是 Hugging Face 维护的 PyTorch 机器人学习库，目标是将**数据集、策略模型、硬件控制、仿真评估**统一到一套 Python 接口中，并通过 Hugging Face Hub 共享模型与数据。

### 解决的核心问题

| 痛点 | LeRobot 方案 |
|------|-------------|
| 机器人数据格式碎片化 | **LeRobotDataset v3**（Parquet + MP4）统一 schema |
| 硬件接口各异 | **`Robot` / `Teleoperator` 抽象** + ChoiceRegistry 配置 |
| 策略与数据预处理耦合 | **`ProcessorPipeline`** 可序列化、可 Hub 共享 |
| 训练/评估/部署脚本分散 | **统一 CLI**（`lerobot-train` / `eval` / `rollout`） |
| 大模型依赖重 | **可选 extras** + 工厂懒加载 |

### 技术栈

- **Python 3.12+** · **PyTorch 2.7+** · **draccus**（配置/CLI）· **Hugging Face Hub** · **Gymnasium** · **uv**（包管理）

---

## 2. 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  CLI 层 (scripts/)                                          │
│  lerobot-train | eval | record | rollout | edit-dataset …   │
├─────────────────────────────────────────────────────────────┤
│  配置层 (configs/) — Draccus ChoiceRegistry + HubMixin      │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ policies │ datasets │ processor│ envs     │ rollout         │
│ rewards  │          │          │          │ async_inference │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│  硬件抽象 (robots / teleoperators / cameras / motors)       │
├─────────────────────────────────────────────────────────────┤
│  基础设施 (utils / optim / transport / types)               │
└─────────────────────────────────────────────────────────────┘
```

### 各层职责

| 层 | 模块 | 职责 |
|----|------|------|
| **CLI** | `scripts/` | 解析配置、编排工厂、执行主循环 |
| **配置** | `configs/` | 声明式 hyperparameter、多态组件选择 |
| **策略** | `policies/` | `PreTrainedPolicy` 子类：训练 loss、推理 action |
| **数据** | `datasets/` | 读写 LeRobotDataset、统计、视频编解码 |
| **处理器** | `processor/` | `EnvTransition` 上的可组合变换 |
| **环境** | `envs/` | Gymnasium 向量环境封装 |
| **硬件** | `robots/` 等 | 真机 connect / get_observation / send_action |
| **部署** | `rollout/` | 真机策略循环、RTC、录制策略 |
| **扩展** | `rewards/`, `rl/`, `annotations/` | 奖励模型、在线 RL、VLM 标注 |

---

## 3. 核心设计：EnvTransition 统一数据载体

LeRobot 用 **`EnvTransition`**（见 `types.py`）作为机器人、仿真、策略、训练批次之间的**通用字典**：

```python
# src/lerobot/types.py（概念示意）
class TransitionKey(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    REWARD = "reward"
    DONE = "done"
    TRUNCATED = "truncated"
    INFO = "info"
    COMPLEMENTARY_DATA = "complementary_data"
```

**为什么需要它？**

- 机器人观测是 `dict[str, Any]`（关节 + 图像）
- Gym 动作是 `np.ndarray`
- 策略输出是 `torch.Tensor`
- 训练 batch 是 `dict[str, Tensor]`

Processor 管道通过 `batch_to_transition` / `transition_to_batch` 等转换器，在这些表示之间**无损切换**，避免每个脚本重复写归一化/设备迁移逻辑。

---

## 4. 核心设计：ChoiceRegistry 多态配置

所有可替换组件（策略、机器人、环境、优化器等）继承 **`draccus.ChoiceRegistry`**：

```python
@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    chunk_size: int = 100
    ...
```

CLI 选择：

```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

预训练加载：

```bash
lerobot-eval --policy.path=lerobot/pi0_libero_finetuned
```

**优点**：声明式、可序列化到 JSON、与 Hub 集成。  
**缺点**：新类型必须 `@register_subclass` 并在工厂中注册或走插件发现。

---

## 5. 端到端生命周期

### 5.1 数据采集（Record）

```
Teleoperator.get_action()
  → teleop_action_processor
  → robot.send_action()
Robot.get_observation()
  → robot_observation_processor
  → LeRobotDataset.add_frame() / save_episode()
```

入口：`lerobot-record`（见 [10-cli-reference.md](./10-cli-reference.md)）

### 5.2 训练（Train）

```
TrainPipelineConfig
  → make_train_eval_datasets()
  → make_policy() + make_pre_post_processors()
  → Accelerate 训练循环: batch → preprocessor → forward() → backward
  → checkpoint / push_to_hub
```

入口：`lerobot-train`

### 5.3 仿真评估（Eval）

```
EvalPipelineConfig
  → make_env() + make_policy()
  → vectorized rollouts → success rate / reward
```

入口：`lerobot-eval`（仅仿真；真机用 rollout）

### 5.4 真机部署（Rollout）

```
RolloutConfig
  → build_rollout_context()
  → Strategy (base/sentry/dagger/…)
  → InferenceEngine (sync/rtc)
  → robot.send_action()
```

入口：`lerobot-rollout`

备选路径：**async_inference** — GPU 服务器跑策略，机器人端轻量 gRPC 客户端。

---

## 6. 模块关联矩阵

|  | datasets | processor | policies | envs | robots | rollout |
|--|:--------:|:---------:|:--------:|:----:|:------:|:-------:|
| **lerobot-train** | ✓ | ✓ | ✓ | 可选 eval | — | — |
| **lerobot-eval** | 可选录制 | ✓ | ✓ | ✓ | — | — |
| **lerobot-record** | ✓ 写入 | ✓ | — | — | ✓ | — |
| **lerobot-replay** | ✓ 读取 | ✓ | — | — | ✓ | — |
| **lerobot-rollout** | 可选录制 | ✓ | ✓ | — | ✓ | ✓ |

---

## 7. 设计权衡与适用场景

### 7.1 单体库 vs 微服务

| 方案 | 适用场景 |
|------|----------|
| **同进程**（record/train/rollout） | 单机开发、SO-101 等低成本臂、快速迭代 |
| **async_inference gRPC** | VLA 大模型需独立 GPU；机器人算力有限 |
| **RTC 推理引擎** | 慢策略（flow-matching chunk）需与控制频率解耦 |

### 7.2 Sync vs RTC 推理

| 引擎 | 行为 | 适用 |
|------|------|------|
| **sync** | 每 tick 同步 `select_action()` | ACT、Diffusion 等较快策略 |
| **rtc** | 后台线程预测 action chunk，主循环 pop | PI0、SmolVLA 等；**必须**用于 relative action 策略 |

> `SyncInferenceEngine` 会拒绝启用 `RelativeActionsProcessorStep` 的策略。

### 7.3 数据集：Parquet + MP4 vs 纯图像

| 格式 | 优点 | 缺点 |
|------|------|------|
| **MP4 视频** | 存储小、Hub 传输快 | 解码开销、需 ffmpeg/torchcodec |
| **嵌入 Parquet 图像** | 随机访问简单 | 体积大 |
| **StreamingLeRobotDataset** | 内存友好 | 不适合随机 episode 采样 |

### 7.4 策略类别选型（概览）

| 类别 | 代表 | 数据需求 | 算力 | 真机迁移 |
|------|------|----------|------|----------|
| 模仿学习 | ACT, Diffusion | 中等演示 | 中 | 较好 |
| RL | TDMPC, HIL-SERL | 交互/奖励 | 中–高 | 任务相关 |
| VLA | PI0, SmolVLA, GR00T | 多模态+语言 | 高 | SOTA 方向 |

详细策略解读见 [05-policies.md](./05-policies.md)。

---

## 8. 可选依赖（extras）策略

`pyproject.toml` 将策略、环境、硬件拆成 **extras**，例如：

```bash
uv sync --locked --extra training --extra pi --extra feetech
```

**设计原因**：

1. VLA 依赖 `transformers`、flash-attn 等，体积极大
2. 仿真环境（LIBERO、MetaWorld）有平台/版本约束
3. 工厂使用 **懒 import**，未安装 extra 时不加载对应模块

常见组合：

| 任务 | 建议 extras |
|------|-------------|
| 仅读数据集 | `dataset` |
| SO-101 录数据 | `core_scripts`, `feetech` |
| ACT 训练 | `training`, `act`（若有独立 extra）或 base + diffusers |
| PI0 微调 | `training`, `pi` |
| LIBERO 评估 | `evaluation`, `libero`, `pi` |

---

## 9. Hub 优先工作流

配置与权重通过 **HubMixin** 持久化：

```
checkpoint/
├── config.json           # PreTrainedConfig
├── model.safetensors     # 策略权重
├── policy_preprocessor/  # ProcessorPipeline JSON + safetensors
└── policy_postprocessor/
```

`--policy.path` 在 train / eval / rollout 中统一加载路径，减少「训练配置与部署配置不一致」问题。

---

## 10. 扩展点一览

| 扩展什么 | 基类 / 注册 | 文档 |
|----------|-------------|------|
| 新策略 | `PreTrainedConfig` + `PreTrainedPolicy` | [05-policies.md](./05-policies.md) |
| 新机器人 | `RobotConfig` + `Robot` | [07-hardware-layer.md](./07-hardware-layer.md) |
| 新 Processor 步骤 | `@ProcessorStepRegistry.register` | [04-processor-pipeline.md](./04-processor-pipeline.md) |
| 新仿真环境 | `EnvConfig` | [08-environments.md](./08-environments.md) |
| 第三方插件 | `--env.discover_packages_path` | [02-core-types-and-config.md](./02-core-types-and-config.md) |

---

## 11. 示例：最小推理流程（可运行）

```python
"""最小示例：加载 Hub 数据集并打印一帧 action shape。
依赖: uv sync --extra dataset
"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")
sample = dataset[0]
print(f"action shape: {sample['action'].shape}")
print(f"keys: {list(sample.keys())}")
```

```python
"""最小示例：从 Hub 加载 ACT 策略配置（需网络）。
依赖: 已安装 lerobot
"""
from lerobot.policies.factory import make_policy_config

cfg = make_policy_config("act")
print(f"policy type: {cfg.type}")
print(f"n_obs_steps: {cfg.n_obs_steps}")
```

---

## 下一章

- 类型与配置细节 → [02-core-types-and-config.md](./02-core-types-and-config.md)
- 数据流实现 → [03-dataset-system.md](./03-dataset-system.md)
