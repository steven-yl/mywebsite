# 02 — 核心类型与配置系统

## 1. 模块边界

| 路径 | 职责 |
|------|------|
| `types.py` | 运行时类型别名、`TransitionKey`、`EnvTransition` |
| `configs/types.py` | 特征 schema：`FeatureType`、`PolicyFeature`、`NormalizationMode` |
| `configs/policies.py` | `PreTrainedConfig` 基类 |
| `configs/train.py` | `TrainPipelineConfig` 训练顶层配置 |
| `configs/eval.py` | `EvalPipelineConfig` |
| `configs/default.py` | `DatasetConfig`、`WandBConfig`、`EvalConfig`、`PeftConfig` |
| `configs/dataset.py` | `DatasetRecordConfig` 录制专用 |
| `configs/parser.py` | Draccus 增强：`wrap()`、插件、`--*.path` |
| `configs/video.py` | 视频编码器配置 |
| `configs/rewards.py` | `RewardModelConfig` |
| `configs/recipe.py` | VLA 语言 `TrainingRecipe` YAML |

---

## 2. 核心类型（`types.py`）

### 2.1 TransitionKey

枚举 transition 字典的键，避免魔法字符串：

| 成员 | 值 | 含义 |
|------|-----|------|
| `OBSERVATION` | `"observation"` | 观测（dict 或嵌套 tensor） |
| `ACTION` | `"action"` | 动作（多种类型，见下） |
| `REWARD` | `"reward"` | 标量奖励 |
| `DONE` | `"done"` | episode 终止 |
| `TRUNCATED` | `"truncated"` | 时间截断 |
| `INFO` | `"info"` | 辅助信息 dict |
| `COMPLEMENTARY_DATA` | `"complementary_data"` | 如 teleop action、语言 |

### 2.2 类型别名

```python
PolicyAction = torch.Tensor          # 策略输出，通常 (B, action_dim) 或 chunk
RobotAction = dict[str, Any]         # 如 {"joint_1.pos": 45.2, ...}
EnvAction = np.ndarray               # Gym 连续/离散动作向量
RobotObservation = dict[str, Any]    # 关节 + 相机图像
BatchType = dict[str, Any]           # DataLoader 批次
```

**为什么 action 有多种类型？**

- 硬件层用**命名关节字典**，便于不同机器人 schema 对齐到 dataset feature
- 策略层用 **Tensor**，便于 GPU 批量计算
- Gym 用 **ndarray**，符合 Gymnasium API

Processor 负责在边界处转换（见 [04-processor-pipeline.md](./04-processor-pipeline.md)）。

### 2.3 EnvTransition

```python
EnvTransition = TypedDict("EnvTransition", {
    "observation": RobotObservation | None,
    "action": PolicyAction | RobotAction | EnvAction | None,
    "reward": float | torch.Tensor | None,
    "done": bool | torch.Tensor | None,
    "truncated": bool | torch.Tensor | None,
    "info": dict[str, Any] | None,
    "complementary_data": dict[str, Any] | None,
})
```

每个 `ProcessorStep.__call__(transition)` 接收并返回完整的 `EnvTransition`，可按需只修改部分字段。

---

## 3. 特征 Schema（`configs/types.py`）

### 3.1 FeatureType

| 枚举 | 用途 |
|------|------|
| `STATE` | 本体状态（关节角等） |
| `VISUAL` | RGB 图像 |
| `ENV` | 环境状态（仿真） |
| `ACTION` | 动作向量 |
| `REWARD` | 奖励 |
| `LANGUAGE` | 任务语言 / 对话 |

### 3.2 PolicyFeature

```python
@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]
```

描述策略输入/输出的**语义类型与形状**。`make_policy()` 从 dataset metadata 或 env 推断并填充 `input_features` / `output_features`。

### 3.3 NormalizationMode

| 模式 | 公式（示意） | 适用 |
|------|-------------|------|
| `MEAN_STD` | `(x - μ) / σ` | 近似高斯的状态/动作 |
| `MIN_MAX` | `(x - min) / (max - min)` | 有界关节 |
| `IDENTITY` | 不变换 | 已归一化数据 |
| `QUANTILES` / `QUANTILE10` | 分位数缩放 | 鲁棒归一化、RA-BC 等 |

映射由 `PreTrainedConfig.normalization_mapping` 指定，例如 `{"STATE": NormalizationMode.MEAN_STD}`。

---

## 4. PreTrainedConfig 基类

**文件**：`configs/policies.py`

### 4.1 关键字段

| 字段 | 说明 |
|------|------|
| `n_obs_steps` | 输入历史帧数 |
| `input_features` / `output_features` | 特征 schema；可从 dataset 推断 |
| `device` | `cuda` / `cpu` / `mps` |
| `use_amp` | 混合精度训练 |
| `use_peft` | 是否 LoRA 等 PEFT |
| `pretrained_path` | Hub ID 或本地目录 |
| `optimizer` / `scheduler` | 嵌套 `OptimizerConfig` / `LRSchedulerConfig` |
| `observation_delta_indices` / `action_delta_indices` | 多步预测时间索引 |

### 4.2 核心方法

| 方法 | 作用 |
|------|------|
| `validate()` | 校验 feature 与 normalization 一致性 |
| `from_pretrained(path)` | 从 Hub/本地加载 JSON |
| `save_pretrained(dir)` | 保存 config.json |

子类通过 `@PreTrainedConfig.register_subclass("name")` 注册 CLI 类型名。

---

## 5. TrainPipelineConfig

**文件**：`configs/train.py`

顶层训练配置，聚合：

```python
@dataclass
class TrainPipelineConfig:
    dataset: DatasetConfig
    policy: PreTrainedConfig | None = None
    reward_model: RewardModelConfig | None = None  # 可选
    env: EnvConfig | None = None                   # 训练中 sim eval
    eval: EvalConfig
    wandb: WandBConfig
    peft: PeftConfig
    # steps, batch_size, save_freq, env_eval_freq, ...
```

**`validate()`** 检查：policy 或 reward_model 二选一、dataset repo 存在、eval 与 env 配置一致等。

---

## 6. 配置解析器（`configs/parser.py`）

### 6.1 `@wrap()` 装饰器

包装 CLI `main()`：

1. 解析 draccus CLI + YAML
2. 处理 `--policy.path` → 加载预训练 config 并合并 CLI override
3. 加载 `--env.discover_packages_path` 插件
4. 调用被装饰函数

### 6.2 关键函数

| 函数 | 作用 |
|------|------|
| `wrap()` | CLI 入口装饰器 |
| `load_plugin(path)` | 动态 import 包以注册 ChoiceRegistry 子类 |
| `get_path_arg(args, field)` | 提取 `--policy.path` 等 |
| `get_cli_overrides(args, field)` | 获取 CLI 覆盖项供 PEFT 等使用 |

### 6.3 CLI 覆盖示例

```bash
lerobot-train \
  --policy.type=act \
  --policy.chunk_size=50 \
  --dataset.repo_id=user/my_dataset \
  --steps=100000 \
  --batch_size=64 \
  --policy.device=cuda
```

预训练微调：

```bash
lerobot-train \
  --policy.path=lerobot/act_aloha \
  --dataset.repo_id=user/new_task \
  --steps=20000
```

---

## 7. ChoiceRegistry 完整注册表

### 7.1 策略（16 种，`policies/factory.py`）

`tdmpc`, `diffusion`, `act`, `multi_task_dit`, `vqbet`, `pi0`, `pi0_fast`, `pi05`, `gaussian_actor`, `smolvla`, `groot`, `xvla`, `wall_x`, `eo1`, `molmoact2`, `vla_jepa`

### 7.2 奖励模型（4 种）

`reward_classifier`, `sarm`, `robometer`, `topreward`

### 7.3 环境（12 种）

`aloha`, `pusht`, `gym_manipulator`, `libero`, `libero_plus`, `metaworld`, `robocasa`, `vlabench`, `isaaclab_arena`, `robotwin`, `robomme`

### 7.4 机器人（17 种）

见 [07-hardware-layer.md](./07-hardware-layer.md)

### 7.5 遥操作（19 种）

见 [07-hardware-layer.md](./07-hardware-layer.md)

### 7.6 相机（4 种）

`opencv`, `intelrealsense`, `reachy2_camera`, `zmq`

### 7.7 优化器（5 种）

`adam`, `adamw`, `sgd`, `xvla-adamw`, `multi_adam`

### 7.8 学习率调度（3 种）

`diffuser`, `vqbet`, `cosine_decay_with_warmup`

### 7.9 Rollout 策略（5 种）

`base`, `sentry`, `highlight`, `episodic`, `dagger`

### 7.10 推理引擎（2 种）

`sync`, `rtc`

### 7.11 数据集编辑操作（9 种）

`delete_episodes`, `split`, `merge`, `remove_feature`, `modify_tasks`, `convert_image_to_video`, `recompute_stats`, `reencode_videos`, `info`

---

## 8. 插件扩展

第三方包可通过 import 时注册子类：

```python
# my_robot_pkg/config.py
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("my_robot")
@dataclass
class MyRobotConfig(RobotConfig):
    port: str = "/dev/ttyUSB0"
```

启动时加载：

```bash
lerobot-record \
  --env.discover_packages_path=my_robot_pkg \
  --robot.type=my_robot \
  --robot.port=/dev/ttyUSB0
```

`register_third_party_plugins()` 在 record/replay 等脚本中自动扫描常见入口。

---

## 9. HubMixin 与配置持久化

`PreTrainedConfig` 与 `ProcessorPipeline` 均实现 `HubMixin`：

- 保存：`save_pretrained(local_dir)` → `config.json`
- 加载：`from_pretrained(hub_id_or_path)`
- 上传：`push_to_hub(repo_id)`

与 `model.safetensors` 同目录，保证**权重与预处理配置版本一致**。

---

## 10. 示例：自定义配置类

```python
"""定义并解析自定义 ACT 配置片段。"""
from dataclasses import dataclass
from lerobot.policies.act.configuration_act import ACTConfig

@dataclass
class MyACTConfig(ACTConfig):
    # ACTConfig 已 register_subclass("act")
    chunk_size: int = 30

# CLI 等效: --policy.type=act --policy.chunk_size=30
cfg = MyACTConfig()
cfg.validate()
print(cfg.chunk_size)
```

```python
"""从 Hub 加载策略配置（需网络）。"""
from lerobot.configs.policies import PreTrainedConfig

cfg = PreTrainedConfig.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")
print(type(cfg).__name__, cfg.type)
```

---

## 下一章

- 数据集 → [03-dataset-system.md](./03-dataset-system.md)
- 处理器 → [04-processor-pipeline.md](./04-processor-pipeline.md)
