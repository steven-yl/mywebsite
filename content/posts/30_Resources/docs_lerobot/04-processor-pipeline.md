# 04 — Processor 处理器管道

## 1. 为什么需要 Processor？

机器人学习数据在多个表示之间流转：

```
Robot dict  →  Dataset parquet  →  Training batch  →  Policy tensor  →  Robot dict
```

若每个脚本独立做归一化、设备迁移、相对/绝对动作转换，会导致：

- 训练与部署**预处理不一致**（常见 bug）
- 无法随 checkpoint **Hub 共享**预处理逻辑

**ProcessorPipeline** 将变换链抽象为可序列化的 `ProcessorStep` 列表。

---

## 2. 模块边界

```
processor/
├── pipeline.py           # ProcessorStep, DataProcessorPipeline, Registry
├── converters.py         # batch ↔ transition 转换
├── factory.py            # 默认 robot/teleop pipelines
├── normalize_processor.py
├── batch_processor.py
├── device_processor.py
├── relative_action_processor.py
├── policy_robot_bridge.py
├── delta_action_processor.py
├── env_processor.py
├── hil_processor.py
├── tokenizer_processor.py
└── ...
```

---

## 3. ProcessorStep 抽象基类

**文件**：`pipeline.py`

| 方法 | 必须/可选 | 作用 |
|------|-----------|------|
| `__call__(transition)` | **必须** | 输入输出 `EnvTransition` |
| `transform_features(features)` | **必须** | 静态推断 feature shape/type 变化 |
| `get_config()` | **必须** | JSON 可序列化配置 |
| `state_dict()` | 可选 | 有状态步骤（如 norm stats） |
| `load_state_dict(state)` | 可选 | 恢复状态 |
| `reset()` | 可选 | episode 边界清 cache |

### 3.1  specialized 基类

| 基类 | 只修改 |
|------|--------|
| `ObservationProcessorStep` | `transition["observation"]` |
| `ActionProcessorStep` | `transition["action"]` |
| `RobotActionProcessorStep` | RobotAction |
| `PolicyActionProcessorStep` | PolicyAction |
| `ComplementaryDataProcessorStep` | complementary_data |

---

## 4. DataProcessorPipeline

泛型管道：`DataProcessorPipeline[TInput, TOutput]`

### 4.1 执行流程

```
input
  → to_transition(input)     # 如 batch_to_transition
  → step_1(transition)
  → step_2(transition)
  → ...
  → to_output(transition)    # 如 transition_to_batch
  → output
```

### 4.2 核心 API

| 方法 | 作用 |
|------|------|
| `__call__(data)` | 端到端处理 |
| `step_through(data)` | 生成器：逐步 yield 中间 transition（调试） |
| `transform_features(initial)` | 链式 feature 推断 |
| `save_pretrained(dir)` / `from_pretrained(path)` | Hub 持久化 |
| `state_dict()` / `load_state_dict()` | 聚合步骤状态 |
| `register_before/after_step_hook` | 调试钩子 |
| `reset()` | 重置所有步骤 |

### 4.3 类型别名

| 别名 | 用途 |
|------|------|
| `PolicyProcessorPipeline` | 策略 pre/postprocessor |
| `RobotProcessorPipeline` | 机器人 obs/action 管道 |

默认 checkpoint 名：`policy_preprocessor`、`policy_postprocessor`

---

## 5. 全部注册步骤（30 个）

| Registry 名 | 类 | 作用 |
|-------------|-----|------|
| `observation_processor` | `VanillaObservationProcessorStep` | Gym obs → LeRobot 格式（CHW float） |
| `normalizer_processor` | `NormalizerProcessorStep` | 按 stats 归一化 obs/action |
| `unnormalizer_processor` | `UnnormalizerProcessorStep` | 反归一化（策略输出后） |
| `rename_observations_processor` | `RenameObservationsProcessorStep` | 重命名 obs keys |
| `device_processor` | `DeviceProcessorStep` | tensor → device/dtype |
| `to_batch_processor` | `AddBatchDimensionProcessorStep` | 全 transition 加 batch 维 |
| `to_batch_processor_observation` | `AddBatchDimensionObservationStep` | 仅 observation |
| `to_batch_processor_action` | `AddBatchDimensionActionStep` | 仅 action |
| `to_batch_processor_complementary_data` | `AddBatchDimensionComplementaryDataStep` | 仅 complementary |
| `tokenizer_processor` | `TokenizerProcessorStep` | 语言 tokenize |
| `action_tokenizer_processor` | `ActionTokenizerProcessorStep` | 动作 tokenize（FAST 等） |
| `render_messages_processor` | `RenderMessagesStep` | 渲染对话模板 |
| `smolvla_new_line_processor` | `NewLineTaskProcessorStep` | SmolVLA task 换行 |
| `relative_actions_processor` | `RelativeActionsProcessorStep` | action − state（掩码） |
| `absolute_actions_processor` | `AbsoluteActionsProcessorStep` | 相对 → 绝对 |
| `robot_action_to_policy_action_processor` | `RobotActionToPolicyActionProcessorStep` | dict → tensor |
| `policy_action_to_robot_action_processor` | `PolicyActionToRobotActionProcessorStep` | tensor → dict |
| `map_tensor_to_delta_action_dict` | `MapTensorToDeltaActionDictStep` | tensor → delta dict |
| `map_delta_action_to_robot_action` | `MapDeltaActionToRobotActionStep` | delta → robot cmd |
| `torch2numpy_action_processor` | `Torch2NumpyActionProcessorStep` | 送 Gym |
| `numpy2torch_action_processor` | `Numpy2TorchActionProcessorStep` | Gym → torch |
| `libero_processor` | `LiberoProcessorStep` | LIBERO obs 格式 |
| `isaaclab_arena_processor` | `IsaaclabArenaProcessorStep` | Isaac Lab Arena obs |
| `add_teleop_action_as_complementary_data` | `AddTeleopActionAsComplimentaryDataStep` | teleop 写入 complementary |
| `add_teleop_action_as_info` | `AddTeleopEventsAsInfoStep` | 键盘/手柄事件 → info |
| `image_crop_resize_processor` | `ImageCropResizeProcessorStep` | HIL 图像裁剪 |
| `time_limit_processor` | `TimeLimitProcessorStep` | 超时 truncated |
| `gym_hil_adapter_processor` | `GymHILAdapterProcessorStep` | HIL Gym 适配 |
| `gripper_penalty_processor` | `GripperPenaltyProcessorStep` | 夹爪惩罚 shaping |
| `intervention_action_processor` | `InterventionActionProcessorStep` | 人工干预覆盖 action |
| `reward_classifier_processor` | `RewardClassifierProcessorStep` | 分类器 reward |

**未注册**：`IdentityProcessorStep`（空操作，默认 pipeline 使用）

---

## 6. 关键步骤详解

### 6.1 NormalizerProcessorStep

**是什么**：按 `meta/stats.json` 对指定 feature 做 MEAN_STD / MIN_MAX 等变换。

**为什么**：神经网络训练数值稳定；不同机器人关节范围统一。

**公式（MEAN_STD）**：

\[
x' = \frac{x - \mu}{\sigma + \epsilon}
\]

配置：`features`, `norm_map`, `stats`（来自 dataset 或 checkpoint state_dict）

### 6.2 RelativeActionsProcessorStep / AbsoluteActionsProcessorStep

**是什么**：将动作从**绝对关节空间**转为**相对当前状态**的增量，或反向。

**为什么**：部分 VLA（PI0 等）在 relative 空间预测更稳；部署时需加回 state。

**配对**：postprocessor 中 `AbsoluteActionsProcessorStep.relative_step` 在 load 后由 `_reconnect_relative_absolute_steps()` 重新链接。

### 6.3 Policy ↔ Robot Bridge

```
RobotAction {"joint_1.pos": v, ...}
  → RobotActionToPolicyActionProcessorStep
  → PolicyAction tensor [d]
  → ... policy ...
  → PolicyActionToRobotActionProcessorStep
  → RobotAction
```

feature 顺序由 dataset `names` 或 robot `action_features` 决定。

---

## 7. 转换器（`converters.py`）

| 函数 | 方向 |
|------|------|
| `batch_to_transition` | DataLoader batch → EnvTransition |
| `transition_to_batch` | EnvTransition → batch |
| `observation_to_transition` | RobotObservation → EnvTransition |
| `transition_to_robot_action` | → RobotAction |
| `policy_action_to_transition` | PolicyAction → EnvTransition |
| `transition_to_policy_action` | → PolicyAction |
| `create_transition(**kwargs)` | 构造部分填充的 transition |

---

## 8. 默认 Factory（`factory.py`）

| 函数 | 返回 |
|------|------|
| `make_default_teleop_action_processor()` | Identity teleop pipeline |
| `make_default_robot_action_processor()` | Identity robot action pipeline |
| `make_default_robot_observation_processor()` | Identity obs pipeline |
| `make_default_processors()` | 上述三者 tuple |

`lerobot-record` 在此基础上可替换为自定义 pipeline；feature schema 由 `aggregate_pipeline_dataset_features()` 推断。

---

## 9. 策略侧集成

`make_pre_post_processors()`（`policies/factory.py`）为每种策略构建：

1. **Preprocessor**：rename → normalize → relative → device → batch
2. **Postprocessor**：unnormalize → absolute → device → robot dict

保存于 checkpoint，**部署时必须加载**，否则动作尺度错误。

---

## 10. 序列化格式

```
policy_preprocessor/
├── config.json       # steps: [{registry_name, config}, ...]
└── step_0.safetensors  # 有 state 的步骤
```

`from_pretrained(..., overrides={...})` 可覆盖单步配置（如 device）。

---

## 11. 调试

- `step_through()` 逐步检查 transition
- 文档 `docs/source/debug_processor_pipeline.mdx` 对应官方调试指南
- `lerobot-imgtransform-viz` 可视化图像增强

---

## 12. 示例

### 12.1 手动管道

```python
"""构建最小 normalize + device 管道。"""
import torch
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.configs.types import NormalizationMode, FeatureType

stats = {
    "observation.state": {"mean": [0.0], "std": [1.0]},
}
norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}

pipe = DataProcessorPipeline(
    steps=[
        NormalizerProcessorStep(
            features={"observation.state": FeatureType.STATE},
            norm_map=norm_map,
            stats=stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ],
    to_transition=batch_to_transition,
    to_output=transition_to_batch,
)

batch = {"observation.state": torch.randn(2, 1)}
out = pipe(batch)
print(out["observation.state"].shape)
```

### 12.2 从策略 checkpoint 加载

```python
"""加载 Hub 策略的 preprocessor（需网络与对应 policy extra）。"""
from lerobot.processor.pipeline import DataProcessorPipeline

pre = DataProcessorPipeline.from_pretrained(
    "lerobot/act_aloha_sim_transfer_cube_human",
    config_filename="policy_preprocessor.json",
)
print(len(pre.steps), [s.__class__.__name__ for s in pre.steps])
```

---

## 下一章

- 策略工厂如何使用 processor → [05-policies.md](./05-policies.md)
- 录制时 pipeline → [07-hardware-layer.md](./07-hardware-layer.md)
