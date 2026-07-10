---
title: "12 核心 API 完整参考"
subtitle: ""
date: 2026-07-10T17:44:00+08:00
draft: false
authors: [Steven]
description: "9 个核心抽象基类的全部 public 方法与属性速查。"
summary: "LeRobot 核心基类完整 API 参考。"
tags: [lerobot, robots]
categories: [docs lerobot, robots]
series: [lerobot-docs]
weight: 12
series_weight: 12
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 12 — 核心 API 完整参考

> 本章列出 LeRobot **9 个核心抽象基类**的全部 **public** 方法/属性。  
> 私有方法（`_` 前缀）及子类特有扩展见各实现文件。  
> 源码路径均相对于 `src/lerobot/`。

---

## 1. PreTrainedPolicy

**文件**：`policies/pretrained.py`  
**继承**：`nn.Module`, `HubMixin`, `ABC`  
**子类必须定义**：`config_class`, `name`

| 成员 | 类型 | 签名 / 说明 | 用途 |
|------|------|-------------|------|
| `__init__` | 构造 | `(config: PreTrainedConfig, *inputs, **kwargs)` | 校验并保存 config |
| `save_pretrained` | 实例 | `(save_directory, *, state_dict=None, repo_id=None, push_to_hub=False, ...)` | 保存 config.json + model.safetensors |
| `from_pretrained` | 类方法 | `(cls, pretrained_name_or_path, *, config=None, strict=False, ...)` | Hub/本地加载；`eval()` 模式 |
| `get_optim_params` | **抽象** | `(self) -> dict` | 优化器参数组（如 backbone 低 LR） |
| `reset` | **抽象** | `(self) -> None` | episode 边界清空 queue/cache |
| `forward` | **抽象** | `(batch) -> tuple[Tensor, dict \| None]` | 训练损失 + 日志 dict |
| `predict_action_chunk` | **抽象** | `(batch, **kwargs) -> Tensor` | 形状 `(B, chunk, action_dim)` |
| `select_action` | **抽象** | `(batch, **kwargs) -> Tensor` | 单步动作 `(B, action_dim)` |
| `push_model_to_hub` | 实例 | `(cfg: TrainPipelineConfig, peft_model=None, state_dict=None)` | 上传权重+训练配置+model card |
| `generate_model_card` | 实例 | `(dataset_repo_id, model_type, license, tags, cfg=None)` | 生成 HF ModelCard |
| `wrap_with_peft` | 实例 | `(peft_config=None, peft_cli_overrides=None)` | LoRA 等 PEFT 包装 |
| `push_to_hub` | 继承 | HubMixin 通用上传 | 直接 push 目录 |

**类属性**：`config_class`, `name`

---

## 2. Robot

**文件**：`robots/robot.py`

| 成员 | 类型 | 说明 |
|------|------|------|
| `__init__` | 构造 | `(config: RobotConfig)` — 加载 calibration JSON |
| `__str__` | 魔术 | `"<id> ClassName"` |
| `__enter__` / `__exit__` | 上下文 | connect / disconnect |
| `__del__` | 析构 | 若仍连接则 disconnect |
| `observation_features` | **抽象 property** | 观测 schema |
| `action_features` | **抽象 property** | 动作 schema |
| `is_connected` | **抽象 property** | 连接状态 |
| `connect` | **抽象** | `(calibrate: bool = True)` |
| `is_calibrated` | **抽象 property** | 标定状态 |
| `calibrate` | **抽象** | 运行标定流程 |
| `configure` | **抽象** | 一次性电机/控制配置 |
| `get_observation` | **抽象** | `-> RobotObservation` |
| `send_action` | **抽象** | `(action) -> RobotAction`（可能 clip） |
| `disconnect` | **抽象** | 释放资源 |

**实例属性**：`robot_type`, `id`, `calibration_dir`, `calibration_fpath`, `calibration`

---

## 3. Teleoperator

**文件**：`teleoperators/teleoperator.py`

| 成员 | 类型 | 说明 |
|------|------|------|
| `__init__` | 构造 | `(config: TeleoperatorConfig)` |
| `__str__` | 魔术 | 可读字符串 |
| `__enter__` / `__exit__` | 上下文 | connect / disconnect |
| `__del__` | 析构 | 安全 disconnect |
| `action_features` | **抽象 property** | 输出动作 schema |
| `feedback_features` | **抽象 property** | 反馈输入 schema |
| `is_connected` | **抽象 property** | |
| `connect` | **抽象** | `(calibrate=True)` |
| `is_calibrated` | **抽象 property** | |
| `calibrate` | **抽象** | |
| `configure` | **抽象** | |
| `get_action` | **抽象** | `-> RobotAction` |
| `send_feedback` | **抽象** | `(feedback: dict) -> None` |
| `disconnect` | **抽象** | |

---

## 4. Camera

**文件**：`cameras/camera.py`

| 成员 | 类型 | 说明 |
|------|------|------|
| `__init__` | 构造 | `(config: CameraConfig)` — 存 fps/width/height |
| `__enter__` / `__exit__` | 上下文 | connect / disconnect |
| `__del__` | 析构 | |
| `is_connected` | **抽象 property** | |
| `find_cameras` | **抽象 static** | `-> list[dict]` 系统发现 |
| `connect` | **抽象** | `(warmup=True)` |
| `read` | **抽象** | 同步阻塞读帧 `-> ndarray` |
| `async_read` | **抽象** | `(timeout_ms)` 异步读 |
| `read_latest` | 具体 | `(max_age_ms=500)` 非阻塞最新帧 |
| `disconnect` | **抽象** | |

**属性**：`fps`, `width`, `height`

---

## 5. ProcessorStep

**文件**：`processor/pipeline.py`

| 成员 | 类型 | 说明 |
|------|------|------|
| `transition` | property | 当前处理中的 EnvTransition |
| `__call__` | **抽象** | `(transition) -> transition` |
| `get_config` | 具体 | JSON 可序列化配置 |
| `state_dict` | 具体 | 持久化 tensor 状态 |
| `load_state_dict` | 具体 | 恢复状态 |
| `reset` | 具体 | 清空内部状态（默认 no-op） |
| `transform_features` | **抽象** | 静态 feature shape 传播 |

**专用子类**（均继承 `ProcessorStep`）：  
`ObservationProcessorStep`, `ActionProcessorStep`, `RobotActionProcessorStep`, `PolicyActionProcessorStep`, `RewardProcessorStep`, `DoneProcessorStep`, `TruncatedProcessorStep`, `InfoProcessorStep`, `ComplementaryDataProcessorStep`, `IdentityProcessorStep`

---

## 6. DataProcessorPipeline

**文件**：`processor/pipeline.py`  
**别名**：`PolicyProcessorPipeline`, `RobotProcessorPipeline`

| 成员 | 类型 | 说明 |
|------|------|------|
| `__call__` | 具体 | `input -> output` 全流程 |
| `step_through` | 具体 | 逐步 yield transition（调试） |
| `get_config` | 具体 | 管道 JSON 配置 |
| `state_dict` | 具体 | 各 step 状态聚合 |
| `load_state_dict` | 具体 | |
| `save_pretrained` | 具体 | JSON + step_*.safetensors |
| `from_pretrained` | 类方法 | Hub/本地加载 |
| `from_config` | 类方法 | 内存 config 构建 |
| `__len__` | 具体 | step 数量 |
| `__getitem__` | 具体 | 索引/切片 step |
| `register_before_step_hook` | 具体 | 调试钩子 |
| `unregister_before_step_hook` | 具体 | |
| `register_after_step_hook` | 具体 | |
| `unregister_after_step_hook` | 具体 | |
| `reset` | 具体 | 重置所有 step |
| `__repr__` | 具体 | |
| `transform_features` | 具体 | 链式 feature 推断 |
| `process_observation` | 具体 | 仅处理 observation 字段 |
| `process_action` | 具体 | 仅处理 action |
| `process_reward` | 具体 | |
| `process_done` | 具体 | |
| `process_truncated` | 具体 | |
| `process_info` | 具体 | |
| `process_complementary_data` | 具体 | |
| `push_to_hub` | 继承 | HubMixin |

**字段**：`steps`, `name`, `to_transition`, `to_output`, hooks

---

## 7. LeRobotDataset

**文件**：`datasets/lerobot_dataset.py`

| 成员 | 类型 | 说明 |
|------|------|------|
| `__init__` | 构造 | 读模式加载 Hub/本地；可选 episode 过滤 |
| `create` | 类方法 | 创建写模式空数据集 |
| `resume` | 类方法 | 追加 episode |
| `fps` | property | |
| `num_frames` | property | |
| `num_episodes` | property | |
| `features` | property | feature spec |
| `hf_dataset` | property | 底层 HF Dataset |
| `absolute_to_relative_idx` | property | index 映射 |
| `add_frame` | 写 | 追加帧到 buffer |
| `save_episode` | 写 | 刷盘 parquet+video |
| `clear_episode_buffer` | 写 | 丢弃 buffer |
| `has_pending_frames` | 写 | buffer 非空？ |
| `finalize` | 写 | 关闭 writer |
| `__len__` | 读 | 帧数 |
| `__getitem__` | 读 | 完整样本（delta+decode） |
| `get_raw_item` | 读 | 无 decode/transform |
| `select_columns` | 读 | 列子集 |
| `set_image_transforms` | 读 | 设置增强 |
| `clear_image_transforms` | 读 | |
| `push_to_hub` | | 上传数据集 |
| `__repr__` | | |

---

## 8. InferenceEngine

**文件**：`rollout/inference/base.py`  
**实现**：`SyncInferenceEngine`, `RTCInferenceEngine`

| 成员 | 类型 | 说明 |
|------|------|------|
| `start` | **抽象** | 启动（RTC 开后台线程） |
| `stop` | **抽象** | 关闭 |
| `reset` | **抽象** | episode 状态清零 |
| `get_action` | **抽象** | `(obs_frame?) -> Tensor \| None` |
| `notify_observation` | 默认 | 异步引擎更新 obs |
| `pause` | 默认 | 暂停后台 |
| `resume` | 默认 | 恢复 |
| `ready` | property 默认 | warmup 完成 |
| `failed` | property 默认 | 不可恢复错误 |

---

## 9. MotorsBusBase 与 SerialMotorsBus

### 9.1 MotorsBusBase（最小接口）

**文件**：`motors/motors_bus.py`

| 成员 | 说明 |
|------|------|
| `__init__(port, motors, calibration=None)` | |
| `connect(handshake=True)` | **抽象** |
| `disconnect(disable_torque=True)` | **抽象** |
| `is_connected` | **抽象 property** |
| `read(data_name, motor)` | **抽象** 单寄存器读 |
| `write(data_name, motor, value)` | **抽象** 单寄存器写 |
| `sync_read(data_name, motors=None)` | **抽象** 批量读 |
| `sync_write(data_name, values)` | **抽象** 批量写 |
| `enable_torque(motors=None, num_retry=0)` | **抽象** |
| `disable_torque(motors=None, num_retry=0)` | **抽象** |
| `read_calibration()` | **抽象** |
| `write_calibration(calibration_dict, cache=True)` | **抽象** |

### 9.2 SerialMotorsBus 扩展 public API

LeRobot 机器人实际使用 `MotorsBus = SerialMotorsBus`：

| 方法 | 说明 |
|------|------|
| `__len__` | motor 数量 |
| `__repr__` | |
| `models` / `ids` | property：型号与 ID 列表 |
| `scan_port(port)` | 类方法：扫描端口上 motor ID |
| `setup_motor(...)` | 配置单个 motor ID |
| `configure_motors()` | 批量配置 |
| `torque_disabled(...)` | 上下文：临时关力矩 |
| `set_timeout(timeout_ms)` | 通信超时 |
| `get_baudrate()` / `set_baudrate(baudrate)` | 波特率 |
| `is_calibrated()` | 是否与文件标定一致 |
| `reset_calibration(motors=None)` | 重置标定 |
| `set_half_turn_homings(...)` | 半圈 homing |
| `record_ranges_of_motion(...)` | 交互式记录运动范围 |
| `ping(motor, ...)` | 探测 motor |
| `broadcast_ping(...)` | 广播 ping |

**子类**：`DynamixelMotorsBus`, `FeetechMotorsBus`（串口）；`DamiaoMotorsBus`, `RobstrideMotorsBus`（CAN）

---

## 10. LeRobotDatasetMetadata 公共 API

**文件**：`datasets/dataset_metadata.py`

| 成员 | 说明 |
|------|------|
| `create(...)` | 类方法：空 meta |
| `finalize()` / `__del__` | flush episode parquet |
| `ensure_readable()` | 写→读切换 reload |
| `filter_episodes(predicate, candidates?)` | 谓词过滤 episode index |
| `get_data_file_path(ep_index)` | data parquet 路径 |
| `get_video_file_path(ep_index, vid_key)` | mp4 路径 |
| `save_episode_tasks(tasks)` | 写 tasks.parquet |
| `save_episode(...)` | episode 元数据 + 计数器 |
| `update_video_info(...)` | ffprobe 更新 feature |
| `update_chunk_settings` / `get_chunk_settings` | chunk 配置 |
| `get_task_index(task)` | 字符串→index |

**属性**：`features`, `fps`, `robot_type`, `image_keys`, `video_keys`, `depth_keys`, `camera_keys`, `has_language_columns`, `total_episodes`, `total_frames`, `total_tasks`, `data_path`, `video_path`, `chunks_size`

---

## 11. PreTrainedConfig 核心 API

**文件**：`configs/policies.py`

| 成员 | 说明 |
|------|------|
| `validate()` | 校验 features/normalization |
| `from_pretrained(path)` | 加载 config.json |
| `save_pretrained(dir)` | 保存 config |
| `get_optimizer_preset()` | 默认 OptimizerConfig |
| `get_scheduler_preset()` | 默认 LRSchedulerConfig |
| `choice_registry` | Draccus 多态 |

关键字段见 [02-core-types-and-config.md](./02-core-types-and-config.md)。

---

## 12. 工厂函数索引

| 函数 | 文件 | 作用 |
|------|------|------|
| `make_policy` | `policies/factory.py` | 构建 Policy |
| `make_pre_post_processors` | 同上 | pre/post pipeline |
| `make_dataset` | `datasets/factory.py` | 训练 Dataset |
| `make_train_eval_datasets` | 同上 | train/eval 划分 |
| `make_env` | `envs/factory.py` | 向量仿真环境 |
| `make_robot_from_config` | `robots/utils.py` | Robot 实例 |
| `make_teleoperator_from_config` | `teleoperators/utils.py` | Teleoperator |
| `make_cameras_from_configs` | `cameras/utils.py` | 相机 dict |
| `make_optimizer_and_scheduler` | `optim/factory.py` | 优化器+调度器 |
| `create_inference_engine` | `rollout/inference/factory.py` | Sync/RTC |
| `create_strategy` | `rollout/strategies/factory.py` | Rollout 策略 |
| `build_rollout_context` | `rollout/context.py` | 部署上下文 |
| `make_reward_model` | `rewards/factory.py` | 奖励模型 |

---

## 返回索引

[← README](./README.md) · [算法与数学 →](./13-algorithms-and-mathematics.md)
