---
title: "07 硬件层"
subtitle: ""
date: 2026-07-10T17:44:00+08:00
draft: false
authors: [Steven]
description: "Robot、Teleoperator、Camera、Motors 抽象与真机控制接口。"
summary: "LeRobot 硬件抽象层：机器人、遥操作、相机与电机。"
tags: [lerobot, robots]
categories: [docs lerobot, robots]
series: [lerobot-docs]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 07 — 硬件层（Robot / Teleoperator / Camera / Motors）

## 1. 设计目标

**硬件无关的控制接口**：上层 `lerobot-record`、`lerobot-rollout` 只依赖抽象 API，具体电机协议、相机 SDK 封装在子类中。

```
┌──────────────┐     get_action      ┌─────────────────┐
│ Teleoperator │ ──────────────────► │ record / rollout │
└──────────────┘                     └────────┬────────┘
                                                │ send_action / get_observation
┌──────────────┐                                ▼
│   Cameras    │ ◄── embedded in ──── ┌─────────────────┐
└──────────────┘                       │     Robot       │
                                       └────────┬────────┘
                                                │
                                       ┌────────▼────────┐
                                       │   MotorsBus     │
                                       └─────────────────┘
```

---

## 2. Robot 抽象

**文件**：`robots/robot.py`

### 2.1 必须实现的 API

| 方法/属性 | 作用 |
|-----------|------|
| `observation_features` | 观测 schema（dtype/shape 描述） |
| `action_features` | 动作 schema |
| `connect()` | 打开电机、相机 |
| `disconnect()` | 释放资源 |
| `configure()` | 一次性参数（PID 等） |
| `get_observation()` | → `RobotObservation` dict |
| `send_action(action)` | ← `RobotAction` dict |
| `calibrate()` | 交互式标定 |
| `is_connected` / `is_calibrated` | 状态 |

### 2.2 校准持久化

路径：`$HF_LEROBOT_CALIBRATION/robots/{robot_name}/{id}.json`

含关节零位、限位、方向等，`MotorsBus` 读写。

### 2.3 RobotConfig

**文件**：`robots/config.py`

- `id`：多机区分
- `calibration_dir`：覆盖默认校准目录
- `cameras: dict[str, CameraConfig]` — **必须**指定 width/height/fps

---

## 3. 全部机器人实现（17 种）

| `--robot.type` | 类 | 通信/电机 |
|----------------|-----|-----------|
| `so100_follower` | `SO100Follower` | Feetech STS3215 |
| `so101_follower` | `SO101Follower` | 同上（配置区分型号） |
| `koch_follower` | `KochFollower` | Dynamixel |
| `omx_follower` | `OmxFollower` | Dynamixel |
| `openarm_follower` | `OpenArmFollower` | Damiao CAN |
| `rebot_b601_follower` | `RebotB601Follower` | motorbridge CAN |
| `hope_jr_arm` | `HopeJrArm` | Feetech |
| `hope_jr_hand` | `HopeJrHand` | Feetech |
| `lekiwi` | `LeKiwi` | Feetech + 全向轮 |
| `lekiwi_client` | `LeKiwiClient` | ZMQ 远程 LeKiwi |
| `reachy2` | `Reachy2Robot` | Pollen Reachy SDK |
| `earthrover_mini_plus` | `EarthRoverMiniPlus` | 云端 WebRTC API |
| `unitree_g1` | `UnitreeG1` | Unitree SDK2 + 可选 ONNX  locomotion |
| `bi_so_follower` | `BiSOFollower` | 双臂 SO（`BimanualMixin`） |
| `bi_openarm_follower` | `BiOpenArmFollower` | 双臂 OpenArm |
| `bi_rebot_b601_follower` | `BiRebotB601Follower` | 双臂 Rebot |

**工厂**：`make_robot_from_config()`（`robots/utils.py`）

---

## 4. Teleoperator 抽象

**文件**：`teleoperators/teleoperator.py`

| API | 作用 |
|-----|------|
| `get_action()` | 人类输入 → `RobotAction` |
| `send_feedback(obs)` | 可选力反馈/镜像（如 G1） |
| `action_features` / `feedback_features` | schema |
| 生命周期 | 同 Robot：connect / calibrate / disconnect |

### 4.1 TeleopEvents（`teleoperators/utils.py`）

录制时键盘/手柄事件：

- `success`, `failure`, `rerecord_episode`
- `is_intervention`, `terminate_episode`

### 4.2 全部遥操作设备（19 种）

| `--teleop.type` | 类 | 输入 |
|-----------------|-----|------|
| `so100_leader` / `so101_leader` | `SOLeader` | Feetech 主从臂 |
| `koch_leader` | `KochLeader` | Dynamixel |
| `omx_leader` | `OmxLeader` | Dynamixel |
| `openarm_leader` | `OpenArmLeader` | Damiao |
| `openarm_mini` | `OpenArmMini` | Feetech |
| `rebot_102_leader` | `RebotArm102Leader` | FashionStar UART |
| `bi_so_leader` | `BiSOLeader` | 双 SO leader |
| `bi_openarm_leader` | `BiOpenArmLeader` | 双 OpenArm |
| `bi_openarm_mini` | `BiOpenArmMini` | 双 mini |
| `bi_rebot_102_leader` | `BiRebot102Leader` | 双 Rebot 102 |
| `keyboard` | `KeyboardTeleop` | pynput |
| `keyboard_ee` | `KeyboardEndEffectorTeleop` | 键盘 EE 增量 |
| `keyboard_rover` | `KeyboardRoverTeleop` | 移动底盘 |
| `gamepad` | `GamepadTeleop` | HID 手柄 |
| `phone` | `Phone` | HEBI 手机 IMU |
| `homunculus_glove` / `homunculus_arm` | Homunculus | 串口自定义协议 |
| `reachy2_teleoperator` | `Reachy2Teleoperator` | Reachy SDK |
| `unitree_g1` | `UnitreeG1Teleoperator` | 外骨骼 + IK |

---

## 5. Camera 抽象

**文件**：`cameras/camera.py`

| 方法 | 作用 |
|------|------|
| `connect(warmup=True)` | 打开设备 |
| `read()` | 同步读一帧 |
| `async_read(timeout_ms)` | 异步读 |
| `read_latest(max_age_ms)` | 最新帧（降延迟） |
| `find_cameras()` (classmethod) | 发现设备 |
| `fps`, `width`, `height` | 属性 |

### 5.1 四种后端

| type | 实现 | 场景 |
|------|------|------|
| `opencv` | `OpenCVCamera` | USB 摄像头、视频文件 |
| `intelrealsense` | `RealSenseCamera` | RealSense RGB-D |
| `reachy2_camera` | `Reachy2Camera` | Reachy2  onboard |
| `zmq` | `ZMQCamera` | 远程相机（LeKiwi） |

工厂：`make_cameras_from_configs()` — 嵌入 `RobotConfig.cameras`。

---

## 6. Motors 驱动

**文件**：`motors/motors_bus.py`

### 6.1 类层次

```
MotorsBusBase
├── SerialMotorsBus
│   ├── DynamixelMotorsBus    (dynamixel_sdk)
│   └── FeetechMotorsBus      (scservo_sdk)
├── DamiaoMotorsBus           (python-can, MIT mode)
└── RobstrideMotorsBus        (python-can, MIT mode)
```

### 6.2 核心 API（MotorsBusBase）

| 方法 | 作用 |
|------|------|
| `connect()` / `disconnect()` | 总线生命周期 |
| `read()` / `write()` | 单寄存器 |
| `sync_read()` / `sync_write()` | 批量读写（低延迟） |
| `enable_torque()` / `disable_torque()` | 力矩模式 |
| `calibrate()` | 校准流程 |

### 6.3 数据类型

- `Motor` — id, model, norm_mode
- `MotorCalibration` — 零位、范围
- `MotorNormMode` — `RANGE_0_100`, `RANGE_M100_100`, `DEGREES`

### 6.4 使用映射

| 驱动 | 使用方 |
|------|--------|
| Feetech | SO-100/101, Hope Jr, LeKiwi, OpenArm mini |
| Dynamixel | Koch, OMX |
| Damiao | OpenArm |
| motorbridge | Rebot B601 |
| Robstride | **已实现，尚未接入 robot** |

交互式校准 GUI：`motors/calibration_gui.py`

---

## 7. 与 record / replay 集成

### 7.1 record_loop 数据路径

```
teleop.get_action()
  → teleop_action_processor(action, obs)
  → 写入 dataset["action"]
  → robot_action_processor(action, obs)
  → robot.send_action()

robot.get_observation()
  → robot_observation_processor(obs)
  → 写入 dataset["observation.*"]
```

### 7.2 replay

```
dataset[i]["action"]
  → 按 feature names 解码为 dict
  → robot_action_processor
  → robot.send_action()
```

**无 teleoperator**；sleep 对齐 `dataset.fps`。

### 7.3 LeKiwi 特殊配置

`teleop` 可为 **list**：键盘控底盘 + leader 控 arm。

---

## 8. 相关 CLI

| 命令 | 作用 |
|------|------|
| `lerobot-find-port` | 列出串口 |
| `lerobot-find-cameras` | 发现相机 |
| `lerobot-setup-motors` | 配置 motor ID |
| `lerobot-setup-can` | CAN 接口 setup |
| `lerobot-calibrate` | 标定 robot/teleop |
| `lerobot-find-joint-limits` | 遥操作探测限位 |
| `lerobot-teleoperate` | 纯遥操作不录数据 |

---

## 9. 示例

### 9.1 列出 observation features（mock）

```python
"""连接前可 inspect config 推断 features（无需硬件）。"""
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

cfg = SO101FollowerConfig(port="/dev/ttyACM0", id="my_arm")
# 实际 features 在 Robot 实例化后来自 observation_features 属性
print(cfg.type, cfg.cameras)
```

### 9.2 录制命令模板

```bash
uv sync --locked --extra core_scripts --extra feetech

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader \
  --dataset.repo_id=user/so101_pick \
  --dataset.num_episodes=10 \
  --dataset.single_task="pick the cube"
```

### 9.3 回放

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --dataset.repo_id=user/so101_pick \
  --dataset.episode=0
```

---

## 10. 扩展自定义 Robot

1. 子类 `RobotConfig` + `@register_subclass("my_robot")`
2. 子类 `Robot` 实现全部抽象方法
3. 在 `make_robot_from_config` 添加分支，或依赖 `make_device_from_device_class` 插件
4. 可选：自定义 `RobotProcessorPipeline` 处理坐标系/安全限位

详见官方 [integrate_hardware.mdx](https://huggingface.co/docs/lerobot/integrate_hardware)。

---

## 下一章

- 仿真环境 → [08-environments.md](./08-environments.md)
- CLI 详情 → [10-cli-reference.md](./10-cli-reference.md)
