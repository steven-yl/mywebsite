---
title: "第四章：物理引擎（mimickit/engines/）"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第四章：物理引擎（mimickit/engines/）。"
summary: "第四章：物理引擎（mimickit/engines/）。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 4
series_weight: 4
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第四章：物理引擎（`mimickit/engines/`）

## 4.1 模块边界

`engines/` 包提供**统一的物理仿真抽象接口**，屏蔽 Isaac Gym、Isaac Lab、Newton 的后端差异。环境与学习算法仅通过 `Engine` API 交互，不直接调用模拟器 SDK。

```
engines/
├── engine.py              # 抽象基类 Engine
├── engine_builder.py      # 工厂：按 engine_name 构建
├── isaac_gym_engine.py    # Isaac Gym 实现
├── isaac_lab_engine.py    # Isaac Lab 实现
├── newton_engine.py       # Newton 实现
├── video_recorder.py      # 录制抽象
├── isaac_gym_recorder.py
├── isaac_lab_recorder.py
└── newton_recorder.py
```

---

## 4.2 控制模式 `ControlMode`

| 枚举值 | 含义 | 动作空间 |
|--------|------|---------|
| `none` | 无控制 | - |
| `pos` | 位置 PD 控制（默认） | 目标关节位置 |
| `vel` | 速度控制 | 目标关节速度 |
| `torque` | 力矩控制 | 关节力矩 |
| `pd_explicit` | 显式 PD（位置+增益） | 扩展动作向量 |

### 4.2.1 为什么默认用 `pos` 模式

位置 PD 控制将 RL 动作映射为**目标关节角度**，物理引擎内置 PD 控制器跟踪。这样：

- 动作空间有界，与关节限位对齐
- 策略输出更平滑，适合运动模仿
- 与 DeepMimic 原论文设置一致

---

## 4.3 抽象类 `Engine`

### 4.3.1 生命周期方法

| 方法 | 说明 |
|------|------|
| `get_name()` | 返回引擎名称字符串 |
| `create_env(env_id)` | 创建单个并行环境实例 |
| `create_obj(env_id, obj_type, asset_file, name, ...)` | 在环境中加载刚体/关节体 |
| `initialize_sim()` | 初始化仿真（分配缓冲区、准备 GPU） |
| `set_cmd(obj_id, cmd)` | 设置控制命令（关节目标） |
| `step()` | 执行一个控制周期的物理仿真 |
| `render()` | 渲染一帧（带帧率限制） |

### 4.3.2 相机方法

| 方法 | 说明 |
|------|------|
| `set_camera_pose(pos, look_at)` | 设置相机位置与朝向 |
| `get_camera_pos()` | 获取相机位置 |
| `get_camera_dir()` | 获取相机方向 |

### 4.3.3 状态查询方法

| 方法 | 返回值 |
|------|--------|
| `get_timestep()` | 控制周期$\Delta t$|
| `get_num_envs()` | 并行环境数 |
| `get_gravity()` | 重力向量 |
| `get_root_pos(obj_id)` | 根位置 [num_envs, 3] |
| `get_root_rot(obj_id)` | 根旋转四元数 [num_envs, 4] |
| `get_root_vel(obj_id)` | 根线速度 |
| `get_root_ang_vel(obj_id)` | 根角速度 |
| `get_dof_pos(obj_id)` | 关节位置 |
| `get_dof_vel(obj_id)` | 关节速度 |
| `get_body_pos(obj_id)` | 各 body 世界位置 |
| `get_body_rot(obj_id)` | 各 body 世界旋转 |
| `get_body_vel(obj_id)` | 各 body 线速度 |
| `get_body_ang_vel(obj_id)` | 各 body 角速度 |
| `get_contact_forces(obj_id)` | 接触力 |
| `get_ground_contact_forces(obj_id)` | 地面接触力（跌倒检测） |

### 4.3.4 状态设置方法

| 方法 | 说明 |
|------|------|
| `set_root_pos(env_ids, obj_id, pos)` | 设置根位置 |
| `set_root_rot(env_ids, obj_id, rot)` | 设置根旋转 |
| `set_root_vel(env_ids, obj_id, vel)` | 设置根速度 |
| `set_root_ang_vel(env_ids, obj_id, ang_vel)` | 设置根角速度 |
| `set_dof_pos(env_ids, obj_id, pos)` | 设置关节位置 |
| `set_dof_vel(env_ids, obj_id, vel)` | 设置关节速度 |
| `set_body_vel(env_ids, obj_id, vel)` | 设置 body 速度 |
| `set_body_ang_vel(env_ids, obj_id, ang_vel)` | 设置 body 角速度 |

### 4.3.5 对象类型 `ObjType`

| 值 | 含义 |
|----|------|
| `rigid` | 刚体（箱子、球等） |
| `articulated` | 关节体（角色） |

---

## 4.4 `engine_builder.py`

```python
def build_engine(config, num_envs, device, visualize, record_video):
    engine_name = config["engine_name"]
    if engine_name == "isaac_gym":
        return IsaacGymEngine(...)
    elif engine_name == "isaac_lab":
        return IsaacLabEngine(...)
    elif engine_name == "newton":
        return NewtonEngine(...)
```

---

## 4.5 三种后端对比

| 特性 | Isaac Gym | Isaac Lab | Newton |
|------|-----------|-----------|--------|
| 并行环境 | ✅ GPU 数千 env | ✅ | 视版本而定 |
| 默认 sim_freq | 120 Hz | 依配置 | 240 Hz |
| 资产格式 | MJCF `.xml` | USD `.usd` | MJCF/URDF |
| 安装复杂度 | 中等 | 高（需 Isaac Sim） | 中等 |
| 推荐场景 | 默认训练 | 需要 Omniverse 生态 | 替代/实验 |
| 本项目测试版本 | 最新 | commit `2ed331a` | v1.0.0 |

### 4.5.1 选型建议

- **大规模 RL 训练**：Isaac Gym + `num_envs=4096`
- **USD 资产 / Isaac Sim 集成**：Isaac Lab
- **对比实验 / 新物理引擎**：Newton

---

## 4.6 仿真步进时序

```
每个 RL step（控制周期 Δt_ctrl = 1/30s）:
    for substep in range(sim_freq / control_freq):  # 例：120/30 = 4 次
        Engine.step()  # 物理子步 Δt_sim = 1/120s
    Engine.render()    # 可选
```

示意图：

```
时间轴 →
|--sim--|--sim--|--sim--|--sim--|  ← 1 个 control step (30Hz)
  120Hz  120Hz  120Hz  120Hz
```

---

## 4.7 视频录制

`VideoRecorder` 抽象类，各引擎有对应实现：

- `IsaacGymRecorder`
- `IsaacLabRecorder`
- `NewtonRecorder`

通过 `--video true` 启用无头录制，视频由 Logger 管理输出。

---

## 4.8 与环境的交互

```python
# char_env.py 典型调用链
self._engine.set_cmd(char_id, actions)   # 应用 PD 目标
self._engine.step()                       # 物理仿真
root_pos = self._engine.get_root_pos(char_id)
dof_pos = self._engine.get_dof_pos(char_id)
```

---

## 4.9 参考资料

| 资源 | 链接 |
|------|------|
| Isaac Gym 文档 | https://developer.nvidia.com/isaac-gym |
| Isaac Lab 安装 | https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html |
| Newton 安装 | https://newton-physics.github.io/newton/guide/installation.html |
| PD 控制原理 | [PID controller (Wikipedia)](https://en.wikipedia.org/wiki/PID_controller) |

---

[← 动画模块](03-animation-module.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：环境 →](05-environments.md)
