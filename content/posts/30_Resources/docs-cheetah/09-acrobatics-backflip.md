---
title: "09 特技控制 BackFlip 与 FrontJump"
subtitle: ""
date: 2026-07-06T12:00:00+08:00
draft: false
authors: [Steven]
description: "BackFlip 后空翻、FrontJump 前跳轨迹回放与 DataReader 特技控制实现。"
summary: "后空翻与前跳特技控制与轨迹回放详解。"
tags: [cheetah, robots]
categories: [docs cheetah, robots]
series: [cheetah-docs]
weight: 10
series_weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 09 — 特技控制：BackFlip 与 FrontJump

## 1. 模块边界

```
user/MIT_Controller/Controllers/BackFlip/
├── DataReader.hpp/.cpp      # 预录轨迹加载
├── DataReadCtrl.hpp         # 回放基类
├── BackFlipCtrl.hpp/.cpp    # 后空翻
└── FrontJumpCtrl.hpp/.cpp   # 前跳

FSM_States/
├── FSM_State_BackFlip.h/.cpp
└── FSM_State_FrontJump.h/.cpp
```

**思路**：离线优化/录制完整轨迹（关节角、力矩、力），在线 **开环回放 + 关节 PD**，Landing 相检测触地后切换。

---

## 2. DataReader — 轨迹二进制格式

### 2.1 plan_offsets 列布局（22 列）

| 偏移 | 枚举 | 列内容 |
|------|------|--------|
| 0–6 | `q0_offset` | 浮基 x,z,yaw + 前髋/膝 + 后髋/膝（对称简化） |
| 7–13 | `qd0_offset` | 对应速度 |
| 14–17 | `tau_offset` | 前髋、前膝、后髋、后膝 力矩 |
| 18–21 | `force_offset` | 前 x/z、后 x/z 力 |

### 2.2 方法完整列表

| 方法 | 说明 |
|------|------|
| `DataReader(RobotType, FSM_StateName)` | 按机器人类型选默认 plan 文件 |
| `load_control_plan(filename)` | 二进制读入 `plan_buffer` |
| `unload_control_plan()` | 释放 buffer |
| `get_initial_configuration()` | 返回 `plan_buffer + q0_offset` |
| `get_plan_at_time(timestep)` | 第 k 行 22 维 float 指针 |
| `plan_timesteps` | 总行数（public） |
| `plan_cols` | 静态常量 22 |

**数据文件**：
- 后空翻 Mini：`config/mc_flip.dat`
- 前跳 Mini：`config/front_jump_pitchup_v2.dat`

---

## 3. DataReadCtrl 基类 — 完整 API

| 方法 | 说明 |
|------|------|
| `DataReadCtrl(DataReader*, dt)` | `_key_pt_step = ceil(dt*1000)` |
| `~DataReadCtrl()` | 析构 |
| `OneStep(curr_time, b_preparation, command)` | **纯虚** |
| `FirstVisit(curr_time)` | `_ctrl_start_time`, 重置 iteration |
| `LastVisit()` | 空 |
| `EndOfPhase(legData)` | 时间 > `_end_time-2dt` 或膝角>2.0 且膝速>2.0 |
| `SetParameter()` | Kp=10, Kd=1（关节） |

**Protected 成员**：`_key_pt_step`, `current_iteration`, `pre_mode_count`, `_end_time=5.5`, `_q_knee_max=2.0`, `_qdot_knee_max=2.0`

---

## 4. BackFlipCtrl

### 4.1 方法

| 方法 | 说明 |
|------|------|
| `BackFlipCtrl(DataReader*, dt)` | 构造 |
| `~BackFlipCtrl()` | 析构 |
| `OneStep(...)` | 调 `_update_joint_command` 并写 LegController |
| `_update_joint_command()` | protected：核心回放逻辑 |

### 4.2 `_update_joint_command` 算法（常量来自源码）

| 阶段 | 条件 | 行为 |
|------|------|------|
| Preparation | `pre_mode_count < 2000` 或 `b_preparation` | 保持初始 pose，`tau_mult=0` |
| Playback | 否则 | 读 plan 行，对称映射到四腿 |
| Landing ramp | `iteration >= 600` | s 从 0→1 插值到固定 landing 角 |
| Abduction | 全程 | landing 时 s 控制外展 ±0.2 rad |

**增益**：正常 Kp/Kd = 10/1；landing Kp/Kd = 25/1.5  
**力矩**：`tau_mult=1.2`，左右腿各分一半 hip/knee tau

### 4.3 控制流程

1. **等待**：进入后约 6 tick 稳定  
2. **Preparation**（RC `BACKFLIP_PRE`）：预备姿势  
3. **Playback**：0.5 kHz（2ms）索引 plan  
   - `tauFeedForward` ← plan  
   - `qDes`, `qdDes` ← plan + PD 增益  
4. **Landing**（iteration > 600）：`EndOfPhase()` 检测膝关节 contact → 渐退  
5. FSM 转 BALANCE 或 RECOVERY  

---

## 5. FrontJumpCtrl

与 `BackFlipCtrl` **相同 public API**，不同预录轨迹与 `DataReader` 状态名。

---

## 6. FSM_State_BackFlip / FrontJump

标准 FSM 五件套 + `testTransition()`。

**BackFlip 内部**：
- `DataReader` robot type `BACKFLIP`  
- RC `BACKFLIP_PRE` → preparation  
- 完成 → `checkTransition` → BALANCE/LOCOMOTION/RECOVERY  

---

## 7. 与 MPC/WBC 对比

| | Locomotion | BackFlip |
|---|------------|----------|
| 规划 | 在线 MPC | 离线轨迹 |
| 反馈 | 状态估计闭环 |  mainly  joint PD |
| 适用 | 周期 gait | 单次特技 |
| 安全 | SafetyChecker 全程 | 依赖 ESTOP/Recovery |

---

## 8. 使用注意

1. 仅在 **平坦硬地** 且 **足够空间** 测试  
2. 先 `BACKFLIP_PRE` 确认预备 pose  
3. 失败时切 `RECOVERY_STAND`（RC 或 control_mode 6）  
4. 轨迹与 **Mini Cheetah** 机体参数绑定，不可直接用于 Cheetah 3  

---

上一章：[08-fsm-and-mit-controller.md](./08-fsm-and-mit-controller.md)  
下一章：[10-robot-runtime-and-simulation.md](./10-robot-runtime-and-simulation.md)
