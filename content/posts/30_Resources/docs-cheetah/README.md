---
title: "Cheetah-Software 技术文档索引"
subtitle: ""
date: 2026-07-06T12:00:00+08:00
draft: false
authors: [Steven]
description: "MIT Cheetah 四足机器人控制与仿真软件栈的完整技术解读索引，涵盖架构、动力学、MPC、WBC 与运行时。"
summary: "Cheetah-Software 技术文档索引、阅读路径与源码目录对照。"
tags: [cheetah, robots]
categories: [docs cheetah, robots]
series: [cheetah-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# Cheetah-Software 技术文档索引

MIT Cheetah 四足机器人控制与仿真软件栈的完整技术解读。本文档集从总体架构出发，逐层展开动力学、估计、MPC、全身控制、状态机与运行时系统。

---

## 文档结构

| 章节 | 文件 | 主题 |
|------|------|------|
| 总览 | [00-architecture-overview.md](./00-architecture-overview.md) | 整体架构、模块关系、技术选型对比、适用场景 |
| 动力学 | [01-dynamics-and-kinematics.md](./01-dynamics-and-kinematics.md) | 空间向量代数、FloatingBaseModel、Quadruped、执行器、仿真器 |
| 腿控与步态 | [02-leg-control-and-gait.md](./02-leg-control-and-gait.md) | LegController、GaitScheduler、FootSwingTrajectory、DesiredStateCommand |
| 状态估计 | [03-state-estimation.md](./03-state-estimation.md) | 姿态/位置速度 KF、接触估计、Cheater 模式 |
| Convex MPC | [04-convex-mpc.md](./04-convex-mpc.md) | 质心凸 MPC、Gait、SolverMPC、qpOASES/JCQP |
| Vision/Sparse MPC | [05-vision-mpc-and-sparse-cmpc.md](./05-vision-mpc-and-sparse-cmpc.md) | 地形感知 MPC、SparseCMPC、FootstepPlanner |
| 全身控制 WBC | [06-whole-body-control.md](./06-whole-body-control.md) | KinWBC、WBIC、Task/Contact、LocomotionCtrl |
| 平衡控制器 | [07-balance-controller.md](./07-balance-controller.md) | BalanceController QP、VBL 变体、ReferenceGRF |
| FSM 与主控 | [08-fsm-and-mit-controller.md](./08-fsm-and-mit-controller.md) | ControlFSM、11 种状态、SafetyChecker、MIT_Controller |
| 特技控制 | [09-acrobatics-backflip.md](./09-acrobatics-backflip.md) | BackFlip、FrontJump、DataReader 轨迹回放 |
| 运行时与仿真 | [10-robot-runtime-and-simulation.md](./10-robot-runtime-and-simulation.md) | RobotRunner、HardwareBridge、Simulation、LCM、实时线程 |
| 数学与碰撞 | [11-math-collision-utilities.md](./11-math-collision-utilities.md) | 姿态工具、滤波、碰撞检测、ContactImpulse/SpringDamper |
| 参数与示例 | [12-user-parameters-and-example-controllers.md](./12-user-parameters-and-example-controllers.md) | MIT_UserParameters 全表、JPos/InvDyn 示例控制器 |

---

## 阅读顺序建议

**速查**：README 索引 → 对应章节「模块边界」与 API 表。  
**进阶**：00 总览 → 01 动力学 → 03 估计 → 04 MPC → 06 WBC → 08 FSM → 10 运行时。  
**调参**：12 参数表 + SimControlPanel 三栏 YAML 对照。

---

## 快速导航：按使用场景

| 场景 | 推荐阅读路径 |
|------|-------------|
| 编写自定义控制器 | 总览 → 运行时 → 腿控 → 动力学 |
| 理解 Trot  locomotion | 步态 → Convex MPC → WBC → FSM |
| 调试状态估计 | 状态估计 → 运行时（cheater_mode） |
| 仿真开发 | 运行时 → 碰撞 → 数学工具 |
| 真机部署 | 运行时 → FSM → SafetyChecker |

---

## 源码目录对照

```
Cheetah-Software/
├── common/          # 共享库 biomimetics：动力学、控制器、MPC、碰撞
├── robot/           # RobotRunner、HardwareBridge、RT 接口
├── sim/             # Qt/OpenGL 仿真器
├── user/            # 用户控制器（MIT_Controller、JPos_Controller 等）
├── lcm-types/       # LCM 消息定义
├── config/          # YAML 参数配置
└── third-party/     # qpOASES、OSQP、JCQP、SOEM 等
```

---

## 相关外部文档

- 项目入门：`documentation/getting_started.md`
- Mini Cheetah 真机：`documentation/running_mini_cheetah.md`
- 论文参考：Convex MPC locomotion (MIT Cheetah 3)、Mini Cheetah 全身控制

---

*文档基于 Cheetah-Software 源码梳理，覆盖 common、robot、sim、user/MIT_Controller 全部公开 API 与核心算法。*

**文档版本**：共 13 章（00–12），约 3500+ 行；涵盖动力学、估计、MPC/WBC、FSM、运行时、碰撞与参数示例。
