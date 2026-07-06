---
title: "01 动力学与运动学"
subtitle: ""
date: 2026-07-06T12:00:00+08:00
draft: false
authors: [Steven]
description: "空间向量代数、FloatingBaseModel 浮基动力学、Quadruped 参数封装与 DynamicsSimulator 仿真步进。"
summary: "Cheetah 浮基多体动力学与四足运动学模型详解。"
tags: [cheetah, robots]
categories: [docs cheetah, robots]
series: [cheetah-docs]
weight: 2
series_weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 01 — 动力学与运动学

## 1. 模块边界

| 路径 | 内容 |
|------|------|
| `common/include/Dynamics/spatial.h` | 空间向量代数原语 |
| `common/include/Dynamics/SpatialInertia.h` | 空间惯性 |
| `common/include/Dynamics/FloatingBaseModel.h` | 浮基多体动力学核心 |
| `common/include/Dynamics/Quadruped.h` | 四足参数封装 |
| `common/include/Dynamics/ActuatorModel.h` | 电机力矩模型 |
| `common/include/Dynamics/DynamicsSimulator.h` | 仿真步进 |
| `common/include/Dynamics/MiniCheetah.h` | Mini Cheetah CAD 参数 |
| `common/include/Dynamics/Cheetah3.h` | Cheetah 3 LMI 参数 |

**依赖**：`Math/orientation_tools` → `spatial` → `FloatingBaseModel` → `Quadruped` / `DynamicsSimulator`

---

## 2. 空间向量代数 (`spatial.h`)

### 2.1 为什么需要

传统 Newton-Euler 逐连杆写方程繁琐且易错。Featherstone **空间向量**将 6D 速度/力统一表示，使递归动力学（RNEA/ABA）为 O(n) 复杂度。

### 2.2 关键概念

**空间运动向量** \(\hat{v} = [ \omega^T,\ v^T ]^T\)（角速度在前）  
**空间力向量** \(\hat{f} = [ \tau^T,\ f^T ]^T\)

**空间变换** \(X\)：将向量从坐标系 A 变换到 B。

### 2.3 函数完整说明

| 函数 | 作用 | 典型调用 |
|------|------|----------|
| `spatialRotation(axis, theta)` | 绕轴空间旋转 6×6 | 关节变换 |
| `motionCrossMatrix(v)` | 运动叉乘矩阵 | 加速度传播 |
| `forceCrossMatrix(v)` | 力叉乘矩阵 | 力传播 |
| `motionCrossProduct(a,b)` | 快速叉乘 | ABA 内部 |
| `forceCrossProduct(a,b)` | 快速力叉乘 | ABA 内部 |
| `createSXform(R, r)` | 由旋转 R 与平移 r 构造 X | FK |
| `invertSXform(X)` | 空间变换求逆 | 父→子链路 |
| `jointMotionSubspace(joint, axis)` | 关节运动子空间 S | 约束投影 |
| `jointXform(joint, axis, q)` | 关节角 q 对应 X | FK |
| `rotInertiaOfBox(mass, dims)` | 均匀盒体惯性 | 建模 |
| `spatialToLinearVelocity(v, x)` | 空间速度 → 点线速度 | 足端速度 |
| `spatialToAngularVelocity(v)` | 提取 ω | 状态输出 |
| `spatialToLinearAcceleration(a,v)` | 经典线加速度 | 接触 |
| `sXFormPoint(X, p)` | 变换 3D 点 | 几何 |
| `forceToSpatialForce(f, p)` | 点力 → 空间力 | 外力施加 |

**关节类型** `JointType`：`Prismatic`, `Revolute`, `FloatingBase`, `Nothing`

---

## 3. SpatialInertia

### 3.1 是什么

6×6 空间惯性矩阵，统一表达质量、质心与转动惯量。

### 3.2 公式

给定质量 \(m\)、质心位置 \(\mathbf{c}\)（相对连杆原点）、关于质心的惯性张量 \(\mathbf{I}_c\)：

\[
\mathcal{I} = \begin{bmatrix}
\mathbf{I}_c + m[\mathbf{c}]_\times[\mathbf{c}]_\times^T & m[\mathbf{c}]_\times \\
m[\mathbf{c}]_\times^T & m\mathbf{I}
\end{bmatrix}
\]

其中 \([\mathbf{c}]_\times\) 为反对称矩阵（`ori::crossMatrix`）。

### 3.3 类方法

| 方法 | 说明 |
|------|------|
| `SpatialInertia(mass, com, inertia)` | 标量参数构造 |
| `SpatialInertia(Mat6)` | 矩阵构造 |
| `SpatialInertia(MassProperties)` | 10 元质量属性向量 |
| `SpatialInertia(Mat4 P)` | 伪惯性（Wensing 参数化） |
| `asMassPropertyVector()` | 导出 10 元向量 |
| `getMatrix()` / `setMatrix()` | 读写 6×6 |
| `addMatrix(mat)` | 累加惯性 |
| `getMass()` / `getCOM()` / `getInertiaTensor()` | 分量访问 |
| `getPseudoInertia()` | 4×4 伪惯性 |
| `flipAlongAxis(axis)` | 镜像（左右腿对称） |

---

## 4. FloatingBaseModel — 核心动力学引擎

### 4.1 状态结构

**`FBModelState<T>`**
- `bodyOrientation`：浮基四元数
- `bodyPosition`：世界系位置
- `bodyVelocity`：浮基空间速度 6D
- `q`, `qd`：关节角与角速度

**`FBModelStateDerivative<T>`**
- `dBodyPosition`, `dBodyVelocity`, `qdd`

### 4.2 建模 API

| 方法 | 说明 |
|------|------|
| `addBase(inertia)` / `addBase(mass, com, I)` | 添加浮基 |
| `addBody(...)` | 添加连杆 + 转子（含 gearRatio） |
| `addGroundContactPoint(bodyID, location, isFoot)` | 点接触（足端） |
| `addGroundContactBoxPoints(bodyId, dims)` | 盒体角点接触 |
| `check()` | 模型一致性检查 |
| `setGravity(g)` | 重力（默认 (0,0,-9.81)） |
| `setContactComputeFlag(gc_index, flag)` | 开关接触点 |

### 4.3 运动学

| 方法 | 说明 |
|------|------|
| `setState(state)` | 设置状态并失效缓存 |
| `forwardKinematics()` | 递归 FK |
| `biasAccelerations()` | 偏置加速度 |
| `forwardAccelerationKinematics()` | 加速度 FK |
| `contactJacobians()` | 接触点 Jacobian \(J_c\) |
| `getPosition(link_idx, local_pos?)` | 连杆/点位置 |
| `getOrientation(link_idx)` | 连杆姿态 |
| `getLinearVelocity(link_idx, point?)` | 线速度 |
| `getLinearAcceleration(link_idx, point?)` | 线加速度 |
| `getAngularVelocity/Acceleration(link_idx)` | 角速度/加速度 |

### 4.4 动力学

| 方法 | 算法 | 说明 |
|------|------|------|
| `generalizedGravityForce()` | — | \(G(q)\) |
| `generalizedCoriolisForce()` | — | \(C(q,\dot{q})\) |
| `massMatrix()` | CRBA | \(H(q)\) |
| `inverseDynamics(dState)` | RNEA | 给定 \(\ddot{q}\) 求 \(\tau\) |
| `runABA(tau, dstate)` | ABA | 给定 \(\tau\) 求 \(\ddot{q}\) |
| `getMassMatrix()` / `getGravityForce()` / `getCoriolisForce()` | 缓存访问 | 避免重复计算 |

### 4.5 接触与测试力

| 方法 | 说明 |
|------|------|
| `invContactInertia(gc_index, force_directions)` | 接触逆惯性 \(\Lambda\) |
| `applyTestForce(gc_index, force, dstate_out)` | 单位测试力，返回 \(\Lambda\) |
| `resetExternalForces()` | 清零外力 |

**接触逆惯性**：用于 `ContactImpulse` 的 sequential impulse 求解，\(\Lambda = (J H^{-1} J^T)^{-1}\) 的等价实现。

### 4.6 转子建模

`addBody` 的 `rotorInertia` + `gearRatio` 实现 Jain 第 12 章风格的 reflected inertia，更精确反映电机-减速器-关节链。

---

## 5. Quadruped 与机器人构建

### 5.1 常量 (`cheetah` 命名空间)

- `_numActJoint = 12`, `_numLegJoint = 3`, `_numLeg = 4`
- `_dimConfig = 19`, `_dimActuated = 12`

### 5.2 Quadruped 类

| 方法 | 说明 |
|------|------|
| `buildModel()` | 返回完整 `FloatingBaseModel` |
| `buildModel(model)` | 填充已有 model |
| `buildActuatorModels()` | 12 个 `ActuatorModel` |
| `getSideSign(leg)` | 左 +1 / 右 -1 |
| `getHipLocation(leg)` | 髋关节 body 系坐标 |

**自由函数** `withLegSigns(v, legID)`：左右腿符号镜像。

### 5.3 预置构建

- `buildMiniCheetah<T>()`：CAD 惯性、连杆长度
- `buildCheetah3<T>()`：LMI 辨识惯性

---

## 6. ActuatorModel — 电机力矩

### 6.1 为什么需要

仿真与力矩限幅需反映 DC 电机电压-速度-力矩关系，而非理想力矩源。

### 6.2 公式

```
BEMF = qd * gearRatio * KT * 2
i = tau_motor / (KT * 1.5)
V = i * R + BEMF
V_clamped = clamp(V, ±batteryV)
tau_joint = gearRatio * clamp(KT*1.5*(V_clamped - BEMF)/R, ±tauMax)
           - damping * qd - dryFriction * sgn(qd)
```

### 6.3 方法

| 方法 | 说明 |
|------|------|
| `ActuatorModel(gearRatio, motorKT, motorR, batteryV, damping, dryFriction, tauMax)` | 构造 |
| `getTorque(tauDes, qd)` | 限幅后实际力矩 |
| `setFriction(enabled)` | 开关摩擦 |

---

## 7. DynamicsSimulator

### 7.1 仿真步 `step(dt, tau, kp, kd)`

```
runABA(tau)           → 计算 qdd（含接触外力）
integrate(dt)         → 积分状态
[ContactConstraint]   → 速度级冲量或弹簧阻尼力
low-level PD          → 由 kp/kd 产生额外力矩（可选）
```

### 7.2 方法一览

| 方法 | 说明 |
|------|------|
| `DynamicsSimulator(model, useSpringDamper)` | 构造；选择 ContactSpringDamper 或 ContactImpulse |
| `step(dt, tau, kp, kd)` | 完整步进 |
| `runABA(tau)` | ABA 前向动力学 |
| `forwardKinematics()` | 足端 FK |
| `integrate(dt)` | 半隐式/显式积分 |
| `setState` / `getState` / `getDState` | 状态读写 |
| `setHoming(homing)` | 复位 PD 参数 |
| `setAllExternalForces(forces)` | 外空间力 |
| `addCollisionPlane/Box/Mesh` | 地形 |
| `getContactForce(idx)` | 接触力 |

---

## 8. 可运行示例：逆动力学

项目自带 `user/Example_Leg_InvDyn`，核心逻辑：

```cpp
// 从估计器与 LegController 组装 FBModelState
FBModelState<float> state;
state.bodyOrientation = _stateEstimate->orientation;
state.bodyPosition = _stateEstimate->position;
state.q = /* 12 关节角 */;

// 期望轨迹：正弦关节角
state.qd = /* ... */;
FBModelStateDerivative<float> dState;
dState.qdd = /* 解析二阶导 */;

// 逆动力学
Vec12<float> tau = _model->inverseDynamics(dState);
_legController->commands[leg].tauFeedForward = /* 分配到各腿 */;
```

运行：

```bash
cd build
./user/Example_Leg_InvDyn/leg_invdyn_ctrl m s
```

---

## 9. 与后续章节关系

- **WBC**（第 06 章）直接使用 `massMatrix()`、`generalizedCoriolisForce()`、`generalizedGravityForce()`
- **Convex MPC**（第 04 章）使用 `Quadruped` 质量与足端相对质心位置
- **LegController**（第 02 章）使用 `computeLegJacobianAndPosition` 解析 Jacobian

上一章：[00-architecture-overview.md](./00-architecture-overview.md)  
下一章：[02-leg-control-and-gait.md](./02-leg-control-and-gait.md)
