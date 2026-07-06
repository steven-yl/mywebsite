---
title: "11 数学工具、碰撞与基础设施"
subtitle: ""
date: 2026-07-06T12:00:00+08:00
draft: false
authors: [Steven]
description: "姿态/滤波数学工具、碰撞检测与 ContactImpulse/SpringDamper 接触仿真模型。"
summary: "数学库、碰撞检测与接触仿真基础设施详解。"
tags: [cheetah, robots]
categories: [docs cheetah, robots]
series: [cheetah-docs]
weight: 12
series_weight: 12
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 11 — 数学工具、碰撞与基础设施

## 1. 模块边界

| 路径 | 内容 |
|------|------|
| `common/include/Math/` | 姿态、插值、IIR |
| `common/include/Utilities/` | 滤波、定时、共享内存、Bezier 等 |
| `common/include/Collision/` | 碰撞几何与接触求解 |
| `common/include/ControlParameters/` | 运行时参数系统 |
| `common/include/cppTypes.h` | Eigen 别名 |

---

## 2. orientation_tools (`ori` 命名空间)

### 2.1 为什么需要

四足控制全程使用 **旋转矩阵、四元数、RPY** 及 skew 矩阵；需统一约定（world→body）与数值稳定算法。

### 2.2 函数完整列表

| 函数 | 说明 |
|------|------|
| `rad2deg` / `deg2rad` | 角度转换 |
| `coordinateRotation(axis, theta)` | 单轴旋转矩阵 |
| `crossMatrix(v)` | 向量→反对称矩阵 \([v]_\times\) |
| `rpyToRotMat(v)` | RPY→R（X-Y-Z 内禀） |
| `vectorToSkewMat` / `matToSkewVec` | 双向 skew |
| `rotationMatrixToQuaternion(R)` | R→q |
| `quaternionToRotationMatrix(q)` | q→R |
| `quatToRPY` / `rpyToQuat` | 四元数↔RPY |
| `quatToso3(q)` | q→so(3) 向量 |
| `rotationMatrixToRPY(R)` | R→RPY |
| `quatDerivative(q, omega)` | \(\dot{q} = \frac{1}{2} q \otimes \omega_{body}\) |
| `quatProduct(q1, q2)` | Hamilton 积 |
| `integrateQuat` / `integrateQuatImplicit` | 离散积分 |
| `quaternionToso3` / `so3ToQuat` | 小角度转换 |

**Enum** `CoordinateAxis { X, Y, Z }`

---

## 3. Interpolation (`Interpolate` 命名空间)

| 函数 | 公式 |
|------|------|
| `lerp(y0, yf, x)` | \(y = y_0 + x(y_f - y_0)\) |
| `cubicBezier(y0, yf, x)` | 端点导数 0 的三次 Bézier |
| `cubicBezierFirstDerivative` | \(6x(1-x)\) 缩放 |
| `cubicBezierSecondDerivative` | \((6-12x)\) 缩放 |

---

## 4. FirstOrderIIRFilter

一阶低通：

\[
\alpha = 1 - e^{-2\pi f_c / f_s}, \quad y_k = \alpha x_k + (1-\alpha) y_{k-1}
\]

| 方法 | 说明 |
|------|------|
| `FirstOrderIIRFilter(fc, fs, init)` | 由截止频率构造 |
| `FirstOrderIIRFilter(alpha, init)` | 直接给 α |
| `update(x)` | 滤波步 |
| `get()` / `reset()` | 读状态/复位 |

---

## 5. MathUtilities

| 函数 | 说明 |
|------|------|
| `square(a)` | 平方 |
| `almostEqual(a, b, tol)` | 矩阵近似相等 |

---

## 6. Utilities 核心

### 6.1 utilities.h

| 函数 | 说明 |
|------|------|
| `fpEqual`, `vectorEqual` | 浮点/向量比较 |
| `coerce`, `deadband`, `eigenDeadband` | 限幅与死区 |
| `sgn` | 符号函数 |
| `fillEigenWithRandom`, `generator_gaussian_noise` | 随机 |
| `mapToRange` | 线性映射 |
| `eigenToString`, `numberToString`, `boolToString` | 字符串化 |
| `writeStringToFile`, `getCurrentTimeAndDate` | IO |
| `getConfigDirectoryPath` | 配置路径 |
| `EulerZYX_2_SO3` | ZYX→旋转 |
| `smooth_change` / `_vel` / `_acc` | 余弦 smoothstep |
| `stringToNumber`, `stringToVec3` | 解析 |
| `getLcmUrl(ttl)` | LCM multicast URL |

### 6.2 filters.h

| 类 | 说明 |
|----|------|
| `filter<T>` | 抽象：`input()`, `output()`, `clear()` |
| `butterworth_filter<T>` | Butterworth LP |
| `digital_lp_filter<T>` | 数字 LP |
| `moving_average_filter<T>` | 滑动平均 |
| `deriv_lp_filter<T>` | 滤波微分 |
| `ff01_filter` / `ff02_filter` | 前馈滤波 |
| `AverageFilter<T>` | 指数平均+限幅 |

### 6.3 PeriodicTask

| 类/方法 | 说明 |
|---------|------|
| `PeriodicTask::start/stop` | 启停线程 |
| `init/run/cleanup` | 子类实现 |
| `getPeriod/getRuntime/getMaxPeriod/getMaxRuntime` | 统计 |
| `isSlow/printStatus/clearMax` | 监控 |
| `PeriodicTaskManager::addTask/stopAll` | 管理 |
| `PeriodicMemberFunction<T>` | 成员函数包装 |

### 6.4 其他

| 文件 | 功能 |
|------|------|
| `Timer` | 毫秒/纳秒计时 |
| `SharedMemoryObject` / `SharedMemorySemaphore` | 进程间同步 |
| `pseudoInverse` | SVD 伪逆 + 奇异值阈值 |
| `BezierCurve` / `BSplineBasic` | 轨迹曲线 |
| `EdgeTrigger` | 边沿检测 |
| `SegfaultHandler` | 崩溃捕获 |
| `save_file` | 向量/标量落盘 |

---

## 7. 碰撞系统

### 7.1 Collision 抽象

| 方法 | 说明 |
|------|------|
| `ContactDetection(cp_pos, penetration, cp_frame)` | 纯虚：点检测 |
| `getFrictionCoeff()` / `getRestitutionCoeff()` | μ, e |

### 7.2 几何实现

| 类 | 说明 |
|----|------|
| `CollisionPlane` | 水平面 z=height |
| `CollisionBox` | 有向盒 |
| `CollisionMesh` | 高度图（双线性插值） |

### 7.3 ContactConstraint

| 方法 | 说明 |
|------|------|
| `AddCollision(collision)` | 注册地形 |
| `UpdateExternalForces(K, D, dt)` | 力级接触 |
| `UpdateQdot(state)` | 速度级冲量 |
| `getContactPosList()` | 可视化 |
| `getGCForce(idx)` | 接触力 |

### 7.4 ContactImpulse — Sequential Impulse

**算法**：
1. 对每个接触点，用法向/切向 impulse 修正 \(\dot{q}\)  
2. 用 `applyTestForce` 得 \(\Lambda = (J H^{-1} J^T)^{-1}\)  
3. Projected Gauss-Seidel 迭代  
4. 摩擦金字塔 + restitution on \(v_n\)  

| 方法 | 说明 |
|------|------|
| `UpdateExternalForces` | 存 dt |
| `UpdateQdot` | 冲量求解 |

### 7.5 ContactSpringDamper

| 方法 | 说明 |
|------|------|
| `UpdateExternalForces` | \(F_n = K \cdot \delta + D \cdot v_n\)，切向带 deflection 记忆 |
| `UpdateQdot` | 无操作 |

**选型**：`SimulatorControlParameters.use_spring_damper` — 弹簧阻尼更平滑，冲量更硬。

---

## 8. ControlParameters 系统

### 8.1 类型

`ControlParameterValueKind`: DOUBLE, FLOAT, S64, VEC3_DOUBLE, VEC3_FLOAT

### 8.2 ControlParameter 方法

`initializeDouble/Float/Integer/Vec3f/Vec3d`, `set`, `get`, `toString`, `setFromString`, `truncateName`

### 8.3 ControlParameters 基类

| 方法 | 说明 |
|------|------|
| `initializeFromYamlFile` / `writeToYamlFile` | YAML |
| `initializeFromIniFile` / `writeToIniFile` | INI |
| `isFullyInitialized` | 是否全部 SET |
| `generateUnitializedList` | 缺失列表 |
| `lockMutex/unlockMutex` | 线程安全 |

### 8.4 预定义集合

- **RobotControlParameters**：controller_dt, cheater_mode, KF 噪声, use_rc, stand gains…  
- **SimulatorControlParameters**：IMU 噪声, simulation_speed, floor K/D, use_spring_damper…  

### 8.5 ControlParameterInterface

仿真 GUI ↔ 控制器参数请求/响应消息。

---

## 9. cppTypes 别名（常用）

```cpp
template<typename T> using Vec3 = Matrix<T, 3, 1>;
template<typename T> using Mat3 = Matrix<T, 3, 3>;
template<typename T> using Quat = Matrix<T, 4, 1>;
using RobotType = enum { MINI_CHEETAH, CHEETAH_3, ... };
```

---

## 10. 单元测试

```bash
cd build
./common/test-common
# 18 tests: dynamics, spatial, filters, JCQP, OSQP, LegController, ...
```

| 测试文件 | 覆盖 |
|----------|------|
| `test_dynamics.cpp` | ABA vs FD |
| `test_spatial.cpp` | 空间代数 |
| `test_footswing.cpp` | Bézier 摆腿 |
| `test_orientation_tools.cpp` | 姿态转换 |
| `test_filters.cpp` | 滤波器 |
| `test_LegController.cpp` | Jacobian |

---

## 11. pseudoInverse 示例

```cpp
Mat3<float> J, Jinv;
pseudoInverse(J, 0.001f, Jinv);  // sigma < 0.001 截断
Vec3<float> tau = Jinv * force;
```

SVD：\(J = U \Sigma V^T\)，\(\Sigma_i^{-1}\) 仅在 \(\Sigma_i > \sigma_{th}\) 时使用。

---

## 12. 地形配置示例

`config/default-terrain.yaml`（注释示例）：

```yaml
# mesh, box, stairs 示例见文件内注释
floor:
  mu: 0.5
  restitution: 0.0
  height: 0.0
```

仿真启动时加载，配合 `CollisionPlane/Mesh/Box`。

---

上一章：[10-robot-runtime-and-simulation.md](./10-robot-runtime-and-simulation.md)  
下一章：[12-user-parameters-and-example-controllers.md](./12-user-parameters-and-example-controllers.md)
