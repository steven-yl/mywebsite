---
title: "第三章：动画模块（mimickit/anim/）"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第三章：动画模块（mimickit/anim/）。"
summary: "第三章：动画模块（mimickit/anim/）。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 3
series_weight: 3
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第三章：动画模块（`mimickit/anim/`）

## 3.1 模块边界与职责

`anim/` 包负责**运动数据表示、加载、插值**以及**运动学树解析与前向运动学**，不涉及物理仿真或 RL 训练逻辑。

```
anim/
├── motion.py           # 单片段 Motion 类、.pkl I/O
├── motion_lib.py       # 多片段库、加权采样、插值
├── kin_char_model.py   # 运动学树抽象、FK、DOF↔旋转
├── mjcf_char_model.py  # MuJoCo XML 解析
├── urdf_char_model.py  # URDF 解析（G1、Go2）
└── usd_char_model.py   # USD 解析（Isaac Lab）
```

---

## 3.2 运动数据格式（`.pkl`）

### 3.2.1 文件结构

```python
{
    "loop_mode": 0,      # 0=CLAMP（不循环）, 1=WRAP（循环）
    "fps": 30.0,
    "frames": [
        [root_pos(3), root_rot_expmap(3), joint_dofs...],
        ...
    ]
}
```

### 3.2.2 帧向量布局

每帧按**运动学树深度优先遍历**顺序排列：

```
[root position (3D), root rotation exp-map (3D), joint rotation DOFs...]
```

以 `humanoid.xml` 为例：

```
[root(3), root_rot(3), abdomen(3), neck(3), right_shoulder(3), right_elbow(1),
 left_shoulder(3), left_elbow(1), right_hip(3), right_knee(1), right_ankle(3),
 left_hip(3), left_knee(1), left_ankle(3)]
```

- **3D 关节**：指数映射（exponential map）表示旋转
- **1D 关节**（铰链）：标量角度

### 3.2.3 多片段数据集（`.yaml`）

```yaml
motions:
  - file: "data/motions/humanoid/humanoid_walk.pkl"
    weight: 3.0
  - file: "data/motions/humanoid/humanoid_run.pkl"
    weight: 1.0
```

`MotionLib` 按 `weight` 加权随机采样片段。

---

## 3.3 `motion.py` — Motion 类

### 3.3.1 枚举

```python
class LoopMode(enum.Enum):
    CLAMP = 0   # 到达末尾后保持最后一帧
    WRAP = 1    # 循环播放，根位置累积 wrap_delta
```

### 3.3.2 类 `Motion`

| 方法/属性 | 说明 |
|----------|------|
| `__init__(loop_mode, fps, frames)` | 构造运动对象 |
| `get_length()` | 运动时长（秒）= `(num_frames - 1) / fps` |
| `get_num_frames()` | 帧数 |
| `get_frame(i)` | 获取第 i 帧数据 |
| `save(file)` | 序列化到 `.pkl` |

### 3.3.3 模块函数

| 函数 | 说明 |
|------|------|
| `load_motion(file)` | 从 `.pkl` 加载，返回 `Motion` 对象 |

---

## 3.4 `motion_lib.py` — MotionLib 类

### 3.4.1 为什么需要 MotionLib

单片段 `Motion` 无法表达**多动作数据集**的训练需求。`MotionLib` 将多个 `.pkl` 预加载到 GPU 张量，支持：

- 加权随机采样片段
- 批量插值查询任意时刻姿态
- 预计算速度、循环偏移

### 3.4.2 公共 API（完整）

| 方法 | 说明 |
|------|------|
| `get_num_motions()` | 数据集片段数量 |
| `get_total_length()` | 所有片段总时长 |
| `sample_motions(n)` | 按权重采样 n 个 motion_id |
| `sample_time(motion_ids, truncate_time=None)` | 在片段内均匀随机采样时间 |
| `get_motion_file(motion_id)` | 返回源文件路径 |
| `get_motion_length(motion_ids)` | 各片段时长 |
| `get_motion_loop_mode(motion_ids)` | 循环模式 |
| `calc_motion_phase(motion_ids, times)` | 归一化相位 ∈ [0,1] |
| `calc_motion_frame(motion_ids, motion_times)` | **核心**：插值得到姿态 |
| `joint_rot_to_dof(joint_rot)` | 关节四元数 → DOF 向量 |
| `get_motion_lengths()` | 全部时长张量 |
| `get_motion_weights()` | 全部权重张量 |
| `get_motion_frame_size()` | 单帧向量维度 = 6 + dof_size |
| `get_num_joints()` | 关节数 |

### 3.4.3 帧插值算法

给定 `motion_ids` 和 `motion_times`：

1. 计算相位$\phi = t / T$（WRAP 模式取$\phi \mod 1$）
2. 帧索引：$i_0 = \lfloor \phi (N-1) \rfloor$，$i_1 = \min(i_0+1, N-1)$
3. 混合系数：$\beta = \phi(N-1) - i_0$
4. 位置线性插值，旋转 **SLERP** 插值
5. WRAP 模式叠加根位置循环偏移$\Delta p_{wrap}$

```python
# 调用示例
motion_ids = motion_lib.sample_motions(n=4)
motion_times = motion_lib.sample_time(motion_ids)
root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
    motion_lib.calc_motion_frame(motion_ids, motion_times)
```

### 3.4.4 内部方法

| 方法 | 说明 |
|------|------|
| `_load_motions(motion_file)` | 加载单 `.pkl` 或数据集 `.yaml` |
| `_extract_frame_data(frame)` | 帧 → 张量 + 四元数 |
| `_calc_frame_blend(motion_ids, times)` | 计算帧索引与混合系数 |
| `_calc_loop_offset(motion_ids, times)` | WRAP 模式根位置偏移 |
| `_load_motion_pkl(file)` | 加载单个 pkl |
| `_load_dataset_yaml(file)` | 加载数据集配置 |

### 3.4.5 模块函数

```python
def extract_pose_data(frame):
    root_pos = frame[..., 0:3]
    root_rot = frame[..., 3:6]
    joint_dof = frame[..., 6:]
    return root_pos, root_rot, joint_dof
```

---

## 3.5 `kin_char_model.py` — 运动学模型

### 3.5.1 关节类型 `JointType`

| 类型 | DOF 维度 | 旋转表示 |
|------|---------|---------|
| `ROOT` | 0 | 由 root_rot 处理 |
| `HINGE` | 1 | 绕轴旋转角 |
| `SPHERICAL` | 3 | 四元数（指数映射存储） |
| `FIXED` | 0 | 无自由度 |

### 3.5.2 类 `Joint`

| 方法 | 说明 |
|------|------|
| `get_dof_dim()` | 关节 DOF 维度 |
| `get_joint_dof(dof)` | 从 DOF 向量提取本关节 DOF |
| `set_joint_dof(j_dof, out_dof)` | 写入 DOF 到输出向量 |
| `dof_to_rot(dof)` | DOF → 四元数 |
| `rot_to_dof(rot)` | 四元数 → DOF |

### 3.5.3 类 `KinCharModel`

| 方法 | 说明 |
|------|------|
| `init(body_names, parent_indices, ...)` | 从解析数据初始化树 |
| `load(char_file)` | 抽象方法，子类实现 |
| `save(output_file)` | 导出运动学模型 |
| `get_body_names()` | 全部 body 名称 |
| `get_joint(j)` | 获取关节对象 |
| `get_parent_id(j)` | 父关节索引 |
| `get_dof_size()` | 总 DOF 数 |
| `get_joint_dof_idx(j)` | 关节在 DOF 向量中的起始索引 |
| `get_joint_dof_dim(j)` | 关节 DOF 维度 |
| `get_num_joints()` | 关节总数 |
| `dof_to_rot(dof)` | 完整 DOF 向量 → 各关节四元数 |
| `rot_to_dof(rot)` | 各关节四元数 → DOF 向量 |
| `forward_kinematics(root_pos, root_rot, joint_rot)` | **FK**：计算各 body 世界位姿 |
| `compute_frame_dof_vel(joint_rot, dt)` | 单帧 DOF 速度 |
| `compute_dof_vel(joint_rot0, joint_rot1, dt)` | 两帧间 DOF 速度 |
| `get_body_name(body_id)` | ID → 名称 |
| `get_body_id(body_name)` | 名称 → ID |
| `get_joint_id(body_name)` | 名称 → 关节 ID |

### 3.5.4 前向运动学公式

对每个关节$j$（父节点$p$）：

$$
T_j^{world} = T_p^{world} \cdot T_{local}(p_j, R_{local,j})
$$

其中$T$为 4×4 变换矩阵，$p_j$为局部平移，$R_{local,j}$为关节旋转。

---

## 3.6 资产解析器

### 3.6.1 `MJCFCharModel`（`.xml`）

- 解析 MuJoCo MJCF 格式
- 用于 Humanoid、SMPL、剑盾角色
- `MJCFCharModel.load(char_file)` → `KinCharModel`

### 3.6.2 `URDFCharModel`（`.urdf`）

- 解析机器人 URDF
- 用于 Unitree G1、Go2、Pi Plus
- 处理 `<joint type="revolute">` 等

### 3.6.3 `USDCharModel`（`.usd`）

- 解析 Isaac Lab USD 舞台
- 用于 Isaac Lab 引擎下的角色加载

### 3.6.4 解析器选择逻辑

```python
# char_env.py _build_kin_char_model
ext = os.path.splitext(char_file)[1]
if ext == ".xml":
    model = MJCFCharModel(device)
elif ext == ".urdf":
    model = URDFCharModel(device)
elif ext == ".usd":
    model = USDCharModel(device)
```

---

## 3.7 旋转数学约定

MimicKit 使用 `util/torch_util.py` 中的四元数工具：

| 操作 | 函数 |
|------|------|
| 指数映射 → 四元数 | `exp_map_to_quat()` |
| 四元数乘法 | `quat_mul()` |
| 四元数旋转向量 | `quat_rotate()` |
| 球面线性插值 | `slerp()` |
| 四元数角度差 | `quat_diff_angle()` |
| 航向四元数逆 | `calc_heading_quat_inv()` |

四元数格式：**(x, y, z, w)**，实部 w 在后。

---

## 3.8 与环境的交互

```
MotionLib.calc_motion_frame()
    → root_pos, root_rot, joint_rot, dof_vel
    → KinCharModel.forward_kinematics() → body_pos, body_rot
    → DeepMimicEnv 设置参考角色状态
    → compute_reward() 计算跟踪误差
```

---

## 3.9 参考资料

| 资源 | 链接 |
|------|------|
| 指数映射旋转 | [Exponential map (Lie theory)](https://en.wikipedia.org/wiki/Exponential_map_(Lie_theory)) |
| MuJoCo MJCF | https://mujoco.readthedocs.io/en/latest/XMLreference.html |
| AMASS 数据集 | https://amass.is.tue.mpg.de |
| GMR 重定向 | https://github.com/YanjieZe/GMR |

---

[← 配置与流程](02-config-and-workflow.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：物理引擎 →](04-physics-engines.md)
