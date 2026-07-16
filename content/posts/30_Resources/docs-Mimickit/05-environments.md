---
title: "第五章：RL 环境（mimickit/envs/）"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第五章：RL 环境（mimickit/envs/）。"
summary: "第五章：RL 环境（mimickit/envs/）。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 5
series_weight: 5
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第五章：RL 环境（`mimickit/envs/`）

## 5.1 模块边界

`envs/` 包实现全部强化学习环境：观测构建、奖励计算、终止判断、动作应用。所有环境通过 `env_builder.build_env` 按 `env_name` 实例化。

---

## 5.2 继承体系与类清单

| 类 | `env_name` | 父类 | 用途 |
|----|-----------|------|------|
| `CharEnv` | `char` | `SimEnv` | 关节角色基类 |
| `DeepMimicEnv` | `deepmimic` | `CharEnv` | 参考动作跟踪 |
| `AMPEnv` | `amp` | `DeepMimicEnv` | 判别器观测 |
| `ASEEnv` | `ase` | `AMPEnv` | 默认姿态随机重置 |
| `ADDEnv` | `add` | `AMPEnv` | 配对 demo/agent 判别观测 |
| `SMPEnv` | `smp` | `AMPEnv` | GSI 缓冲区支持 |
| `TaskLocationEnv` | `task_location` | `SMPEnv` | 行走到目标点 |
| `TaskSteeringEnv` | `task_steering` | `SMPEnv` | 速度+朝向控制 |
| `TaskDodgeballEnv` | `task_dodgeball` | `SMPEnv` | 躲避球体 |
| `StaticObjectsEnv` | `static_objects` | `DeepMimicEnv` | 带障碍物（vault） |
| `ViewMotionEnv` | `view_motion` | `CharEnv` | 动作预览（无 RL） |
| `CharDofTestEnv` | `char_dof_test` | `CharEnv` | 关节范围测试 |

---

## 5.3 `BaseEnv` / `SimEnv`

### 5.3.1 枚举

```python
class EnvMode(enum.Enum):
    TRAIN = 0
    TEST = 1

class DoneFlags(enum.IntEnum):
    NULL = 0    # 未结束
    TIME = 1    # 超时
    FAIL = 2    # 失败（跌倒等）
    SUCC = 3    # 成功（动作完成）
```

### 5.3.2 核心接口

| 方法 | 说明 |
|------|------|
| `reset(env_ids)` | 重置指定环境，返回 `(obs, info)` |
| `step(action)` | 执行动作，返回 `(obs, reward, done, info)` |
| `get_obs_space()` | Gymnasium Box 观测空间 |
| `get_action_space()` | Gymnasium Box/Discrete 动作空间 |
| `get_num_envs()` | 并行环境数 |
| `set_mode(mode)` | 训练/测试模式 |
| `get_reward_succ()` / `get_reward_fail()` | 终止时的 bootstrap 奖励 |
| `record_diagnostics()` | 返回诊断指标字典 |

---

## 5.4 `CharEnv` — 角色基类

### 5.4.1 初始化流程

1. `_build_kin_char_model(char_file)` — 加载运动学模型
2. `_build_envs(num_envs)` — 创建并行仿真环境
3. `_build_character()` — 在引擎中加载角色
4. `_build_sim_tensors()` — 分配 GPU 张量
5. `_build_action_space()` — 根据控制模式定义动作边界
6. `_build_camera()` — 相机跟踪设置

### 5.4.2 关键方法

| 方法 | 说明 |
|------|------|
| `_compute_obs(env_ids)` | 构建角色本体观测 |
| `_apply_action(actions)` | 动作 → 引擎控制命令 |
| `_update_reward()` | 更新奖励（子类重写） |
| `_update_done()` | 更新终止标志 |
| `_reset_char(env_ids)` | 重置角色状态 |
| `_build_body_ids_tensor(body_names)` | 名称列表 → body ID 张量 |
| `_has_key_bodies()` | 是否配置了 key_bodies |

### 5.4.3 角色观测 `compute_char_obs`

输入：根位置、根旋转、根速度、根角速度、关节旋转、DOF 速度、关键点位置

输出向量包含（依配置裁剪）：

- 根高度（若 `root_height_obs`）
- 局部/全局根方向
- 关节旋转（tan-norm 或四元数）
- DOF 速度
- 关键点相对位置

### 5.4.4 动作空间构建

| 控制模式 | 动作边界来源 |
|---------|-------------|
| `pos` | 关节位置上下限（XML `limited=true`） |
| `vel` | 对称速度限幅 |
| `torque` | 力矩限幅 |

---

## 5.5 `DeepMimicEnv` — 参考动作跟踪

### 5.5.1 为什么需要 DeepMimic 环境

将 RL 问题定义为：**让仿真角色尽可能跟踪参考动作序列**。奖励直接度量当前姿态与参考姿态的偏差，无需对抗训练。

### 5.5.2 特有配置参数

| 参数 | 说明 |
|------|------|
| `enable_early_termination` | 跌倒/偏差过大时提前终止 |
| `pose_termination` | 按姿态距离终止 |
| `pose_termination_dist` | 姿态终止阈值（米） |
| `enable_phase_obs` | 动作相位编码加入观测 |
| `num_phase_encoding` | 相位 sin/cos 编码维度 |
| `enable_tar_obs` | 未来参考帧作为观测 |
| `tar_obs_steps` | 未来帧步数列表 |
| `rand_reset` | 随机采样动作起始时间 |
| `ref_char_offset` | 参考角色可视化偏移 |
| `joint_err_w` | 各关节误差权重 |
| `reward_*_w` / `reward_*_scale` | 奖励分量权重与尺度 |

### 5.5.3 关键方法

| 方法 | 说明 |
|------|------|
| `_load_motions(motion_file)` | 创建 `MotionLib` |
| `_reset_ref_motion(env_ids)` | 采样动作片段与时间 |
| `_update_ref_motion()` | 每步推进参考动作 |
| `_update_reward()` | 计算跟踪奖励 |
| `_update_done()` | 超时/成功/失败判断 |
| `_ref_state_init(env_ids)` | 从参考状态初始化角色 |
| `_enable_ref_char()` | 是否显示幽灵参考角色 |
| `_parse_joint_err_weights()` | 关节权重 → DOF 权重 |

### 5.5.4 奖励函数 `compute_reward`

DeepMimic 使用**指数衰减**奖励，各分量加权求和：

**误差计算：**

$$
e_{pose} = \sum_j w_j \cdot \Delta q_j^2, \quad \Delta q_j = \text{quat\_diff\_angle}(q_j, q_j^{tar})
$$

$$
e_{vel} = \sum_j w_j \cdot (\dot{q}_j^{tar} - \dot{q}_j)^2
$$

$$
e_{root\_pos} = \|p^{tar} - p\|^2 \quad (\text{可选忽略水平/高度分量})
$$

$$
e_{root\_rot} = \text{quat\_diff\_angle}(R, R^{tar})^2
$$

**奖励：**

$$
r_{pose} = \exp(-\lambda_{pose} \cdot e_{pose})
$$
$$
r_{vel} = \exp(-\lambda_{vel} \cdot e_{vel})
$$
$$
r_{root} = \exp(-\lambda_{root} \cdot (e_{root\_pos} + 0.1 \cdot e_{root\_rot}))
$$
$$
r_{key} = \exp(-\lambda_{key} \cdot e_{key\_pos})
$$

$$
r = w_{pose} r_{pose} + w_{vel} r_{vel} + w_{root} r_{root} + w_{vel\_root} r_{vel\_root} + w_{key} r_{key}
$$

实现见 `deepmimic_env.py` 的 `@torch.jit.script def compute_reward(...)`。

### 5.5.5 终止条件 `compute_done`

| 条件 | `DoneFlags` |
|------|-------------|
| `time >= episode_length` | `TIME` |
| `motion_time >= motion_length` | `SUCC` |
| 非允许 body 接触地面 | `FAIL` |
| 姿态偏差 > `pose_termination_dist` | `FAIL` |

### 5.5.6 目标观测 `compute_tar_obs`

构建相对于参考根的位置/旋转观测，支持：

- **局部坐标**（`global_obs=false`）：用航向四元数逆旋转
- **根高度**（`root_height_obs`）：仅保留 z 分量
- **关键点**：相对根的位置

---

## 5.6 `AMPEnv` — 对抗运动先验环境

### 5.6.1 与 DeepMimic 的差异

| 方面 | DeepMimic | AMP |
|------|-----------|-----|
| 奖励 | 跟踪参考帧 | 由 Agent 端判别器提供 |
| `_update_reward()` | 计算跟踪奖励 | **空操作**（奖励在 Agent 计算） |
| 参考角色可视化 | 默认开启 | 默认关闭 |
| 额外输出 | - | `info["disc_obs"]` |
| 动作结束终止 | 可启用 | 通常禁用 |

### 5.6.2 判别器观测 `compute_disc_obs`

拼接**位置观测**与**速度观测**，跨 `num_disc_obs_steps` 历史步展平：

```
disc_obs = flatten([pos_obs_t-N+1, ..., pos_obs_t, vel_obs_t-N+1, ..., vel_obs_t])
```

位置部分复用 `compute_tar_obs`；速度部分 `compute_disc_vel_obs` 含根速度、根角速度、可选 DOF 速度。

### 5.6.3 关键方法

| 方法 | 说明 |
|------|------|
| `get_disc_obs_space()` | 判别器观测空间 |
| `fetch_disc_obs_demo(n)` | 从动作库采样 n 个 demo 观测 |
| `_compute_disc_obs_demo(motion_ids, motion_times0)` | 构建 demo 观测 |
| `_fetch_disc_demo_data(...)` | 获取多步历史 demo 数据 |
| `_build_disc_obs_buffers()` | 环形缓冲区存储历史 |
| `_update_disc_hist()` | 每步推入当前状态 |
| `_update_disc_obs(env_ids)` | 构建当前 disc_obs |
| `_reset_disc_hist(env_ids)` | 重置历史为 demo 数据 |

### 5.6.4 历史缓冲 `CircularBuffer`

AMP 需要最近 N 步状态构建时序判别器输入。`CircularBuffer` 提供：

- `push(data)` — 推入新帧
- `get_all()` — 获取全部历史 [num_envs, N, ...]
- `fill(env_ids, data)` — 重置时填充

---

## 5.7 `ASEEnv`

在 `AMPEnv` 基础上增加 `default_reset_prob`：以一定概率从**默认姿态**而非参考动作状态重置，促进探索与技能多样性。

---

## 5.8 `ADDEnv`

### 5.8.1 与 AMP 的差异

同时输出**配对观测**：

```python
info["disc_obs"]       # 当前 agent 状态
info["disc_obs_demo"]  # 同时刻参考 demo 状态
```

Agent 端计算差分$\Delta = \text{demo} - \text{agent}$送入差分判别器。详见 [11-algorithm-add](11-algorithm-add.md)。

---

## 5.9 `SMPEnv`

### 5.9.1 GSI 缓冲区

支持 Generative State Initialization：

| 方法 | 说明 |
|------|------|
| `init_gsi_buffer(samples)` | 初始化生成状态池 |
| `add_gsi_samples(samples)` | 追加新生成状态 |
| `_sample_gsi_reset_state(env_ids)` | 从池中采样重置状态 |

要求：`enable_tar_obs=false`，`pose_termination=false`。

---

## 5.10 任务环境

### 5.10.1 `TaskLocationEnv`

- 随机采样地面目标点
- 任务奖励：接近目标（位置 + 朝向）
- 与 SMP/AMP 判别器奖励加权组合

### 5.10.2 `TaskSteeringEnv`

- 指定目标线速度与朝向
- 任务奖励：速度/朝向跟踪

### 5.10.3 `TaskDodgeballEnv`

- 发射物飞向角色
- 任务奖励：躲避成功

---

## 5.11 辅助环境

### 5.11.1 `ViewMotionEnv`

- 播放 `.pkl` 动作，无 RL 训练
- 用于检查动作数据质量

```bash
python mimickit/run.py --mode test --arg_file args/view_motion_humanoid_args.txt --visualize true
```

### 5.11.2 `StaticObjectsEnv`

- DeepMimic + 静态障碍物（vault 等）
- 额外加载场景物体

### 5.11.3 `CharDofTestEnv`

- 逐关节测试运动范围
- 用于资产验证

---

## 5.12 观测向量结构示意

```
完整观测（DeepMimic + tar_obs）≈
[char_obs (本体感知), tar_obs (未来参考帧), phase_obs (可选)]
```

局部坐标模式下，水平位置/速度相对于**角色航向**表达，使策略对全局朝向不变。

---

## 5.13 参考资料

| 论文 | 链接 |
|------|------|
| DeepMimic (TOG 2018) | https://xbpeng.github.io/projects/DeepMimic/ |
| AMP (TOG 2021) | https://xbpeng.github.io/projects/AMP/ |
| ASE (TOG 2022) | https://xbpeng.github.io/projects/ASE/ |

---

[← 物理引擎](04-physics-engines.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：学习核心 →](06-learning-core.md)
