---
title: "第二章：配置系统与训练流程"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第二章：配置系统与训练流程。"
summary: "第二章：配置系统与训练流程。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 2
series_weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第二章：配置系统与训练流程

## 2.1 三层配置架构

MimicKit 通过三个 YAML 文件解耦**物理引擎**、**RL 环境**、**学习算法**：

```
args/deepmimic_humanoid_ppo_args.txt
    ├── engine_config  → data/engines/isaac_gym_engine.yaml
    ├── env_config     → data/envs/deepmimic_humanoid_env.yaml
    └── agent_config   → data/agents/deepmimic_humanoid_ppo_agent.yaml
```

环境配置可内嵌 `engine:` 块覆盖引擎参数（`env_builder.override_engine_config`）。

---

## 2.2 入口：`mimickit/run.py`

### 2.2.1 函数说明

| 函数 | 作用 |
|------|------|
| `load_args(argv)` | 解析 CLI，支持 `--arg_file` 加载预设参数 |
| `build_env(args, num_envs, device, visualize)` | 调用 `env_builder.build_env` |
| `build_agent(args, env, device)` | 调用 `agent_builder.build_agent` |
| `train(agent, max_samples, out_dir, ...)` | 调用 `agent.train_model` |
| `test(agent, test_episodes)` | 调用 `agent.test_model`，打印 Mean Return |
| `save_config_files(args, out_dir)` | 复制三份 YAML 到输出目录 |
| `set_rand_seed(args)` | 基于时间戳 + 进程 rank 设置随机种子 |
| `run(rank, num_procs, device, master_port, args)` | 单进程主逻辑 |
| `main(argv)` | 多进程 spawn，每个 device 一个进程 |

### 2.2.2 训练主循环（`BaseAgent.train_model`）

```python
# 伪代码：对应 base_agent.py
while sample_count < max_samples:
    rollout(steps_per_iter)          # 收集经验
    build_train_data()               # 计算 TD(λ)、优势等
    update_model()                   # 更新网络
    update_normalizers()             # 更新观测归一化（可选）
    if iter % iters_per_output == 0:
        test_model(test_episodes)    # 评估
        save(model.pt)               # 保存检查点
```

### 2.2.3 命令行参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--mode` | `train` 或 `test` | `train` |
| `--num_envs` | 并行环境数 | `4096`（Isaac Gym）/ `1`（非并行） |
| `--engine_config` | 引擎 YAML | `data/engines/isaac_gym_engine.yaml` |
| `--env_config` | 环境 YAML | `data/envs/deepmimic_humanoid_env.yaml` |
| `--agent_config` | Agent YAML | `data/agents/deepmimic_humanoid_ppo_agent.yaml` |
| `--arg_file` | 预设参数文件 | `args/deepmimic_humanoid_ppo_args.txt` |
| `--visualize` | 是否渲染 | 训练 `false`，测试 `true` |
| `--out_dir` | 输出目录 | `output/` |
| `--model_file` | 测试用检查点 | `data/models/xxx.pt` |
| `--logger` | 日志后端 | `txt` / `tb` / `wandb` |
| `--video` | 无头视频录制 | `true` / `false` |
| `--devices` | 多设备训练 | `cuda:0 cuda:1` |
| `--max_samples` | 训练样本上限 | 默认 `int64.max` |
| `--rand_seed` | 随机种子 | 可选 |
| `--master_port` | 分布式端口 | 默认随机 6000–7000 |

### 2.2.4 可运行示例

**训练 DeepMimic 人形 Spin Kick：**

```bash
python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/deepmimic_humanoid_env.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --visualize false \
  --out_dir output/
```

**使用 arg_file（等价写法）：**

```bash
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --visualize false --out_dir output/
```

**测试预训练模型：**

```bash
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --num_envs 4 \
  --mode test \
  --visualize true \
  --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```

**多 GPU 分布式：**

```bash
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --devices cuda:0 cuda:1
```

---

## 2.3 引擎配置（`data/engines/`）

| 文件 | `engine_name` | `sim_freq` | 说明 |
|------|---------------|------------|------|
| `isaac_gym_engine.yaml` | `isaac_gym` | 120 Hz | 默认，GPU 并行 |
| `isaac_lab_engine.yaml` | `isaac_lab` | - | Isaac Sim / USD |
| `newton_engine.yaml` | `newton` | 240 Hz | 替代引擎 |

公共字段：

```yaml
control_mode: "pos"      # pos / vel / torque / pd_explicit
control_freq: 30         # 控制频率 Hz
env_spacing: 5.0         # 并行环境间距
```

控制周期：$\Delta t_{ctrl} = 1 / \text{control\_freq}$，每控制步包含$\text{sim\_freq} / \text{control\_freq}$个物理子步。

---

## 2.4 环境配置（`data/envs/`）

### 2.4.1 通用字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `env_name` | string | 环境类名（`deepmimic`/`amp`/`ase`/`add`/`smp`/...） |
| `char_file` | string | 角色资产路径 |
| `motion_file` | string | 单片段 `.pkl` 或数据集 `.yaml` |
| `episode_length` | float | 回合最大时长（秒） |
| `global_obs` | bool | 是否使用全局坐标观测 |
| `root_height_obs` | bool | 根高度是否纳入观测 |
| `key_bodies` | list | 关键点 body 名称 |
| `contact_bodies` | list | 允许接触地面的 body（跌倒检测用） |

### 2.4.2 DeepMimic 专用字段

```yaml
pose_termination: true          # 姿态偏差过大则终止
pose_termination_dist: 1.0        # 终止距离阈值（米）
enable_tar_obs: true            # 是否包含目标姿态观测
tar_obs_steps: [1, 2, 3]        # 未来参考帧步数
reward_pose_w: 0.5              # 关节姿态奖励权重
reward_vel_w: 0.1
reward_pose_scale: 0.25         # 指数衰减尺度
joint_err_w: [1.0, 0.6, ...]    # 各关节误差权重
```

### 2.4.3 AMP/ADD/SMP 专用字段

```yaml
num_disc_obs_steps: 10          # 判别器观测历史步数
disc_dof_vel_obs: true          # 判别器观测是否含 DOF 速度
```

### 2.4.4 环境配置清单（35 个）

按 `env_name` 分组：

- **deepmimic (6)**：humanoid, g1, go2, smpl, pi_plus, sword_shield
- **amp (9)**：humanoid, g1, go2, smpl, pi_plus, location, steering, sword_shield 变体
- **ase (2)**：humanoid, sword_shield
- **add (5)**：humanoid, g1, go2, smpl, pi_plus
- **smp (4)**：humanoid, location, steering, dodgeball
- **view_motion (6)**：各角色动作预览
- **static_objects (2)**：humanoid/g1 vault
- **char_dof_test (1)**：关节范围测试

---

## 2.5 Agent 配置（`data/agents/`）

### 2.5.1 通用字段

```yaml
agent_name: "PPO"               # PPO/AWR/AMP/ASE/ADD/LCP/SMP

model:
  actor_net: "fc_2layers_1024units"
  critic_net: "fc_2layers_1024units"
  actor_std_type: "FIXED"         # FIXED / CONSTANT / VARIABLE
  action_std: 0.05

discount: 0.99
steps_per_iter: 32
iters_per_output: 100
td_lambda: 0.95
```

### 2.5.2 PPO 专用

```yaml
ppo_clip_ratio: 0.2
norm_adv_clip: 4.0
actor_epochs: 5
critic_epochs: 2
action_bound_weight: 10.0
```

### 2.5.3 AMP 专用

```yaml
disc_reward_weight: 1.0
task_reward_weight: 0.0
disc_reward_scale: 2.0
disc_grad_penalty: 5.0
disc_buffer_size: 100000
```

### 2.5.4 SMP 专用

```yaml
smp_prior_cfg: "path/to/diffusion_config.yaml"
smp_prior_model: "path/to/model.pt"
smp_reward_weight: 1.0
sds_loss_scale: 0.5
diffusion_steps: [22, 15, 8]
enable_gsi: true
```

---

## 2.6 Builder 模式

### 2.6.1 `env_builder.build_env`

根据 `env_name` 分发到具体环境类：

```python
# env_builder.py 支持的环境名
"char", "deepmimic", "amp", "ase", "add", "smp",
"char_dof_test", "view_motion",
"task_location", "task_steering", "task_dodgeball", "static_objects"
```

### 2.6.2 `agent_builder.build_agent`

根据 `agent_name` 分发：

```python
"Dummy", "PPO", "AWR", "AMP", "SMP", "ASE", "ADD", "LCP"
```

### 2.6.3 `engine_builder.build_engine`

根据 `engine_name` 分发：`isaac_gym` / `isaac_lab` / `newton`

### 2.6.4 `net_builder.build_net`

动态导入 `learning/nets/fc_2layers_*units.py` 等模块。

---

## 2.7 分布式训练

`run.py` 使用 `torch.multiprocessing` spawn 模式：

1. 每个 `--devices` 条目启动一个进程
2. `mp_util.init(rank, num_procs, device, master_port)` 初始化通信
3. `MPOptimizer` 在 `step()` 时跨进程平均梯度
4. `Normalizer` 跨进程合并统计量
5. 仅 root 进程（rank 0）写日志和保存模型

---

## 2.8 日志与可视化

### 2.8.1 日志类型

| Logger | 输出 |
|--------|------|
| `txt` | `out_dir/log.txt` |
| `tb` | TensorBoard events 文件 |
| `wandb` | Weights & Biases 云端 |

### 2.8.2 可视化 UI 快捷键

- **Alt + 左键拖拽**：平移相机
- **滚轮**：缩放
- **Enter**：暂停/继续
- **Space**：单步仿真

### 2.8.3 绘制训练曲线

```bash
tensorboard --logdir=output/ --port=6006
# 或
python tools/plot_log/plot_log.py  # 绘制 log.txt
```

---

## 2.9 输出目录结构

训练完成后 `out_dir/` 包含：

```
output/
├── model.pt              # 最终模型
├── log.txt               # 训练日志
├── engine_config.yaml    # 配置副本
├── env_config.yaml
└── agent_config.yaml
```

可选 `int_models/model_XXXXXXXXXX.pt` 中间检查点（`--save_int_models true`）。

---

[← 架构总览](01-architecture-overview.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：动画模块 →](03-animation-module.md)
