---
title: "08 仿真环境"
subtitle: ""
date: 2026-07-10T17:44:00+08:00
draft: false
authors: [Steven]
description: "12 种 EnvConfig、Gymnasium 封装与 HIL-SERL 人机在环强化学习。"
summary: "LeRobot 仿真环境配置与 HIL-SERL。"
tags: [lerobot, robots]
categories: [docs lerobot, robots]
series: [lerobot-docs]
weight: 8
series_weight: 8
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 08 — 仿真环境（Envs）

## 1. 模块边界

```
envs/
├── configs.py        # EnvConfig 基类 + 各环境 dataclass
├── factory.py        # make_env, make_env_config
├── utils.py          # 辅助
├── libero.py         # LIBERO 向量环境
├── metaworld.py
├── robocasa.py
├── vlabench.py
├── robotwin.py
└── robomme.py
```

**依赖**：Gymnasium（核心依赖）；各 benchmark 为 optional extras。

---

## 2. EnvConfig 抽象

**文件**：`envs/configs.py`

### 2.1 公共字段

| 字段 | 说明 |
|------|------|
| `task` | 任务名/suite |
| `fps` | 控制频率（与 policy 对齐） |
| `features` / `features_map` | 观测/动作 feature 定义 |
| `max_parallel_tasks` | 并行任务数上限 |

### 2.2 抽象接口

| 成员 | 作用 |
|------|------|
| `gym_id` | 如 `gym_aloha/AlohaInsertion-v0` |
| `gym_kwargs` | 传给 `gym.make()` |
| `create_envs(n_envs, use_async_envs)` | → `{suite: {task_id: VectorEnv}}` |
| `get_env_processors()` | 环境专用 pre/post processor |

### 2.3 Hub 环境

`HubEnvConfig.hub_path` — 从 Hugging Face Hub 拉取第三方 env 包（EnvHub 机制）。

---

## 3. 全部 12 种环境

| `--env.type` | Config | 来源/特点 |
|--------------|--------|-----------|
| `aloha` | `AlohaEnv` | `gym-aloha`，14-DOF 双臂插入等 |
| `pusht` | `PushtEnv` | `gym-pusht`，2D 推块 |
| `gym_manipulator` | `HILSerlRobotEnvConfig` | **真机** HIL-SERL；内嵌 `RobotConfig`+`TeleoperatorConfig` |
| `libero` | `LiberoEnv` | LIBERO 多任务；需 Linux + `hf-libero` |
| `libero_plus` | `LiberoPlusEnv` | LIBERO 鲁棒性变体 |
| `metaworld` | `MetaworldEnv` | MetaWorld MT1/MT50 |
| `robocasa` | `RoboCasaEnv` | 厨房场景（手动安装 robocasa） |
| `vlabench` | `VLABenchEnv` | VLA 基准（GitHub 安装 VLABench） |
| `isaaclab_arena` | `IsaaclabArenaEnv` | Hub: `nvidia/isaaclab-arena-envs` |
| `robotwin` | `RoboTwinEnvConfig` | SAPIEN 双臂 50 任务 |
| `robomme` | `RoboMMEEnv` | ManiSkill 记忆增强（Docker 镜像） |

---

## 4. make_env 工厂

**文件**：`envs/factory.py`

```python
envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=True)
# envs: dict[suite_name, dict[task_id, VectorEnv]]
```

- 向量化：`gymnasium.vector.SyncVectorEnv` / `AsyncVectorEnv`
- LIBERO 等使用专用 `create_libero_envs()`

---

## 5. Feature 与 Policy 对齐

`env_to_policy_features(env_cfg)` 将环境观测/动作空间转为 `PolicyFeature` dict，供 `make_policy(..., env_cfg=cfg.env)` 在**无 dataset** 时推断输入输出维度。

环境 processor（如 `LiberoProcessorStep`）在 eval 时将 raw obs 转为 LeRobot 命名约定。

---

## 6. HIL-SERL 真机环境

`gym_manipulator` **不是纯仿真**：

```
HILSerlRobotEnvConfig
├── robot: RobotConfig
├── teleop: TeleoperatorConfig
└── processor: HILSerlProcessorConfig
    ├── gamepad 控制
    ├── 可选 IK (placo)
    └── reward classifier processor
```

用于 `lerobot[hilserl]` 在线 RL + 人类干预。

---

## 7. Eval 中的环境交互

```
obs = vector_env.reset()
loop:
  batch = preprocessor(obs)
  action = policy.select_action(batch)
  action = postprocessor(action)
  obs, reward, done, truncated, info = vector_env.step(action)
  if done: policy.reset()
```

`lerobot_eval.py` 聚合 multi-episode success rate。

---

## 8. 环境 extras 与平台限制

| 环境 | extra | 限制 |
|------|-------|------|
| aloha | `lerobot[aloha]` | |
| pusht | `lerobot[pusht]` | |
| libero | `lerobot[libero]` | Linux |
| metaworld | `lerobot[metaworld]` | |
| robocasa | 手动安装 | 与 lerobot 版本 pin 冲突 |
| vlabench | 手动安装 | 无 PyPI |
| robomme | Docker | numpy 版本冲突 |

---

## 9. 示例

### 9.1 PushT 评估

```bash
uv sync --locked --extra evaluation --extra pusht --extra training

lerobot-eval \
  --policy.path=lerobot/act_pusht \
  --env.type=pusht \
  --eval.n_episodes=20 \
  --eval.batch_size=10
```

### 9.2 编程式创建环境

```python
"""创建 PushT 向量环境（需 gym-pusht）。"""
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env

cfg = PushtEnv()
envs = make_env(cfg, n_envs=2)
suite = next(iter(envs.values()))
vec = next(iter(suite.values()))
obs, info = vec.reset()
print(obs.keys() if isinstance(obs, dict) else type(obs))
vec.close()
```

---

## 10. EnvHub 扩展

第三方可发布 Hub repo，内含：

- 注册 `@EnvConfig.register_subclass`
- `create_envs` 实现
- 依赖声明

用户：

```bash
lerobot-eval --env.type=my_env --env.hub_path=user/my-env-package
```

详见 [envhub.mdx](https://huggingface.co/docs/lerobot/envhub)。

---

## 下一章

- 真机部署 → [09-rollout-inference.md](./09-rollout-inference.md)
