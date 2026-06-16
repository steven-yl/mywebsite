---
title: 08 — 客户端与运行时
subtitle: ""
date: 2026-06-17T10:26:59+08:00
# lastmod: 2026-06-17T10:26:59+08:00
draft: false
authors: [Steven]
description: ""
tags: [openpi]
categories: [openpi]
series: [openpi-docs]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 8 章：客户端与运行时

源码：`packages/openpi-client/`。独立于 JAX，可安装在机器人控制机。

## 8.1 包结构

```text
openpi_client/
├── base_policy.py          # BasePolicy 抽象
├── websocket_client_policy.py
├── action_chunk_broker.py
├── image_tools.py
├── msgpack_numpy.py
└── runtime/
    ├── environment.py    # Environment 抽象
    ├── agent.py            # Agent 抽象
    ├── subscriber.py       # Subscriber 抽象
    ├── agents/policy_agent.py
    └── runtime.py          # 控制循环
```

## 8.2 `BasePolicy`

```python
class BasePolicy(abc.ABC):
    def infer(self, obs: dict) -> dict: ...
    def reset(self) -> None: ...  # 默认空实现
```

服务端 `Policy` 与客户端 `WebsocketClientPolicy` 均实现此接口，便于嵌套包装（如 `ActionChunkBroker`）。

## 8.3 `WebsocketClientPolicy`

| 成员 | 说明 |
|------|------|
| `__init__(host, port)` | 连接 `ws://host:port` |
| `infer(obs)` | msgpack 发送观测，接收动作 dict |
| `reset()` | 可选通知服务端重置 |
| `_ws` | 持久连接；断线重连逻辑见源码 |

首包接收 server `metadata`（与 `Policy.metadata` 一致）。

依赖：`websockets`、`msgpack_numpy` 扩展。

## 8.4 `msgpack_numpy`

| 函数 | 作用 |
|------|------|
| `pack_array` / `unpack_array` | ndarray ↔ msgpack 二进制 |
| 注册 `msgpack` default/hook | 避免 JSON 浮点精度损失 |

WebSocket 载荷为 **msgpack**，非 JSON。

## 8.5 `image_tools`

| 函数 | 说明 |
|------|------|
| `convert_to_uint8` | float [0,1] 或 int 转 uint8 |
| `resize_with_pad` | 保持宽高比 resize + pad 到目标 `(H,W)` |
| `_resize_with_pad_pil` | PIL 实现 |

与训练侧 `transforms.ResizeImages` 共用，保证机器人端相机预处理一致。

## 8.6 `ActionChunkBroker`

**问题**：策略一次返回 `[action_horizon, action_dim]`，控制循环通常每步只要一个动作。

**机制**：

1. 首次 `infer` 调用内层 policy，缓存完整结果。
2. 随后 `action_horizon-1` 次逐步切片 `x[cur_step]`。
3. 耗尽后下次 `infer` 再请求新 chunk。

```python
broker = ActionChunkBroker(
    WebsocketClientPolicy(host="localhost", port=8000),
    action_horizon=50,
)
for t in range(1000):
    action = broker.infer(obs)["actions"]  # shape [D] 每步
```

`reset()` 清空缓存并转发内层 `reset()`。

## 8.7 Runtime 框架

用于示例与评测脚本的标准 **感知-决策-执行** 循环。

### `Environment`（抽象）

- `reset()` → 初始观测
- `get_observation()` → 当前观测 dict
- `apply_action(action)` → 执行
- `is_episode_complete()` → 终止条件

### `Agent`（抽象）

- `get_action(obs)` → 动作
- `reset()` → episode 开始

### `PolicyAgent`

将 `BasePolicy.infer(obs)["actions"]` 作为 `get_action` 输出（常与 `ActionChunkBroker` 联用）。

### `Subscriber`（抽象）

- `on_episode_start` / `on_episode_end`
- `on_step(obs, action)` — 日志、录屏、指标

### `Runtime`

| 参数 | 说明 |
|------|------|
| `max_hz` | 控制频率上限；0 表示不睡眠 |
| `num_episodes` | episode 数 |
| `max_episode_steps` | 0 为无限，直到 `mark_episode_complete` |

**`run()`**：重复 `_run_episode` → 最后 `environment.reset()` 回 home。

**`_step()`**：`obs = env.get_observation()` → `action = agent.get_action(obs)` → `env.apply_action(action)` → 通知 subscribers。

**`run_in_new_thread()`**：后台线程跑循环。

## 8.8 端到端部署架构

```text
┌──────────────────── GPU 服务器 ────────────────────┐
│  serve_policy.py → WebsocketPolicyServer         │
│       → Policy (JAX/PyTorch π₀)                  │
└───────────────────────┬──────────────────────────┘
                        │ ws + msgpack
┌───────────────────────▼──────────────────────────┐
│  机器人控制机                                      │
│  ActionChunkBroker(WebsocketClientPolicy)        │
│       → Runtime(PolicyAgent, RobotEnvironment)   │
└──────────────────────────────────────────────────┘
```

## 8.9 最小客户端示例

```python
import numpy as np
from openpi_client import websocket_client_policy as wcp
from openpi_client import action_chunk_broker as acb

policy = acb.ActionChunkBroker(
    wcp.WebsocketClientPolicy(host="127.0.0.1", port=8000),
    action_horizon=50,
)

obs = {
    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(8, dtype=np.float32),
    "prompt": "do something",
}
for _ in range(10):
    out = policy.infer(obs)
    print(out["actions"].shape)  # 单步动作维
```

需与服务器端 `TrainConfig` 对应的键名一致（或通过 repack 对齐）。

## 8.10 与主库关系

| 组件 | 主库 `openpi` | `openpi-client` |
|------|---------------|-----------------|
| 模型推理 | ✓ | ✗ |
| WebSocket 服务 | ✓ | ✗ |
| WebSocket 客户端 | ✗ | ✓ |
| 图像 resize | `shared/image_tools` | `image_tools`（同算法） |

## 8.11 章节边界

- 服务端 Policy 与变换 → [06-inference-policy-serving.md](./06-inference-policy-serving.md)
- 网络部署 → [../remote_inference.md](../remote_inference.md)
