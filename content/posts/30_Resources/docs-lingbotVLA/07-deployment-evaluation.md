---
title: "7. 部署与评估"
subtitle: ""
date: 2026-07-13T12:00:00+08:00
draft: false
authors: [Steven]
description: "7. 部署与评估。"
summary: "7. 部署与评估。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA, robots]
series: [lingbotVLA-docs]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 7. 部署与评估

---

## 7.1 部署架构

```
Robot / Simulator Client
    │  WebSocket (msgpack+numpy)
    ▼
websocket_policy_server.py
    │  infer(obs) / reset()
    ▼
LingbotVLAServer
    ├── LingBotVlaInferencePolicy (LingbotVlaPolicy + mixin)
    ├── FeatureTransform (robot config + norm)
    └── sample_actions (Flow Matching)
```

**特点：** 单 GPU 推理，无分布式；支持 action chunk 逐步执行或 ensemble。

---

## 7.2 核心文件

| 文件 | 职责 |
|------|------|
| `deploy/lingbot_vla_policy.py` | 模型加载、预处理、CLI 入口 |
| `deploy/websocket_policy_server.py` | Async WebSocket 服务，`/healthz` |
| `deploy/websocket_client_policy.py` | 同步客户端 `infer()` / `reset()` |
| `deploy/msgpack_numpy.py` | numpy 数组高效序列化 |
| `deploy/image_tools.py` | uint8 转换、resize-with-pad |

---

## 7.3 `LingbotVLAServer`

### 初始化参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `path_to_pi_model` | — | HF checkpoint 目录 |
| `use_length` | 1 | 每次执行 action chunk 前 N 步；-1 用 ensemble |
| `num_denoising_step` | 10 | Flow Matching Euler 步数 |
| `use_compile` | False | torch.compile qwenvl_with_expert |
| `use_bf16` / `use_fp32` | bf16 | 精度（互斥） |
| `robot_norm_path` | None | 覆盖 norm stats |
| `action_ensemble_horizon` | 8 | temporal ensemble 窗口 |
| `adaptive_ensemble_alpha` | 0.1 | ensemble 衰减系数 |

### `load_vla(model_path)`

1. 读取 `config.json` + `lingbotvla_cli.yaml`
2. `merge_qwen_config()` — 合并 Qwen2.5-VL text/vision 配置
3. `LingBotVlaInferencePolicy.from_pretrained` 或 safetensors 手动加载
4. 构建 `FeatureTransform`（robot config + norm stats）
5. 可选 `torch.compile(model.qwenvl_with_expert)`

**环境变量：** `QWEN25_PATH=path/to/Qwen2.5-VL-3B-Instruct`（合并 config 用）

### Checkpoint 要求

`train.output_dir/checkpoints/*/hf_ckpt/` 需包含：

- `*.safetensors`
- `config.json`
- `lingbotvla_cli.yaml`

---

## 7.4 `PolicyPreprocessMixin.select_action`

```python
@torch.no_grad()
def select_action(self, observation, use_bf16=False, noise=None, num_denoising_step=10):
    # observation: images, img_masks, state, lang_tokens, lang_masks
    actions = self.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state,
        num_steps=num_denoising_step
    )
    observation['actions'] = actions
    data = self.feature_transform.unapply(observation)
    return data  # 物理单位动作
```

**推理 attention：** deploy 通常设 `attention_implementation: eager`（低延迟、兼容性好）。

---

## 7.5 WebSocket 协议

### Server (`websocket_policy_server.py`)

- 监听 `--port`（默认 8006）
- 接收 msgpack 编码 obs dict
- 返回 `{actions, server_timing}`

### Client 示例

```python
from deploy.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="localhost", port=8006)
client.reset()
action = client.infer({
    "images": ...,      # uint8 or float tensor
    "state": ...,
    "prompt": "open the microwave",
})
```

---

## 7.6 启动命令

### RoboTwin 仿真

```bash
export QWEN25_PATH=path/to/Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_vla_policy \
    --model_path output/checkpoints/global_step_20000/hf_ckpt \
    --use_compile \
    --use_length 50 \
    --port 8006
```

### 真实机器人

```bash
python -m deploy.lingbot_vla_policy \
    --model_path path/to/hf_ckpt \
    --use_compile \
    --use_length 25
# 加速：--num_denoising_step 5
```

---

## 7.7 开环评估

**脚本：** `scripts/open_loop_eval.py`

在 LeRobot 验证集上对比 GT action vs 预测 action，绘制曲线。

```bash
export QWEN25_PATH=Qwen/Qwen2.5-VL-3B-Instruct
python scripts/open_loop_eval.py \
    --model_path path/to/hf_ckpt \
    --data_path path/to_val_data \
    --use_length 50
```

省略 `--data_path` 时使用训练 YAML 中的 `data.train_path`。

---

## 7.8 仿真 benchmark

| 平台 | 目录 | 说明 |
|------|------|------|
| RoboTwin 2.0 | `experiment/robotwin/` | 连接 WebSocket policy |
| LIBERO | `experiment/libero/` | `run_libero_eval.py` |

RoboTwin 客户端连接 deploy server，逐步发送 obs 接收 action chunk。

---

## 7.9 Action Chunk 执行策略

VLA 一次预测 `chunk_size`（如 50）步未来动作。部署时：

| 模式 | `use_length` | 行为 |
|------|--------------|------|
| 逐步执行 | 1–N | 每 N 步重新 inference |
| 长 horizon | 50 | 一次 chunk 内开环执行 |
| Ensemble | -1 | 重叠 chunk 加权平均（PI0 风格） |

**权衡：** 短 `use_length` → 更闭环、更慢；长 horizon → 快但误差累积。

---

## 7.10 推理优化

| 优化 | 效果 |
|------|------|
| `--use_compile` | 首次慢，后续 10–30% 加速 |
| `--num_denoising_step 5` | 线性减少 FM 步数 |
| bf16 | 默认，VRAM↓ |
| prefix KV cache | 已实现，每步只算 suffix |
| eager vs flex | deploy 用 eager 避免 compile 依赖 |

---

## 7.11 辅助脚本

| 脚本 | 用途 |
|------|------|
| `scripts/download_hf_model.py` | 下载 HF checkpoint |
| `scripts/download_hf_data.py` | 下载数据集 |
| `scripts/mereg_dcp_to_hf.py` | DCP → HF safetensors |
| `scripts/compute_norm.py` | 归一化统计 |
| `scripts/merge_lerobot_v21.py` | 合并 LeRobot 数据集 |

---

## 7.12 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| PI0Config 字段解码错误 | 旧版 checkpoint | 重下载或迁移字段到 `lingbotvla_cli.yaml` |
| OOM | chunk_size 过大 / 多相机 | 减 batch、用 bf16 |
| 动作漂移 | norm stats 不匹配 | 确认 `norm_stats_file` 与训练一致 |
| 延迟高 | num_steps=10 | 降至 5，启用 compile |

---

## 7.13 相关链接

- [LingBot-VLA HuggingFace Collection](https://huggingface.co/collections/robbyant/lingbot-vla)
- [RoboTwin 2.0](https://robotwin-platform.github.io/)
- [LeRobot Policy API](https://github.com/huggingface/lerobot)
