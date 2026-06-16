---
title: 02 — π₀ 与 π₀.₅ — Flow Matching 动作模型
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

# 第 2 章：π₀ 与 π₀.₅ — Flow Matching 动作模型

源码：`src/openpi/models/pi0.py`、`pi0_config.py`，PyTorch 镜像：`models_pytorch/pi0_pytorch.py`。

## 2.1 算法动机

传统行为克隆直接回归 \(a = f(o)\)，多模态观测下易平均化。π₀ 将**未来动作轨迹** \(\mathbf{a}_{1:H} \in \mathbb{R}^{H \times D}\) 视为连续变量，用 **Flow Matching** 学习从噪声到真实动作的速度场 \(v_\theta(o, \mathbf{x}_t, t)\)，推理时沿场积分得到动作块。

**为何用动作块（chunk）**：一次预测未来 \(H\) 步，配合 `ActionChunkBroker` 开环执行若干步后再重规划，平衡延迟与稳定性。

## 2.2 π₀ 与 π₀.₅ 的结构差异

由 `Pi0Config.pi05` 控制（`pi05=True` 时 `model_type` 为 `PI05`）。

| 组件 | π₀ | π₀.₅ |
|------|-----|------|
| 状态 | `state_proj` → 1 个 suffix token | 256-bin 离散化写入 prompt（`PaligemmaTokenizer` + `discrete_state_input`） |
| 时间条件 | `action_time_mlp` 与 action token 拼接 | `time_mlp_in/out` → **adaRMSNorm** 条件 `adarms_cond` |
| `max_token_len` 默认 | 48 | 200 |
| Action Expert | Gemma 300M，标准 RMSNorm | 同结构 + adaRMS |

配置类 `Pi0Config` 字段：

- `dtype`：默认 `"bfloat16"`
- `paligemma_variant` / `action_expert_variant`：如 `gemma_2b` / `gemma_300m`，支持 `*_lora` 变体
- `action_dim`（32）、`action_horizon`（50）
- `get_freeze_filter()`：LoRA 微调时冻结非 LoRA 的 LLM 权重

## 2.3 网络结构示意

```text
Prefix（双向注意力）                    Suffix（因果 + 前缀可见）
┌─────────────────────────────────┐   ┌──────────────────────────┐
│ SigLIP tokens (每相机一路)       │   │ [state_token] (仅 π₀)     │
│ + PaliGemma 文本 token          │   │ action tokens × H         │
│  图像↔语言 全连接               │   │  仅看 prefix + 因果看自身  │
└─────────────────────────────────┘   └──────────────────────────┘
         │                                      │
         └────────── Dual Gemma LLM ────────────┘
                              │
                    action_out_proj → v_t 或推理积分
```

`PaliGemma` 为 `nnx.Dict(llm=..., img=...)`：`llm` 为**双配置** Gemma（前缀 2B + 动作专家 300M），单次 forward 接收 `[prefix_tokens, suffix_tokens]`。

## 2.4 注意力掩码：`make_attn_mask`

```python
# pi0.py — 逻辑摘要
cumsum = jnp.cumsum(mask_ar, axis=1)
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
return jnp.logical_and(attn_mask, valid_mask)
```

**`mask_ar`**：在位置为 True 处开始新的自回归块；False 表示与前一 token 共享同一注意力组。

π₀ 中典型模式：

- 图像、语言：`ar_mask=False`（块内双向）
- π₀ 的 state token：`True`（suffix 起点）
- 动作 token：首个 `True`，其余 `False`（动作序列内因果）

这使 prefix 不能 attend 到 suffix，suffix 可 attend prefix。

## 2.5 位置编码：`posemb_sincos`

对标量时间步 \(t\) 生成 sin/cos 特征，频率在 `[min_period, max_period]` 对数间隔（默认 4e-3～4.0），用于条件化动作 token。

## 2.6 类 `Pi0` 方法详解

### `__init__(config, rngs)`

构建 `PaliGemma.llm`、`PaliGemma.img`（SigLIP So400m/14）、`action_in_proj`、`action_out_proj`，及 π₀/π₀.₅ 分支中的 state/time MLP。

### `embed_prefix(obs) -> (tokens, input_mask, ar_mask)`

1. 遍历 `obs.images`：SigLIP 编码 → 展平为 token 序列；mask 按 batch 广播。
2. 若有 `tokenized_prompt`：`llm.embed` 得到语言 token。
3. 拼接所有 prefix token 与掩码。

### `embed_suffix(obs, noisy_actions, timestep) -> (tokens, input_mask, ar_mask, adarms_cond)`

1. π₀：投影 `obs.state` 为单 token。
2. `action_in_proj(noisy_actions)` 得动作 token。
3. `posemb_sincos(timestep)` + π₀.₅ 的 time MLP 或 π₀ 的 action-time MLP。
4. 拼接 suffix；返回 `adarms_cond`（π₀.₅）或 `None`。

### `compute_loss(rng, observation, actions, train=False)`

**Flow Matching 训练目标**（代码约定 \(t=1\) 为噪声，\(t=0\) 为数据）：

采样：
\[
\epsilon \sim \mathcal{N}(0, I), \quad t \sim \mathrm{Beta}(1.5, 1) \cdot 0.999 + 0.001
\]
\[
\mathbf{x}_t = t \cdot \epsilon + (1-t) \cdot \mathbf{a}, \quad \mathbf{u}_t = \epsilon - \mathbf{a}
\]

网络预测速度 \(\mathbf{v}_t = f_\theta(o, \mathbf{x}_t, t)\)，损失：
\[
\mathcal{L} = \mathrm{mean}_{d} \big\| \mathbf{v}_t - \mathbf{u}_t \big\|^2
\]
（对 `action_dim` 维求均值，返回 `[*batch, action_horizon]`。）

流程：合并 prefix+suffix → `make_attn_mask` → 双塔 LLM → `action_out_proj` 取最后 \(H\) 个 suffix 位置。

### `sample_actions(rng, observation, num_steps=10, noise=None)`

**Euler ODE 求解**：

1. 初始化 \(\mathbf{x} \leftarrow \epsilon\)（或传入 `noise`），\(t \leftarrow 1\)。
2. **Prefix 一次前向**填充 KV cache（不再重算图像/语言）。
3. 循环 `num_steps` 次：
   - \(\mathbf{v}_t \leftarrow f_\theta(\ldots)\)
   - \(\mathbf{x} \leftarrow \mathbf{x} + \Delta t \cdot \mathbf{v}_t\)，\(\Delta t = -1/\texttt{num\_steps}\)
   - \(t \leftarrow t + \Delta t\)
4. 当 \(t < -\Delta t/2\) 停止，返回 \(\mathbf{x}\) 作为动作块。

```python
# 推理积分（概念代码，与 pi0.py 一致）
dt = -1.0 / num_steps
x_t, time = noise, 1.0
while time >= -dt / 2:
    v_t = model_forward_suffix_with_kv_cache(obs, x_t, time)
    x_t = x_t + dt * v_t
    time = time + dt
return x_t  # shape [B, H, D]
```

## 2.7 配置 `Pi0Config` 完整 API

| 成员 | 说明 |
|------|------|
| `__post_init__` | 默认 `max_token_len`、`discrete_state_input` 随 `pi05` 设置 |
| `model_type` | `PI05` 或 `PI0` |
| `create(rng)` | 返回 `Pi0` 实例 |
| `inputs_spec(batch_size)` | 三路 224 图像 + state + tokenized_prompt |
| `get_freeze_filter()` | LoRA 训练冻结策略 |

## 2.8 使用场景与超参

| 场景 | 建议 |
|------|------|
| 双臂 ALOHA、固定台面 | `pi0_*` / `pi05_*` aloha 配置 |
| 语言跟随、新物体 | 优先 `pi05_*` checkpoint |
| 内存紧张推理 | 减少 `num_steps`（略损质量）或 LoRA checkpoint |
| 微调 | `compute_norm_stats` → `train.py`；可选 `AssetsConfig` 复用预训练 norm |

## 2.9 与 PyTorch 实现对应关系

`PI0Pytorch` 提供 `forward`（训练损失）、`sample_actions`、`embed_prefix`、`embed_suffix`、`denoise_step`，逻辑与 JAX 对齐；通过 `train_config.model.load_pytorch` 加载。

## 2.10 章节边界

- 自回归 FAST 模型见 [03-models-pi0-fast.md](./03-models-pi0-fast.md)
- Gemma/SigLIP 实现见 [07-backbone-tokenizers.md](./07-backbone-tokenizers.md)
- 训练循环见 [05-training-system.md](./05-training-system.md)
