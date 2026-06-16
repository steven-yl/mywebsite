---
title: 03 — π₀-FAST — 自回归动作 Token 模型
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

# 第 3 章：π₀-FAST — 自回归动作 Token 模型

源码：`src/openpi/models/pi0_fast.py`、`tokenizer.FASTTokenizer`。

## 3.1 算法动机

π₀-FAST 将**动作轨迹离散化**为 token 序列，在 PaliGemma 词表子空间内用**标准因果语言建模**预测动作，训练目标为 token 交叉熵。优势是与大规模 VLM 预训练对齐、推理可用 KV cache；代价是自回归步数与序列长度相关。

FAST（Fine-grained Action Sequence Tokenization）由 Physical Intelligence 提出，将连续动作压缩为较短 token 串，再映射到 PaliGemma 词表尾部。

## 3.2 与 π₀ 的架构对比

| 项目 | π₀ / π₀.₅ | π₀-FAST |
|------|-----------|---------|
| LLM | 双塔（2B + 300M expert） | 单 `gemma_fast.Module`（2B） |
| 动作表示 | 连续 `[H,D]` flow | Token 序列（在 prompt postfix） |
| 损失 | Flow MSE | Token CE |
| `sample_actions` 返回 | `[B,H,D]` float | **Token id** `[B, max_decoding_steps]`（需 `ExtractFASTActions`） |
| 默认相机键 | 3× manipulator | `base_0_rgb`, `base_1_rgb`, `wrist_0_rgb` |

## 3.3 序列格式（FASTTokenizer）

训练时构造（见 `tokenizer.py`）：

```text
[ BOS ] Task: {prompt}, State: {离散状态}; \n Action: {FAST tokens...} | [EOS]
|<──────── prefix: ar_mask=0 ────────>|← postfix: ar_mask=1, loss_mask=True →|
```

- 状态：归一化到 \([-1,1]\) 后 **256-bin** `digitize`，空格分隔写入文本。
- FAST token 经 `_act_tokens_to_paligemma_tokens` 映射到 `vocab_size - 1 - 128 - fast_id`（保留 PaliGemma 特殊 token）。

推理时无 postfix，模型自回归生成直至 `PALIGEMMA_EOS_TOKEN = 1` 或达到 `max_decoding_steps`。

## 3.4 配置 `Pi0FASTConfig`

| 字段 | 默认 | 说明 |
|------|------|------|
| `dtype` | bfloat16 | |
| `paligemma_variant` | gemma_2b | 可 `gemma_2b_lora` |
| `action_dim` | 32 | state pad 维 |
| `action_horizon` | 32 | 解码后动作步数 |
| `max_token_len` | 250 | |
| `fast_model_tokenizer` | None | 可注入 HF processor |
| `model_type` | `PI0_FAST` | |
| `create` / `inputs_spec` / `get_freeze_filter` | 同 `Pi0Config` 模式 | |

`inputs_spec` 额外要求 `token_ar_mask`、`token_loss_mask`。

## 3.5 类 `Pi0FAST` 方法

### 辅助函数

| 函数 | 作用 |
|------|------|
| `make_attn_mask` | 与 π₀ 相同语义的块因果掩码 |
| `left_to_right_align` | 将变长序列右对齐，便于解码与 cache 布局 |
| `put_along_last_axis` | JAX 版 `put_along_axis`，写入解码 token |

### `embed_inputs(obs)`

1. 各图像 → SigLIP token，`ar_mask=0`（双向）。
2. `tokenized_prompt` → `llm(..., embed_only=True)`。
3. 拼接 `token_ar_mask`（来自 tokenizer）。
4. 返回 `(embeddings, input_mask, ar_mask)`。

### `compute_loss(rng, observation, actions, train=False)`

1. `preprocess_observation`（训练时可增强）。
2. `embed_inputs` → `attn_mask`。
3. **Teacher forcing**：输入 `embedded_prefix[:, :-1]`，预测 `tokenized_prompt[:, 1:]` 的 one-hot 目标。
4. 仅对 `pre_logits` 最后 `targets.shape[1]` 段做 vocab 投影，节省显存。
5. 损失：
\[
\mathcal{L} = -\frac{\sum_i \log p(x_i \mid x_{<i}) \cdot m_i}{\sum_i m_i}
\]
其中 \(m_i\) 为 `token_loss_mask[:,1:]`（prefix 不计算 loss）。

### `sample_actions(rng, observation, max_decoding_steps=256, temperature=0.0)`

1. `embed_inputs` + `left_to_right_align`。
2. **Prefill**：`decode=True` 填充 KV cache，长度 `prefill_size + max_decoding_steps`。
3. **自回归循环**（`jax.lax.while_loop`）：
   - `temperature>0`：`categorical(logits/T)`；否则 `argmax`。
   - 写入 `output_tokens`；检测 EOS 全员则停。
   - 单 token embed → 带因果 mask 的 decode 步更新 cache。
4. 返回 `output_tokens`（**非**连续动作；下游 `ExtractFASTActions` 解析）。

```python
# 推理后处理（transforms.ExtractFASTActions）
actions = FASTTokenizer.extract_actions(
    tokens, action_horizon=32, action_dim=32
)  # -> ndarray [H, D]
```

### `extract_actions`（Tokenizer）

1. `decode` 全文，截取 `Action: ` 与 `|` 之间子串。
2. 再 encode 得 raw token → 逆映射 FAST id → `fast_tokenizer.decode(..., time_horizon, action_dim)`。

## 3.6 数据流中的 FAST 专用变换

| Transform | 阶段 | 作用 |
|-----------|------|------|
| `TokenizeFASTInputs` | 训练/推理输入 | 生成 token 四元组 |
| `ExtractFASTActions` | 推理输出 | token → 连续动作 |

## 3.7 适用场景

- **DROID  tabletop**：`pi0_fast_droid` 等 checkpoint，语言+视觉泛化较好。
- **低延迟批量推理**：相比 flow 多步积分，单次 decode 步数可调。
- **不适合**：需要 PyTorch 仅推理栈时（当前无 PyTorch FAST）。

## 3.8 与第 2 章、第 7 章关系

- Flow 模型数学与 KV cache 用法不同，但共享 `preprocess_observation` 与 SigLIP 视觉塔。
- FAST 词表与 `gemma_fast` 见 [07-backbone-tokenizers.md](./07-backbone-tokenizers.md)。
