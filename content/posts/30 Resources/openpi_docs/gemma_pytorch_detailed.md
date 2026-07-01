---
title: "PaliGemmaWithExpertModel 详细技术文档"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "PaliGemmaWithExpertModel 双塔共享注意力架构的 PyTorch 实现、前向流程与张量形状详解。"
summary: "OpenPI 核心模块 PaliGemmaWithExpertModel 深度技术解读。"
tags: [openpi, robots, PyTorch]
categories: [docs openpi]
series: [openpi-docs]
weight: 11
series_weight: 11
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# PaliGemmaWithExpertModel 详细技术文档

## 概述

`PaliGemmaWithExpertModel` 是 OpenPI (π₀) 项目的核心神经网络模块，位于 `src/openpi/models_pytorch/gemma_pytorch.py`。它实现了一种**双塔共享注意力架构**，将视觉语言模型 (VLM) 与动作专家模型结合，用于从多模态观测（图像+语言指令）生成机器人动作序列。

## 架构总览

```
输入层:
  ┌─────────────────┐        ┌─────────────────────┐
  │  图像 + 语言Token │        │   动作Token (Flow)   │
  │  (Observation)   │        │  (Action Sequence)   │
  └────────┬────────┘        └──────────┬──────────┘
           │                            │
           ▼                            ▼
  ┌─────────────────┐        ┌─────────────────────┐
  │  PaliGemma VLM  │        │  Gemma Action Expert │
  │  embed_image()  │        │  (外部提供embedding)  │
  │  embed_tokens() │        │                     │
  └────────┬────────┘        └──────────┬──────────┘
           │                            │
           ▼                            ▼
  ┌──────────────────────────────────────────────────┐
  │        逐层共享注意力 (Shared Attention)            │
  │                                                  │
  │  Layer 0: [VLM_tokens | Expert_tokens] → Attn   │
  │  Layer 1: [VLM_tokens | Expert_tokens] → Attn   │
  │  ...                                            │
  │  Layer N: [VLM_tokens | Expert_tokens] → Attn   │
  └──────────────────────────────────────────────────┘
           │                            │
           ▼                            ▼
  ┌─────────────────┐        ┌─────────────────────┐
  │  prefix_output  │        │   suffix_output      │
  │  (VLM隐藏状态)   │        │  (动作专家隐藏状态)    │
  └─────────────────┘        └─────────────────────┘
```

## 类定义与依赖

```python
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
```

- **PaliGemmaForConditionalGeneration**: HuggingFace 的 PaliGemma 实现，包含 SigLIP 视觉编码器 + Gemma 语言模型
- **GemmaForCausalLM**: HuggingFace 的 Gemma 语言模型，这里用作动作专家
- **CONFIG_MAPPING**: 用于动态创建模型配置
- **modeling_gemma**: 包含底层工具函数（RoPE、attention forward、gated residual 等）

---

## `__init__` 方法详解

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `vlm_config` | object | VLM 配置，包含 width, mlp_dim, num_heads, head_dim, depth, num_kv_heads |
| `action_expert_config` | object | 动作专家配置，字段同上 |
| `use_adarms` | list[bool] | 长度为2的列表，分别控制 VLM 和 Expert 是否使用 AdaRMS 归一化 |
| `precision` | "bfloat16" \| "float32" | 模型精度 |

### VLM 配置 (PaliGemma)

```python
vlm_config_hf = CONFIG_MAPPING["paligemma"]()
vlm_config_hf._vocab_size = 257152
vlm_config_hf.image_token_index = 257152
vlm_config_hf.text_config.hidden_size = vlm_config.width
vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
# ... 更多配置
```

**关键配置项：**
- `vocab_size = 257152`: Gemma tokenizer 的标准词表大小
- `image_token_index = 257152`: 图像占位符 token 的索引，等于词表大小（即最后一个位置之后）
- `hidden_activation = "gelu_pytorch_tanh"`: 使用近似 GELU 激活函数
- `use_adarms`: 是否启用自适应 RMS 归一化（Adaptive RMS Normalization）
- `adarms_cond_dim`: AdaRMS 条件维度，等于模型宽度（用于注入时间步等条件信息）

**视觉编码器配置：**
- `intermediate_size = 4304`: SigLIP ViT 的 FFN 中间维度
- `projection_dim = 2048`: 视觉特征投影到语言空间的维度
- `projector_hidden_act = "gelu_fast"`: 投影层激活函数

### Action Expert 配置 (Gemma)

```python
action_expert_config_hf = CONFIG_MAPPING["gemma"](
    head_dim=action_expert_config.head_dim,
    hidden_size=action_expert_config.width,
    # ...
)
```

动作专家是一个独立的 Gemma 模型，参数量通常比 VLM 小。关键区别：

```python
self.gemma_expert.model.embed_tokens = None
```

**删除了 embed_tokens 层**，因为动作专家不需要从离散 token 做 embedding，它的输入是由外部（`pi0_pytorch.py`）计算好的连续向量嵌入（流匹配的噪声动作 + 时间步编码）。

---

## `to_bfloat16_for_selected_params` 方法

### 混合精度策略

```python
def to_bfloat16_for_selected_params(self, precision):
```

**目的：** 在节省显存的同时保持数值稳定性。

**策略：**
1. 将整个模型转为 `bfloat16`（节省约50%显存）
2. 将以下关键参数保持为 `float32`：

| 参数 | 保持float32的原因 |
|------|-----------------|
| `patch_embedding.weight/bias` | 视觉输入的第一层卷积，低精度会导致图像特征提取不准确 |
| `position_embedding.weight` | 位置编码需要精确的数值表示 |
| `input_layernorm` | RMS Norm 涉及方差计算，对精度敏感 |
| `post_attention_layernorm` | 同上 |
| `model.norm` | 最终层归一化，影响输出稳定性 |

这种做法是大模型训练的标准实践——归一化层在低精度下容易产生数值溢出或下溢。

---

## `embed_image` 方法

```python
def embed_image(self, image: torch.Tensor):
    return self.paligemma.model.get_image_features(image)
```

**功能：** 将原始图像张量通过 SigLIP 视觉编码器转换为视觉特征序列。

**输入：** `image` — shape `[batch, channels, height, width]`
**输出：** 视觉 token 序列 — shape `[batch, num_patches, hidden_dim]`

内部流程：
1. Patch Embedding（将图像分割为 patch 并线性投影）
2. 加上位置编码
3. 通过 ViT Transformer 层
4. 线性投影到语言模型的隐藏维度

---

## `embed_language_tokens` 方法

```python
def embed_language_tokens(self, tokens: torch.Tensor):
    return self.paligemma.language_model.embed_tokens(tokens)
```

**功能：** 将离散语言 token ID 转换为连续嵌入向量。

**输入：** `tokens` — shape `[batch, seq_len]`，整数型
**输出：** 嵌入向量 — shape `[batch, seq_len, hidden_dim]`

---

## `forward` 方法详解

### 参数说明

| 参数 | 形状 | 说明 |
|------|------|------|
| `attention_mask` | `[batch, 1, total_seq, total_seq]` | 注意力掩码（causal mask + padding） |
| `position_ids` | `[batch, total_seq]` | 位置编码索引 |
| `past_key_values` | list of tensors | KV缓存，推理时使用 |
| `inputs_embeds` | `[prefix_embeds, suffix_embeds]` | 两个模型的输入嵌入 |
| `use_cache` | bool | 是否返回KV缓存 |
| `adarms_cond` | `[cond_vlm, cond_expert]` | AdaRMS 条件信号（如时间步编码） |

### 三种运行模式

#### 模式1：仅前缀 (Prefill) — `inputs_embeds[1] is None`

```python
if inputs_embeds[1] is None:
    prefix_output = self.paligemma.language_model.forward(
        inputs_embeds=inputs_embeds[0],
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        adarms_cond=adarms_cond[0],
    )
```

**使用场景：** 推理阶段的第一步——编码 observation（图像+语言指令）。

**流程：**
1. 仅运行 PaliGemma 语言模型处理前缀 token
2. 生成 KV Cache（`past_key_values`）供后续动作生成步骤复用
3. 返回 `prefix_output`（最终隐藏状态）和 KV Cache

**意义：** 在多步动作生成中，observation 只需编码一次，后续生成步骤通过 KV Cache 复用计算结果，避免重复计算。

#### 模式2：仅后缀 (Decode with Cache) — `inputs_embeds[0] is None`

```python
elif inputs_embeds[0] is None:
    suffix_output = self.gemma_expert.model.forward(
        inputs_embeds=inputs_embeds[1],
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        adarms_cond=adarms_cond[1],
    )
```

**使用场景：** 推理阶段的后续步骤——利用已缓存的 KV 值生成动作。

**流程：**
1. 仅运行 Gemma 动作专家
2. 通过 `past_key_values` 访问之前编码的 observation 信息
3. 输出动作隐藏状态

#### 模式3：前缀+后缀联合 (Joint Forward) — 训练模式

```python
else:
    # 两个模型逐层共享注意力
```

**使用场景：** 训练阶段，前缀和后缀同时存在，需要全序列的梯度流动。

这是最复杂的分支，下面详细展开。

---

### 模式3：联合前向传播的详细流程

#### 第一步：梯度检查点设置

```python
use_gradient_checkpointing = (
    hasattr(self.gemma_expert.model, "gradient_checkpointing")
    and self.gemma_expert.model.gradient_checkpointing
    and self.training
)
```

**Gradient Checkpointing 原理：**
- 正常训练时，每一层的中间激活都保存在显存中用于反向传播
- 开启 checkpointing 后，中间激活不保存，反向传播时重新前向计算
- 代价：计算量增加约 33%
- 收益：显存占用大幅降低（从 O(N) 降至 O(√N)，N 为层数）

对于 π₀ 这样的大模型（VLM + Expert 双塔），显存节省是必要的。

#### 第二步：逐层计算 (`compute_layer_complete`)

这是核心函数，对每一个 Transformer 层执行完整的前向计算：

##### 2.1 Input LayerNorm + QKV 投影

```python
for i, hidden_states in enumerate(inputs_embeds):  # i=0: VLM, i=1: Expert
    layer = models[i].layers[layer_idx]
    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
    gates.append(gate)

    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_state   = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

**要点：**
- 两个模型各自使用**自己的** LayerNorm 和 QKV 投影权重
- `input_layernorm` 是 AdaRMS Norm，接受条件参数 `cond`（如扩散时间步），返回归一化结果和门控值 `gate`
- `gate` 用于后续的门控残差连接
- QKV 张量形状变换：`[batch, seq, hidden] → [batch, num_heads, seq, head_dim]`

##### 2.2 序列拼接 — 实现跨模态注意力

```python
query_states = torch.cat(query_states, dim=2)  # dim=2 是序列维度
key_states   = torch.cat(key_states, dim=2)
value_states = torch.cat(value_states, dim=2)
```

**这是架构的核心设计：**

```
拼接前:                          拼接后:
VLM:    [batch, heads, S1, dim]   →  [batch, heads, S1+S2, dim]
Expert: [batch, heads, S2, dim]
```

拼接后的注意力矩阵（假设无mask）：
```
         VLM_tokens  Expert_tokens
VLM     [    ✓           ✓       ]   ← VLM token 可以看到所有 token
Expert  [    ✓           ✓       ]   ← Expert token 也可以看到所有 token
```

通过 `attention_mask` 可以控制可见性（如 causal mask 让 expert 只能看到前面的 VLM token）。

##### 2.3 旋转位置编码 (RoPE)

```python
cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
    query_states, key_states, cos, sin, unsqueeze_dim=1
)
```

RoPE 对 Q 和 K 施加旋转变换，编码相对位置信息。`position_ids` 涵盖了整个拼接序列的位置。

##### 2.4 注意力计算

```python
att_output, _ = modeling_gemma.eager_attention_forward(
    self.paligemma.language_model.layers[layer_idx].self_attn,
    query_states, key_states, value_states,
    attention_mask, scaling,
)
```

使用 eager（非 flash）attention 实现：
- `scaling = 1 / sqrt(head_dim)` — 标准缩放因子
- `attention_mask` 控制哪些 token 对可以互相 attend
- 输出形状：`[batch, total_seq, num_heads * head_dim]`

##### 2.5 输出分割与后处理

```python
att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

for i, hidden_states in enumerate(inputs_embeds):
    end_pos = start_pos + hidden_states.shape[1]
    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])  # 各自的输出投影
```

将拼接的注意力输出按位置拆分，分别送入各模型自己的输出投影层。

##### 2.6 门控残差连接 + FFN

```python
# 第一个残差连接（attention 输出 + 原始输入）
out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])

after_first_residual = out_emb.clone()

# Post-attention LayerNorm
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])

# FFN (MLP)
out_emb = layer.mlp(out_emb)

# 第二个残差连接（MLP 输出 + attention 残差输出）
out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
```

**门控残差连接 (`_gated_residual`)：**

标准残差：`output = input + sublayer(input)`

门控残差：`output = input + gate * sublayer(input)`

其中 `gate` 来自 AdaRMS Norm，由条件信号（时间步）控制。这允许模型在不同扩散时间步动态调整残差贡献的强度——在噪声较大的早期步骤中可能减弱残差，在接近收敛时增强。

##### 2.7 精度转换

```python
if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
    out_emb = out_emb.to(dtype=torch.bfloat16)
```

由于 LayerNorm 保持 float32 而 MLP 权重为 bfloat16，需要在进入 MLP 前显式转换精度，避免 dtype 不匹配错误。

#### 第三步：最终归一化

```python
def compute_final_norms(inputs_embeds, adarms_cond):
    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
        outputs_embeds.append(out_emb)
    return outputs_embeds
```

所有层处理完成后，对最终隐藏状态做一次归一化（Gemma 标准做法），各模型使用自己的 norm 层。

---

## AdaRMS 归一化机制

AdaRMS (Adaptive RMS Normalization) 是 π₀ 的关键创新之一：

```
标准 RMS Norm:  output = x / RMS(x) * gamma

AdaRMS Norm:    scale, gate = MLP(condition)
                output = x / RMS(x) * (gamma * (1 + scale))
                return output, gate
```

`condition` 通常是扩散模型的时间步编码（noised timestep），这使得：
- **归一化的缩放因子随时间步变化** — 模型在不同去噪阶段采用不同的特征缩放
- **门控值控制残差强度** — 模型可以学习在某些时间步"跳过"某些层

这是将扩散模型（Flow Matching）与 Transformer 结合的优雅方式。

---

## 返回值

```python
return [prefix_output, suffix_output], prefix_past_key_values
```

| 返回值 | 形状 | 用途 |
|--------|------|------|
| `prefix_output` | `[batch, prefix_seq, hidden_dim]` | VLM 最终隐藏状态，可用于语言建模损失 |
| `suffix_output` | `[batch, suffix_seq, hidden_dim]` | 动作专家最终隐藏状态，送入 action head 解码为连续动作 |
| `prefix_past_key_values` | list of KV tensors | 仅模式1有值，用于推理时的 KV Cache |

---

## 运行模式与数据流总结

### 训练时 (模式3)

```
图像 → embed_image() → image_embeds ─┐
                                      ├─ concat → prefix_embeds (inputs_embeds[0])
语言tokens → embed_language_tokens() ─┘

噪声动作 + 时间步编码 → suffix_embeds (inputs_embeds[1])

[prefix_embeds, suffix_embeds] → forward() → [prefix_output, suffix_output]
                                                                │
                                          suffix_output → action_head → predicted_velocity
                                                                │
                                          loss = MSE(predicted_velocity, target_velocity)
```

### 推理时 (模式1 → 模式2 循环)

```
Step 1 (Prefill, 模式1):
  observation → prefix_embeds → forward([prefix_embeds, None])
                               → prefix_output, kv_cache

Step 2..N (Decode, 模式2, 迭代去噪):
  noised_action(t) → suffix_embeds → forward([None, suffix_embeds], past_kv=kv_cache)
                                    → suffix_output → action_head → velocity
  action(t-1) = action(t) - velocity * dt   (Flow Matching ODE step)
```

---

## 关键设计决策与权衡

| 设计 | 选择 | 原因 |
|------|------|------|
| 注意力共享方式 | 序列拼接 | 比 cross-attention 更简洁，允许双向信息流 |
| Expert 无 embed_tokens | 删除 | 动作是连续值，不走离散 tokenization |
| 梯度检查点 | 训练时强制开启 | 双塔模型显存压力大 |
| 混合精度 | 主体bf16 + norm层fp32 | 平衡速度与稳定性 |
| eager attention | 非 Flash Attention | 需要自定义mask支持跨模态注意力模式 |
| 门控残差 | AdaRMS 提供 gate | 条件生成模型需要时间步自适应 |

---

## 与 π₀ 论文的对应关系

这个实现对应 π₀ 论文中的以下概念：

1. **VLM Backbone** → `self.paligemma` (PaliGemma = SigLIP + Gemma)
2. **Action Expert** → `self.gemma_expert` (smaller Gemma model)
3. **Flow Matching** → 由 `pi0_pytorch.py` 实现，本模块提供网络前向传播
4. **Action Chunking** → suffix 的序列长度对应 action chunk size
5. **Cross-modal Attention** → 模式3中序列拼接实现的联合注意力
