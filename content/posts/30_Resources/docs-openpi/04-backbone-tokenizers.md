---
title: "04 骨干网络与分词器"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "解读 Gemma 混合专家、SigLIP/ViT、LoRA 与四类 Tokenizer 等 openpi 模型底层构件。"
summary: "openpi 骨干网络与分词器模块详解。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 4
series_weight: 4
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 04 骨干网络与分词器

> 本章解读模型的"零件库"：`gemma.py`（混合专家 Transformer）、`gemma_fast.py`（FAST 用、带 KV 缓存的 Gemma）、`siglip.py`（视觉塔/ViT）、`lora.py`（低秩适配）、`tokenizer.py`（四种分词器）。这些是 [02](02-models-flow-matching.md)/[03](03-models-pi0-fast.md) 章模型的底层构件。

---

## 4.1 Gemma 混合专家 Transformer（gemma.py）

`models/gemma.py` 改编自 Google big_vision，是 π₀/π₀.₅ 的核心骨干。它最特别的地方是支持**一个序列里跑多个"专家"**（不同 token 段用不同权重），从而实现 PaliGemma 主干 + 动作专家的混合。

### 4.1.1 配置与变体 `Config` / `get_config`

```python
@dataclasses.dataclass
class Config:
    width: int; depth: int; mlp_dim: int
    num_heads: int; num_kv_heads: int; head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = {}
```

`get_config(variant)` 返回预设变体：

| 变体 | width | depth | mlp_dim | heads/kv | head_dim | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `dummy` | 64 | 4 | 128 | 8/1 | 16 | 调试 |
| `gemma_300m` | 1024 | 18 | 4096 | 8/1 | 256 | 动作专家（311M） |
| `gemma_2b` | 2048 | 18 | 16384 | 8/1 | 256 | PaliGemma 主干 |
| `gemma_2b_lora` | 同上 | | | | | + LoRA(rank16) |
| `gemma_300m_lora` | 同上 | | | | | + LoRA(rank32) |

注意所有专家共享 `depth`、`head_dim`、`num_heads`、`num_kv_heads`（自注意力要求一致，见 §4.1.4）。`num_kv_heads=1` 表示用了**多查询注意力（MQA）**——8 个查询头共享 1 个 KV 头，省显存。

### 4.1.2 `RMSNorm`：普通 + 自适应（AdaRMS）

```python
class RMSNorm(nn.Module):
    def __call__(self, x, cond):
        var = mean(square(x.astype(float32)), -1, keepdims=True)   # 方差用 float32 算
        normed = x * reciprocal(sqrt(var + 1e-6))
        if cond is None:
            scale = self.param("scale", zeros_init, (x.shape[-1]))
            return (normed * (1 + scale)).astype(dtype), None       # 普通 RMSNorm
        # 自适应 RMSNorm：用条件向量产出 scale/shift/gate
        modulation = nn.Dense(x.shape[-1]*3, kernel_init=zeros)(cond)
        scale, shift, gate = split(modulation[:, None, :], 3, axis=-1)
        return (normed * (1 + scale) + shift).astype(dtype), gate
```

**为什么需要 AdaRMS**：π₀.₅ 要把"时间步"这个条件信息注入网络。AdaRMS 让条件向量（时间嵌入）动态生成归一化的缩放（scale）、平移（shift）和门控（gate），即 [FiLM 式调制](https://arxiv.org/abs/1709.07871)。`cond=None` 时退化为标准 RMSNorm（π₀ 及 PaliGemma 主干用）。注意方差始终在 float32 下计算以保证数值稳定。

### 4.1.3 `Embedder`：词嵌入

```python
class Embedder(nn.Module):
    def encode(self, x):  # token → 向量，并乘 sqrt(embed_dim) 缩放
        x = self.input_embedding_table[(x,)]; return x * sqrt(embed_dim)
    def decode(self, x):  # 向量 → 词表 logits（与嵌入表共享权重）
        return jnp.dot(x, self.input_embedding_table.T)
```

### 4.1.4 `Attention`：多专家共享注意力

这是混合专家的核心。多个专家的 token 各自算 QKV（用各自权重），然后**拼接到一起算注意力**，最后再切回各专家：

```python
class Attention(nn.Module):
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # 断言所有专家 head_dim / num_heads / num_kv_heads 一致
        for x, config in zip(xs, self.configs):
            if x is None: continue
            # 每个专家用自己的 einsum 权重算 QKV（MQA：q 多头，kv 单头）
            ...
        q, k, v = (concat(y, axis=1) for y in zip(*qkvs))   # 沿序列维拼接各专家
        q = _apply_rope(q, positions); q *= head_dim ** -0.5
        k = _apply_rope(k, positions)
        if kv_cache is not None:                            # 推理：拼接历史 KV
            cache_k, cache_v = kv_cache
            k = concat([cache_k, k], axis=1); v = concat([cache_v, v], axis=1)
        # MQA 注意力
        logits = einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=float32)
        masked = where(attn_mask, logits, -2.38e38)         # 大负数屏蔽
        probs = softmax(masked, -1).astype(dtype)
        encoded = einsum("BKGTS,BSKH->BTKGH", probs, v)
        # 切回各专家，用各自的输出投影
        out = []; start = 0
        for x, config in zip(xs, self.configs):
            if x is not None:
                out.append(out_einsum(encoded[:, start:start+x.shape[1]])); start += x.shape[1]
            else: out.append(None)
        return out, (k, v)
```

**关键点**：
- `xs` 是专家 token 列表（如 `[prefix, suffix]`），某个为 `None` 表示该专家本步不参与（推理时前缀已缓存）。
- QKV 各专家独立投影，但**拼接后统一做注意力**——这让动作 token 能注意到图文 token（跨专家信息流动）。
- 返回 `(k, v)` 供 KV 缓存。
- 屏蔽用大负数 `-2.3819763e38`（对齐 gemma 官方实现，而非 dtype 最小值）。

### 4.1.5 `FeedForward`：GeGLU

```python
class FeedForward(nn.Module):
    def __call__(self, x):
        ff_gate = dot(x, w_gating[0]); gate = nn.gelu(ff_gate)   # 门控分支
        ff1 = dot(x, w_gating[1])
        return dot(gate * ff1, w_linear)                         # GeGLU
```

GeGLU（Gated GELU）：两路线性变换，一路过 GELU 当门控，逐元素相乘后再投影回去。比普通 MLP 表达力更强。

### 4.1.6 `Block` 与门控残差

```python
class Block(nn.Module):
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):
        # 1) 注意力前归一化（可能 AdaRMS，返回 gate）
        pre_attn, gates = [], []
        for i, x in enumerate(xs):
            x, gate = RMSNorm(...)(x, adarms_cond[i]); pre_attn.append(x); gates.append(gate)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates)]  # 门控残差
        # 2) FFN 前归一化 + GeGLU + 门控残差
        out, gates = [], []
        for i, (x, config) in enumerate(zip(xs, self.configs)):
            x, gate = RMSNorm(...)(x, adarms_cond[i]); x = lora.FeedForward(...)(x)
            out.append(x); gates.append(gate)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates)]
        return xs, kv_cache
```

`_gated_residual(x, y, gate)`：当 `gate is None` 时是普通残差 `x + y`；否则 `x + y * gate`（AdaRMS 的门控）。这让 π₀.₅ 能通过时间条件动态调节每个子层的残差强度。

### 4.1.7 `Module`：组装与扫描

`Module` 把所有 Block 用 `nn.scan` + `nn.remat` 堆叠：
- `nn.scan`：把 depth 层折叠成一次扫描，参数沿第 0 轴堆叠——大幅减少编译时间与代码膨胀。
- `nn.remat`（梯度检查点）：用 `nothing_saveable` 策略，反向时重算激活省显存。
- `__call__` 接受专家 token 列表，逐层处理后对每个专家过 `final_norm`。
- `embed(tokens)`：词嵌入入口（`method="embed"`）。
- `init(use_adarms)`：linen 的初始化便捷方法。

`_apply_rope`：旋转位置编码（RoPE），在 float32 下计算 sin/cos 后施加到 Q/K。`_name(name, i)`：第一个专家无后缀（可加载 PaliGemma 权重），后续加 `_1`。

---

## 4.2 FAST 版 Gemma（gemma_fast.py）

`models/gemma_fast.py` 是给 π₀-FAST 用的 Gemma 变体。与 `gemma.py` 的核心区别：**单专家 + 真正的增量 KV 缓存**（支持逐 token 解码）。

主要差异点：
- 只用 `ml_collections.ConfigDict` 配置（`gemma_2b` / `gemma_2b_lora`），无多专家。
- `Attention` 内含 `_init_cache` / `_update_cache`：
  - `_init_cache(k, v, cache_size)`：把 K/V padding 到 `cache_size`，返回写入位置索引 `idx`。
  - `_update_cache(k, v, idx, k_cache, v_cache)`：用 `jax.lax.dynamic_update_slice` 在 `idx` 处写入新的单步 K/V（只支持长度 1 的更新——典型自回归解码）。
- `Module.__call__` 支持多种模式：`embed_only`（只嵌入）、`pre_logits`（从预 logits 解码）、`return_prelogits`（返回预 logits）、`decode`（用 KV 缓存）。这些模式恰好支撑了 [03 章](03-models-pi0-fast.md) 里 `compute_loss`（拆分前向/解码省显存）与 `sample_actions`（prefill + 逐步解码）的调用方式。
- `RMSNorm` 只有普通版（无 AdaRMS，FAST 不需要时间条件）。

| 能力 | gemma.py | gemma_fast.py |
| --- | --- | --- |
| 多专家 | ✅ | ❌（单专家） |
| AdaRMS | ✅ | ❌ |
| 增量 KV 缓存 | 简单拼接 | ✅ 带索引的 dynamic_update_slice |
| 用于 | π₀ / π₀.₅ | π₀-FAST |

---

## 4.3 SigLIP 视觉塔（siglip.py）

`models/siglip.py` 是 ViT 实现（改编自 big_vision），把图像编码成 token 序列。π₀ 系列用 `So400m/14` 变体（width 1152, depth 27, patch 14×14）。

### 组件
- `posemb_sincos_2d(h, w, width)`：2D sin-cos 位置编码（MoCo v3 风格）。
- `get_posemb(...)`：选择位置编码类型（`learn` 可学习 / `sincos2d`）。
- `MlpBlock`：标准 ViT 前馈块（Dense → GELU → Dropout → Dense）。
- `Encoder1DBlock`：单个 Transformer 编码块（多头自注意力 + MLP + 残差 + LayerNorm，pre-norm 结构）。
- `Encoder`：堆叠 depth 个 `Encoder1DBlock`，支持 `scan`（折叠层）与 `remat`（检查点），末尾接 LayerNorm。
- `MAPHead`：多头注意力池化（Multihead Attention Pooling），用一个可学习 probe 向量聚合序列。
- `_Module`：完整 ViT——patch 提取（Conv）→ 加位置编码 →（可选 cls token）→ Encoder → 池化（`gap`/`map`/`tok`/`none`）。π₀ 用 `pool_type="none"`，即**保留所有 patch token**作为图像 token 喂给 Gemma（而非池化成单一向量）。
- `Module(num_classes, *, variant, **kw)`：工厂函数。
- `decode_variant(variant)`：把 `"So400m/14"` 这类字符串解码成 width/depth/mlp_dim/num_heads/patch_size 参数字典（内置一张参数表）。

> **精度处理细节**：patch 提取与位置编码刻意在 float32 下做（注释里 "Kevin edit: 感觉更安全"），之后才转回 `dtype_mm`（可能半精度）。这是数值稳定性的工程考量。

```
图像 [B,224,224,3]
  │ Conv(patch=14, stride=14)        # patch 提取
  ▼ [B,16,16,1152] → reshape [B,256,1152]
  │ + 2D 位置编码
  ▼
  Encoder × 27（自注意力 + MLP）
  ▼
  pool_type="none" → 输出 256 个图像 token [B,256,1152]
```

---

## 4.4 LoRA 低秩适配（lora.py）

`models/lora.py` 实现 LoRA（Low-Rank Adaptation），用于低显存微调：冻结原权重 $W$，只训练低秩增量 $\Delta W = BA$。

### `LoRAConfig`
```python
@struct.dataclass
class LoRAConfig:
    rank: int                          # 低秩维度
    alpha: float = 1.0                 # 缩放系数
    init_fn = nn.initializers.normal(stddev=0.01)
    rslora: bool = False               # rank-stabilized LoRA
    axes: tuple = (-2, -1)             # 在权重的哪两个轴上加 LoRA
    label: str = "L"                   # einsum 中的低秩轴标签

    @property
    def scaling_value(self):           # alpha/sqrt(rank) 或 alpha/rank
        return alpha/sqrt(rank) if rslora else alpha/rank
```

### `Einsum`（支持 LoRA 的 einsum）
作为 Gemma `Einsum` 的替代：在原 einsum 结果上加 `lora_a · lora_b · scaling`。核心是 `_make_lora_eqns` 把原 einsum 方程 `"BSD,3KDH->3BSKH"` 拆成两步低秩方程（插入低秩轴 `L`）：

$$y = Wx + \frac{\alpha}{r}\,(B(Ax))$$

### `FeedForward`（支持 LoRA 的 GeGLU）
对 gating 和 linear 两个权重分别挂上 LoRA 的 a/b 矩阵，`_dot` 方法在基础矩阵乘法上叠加低秩项。

> 配合 [02 章](02-models-flow-matching.md) 的 `get_freeze_filter`：LoRA 微调时主干权重冻结（且转 bf16），只有 LoRA 参数可训练，大幅降低显存（README：LoRA 微调仅需 >22.5GB，全参需 >70GB）。

---

## 4.5 分词器（tokenizer.py）

`models/tokenizer.py` 提供四种分词器。π₀ 系列只用前两种，后两种用于 RoboArena 基线。

### 4.5.1 `PaligemmaTokenizer`（π₀ / π₀.₅ 用）

基于 SentencePiece（从 `gs://big_vision/paligemma_tokenizer.model` 下载）。`tokenize(prompt, state=None)`：

```python
def tokenize(self, prompt, state=None):
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    if state is not None:   # π0.5 格式：状态进离散语言
        discretized = digitize(state, bins=linspace(-1,1,257)[:-1]) - 1   # 状态离散成 256 桶
        state_str = " ".join(map(str, discretized))
        full = f"Task: {cleaned}, State: {state_str};\nAction: "
        tokens = encode(full, add_bos=True)
    else:                   # π0 格式：状态走连续动作专家
        tokens = encode(cleaned, add_bos=True) + encode("\n")  # "\n" 作为"答案起始"
    # padding / 截断到 max_len，返回 tokens + mask
```

**两种格式的差异**正对应 π₀ vs π₀.₅：π₀.₅ 把状态离散化后拼进语言提示（`discrete_state_input`），π₀ 则只编码文本（状态走后缀的 `state_proj`）。

### 4.5.2 `FASTTokenizer`（π₀-FAST 用）

用 HuggingFace 的 `physical-intelligence/fast` 处理器把连续动作编码成离散 token。

- `tokenize(prompt, state, actions)` → `(tokens, token_mask, ar_mask, loss_mask)`：
  - 前缀：`Task: {prompt}, State: {离散状态};\n`（`add_bos`）。
  - 后缀：`Action: ` + FAST 动作 token + `|`（`add_eos`）。
  - `ar_mask`：前缀 0（双向）、后缀 1（因果）；`loss_mask`：只在后缀为 True。
  - 动作 token 经 `_act_tokens_to_paligemma_tokens` 映射到 PaliGemma 词表末尾（`vocab_size - 1 - 128 - token`，跳过最后 128 个特殊 token）。
- `extract_actions(tokens, action_horizon, action_dim)`：推理时反向——解码 token 串，定位 `Action: ... |` 之间的内容，映射回 FAST token，再用 FAST 处理器 `decode` 还原连续动作。
- `_act_tokens_to_paligemma_tokens`：动作 token ↔ PaliGemma 词表的双向映射。

这正是 [03 章](03-models-pi0-fast.md) 里序列结构的来源，也是 `TokenizeFASTInputs` / `ExtractFASTActions` 变换的底层。

### 4.5.3 `BinningTokenizer` 与 `FSQTokenizer`（RoboArena 基线，非 π₀ 用）

- `BinningTokenizer`：RT-2 / OpenVLA 风格的均匀分桶分词器（256 桶）。只支持推理时 `extract_actions`（不支持编码动作）。
- `FSQTokenizer`：FAST 论文里的 FSQ（Finite Scalar Quantization）分词器，从 Orbax 检查点加载一个 `FsqAttentionTokenizer`，用 JAX jit 的 tokenize/detokenize。

两者结构与 `FASTTokenizer` 类似（都拼 `Task:...State:...;` 前缀），但动作编/解码方式不同。它们标注为"用于 RoboArena 基线实现，不用于 π₀ 系列"。

---

## 4.6 小结

- `gemma.py`：多专家 Transformer，支持 AdaRMS 与跨专家共享注意力，是 π₀/π₀.₅ 的骨架。
- `gemma_fast.py`：单专家 + 增量 KV 缓存，支撑 π₀-FAST 的自回归解码。
- `siglip.py`：ViT 视觉塔，`pool_type="none"` 输出全部 patch token。
- `lora.py`：低秩适配，配合冻结过滤器做低显存微调。
- `tokenizer.py`：Paligemma（π₀ 文本/π₀.₅ 文本+状态）、FAST（动作 token）、Binning/FSQ（基线）。

下一章 [05 数据管线](05-data-pipeline.md) 讲这些 token/图像/状态是怎么从原始数据集一步步变换出来的。
