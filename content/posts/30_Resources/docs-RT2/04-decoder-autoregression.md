---
title: "4. 语言解码器与自回归生成"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "Transformer、Decoder、AutoregressiveWrapper 的原理、API 与自回归生成策略。"
summary: "RT-2 语言解码器与自回归动作生成详解。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 5
series_weight: 5
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 4. 语言解码器与自回归生成

本章详解 RT-2 的语言/动作解码路径：`Transformer`、`Decoder`、`AutoregressiveWrapper` 的原理、API 与生成策略。

---

## 4.1 为什么需要解码器

视觉编码器输出图像 Token 序列 $\mathbf{H}_{img} \in \mathbb{R}^{N \times d}$，任务要求自回归预测下一个 Token（语言或动作）：

$$
P(y_t \mid y_{<t}, \mathbf{I}) = \text{softmax}(\text{Decoder}(y_{<t}, \mathbf{H}_{img}))
$$

解码器必须：
1. **因果性**：位置 $t$ 只能看到 $y_{<t}$，不能偷看未来
2. **多模态融合**：通过 Cross-Attention 访问完整图像上下文
3. **大词表输出**：对每个位置输出 `num_tokens` 维 logits

---

## 4.2 Transformer 类

**来源**：`zeta.structs.transformer.Transformer`

### 4.2.1 `__init__` 关键参数

| 参数 | RT2 默认值 | 说明 |
|------|------------|------|
| `num_tokens` | 20000 | 词表大小 |
| `max_seq_len` | 1024 | 最大序列长度 |
| `attn_layers` | `Decoder(...)` | 必须是 AttentionLayers 子类 |
| `use_abs_pos_emb` | False | 是否使用绝对位置编码 |
| `emb_dim` | None (=dim) | 嵌入维度 |
| `tie_embedding` | False | 是否 tying 输入嵌入与输出投影 |

**内部子模块**：

| 子模块 | 作用 |
|--------|------|
| `token_emb` | `TokenEmbedding(emb_dim, num_tokens)` |
| `pos_emb` | 绝对/正弦位置编码，或 `always(0)` |
| `project_emb` | 嵌入维度 → 注意力维度 |
| `attn_layers` | `Decoder` 层堆叠 |
| `to_logits` | `Linear(dim, num_tokens)` 输出 logits |

### 4.2.2 `forward` 完整签名

```python
def forward(
    self,
    x,                          # (B, seq_len) Token IDs
    return_embeddings=False,
    return_logits_and_embeddings=False,
    return_intermediates=False,
    mask=None,
    return_mems=False,
    return_attn=False,
    mems=None,
    pos=None,
    prepend_embeds=None,        # PaLI 风格：前置图像嵌入
    sum_embeds=None,
    return_attn_z_loss=False,
    attn_z_loss_weight=1e-4,
    **kwargs,                   # context=encoded 通过 kwargs 传入 Decoder
):
```

#### RT2 使用的调用路径

```python
# AutoregressiveWrapper 内部
logits = self.net(inp, context=encoded)
```

等价于：

```python
x = token_emb(inp) + pos_emb(inp)
x = project_emb(x)
x = attn_layers(x, context=encoded)  # Cross-Attn 层使用 encoded
logits = to_logits(x)
```

#### `prepend_embeds` vs `context`（Cross-Attention）

| 方式 | 机制 | 使用方 |
|------|------|--------|
| `prepend_embeds` | 图像 Token 与文本 Token **拼接** | PaLI, PaLM-E |
| `context=` (Cross-Attn) | 图像作为 K/V，文本作为 Q | 本仓库 RT2 |

本仓库选择 **Cross-Attention**，图像与文本序列长度解耦，适合固定 Patch 数的 ViT 输出。

---

## 4.3 Decoder 类

```python
class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)
```

### 4.3.1 RT2 中的 Decoder 配置

```python
Decoder(
    dim=512,
    depth=6,
    heads=8,
    cross_attend=True,      # 启用 cross-attention 层
    attn_kv_heads=2,        # Grouped Query Attention (GQA)
    attn_flash=True,        # Flash Attention
    qk_norm=True,           # Query/Key L2 归一化
)
```

### 4.3.2 层结构（cross_attend=True）

默认 `layer_types = ("a", "c", "f") * depth` 或类似交替模式：

| 层类型 | 符号 | 作用 |
|--------|------|------|
| Self-Attention | `a` | 因果自注意力，建模 Token 序列依赖 |
| Cross-Attention | `c` | Query=文本，Key/Value=图像嵌入 |
| FeedForward | `f` | 逐 Token 非线性变换 |

### 4.3.3 因果 Self-Attention

因果掩码 $M$：

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

$$
\text{Attn} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
$$

### 4.3.4 Cross-Attention

$$
Q = X W_Q \quad (\text{来自解码器状态})
$$
$$
K = \mathbf{H}_{img} W_K, \quad V = \mathbf{H}_{img} W_V
$$

$$
\text{CrossAttn}(X, \mathbf{H}_{img}) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

**无因果掩码**：每个文本位置可 attend 到**所有**图像 Patch。

### 4.3.5 Grouped Query Attention (GQA)

当 `attn_kv_heads=2 < heads=8` 时，K/V 头数少于 Q 头数，通过 repeat 共享：

$$
\text{KV heads} = 2, \quad \text{Q heads} = 8, \quad \text{repeat factor} = 4
$$

**优势**：减少 KV Cache 显存，加速推理（[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)）。

### 4.3.6 QK-Norm

对 Q、K 做 L2 归一化后再缩放：

$$
Q' = \text{l2norm}(Q) \cdot s_q, \quad K' = \text{l2norm}(K) \cdot s_k
$$

提升深层 Transformer 训练稳定性。

### 4.3.7 Flash Attention

使用 PyTorch `scaled_dot_product_attention` 融合算子，$O(N)$ 显存复杂度替代 materialized $N \times N$ 注意力矩阵（[Dao et al., 2022](https://arxiv.org/abs/2205.14135)）。

---

## 4.4 AutoregressiveWrapper 类

**来源**：`zeta.structs.auto_regressive_wrapper.AutoRegressiveWrapper`

### 4.4.1 `__init__`

```python
def __init__(
    self,
    net: nn.Module,           # Transformer 实例
    ignore_index: int = -100,
    pad_value: int = 0,
    mask_prob: float = 0.0,   # MLM 风格 mask 概率
    speculative: bool = False,
):
```

| 属性 | 说明 |
|------|------|
| `self.net` | 被包装的 Transformer |
| `self.max_seq_len` | 继承自 `net.max_seq_len` |
| `self.pad_value` | 生成时 padding 值 |
| `self.ignore_index` | 损失计算忽略的标签 |
| `self.mask_prob` | >0 时在训练中随机 mask 输入 token |

### 4.4.2 `forward`

```python
def forward(self, x, return_loss=True, **kwargs):
```

#### Teacher Forcing 算法

```
输入 x: (B, seq_len)  完整 token 序列

1. inp   = x[:, :-1]   # 输入：前 seq_len-1 个 token
2. target = x[:, 1:]   # 目标：后 seq_len-1 个 token（移位一位）

3. logits = net(inp, **kwargs)   # (B, seq_len-1, vocab)

4. loss = CrossEntropy(logits, target, ignore_index)

5. 返回 (logits, loss) 或 logits
```

#### 损失函数

$$
\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_{\leq t}, \mathbf{H}_{img})
$$

### 4.4.3 `generate`

```python
@torch.no_grad()
def generate(
    self,
    start_tokens,        # 起始 prompt
    seq_len: int,        # 要生成的 token 数
    eos_token=None,
    strategy="temperature",
    temperature=1.0,
    filter_logits_fn=top_k,
    filter_thres=0.9,
    min_p_pow=2.0,
    min_p_ratio=0.02,
    gamma=5,
    **kwargs,            # context=encoded 传入 net
):
```

#### 自回归采样循环

```
out = start_tokens
for _ in range(seq_len):
    x = out[:, -max_seq_len:]
    logits = net(x, **kwargs)[:, -1]      # 取最后位置 logits
    filtered = top_k(logits, thres=0.9)     # 可选过滤
    probs = softmax(filtered / temperature)
    sample = multinomial(probs, 1)
    out = cat(out, sample)
    if eos_token: break
return out[:, len(start_tokens):]
```

#### 采样策略

| 策略 | 函数 | 公式/说明 |
|------|------|-----------|
| Temperature | `temperature` | $P \propto \exp(z/T)$，$T$ 越小越贪婪 |
| Top-K | `top_k` | 只保留概率最高的 K 个 token |
| Top-P (Nucleus) | `top_p` | 保留累积概率 ≥ p 的最小集合 |
| Top-A | `top_a` | 自适应截断 |

### 4.4.4 其他方法

| 方法 | 说明 |
|------|------|
| `generate_n_solutions` | 生成 n 个候选序列 |
| `evaluate_and_select_best_solution` | 用 reward 函数选最优 |
| `classifier_free_guidance` (模块级) | $\text{logits} = \text{uncond} + \alpha(\text{cond} - \text{uncond})$ |

---

## 4.5 RT-2 推理完整流程

```python
import torch
from rt2 import RT2

model = RT2()
model.eval()

img = torch.randn(1, 3, 256, 256)
prompt = torch.tensor([[101, 2023, 102]])  # 示例 token ids

with torch.no_grad():
    encoded = model.encoder(img, return_embeddings=True)
    generated = model.decoder.generate(
        prompt,
        seq_len=8,           # RT-2 动作 8 维 → 8 tokens
        context=encoded,
        temperature=0.7,
        filter_logits_fn=top_k,
        filter_thres=0.9,
    )

print(generated.shape)  # (1, 8)
```

**论文 Output Constraint**：推理机器人任务时，应将 `filter_logits_fn` 替换为仅在 256 个动作 Token 上采样（见 [05-action-tokenization.md](./05-action-tokenization.md)）。

---

## 4.6 数据流示意图

```
text tokens (B, T)
        │
        ▼
┌─────────────────┐
│ Token Embedding │
│ + Position Emb  │
└─────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Decoder Block × 6                       │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ Causal      │  │ Cross-Attention  │  │
│  │ Self-Attn   │→ │ Q: text          │  │
│  │             │  │ K,V: image (B,64,512)│
│  └─────────────┘  └──────────────────┘  │
│         ↓                                │
│  ┌─────────────┐                        │
│  │ FeedForward │                        │
│  └─────────────┘                        │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│ Linear → Logits │  (B, T, 20000)
└─────────────────┘
```

---

## 4.7 与论文 PaLM-E / PaLI-X 解码对比

| 特性 | PaLM-E (Dec-Only) | PaLI-X (Enc-Dec) | 本仓库 |
|------|-------------------|------------------|--------|
| 架构 | Decoder-Only | Encoder-Decoder | Decoder + Cross-Attn |
| 图像融合 | Token 拼接 | Enc-Dec Cross-Attn | Cross-Attn |
| CoT | 支持 Plan+Action | 有限 | 需自行扩展 prompt |
| 词表 | SentencePiece | SentencePiece | 自定义 20000 |

---

## 4.8 参考文献

| 文献 | 链接 |
|------|------|
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 |
| FlashAttention | https://arxiv.org/abs/2205.14135 |
| GQA | https://arxiv.org/abs/2305.13245 |
| Autoregressive Wrapper 源码 | https://github.com/kyegomez/zeta/blob/master/zeta/structs/auto_regressive_wrapper.py |
| Transformer 源码 | https://github.com/kyegomez/zeta/blob/master/zeta/structs/transformer.py |

---

## 4.9 相关章节

- 动作 Token 如何映射到 logits → [05-action-tokenization.md](./05-action-tokenization.md)
- RT2 类如何组装 → [02-implementation.md](./02-implementation.md)
