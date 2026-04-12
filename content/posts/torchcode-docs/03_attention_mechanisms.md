---
title: "第三章：注意力机制（TorchCode）"
date: 2026-04-01T10:00:00+08:00
draft: false
authors: [Steven]
description: "SDPA、MHA、因果/交叉注意力、GQA、滑动窗口、线性注意力、KV Cache、RoPE、Flash Attention 等。"
summary: "TorchCode 文档第三章：注意力机制从基础到高效实现。"

tags: [PyTorch, TorchCode]
categories: [PyTorch]
series: [TorchCode 系列]
weight: 4
series_weight: 4
hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

# 第三章：注意力机制

注意力机制是 Transformer 的核心。本章从基础的 Scaled Dot-Product Attention 出发，逐步展开到所有主流变体。

---

## 3.1 Scaled Dot-Product Attention（缩放点积注意力）

### 是什么
注意力机制的核心计算单元。给定查询（Q）、键（K）、值（V），计算每个查询对所有键的相关性权重，然后对值进行加权求和。

### 数学定义

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 步骤分解
1. 计算相似度矩阵：$S = QK^T$，形状 `(B, seq_q, seq_k)`
2. 缩放：$S = S / \sqrt{d_k}$，防止点积值过大导致 softmax 饱和
3. 归一化：对每行做 softmax，得到注意力权重
4. 加权求和：$\text{output} = \text{weights} \times V$

### 为什么要缩放
当 $d_k$ 较大时，$QK^T$ 的方差约为 $d_k$。除以 $\sqrt{d_k}$ 使方差回到 1，避免 softmax 输入过大导致梯度消失。

### 代码示例

```python
import torch
import math

def scaled_dot_product_attention(Q, K, V):
    # Q: (B, seq_q, d_k), K: (B, seq_k, d_k), V: (B, seq_k, d_v)
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V  # (B, seq_q, d_v)

# 自注意力
Q = K = V = torch.randn(2, 6, 64)
out = scaled_dot_product_attention(Q, K, V)
print(out.shape)  # (2, 6, 64)

# 交叉注意力（seq_q != seq_k）
Q2 = torch.randn(1, 3, 16)
K2 = torch.randn(1, 10, 16)
V2 = torch.randn(1, 10, 32)
out2 = scaled_dot_product_attention(Q2, K2, V2)
print(out2.shape)  # (1, 3, 32)
```

### 适用场景
- 所有 Transformer 变体的基础
- 自注意力（Q=K=V 来自同一输入）和交叉注意力（Q 和 K/V 来自不同输入）

---

## 3.2 Multi-Head Attention（多头注意力）

### 是什么
将输入投影到多个子空间（"头"），在每个子空间独立计算注意力，最后拼接并投影回原始维度。

### 数学定义

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 为什么需要多头
- 单头注意力只能学习一种注意力模式
- 多头允许模型同时关注不同位置的不同特征（如语法关系、语义关系、位置关系）
- 每个头的维度 $d_k = d_{model} / h$，总计算量与单头相同

### 结构说明
- `W_q, W_k, W_v`：投影矩阵，形状 `(d_model, d_model)`
- `W_o`：输出投影，形状 `(d_model, d_model)`
- `d_k = d_model // num_heads`：每个头的维度

### 代码示例

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]

        # 1. 线性投影
        q = self.W_q(Q)  # (B, S_q, d_model)
        k = self.W_k(K)  # (B, S_k, d_model)
        v = self.W_v(V)  # (B, S_k, d_model)

        # 2. 拆分为多头: (B, S, d_model) → (B, num_heads, S, dk)
        q = q.view(B, S_q, self.num_heads, self.dk).transpose(1, 2)
        k = k.view(B, S_k, self.num_heads, self.dk).transpose(1, 2)
        v = v.view(B, S_k, self.num_heads, self.dk).transpose(1, 2)

        # 3. 每个头独立计算注意力
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        weights = torch.softmax(scores, dim=-1)
        attn = weights @ v  # (B, num_heads, S_q, dk)

        # 4. 拼接所有头: (B, S_q, d_model)
        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)

        # 5. 输出投影
        return self.W_o(out)

# 测试
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
print(mha.forward(x, x, x).shape)  # (2, 10, 64)
```

---

## 3.3 Causal Self-Attention（因果自注意力）

### 是什么
在标准注意力基础上添加因果掩码，确保每个位置只能看到自己和之前的位置，不能"偷看"未来的 token。这是 GPT 等自回归语言模型的核心。

### 数学定义

$$\text{scores}_{ij} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{d_k}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

将未来位置的分数设为 $-\infty$，经过 softmax 后权重变为 0。

### 掩码实现
使用上三角矩阵（`torch.triu`）填充 $-\infty$：

```python
import torch
import math

def causal_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    # 创建上三角掩码（对角线以上为 -inf）
    mask = torch.triu(torch.full_like(scores, float('-inf')), diagonal=1)
    scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

# 测试
Q = K = V = torch.randn(1, 4, 8)
out = causal_attention(Q, K, V)
# 第 0 个位置只能看到自己，所以输出 = V[0]
print(torch.allclose(out[:, 0], V[:, 0], atol=1e-5))  # True
```

### 掩码矩阵可视化
```
位置:  0  1  2  3
  0  [ 0 -∞ -∞ -∞ ]   位置 0 只看自己
  1  [ 0  0 -∞ -∞ ]   位置 1 看 0,1
  2  [ 0  0  0 -∞ ]   位置 2 看 0,1,2
  3  [ 0  0  0  0 ]   位置 3 看所有
```

---

## 3.4 Multi-Head Cross-Attention（多头交叉注意力）

### 是什么
Q 来自一个序列（如解码器），K 和 V 来自另一个序列（如编码器）。用于 encoder-decoder 架构中，让解码器"查询"编码器的信息。

### 与自注意力的区别
- 自注意力：Q = K = V 来自同一输入
- 交叉注意力：Q 来自解码器，K/V 来自编码器
- 交叉注意力不需要因果掩码（编码器的所有位置对解码器可见）

### 代码示例

```python
import torch
import torch.nn as nn
import math

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        B, S_q, _ = x_q.shape
        S_kv = x_kv.shape[1]

        q = self.W_q(x_q).view(B, S_q, self.num_heads, self.dk).transpose(1, 2)
        k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.dk).transpose(1, 2)
        v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        weights = torch.softmax(scores, dim=-1)
        attn = (weights @ v).transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.W_o(attn)

# 测试：解码器 6 个 token 查询编码器 10 个 token
attn = MultiHeadCrossAttention(64, 4)
x_q = torch.randn(2, 6, 64)    # 解码器
x_kv = torch.randn(2, 10, 64)  # 编码器
print(attn(x_q, x_kv).shape)   # (2, 6, 64)
```

### 适用场景
- Transformer encoder-decoder（机器翻译、语音识别）
- 多模态模型（图像特征作为 KV，文本作为 Q）

---

## 3.5 Grouped Query Attention（GQA，分组查询注意力）

### 是什么
MHA 的高效变体。Q 保持完整的头数，但 K 和 V 使用更少的头（多个 Q 头共享同一组 K/V 头）。

### 为什么需要
- 标准 MHA 中 KV cache 的大小与头数成正比，是推理时的内存瓶颈
- GQA 减少 KV 头数，直接减少 KV cache 大小
- 当 `num_kv_heads = 1` 时退化为 Multi-Query Attention（MQA）
- 当 `num_kv_heads = num_heads` 时等价于标准 MHA

### 实现关键
使用 `repeat_interleave` 将少量 KV 头扩展到与 Q 头数量匹配：

```python
import torch
import torch.nn as nn
import math

class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dk = d_model // num_heads
        self.groups = num_heads // num_kv_heads  # 每组 Q 头共享一个 KV 头

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.dk)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.dk)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.dk).transpose(1, 2)

        # 扩展 KV 头以匹配 Q 头数量
        k = k.repeat_interleave(self.groups, dim=1)  # (B, num_heads, S, dk)
        v = v.repeat_interleave(self.groups, dim=1)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        weights = torch.softmax(scores, dim=-1)
        attn = (weights @ v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(attn)

# 测试：8 个 Q 头，2 个 KV 头
gqa = GroupQueryAttention(d_model=32, num_heads=8, num_kv_heads=2)
x = torch.randn(2, 6, 32)
print(gqa.forward(x).shape)  # (2, 6, 32)
print("W_k 参数量:", gqa.W_k.weight.shape)  # (8, 32) 而非 (32, 32)
```

### 适用场景
- LLaMA 2（num_kv_heads=8, num_heads=32）
- Mistral 7B（num_kv_heads=8, num_heads=32）
- 所有需要减少 KV cache 内存的场景

---

## 3.6 Sliding Window Attention（滑动窗口注意力）

### 是什么
每个位置只关注其周围固定大小窗口内的位置，而非全部位置。将注意力复杂度从 O(S²) 降低到 O(S·W)。

### 掩码规则
位置 $i$ 只能关注位置 $j$，当且仅当 $|i - j| \le w$（窗口大小）。

### 边界情况
- `window_size = 0`：每个位置只看自己，输出等于 V
- `window_size >= seq_len`：等价于全注意力

### 代码示例

```python
import torch
import math

def sliding_window_attention(Q, K, V, window_size):
    d_k = Q.shape[-1]
    S = Q.shape[1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)

    # 创建滑动窗口掩码
    positions = torch.arange(S)
    mask = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    return weights @ V

# 测试
Q = K = V = torch.randn(1, 8, 16)
out = sliding_window_attention(Q, K, V, window_size=2)
print(out.shape)  # (1, 8, 16)

# window=0 时输出等于 V
out0 = sliding_window_attention(Q, K, V, window_size=0)
print(torch.allclose(out0, V, atol=1e-5))  # True
```

### 适用场景
- Longformer（结合全局注意力 token）
- Mistral（结合 GQA）
- 处理长文档（>4k tokens）

---

## 3.7 Linear Attention（线性注意力）

### 是什么
用核函数特征映射替代 softmax，将注意力复杂度从 O(S²·D) 降低到 O(S·D²)。

### 核心思想
标准注意力：$\text{softmax}(QK^T)V$ 需要计算 S×S 矩阵。

线性注意力利用结合律改变计算顺序：
$$\text{LinearAttn}(Q,K,V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q) \sum \phi(K)}$$

先计算 $\phi(K)^T V$（D×D 矩阵），再乘以 $\phi(Q)$，避免了 S×S 矩阵。

### 特征映射
使用 $\phi(x) = \text{elu}(x) + 1$ 确保非负性（模拟 softmax 的非负权重）。

### 代码示例

```python
import torch
import torch.nn.functional as F

def linear_attention(Q, K, V):
    # 特征映射：elu(x) + 1 保证非负
    Q_prime = F.elu(Q) + 1  # (B, S, D_k)
    K_prime = F.elu(K) + 1  # (B, S, D_k)

    # 关键：先算 K^T V（D_k × D_v），再乘 Q
    KV = torch.bmm(K_prime.transpose(1, 2), V)  # (B, D_k, D_v)
    numerator = torch.bmm(Q_prime, KV)  # (B, S, D_v)

    # 归一化
    K_sum = K_prime.sum(dim=1, keepdim=True)  # (B, 1, D_k)
    denominator = torch.bmm(Q_prime, K_sum.transpose(1, 2))  # (B, S, 1)

    return numerator / (denominator + 1e-6)

# 测试
Q = torch.randn(1, 1024, 64)
K = torch.randn(1, 1024, 64)
V = torch.randn(1, 1024, 64)
out = linear_attention(Q, K, V)
print(out.shape)  # (1, 1024, 64)
```

### 优缺点
- 优点：长序列时计算效率高（O(S·D²) vs O(S²·D)）
- 缺点：近似质量不如标准 softmax 注意力，在某些任务上性能下降

---

## 3.8 KV Cache Attention

### 是什么
在自回归生成（逐 token 生成）时，缓存已计算的 K 和 V，避免重复计算。

### 为什么需要
生成第 $t$ 个 token 时，标准注意力需要对前 $t$ 个 token 全部重新计算 K 和 V。KV Cache 将之前的 K/V 存起来，每步只需计算新 token 的 K/V 并追加。

### 工作流程
```
Prefill（首次）: 输入 [t0, t1, t2, t3]
  → 计算完整因果注意力
  → 缓存 K_{0:3}, V_{0:3}

Decode（逐步）: 输入 [t4]
  → 只计算 t4 的 Q, K, V
  → K_all = concat(K_cache, K_new)
  → V_all = concat(V_cache, V_new)
  → Q_new 对 K_all 做注意力
  → 更新缓存
```

### 代码示例

```python
import torch
import torch.nn as nn
import math

class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, cache=None):
        B, S_new, _ = x.shape
        q = self.W_q(x).view(B, S_new, self.num_heads, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, S_new, self.num_heads, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, S_new, self.num_heads, self.dk).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)  # 拼接历史 K
            v = torch.cat([cache[1], v], dim=2)  # 拼接历史 V

        S_total = k.shape[2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)

        # Prefill 时需要因果掩码；单 token decode 时不需要
        if S_new > 1:
            mask = torch.triu(torch.full((S_new, S_total), float('-inf'),
                              device=x.device), diagonal=S_total - S_new + 1)
            scores = scores + mask

        weights = torch.softmax(scores, dim=-1)
        attn = (weights @ v).transpose(1, 2).contiguous().view(B, S_new, -1)
        return self.W_o(attn), (k, v)

# 测试：增量解码与全量前向结果一致
torch.manual_seed(0)
attn = KVCacheAttention(64, 4)
x = torch.randn(1, 6, 64)

full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)
print("匹配:", torch.allclose(full_out, inc_out, atol=1e-5))
```

---

## 3.9 Rotary Position Embedding（RoPE，旋转位置编码）

### 是什么
RoPE 通过旋转 Q 和 K 向量来编码位置信息，使得注意力分数自然地依赖于相对位置。

### 核心思想
将每个向量的相邻维度对视为二维平面上的点，按位置相关的角度旋转：

$$[x_0, x_1] \rightarrow [x_0 \cos\theta - x_1 \sin\theta, \; x_0 \sin\theta + x_1 \cos\theta]$$

角度 $\theta = \text{pos} / 10000^{2i/D}$，其中 pos 是位置，i 是维度对的索引。

### 关键性质
旋转后 $\text{dot}(q_{rot}[i], k_{rot}[j])$ 只依赖于 $i - j$（相对位置），而非绝对位置。这是因为旋转矩阵的正交性：$R_i^T R_j = R_{i-j}$。

### 代码示例

```python
import torch
import math

def apply_rope(q, k):
    B, S, D = q.shape
    assert D % 2 == 0

    # 计算频率
    positions = torch.arange(S, device=q.device).float()
    dim_pairs = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (10000.0 ** (dim_pairs / D))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (S, D/2)

    cos_vals = torch.cos(angles).unsqueeze(0)  # (1, S, D/2)
    sin_vals = torch.sin(angles).unsqueeze(0)

    # 拆分为偶数和奇数维度
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]

    # 旋转
    q_rot = torch.stack([
        q_even * cos_vals - q_odd * sin_vals,
        q_even * sin_vals + q_odd * cos_vals
    ], dim=-1).flatten(-2)

    k_rot = torch.stack([
        k_even * cos_vals - k_odd * sin_vals,
        k_even * sin_vals + k_odd * cos_vals
    ], dim=-1).flatten(-2)

    return q_rot, k_rot

# 测试
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
qr, kr = apply_rope(q, k)
print("形状保持:", qr.shape == q.shape)  # True
print("范数保持:", torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))  # True
```

### 适用场景
- LLaMA、GPT-NeoX、Mistral 等几乎所有现代 LLM
- 替代了传统的正弦位置编码和可学习位置编码

---

## 3.10 Flash Attention（分块注意力）

### 是什么
Flash Attention 通过分块计算和在线 softmax 算法，在不显式构建 S×S 注意力矩阵的情况下计算精确的注意力结果，大幅减少内存使用。

### 核心算法：Online Softmax
标准 softmax 需要两遍扫描（先求 max，再求 exp/sum）。Online softmax 在一遍扫描中完成，通过动态更新 running max 和 running sum：

```
对每个 KV 块:
  1. 计算局部分数 S_block = Q_block @ K_block^T / sqrt(d)
  2. 更新 running max: new_max = max(old_max, block_max)
  3. 修正累加器: acc *= exp(old_max - new_max)
  4. 累加新块: acc += exp(S_block - new_max) @ V_block
  5. 更新 running sum
最终: output = acc / row_sum
```

### 代码示例

```python
import torch
import math

def flash_attention(Q, K, V, block_size=32):
    B, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    output = torch.zeros_like(Q)
    row_max = torch.full((B, S, 1), float('-inf'), device=Q.device)
    row_sum = torch.zeros((B, S, 1), device=Q.device)

    for j_start in range(0, S, block_size):
        j_end = min(j_start + block_size, S)
        K_block = K[:, j_start:j_end]
        V_block = V[:, j_start:j_end]

        scores = torch.bmm(Q, K_block.transpose(1, 2)) * scale  # (B, S, block)
        block_max = scores.max(dim=-1, keepdim=True).values

        new_max = torch.maximum(row_max, block_max)
        # 修正已有累加器
        correction = torch.exp(row_max - new_max)
        output = output * correction
        row_sum = row_sum * correction

        # 累加新块
        exp_scores = torch.exp(scores - new_max)
        output = output + torch.bmm(exp_scores, V_block)
        row_sum = row_sum + exp_scores.sum(dim=-1, keepdim=True)
        row_max = new_max

    return output / row_sum

# 验证与标准注意力结果一致
Q = K = V = torch.randn(1, 64, 32)
out_flash = flash_attention(Q, K, V, block_size=16)
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(32)
out_ref = torch.bmm(torch.softmax(scores, dim=-1), V)
print("匹配:", torch.allclose(out_flash, out_ref, atol=1e-4))
```

### 优势
- 内存从 O(S²) 降低到 O(S)（不存储完整注意力矩阵）
- 利用 GPU SRAM 的高带宽，减少 HBM 访问
- 结果与标准注意力完全一致（不是近似）

---

## 3.11 注意力变体总结

```
                    标准 Attention
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Multi-Head     Causal        Cross-Attn
          │              │
     ┌────┴────┐    ┌────┴────┐
     ▼         ▼    ▼         ▼
   GQA    Sliding  KV Cache  Flash
  (省KV)  Window   (省计算)  (省内存)
          (省计算)

位置编码: RoPE（旋转编码，现代 LLM 标配）
线性化:   Linear Attention（O(n) 但近似）
```
