# 第四章：架构与模型组件

本章介绍如何将基础组件组装为完整的模型模块：GPT-2 Block、SwiGLU MLP、Conv2d、ViT Patch Embedding、LoRA、Mixture of Experts。

---

## 4.1 GPT-2 Transformer Block

### 是什么
GPT-2 Block 是 GPT 系列模型的基本构建单元，采用 Pre-Norm 架构，包含因果自注意力和前馈网络，通过残差连接组合。

### 架构（Pre-Norm）

```
输入 x
  │
  ├──────────────────────────┐
  │                          │ (残差连接)
  ▼                          │
LayerNorm(x)                 │
  │                          │
  ▼                          │
Causal Self-Attention        │
  │                          │
  ▼                          │
  + ◄────────────────────────┘
  │
  ├──────────────────────────┐
  │                          │ (残差连接)
  ▼                          │
LayerNorm(x)                 │
  │                          │
  ▼                          │
MLP (Linear→GELU→Linear)    │
  │                          │
  ▼                          │
  + ◄────────────────────────┘
  │
  ▼
输出
```

### Pre-Norm vs Post-Norm
- Pre-Norm：先归一化再做注意力/MLP（GPT-2、LLaMA 使用）
- Post-Norm：先做注意力/MLP 再归一化（原始 Transformer 论文）
- Pre-Norm 训练更稳定，不需要 warmup

### 代码示例

```python
import torch
import torch.nn as nn
import math

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # 注意力投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        # MLP: d → 4d → d
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def causal_attn(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1) @ v
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S, -1))

    def forward(self, x):
        x = x + self.causal_attn(self.ln1(x))  # 残差 + 注意力
        x = x + self.mlp(self.ln2(x))           # 残差 + MLP
        return x

# 测试
block = GPT2Block(d_model=64, num_heads=4)
x = torch.randn(2, 8, 64)
print(block(x).shape)  # (2, 8, 64)
print("参数量:", sum(p.numel() for p in block.parameters()))
```

### 关键设计决策
- MLP 隐藏层维度 = 4 × d_model（经验值）
- 使用 GELU 激活（而非 ReLU）
- 残差连接确保梯度流通

---

## 4.2 SwiGLU MLP

### 是什么
SwiGLU 是现代 LLM（LLaMA、Mistral、PaLM）使用的前馈网络，用门控机制替代了 GPT-2 的简单 FFN。

### 数学定义

$$\text{SwiGLU}(x) = \text{down\_proj}\big(\text{SiLU}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x)\big)$$

其中 SiLU（Swish）= $x \cdot \sigma(x)$，$\odot$ 是逐元素乘法。

### 与标准 FFN 的区别

| 标准 FFN (GPT-2) | SwiGLU MLP (LLaMA) |
|-------------------|---------------------|
| Linear(d, 4d) → GELU → Linear(4d, d) | gate_proj(d, d_ff) + up_proj(d, d_ff) → SiLU·gate → down_proj(d_ff, d) |
| 2 个线性层 | 3 个线性层 |
| 无门控 | 门控机制 |

### 门控机制的直觉
- `gate_proj` 决定"哪些信息通过"（经过 SiLU 激活后作为门）
- `up_proj` 提供"要传递的内容"
- 两者逐元素相乘，实现选择性信息传递

### 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))  # SiLU = x * sigmoid(x)
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# 测试
mlp = SwiGLUMLP(d_model=64, d_ff=128)
x = torch.randn(2, 8, 64)
print(mlp(x).shape)  # (2, 8, 64)
# 参数量：3 个线性层 = 64*128 + 64*128 + 128*64 + biases
print("参数量:", sum(p.numel() for p in mlp.parameters()))
```

---

## 4.3 2D Convolution（二维卷积）

### 是什么
卷积操作用滑动窗口（卷积核）在输入特征图上提取局部特征，是 CNN 的核心操作。

### 数学定义
对于输入 `(B, C_in, H, W)` 和卷积核 `(C_out, C_in, kH, kW)`：

$$\text{out}[b, c_{out}, i, j] = \sum_{c_{in}} \sum_{m,n} x[b, c_{in}, i \cdot s + m, j \cdot s + n] \cdot w[c_{out}, c_{in}, m, n] + \text{bias}[c_{out}]$$

### 输出尺寸计算

$$H_{out} = \lfloor\frac{H + 2 \cdot \text{padding} - kH}{\text{stride}}\rfloor + 1$$

### 代码示例

```python
import torch
import torch.nn.functional as F

def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    if padding > 0:
        x = F.pad(x, [padding] * 4)

    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    output = torch.zeros(B, C_out, H_out, W_out, device=x.device)
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            patch = x[:, :, h_start:h_start+kH, w_start:w_start+kW]
            # patch: (B, C_in, kH, kW), weight: (C_out, C_in, kH, kW)
            output[:, :, i, j] = (patch.unsqueeze(1) * weight.unsqueeze(0)).sum(dim=(2, 3, 4))

    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    return output

# 测试
x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
out = my_conv2d(x, w, stride=1, padding=1)
print(out.shape)  # (1, 16, 8, 8)
ref = F.conv2d(x, w, padding=1)
print("匹配:", torch.allclose(out, ref, atol=1e-4))
```

---

## 4.4 ViT Patch Embedding

### 是什么
Vision Transformer（ViT）的第一步：将图像分割为不重叠的 patch，每个 patch 展平后通过线性层投影为嵌入向量。

### 算法步骤
1. 将 `(B, C, H, W)` 图像分割为 `(H/P × W/P)` 个 patch，每个 patch 大小 `(C, P, P)`
2. 展平每个 patch 为 `(C × P × P)` 维向量
3. 线性投影到 `embed_dim` 维

### 代码示例

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        # 重排为 patches: (B, num_patches, C*P*P)
        x = x.unfold(2, P, P).unfold(3, P, P)  # (B, C, H/P, W/P, P, P)
        x = x.contiguous().view(B, C, -1, P, P)  # (B, C, N, P, P)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * P * P)
        return self.proj(x)

# 测试：32×32 图像，8×8 patch，3 通道，64 维嵌入
pe = PatchEmbedding(32, 8, 3, 64)
x = torch.randn(2, 3, 32, 32)
print(pe(x).shape)       # (2, 16, 64)  — 16 个 patch
print(pe.num_patches)    # 16
```

### 适用场景
- ViT（Vision Transformer）
- CLIP 的视觉编码器
- 任何将图像输入 Transformer 的场景

---

## 4.5 LoRA（Low-Rank Adaptation）

### 是什么
LoRA 是一种参数高效微调方法。冻结预训练权重，只训练两个低秩矩阵 A 和 B，使得微调参数量大幅减少。

### 数学定义

$$h = W_0 x + \frac{\alpha}{r} B A x$$

- $W_0$：冻结的预训练权重 `(out, in)`
- $A$：低秩矩阵 `(rank, in)`，随机初始化
- $B$：低秩矩阵 `(out, rank)`，零初始化
- $\alpha / r$：缩放因子

### 为什么 B 零初始化
训练开始时 $BA = 0$，LoRA 不改变原始模型的输出。随着训练进行，B 逐渐学习到有意义的值。

### 参数效率
原始 Linear(1024, 1024) 有 ~1M 参数。LoRA rank=8 只需 1024×8 + 8×1024 = 16K 参数（1.6%）。

### 代码示例

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

# 测试
layer = LoRALinear(16, 8, rank=4)
x = torch.randn(2, 16)
print(layer(x).shape)  # (2, 8)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"可训练参数: {trainable}, 总参数: {total}, 比例: {trainable/total:.1%}")
```

---

## 4.6 Mixture of Experts（MoE，混合专家）

### 是什么
MoE 层包含多个"专家"（独立的 MLP），通过路由网络为每个 token 选择 top-k 个专家，只激活部分专家进行计算。

### 架构

```
输入 token
    │
    ▼
Router (Linear → Softmax)
    │
    ├── Expert 0 (MLP)  ← 权重 0.7
    ├── Expert 1 (MLP)  ← 权重 0.3
    ├── Expert 2 (MLP)  ← 未选中
    └── Expert 3 (MLP)  ← 未选中
    │
    ▼
加权求和输出
```

### 为什么需要
- 增加模型容量而不成比例增加计算量
- Mixtral 8x7B：8 个专家，每次选 2 个，总参数 47B 但每个 token 只用 ~13B
- 实现"条件计算"：不同 token 走不同路径

### 代码示例

```python
import torch
import torch.nn as nn

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (B*S, D)

        # 路由：选择 top-k 专家
        router_logits = self.router(x_flat)  # (B*S, num_experts)
        topk_vals, topk_idx = router_logits.topk(self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)  # 归一化权重

        # 计算每个选中专家的输出并加权求和
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]  # (B*S,)
            weight = topk_weights[:, k].unsqueeze(-1)  # (B*S, 1)
            for e_idx in range(len(self.experts)):
                mask = (expert_idx == e_idx)
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e_idx](x_flat[mask])

        return output.view(B, S, D)

# 测试
moe = MixtureOfExperts(32, 64, num_experts=4, top_k=2)
x = torch.randn(2, 8, 32)
print(moe(x).shape)  # (2, 8, 32)
```

### 适用场景
- Mixtral 8x7B、Switch Transformer、GShard
- 需要大模型容量但计算预算有限的场景
