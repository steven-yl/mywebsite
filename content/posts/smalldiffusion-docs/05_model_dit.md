---
title: "smalldiffusion 模型：model_dit.py"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 模型：model_dit.py"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 5
hiddenFromHomePage: false
hiddenFromSearch: false
---

> 本文件实现了 [DiT (Peebles & Xie, 2022)](https://arxiv.org/abs/2212.09748) 架构，一种基于 Transformer 的扩散模型。

## 5.1 模块结构

```
model_dit.py
├── PatchEmbed          # 图像 Patch 嵌入
├── Modulation          # 自适应调制层
├── ModulatedLayerNorm  # 调制 LayerNorm
├── DiTBlock            # DiT Transformer 块
├── get_pos_embed()     # 2D 正弦余弦位置编码
└── DiT                 # 完整 DiT 模型
```

---

## 5.2 整体架构

```
输入图像 (B, C, H, W)
    │
    ▼
PatchEmbed ──→ (B, N, D)  ← + pos_embed (1, N, D)
    │
    │  σ ──→ SigmaEmbedderSinCos ──→ y (B, D)
    │  cond ──→ CondEmbedderLabel ──→ + y
    │
    ▼
DiTBlock × depth  ← y 作为条件
    │
    ▼
ModulatedLayerNorm ← y
    │
    ▼
Linear (D → patch_size² × C)
    │
    ▼
Unpatchify ──→ 输出 (B, C, H, W)
```

---

## 5.3 PatchEmbed

### 是什么

将图像分割为不重叠的 patch 并线性嵌入到高维空间。

```python
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, channels=3, embed_dim=768, bias=True):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim,
                              stride=patch_size, kernel_size=patch_size, bias=bias)
        self.init()

    def init(self):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        return rearrange(self.proj(x), 'b c h w -> b (h w) c')
```

### 工作原理

使用 `Conv2d` 实现 patch 化：
- `kernel_size=patch_size, stride=patch_size`：不重叠地将图像分割为 patch
- 输入 `(B, C, H, W)` → 卷积输出 `(B, D, H/p, W/p)` → 重排为 `(B, N, D)`
- 其中 `N = (H/p) × (W/p)` 是 patch 数量，`D = embed_dim`

### 初始化

使用 Xavier 均匀初始化（模拟 `nn.Linear` 的初始化），偏置初始化为 0。

---

## 5.4 Modulation

### 是什么

从条件向量 $y$ 生成 $n$ 组调制参数的模块。

```python
class Modulation(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.n = n
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, n * dim, bias=True))
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)

    def forward(self, y):
        return [m.unsqueeze(1) for m in self.proj(y).chunk(self.n, dim=1)]
```

### 工作原理

1. 输入条件向量 $y$：形状 `(B, D)`
2. 通过 `SiLU → Linear` 映射到 `(B, n*D)`
3. 沿维度 1 分割为 $n$ 组，每组 `(B, D)`
4. 每组增加维度变为 `(B, 1, D)` 用于广播

### 零初始化

线性层的权重和偏置初始化为 0，使得训练初期调制参数为零向量，模型行为接近无条件的标准 Transformer。这是一种常见的稳定训练技巧。

---

## 5.5 ModulatedLayerNorm

### 是什么

带自适应调制的 LayerNorm，即 adaLN（adaptive Layer Normalization）。

```python
class ModulatedLayerNorm(nn.LayerNorm):
    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)
        self.modulation = Modulation(dim, 2)

    def forward(self, x, y):
        scale, shift = self.modulation(y)
        return super().forward(x) * (1 + scale) + shift
```

### 数学公式

$$\text{adaLN}(x, y) = \text{LN}(x) \cdot (1 + \gamma(y)) + \beta(y)$$

其中 $\gamma, \beta$ 是从条件 $y$ 生成的 scale 和 shift 参数。

### 为什么使用 adaLN

标准 LayerNorm 的 scale/shift 是可学习参数，对所有输入相同。adaLN 使这些参数依赖于条件信息（噪声水平 + 类别标签），让模型在不同条件下有不同的归一化行为。这是 DiT 论文中性能最好的条件注入方式。

---

## 5.6 DiTBlock

### 是什么

DiT 的核心 Transformer 块，包含调制注意力和调制 MLP。

```python
class DiTBlock(nn.Module):
    def __init__(self, head_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        dim = head_dim * num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(head_dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim, bias=True),
        )
        self.scale_modulation = Modulation(dim, 2)

    def forward(self, x, y):
        gate_msa, gate_mlp = self.scale_modulation(y)
        x = x + gate_msa * self.attn(self.norm1(x, y))
        x = x + gate_mlp * self.mlp(self.norm2(x, y))
        return x
```

### 计算流程

```
输入 x (B, N, D), y (B, D)
│
├─ gate_msa, gate_mlp = Modulation(y)     # 门控参数
│
├─ h = adaLN_1(x, y)                      # 调制归一化
├─ h = Attention(h)                        # 多头自注意力
├─ x = x + gate_msa * h                   # 门控残差连接
│
├─ h = adaLN_2(x, y)                      # 调制归一化
├─ h = MLP(h)                             # 前馈网络
├─ x = x + gate_mlp * h                   # 门控残差连接
│
输出 x (B, N, D)
```

### 设计要点

1. **adaLN-Zero**：使用 `elementwise_affine=False` 的 LayerNorm，所有 scale/shift 完全由条件 $y$ 决定
2. **门控机制**：`gate_msa` 和 `gate_mlp` 对注意力和 MLP 的输出进行缩放，初始化为 0 使训练初期残差连接为恒等映射
3. **MLP 扩展比**：`mlp_ratio=4.0`，隐藏层维度是输入的 4 倍（标准 Transformer 设计）
4. **近似 GELU**：使用 `tanh` 近似的 GELU 激活函数

---

## 5.7 get_pos_embed 函数

### 是什么

生成 2D 正弦余弦位置编码，为每个 patch 提供空间位置信息。

```python
def get_pos_embed(in_dim, patch_size, dim, N=10000):
    n = in_dim // patch_size                                          # 每边 patch 数
    assert dim % 4 == 0, 'Embedding dimension must be multiple of 4!'
    omega = 1/N**np.linspace(0, 1, dim // 4, endpoint=False)          # 频率
    freqs = np.outer(np.arange(n), omega)                             # (n, dim/4)
    embeds = repeat(np.stack([np.sin(freqs), np.cos(freqs)]),
                       ' b n d -> b n k d', k=n)                      # (2, n, n, dim/4)
    embeds_2d = np.concatenate([
        rearrange(embeds, 'b n k d -> (k n) (b d)'),                  # (n², dim/2)
        rearrange(embeds, 'b n k d -> (n k) (b d)'),                  # (n², dim/2)
    ], axis=1)                                                        # (n², dim)
    return nn.Parameter(torch.tensor(embeds_2d).float().unsqueeze(0),
                        requires_grad=False)                           # (1, n², dim)
```

### 工作原理

1. 计算每边的 patch 数 `n = in_dim / patch_size`
2. 生成频率向量 $\omega_k = 1/N^{k/(d/4)}$，其中 $N=10000$
3. 对行坐标和列坐标分别计算正弦/余弦编码
4. 将行编码和列编码拼接，得到 `dim` 维的 2D 位置编码
5. 返回不可训练的参数（`requires_grad=False`）

### 维度要求

嵌入维度 `dim` 必须是 4 的倍数，因为需要分配给：行-sin、行-cos、列-sin、列-cos 各 `dim/4` 维。

---

## 5.8 DiT 完整模型

### 是什么

完整的 Diffusion Transformer 模型，组合上述所有组件。

```python
class DiT(nn.Module, ModelMixin):
    def __init__(self, in_dim=32, channels=3, patch_size=2, depth=12,
                 head_dim=64, num_heads=6, mlp_ratio=4.0,
                 sig_embed=None, cond_embed=None):
        super().__init__()
        self.input_dims = (channels, in_dim, in_dim)
        dim = head_dim * num_heads

        self.pos_embed = get_pos_embed(in_dim, patch_size, dim)
        self.x_embed = PatchEmbed(patch_size, channels, dim, bias=True)
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(dim)
        self.cond_embed = cond_embed

        self.blocks = CondSequential(*[
            DiTBlock(head_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_norm = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(dim, patch_size**2 * channels)
        self.init()
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_dim` | 32 | 输入图像边长 |
| `channels` | 3 | 输入通道数 |
| `patch_size` | 2 | Patch 大小 |
| `depth` | 12 | Transformer 块数量 |
| `head_dim` | 64 | 每个注意力头的维度 |
| `num_heads` | 6 | 注意力头数量 |
| `mlp_ratio` | 4.0 | MLP 隐藏层扩展比 |
| `sig_embed` | None | 自定义 σ 嵌入器（默认 SigmaEmbedderSinCos） |
| `cond_embed` | None | 条件嵌入器（None 表示无条件模型） |

### init 方法

```python
def init(self):
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    self.apply(_basic_init)
    nn.init.normal_(self.sig_embed.mlp[0].weight, std=0.02)
    nn.init.normal_(self.sig_embed.mlp[2].weight, std=0.02)
    nn.init.constant_(self.final_linear.weight, 0)
    nn.init.constant_(self.final_linear.bias, 0)
```

初始化策略：
1. 所有 Linear 层使用 Xavier 均匀初始化
2. σ 嵌入 MLP 使用小标准差正态初始化
3. 最终输出层零初始化（训练初期输出接近零，即预测"无噪声"）

### unpatchify 方法

将 patch 序列重组为图像：

```python
def unpatchify(self, x):
    patches = self.in_dim // self.patch_size
    return rearrange(x, 'b (ph pw) (psh psw c) -> b c (ph psh) (pw psw)',
                     ph=patches, pw=patches,
                     psh=self.patch_size, psw=self.patch_size)
```

`(B, N, patch_size² × C)` → `(B, C, H, W)`

### forward 方法

```python
def forward(self, x, sigma, cond=None):
    x = self.x_embed(x) + self.pos_embed            # (B, N, D)
    y = self.sig_embed(x.shape[0], sigma.squeeze())  # (B, D)
    if self.cond_embed is not None:
        assert cond is not None and x.shape[0] == cond.shape[0]
        y += self.cond_embed(cond)                   # (B, D)
    x = self.blocks(x, y)                            # (B, N, D)
    x = self.final_linear(self.final_norm(x, y))     # (B, N, p²C)
    return self.unpatchify(x)                         # (B, C, H, W)
```

条件信息（σ 嵌入 + 可选的类别嵌入）通过相加合并为统一的条件向量 $y$，然后通过 adaLN 注入每个 Transformer 块。

### 使用示例

```python
from smalldiffusion import DiT, CondEmbedderLabel, ScheduleDDPM, training_loop, samples
from torch.utils.data import DataLoader

# 无条件 DiT
model = DiT(in_dim=28, channels=1, patch_size=2, depth=6,
            head_dim=32, num_heads=6)

# 条件 DiT（10 类）
model = DiT(in_dim=28, channels=1, patch_size=2, depth=6,
            head_dim=32, num_heads=6,
            cond_embed=CondEmbedderLabel(32*6, 10, dropout_prob=0.1))
```
