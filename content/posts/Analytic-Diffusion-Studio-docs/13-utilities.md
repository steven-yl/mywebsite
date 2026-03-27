---
title: "Analytic Diffusion Studio — 工具模块"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 工具模块"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 13
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 工具模块"
---

# 13 — 工具模块

目录：`src/local_diffusion/utils/`

## 13.1 模块结构

```
utils/
├── __init__.py              # 导出公共 API
├── wiener.py                # Wiener 滤波计算与存储
└── neural_networks.py       # UNet 网络定义
```

`__init__.py` 导出：

```python
from .wiener import compute_wiener_filter, load_wiener_filter, save_wiener_filter
from .neural_networks import UNet
```

---

## 13.2 Wiener 滤波工具 (`wiener.py`)

### compute_wiener_filter()

```python
def compute_wiener_filter(dataloader, device, resolution, n_channels):
    """从数据集计算协方差矩阵和均值。"""
```

**两遍扫描算法**：

第一遍：计算均值
```python
for batch in dataloader:
    images = batch[0].to(device).flatten(start_dim=1)  # [batch, n_pixels]
    sum_images += images.sum(dim=0)
    total_samples += images.shape[0]
mean = sum_images / total_samples
```

第二遍：计算协方差
```python
for batch in dataloader:
    images = batch[0].to(device).flatten(start_dim=1)
    centered = images - mean.unsqueeze(0)
    cov_accumulator += centered.T @ centered  # [n, n]
S = cov_accumulator / (total_samples - 1)
```

**返回值**：
- `S`：协方差矩阵 `[n_pixels, n_pixels]`
- `mean`：均值向量 `[n_pixels]`

**内存需求**：协方差矩阵大小为 $n^2$，其中 $n = C \times H \times W$：
- MNIST (784)：约 2.3 MB
- CIFAR-10 (3072)：约 36 MB
- 64×64 RGB (12288)：约 576 MB

**为什么用两遍扫描？** 一遍扫描的在线协方差算法（如 Welford）数值稳定性更好，但两遍扫描更简单且在 GPU 上更高效（可以利用矩阵乘法加速）。

### save_wiener_filter()

```python
def save_wiener_filter(U, LA, Vh, mean, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(U.cpu(), save_path / "U.pt")
    torch.save(LA.cpu(), save_path / "LA.pt")
    torch.save(Vh.cpu(), save_path / "Vh.pt")
    torch.save(mean.cpu(), save_path / "mean.pt")
```

保存 SVD 分解的四个组件到指定目录。

### load_wiener_filter()

```python
def load_wiener_filter(load_path, device=None):
    U = torch.load(load_path / "U.pt", map_location=device, weights_only=True)
    LA = torch.load(load_path / "LA.pt", map_location=device, weights_only=True)
    Vh = torch.load(load_path / "Vh.pt", map_location=device, weights_only=True)
    mean = torch.load(load_path / "mean.pt", map_location=device, weights_only=True)
    return U, LA, Vh, mean
```

如果文件不存在，抛出 `FileNotFoundError`（调用方负责处理）。

**`weights_only=True`**：PyTorch 安全加载模式，防止反序列化攻击。

---

## 13.3 UNet 网络 (`neural_networks.py`)

这是一个标准的 DDPM UNet 实现，用于 `BaselineUNet` 模型。

### 整体架构

```
输入 x_t [B, C, H, W] + 时间步 t [B]
    │
    ▼
TimeEmbedding(t) → temb [B, tdim]
    │
    ▼
Head Conv (C → ch)
    │
    ▼
┌─ Encoder ──────────────────────┐
│  Level 0: ResBlock × 2         │
│  DownSample                    │
│  Level 1: ResBlock × 2         │
│  DownSample                    │
│  ...                           │
│  Level N: ResBlock × 2         │
└────────────────────────────────┘
    │
    ▼
Middle: ResBlock(attn=True) + ResBlock(attn=False)
    │
    ▼
┌─ Decoder ──────────────────────┐
│  Level N: ResBlock × 3 (+ skip)│
│  UpSample                      │
│  ...                           │
│  Level 0: ResBlock × 3 (+ skip)│
└────────────────────────────────┘
    │
    ▼
Tail: GroupNorm → Swish → Conv (ch → C)
    │
    ▼
输出 ε̂ [B, C, H, W]
```

### Swish 激活函数

```python
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```

$\text{Swish}(x) = x \cdot \sigma(x)$，是一种平滑的非线性激活，在扩散模型中广泛使用。

### TimeEmbedding

```python
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        # 正弦位置编码
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1).view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),  # 查表
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
```

将离散时间步 $t \in \{0, 1, ..., T-1\}$ 编码为连续向量：
1. 正弦/余弦位置编码（类似 Transformer）
2. 两层 MLP 投影到 `tdim = ch * 4` 维

### DownSample

```python
class DownSample(nn.Module):
    def __init__(self, in_ch):
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
```

使用 stride=2 的卷积实现 2× 下采样。

### UpSample

```python
class UpSample(nn.Module):
    def __init__(self, in_ch):
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, temb):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x
```

先最近邻插值 2× 上采样，再卷积平滑。

### AttnBlock

```python
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)
```

标准自注意力机制：
1. GroupNorm 归一化
2. 1×1 卷积生成 Q、K、V
3. 注意力权重：$W = \text{softmax}(QK^T / \sqrt{C})$
4. 输出：$h = WV$
5. 残差连接：$x + \text{proj}(h)$

### ResBlock

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        self.block1 = nn.Sequential(GroupNorm, Swish, Conv2d)
        self.temb_proj = nn.Sequential(Swish, Linear(tdim, out_ch))
        self.block2 = nn.Sequential(GroupNorm, Swish, Dropout, Conv2d)
        self.shortcut = Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else Identity()
        self.attn = AttnBlock(out_ch) if attn else Identity()
```

前向传播：
```python
def forward(self, x, temb):
    h = self.block1(x)
    h += self.temb_proj(temb)[:, :, None, None]  # 时间嵌入注入
    h = self.block2(h)
    h = h + self.shortcut(x)                      # 残差连接
    h = self.attn(h)                               # 可选注意力
    return h
```

### FlattenLinear

```python
class FlattenLinear(nn.Module):
    def __init__(self, channels, height, width, tdim):
        self.linear = nn.Linear(channels * height * width + tdim,
                                channels * height * width)
```

将特征图展平后与时间嵌入拼接，通过全连接层处理。在当前配置中未使用（为扩展预留）。

### UNet 主类

```python
class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout,
                 in_channels=3, out_channels=3):
```

#### 构造函数参数

| 参数 | 说明 |
|------|------|
| `T` | 时间步总数（1000） |
| `ch` | 基础通道数 |
| `ch_mult` | 各级通道倍率列表 |
| `attn` | 使用注意力的级别索引列表 |
| `num_res_blocks` | 每级残差块数 |
| `dropout` | Dropout 概率 |
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数 |

#### forward() 方法

```python
def forward(self, x, t, return_middle_feature=False, return_all_features=False):
    temb = self.time_embedding(t)
    h = self.head(x)
    hs = [h]

    # 编码器
    for layer in self.downblocks:
        h = layer(h, temb)
        hs.append(h)

    # 中间层
    for layer in self.middleblocks:
        h = layer(h, temb)

    # 解码器（带跳跃连接）
    for layer in self.upblocks:
        if isinstance(layer, ResBlock):
            h = torch.cat([h, hs.pop()], dim=1)  # 跳跃连接
        h = layer(h, temb)

    h = self.tail(h)
    return h
```

可选返回中间特征（用于分析）：
- `return_middle_feature=True`：返回 `(output, middle_feature, temb)`
- `return_all_features=True`：返回 `(output, middle_feature, pretail_features, temb)`

#### 权重初始化

```python
def initialize(self):
    init.xavier_uniform_(self.head.weight)
    init.zeros_(self.head.bias)
    init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)  # 小增益
    init.zeros_(self.tail[-1].bias)
```

尾部卷积使用极小的增益（`1e-5`），使初始输出接近零——这是扩散模型训练的常见技巧。

### 编码器-解码器通道数示例

以 CIFAR-10 (32×32, ch=128, ch_mult=[1,2,3,4]) 为例：

```
编码器：
  Level 0: 128 → 128 (ResBlock ×2), DownSample → 16×16
  Level 1: 128 → 256 (ResBlock ×2), DownSample → 8×8
  Level 2: 256 → 384 (ResBlock ×2), DownSample → 4×4
  Level 3: 384 → 512 (ResBlock ×2)

中间层：
  512 → 512 (ResBlock with Attn)
  512 → 512 (ResBlock)

解码器：
  Level 3: 512+512 → 512 (ResBlock ×3)
  UpSample → 8×8
  Level 2: 512+384 → 384 (ResBlock ×3)
  UpSample → 16×16
  Level 1: 384+256 → 256 (ResBlock ×3)
  UpSample → 32×32
  Level 0: 256+128 → 128 (ResBlock ×3)
```
