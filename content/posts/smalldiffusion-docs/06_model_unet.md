---
title: "smalldiffusion 模型：model_unet.py"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 模型：model_unet.py"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false
---

> 本文件实现了经典的 U-Net 扩散模型架构，改编自 [PNDM](https://github.com/luping-liu/PNDM) 和 [DDIM](https://github.com/ermongroup/ddim) 的实现。

## 6.1 模块结构

```
model_unet.py
├── Normalize()     # GroupNorm 工厂函数
├── Upsample()      # 上采样模块
├── Downsample()    # 下采样模块
├── ResnetBlock     # 残差块
├── AttnBlock       # 注意力块
└── Unet            # 完整 U-Net 模型
```

---

## 6.2 整体架构

```
输入 (B, C_in, H, W)
    │
    ▼
Conv_in (C_in → ch)
    │
    ▼
┌─ Down Block 1 ──┐  ← ResnetBlock × num_res_blocks [+ AttnBlock]
│  Downsample      │
├─ Down Block 2 ──┤  ← ResnetBlock × num_res_blocks [+ AttnBlock]
│  Downsample      │
├─ ...            ─┤
│  (no downsample) │  ← 最后一级不下采样
└──────────────────┘
    │
    ▼
Middle: ResnetBlock → AttnBlock → ResnetBlock
    │
    ▼
┌─ Up Block 1 ────┐  ← ResnetBlock × (num_res_blocks+1) [+ AttnBlock]
│  Upsample        │     每个 ResnetBlock 接收 skip connection
├─ Up Block 2 ────┤
├─ ...            ─┤
│  (no upsample)   │  ← 最后一级不上采样
└──────────────────┘
    │
    ▼
Normalize → SiLU → Conv_out (ch → C_out)
    │
    ▼
输出 (B, C_out, H, W)
```

---

## 6.3 Normalize 工厂函数

```python
def Normalize(ch):
    return torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)
```

使用 GroupNorm（32 组）代替 BatchNorm。GroupNorm 不依赖 batch 统计量，在小 batch 和分布式训练中更稳定。

---

## 6.4 Upsample 和 Downsample

### Upsample

```python
def Upsample(ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        torch.nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
    )
```

最近邻插值 2× 上采样 + 3×3 卷积平滑。

### Downsample

```python
def Downsample(ch):
    return nn.Sequential(
        nn.ConstantPad2d((0, 1, 0, 1), 0),
        torch.nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0),
    )
```

先在右边和下边各填充 1 像素（零填充），然后用 stride=2 的 3×3 卷积实现 2× 下采样。填充确保奇数尺寸的输入也能正确处理。

---

## 6.5 ResnetBlock

### 是什么

带时间嵌入注入的残差块，是 U-Net 的基本构建单元。

```python
class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch

        self.layer1 = nn.Sequential(
            Normalize(in_ch), nn.SiLU(),
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, out_ch),
        )
        self.layer2 = nn.Sequential(
            Normalize(out_ch), nn.SiLU(), torch.nn.Dropout(dropout),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        if self.in_ch != self.out_ch:
            kernel_stride_padding = (3,1,1) if self.use_conv_shortcut else (1,1,0)
            self.shortcut = torch.nn.Conv2d(in_ch, out_ch, *kernel_stride_padding)

    def forward(self, x, temb):
        h = self.layer1(x)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.layer2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h
```

### 计算流程

```
输入 x (B, C_in, H, W), temb (B, temb_ch)
│
├─ h = Norm → SiLU → Conv3×3 (C_in → C_out)     # layer1
├─ h = h + Linear(temb)[:,:,None,None]             # 时间嵌入注入（广播到空间维度）
├─ h = Norm → SiLU → Dropout → Conv3×3            # layer2
│
├─ if C_in ≠ C_out: x = Conv(x)                   # shortcut 对齐通道数
│
└─ output = x + h                                  # 残差连接
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `in_ch` | 输入通道数 |
| `out_ch` | 输出通道数（默认等于 in_ch） |
| `conv_shortcut` | shortcut 使用 3×3 卷积还是 1×1 卷积 |
| `dropout` | Dropout 概率 |
| `temb_channels` | 时间嵌入维度 |

### 时间嵌入注入方式

时间嵌入通过线性投影后加到特征图上：`h + proj(temb)[:, :, None, None]`。`None, None` 将 `(B, C)` 扩展为 `(B, C, 1, 1)` 以广播到所有空间位置。

---

## 6.6 AttnBlock

### 是什么

U-Net 中的自注意力块，在特定分辨率下对空间特征做自注意力。

```python
class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=1):
        super().__init__()
        self.norm = Normalize(ch)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads)
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        B, C, H, W = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attn(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=H, w=W)
        return x + self.proj_out(h_)
```

### 计算流程

1. GroupNorm 归一化
2. 将空间维度展平为序列：`(B, C, H, W) → (B, H×W, C)`
3. 多头自注意力
4. 恢复空间维度：`(B, H×W, C) → (B, C, H, W)`
5. 1×1 卷积投影 + 残差连接

### 设计细节

- `temb` 参数未使用，但保留以兼容 `CondSequential` 的接口
- 默认 `num_heads=1`，即单头注意力
- 复用 `model.py` 中的 `Attention` 模块

---

## 6.7 Unet 完整模型

### 是什么

完整的 U-Net 扩散模型，支持多分辨率特征提取和 skip connection。

```python
class Unet(nn.Module, ModelMixin):
    def __init__(self, in_dim, in_ch, out_ch,
                 ch=128, ch_mult=(1,2,2,2), embed_ch_mult=4,
                 num_res_blocks=2, attn_resolutions=(16,),
                 dropout=0.1, resamp_with_conv=True,
                 sig_embed=None, cond_embed=None):
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_dim` | - | 输入图像边长 |
| `in_ch` | - | 输入通道数 |
| `out_ch` | - | 输出通道数 |
| `ch` | 128 | 基础通道数 |
| `ch_mult` | (1,2,2,2) | 各级通道倍数 |
| `embed_ch_mult` | 4 | 嵌入通道倍数 |
| `num_res_blocks` | 2 | 每级残差块数量 |
| `attn_resolutions` | (16,) | 使用注意力的分辨率 |
| `dropout` | 0.1 | Dropout 概率 |
| `sig_embed` | None | σ 嵌入器 |
| `cond_embed` | None | 条件嵌入器 |

### 通道数计算

以 `ch=128, ch_mult=(1,2,2,2)` 为例：

```
级别 0: 128 × 1 = 128
级别 1: 128 × 2 = 256
级别 2: 128 × 2 = 256
级别 3: 128 × 2 = 256
```

`in_ch_dim = [ch * m for m in (1,) + ch_mult]` = `[128, 128, 256, 256, 256]`

### 下采样路径

```python
self.conv_in = torch.nn.Conv2d(in_ch, self.ch, kernel_size=3, stride=1, padding=1)
self.downs = nn.ModuleList()
for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
    down = nn.Module()
    down.blocks = nn.ModuleList()
    for _ in range(self.num_res_blocks):
        block = [make_block(block_in, block_out)]
        if curr_res in attn_resolutions:
            block.append(AttnBlock(block_out))
        down.blocks.append(CondSequential(*block))
        block_in = block_out
    if i < self.num_resolutions - 1:
        down.downsample = Downsample(block_in)
        curr_res = curr_res // 2
    self.downs.append(down)
```

每级包含：
- `num_res_blocks` 个 ResnetBlock（可选 AttnBlock）
- 除最后一级外，末尾有 Downsample

### 中间层

```python
self.mid = CondSequential(
    make_block(block_in, block_in),
    AttnBlock(block_in),
    make_block(block_in, block_in)
)
```

ResnetBlock → AttnBlock → ResnetBlock，在最低分辨率处理全局信息。

### 上采样路径

```python
self.ups = nn.ModuleList()
for i_level, (block_out, next_skip_in) in enumerate(pairwise(reversed(in_ch_dim))):
    up = nn.Module()
    up.blocks = nn.ModuleList()
    skip_in = block_out
    for i_block in range(self.num_res_blocks + 1):
        if i_block == self.num_res_blocks:
            skip_in = next_skip_in
        block = [make_block(block_in + skip_in, block_out)]
        if curr_res in attn_resolutions:
            block.append(AttnBlock(block_out))
        up.blocks.append(CondSequential(*block))
        block_in = block_out
    if i_level < self.num_resolutions - 1:
        up.upsample = Upsample(block_in)
        curr_res = curr_res * 2
    self.ups.append(up)
```

每级包含：
- `num_res_blocks + 1` 个 ResnetBlock（比下采样多一个，用于处理 skip connection）
- 每个 ResnetBlock 的输入通道数 = 当前通道 + skip 通道
- 除最后一级外，末尾有 Upsample

### forward 方法

```python
def forward(self, x, sigma, cond=None):
    assert x.shape[2] == x.shape[3] == self.in_dim

    # 嵌入
    emb = self.sig_embed(x.shape[0], sigma.squeeze())
    if self.cond_embed is not None:
        emb += self.cond_embed(cond)

    # 下采样（收集 skip connections）
    hs = [self.conv_in(x)]
    for down in self.downs:
        for block in down.blocks:
            h = block(hs[-1], emb)
            hs.append(h)
        if hasattr(down, 'downsample'):
            hs.append(down.downsample(hs[-1]))

    # 中间层
    h = self.mid(hs[-1], emb)

    # 上采样（消费 skip connections）
    for up in self.ups:
        for block in up.blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), emb)
        if hasattr(up, 'upsample'):
            h = up.upsample(h)

    # 输出
    return self.out_layer(h)
```

### Skip Connection 机制

下采样路径中，每个中间特征图都被压入 `hs` 栈。上采样路径中，每个 ResnetBlock 从 `hs` 栈弹出对应的特征图，与当前特征图在通道维度拼接。这种对称的 skip connection 是 U-Net 的核心设计，帮助保留高分辨率细节。

### 使用示例

```python
from smalldiffusion import Unet, Scaled, ScheduleLogLinear, training_loop, samples

# FashionMNIST (28×28, 灰度)
model = Scaled(Unet)(28, 1, 1, ch=64, ch_mult=(1, 1, 2), attn_resolutions=(14,))

# CIFAR-10 (32×32, RGB)
model = Scaled(Unet)(32, 3, 3, ch=128, ch_mult=(1, 2, 2, 2), attn_resolutions=(16,))

# 采样
schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
*xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6, batchsize=64)
```

### DiT vs U-Net 架构对比

| 方面 | DiT | U-Net |
|------|-----|-------|
| 核心操作 | 全局自注意力 | 局部卷积 + 选择性注意力 |
| 多尺度处理 | Patch 化（单一分辨率） | 编码器-解码器（多分辨率） |
| 条件注入 | adaLN（调制归一化） | 加法注入到 ResnetBlock |
| Skip Connection | 无 | 编码器→解码器对称连接 |
| 位置编码 | 2D 正弦余弦 | 隐式（卷积的平移等变性） |
| 计算复杂度 | O(N²) 注意力 | O(N) 卷积为主 |
| 适合场景 | 中小分辨率、需要全局一致性 | 各种分辨率、需要局部细节 |
