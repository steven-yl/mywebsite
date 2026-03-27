---
title: "smalldiffusion 模型基础：model.py"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 模型基础：model.py"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 4
hiddenFromHomePage: false
hiddenFromSearch: false
---

> 本文件定义了所有模型共享的基类、预测模式修饰器、通用组件（注意力、嵌入）、玩具模型和理想去噪器。

## 4.1 模块结构

```
model.py
├── ModelMixin                # 模型基类 Mixin
├── get_sigma_embeds()        # σ 嵌入函数
├── SigmaEmbedderSinCos      # σ 嵌入模块
├── alpha()                   # σ → α̅ 转换
├── Scaled()                  # 输入缩放修饰器
├── PredX0()                  # 预测 x0 修饰器
├── PredV()                   # 预测 v 修饰器
├── CondSequential            # 条件顺序容器
├── Attention                 # 多头注意力
├── CondEmbedderLabel         # 标签条件嵌入
├── TimeInputMLP              # 玩具 MLP 模型
├── ConditionalMLP            # 条件 MLP 模型
├── sq_norm()                 # 辅助函数
└── IdealDenoiser             # 理想去噪器
```

---

## 4.2 ModelMixin

### 是什么

所有扩散模型的 Mixin 基类，提供三个关键方法。

### 模型协议

smalldiffusion 中的模型必须满足以下协议：

1. 继承 `torch.nn.Module` 和 `ModelMixin`
2. 设置 `input_dims` 属性（不含 batch 维度的输入形状元组）
3. 实现 `forward(self, x, sigma, cond=None)` 方法，返回与 `x` 同形状的预测噪声

```python
class ModelMixin:
    def rand_input(self, batchsize):
        """生成标准正态随机输入，形状为 [batchsize, *input_dims]"""
        assert hasattr(self, 'input_dims'), 'Model must have "input_dims" attribute!'
        return torch.randn((batchsize,) + self.input_dims)

    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        """计算训练损失：预测噪声与真实噪声的 MSE"""
        return loss()(eps, self(x0 + sigma * eps, sigma, cond=cond))

    def predict_eps(self, x, sigma, cond=None):
        """预测噪声 ε（默认直接调用 forward）"""
        return self(x, sigma, cond=cond)

    def predict_eps_cfg(self, x, sigma, cond, cfg_scale):
        """带 Classifier-Free Guidance 的噪声预测"""
        ...
```

### rand_input(batchsize)

生成采样初始噪声。形状由模型的 `input_dims` 决定：
- 2D 模型：`input_dims = (2,)` → 输出形状 `[B, 2]`
- 图像模型：`input_dims = (3, 32, 32)` → 输出形状 `[B, 3, 32, 32]`

### get_loss(x0, sigma, eps, cond)

默认实现假设模型预测噪声 $\varepsilon$：

$$\mathcal{L} = \text{MSE}(\varepsilon, f_\theta(x_0 + \sigma \cdot \varepsilon, \sigma))$$

其中 $x_0 + \sigma \cdot \varepsilon$ 是加噪后的样本。此方法可被 `PredX0` 和 `PredV` 修饰器覆盖。

### predict_eps_cfg(x, sigma, cond, cfg_scale)

实现 [Classifier-Free Guidance (CFG)](https://arxiv.org/abs/2207.12598)：

```python
def predict_eps_cfg(self, x, sigma, cond, cfg_scale):
    if cond is None or cfg_scale == 0:
        return self.predict_eps(x, sigma, cond=cond)
    assert sigma.shape == tuple(), 'CFG sampling only supports singleton sigma!'
    uncond = torch.full_like(cond, self.cond_embed.null_cond)
    eps_cond, eps_uncond = self.predict_eps(
        torch.cat([x, x]), sigma, torch.cat([cond, uncond])
    ).chunk(2)
    return eps_cond + cfg_scale * (eps_cond - eps_uncond)
```

**CFG 公式：**

$$\hat{\varepsilon} = \varepsilon_{\text{cond}} + s \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}})$$

其中 $s$ 是 `cfg_scale`。当 $s > 0$ 时，模型输出被推向条件方向，远离无条件方向。

**实现技巧：** 将条件和无条件输入拼接成一个 batch 一次前向传播，避免两次调用模型。

---

## 4.3 get_sigma_embeds 函数

### 是什么

将标量 $\sigma$ 值编码为 2 维嵌入向量的函数。

```python
def get_sigma_embeds(batches, sigma, scaling_factor=0.5, log_scale=True):
    if sigma.shape == torch.Size([]):
        sigma = sigma.unsqueeze(0).repeat(batches)
    else:
        assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
    if log_scale:
        sigma = torch.log(sigma)
    s = sigma.unsqueeze(1) * scaling_factor
    return torch.cat([torch.sin(s), torch.cos(s)], dim=1)
```

### 工作原理

1. **标量处理**：若 $\sigma$ 是标量，扩展为 batch 大小
2. **对数缩放**：默认取 $\log(\sigma)$，将指数级变化的 $\sigma$ 压缩到线性范围
3. **正弦/余弦编码**：$[\sin(s \cdot f), \cos(s \cdot f)]$，其中 $f$ 是缩放因子

输出形状：`[B, 2]`。这是一种极简的时间嵌入，仅用 2 维就能有效编码噪声水平。

### 与标准 Sinusoidal Embedding 的区别

标准 Transformer 位置编码使用多个频率，输出维度通常为 128 或 256。smalldiffusion 的实现仅用 1 个频率（2 维），但论文表明在扩散模型中效果相当。

---

## 4.4 SigmaEmbedderSinCos

### 是什么

将 `get_sigma_embeds` 的 2 维输出通过 MLP 映射到高维空间的模块。

```python
class SigmaEmbedderSinCos(nn.Module):
    def __init__(self, hidden_size, scaling_factor=0.5, log_scale=True):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.log_scale = log_scale
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, batches, sigma):
        sig_embed = get_sigma_embeds(batches, sigma,
                                     self.scaling_factor, self.log_scale)  # (B, 2)
        return self.mlp(sig_embed)                                         # (B, D)
```

### 结构

```
σ → [sin, cos] (2维) → Linear(2, D) → SiLU → Linear(D, D) → 输出 (D维)
```

被 `DiT` 和 `Unet` 使用，将噪声水平信息注入模型。

---

## 4.5 alpha 函数

### 是什么

$\sigma$ 参数化到 $\bar{\alpha}$ 参数化的转换函数。

```python
def alpha(sigma):
    return 1 / (1 + sigma**2)
```

### 数学关系

$$\bar{\alpha} = \frac{1}{1 + \sigma^2}, \quad \sigma = \sqrt{\frac{1}{\bar{\alpha}} - 1}$$

被 `Scaled`、`PredV` 修饰器和 `diffusers_wrapper.py` 使用。

---

## 4.6 Scaled 修饰器

### 是什么

一个类修饰器（class decorator），对模型输入进行缩放，使不同噪声水平下输入的范数保持恒定。

```python
def Scaled(cls: ModelMixin):
    def forward(self, x, sigma, cond=None):
        return cls.forward(self, x * alpha(sigma).sqrt(), sigma, cond=cond)
    return type(cls.__name__ + 'Scaled', (cls,), dict(forward=forward))
```

### 数学原理

加噪样本 $x_t = x_0 + \sigma \varepsilon$ 的期望范数随 $\sigma$ 增大而增大。缩放因子 $\sqrt{\bar{\alpha}} = \frac{1}{\sqrt{1+\sigma^2}}$ 将输入归一化：

$$\tilde{x}_t = \sqrt{\bar{\alpha}} \cdot x_t$$

使得 $\mathbb{E}[\|\tilde{x}_t\|^2]$ 对所有 $\sigma$ 近似恒定。

### 使用方式

```python
from smalldiffusion import Scaled, Unet

# 创建带输入缩放的 U-Net
model = Scaled(Unet)(28, 1, 1, ch=64, ch_mult=(1, 1, 2))
# 等价于创建了一个名为 "UnetScaled" 的新类
```

### 实现细节

`Scaled` 使用 Python 的 `type()` 动态创建新类，继承原始类但覆盖 `forward` 方法。新类名为原类名 + "Scaled"。

---

## 4.7 PredX0 修饰器

### 是什么

将模型从预测噪声 $\varepsilon$ 改为预测干净数据 $x_0$ 的类修饰器。

```python
def PredX0(cls: ModelMixin):
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        return loss()(x0, self(x0 + sigma * eps, sigma, cond=cond))
    def predict_eps(self, x, sigma, cond=None):
        x0_hat = self(x, sigma, cond=cond)
        return (x - x0_hat) / sigma
    return type(cls.__name__ + 'PredX0', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))
```

### 数学原理

若模型预测 $\hat{x}_0$，可以反推噪声预测：

$$\hat{\varepsilon} = \frac{x_t - \hat{x}_0}{\sigma}$$

因为 $x_t = x_0 + \sigma \varepsilon$，所以 $\varepsilon = (x_t - x_0) / \sigma$。

### 覆盖的方法

- `get_loss`：损失变为 $\text{MSE}(x_0, f_\theta(x_t, \sigma))$
- `predict_eps`：从 $\hat{x}_0$ 反推 $\hat{\varepsilon}$，使采样代码无需修改

---

## 4.8 PredV 修饰器

### 是什么

将模型改为预测 velocity $v$ 的类修饰器，来自 [Progressive Distillation](https://arxiv.org/abs/2202.00512)。

```python
def PredV(cls: ModelMixin):
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        xt = x0 + sigma * eps
        v = alpha(sigma).sqrt() * eps - (1 - alpha(sigma)).sqrt() * x0
        return loss()(v, self(xt, sigma, cond=cond))
    def predict_eps(self, x, sigma, cond=None):
        v_hat = self(x, sigma, cond=cond)
        return alpha(sigma).sqrt() * (v_hat + (1 - alpha(sigma)).sqrt() * x)
    return type(cls.__name__ + 'PredV', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))
```

### 数学原理

Velocity 定义为：

$$v = \sqrt{\bar{\alpha}} \cdot \varepsilon - \sqrt{1 - \bar{\alpha}} \cdot x_0$$

从 $\hat{v}$ 反推噪声：

$$\hat{\varepsilon} = \sqrt{\bar{\alpha}} \cdot (\hat{v} + \sqrt{1 - \bar{\alpha}} \cdot x_t)$$

### 为什么使用 v-prediction

在高噪声水平下，预测 $\varepsilon$ 的信噪比很低；在低噪声水平下，预测 $x_0$ 的信噪比很低。v-prediction 在两种极端情况下都有更均衡的信噪比。

### 修饰器组合

修饰器可以组合使用：

```python
from smalldiffusion import Scaled, PredX0, PredV, DiT

# 带输入缩放 + 预测 x0
model = Scaled(PredX0(DiT))(in_dim=16, channels=3, patch_size=2, depth=4)

# 带输入缩放 + 预测 v
model = Scaled(PredV(DiT))(in_dim=16, channels=3, patch_size=2, depth=4)
```

---

## 4.9 CondSequential

### 是什么

支持条件输入的 `nn.Sequential` 变体。

```python
class CondSequential(nn.Sequential):
    def forward(self, x, cond):
        for module in self._modules.values():
            x = module(x, cond)
        return x
```

### 为什么需要

标准 `nn.Sequential` 只支持单输入。扩散模型的中间层需要同时接收特征 `x` 和条件信息 `cond`（如时间嵌入）。`CondSequential` 将 `(x, cond)` 传递给每个子模块。

被 `DiT` 的 Transformer Block 序列和 `Unet` 的中间层使用。

---

## 4.10 Attention

### 是什么

标准多头自注意力模块。

```python
class Attention(nn.Module):
    def __init__(self, head_dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        dim = head_dim * num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, D) → (B, N, D)
        q, k, v = rearrange(self.qkv(x), 'b n (qkv h k) -> qkv b h n k',
                            h=self.num_heads, k=self.head_dim)
        x = rearrange(F.scaled_dot_product_attention(q, k, v),
                      'b h n k -> b n (h k)')
        return self.proj(x)
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `head_dim` | 每个注意力头的维度 |
| `num_heads` | 注意力头数量 |
| `qkv_bias` | QKV 投影是否使用偏置 |

### 计算流程

1. 线性投影生成 Q, K, V：`(B, N, D) → (B, N, 3D) → 3 × (B, H, N, d)`
2. 缩放点积注意力：`F.scaled_dot_product_attention(q, k, v)`（PyTorch 原生实现，自动选择 Flash Attention 等优化）
3. 拼接多头并投影：`(B, H, N, d) → (B, N, D)`

被 `DiT` 和 `Unet` 的 `AttnBlock` 共同使用。

---

## 4.11 CondEmbedderLabel

### 是什么

将离散类别标签嵌入为连续向量的模块，支持训练时的随机 dropout（用于 Classifier-Free Guidance）。

```python
class CondEmbedderLabel(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_prob=0.1):
        super().__init__()
        self.embeddings = nn.Embedding(num_classes + 1, hidden_size)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.null_cond = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels):  # (B,) → (B, D)
        if self.training:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.null_cond, labels)
        return self.mlp(self.embeddings(labels))
```

### 设计细节

- **嵌入表大小**：`num_classes + 1`，额外的一个位置用于"无条件"标签
- **null_cond**：值为 `num_classes`，表示无条件输入
- **训练时 dropout**：以 `dropout_prob` 概率将标签替换为 `null_cond`，使模型同时学习条件和无条件生成
- **MLP**：`SiLU → Linear`，将嵌入映射到模型隐藏维度

### 为什么需要 dropout

CFG 采样需要模型同时能做条件和无条件预测。训练时随机丢弃条件信息，使模型学会在无条件下也能生成合理输出。

---

## 4.12 TimeInputMLP

### 是什么

用于 2D 玩具数据的简单 MLP 模型。

```python
class TimeInputMLP(nn.Module, ModelMixin):
    sigma_dim = 2
    def __init__(self, dim=2, output_dim=None, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + self.sigma_dim,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim or dim))
        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def forward(self, x, sigma, cond=None):
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze())  # (B, 2)
        nn_input = torch.cat([x, sigma_embeds], dim=1)                 # (B, dim+2)
        return self.net(nn_input)
```

### 网络结构

默认配置 `hidden_dims=(16,128,256,128,16)`：

```
输入 (dim+2) → Linear → GELU → Linear → GELU → ... → Linear → 输出 (dim)
     4          16       128      256      128     16      2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dim` | 2 | 数据维度 |
| `output_dim` | None | 输出维度（默认等于 dim） |
| `hidden_dims` | (16,128,256,128,16) | 隐藏层维度序列 |

### 使用示例

```python
from smalldiffusion import TimeInputMLP
import torch

model = TimeInputMLP(dim=2, hidden_dims=(16, 128, 128, 16))
x = torch.randn(32, 2)       # batch of 2D points
sigma = torch.tensor(1.0)     # noise level
output = model(x, sigma)      # (32, 2)
```

---

## 4.13 ConditionalMLP

### 是什么

`TimeInputMLP` 的条件版本，支持类别标签条件生成。

```python
class ConditionalMLP(TimeInputMLP):
    def __init__(self, dim=2, hidden_dims=(16,128,256,128,16),
                 cond_dim=4, num_classes=10, dropout_prob=0.1):
        super().__init__(dim=dim+cond_dim, output_dim=dim, hidden_dims=hidden_dims)
        self.input_dims = (dim,)  # 覆盖父类设置
        self.cond_embed = CondEmbedderLabel(cond_dim, num_classes, dropout_prob)

    def forward(self, x, sigma, cond):
        cond_embeds = self.cond_embed(cond)                           # (B, cond_dim)
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze())  # (B, sigma_dim)
        nn_input = torch.cat([x, sigma_embeds, cond_embeds], dim=1)   # (B, dim+sigma_dim+cond_dim)
        return self.net(nn_input)
```

### 设计细节

- 继承 `TimeInputMLP`，但将输入维度扩展为 `dim + cond_dim`
- `input_dims` 仍设为 `(dim,)`，因为 `rand_input` 只需生成数据维度的噪声
- 条件嵌入通过 `CondEmbedderLabel` 将标签映射为 `cond_dim` 维向量
- 三部分拼接后送入 MLP：`[x, sigma_embed, cond_embed]`

### 使用示例

```python
from smalldiffusion import ConditionalMLP
import torch

model = ConditionalMLP(dim=2, cond_dim=4, num_classes=10)
x = torch.randn(32, 2)
sigma = torch.tensor(1.0)
cond = torch.randint(0, 10, (32,))
output = model(x, sigma, cond)  # (32, 2)
```

---

## 4.14 sq_norm 辅助函数

### 是什么

计算矩阵每行的平方范数并重复 k 次的辅助函数，被 `IdealDenoiser` 使用。

```python
def sq_norm(M, k):
    # M: (b, n) → (b,) → (b, k)
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1, k)
```

用于高效计算成对平方距离矩阵 $\|x_i - d_j\|^2$。

---

## 4.15 IdealDenoiser

### 是什么

给定数据集的理论最优去噪器（Bayes 最优估计器），用于验证和基准测试。

```python
class IdealDenoiser(nn.Module, ModelMixin):
    def __init__(self, dataset):
        super().__init__()
        self.data = torch.stack([dataset[i] for i in range(len(dataset))])
        self.input_dims = self.data.shape[1:]

    def forward(self, x, sigma, cond=None):
        data = self.data.to(x)
        x_flat = x.flatten(start_dim=1)
        d_flat = data.flatten(start_dim=1)
        xb, xr = x_flat.shape
        db, dr = d_flat.shape
        # 计算成对平方距离
        sq_diffs = sq_norm(x_flat, db).T + sq_norm(d_flat, xb) - 2 * d_flat @ x_flat.T
        # Softmax 权重
        weights = F.softmax(-sq_diffs / 2 / sigma.squeeze()**2, dim=0)
        # 加权平均
        eps = torch.einsum('ij,i...->j...', weights, data)
        return (x - eps) / sigma
```

### 数学原理

对于高斯噪声模型 $x_t = x_0 + \sigma \varepsilon$，Bayes 最优去噪器为：

$$\hat{x}_0(x_t) = \mathbb{E}[x_0 | x_t] = \frac{\sum_{i} x_0^{(i)} \exp\left(-\frac{\|x_t - x_0^{(i)}\|^2}{2\sigma^2}\right)}{\sum_{i} \exp\left(-\frac{\|x_t - x_0^{(i)}\|^2}{2\sigma^2}\right)}$$

即数据集中所有点的 softmax 加权平均，权重与距离的负指数成正比。

然后转换为噪声预测：$\hat{\varepsilon} = (x_t - \hat{x}_0) / \sigma$

### 计算优化

使用展开的平方距离公式避免显式计算差值矩阵：

$$\|x_i - d_j\|^2 = \|x_i\|^2 + \|d_j\|^2 - 2 x_i \cdot d_j$$

### 适用场景

- 验证采样算法的正确性
- 作为训练模型的性能上界
- 不需要训练，直接从数据集构造

### 使用示例

```python
from smalldiffusion import IdealDenoiser, Swissroll, samples, ScheduleLogLinear
import numpy as np

dataset = Swissroll(np.pi/2, 5*np.pi, 100)
model = IdealDenoiser(dataset)
schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
*xt, x0 = samples(model, schedule.sample_sigmas(20), gam=2)
```
