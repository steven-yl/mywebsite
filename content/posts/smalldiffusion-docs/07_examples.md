---
title: "smalldiffusion 实战示例"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 实战示例"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
---

> 本章解读项目提供的所有示例，从 2D 玩具模型到 Stable Diffusion 级别的预训练模型。

## 7.1 示例总览

| 示例 | 数据 | 模型 | 调度 | 条件 | 运行方式 |
|------|------|------|------|------|----------|
| toyexample.ipynb | Swissroll 2D | TimeInputMLP | LogLinear | 无 | Jupyter |
| cond_tree_model.ipynb | TreeDataset 2D | ConditionalMLP | LogLinear | 类别标签 + CFG | Jupyter |
| fashion_mnist_dit.py | FashionMNIST 28×28 | DiT | DDPM | 无 | accelerate launch |
| fashion_mnist_dit_cond.py | FashionMNIST 28×28 | DiT + CondEmbedder | DDPM | 类别标签 + CFG | accelerate launch |
| fashion_mnist_unet.py | FashionMNIST 28×28 | Scaled(Unet) | LogLinear | 无 | accelerate launch |
| cifar_unet.py | CIFAR-10 32×32 | Scaled(Unet) | Sigmoid(训练)/LogLinear(采样) | 无 | accelerate launch |
| diffusers_wrapper.py | - | ModelLatentDiffusion | LDM | 文本 | Python 模块 |
| stablediffusion.py | - | ModelLatentDiffusion | LDM | 文本 | python |

---

## 7.2 玩具模型示例 (toyexample.ipynb)

### 最小可运行代码

```python
import numpy as np
from torch.utils.data import DataLoader
from smalldiffusion import Swissroll, TimeInputMLP, ScheduleLogLinear, training_loop, samples

# 1. 数据
dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
loader   = DataLoader(dataset, batch_size=2048)

# 2. 模型
model    = TimeInputMLP(hidden_dims=(16,128,128,128,128,16))

# 3. 调度
schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)

# 4. 训练
trainer  = training_loop(loader, model, schedule, epochs=15000)
losses   = [ns.loss.item() for ns in trainer]

# 5. 采样
*xt, x0  = samples(model, schedule.sample_sigmas(20), gam=2)
# x0.shape: (1, 2) — 一个生成的 2D 点
# xt: 20 个中间步的结果列表
```

### 关键参数选择

- `batch_size=2048`：数据集只有 100 个点，大 batch 确保每次迭代覆盖全部数据
- `N=200`：训练调度步数，玩具模型不需要太多
- `sigma_min=0.005, sigma_max=10`：噪声范围，最小值接近 0（几乎无噪声），最大值远大于数据范围
- `epochs=15000`：MLP 模型收敛较慢，需要较多轮次
- `gam=2`：使用加速采样，仅需 20 步即可生成高质量样本

---

## 7.3 条件树模型示例 (cond_tree_model.ipynb)

### 核心代码

```python
import torch
import numpy as np
from torch.utils.data import DataLoader
from smalldiffusion import (
    TreeDataset, ConditionalMLP, ScheduleLogLinear,
    training_loop, samples
)

# 数据：4 分支、3 层深度的树
dataset = TreeDataset(branching_factor=4, depth=3)
loader = DataLoader(dataset, batch_size=2048, shuffle=True)

# 条件模型
model = ConditionalMLP(
    dim=2,
    hidden_dims=(16, 128, 256, 128, 16),
    cond_dim=4,
    num_classes=dataset.total_leaves,  # 64 类
    dropout_prob=0.1
)

# 训练（conditional=True）
schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10)
trainer = training_loop(loader, model, schedule, epochs=15000, conditional=True)
losses = [ns.loss.item() for ns in trainer]

# 条件采样 + CFG
N_sample = 64
cond = torch.arange(dataset.total_leaves)  # 每个类别各生成一个样本
*xt, x0 = samples(model, schedule.sample_sigmas(20),
                   gam=2, batchsize=N_sample,
                   cond=cond, cfg_scale=4.0)
```

### 关键点

- `conditional=True` 使 `training_loop` 将 DataLoader 的输出解包为 `(data, labels)`
- `dropout_prob=0.1` 使 10% 的训练样本以无条件方式训练
- `cfg_scale=4.0` 在采样时增强条件引导

---

## 7.4 FashionMNIST DiT 示例

### 无条件版本 (fashion_mnist_dit.py)

```python
from smalldiffusion import (
    ScheduleDDPM, samples, training_loop, MappedDataset, DiT,
    img_train_transform, img_normalize
)
from torch_ema import ExponentialMovingAverage as EMA

# 数据：丢弃标签
dataset = MappedDataset(
    FashionMNIST('datasets', train=True, download=True, transform=img_train_transform),
    lambda x: x[0]
)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# 模型
schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
model = DiT(in_dim=28, channels=1, patch_size=2, depth=6,
            head_dim=32, num_heads=6, mlp_ratio=4.0)

# 训练 + EMA
ema = EMA(model.parameters(), decay=0.99)
ema.to(accelerator.device)
for ns in training_loop(loader, model, schedule, epochs=300, lr=1e-3, accelerator=a):
    ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
    ema.update()

# 采样（使用 EMA 参数）
with ema.average_parameters():
    *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6, batchsize=64)
    save_image(img_normalize(make_grid(x0)), 'samples.png')
```

### 关键技术点

- **EMA (Exponential Moving Average)**：维护模型参数的指数移动平均，采样时使用 EMA 参数，生成质量更稳定
- **MappedDataset**：丢弃 FashionMNIST 的标签
- **img_train_transform**：随机翻转 + 归一化到 [-1, 1]
- **gam=1.6**：介于 DDIM (1.0) 和加速采样 (2.0) 之间的折中值
- **运行方式**：`accelerate launch examples/fashion_mnist_dit.py`（需先运行 `accelerate config`）

### 条件版本 (fashion_mnist_dit_cond.py)

与无条件版本的区别：

```python
# 不丢弃标签
dataset = FashionMNIST('datasets', train=True, download=True, transform=img_train_transform)

# 添加条件嵌入
model = DiT(in_dim=28, channels=1, patch_size=2, depth=6,
            head_dim=32, num_heads=6, mlp_ratio=4.0,
            cond_embed=CondEmbedderLabel(32*6, 10, 0.1))  # 10 类，10% dropout

# 条件训练
for ns in training_loop(loader, model, schedule, epochs=300, conditional=True, ...):
    ...

# 条件采样
cond = torch.tensor([i % 10 for i in range(40)])  # 每类 4 个样本
*xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6, batchsize=40, cond=cond)
```

---

## 7.5 FashionMNIST U-Net 示例

```python
from smalldiffusion import ScheduleLogLinear, Unet, Scaled

# 使用 LogLinear 调度 + Scaled U-Net
schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
model = Scaled(Unet)(28, 1, 1, ch=64, ch_mult=(1, 1, 2), attn_resolutions=(14,))
```

### 与 DiT 版本的区别

| 方面 | DiT 版本 | U-Net 版本 |
|------|---------|-----------|
| 调度 | ScheduleDDPM | ScheduleLogLinear |
| 模型 | DiT | Scaled(Unet) |
| 输入缩放 | 无 | Scaled 修饰器 |
| 注意力分辨率 | 全局（所有 patch） | 14×14 |
| 基础通道数 | 192 (32×6) | 64 |

---

## 7.6 CIFAR-10 U-Net 示例

```python
from smalldiffusion import Unet, Scaled, ScheduleLogLinear, ScheduleSigmoid

# 训练和采样使用不同调度
train_schedule = ScheduleSigmoid(N=1000)
model = Scaled(Unet)(32, 3, 3, ch=128, ch_mult=(1, 2, 2, 2), attn_resolutions=(16,))

# 训练...

# 采样使用不同调度
sample_schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=35, N=1000)
*xt, x0 = samples(model, sample_schedule.sample_sigmas(10), gam=2.1, batchsize=64)
```

### 关键设计

- **训练/采样分离调度**：训练用 `ScheduleSigmoid`，采样用 `ScheduleLogLinear`。这是因为训练调度决定模型学习的噪声分布，而采样调度决定去噪路径，两者可以独立优化
- **更大模型**：`ch=128`（FashionMNIST 用 64），因为 CIFAR-10 是 RGB 且更复杂
- **gam=2.1**：略大于 2 的加速采样参数
- **仅 10 步采样**：加速采样允许极少步数生成高质量图像
- **FID ~3-4**：在 CIFAR-10 无条件生成上达到竞争力的结果

---

## 7.7 Stable Diffusion 包装器 (diffusers_wrapper.py)

### 是什么

将 HuggingFace Diffusers 的预训练潜空间扩散模型包装为 smalldiffusion 兼容的模型接口。

### ModelLatentDiffusion 类

```python
class ModelLatentDiffusion(nn.Module, ModelMixin):
    def __init__(self, model_key, accelerator=None):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.input_dims = (self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
```

### 关键方法

**set_text_condition(prompt, negative_prompt, text_guidance_scale)**

将文本提示编码为 CLIP 嵌入，存储条件和无条件嵌入：

```python
def set_text_condition(self, prompt, negative_prompt='', text_guidance_scale=7.5):
    with torch.no_grad():
        prompt_emb = self.text_encoder(self.tokenize(prompt))[0]
        uncond_emb = self.text_encoder(self.tokenize(negative_prompt))[0]
    self.text_condition = torch.cat([uncond_emb, prompt_emb])
    self.text_guidance_scale = text_guidance_scale
```

**sigma_to_t(sigma)**

将 smalldiffusion 的 $\sigma$ 参数化转换为 Diffusers 的时间步 $t$：

```python
def sigma_to_t(self, sigma):
    idx = torch.searchsorted(reversed(self.scheduler.alphas_cumprod.to(sigma)), alpha_bar(sigma))
    return self.scheduler.num_train_timesteps - 1 - idx
```

**forward(x, sigma, cond)**

在潜空间中执行去噪，内置文本引导：

```python
def forward(self, x, sigma, cond=None):
    z = alpha_bar(sigma).sqrt() * x                    # 缩放到 Diffusers 的参数化
    z2 = torch.cat([z, z])                              # 条件 + 无条件
    eps = self.unet(z2, self.sigma_to_t(sigma),
                    encoder_hidden_states=self.text_condition).sample
    eps_uncond, eps_prompt = eps.chunk(2)
    return eps_prompt + self.text_guidance_scale * (eps_prompt - eps_uncond)
```

**decode_latents(latents)**

将潜空间表示解码为像素图像：

```python
def decode_latents(self, latents):
    return self.vae.decode(latents / 0.18215).sample
```

`0.18215` 是 Stable Diffusion VAE 的缩放因子。

---

## 7.8 Stable Diffusion 采样示例

```python
from diffusers_wrapper import ModelLatentDiffusion
from smalldiffusion import ScheduleLDM, samples
from torchvision.utils import save_image

schedule = ScheduleLDM(1000)
model = ModelLatentDiffusion('CompVis/stable-diffusion-v1-4')
model.set_text_condition('An astronaut riding a horse')
*xts, x0 = samples(model, schedule.sample_sigmas(50))
decoded = model.decode_latents(x0)
save_image(((decoded.squeeze()+1)/2).clamp(0,1), 'output.png')
```

### 这个示例展示的核心价值

smalldiffusion 的采样器是通用的——同一个 `samples` 函数既能用于 2D 玩具模型，也能用于 Stable Diffusion 级别的预训练模型。只需要模型实现 `ModelMixin` 协议（`input_dims`, `rand_input`, `predict_eps`）。

### 实验不同采样参数

```python
# 更少步数 + 加速采样
*xts, x0 = samples(model, schedule.sample_sigmas(20), gam=2)

# DDIM 采样
*xts, x0 = samples(model, schedule.sample_sigmas(50), gam=1, mu=0)

# DDPM 采样
*xts, x0 = samples(model, schedule.sample_sigmas(100), gam=1, mu=0.5)
```

---

## 7.9 运行示例的前置条件

### 玩具模型

```bash
pip install smalldiffusion
# 或本地开发
uv sync --extra dev --extra test --extra examples
```

### 图像模型（需要 GPU）

```bash
pip install smalldiffusion torch_ema accelerate
accelerate config  # 配置 GPU
accelerate launch examples/fashion_mnist_dit.py
```

### Stable Diffusion

```bash
pip install smalldiffusion diffusers transformers
# 需要 HuggingFace 账号和模型访问权限
python examples/stablediffusion.py
```
