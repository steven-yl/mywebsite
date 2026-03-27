---
title: "Analytic Diffusion Studio — 基线 UNet 模型"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 基线 UNet 模型"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 11
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 基线 UNet 模型"
---

# 11 — 基线 UNet 模型

文件：`src/local_diffusion/models/baseline_unet.py`

## 11.1 概述

BaselineUNet 是一个预训练的 UNet 扩散模型，用作对比基线。它不是本项目的核心方法，而是用来评估解析方法与神经网络方法之间的差距。

**用途**：
- 计算解析方法与 UNet 预测的 R² 和 MSE
- 生成对比可视化（同一噪声输入下两种方法的输出对比）
- 提供"训练过的神经网络"这一上界参考

## 11.2 UNet 架构配置

根据数据集分辨率自动选择架构：

```python
def _get_unet_config(self, dataset_name, img_size, in_channels, out_channels):
    if img_size == 28:      # MNIST, Fashion-MNIST
        channel = 64
        channel_mult = [1, 2, 2]       # 28→14→7→3
    elif img_size == 32:    # CIFAR-10
        channel = 128
        channel_mult = [1, 2, 3, 4]    # 32→16→8→4→2
    elif img_size == 64:    # CelebA-HQ, AFHQ
        channel = 128
        channel_mult = [1, 2, 3, 4]    # 64→32→16→8→4

    return {
        "T": 1000,              # 时间嵌入维度
        "channel": channel,      # 基础通道数
        "channel_mult": channel_mult,
        "attn": [],              # 注意力层位置（空=无注意力）
        "num_res_blocks": 2,     # 每级残差块数
        "dropout": 0.15,
    }
```

| 分辨率 | 基础通道 | 通道倍率 | 下采样级数 |
|--------|---------|---------|-----------|
| 28 | 64 | [1, 2, 2] | 3 |
| 32 | 128 | [1, 2, 3, 4] | 4 |
| 64 | 128 | [1, 2, 3, 4] | 4 |

## 11.3 类定义

```python
@register_model("baseline_unet")
class BaselineUNet(BaseDenoiser):
    def __init__(self, resolution, device, num_steps, model_path,
                 dataset_name="cifar10", in_channels=3, out_channels=3, **kwargs):
```

### 构造函数参数

| 参数 | 说明 |
|------|------|
| `resolution` | 图像分辨率 |
| `device` | 计算设备 |
| `num_steps` | DDIM 步数 |
| `model_path` | 预训练权重路径 |
| `dataset_name` | 数据集名（决定架构） |
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数 |

### 初始化流程

```python
# 1. 根据分辨率确定 UNet 配置
self.unet_config = self._get_unet_config(dataset_name, resolution, ...)

# 2. 实例化 UNet
self.model = UNet(
    T=1000, ch=config["channel"], ch_mult=config["channel_mult"],
    attn=[], num_res_blocks=2, dropout=0.15,
    in_channels=in_channels, out_channels=out_channels,
)

# 3. 加载预训练权重
self._load_weights()

# 4. 设为评估模式
self.model.eval()
```

## 11.4 _load_weights() 方法

```python
def _load_weights(self):
    checkpoint = torch.load(self.model_path, map_location=self.device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        self.model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        self.model.load_state_dict(checkpoint)
    else:
        self.model.load_state_dict(checkpoint.state_dict())
```

支持三种权重格式：
1. 包含 `model_state_dict` 键的字典（标准训练保存格式）
2. 直接的 state_dict 字典
3. 包含 `state_dict()` 方法的对象

权重通过 `download_baseline_weights.py` 从 HuggingFace 下载：

```python
# download_baseline_weights.py
repo_id = "ottogin/locality-diffusion-baselines"
# 下载 data/models/baseline_unet/{dataset}/ckpt_epoch_200.pt
```

## 11.5 denoise() 方法

```python
def denoise(self, latents, timestep, *, generator=None, **kwargs):
    B = latents.shape[0]

    # 1. 准备时间步张量
    ts = timestep.expand(B).long().to(self.device)  # [B]

    # 2. UNet 前向传播（预测噪声 ε）
    eps = self.model(latents, ts)  # [B, C, H, W]

    # 3. 从预测噪声反推 x̂₀
    alpha_prod_t = self.scheduler.alphas_cumprod[ts.cpu().long()]
    alpha_prod_t = alpha_prod_t.to(self.device).reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_x0 = (latents - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
    return pred_x0
```

### 步骤详解

1. UNet 接收噪声图像 $x_t$ 和时间步 $t$，预测噪声 $\hat{\epsilon}$
2. 利用前向过程公式反推干净图像：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

## 11.6 train() 方法

```python
def train(self, dataset):
    pass  # 预训练模型，无需额外操作
```

## 11.7 在 generate.py 中的使用

BaselineUNet 不作为主模型使用，而是在 `evaluate_comparison()` 中作为对比基线：

```python
# generate.py 中的调用
if baseline_path:
    baseline_model = BaselineUNet(
        resolution=dataset.resolution,
        device=cfg.experiment.device,
        num_steps=cfg.sampling.num_inference_steps,
        model_path=baseline_path,
        dataset_name=cfg.dataset.name,
        in_channels=dataset.in_channels,
        out_channels=dataset.in_channels,
    )

    baseline_result = baseline_model.sample(
        num_samples=cfg.sampling.num_samples,
        batch_size=cfg.sampling.batch_size,
        generator=baseline_gen,  # 相同种子
        return_intermediates=True,
    )
```

对比方式：
1. **轨迹对比**：同一时间步下，解析方法和 UNet 的 $\hat{x}_0$ 预测
2. **单步对比**：将解析方法的 $x_t$ 送入 UNet，比较两者的 $\hat{x}_0$

## 11.8 预训练权重

通过 `download_baseline_weights.py` 下载：

```bash
uv run download_baseline_weights.py
```

下载路径：`data/models/baseline_unet/{dataset}/ckpt_epoch_200.pt`

支持的数据集：mnist, fashion_mnist, cifar10, celeba_hq, afhq
