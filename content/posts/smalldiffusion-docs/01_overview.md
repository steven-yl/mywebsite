---
title: "smalldiffusion 项目总览"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 项目总览"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
---
## 1.1 项目定位

smalldiffusion 是一个教学与实验导向的扩散模型库，核心训练和采样代码不到 100 行。它的设计目标是：

- 提供可读、可理解的扩散模型实现
- 支持从 2D 玩具数据到 Stable Diffusion 级别的预训练模型
- 方便研究者快速实验新的采样算法和模型架构

论文参考：[Permenter and Yuan, arXiv:2306.04848](https://arxiv.org/abs/2306.04848)

## 1.2 扩散模型数学基础

### 前向过程（加噪）

扩散模型的核心思想是：将数据逐步加噪直到变成纯噪声，然后学习逆过程。

给定干净数据 $x_0$，前向过程生成带噪样本：

$$x_t = x_0 + \sigma \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

其中 $\sigma$ 是噪声水平。smalldiffusion 使用 **$\sigma$ 参数化**（而非常见的 $\alpha$/$\bar{\alpha}$ 参数化），两者的关系为：

$$\sigma = \sqrt{\frac{1}{\bar{\alpha}} - 1}, \quad \bar{\alpha} = \frac{1}{1 + \sigma^2}$$

代码中的 `alpha(sigma)` 函数即计算 $\bar{\alpha}$：

```python
def alpha(sigma):
    return 1 / (1 + sigma**2)
```

### 反向过程（去噪/采样）

模型学习预测噪声 $\varepsilon_\theta(x_t, \sigma)$，采样时从纯噪声 $x_T \sim \mathcal{N}(0, \sigma_T^2 I)$ 出发，逐步去噪。smalldiffusion 的采样公式（5 行代码）统一了 DDPM、DDIM 和加速采样：

$$x_{t-1} = x_t - (\sigma_t - \sigma_p) \cdot \bar{\varepsilon} + \eta \cdot z$$

其中：
- $\bar{\varepsilon}$ 是当前和上一步噪声预测的加权平均（由 `gam` 控制）
- $\sigma_p$ 和 $\eta$ 由参数 `mu` 控制确定性/随机性比例
- $z \sim \mathcal{N}(0, I)$ 是随机噪声

## 1.3 架构设计

项目由三个核心概念组成：**数据（Data）**、**模型（Model）**、**调度（Schedule）**，它们通过 `training_loop` 和 `samples` 两个函数协作。

```
┌─────────────────────────────────────────────────────┐
│                    diffusion.py                      │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ training_loop│  │   samples    │                 │
│  └──────┬───────┘  └──────┬───────┘                 │
│         │                 │                          │
│  ┌──────┴─────────────────┴──────┐                  │
│  │         Schedule 系列          │                  │
│  │ LogLinear│DDPM│LDM│Sigmoid│Cos│                  │
│  └───────────────────────────────┘                  │
└─────────────────────────────────────────────────────┘
         │                 │
    ┌────┴────┐      ┌────┴────┐
    │ data.py │      │model.py │
    │Swissroll│      │ModelMixin│
    │Datasaur.│      │Scaled   │
    │TreeData │      │PredX0   │
    │MappedDS │      │PredV    │
    └─────────┘      │TimeInput│
                     │CondMLP  │
                     │IdealDen.│
                     ├─────────┤
                     │model_dit│
                     │  DiT    │
                     ├─────────┤
                     │model_unet│
                     │  Unet   │
                     └─────────┘
```

### 模块职责

| 模块 | 职责 | 关键导出 |
|------|------|----------|
| `diffusion.py` | 噪声调度、训练循环、采样算法 | `Schedule`, `ScheduleLogLinear`, `ScheduleDDPM`, `ScheduleLDM`, `ScheduleSigmoid`, `ScheduleCosine`, `training_loop`, `samples` |
| `data.py` | 数据集定义与预处理工具 | `Swissroll`, `DatasaurusDozen`, `TreeDataset`, `MappedDataset`, `img_train_transform`, `img_normalize` |
| `model.py` | 模型基类、修饰器、基础组件 | `ModelMixin`, `Scaled`, `PredX0`, `PredV`, `TimeInputMLP`, `ConditionalMLP`, `IdealDenoiser`, `Attention`, `SigmaEmbedderSinCos`, `CondEmbedderLabel`, `CondSequential` |
| `model_dit.py` | Diffusion Transformer 实现 | `DiT` |
| `model_unet.py` | U-Net 实现 | `Unet` |

### 数据流

**训练阶段：**
1. `DataLoader` 产出一批数据 $x_0$
2. `Schedule.sample_batch()` 随机采样噪声水平 $\sigma$
3. `generate_train_sample()` 生成 $(x_0, \sigma, \varepsilon)$ 三元组
4. 模型前向传播预测噪声，计算 MSE 损失
5. 反向传播更新参数

**采样阶段：**
1. `Schedule.sample_sigmas(steps)` 生成递减的 $\sigma$ 序列
2. 从 $x_T \sim \mathcal{N}(0, \sigma_0^2 I)$ 开始
3. 每步调用模型预测噪声，按采样公式更新 $x_t$
4. 最终得到生成样本 $x_0$

## 1.4 模型架构对比

| 特性 | TimeInputMLP | DiT | Unet |
|------|-------------|-----|------|
| 适用数据 | 2D 玩具数据 | 图像 | 图像 |
| 参数量级 | ~10K | ~10M | ~10M |
| 时间嵌入 | sin/cos (2维) | SigmaEmbedderSinCos + MLP | SigmaEmbedderSinCos + MLP |
| 核心结构 | 全连接层 + GELU | Transformer Block + Modulation | ResNet Block + Attention + Skip Connection |
| 条件生成 | ConditionalMLP 变体 | 通过 `cond_embed` 参数 | 通过 `cond_embed` 参数 |
| 输入缩放 | 可选 (Scaled) | 可选 (Scaled) | 通常使用 Scaled |
| 训练速度 | 快（秒级） | 中等（小时级） | 中等（小时级） |
| 生成质量 | 仅适合简单分布 | FashionMNIST FID ~5-6 | CIFAR-10 FID ~3-4 |

## 1.5 Schedule 对比

| Schedule | 公式特点 | 典型用途 | 默认参数 |
|----------|---------|---------|---------|
| `ScheduleLogLinear` | $\sigma$ 在对数空间线性增长 | 玩具模型、小数据集 | N=200, σ_min=0.02, σ_max=10 |
| `ScheduleDDPM` | 线性 $\beta$ 调度 | 像素空间图像扩散 | N=1000, β_start=0.0001, β_end=0.02 |
| `ScheduleLDM` | 缩放线性 $\beta$ 调度 | 潜空间扩散 (Stable Diffusion) | N=1000, β_start=0.00085, β_end=0.012 |
| `ScheduleSigmoid` | Sigmoid 形状的 $\beta$ 调度 | 分子构象生成 (GeoDiff) | N=1000, β_start=0.0001, β_end=0.02 |
| `ScheduleCosine` | 余弦 $\bar{\alpha}$ 调度 | 改进的 DDPM (iDDPM) | N=1000, max_beta=0.999 |

## 1.6 采样算法对比

| 算法 | `gam` | `mu` | 特点 |
|------|-------|------|------|
| DDPM | 1 | 0.5 | 随机采样，需要较多步数 |
| DDIM | 1 | 0 | 确定性采样，可用较少步数 |
| 加速采样 | 2 | 0 | 利用历史噪声预测加速收敛 |

## 1.7 依赖关系

```toml
dependencies = [
  "accelerate",   # 多 GPU 训练支持
  "numpy",        # 数值计算
  "torchvision",  # 图像变换、数据集
  "torch",        # 深度学习框架
  "tqdm",         # 进度条
  "einops",       # 张量重排
]
```
