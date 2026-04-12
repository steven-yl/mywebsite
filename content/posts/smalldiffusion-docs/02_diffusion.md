---
title: "smalldiffusion 核心模块：diffusion.py"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "smalldiffusion 核心模块：diffusion.py"
tags: [diffusion/flow, smalldiffusion]
categories: [diffusion/flow, smalldiffusion]
series: [smalldiffusion系列]
weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false
---

> 本文件是 smalldiffusion 的核心，包含噪声调度（Schedule）、训练循环（training_loop）和采样算法（samples），总计不到 100 行代码。

## 2.1 模块结构

```
diffusion.py
├── Schedule (基类)
│   ├── ScheduleLogLinear
│   ├── ScheduleDDPM
│   ├── ScheduleLDM
│   ├── ScheduleSigmoid
│   └── ScheduleCosine
├── sigmas_from_betas()      # 辅助函数：β → σ 转换
├── generate_train_sample()  # 辅助函数：生成训练样本
├── training_loop()          # 训练循环生成器
└── samples()                # 采样生成器
```

---

## 2.2 Schedule 基类

### 是什么

`Schedule` 管理扩散过程中噪声水平 $\sigma$ 的递增序列。它是所有调度策略的基类。

### 为什么需要

扩散模型的训练和采样都依赖于一个预定义的噪声调度：
- **训练时**：随机采样一个 $\sigma$ 值，决定给数据加多少噪声
- **采样时**：按递减的 $\sigma$ 序列逐步去噪

### 接口定义

```python
class Schedule:
    def __init__(self, sigmas: torch.FloatTensor):
        self.sigmas = sigmas  # 递增的 σ 序列

    def __getitem__(self, i) -> torch.FloatTensor:
        return self.sigmas[i]  # 支持索引和切片

    def __len__(self) -> int:
        return len(self.sigmas)

    def sample_sigmas(self, steps: int) -> torch.FloatTensor:
        """采样时使用：从完整调度中子采样 steps+1 个递减的 σ 值"""
        ...

    def sample_batch(self, x0: torch.FloatTensor) -> torch.FloatTensor:
        """训练时使用：随机采样一批 σ 值"""
        ...
```

### `sample_sigmas(steps)` 详解

该方法在采样阶段调用，从完整的 $N$ 步调度中选取 `steps` 个时间步，返回 `steps + 1` 个递减的 $\sigma$ 值（包含起始和终止值）。

采用 "trailing" 间距策略（参考 [Table 2, arXiv:2305.08891](https://arxiv.org/abs/2305.08891)）：

```python
def sample_sigmas(self, steps: int) -> torch.FloatTensor:
    indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                   .round().astype(np.int64) - 1)
    return self[indices + [0]]
```

**工作原理：**
- `np.arange(0, steps)/steps` 生成 `[0, 1/steps, 2/steps, ..., (steps-1)/steps]`
- `1 - ...` 翻转为递减序列
- 乘以 `len(self)` 并四舍五入得到索引
- 末尾追加索引 `0`（最小 $\sigma$）

**示例：** 若 `N=1000, steps=5`，则选取索引约为 `[999, 799, 599, 399, 199, 0]`，返回 6 个 $\sigma$ 值。

### `sample_batch(x0)` 详解

训练时调用，为每个样本随机采样一个 $\sigma$ 值：

```python
def sample_batch(self, x0: torch.FloatTensor) -> torch.FloatTensor:
    batchsize = x0.shape[0]
    return self[torch.randint(len(self), (batchsize,))].to(x0)
```

从 `[0, N)` 均匀随机选取索引，返回对应的 $\sigma$ 值，并转移到与 `x0` 相同的设备。

---

## 2.3 sigmas_from_betas 辅助函数

### 是什么

将 $\beta$ 序列转换为 $\sigma$ 序列的工具函数。

### 数学推导

给定 $\beta_t$ 序列，累积乘积 $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$，则：

$$\sigma_t = \sqrt{\frac{1}{\bar{\alpha}_t} - 1}$$

```python
def sigmas_from_betas(betas: torch.FloatTensor):
    return (1/torch.cumprod(1.0 - betas, dim=0) - 1).sqrt()
```

### 为什么需要

大多数经典扩散模型论文（DDPM、LDM 等）使用 $\beta$ 参数化定义调度，而 smalldiffusion 内部统一使用 $\sigma$ 参数化。此函数是两种参数化之间的桥梁。

---

## 2.4 ScheduleLogLinear

### 是什么

在对数空间中线性插值的简单调度，$\sigma$ 从 `sigma_min` 到 `sigma_max` 呈对数线性增长。

```python
class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float=0.02, sigma_max: float=10):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))
```

### 适用场景

- 玩具模型和小数据集
- 快速实验和原型验证
- 与 `Scaled` 修饰器配合使用效果好（U-Net 示例中使用）

### 使用示例

```python
from smalldiffusion import ScheduleLogLinear

schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)
print(f"σ 范围: [{schedule[0]:.4f}, {schedule[-1]:.4f}]")
print(f"总步数: {len(schedule)}")
# 采样时子采样 20 步
sigmas = schedule.sample_sigmas(20)
print(f"采样 σ 序列长度: {len(sigmas)}")  # 21
```

---

## 2.5 ScheduleDDPM

### 是什么

复现 DDPM 论文 ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) 中的线性 $\beta$ 调度。

```python
class ScheduleDDPM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start, beta_end, N)))
```

### 数学细节

$\beta$ 从 `beta_start` 到 `beta_end` 线性增长：

$$\beta_t = \beta_{\text{start}} + \frac{t}{N-1}(\beta_{\text{end}} - \beta_{\text{start}})$$

然后通过 `sigmas_from_betas` 转换为 $\sigma$ 序列。

### 适用场景

- 像素空间图像扩散模型
- 与 HuggingFace Diffusers 的 `DDIMScheduler` / `DDPMScheduler` 等价

---

## 2.6 ScheduleLDM

### 是什么

复现潜空间扩散模型（如 Stable Diffusion）使用的 "scaled linear" $\beta$ 调度。

```python
class ScheduleLDM(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.00085, beta_end: float=0.012):
        super().__init__(sigmas_from_betas(torch.linspace(beta_start**0.5, beta_end**0.5, N)**2))
```

### 数学细节

与 DDPM 不同，LDM 对 $\sqrt{\beta}$ 做线性插值后再平方：

$$\beta_t = \left(\sqrt{\beta_{\text{start}}} + \frac{t}{N-1}(\sqrt{\beta_{\text{end}}} - \sqrt{\beta_{\text{start}}})\right)^2$$

这使得 $\beta$ 的增长更平缓，适合潜空间中的扩散。

### 适用场景

- Stable Diffusion 等潜空间扩散模型
- 默认参数与 `CompVis/stable-diffusion-v1-4` 的调度一致

---

## 2.7 ScheduleSigmoid

### 是什么

使用 Sigmoid 函数定义 $\beta$ 调度，来自 [GeoDiff](https://arxiv.org/abs/2203.02923)。

```python
class ScheduleSigmoid(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02):
        betas = torch.sigmoid(torch.linspace(-6, 6, N)) * (beta_end - beta_start) + beta_start
        super().__init__(sigmas_from_betas(betas))
```

### 数学细节

$$\beta_t = \text{sigmoid}\left(-6 + \frac{12t}{N-1}\right) \cdot (\beta_{\text{end}} - \beta_{\text{start}}) + \beta_{\text{start}}$$

Sigmoid 形状使得 $\beta$ 在中间区域变化最快，两端变化缓慢，形成 S 形曲线。

### 适用场景

- 分子构象生成（GeoDiff）
- CIFAR-10 训练示例中使用此调度

---

## 2.8 ScheduleCosine

### 是什么

使用余弦函数定义 $\bar{\alpha}$ 调度，来自 [iDDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)。

```python
class ScheduleCosine(Schedule):
    def __init__(self, N: int=1000, beta_start: float=0.0001, beta_end: float=0.02, max_beta: float=0.999):
        alpha_bar = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = [min(1 - alpha_bar((i+1)/N)/alpha_bar(i/N), max_beta)
                 for i in range(N)]
        super().__init__(sigmas_from_betas(torch.tensor(betas, dtype=torch.float32)))
```

### 数学细节

$$\bar{\alpha}(t) = \cos^2\left(\frac{t + 0.008}{1.008} \cdot \frac{\pi}{2}\right)$$

$$\beta_t = \min\left(1 - \frac{\bar{\alpha}(t+1/N)}{\bar{\alpha}(t/N)},\ \beta_{\max}\right)$$

偏移量 0.008 防止 $t=0$ 时 $\beta$ 过小；`max_beta` 截断防止数值不稳定。

### 适用场景

- 改进的 DDPM 训练
- 在低噪声区域提供更均匀的信噪比变化

---

## 2.9 generate_train_sample 函数

### 是什么

为训练生成 $(x_0, \sigma, \varepsilon, \text{cond})$ 四元组的辅助函数。

```python
def generate_train_sample(x0, schedule, conditional=False):
    cond = x0[1] if conditional else None
    x0   = x0[0] if conditional else x0
    sigma = schedule.sample_batch(x0)
    while len(sigma.shape) < len(x0.shape):
        sigma = sigma.unsqueeze(-1)
    eps = torch.randn_like(x0)
    return x0, sigma, eps, cond
```

### 工作流程

1. **条件处理**：若 `conditional=True`，`x0` 是 `(data, labels)` 元组，拆分为数据和条件
2. **采样 σ**：从调度中随机采样一批 $\sigma$ 值
3. **维度对齐**：将 $\sigma$ 扩展到与 $x_0$ 相同的维度数（用于广播），例如图像数据 `[B, C, H, W]` 需要 $\sigma$ 形状为 `[B, 1, 1, 1]`
4. **生成噪声**：采样与 $x_0$ 同形状的标准正态噪声 $\varepsilon$

---

## 2.10 training_loop 函数

### 是什么

一个 Python 生成器函数，实现完整的扩散模型训练循环。

```python
def training_loop(loader, model, schedule, accelerator=None,
                  epochs=10000, lr=1e-3, conditional=False):
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    for _ in (pbar := tqdm(range(epochs))):
        for x0 in loader:
            model.train()
            optimizer.zero_grad()
            x0, sigma, eps, cond = generate_train_sample(x0, schedule, conditional)
            loss = model.get_loss(x0, sigma, eps, cond=cond)
            yield SimpleNamespace(**locals())
            accelerator.backward(loss)
            optimizer.step()
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `loader` | `DataLoader` | PyTorch 数据加载器 |
| `model` | `nn.Module` | 扩散模型（需实现 `get_loss` 方法） |
| `schedule` | `Schedule` | 噪声调度 |
| `accelerator` | `Accelerator` | HuggingFace Accelerate 实例（可选，默认自动创建） |
| `epochs` | `int` | 训练轮数，默认 10000 |
| `lr` | `float` | 学习率，默认 1e-3 |
| `conditional` | `bool` | 是否为条件生成，默认 False |

### 设计亮点

1. **生成器模式**：使用 `yield` 而非回调，调用者可以在每个训练步后执行自定义逻辑（如记录损失、保存检查点）
2. **Accelerate 集成**：通过 `accelerator.prepare()` 自动处理多 GPU 分布式训练
3. **命名空间暴露**：`yield SimpleNamespace(**locals())` 将所有局部变量（`loss`, `x0`, `sigma`, `eps`, `pbar` 等）暴露给调用者

### 使用示例

```python
from smalldiffusion import training_loop

# 基本用法：收集损失
trainer = training_loop(loader, model, schedule, epochs=100)
losses = [ns.loss.item() for ns in trainer]

# 高级用法：自定义训练逻辑
for ns in training_loop(loader, model, schedule, epochs=100):
    ns.pbar.set_description(f'Loss={ns.loss.item():.5f}')
    if ns.loss.item() < 0.01:
        break  # 提前停止
```

### 训练流程图

```
for each epoch:
    for each batch x0 from loader:
        1. model.train()
        2. optimizer.zero_grad()
        3. (x0, σ, ε, cond) = generate_train_sample(x0, schedule)
        4. loss = model.get_loss(x0, σ, ε, cond)
        5. yield namespace  ← 调用者可在此处插入逻辑
        6. loss.backward()
        7. optimizer.step()
```

---

## 2.11 samples 函数

### 是什么

扩散模型的通用采样生成器，仅用 5 行核心代码统一了 DDPM、DDIM 和加速采样算法。

```python
@torch.no_grad()
def samples(model, sigmas, gam=1., mu=0., cfg_scale=0.,
            batchsize=1, xt=None, cond=None, accelerator=None):
    model.eval()
    accelerator = accelerator or Accelerator()
    xt = model.rand_input(batchsize).to(accelerator.device) * sigmas[0] if xt is None else xt
    if cond is not None:
        assert cond.shape[0] == xt.shape[0], 'cond must have same shape as x!'
        cond = cond.to(xt.device)
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps_prev, eps = eps, model.predict_eps_cfg(xt, sig.to(xt), cond, cfg_scale)
        eps_av = eps * gam + eps_prev * (1-gam)  if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu))
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(xt.shape[0]).to(xt)
        yield xt
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `nn.Module` | - | 扩散模型 |
| `sigmas` | `FloatTensor` | - | 递减的 σ 序列（N+1 个值对应 N 步采样） |
| `gam` | `float` | 1.0 | 噪声预测平均权重，建议 ≥ 1 |
| `mu` | `float` | 0.0 | 随机性控制参数，范围 [0, 1) |
| `cfg_scale` | `float` | 0.0 | Classifier-Free Guidance 强度，0 表示不使用 |
| `batchsize` | `int` | 1 | 生成样本数 |
| `xt` | `FloatTensor` | None | 自定义初始噪声（可选） |
| `cond` | `Tensor` | None | 条件信息（可选） |
| `accelerator` | `Accelerator` | None | 多 GPU 支持 |

### 核心采样公式推导

每一步从 $\sigma_t$ 到 $\sigma_{t-1}$ 的更新：

**第 1 步：噪声预测平均**

$$\bar{\varepsilon} = \gamma \cdot \varepsilon_t + (1-\gamma) \cdot \varepsilon_{t+1}$$

当 `gam=1` 时退化为仅使用当前预测；`gam=2` 时利用历史预测进行外推加速。

**第 2 步：计算中间 σ**

$$\sigma_p = \left(\frac{\sigma_{t-1}}{\sigma_t^\mu}\right)^{1/(1-\mu)}$$

当 `mu=0` 时 $\sigma_p = \sigma_{t-1}$（确定性）；当 `mu=0.5` 时引入随机性（DDPM 行为）。

**第 3 步：计算随机噪声幅度**

$$\eta = \sqrt{\sigma_{t-1}^2 - \sigma_p^2}$$

**第 4 步：更新**

$$x_{t-1} = x_t - (\sigma_t - \sigma_p) \cdot \bar{\varepsilon} + \eta \cdot z, \quad z \sim \mathcal{N}(0, I)$$

### 采样算法对应关系

| 算法 | `gam` | `mu` | 行为 |
|------|-------|------|------|
| DDPM | 1 | 0.5 | $\sigma_p < \sigma_{t-1}$，$\eta > 0$，随机采样 |
| DDIM | 1 | 0 | $\sigma_p = \sigma_{t-1}$，$\eta = 0$，确定性采样 |
| 加速 | 2 | 0 | 利用 $\varepsilon_{t+1}$ 外推，确定性，更少步数 |

### 使用示例

```python
from smalldiffusion import samples, ScheduleLogLinear

schedule = ScheduleLogLinear(N=200, sigma_min=0.005, sigma_max=10)

# DDIM 采样（确定性）
*intermediates, x0 = samples(model, schedule.sample_sigmas(20), gam=1, mu=0)

# DDPM 采样（随机）
*intermediates, x0 = samples(model, schedule.sample_sigmas(50), gam=1, mu=0.5)

# 加速采样
*intermediates, x0 = samples(model, schedule.sample_sigmas(10), gam=2)

# 条件采样 + CFG
import torch
cond = torch.tensor([0, 1, 2, 3])  # 4 个类别标签
*intermediates, x0 = samples(model, schedule.sample_sigmas(20),
                              gam=1.6, batchsize=4,
                              cond=cond, cfg_scale=4.0)
```

### 生成器特性

`samples` 是生成器，每步 `yield` 当前的 $x_t$。这允许：
- 可视化去噪过程的中间结果
- 使用 `*xt, x0 = samples(...)` 解包获取所有中间步和最终结果
- 提前终止采样
