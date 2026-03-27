---
title: "Analytic Diffusion Studio — 模型基类与采样循环"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 模型基类与采样循环"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 5
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 模型基类与采样循环"
---

summary: "Analytic Diffusion Studio — 模型基类与采样循环"

文件：`src/local_diffusion/models/base.py`、`src/local_diffusion/models/__init__.py`

## 5.1 模型注册表

与数据集类似，模型也使用注册表模式：

```python
# models/__init__.py
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_model(name: str):
    def decorator(cls_or_factory):
        MODEL_REGISTRY[name.lower()] = cls_or_factory
        return cls_or_factory
    return decorator

def create_model(name: str, **kwargs) -> Any:
    factory = MODEL_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return factory(**kwargs)
```

已注册的模型在模块底部通过导入触发：

```python
from . import nearest_dataset   # "nearest_dataset"
from . import optimal           # "optimal"
from . import scfdm             # "scfdm"
from . import wiener            # "wiener"
from . import pca_locality      # "pca_locality"
```

在 `generate.py` 中的调用方式：

```python
model = create_model(
    cfg.model.name,           # 例如 "pca_locality"
    dataset=dataset,          # DatasetBundle
    device=cfg.experiment.device,
    num_steps=cfg.sampling.num_inference_steps,
    params=model_params,      # 模型特定超参数 dict
)
```

## 5.2 SamplingOutput 数据类

采样结果的统一封装：

```python
@dataclass
class SamplingOutput:
    images: torch.Tensor                    # 最终生成图像 [N, C, H, W]
    timesteps: Optional[List[int]]          # 时间步列表（如 [999, 888, ...]）
    trajectory_xt: Optional[List[Tensor]]   # 每步的噪声图像 x_t
    trajectory_x0: Optional[List[Tensor]]   # 每步的预测干净图像 x̂₀
```

`trajectory_xt` 和 `trajectory_x0` 是列表，每个元素对应一个时间步，形状为 `[N, C, H, W]`。仅在 `return_intermediates=True` 时记录。

## 5.3 BaseDenoiser 类

所有去噪模型的抽象基类，继承自 `torch.nn.Module`。

### 构造函数

```python
class BaseDenoiser(torch.nn.Module):
    prediction_type: str = "epsilon"  # 预测类型：噪声 ε

    def __init__(self, resolution, device, num_steps, *,
                 beta_1=0.0001, beta_T=0.02,
                 dataset_name="cifar10",
                 scheduler_kwargs=None, **kwargs):
        self.device = torch.device(device)
        self.n_channels = kwargs.get("in_channels", 3)
        self.resolution = resolution
        self.num_steps = num_steps

        # 初始化 DDIM 调度器
        self.scheduler = DDIMScheduler(
            beta_start=beta_1, beta_end=beta_T,
            beta_schedule="linear",
            prediction_type=self.prediction_type,
        )
        self.scheduler.set_timesteps(num_steps)
```

关键属性：
- `self.scheduler`：HuggingFace diffusers 的 `DDIMScheduler`，管理噪声调度表和时间步
- `self.scheduler.alphas_cumprod`：累积 $\bar{\alpha}_t$ 数组，长度 1000
- `self.scheduler.timesteps`：采样时间步（如 10 步时为 `[999, 899, 799, ...]`）

### denoise() — 抽象方法

```python
def denoise(self, latents, timestep, *, generator=None, **kwargs):
    """预测去噪后的干净图像 x̂₀。子类必须实现此方法。"""
    raise NotImplementedError
```

输入：
- `latents`：当前噪声图像 $x_t$，形状 `[B, C, H, W]`
- `timestep`：当前时间步（标量或 0-d 张量）

输出：
- 预测的干净图像 $\hat{x}_0$，形状 `[B, C, H, W]`

### train() — 抽象方法

```python
def train(self, dataset):
    """预计算数据集相关参数。子类必须实现。"""
    raise NotImplementedError
```

不同模型的 `train()` 做不同的事：
- Wiener：计算协方差矩阵 SVD
- Optimal：构建 FAISS 索引
- PCA Locality：计算协方差 SVD + 保留数据集引用
- Nearest：将数据集加载到 GPU 内存

### prepare_latents()

```python
def prepare_latents(self, batch_size, generator=None):
    shape = (batch_size, self.n_channels, self.resolution, self.resolution)
    latents = torch.randn(shape, generator=generator, device=self.device)
    return latents * self.scheduler.init_noise_sigma
```

生成初始噪声 $x_T \sim \mathcal{N}(0, I)$，乘以调度器的初始噪声标准差。

### compute_noise_from_x0()

```python
def compute_noise_from_x0(self, x_t, pred_x0, timestep):
    alpha_prod = self.scheduler.alphas_cumprod[t]
    beta_prod = 1 - alpha_prod
    sqrt_alpha = torch.sqrt(alpha_prod)
    sqrt_beta = torch.sqrt(beta_prod + 1e-8)
    return (x_t - sqrt_alpha * pred_x0) / sqrt_beta
```

从预测的 $\hat{x}_0$ 反推预测噪声 $\hat{\epsilon}$：

$$\hat{\epsilon} = \frac{x_t - \sqrt{\bar{\alpha}_t} \hat{x}_0}{\sqrt{1 - \bar{\alpha}_t}}$$

这是因为 DDIMScheduler 期望接收噪声预测（`prediction_type="epsilon"`）。

### set_timesteps()

```python
def set_timesteps(self, num_steps):
    self.scheduler.set_timesteps(num_steps)
    self.num_steps = num_steps
```

更新采样步数。DDIMScheduler 会自动计算等间距时间步。

### _image_preprocess() / _image_postprocess()

```python
def _image_preprocess(self, img):
    # 插值到目标分辨率 + 归一化到 [-1, 1]
    imgs = F.interpolate(img, size=(self.resolution, self.resolution), mode="bilinear")
    return (imgs - 0.5) * 2

def _image_postprocess(self, img):
    # [-1, 1] → [0, 1]
    return ((img + 1) / 2).clamp(0, 1)
```

这两个方法在基类中定义但未被所有子类使用（数据预处理主要在 `data/utils.py` 中完成）。

## 5.4 采样循环

### sample() — 公共接口

```python
@torch.no_grad()
def sample(self, *, num_samples, batch_size, generator=None,
           return_intermediates=False) -> SamplingOutput:
```

处理多批次采样：
1. 计算需要的批次数
2. 对每批调用 `_sample_batch()`
3. 拼接所有批次的结果（images、trajectory_xt、trajectory_x0）
4. 返回统一的 `SamplingOutput`

### _sample_batch() — 单批次 DDIM 循环

```python
def _sample_batch(self, *, batch_size, generator, return_intermediates):
    latents = self.prepare_latents(batch_size, generator)

    for step_idx, timestep in enumerate(self.scheduler.timesteps):
        # 1. 调用子类的 denoise() 预测 x̂₀
        pred_x0 = self.denoise(latents, timestep, generator=generator)

        # 2. 从 x̂₀ 反推预测噪声 ε̂
        predicted_noise = self.compute_noise_from_x0(latents, pred_x0, timestep)

        # 3. DDIM 调度器计算 x_{t-1}
        step_output = self.scheduler.step(
            model_output=predicted_noise,
            timestep=timestep,
            sample=latents,
        )

        # 4. 记录轨迹（可选）
        if return_intermediates:
            trajectory_xt.append(latents.detach().cpu())
            trajectory_x0.append(pred_x0.detach().cpu())

        latents = step_output.prev_sample  # 更新为 x_{t-1}

    return SamplingOutput(images=last_pred_x0, ...)
```

注意：最终返回的 `images` 是最后一步的 `pred_x0`（而非 `latents`），因为 `pred_x0` 是对干净图像的直接预测。

## 5.5 采样流程图

```
x_T (纯噪声)
  │
  ▼ t = 999
  denoise(x_T, 999) → x̂₀⁽¹⁾
  compute_noise → ε̂⁽¹⁾
  scheduler.step → x_{t-1}
  │
  ▼ t = 899
  denoise(x_{899}, 899) → x̂₀⁽²⁾
  compute_noise → ε̂⁽²⁾
  scheduler.step → x_{t-2}
  │
  ... (重复 num_inference_steps 次)
  │
  ▼ t = 99
  denoise(x_{99}, 99) → x̂₀⁽ᴺ⁾  ← 最终输出
```

## 5.6 build_sample_output()

```python
def build_sample_output(self, images, trajectory_xt, trajectory_x0, timesteps):
    return SamplingOutput(
        images=images,
        trajectory_xt=trajectory_xt,
        trajectory_x0=trajectory_x0,
        timesteps=timesteps,
    )
```

简单的工厂方法，将采样结果封装为 `SamplingOutput`。
