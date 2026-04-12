---
title: "Analytic Diffusion Studio — 数据模块"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 数据模块"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 4
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 数据模块"
---

目录：`src/local_diffusion/data/`

数据模块负责数据集的注册、加载、预处理和后处理。采用工厂模式 + 注册表实现可扩展的数据集管理。

## 4.1 模块结构

```
data/
├── __init__.py                  # 导出公共 API，触发数据集注册
├── datasets.py                  # 注册表、DatasetBundle、build_dataset()
├── torchvision_datasets.py      # MNIST、Fashion-MNIST、CIFAR-10 注册
├── image_folder_datasets.py     # CelebA-HQ、AFHQ 注册
└── utils.py                     # 图像变换、后处理、子集截取
```

## 4.2 注册机制

### 注册表

`DATASET_REGISTRY` 是一个全局字典，将数据集名称映射到工厂函数：

```python
DATASET_REGISTRY: Dict[str, DatasetFactory] = {}
```

### @register_dataset 装饰器

```python
@register_dataset("mnist")
def build_mnist(cfg: DatasetConfig) -> DatasetFactoryOutput:
    ...
```

装饰器将函数注册到 `DATASET_REGISTRY["mnist"]`。注册在模块导入时自动完成——`data/__init__.py` 中显式导入了 `torchvision_datasets` 和 `image_folder_datasets`。

### DatasetFactoryOutput

工厂函数的返回类型：

```python
@dataclass
class DatasetFactoryOutput:
    dataset: Dataset          # PyTorch Dataset 实例
    resolution: int           # 图像分辨率（正方形边长）
    in_channels: int          # 通道数（1=灰度，3=RGB）
    postprocess: Optional[Callable]  # 后处理函数（[-1,1] → [0,1]）
```

### DatasetBundle

`build_dataset()` 的返回类型，封装了数据集的所有信息：

```python
@dataclass
class DatasetBundle:
    name: str                 # 数据集名称
    dataset: Dataset          # PyTorch Dataset
    dataloader: DataLoader    # 预配置的 DataLoader
    resolution: int           # 图像分辨率
    in_channels: int          # 通道数
    split: str                # 数据划分（train/test）
    postprocess: Callable     # 后处理函数
```

## 4.3 build_dataset() 流程

```python
def build_dataset(cfg: DatasetConfig) -> DatasetBundle:
```

执行步骤：
1. 从 `DATASET_REGISTRY` 查找工厂函数
2. 调用工厂函数，传入 `DatasetConfig`
3. 可选：应用子集截取 (`maybe_apply_subset`)
4. 创建 `DataLoader`（`shuffle=False`, `pin_memory=True`）
5. 封装为 `DatasetBundle` 返回

```python
# DataLoader 配置
DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=False,          # 不打乱（保证可复现）
    num_workers=cfg.num_workers,
    pin_memory=True,        # GPU 加速
    drop_last=False,
)
```

## 4.4 支持的数据集

### MNIST (`torchvision_datasets.py`)

```python
@register_dataset("mnist")
def build_mnist(cfg: DatasetConfig) -> DatasetFactoryOutput:
    transform = utils.compose_transform(28, in_channels=1)
    dataset = datasets.MNIST(root=cfg.root, train=cfg.split == "train",
                             download=cfg.download, transform=transform)
    postprocess = utils.get_postprocess_fn()
    return DatasetFactoryOutput(dataset=dataset, resolution=28,
                                in_channels=1, postprocess=postprocess)
```

- 分辨率：28×28，灰度（1 通道）
- 支持自动下载

### Fashion-MNIST (`torchvision_datasets.py`)

```python
@register_dataset("fashion_mnist")
def build_fashion_mnist(cfg: DatasetConfig) -> DatasetFactoryOutput:
```

- 与 MNIST 结构相同，分辨率 28×28，灰度
- 使用 `datasets.FashionMNIST`

### CIFAR-10 (`torchvision_datasets.py`)

```python
@register_dataset("cifar10")
def build_cifar10(cfg: DatasetConfig) -> DatasetFactoryOutput:
    transform = utils.compose_transform(32, in_channels=3)
    dataset = datasets.CIFAR10(...)
```

- 分辨率：32×32，RGB（3 通道）
- 支持自动下载

### CelebA-HQ (`image_folder_datasets.py`)

```python
@register_dataset("celeba_hq")
def build_celeba_hq(cfg: DatasetConfig) -> DatasetFactoryOutput:
```

- 默认分辨率：256（可通过 `cfg.resolution` 覆盖，通常降至 64）
- 使用自定义 `ImageFolderDataset`（支持扁平目录结构）
- 可选自动下载（从 Kaggle，需要 curl）
- 期望目录：`data/datasets/celebahq-resized-256x256/versions/1/celeba_hq_256/`

### AFHQ (`image_folder_datasets.py`)

```python
@register_dataset("afhq")
def build_afhq(cfg: DatasetConfig) -> DatasetFactoryOutput:
```

- 默认分辨率：512（通常降至 64）
- 使用 `torchvision.datasets.ImageFolder`（按类别子目录组织）
- 可选自动下载（从 Dropbox）
- 期望目录：`data/datasets/afhq/{split}/{class}/`

### ImageFolderDataset 自定义类

```python
class ImageFolderDataset(Dataset):
    """支持扁平目录结构的图像数据集（不要求按类别子目录组织）。"""
    
    def __init__(self, root_dir: str, transform=None):
        # 扫描目录中所有 .png/.jpg/.jpeg/.bmp/.webp 文件
        self.image_files = sorted([f for f in os.listdir(root_dir)
                                    if f.lower().endswith((...)) ])
    
    def __getitem__(self, idx):
        image = Image.open(img_path).convert("RGB")
        return image, 0  # 返回 0 作为虚拟标签
```

与 `torchvision.datasets.ImageFolder` 的区别：后者要求图像按类别放在子目录中，而 `ImageFolderDataset` 支持所有图像直接放在一个目录下。

## 4.5 图像预处理管线

`utils.compose_transform(resolution, in_channels)` 构建变换链：

```python
def compose_transform(resolution, *, in_channels):
    ops = []
    if in_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.extend([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),                    # [0, 255] → [0, 1]
        transforms.Normalize((0.5,)*C, (0.5,)*C), # [0, 1] → [-1, 1]
    ])
    return transforms.Compose(ops)
```

变换步骤：
1. **灰度转换**（仅单通道数据集）
2. **Resize** 到目标分辨率（双线性插值）
3. **ToTensor**：PIL Image → `[0, 1]` 浮点张量
4. **Normalize**：`(x - 0.5) / 0.5` → `[-1, 1]` 范围

**为什么归一化到 [-1, 1]？** 扩散模型假设数据分布以 0 为中心，这与高斯噪声的分布一致，有利于训练和采样的数值稳定性。

## 4.6 后处理

```python
def get_postprocess_fn():
    def postprocess(tensor):
        return ((tensor + 1.0) / 2.0).clamp(0, 1)  # [-1, 1] → [0, 1]
    return postprocess
```

后处理将模型输出从 `[-1, 1]` 映射回 `[0, 1]`，用于图像保存和可视化。

## 4.7 子集截取

```python
def maybe_apply_subset(dataset, subset_size):
    if subset_size is None:
        return dataset
    indices = torch.arange(subset_size)
    return Subset(dataset, indices.tolist())
```

通过 `dataset.subset_size` 配置项限制数据集大小，用于快速调试或降低计算量。取前 N 个样本（非随机采样）。

## 4.8 添加新数据集

```python
# 在 data/ 下新建文件或在已有文件中添加：
from .datasets import register_dataset, DatasetFactoryOutput

@register_dataset("my_dataset")
def build_my_dataset(cfg: DatasetConfig) -> DatasetFactoryOutput:
    transform = utils.compose_transform(cfg.resolution or 64, in_channels=3)
    dataset = MyCustomDataset(root=cfg.root, transform=transform)
    return DatasetFactoryOutput(
        dataset=dataset,
        resolution=cfg.resolution or 64,
        in_channels=3,
        postprocess=utils.get_postprocess_fn(),
    )
```

然后在 `data/__init__.py` 中导入新模块以触发注册。
