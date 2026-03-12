---

title: "PyTorch Dataset 体系技术文档"
subtitle: ""
date: 2026-03-12T00:00:00+08:00
draft: false
authors: [Steven]
description: "系统梳理 PyTorch 数据集体系：Dataset 基类（**len**/**getitem**/**getitems**/**add**）、内置扩展（TensorDataset/ConcatDataset/Subset/StackDataset/IterableDataset/ChainDataset/random_split）、PyG/HuggingFace、领域库，以及 padding、collate 与 DataLoader 的协作。"
summary: "覆盖 map-style/IterableDataset、全部内置 Dataset 扩展、图数据与 HF datasets、典型项目扩展模式、padding 与 collate 职责划分，以及与 DataLoader 的衔接。"

tags: ["PyTorch", "Dataset", "Deep Learning"]
categories: ["PyTorch"]
series: ["PyTorch 实践指南"]
weight: 2
series_weight: 2

hiddenFromHomePage: false
hiddenFromSearch: false

## featuredImage: ""
featuredImagePreview: ""
---

## 文档索引


| 章节                                                                    | 主题                                                                                                    | 主要内容                                                                     |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [一、总览](#一总览)                                                          | 整体架构与文档脉络                                                                                             | 知识结构、各部分作用与关联、优缺点与适用场景                                                   |
| [二、torch.utils.data.Dataset](#二torchutilsdatadataset--根基)             | PyTorch 数据集根基                                                                                         | 抽象基类协议、`__len__`/`__getitem__`/`__getitems__`/`__add__`、与 DataLoader 的契约 |
| [三、torch.utils.data 内置扩展](#三torchutilsdata-内置扩展)                      | TensorDataset / ConcatDataset / Subset / StackDataset / IterableDataset / ChainDataset / random_split | 结构、完整接口、使用方式与可运行示例                                                       |
| [四、torch_geometric.data.Dataset](#四torch_geometricdatadataset--图数据专用) | 图数据专用 Dataset                                                                                         | 文件管理、三阶段 transform、Dataset vs InMemoryDataset                            |
| [五、HuggingFace datasets](#五huggingface-datasetsdataset--独立体系)         | 独立数据体系                                                                                                | Arrow、map/filter、set_format 与 PyTorch 互转                                 |
| [六、领域库数据集](#六领域库数据集)                                                  | torchvision / torchaudio                                                                              | transform 管线、典型用法                                                        |
| [七、典型项目扩展模式](#七典型项目扩展模式)                                              | 多任务/RL/图风格数据入口                                                                                        | BaseDataset/RLDataset/GTDataset 等概念与调用关系（通用模式）                           |
| [八、Padding 与 collate 协作](#八padding-与-collate-协作)                      | 变长样本成 batch                                                                                           | 单条 padding 与 collate_fn 的职责划分、典型实现模式                                     |
| [九、与 DataLoader 的协作](#九与-dataloader-的协作)                              | Dataset → DataLoader → 训练                                                                             | collate_fn、多进程、与 [dataloader_guide.md](./dataloader_guide.md) 的衔接        |


---

## 一、总览

### 1.1 整体架构与知识结构

PyTorch 的数据集体系以 `**torch.utils.data.Dataset`** 为根基，向上衍生出多条线：

```
torch.utils.data.Dataset              ← 所有 PyTorch 系的根基（map-style）
    ├── TensorDataset                  包装多个等长 tensor
    ├── ConcatDataset                  逻辑拼接多个 map-style 数据集
    ├── Subset                         按索引取子集（train/val split）
    ├── StackDataset                   多数据集按样本对齐堆叠（tuple/dict）
    ├── IterableDataset                流式数据，另一条线，不支持随机访问
    │       └── ChainDataset           串联多个 IterableDataset（+ 运算符）
    ├── random_split(dataset, lengths) 随机划分成多个 Subset
    │
    ├── torchvision / torchaudio 各数据集
    │       └── 在基类上加了 PIL/波形 transform 管线
    │
    └── torch_geometric.data.Dataset   图数据：raw/processed 文件管理 + 三阶段 transform
            └── InMemoryDataset        全量加载进内存，适合小图集

datasets.Dataset (HuggingFace)        ← 独立体系，Arrow 格式，可通过 set_format 转 torch
```

### 1.2 各部分作用与主题


| 部分                         | 作用/主题                             | 解决的核心问题                                          |
| -------------------------- | --------------------------------- | ------------------------------------------------ |
| **Dataset 基类**             | 定义「长度 + 按索引取一条」的协议                | 让 DataLoader 能统一驱动任意数据源，而不关心存储与格式                |
| **TensorDataset**          | 把已有 tensor 包装成 Dataset            | 快速从内存 tensor 建可迭代数据源                             |
| **ConcatDataset / Subset** | 逻辑组合与切分                           | 多数据源合并、train/val 划分，无需拷贝数据                       |
| **StackDataset**           | 多集按样本对齐堆叠为 tuple/dict             | 多模态（图+文等）同索引对齐                                   |
| **IterableDataset**        | 流式、无索引                            | 大文件、管道、无法事先知道总条数的场景                              |
| **ChainDataset**           | 串联多个 IterableDataset              | 流式多源顺序拼接；IterableDataset 的 `+` 运算符               |
| **random_split**           | 按长度或比例随机划分                        | 可复现的 train/val/test 划分                           |
| **PyG Dataset**            | 图数据 + 文件与预处理管线                    | 下载、process、pre_filter/pre_transform/transform 分工 |
| **HuggingFace datasets**   | 列式存储、map/filter、格式转换              | 大表、并行预处理、与 PyTorch 无缝对接                          |
| **领域库**                    | 图像/音频等领域的标准数据集与 transform         | 常见任务开箱即用                                         |
| **本项目 BaseDataset 系**      | 多任务 pkl 文件列表、任务比例、padding、collate | 统一多任务驾驶数据与仿真入口                                   |
| **本项目 GTDataset**          | 借 PyG 外壳、pkl 路径列表、按需加载与 padding   | 与 BaseDataset 类似但走 PyG 目录约定，便于与现有脚本统一            |


### 1.3 彼此关联

- **DataLoader** 只依赖「长度 + 按索引取一条」或「可迭代流」；**Dataset 决定单条样本格式**，**collate_fn 决定多条如何成 batch**。
- **BaseDataset / RLDataset / GTDataset** 都产出「dict of tensors + labels」形态；差异在于：BaseDataset 来自 `code_train` 配置与 pkl 列表，RLDataset 增加轨迹采样与专用 collate，GTDataset 来自 `datasets/gt_dataset.py`、用 PyG 外壳。
- **Padding** 在 `__getitem`__/`get()` 内对单条样本执行，**collate_fn** 在 DataLoader 内对 list of 样本执行；两者配合得到固定形状的 batch。

### 1.4 优缺点与适用场景对比


| 类型                       | 优点                       | 缺点                             | 适用场景                         |
| ------------------------ | ------------------------ | ------------------------------ | ---------------------------- |
| **map-style Dataset**    | 支持随机访问、shuffle、多进程索引分片   | 需事先能确定长度                       | 绝大多数训练/验证                    |
| **IterableDataset**      | 可流式、不必装全量进内存             | 不能真正 shuffle，多进程需自行分片          | 大日志、管道、实时流                   |
| **TensorDataset**        | 实现极简、零拷贝                 | 仅适合已在内存的等长 tensor              | 小规模、快速实验                     |
| **ConcatDataset/Subset** | 逻辑视图、无拷贝                 | 仅做索引映射                         | 多源合并、比例划分                    |
| **StackDataset**         | 多集对齐、tuple/dict 输出       | 要求各集长度一致                       | 多模态按样本对齐                     |
| **ChainDataset**         | 流式串联、无预加载                | 仅支持 IterableDataset            | 多流顺序拼接                       |
| **PyG Dataset**          | 标准化的下载/process/transform | 需实现 len/get 与文件列表              | 图结构、需持久化处理结果                 |
| **HuggingFace datasets** | map/filter 强、Arrow 大表友好  | 与 PyTorch 无继承关系                | NLP/表格、Hub 数据、大批量预处理         |
| **本项目 BaseDataset**      | 多任务、比例采样、与训练脚本深度集成       | 依赖配置与 pkl 目录结构                 | 多任务驾驶模型训练与仿真                 |
| **本项目 GTDataset**        | 与 PyG 生态一致、接口简单          | 未用 PyG 的 process()，本质仍是 pkl 列表 | 需要 DataLoader + PyG 风格时的数据入口 |


---

## 二、`torch.utils.data.Dataset` — 根基

### 2.1 结构与边界

所有 PyTorch **map-style** 数据集的抽象基类。基类本身**不包含**任何文件管理、transform 或下载逻辑，只定义与 DataLoader 的**协议**：

- **必须实现**：`__getitem__(self, index)` — 按索引返回一条样本。
- **强烈建议实现**：`__len__(self)` — 返回数据集大小；多数 Sampler 和 DataLoader 默认行为依赖它。
- **可选**：`__getitems__(self, indices)` — 批量取多条样本，用于 DataLoader 加速；未实现时内部会多次调用 `__getitem__`。
- **内置便捷方法**：`__add__(self, other)` — 返回 `ConcatDataset([self, other])`，可用 `ds1 + ds2` 拼接。

**与 DataLoader 的契约**：DataLoader 通过 `len(dataset)` 得到长度，通过 `dataset[i]`（或 `dataset.__getitems__(indices)`）取样本；索引由 Sampler 提供，通常为 `0 .. len(dataset)-1` 的某种排列。若使用非整数 key 或不可求长度，需自定义 Sampler。

### 2.2 关键概念

- **map-style**：有固定长度、支持整数索引、可随机访问；与 **IterableDataset**（仅 `__iter__`，无索引）相对。
- **为何以「长度 + 按索引取一条」为核心**：DataLoader 只关心「有多少条」和「第 i 条是什么」；存储、解码、增强等都在子类的 `__getitem__` 内完成，接口最小且稳定。
- `**__getitems__` 的作用**：当 DataLoader 一次要取一批索引时，若子类实现了 `__getitems__(indices)`，会优先调用它以减少 Python 调用次数、便于做批量 I/O，否则会逐次调用 `__getitem__`。
- `**__add__`**：语法糖，`ds1 + ds2` 等价于 `ConcatDataset([ds1, ds2])`，逻辑拼接、无数据拷贝。

### 2.3 使用方式与示例

继承并实现 `__len__` 与 `__getitem__` 即可接入 DataLoader；样本格式由业务决定，collate_fn 需与之匹配。需要加速时可实现 `__getitems_`_。

```python
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data        # shape: (N, ...)
        self.labels = labels    # shape: (N,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    # 可选：批量取样本，供 DataLoader 优化用
    def __getitems__(self, indices):
        return [(self.data[i], self.labels[i]) for i in indices]

ds = MyDataset(torch.randn(100, 32), torch.randint(0, 10, (100,)))
print(len(ds))         # 100
print(ds[0][0].shape)  # torch.Size([32])

# __add__ 示例
ds2 = MyDataset(torch.randn(50, 32), torch.randint(0, 10, (50,)))
combined = ds + ds2    # ConcatDataset
print(len(combined))   # 150
```

---

## 三、`torch.utils.data` 内置扩展

### 3.1 结构说明与模块划分

基于 `Dataset` / `IterableDataset`，PyTorch 在 `torch.utils.data` 中提供以下组件，按用途分为：**包装类**（TensorDataset、StackDataset）、**组合类**（ConcatDataset、Subset、ChainDataset）、**流式基类**（IterableDataset）、**工具函数**（random_split）。


| 类/函数                | 用途                           | 核心接口/行为                                                                                                                                                            |
| ------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **TensorDataset**   | 包装多个第 0 维等长的 tensor          | `__init__(*tensors)`；`__getitem__(idx)` → `(t[0][idx], t[1][idx], ...)`；`__len_`_；属性 `.tensors`                                                                    |
| **ConcatDataset**   | 逻辑拼接多个 **map-style** Dataset | `__init__(datasets)`；`__len__` 为各集长度之和；`__getitem__(idx)` 按 `cumulative_sizes` 映射到对应子集；静态方法 `cumsum(sequence)`；属性 `cumulative_sizes`；**不支持 IterableDataset**       |
| **Subset**          | 按索引列表取子集                     | `__init__(dataset, indices)`；`__getitem__(idx)` → `dataset[indices[idx]]`，支持 list 索引；`__getitems__(indices)` 委托底层以支持 DataLoader 批取；`__len__` → `len(indices)`      |
| **StackDataset**    | 多数据集「按样本对齐」堆叠为 tuple 或 dict  | `__init__(*args)` 或 `__init__(**kwargs)`；`__getitem__(idx)` 返回 `(ds0[idx], ds1[idx], ...)` 或 `{k: ds[idx] for k, ds}`；`__getitems__` 若子集支持则批取；`__len__` 要求各数据集长度一致 |
| **IterableDataset** | 流式数据基类                       | 实现 `__iter__`，无 `__len__`/`__getitem__`；`__add__` 返回 **ChainDataset**；多 worker 时需用 `get_worker_info()` 分片                                                          |
| **ChainDataset**    | 串联多个 IterableDataset         | `__init__(datasets)`；`__iter__` 依次 `yield from` 各集；`__len__` 为各集长度之和（若均有 `__len__`）                                                                                |
| **random_split**    | 按长度或比例随机划分                   | `random_split(dataset, lengths, generator=None)` 返回 `list[Subset]`；`lengths` 可为整数列表或和为 1 的比例列表                                                                     |


### 3.2 关键概念与注意点

- **TensorDataset**：所有传入 tensor 的第 0 维必须相同；适合已在内存的简单特征+标签；实现极简、零拷贝。
- **ConcatDataset**：子集类型可不同；索引自动偏移（含负索引），无数据拷贝；**不能包含 IterableDataset**（会 AssertionError）。
- **Subset**：`indices` 可为任意整数序列，可重复；不拷贝数据。若子类重写 `__getitem_`_，必须同时重写 `__getitems__`，否则 DataLoader 会报错。
- **StackDataset**：多个 Dataset 长度必须一致；按「同一索引」从各集取一条再组成 tuple/dict，适合多模态（如图+文）对齐。
- **IterableDataset**：不能使用 DataLoader 的 `shuffle=True`（需在数据源侧打乱）；多 worker 时必须在 `__iter__` 或 `worker_init_fn` 中按 `get_worker_info()` 分片，否则每个 worker 会重复全量数据。
- **ChainDataset**：仅接受 IterableDataset；流式串联，不预加载。
- **random_split**：`lengths` 若为比例（和为一），会先换算为整数并处理余数；可用 `generator` 固定随机种子以保证可复现。

### 3.3 示例（可运行）

```python
import math
import torch
from torch.utils.data import (
    TensorDataset, ConcatDataset, Subset, StackDataset,
    IterableDataset, ChainDataset, random_split,
)

# ---------- TensorDataset ----------
X = torch.randn(200, 16)
y = torch.randint(0, 5, (200,))
tensor_ds = TensorDataset(X, y)
print(len(tensor_ds), tensor_ds[0][0].shape)  # 200, torch.Size([16])

# ---------- ConcatDataset：索引映射与负索引 ----------
combined = ConcatDataset([tensor_ds, tensor_ds])
print(len(combined))  # 400
print(combined.cumulative_sizes)  # [200, 400]
print(combined[0] is tensor_ds[0], combined[200] is tensor_ds[0])  # True True

# ---------- Subset：train/val ----------
train_ds = Subset(tensor_ds, range(0, 160))
val_ds = Subset(tensor_ds, range(160, 200))

# ---------- random_split ----------
train_sub, val_sub = random_split(tensor_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
assert len(train_sub) + len(val_sub) == len(tensor_ds)

# ---------- StackDataset：多模态对齐 ----------
text_len = len(tensor_ds)
# 假设有两个等长数据集
stacked = StackDataset(tensor_ds, Subset(tensor_ds, range(text_len)))
print(stacked[0])  # ((tensor(...), tensor(...)), (tensor(...), tensor(...)))
# dict 形式：StackDataset(a=X_ds, b=y_ds) -> stacked[i] == {"a": ..., "b": ...}

# ---------- IterableDataset + 多 worker 分片 ----------
class RangeIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start, self.end = start, end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start, iter_end = self.start, self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / worker_info.num_workers))
            iter_start = self.start + worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))

it_ds = RangeIterableDataset(0, 10)
# 单进程：list(DataLoader(it_ds, num_workers=0)) 得到 0..9 各一次
# 多进程需上述分片，否则会重复

# ---------- ChainDataset ----------
chained = RangeIterableDataset(0, 3) + RangeIterableDataset(10, 13)  # 0,1,2,10,11,12
print(list(chained))  # [0, 1, 2, 10, 11, 12]
```

---

## 四、`torch_geometric.data.Dataset` — 图数据专用

### 4.1 结构与目录约定

继承自 `torch.utils.data.Dataset`，增加：

1. **文件管理**：`raw_dir`（原始文件）与 `processed_dir`（处理后的 `.pt`），按需 `download()` / `process()`。
2. **三阶段 transform**：
  - **pre_filter**：在 `process()` 时过滤，不存盘。  
  - **pre_transform**：在 `process()` 时执行，结果存盘。  
  - **transform**：每次 `get()` 时执行，不存盘。

目录与调用链：

```
root/
  raw/          ← raw_dir
  processed/    ← processed_dir

__init__ → 检查 raw → download()；检查 processed → process()
dataset[idx] → get(idx) → [pre_filter / pre_transform 已在 process 中] → transform(data)
```

### 4.2 必须实现的接口（PyG Dataset）


| 接口                                | 说明                                                       |
| --------------------------------- | -------------------------------------------------------- |
| `raw_file_names` (property)       | 原始文件名列表，用于判断是否需要 download                                |
| `processed_file_names` (property) | 处理后文件名列表，用于判断是否需要 process                                |
| `len()`                           | 数据集大小（注意是 `len()` 不是 `__len__`，PyG 内部会调）                 |
| `get(idx)`                        | 按索引返回单条样本（注意是 `get` 不是 `__getitem__`，基类会在此上应用 transform） |


可选：`download()`、`process()`。

### 4.3 Dataset vs InMemoryDataset


| 对比项            | Dataset     | InMemoryDataset     |
| -------------- | ----------- | ------------------- |
| 存储             | 每样本一个 `.pt` | 全部合并为一个文件           |
| `get()`        | 每次从磁盘读      | 内存切片（data + slices） |
| 规模             | 大图集         | 小/中图集               |
| 是否必须实现 `get()` | 是           | 否（内部用 data/slices）  |


### 4.4 三阶段 transform 对比


| 阶段            | 触发时机         | 是否存盘 | 典型用途                 |
| ------------- | ------------ | ---- | -------------------- |
| pre_filter    | process() 一次 | 否    | 过滤无效图（如 0 节点）        |
| pre_transform | process() 一次 | 是    | 特征归一化、边属性            |
| transform     | 每次 get()     | 否    | 随机 dropout edge、数据增强 |


### 4.5 与 DataLoader 的配合

PyG 的 `Data` 对象可直接作为样本；DataLoader 的 `collate_fn` 通常使用 `torch_geometric.loader.DataLoader` 自带的 `collate`，将 list of `Data` 聚成 `Batch`（含 `batch` 向量等）。若使用 `torch.utils.data.DataLoader`，需自定义 collate 或封装成 tensor/dict。

---

## 五、HuggingFace `datasets.Dataset` — 独立体系

### 5.1 结构与边界

HuggingFace 的 `datasets.Dataset` 与 PyTorch **无继承关系**，底层为 **Apache Arrow** 列式存储，支持内存映射、大表高效访问。常用能力与调用关系：

- **load_dataset()**：从 Hub 或本地（CSV/JSON/Parquet 等）加载，返回 `Dataset` 或 `DatasetDict`。
- **.map(fn, batched=True, ...)**：对每行或每批应用函数，并行预处理（如 tokenize）；`batched=True` 时传入批，性能更好。
- **.filter(fn)**：按条件筛样本，返回新 Dataset。
- **.set_format("torch", columns=...)**: 使迭代时输出变为 torch tensor，可直接配合 `torch.utils.data.DataLoader`（需用 `IterableDataset` 包装或通过 `.with_format` 迭代）。
- **.train_test_split()**：划分 train/test，返回 `DatasetDict`。

### 5.2 关键概念

- **为何独立于 PyTorch**：面向多框架（PyTorch/TF/JAX）与 Hub 生态，列式存储便于列级操作与流式。
- **与 PyTorch 对接**：`set_format("torch")` 后，迭代得到的是 tensor；可包成 `IterableDataset` 或逐批取用，也可用 `Dataset` 的 `__getitem_`_ 封装成 map-style。

### 5.3 使用方式与示例（需安装 `datasets`）

```python
# 需安装: pip install datasets
from datasets import load_dataset

ds = load_dataset("imdb", split="train[:100]")  # 取前 100 条
ds = ds.map(lambda x: {"len": len(x["text"].split())}, batched=False)
ds = ds.filter(lambda x: x["len"] > 10)
# set_format("torch", columns=...) 在列已是数值/tensor 时使用，便于进 DataLoader
# 划分
splits = ds.train_test_split(test_size=0.2, seed=42)
train_ds, test_ds = splits["train"], splits["test"]
```

---

## 六、领域库数据集

### 6.1 结构说明

**torchvision** / **torchaudio** 等均继承 `torch.utils.data.Dataset`，在基类上增加**领域 transform** 与标准数据接口：

- **torchvision**：`datasets.CIFAR10`、`ImageFolder` 等；transform 为 PIL → Tensor 或增强（Resize、Normalize、RandomCrop 等）；`transforms.Compose` 串联。
- **torchaudio**：`datasets.LIBRISPEECH` 等；transform 为波形/频谱（Resample、MelSpectrogram 等）。
- **torchtext**：文本 pipeline，逐渐被 HuggingFace 替代。

`transform` 在 `__getitem__` 中对**单样本**执行；与 DataLoader 的 collate_fn 无关。

### 6.2 典型用法示例（可运行）

```python
import torch
from torchvision import datasets, transforms

# 使用 CIFAR10（需能访问或已下载）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# root 需存在或留空并允许下载
# cifar = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
# 仅演示结构：用 TensorDataset 模拟「图像+标签」
X = torch.rand(100, 3, 32, 32)
y = torch.randint(0, 10, (100,))
ds = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
batch = next(iter(loader))
print(batch[0].shape, batch[1].shape)  # torch.Size([8, 3, 32, 32]), torch.Size([8])
```

`ImageFolder`：目录结构为 `root/class_a/xx.png`，自动按子目录名作标签，配合 `transform` 即可用于分类。

---

## 七、典型项目扩展模式

### 7.1 结构/边界说明

许多项目会在 `torch.utils.data.Dataset` 或 PyG/HuggingFace 之上再封一层，以统一**多任务、多数据源、比例采样、padding 与 collate**。本节描述**通用模式**；若你的项目中有类似 `BaseDataset`、`RLDataset`、`SplitMergeDataset`、`GTDataset` 等，可对应下列概念与调用关系。

- **BaseDataset 系**：从配置（如 `code_train`）与 pkl 文件列表构建；产出「dict of tensors + labels」；支持多任务比例采样（通过 Sampler 或自定义 DataLoader 逻辑）。
- **RLDataset**：在 BaseDataset 思路上增加轨迹/步采样、专用 collate（如按步长 pad）。
- **SplitMergeDataset**：多数据源按比例或顺序合并，常与 `ConcatDataset` 或自定义 Sampler 配合。
- **GTDataset**：借 PyG 的 `Dataset` 外壳（目录约定、`len`/`get`），但数据来自 pkl 路径列表、按需加载并做 padding，与 BaseDataset 产出形态类似，便于与既有 PyG 风格脚本统一。

### 7.2 关键概念与调用关系

- **单条形态**：通常为 `dict of tensors + labels`，便于 collate_fn 统一拼成 batch（如 `default_collate` 或自定义 `gt_collate_fn`）。
- **Padding**：变长序列/图在 **Dataset 的 `__getitem__`/`get()`** 内对单条做 pad，或留到 **collate_fn** 对 list 做 pad；两者分工见下一节。
- **与 DataLoader 的衔接**：Dataset 只负责「单条样本」；Sampler 负责「取哪些、顺序」；collate_fn 负责「多条 → batch」。

### 7.3 适用场景

- 多任务驾驶/仿真：多 pkl 源、任务比例、统一入口。
- 需要 PyG 风格目录与 DataLoader 时：用 GTDataset 封装 pkl 列表，配合 PyG DataLoader 或自定义 collate。

---

## 八、Padding 与 collate 协作

### 8.1 职责划分

- **单条 padding（在 Dataset 内）**：在 `__getitem_`_ 或 PyG 的 `get()` 中，对单条样本做长度对齐（如 pad 到固定长度或到本 batch 内最大长度前的某个上界）。适合「每条独立 pad 到同一规则」的场景。
- **batch 内 padding（在 collate_fn 内）**：DataLoader 收集到 list of 样本后，在 collate_fn 里按本 batch 最大长度做 pad，再 stack 成 tensor。更灵活，且只 pad 到当前 batch 所需长度，节省显存。

两者可组合：例如在 Dataset 里做简单截断/最小 pad，在 collate_fn 里再做 batch 维度的 pad 与 mask。

### 8.2 关键概念

- **为何需要 padding**：变长序列/图无法直接 `torch.stack` 成规则 tensor；padding 后得到固定形状，便于模型前向。
- **mask**：通常与 padding 一起维护一个 mask 张量（如 1 为有效、0 为 pad），供模型忽略 pad 位置。

### 8.3 典型 collate_fn 模式（可运行）

```python
import torch

def pad_collate(batch):
    """batch: list of (seq, label), seq 为 1D tensor 变长"""
    seqs = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s
        mask[i, :s.size(0)] = True
    return padded, mask, labels

# 使用示例
data = [(torch.randn(i + 1), torch.tensor(i % 2)) for i in range(5)]
padded, mask, labels = pad_collate(data)
print(padded.shape, mask.shape, labels.shape)  # [5, 5], [5, 5], [5]
```

---

## 九、与 DataLoader 的协作

### 9.1 数据流与职责

- **Dataset**：提供 `__len__` 与 `__getitem__`（或 IterableDataset 的 `__iter__`），定义「单条样本」形态。
- **DataLoader**：根据 Sampler 产生索引（或直接消费迭代器），在 worker 中取样本，经 **collate_fn** 聚成 batch，可选多进程与 pin_memory。

因此：**Dataset 决定单条格式；collate_fn 决定多条如何成 batch；Sampler 决定取哪些、顺序。**

### 9.2 关键衔接点

- **map-style**：DataLoader 调用 `len(dataset)` 与 `dataset[indices]`（或 `__getitems__(indices)`）；collate_fn 的输入是「list of 单条样本」。
- **IterableDataset**：无索引，DataLoader 从 `iter(dataset)` 取数据；多 worker 时需在 dataset 或 `worker_init_fn` 中分片，避免重复。
- **自定义 batch**：若需自定义每批的组成（如加权、多任务比例），可传 `batch_sampler`，或使用 `Sampler` 控制索引。

