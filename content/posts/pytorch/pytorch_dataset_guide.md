# PyTorch Dataset 体系技术文档

## 文档索引


| 章节                                                                    | 主题                                                       | 主要内容                                                              |
| --------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------- |
| [一、总览](#一总览)                                                          | 整体架构与文档脉络                                                | 知识结构、各部分作用与关联、优缺点与适用场景                                            |
| [二、torch.utils.data.Dataset](#二torchutilsdatadataset--根基)             | PyTorch 数据集根基                                            | 抽象基类协议、`__len__`/`__getitem__`、与 DataLoader 的契约                   |
| [三、内置扩展](#三torchutilsdata-内置扩展)                                       | TensorDataset / ConcatDataset / Subset / IterableDataset | 结构、使用方式、示例与注意点                                                    |
| [四、torch_geometric.data.Dataset](#四torch_geometricdatadataset--图数据专用) | 图数据专用 Dataset                                            | 文件管理、三阶段 transform、Dataset vs InMemoryDataset                     |
| [五、HuggingFace datasets](#五huggingface-datasetsdataset--独立体系)         | 独立数据体系                                                   | Arrow、map/filter、set_format 与 PyTorch 互转                          |
| [六、领域库数据集](#六领域库数据集)                                                  | torchvision / torchaudio                                 | transform 管线、典型用法                                                 |
| [七、本项目数据集体系](#七本项目数据集体系)                                              | BaseDataset / RLDataset / SplitMergeDataset / GTDataset  | 模块划分、关键概念、完整接口与调用关系                                               |
| [八、Padding 与工具模块](#八padding-与工具模块)                                    | padding.py、dataset_util.py                               | 各 padding 函数作用、工具函数、与 Dataset 的协作                                 |
| [九、与 DataLoader 的协作](#九与-dataloader-的协作)                              | Dataset → DataLoader → 训练                                | collate_fn、多进程、与 [dataloader_guide.md](./dataloader_guide.md) 的衔接 |


---

## 一、总览

### 1.1 整体架构与知识结构

PyTorch 的数据集体系以 `**torch.utils.data.Dataset**` 为根基，向上衍生出多条线：

```
torch.utils.data.Dataset              ← 所有 PyTorch 系的根基（map-style）
    ├── TensorDataset                  包装多个等长 tensor
    ├── ConcatDataset                  逻辑拼接多个数据集
    ├── Subset                         按索引取子集（train/val split）
    ├── IterableDataset                流式数据，另一条线，不支持随机访问
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
| **IterableDataset**        | 流式、无索引                            | 大文件、管道、无法事先知道总条数的场景                              |
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
| **PyG Dataset**          | 标准化的下载/process/transform | 需实现 len/get 与文件列表              | 图结构、需持久化处理结果                 |
| **HuggingFace datasets** | map/filter 强、Arrow 大表友好  | 与 PyTorch 无继承关系                | NLP/表格、Hub 数据、大批量预处理         |
| **本项目 BaseDataset**      | 多任务、比例采样、与训练脚本深度集成       | 依赖配置与 pkl 目录结构                 | 多任务驾驶模型训练与仿真                 |
| **本项目 GTDataset**        | 与 PyG 生态一致、接口简单          | 未用 PyG 的 process()，本质仍是 pkl 列表 | 需要 DataLoader + PyG 风格时的数据入口 |


---

## 二、`torch.utils.data.Dataset` — 根基

### 2.1 结构与边界

所有 PyTorch map-style 数据集的抽象基类，**只定义两个协议**，不包含任何文件管理、transform 或下载逻辑：

```python
class Dataset:
    def __len__(self) -> int:
        """返回数据集大小，DataLoader 用其决定迭代次数与采样范围"""
        raise NotImplementedError

    def __getitem__(self, idx):
        """按整数索引返回单个样本，格式不限（tensor / dict / tuple / 自定义对象）"""
        raise NotImplementedError
```

**与 DataLoader 的契约**：DataLoader 通过 `len(dataset)` 得到长度，通过 `dataset[i]` 取样本；索引 `i` 由 Sampler 提供，通常为 `0 .. len(dataset)-1` 的某种排列。

### 2.2 关键概念

- **map-style**：有固定长度、支持整数索引，可随机访问；与 **IterableDataset**（仅 `__iter__`）相对。
- **为何只需这两个方法**：DataLoader 只关心「有多少条」和「第 i 条是什么」，其余（存储、增强、解码）都由子类在 `__getitem__` 内完成，保持接口最小且稳定。

### 2.3 使用方式与示例

继承并实现 `__len__` 与 `__getitem__` 即可接入 DataLoader；样本格式由业务决定，collate_fn 需与之匹配。

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

ds = MyDataset(torch.randn(100, 32), torch.randint(0, 10, (100,)))
print(len(ds))         # 100
print(ds[0][0].shape)  # torch.Size([32])
```

---

## 三、`torch.utils.data` 内置扩展

### 3.1 结构说明

基于 `Dataset`，PyTorch 提供四类开箱即用组件：


| 类                   | 用途                  | 核心接口/行为                                                         |
| ------------------- | ------------------- | --------------------------------------------------------------- |
| **TensorDataset**   | 包装多个第 0 维等长的 tensor | `__getitem__(idx)` 返回 `(tensors[0][idx], tensors[1][idx], ...)` |
| **ConcatDataset**   | 逻辑拼接多个 Dataset      | 索引连续映射到各子集，透明访问                                                 |
| **Subset**          | 按索引列表取子集            | 常用于 train/val split，支持重复索引                                      |
| **IterableDataset** | 流式数据                | 实现 `__iter__`，无 `__len__`/`__getitem__`，DataLoader 行为不同         |


### 3.2 关键概念与注意点

- **TensorDataset**：所有传入 tensor 的第 0 维必须相同；适合已在内存的简单特征+标签。
- **ConcatDataset**：子集类型可不同；索引自动偏移，无数据拷贝。
- **Subset**：`indices` 可为任意整数列表；不拷贝数据，只做索引映射。
- **IterableDataset**：不能 shuffle（需在数据源侧打乱）；多 worker 时需在 `__iter__` 内用 `get_worker_info()` 做分片，否则每个 worker 会重复全量数据。

### 3.3 示例

```python
from torch.utils.data import TensorDataset, ConcatDataset, Subset, IterableDataset
import torch

# TensorDataset
X = torch.randn(200, 16)
y = torch.randint(0, 5, (200,))
tensor_ds = TensorDataset(X, y)
print(tensor_ds[0])  # (tensor([...]), tensor(3))

# ConcatDataset
combined = ConcatDataset([tensor_ds, tensor_ds])
print(len(combined))  # 400

# Subset：train/val
n = len(tensor_ds)
train_ds = Subset(tensor_ds, range(0, int(n * 0.8)))
val_ds   = Subset(tensor_ds, range(int(n * 0.8), n))

# IterableDataset：按行读文件，多 worker 分片
class LineDataset(IterableDataset):
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with open(self.filepath) as f:
            for i, line in enumerate(f):
                if worker_info is None or i % worker_info.num_workers == worker_info.id:
                    yield line.strip()
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


---

## 五、HuggingFace `datasets.Dataset` — 独立体系

与 PyTorch 无继承关系；底层 Apache Arrow，支持 mmap、大表。常用能力：

- **load_dataset()**：从 Hub 或本地加载。  
- **.map(fn, batched=True)**：并行预处理（如 tokenize）。  
- **.filter(fn)**：按条件筛样本。  
- **.set_format("torch")**：输出转为 torch tensor，可直接进 DataLoader。  
- **.train_test_split()**：划分 train/test。

使用要点：`map(..., batched=True)` 比逐条快很多；`set_format("torch")` 后无需再包一层 Dataset。

---

## 六、领域库数据集

**torchvision** / **torchaudio** 等均继承 `torch.utils.data.Dataset`，在基类上增加领域 transform：

- **torchvision**：PIL/张量变换（Resize、Normalize、RandomCrop 等）；`ImageFolder` 按目录建分类数据集。  
- **torchaudio**：波形/频谱（Resample、MelSpectrogram 等）。  
- **torchtext**：文本 pipeline（逐渐被 HuggingFace 替代）。

`transform` 在 `__getitem`__ 中对单样本执行；可用 `transforms.Compose` 串联。

