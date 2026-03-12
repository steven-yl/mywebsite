# PyTorch DataLoader 技术解读

## 文档索引

| 章节 | 主题 | 主要内容 |
|------|------|----------|
| [一、总览](#一总览) | 整体架构与文档脉络 | 知识结构、数据流、各部分职责与关联、优缺点与适用场景 |
| [二、与 Dataset 的接口](#二与-dataset-的接口) | DataLoader 与数据源的契约 | map-style / IterableDataset、`__len__`/`__getitem__`/`__iter__`、ConcatDataset/Subset |
| [三、Sampler 与索引流](#三sampler-与索引流) | 取哪些、按何顺序 | Sampler 协议、Random/Sequential/BatchSampler、batch_sampler、DistributedSampler |
| [四、batch 与 collate_fn](#四batch-与-collate_fn) | 聚批与默认/自定义 collate | batch_size、drop_last、default_collate 行为、自定义 collate、padding |
| [五、多进程 Worker 与传输优化](#五多进程-worker-与传输优化) | 加速取数与 CPU→GPU | num_workers、worker_init_fn、persistent_workers、prefetch_factor、pin_memory、multiprocessing_context |
| [六、迭代行为与使用方式](#六迭代行为与使用方式) | 遍历与长度 | `for batch in loader`、`iter`/`next`、`len(loader)`、多 epoch 注意点 |
| [七、本项目中的用法](#七本项目中的用法) | pl_train / GTDataset / gt_collate_fn | 配置、ConcatDataset、Hydra instantiate、与 [pytorch_dataset_guide.md](./pytorch_dataset_guide.md) 衔接 |
| [八、小结与速查](#八小结与速查) | 汇总与速查表 | 模块对照表、参数速查、数据流一句话 |

---

## 一、总览

### 1.1 DataLoader 的职责与位置

`torch.utils.data.DataLoader` 是 PyTorch 中**将「数据集」变为「可按 batch 迭代的输入流」**的组件。它不定义数据从哪来、长什么样（由 **Dataset** 负责），只负责：

- **按何种顺序、每次取多少条**：Sampler / batch_size / batch_sampler  
- **多条样本如何拼成一个 batch**：collate_fn  
- **是否用多进程预取、是否锁页内存**：num_workers、pin_memory、persistent_workers、prefetch_factor  

在训练/验证流程中的位置：

```
配置/代码 → 构建 Dataset（如 GTDataset、ConcatDataset）
         → 传入 DataLoader（batch 大小、采样、collate_fn、多进程等）
         → 训练循环：for batch in train_dataloader: loss = model(batch) ...
```

### 1.2 整体架构与知识结构

DataLoader 内部可拆成**五条逻辑线**，同一次迭代中的关系如下：

```
                    ┌─────────────────────────────────────────────────────────────────┐
                    │                         DataLoader                                │
                    │                                                                   │
  ┌──────────────┐  │  ① 索引流          ② 取样本              ③ 成 batch             │
  │   Dataset    │  │  Sampler      →    Worker 取        →    collate_fn     → 输出   │
  │ (数据源)     │  │  (顺序/批次)      dataset[i]             (list→batch)          │
  └──────┬───────┘  │      │                  │                       │                │
         │          │      │                  │                       │                │
         │          │  ┌───▼───┐         ┌────▼────┐             ┌────▼────┐          │
         └──────────┼─►│indexes│────────►│ samples │─────────────►│  batch  │──────────┼──► 训练/验证
                    │  │0,3,1..│         │ (单条)  │             │(tensor  │          │
                    │  └───────┘         └─────────┘             │ or dict) │          │
                    │       ▲                 ▲                  └─────────┘          │
                    │  batch_size        num_workers                                  │
                    │  shuffle           pin_memory                                   │
                    └─────────────────────────────────────────────────────────────────┘
```

- **① 索引流**：Sampler（或由 shuffle 推导的 RandomSampler/SequentialSampler）产生索引；若指定 `batch_sampler` 则直接产生「一批索引」，此时忽略 `batch_size` / `shuffle`。  
- **② 取样本**：主进程或 Worker 根据索引调用 `dataset[idx]`（或 IterableDataset 的迭代），得到多条单样本。  
- **③ 成 batch**：`collate_fn` 把「list of 单样本」聚合成一个 batch（如 stack 成 tensor、或 dict of tensors），供模型使用。

因此：**Dataset 决定「单条是什么」；Sampler 决定「取哪些、按什么顺序」；collate_fn 决定「多条怎么变成 batch」；Worker 与 pin_memory 等决定「取数据与传输的效率」。**

### 1.3 各部分职责与彼此关联

| 部分 | 职责/主题 | 与其它部分的关系 |
|------|-----------|------------------|
| **Dataset 接口** | 数据从哪来、单条长什么样 | DataLoader 只依赖「长度 + 按索引取一条」或「可迭代流」；格式、增强、IO 全由 Dataset 负责 |
| **Sampler** | 取哪些索引、顺序 | 产出索引序列（或一批批索引）；不接触实际数据，只驱动「谁被取」 |
| **batch / collate_fn** | 多少条一批、如何聚批 | Worker 取到的是 list of 单样本；collate_fn 把 list 变成模型可用的 batch |
| **多进程与传输** | 加速取数、加速 CPU→GPU | Sampler 产索引 → 在 worker 中执行 dataset[idx] → 预取若干批 → 主进程 collate，可选 pin_memory |
| **迭代** | 如何遍历 batch | 对上述链条的封装：`for batch in loader` 即重复「取索引→取样本→collate→产出」 |

### 1.4 优缺点与适用场景对比

| 选项/场景 | 优点 | 缺点 / 注意点 | 适用场景 |
|-----------|------|----------------|----------|
| **map-style Dataset + DataLoader** | 支持随机访问、shuffle、多进程索引分片 | 需事先能确定 `__len__` | 绝大多数训练/验证 |
| **IterableDataset + DataLoader** | 流式、不必装全量进内存 | 不能使用 `shuffle=True`（需在 dataset 内自行打乱），多进程需自行分片 | 大文件、管道、无法事先知道总条数 |
| **shuffle=True（默认 RandomSampler）** | 每 epoch 随机顺序，利于泛化 | 顺序不可复现（需设 seed） | 训练集 |
| **shuffle=False（默认 SequentialSampler）** | 顺序固定、可复现 | 不适合训练 | 验证/测试 |
| **自定义 sampler** | 可做加权、子集、分布式分片 | 与 shuffle 互斥，需自己保证不重复不遗漏 | 多任务比例采样、DDP、子集验证 |
| **batch_sampler** | 完全控制每批的索引列表 | 忽略 batch_size、shuffle、sampler、drop_last | 自定义批组成、某些分布式策略 |
| **num_workers=0** | 无多进程开销、调试简单 | 取数在主进程，可能拖慢 GPU | 小数据、调试 |
| **num_workers>0** | 取数与训练并行，减轻 GPU 等待 | 进程切换与内存占用；Windows 需 `if __name__=="__main__"` | 大数据、生产训练 |
| **pin_memory=True** | 加速 CPU→GPU 拷贝 | 仅在与 `.to(device, non_blocking=True)` 配合时效果明显 | CUDA 训练 |
| **persistent_workers=True** | epoch 间不销毁 worker，减少 fork 开销 | 首 epoch 后不会重新初始化 worker | 多 epoch 训练 |

---

## 二、与 Dataset 的接口

### 2.1 结构说明

DataLoader 的第一个核心参数是 `dataset`，类型为：

- **`torch.utils.data.Dataset`**（map-style）：需实现 `__len__` 与 `__getitem__(idx)`，支持整数索引。  
- **`torch.utils.data.IterableDataset`**：只需实现 `__iter__`，无索引；DataLoader 从迭代器取样本并按 batch_size 聚批。

DataLoader 在迭代时**只通过「索引」或「迭代器」**与 dataset 交互，不关心 dataset 内部如何存数据、是否做 transform。

- 对 **Dataset**：内部用 Sampler 得到索引序列，再在某个进程里执行 `dataset[i]` 得到单条样本。  
- 对 **IterableDataset**：由 dataset 自己产出样本流，DataLoader 负责按 `batch_size` 聚批并调用 `collate_fn`；此时不能使用 `shuffle=True`（需在 dataset 内自行打乱），且 Sampler 行为受限。

### 2.2 关键概念

- **map-style**：有长度、可随机访问。适合绝大多数「已知条数、需 shuffle」的场景。  
- **IterableDataset**：无长度（或长度仅作提示）、只能顺序迭代。适合流式数据、大文件、无法事先知道总条数的场景。  
- **为什么需要两种**：map-style 便于 Sampler 任意排列索引；Iterable 便于与管道、生成器、分布式按 worker 分片对接。

### 2.3 使用方式

- 使用 **map-style Dataset** 时：必须实现 `__len__` 和 `__getitem__(idx)`，且 `idx` 为整数；DataLoader 会用 Sampler 给出的索引调用 `__getitem__`。  
- 使用 **IterableDataset** 时：只需实现 `__iter__`；多进程时每个 worker 会得到同一迭代器的一份拷贝，若需分片需在 dataset 内根据 `worker_info` 自行划分，避免重复数据。  
- **ConcatDataset**、**Subset** 等包装多个 Dataset，再整体传给 DataLoader，无需改 DataLoader 调用方式；长度与索引映射由这些包装类负责。

### 2.4 示例代码

```python
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class SimpleDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.randn(size, 8)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]  # 单条 shape: (8,)

ds1 = SimpleDataset(100)
ds2 = SimpleDataset(50)
combined = ConcatDataset([ds1, ds2])  # 长度 150，索引 0..99 来自 ds1，100..149 来自 ds2

loader = DataLoader(combined, batch_size=16, shuffle=True)
batch = next(iter(loader))
print(batch.shape)  # torch.Size([16, 8])
```

---

## 三、Sampler 与索引流

### 3.1 结构说明

Sampler 决定**每次迭代时，以什么顺序、用哪些索引**访问 Dataset。协议为：实现 `__iter__`，产出索引序列（整数）；可选实现 `__len__`（返回索引总数，用于 `len(loader)` 等）。

DataLoader 内部逻辑可简化为：

- **未指定 `batch_sampler`**：使用 `sampler`（或由 `shuffle` 推导出的 RandomSampler/SequentialSampler）得到整集索引序列，再按 `batch_size` 切分成一批批索引，每批交给 Worker 取样本，最后经 `collate_fn` 成 batch。  
- **指定了 `batch_sampler`**：直接使用它产出「一批索引」（每个元素为 index 的 list），此时**忽略** `batch_size`、`shuffle`、`sampler`、`drop_last`。

### 3.2 常用 Sampler 与关系

| 类型 | 作用 | 典型用法 |
|------|------|----------|
| **SequentialSampler** | 顺序 0,1,...,N-1 | 验证集默认（shuffle=False） |
| **RandomSampler** | 随机打乱索引（可带 replacement） | 训练集 shuffle=True |
| **BatchSampler** | 在某个 Sampler 上再按 batch 切分 | 自定义每批索引、与其它 Sampler 组合 |
| **DistributedSampler** | 多卡下每卡不同子集，不重复 | DDP 训练 |
| **SubsetRandomSampler** | 仅从给定索引中随机采样 | 子集训练、部分验证 |

索引流 → 取样本 → collate 的链条中，**Sampler 只负责「索引」这一环**，不接触实际数据。

### 3.3 关键概念

- **为什么需要 Sampler**：将「取哪些、按什么顺序」从 Dataset 中解耦，便于实现 shuffle、加权、分布式、子集等，而不必改 Dataset。  
- **batch_sampler 与 sampler 互斥**：指定 `batch_sampler` 后，DataLoader 不再使用 `sampler`、`batch_size`、`shuffle`、`drop_last`；若只指定 `sampler`，则仍用 `batch_size` 在内部做批切分。

### 3.4 使用方式

- 不传 `sampler` / `batch_sampler` 时：`shuffle=True` → 内部使用 `RandomSampler(dataset)`；`shuffle=False` → `SequentialSampler(dataset)`。  
- 传了 `sampler` 后，**不能再传 `shuffle`**（会冲突）。  
- DDP 时通常为训练集传 `DistributedSampler`，并在每 epoch 前调用 `sampler.set_epoch(epoch)` 以保证各 epoch 打乱不同。

### 3.5 示例代码

```python
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler, SubsetRandomSampler

dataset = SimpleDataset(100)

# 等价：不传 sampler，shuffle=False → 顺序
loader_seq = DataLoader(dataset, batch_size=8, shuffle=False)
loader_seq2 = DataLoader(dataset, batch_size=8, sampler=SequentialSampler(dataset))

# 训练常用：随机打乱
loader_shuffle = DataLoader(dataset, batch_size=8, shuffle=True)

# 自定义：只取前 50 个样本，且每批 4 条
indices = list(range(50))
sampler = SubsetRandomSampler(indices)
batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
loader_custom = DataLoader(dataset, batch_sampler=batch_sampler)
# 使用 batch_sampler 时不能再传 batch_size、shuffle、sampler
```

---

## 四、batch 与 collate_fn

### 4.1 结构说明

- **batch_size**：每批包含的样本数（若使用 `batch_sampler` 则无效）。  
- **drop_last**：若最后不足一整批，是否丢弃该批；`True` 时 `len(loader) == floor(len(dataset)/batch_size)`，常用于训练时保持每批大小一致（如 BN）。  
- **collate_fn**：签名为 `(list of 单样本) -> batch`。Worker 取到的是「一批索引对应的多条单样本」的 list；collate_fn 把这条 list 变成「一个 batch 对象」，训练代码拿到的就是其返回值。

### 4.2 默认 collate_fn 行为

未传 `collate_fn` 时，DataLoader 使用 `torch.utils.data.default_collate`：

- 若单样本为 **tensor**：对 list 做 `torch.stack(batch)`。  
- 若单样本为 **tuple/list**：按元素位置分别 stack（要求各位置类型一致、可 stack）。  
- 若单样本为 **dict**：按 key 分别对 value 做 default_collate（递归）。  
- 若为 **数字**：转为 tensor。  
- **不可 stack 的类型**（如不等长序列、str、自定义对象）：会报错或行为未定义，此时需自定义 collate_fn。

### 4.3 关键概念

- **为什么需要 collate_fn**：单条样本可能是 tensor、tuple、dict 或不等长序列；模型需要的是固定形状的 batch（如 `(B, ...)` 的 tensor 或 dict of tensors）。collate_fn 统一完成「多条 → 一批」的转换，并可在此做 padding、mask 等。  
- **padding 放在哪**：可在 Dataset 的 `__getitem__` 里对单条做固定长度 padding，或在 collate_fn 里按本批最大长度做 padding；后者更省内存、更灵活。

### 4.4 使用方式

- 单样本为等长 tensor 或简单 tuple/list/dict of tensors：可不传 `collate_fn`，用默认即可。  
- 单样本为不等长序列、或需要 padding/掩码：必须自定义 `collate_fn`，返回一个 batch（例如 `{"x": padded_tensor, "mask": mask_tensor}`）。  
- `drop_last=True` 时，最后一个不完整 batch 不会出现。

### 4.5 示例代码

```python
def simple_collate(batch):
    """batch: list of (tensor,) or (tensor, label)"""
    if isinstance(batch[0], tuple):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.tensor([b[1] for b in batch])
        return xs, ys
    return torch.stack(batch)

# 不等长序列：padding 到本批最大长度
def pad_collate(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch])
    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros(len(seqs), max_len)
    for i, s in enumerate(seqs):
        padded[i, :s.size(0)] = s
    return padded, labels

loader_default = DataLoader(dataset, batch_size=16)  # 默认 stack
loader_custom = DataLoader(dataset, batch_size=16, collate_fn=simple_collate)
```

---

## 五、多进程 Worker 与传输优化

### 5.1 结构说明

- **num_workers**：用于加载数据的子进程数。0 表示主进程加载；>0 时，主进程只负责组 batch 和送进 GPU，取样本在 worker 进程里执行，通过队列把样本传给主进程。  
- **worker_init_fn**：每个 worker 启动时调用一次，可用于设随机种子、改线程数等，签名为 `(worker_id: int) -> None`。  
- **persistent_workers**：若 True，worker 进程在 epoch 之间不销毁，避免每个 epoch 重新 fork，适合多 epoch 训练（需 num_workers>0）。  
- **prefetch_factor**：每个 worker 预取的 batch 数（仅 num_workers>0 时有效），默认 2。  
- **pin_memory**：若 True，主进程会把 CPU 上的 batch 放在锁页内存，加速 CPU→GPU 拷贝；需在 `.to(device, non_blocking=True)` 时才能更好利用。  
- **multiprocessing_context**：多进程启动方式，如 `'spawn'`、`'fork'`；Windows 上默认 spawn，Linux 上多为 fork。

关系简要：**Sampler 决定取哪些索引；这些索引在 worker 中转为 `dataset[idx]`；多个 worker 并行取，预取若干批；主进程把收到的样本 list 交给 collate_fn 成 batch，再可选地 pin_memory。**

### 5.2 关键概念

- **为什么需要多进程**：数据加载往往受 IO 或 CPU 预处理限制，多进程可让「取下一批」与「当前批在 GPU 上训练」并行，减轻 GPU 空转。  
- **为什么需要 persistent_workers**：每次 fork 会复制进程、重新导入模块，多 epoch 时重复开销大；保持 worker 存活可减少这部分成本。  
- **为什么需要 pin_memory**：锁页内存在 DMA 传输时不需要先拷到可换页内存，可减少 CPU→GPU 拷贝延迟。

### 5.3 使用方式

- Windows 上多进程需把数据/模型构建放在 `if __name__ == "__main__"` 内，避免重复 fork 出错。  
- `num_workers` 过大反而可能因进程切换和内存占用变慢，一般 4～8 常见；数据很轻时可设为 0。  
- `persistent_workers=True` 时，第一个 epoch 后不会重新创建 worker；若设为 True，则 num_workers 必须 >0。  
- `pin_memory=True` 通常与 CUDA 训练一起使用，且配合 `non_blocking=True` 的 `.to(device)`。

### 5.4 示例代码

```python
def worker_init(worker_id):
    import numpy as np
    np.random.seed(worker_id)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    worker_init_fn=worker_init,
)
```

---

## 六、迭代行为与使用方式

### 6.1 结构说明

DataLoader 是**可迭代对象**：

- **for batch in loader**：遍历所有 batch，每个 batch 是 `collate_fn` 的返回值。  
- **iter(loader)**：得到迭代器；**next(iterator)** 得到一个 batch。多 epoch 时每轮应重新 `iter(loader)` 或直接 `for batch in loader`。  
- **len(loader)**：batch 的个数。使用默认 sampler 且 `drop_last=False` 时为 `ceil(len(dataset)/batch_size)`；`drop_last=True` 时为 `floor(len(dataset)/batch_size)`。若使用 `batch_sampler`，则 `len(loader) == len(batch_sampler)`。

迭代时内部按「Sampler → 取样本 → collate」的顺序产出 batch，与前面各模块一一对应。

### 6.2 使用方式

- 训练循环中不要重复使用同一个迭代器对象跨 epoch，应每个 epoch 重新 `for batch in train_loader` 或重新 `iter(train_loader)`。  
- 若 DataLoader 使用 **IterableDataset** 且未实现 `__len__`，`len(loader)` 可能不可用或仅为估计。  
- 验证/测试时通常 `shuffle=False`，保证结果可复现。

### 6.3 示例代码

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
print("batches per epoch:", len(loader))

for epoch in range(3):
    for batch in loader:
        pass

# 或手动取一个 batch
it = iter(loader)
batch0 = next(it)
batch1 = next(it)
```

---

## 七、小结与速查

### 7.1 模块对照表

| 模块 | 作用 | 主要参数/组件 |
|------|------|----------------|
| Dataset 接口 | 数据从哪来、单条长什么样 | `dataset`（Dataset / IterableDataset）、ConcatDataset / Subset |
| Sampler | 取哪些索引、顺序 | `sampler`、`batch_sampler`、`shuffle`、RandomSampler、SequentialSampler、BatchSampler、DistributedSampler |
| batch 与 collate | 多少条一批、如何聚批 | `batch_size`、`drop_last`、`collate_fn`、default_collate、自定义 padding/mask |
| 多进程与传输 | 加速取数、加速到 GPU | `num_workers`、`worker_init_fn`、`persistent_workers`、`prefetch_factor`、`pin_memory`、`multiprocessing_context` |
| 迭代 | 如何遍历 batch | `for batch in loader`、`iter(loader)`、`next(it)`、`len(loader)` |

### 7.2 数据流一句话

**Sampler 产索引 → Worker 用索引调 Dataset 取样本 → collate_fn 把样本 list 合成 batch → 迭代输出给训练/验证循环。**

### 7.3 参数速查（DataLoader 常用）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| dataset | Dataset / IterableDataset | 必填 | 数据源 |
| batch_size | int | 1 | 每批样本数（batch_sampler 存在时无效） |
| shuffle | bool | False | 是否打乱（与 sampler 互斥） |
| sampler | Sampler | None | 索引采样器（与 shuffle 互斥） |
| batch_sampler | Sampler\[list] | None | 若指定，忽略 batch_size/shuffle/sampler/drop_last |
| num_workers | int | 0 | 加载数据的子进程数 |
| collate_fn | callable | default_collate | (list of 样本) -> batch |
| pin_memory | bool | False | 是否使用锁页内存 |
| drop_last | bool | False | 是否丢弃最后不完整批 |
| persistent_workers | bool | False | 是否在 epoch 间保持 worker 存活 |
| worker_init_fn | callable | None | 每个 worker 启动时调用 (worker_id) |
| prefetch_factor | int | 2 | 每 worker 预取批数（num_workers>0 时有效） |
| multiprocessing_context | str/callable | None | 多进程上下文，如 'spawn'/'fork' |

理解其中一环即可对应到本文相应小节进行查阅或扩展。
