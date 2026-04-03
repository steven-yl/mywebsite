---

title: "PyTorch Tensor 工具函数技术文档：创建、计算、拼接与索引"
subtitle: ""
date: 2026-03-12T00:00:00+08:00
draft: false
authors: [Steven]
description: "系统解读 PyTorch tensor工具函数：从总览到创建、形状与视图、拼接/堆叠/拆分、逐元与矩阵运算、归约、索引与高级索引、比较逻辑及 einsum 等，覆盖完整 API 与可运行示例。"
summary: "总览 PyTorch 张量运算知识结构；创建/reshape/cat/stack/split；逐元与矩阵运算；归约；索引、gather/scatter；比较与逻辑；einsum 等工具；速查与延伸。"

tags: ["PyTorch", "Deep Learning"]
categories: ["PyTorch"]
series: ["PyTorch实践指南"]
weight: 5
series_weight: 5

hiddenFromHomePage: false
hiddenFromSearch: false

## featuredImage: ""

## featuredImagePreview: ""
---

## 文档索引


| 章节                     | 主题                        | 内容概要                                                                                                                                                                                                                                                        |
| ---------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [一、总览](#一总览整体架构与知识结构)  | 整体架构与知识结构                 | 张量运算分类、各部分作用与关联、cat vs stack / view vs reshape 等对比、适用场景                                                                                                                                                                                                     |
| [二、张量创建与属性](#二张量创建与属性) | 创建与 dtype/device/shape    | tensor/zeros/ones/empty/full、arange/linspace/logspace、rand/randn、from_numpy、dtype/device/ndim/shape                                                                                                                                                         |
| [三、形状与视图](#三形状与视图)     | 变形与视图                     | view/reshape、squeeze/unsqueeze、flatten、transpose/permute、contiguous、unfold/movedim/t()、view 与 copy 区别                                                                                                                                                       |
| [四、拼接与堆叠](#四拼接与堆叠)     | cat 与 stack               | cat（沿已有维拼接）、stack（新维堆叠）、参数与约束、典型用法                                                                                                                                                                                                                          |
| [五、拆分](#五拆分)           | 沿维拆分                      | chunk、split、unbind、vsplit/hsplit/dsplit、与 cat 的互逆关系                                                                                                                                                                                                         |
| [六、逐元与广播运算](#六逐元与广播运算) | 四则与广播及更多逐元                | add/sub/mul/div、广播、log2/log10/exp2/expm1、floor_divide/div(rounding_mode)、pow/sqrt/rsqrt、minimum/maximum、lerp/frac、atan2/hypot、inplace                                                                                                                       |
| [七、矩阵与线性代数](#七矩阵与线性代数) | 矩阵乘、外积、叉积与 linalg         | matmul/mm/bmm、dot/vdot、mv、ger/outer、cross、trace、linalg.inv/solve/det/svd/cholesky/qr/eig、linalg.norm/pinv/matrix_rank、tensordot、meshgrid                                                                                                                    |
| [八、归约](#八归约)           | 聚合、累积、排序与去重               | sum/mean/max/min/prod、argmax/argmin、norm/std/var、any/all/count_nonzero、cumsum/cumprod/cummax/cummin/logcumsumexp、median/nanmedian、nanmean/nansum、quantile/logsumexp/dist、sort/topk/kthvalue、unique                                                          |
| [九、索引与切片](#九索引与切片)     | 基础与高级索引                   | 下标/切片、高级索引、masked_select/index_select/take、put_、nonzero、index_fill_、masked_fill_、masked_scatter_、index_put_                                                                                                                                                 |
| [十、高级索引写回](#十高级索引写回)   | gather/scatter            | gather、scatter/scatter_、index_add_/index_copy_、scatter_reduce_/index_reduce_、公式与示例                                                                                                                                                                          |
| [十一、比较与逻辑](#十一比较与逻辑)   | 比较、where 与数值判断            | eq/ne/gt/ge/lt/le、where、logical_and/or/not、isfinite/isnan/isinf、isclose、equal/allclose、is_tensor/is_floating_point/is_complex                                                                                                                               |
| [十二、其它工具](#十二其它工具)     | clone/expand/einsum/设备转换等 | clone、expand/repeat、narrow、einsum、roll、clamp、addcdiv/addcmul、tril/triu、renorm、设备与 NumPy 转换、复数与 torch.special、diff/broadcast_tensors/atleast_1d/bucketize/searchsorted、resize_/resize_as_、result_type/promote_types、uniform_/normal_/exponential_/bernoulli_ |
| [十三、速查与小结](#十三速查与小结)   | 速查与延伸                     | 分类速查表、**常见遗漏清单**、与 [training_solver_guide](./training_solver_guide.md)/[dataloader_guide](./dataloader_guide.md) 的衔接                                                                                                                                        |


**阅读建议**：先读总览建立「创建 → 形状 → 拼接/拆分 → 运算 → 归约 → 索引」的全局图景，再按需跳转具体 API；实现时按「先创建/再变形/再运算/再索引」顺序对照各章。

---

## 一、总览：整体架构与知识结构

### 1.1 张量工具函数在做什么

**目标**：在 PyTorch 中，张量（Tensor）是模型输入输出与中间计算的基本单位；**工具函数**负责对张量进行**创建、变形、拼接/拆分、逐元/矩阵运算、归约、索引**等，不涉及自动求导或优化器，但构成前向计算与数据准备的基础。

**分类概览**：

```
张量工具函数
├── 创建与属性     tensor(), zeros(), arange(), dtype, device, shape
├── 形状与视图     view(), reshape(), squeeze(), unsqueeze(), transpose(), permute()
├── 拼接与堆叠     cat(), stack()
├── 拆分           chunk(), split(), unbind()
├── 逐元与广播     add(), mul(), pow(), sqrt(), minimum(), maximum(), lerp(), atan2(), 四则与广播
├── 矩阵运算       matmul(), mm(), bmm(), dot(), ger(), cross(), linalg.inv(), trace()
├── 归约           sum(), mean(), max(), min(), cumsum(), cumprod(), cummax(), cummin(), logcumsumexp(), sort(), topk(), unique(), any(), all()
├── 索引与切片     下标/切片, index_select(), masked_select(), take(), put_()
├── 高级索引       gather(), scatter(), index_add_(), index_copy_()
├── 比较与逻辑     eq(), gt(), where(), logical_and()
└── 其它           clone(), expand(), repeat(), einsum(), narrow(), clamp(), addcdiv(), tril(), triu(), renorm()
```

### 1.2 各部分作用与关联


| 部分        | 作用/主题                                    | 与其它部分的关系                         |
| --------- | ---------------------------------------- | -------------------------------- |
| **创建与属性** | 从数值/序列/NumPy 得到张量，并控制 dtype/device/shape | 所有后续运算的前提；属性被 reshape/cat 等依赖    |
| **形状与视图** | 在不改数据的前提下改变逻辑形状，或压缩/增加维度                 | 为 cat/stack、矩阵乘、归约提供正确维度         |
| **拼接与堆叠** | 沿已有维拼接（cat）或沿新维堆叠（stack）                 | 与拆分互逆；常与 DataLoader 的 batch 维度配合 |
| **拆分**    | 沿某维切成多块（chunk/split）或拆成多个张量（unbind）      | 与 cat 互逆；用于序列/多尺度处理              |
| **逐元与广播** | 标量/同形/广播下的四则与函数（exp/log 等）               | 损失、归一化、掩码运算的基础                   |
| **矩阵运算**  | 向量/矩阵/批量矩阵乘与点积                           | 线性层、注意力、全连接的核心                   |
| **归约**    | 沿维或全局聚合成标量或更小形状                          | 损失标量、全局池化、统计量                    |
| **索引与切片** | 按下标/切片/布尔/索引张量取值                         | 为 gather/scatter、掩码、条件选数提供基础     |
| **高级索引**  | 按索引张量聚集（gather）或写回（scatter）              | 灵活重排、one-hot、稀疏更新                |
| **比较与逻辑** | 比较与逻辑运算，常得到布尔张量                          | 与 where、掩码、条件分支配合                |
| **其它**    | clone/expand/repeat/einsum/narrow/clamp  | 复制、扩维、爱因斯坦求和、边界裁剪等               |


### 1.3 关键对比：何时用谁

#### 1.3.1 cat 与 stack


| 维度     | torch.cat               | torch.stack                 |
| ------ | ----------------------- | --------------------------- |
| **含义** | 沿**已有**维度拼接，维数不变        | 在**新**维度上堆叠，维数 +1           |
| **形状** | 除 `dim` 外同形，`dim` 上相加   | 所有输入同形，输出多一维                |
| **典型** | 多个 batch 拼成一个大 batch    | 多个同形张量做成「批量」                |
| **示例** | (2,3) cat (2,3) → (4,3) | (2,3) stack (2,3) → (2,2,3) |


#### 1.3.2 view 与 reshape


| 维度      | view()                   | reshape()         |
| ------- | ------------------------ | ----------------- |
| **连续性** | 要求内存连续，否则先 .contiguous() | 不连续时可返回 copy，更宽松  |
| **返回值** | 恒为视图（共享存储）               | 可能视图或 copy        |
| **建议**  | 确定连续时用 view（略快）          | 不确定或跨设备时用 reshape |


#### 1.3.3 expand 与 repeat


| 维度     | expand()          | repeat() |
| ------ | ----------------- | -------- |
| **实现** | 仅改步长，不拷贝数据        | 真实复制数据   |
| **约束** | 只能把 1 扩成 n，不能 2→3 | 任意倍数     |
| **内存** | 不增显存              | 随倍数增加    |


---

## 二、张量创建与属性

### 2.1 结构说明

创建类 API 分为：**从数值/列表**、**填充常数**、**序列**、**随机**、**从 NumPy/共享内存**、**类同形状**。属性包括 dtype、device、ndim、shape、stride、is_contiguous 等。

### 2.2 从数值或列表创建

- **torch.tensor(data, dtype=None, device=None)**  
**是什么**：从 Python 列表/标量构造新张量，数据会被拷贝。  
**为什么需要**：训练中把标量损失、列表标签等转为张量以便参与计算与 backward。  
**注意**：大列表用 tensor 会慢，可考虑 as_tensor 或 from_numpy 减少拷贝。

```python
import torch
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1., 2], [3, 4]], dtype=torch.float32)
```

- **torch.as_tensor(data, dtype=None, device=None)**  
尽量共享 data 内存（若已是 tensor/ndarray 且 dtype/device 兼容则不拷贝），否则等价于 tensor。
- **torch.from_numpy(ndarray)**  
与 NumPy 共享内存，仅支持 CPU；改一边另一边会变。

### 2.3 填充创建

- **torch.zeros(size, dtype=None, device=None)**  
全 0；常用于初始化或占位。  
- **torch.ones(size, ...)**  
全 1；如初始化偏置、mask。  
- **torch.empty(size, ...)**  
未初始化，值未定义；适合紧接着就地写入（如 uniform_）。  
- **torch.full(size, fill_value, ...)**  
用标量 fill_value 填满。

```python
a = torch.zeros(2, 3)
b = torch.ones(2, 3)
c = torch.full((2, 3), 3.14)
```

### 2.4 序列创建

- **torch.arange(start=0, end, step=1, ...)**  
一维序列，不包含 end；整数或浮点。  
- **torch.linspace(start, end, steps, ...)**  
等间隔 steps 个点，包含两端。  
- **torch.logspace(start, end, steps, base=10.0, ...)**  
对数值等间隔。

```python
torch.arange(5)           # [0,1,2,3,4]
torch.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1]
```

### 2.5 随机创建

- **torch.rand(size)**
[0,1) 均匀分布。  
- **torch.randn(size)**  
标准正态。  
- **torch.randint(low, high, size)**  
离散均匀整数，区间 [low, high)。  
- **torch.randperm(n)**  
0 到 n-1 的随机排列。

### 2.6 特殊矩阵

- **torch.eye(n, m=None, ...)**  
单位矩阵；m 省略时方阵。  
- **torch.diag(input, diagonal=0)**  
提取对角线或构造对角阵：input 1D 则构造 2D 对角阵，input 2D 则取对角线为 1D。

```python
torch.eye(3)           # (3,3) 单位阵
torch.diag(torch.tensor([1.,2.,3.]))  # (3,3) 对角阵
```

### 2.7 类同形状

- **torch.zeros_like(input)**, **torch.ones_like(input)**, **torch.empty_like(input)**  
与 input 同 shape（默认同 dtype/device）。  
- **torch.full_like(input, fill_value)**  
与 input 同形并填常量 fill_value；与 full 对应。  
- **torch.complex(real, imag)**  
用实部张量 real 与虚部张量 imag（同形）构造复数张量，dtype 为 complex。

### 2.8 常用属性

- **tensor.dtype**：元素类型（如 torch.float32, torch.int64）。  
- **tensor.device**：所在设备（cpu 或 cuda:0 等）。  
- **tensor.shape** / **tensor.size()**：形状。  
- **tensor.ndim**：维数。  
- **tensor.numel()**：元素总数。  
- **tensor.stride()**：各维步长。  
- **tensor.is_contiguous()**：是否内存连续。

```python
x = torch.randn(2, 3)
print(x.shape, x.dtype, x.device, x.numel())
```

---

## 三、形状与视图

### 3.1 结构说明

变形分为：**不增删维度**（view/reshape）、**删维**（squeeze/flatten）、**增维**（unsqueeze）、**换维顺序**（transpose/permute）。视图与存储布局、contiguous 紧密相关。

### 3.2 view 与 reshape

- **tensor.view(shape)**  
***是什么**：在满足元素总数不变的前提下，把张量解释为新形状，返回视图**（共享存储）。  
**为什么需要**：线性层输入要二维 (N, C)、卷积要四维 (N,C,H,W)，需把原始维度「压平」或重组。  
**约束**：当前张量必须内存连续；否则先 `.contiguous()` 再 view。
- **tensor.reshape(shape)**  
语义同 view，但若形状兼容且不连续，PyTorch 会返回 copy 而非视图，因此更安全、略慢。

```python
x = torch.arange(6)
y = x.view(2, 3)
z = x.reshape(2, 3)
assert y.data_ptr() == x.data_ptr()
```

### 3.3 squeeze 与 unsqueeze

- **tensor.squeeze(dim=None)**  
**是什么**：去掉大小为 1 的维度；若 dim 为 None 则去掉所有 1 维。  
**为什么需要**：如 (1, C, 1, 1) 经池化后要去掉 1 维以便后续全连接。
- **tensor.unsqueeze(dim)**  
在 dim 处插入大小为 1 的维度；常用于「加 batch 维」或「加通道维」以符合广播或 API 要求。

```python
x = torch.randn(1, 3, 1, 4)
y = x.squeeze()      # (3, 4)
z = y.unsqueeze(0)   # (1, 3, 4)
```

### 3.4 flatten

- **tensor.flatten(start_dim=0, end_dim=-1)**  
从 start_dim 到 end_dim（含）拉成一维，等价于该区间内 shape 连乘；常用于全连接前把空间维压平。

```python
x = torch.randn(2, 3, 4, 5)
y = x.flatten(2)     # (2, 3, 20)
z = x.flatten()      # (120,)
```

### 3.5 transpose 与 permute

- **tensor.transpose(dim0, dim1)**  
交换两维；常用于「通道最后」与「通道在前」格式互转（如 HWC ↔ CHW）。
- **tensor.permute(dims)**  
按 dims 重排所有维度，可一次完成多维重排。

```python
x = torch.randn(2, 3, 4)  # (N, C, L)
y = x.permute(0, 2, 1)    # (N, L, C)
```

### 3.6 contiguous

- **tensor.contiguous()**  
若已连续则返回自身，否则返回一份连续拷贝；在 view/reshape 前若报错「non-contiguous」可先调用。

### 3.7 unfold 与 movedim

- **tensor.unfold(dim, size, step)**  
沿 dim 做滑动窗口，窗口大小为 size、步长为 step；输出多一维（该维长度为窗口个数）。常用于 patch 提取、局部卷积。  
- **torch.movedim(input, source, destination)**  
把 source 维移到 destination 位置；单维移动时比 permute 更直观。  
- **tensor.t()**  
仅对 2D 有效，转置的简写。  
- **torch.swapdims(input, dim0, dim1)**  
交换两维，与 transpose(dim0, dim1) 等价；语义更直观。

---

## 四、拼接与堆叠

### 4.1 边界与概念

- **torch.cat(tensors, dim=0, , out=None)**  
***是什么**：在**已有**维度 dim 上把多个张量首尾相接，输出维数与输入相同。  
**为什么需要**：把多个小 batch 拼成一个大 batch、把多段序列拼成一条等。  
**约束**：除 dim 外，其它维度形状必须相同。
- **torch.stack(tensors, dim=0, , out=None)**  
***是什么**：在**新**维度 dim 上堆叠，所有输入形状相同，输出多一维。  
**为什么需要**：把多个同形张量做成「批量」以便批量运算（如多张图 stack 成 (N,C,H,W)）。

### 4.2 使用方式与示例

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
torch.cat([a, b], dim=0)   # (4, 2)
torch.cat([a, b], dim=1)   # (2, 4)
torch.stack([a, b], dim=0) # (2, 2, 2)
```

公式（概念）：  

- cat：沿 dim，$(\ldots, L_i, \ldots)$ 与 $(\ldots, L_j, \ldots)$ → $(\ldots, L_i+L_j, \ldots)$。  
- stack：在 dim 插入新维，$(\ldots)$ 与 $(\ldots)$ → $(2, \ldots)$（2 为列表长度）。

---

## 五、拆分

### 5.1 chunk 与 split

- **torch.chunk(input, chunks, dim=0)**  
把 dim 维**均分**成 chunks 块（若不能整除则前几块多 1）；返回 tuple  of tensors。
- **torch.split(tensor, split_size_or_sections, dim=0)**  
**split_size_or_sections** 为整数时，按该大小切块（最后一块可更小）；为 list 时，按各段长度切分。  
与 cat 互逆：`torch.cat(torch.split(x, sizes, dim), dim) == x`。

```python
x = torch.arange(10).view(2, 5)
list(torch.chunk(x, 2, dim=0))   # 两个 (1,5)
list(torch.split(x, [2, 3], dim=1))  # (2,2) 与 (2,3)
```

### 5.2 unbind

- **torch.unbind(input, dim=0)**  
沿 dim 拆成多个张量，每个在 dim 上长度为 1 并去掉该维；返回 tuple。

```python
x = torch.randn(3, 4)
list(torch.unbind(x, dim=0))  # 3 个 (4,)
```

### 5.3 vsplit / hsplit / dsplit

- **torch.vsplit(input, indices_or_sections)**  
沿 dim=0 切分，等价于 `split(..., dim=0)`；2D 时按行切块。  
- **torch.hsplit(input, indices_or_sections)**  
沿 dim=1 切分；2D 时按列切块。  
- **torch.dsplit(input, indices_or_sections)**  
沿 dim=2 切分；用于 3D 及以上（如多通道特征图按通道切）。

```python
x = torch.arange(12).view(3, 4)
torch.vsplit(x, 3)   # 3 个 (1,4)
torch.hsplit(x, [2, 3])  # (3,2), (3,1), (3,1)
```

---

## 六、逐元与广播运算

### 6.1 逐元四则

- **torch.add / sub / mul / div**（或 `+ - * /`）  
逐元素运算；若一个为标量则广播到另一张量形状。

**为什么需要广播**：向量与标量运算、通道级缩放（如 (C,) 与 (N,C,H,W)）无需显式扩展，代码简洁且高效。

**广播规则**：从右对齐 shape，维数不足的左侧补 1；每维要么相等要么其一为 1。

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = a + b
d = a * 2
e = torch.randn(3)
f = a + e  # (2,3) + (3,) -> (2,3)
```

### 6.2 逐元数学函数

- **torch.exp / log / log1p / sqrt / abs**  
指数、自然对数、log(1+x)、平方根、绝对值。  
- **torch.log2(input)** / **torch.log10(input)**  
以 2、10 为底的对数；信息论、量纲常用。  
- **torch.exp2(input)** / **torch.expm1(input)**  
$2^x$、$e^x - 1$；expm1 在 x 接近 0 时数值更稳。  
- **torch.sin / cos / tan / sinh / cosh / tanh**  
三角函数与双曲函数。  
- **torch.sigmoid / tanh**  
常用激活形状。  
- **torch.round / floor / ceil / trunc**  
取整。

**用途**：激活函数、损失中的 log、归一化中的 sqrt、角度计算等。

```python
x = torch.randn(2, 3)
y = torch.exp(x)
z = torch.sigmoid(x)
```

### 6.3 幂、符号与倒数

- **torch.pow(input, exponent)**  
逐元素幂；exponent 可为标量或同形张量。  
- **torch.sqrt(input)** / **torch.rsqrt(input)**  
平方根、平方根倒数（$1/\sqrt{x}$），常用于归一化。  
- **torch.neg(input)** / **torch.reciprocal(input)**  
取负、倒数；reciprocal 在 0 处为 inf。  
- **torch.sign(input)**  
符号：正为 1、负为 -1、零为 0。  
- **torch.abs(input)**  
绝对值（已有）；**torch.positive** / **torch.negative** 恒等与取负。

```python
x = torch.tensor([1., 4., 9.])
torch.sqrt(x)      # [1, 2, 3]
torch.pow(x, 0.5)  # 同 sqrt
torch.sign(torch.tensor([-0.1, 0., 0.1]))  # [-1, 0, 1]
```

### 6.4 逐元最值、取模与插值

- **torch.minimum(input, other)** / **torch.maximum(input, other)**  
两张量逐元素取小/取大；与归约 min/max（单输入）不同。  
- **torch.fmod(input, divisor)** / **torch.remainder(input, divisor)**  
逐元素取模；fmod 符号与 input 一致，remainder 与 divisor 一致。  
- **torch.lerp(start, end, weight)**  
线性插值：`start + weight * (end - start)`；weight 可为标量或与 start 同形。  
- **torch.frac(input)**  
逐元素取小数部分（等价于 `x - trunc(x)`）。  
- **torch.heaviside(input, values)**  
阶跃：input < 0 为 0，input > 0 为 1，input == 0 取 values 对应位置。  
- **torch.floor_divide(input, other)** / **a // b**  
向下取整除法；与 trunc 向零取整不同。  
- **torch.div(input, other, rounding_mode=None)**  
rounding_mode 为 `'floor'` 或 `'trunc'` 时可控制舍入方式，替代 `/` 的默认行为。

```python
a = torch.tensor([1., 3., 2.])
b = torch.tensor([2., 1., 4.])
torch.minimum(a, b)   # [1, 1, 2]
torch.lerp(a, b, 0.5) # [1.5, 2, 3]
```

### 6.5 三角函数与反三角

- **torch.atan2(input, other)**  
逐元素 atan2(y, x)，返回值域 $(-\pi,\pi]$，常用于角度/方向。  
- **torch.hypot(input, other)**  
逐元素 $\sqrt{\texttt{input}^2 + \texttt{other}^2}$。  
- **torch.atan / asin / acos**  
反三角函数，定义域需在 [-1,1]（asin/acos）内。

### 6.6 inplace 与运算符

- **a.add_(b)**、**a.mul_(b)** 等带下划线版本为 inplace；`a += b` 等价于 `a.add_(b)`。  
- 多数逐元函数有 **out** 参数可写回指定张量，如 `torch.add(a, b, out=c)`。

---

## 七、矩阵与线性代数

### 7.1 概念与区别

- **torch.matmul(input, other)**  
通用矩阵乘：1d@1d 标量，2d@2d 矩阵乘，2d@1d 矩阵乘向量，(batch,2d)@(batch,2d) 批量乘等。  
**为什么需要**：线性层、注意力、全连接都依赖矩阵乘；matmul 统一处理多种形状。
- **torch.mm(a, b)**  
严格 2D×2D，无广播。  
- **torch.bmm(a, b)**  
批量 3D×3D，(B,m,n) @ (B,n,p) → (B,m,p)。  
- **torch.dot(a, b)**  
仅 1D，内积标量。  
- **torch.vdot(a, b)**  
1D，先展平再共轭内积（实数为普通内积）。  
- **torch.mv(matrix, vec)**  
2D×1D → 1D。

### 7.2 外积与叉积

- **torch.ger(vec1, vec2)**  
两 1D 张量外积 → 2D：`out[i,j] = vec1[i] * vec2[j]`。  
- **torch.cross(input, other, dim=-1)**  
叉积；默认最后一维为 3 维向量，沿该维做叉积。  
- **torch.outer(input, other)**  
与 ger 等价，两 1D 的外积。

```python
u = torch.tensor([1., 2., 3.])
v = torch.tensor([4., 5., 6.])
torch.outer(u, v)   # (3,3) 外积
torch.cross(u, v)   # (3,) 叉积
```

### 7.3 线性代数进阶（torch.linalg）

- **torch.trace(input)**  
方阵迹，标量。  
- **torch.linalg.inv(A)**  
方阵求逆；方程组求解用 **torch.linalg.solve(A, B)**（AX=B 求 X）。  
- **torch.linalg.det(A)**  
行列式。  
- **torch.linalg.svd(A)**  
奇异值分解，返回 U, S, Vh。  
- **torch.linalg.cholesky(A)**  
Cholesky 分解（A 需对称正定）。  
- **torch.linalg.qr(A)**  
QR 分解。  
- **torch.linalg.eig(A)** / **torch.linalg.eigh(A)**  
特征值与特征向量；eigh 用于厄米矩阵。  
- **torch.linalg.norm(input, ord=None, dim=None)**  
矩阵/向量范数：ord 为 `'fro'` 为 Frobenius 范数，`'nuc'` 为核范数，整数为对应 Lp。  
- **torch.linalg.pinv(input)**  
摩尔-彭罗斯伪逆；当矩阵非方或奇异时求「广义逆」。  
- **torch.linalg.matrix_rank(input)**  
矩阵秩（基于 SVD 的数值秩）。  
- **torch.tensordot(a, b, dims=2)**  
指定 dims 维做内积收缩：dims 为 int 表示两张量各取最后 dims 维收缩；为 (dims_a, dims_b) 表示 a 的 dims_a 维与 b 的 dims_b 维对应收缩。用于高阶张量乘法、注意力中的收缩等。  
- **torch.meshgrid(tensors, indexing='ij')**  
由一维坐标张量生成网格；返回 N 个张量，形状由各维长度决定；indexing='xy' 时前两维按笛卡尔（绘图常用），'ij' 为矩阵下标。用于生成 2D/3D 坐标网格、插值、物理场等。

```python
A = torch.randn(3, 3)
A = A @ A.T + torch.eye(3)  # 对称正定
L = torch.linalg.cholesky(A)
torch.trace(A)
```

---

## 八、归约

### 8.1 常用归约

- **sum / mean / prod**：和、平均、乘积。  
- **max / min**：最大值、最小值；可返回 (values, indices)。  
- **argmax / argmin**：最大/最小索引。  
- **norm(p=2)**：范数，默认 L2。  
- **std / var**：标准差、方差。

**dim**：沿哪一维归约，该维消失（或保留见下）。  
**keepdim=True**：保留该维为 1，便于后续广播（如对每通道求 mean 得 (1,C,1,1) 再减）。

### 8.2 逻辑归约与计数

- **torch.any(input, dim=None)** / **torch.all(input, dim=None)**  
沿 dim 做逻辑或/与归约；dim 为 None 时对全部元素，返回 bool。  
- **torch.count_nonzero(input, dim=None)**  
沿 dim（或全局）统计非零个数。

### 8.3 累积与排序类（递进/逐步聚合）

- **torch.cumsum(input, dim)** / **torch.cumprod(input, dim)**  
沿 dim 的累积和/累积积，输出形状与 input 相同；递进相加、递进相乘。  
- **torch.cummax(input, dim)** / **torch.cummin(input, dim)**  
沿 dim 的累积最大值/最小值，返回 (values, indices)；递进取大、递进取小，常用于单调序列或分段最值。  
- **torch.logcumsumexp(input, dim)**  
$\log(\sum_{k\le i} \exp(x_k))$，数值稳定的递进 log-sum-exp；用于 CRF、序列 log 域累积。  
- **torch.median(input, dim=None)** / **torch.nanmedian(input, dim=None)**  
中位数；nanmedian 忽略 NaN。  
- **torch.nanmean(input, dim=None)** / **torch.nansum(input, dim=None)**  
忽略 NaN 的均值与和；含缺失值时常用。  
- **torch.quantile(input, q, dim=None)** / **torch.nanquantile(input, q, dim=None)**  
分位数；q 为标量或 1D，输出对应分位值；nanquantile 忽略 NaN。  
- **torch.logsumexp(input, dim, keepdim=False)**  
$\log\sum\exp(x)$，数值稳定；常用于 log-softmax、对数域求和。  
- **torch.mode(input, dim=-1)**  
沿 dim 取众数，返回 (values, indices)。  
- **torch.sort(input, dim=-1, descending=False)**  
沿 dim 排序，返回 (values, indices)。  
- **torch.topk(input, k, dim=-1)**  
沿 dim 取最大的 k 个，返回 (values, indices)。  
- **torch.kthvalue(input, k, dim=-1)**  
沿 dim 取第 k 小值及索引。

```python
x = torch.tensor([1., 2., 3., 4.])
torch.cumsum(x, dim=0)   # [1, 3, 6, 10] 递进相加
v, i = torch.cummax(x, dim=0)  # values [1,2,3,4], indices [0,1,2,3]
torch.logcumsumexp(torch.randn(5), dim=0)  # (5,) 递进 log-sum-exp
v, i = torch.topk(x, 2)  # values [4,3], indices [3,2]
```

### 8.4 去重

- **torch.unique(input, sorted=True, return_inverse=False, return_counts=False)**  
展平后去重；可返回逆映射、计数。  
- **torch.unique_consecutive(input)**  
仅去掉连续重复，保持顺序。  
- **torch.dist(input, other, p=2)**  
两张量间的 p-范数距离（标量）；p=2 为欧氏距离。

```python
x = torch.tensor([1, 2, 2, 3, 1])
torch.unique(x)  # [1, 2, 3]
torch.logsumexp(torch.randn(2, 3), dim=1)  # (2,) 数值稳定
```

---

## 九、索引与切片

### 9.1 基础索引与切片

与 NumPy 类似：整数、切片 `start:stop:step`、省略号 `...`。  
**注意**：高级索引（张量索引、布尔张量）会触发复制而非视图。

```python
x = torch.randn(3, 4, 5)
x[0]        # (4, 5)
x[:, 1:3]   # (3, 2, 5)
x[..., 0]   # (3, 4)
```

### 9.2 index_select / masked_select / take

- **torch.index_select(input, dim, index)**  
在 dim 维上按 1D 长整型 index 取子张量；输出在 dim 上长度为 len(index)。
- **torch.masked_select(input, mask)**  
按与 input 同形的 bool mask 选出 True 位置元素，返回 1D（复制）。
- **torch.take(input, index)**  
把 input 视为 1D，按 index 中线性下标取值，输出形状同 index。  
- **tensor.put_(index, source, accumulate=False)**  
与 take 互逆：把 input 视为 1D，在 index 指定的线性下标位置写入 source 中对应元素；accumulate=True 时累加而非覆盖。用于按「展平下标」批量写回。
- **torch.nonzero(input)**  
返回非零元素的坐标，形状为 (N, ndim)；与布尔掩码配合取下标、或做稀疏索引。  
- **tensor.index_fill_(dim, index, value)**  
在 dim 维的 index 指定位置填 value。  
- **tensor.masked_fill_(mask, value)**  
在 mask 为 True 的位置填 value；常用于把 padding 位置设为 0 或 -inf。  
- **tensor.masked_scatter_(mask, source)**  
在 mask 为 True 的位置依次从 source 取元素写入 self；source 中参与写入的元素数需 ≥ mask 中 True 的个数。  
- **tensor.index_put_(indices, value)** / **torch.index_put(input, indices, value)**  
按多维索引元组 indices（每维一个 1D 索引张量）写入 value；适合多维高级索引写回。

```python
x = torch.randn(3, 4)
idx = torch.tensor([0, 2])
torch.index_select(x, 0, idx)   # (2, 4)
mask = x > 0
torch.masked_select(x, mask)    # 1D
idx = torch.nonzero(mask)       # (N, 2) 坐标
x.masked_fill_(~mask, -1e9)     # 非正位置填 -1e9
flat_idx = torch.tensor([0, 5, 11])
y = torch.zeros(12)
y.put_(flat_idx, torch.tensor([1., 2., 3.]))  # 按线性下标写入
```

---

## 十、高级索引写回

### 10.1 gather

- **torch.gather(input, dim, index, , sparse_grad=False)**  
**是什么**：沿 dim 按 index 张量「聚拢」元素；index 与 input 维数相同，输出形状与 index 相同。  
**公式**（以 dim=0 为例）：`out[i][j][k] = input[index[i][j][k]][j][k]`。  
**典型用途**：从 logits 中按 target 取对应类别的 logit（如 CE 的 -log(p_true)）、one-hot 式取值。

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
idx = torch.tensor([[0, 1, 0], [2, 0, 1]])
torch.gather(x, 0, idx)  # [[1,5,3], [7,2,6]]
```

### 10.2 scatter 与 scatter_

- **tensor.scatter_(dim, index, src)**  
**是什么**：gather 的逆——把 src 按 index 写到 tensor 的 dim 维对应位置。  
**公式**（dim=0）：`self[index[i][j][k]][j][k] = src[i][j][k]`。  
**用途**：one-hot 构造、稀疏更新、将类别下标转为 one-hot 嵌入。

```python
x = torch.zeros(3, 4)
idx = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2], [2, 0, 1, 1]])
src = torch.ones(3, 4)
x.scatter_(0, idx, src)
```

### 10.3 index_add_ / index_copy_

- **index_add_(dim, index, source)**：按 index 在 dim 上把 source 加到 self。  
- **index_copy_(dim, index, source)**：按 index 在 dim 上把 source 拷贝到 self。

### 10.4 scatter_reduce_ 与 index_reduce_

- **tensor.scatter_reduce_(dim, index, src, reduce='sum'|'mean'|'amax'|'amin')**  
与 scatter_ 类似，但同一下标被多次写入时按 reduce 归约（求和、平均、取大、取小）；适合稀疏梯度聚合等。  
- **tensor.index_reduce_(dim, index, source, reduce, include_self=True)**  
沿 dim 按 index 将 source 归约写回 self；reduce 同上；include_self 控制是否先保留 self 再与 source 归约。

---

## 十一、比较与逻辑

### 11.1 比较

- **eq / ne / gt / ge / lt / le**（或 `== != > >= < <=`）  
逐元素比较，返回 bool 张量；可与 where 结合做条件赋值。

### 11.2 where

- **torch.where(condition, x, y)**  
condition 为 True 取 x，否则取 y；x/y 可广播到同一形状。  
**用途**：裁剪、阈值、条件分支（如梯度裁剪、ReLU 式选择）。

```python
a = torch.randn(2, 3)
b = torch.where(a > 0, a, torch.zeros_like(a))
```

### 11.3 logical_and / logical_or / logical_not

- **torch.logical_and / logical_or / logical_not / logical_xor**  
逐元素逻辑运算，常用于组合多个条件掩码。

### 11.4 数值与类型判断

- **torch.isfinite(input)** / **torch.isinf(input)** / **torch.isnan(input)**  
逐元素判断是否有限、无穷、NaN；调试与掩码常用。  
- **torch.isposinf(input)** / **torch.isneginf(input)**  
是否为正无穷、负无穷。  
- **torch.isclose(a, b, rtol=1e-5, atol=1e-8)**  
逐元素判断是否「近似相等」：`|a - b| ≤ atol + rtol * |b|`。  
- **torch.isreal(input)**  
逐元素是否实数（复数时虚部为 0 则为 True）。  
- **torch.equal(a, b)**  
若 a、b 同形且逐元相等则返回 True（单布尔）；与 `(a == b).all()` 等价但更高效。  
- **torch.allclose(a, b, rtol=1e-5, atol=1e-8)**  
整体是否「近似相等」：所有元素满足 isclose 则 True；常用于数值测试。  
- **torch.is_tensor(obj)**  
是否为 PyTorch 张量。  
- **tensor.is_floating_point()** / **tensor.is_complex()**  
元素类型是否为浮点或复数；用于分支选择（如是否做归一化）。

```python
x = torch.tensor([1.0, float('nan'), float('inf')])
torch.isfinite(x)  # [True, False, False]
torch.isnan(x)     # [False, True, False]
torch.isclose(torch.tensor(1.0), torch.tensor(1.0 + 1e-6))  # True
torch.equal(torch.ones(2, 3), torch.ones(2, 3))  # True
torch.allclose(torch.randn(3), torch.randn(3) + 1e-7)  # 可能 True
```

---

## 十二、其它工具

### 12.1 clone / expand / repeat

- **tensor.clone()**：拷贝一份，新存储，梯度可独立回传。  
- **tensor.expand(*sizes)**：仅把大小为 1 的维扩展，不拷贝数据。  
- **tensor.repeat(*repeats)**：按 repeats 各维重复，真实复制。

```python
x = torch.randn(1, 3)
x.expand(4, 3)   # (4,3) 视图式
x.repeat(4, 2)   # (4, 6) 复制
```

### 12.2 narrow

- **tensor.narrow(dim, start, length)**  
在 dim 维从 start 取 length 个，返回视图。

### 12.3 einsum

- **torch.einsum(equation, operands)**  
**是什么**：用爱因斯坦求和约定写张量运算，一次表达乘法与求和维度。  
**用途**：矩阵乘、外积、迹、对角、batch 乘等，公式简洁。

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum("ij,jk->ik", A, B)  # 矩阵乘
x = torch.randn(4)
y = torch.einsum("i,i->", x, x)     # 内积
```

### 12.4 roll / clamp

- **torch.roll(input, shifts, dims)**  
沿 dims 循环移位。  
- **torch.clamp(input, min=None, max=None)**  
将元素限制在 [min, max]；可用于梯度裁剪、ReLU 式截断。  
- **torch.clamp_min(input, min)** / **torch.clamp_max(input, max)**  
仅设下限或上限。

### 12.5 带系数的加减乘除

- **torch.addcdiv(input, tensor1, tensor2, value=1)**  
`input + value * (tensor1 / tensor2)`，逐元素；常用于带除法的更新。  
- **torch.addcmul(input, tensor1, tensor2, value=1)**  
`input + value * (tensor1 * tensor2)`，逐元素。

### 12.6 三角矩阵与重归一化

- **torch.tril(input, diagonal=0)** / **torch.triu(input, diagonal=0)**  
取下三角/上三角（含对角线）；diagonal 控制偏移。  
- **torch.renorm(input, p, dim, maxnorm)**  
沿 dim 将子向量重新归一化，使其 p-范数不超过 maxnorm；常用于梯度稳定。

```python
x = torch.randn(3, 3)
torch.tril(x)   # 下三角，其余为 0
torch.triu(x, diagonal=1)  # 严格上三角
```

### 12.7 设备、类型与 Python/NumPy 转换

- **tensor.to(device=None, dtype=None)**  
设备或 dtype 迁移；`.cpu()`, `.cuda(device=None)` 为常用简写。  
- **tensor.numpy()**  
转 NumPy 数组（与 CPU 张量共享内存时注意 inplace 修改）。  
- **tensor.tolist()**  
转 Python 嵌套 list。  
- **tensor.item()**  
单元素张量取 Python 标量（如 loss.item()）。  
- **tensor.fill_(value)** / **tensor.zero_()**  
原地填常量或置零。  
- **tensor.copy_(src)**  
从 src 拷贝到 self（形状需可广播或一致）。

### 12.8 复数与特殊函数（选读）

- **torch.real(input)** / **torch.imag(input)**  
取复数的实部/虚部。  
- **torch.angle(input)**  
幅角。  
- **torch.conj(input)**  
共轭。  
- **torch.special** 命名空间下有 **erf, erfc, gammaln, multigammaln** 等；科学计算与分布时常用。

### 12.9 差分、广播与维度保证

- **torch.diff(input, n=1, dim=-1)**  
沿 dim 做 n 阶差分；一阶即相邻相减，输出长度减 n。  
- **torch.broadcast_tensors(tensors)**  
将多个张量广播到相同形状后返回元组；在手动广播前做形状对齐时有用。  
- **torch.atleast_1d(input)** / **atleast_2d** / **atleast_3d**  
保证至少 1/2/3 维；标量变 (1,)，1D 保持，2D 保持。  
- **torch.bucketize(input, boundaries, right=False)**  
将 input 按 boundaries 分桶，返回每个元素所属桶的下标（整数）；right=True 时右闭。  
- **torch.searchsorted(sorted_sequence, value, right=False)**  
在有序 1D 张量 sorted_sequence 中找 value 的插入位置，保持有序；用于分位数、插值。

```python
x = torch.tensor([1., 2., 4., 7.])
torch.diff(x)  # [1., 2., 3.]
a, b = torch.broadcast_tensors(torch.randn(3, 1), torch.randn(1, 4))  # 均为 (3, 4)
torch.bucketize(torch.tensor([0.5, 2.5, 5.]), torch.tensor([1., 2., 3., 4.]))  # 桶下标
```

### 12.10 形状原地修改、类型提升与原地随机

- **tensor.resize_(sizes)** / **tensor.resize_as_(other)**  
原地修改形状；可改变元素总数（多则未初始化、少则截断），与 view 不同。多用于与 C++/Legacy 接口兼容，新代码优先用 view/reshape。  
- **torch.result_type(tensors_and_dtypes)** / **torch.promote_types(type1, type2)**  
根据类型提升规则得到运算结果 dtype；混合 int/float 或不同精度时，用于事先确定输出类型或做类型检查。  
- **tensor.uniform_(from=0, to=1)** / **tensor.normal_(mean=0, std=1)**  
原地用均匀分布或正态分布填充；初始化权重、生成噪声时常用。  
- **tensor.exponential_(lambd=1)** / **tensor.bernoulli_(p=0.5)**  
原地用指数分布或伯努利分布填充；采样、Dropout 掩码等。

```python
x = torch.empty(2, 3)
x.uniform_(-0.1, 0.1)   # 原地均匀初始化
torch.result_type(torch.tensor(1), torch.tensor(1.0))  # torch.float32
```

---

## 十三、速查与小结

### 13.1 分类速查


| 类别    | 常用 API                                                                                                                                                                                                                                                                                                                                     |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 创建    | tensor, zeros, ones, arange, linspace, randn, from_numpy                                                                                                                                                                                                                                                                                   |
| 形状    | view, reshape, squeeze, unsqueeze, flatten, transpose, permute, contiguous                                                                                                                                                                                                                                                                 |
| 拼接/拆分 | cat, stack, chunk, split, unbind                                                                                                                                                                                                                                                                                                           |
| 逐元/广播 | add, sub, mul, div；pow, sqrt, rsqrt, neg, reciprocal, sign；minimum, maximum；fmod, remainder；lerp, frac；atan2, hypot；exp, log, sin, cos, round, floor, ceil                                                                                                                                                                                 |
| 矩阵    | matmul, mm, bmm, mv, dot, vdot；ger, outer, cross；trace, linalg.inv, solve, det, svd, cholesky, qr, eig；tensordot, meshgrid                                                                                                                                                                                                                 |
| 归约    | sum, mean, max, min, prod, argmax, argmin, norm, std, var；any, all, count_nonzero；cumsum, cumprod, cummax, cummin, logcumsumexp；median, mode；nanmean, nansum, quantile, logsumexp, dist；sort, topk, kthvalue；unique                                                                                                                        |
| 索引    | [], index_select, masked_select, take, put_；nonzero；index_fill_, masked_fill_, masked_scatter_, index_put_                                                                                                                                                                                                                                 |
| 高级    | gather, scatter_, index_add_, index_copy_, scatter_reduce_, index_reduce_                                                                                                                                                                                                                                                                  |
| 比较逻辑  | eq, ne, gt, where, logical_and/or/not；isfinite, isnan, isinf, isclose；equal, allclose；is_tensor, is_floating_point, is_complex                                                                                                                                                                                                             |
| 其它    | clone, expand, repeat, narrow, einsum, roll, clamp, addcdiv, addcmul, tril, triu, renorm；real, imag, angle, conj；to/cpu/cuda, numpy/tolist/item, fill_/zero_/copy_；diff, broadcast_tensors, atleast_1d/2d/3d, bucketize, searchsorted；resize_/resize_as_, result_type/promote_types；uniform_/normal_/exponential_/bernoulli_；torch.special |


### 13.2 常见遗漏清单（按类补充）

以下 API 在写代码时常用，若在正文未单独展开，可在此按类查缺补漏。


| 类别          | 易遗漏的 API                                                 | 用途简述                                              |
| ----------- | -------------------------------------------------------- | ------------------------------------------------- |
| **创建**      | `full_like`, `empty_like`                                | 与输入同形填常量或未初始化；`torch.complex(real, imag)` 构造复数张量。 |
| **矩阵/张量**   | `torch.tensordot(a, b, dims)`                            | 指定维做内积收缩；高阶张量乘、注意力收缩。                             |
|             | `torch.meshgrid(*tensors, indexing='ij')`                | 由 1D 坐标生成网格；绘图、插值、物理场。                            |
| **形状/视图**   | `tensor.t()`                                             | 2D 转置 shorthand（仅 2D）。                            |
|             | `tensor.unfold(dim, size, step)`                         | 沿 dim 滑动窗口，得到局部块；做 patch、局部卷积时常用。                 |
|             | `torch.movedim(input, source, destination)`              | 把 source 维移到 destination；比 permute 更直观地「挪一维」。     |
|             | `torch.swapdims(input, dim0, dim1)`                      | 交换两维，transpose 的别名。                               |
| **索引/选数**   | `tensor.put_(index, source, accumulate=False)`           | 按线性下标写回，与 take 互逆；accumulate 为 True 时累加。          |
|             | `torch.nonzero(input)` / `input.nonzero()`               | 返回非零元素的坐标张量 (N, ndim)；与 mask 配合取下标。               |
|             | `tensor.index_fill_(dim, index, value)`                  | 在 dim 维的 index 位置填 value。                         |
|             | `tensor.masked_fill_(mask, value)`                       | 在 mask 为 True 处填 value；常用于 padding 掩码。            |
|             | `tensor.masked_scatter_(mask, source)`                   | 按 mask 从 source 依次取数填到 self。                      |
| **Scatter** | `tensor.scatter_reduce_(dim, index, src, reduce='sum'    | 'mean'                                            |
| **逐元**      | `log2`, `log10`, `exp2`, `expm1`                         | 以 2/10 为底对数、2^x、exp(x)-1。                         |
|             | `floor_divide` / `//`                                    | 向下取整除；`div(..., rounding_mode='floor'             |
| **归约**      | `cummax`, `cummin`                                       | 递进取大/取小，返回 (values, indices)。                     |
|             | `logcumsumexp`                                           | 递进 log-sum-exp，数值稳定。                              |
|             | `nanmean`, `nansum`                                      | 忽略 NaN 的均值/和。                                     |
|             | `quantile`, `nanquantile`                                | 分位数。                                              |
| **比较/判断**   | `torch.equal(a, b)`                                      | 两张量逐元相等且同形则 True（单布尔）。                            |
|             | `torch.allclose(a, b, rtol, atol)`                       | 整体近似相等，常用于数值测试。                                   |
|             | `is_tensor`, `is_floating_point`, `is_complex`           | 类型/数值类型判断。                                        |
| **写回**      | `tensor.index_put_(indices, value)`                      | 按多维索引元组写 value。                                   |
|             | `tensor.index_reduce_(dim, index, source, reduce)`       | 沿 dim 按 index 将 source 归约写回。                      |
| **设备与转换**   | `tensor.to(device                                        | dtype)`,` .cpu()`,` .cuda()`                      |
|             | `tensor.numpy()`, `tensor.tolist()`, `tensor.item()`     | 转 NumPy、转 Python list、标量取 Python 标量。              |
| **其它**      | `tensor.resize_(*sizes)` / `resize_as_(other)`           | 原地改形状（可改元素总数）；慎用，优先 view/reshape。                 |
|             | `torch.result_type(...)` / `torch.promote_types(t1, t2)` | 类型提升，得到运算结果 dtype。                                |
|             | `tensor.uniform_(from, to)` / `normal_(mean, std)`       | 原地均匀/正态填充；权重初始化、噪声。                               |
|             | `tensor.exponential_(lambd)` / `bernoulli_(p)`           | 原地指数/伯努利填充；采样、Dropout 等。                          |
|             | `torch.diff(input, n=1, dim=-1)`                         | 沿 dim 做 n 阶差分。                                    |
|             | `torch.broadcast_tensors(*tensors)`                      | 把多个张量广播到相同形状后返回。                                  |
|             | `vsplit`, `hsplit`, `dsplit`                             | 沿第 0/1/2 维 split 的便捷写法（2D/3D）。                    |
|             | `atleast_1d`, `atleast_2d`, `atleast_3d`                 | 保证至少 1/2/3 维。                                     |
|             | `tensor.fill_(value)`, `zero_()`, `copy_(src)`           | 原地填值、置零、从 src 拷贝。                                 |
|             | `torch.bucketize(input, boundaries)`                     | 将 input 按 boundaries 分桶，返回桶下标。                    |
|             | `torch.searchsorted(sorted, value)`                      | 在有序序列中找 value 的插入位置。                              |


正文已覆盖绝大部分「创建、形状、拼接、逐元、矩阵、归约、索引、gather/scatter、比较、einsum」；上表用于查缺补漏与速查，需要时查阅官方 [torch](https://pytorch.org/docs/stable/torch.html) 与 [torch.Tensor](https://pytorch.org/docs/stable/tensors.html) 文档。

### 13.3 小结

PyTorch 张量工具函数覆盖「创建 → 变形 → 拼接/拆分 → 运算 → 归约 → 索引」全流程；掌握 cat/stack、view/reshape、广播、matmul、归约 dim/keepdim、gather/scatter 和 where，即可覆盖绝大多数模型与数据脚本中的张量操作。需要时以本文索引与速查表按类查找具体 API 与示例。