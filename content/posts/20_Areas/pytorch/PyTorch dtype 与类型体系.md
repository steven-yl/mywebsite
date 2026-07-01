---
title: "PyTorch dtype 与类型体系"
subtitle: ""
date: 2026-03-25T00:00:00+08:00
draft: true
authors: [Steven]
summary: "系统说明 PyTorch 张量的 dtype：分类、默认推断、转换、类型提升、应用场景及与 NumPy 的对应关系。"
description: "梳理 PyTorch dtype 的精度与内存、默认类型、指定与转换方法、GPU/MPS 限制及实用建议。"
tags: ["PyTorch", "dtype", "Tensor", "混合精度"]
categories: ["PyTorch"]
series: [PyTorch实践指南]
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

PyTorch 中的 **dtype**（数据类型）用于定义张量中每个元素的存储类型，它决定了数值的精度、内存占用以及支持的操作。理解和正确使用 dtype 对于模型训练、内存优化和避免错误至关重要。

---

## 1. PyTorch 支持的数据类型

PyTorch 提供了多种 dtype，主要分为以下几类（以 CPU 为例，CUDA 支持情况类似但部分类型可能受限）：

| 数据类型 | PyTorch dtype | 对应 C 类型 | 内存 (字节/元素) | 说明 |
|---------|--------------|-----------|-----------------|------|
| **浮点型** | | | | |
| 半精度浮点 | `torch.float16` / `torch.half` | `half` | 2 | 用于混合精度训练，节省显存 |
| 单精度浮点 | `torch.float32` / `torch.float` | `float` | 4 | **默认类型**，最常用 |
| 双精度浮点 | `torch.float64` / `torch.double` | `double` | 8 | 高精度计算，但内存和计算开销大 |
| **整型** | | | | |
| 8 位有符号整型 | `torch.int8` | `int8_t` | 1 | 用于量化或小范围整数 |
| 8 位无符号整型 | `torch.uint8` | `uint8_t` | 1 | 常用于图像数据（像素值 0-255） |
| 16 位有符号整型 | `torch.int16` / `torch.short` | `int16_t` | 2 | 较少使用 |
| 32 位有符号整型 | `torch.int32` / `torch.int` | `int32_t` | 4 | 中等范围整数 |
| 64 位有符号整型 | `torch.int64` / `torch.long` | `int64_t` | 8 | **用于索引和标签**（因为索引通常需要 64 位） |
| **布尔型** | `torch.bool` | `bool` | 1 | 逻辑运算结果 |
| **复数型** | | | | |
| 复数（单精度） | `torch.complex64` | `complex float` | 8 | 实部和虚部各 4 字节 |
| 复数（双精度） | `torch.complex128` | `complex double` | 16 | 实部和虚部各 8 字节 |

> **注意**：`torch.float` 是 `torch.float32` 的别名，`torch.int` 是 `torch.int32`，`torch.long` 是 `torch.int64`，`torch.double` 是 `torch.float64`。

---

## 2. 默认 dtype 与类型推断

- **浮点默认类型**：创建未指定 dtype 的浮点张量时，默认使用 `torch.float32`。可以通过 `torch.set_default_dtype(torch.float64)` 更改默认浮点类型（仅影响浮点型，不影响整型）。
- **整型默认推断**：使用 `torch.tensor([1,2,3])` 时，默认推断为 `torch.int64`（即 `torch.long`），因为 Python 整数是任意精度，PyTorch 选择 64 位作为安全默认。
- **其他类型**：`torch.tensor([True, False])` 默认为 `torch.bool`；`torch.tensor([1+1j])` 默认为 `torch.complex64`。

示例：
```python
import torch

# 默认浮点
a = torch.tensor([1.0, 2.0])
print(a.dtype)  # torch.float32

# 更改默认浮点类型
torch.set_default_dtype(torch.float64)
b = torch.tensor([1.0, 2.0])
print(b.dtype)  # torch.float64

# 整数列表默认
c = torch.tensor([1, 2, 3])
print(c.dtype)  # torch.int64
```

---

## 3. 指定 dtype 的方法

### 创建时指定
```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.zeros(2, 3, dtype=torch.int8)
z = torch.randn(100, dtype=torch.float16)  # 半精度
```

### 转换类型
```python
# 使用 .to()
x_float = x.to(torch.float32)
x_half = x.to(torch.float16)
x_long = x.to(torch.long)

# 使用快捷方法
x_float = x.float()   # 转为 float32
x_half = x.half()     # 转为 float16
x_long = x.long()     # 转为 int64
x_int = x.int()       # 转为 int32
x_double = x.double() # 转为 float64
```

### 类型推断与混合运算
PyTorch 遵循**类型提升**规则：在二元操作中，结果 dtype 通常是两个操作数中**更宽**的类型（例如 float32 + float64 → float64；int + float → float；int + bool → int）。但整型与浮点运算时，结果会转为浮点型。

```python
a = torch.tensor([1, 2], dtype=torch.int32)
b = torch.tensor([1.0, 2.0], dtype=torch.float32)
c = a + b          # 自动提升为 float32
print(c.dtype)     # torch.float32
```

---

## 4. 各 dtype 的应用场景

| dtype | 典型用途 |
|-------|---------|
| `torch.float32` | **绝大多数深度学习模型**，梯度、权重、激活值的默认选择，平衡精度与性能。 |
| `torch.float16` | 混合精度训练（如 `torch.cuda.amp`），大幅降低显存占用并加速计算，但精度稍低。 |
| `torch.float64` | 科学计算、对数值稳定性要求极高的场景（如计算 Hessian 矩阵）。 |
| `torch.int64` | **索引操作**（如 `tensor[indices]`）、分类任务标签（`CrossEntropyLoss` 要求）、张量形状（`size()` 返回的元组）。 |
| `torch.uint8` | 图像数据（0-255）、掩码（mask）或布尔值的高效存储。 |
| `torch.bool` | 逻辑运算结果（`>`、`==` 等），掩码索引。 |
| `torch.int8` / `torch.int16` | 量化推理（如 `torch.quantization`）或小范围整数运算。 |
| `torch.complex64/128` | 信号处理、傅里叶变换（`torch.fft`）等涉及复数的领域。 |

---

## 5. 类型转换的注意事项

### 精度损失
将高精度类型转换为低精度类型可能导致精度损失或溢出：
```python
x = torch.tensor([1e-8], dtype=torch.float64)
x_half = x.to(torch.float16)
print(x_half)  # 可能变为 0.0（下溢）
```

### 整数溢出
将大整数转换为小范围整型会溢出：
```python
x = torch.tensor([300], dtype=torch.int64)
x_int8 = x.to(torch.int8)  # 结果可能是 44（因为 300 mod 256 再转为有符号）
```

### 布尔与整数的转换
`bool` → `int`：`True` 变为 1，`False` 变为 0；`int` → `bool`：非零为 `True`，零为 `False`。

---

## 6. 设备相关的 dtype 限制

- **CPU**：支持所有上述 dtype。
- **CUDA（GPU）**：通常支持 `float32`、`float16`、`int64`、`int32`、`uint8`、`bool` 等。`float64` 在 GPU 上计算速度较慢且显存占用大，一般不推荐。部分旧 GPU 可能对某些类型支持不佳，但现代 GPU（Volta 架构及以上）对 `float16` 有硬件加速。
- **MPS（Apple Silicon）**：支持大部分 dtype，但 `float64` 支持有限。

使用 `.to(device)` 时，类型与设备可以同时转换：
```python
x = torch.tensor([1.0])
x_cuda = x.to('cuda', dtype=torch.float16)
```

---

## 7. 与 NumPy dtype 的对应关系

PyTorch 的 dtype 与 NumPy 的 dtype 基本一一对应，便于互操作：

| PyTorch | NumPy |
|---------|-------|
| `torch.float32` | `np.float32` |
| `torch.float64` | `np.float64` |
| `torch.int64` | `np.int64` |
| `torch.uint8` | `np.uint8` |
| `torch.bool` | `np.bool_` |

转换时，dtype 会自动匹配：
```python
import numpy as np
np_arr = np.array([1.0, 2.0], dtype=np.float32)
torch_arr = torch.from_numpy(np_arr)
print(torch_arr.dtype)  # torch.float32
```

---

## 8. 实用技巧与最佳实践

1. **优先使用 `float32`**：除非有明确的精度或内存需求。
2. **索引和标签使用 `int64`**：PyTorch 的许多函数（如 `scatter`、`index_select`）强制要求索引为 `int64`。
3. **内存敏感场景使用 `float16` 或量化类型**：在模型推理或混合精度训练中显著降低显存。
4. **避免不必要的类型转换**：频繁的 `to(dtype)` 会带来额外开销。
5. **检查 dtype**：使用 `tensor.dtype` 或 `print(tensor)`（会显示类型）来调试。

---

## 9. 示例代码集锦

```python
import torch

# 1. 默认整数张量
idx = torch.tensor([0, 2, 1])
print(idx.dtype)  # torch.int64

# 2. 交叉熵损失要求标签为 long
criterion = torch.nn.CrossEntropyLoss()
logits = torch.randn(3, 5)
labels = torch.tensor([1, 0, 3], dtype=torch.long)  # 必须是 long
loss = criterion(logits, labels)

# 3. 混合精度训练常用设置
with torch.cuda.amp.autocast():
    output = model(input.half())  # 输入转为 half，模型参数自动转换
    loss = criterion(output, labels)

# 4. 类型转换与设备迁移
x = torch.randn(100, 100).cuda()  # float32
x = x.half()  # 转为 float16
x = x.to(torch.float64)  # 转为 float64，但 GPU 上可能较慢
```

---

## 总结
在深度学习任务中，`float32` 和 `int64` 是最常见的两个 dtype，而 `float16` 和量化类型则在优化性能时发挥重要作用。