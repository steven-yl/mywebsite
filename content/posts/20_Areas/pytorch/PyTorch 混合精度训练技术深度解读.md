---
title: PyTorch混合精度训练技术深度解读
subtitle: ""
date: 2026-03-24T00:00:00+08:00
draft: false
authors:
  - Steven
description: "系统梳理 PyTorch 自动混合精度（AMP）训练技术文档，包括 torch.autocast 与 torch.amp.GradScaler 的原理、API 详解、内部转换策略、典型用法示例、重要注意事项等。"
summary: "系统梳理 PyTorch 自动混合精度（AMP）训练技术文档，包括 torch.autocast 与 torch.amp.GradScaler 的原理、API 详解、内部转换策略、典型用法示例、重要注意事项等。"
tags:
  - PyTorch
  - AMP
  - Deep Learning
categories:
  - PyTorch
  - AMP
  - Deep Learning
series:
  - PyTorch实践指南
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

## 1. 总览

### 1.1 什么是混合精度训练

混合精度训练（Mixed Precision Training）是一种在神经网络训练中**同时使用多种数值精度**（如 FP16 与 FP32）来加速计算、减少显存占用，同时保持模型精度的技术。核心思想是利用低精度（如 FP16 或 BF16）的高速算术运算能力，配合高精度（FP32）的关键参数存储，在不牺牲最终模型质量的前提下大幅提升训练效率。

### 1.2 整体架构

混合精度训练的完整技术栈可由下图表示：

```
┌────────────────────────────────────────────────────────────────┐
│                      应用层（训练脚本）                        │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│   │ 模型定义    │  │ 自动混合精度 │  │ 损失缩放           │  │
│   │ (FP32/BF16) │  │ (Autocast)   │  │ (GradScaler)       │  │
│   └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘  │
├──────────┼────────────────┼────────────────────┼──────────────┤
│          │       框架层（PyTorch/TensorFlow）   │              │
│   ┌──────▼──────┐  ┌──────▼───────┐  ┌─────────▼──────────┐  │
│   │ 前向传播    │  │ 反向传播     │  │ 优化器步（含Master │  │
│   │ (FP16 路径) │  │ (FP16 梯度)  │  │ Weights更新）      │  │
│   └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘  │
├──────────┼────────────────┼────────────────────┼──────────────┤
│          │       硬件层（GPU/NPU）              │              │
│   ┌──────▼────────────────▼────────────────────▼──────────┐   │
│   │          Tensor Cores / 矩阵加速单元                  │   │
│   │          (FP16 吞吐量 8x/16x of FP32)                │   │
│   └───────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

**图1：混合精度训练架构图**。应用层通过 autocast 自动将部分算子转换为低精度；GradScaler 动态调整损失缩放因子，防止梯度下溢。框架层保证高精度主权重（Master Weights）存储与更新。硬件层利用专用单元加速低精度运算。

### 1.3 各组件作用与关联

| 组件 | 作用 | 与其他组件的关系 |
|------|------|------------------|
| **低精度计算（FP16/BF16）** | 加速前向/反向传播中的矩阵运算，减少显存占用 | 是混合精度训练的基石，需与损失缩放、主权重配合解决精度问题 |
| **自动混合精度（Autocast）** | 自动决定哪些操作使用低精度，哪些保持 FP32 | 根据“安全操作列表”将大部分运算转换为半精度，同时保护精度敏感层 |
| **损失缩放（Loss Scaling）** | 放大损失值，使小梯度值进入半精度可表示范围，避免下溢 | 与 FP16 梯度计算紧密耦合；**BF16 训练通常不需要此组件** |
| **主权重副本（Master Weights）** | 保留一份 FP32 权重，用半精度梯度更新，避免每次更新的舍入误差累积 | 在前向传播前将权重降级为 FP16/BF16，与优化器步骤衔接 |
| **分布式梯度通信** | AllReduce 等通信时，可先对 FP16 梯度反缩放并保持 FP16 传输，减少带宽 | 与损失缩放和精度策略共同决定是否在传输前转为 FP32 |

### 1.4 优缺点与适用场景对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **纯 FP32 训练** | 精度可靠，无需特殊处理 | 计算慢，显存占用大 | 小规模模型、对精度极其敏感的任务 |
| **FP16 混合精度** | 利用 Tensor Cores，吞吐量提升2~6倍；显存减少约一半 | 需损失缩放、主权重，调试复杂；某些层需强制 FP32 | NVIDIA GPU (Volta及以后)，视觉/语言模型训练 |
| **BF16 混合精度** | 动态范围与 FP32 相同，无需损失缩放；部署支持愈发广泛 | 需要硬件支持（A100、H100、TPU 等），数值精度略低于 FP16 | 大规模训练、TPU 训练、对稳定性要求高的任务 |
| **纯 FP16（无主权重）** | 显存最小 | 极易出现梯度消失/爆炸，精度难以保证 | 极少数实验性质尝试，不推荐生产使用 |

---

## 2. 数值格式：FP16、BF16 与 FP32

### 2.1 浮点表示回顾

浮点数由符号位 $s$、指数位 $e$ 和尾数位 $m$ 组成，值为：
$$
\text{value} = (-1)^s \times 2^{e - \text{bias}} \times (1.m)
$$
表数范围由指数位数决定，精度由尾数位数决定。

### 2.2 FP16 格式（IEEE 754 half precision）

- **总位数**：16 bits
- **符号**：1 bit
- **指数**：5 bits（bias=15）
- **尾数**：10 bits
- **动态范围**：约 $6 \times 10^{-8} \sim 65504$
- **最小正正规数**：$2^{-14} \approx 6.1 \times 10^{-5}$，支持次正规数（denormal）可达 $5.96 \times 10^{-8}$
- **特点**：动态范围较窄，许多小梯度容易下溢为0；精度约为十进制3.3位。

### 2.3 BF16 格式（Brain Floating Point）

- **总位数**：16 bits
- **符号**：1 bit
- **指数**：8 bits（bias=127，同 FP32）
- **尾数**：7 bits
- **动态范围**：与 FP32 完全一致，约 $1.18 \times 10^{-38} \sim 3.4 \times 10^{38}$
- **特点**：牺牲尾数精度换取动态范围，因此**不存在梯度下溢问题**，训练时通常无需损失缩放。非常适合深度学习。

### 2.4 FP32 格式（单精度）

- **总位数**：32 bits
- **指数**：8 bits，尾数：23 bits
- 作为基准精度，用于关键参数存储和敏感层计算。

### 2.5 格式对比与选择

| 特性 | FP32 | FP16 | BF16 |
|------|------|------|------|
| 动态范围 | 高 | 低（易下溢） | 高（同 FP32） |
| 数值精度 | 高（约7.2位十进制） | 中（约3.3位） | 低（约2位） |
| 硬件加速 | 基础 | NVIDIA Tensor Cores 支持 | NVIDIA Ampere+ 、TPU 支持 |
| 损失缩放 | 不需要 | **需要** | 不需要 |

**选择建议**：NVIDIA Volta/Turing 用户使用 FP16+损失缩放；Ampere 及以上（A100/H100）优先使用 BF16 以简化训练；Google TPU 使用 BF16。

---

## 3. 混合精度训练的核心挑战

### 3.1 下溢与上溢

FP16 能表示的最小正正规数是 $6.1 \times 10^{-5}$，而神经网络训练中很多梯度值（如权重衰减梯度、深层浅层梯度）可能远小于此（例如 $1 \times 10^{-7}$），这些值将变为 0，即下溢。反之，超过 65504 会发生上溢导致 inf/NaN。

### 3.2 舍入误差与权重更新丢失

即使梯度值未下溢，用 FP16 梯度更新 FP16 权重时，由于权重值远大于更新量，加法可能无效果：
$$
W_{t+1} = W_t - \eta \cdot g_t
$$
若 $W_t = 1.0$，$\eta g_t = 2^{-11} = 0.000488$，在 FP16 中加上 1.0 后可能舍入回 1.0（因为 FP16 的 10 位尾数只能区分约 1/1024 的差异）。更新被“吃掉”，模型不收敛。

### 3.3 精度敏感层

某些操作对精度高度敏感，必须在 FP32 下计算：
- **Softmax、LayerNorm、BatchNorm**：涉及大量指数、求和、除法，FP16 容易产生较大误差。
- **Loss 函数**（如交叉熵）：计算过程中值的范围可能很大。
- **梯度累积与统计**：需要高精度累加。

混合精度策略通常对这些层使用 FP32 计算路径。

---

## 4. 关键技术：损失缩放 (Loss Scaling)

### 4.1 为什么需要损失缩放

反向传播的梯度值正比于损失函数对权重的偏导。许多梯度值很小（尤其是在深层网络尾部）。直接使用 FP16 存储和计算这些梯度，大量值会下溢为 0，导致模型停止学习。**解决思路**：在反向传播前，将损失值人为放大（如乘以一个常数因子），使得得到的放大梯度可被 FP16 表示；在更新权重前，再将梯度除以相同的因子，恢复原始尺度。

### 4.2 静态损失缩放

选择一个固定的缩放因子（如 $2^{15} = 32768$），训练全程不变。优点是简单，缺点是无法自适应：因子太小则保护不足，因子太大可能导致梯度上溢。

### 4.3 动态损失缩放

自适应的损失缩放算法，每 N 次迭代检查梯度是否存在 inf/NaN：
- 若不存在 inf/NaN，可尝试增大缩放因子（乘以 $\alpha$，如 2.0），或保持不变。
- 若存在 inf/NaN，跳过此次更新，不应用权重更新，并将缩放因子减小（除以 $\alpha$，如 2.0），以保证后续训练稳定。

典型动态算法（NVIDIA Apex / PyTorch GradScaler）：

```
初始化 scale = 2^16
每经过 growth_interval 步无 inf/NaN，执行 scale *= growth_factor
当 step 出现 inf/NaN：
    scale *= backoff_factor
    跳过本次 optimizer.step()
```

PyTorch 默认参数：`init_scale=65536.0`，`growth_factor=2.0`，`backoff_factor=0.5`，`growth_interval=2000`。

### 4.4 实现细节与示例

**损失缩放的完整流程**：
1. 前向：output = model(input)  # 混合精度
2. loss = criterion(output, target)
3. 将 loss 乘以当前 scale：`scaled_loss = loss * scale`
4. 反向：`scaled_loss.backward()`，计算出的梯度自动被 scale 倍扩大
5. 反缩放梯度：`scaler.unscale_(optimizer)`，将 optimizer 中参数的 grad 除以 scale
6. 梯度裁剪（若需要）：应在反缩放之后进行
7. 执行 optimizer.step()（使用 FP32 Master Weights）
8. 更新 scaler：`scaler.update()`，内部判断梯度是否有 inf/NaN，并调整 scale

---

## 5. 关键技术：主权重副本 (Master Weights)

### 5.1 梯度更新在 FP16 下的问题

如 3.2 节所述，即使有损失缩放，权重值本身以 FP16 存储仍会导致更新丢失。为解决此问题，我们保持一份**FP32 的主权重副本**。

### 5.2 Master Weights 机制

1. **存储**：模型权重（parameters）实际存储为 FP32。
2. **前向传播**：临时将权重从 FP32 转换为 FP16（或 BF16），用于矩阵乘法等计算。
3. **反向传播**：得到 FP16（或 FP32，取决于框架设定）梯度。
4. **梯度反缩放**：将梯度恢复为真实尺度。
5. **更新主权重**：在 FP32 精度下执行：
   $$
   W_{master} \leftarrow W_{master} - \eta \cdot g_{FP32}
   $$
   其中 $g_{FP32}$ 由反缩放后的 FP16 梯度转换为 FP32。
6. **下一次前向**：重复步骤2，使用更新后的 FP32 权重转 FP16。

### 5.3 内存与性能权衡

保留 FP32 主权重增加了约 50% 的权重内存开销（因为除了 FP16 的活动副本，还有 FP32 的主副本）。但激活值（activations）在 FP16 下已经减少了一半，总体显存通常仍降低约 30%~40%。性能上，计算密集部分（矩阵乘法）由 Tensor Cores 加速，而额外的精度转换开销很小。

---

## 6. 自动混合精度 (AMP)

### 6.1 什么是 AMP

自动混合精度（Automatic Mixed Precision）是一种**自动化框架功能**，它根据预先定义的安全列表（Op list）决定每个算子在训练中的执行精度。用户只需通过上下文管理器或装饰器包裹前向传播，并配合梯度缩放器，即可实现混合精度训练。

### 6.2 框架支持：PyTorch AMP

PyTorch 在 1.6+ 提供 `torch.cuda.amp` 模块，核心包含：

- **`torch.cuda.amp.autocast()`**：上下文管理器，进入此上下文后，CUDA 算子将自动选择合适精度。符合条件的算子（如 `nn.Linear`、`nn.Conv2d`、`matmul`）使用 FP16，不适用的（如 softmax、normalization）保持 FP32。可指定 `dtype=torch.float16` 或 `torch.bfloat16`。
- **`torch.cuda.amp.GradScaler`**：实现动态损失缩放。提供 `scale(loss)` 方法返回 scaled loss，`step(optimizer)` 进行优化步骤前自动反缩放梯度，`update()` 调整缩放因子。

**完整训练步骤（伪代码）**：

```python
scaler = torch.cuda.amp.GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**关键函数详解**：

- `scaler.scale(loss)`：返回 `loss * current_scale`，此结果作为 backward 的输入。
- `scaler.step(optimizer)`：
    1. 检查 optimizer 中所有参数的 `grad` 是否有 inf/NaN。
    2. 如果有 inf/NaN，跳过 `optimizer.step()`，并准备在 `update()` 中减小 scale。
    3. 如果无 inf/NaN，则反缩放梯度（`param.grad = param.grad / scale`）并执行 `optimizer.step()`。因此 `step()` 完成后，参数的 `.grad` 已恢复为正常尺度。
- `scaler.update()`：根据 inf/NaN 出现情况，更新内部 scale 值，策略为增长/回退。

### 6.3 TensorFlow 混合精度 API

TensorFlow 2.x 提供了 `tf.keras.mixed_precision` 模块：

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
# 构建模型时，会自动将各层计算 dtype 设为 float16，变量 dtype 保持 float32
# 需要将损失输出强制转为 float32，否则可能下溢
# 优化器使用 LossScaleOptimizer 包裹：
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
```

内部同样维护 Master Weights 和动态损失缩放。

### 6.4 操作自动转换策略 (Ops list)

AMP 的转换策略由框架维护，大致原则：

- **通常转换为 FP16 的算子**：卷积、全连接、矩阵乘法、激活函数（ReLU、GeLU 等，部分除外）、元素级加法/乘法（部分条件）。
- **通常保持 FP32 的算子**：Softmax、LayerNorm、BatchNorm、损失函数（交叉熵、NLLLoss等）、指数/对数、梯度归一化等。

具体列表可参考 PyTorch 文档中的“Autocast Op Reference”。

### 6.5 黑白名单机制

一些特殊场景可能需要用户自定义强制精度，例如：

- 强制某些层使用 FP32：可通过 `with torch.cuda.amp.autocast(enabled=False):` 包裹或对输入调用 `.float()`。
- 自定义算子的 dtype 规则：可通过 `torch.cuda.amp.custom_fwd/custom_bwd` 装饰器实现。

---

## 7. 训练流程与算法步骤

结合前面组件，混合精度训练的完整流程如下。

### 7.1 前向传播

- 输入数据通常保持 FP32（或根据需要 cast）。
- 进入 `autocast` 上下文。
- 模型各层：
  - 权重从 Master FP32 转换为 FP16/BF16 用于矩阵乘法。
  - 安全操作使用 FP16。
  - 敏感操作在上下文内自动使用 FP32。
- 输出 logits 在 autocast 中可能是 FP16。

### 7.2 损失计算与缩放

- 损失函数通常在 FP32 下计算（框架自动处理）。
- 对 loss 执行：`scaled_loss = loss * loss_scale`（由 GradScaler.scale() 完成）。

### 7.3 反向传播

- 调用 `scaled_loss.backward()`。
- 计算得到的梯度已被放大 scale 倍，并且大部分以 FP16 存储（部分为 FP32）。
- 梯度值在 FP16 可表示范围内，避免了下溢。

### 7.4 梯度反缩放与 Master Weights 更新

- `scaler.step(optimizer)` 前先反缩放：`grad = grad / scale`。
- 此后梯度恢复为真实尺度。
- 优化器使用 FP32 Master Weights 进行更新：
  - 将梯度转换为 FP32（若之前是 FP16）。
  - 执行动量、Adam 等更新公式（此时所有状态可能使用 FP32，但也可配合低精度优化器）。
- 更新后的 FP32 权重成为新的 Master Weights。

### 7.5 优化器步骤

- PyTorch 中 `scaler.step(optimizer)` 自动完成反缩放 + `optimizer.step()`。
- TensorFlow 中 `LossScaleOptimizer` 在 `apply_gradients` 中完成反缩放。
- `scaler.update()` 根据这次 step 中梯度的 inf/NaN 状态调整损失缩放因子。

**图示：混合精度训练数据流**

```
输入 x (FP32)
    │
    ▼
[Autocast 上下文]
    │
    ▼
模型层: 权重 → cast 到 FP16/BF16 → 矩阵乘法(FP16/BF16) ────► 激活值(FP16/BF16)
    │                                                    （部分层保持 FP32）
    ▼
输出 logits (FP16/BF16)
    │
    ▼
损失函数 (FP32) → loss (FP32)
    │
    ▼
scale(loss) → scaled_loss (FP32)
    │
    ▼
backward() → 梯度 grad (FP16, 已放大)
    │
    ▼
unscale_(optimizer) → grad /= scale (FP32/FP16)
    │
    ▼
optimizer.step() 更新 FP32 Master Weights
    │
    ▼
scaler.update() 调整 scale
```

---

## 8. 分布式训练中的混合精度

### 8.1 梯度 AllReduce 的精度

多 GPU 数据并行中，每个 GPU 计算本地梯度，然后通过 AllReduce 求和取平均。如果将 FP16 梯度直接通信，可以减少一半带宽。但需注意：

- 未反缩放的梯度是扩大后的值，通信时若使用 FP16 可能上溢。
- 正确做法：先在每个 GPU 上对梯度进行**反缩放**，然后保持在 FP16 进行 AllReduce，最后各 GPU 使用 FP32 进行 Master Weights 更新。或者反缩放后在 FP32 下通信（更安全但带宽高）。

PyTorch 的 `DistributedDataParallel` 配合 `GradScaler` 时，如果使用 `find_unused_parameters=False`，可自动处理：在 backward 后，bucket 中梯度已就绪，`scaler.step()` 会反缩放并启动 AllReduce（DDP 的 allreduce hook 在参数梯度 ready 时触发）。通信缓冲区精度由 DDP 的 `gradient_as_bucket_view` 和参数类型决定；为了使用 FP16 通信，可将参数存储为 FP16 主参数但需特殊处理，一般推荐保持默认（FP32 通信）以避免精度问题，除非带宽是明确瓶颈。

### 8.2 梯度通信前反缩放

若手动管理，正确顺序为：

```python
# 在 backward() 后，scaler.step(optimizer) 之前，禁止手动调用 allreduce
# 默认 DDP 会在 backward 的 hook 中自动进行 allreduce（梯度是放大后的 FP16 或 FP32）。
# scaler.step() 里会反缩放梯度，之后 optimizer.step()。
# 若希望在 FP16 下通信，可使用 NVIDIA Apex 的 Distributed AMP 扩展，
# 它在 unscale 后、执行 allreduce 前将梯度转换为 FP16，allreduce 后再转回 FP32。
```

### 8.3 通信缓冲精度

混合精度下的分布式最佳实践：

- 使用 `DistributedDataParallel` + `GradScaler` 默认设置，无需修改。
- 如使用 `torch.nn.parallel.DistributedDataParallel`，且 `device_ids` 指定 GPU，梯度同步基于 FP32 bucket，因其存储的是 FP32 梯度视图。
- 若想利用 FP16 降低通信量，需使用 `NVIDIA/apex` 的 `DDP` + `amp` 的 `distributed` 选项，但会增加复杂度。新项目建议直接使用 BF16 或默认 FP32 通信，因为网络带宽在现代集群中通常不是瓶颈。

---

## 9. 硬件加速：Tensor Cores 与 NPU

### 9.1 NVIDIA Tensor Cores 支持 FP16/BF16 矩阵运算

- **Volta (V100)**：支持 FP16 矩阵乘加，吞吐量是 FP32 的 8 倍（125 TFLOPS vs 15.7 TFLOPS）。
- **Turing (T4)**：引入 INT8/INT4 等，FP16 也有加速。
- **Ampere (A100)**：新增 BF16 支持，FP16/BF16 吞吐量达到 312 TFLOPS（稀疏），FP32 为 19.5 TFLOPS，加速比更显著。同时支持 TF32（19位尾数），但严格意义上不是混合精度。
- **Hopper (H100)**：进一步强化 FP8 支持，FP16/BF16 仍然是主力。

Tensor Cores 通过 `mma.sync` 等指令并行执行 $D = A \times B + C$，要求输入矩阵形状满足一定对齐要求（通常 m,n,k 为 8 的倍数）。

### 9.2 性能特征

- 计算密集层（Conv, Linear, Attention 中的 MatMul）可提速 2~6 倍。
- 内存操作（Normalization, activation 等）提升不明显，因为受限于带宽。
- 整体训练吞吐量提升通常在 1.5x ~ 3x 之间（与实际模型结构相关）。

### 9.3 硬件要求

- 混合精度（FP16）需要 CUDA compute capability >= 7.0（V100 及以上）。GTX 10 系列不支持 Tensor Cores 加速 FP16。
- BF16 需要 compute capability >= 8.0 (A100)。
- 确保 NVIDIA 驱动、CUDA 工具包和 cuDNN 版本兼容。

---

## 10. 实践指南与代码示例

### 10.1 PyTorch 完整示例（可运行）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 简单 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # 模拟数据
    data = torch.randn(8, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (8,), device=device)

    for epoch in range(5):
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        # 缩放损失并反向
        scaler.scale(loss).backward()
        # 反缩放梯度并执行优化器步
        scaler.step(optimizer)
        # 更新缩放器
        scaler.update()
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Scale: {scaler.get_scale()}')

if __name__ == '__main__':
    train()
```

### 10.2 TensorFlow/Keras 示例

```python
import tensorflow as tf

# 设置全局混合精度策略
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, dtype='float32')  # 输出层保持 float32
])

optimizer = tf.keras.optimizers.SGD(0.01)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 假数据
import numpy as np
data = np.random.randn(64, 32, 32, 3).astype('float32')
label = np.random.randint(0, 10, size=(64,))
model.fit(data, label, epochs=5)
```

### 10.3 使用 Apex 的旧方法（供维护旧代码参考）

NVIDIA Apex 是混合精度训练的早期实现，提供 `amp.initialize` 和 `amp.scale_loss`。PyTorch 1.6 后官方 AMP 已成为首选，但 Apex 仍在一些场景中使用。

```python
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
optimizer.step()
```

opt_level 控制优化级别（O0: FP32 训练，O1: 自动混合精度，O2: 几乎纯 FP16 训练带主权重，O3: 纯 FP16 无主权重）。

### 10.4 训练与推理一致性检查

混合精度训练后，需确保推理时（FP32 或 FP16）的结果与期望一致。可在训练结束后，对模型执行 `model.float()` 将所有权重转换回 FP32，然后进行推理测试。若训练过程中 Master Weights 正常工作，推理精度应与纯 FP32 训练相当。

---

## 11. 常见问题与调试

### 11.1 损失 NaN / 发散

**原因**：
- 损失缩放因子过高，导致梯度上溢。
- 学习率过大，与混合精度下数值不稳定叠加。
- 某些层在 FP16 下运算不稳定（如 Attention 中的 softmax 除以极小的缩放）。

**解决方法**：
- 降低初始损失缩放因子，如从 65536 降至 2048。
- 使用动态缩放（默认）并耐心观察几轮迭代后，scale 会自适应。
- 对不稳定层强制使用 FP32（如 autocast 下对注意力部分手动 `.float()`）。
- 使用 BF16 替代 FP16。

### 11.2 精度不收敛

- 检查是否正确使用主权重：权重应存储在 FP32，检查 `model.parameters()` 的 dtype。
- 确保未对损失缩放后的梯度进行了额外操作（如手动 `grad /= scale`），因为 `scaler.step()` 已自动反缩放。
- 验证数据增强和预处理没有精度问题。

### 11.3 性能没有提升

- 确认使用了 Tensor Cores 支持的 GPU 和 CUDA 版本。
- 模型计算密集度不够高（如大量小卷积或 batch 过小），导致 Tensor Cores 利用不充分。
- 检查是否使用了 `torch.backends.cudnn.benchmark = True` 以优化卷积算法。
- 确保没有频繁的 CPU-GPU 同步（如打印 loss 时 `.item()` 调用过多）。

---

## 12. 进阶主题

### 12.1 BF16 训练与混合精度

BF16 训练因动态范围大，无需损失缩放。PyTorch 中只需将 `autocast` 的 `dtype=torch.bfloat16`，并移除 GradScaler：

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
loss.backward()
optimizer.step()
```

某些操作（如 LayerNorm）在 BF16 autocast 下仍保持 FP32。整体流程大大简化，稳定性极佳。

### 12.2 低精度优化器状态 (8-bit optimizers)

除了前向/反向计算，优化器状态（如 Adam 的动量 m, v）也占据大量显存。结合混合精度训练，可使用 Bitsandbytes 的 8-bit 优化器（`bnb.optim.Adam8bit`）将状态量化为 8-bit，进一步降低内存，适合大模型微调。

### 12.3 与模型剪枝/量化结合

混合精度训练与量化感知训练（QAT）可以叠加：模型权重以 FP32 Master 存储，通过伪量化节点模拟低精度推理，前向/反向仍可利用 FP16 加速。这要求 QAT 节点不影响 AMP 的 cast 策略，一般需在量化节点前后强制 FP32。

### 12.4 自动损失缩放算法详解

动态损失缩放的核心是一个有限状态机或基于计数器的方法。NVIDIA 的算法（PyTorch 默认实现）：

1. 初始 scale = 65536。
2. 在 `scaler.step(optimizer)` 中检查梯度 inf/NaN。
3. 若存在 inf/NaN，本次 step 跳过（不更新权重），且 `mark_overflow()` 被调用，内部增加连续溢出计数。
4. `update()` 被调用时：
   - 如果本次 step 无溢出：增长计数器 +1；当累积 `growth_interval` 次无溢出，scale *= `growth_factor`（默认 2.0），重置计数器。
   - 如果本次 step 有溢出：scale *= `backoff_factor`（默认 0.5）；如果 scale 低于某个最小值（默认 1.0），强制设为 1.0；重置增长计数器。
5. 这样 scale 会在安全范围内自适应调整，训练初期可能频繁回退，随着模型稳定 scale 会升高。

**数学依据**：以 scale 放大损失相当于放大梯度分布的标准差，使更多梯度值落入 FP16 的正规数范围，而不改变梯度方向。

---

## 13. 总结

混合精度训练是一种成熟且必备的高效训练技术。通过理解 FP16/BF16 的数值特性、掌握损失缩放和主权重机制，并利用现代框架的自动混合精度 API，开发者可以用最小的代码改动获得 1.5~3x 的训练加速和约 30%~50% 的显存节省。实践中推荐优先使用 BF16（若硬件支持）以避免损失缩放的复杂性；否则使用 PyTorch 的 GradScaler + autocast 或 TensorFlow 的 mixed_float16 策略。在分布式、大模型、低精度优化器等扩展场景下，混合精度是构建高性能训练系统的基石。