---
title: "Flash Attention 原理公式与推导"
date: 2026-04-09T10:00:00+08:00
draft: false
authors: [Steven]
description: "Flash Attention 原理公式与推导"
summary: "Flash Attention 原理公式与推导"

tags: [Deep Learning]
categories: [Deep Learning]
series: [Deep Learning系列]

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

## Flash Attention 原理公式与推导

Flash Attention 是一种 **IO 感知**的精确注意力算法，其核心数学技巧在于 **Online Softmax** 的迭代实现。下面从标准注意力出发，逐步推导分块计算所需的统计量更新公式。

---

### 1. 标准注意力公式

给定矩阵 \( Q, K, V \in \mathbb{R}^{N \times d} \)，标准注意力输出为：

\[
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right) V
\]

其中 \(\text{softmax}\) 按行进行：

\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
\]

为防止指数溢出，实际计算时采用 **数值稳定的 softmax**：对每行 \( \mathbf{s} = (s_1, \dots, s_N) \)，令 \( m = \max(s_j) \)，则

\[
\text{softmax}(s_i) = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}.
\]

令 \( S = \frac{QK^T}{\sqrt{d}} \in \mathbb{R}^{N \times N} \)，\( P = \text{softmax}(S) \)（逐行）。输出 \( O = P V \)。

**标准算法的缺陷**：需要显式存储整个 \( S \) 和 \( P \) 矩阵，内存复杂度 \( O(N^2) \)，且对 HBM 的读写次数多。

---

### 2. Online Softmax 基本思想

为了在不存储完整 \( S \) 的情况下逐行计算 softmax，可以维护两个累积统计量：

- **行最大值** \( m(x) \)
- **行归一化因子** \( \ell(x) = \sum_j e^{x_j - m(x)} \)

最终 softmax 输出为 \( p_i = e^{x_i - m} / \ell \)。

如果我们**分块**处理 \( S \) 的行（对应 Q 的分块），并且每个块内只能看到部分列（对应 K 的分块），就需要**合并不同块的统计量**。这正是 Flash Attention 中分块合并公式的来源。

---

### 3. 分块合并公式推导

假设我们正在计算输出矩阵 \( O \) 的第 \( i \) 行（对应 \( Q_i \)），但整个 \( K, V \) 被分成若干列块 \( K^{(j)}, V^{(j)} \)。对于当前块 \( j \)，我们只能计算该块对应的部分 softmax：

\[
S^{(j)} = Q_i (K^{(j)})^T / \sqrt{d} \in \mathbb{R}^{1 \times B_c}
\]

记该块内的原始得分向量为 \( \mathbf{s}^{(j)} \)（长度为 \( B_c \)）。为进行数值稳定，需要该块内的局部最大值 \( m^{(j)} = \max(\mathbf{s}^{(j)}) \)，局部指数和 \( \ell^{(j)} = \sum e^{s^{(j)}_t - m^{(j)}} \)。

但最终 softmax 需要**全局**最大值 \( m = \max( \text{all blocks' } \mathbf{s} ) \) 和全局指数和 \( \ell = \sum_{\text{all } j} \sum_t e^{s^{(j)}_t - m} \)。当按顺序处理块时，我们需要一种方式将前 \( j-1 \) 个块的统计量 \( (m_{\text{old}}, \ell_{\text{old}}) \) 与当前块的统计量 \( (m^{(j)}, \ell^{(j)}) \) 合并为新的全局统计量 \( (m_{\text{new}}, \ell_{\text{new}}) \)。

#### 合并公式

设已有统计量（来自前 \( j-1 \) 个块）：
\[
m_{\text{old}} = \max( \text{已处理的所有得分} ), \quad \ell_{\text{old}} = \sum_{\text{已处理}} e^{s_t - m_{\text{old}}}.
\]
当前块的最大值为 \( m^{(j)} \)，局部指数和为 \( \ell^{(j)} = \sum_{t \in \text{block } j} e^{s_t - m^{(j)}} \)。

**情况 1**：\( m^{(j)} \le m_{\text{old}} \)  
新全局最大值 \( m_{\text{new}} = m_{\text{old}} \)。  
旧块的指数需要重新按新最大值缩放：旧贡献变为 \( \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} = \ell_{\text{old}} \)。（因为 \( m_{\text{new}}=m_{\text{old}} \)）  
当前块贡献需从 \( e^{s_t - m^{(j)}} \) 调整为 \( e^{s_t - m_{\text{new}}} = e^{s_t - m^{(j)}} \cdot e^{m^{(j)} - m_{\text{new}}} = e^{s_t - m^{(j)}} \cdot e^{m^{(j)} - m_{\text{old}}} \)。  
所以当前块调整后的和为 \( \ell^{(j)} \cdot e^{m^{(j)} - m_{\text{old}}} \)。  
因此：
\[
\ell_{\text{new}} = \ell_{\text{old}} + \ell^{(j)} \cdot e^{m^{(j)} - m_{\text{old}}}.
\]

**情况 2**：\( m^{(j)} > m_{\text{old}} \)  
新全局最大值 \( m_{\text{new}} = m^{(j)} \)。  
旧块贡献需从 \( e^{s_t - m_{\text{old}}} \) 调整为 \( e^{s_t - m_{\text{new}}} = e^{s_t - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} \)，所以旧块调整后和为 \( \ell_{\text{old}} \cdot e^{m_{\text{old}} - m^{(j)}} \)。  
当前块无需调整（因为它的局部最大值就是新全局最大值），其和为 \( \ell^{(j)} \)。  
因此：
\[
\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m^{(j)}} + \ell^{(j)}.
\]

综合两种情形，可以统一写成：
\[
m_{\text{new}} = \max(m_{\text{old}}, m^{(j)})
\]
\[
\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \ell^{(j)} \cdot e^{m^{(j)} - m_{\text{new}}}.
\]

这就是 Flash Attention 中合并块统计量的核心公式。

---

### 4. 输出值的增量更新

除了统计量，还需要逐步累积输出 \( O_i = \sum_{\text{all blocks}} \text{softmax\_block} \cdot V_{\text{block}} \)。假设已经计算出前 \( j-1 \) 个块的部分输出 \( O_{\text{old}} \)，且当时使用的全局最大值是 \( m_{\text{old}} \)，归一化因子是 \( \ell_{\text{old}} \)。现在处理第 \( j \) 个块，该块内未归一化的权重为 \( \tilde{P}^{(j)} = e^{S^{(j)} - m^{(j)}} \)，归一化后的块内 softmax 为 \( P^{(j)} = \tilde{P}^{(j)} / \ell^{(j)} \)，该块对输出的贡献为 \( P^{(j)} V^{(j)} \)。

但当我们合并统计量后，最终的 softmax 权重应该用全局最大值 \( m_{\text{new}} \) 和全局归一化因子 \( \ell_{\text{new}} \) 来重新缩放。因此，旧的输出 \( O_{\text{old}} \) 是按旧统计量计算的，需要按比例缩放，再加上新块的贡献。

具体推导：

最终输出 \( O_{\text{new}} = \sum_{\text{all blocks}} \frac{e^{S^{(b)} - m_{\text{new}}}}{\ell_{\text{new}}} V^{(b)} \)。

将和分解为“旧块部分”和“当前块部分”：

旧块部分：之前已经计算了 \( O_{\text{old}} = \sum_{\text{old blocks}} \frac{e^{S^{(b)} - m_{\text{old}}}}{\ell_{\text{old}}} V^{(b)} \)。  
为了将其转换为以 \( m_{\text{new}} \) 为最大值的形式，我们需要将每个 \( e^{S^{(b)} - m_{\text{old}}} \) 变为 \( e^{S^{(b)} - m_{\text{new}}} = e^{S^{(b)} - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} \)。同时分母从 \( \ell_{\text{old}} \) 变为 \( \ell_{\text{new}} \)。因此旧块的正确贡献为：

\[
O_{\text{old}} \cdot \frac{\ell_{\text{old}}}{\ell_{\text{new}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}.
\]

当前块贡献：当前块未归一化权重 \( \tilde{P}^{(j)} = e^{S^{(j)} - m^{(j)}} \)，将其缩放为全局最大值后，权重变为 \( e^{S^{(j)} - m_{\text{new}}} = \tilde{P}^{(j)} \cdot e^{m^{(j)} - m_{\text{new}}} \)。然后除以 \( \ell_{\text{new}} \) 得到最终 softmax，乘以 \( V^{(j)} \) 后加和。当前块对输出的贡献为：

\[
\frac{ e^{m^{(j)} - m_{\text{new}}} }{\ell_{\text{new}}} \cdot \left( \tilde{P}^{(j)} V^{(j)} \right).
\]

注意 \( \tilde{P}^{(j)} V^{(j)} \) 正是我们计算当前块时得到的“局部输出”（按局部最大值归一化）。记 \( O_{\text{local}}^{(j)} = \tilde{P}^{(j)} V^{(j)} \)。

于是输出更新公式为：

\[
O_{\text{new}} = O_{\text{old}} \cdot \frac{\ell_{\text{old}}}{\ell_{\text{new}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \frac{e^{m^{(j)} - m_{\text{new}}}}{\ell_{\text{new}}} \cdot O_{\text{local}}^{(j)}.
\]

这等价于：

\[
O_{\text{new}} = \frac{ \ell_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}} O_{\text{old}} + e^{m^{(j)} - m_{\text{new}}} O_{\text{local}}^{(j)} }{ \ell_{\text{new}} }.
\]

---

### 5. Flash Attention 完整算法流程（前向）

输入：\( Q, K, V \in \mathbb{R}^{N \times d} \)，分块大小 \( B_c, B_r \)（通常 \( B_c \times d \) 能放入 SRAM）

1. 将 \( Q \) 分成 \( T_r = \lceil N / B_r \rceil \) 个块 \( Q_1, \dots, Q_{T_r} \)，每块大小 \( B_r \times d \)。
2. 将 \( K, V \) 分成 \( T_c = \lceil N / B_c \rceil \) 个块 \( K_1, V_1, \dots, K_{T_c}, V_{T_c} \)，每块大小 \( B_c \times d \)。
3. 初始化输出 \( O = \mathbf{0} \in \mathbb{R}^{N \times d} \)，以及每个 Q 块对应的统计量 \( \ell_i, m_i \)（但算法中逐块处理）。
4. **外层循环**：对于每个 Q 块 \( Q_i \)：
   - 初始化该 Q 块的统计量：\( m = -\infty \), \( \ell = 0 \), \( O_i = \mathbf{0} \)（对应输出的第 \( i \) 块行）。
   - **内层循环**：对于每个 KV 块 \( (K_j, V_j) \)：
        - 将 \( Q_i, K_j, V_j \) 加载到 SRAM。
        - 计算 \( S_{ij} = Q_i K_j^T / \sqrt{d} \)（形状 \( B_r \times B_c \)）。
        - 计算块内最大值 \( m_{ij} = \max(S_{ij}) \)（按行，形状 \( B_r \)）。
        - 计算块内指数和 \( \ell_{ij} = \sum_{\text{cols}} e^{S_{ij} - m_{ij}} \)（按行，形状 \( B_r \)）。
        - 更新全局统计量（对每个行位置独立）：
            \[
            m_{\text{new}} = \max(m, m_{ij}), \quad \ell_{\text{new}} = \ell \cdot e^{m - m_{\text{new}}} + \ell_{ij} \cdot e^{m_{ij} - m_{\text{new}}}.
            \]
        - 计算局部输出 \( O_{\text{local}} = \left( e^{S_{ij} - m_{ij}} \right) V_j \)（形状 \( B_r \times d \)）。
        - 更新输出：
            \[
            O_i = O_i \cdot \frac{\ell}{\ell_{\text{new}}} \cdot e^{m - m_{\text{new}}} + \frac{e^{m_{ij} - m_{\text{new}}}}{\ell_{\text{new}}} \cdot O_{\text{local}}.
            \]
        - 将 \( m \leftarrow m_{\text{new}}, \ell \leftarrow \ell_{\text{new}} \)。
   - 将 \( O_i \) 写回 HBM 中 \( O \) 的对应行。

算法结束时 \( O \) 即为精确的注意力输出。

---

### 6. 反向传播中的重计算

Flash Attention 在反向传播时**不存储**前向的 \( S \) 矩阵（即 \( QK^T \)），而是存储每个块的统计量 \( m_{ij}, \ell_{ij} \) 以及输出 \( O \)（或等效信息）。当需要计算梯度时，根据存储的统计量在 SRAM 中**重新计算** \( S_{ij} \) 和 \( P_{ij} = \text{softmax}(S_{ij}) \)，然后计算对 \( Q, K, V \) 的梯度。由于重计算只涉及分块加载和少量浮点运算，其额外开销远小于从 HBM 读取巨大 \( S \) 矩阵的成本，从而既节省显存又保持高速。


## 延迟归一化 FlashAttention V1 vs V2

### 1. 原始公式（FlashAttention V1 风格）

假设已经处理完前 \(j-1\) 个 KV 块，得到：
- 全局最大值 \(m_{\text{old}}\)
- 全局指数和 \(\ell_{\text{old}}\)
- 已归一化的输出 \(O_{\text{old}} = \frac{ \sum_{k=1}^{j-1} e^{S_{ik} - m_{\text{old}}} V_k }{ \ell_{\text{old}} }\)

现在处理第 \(j\) 个块，计算：
- 块内最大值 \(m^{(j)} = \max(S_{ij})\)
- 块内指数和 \(\ell_{\text{local}}^{(j)} = \sum e^{S_{ij} - m^{(j)}}\)
- 块内局部输出（已归一化）\(O_{\text{local}}^{(j)} = \frac{ \sum e^{S_{ij} - m^{(j)}} V_j }{ \ell_{\text{local}}^{(j)} }\)

然后更新全局统计量：
- \(m_{\text{new}} = \max(m_{\text{old}}, m^{(j)})\)
- \(\ell_{\text{new}} = \ell_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}} + \ell_{\text{local}}^{(j)} e^{m^{(j)} - m_{\text{new}}}\)

最终新输出为：
\[
O_{\text{new}} = \frac{ \ell_{\text{old}} e^{m_{\text{old}} - m_{\text{new}}} O_{\text{old}} + \ell_{\text{local}}^{(j)} e^{m^{(j)} - m_{\text{new}}} O_{\text{local}}^{(j)} }{ \ell_{\text{new}} }
\tag{1}
\]
这正是你给出的公式。

### 2. 引入“未归一化累加器”

定义**未归一化的累积输出**：
\[
\tilde{O}_{\text{old}} = \ell_{\text{old}} O_{\text{old}} = \sum_{k=1}^{j-1} e^{S_{ik} - m_{\text{old}}} V_k
\]
\[
\tilde{O}_{\text{local}}^{(j)} = \ell_{\text{local}}^{(j)} O_{\text{local}}^{(j)} = \sum e^{S_{ij} - m^{(j)}} V_j
\]

注意 \(\tilde{O}\) 直接是 **指数乘以 V 的和**，没有除以任何分母。

用 \(\tilde{O}\) 重写 (1) 的分子：
\[
\text{分子} = e^{m_{\text{old}} - m_{\text{new}}} \tilde{O}_{\text{old}} + e^{m^{(j)} - m_{\text{new}}} \tilde{O}_{\text{local}}^{(j)}
\]
分母仍是 \(\ell_{\text{new}}\)。

于是：
\[
O_{\text{new}} = \frac{ e^{m_{\text{old}} - m_{\text{new}}} \tilde{O}_{\text{old}} + e^{m^{(j)} - m_{\text{new}}} \tilde{O}_{\text{local}}^{(j)} }{ \ell_{\text{new}} }
\tag{2}
\]

### 3. 递推更新未归一化累加器

定义新的未归一化累加器：
\[
\tilde{O}_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \tilde{O}_{\text{old}} + e^{m^{(j)} - m_{\text{new}}} \tilde{O}_{\text{local}}^{(j)}
\]
那么 (2) 简化为：
\[
O_{\text{new}} = \frac{ \tilde{O}_{\text{new}} }{ \ell_{\text{new}} }
\tag{3}
\]

**关键观察**：在每一步更新中，我们**不需要**立即计算 \(O_{\text{new}}\)（即做除法）。我们只需要：
- 更新 \(\tilde{O}\)（使用乘加运算）
- 更新 \(\ell_{\text{new}}\)（同样用乘加）
- 更新 \(m_{\text{new}}\)

直到**所有 KV 块处理完毕**，得到最终的 \(\tilde{O}_{\text{final}}\) 和 \(\ell_{\text{final}}\)，然后只做**一次**除法：
\[
O_{\text{final}} = \frac{ \tilde{O}_{\text{final}} }{ \ell_{\text{final}} }
\]

### 4. 这就是“延迟归一化”的简化形式

原始 V1 公式每处理一个块就要计算 \(O_{\text{new}}\)，涉及昂贵的逐元素除法（除以 \(\ell_{\text{new}}\)）。  
而简化后的递推：
\[
\boxed{ \tilde{O}_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \tilde{O}_{\text{old}} + e^{m^{(j)} - m_{\text{new}}} \tilde{O}_{\text{local}}^{(j)} }
\]
只包含乘法、加法，没有除法。除法被推迟到最后一步，因此计算量大幅降低。

### 5. 直观总结

| 步骤 | 原始 V1 做法 | 简化 V2 做法 |
|------|--------------|----------------|
| 每个块后 | 更新 \(m, \ell, O\)，其中 \(O\) 需要除以 \(\ell\) | 更新 \(m, \ell, \tilde{O}\)，\(\tilde{O}\) 不用除 |
| 所有块后 | 已得到最终 \(O\) | 做一次 \(O = \tilde{O} / \ell\) |
| 除法次数 | 块数次 | **1 次** |



## V1-V4 FlashAttention 系列演进
始于一个简单的洞察：标准注意力机制的最大瓶颈并非计算本身，而是 GPU 内不同存储层级之间频繁的数据搬运。每一代版本都如同一场手术，精准地切入并优化了某个特定的效率瓶颈，以逼近硬件的理论极限。

下面是各代版本的核心差异总览：

| 特性 | FlashAttention V1 | FlashAttention V2 | FlashAttention V3 | FlashAttention V4 |
| :--- | :--- | :--- | :--- | :--- |
| **发布年份** | 2022 | 2023 | 2024 | 2026 |
| **核心瓶颈** | HBM 访存是主要瓶颈 | 并行度不足，GPU 利用率低 | 计算与访存无法重叠 | Tensor Core 与其他单元性能增长不匹配 |
| **硬件架构** | 通用 GPU (如 A100) | Ampere (如 A100) | Hopper (如 H100) | Blackwell (如 B200) |
| **核心算法创新** | **分块 + 在线 Softmax**，将大矩阵分块在 SRAM 中计算，避免存储 `NxN` 中间矩阵，将访存复杂度从 `O(N^2)` 降至 `O(N)`。 | **延迟归一化** + **序列长度维度并行** + **任务划分优化**，大幅减少非矩阵运算，提高 GPU 占用率。 | **Warp 专用化** + **GEMM-Softmax 交错执行** + **FP8 低精度支持**，利用 Hopper 架构特性实现异步执行，隐藏访存延迟。 | **算法-内核协同流水线设计** + **软件模拟指数函数** + **全异步 MMA**，进一步精细化流水线，压榨 Blackwell 新架构的性能。 |
| **性能提升 (相对)** | 相较标准 Attention 实现，速度提升 **2-4倍**。 | 相较 V1，速度提升约 **2倍**，A100 上可达 **225 TFLOPs/s**。 | 相较 V2 (H100 FP16)，速度提升 **1.5-2.0倍**，FP16 性能达 **740 TFLOPs/s**，硬件利用率提升至 **75%**。 | 相较 cuDNN (B200 BF16)，速度提升 **1.3倍**，性能达 **1613 TFLOPs/s**，硬件利用率达 **71%**。 |
| **主要限制/不足** | 并行度不高，GPU 利用率仅 25-40%。 | 部分非矩阵运算未完全消除，在特定架构下仍有优化空间。 | 主要为 Hopper 架构设计，对 Blackwell 等新架构适配不足。 | 目前主要针对前向推理优化，暂不支持反向传播、变长序列等高级特性。 |

### 💡 V1: 奠基之作，以分块思想突破显存墙

FlashAttention V1 的核心洞察是：标准 Attention 的内存瓶颈在于需要存储巨大的 `S` 和 `P` 矩阵。其解决方案是**分块 (Tiling)** 和 **在线 Softmax (Online Softmax)**。

它将庞大的 `Q, K, V` 矩阵切成小块，使其能在容量有限但速度极快的 GPU 片上高速缓存 (SRAM) 中完成计算。每一步，它只加载一小块 `Q`，一小块 `K` 和一小块 `V`，在 SRAM 中计算该块的注意力分数并立即与对应的 `V` 块相乘，得到部分输出 `O`。通过在线 Softmax 算法维护的全局统计量（最大值 `m` 和指数和 `l`），V1 能正确地将这些部分输出合并成最终结果，全程无需将巨大的中间矩阵写回高带宽内存 (HBM)。

这项创新将内存复杂度从 `O(N²)` 降至 `O(N)`，并将标准 Attention 的速度提升了 2-4 倍。然而，V1 在 GPU 利用率上存在短板，仅能发挥 A100 约 25-40% 的理论峰值算力。

### 🚀 V2: 并行革命，向算法要效率

V2 敏锐地发现了 V1 的症结：**并行度不足**和**大量低效的非矩阵运算**。它通过三项关键优化，实现了约 2 倍的性能飞跃。

*   **增加序列维度的并行**：V1 的并行主要在 batch 和 head 上，当 batch 或 head 较少时，GPU 大量核心闲置。V2 则允许**多个线程块 (Thread Block) 并行处理同一个注意力头内的不同 Query 行**，极大提升了 GPU 的并行度和占用率。
*   **削减非矩阵乘运算 (Non-Matmul)**：V2 的核心优化是**延迟归一化 (Delayed Normalization)**，它让 Softmax 计算过程中的除法操作被推迟到循环的最后，大幅减少了迭代过程中的开销。这个看似微小的改动，却因 GPU 执行非矩阵运算的速度远慢于矩阵乘，而带来了显著的性能提升。
*   **优化 Warp 级别任务划分**：V2 还优化了线程束 (Warp) 之间的任务分配，减少了通信和同步开销，进一步压榨了 GPU 性能。

这些改进使 V2 在 A100 上实现了高达 225 TFLOPs/s 的训练速度，将 V1 的 GPU 利用率从 25-40% 提升至 **50-73%**。

### ⚙️ V3: 架构专精，拥抱异步新纪元

V3 是为 NVIDIA **Hopper** 架构（如 H100）量身打造的版本，不再局限于算法，而是深入利用了硬件的新特性。

*   **Warp 专用化与异步执行**：V3 利用 Hopper 的张量内存加速器 (TMA) 和新的 **WGMMA（Warp Group Matrix Multiply-Accumulate）** 指令，将部分线程束专门用于异步数据加载（生产者），另一部分专用于计算（消费者），实现了计算与数据搬运的高度重叠，有效隐藏了访存延迟。
*   **GEMM 与 Softmax 流水线交错 (Block Interleaving)**：V3 通过精细的调度，使得矩阵乘法 (GEMM) 和 Softmax 操作可以在时间上重叠执行，避免了 GPU 执行单元的空闲，进一步提升了流水线效率。
*   **FP8 低精度支持**：V3 开始支持 **FP8 低精度计算**。在 H100 上，FP8 的 Tensor Core 吞吐量是 FP16 的两倍。V3 通过引入块级量化和非相干处理等技术，在保证精度的前提下，充分挖掘了这一潜力。

凭借这些硬件级优化，V3 在 H100 上达到了 75% 的硬件利用率，速度相比 V2 再提升 1.5-2 倍。

### 🏭 V4: 软硬协同，为 Blackwell 重构流水线

面对算力更强劲的 **Blackwell** 架构（如 B200），V4 发现新的瓶颈：Tensor Core 的算力翻倍，但其他单元（如共享内存带宽、指数计算单元）的性能并未同步提升。V4 的策略是“算法与内核协同设计 (Algorithm and Kernel Co-Design)”，通过更精细的流水线来解决新的不平衡。

*   **全异步 MMA (矩阵乘-累加) 与流水线重设计**：V4 针对 Blackwell 支持的全异步 MMA 操作重新设计了流水线，使其能够更大程度地并行化。
*   **软件模拟指数函数**：为了缓解专用单元 (SFU) 的瓶颈，V4 使用 CUDA 核心上的普通指令**通过软件模拟指数函数 (exp2)**，从而释放了宝贵的 SFU 资源，避免了硬件争用。
*   **条件性在线重缩放**：V4 引入了“条件性”的在线重缩放，仅在统计量发生显著变化时才执行昂贵的重缩放操作，进一步减少了计算量。

V4 最终在 B200 GPU 上实现了高达 1613 TFLOPs/s 的 BF16 性能（71% 利用率），甚至超越了 NVIDIA 官方优化库 cuDNN 的实现。


### 💎 总结
FlashAttention 的演进史，也是一部如何与硬件对话、挖掘极致性能的编年史。V1 解决了内存墙，V2 提升了并行度，V3 利用了硬件异步性，V4 则通过精细的流水线协同，在新一代硬件上实现了软硬件的完美配合。