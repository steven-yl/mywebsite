---
title: "Pytorch 权重初始化方法"
date: 2026-03-12T00:00:00+08:00
draft: false
authors: [Steven]
description: "系统梳理深度学习中所有主流权重初始化方法，从零初始化到 Xavier、He、正交初始化，涵盖原理推导、方差分析、PyTorch 代码示例与 Transformer 初始化实践。"
summary: "全面对比深度学习权重初始化方法的原理、公式推导、优缺点与适用场景，附 PyTorch 代码示例和 Transformer 架构初始化最佳实践。"
tags: ["PyTorch", "Deep Learning", "initialization", "Xavier", "Kaiming"]
categories: ["PyTorch"]
series: ["PyTorch 实践指南"]
weight: 2
series_weight: 2
---


深度学习的初始化方法是神经网络训练的基础，它决定了模型能否有效收敛、收敛速度以及最终性能。下面我将系统性地梳理所有主流初始化方法，从原理、优缺点到适用场景进行全面对比。

---

## 一、初始化方法分类

按设计思想，可分为以下几类：

1. **基础随机初始化**
2. **方差缩放类**（Xavier / He 等）
3. **正交/结构化初始化**
4. **自适应/数据驱动初始化**
5. **预训练初始化**

---

## 二、各类初始化详解

### 1. 零初始化与常数初始化

**做法**：所有权重设为相同常数（如0、0.1等），偏置常设为0或小常数。

**原理**：极其简单。

**优点**：实现简单，适合调试或偏置初始化。

**缺点**：
- 对称性问题：同一层所有神经元输出完全相同，梯度相同，导致所有神经元学习相同特征，网络无法训练。
- 几乎从不用于权重初始化（除极特殊结构）。

**适用场景**：仅偏置可初始化为0，权重几乎不用。

**PyTorch 示例**：
```python
import torch.nn as nn

# 偏置初始化为 0（默认行为）
nn.init.zeros_(layer.bias)

# 常数初始化（仅用于特殊场景）
nn.init.constant_(layer.weight, 0.01)
```

---

### 2. 随机初始化（Uniform / Normal）

**做法**：从均匀分布或高斯分布中随机采样，常配合一个缩放因子，如：
$$
W \sim U[-a, a] \quad \text{或} \quad W \sim N(0, \sigma^2)
$$
早期常用 $ a = 1/\sqrt{n_{\text{in}}} $ 或简单取 $ a=0.01 $。

**优点**：打破对称性，简单。

**缺点**：
- 若缩放因子不当，易导致梯度消失或爆炸（尤其深层网络）。
- 对层数敏感，不适用于深度网络。

**适用场景**：浅层网络（如2-3层），或作为其他方法的基。

**PyTorch 示例**：
```python
# 均匀分布
nn.init.uniform_(layer.weight, a=-0.1, b=0.1)

# 正态分布
nn.init.normal_(layer.weight, mean=0.0, std=0.01)
```

---

### 3. Xavier / Glorot 初始化

**提出**：Glorot & Bengio, 2010

**核心思想**：保持前向传播与反向传播中信号的方差一致，避免梯度消失/爆炸。

**公式**：
- **均匀分布**：
  $$
  W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right]
  $$
- **正态分布**：
  $$
  W \sim N\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
  $$
其中 $ n_{\text{in}} $ 为输入维度，$ n_{\text{out}} $ 为输出维度。

**优点**：
- 有效缓解深层网络的梯度消失/爆炸问题（针对S型激活函数）。
- 理论扎实，被广泛使用。

**缺点**：
- 假设激活函数是线性的，对于ReLU等非线性激活函数效果不佳（因ReLU会屏蔽一半神经元，方差减半）。

{{< admonition tip "Xavier 方差推导" false >}}
考虑一个线性层 $ y = Wx $，其中 $ W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}} $，$ x \in \mathbb{R}^{n_{\text{in}}} $。

假设：
- $ w_{ij} $ 和 $ x_j $ 相互独立
- $ \mathbb{E}[w_{ij}] = 0 $，$ \mathbb{E}[x_j] = 0 $
- 所有 $ w_{ij} $ 同分布，所有 $ x_j $ 同分布

对于输出的第 $ i $ 个元素：
$$
y_i = \sum_{j=1}^{n_{\text{in}}} w_{ij} x_j
$$

计算方差（利用独立性和零均值）：
$$
\text{Var}(y_i) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(w_{ij} x_j) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x)
$$

**前向传播约束**：为保持 $ \text{Var}(y) = \text{Var}(x) $，需要：
$$
n_{\text{in}} \cdot \text{Var}(w) = 1 \quad \Rightarrow \quad \text{Var}(w) = \frac{1}{n_{\text{in}}}
$$

**反向传播约束**：类似地，为保持梯度方差不变，需要：
$$
n_{\text{out}} \cdot \text{Var}(w) = 1 \quad \Rightarrow \quad \text{Var}(w) = \frac{1}{n_{\text{out}}}
$$

Xavier 取两者的折中：
$$
\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

对于均匀分布 $ U[-a, a] $，其方差为 $ \frac{a^2}{3} $，令 $ \frac{a^2}{3} = \frac{2}{n_{\text{in}} + n_{\text{out}}} $，解得：
$$
a = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}
$$
{{< /admonition >}}

**适用场景**：
- 激活函数为 **tanh、sigmoid、线性** 的全连接层或卷积层。
- 在早期深度网络（如RNN中的tanh）中表现良好。

**PyTorch 示例**：
```python
# Xavier 均匀分布（默认 gain=1，适用于线性/tanh）
nn.init.xavier_uniform_(layer.weight, gain=1.0)

# Xavier 正态分布
nn.init.xavier_normal_(layer.weight, gain=1.0)

# 对于 tanh 激活函数，可使用推荐的 gain 值
gain = nn.init.calculate_gain('tanh')  # ≈ 5/3
nn.init.xavier_uniform_(layer.weight, gain=gain)
```

---

### 4. He / Kaiming 初始化

**提出**：He et al., 2015（针对ReLU及其变体）

**核心思想**：考虑ReLU激活函数会使方差减半，故将方差放大一倍以补偿。

**公式**：
- **均匀分布**：
  $$
  W \sim U\left[-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right]
  $$
- **正态分布**：
  $$
  W \sim N\left(0, \frac{2}{n_{\text{in}}}\right)
  $$
对于ReLU使用 $ \frac{2}{n_{\text{in}}} $，对于Leaky ReLU等变体有相应调整（如 $ \frac{2}{(1+a^2)n_{\text{in}}} $）。

**优点**：
- 在ReLU系列激活函数下，能有效维持梯度健康，支持极深网络（如ResNet）。
- 现代深度学习默认初始化之一。

**缺点**：
- 对sigmoid/tanh等饱和激活函数不适用（方差仍会过大，导致饱和）。

**适用场景**：
- 激活函数为 **ReLU、Leaky ReLU、PReLU、ELU** 等的网络。
- 卷积神经网络（CNN）、残差网络（ResNet）、Transformer中的FFN等。

{{< admonition tip "He 初始化方差推导（ReLU 情形）" false >}}
在 Xavier 推导的基础上，考虑 ReLU 激活函数 $ f(x) = \max(0, x) $。

设输入 $ x $ 服从对称分布（均值为0），经过 ReLU 后：
$$
\text{Var}(f(x)) = \frac{1}{2} \text{Var}(x)
$$
因为 ReLU 将约一半的值置零，方差减半。

因此，经过一层线性变换 + ReLU 后：
$$
\text{Var}(y) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x) \cdot \frac{1}{2}
$$

为保持 $ \text{Var}(y) = \text{Var}(x) $：
$$
\frac{n_{\text{in}}}{2} \cdot \text{Var}(w) = 1 \quad \Rightarrow \quad \text{Var}(w) = \frac{2}{n_{\text{in}}}
$$

对于 Leaky ReLU（负半轴斜率为 $ a $），$ \text{Var}(f(x)) = \frac{1+a^2}{2} \text{Var}(x) $，因此：
$$
\text{Var}(w) = \frac{2}{(1+a^2) n_{\text{in}}}
$$
{{< /admonition >}}

**PyTorch 示例**：
```python
# He 均匀分布（fan_in 模式，适用于前向传播）
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

# He 正态分布
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# 对于 Leaky ReLU（负斜率 a=0.01）
nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

# fan_out 模式：保持反向传播方差稳定（某些场景更优）
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
```

---

### 5. LSUV（Layer-sequential Unit Variance）

**做法**：先正交初始化，再逐层归一化（通过前向数据调整每层输出的方差为1，类似批量归一化前的预调节）。

**步骤**：
1. 正交初始化所有权重。
2. 输入一批数据，逐层前向，计算每层输出的标准差。
3. 将权重除以该标准差，使输出方差为1。

**优点**：
- 可自动适配不同激活函数与网络结构。
- 训练初期梯度稳定，收敛更快。

**缺点**：
- 需要额外的前向过程（数据驱动），增加初始化开销。
- 对超参数（如输入数据分布）敏感。

**适用场景**：
- 大型或结构复杂的网络，尤其是不方便使用批量归一化（BN）但希望快速稳定训练的场景。

**PyTorch 示例**（伪代码，需根据具体网络结构调整）：
```python
import torch

def lsuv_init(model, data_batch, tol=0.1, max_iter=10):
    """Layer-sequential Unit Variance 初始化"""
    # 第一步：正交初始化所有权重
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight.dim() >= 2:
            nn.init.orthogonal_(module.weight)

    # 第二步：逐层调整方差
    model.eval()
    hooks = []
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # 注册 hook 记录每层输出
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 前向传播并调整
    with torch.no_grad():
        model(data_batch)
        for name, module in model.named_modules():
            if name in activations:
                for _ in range(max_iter):
                    std = activations[name].std()
                    if abs(std - 1.0) < tol:
                        break
                    module.weight.data /= (std + 1e-8)
                    model(data_batch)  # 重新前向

    # 移除 hooks
    for h in hooks:
        h.remove()
```

---

### 6. 正交初始化

**做法**：将权重矩阵初始化为正交矩阵（或半正交），即 $ W^T W = I $（对于方阵）或接近正交。

常用方法：对随机矩阵做QR分解或SVD得到正交基。

**优点**：
- 严格保持前向/反向信号的范数，梯度传播稳定。
- 在RNN中特别有效，可缓解梯度消失/爆炸。

**缺点**：
- 实现稍复杂（需分解）。
- 对卷积层的应用需特殊处理（如正交卷积核）。

**适用场景**：
- 循环神经网络（RNN、LSTM）的隐层权重。
- 需要严格保持范数的深度网络（如某些归一化流、等变网络）。

**PyTorch 示例**：
```python
# 正交初始化（gain=1 保持范数不变）
nn.init.orthogonal_(layer.weight, gain=1.0)

# RNN 隐层权重正交初始化
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2)
for name, param in rnn.named_parameters():
    if 'weight_hh' in name:  # 隐层到隐层的权重
        nn.init.orthogonal_(param)
    elif 'weight_ih' in name:  # 输入到隐层的权重
        nn.init.xavier_uniform_(param)

# LSTM 遗忘门偏置初始化为 1
lstm = nn.LSTM(input_size=128, hidden_size=256)
for name, param in lstm.named_parameters():
    if 'bias' in name:
        n = param.size(0)
        # LSTM 偏置顺序：[input_gate, forget_gate, cell_gate, output_gate]
        param.data[n // 4 : n // 2].fill_(1.0)  # 遗忘门偏置设为 1
```

---

### 7. 稀疏初始化

**做法**：大部分权重为0，仅少量非零（如按一定比例随机选择连接），非零值通常从高斯分布中采样。

**优点**：
- 天然产生稀疏连接，可能节省计算。
- 在某些自编码器、稀疏编码任务中有理论依据。

**缺点**：
- 现代GPU对密集矩阵运算更友好，稀疏性不总能带来加速。
- 需谨慎设置稀疏度，否则信息流动不足。

**适用场景**：
- 早期深度学习研究，或需强稀疏性约束的任务（如稀疏自编码器）。

**PyTorch 示例**：
```python
# 稀疏初始化：每列保留 10% 的非零连接
nn.init.sparse_(layer.weight, sparsity=0.9, std=0.01)
```

---

### 8. 数据驱动初始化（如自编码器逐层预训练）

**做法**：使用无监督预训练（如受限玻尔兹曼机、自编码器）逐层初始化网络，再微调。

**历史地位**：在深度学习早期（2006-2010）是训练深层网络的关键技术。

**优点**：
- 能有效初始化深层网络，避免梯度消失。
- 可学习数据相关的特征表示。

**缺点**：
- 训练成本高，步骤繁琐。
- 自ReLU、BN、He初始化等出现后，已被逐步取代。

**适用场景**：
- 现代几乎不再使用，仅在无监督预训练+微调的特定场景（如少量数据+极深网络）中可能重现。

---

### 9. 动态初始化（如Fixup、Zero Init with Skip）

**做法**：设计特殊的初始化策略，使残差网络或Transformer在无归一化层时也能稳定训练。例如Fixup将残差分支的最后一层初始化为0，其他层用He初始化。

**优点**：
- 允许去除Batch Normalization，减少内存和计算。
- 训练仍能保持稳定。

**缺点**：结构依赖性强，需根据网络定制。

**适用场景**：
- 无归一化层的深度网络（如某些高效推理模型、小批量训练场景）。

**PyTorch 示例**（Fixup 风格的残差网络初始化）：
```python
def fixup_init(model, num_layers):
    """Fixup 初始化：残差分支最后一层置零，其他层缩放"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if 'residual_last' in name:
                # 残差分支最后一层初始化为 0
                nn.init.zeros_(module.weight)
            else:
                # 其他层用 He 初始化后按层数缩放
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                module.weight.data *= num_layers ** (-0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

---

## 三、优缺点对比总结表

| 初始化方法       | 优点                                      | 缺点                                      | 适用激活函数         | 适用网络结构         |
|------------------|-------------------------------------------|-------------------------------------------|----------------------|----------------------|
| 零/常数初始化     | 简单                                      | 对称性问题，无法训练                       | 任意                 | 仅偏置，不能用于权重 |
| 随机初始化       | 打破对称性                                 | 深层梯度消失/爆炸，需精细调参               | 任意                 | 浅层网络             |
| Xavier / Glorot  | 保持信号方差，缓解梯度问题                  | 对ReLU效果不佳                            | tanh, sigmoid, 线性  | 中等深度全连接/卷积   |
| He / Kaiming     | 适配ReLU系列，支持极深网络                  | 不适用于饱和激活函数                        | ReLU, Leaky ReLU等   | CNN, ResNet, Transformer |
| 正交初始化       | 严格保持范数，RNN友好                       | 实现复杂，卷积不易适配                      | 任意（尤其tanh/线性）| RNN, LSTM, 深层线性网络 |
| LSUV             | 自动调节方差，收敛快                        | 需数据前向，对输入敏感                      | 任意                 | 复杂网络，无BN场景    |
| 稀疏初始化       | 强稀疏性，理论解释好                        | 计算不友好，稀疏度需调参                    | 任意                 | 稀疏自编码器等        |
| 数据驱动预训练    | 学习数据特征，深度网络可行                   | 计算成本高，流程复杂                        | 任意                 | 早期深度学习，现少用 |
| 动态初始化(Fixup)| 可去除BN，训练稳定                          | 结构定制，通用性差                          | ReLU等               | 无归一化的残差网络    |

---

## 四、选择建议

1. **现代通用默认**：
   - 卷积网络 + ReLU → **He均匀或正态初始化**
   - Transformer（ReLU/GeLU） → **He初始化**，且通常配合预归一化（Pre-LN）
   - 全连接网络 + ReLU → **He初始化**

2. **若使用 tanh / sigmoid**：
   - **Xavier初始化** 仍是最佳选择

3. **循环网络（RNN/LSTM）**：
   - 隐层权重建议 **正交初始化**，输入/输出权重可用Xavier/He
   - LSTM中遗忘门偏置常初始化为1或较大值

4. **无归一化层的深度网络**：
   - 可尝试 **Fixup** 或 **LSUV** 等方法

5. **偏置初始化**：
   - 通常设为0，但某些门控网络（如LSTM遗忘门）可设为1以提高初始记忆能力

6. **特殊场景**：
   - 迁移学习/微调：用预训练权重初始化（如ImageNet预训练）
   - 强化学习：常用较小的正交初始化，以保持初始探索的稳定性

---

## 五、实践注意

- 初始化的 **缩放因子** 需要与网络深度、激活函数、归一化层配合。
- 即使使用合适初始化，**学习率** 与 **优化器** 的选择仍会显著影响最终效果。
- 批量归一化（Batch Normalization）可以在一定程度上 **缓解对初始化的依赖**，但好的初始化依然能加速收敛。

---

## 六、Transformer 初始化实践

Transformer 架构中各组件的初始化策略有其特殊性，以下是主流做法的总结。

### 1. GPT 风格初始化

GPT-2 / GPT-3 采用的初始化策略：

```python
def gpt_init(model, n_layer):
    """GPT 风格初始化"""
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if 'wte' in name or 'wpe' in name:
            # Token / Position Embedding：正态分布 N(0, 0.02)
            nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'c_proj' in name:
            # 残差路径的输出投影：按层数缩放，防止残差累积导致方差爆炸
            nn.init.normal_(param, mean=0.0, std=0.02 / (2 * n_layer) ** 0.5)
        else:
            # 其他线性层：N(0, 0.02)
            nn.init.normal_(param, mean=0.0, std=0.02)
```

关键点：
- 残差路径输出投影的标准差按 $ \frac{1}{\sqrt{2N}} $ 缩放（$ N $ 为层数），防止深层残差累积导致激活值方差线性增长。
- Embedding 层使用较小的标准差（0.02），避免初始 token 表示过于分散。

### 2. BERT / Pre-LN Transformer 初始化

```python
def bert_init(model, hidden_size):
    """BERT 风格初始化"""
    std = 1.0 / hidden_size ** 0.5
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

### 3. 各组件初始化总结

| 组件 | 推荐初始化 | 说明 |
|------|-----------|------|
| Token Embedding | $ N(0, 0.02) $ 或 $ N(0, 1/\sqrt{d}) $ | 标准差不宜过大 |
| Position Embedding | 同 Token Embedding | 可学习或固定（如 sinusoidal） |
| Q/K/V 投影 | Xavier 或 $ N(0, 1/\sqrt{d}) $ | 保持注意力分数方差合理 |
| Attention 输出投影 | $ N(0, \sigma/\sqrt{2N}) $ | 残差路径缩放 |
| FFN 第一层 | He 初始化（ReLU/GeLU） | 配合激活函数 |
| FFN 第二层（输出） | $ N(0, \sigma/\sqrt{2N}) $ | 残差路径缩放 |
| LayerNorm | weight=1, bias=0 | 标准做法 |
| 分类头 / LM Head | $ N(0, 0.02) $ 或 Xavier | 视任务而定 |

---

## 七、PyTorch 通用初始化工具函数

以下是一个可直接复用的通用初始化函数，覆盖常见网络结构：

```python
import torch.nn as nn

def init_weights(model, init_type='kaiming', gain=1.0):
    """
    通用权重初始化函数

    Args:
        model: nn.Module 实例
        init_type: 初始化类型，可选 'kaiming', 'xavier', 'orthogonal', 'normal'
        gain: 缩放因子
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight, gain=gain)
            elif init_type == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                raise ValueError(f"不支持的初始化类型: {init_type}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            for param_name, param in module.named_parameters():
                if 'weight_hh' in param_name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in param_name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in param_name:
                    nn.init.zeros_(param)
                    # LSTM 遗忘门偏置设为 1
                    if isinstance(module, nn.LSTM):
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(1.0)

# 使用示例
model = MyModel()
init_weights(model, init_type='kaiming')
```

---

## 八、`calculate_gain` 参考表

PyTorch 提供 `nn.init.calculate_gain(nonlinearity, param=None)` 来获取激活函数对应的推荐增益系数，用于 Xavier 初始化中的 `gain` 参数：

| 激活函数 | gain 值 | 说明 |
|---------|---------|------|
| `'linear'` / `'identity'` | 1 | 线性激活 |
| `'sigmoid'` | 1 | S 型激活 |
| `'tanh'` | 5/3 ≈ 1.6667 | 双曲正切 |
| `'relu'` | $ \sqrt{2} $ ≈ 1.4142 | 修正线性单元 |
| `'leaky_relu'` | $ \sqrt{2 / (1 + a^2)} $ | 参数 $ a $ 为负斜率，默认 0.01 |
| `'selu'` | 3/4 = 0.75 | 自归一化线性单元 |

```python
# 使用示例
gain_relu = nn.init.calculate_gain('relu')           # 1.4142
gain_tanh = nn.init.calculate_gain('tanh')           # 1.6667
gain_lrelu = nn.init.calculate_gain('leaky_relu', 0.2)  # 1.3868

nn.init.xavier_uniform_(layer.weight, gain=gain_tanh)
```