# 第一章：激活函数与基础组件

本章涵盖神经网络最底层的构建模块：激活函数、线性变换、嵌入、正则化和损失函数。

---

## 1.1 ReLU（Rectified Linear Unit）

### 是什么
ReLU 是最简单也最常用的激活函数，将所有负值置零，正值保持不变。

### 数学定义

$$\text{ReLU}(x) = \max(0, x)$$

等价实现：`x * (x > 0)`，利用布尔掩码与逐元素乘法。

### 为什么需要
神经网络需要非线性变换来拟合复杂函数。没有激活函数，多层线性层的叠加仍然是线性变换。ReLU 相比 Sigmoid/Tanh 的优势：
- 计算极快（仅比较和乘法）
- 正区间梯度恒为 1，缓解梯度消失
- 产生稀疏激活（负值区域输出为 0）

### 已知问题
- **Dead ReLU**：如果某个神经元的输入始终为负，梯度永远为 0，该神经元"死亡"
- 输出非零中心化，可能影响后续层的收敛

### 代码示例

```python
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0).float()

# 测试
x = torch.tensor([-2., -1., 0., 1., 2.])
print(relu(x))  # tensor([0., 0., 0., 1., 2.])

# 验证梯度流
x = torch.tensor([-1., 0., 1.], requires_grad=True)
y = relu(x).sum()
y.backward()
print(x.grad)  # tensor([0., 0., 1.])  — 负值区域梯度为 0
```

### 适用场景
- CNN 隐藏层的默认激活函数
- 简单前馈网络
- 不适合输出层（因为只有非负输出）

---

## 1.2 GELU（Gaussian Error Linear Unit）

### 是什么
GELU 是一种平滑的激活函数，可以看作 ReLU 的概率化版本。它根据输入值的大小，以一定概率"保留"或"丢弃"该值。

### 数学定义

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot 0.5 \cdot (1 + \text{erf}(x / \sqrt{2}))$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数（CDF），$\text{erf}$ 是误差函数。

### 为什么需要
- 比 ReLU 更平滑，在 x=0 附近有连续的梯度过渡
- 允许小的负值通过（不像 ReLU 完全截断），缓解 Dead ReLU 问题
- 在 Transformer 架构中表现优于 ReLU

### 直觉理解
GELU(x) ≈ x（当 x 很大时），GELU(x) ≈ 0（当 x 很小时）。过渡区域是平滑的 S 形曲线，而非 ReLU 的硬拐点。

### 代码示例

```python
import torch
import math

def my_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 测试
x = torch.tensor([-2., -1., 0., 1., 2.])
print(my_gelu(x))
# tensor([-0.0454, -0.1587,  0.0000,  0.8413,  1.9546])
# 注意：负值区域不完全为 0，而是接近 0 的小值
```

### 适用场景
- Transformer 中 FFN 的默认激活函数（BERT、GPT-2）
- 现代 LLM 中常被 SiLU/Swish 替代（用于 SwiGLU）

---

## 1.3 Softmax

### 是什么
Softmax 将任意实数向量转换为概率分布（所有元素非负且和为 1）。

### 数学定义

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

### 数值稳定性
直接计算 $e^{x_i}$ 在 $x_i$ 很大时会溢出。解决方法：先减去最大值。

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

这不改变结果（分子分母同乘常数），但避免了数值溢出。

### 代码示例

```python
import torch

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

# 测试
x = torch.tensor([1.0, 2.0, 3.0])
print(my_softmax(x))  # tensor([0.0900, 0.2447, 0.6652])
print(my_softmax(x).sum())  # tensor(1.)

# 数值稳定性测试
x_large = torch.tensor([1000., 1001., 1002.])
print(my_softmax(x_large))  # 正常输出，不会 NaN
```

### 适用场景
- 分类任务的输出层
- 注意力权重计算
- 任何需要将 logits 转为概率的场景

---

## 1.4 Linear Layer（全连接层）

### 是什么
线性层执行仿射变换 $y = xW^T + b$，是神经网络最基本的参数化组件。

### 结构说明
- `weight`：形状 `(out_features, in_features)`，可学习参数
- `bias`：形状 `(out_features,)`，可学习参数
- 前向计算：`x @ weight.T + bias`

### 权重初始化
使用 $W \sim \mathcal{N}(0, 1/\sqrt{n_{in}})$ 初始化，确保输出方差不随层数增长而爆炸或消失。

### 代码示例

```python
import torch

class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        # Xavier/LeCun 风格初始化
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features) / (in_features ** 0.5)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias

# 测试
layer = SimpleLinear(8, 4)
x = torch.randn(2, 8)
y = layer.forward(x)
print(y.shape)  # torch.Size([2, 4])
```

### 关键概念：requires_grad
`nn.Parameter` 自动设置 `requires_grad=True`，使得 PyTorch 的 autograd 系统能追踪梯度，实现反向传播。

---

## 1.5 Embedding Layer（嵌入层）

### 是什么
嵌入层是一个可学习的查找表，将离散的整数索引（如 token ID）映射为连续的稠密向量。

### 工作原理
本质上就是一个矩阵索引操作：`weight[indices]`。权重矩阵形状为 `(vocab_size, embed_dim)`，输入是整数张量，输出是对应行的向量。

### 为什么需要
- 神经网络无法直接处理离散符号（如单词、字符）
- 嵌入向量可以学习到语义关系（相似词的向量距离近）
- 比 one-hot 编码高效得多（one-hot 维度 = 词表大小，通常 30k-100k）

### 代码示例

```python
import torch
import torch.nn as nn

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]

# 测试
emb = MyEmbedding(1000, 64)  # 词表大小 1000，嵌入维度 64
idx = torch.tensor([0, 42, 999])
print(emb(idx).shape)  # torch.Size([3, 64])

# 支持任意形状的索引
batch_idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(emb(batch_idx).shape)  # torch.Size([2, 3, 64])
```

### 适用场景
- NLP 模型的第一层（word embedding）
- 推荐系统中的 item/user embedding
- 任何需要将离散 ID 转为向量的场景

---

## 1.6 Dropout

### 是什么
Dropout 是一种正则化技术。训练时随机将一部分神经元的输出置零，推理时不做任何操作。

### 算法
- 训练模式：以概率 $p$ 将每个元素置零，剩余元素乘以 $\frac{1}{1-p}$（inverted dropout）
- 推理模式：直接返回输入（恒等映射）

### 为什么要缩放
训练时丢弃了 $p$ 比例的神经元，如果不缩放，推理时（所有神经元都活跃）的期望输出会比训练时大。乘以 $\frac{1}{1-p}$ 使得训练和推理时的期望输出一致。

### 代码示例

```python
import torch
import torch.nn as nn

class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1.0 - self.p)

# 测试
d = MyDropout(p=0.5)
x = torch.ones(10)

d.train()
print(d(x))  # 约一半为 0，其余为 2.0（= 1.0 / 0.5）

d.eval()
print(d(x))  # tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

### 适用场景
- Transformer 中的注意力权重和 FFN 输出
- 全连接层之间
- 通常 p=0.1（Transformer）或 p=0.5（经典网络）

---

## 1.7 Cross-Entropy Loss（交叉熵损失）

### 是什么
交叉熵损失是分类任务的标准损失函数，衡量模型预测的概率分布与真实标签之间的差异。

### 数学定义

$$\text{CE}(x, y) = -\log\frac{e^{x_y}}{\sum_j e^{x_j}}$$

其中 $x$ 是 logits 向量（未归一化的分数），$y$ 是正确类别的索引。

展开后等价于：

$$\text{CE}(x, y) = -x_y + \log\sum_j e^{x_j}$$

### 数值稳定性：LogSumExp 技巧

$$\log\sum_j e^{x_j} = m + \log\sum_j e^{x_j - m}, \quad m = \max(x)$$

### 代码示例

```python
import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: (B, C), targets: (B,)
    # 使用 logsumexp 保证数值稳定
    log_sum_exp = torch.logsumexp(logits, dim=-1)  # (B,)
    # 取出正确类别的 logit
    correct_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
    # CE = -log(softmax) = -(logit - logsumexp)
    loss = log_sum_exp - correct_logits
    return loss.mean()

# 测试
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])
print(cross_entropy_loss(logits, targets))
# 与 PyTorch 内置实现对比
print(torch.nn.functional.cross_entropy(logits, targets))
```

### 与 Softmax 的关系
Cross-Entropy Loss = Softmax + Negative Log-Likelihood。实践中直接对 logits 计算（而非先 softmax 再取 log），数值更稳定。

### 适用场景
- 所有分类任务（图像分类、语言模型的 next-token prediction）
- 语言模型训练中，对每个位置的 token 预测计算 CE loss
