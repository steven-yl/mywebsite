---
title: "PyTorch 概率分 (`torch.distributions`)"
date: 2026-04-13T10:00:00+08:00
draft: false
authors: [Steven]
description: "PyTorch 概率分布库 (`torch.distributions`) 完全指南"
summary: "PyTorch 概率分布库 (`torch.distributions`) 完全指南"

tags: [PyTorch]
categories: [PyTorch]
series: [PyTorch系列]
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

## 目录

1. [概述](#概述)
2. [核心概念与统一接口](#核心概念与统一接口)
3. [离散分布](#离散分布)
   - 3.1 `Categorical`（类别分布）
   - 3.2 `Bernoulli`（伯努利分布）
   - 3.3 `Binomial`（二项分布）
   - 3.4 `Poisson`（泊松分布）
   - 3.5 `OneHotCategorical` 与 `RelaxedOneHotCategorical`
4. [连续分布](#连续分布)
   - 4.1 `Normal`（正态分布）
   - 4.2 `Uniform`（均匀分布）
   - 4.3 `Beta` 与 `Gamma`
   - 4.4 `Exponential` 与 `Laplace`
   - 4.5 `Chi2` 与 `StudentT`
5. [多元分布](#多元分布)
   - 5.1 `MultivariateNormal`
   - 5.2 `Dirichlet`
6. [变换与工具](#变换与工具)
   - 6.1 `TransformedDistribution`
   - 6.2 `kl_divergence` 与 `constraints`
7. [完整示例：从离散采样到梯度传播](#完整示例)
8. [常见问题与最佳实践](#常见问题与最佳实践)

---

## 概述

`torch.distributions` 是 PyTorch 内置的概率分布库，提供了数十种常见的离散、连续及多元分布，并统一实现了采样 (`sample`)、对数概率 (`log_prob`)、熵 (`entropy`) 等核心方法。它无缝集成 PyTorch 的自动微分系统，支持 **重参数化采样** (`rsample`)，非常适合深度概率编程。

---

## 核心概念与统一接口

每个分布类都遵循相同的设计模式：

| 属性/方法               | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `sample(sample_shape)`  | 生成形状为 `sample_shape` 的随机样本（无梯度）               |
| `rsample(sample_shape)` | 生成可微分的样本（仅适用于连续分布或使用了重参数化技巧的分布） |
| `log_prob(value)`       | 计算给定值的对数概率密度/质量函数                            |
| `entropy()`             | 计算分布的熵（微分熵）                                       |
| `probs` / `logits`      | 获取概率或对数几率（对离散分布有效）                         |
| `batch_shape`           | 批处理形状（独立同分布的维度）                               |
| `event_shape`           | 事件形状（每个样本的维度）                                   |

---

## 离散分布

### 3.1 `Categorical`

**描述**：类别分布，随机变量取值为 `0, 1, ..., K-1`，概率由 `probs` 或 `logits` 指定。

**参数**：
- `probs` (Tensor) – 概率值，必须非负且和为 1
- `logits` (Tensor) – 未归一化的分数，内部通过 softmax 转换

**示例**：
```python
from torch.distributions import Categorical

# 使用概率
probs = torch.tensor([0.1, 0.2, 0.7])
dist = Categorical(probs=probs)

# 采样
sample = dist.sample()           # tensor(2)
samples = dist.sample_n(5)       # tensor([2, 1, 2, 0, 2])

# 对数概率
logp = dist.log_prob(torch.tensor(2))   # tensor(-0.3567)

# 批量分布
batch_probs = torch.tensor([[0.8, 0.1, 0.1],
                            [0.1, 0.8, 0.1]])
batch_dist = Categorical(probs=batch_probs)
batch_sample = batch_dist.sample()       # tensor([0, 1])
```

### 3.2 `Bernoulli`

**描述**：伯努利分布，单次二元试验（0/1）。  
**参数**：`probs` (成功概率) 或 `logits`。

```python
from torch.distributions import Bernoulli

dist = Bernoulli(probs=0.7)
samples = dist.sample((10,))   # 10个0/1样本
logp = dist.log_prob(torch.tensor(1))  # ln(0.7)
```

### 3.3 `Binomial`

**描述**：二项分布，`n` 次独立伯努利试验中的成功次数。  
**参数**：`total_count` (试验次数), `probs` (单次成功概率)。

```python
from torch.distributions import Binomial

dist = Binomial(total_count=10, probs=0.5)
samples = dist.sample((5,))   # 5个样本，每个介于0~10之间
```

### 3.4 `Poisson`

**描述**：泊松分布，模拟单位时间内随机事件的发生次数。  
**参数**：`rate` (λ，事件发生率)。

```python
from torch.distributions import Poisson

dist = Poisson(rate=3.0)
samples = dist.sample((100,))
```

### 3.5 `OneHotCategorical` 与 `RelaxedOneHotCategorical`

- **`OneHotCategorical`**：与 `Categorical` 相同，但采样结果为 one-hot 向量。
- **`RelaxedOneHotCategorical`**：基于 Gumbel-Softmax，输出为连续近似的 one-hot 向量（可微分）。

```python
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical

# One-hot 版本
dist = OneHotCategorical(probs=torch.tensor([0.2, 0.5, 0.3]))
sample = dist.sample()   # tensor([0., 1., 0.])

# 可微分近似（需要温度参数）
relaxed = RelaxedOneHotCategorical(temperature=0.5, probs=torch.tensor([0.2, 0.5, 0.3]))
soft_sample = relaxed.rsample()   # 例如 [0.1, 0.8, 0.1]，梯度可传播
```

---

## 连续分布

### 4.1 `Normal`

**描述**：正态分布（高斯分布）。  
**参数**：`loc` (均值 μ), `scale` (标准差 σ > 0)。

```python
from torch.distributions import Normal

dist = Normal(loc=0.0, scale=1.0)
samples = dist.sample((1000,))
log_probs = dist.log_prob(samples)
entropy = dist.entropy()   # 0.5 * ln(2πeσ^2)
```

### 4.2 `Uniform`

**描述**：均匀分布。  
**参数**：`low` (下界), `high` (上界)。

```python
from torch.distributions import Uniform

dist = Uniform(low=0.0, high=1.0)
sample = dist.sample()
```

### 4.3 `Beta` 与 `Gamma`

- **`Beta`**：定义在 [0,1] 区间，常用作伯努利/二项分布的共轭先验。参数 `concentration1` (α), `concentration0` (β)。
- **`Gamma`**：正连续分布，参数 `concentration` (形状 k), `rate` (逆尺度 β)。

```python
from torch.distributions import Beta, Gamma

beta = Beta(concentration1=2.0, concentration0=5.0)
gamma = Gamma(concentration=1.0, rate=2.0)
```

### 4.4 `Exponential` 与 `Laplace`

- **`Exponential`**：指数分布，参数 `rate` (λ)。
- **`Laplace`**：拉普拉斯分布（双指数），参数 `loc`, `scale`。

```python
from torch.distributions import Exponential, Laplace

exp_dist = Exponential(rate=1.0)
lap_dist = Laplace(loc=0.0, scale=1.0)
```

### 4.5 `Chi2` 与 `StudentT`

- **`Chi2`**：卡方分布，参数 `df` (自由度)。
- **`StudentT`**：t 分布，参数 `df`。

```python
from torch.distributions import Chi2, StudentT

chi2 = Chi2(df=3)
t = StudentT(df=2.5)
```

---

## 多元分布

### 5.1 `MultivariateNormal`

**描述**：多元正态分布。  
**参数**：`loc` (均值向量), `covariance_matrix` 或 `precision_matrix` 或 `scale_tril`。

```python
from torch.distributions import MultivariateNormal

mean = torch.zeros(2)
cov = torch.eye(2)
mvn = MultivariateNormal(mean, cov)
sample = mvn.sample()          # shape (2,)
samples = mvn.sample((5,))     # shape (5, 2)
```

### 5.2 `Dirichlet`

**描述**：狄利克雷分布，样本是位于单纯形上的正向量（各分量和为 1）。常用于多项分布的共轭先验。  
**参数**：`concentration` (浓度参数，α 向量)。

```python
from torch.distributions import Dirichlet

conc = torch.tensor([1.0, 1.0, 1.0])
dirichlet = Dirichlet(conc)
sample = dirichlet.sample()   # 如 [0.2, 0.3, 0.5]
```

---

## 变换与工具

### 6.1 `TransformedDistribution`

通过对基础分布应用可逆变换构造新分布。例如：对数正态分布可由正态分布经 `ExpTransform` 得到。

```python
from torch.distributions import TransformedDistribution, Normal, ExpTransform

base = Normal(0, 1)
transform = ExpTransform()
log_normal = TransformedDistribution(base, [transform])
sample = log_normal.sample()   # 恒为正
```

### 6.2 `kl_divergence` 与 `constraints`

- **`kl_divergence(p, q)`**：计算两个相同类型分布之间的 KL 散度。
- **`constraints`**：定义参数有效域，如 `constraints.positive`、`constraints.unit_interval`。

```python
from torch.distributions import kl_divergence, Normal, constraints

p = Normal(0, 1)
q = Normal(1, 1)
kl = kl_divergence(p, q)   # tensor(0.5)

# 检查约束
constraints.positive.check(torch.tensor(-1.0))   # False
```

---

## 完整示例：从离散采样到梯度传播

```python
import torch
from torch.distributions import Categorical, Normal

# 1. 类别分布采样（无梯度）
logits = torch.tensor([1.0, 2.0, 3.0])
dist_cat = Categorical(logits=logits)
action = dist_cat.sample()
print(f"Sampled action: {action}")

# 2. 使用 log_prob 计算策略梯度
log_prob = dist_cat.log_prob(action)
loss = -log_prob  # 假设 reward=1
loss.backward()   # 梯度会流向 logits

# 3. 可微分采样（重参数化）
normal = Normal(loc=torch.tensor(0.0, requires_grad=True), scale=1.0)
sample = normal.rsample()   # 可微分
loss = sample.pow(2).sum()
loss.backward()             # 梯度流向 loc

# 4. 批量分布
batch_probs = torch.softmax(torch.randn(3, 5), dim=-1)
batch_dist = Categorical(probs=batch_probs)
batch_actions = batch_dist.sample()   # shape (3,)
batch_log_probs = batch_dist.log_prob(batch_actions)
```

---

## 常见问题与最佳实践

### Q1：何时使用 `logits` 而非 `probs`？
**A**：使用 `logits` 数值更稳定（避免极小的概率值下溢），且梯度传播更友好。从神经网络输出时直接传入 `logits` 即可。

### Q2：`sample()` 与 `rsample()` 的区别？
**A**：`rsample()` 仅适用于连续分布（或重参数化的分布），生成的样本携带梯度；`sample()` 则阻断梯度。在 VAE、强化学习的重参数化技巧中必须使用 `rsample()`。

### Q3：如何计算分布之间的 KL 散度？
**A**：使用 `kl_divergence(p, q)`，要求 `p` 和 `q` 属于同一分布族。许多分布（如 `Normal`）已实现解析解。

### Q4：批量分布与事件形状如何理解？
**A**：`batch_shape` 表示独立同分布的分布数量，`event_shape` 表示每个样本的结构。例如 `MultivariateNormal` 的事件形状是向量长度，`Categorical` 的事件形状是空（标量）。

### Q5：可以自定义分布吗？
**A**：可以，继承 `torch.distributions.Distribution` 并实现 `sample`、`log_prob` 等方法，也可使用 `TransformedDistribution` 组合现有分布。

---

## 总结

`torch.distributions` 提供了一套完整、统一的概率分布接口，覆盖了从离散到连续、从单变量到多变量、从基础到高级变换的各种需求。熟练掌握这些分布能够极大地简化概率模型、强化学习算法以及生成模型的实现，并充分利用 PyTorch 的自动微分能力。

官方文档：[PyTorch Distributions](https://pytorch.org/docs/stable/distributions.html)