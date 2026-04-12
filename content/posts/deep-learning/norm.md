---
title: "深度学习中的常见归一化方法"
date: 2026-04-01T10:00:00+08:00
draft: false
authors: [Steven]
description: "深度学习中的常见归一化方法"
summary: "深度学习中的常见归一化方法"
tags: [Deep Learning, Normalization]
categories: [Deep Learning]
series: [Deep Learning系列]
weight: 1
series_weight: 1

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

# 深度学习归一化方法全解析

归一化层是深度神经网络中不可或缺的组件，其核心作用是通过对网络中间层的激活值（或权重）进行统计标准化，以**稳定训练过程**、**加速收敛**、**提升泛化能力**。不同的归一化方法在统计维度、依赖关系、实现方式上有所差异，从而适应不同的网络结构和任务需求。

本文系统梳理了深度学习中常见的归一化方法，涵盖从基础到前沿的多种技术，并详细介绍其原理、公式、特点、适用场景及PyTorch实现示例。

---

## 一、基础归一化（按维度划分）

这类方法直接对激活值进行标准化，依据所选维度的不同分为四类。

### 1. Batch Normalization (BN)
**提出**：Ioffe & Szegedy, 2015

**核心思想**：沿批次维度与空间维度对每个通道独立标准化，使每一层输入分布稳定，从而允许更高的学习率和更灵活的初始化。

**公式**：
对于输入张量 \(x \in \mathbb{R}^{N \times C \times H \times W}\)：
\[
\mu_c = \frac{1}{NHW}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{nchw}
\]
\[
\sigma_c^2 = \frac{1}{NHW}\sum_{n=1}^{N}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{nchw} - \mu_c)^2
\]
\[
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
\]
\[
y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
\]

**训练与推理**：
- 训练时使用当前 mini-batch 的统计量，同时通过指数移动平均更新全局统计量（running_mean, running_var）。
- 推理时使用全局统计量。

**特点**：
- ✅ 显著加速收敛，允许更大学习率，对初始化不敏感。
- ❌ 对批次大小敏感，小批量时统计不稳定；不适用于 RNN 或变长序列。

**适用场景**：CNN 视觉任务（分类、检测、分割），通常放在卷积层之后、激活函数之前。

**PyTorch 示例**：
```python
bn = nn.BatchNorm2d(num_features=64)
```

---

### 2. Layer Normalization (LN)
**提出**：Ba et al., 2016

**核心思想**：对每个样本独立，在所有通道和空间维度上标准化，消除样本间依赖，适合序列模型。

**公式**：
\[
\mu_n = \frac{1}{CHW}\sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{nchw}
\]
\[
\sigma_n^2 = \frac{1}{CHW}\sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{nchw} - \mu_n)^2
\]
\[
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_n}{\sqrt{\sigma_n^2 + \epsilon}}
\]
\[
y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
\]

**特点**：
- ✅ 不依赖批次，可处理变长序列和小批量。
- ❌ 在 CNN 中表现通常不如 BN，因为卷积权重共享可能与之不匹配。

**适用场景**：自然语言处理（Transformer、RNN）、强化学习、小批量训练。

**PyTorch 示例**：
```python
# 对 (C, H, W) 进行归一化
ln = nn.LayerNorm(normalized_shape=[64, 32, 32])
# 或对最后一维（常见于 NLP）
ln = nn.LayerNorm(normalized_shape=512)
```

---

### 3. Instance Normalization (IN)
**提出**：Ulyanov et al., 2016

**核心思想**：对每个样本的每个通道独立，仅在空间维度上标准化，去除样本间和通道间的统计关联，常用于风格迁移。

**公式**：
\[
\mu_{nc} = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{nchw}
\]
\[
\sigma_{nc}^2 = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{nchw} - \mu_{nc})^2
\]
\[
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_{nc}}{\sqrt{\sigma_{nc}^2 + \epsilon}}
\]
\[
y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
\]

**特点**：
- ✅ 解耦内容与风格，保留单个样本内通道间的相对差异。
- ❌ 丢弃了通道间相关性，不适合分类等任务。

**适用场景**：风格迁移、图像生成（GAN）、单图像超分辨率。

**PyTorch 示例**：
```python
in_norm = nn.InstanceNorm2d(num_features=64)
```

---

### 4. Group Normalization (GN)
**提出**：Wu & He, 2018

**核心思想**：将通道分为若干组，在组内通道与空间维度上标准化，介于 LN 和 IN 之间，不受批次大小限制。

**公式**（设组数为 \(G\)，每组通道数 \(C_g = C/G\)）：
\[
\mu_{ng} = \frac{1}{C_g HW}\sum_{c \in \text{group}_g}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{nchw}
\]
\[
\sigma_{ng}^2 = \frac{1}{C_g HW}\sum_{c \in \text{group}_g}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{nchw} - \mu_{ng})^2
\]
\[
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_{ng}}{\sqrt{\sigma_{ng}^2 + \epsilon}}
\]
\[
y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
\]

**特点**：
- ✅ 不依赖批次，性能与 BN 接近，适合小批量训练。
- ❌ 需要调整组数超参数（常用 G=32 或通道数/16）。

**适用场景**：小批量训练、内存受限场景、视频理解、检测分割。

**PyTorch 示例**：
```python
gn = nn.GroupNorm(num_groups=32, num_channels=64)
```

---

## 二、改进与变体（解决 BN 的缺点）

这类方法旨在克服 BN 对批次大小的依赖或将其推广到更多场景。

### 5. Batch Renormalization
**提出**：Ioffe, 2017

**核心思想**：在 BN 基础上引入两个额外参数 \(r\) 和 \(d\)，对小批次统计量进行约束，使模型在小批量时也能保持 BN 的性质。

**公式**：
\[
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \cdot r + d
\]
其中 \(r = \text{clip}(\frac{\sigma_{\text{running}}}{\sigma_{\mathcal{B}}}, 1/r_{\max}, r_{\max})\)，\(d = \text{clip}(\frac{\mu_{\text{running}} - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}}, -d_{\max}, d_{\max})\)，\(r_{\max}, d_{\max}\) 是预设阈值。

**特点**：
- ✅ 允许小批量训练（如 batch size=1），同时保持 BN 的加速能力。
- ❌ 增加额外超参数，需微调。

**适用场景**：大批量受限的视觉任务。

**PyTorch 示例**（需自定义或使用第三方库，如 `torch.nn.BatchNorm2d` 未原生支持）：
```python
# 可使用 pip install inplace_abn 中的 InPlaceABN 实现
```

---

### 6. Moving Average Batch Normalization (MABN)
**提出**：Yan et al., 2019

**核心思想**：在 BN 中引入移动平均统计量作为全局估计，并在训练中与当前 batch 统计量融合，增强小批量稳定性。

**特点**：
- ✅ 适合在线学习和持续学习场景，统计量平滑变化。
- ❌ 实现相对复杂。

**适用场景**：在线学习、小批量连续训练。

---

### 7. Cross-Iteration Batch Normalization (CBN)
**提出**：Yao et al., 2020

**核心思想**：将多个迭代的统计量融合，通过历史信息补偿小批量统计的方差，使 BN 在大批量受限时依然有效。

**特点**：
- ✅ 显著提高小批量下的性能，可与 GN 媲美。
- ❌ 需要存储历史统计量，增加显存和计算。

**适用场景**：大批量受限的视觉任务，如目标检测、语义分割。

---

### 8. Filter Response Normalization (FRN)
**提出**：Singh & Krishnan, 2019

**核心思想**：摒弃均值，仅用方差进行归一化，并结合一个可学习的偏移量，避免对 batch 的依赖。

**公式**：
\[
\nu_{nc}^2 = \frac{1}{HW}\sum_{h,w} (x_{nchw})^2
\]
\[
\hat{x}_{nchw} = \frac{x_{nchw}}{\sqrt{\nu_{nc}^2 + \epsilon}}
\]
\[
y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
\]
再经过一个门控单元（TLU）处理。

**特点**：
- ✅ 不依赖 batch，适合小批量；计算简单。
- ❌ 对某些任务可能不如 BN 稳定。

**适用场景**：图像分类、目标检测（小批量）。

**PyTorch 示例**（可使用第三方实现）：
```python
# 如 https://github.com/facebookresearch/ResNeXt/blob/master/models/frn.py
```

---

### 9. Temporal Batch Normalization (TBN)
**提出**：Vaswani et al., 2016（用于视频）

**核心思想**：将 BN 扩展到 3D 卷积中，沿时间维度（帧）和空间维度标准化，同时保留 batch 维度。

**特点**：
- ✅ 有效稳定视频模型训练。
- ❌ 需处理变长时间序列，需小心设计统计范围。

**适用场景**：视频分类、动作识别。

---

## 三、权重归一化（不归一化激活值）

这类方法直接对权重进行重参数化或约束，不依赖输入统计。

### 10. Weight Normalization (WN)
**提出**：Salimans & Kingma, 2016

**核心思想**：将权重向量分解为方向 \(v\) 和幅度 \(g\)：\(w = g \cdot \frac{v}{\|v\|}\)，训练时优化 \(v\) 和 \(g\)。

**公式**：
\[
w = \frac{g}{\|v\|} v
\]
梯度通过分解计算，解耦了权重大小与方向。

**特点**：
- ✅ 不依赖批次，适合 RNN、强化学习；加速收敛。
- ❌ 与 BN 相比在某些任务上性能稍逊。

**适用场景**：RNN、小批量训练、生成模型。

**PyTorch 示例**：
```python
import torch.nn.utils as utils
# 对线性层使用权重归一化
linear = nn.Linear(10, 5)
linear = utils.weight_norm(linear)
```

---

### 11. Spectral Normalization (SN)
**提出**：Miyato et al., 2018

**核心思想**：约束权重的谱范数（最大奇异值），使网络满足 Lipschitz 条件，常用在 GAN 判别器上。

**公式**：
\[
W_{\text{SN}} = \frac{W}{\sigma(W)}
\]
其中 \(\sigma(W)\) 是 \(W\) 的最大奇异值，通常通过幂迭代法近似。

**特点**：
- ✅ 稳定 GAN 训练，防止梯度爆炸。
- ❌ 计算开销稍大（需迭代估计奇异值）。

**适用场景**：GAN 判别器、需要 Lipschitz 约束的模型。

**PyTorch 示例**：
```python
import torch.nn.utils as utils
# 对卷积层应用谱归一化
conv = nn.Conv2d(3, 64, 3)
conv = utils.spectral_norm(conv)
```

---

### 12. Weight Standardization (WS)
**提出**：Qiao et al., 2019

**核心思想**：对卷积核权重进行标准化，使其均值为 0、方差为 1（沿输出通道维度），从而平滑损失景观。

**公式**：
\[
\hat{W}_{i,j} = \frac{W_{i,j} - \mu_{i}}{\sqrt{\sigma_i^2 + \epsilon}}
\]
其中 \(\mu_i = \frac{1}{K}\sum_{j} W_{i,j}\)，\(\sigma_i^2 = \frac{1}{K}\sum_{j} (W_{i,j} - \mu_i)^2\)，\(K\) 是卷积核参数量。

**特点**：
- ✅ 与 BN 配合可进一步提升性能；可单独使用于小批量。
- ❌ 单独使用时收敛速度可能不如 BN。

**适用场景**：大规模图像分类、目标检测，尤其是与 BN 结合。

**PyTorch 示例**（需自定义）：
```python
def weight_standardization(weight, eps=1e-5):
    mean = weight.mean(dim=[1,2,3], keepdim=True)
    var = weight.var(dim=[1,2,3], keepdim=True, unbiased=False)
    return (weight - mean) / torch.sqrt(var + eps)
```

---

## 四、自适应与条件归一化（融入外部信息）

这类方法将额外信息（如图像风格、语义布局、类别）注入归一化参数。

### 13. Adaptive Instance Normalization (AdaIN)
**提出**：Huang & Belongie, 2017

**核心思想**：用风格图像的均值/方差替换 Instance Normalization 中的统计量，实现风格迁移。

**公式**：
\[
\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)
\]
其中 \(x\) 是内容特征，\(y\) 是风格特征，\(\mu, \sigma\) 在每个样本每个通道的空间上计算。

**特点**：
- ✅ 实现任意风格迁移，风格与内容解耦。
- ❌ 要求风格特征需预先提取（通常来自预训练网络）。

**适用场景**：风格迁移、图像生成。

**PyTorch 示例**：
```python
def adain(content, style):
    mean_c = content.mean(dim=[2,3], keepdim=True)
    std_c = content.std(dim=[2,3], keepdim=True)
    mean_s = style.mean(dim=[2,3], keepdim=True)
    std_s = style.std(dim=[2,3], keepdim=True)
    return std_s * (content - mean_c) / (std_c + 1e-5) + mean_s
```

---

### 14. Spatially-Adaptive Denormalization (SPADE)
**提出**：Park et al., 2019

**核心思想**：利用语义分割图（mask）学习空间可变的 \(\gamma, \beta\)，实现对生成图像的精细控制。

**公式**：
\[
y = \gamma_{\text{spatial}}(mask) \cdot \hat{x} + \beta_{\text{spatial}}(mask)
\]
其中 \(\gamma_{\text{spatial}}, \beta_{\text{spatial}}\) 通过卷积层从 mask 学习。

**特点**：
- ✅ 能根据语义布局生成高质量图像，保留结构。
- ❌ 需要输入 mask，增加了计算量。

**适用场景**：图像合成、语义图像生成。

**PyTorch 示例**（使用官方库 `SPADE` 或 `pytorch-CycleGAN-and-pix2pix`）：
```python
# 可从 https://github.com/NVlabs/SPADE 获取实现
```

---

### 15. Conditional Batch Normalization (CBN)
**提出**：De Vries et al., 2017

**核心思想**：根据类别或域条件学习不同的 \(\gamma, \beta\)，使 BN 具备条件生成能力。

**公式**：
\[
y = \gamma_c \odot \hat{x} + \beta_c
\]
其中 \(\gamma_c, \beta_c\) 由条件向量（如类别嵌入）通过 MLP 生成。

**特点**：
- ✅ 适合多模态生成、域自适应。
- ❌ 需提供条件信息。

**适用场景**：条件图像生成、少样本学习。

**PyTorch 示例**：
```python
class ConditionalBN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)
    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(y).unsqueeze(-1).unsqueeze(-1)
        return out * gamma + beta
```

---

### 16. Modulated Group Normalization (ModGN)
**提出**：Lee et al., 2021（常见于 StyleGAN2）

**核心思想**：在 GroupNorm 基础上，根据隐码（style code）学习调制参数，实现对生成图像的精细控制。

**特点**：
- ✅ 可生成高质量、多样化的图像。
- ❌ 需要额外的 style 输入。

**适用场景**：生成对抗网络（StyleGAN 系列）。

---

## 五、动态与混合归一化（自动选择/融合）

这类方法通过学习或进化方式自动选择或组合不同归一化操作。

### 17. Switchable Normalization (SN)
**提出**：Luo et al., 2018

**核心思想**：为每个归一化层学习 BN、LN、IN 的权重，动态融合三种统计量。

**公式**：
\[
\hat{x} = w_1 \hat{x}_{\text{BN}} + w_2 \hat{x}_{\text{LN}} + w_3 \hat{x}_{\text{IN}}
\]
权重 \(w_i\) 通过 softmax 学习得到，且可随通道、位置变化。

**特点**：
- ✅ 自适应不同任务和批次大小，减少调参。
- ❌ 增加计算开销和参数量。

**适用场景**：统一框架下的各类任务，如分类、检测、分割。

**PyTorch 示例**（需第三方实现，如 `switchable_norm` 库）。

---

### 18. SkipNorm
**提出**：Lee et al., 2020

**核心思想**：在残差连接中，选择性地跳过归一化层，以改善梯度流。

**特点**：
- ✅ 提升训练稳定性，尤其深层网络。
- ❌ 引入额外选择机制。

**适用场景**：深层残差网络。

---

### 19. EvoNorm
**提出**：Liu et al., 2021

**核心思想**：通过进化搜索发现新的归一化-激活函数组合，将 BN 与激活函数融合为单一操作。

**公式**（EvoNorm-S0 为例）：
\[
y = \frac{x}{\sqrt{\text{Var}(x) + \epsilon}} \cdot \text{sigmoid}(s \cdot x)
\]
其中 \(s\) 为可学习参数。

**特点**：
- ✅ 性能优于 BN+ReLU，且可替换后直接使用。
- ❌ 搜索计算量大，实现复杂。

**适用场景**：大规模图像分类、检测。

**PyTorch 示例**（使用官方实现 `EvoNorm` 库）。

---

## 六、Transformer 与序列模型专用

针对 Transformer 类模型设计的归一化方法，往往更关注序列长度和特征维度。

### 20. RMSNorm
**提出**：Zhang & Sennrich, 2019

**核心思想**：仅使用均方根进行缩放，省去均值平移，计算更高效。

**公式**：
\[
\hat{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma
\]

**特点**：
- ✅ 与 LayerNorm 性能相当，但计算量更小。
- ❌ 缺少可学习的偏置。

**适用场景**：Transformer 模型（如 T5、LLaMA 中使用）。

**PyTorch 示例**：
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma
```

---

### 21. ScaleNorm
**提出**：Nguyen & Salazar, 2018

**核心思想**：仅用 L2 范数进行缩放，类似 RMSNorm，但使用 L2 范数而非 RMS。

**公式**：
\[
\hat{x} = \frac{x}{\|x\|_2} \cdot g
\]
其中 \(g\) 是可学习标量。

**特点**：
- ✅ 简单高效，适合极深 Transformer。
- ❌ 表达能力弱于 LayerNorm。

**适用场景**：深层 Transformer 预训练。

---

### 22. DeepNorm
**提出**：Wang et al., 2022

**核心思想**：在 Transformer 的残差连接后引入缩放因子，并结合 LayerNorm，使模型能稳定训练上千层。

**公式**：
\[
x_{l+1} = \alpha \cdot \text{LN}(x_l + \text{Attn}(x_l)) + \beta \cdot \text{LN}(x_l + \text{FFN}(x_l))
\]
其中 \(\alpha, \beta\) 为可学习或固定参数。

**特点**：
- ✅ 支持超深 Transformer（如 1000 层以上）。
- ❌ 需调整缩放参数。

**适用场景**：超深 Transformer 预训练。

---

## 七、图神经网络 (GNN) 专用

这类归一化方法针对图结构数据设计，处理节点、边的不规则性。

### 23. GraphNorm
**提出**：Cai et al., 2020

**核心思想**：在图级别上标准化节点特征，同时保留图间差异。

**公式**：
\[
\hat{x}_{v} = \frac{x_v - \mu_{\mathcal{G}}}{\sqrt{\sigma_{\mathcal{G}}^2 + \epsilon}} \cdot \gamma + \beta
\]
其中 \(\mu_{\mathcal{G}}, \sigma_{\mathcal{G}}^2\) 是对全图所有节点特征计算的统计量。

**特点**：
- ✅ 比 BN 更适应图数据，提升 GNN 性能。
- ❌ 对单图训练不友好。

**适用场景**：图分类、节点分类（多图）。

**PyTorch 示例**（可使用 `torch_geometric.nn.norm.GraphNorm`）。

---

### 24. PairNorm
**提出**：Zhao & Akoglu, 2019

**核心思想**：控制节点表示的全局方差，防止深层 GNN 的过平滑现象。

**公式**：
\[
\hat{x}_v = s \cdot \frac{x_v - \frac{1}{n}\sum_{u} x_u}{\sqrt{\frac{1}{n}\sum_{u} \|x_u - \bar{x}\|^2}}
\]
其中 \(s\) 是可学习缩放参数。

**特点**：
- ✅ 有效缓解过平滑，支持更深的 GNN。
- ❌ 可能牺牲局部结构信息。

**适用场景**：深层图神经网络。

**PyTorch 示例**（使用 `torch_geometric.nn.norm.PairNorm`）。

---

## 八、其他新兴方法

### 25. Correlation Normalization (CorrNorm)
**提出**：Liu et al., 2021

**核心思想**：对特征之间的相关性进行标准化，解耦通道间的冗余信息。

**公式**：
\[
\hat{x} = \frac{x - \mu}{\sigma} \quad \text{(传统)}
\]
CorrNorm 进一步将特征白化，使协方差矩阵接近单位阵。

**特点**：
- ✅ 提升特征多样性，改善模型鲁棒性。
- ❌ 计算代价高（需计算协方差矩阵）。

**适用场景**：对抗训练、域泛化。

---

### 26. Positional Normalization (PN)
**提出**：Li et al., 2020

**核心思想**：沿空间位置（H, W）进行归一化，类似 LayerNorm 但仅作用于空间维度。

**公式**：
\[
\mu_{nhw} = \frac{1}{C}\sum_{c} x_{nchw}, \quad \sigma_{nhw}^2 = \frac{1}{C}\sum_{c} (x_{nchw} - \mu_{nhw})^2
\]
\[
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_{nhw}}{\sqrt{\sigma_{nhw}^2 + \epsilon}}
\]

**特点**：
- ✅ 适合 3D 数据（视频、体数据），保留位置差异。
- ❌ 在 2D 图像上可能不如 BN。

**适用场景**：视频分类、3D 分割。

---

### 27. T-Norm
**提出**：Khrulkov et al., 2023

**核心思想**：引入温度系数 \(T\) 控制归一化强度，使模型在训练初期柔和地标准化，后期逐渐增强。

**公式**：
\[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \frac{1}{T}
\]
其中 \(T\) 可随训练步数衰减。

**特点**：
- ✅ 提升训练稳定性和最终性能。
- ❌ 需要调整温度调度策略。

**适用场景**：大规模训练（如 CLIP、BERT 预训练）。

---

## 总结

归一化方法的核心在于选择在哪些维度上聚合统计量，并根据任务需求引入自适应、条件化或动态融合机制。从最早的 BatchNorm 到如今针对特定架构（Transformer、GNN）和任务（风格迁移、生成对抗网络）的专用归一化，该领域已发展出丰富的技术栈。在实际应用中，应根据以下因素选择合适的方法：

- **网络结构**：CNN 优先 BN/GN，Transformer 优先 LN/RMSNorm。
- **训练批量**：大批量可用 BN，小批量可选用 GN、LN、WeightNorm。
- **任务特性**：风格迁移用 IN/AdaIN，生成对抗网络用 SN，图网络用 GraphNorm。
- **计算资源**：若显存受限，避免使用需要额外存储历史统计量的方法（如 CBN）。

掌握这些方法的原理与适用场景，有助于更高效地设计和调试深度神经网络。