---
title: RMSNorm 与 “均值平移"
date:
draft: false
authors:
  - Steven
description:
summary:
tags: [RMSNorm, LayerNorm, 归一化, 知识点]
categories:
series:
weight: 0
series_weight: 0
hiddenFromHomePage: false
seriesNavigation: false
hiddenFromSearch: false
---


## 1. 定义
在归一化方法中，**均值平移** 指将激活值的统计中心调整到 0 的操作，即 `x_i ← x_i - μ`。

## 2. LayerNorm（包含均值平移）
对输入向量 $x \in \mathbb{R}^d$ 的归一化步骤：
1. **中心化（均值平移）**  
   $x_i ← x_i - μ$，其中 $μ = \frac{1}{d}\sum_{j=1}^d x_j$
2. **缩放**  
   $x_i ← x_i / σ$，其中 $σ = \sqrt{\frac{1}{d}\sum (x_j-μ)^2}$

> 完整 LayerNorm 还会加上可学习的 γ 和 β，但核心归一化是上述两步。

## 3. RMSNorm 如何“省去均值平移”
RMSNorm 公式：

$$ \hat{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma $$

- **无均值平移**：直接用原始 $x_i$ 除以 **均方根 (RMS)**
- RMS 定义：$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_j^2}$
- 效果：**只做缩放**，不显式将均值移至 0

## 4. 为什么可以省略均值平移？
| 原因 | 说明 |
|------|------|
| 残差连接提供隐含中心化 | Transformer 的残差结构抑制了激活值的持续偏移 |
| 收益有限 | 实验证明去掉均值平移后性能不降，稳定性甚至略升 |
| 降低计算开销 | 省去求和与除法，对大模型可累积可观加速 |

## 5. 对比总结

| 操作 | 物理意义 | RMSNorm 是否保留 |
|------|----------|:---:|
| 均值平移（去均值） | 将分布中心移至 0，消除整体偏移 | ❌ 省略 |
| 缩放（除以 RMS） | 控制能量尺度，防止数值爆炸/消失 | ✅ 保留 |

> **一句话概括**：RMSNorm 只关心输入的“能量”（RMS），不关心“位置”（均值），利用残差网络让均值自然稳定，同时用简单缩放维持数值范围。

## 6. 典型应用
- T5、LLaMA 等流行架构
- 需要降低计算量并保持训练稳定性的场景
```