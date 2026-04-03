---
title: "Kaiming（He）初始化：方差推导与 ReLU 网络"
date: 2026-04-03T00:00:00+08:00
draft: false
authors: [Steven]
description: "从方差传播角度推导 Kaiming / He 初始化中分子为 2 的原因，说明 ReLU 对方差的折半效应、与 Xavier 的差异，以及 PyTorch 中 kaiming_normal_ 的 std 公式。"
summary: "用前向方差分析解释为何 ReLU 网络宜用方差 2/fan_in 的权重初始化，并对比 Xavier、给出 PyTorch 中的对应实现。"
tags: ["PyTorch", "Deep Learning", "Kaiming", "Xavier"]
categories: ["PyTorch"]
series: ["PyTorch实践指南"]
weight: 6
series_weight: 6
featuredImage: ""
featuredImagePreview: ""
---

Kaiming 初始化（也称为 He 初始化）的核心目标是：**让输入信号在前向传播和反向传播时，其方差保持稳定**，避免随网络加深而指数级爆炸或消失。

对于使用 **ReLU** 激活函数的网络，Kaiming 初始化建议将权重方差设为 \(\frac{2}{\text{fan\_in}}\)（分子是 2，而不是 Xavier 初始化中的 1）。这个“2”正是为了补偿 ReLU 将大约一半的神经元输出置为 0 的特性。

---

## 1. 数学推导的关键步骤

假设第 \(l\) 层的权重 \(W_l\) 独立同分布，均值为 0，方差为 \(\sigma_l^2\)。该层输入为 \(x_{l-1}\)（来自上一层的 ReLU 输出），且 \(x_{l-1}\) 均值为 0，方差为 \(\text{Var}[x_{l-1}]\)。

前向传播：
\[
y_l = W_l x_{l-1}, \quad x_l = \text{ReLU}(y_l)
\]
我们希望 \(\text{Var}[x_l] = \text{Var}[x_{l-1}]\)。

---

### ① 先计算 \(\text{Var}[y_l]\)（ReLU 之前）

设 \(W_l\) 形状为 \(d_l \times d_{l-1}\)，其中 \(d_{l-1}\) 是 fan_in。对于第 \(j\) 个神经元：
\[
y_{l,j} = \sum_{k=1}^{d_{l-1}} W_{l,jk} x_{l-1,k}
\]
由于 \(W\) 与 \(x_{l-1}\) 独立且零均值，方差为：
\[
\text{Var}[y_{l,j}] = d_{l-1} \cdot \sigma_l^2 \cdot \text{Var}[x_{l-1,k}]
\]
假设各分量同分布，记 \(\text{Var}[x_{l-1}] = \sigma_{x,l-1}^2\)，则：
\[
\text{Var}[y_l] = d_{l-1} \sigma_l^2 \sigma_{x,l-1}^2
\]

---

### ② ReLU 对方差的影响

ReLU 函数：\(x_l = \max(0, y_l)\)。若 \(y_l\) 服从均值为 0 的对称分布（通常如此），则 ReLU 后恰好有一半的数值被置为零，另一半保留原值。此时：
\[
\text{Var}[x_l] = \frac{1}{2} \text{Var}[y_l]
\]
推导：对于零均值对称分布，\(E[y_l] = 0\)，则
\[
E[x_l^2] = E[\max(0, y_l)^2] = \frac{1}{2} E[y_l^2]
\]
因为一半的概率 y_l ≤ 0 贡献为 0，另一半 y_l > 0 贡献为 y_l^2。而 \(\text{Var}[x_l] = E[x_l^2] - (E[x_l])^2\)，且 \(E[x_l] = E[\max(0, y_l)] = \frac{1}{\sqrt{2\pi}}\sigma_{y_l}\) 不为零，但主要影响是缩放因子约 1/2。更精确的推导（He et al., 2015）得到：
\[
\text{Var}[x_l] = \frac{1}{2} \text{Var}[y_l]
\]

---

### ③ 令方差不变

要求 \(\text{Var}[x_l] = \sigma_{x,l-1}^2\)，代入：
\[
\frac{1}{2} \cdot d_{l-1} \sigma_l^2 \sigma_{x,l-1}^2 = \sigma_{x,l-1}^2
\]
消去 \(\sigma_{x,l-1}^2\)，得：
\[
\sigma_l^2 = \frac{2}{d_{l-1}}
\]
因此，权重的方差应该设为 \(\frac{2}{\text{fan\_in}}\)。如果使用反向传播推导（考虑梯度），会得到类似结果，分子依然是 2。

---

## 2. 对比 Xavier 初始化（分子为 1）

Xavier 初始化针对的是 **tanh** 或 **sigmoid** 等关于原点对称且输出范围有限的激活函数。在理想情况下（激活函数近似线性区），梯度流可以保持方差不变，推导结果：
\[
\sigma_l^2 = \frac{1}{\text{fan\_in}}
\]
这相当于假设激活函数不改变信号的方差（即线性激活）。而 ReLU 会主动砍掉一半的激活，导致信号方差减半，所以需要将输入权重的方差扩大一倍来补偿。

---

## 3. 直观理解

- **没有 ReLU**：信号经过一层，方差乘以 \(d_{l-1} \sigma_l^2\)。要维持不变，需要 \(\sigma_l^2 = 1/d_{l-1}\)（分子=1）。
- **有 ReLU**：信号经过 ReLU，方差损失一半，因此需要在前一步将方差放大一倍，即 \(\sigma_l^2 = 2/d_{l-1}\)。

---

## 4. 在 PyTorch 中的体现

PyTorch 的 `nn.init.kaiming_normal_` 默认 `nonlinearity='relu'`，此时会使用 `a=0`（负斜率），方差计算公式为：
\[
\text{std} = \sqrt{\frac{2}{(1 + a^2) \cdot \text{fan\_in}}}
\]
当 `a=0`（ReLU），std = \(\sqrt{2/\text{fan\_in}}\)，方差 = \(2/\text{fan\_in}\)。如果使用 `leaky_relu` 且 `a=0.01`，分母中的 \((1+a^2)\) 略大于 1，方差略小于 \(2/\text{fan\_in}\)，因为负轴也保留了一点梯度。

---

## 5. 总结

- **分子为 2** 正是因为 ReLU 在正半轴保留信号、负半轴置零，导致输出方差约为输入方差的一半。
- 为了保持信号幅度稳定，需要将权重的方差加倍补偿。
- 这一设计让 Kaiming 初始化在 ResNet 等深层 ReLU 网络中表现出色，成为事实标准。
