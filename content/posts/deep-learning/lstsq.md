---
title: "线性回归的最小二乘闭式解的推导"
date: 2026-04-13T10:00:00+08:00
draft: false
authors: [Steven]
description: "线性回归的最小二乘闭式解的推导"
summary: "线性回归的最小二乘闭式解的推导"
tags: [Deep Learning]
categories: [Deep Learning]
series: [Deep Learning系列]
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

推导线性回归中最小二乘解的闭式（正规方程） \(\theta = (X^T X)^{-1} X^T y\)。

---

## 1. 问题设定

假设我们有 \(m\) 个训练样本，每个样本有 \(n\) 个特征（加上截距项后为 \(n+1\) 维，但为了符号简洁，我们直接用 \(X\) 表示包含截距的设计矩阵）。  
线性模型：

\[
y = X\theta + \varepsilon
\]

其中：
- \(y\) 是 \(m \times 1\) 的观测值向量
- \(X\) 是 \(m \times n\) 的设计矩阵（每一行是一个样本，每一列是一个特征）
- \(\theta\) 是 \(n \times 1\) 的待求参数向量
- \(\varepsilon\) 是误差项

目标：找到 \(\theta\) 使**残差平方和**最小：

\[
J(\theta) = \|y - X\theta\|^2 = (y - X\theta)^T (y - X\theta)
\]

---

## 2. 展开损失函数

\[
J(\theta) = y^T y - y^T X\theta - (X\theta)^T y + (X\theta)^T (X\theta)
\]

因为 \(y^T X\theta\) 是标量，有 \(y^T X\theta = (X\theta)^T y\)，所以：

\[
J(\theta) = y^T y - 2\theta^T X^T y + \theta^T X^T X \theta
\]

---

## 3. 对 \(\theta\) 求梯度（向量导数）

使用以下向量求导公式（假设对称矩阵）：
- \(\frac{\partial (a^T \theta)}{\partial \theta} = a\)
- \(\frac{\partial (\theta^T A \theta)}{\partial \theta} = (A + A^T)\theta\)，当 \(A\) 对称时为 \(2A\theta\)

这里 \(X^T X\) 是对称半正定矩阵，因此：

\[
\nabla_{\theta} J(\theta) = -2 X^T y + 2 X^T X \theta
\]

---

## 4. 令梯度为零

\[
-2 X^T y + 2 X^T X \theta = 0
\]

除以 2：

\[
X^T X \theta = X^T y
\]

这就是**正规方程**。

---

## 5. 解出 \(\theta\)

如果 \(X^T X\) 可逆（即特征之间线性无关，矩阵满秩），两边左乘 \((X^T X)^{-1}\)：

\[
\theta = (X^T X)^{-1} X^T y
\]

---

## 6. 几何解释（可选）

最小二乘解可以理解为：\(X\theta\) 是 \(y\) 在 \(X\) 的列空间上的正交投影。  
投影矩阵为 \(P = X (X^T X)^{-1} X^T\)，则投影向量 \(X\theta = P y\)。  
左乘 \(X^T\) 得到 \(X^T X \theta = X^T y\)，与代数推导一致。

---

## 7. 需要注意的地方

- **可逆性要求**：\(X^T X\) 可逆等价于 \(X\) 列满秩（各特征线性无关）。若不可逆，解不唯一，通常使用伪逆 \(\theta = (X^T X)^+ X^T y\)，或用岭回归（L2 正则化）改为 \((X^T X + \lambda I)^{-1} X^T y\)。
- **数值稳定性**：实际应用中常用 QR 分解或 SVD 来求解正规方程，而不是直接求逆。

---

## 总结

正规方程的推导就是**最小化平方误差**的解析求解过程：  
1. 写出损失函数 \(J(\theta) = \|y - X\theta\|^2\)  
2. 展开并对 \(\theta\) 求导  
3. 令导数为零得到 \(X^T X \theta = X^T y\)  
4. 解出 \(\theta = (X^T X)^{-1} X^T y\)（当逆存在时）
