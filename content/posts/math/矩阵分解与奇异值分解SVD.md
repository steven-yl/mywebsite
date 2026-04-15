---
title: "矩阵分解与奇异值分解（SVD）" 
date: 2026-04-14T10:00:00+08:00
draft: false
authors: [Steven]
description: "矩阵分解与奇异值分解（SVD）"
summary: "矩阵分解与奇异值分解（SVD）"
tags: [Deep Learning]
categories: [Deep Learning]
series: [Deep Learning系列]
weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
---

## 1. 特征值与特征向量（方阵）

### 1.1 定义

对于 **方阵** \( A \in \mathbb{R}^{n \times n} \)，若存在非零向量 \( v \in \mathbb{R}^n \) 和标量 \( \lambda \) 满足：

\[
A v = \lambda v
\]

则称 \( \lambda \) 为 \( A \) 的**特征值**，\( v \) 为对应的**特征向量**。

### 1.2 几何意义

- 特征向量方向在 \( A \) 作用下保持不变（或反向）。
- 特征值 \( \lambda \) 是该方向上的缩放倍数：\( |\lambda| > 1 \) 拉伸，\( |\lambda| < 1 \) 压缩，\( \lambda = 0 \) 降维。

### 1.3 计算（Python）

```python
import numpy as np

A = np.array([[2, 1],
              [1, 2]])
eigvals, eigvecs = np.linalg.eig(A)
print("特征值:", eigvals)
print("特征向量（列）:\n", eigvecs)
```

### 1.4 注意事项

- 特征值和特征向量**仅对方阵定义**。
- 非方阵没有特征值/特征向量。

---

## 2. 协方差矩阵与数据中心化（PCA 基础）

### 2.1 为什么需要中心化？

PCA（主成分分析）的目标是寻找数据方差最大的方向。若不减去均值，协方差会受到数据绝对位置的影响，无法正确反映内部波动。

**中心化公式**：
\[
X_{\text{centered}} = X - \frac{1}{n}\sum_{i=1}^n X_{i}
\]

### 2.2 协方差矩阵

对于中心化数据矩阵 \( X_{\text{centered}} \in \mathbb{R}^{n \times p} \)（\( n \) 个样本，\( p \) 个特征），协方差矩阵为：

\[
C = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}
\]

- \( C \) 是 \( p \times p \) 对称半正定矩阵。
- 对角元：各特征的方差。
- 非对角元：特征间的协方差。

### 2.3 PCA 与特征分解

- 协方差矩阵的特征向量 → 主成分方向。
- 协方差矩阵的特征值 → 沿主成分方向的方差。

---

## 3. 半正定矩阵

### 3.1 定义

实对称矩阵 \( A \) 称为**半正定**，若对所有非零向量 \( x \) 有：

\[
x^T A x \ge 0
\]

若严格 \( >0 \) 则称为**正定**。

### 3.2 等价条件

- 所有特征值 \( \lambda_i \ge 0 \)。
- 存在矩阵 \( B \) 使得 \( A = B^T B \)。
- 所有主子式（principal minors）非负（Sylvester 准则推广）。

### 3.3 常见例子

- 协方差矩阵。
- \( A^T A \) 或 \( A A^T \)（对任意实矩阵 \( A \)）。
- 核矩阵（Gram 矩阵）。
- 图拉普拉斯矩阵。

---

## 4. 奇异值与奇异值分解（SVD）

### 4.1 奇异值定义

对于任意实矩阵 \( A \in \mathbb{R}^{m \times n} \)（不必为方阵），其**奇异值**定义为：

\[
\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(A A^T)}, \quad i = 1, \dots, r
\]

其中 \( r = \mathrm{rank}(A) \)，且 \( \sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0 \)。奇异值总是非负实数。

### 4.2 奇异值分解（SVD）

任何矩阵 \( A \) 可分解为：

\[
A = U \Sigma V^T
\]

- \( U \in \mathbb{R}^{m \times m} \) 正交矩阵（左奇异向量）。
- \( V \in \mathbb{R}^{n \times n} \) 正交矩阵（右奇异向量）。
- \( \Sigma \in \mathbb{R}^{m \times n} \) 对角矩阵，对角元为奇异值 \( \sigma_i \)。

### 4.3 几何解释

将单位球（\( \mathbb{R}^n \) 中的球面）通过 \( A \) 映射到 \( \mathbb{R}^m \) 得到一个**超椭球**：

- 椭球的半轴长度 = 奇异值 \( \sigma_i \)。
- 半轴方向 = 左奇异向量 \( u_i \)。
- 原始球上与右奇异向量 \( v_i \) 重合的点映射到 \( \sigma_i u_i \)。

### 4.4 与特征值的关系

- 对对称半正定矩阵：奇异值 = 特征值。
- 对一般方阵：奇异值 ≠ |特征值|，但奇异值总是非负实数，而特征值可以是复数。
- 非方阵没有特征值，奇异值是其自然替代。

---

## 5. 左奇异向量与右奇异向量

### 5.1 定义与关系

- **右奇异向量** \( v_i \)：\( V \) 的列，是 \( A^T A \) 的特征向量（特征值 \( \sigma_i^2 \)）。
- **左奇异向量** \( u_i \)：\( U \) 的列，是 \( A A^T \) 的特征向量（特征值 \( \sigma_i^2 \)）。

基本关系：
\[
A v_i = \sigma_i u_i, \quad A^T u_i = \sigma_i v_i
\]

### 5.2 几何角色

- \( v_1, \dots, v_r \) 张成 \( A \) 的行空间，\( v_{r+1}, \dots, v_n \) 张成零空间。
- \( u_1, \dots, u_r \) 张成 \( A \) 的列空间，\( u_{r+1}, \dots, u_m \) 张成左零空间。

### 5.3 与特征向量的区别

| 概念               | 适用矩阵   | 是否正交 | 是否非负实数 |
| ------------------ | ---------- | -------- | ------------ |
| 特征向量           | 方阵       | 一般不是 | 否（可复）   |
| 左/右奇异向量      | 任意矩阵   | 是       | 不适用（向量）|
| 奇异值             | 任意矩阵   | -        | 是           |

---

## 6. SVD 的数值实现

### 6.1 算法概览（以 \( m \ge n \) 为例）

1. **双对角化**（Bidiagonalization）  
   使用 Householder 或 Givens 变换将 \( A \) 化为上双对角矩阵 \( B \)：
   \[
   U_1^T A V_1 = B = \begin{pmatrix}
   b_{11} & b_{12} & & \\
   & b_{22} & b_{23} & \\
   & & \ddots & \ddots \\
   & & & b_{nn}
   \end{pmatrix}
   \]

2. **双对角 SVD**  
   对 \( B \) 计算 SVD：\( B = U_2 \Sigma V_2^T \)。常用方法：
   - **隐式对称 QR 迭代**（Golub-Kahan）：隐式处理 \( B^T B \) 而不显式形成。
   - **分治法**（Divide-and-Conquer，LAPACK `dgesdd`）：更快，适合大矩阵。
   - **Jacobi 旋转**：高精度，较慢（LAPACK `dgesvj`）。

3. **合成**：
   \[
   U = U_1 U_2,\quad V = V_1 V_2,\quad A = U \Sigma V^T
   \]

### 6.2 为什么不用 \( A^T A \) 特征分解？

- 显式计算 \( A^T A \) 会**平方条件数**，导致数值精度损失。
- 直接对 \( A \) 或双对角矩阵操作更稳定。

### 6.3 Python 调用

```python
import numpy as np

A = np.random.randn(5, 3)
U, s, Vt = np.linalg.svd(A, full_matrices=False)   # 调用 LAPACK
print("奇异值:", s)
```

### 6.4 仅求奇异值（更快）

```python
s = np.linalg.svd(A, compute_uv=False)
```

---

## 7. 主要应用

| 应用领域               | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| 主成分分析（PCA）      | 中心化数据矩阵的右奇异向量 = 主成分方向，奇异值平方 = 方差 × (n-1) |
| 低秩近似与压缩         | 保留前 \( k \) 个最大奇异值及对应奇异向量，得到最优低秩逼近（Eckart-Young） |
| 矩阵伪逆与最小二乘     | \( A^+ = V \Sigma^+ U^T \)，用于求解超定或欠定方程组         |
| 条件数与数值稳定性分析 | \( \kappa(A) = \sigma_{\max} / \sigma_{\min} \)              |
| 推荐系统（协同过滤）   | 对用户-物品评分矩阵做 SVD 降维，预测缺失值                   |
| 自然语言处理（LSA）    | 词-文档矩阵的 SVD 提取潜在语义主题                           |
| 图像处理               | 对图像矩阵做 SVD，保留大奇异值实现压缩/去噪                  |

---

## 8. 常见误区澄清

| 错误说法                             | 正确理解                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------ |
| “非方阵有特征值”                     | 非方阵没有特征值，但可以用奇异值                                       |
| “奇异值就是特征值的绝对值”           | 仅对正规矩阵成立；一般矩阵无此关系                                     |
| “SVD 就是求 \( A^T A \) 的特征分解” | 数值上从不这样做，而是直接双对角化+隐式QR/分治，避免精度损失           |
| “左/右奇异向量总是一样”             | 仅当 \( A \) 对称时才可能（且需半正定）；一般情况不同                   |
| “特征值/奇异值为零的矩阵必为零矩阵”   | 零奇异值只表示矩阵不满秩，不一定全零（如投影矩阵）                     |

---

## 9. 完整示例：PCA 与 SVD

```python
import numpy as np
import matplotlib.pyplot as plt

# 原始数据（未中心化）
X = np.array([[2.5, 1.2],
              [2.2, 1.0],
              [1.8, 0.8],
              [2.0, 1.1],
              [2.3, 1.3]])

# 1. 中心化
X_c = X - np.mean(X, axis=0)

# 2. SVD
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

# 3. 主成分方向 = V 的列
pc1 = Vt[0, :]
pc2 = Vt[1, :]

# 4. 投影到第一主成分
proj_coords = X_c @ pc1
proj_points = np.outer(proj_coords, pc1)

# 5. 绘图
plt.figure(figsize=(6,6))
plt.scatter(X_c[:,0], X_c[:,1], label='Centered data')
plt.scatter(proj_points[:,0], proj_points[:,1], marker='x', c='red', label='Projection on PC1')
for i in range(len(X_c)):
    plt.plot([X_c[i,0], proj_points[i,0]], [X_c[i,1], proj_points[i,1]], 'gray', linestyle='--')
plt.arrow(0, 0, pc1[0]*2*np.sqrt(s[0]), pc1[1]*2*np.sqrt(s[0]), head_width=0.05, color='g', label='PC1')
plt.arrow(0, 0, pc2[0]*2*np.sqrt(s[1]), pc2[1]*2*np.sqrt(s[1]), head_width=0.05, color='orange', label='PC2')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()

print("奇异值:", s)
print("方差比例:", s**2 / np.sum(s**2))
```

---

## 10. 总结

- **特征值/特征向量**：仅对方阵，描述线性变换在特定方向上的缩放。
- **奇异值/奇异向量**：对任意矩阵，描述将单位球映射为椭球的半轴长度和方向。
- **SVD**：最通用的矩阵分解，是数值线性代数、数据科学、信号处理等领域的基石。
- **中心化与协方差**：是 PCA 的关键预处理，使协方差矩阵能正确反映数据内部波动。
