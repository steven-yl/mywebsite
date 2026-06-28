---
title: EDM算法原理和公式推导
date: 2026-06-26
draft: false
authors: [Steven]
description: ""
tags: [diffusion/flow, EDM]
categories: [diffusion/flow, EDM]
series: [diffusion/flow-tutorial]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

### 0. 预备符号与设定
设数据集的 **像素级标准差** 为 $\sigma_d$（通常对归一化到 $[0,1]$ 或 $[-1,1]$ 的图像取 $\sigma_d \approx 0.5$ 或 $1$）。去噪网络 $D_\theta(x;\sigma)$ 旨在从被高斯噪声扰动 $\sigma$ 的样本 $x_\sigma$ 中预测干净的原始样本 $x_0$。

---

### 1. 前向加噪（扰动数据）
$$
x_\sigma = x_0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$
**推导动机**：这是扩散模型（或得分匹配）的标准前向过程。给定噪声水平 $\sigma$，条件分布为 $p(x_\sigma \mid x_0) = \mathcal{N}(x_0, \sigma^2 \mathbf{I})$。在此设定下，**得分函数**（Score Function）$\nabla_{x_\sigma} \log p(x_\sigma \mid x_0)$ 存在解析闭式解：
$$
\nabla_{x_\sigma} \log p(x_\sigma \mid x_0) = -\frac{x_\sigma - x_0}{\sigma^2} = \frac{x_0 - x_\sigma}{\sigma^2}
$$
这正是后续 ODE 和去噪网络的关键桥梁。

---

### 2. 去噪网络预处理封装（网络参数化）
$$
D_\theta(x;\sigma) = c_{\text{skip}}(\sigma) \cdot x + c_{\text{out}}(\sigma) \cdot F_\theta\!\left( c_{\text{in}}(\sigma) \cdot x;\ c_{\text{noise}}(\sigma) \right)
$$
其中：
$$
c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_d^2}},\quad
c_{\text{out}} = \frac{\sigma \cdot \sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}},\quad
c_{\text{skip}} = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2},\quad
c_{\text{noise}} = \frac{1}{4}\ln \sigma
$$

**推导与设计原理**（旨在改善网络 $F_\theta$ 的训练条件）：

- **输入归一化 $c_{\text{in}}$**：原始输入 $x$ 的方差为 $\mathbb{E}[\|x\|^2] = \sigma_d^2 + \sigma^2$。将其乘以 $c_{\text{in}}$ 后，使得网络 $F_\theta$ 的输入始终具有单位方差（$\approx 1$），避免了网络需要适应不同幅值输入的问题。

- **跳跃连接 $c_{\text{skip}}$ 与输出缩放 $c_{\text{out}}$**：
  我们希望 $D_\theta$ 是 **最优去噪器**。当 $\sigma \to 0$ 时，$D_\theta(x;\sigma) \to x$（恒等映射）；当 $\sigma \to \infty$ 时，先验信息失效，最优估计应趋向于数据集的均值（归一化后为 $0$），即 $D_\theta \to 0$。
  令 $c_{\text{skip}} + c_{\text{out}} \cdot F_\theta = 1$ 且 $F_\theta \to 0$ 时，自然满足极端情况。更关键的是，**最优去噪器**（MMSE估计）的解析解为：
  $$
  D^* = x_0 = \mathbb{E}[x_0 \mid x_\sigma]
  $$
  若 $F_\theta$ 只是普通的残差网络，其输出幅度会随 $\sigma$ 剧烈变化。通过设定 $c_{\text{out}} = \sigma \sigma_d / \sqrt{\sigma^2 + \sigma_d^2}$，使得 **无论 $\sigma$ 如何变化，$D_\theta$ 的输出幅值始终保持在 $\mathcal{O}(\sigma_d)$ 级别**，极大稳定了梯度流。

- **噪声条件 $c_{\text{noise}} = \frac{1}{4}\ln \sigma$**：直接将 $\sigma$ 或 $\ln \sigma$ 输入网络会导致正弦位置编码（Sinusoidal Embedding）的输入范围过大或过小。乘以 $1/4$ 可以确保在典型的 $\sigma \in [0.002, 80]$ 范围内，输入值大致落在 $[-2, 2]$ 之间，恰好匹配正弦编码的最佳响应区间。

---

### 3. 训练损失与均衡权重函数
$$
\mathcal{L} = \mathbb{E}_{\sigma, x_0, \epsilon}\left[
\underbrace{\frac{\sigma^2 + \sigma_d^2}{(\sigma \cdot \sigma_d)^2}}_{\lambda(\sigma)}
\cdot \left\| D_\theta(x_0 + \sigma\epsilon;\sigma) - x_0 \right\|^2
\right]
$$

**推导过程（关键步骤——梯度平衡）**：

1. 将 $D_\theta = c_{\text{skip}} x + c_{\text{out}} F_\theta$ 代入损失（暂不考虑权重 $\lambda$），并计算损失对网络 $F_\theta$ 参数的梯度。
2. 梯度中必然会乘以一个因子 $c_{\text{out}}$（链式法则）。即：
   $$
   \nabla_\theta \mathcal{L} \propto c_{\text{out}}(\sigma) \cdot (D_\theta - x_0) \cdot \nabla_\theta F_\theta
   $$
3. 在训练初期或最优解附近，残差项 $D_\theta - x_0$ 的幅值近似与 $c_{\text{out}}(\sigma)$ 同阶。
4. 因此，梯度幅值大致正比于 $c_{\text{out}}^2(\sigma)$。若不加权重，大 $\sigma$ 的梯度会远大于小 $\sigma$ 的梯度，导致训练偏向高噪声水平。
5. 为了 **抹平不同 $\sigma$ 下的梯度尺度**，设计权重 $\lambda(\sigma)$ 使得：
   $$
   \lambda(\sigma) \cdot c_{\text{out}}^2(\sigma) = \text{常数}
   $$
   取常数为 $1$，则：
   $$
   \lambda(\sigma) = \frac{1}{c_{\text{out}}^2(\sigma)} = \frac{\sigma^2 + \sigma_d^2}{\sigma^2 \sigma_d^2}
   $$
   至此，公式中的权重函数便严格推导得出。这保证了网络 $F_\theta$ 在所有噪声水平上都能接收到均衡的更新信号。

---

### 4. 概率流常微分方程（Probability Flow ODE）
$$
\frac{dx}{d\sigma} = \frac{x - D_\theta(x;\sigma)}{\sigma}
$$

**推导**：
根据随机微分方程（SDE）理论，Variance-Exploding（VE）SDE 的前向过程为 $dx = \sqrt{2\dot{\sigma}(t)\sigma(t)} \, dw$。其对应的 **概率流 ODE** 为：
$$
\frac{dx}{dt} = -\dot{\sigma}(t) \sigma(t) \nabla_x \log p_t(x)
$$
将其从时间 $t$ 变量换元为噪声水平 $\sigma$（即 $d\sigma = \dot{\sigma}(t) dt$）：
$$
\frac{dx}{d\sigma} = \frac{dx/dt}{d\sigma/dt} = -\sigma \nabla_x \log p_\sigma(x)
$$
由得分匹配的性质，网络 $D_\theta$ 估计的是干净数据 $x_0$，而得分函数 $\nabla_x \log p_\sigma(x) = (D_\theta - x) / \sigma^2$。代入上式得：
$$
\frac{dx}{d\sigma} = -\sigma \cdot \frac{D_\theta - x}{\sigma^2} = \frac{x - D_\theta}{\sigma}
$$
证毕。

---

### 5. 离散化采样（二阶 Heun 预测-校正）
$$
\begin{cases}
\tilde{x}_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) \cdot \dfrac{x_i - D_\theta(x_i;\sigma_i)}{\sigma_i} & (\text{预测步}) \$$6pt]
x_{i+1} = x_i + \dfrac{\sigma_{i+1} - \sigma_i}{2} \cdot \left[ \dfrac{x_i - D_\theta(x_i;\sigma_i)}{\sigma_i} + \dfrac{\tilde{x}_{i+1} - D_\theta(\tilde{x}_{i+1};\sigma_{i+1})}{\sigma_{i+1}} \right] & (\text{校正步})
\end{cases}
$$

**推导**：
将 ODE 写为紧凑形式 $dx/d\sigma = f(x, \sigma)$，其中 $f(x, \sigma) = (x - D_\theta(x;\sigma))/\sigma$。对区间 $[\sigma_i, \sigma_{i+1}]$ 进行数值积分：

- **显式欧拉（预测）**：
  $$
  \tilde{x}_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) \cdot f(x_i, \sigma_i)
  $$

- **梯形法则（校正）**：为了提高精度，避免显式欧拉的累积误差，使用二阶 Runge-Kutta（即 Heun 方法），将预测步的斜率与当前步的斜率取算术平均：
  $$
  x_{i+1} = x_i + \frac{\Delta \sigma}{2} \left[ f(x_i, \sigma_i) + f(\tilde{x}_{i+1}, \sigma_{i+1}) \right]
  $$
  代入 $f$ 的定义，即得到框中的校正步公式。该二阶方法仅比欧拉多计算一次网络前向，但显著降低了局部截断误差至 $\mathcal{O}(\Delta \sigma^2)$。

---

### 6. 噪声水平调度（幂律分布）
$$
\sigma_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left( \sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho} \right) \right)^{\rho}, \quad i \in [0, N-1]
$$

**推导**：
概率流 ODE 在 $\sigma$ 较大时变化平缓，在 $\sigma$ 较小时（接近 0）变化剧烈且刚性较强。若采用线性间隔（$\rho=1$），则大部分采样点会浪费在高噪声区，而低噪声区步长过大导致离散误差爆炸。

- 设变换 $y = \sigma^{1/\rho}$，则 $\ln \sigma = \rho \ln y$。
- 为了使 $\sigma$ 在对数尺度（即 $\ln \sigma$）上均匀分布，我们令 $y$ 在线性空间均匀分布。
- 当 $\rho > 1$ 时，幂律映射使得 $\sigma$ 的分布向低噪声区域倾斜（因为指数 $\rho$ 压缩了小 $y$ 对应的 $\sigma$ 间隔，展开了大 $y$ 的间隔）。

**具体构造**：
令 $y_i$ 在 $[y_{\max}, y_{\min}]$ 上线性插值（注意从最大到最小）：
$$
y_i = \sigma_{\max}^{1/\rho} + \frac{i}{N-1} \left( \sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho} \right)
$$
代回 $\sigma_i = y_i^{\rho}$，即得到上述调度公式。实际应用中常取 $\rho \approx 7$，在低噪声区精细采样以保证生成质量。

---

### 总结
整个框架的推导形成了一个完整的闭环：
1. **网络封装** 通过 $c_{in}, c_{out}, c_{skip}$ 改善网络条件；
2. **损失权重** $\lambda$ 通过抵消 $c_{out}^2$ 均衡训练信号；
3. **ODE** 将去噪问题转化为确定性的常微分方程求解；
4. **Heun采样器** 提供高精度的离散化积分；
5. **幂律调度** 针对 ODE 的刚性分配最优步长。

这些推导细节共同保证了 EDM 在扩散模型采样质量和速度上的领先地位。