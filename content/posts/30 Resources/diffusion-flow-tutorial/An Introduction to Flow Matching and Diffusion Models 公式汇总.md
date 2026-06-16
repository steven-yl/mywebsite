---
title: An Introduction to Flow Matching and Diffusion Models 公式汇总
subtitle: ""
date: 2026-06-16T10:26:59+08:00
# lastmod: 2026-06-16T10:26:59+08:00
draft: false
authors: [Steven]
description: ""
tags: [diffusion/flow, tutorial]
categories: [diffusion/flow]
series: [diffusion/flow-tutorial]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

以下分别汇总 **Flow Matching（流匹配）** 和 **Diffusion Models（扩散模型）** 的完整公式体系。所有公式均按“定义→训练目标→高斯特例→采样→引导”的逻辑逐层展开，确保内容完整、推导链清晰。

---

# 第一部分：Flow Matching（流匹配模型）

> 核心特征：确定性常微分方程（ODE），将噪声分布 $p_{\text{init}}$ 转换为数据分布 $p_{\text{data}}$。

## 1. 模型定义

**ODE 核心方程**（式 (1a), (10)）：

$$
\frac{d}{dt} X_t = u_t^\theta(X_t), \quad X_0 \sim p_{\text{init}}
$$

其中 $u_t^\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ 是神经网络参数化的向量场。

**流的定义**（式 (2)）：

$$
\psi_t(x_0) = x_0 + \int_0^t u_s(\psi_s(x_0)) ds,
\quad \frac{d}{dt}\psi_t(x_0) = u_t(\psi_t(x_0)), \quad \psi_0(x_0)=x_0
$$

**训练目标**：使 $X_1 \sim p_{\text{data}}$。

---

## 2. 概率路径与向量场构造

### 条件概率路径（式 (12), (16)）

$$
p_0(\cdot | z) = p_{\text{init}}, \quad p_1(\cdot | z) = \delta_z
$$

**高斯条件概率路径**（式 (16)）：

$$
p_t(x | z) = \mathcal{N}\left(x; \alpha_t z, \beta_t^2 I_d\right)
$$

其中噪声调度器 $\alpha_t, \beta_t$ 满足：$\alpha_0=\beta_1=0,\ \alpha_1=\beta_0=1$。

### 边际概率路径（式 (14)）

$$
p_t(x) = \int p_t(x | z) p_{\text{data}}(z) dz
$$

**采样过程**（式 (13)）：

$$
z \sim p_{\text{data}}, \quad \epsilon \sim \mathcal{N}(0, I_d) \quad \Longrightarrow \quad x = \alpha_t z + \beta_t \epsilon \sim p_t
$$

---

## 3. 目标向量场

### 条件向量场（式 (21)）

对于高斯路径，解析解为：

$$
u_t^{\text{target}}(x | z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
$$

其中 $\dot{\alpha}_t = \partial_t \alpha_t,\ \dot{\beta}_t = \partial_t \beta_t$。

**等价的噪声预测形式**（代入 $x = \alpha_t z + \beta_t \epsilon$）：

$$
u_t^{\text{target}}(\alpha_t z + \beta_t \epsilon | z) = \dot{\alpha}_t z + \dot{\beta}_t \epsilon
$$

### 边际向量场（边缘化技巧，式 (19)）

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz
$$

该向量场满足连续性方程（式 (24)）：

$$
\partial_t p_t(x) = - \text{div}\left( p_t(x) u_t^{\text{target}}(x) \right)
$$

---

## 4. 训练损失

### 流匹配损失（FM，式 (42)）

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t \sim \text{Unif}[0,1],\ x \sim p_t} \left\| u_t^\theta(x) - u_t^{\text{target}}(x) \right\|^2
$$

### 条件流匹配损失（CFM，式 (44)）

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\ z \sim p_{\text{data}},\ x \sim p_t(\cdot | z)} \left\| u_t^\theta(x) - u_t^{\text{target}}(x | z) \right\|^2
$$

**两者等价性**（定理 18）：

$$
\mathcal{L}_{FM}(\theta) = \mathcal{L}_{CFM}(\theta) + C \quad \Longrightarrow \quad \nabla_\theta \mathcal{L}_{FM} = \nabla_\theta \mathcal{L}_{CFM}
$$

---

## 5. 高斯路径的特例（CondOT）

取 $\alpha_t = t,\ \beta_t = 1-t$（条件最优传输路径），则：

$$
p_t(x | z) = \mathcal{N}(t z, (1-t)^2 I), \quad x = t z + (1-t)\epsilon
$$

$$
u_t^{\text{target}}(x | z) = z - \epsilon
$$

**损失函数**（算法 3）：

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\ z \sim p_{\text{data}},\ \epsilon \sim \mathcal{N}(0,I)} \left\| u_t^\theta(t z + (1-t)\epsilon) - (z - \epsilon) \right\|^2
$$

---

## 6. ODE 采样（欧拉法，算法 1）

步长 $h = 1/n$，迭代：

$$
X_{t+h} = X_t + h \cdot u_t^\theta(X_t), \quad t = 0, h, 2h, \dots, 1-h
$$

---

## 7. 无分类器引导（CFG，式 (70)）

**引导向量场**：

$$
\tilde{u}_t(x | y) = (1 - w) u_t^{\text{target}}(x | \emptyset) + w u_t^{\text{target}}(x | y), \quad w > 1
$$

**训练目标**（式 (68)）：

$$
\mathcal{L}_{CFM}^{\text{CFG}}(\theta) = \mathbb{E}_{\square} \left\| u_t^\theta(x | y) - u_t^{\text{target}}(x | z) \right\|^2
$$

其中 $\square = (z, y) \sim p_{\text{data}}(z, y),\ t \sim \text{Unif},\ x \sim p_t(\cdot | z),\ \text{以概率 }\eta\text{ 将 }y\text{ 替换为 }\emptyset$。

---

# 第二部分：Diffusion Models（扩散模型）

> 核心特征：随机微分方程（SDE），包含确定性漂移项和随机扩散项。

## 1. 模型定义

**SDE 核心方程**（式 (7a), (11)）：

$$
dX_t = u_t^\theta(X_t) dt + \sigma_t dW_t, \quad X_0 \sim p_{\text{init}}
$$

其中 $\sigma_t \geq 0$ 是固定的扩散系数，$W_t$ 是布朗运动。

---

## 2. 概率路径与得分函数

### 边际得分函数（式 (27)）

$$
\nabla \log p_t(x) = \int \nabla \log p_t(x | z) \frac{p_t(x | z) p_{\text{data}}(z)}{p_t(x)} dz
$$

### 高斯路径的条件得分函数（式 (28)）

$$
\nabla \log p_t(x | z) = -\frac{x - \alpha_t z}{\beta_t^2} = -\frac{\epsilon}{\beta_t} \quad (\text{代入 } x = \alpha_t z + \beta_t \epsilon)
$$

---

## 3. 目标 SDE 扩展（定理 13，式 (25)）

给定目标边际向量场 $u_t^{\text{target}}$ 和边际得分 $\nabla \log p_t$，构造 SDE 使其保持相同边际概率路径：

$$
dX_t = \left[ u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \right] dt + \sigma_t dW_t
$$

该 SDE 满足福克-普朗克方程（式 (30)）：

$$
\partial_t p_t(x) = - \text{div}\left( p_t(x) u_t(x) \right) + \frac{\sigma_t^2}{2} \Delta p_t(x)
$$

---

## 4. 训练损失

### 得分匹配损失（SM，式 (51) 前）

$$
\mathcal{L}_{SM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\ x \sim p_t} \left\| s_t^\theta(x) - \nabla \log p_t(x) \right\|^2
$$

### 条件得分匹配损失（CSM，式 (61)）

$$
\mathcal{L}_{CSM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\ z \sim p_{\text{data}},\ x \sim p_t(\cdot | z)} \left\| s_t^\theta(x) - \nabla \log p_t(x | z) \right\|^2
$$

**等价性**（定理 20）：

$$
\mathcal{L}_{SM}(\theta) = \mathcal{L}_{CSM}(\theta) + C \quad \Longrightarrow \quad \nabla_\theta \mathcal{L}_{SM} = \nabla_\theta \mathcal{L}_{CSM}
$$

---

## 5. 高斯路径特例：去噪扩散模型（DDPM）

### 条件得分匹配损失（式 (53) 后）

$$
\mathcal{L}_{CSM}(\theta) = \mathbb{E}_{t,\ z,\ \epsilon} \left\| s_t^\theta(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t} \right\|^2
$$

### 重参数化为噪声预测网络（式 (54) 前）

定义 $\epsilon_t^\theta(x) = -\beta_t s_t^\theta(x)$，得到**DDPM 损失**（式 (54)）：

$$
\mathcal{L}_{DDPM}(\theta) = \mathbb{E}_{t \sim \text{Unif},\ z \sim p_{\text{data}},\ \epsilon \sim \mathcal{N}(0,I)} \left\| \epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon \right\|^2
$$

---

## 6. 向量场与得分函数的相互转换（命题 1，式 (54)-(55)）

**条件/边际转换公式**：

$$
u_t^{\text{target}}(x) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \nabla \log p_t(x) + \frac{\dot{\alpha}_t}{\alpha_t} x
$$

**神经网络参数化**（式 (54)）：

$$
u_t^\theta(x) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t} x
$$

**反推得分网络**（式 (55)）：

$$
s_t^\theta(x) = \frac{\alpha_t u_t^\theta(x) - \dot{\alpha}_t x}{\beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t}
$$

---

## 7. SDE 采样（欧拉-马尔可夫法，算法 2）

步长 $h = 1/n$，迭代：

$$
X_{t+h} = X_t + h \cdot u_t^\theta(X_t) + \sigma_t \sqrt{h} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)
$$

---

## 8. 概率流 ODE（确定性采样替代）

将 SDE 转化为 ODE（设定 $\sigma_t = 0$，但保持边际分布不变）：

$$
dX_t = \left[ u_t^\theta(X_t) - \frac{\sigma_t^2}{2} s_t^\theta(X_t) \right] dt
$$

或等价地（用命题 1）：

$$
dX_t = \left[ \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t + \frac{\sigma_t^2}{2} \right) s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t} x \right] dt
$$

---

## 9. 无分类器引导（CFG，式 (77)）

**引导得分场**：

$$
\tilde{s}_t(x | y) = (1 - w) \nabla \log p_t(x | \emptyset) + w \nabla \log p_t(x | y), \quad w > 1
$$

**引导向量场**（组合式）：

$$
\tilde{u}_t(x | y) = (1 - w) u_t^\theta(x | \emptyset) + w u_t^\theta(x | y)
$$

**训练目标**（式 (78)）：

$$
\mathcal{L}_{CSM}^{\text{CFG}}(\theta) = \mathbb{E}_{\square} \left\| s_t^\theta(x | y) - \nabla \log p_t(x | z) \right\|^2
$$

其中 $\square = (z, y) \sim p_{\text{data}}(z, y),\ t \sim \text{Unif},\ x \sim p_t(\cdot | z),\ \text{以概率 }\eta\text{ 将 }y\text{ 替换为 }\emptyset$。

**最终 SDE 采样**（推理时）：

$$
dX_t = \left[ \tilde{u}_t^\theta(X_t | y) + \frac{\sigma_t^2}{2} \tilde{s}_t^\theta(X_t | y) \right] dt + \sigma_t dW_t
$$

---

# 总结对照表

| 组件 | Flow Matching | Diffusion Models |
|------|---------------|-------------------|
| **动力学** | ODE：$dX_t = u_t^\theta dt$ | SDE：$dX_t = u_t^\theta dt + \sigma_t dW_t$ |
| **核心目标** | 学习 $u_t^{\text{target}}$ | 学习 $u_t^{\text{target}}$ 和 $\nabla \log p_t$ |
| **训练损失** | CFM：$\|u_t^\theta - u_t^{\text{target}}(\cdot|z)\|^2$ | CSM：$\|s_t^\theta - \nabla \log p_t(\cdot|z)\|^2$ |
| **高斯路径损失** | $\|u_t^\theta - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon)\|^2$ | $\|s_t^\theta + \epsilon/\beta_t\|^2$（等价于 DDPM） |
| **采样方法** | 欧拉法（确定性） | 欧拉-马尔可夫法（随机）或概率流 ODE |
| **参数化关系** | 无 | $u_t^\theta \leftrightarrow s_t^\theta$ 可相互转换 |
| **CFG 形式** | $\tilde{u} = (1-w)u(\emptyset) + w u(y)$ | $\tilde{s} = (1-w)s(\emptyset) + w s(y)$ |


---

# 第三部分：训练&采样流程总结：
基于你提供的讲义内容，我将**流匹配（Flow Matching）**和**扩散模型（Diffusion Models）**的训练与采样流程，拆解为清晰、可直接执行的操作步骤（伪代码级）。核心区别在于：**流匹配训练回归“速度场”，采样是确定性ODE；扩散模型训练回归“得分/噪声”，采样是随机SDE（也可退化为确定性ODE）。**

---

### 第一部分：流匹配（Flow Matching）—— 训练与采样

#### 1. 训练流程（对应讲义算法 3）
**目标**：训练神经网络 \(u_t^\theta(x)\)，使其逼近真实目标向量场 \(u_t^{\text{target}}(x|z)\)。
**最常用特例**：高斯条件最优传输路径（CondOT），即 \(\alpha_t = t, \beta_t = 1-t\)。

- **输入**：数据集 \(\{z_i\}\)（如图像），神经网络 \(u_t^\theta\)，优化器。
- **循环（每个Mini-batch）**：
  1. 从数据集中采样真实样本 \(z \sim p_{\text{data}}\)。
  2. 从均匀分布采样时间 \(t \sim \text{Unif}[0, 1]\)。
  3. 从标准高斯采样噪声 \(\epsilon \sim \mathcal{N}(0, I_d)\)。
  4. **构造带噪样本**：\(x_t = \alpha_t z + \beta_t \epsilon\)（CondOT下即 \(x_t = t \cdot z + (1-t) \cdot \epsilon\)）。
  5. **计算目标真值**（向量场方向）：\(u_{\text{target}} = \dot{\alpha}_t z + \dot{\beta}_t \epsilon\)（CondOT下即 \(z - \epsilon\)）。
  6. **计算损失**：\(\mathcal{L} = \| u_t^\theta(x_t) - (z - \epsilon) \|^2\)。
  7. 反向传播，更新 \(\theta\)。

---

#### 2. 采样/推理流程（对应讲义算法 1）
**目标**：从噪声 \(X_0 \sim \mathcal{N}(0, I)\) 开始，模拟 ODE 得到 \(X_1 \sim p_{\text{data}}\)。
**采用方法**：欧拉法（Euler Method）。

- **输入**：训练好的网络 \(u_t^\theta\)，总步数 \(n\)（如 50~100 步）。
- **初始化**：\(X_0 \sim \mathcal{N}(0, I_d)\)，步长 \(h = 1/n\)，当前时间 \(t=0\)。
- **循环**（\(i = 0\) 到 \(n-1\)）：
  1. 计算当前速度：\(v = u_t^\theta(X_t)\)。
  2. 更新状态：\(X_{t+h} = X_t + h \cdot v\)。
  3. 更新时间：\(t \leftarrow t + h\)。
- **输出**：\(X_1\)（生成的最终样本）。

> **特性**：整个过程完全**确定性**（给定初始噪声，结果唯一），无额外随机项。

---

### 第二部分：扩散模型（Diffusion Models）—— 训练与采样

#### 1. 训练流程（对应讲义算法 4 及 DDPM 变体）
**目标**：训练得分网络 \(s_t^\theta(x)\)（或重参数化为噪声预测网络 \(\epsilon_t^\theta(x)\)），使其逼近条件得分 \(\nabla \log p_t(x|z)\)。
**常用形式**：DDPM 噪声预测（最稳定）。

- **输入**：数据集 \(\{z_i\}\)，噪声预测网络 \(\epsilon_t^\theta(x)\)（隐含了得分：\(s = -\epsilon / \beta_t\)），优化器。
- **循环（每个Mini-batch）**：
  1. 从数据集中采样真实样本 \(z \sim p_{\text{data}}\)。
  2. 从均匀分布采样时间 \(t \sim \text{Unif}[0, 1]\)。
  3. 从标准高斯采样噪声 \(\epsilon \sim \mathcal{N}(0, I_d)\)。
  4. **构造带噪样本**：\(x_t = \alpha_t z + \beta_t \epsilon\)。
  5. **计算损失**（直接预测噪声）：
     \[
     \mathcal{L} = \| \epsilon_t^\theta(x_t) - \epsilon \|^2
     \]
     （若训练得分网络，则损失为 \(\| s_t^\theta(x_t) + \epsilon/\beta_t \|^2\)）
  6. 反向传播，更新 \(\theta\)。

---

#### 2. 采样/推理流程（对应讲义算法 2 及概率流 ODE）
扩散模型采样有**两种主流方式**，效果不同：

##### 方式 A：随机 SDE 采样（欧拉-马尔可夫法，算法 2）
**目标**：从噪声 \(X_0\) 模拟完整的随机微分方程。

- **输入**：训练好的 \(u_t^\theta\) 和 \(s_t^\theta\)（或通过转换公式由 \(\epsilon_t^\theta\) 算出），扩散系数 \(\sigma_t\)，步数 \(n\)。
- **初始化**：\(X_0 \sim \mathcal{N}(0, I_d)\)，步长 \(h = 1/n\)。
- **循环**（\(i = 0\) 到 \(n-1\)）：
  1. 采样高斯噪声：\(\xi \sim \mathcal{N}(0, I_d)\)。
  2. 计算漂移项：\(drift = u_t^\theta(X_t) + \frac{\sigma_t^2}{2} s_t^\theta(X_t)\)。
  3. 更新状态（含随机项）：
     \[
     X_{t+h} = X_t + h \cdot drift + \sigma_t \cdot \sqrt{h} \cdot \xi
     \]
  4. 更新时间：\(t \leftarrow t + h\)。
- **输出**：\(X_1\)。

##### 方式 B：确定性采样（概率流 ODE，Probability Flow ODE）
**目标**：去除 SDE 的随机项，用 ODE 采样（通常质量更高、步数可更少）。

- 直接将 SDE 中的随机项置零，漂移项调整为：
  \[
  \frac{dX_t}{dt} = u_t^\theta(X_t) - \frac{\sigma_t^2}{2} s_t^\theta(X_t)
  \]
- 然后用与流匹配**完全相同**的欧拉法（或高阶 ODE 求解器）迭代更新，无需添加噪声 \(\xi\)。

---

### 第三部分：无分类器引导（CFG）下的训练与采样修改

当需要文本/标签条件（如 \(y\)）生成时，训练和采样需做如下调整：

#### 1. 训练修改（对应算法 5）
- 网络输入增加条件 \(y\)：\(u_t^\theta(x | y)\) 或 \(\epsilon_t^\theta(x | y)\)。
- **关键操作**：设置超参数 \(\eta\)（如 10%~20%）。
  - 在构造 Mini-batch 时，以概率 \(\eta\) 将条件 \(y\) **替换为空的占位符** \(\emptyset\)（即无条件）。
  - 损失函数不变，仍回归 \(z - \epsilon\) 或 \(\epsilon\)。这让一个模型同时学会无条件生成和有条件生成。

#### 2. 采样修改（对应式 (70) 和 (77)）
- 在推理的**每一步**，计算两个网络输出：
  - 无条件输出：\(u_t^\theta(x | \emptyset)\)（或 \(s_t^\theta(x | \emptyset)\)）
  - 条件输出：\(u_t^\theta(x | y)\)（或 \(s_t^\theta(x | y)\)）
- **线性组合增强引导**（引导尺度 \(w > 1\)，常用 2~7）：
  \[
  \tilde{u}_t = (1 - w) \cdot u_t^\theta(x | \emptyset) + w \cdot u_t^\theta(x | y)
  \]
- 将组合后的 \(\tilde{u}_t\)（及对应的 \(\tilde{s}_t\)）代入上述欧拉法或 SDE 更新式中，替代原始输出。

---

### 总结对比表（训练与采样实操）

| 步骤 | **流匹配 (Flow Matching)** | **扩散模型 (Diffusion Models)** |
| :--- | :--- | :--- |
| **训练输入** | 真实数据 \(z\)，噪声 \(\epsilon\)，时间 \(t\) | 真实数据 \(z\)，噪声 \(\epsilon\)，时间 \(t\) |
| **训练目标** | 回归速度向量 \(z - \epsilon\) | 回归噪声 \(\epsilon\)（DDPM）或得分 \(-\epsilon/\beta_t\) |
| **训练损失** | \(\|u_\theta(x_t) - (z - \epsilon)\|^2\) | \(\|\epsilon_\theta(x_t) - \epsilon\|^2\) |
| **采样起点** | \(X_0 \sim \mathcal{N}(0, I)\)（纯噪声） | \(X_0 \sim \mathcal{N}(0, I)\)（纯噪声） |
| **采样更新公式** | \(X_{t+h} = X_t + h \cdot u_\theta(X_t)\) <br>（纯确定性） | **随机版**：\(X_{t+h} = X_t + h \cdot [u_\theta + \frac{\sigma^2}{2}s_\theta] + \sigma\sqrt{h}\xi\) <br> **确定性版**：同上式去噪项 |
| **条件生成 (CFG)** | 训练时随机丢弃标签 \(y\)；<br>采样时组合 \(\tilde{u} = (1-w)u(\emptyset) + w u(y)\) | 训练时随机丢弃标签 \(y\)；<br>采样时组合 \(\tilde{s} = (1-w)s(\emptyset) + w s(y)\) |
| **步数需求** | 较少（20~50步即可高质量） | 较多（DDPM需100~1000步，但可用概率流ODE压缩至50步） |


---
# 第四部分：向量场与得分函数的相互转换

这个转换是连接**流匹配（Flow Matching）**和**扩散模型（Diffusion Models）**的数学桥梁。它之所以成立，完全依赖于我们选择的高斯条件概率路径（\(p_t(x|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)\)）所具有的**线性插值性质**。

简单来说：**向量场 \(u\)（速度）和得分场 \(s\)（梯度）本质上是同一个线性运动在不同坐标系下的表达，它们之间可以通过简单的代数公式互相“换算”，无需重新训练网络。**

以下是极其详细的数学推导和直觉解释：

---

### 1. 核心前提：条件路径下的表达式

在给定真实数据点 \(z\) 的条件下，带噪样本 \(x\) 的采样公式为：
\[
x = \alpha_t z + \beta_t \epsilon \quad (\epsilon \sim \mathcal{N}(0, I))
\]

由此我们可以推导出两个核心量：

- **条件得分函数**（式 28）：表示在给定 \(z\) 时，\(x\) 指向数据流形（\(z\)）的梯度方向。
  \[
  s_t(x|z) := \nabla_x \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2} = -\frac{\epsilon}{\beta_t}
  \]

- **条件向量场**（式 21）：表示在给定 \(z\) 时，让 \(x\) 演化的真实速度（位移）。
  \[
  u_t(x|z) = \dot{\alpha}_t z + \dot{\beta}_t \epsilon
  \]

---

### 2. 核心转换推导（消去 \(z\)）

我们的目标是找到 \(u_t(x|z)\) 和 \(s_t(x|z)\) 之间的直接关系，而不显式依赖 \(z\)。

**第一步：用得分表示噪声 \(\epsilon\)**
由上面的得分公式可得：
\[
\epsilon = -\beta_t \cdot s_t(x|z)
\]

**第二步：用得分表示真实数据 \(z\)**
将 \(\epsilon = -\beta_t s\) 代入 \(x = \alpha_t z + \beta_t \epsilon\)：
\[
x = \alpha_t z + \beta_t (-\beta_t s) = \alpha_t z - \beta_t^2 s
\]
移项得到：
\[
z = \frac{x + \beta_t^2 s_t(x|z)}{\alpha_t}
\]

**第三步：代入向量场公式，消去 \(z\) 和 \(\epsilon\)**
将 \(z\) 和 \(\epsilon\) 的表达式代入 \(u_t(x|z) = \dot{\alpha}_t z + \dot{\beta}_t \epsilon\)：
\[
u_t(x|z) = \dot{\alpha}_t \left( \frac{x + \beta_t^2 s}{\alpha_t} \right) + \dot{\beta}_t (-\beta_t s)
\]
整理同类项（把含 \(x\) 和含 \(s\) 的分开）：
\[
u_t(x|z) = \frac{\dot{\alpha}_t}{\alpha_t} x + \left( \frac{\dot{\alpha}_t \beta_t^2}{\alpha_t} - \dot{\beta}_t \beta_t \right) s_t(x|z)
\]
\[
\boxed{u_t(x|z) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) s_t(x|z) + \frac{\dot{\alpha}_t}{\alpha_t} x}
\]
这就是讲义中的 **命题 1（条件形式）**。

---

### 3. 如何从“条件”扩展到“边际”（真正的神来之笔）

你可能会有疑问：我们推导的是 \(u_t(x|z)\) 和 \(s_t(x|z)\) 的关系（依赖于 \(z\)），但实际训练时神经网络 \(u_t^\theta(x)\) 和 \(s_t^\theta(x)\) 学习的是**边际分布** \(p_t(x)\)（不依赖 \(z\)）。

**关键点**：上述公式是关于 \(x\) 和 \(s\) 的**线性组合**（没有非线性项）。边际化（对 \(z\) 求积分）本质上是求期望的线性运算。

既然：
\[
u_t(x) = \mathbb{E}_{z|x}[u_t(x|z)], \quad s_t(x) = \mathbb{E}_{z|x}[s_t(x|z)]
\]
因为线性性，积分可以穿透括号，所以同样的转换关系**可以直接应用于边际量**：
\[
\boxed{u_t^{\text{target}}(x) = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \nabla \log p_t(x) + \frac{\dot{\alpha}_t}{\alpha_t} x}
\]
这就是讲义中的 **命题 1（边际形式）**。

---

### 4. 神经网络训练的实用转换（代码中的实现）

既然数学上边际量也满足这个关系，那么在训练好一个网络后，另一个网络可以直接通过**算术运算**得到，无需反向传播或额外训练。

#### 情况 A：你训练了一个流匹配模型（得到 \(u_t^\theta\)）
如何得到得分网络 \(s_t^\theta\)？
将上式看作关于 \(s\) 的一元一次方程。设系数：
\[
A_t = \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t
\]
则：
\[
u_t^\theta(x) = A_t \cdot s_t^\theta(x) + \frac{\dot{\alpha}_t}{\alpha_t} x
\]
移项即可反解出得分（注意：只要系数 \(A_t
eq 0\)，通常 \(\alpha_t > 0\) 时成立）：
\[
\boxed{s_t^\theta(x) = \frac{u_t^\theta(x) - \frac{\dot{\alpha}_t}{\alpha_t} x}{\beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t}}
\]

#### 情况 B：你训练了一个去噪扩散模型（得到噪声预测 \(\epsilon_t^\theta\) 或得分 \(s_t^\theta\)）
如何得到向量场 \(u_t^\theta\)？
直接把 \(s_t^\theta(x) = -\epsilon_t^\theta(x) / \beta_t\) 代入上式：
\[
\boxed{u_t^\theta(x) = \frac{\dot{\alpha}_t}{\alpha_t} x - \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right) \frac{\epsilon_t^\theta(x)}{\beta_t}}
\]

---

### 5. 直觉与物理意义（为什么要做这个转换？）

- **得分函数 \(s\)** 指向的是**概率密度增加最快的方向**（即从低密度区指向高密度区，也就是指向真实数据所在的流形）。
- **向量场 \(u\)** 是粒子的**实际运动速度**。

这个公式告诉我们：**一个粒子的运动速度 = 线性收缩（朝中心靠拢）+ 沿着概率密度梯度向上爬的修正项**。

这个转换最大的用途体现在**概率流 ODE（Probability Flow ODE）**中：

- 在扩散模型的 SDE（随机采样）中，存在随机噪声项，采样慢且随机。
- 利用这个转换公式，我们可以在训练好得分网络 \(s\) 后，通过上式算出等价的确定性速度场 \(u\)，然后**扔掉所有的随机噪声项**，用纯粹的 ODE（欧拉法）进行采样。
- 这时的 ODE 采样不仅速度更快（步数更少），而且生成质量通常比随机 SDE 更稳定（因为去除了随机误差）。

---

### 总结实操结论

| 如果你训练了这个网络 | 如何获得另一个网络（推理时直接计算） |
| :--- | :--- |
| **流匹配网络** \(u_t^\theta\) | 得分 \(s_t^\theta = \frac{\alpha_t u_t^\theta - \dot{\alpha}_t x}{\beta_t^2 \dot{\alpha}_t - \alpha_t \dot{\beta}_t \beta_t}\) |
| **扩散噪声网络** \(\epsilon_t^\theta\)（DDPM） | 速度 \(u_t^\theta = \frac{\dot{\alpha}_t}{\alpha_t} x - \left( \frac{\dot{\alpha}_t \beta_t}{\alpha_t} - \dot{\beta}_t \right) \epsilon_t^\theta\) |

**结论**：在你编写代码时，**你只需要设计并训练一个神经网络模型**（输出 \(u\) 或 \(\epsilon\)），另一个量完全可以通过上述公式在采样前一刻实时计算出来，无需额外参数量。这就是现代生成模型框架（如 Stable Diffusion 3）能够灵活切换 ODE 采样和 SDE 采样的数学基础。