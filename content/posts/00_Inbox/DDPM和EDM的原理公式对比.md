---
title: DDPM和EDM的原理公式对比
date: 2026-06-26
draft: false
authors: [Steven]
description: ""
tags: [diffusion/flow, tutorial]
categories: [diffusion/flow, tutorial]
series: [diffusion/flow-tutorial]
weight: 0
series_weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

从**数学定义**出发，分**7个核心步骤**彻底拆解DDPM和EDM的原理公式，并**在每一步中进行逐项对比**。

我们约定：

- DDPM使用离散时间步$t \in [0, T]$（$T=1000$）。
- EDM使用连续噪声水平$\sigma \in [\sigma_{\min}, \sigma_{\max}]$。
- 定义数据分布方差为$\sigma_{\text{data}}$（通常取$0.5$）。

---

### 第1步：前向加噪过程（扩散路径的定义）

这是两者最根本的数学起点，决定了后续所有公式的形态。

- **DDPM 前向过程**：
  定义$\beta_t$为方差调度（Variance Schedule）。给定$x_{t-1}$到$x_t$的转移核为高斯分布：
 $$
  q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})
 $$
  令$\alpha_t = 1 - \beta_t$，累积乘积$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。从$x_0$到$x_t$的闭式解（边缘分布）为：
 $$
  x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) \tag{1-1}
 $$

- **EDM 前向过程**：
  EDM彻底简化，直接使用**方差爆炸（VE）** 的连续形式，定义噪声强度为$\sigma$。前向扰动核直接写为：
 $$
  p(x_\sigma | x_0) = \mathcal{N}(x_\sigma; x_0, \sigma^2 \mathbf{I}) \quad \Longleftrightarrow \quad x_\sigma = x_0 + \sigma \epsilon \tag{1-2}
 $$

- **🔍 本步对比**：
  - DDPM含有信号衰减项$\sqrt{\bar{\alpha}_t} x_0$（数据在加噪中逐渐变暗）。
  - EDM **没有信号衰减**（$x_0$前面的系数恒为1），数据只叠加噪声，保持幅度恒定。

---

### 第2步：变量替换（证明数学同源性）

为了看看它们是否描述同一个东西，我们进行坐标变换。

- 将 DDPM 的公式 (1-1) **两边同时除以**$\sqrt{\bar{\alpha}_t}$：
 $$
  \frac{x_t}{\sqrt{\bar{\alpha}_t}} = x_0 + \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \epsilon \tag{2-1}
 $$

- 定义 EDM 视角下的缩放坐标$\hat{x}_t = \frac{x_t}{\sqrt{\bar{\alpha}_t}}$，以及等效噪声水平：
 $$
  \sigma(t) = \frac{\sqrt{1 - \bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}} \tag{2-2}
 $$

- **结论**：将 DDPM 的坐标轴动态缩放后，得到$\hat{x}_t = x_0 + \sigma(t) \epsilon$，这与 EDM 的公式 (1-2) **数学形式完全一致**。所以，**边缘概率分布$p(x_t)$是同构的**。

---

### 第3步：反向去噪的微分方程（生成过程的底层逻辑）

生成过程是求解逆向扩散的微分方程。

- **DDPM 反向过程（离散SDE视角）**：
  DDPM 基于马尔可夫链，其逆向去噪均值$\mu_\theta(x_t, t)$由贝叶斯公式推导得出。其采样迭代公式（DDPM标准采样）为：
 $$
  x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} z, \quad z \sim \mathcal{N}(0, \mathbf{I}) \tag{3-1}
 $$
  这是一个**一阶离散化**的随机微分方程（SDE）。

- **EDM 反向过程（连续ODE视角）**：
  EDM 避开随机性，利用 Fokker-Planck 方程推导出对应的**概率流常微分方程（Probability Flow ODE）**。在 EDM 的$\sigma$坐标系下，这个 ODE 拥有极其简洁的形式：
 $$
  dx = -\dot{\sigma}(t) \sigma(t) \nabla_x \log p(x; \sigma) \, dt \tag{3-2}
 $$
  利用得分函数与去噪器$D_\theta$的关系$\nabla_x \log p(x; \sigma) = (D_\theta(x; \sigma) - x) / \sigma^2$，可将 ODE 改写为关于$\sigma$的离散迭代：
 $$
  \frac{dx}{d\sigma} = \frac{x - D_\theta(x; \sigma)}{\sigma} \tag{3-3}
 $$

- **🔍 本步对比**：
  - DDPM 求解的是 **SDE（含随机噪声项$\sqrt{\beta_t} z$）**，且是一阶近似，截断误差大，需要极小的步长。
  - EDM 求解的是 **ODE（确定性，无随机项）**，且公式（3-3）极其干净，允许使用高阶数值求解器大步长跳跃。

---

### 第4步：网络参数的预处理与缩放（Preconditioning）

这是 **DDPM 与 EDM 在公式结构上最核心、最本质的分水岭**。DDPM 直接输出结果，EDM 对网络进行了人为的数学封装。

- **DDPM 网络参数化**：
  直接让神经网络$\epsilon_\theta$去拟合添加的噪声$\epsilon$，没有额外的输入输出缩放：
 $$
  \text{网络目标：} \quad \epsilon_\theta(x_t, t) \approx \epsilon \tag{4-1}
 $$

- **EDM 网络参数化**：
  EDM 让神经网络$F_\theta$去预测干净的$x_0$，但**不直接使用$F_\theta$的输出**。而是设计了一个精巧的线性组合器$D_\theta$：
 $$
  D_\theta(x; \sigma) = c_{\text{skip}}(\sigma) \cdot x + c_{\text{out}}(\sigma) \cdot F_\theta\left( c_{\text{in}}(\sigma) \cdot x ; \; c_{\text{noise}}(\sigma) \right) \tag{4-2}
 $$
  其中四个系数的精确数学定义为（令$\sigma_{\text{data}}$为数据标准差）：
 $$
  c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}, \quad
  c_{\text{out}}(\sigma) = \frac{\sigma \cdot \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}
 $$
 $$
  c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \quad
  c_{\text{noise}}(\sigma) = 0.25 \ln(\sigma)
 $$

- **🔍 本步对比**：
  - **DDPM**：网络输入幅度随$t$剧烈变化（因为$x_t$的方差在变），导致训练不稳定。
  - **EDM**：通过$c_{\text{in}}$强制网络输入恒为**单位方差**；通过$c_{\text{skip}}$引入**自适应跳跃连接**（$\sigma$大时，输出基本等于输入，网络只学微小残差）。这极大降低了网络优化的曲率。

---

### 第5步：训练目标函数与损失加权（Loss Weighting）

- **DDPM 训练损失**：
  根据变分下界（VLB）简化，DDPM 去掉权重项，直接使用均方误差（MSE）：
 $$
  \mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right] \tag{5-1}
 $$
  此时，所有时间步$t$的损失贡献权重**完全相等**。

- **EDM 训练损失**：
  EDM 让网络预测$x_0$，并引入依赖于$\sigma$的显式权重函数$\lambda(\sigma)$：
 $$
  \mathcal{L}_{\text{EDM}} = \mathbb{E}_{\sigma, x_0, \epsilon} \left[ \lambda(\sigma) \cdot \| D_\theta(x_0 + \sigma\epsilon; \sigma) - x_0 \|^2 \right] \tag{5-2}
 $$
  其中，EDM 推导出的最优权重（用于平衡各噪声水平的梯度）为：
 $$
  \lambda(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2} \tag{5-3}
 $$

- **🔍 本步数学推导（为什么 EDM 要加权）**：
  由于$\epsilon = (x_\sigma - x_0)/\sigma$，将 DDPM 的$\epsilon$预测转换为$x_0$预测时，天然会带一个$1/\sigma^2$的系数。为了不让大$\sigma$（低信噪比）的样本主导梯度，EDM 特意设计了$\lambda(\sigma)$来抵消这个效应，让模型对**所有噪声水平一视同仁**，从而精细刻画低噪声区域的纹理。

---

### 第6步：离散化步长的调度策略（Schedule）

- **DDPM 调度**：
  使用线性调度（或余弦调度）。对于$t \in [0, T]$，定义$\beta_t$为线性插值。其对应的$\sigma(t)$在离散步长中变化为：
 $$
  \Delta \sigma_t = \sigma(t) - \sigma(t-1)
 $$
  在中间区域（$\sigma \approx 1$附近）变化快，在两端（极低噪声和极高噪声）变化慢，**步长分配不太合理**。

- **EDM 调度**：
  EDM 直接定义在$\sigma$空间上的离散化点$\{\sigma_i\}_{i=0}^{N-1}$。采用了**幂律插值（Power-law interpolation）**：
 $$
  \sigma_i = \left( \sigma_{\text{max}}^{1/\rho} + \frac{i}{N-1} \left( \sigma_{\text{min}}^{1/\rho} - \sigma_{\text{max}}^{1/\rho} \right) \right)^{\rho} \tag{6-1}
 $$
  通常取$\rho = 7$。

- **🔍 本步对比**：
  - DDPM 的$t$是均匀离散的，但对应的$\sigma(t)$是不均匀的。
  - EDM 直接操控$\sigma_i$，由于$\rho=7$，**在低噪声区域（$\sigma$接近 0）采样点极密，在高噪声区域采样点极疏**。这精准对应了生成质量主要取决于“精细纹理恢复”的物理事实。

---

### 第7步：采样求解器的数值精度（迭代公式）

- **DDPM 采样（一阶 Euler-Maruyama）**：
  使用的是公式 (3-1) 的一阶离散格式，每一步只利用当前的梯度信息，局部截断误差为$\mathcal{O}(\Delta t)$。为了控制误差，必须走满 1000 步：
 $$
  x_{t-1} = \text{当前梯度} \times \Delta t + \text{随机噪声项}
 $$

- **EDM 采样（二阶 Heun's 方法）**：
  EDM 求解 ODE 公式 (3-3)，使用预测-校正（Predictor-Corrector）风格的 Heun 二阶方法。

  令去噪输出为$D_\theta(x_i; \sigma_i)$，导数$\dot{x}_i = \frac{x_i - D_\theta(x_i; \sigma_i)}{\sigma_i}$。

  **Step 1 (Predictor 预测)**：根据当前$\sigma_i$向$\sigma_{i+1}$外推：
 $$
  \tilde{x}_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) \cdot \dot{x}_i \tag{7-1}
 $$

  **Step 2 (Corrector 校正)**：在预测点$\tilde{x}_{i+1}$处重新计算导数$\dot{\tilde{x}}_{i+1}$，然后用梯形法则（前后两步导数的平均）进行最终更新：
 $$
  x_{i+1} = x_i + \frac{\sigma_{i+1} - \sigma_i}{2} \left( \dot{x}_i + \dot{\tilde{x}}_{i+1} \right) \tag{7-2}
 $$

- **🔍 本步对比**：
  - DDPM 只算一次梯度（一阶），走一步。
  - EDM 算**两次**梯度（一次预测，一次校正），但因其局部截断误差为$\mathcal{O}(\Delta t^2)$，允许$\Delta \sigma$取得极大。因此，**EDM 用 2 次梯度（NFE=2）走一大步，其精度等效于 DDPM 用数百步**。整体采样效率提升数十倍。

---

### 📊 终极总结表：每一步的公式核心差异

| 对比维度 | **DDPM 公式核心** | **EDM 公式核心** |
| :--- | :--- | :--- |
| **前向加噪** |$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$|$x_\sigma = x_0 + \sigma\epsilon$（更简洁，无衰减） |
| **反向方程** | 离散 SDE：$x_{t-1} = \mu_\theta(x_t) + \beta_t z$| 连续 ODE：$dx/d\sigma = (x - D_\theta)/\sigma$（确定性） |
| **网络输入** | 直接输入$x_t$（幅度剧烈波动） |$c_{\text{in}} \cdot x$（强制归一化为单位方差） |
| **网络输出** | 直接输出$\epsilon_\theta$（拟合纯噪声） |$D_\theta = c_{\text{skip}}x + c_{\text{out}}F_\theta$（自适应跳连） |
| **损失函数** |$\| \epsilon - \epsilon_\theta \|^2$（无权重） |$\lambda(\sigma) \|D_\theta - x_0\|^2$（平衡加权） |
| **步长调度** | 均匀$t$，导致中间段$\sigma$变化剧烈 | 幂律$\sigma_i$（$\rho=7$），低噪区极密 |
| **采样迭代** | 一阶 Euler 法（只用当前梯度） | 二阶 Heun 法（预测+校正，利用前后梯度） |

**解释**：虽然第2步证明了它们在“理想连续极限”下共享同一个概率路径，但 **EDM 并不是在重复造轮子**。它通过第4步（预处理）、第5步（加权）、第6步（非均匀调度）和第7步（高阶求解器），彻底改变了网络优化的难度和数值离散化的误差累积方式。**路径相同，但开车的方法和油门的灵敏度完全不同**，这就是 EDM 能成为后续所有扩散模型（如 Karras 家族）基石的根本原因。


### EDM（Elucidating Diffusion Models）的整个算法原理，**数据定义**、**网络结构**、**训练目标**、**概率流方程**到**离散采样更新**

为了保持逻辑自洽，我们定义数据标准差为$\sigma_d$（通常取$0.5$），噪声调度幂次为$\rho=7$，离散采样步索引为$i$。

---

### 📐 EDM 完整算法原理公式链（训练 + 推理全流程）

<div style="text-align: left;">

$$
\boxed{
\begin{aligned}
& \textbf{1. 前向加噪（扰动数据）}: &&
x_\sigma = x_0 + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}) \\[6pt]
& \textbf{2. 去噪网络（预处理封装）}: &&
D_\theta(x;\sigma) = \underbrace{\frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}}_{\text{跳跃连接 } c_{\text{skip}}} \cdot x
+ \underbrace{\frac{\sigma \cdot \sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}}_{\text{输出缩放 } c_{\text{out}}}
\cdot F_\theta\!\left( \underbrace{\frac{x}{\sqrt{\sigma^2 + \sigma_d^2}}}_{\text{输入归一化 } c_{\text{in}}};
\underbrace{\frac{1}{4}\ln \sigma}_{\text{噪声条件 } c_{\text{noise}}} \right) \\[6pt]
& \textbf{3. 训练损失（均衡加权）}: &&
\mathcal{L} = \mathbb{E}_{\sigma, x_0, \epsilon}\left[
\underbrace{\frac{\sigma^2 + \sigma_d^2}{(\sigma \cdot \sigma_d)^2}}_{\text{权重函数 } \lambda(\sigma)}
\cdot \left\| D_\theta(x_0 + \sigma\epsilon;\sigma) - x_0 \right\|^2
\right] \\[6pt]
& \textbf{4. 概率流常微分方程（ODE）}: &&
\frac{dx}{d\sigma} = \frac{x - D_\theta(x;\sigma)}{\sigma} \\[6pt]
& \textbf{5. 离散化采样（二阶 Heun 预测-校正）}: &&
\begin{cases}
\tilde{x}_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) \cdot \dfrac{x_i - D_\theta(x_i;\sigma_i)}{\sigma_i} & (\text{预测步}) \\[6pt]
x_{i+1} = x_i + \dfrac{\sigma_{i+1} - \sigma_i}{2} \cdot \left[ \dfrac{x_i - D_\theta(x_i;\sigma_i)}{\sigma_i} + \dfrac{\tilde{x}_{i+1} - D_\theta(\tilde{x}_{i+1};\sigma_{i+1})}{\sigma_{i+1}} \right] & (\text{校正步})
\end{cases} \\[6pt]
& \textbf{6. 噪声水平调度（幂律分布）}: &&
\sigma_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left( \sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho} \right) \right)^{\rho}, \quad i \in [0, N-1]
\end{aligned}
}
$$

</div>

---

### 🔍 公式逻辑衔接解读

这套多等号公式从**物理定义**出发，逐步过渡到**工程实现**，其内在逻辑环环相扣：

1.  **第 1 步 → 第 3 步**：用第 1 步定义的带噪数据$x_\sigma$代入第 3 步的损失函数，通过最小化$\mathcal{L}$来训练网络$F_\theta$。
2.  **第 2 步 → 第 5 步**：训练好的$F_\theta$通过第 2 步的封装，形成最终的去噪器$D_\theta$。在推理（生成）时，第 5 步的迭代公式直接调用这个$D_\theta$来更新$x$。
3.  **第 4 步 → 第 5 步**：第 4 步是连续时间的理论ODE，第 5 步是将其在离散时间轴上用二阶泰勒展开（Heun法）进行数值逼近，保证了在大步长下的低误差。
4.  **第 6 步 → 第 5 步**：第 6 步决定了第 5 步中每一步的$\sigma_{i+1}$与$\sigma_i$的间距，其幂律特性（$\rho=7$）使得在生成细节纹理的低噪声区域步长更密，在高噪声轮廓区域步长更疏。

**这份公式链完整覆盖了 EDM 从“数据扰动”到“网络设计”，再到“如何训练”和“如何高效采样”的全部数学核心。**