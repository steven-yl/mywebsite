---
title: "PyTorch 优化器原理"
date: 2026-03-17T00:00:00+08:00
draft: false
authors: [Steven]
description: "解析PyTorch 优化器原理、更新公式与偏差校正推导，并给出超参数建议与优缺点总结。"
summary: "解析PyTorch 优化器原理、更新公式与偏差校正推导，并给出超参数建议与优缺点总结。"
tags: ["PyTorch", "Adam", "optimization"]
categories: ["PyTorch"]
series: ["PyTorch 实践指南"]
---

Adam（Adaptive Moment Estimation）是一种自适应学习率优化算法，它结合了**动量法**（Momentum）和**RMSprop**的优点，通过计算梯度的一阶矩（均值）和二阶矩（未中心化的方差）来动态调整每个参数的学习率。下面详细解释其公式和原理。

---

以下是针对你列出的优化器，结合 PyTorch 接口与核心原理的完整补充说明。每个优化器均包含：使用接口（含常用默认参数）、计算公式、原理解释、优缺点、适用场景。

---

## 1. 十三类优化器接口、公式、原理、优缺点、适用场景

### 1. SGD (Stochastic Gradient Descent)
- **使用接口**  
  `torch.optim.SGD(params, lr=<required>, momentum=0, dampening=0, weight_decay=0, nesterov=False)`
  - `lr`：学习率（必须指定）
  - `momentum`：动量因子，默认 0
  - `dampening`：动量阻尼，默认 0
  - `weight_decay`：权重衰减（L2 惩罚），默认 0
  - `nesterov`：是否启用 Nesterov 动量，默认 False

- **计算公式**  
  不带动量：  
  $$
  \theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
  $$  
  带动量（标准动量）：  
  $$
  \begin{aligned}
  v_{t+1} &= \mu v_t + \nabla L(\theta_t) \\
  \theta_{t+1} &= \theta_t - \eta v_{t+1}
  \end{aligned}
  $$  
  Nesterov 动量：  
  $$
  \begin{aligned}
  \theta_{\text{ahead}} &= \theta_t - \mu v_t \\
  v_{t+1} &= \mu v_t + \nabla L(\theta_{\text{ahead}}) \\
  \theta_{t+1} &= \theta_t - \eta v_{t+1}
  \end{aligned}
  $$

- **原理解释**  
  最基本的参数更新方式，每次沿负梯度方向移动。加入动量后，累积历史梯度方向，加速收敛并抑制振荡。Nesterov 则先按历史方向“前瞻”一步，再计算梯度，提高响应灵敏度。

- **优缺点**  
  - 优点：简单、泛化能力强；配合动量/Nesterov 后收敛快且稳定。  
  - 缺点：对学习率敏感，需要仔细调参；自适应能力弱。

- **什么时候使用**  
  适用于大多数任务，尤其当数据量大、需要强泛化能力时（如 CV 分类任务）。常配合 momentum=0.9 和 Nesterov 使用。

---

### 2. ASGD (Averaged Stochastic Gradient Descent)
- **使用接口**  
  `torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)`
  - `lr`：学习率，默认 0.01
  - `lambd`：衰减项，默认 1e-4
  - `alpha`：eta 更新的指数，默认 0.75
  - `t0`：开始平均的步数，默认 1e6
  - `weight_decay`：权重衰减

- **计算公式**  
  维护两组参数：当前迭代参数 $\theta$ 和平均参数 $\bar{\theta}$。  
  每次更新：  
  $$
  \theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)
  $$  
  当 $t > t_0$ 后，平均参数按递推更新：  
  $$
  \bar{\theta}_{t+1} = \frac{t}{t+1} \bar{\theta}_t + \frac{1}{t+1} \theta_{t+1}
  $$  
  最终输出平均参数。

- **原理解释**  
  通过平均迭代历史中的参数，降低方差，使收敛点更稳定，尤其适合凸或近凸问题。

- **优缺点**  
  - 优点：对强凸问题有理论最优收敛率；解更平滑稳定。  
  - 缺点：需要存储两组参数；在非凸深度网络中优势不明显。

- **什么时候使用**  
  适用于强凸或近凸的优化问题（如某些传统机器学习模型）；深度学习中较少用，但可作为 SGD 的替代探索。

---

### 3. Adam (Adaptive Moment Estimation)
- **使用接口**  
  `torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)`
  - `lr`：学习率，默认 0.001
  - `betas`：一阶、二阶矩的衰减系数，默认 (0.9, 0.999)
  - `eps`：防止除零的小常数，默认 1e-8
  - `weight_decay`：权重衰减（L2 惩罚）
  - `amsgrad`：是否使用 AMSGrad 变体

- **计算公式**  
  $$
  \begin{aligned}
  m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
  v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
  \hat{m}_t &= m_t / (1-\beta_1^t) \\
  \hat{v}_t &= v_t / (1-\beta_2^t) \\
  \theta_{t} &= \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  \end{aligned}
  $$

- **原理解释**  
  结合动量（一阶矩）和 RMSProp（二阶矩），为每个参数自适应调整学习率。偏差校正修正初期矩估计的零偏置。

- **优缺点**  
  - 优点：收敛快、对超参数不敏感、适合稀疏梯度。  
  - 缺点：可能不收敛到最优（尤其是某些 Transformer 训练时），泛化性能有时不如 SGD。

- **什么时候使用**  
  默认首选之一，尤其适合 NLP、强化学习、生成模型等任务。若追求泛化，可尝试 SGD 或 AdamW。

---

### 4. AdamW (Adam with Decoupled Weight Decay)
- **使用接口**  
  `torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)`
  - 参数含义同 Adam，但 `weight_decay` 默认常设为 0.01，且实现上与 Adam 不同（解耦）

- **计算公式**  
  先计算 Adam 更新量（不含 weight decay），然后单独施加衰减：  
  $$
  \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
  $$  
  其中 $\lambda$ 为 `weight_decay`。

- **原理解释**  
  将权重衰减从梯度计算中解耦，使衰减不受自适应学习率缩放，更符合原始权重衰减的意图，提升泛化。

- **优缺点**  
  - 优点：比传统 Adam 泛化更好，尤其适合 Transformer 架构；超参数调整更直观。  
  - 缺点：仍可能比 SGD 稍差（但通常优于 Adam）。

- **什么时候使用**  
  NLP 任务（如 BERT、GPT）和 Vision Transformer 的默认优化器；推荐作为 Adam 的替代。

---

### 5. Adamax (Adam based on Infinity Norm)
- **使用接口**  
  `torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)`
  - `lr`：学习率，默认 0.002（比 Adam 稍大）
  - `betas`：一阶、无穷范数衰减系数，默认 (0.9, 0.999)
  - 其余同 Adam

- **计算公式**  
  一阶矩同 Adam，二阶矩改用无穷范数：  
  $$
  \begin{aligned}
  m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
  u_t &= \max(\beta_2 u_{t-1}, |g_t| + \epsilon) \\
  \theta_t &= \theta_{t-1} - \eta \frac{m_t}{u_t}
  \end{aligned}
  $$  
  注意：这里 $u_t$ 是梯度绝对值的加权最大值，而非平方平均。

- **原理解释**  
  用梯度绝对值的指数衰减最大值替代平方平均，使学习率对梯度尺度更鲁棒，适合某些噪声分布。

- **优缺点**  
  - 优点：在梯度变化剧烈时更稳定；对学习率选择更鲁棒。  
  - 缺点：不如 Adam 常用，理论分析较少。

- **什么时候使用**  
  在个别任务（如某些 NLP 或嵌入学习）中可尝试，当 Adam 不稳定时可作为备选。

---

### 6. NAdam (Adam with Nesterov Momentum)
- **使用接口**  
  `torch.optim.NAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=0.004)`
  - `lr`、`betas`、`eps`、`weight_decay` 同 Adam
  - `momentum_decay`：动量衰减系数，默认 0.004，用于 Nesterov 修正

- **计算公式**  
  在 Adam 基础上引入 Nesterov 前瞻思想，修改一阶矩的更新方式（具体公式较复杂，可参考原论文或源码）。

- **原理解释**  
  结合 Adam 的自适应学习率和 Nesterov 动量的前瞻梯度，有时能进一步加速收敛，尤其在训练初期无需 warmup。

- **优缺点**  
  - 优点：收敛更快，对学习率更不敏感；无需 warmup 阶段。  
  - 缺点：可能在某些任务上不如 Adam 稳定。

- **什么时候使用**  
  可替代 Adam，尤其当希望快速收敛且不想做 warmup 时。

---

### 7. RAdam (Rectified Adam)
- **使用接口**  
  `torch.optim.RAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)`
  - 参数同 Adam，无额外超参数

- **计算公式**  
  动态修正 Adam 早期的二阶矩估计方差，在训练初期使用近似 SGD 的更新，后期平滑过渡到 Adam。

- **原理解释**  
  针对 Adam 在训练初期因二阶矩估计不准导致方差过大而提出的修正，通过整流项使更新更稳定，避免需要 warmup。

- **优缺点**  
  - 优点：训练初期更稳定，无需 warmup；收敛效果与 Adam 相当。  
  - 缺点：计算稍复杂，但影响不大。

- **什么时候使用**  
  当使用 Adam 需要 warmup 时，RAdam 可直接替代，省去调 warmup 步骤。

---

### 8. SparseAdam
- **使用接口**  
  `torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)`
  - 参数同 Adam，但专为稀疏梯度设计（如嵌入层）

- **计算公式**  
  与 Adam 类似，但只更新有非零梯度的参数部分，且二阶矩的存储方式针对稀疏性优化。

- **原理解释**  
  针对嵌入层等高维稀疏特征，避免为所有零梯度参数更新动量，节省内存和计算。

- **优缺点**  
  - 优点：内存占用少，计算快，适合大规模稀疏特征。  
  - 缺点：仅适用于稀疏梯度场景，普通参数无法使用。

- **什么时候使用**  
  模型中包含嵌入层且特征极度稀疏时（如推荐系统、NLP 词嵌入），替代 Adam。

---

### 9. Rprop (Resilient Propagation)
- **使用接口**  
  `torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50))`
  - `lr`：学习率（初始步长），默认 0.01
  - `etas`：步长增减因子 (etaplus, etaminus)，默认 (0.5, 1.2)
  - `step_sizes`：步长上下界 (min_step, max_step)

- **计算公式**  
  根据梯度符号变化调整每个参数的步长：  
  若连续两次梯度符号相同，则步长乘以 `etaplus`；若相反，则步长乘以 `etaminus`；梯度符号改变时，跳过本次更新。  
  更新公式：  
  $$
  \theta_{t+1} = \theta_t - \text{sign}(g_t) \cdot \Delta_t
  $$  
  其中 $\Delta_t$ 为自适应步长。

- **原理解释**  
  仅利用梯度符号，忽略梯度大小，步长自适应调整。适合全批量（batch）学习，对梯度噪声不敏感。

- **优缺点**  
  - 优点：无需设置学习率（但需设初始步长）；对超参数鲁棒。  
  - 缺点：仅适用于全批量（不能用于 mini-batch），否则梯度符号不稳定。

- **什么时候使用**  
  小规模全批量优化（如传统机器学习、小数据集），深度学习几乎不用。

---

### 10. RMSprop
- **使用接口**  
  `torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)`
  - `lr`：学习率，默认 0.01
  - `alpha`：梯度平方移动平均的衰减系数，默认 0.99
  - `eps`：防止除零
  - `momentum`：可选动量项
  - `centered`：是否使用中心化二阶矩（减去均值）

- **计算公式**  
  $$
  \begin{aligned}
  v_t &= \alpha v_{t-1} + (1-\alpha) g_t^2 \\
  \theta_t &= \theta_{t-1} - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}
  \end{aligned}
  $$  
  若启用动量，则对除以缩放因子后的梯度再应用动量。

- **原理解释**  
  通过梯度平方的移动平均调整学习率，解决 Adagrad 学习率单调下降的问题，适合非平稳目标。

- **优缺点**  
  - 优点：自适应学习率，适合 RNN 等时间序列任务。  
  - 缺点：可能在某些任务上不如 Adam 稳定。

- **什么时候使用**  
  常用于 RNN、强化学习（如 DQN）等需要处理非平稳性的场景。

---

### 11. Adadelta
- **使用接口**  
  `torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)`
  - `lr`：学习率（其实已无学习率概念，但保留作为初始系数），默认 1.0
  - `rho`：梯度平方与更新量平方的衰减系数，默认 0.9
  - `eps`：防止除零

- **计算公式**  
  维护两个移动平均：  
  $$
  \begin{aligned}
  E[g^2]_t &= \rho E[g^2]_{t-1} + (1-\rho) g_t^2 \\
  \Delta \theta_t &= - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t \\
  E[\Delta \theta^2]_t &= \rho E[\Delta \theta^2]_{t-1} + (1-\rho) (\Delta \theta_t)^2 \\
  \theta_{t+1} &= \theta_t + \Delta \theta_t
  \end{aligned}
  $$  
  其中 $E[\Delta \theta^2]$ 是参数更新量的平方平均。

- **原理解释**  
  不仅用梯度平方调整步长，还引入更新量的平方平均，使步长单位与参数单位匹配，理论上无需学习率。

- **优缺点**  
  - 优点：无需设置学习率（但需初始系数）；对超参数鲁棒。  
  - 缺点：实践中不如 Adam 常用，收敛速度可能较慢。

- **什么时候使用**  
  当希望避免调学习率时，可作为替代方案；在有些任务上表现良好。

---

### 12. Adagrad
- **使用接口**  
  `torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)`
  - `lr`：学习率，默认 0.01
  - `lr_decay`：学习率衰减因子
  - `initial_accumulator_value`：梯度平方累积初始值
  - `eps`：防止除零

- **计算公式**  
  $$
  \begin{aligned}
  G_t &= G_{t-1} + g_t^2 \quad (\text{逐元素累加}) \\
  \theta_t &= \theta_{t-1} - \eta \frac{g_t}{\sqrt{G_t} + \epsilon}
  \end{aligned}
  $$

- **原理解释**  
  对每个参数累积历史梯度平方，使稀疏特征获得更大更新，频繁特征更新变小。

- **优缺点**  
  - 优点：适合稀疏数据；无需手动调整学习率。  
  - 缺点：累积平方和单调增加，学习率会趋近于零，最终停止学习。

- **什么时候使用**  
  处理稀疏特征（如大规模词嵌入、逻辑回归）时效果很好；深度学习中已被 Adam/RMSprop 取代。

---

### 13. LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
- **使用接口**  
  `torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None)`
  - `lr`：学习率，默认 1.0（实际为步长缩放因子）
  - `max_iter`：每次优化步的最大迭代次数
  - `history_size`：保存的历史梯度/步长数量
  - `line_search_fn`：线搜索方法（如 'strong_wolfe'）

- **计算公式**  
  属于拟牛顿法，利用历史梯度差和参数差近似 Hessian 矩阵的逆，然后计算更新方向，并通过线搜索确定步长。具体公式较复杂，涉及 BFGS 更新。

- **原理解释**  
  通过近似二阶导数信息，实现快速收敛，尤其适合小批量确定性优化（即每次用全部数据计算梯度）。

- **优缺点**  
  - 优点：收敛速度快（二阶信息）；对学习率不敏感。  
  - 缺点：需要多次前向/后向（closure 函数），计算量大；不适合 mini-batch 随机优化。

- **什么时候使用**  
  小规模全批量优化（如小数据集、传统机器学习），或需要精确求解时（如某些生成模型的训练）。在深度学习中很少用。





## 2.Adam的原理

### 1.Adam 更新公式

设 $ g_t $ 为第 $ t $ 步的梯度（关于目标函数的梯度），Adam 维护两个状态变量：

- **一阶动量** $ m_t $：梯度的指数移动平均（即 Momentum）
- **二阶动量** $ v_t $：梯度平方的指数移动平均（即 RMSprop 中的均方根）

具体计算步骤如下：

1. **更新有偏一阶矩估计**  
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   其中 $ \beta_1 $ 通常取 0.9，控制历史梯度的衰减率。

2. **更新有偏二阶矩估计**  
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   其中 $ \beta_2 $ 通常取 0.999，$ g_t^2 $ 表示逐元素的平方。

3. **偏差校正**  
   由于 $ m_0, v_0 $ 初始化为 0，初期估计会偏向于 0，因此需要校正：
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad
   \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$
   这里 $ \beta_1^t, \beta_2^t $ 是 $ \beta $ 的 $ t $ 次幂，随着 $ t $ 增大分母趋近于 1。

4. **参数更新**  
   $$
   \theta_{t} = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$
   其中 $ \alpha $ 是学习率（通常默认 0.001），$ \epsilon $ 是一个极小常数（如 $10^{-8}$）防止除零。

---

### 2.公式含义解读

- **一阶矩 $ m_t $**：相当于带衰减的梯度累积，捕捉梯度的平均方向。如果梯度方向一致，$ m_t $ 会增大，从而加速收敛。
- **二阶矩 $ v_t $**：记录了梯度幅度的方差。对于梯度较大的参数，$ v_t $ 较大，导致学习率 $ \frac{\alpha}{\sqrt{\hat{v}_t}} $ 变小，避免振荡；对于梯度较小的参数，学习率变大，加速更新。
- **偏差校正**：在训练初期，$ m_t $ 和 $ v_t $ 被初始化为 0，导致估计值偏向于 0。校正后能更快进入正确估计。

   {{< admonition tip "偏差校正原理推导" false >}}
   要推导 $ m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i $，我们从 Adam 中一阶矩的递推定义开始：

   $$
   m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad \text{初始 } m_0 = 0
   $$

   ---

   ### 推导过程（反复代入法）

   1. **写出 $ m_t $ 的表达式**  
      $$
      m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
      $$

   2. **代入 $ m_{t-1} $**  
      $$
      m_{t-1} = \beta_1 m_{t-2} + (1-\beta_1) g_{t-1}
      $$
      因此：
      $$
      m_t = \beta_1 \big( \beta_1 m_{t-2} + (1-\beta_1) g_{t-1} \big) + (1-\beta_1) g_t
      $$
      $$
      m_t = \beta_1^2 m_{t-2} + \beta_1(1-\beta_1) g_{t-1} + (1-\beta_1) g_t
      $$

   3. **继续代入 $ m_{t-2} $**  
      $$
      m_{t-2} = \beta_1 m_{t-3} + (1-\beta_1) g_{t-2}
      $$
      $$
      m_t = \beta_1^2 \big( \beta_1 m_{t-3} + (1-\beta_1) g_{t-2} \big) + \beta_1(1-\beta_1) g_{t-1} + (1-\beta_1) g_t
      $$
      $$
      m_t = \beta_1^3 m_{t-3} + \beta_1^2 (1-\beta_1) g_{t-2} + \beta_1(1-\beta_1) g_{t-1} + (1-\beta_1) g_t
      $$

   4. **重复此过程，直到 $ m_0 $**  
      经过 $ t $ 次代入后，$ m_0 = 0 $ 出现：
      $$
      m_t = \beta_1^t m_0 + (1-\beta_1) \big( g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2} + \cdots + \beta_1^{t-1} g_1 \big)
      $$
      由于 $ m_0 = 0 $，第一项消失：
      $$
      m_t = (1-\beta_1) \big( g_t + \beta_1 g_{t-1} + \beta_1^2 g_{t-2} + \cdots + \beta_1^{t-1} g_1 \big)
      $$

   5. **用求和符号表示**  
      $$
      m_t = (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} g_i
      $$
      这里 $ i $ 从 1 到 $ t $，当 $ i = t $ 时，$ \beta_1^{t-t} = \beta_1^0 = 1 $，系数为 $ (1-\beta_1) g_t $，与上述展开一致。

   ---

   ### 权重之和

   注意所有项的权重之和为：
   $$
   (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} = (1-\beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = 1 - \beta_1^t
   $$
   这正好小于 1，解释了为什么需要偏差校正：因为 $ m_t $ 是真实梯度期望的 $ 1-\beta_1^t $ 倍。

   {{< /admonition >}}

- **学习率缩放**：最终更新步长为 $ \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $，实现了对每个参数独立调整学习率。

---

### 3.超参数说明

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| $ \alpha $ | 0.001 | 全局学习率 |
| $ \beta_1 $ | 0.9 | 一阶矩的指数衰减率，控制动量影响 |
| $ \beta_2 $ | 0.999 | 二阶矩的指数衰减率，控制梯度平方的影响范围 |
| $ \epsilon $ | $10^{-8}$ | 数值稳定项，避免分母为零 |

---

### 4.优点与局限

**优点**  
- 自适应学习率，适合处理稀疏梯度（如 NLP、图像 caption 任务）。  
- 超参数鲁棒，通常无需精细调参就能获得不错效果。  
- 计算高效，内存需求小（只需存储 $ m_t, v_t $）。

**局限**  
- 在某些情况下（如 Transformer 训练）可能不收敛到最优，需要使用 **AdamW**（修正了权重衰减实现方式）或 **Adam with decoupled weight decay**。  
- 二阶矩估计可能受到极端梯度的影响，导致学习率骤降。

---

### 5.与其他优化器的关系

- **SGD + Momentum**：只使用一阶矩，没有自适应学习率。  
- **RMSprop**：只使用二阶矩，没有动量。  
- **Adam**：两者结合，是目前最流行的优化器之一。



## 3.weight_decay原理

权重衰减（Weight Decay）是深度学习中最常用的正则化技术之一，它的作用是**防止模型过拟合**，通过在每次参数更新时，以一定比例减小参数的数值，从而限制模型的复杂度。

---

### 1. 权重衰减的基本思想

在标准的梯度下降更新中，参数更新方向仅由损失函数的梯度决定：
$$
\theta \leftarrow \theta - \eta \nabla L(\theta)
$$
权重衰减在更新时额外减去一小部分当前参数值：
$$
\theta \leftarrow \theta - \eta \nabla L(\theta) - \eta \lambda \theta
$$
其中 $\eta$ 是学习率，$\lambda > 0$ 是权重衰减系数。将后两项合并：
$$
\theta \leftarrow (1 - \eta \lambda) \theta - \eta \nabla L(\theta)
$$
可以看到，参数在更新前先被缩小了 $(1 - \eta \lambda)$ 倍，这就是“衰减”名称的由来。

---

### 2. 与 L2 正则化的关系（SGD 情况）

在经典的随机梯度下降（SGD）中，权重衰减等价于在损失函数中加入 L2 正则化项。

**L2 正则化**：在原始损失函数后加上参数的平方和：
$$
\tilde{L}(\theta) = L(\theta) + \frac{\lambda}{2} \|\theta\|^2
$$
对这个新损失函数求梯度：
$$
\nabla \tilde{L}(\theta) = \nabla L(\theta) + \lambda \theta
$$
使用梯度下降更新：
$$
\theta \leftarrow \theta - \eta \left( \nabla L(\theta) + \lambda \theta \right) = (1 - \eta \lambda) \theta - \eta \nabla L(\theta)
$$
这与直接权重衰减的公式完全一致。因此，在 SGD 中，权重衰减和 L2 正则化是等价的。

---

### 3. 自适应优化器中的问题与 AdamW 的解耦

对于自适应优化器（如 Adam、RMSprop），情况变得复杂。这类优化器会为每个参数独立调整学习率，如果直接将 L2 正则化项加到梯度上，那么正则化项也会被自适应地缩放，导致实际衰减效果与 $\lambda$ 不成比例，且难以调节。

**Adam 的传统实现**（带 weight_decay 参数）：
$$
g_t = \nabla L(\theta_{t-1}) + \lambda \theta_{t-1}
$$
然后使用 $g_t$ 计算动量、自适应学习率。这样权重衰减项参与了动量和二阶矩的统计，导致衰减效果不稳定。

**AdamW 的改进**：将权重衰减与梯度更新解耦，直接在参数更新后施加衰减：
$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
$$
此时权重衰减项 **不参与动量计算**，也不被自适应学习率缩放，保持了恒定的衰减强度，效果更可控，也更接近传统 SGD 的权重衰减。实验表明，AdamW 能显著提升模型（尤其是 Transformer）的泛化能力。

---

### 4. 权重衰减的作用总结

- **防止过拟合**：通过约束参数范数，限制模型复杂度，降低对训练数据噪声的拟合。
- **提高泛化能力**：使模型在未见数据上表现更好。
- **数值稳定性**：对参数进行软性收缩，有助于避免参数过大导致的梯度爆炸。
- **超参数 λ**：控制正则化强度，λ 越大，参数衰减越快，模型越简单。通常通过交叉验证选择。

---

### 5. 公式总结

| 优化器 | 权重衰减实现方式 | 更新公式（简化） |
|--------|------------------|------------------|
| SGD（L2） | 通过梯度隐含 | $\theta \leftarrow \theta - \eta \nabla L(\theta) - \eta \lambda \theta$ |
| Adam（传统） | 梯度中加入 λθ | $g_t = \nabla L + \lambda \theta$，然后进行 Adam 更新 |
| AdamW | 解耦更新后直接衰减 | $\theta \leftarrow \theta - \eta \cdot \text{AdamUpdate} - \eta \lambda \theta$ |

其中 $\eta$ 为学习率，$\lambda$ 为权重衰减系数。




## 4.标准动量法和Nesterov方法
Nesterov 加速梯度（Nesterov Accelerated Gradient, NAG）是梯度下降算法的一种改进，其核心思想是**在计算梯度时引入“前瞻”机制，让优化器不仅依赖当前位置的梯度，还能提前看到如果按当前动量继续前进后可能的位置，从而更准确地调整更新方向**。

---

### 1. 直观理解：从“盲目跟随”到“预见性”

- **标准动量法**：想象一个小球滚下山坡，它根据当前所在位置的坡度（梯度）决定下一步速度，然后滚动到新位置。它不知道自己将要滚向哪里，只是被动地响应当前位置的力。
- **Nesterov 方法**：小球不仅看当前脚下的坡度，还“踮起脚尖”向前看一步：它先按照之前的动量往前走一小段，看看那个地方的坡度如何，然后再根据那个“未来位置”的坡度来调整方向。这样能提前感知到地形变化，避免冲过头，尤其在临近谷底时能更平稳地减速。

---

### 2. 数学对比（以标准动量 vs Nesterov）

记 $ \theta_t $ 为参数，$ v_t $ 为动量（速度），$ \eta $ 为学习率，$ \mu $ 为动量系数（通常 0.9）。

#### 标准动量（Momentum）
$$
\begin{aligned}
v_t &= \mu v_{t-1} + \eta \nabla L(\theta_{t-1}) \\
\theta_t &= \theta_{t-1} - v_t
\end{aligned}
$$
先计算当前位置的梯度，再更新速度，最后用新速度更新参数。

#### Nesterov 加速梯度（NAG）
$$
\begin{aligned}
v_t &= \mu v_{t-1} + \eta \nabla L(\theta_{t-1} - \mu v_{t-1}) \\
\theta_t &= \theta_{t-1} - v_t
\end{aligned}
$$
关键区别：梯度不是在当前 $\theta_{t-1}$ 处计算的，而是在 **“前瞻点”** $\theta_{t-1} - \mu v_{t-1}$（即如果按之前动量走一步会到达的位置）处计算的。

---

### 3. 为什么前瞻有效？

- **减小振荡**：当参数接近最优区域时，梯度方向可能来回变化。Nesterov 能提前看到前方梯度方向，从而及时刹车或转向，避免像标准动量那样冲过头再折返。
- **理论加速**：对于凸优化问题，Nesterov 方法能达到最优的收敛速度 $O(1/t^2)$，而普通梯度下降只有 $O(1/t)$。虽然在深度学习中非凸问题没有严格保证，但实际中 NAG 往往比标准动量更稳定，有时收敛更快。

---

### 4. 等价形式与实现

实践中，Nesterov 常被重写为一种更方便实现的形式（例如在 PyTorch 的 SGD 中，通过设置 `nesterov=True` 启用）：

$$
\begin{aligned}
\theta_{\text{ahead}} &= \theta_{t-1} - \mu v_{t-1} \\
g &= \nabla L(\theta_{\text{ahead}}) \\
v_t &= \mu v_{t-1} + \eta g \\
\theta_t &= \theta_{t-1} - v_t
\end{aligned}
$$

这种形式清晰地展示了“先跳一步，再计算梯度，然后更新速度”的过程。

---

### 5. 总结

Nesterov 的核心思想是**让优化器具有“预见性”**：通过在当前动量基础上预先迈出一小步，在那个未来位置评估梯度，从而让更新方向更符合前方的地形，既保留了动量加速的优势，又提升了稳定性和理论收敛率。


