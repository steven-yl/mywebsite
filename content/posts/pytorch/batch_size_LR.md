---
title: "Pytorch Batch Size 与学习率缩放规则"
date: 2026-03-17T00:00:00+08:00
draft: false
authors: [Steven]
description: "从梯度更新期望与凸优化理论推导线性缩放与平方根缩放规则，并说明极大 batch 下线性缩放配合长 warmup 的由来与适用场景。"
summary: "详解分布式训练中 batch size 扩大时学习率的线性缩放、平方根缩放及线性+长 warmup 的推导依据与使用建议。"
tags: ["PyTorch", "优化器", "训练"]
categories: ["PyTorch"]
series: ["PyTorch实践指南"]
---

在分布式深度学习训练中，当改变 batch size 时，需要相应调整学习率以维持训练动态的稳定性与收敛效率。最常见的两种调整规则是 **线性缩放**（$\eta' = k\eta$）和 **平方根缩放**（$\eta' = \sqrt{k}\,\eta$），以及对于极大 batch 采用的 **线性缩放 + 长 warmup**。下面详细解释这些规则是如何从理论分析和实践经验中得出的。

---

## batch size和LR的调整

### 一、线性缩放规则 $\eta' = k\eta$

#### 1.1 直观动机
- **小 batch**：梯度估计噪声大，每一步的方向不太准，因此要用小学习率防止震荡。
- **大 batch**：梯度估计更准确，方向更可靠，可以用更大的步长加速收敛。
  
如果 batch 扩大 $k$ 倍，相当于一次更新看到了 $k$ 倍多的样本，因此更新的“信息量”增加了。为了使一次更新能达到原来 $k$ 步累积的效果，可以将学习率也扩大 $k$ 倍。这样，在相同的 epoch 数内，大 batch 的更新步数变少，但每一步更大，总体的参数变化量大致相当。

#### 1.2 从梯度更新期望推导
考虑标准 SGD 更新：
$$
\theta_{t+1} = \theta_t - \eta \tilde{g}_t, \quad \tilde{g}_t = \frac{1}{B}\sum_{i=1}^B \nabla L_i(\theta_t)
$$
假设梯度变化缓慢，真实梯度为 $g_t \approx \mathbb{E}[\tilde{g}_t]$。忽略噪声，一次更新后参数变化约为 $-\eta g_t$。

若 batch 扩大为 $B' = kB$，学习率设为 $\eta'$，则一次更新后参数变化约为 $-\eta' g_t$。为了使一次更新与原来 $k$ 次更新（每次学习率 $\eta$）的总效果相当，即：
$$
\eta' g_t = k \eta g_t \quad \Rightarrow \quad \eta' = k\eta
$$
这就是线性规则的由来。

#### 1.3 从凸优化理论推导
对于强凸二次损失，可以严格证明：为了在相同的 epoch 数内达到相同的次优性（suboptimality），学习率应与 batch size 成正比（Bottou, 2012）。更一般地，Goyal et al. (2017) 在论文中给出了启发式论证：当 batch 扩大 $k$ 倍时，梯度的方差减小为原来的 $1/k$，为了保持参数更新的信噪比（SNR）不变，需要将学习率扩大 $k$ 倍。

**信噪比（SNR）** 定义为：
$$
\text{SNR} = \frac{\|\mathbb{E}[\Delta \theta]\|}{\sqrt{\text{Tr}(\text{Var}(\Delta \theta))}} = \frac{\eta \|g\|}{\eta \sqrt{\text{Tr}(\Sigma)/B}} = \frac{\|g\|}{\sqrt{\text{Tr}(\Sigma)/B}}
$$
可见 SNR 与 $\eta$ 无关，但与 $B$ 有关。实际上，当我们增大 batch 时，SNR 自动增大（梯度更准）。若希望保持每次更新对损失函数的影响（即期望损失下降量）相似，需要调整 $\eta$。对于 SGD 的期望损失下降，有近似公式：
$$
\mathbb{E}[L(\theta_{t+1}) - L(\theta_t)] \approx -\eta \|\nabla L\|^2 + \frac{\eta^2}{2B} \text{Tr}(\nabla^2 L \cdot \Sigma)
$$
当 batch 扩大 $k$ 倍时，要使一阶项主导并保持下降量相当，可令 $\eta' = k\eta$，这样一阶项扩大 $k$ 倍，但二阶项（噪声项）扩大 $k^2 / (kB) = k/B$，即二阶项也扩大了 $k$ 倍（相对于原 batch 的二阶项）。因此，在初期梯度较大时，一阶项占优，线性缩放可使下降速度匹配。

在随机梯度下降（SGD）中，我们关心损失函数在每次迭代后的期望下降量。下面给出推导过程。


{{< admonition tip "SGD 的期望损失下降公式推导" false >}}

## 1. 基本设定
设损失函数 $L(\theta)$ 关于参数 $\theta$ 光滑。在第 $t$ 步，参数为 $\theta_t$，真实梯度为 $g_t = \nabla L(\theta_t)$。我们从数据集中随机抽取一个 mini-batch，计算梯度估计：
$$
\tilde{g}_t = \frac{1}{B} \sum_{i=1}^{B} \nabla L_i(\theta_t),
$$
其中 $L_i$ 是单个样本的损失，且假设各样本梯度独立同分布，满足：
$$
\mathbb{E}[\nabla L_i] = g_t, \quad \text{Cov}(\nabla L_i) = \Sigma.
$$
于是，梯度估计的均值和协方差为：
$$
\mathbb{E}[\tilde{g}_t] = g_t, \quad \text{Cov}(\tilde{g}_t) = \frac{1}{B} \Sigma.
$$

SGD 更新规则为：
$$
\theta_{t+1} = \theta_t - \eta \tilde{g}_t,
$$
其中 $\eta$ 是学习率。

---

## 2. 泰勒展开
对损失函数 $L$ 在 $\theta_t$ 处进行二阶泰勒展开：
$$
L(\theta_{t+1}) = L(\theta_t) + \nabla L(\theta_t)^\top (\theta_{t+1} - \theta_t) + \frac{1}{2} (\theta_{t+1} - \theta_t)^\top \nabla^2 L(\theta_t) (\theta_{t+1} - \theta_t) + O(\eta^3).
$$
代入 $\theta_{t+1} - \theta_t = -\eta \tilde{g}_t$，并记 $H_t = \nabla^2 L(\theta_t)$，得：
$$
L(\theta_{t+1}) - L(\theta_t) = -\eta g_t^\top \tilde{g}_t + \frac{1}{2} \eta^2 \tilde{g}_t^\top H_t \tilde{g}_t + O(\eta^3).
$$

---

## 3. 取期望
对上述等式两边取期望（关于 mini-batch 的随机性）：
$$
\mathbb{E}[L(\theta_{t+1}) - L(\theta_t)] = -\eta g_t^\top \mathbb{E}[\tilde{g}_t] + \frac{1}{2} \eta^2 \mathbb{E}[\tilde{g}_t^\top H_t \tilde{g}_t] + O(\eta^3).
$$

### 3.1 一阶项
由 $\mathbb{E}[\tilde{g}_t] = g_t$，得：
$$
-\eta g_t^\top g_t = -\eta \|g_t\|^2.
$$

### 3.2 二阶项
计算 $\mathbb{E}[\tilde{g}_t^\top H_t \tilde{g}_t]$。由于 $H_t$ 是确定性的（在给定 $\theta_t$ 下），有：
$$
\mathbb{E}[\tilde{g}_t^\top H_t \tilde{g}_t] = \mathbb{E}[\operatorname{Tr}(H_t \tilde{g}_t \tilde{g}_t^\top)] = \operatorname{Tr}\left(H_t \mathbb{E}[\tilde{g}_t \tilde{g}_t^\top]\right).
$$
而
$$
\mathbb{E}[\tilde{g}_t \tilde{g}_t^\top] = \operatorname{Cov}(\tilde{g}_t) + \mathbb{E}[\tilde{g}_t] \mathbb{E}[\tilde{g}_t]^\top = \frac{1}{B} \Sigma + g_t g_t^\top.
$$
因此，
$$
\mathbb{E}[\tilde{g}_t^\top H_t \tilde{g}_t] = \frac{1}{B} \operatorname{Tr}(H_t \Sigma) + g_t^\top H_t g_t.
$$

于是二阶项贡献为：
$$
\frac{1}{2} \eta^2 \left( g_t^\top H_t g_t + \frac{1}{B} \operatorname{Tr}(H_t \Sigma) \right).
$$

---

## 4. 合并并简化
将一阶和二阶结果合并：
$$
\mathbb{E}[L(\theta_{t+1}) - L(\theta_t)] = -\eta \|g_t\|^2 + \frac{1}{2} \eta^2 g_t^\top H_t g_t + \frac{\eta^2}{2B} \operatorname{Tr}(H_t \Sigma) + O(\eta^3).
$$

在随机梯度下降的经典分析中，常忽略确定性二阶项 $\frac{1}{2} \eta^2 g_t^\top H_t g_t$，因为它相对于一阶项是更高阶的小量（尤其当学习率较小时），且在很多情况下该项并不主导收敛行为。因此得到常用近似：
$$
\boxed{\mathbb{E}[L(\theta_{t+1}) - L(\theta_t)] \approx -\eta \|\nabla L\|^2 + \frac{\eta^2}{2B} \operatorname{Tr}(\nabla^2 L \cdot \Sigma)}.
$$

这个公式揭示了：
- 第一项是确定性梯度下降带来的损失下降，与学习率成正比。
- 第二项是 mini-batch 随机性导致的“噪声项”，它可能减缓下降甚至使损失上升（当 $\eta$ 过大时），且与 batch size $B$ 成反比，说明增大 batch 可减小噪声影响。

{{< /admonition >}}

---

### 二、平方根缩放规则 $\eta' = \sqrt{k}\,\eta$

#### 2.1 从保持更新量的方差不变推导
考虑参数更新的方差。更新量 $\Delta \theta = -\eta \tilde{g}$ 的协方差矩阵为：
$$
\text{Var}(\Delta \theta) = \eta^2 \cdot \frac{1}{B} \Sigma
$$
当 batch 扩大为 $B' = kB$，新学习率为 $\eta'$ 时，更新量的方差变为：
$$
\text{Var}(\Delta \theta') = \eta'^2 \cdot \frac{1}{kB} \Sigma
$$
如果我们希望保持更新量的方差不变（即噪声引起的波动幅度不变），则需：
$$
\eta'^2 \cdot \frac{1}{kB} \Sigma = \eta^2 \cdot \frac{1}{B} \Sigma \quad \Rightarrow \quad \frac{\eta'^2}{k} = \eta^2 \quad \Rightarrow \quad \eta' = \sqrt{k}\,\eta
$$

这种观点在 batch 极大时更为合理。因为当 batch 很大时，梯度估计的方差已经非常小，此时再线性放大学习率会导致更新量过大（因为更新量的期望与方差之比会变得很大），可能使参数跨过最优区域。保持方差不变可以避免训练不稳定。

#### 2.2 与牛顿法或自然梯度法的类比
平方根缩放也可以从二阶优化的角度理解。当 batch 极大时，梯度接近真实梯度，相当于确定性优化。此时，一步最优的步长应与 Hessian 矩阵的 Lipschitz 常数相关。如果采用一阶方法，通常学习率需要与 Lipschitz 常数的倒数成正比，而这个常数与 batch 无关。因此，不能无限放大学习率，平方根缩放是一种折中。

---

### 三、线性缩放 + 长 warmup 的由来

#### 3.1 问题背景
当 batch 极大（例如从 256 扩大到 32k）时，直接使用线性缩放（$\eta' = k\eta$）往往会导致训练初期发散或性能下降。原因在于：
- **初始阶段参数随机初始化**，梯度可能很大，直接使用大学习率容易导致梯度爆炸。
- **模型需要时间适应**，尤其是 Batch Normalization 的统计量也需要调整。
- **梯度噪声极低**，但参数初值离最优解很远，大步长可能使参数跳出合适区域。

#### 3.2 warmup 的作用
Warmup 是指在训练开始的前 $T_w$ 步或前几个 epoch 使用较小的学习率（例如从 0 线性增加到目标学习率）。它的作用是：
- 让模型在初期以温和的步伐探索参数空间，避免剧烈震荡。
- 逐渐积累梯度的稳定性，使 BN 的 running statistics 适应大学习率。
- 为后续的大学习率打下基础。

对于极大 batch，即使采用平方根缩放，也可能需要 warmup。但许多实验表明，**线性缩放配合足够长的 warmup** 可以达到与平方根缩放相当甚至更好的效果。这是因为 warmup 阶段实际上起到了“软化”线性缩放的作用：初期学习率小，相当于学习率自动遵循了某种渐进增长，最终达到 $k\eta$。只要 warmup 长度足够长，模型就能平稳过渡。

#### 3.3 如何确定 warmup 长度？
通常 warmup 的 epoch 数会随 batch 扩大而增加。例如，原始 batch 256 可能用 5 epoch warmup，当 batch 扩大到 8k 时，可能需要 100 epoch 甚至更长。一种经验法则是：warmup 步数与 batch 扩大倍数成正比，即保持 warmup 期间看到的样本总数不变（即 warmup 的 epoch 数不变？不，因为 batch 变大，每 epoch 步数变少，所以需要更多 epoch 才能看到相同数量的样本）。更常见的做法是固定 warmup 的迭代步数，例如从 5000 步增加到 20000 步。

---

### 四、总结

| 缩放方式 | 公式 | 推导依据 | 适用场景 |
|----------|------|----------|----------|
| **线性缩放** | $\eta' = k\eta$ | 保持一次更新与原来 $k$ 次更新的期望变化量相同，或保持收敛速度一致 | batch 扩大倍数适中（如 ≤32 倍），梯度噪声仍明显 |
| **平方根缩放** | $\eta' = \sqrt{k}\,\eta$ | 保持参数更新量的方差不变，避免大步长带来的不稳定 | batch 极大（如 128 倍以上），梯度噪声可忽略 |
| **线性 + 长 warmup** | 先用 warmup 再线性缩放 | 通过初期小学习率平滑过渡，使模型适应大学习率，再逐步放大 | 超大 batch 时作为线性缩放的补充，也可替代平方根缩放 |
