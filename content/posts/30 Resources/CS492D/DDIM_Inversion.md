---
title: DDIM反演（DDIM Inversion）
subtitle: ""
date: 2026-06-12T10:26:59+08:00
# lastmod: 2026-06-12T10:26:59+08:00
draft: false
authors: [Steven]
description: ""
tags: [CS492D]
categories: [CS492D]
series: [CS492D系列]
weight: 0
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
## DDIM反演（DDIM Inversion）
### 1. 背景
- **DDIM**：确定性逆向过程，可加速采样；与 DDPM（随机）不同。
- 给定初始噪声 $\mathbf{x}_T$，生成过程唯一确定。

### 2. 反演定义
- 从真实图像 $\mathbf{x}_0$ 逆向找到对应噪声 $\mathbf{x}_T$，使得从该噪声采样可重建原图。
- DDIM 确定性 → 反演可行；DDPM 随机 → 反演困难。

### 3. 核心公式
**DDIM 逆向（采样，$\sigma_t=0$）**：
$$
\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\alpha_t}}\right) + \sqrt{1-\alpha_{t-1}}\,\epsilon_\theta(\mathbf{x}_t, t)
$$

**DDIM 反演（正向加噪）**：反转上式并近似 $\epsilon_\theta(\mathbf{x}_t,t)\approx\epsilon_\theta(\mathbf{x}_{t-1},t-1)$：
$$
\mathbf{x}_{t} = \sqrt{\alpha_t}\left(\frac{\mathbf{x}_{t-1} - \sqrt{1-\alpha_{t-1}}\,\epsilon_\theta(\mathbf{x}_{t-1}, t-1)}{\sqrt{\alpha_{t-1}}}\right) + \sqrt{1-\alpha_t}\,\epsilon_\theta(\mathbf{x}_{t-1}, t-1)
$$

### 4. 算法步骤
输入 $\mathbf{x}_0$，对 $t=1$ 到 $T$：
1. $\epsilon = \epsilon_\theta(\mathbf{x}_{t-1}, t-1)$
2. $\mathbf{x}_t = \sqrt{\alpha_t}\cdot\frac{\mathbf{x}_{t-1} - \sqrt{1-\alpha_{t-1}}\,\epsilon}{\sqrt{\alpha_{t-1}}} + \sqrt{1-\alpha_t}\,\epsilon$
输出 $\mathbf{x}_T$（噪声潜码）。从 $\mathbf{x}_T$ 执行 DDIM 采样即可近似重建 $\mathbf{x}_0$。


### 5. 局限与改进
- **近似误差**：相邻步噪声预测不变假设导致重建偏差。
- **改进**：更小步长、精确可逆过程（EDICT）、空文本反演（Null-text inversion）。


## Null-Text Inversion 相关公式汇总

### 1. DDIM 确定性采样（逆向，$\sigma_t=0$）
从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t-1}$：
$$
\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(\mathbf{x}_t, t, \mathcal{P})}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_{t-1}}\;\epsilon_\theta(\mathbf{x}_t, t, \mathcal{P})
$$
记为 $\mathbf{x}_{t-1} = \Phi(\mathbf{x}_t, t, \mathcal{P})$，其中 $\mathcal{P}$ 为文本嵌入（有条件或空文本）。

### 2. DDIM 反演近似（正向，从 $\mathbf{x}_{t-1}$ 到 $\mathbf{x}_t$）
使用上一步的噪声预测近似：
$$
\mathbf{x}_t = \sqrt{\alpha_t} \left( \frac{\mathbf{x}_{t-1} - \sqrt{1-\alpha_{t-1}}\,\epsilon_\theta(\mathbf{x}_{t-1}, t-1, \mathcal{P})}{\sqrt{\alpha_{t-1}}} \right) + \sqrt{1-\alpha_t}\;\epsilon_\theta(\mathbf{x}_{t-1}, t-1, \mathcal{P})
$$
记为 $\mathbf{x}_t = \Phi^{-1}(\mathbf{x}_{t-1}, t-1, \mathcal{P})$。

### 3. Classifier-Free Guidance (CFG) 噪声预测
$$
\epsilon_\theta^s(\mathbf{x}_t, t, \mathcal{C}, \varnothing) = \epsilon_\theta(\mathbf{x}_t, t, \varnothing) + s \cdot \bigl( \epsilon_\theta(\mathbf{x}_t, t, \mathcal{C}) - \epsilon_\theta(\mathbf{x}_t, t, \varnothing) \bigr)
$$
其中 $s>1$ 为引导尺度，$\mathcal{C}$ 为条件文本嵌入，$\varnothing$ 为空文本嵌入。

### 4. 带 CFG 的 DDIM 采样一步（$\Phi_{\text{cfg}}$）
将 CFG 预测噪声代入 DDIM 采样公式：
$$
\Phi_{\text{cfg}}(\mathbf{x}_t, t, \mathcal{C}, \varnothing) = \sqrt{\alpha_{t-1}} \left( \frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\,\epsilon_\theta^s(\mathbf{x}_t, t, \mathcal{C}, \varnothing)}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_{t-1}}\;\epsilon_\theta^s(\mathbf{x}_t, t, \mathcal{C}, \varnothing)
$$

### 5. 关键点反演（Pivotal Inversion）
使用无条件模型（空文本 $\varnothing_{\text{default}}$）从真实图像 $\mathbf{x}_0^{\text{real}}$ 正向加噪：
$$
\mathbf{x}_t^{\text{piv}} = \Phi^{-1}(\mathbf{x}_{t-1}^{\text{piv}}, t-1, \varnothing_{\text{default}}), \quad \mathbf{x}_0^{\text{piv}} = \mathbf{x}_0^{\text{real}}
$$
得到轨迹 $\{\mathbf{x}_t^{\text{piv}}\}_{t=0}^T$。

### 6. Null-Text Optimization 优化目标（每时间步 $t$）
固定 $\mathbf{x}_t = \mathbf{x}_t^{\text{piv}}$，优化空文本嵌入 $\varnothing_t$ 使重建误差最小：
$$
\min_{\varnothing_t} \mathcal{L}_t(\varnothing_t) = \left\| \mathbf{x}_{t-1}^{\text{target}} - \Phi_{\text{cfg}}(\mathbf{x}_t^{\text{piv}}, t, \mathcal{C}_{\text{src}}, \varnothing_t) \right\|_2^2
$$
其中 $\mathbf{x}_{t-1}^{\text{target}}$ 通常取关键点反演的 $\mathbf{x}_{t-1}^{\text{piv}}$，$\mathcal{C}_{\text{src}}$ 为源文本嵌入。

### 7. 优化后的状态更新
得到最优 $\varnothing_t^*$ 后，计算下一步的 $\mathbf{x}_{t-1}$：
$$
\mathbf{x}_{t-1} = \Phi_{\text{cfg}}(\mathbf{x}_t^{\text{piv}}, t, \mathcal{C}_{\text{src}}, \varnothing_t^*)
$$

### 8. 编辑阶段生成（目标文本 $\mathcal{C}_{\text{tgt}}$）
从 $\mathbf{x}_T = \mathbf{x}_T^{\text{piv}}$ 开始，逆向使用优化后的空文本：
$$
\mathbf{x}_{t-1} = \Phi_{\text{cfg}}(\mathbf{x}_t, t, \mathcal{C}_{\text{tgt}}, \varnothing_t^*), \quad t = T, T-1, \dots, 1
$$

### 9. 负提示反演 (NPI) 闭式解（可选）
NPI 发现 NTI 优化得到的 $\varnothing_t^*$ 可用解析式表达：
$$
\varnothing_t^* = \lambda_t \cdot \mathcal{C}_{\text{src}} + \mu_t
$$
其中 $\lambda_t, \mu_t$ 由扩散模型参数及 $s, \alpha_t$ 决定，无需迭代优化。
