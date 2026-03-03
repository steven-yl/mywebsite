---
title: "Minimal-Drifting-Models"
subtitle: "Drifting Models 的 2D 玩具实现与扩展（Mean-Shift / Sinkhorn / CF / Autoencoder）"
date: 2026-03-03T12:00:00+08:00
draft: false
authors: [Steven]

tags: [diffusion/flow, drifting-model, papers]
categories: [diffusion/flow, papers]
series: [diffusion/flow系列]
weight: 4
series_weight: 4

hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""

summary: "基于 [Algomancer/Minimal-Drifting-Models](https://github.com/Algomancer/Minimal-Drifting-Models) 与论文 Generative Modeling via Drifting (Deng et al., 2026)，详解漂移场 V、训练损失、推理 1-NFE，以及 Sinkhorn / 特征函数 / 自编码器扩展。"
---

# Minimal-Drifting-Models 算法与实现说明

本文对 GitHub 项目 [Algomancer/Minimal-Drifting-Models](https://github.com/Algomancer/Minimal-Drifting-Models) 做**算法与实现**层面的说明。该项目是论文 [*Generative Modeling via Drifting*](https://arxiv.org/abs/2602.04770)（Deng et al., ICML 2026）的**极简 2D 实现**：在训练时演化 pushforward 分布、推理时**单步前向（1-NFE）**生成，无需 score、ODE 求解器或噪声调度。

---

## 目录

- [Minimal-Drifting-Models 算法与实现说明](#minimal-drifting-models-算法与实现说明)
  - [目录](#目录)
  - [1. 项目与论文背景](#1-项目与论文背景)
  - [2. 核心思想：训练时演化，推理时一步](#2-核心思想训练时演化推理时一步)
  - [3. 漂移场 V 与 Mean-Shift 形式](#3-漂移场-v-与-mean-shift-形式)
  - [4. 训练损失（Algorithm 1）](#4-训练损失algorithm-1)
  - [5. 代码结构概览](#5-代码结构概览)
  - [6. drifting.py：基础实现](#6-driftingpy基础实现)
    - [6.1 数据](#61-数据)
    - [6.2 网络](#62-网络)
    - [6.3 漂移场 `compute_drift(x, y_pos, y_neg, temp)`](#63-漂移场-compute_driftx-y_pos-y_neg-temp)
    - [6.4 损失 `drifting_loss(gen, pos, temp)`](#64-损失-drifting_lossgen-pos-temp)
    - [6.5 训练与生成](#65-训练与生成)
  - [7. sinkhorn\_drifting.py：双随机归一化](#7-sinkhorn_driftingpy双随机归一化)
  - [8. cf\_drifting.py：特征函数漂移](#8-cf_driftingpy特征函数漂移)
    - [8.1 CF 距离](#81-cf-距离)
    - [8.2 漂移 (V(x\_k))](#82-漂移-vx_k)
    - [8.3 复杂度](#83-复杂度)
  - [9. drifting\_autoencoder.py：漂移自编码器](#9-drifting_autoencoderpy漂移自编码器)
  - [10. 运行与可视化](#10-运行与可视化)
  - [11. 小结与延伸阅读](#11-小结与延伸阅读)
    - [小结](#小结)
    - [延伸阅读](#延伸阅读)

---

## 1. 项目与论文背景

- **论文**：[*Generative Modeling via Drifting*](https://arxiv.org/abs/2602.04770)（Deng et al., ICML 2026）。  
- **仓库**：[Algomancer/Minimal-Drifting-Models](https://github.com/Algomancer/Minimal-Drifting-Models)，使用 2D 玩具数据（8 Gaussians、Checkerboard）做最小可复现实现。  
- **特点**：与 diffusion/flow 不同，**迭代发生在训练阶段**；推理时生成器一次前向即可（1-NFE），漂移场 \(V\) 在均衡时趋于 0（生成分布与数据分布一致）。

---

## 2. 核心思想：训练时演化，推理时一步

| 对比项 | Diffusion / Flow | Drifting Models |
| :--- | :--- | :--- |
| **迭代发生时机** | 推理时多步采样 | 训练时分布演化 |
| **推理** | 多步 NFE（ODE/SDE 或离散步） | **单步前向 1-NFE** |
| **训练目标** | score / flow 匹配等 | 漂移目标：\(f(\varepsilon) \to f(\varepsilon) + V(f(\varepsilon))\) |
| **漂移场 \(V\)** | — | 指向数据、排斥生成样本；\(V \to 0\) 即均衡 |

直观上：**漂移场 \(V\)** 告诉每个生成样本「该往哪走」才能更接近数据分布；用 \(V\) 构造**带 stop-gradient 的目标**，让生成器 \(f\) 去拟合「当前输出 + \(V\)」，从而在训练过程中把 pushforward 分布一步步推向数据分布。收敛后 \(V \approx 0\)，推理时只需 \(z \sim \mathcal{N}(0,I),\ x = f(z)\)。

---

## 3. 漂移场 V 与 Mean-Shift 形式

论文中漂移场（Algorithm 2）的**紧凑形式**（mean-shift 核）：

$$
V(x) = \frac{1}{Z_p Z_q} \mathbb{E}_{y^+ \sim p,\, y^- \sim q}\big[ k(x,y^+) k(x,y^-) (y^+ - y^-) \big]
$$

- \(p\)：数据分布（正样本 \(y^+\)）  
- \(q\)：当前生成分布（负样本 \(y^-\)，通常取为当前 batch 的生成点）  
- \(k(x,y)\)：核，实现中为 \(k(x,y) = \exp(-\Vert x-y\Vert/\tau)\)（或与距离相关的 logits），\(\tau\) 为温度 `temp`。

含义：\(V(x)\) 是「被拉向数据 \(y^+\)、被推离生成 \(y^-\)」的加权平均方向；权重由 \(k(x,y^+)\) 与 \(k(x,y^-)\) 共同决定。  
实现时用 **doubly-normalized affinity**（对 x 和 y 两维做 softmax 后取几何平均）或 **Sinkhorn 双随机归一化**，再分解成 \(W_{\mathrm{pos}} y^+ - W_{\mathrm{neg}} y^-\)，避免显式 \(O(N\times N_{\mathrm{pos}}\times N_{\mathrm{neg}})\)。

---

## 4. 训练损失（Algorithm 1）

损失为**带 stop-gradient 的 MSE**：

$$
\mathcal{L} = \mathbb{E}\Big[ \big\Vert f(\varepsilon) - \mathrm{sg}\big(f(\varepsilon) + V(f(\varepsilon))\big) \big\Vert^2 \Big]
$$

- \(\varepsilon \sim \mathcal{N}(0,I)\)，\(f\) 为生成器。  
- \(\mathrm{sg}\) 表示 stop-gradient：梯度只通过 \(f(\varepsilon)\)，不通过 \(f(\varepsilon)+V\)。  
- 目标 = 当前生成点 + 漂移量；损失值等价于 \(\Vert V\Vert^2\)。

这样，优化器是在「把 \(f(\varepsilon)\) 朝 \(f(\varepsilon)+V\) 移动」，即沿 \(V\) 的方向更新生成分布。

---

## 5. 代码结构概览

| 文件 | 作用 |
| :--- | :--- |
| `drifting.py` | 基础 Drifting Model：mean-shift 核 + 几何平均双归一化 affinity，8 Gaussians / Checkerboard 训练与漂移场可视化。 |
| `sinkhorn_drifting.py` | 用 **Sinkhorn**（log 域 5 轮）得到**双随机** affinity，替代几何平均 softmax。 |
| `cf_drifting.py` | **特征函数（CF）距离**驱动的漂移：\(V\) 取为 CF 距离的泛函梯度，复杂度 \(O(N\cdot F)\)，\(F\) 为频率向量数。 |
| `drifting_autoencoder.py` | **漂移自编码器**：编码器 data→latent 向先验漂移，解码器 latent→data 向数据漂移；编码/解码均为 1-NFE。 |

---

## 6. drifting.py：基础实现

### 6.1 数据

- `gen_data(n)`：2D **8 Gaussians**（圆周排列）。  
- `gen_checkerboard(n)`：2D **Checkerboard**（4 块）。

### 6.2 网络

- `Net`：MLP，noise_dim → hidden_dim（4 层 SELU）→ 2；输入为噪声 \(z\)，输出为 2D 样本。

### 6.3 漂移场 `compute_drift(x, y_pos, y_neg, temp)`

- `x`：查询点（当前生成）`[N, D]`。  
- `y_pos`：数据 `[N_pos, D]`。  
- `y_neg`：负样本（通常 = 当前生成 `x`）`[N_neg, D]`。  
- 用 `torch.cdist` 算 `dist_pos`、`dist_neg`，当 \(y_{\mathrm{neg}}=x\) 时屏蔽自交互（`dist_neg` 对角线加一大数）。  
- logits = `-dist / temp`，在最后一维做 softmax 得 \(A_{\mathrm{row}}\)，在倒数第二维做 softmax 得 \(A_{\mathrm{col}}\)，\(A = \sqrt{A_{\mathrm{row}} \cdot A_{\mathrm{col}}}\)。  
- 拆成 \(A_{\mathrm{pos}}\)、\(A_{\mathrm{neg}}\)，因子化权重：
  - \(W_{\mathrm{pos}} = A_{\mathrm{pos}} \cdot \sum_k A_{\mathrm{neg}}[i,k]\)
  - \(W_{\mathrm{neg}} = A_{\mathrm{neg}} \cdot \sum_j A_{\mathrm{pos}}[i,j]\)
- \(V = W_{\mathrm{pos}} @ y_{\mathrm{pos}} - W_{\mathrm{neg}} @ y_{\mathrm{neg}}\)。

### 6.4 损失 `drifting_loss(gen, pos, temp)`

- `gen`：当前生成样本（保留梯度）。  
- 在 `torch.no_grad()` 下算 \(V = \mathrm{compute\_drift}(\mathrm{gen}, \mathrm{pos}, \mathrm{gen}, \mathrm{temp})\)。  
- target = `(gen + V).detach()`，loss = `(gen - target).pow(2).sum(dim=-1)`（每样本 \(\Vert V\Vert^2\)）。

### 6.5 训练与生成

- `DriftingModel.forward(pos, n_gen)`：采样 `n_gen` 个 \(z\)，生成 `gen`，返回 `drifting_loss(gen, pos, temp)`。  
- `DriftingModel.generate(n)`：`torch.no_grad()` 下 \(z \sim \mathcal{N}(0,I)\)，\(x = \mathrm{net}(z)\)，即 1-NFE。

---

## 7. sinkhorn_drifting.py：双随机归一化

与 `drifting.py` 的差异仅在 **affinity 的归一化方式**：

- 同一套 logits（`-dist_pos/temp` 与 `-dist_neg/temp` 拼接）。  
- 不用「行 softmax × 列 softmax 的几何平均」，而用 **Sinkhorn**：在 log 域交替按行、按列做 logsumexp 归一化（代码中 5 轮），再 `exp` 得到**双随机矩阵** \(A\)（行和、列和均为 1）。  
- 之后同样拆成 \(A_{\mathrm{pos}}/A_{\mathrm{neg}}\)、因子化权重、\(V = W_{\mathrm{pos}}@y_{\mathrm{pos}} - W_{\mathrm{neg}}@y_{\mathrm{neg}}\)。

双随机约束比几何平均更严格，有时对分布匹配更稳定。

---

## 8. cf_drifting.py：特征函数漂移

这里**不再用 mean-shift 核**，而是用**特征函数（CF）距离**的泛函梯度作为漂移方向。

### 8.1 CF 距离

- 采样 \(F\) 个频率向量 \(f \sim \mathcal{N}(0, \sigma^2 I)\)。  
- 经验 CF：\(\phi_\mu(f) = \frac{1}{N}\sum_j e^{i f^\top x_j}\)，实部 \(C_\mu(f)=\frac{1}{N}\sum_j \cos(f^\top x_j)\)，虚部 \(S_\mu(f)=\frac{1}{N}\sum_j \sin(f^\top x_j)\)。  
- 距离：\(D = \frac{1}{F}\sum_f |\phi_\mu(f) - \phi_\nu(f)|^2\)（\(\mu\)=生成，\(\nu\)=数据）。

### 8.2 漂移 \(V(x_k)\)

对样本 \(x_k\) 求 \(D\) 的梯度并取负（梯度下降方向），得到：

$$
V(x_k) = -\frac{2}{NF} \sum_f \big[ -\Delta C \sin(f^\top x_k) + \Delta S \cos(f^\top x_k) \big] f
$$

其中 \(\Delta C = C_\mu - C_\nu\)，\(\Delta S = S_\mu - S_\nu\)。实现里会对 \(V\) 做方差归一化（`normalize_drift`），再用同一套 drifting loss：target = `(x + V).detach()`，loss = `(x - target).pow(2).sum(dim=-1)`。

### 8.3 复杂度

\(O(N \cdot F)\)：每个 batch 对 \(N\) 个样本、\(F\) 个频率算内积与三角函数即可，无需 \(N\times M\) 的核矩阵。

---

## 9. drifting_autoencoder.py：漂移自编码器

- **编码器**：data → latent；用 `drifting_loss(encoded, prior)` 让 latent 分布向先验 \(\mathcal{N}(0,I)\) 漂移（prior 每步重采样）。  
- **解码器**：latent → data；用 **L1 重建损失** `decoded vs data`（代码中未对解码端再做一次向数据的漂移，而是直接重建）。  
- 特征归一化（Sec A.6）：在算漂移前对特征做零均值、单位方差，再缩放使平均成对距离约 \(\sqrt{D}\)，便于温度与尺度解耦；漂移算在归一化空间，再映射回原空间并做方差归一化。  
- 编码/解码均为 **1-NFE**；latent 在训练时 detach，编码器与解码器通过不同损失分别更新。

---

## 10. 运行与可视化

- **基础训练**（8 Gaussians + Checkerboard，漂移场与样本对比图）：
  ```bash
  python drifting.py
  ```
- **Sinkhorn 版**：`python sinkhorn_drifting.py`，流程同 `drifting.py`，仅 affinity 为 Sinkhorn。  
- **CF 漂移**：`python cf_drifting.py`，使用 `CFDriftGenerator`，可调 `num_freq`、`freq_std`。  
- **漂移自编码器**：`python drifting_autoencoder.py`，输出 data / encoded / generated / reconstructed 四宫格。

生成图片包括：数据分布、训练中/收敛后生成分布对比、漂移场 quiver 图（数据点、生成点、\(V\) 箭头）。

---

## 11. 小结与延伸阅读

### 小结

- **Minimal-Drifting-Models** 用 2D 玩具数据实现了 Deng et al. 的 Drifting Models：训练时用漂移场 \(V\) 演化 pushforward 分布，推理时 **1-NFE** 生成。  
- **漂移场**：mean-shift 形式 \(V = W_{\mathrm{pos}}y^+ - W_{\mathrm{neg}}y^-\)，由核亲和度双归一化（或 Sinkhorn）得到权重；另有 CF 距离版本，复杂度 \(O(N\cdot F)\)。  
- **损失**：MSE(\(f(\varepsilon)\), \(\mathrm{sg}(f(\varepsilon)+V)\))，梯度只过 \(f\)，等价于最小化 \(\Vert V\Vert^2\)。  
- 扩展包括：Sinkhorn 双随机、CF 漂移、漂移自编码器（编码向先验、解码向数据/重建）。

### 延伸阅读

- 论文：[*Generative Modeling via Drifting*](https://arxiv.org/abs/2602.04770)（Deng et al., ICML 2026）。  
- 项目：[Algomancer/Minimal-Drifting-Models](https://github.com/Algomancer/Minimal-Drifting-Models)。  
- 与 diffusion/flow 的对比可参见本系列中 [SDE-ODE-离散与连续的转换](/posts/diffusion-flow/sde-ode-%E7%A6%BB%E6%95%A3%E8%BF%9E%E7%BB%AD%E7%9A%84%E8%BD%AC%E6%8D%A2/) 等文章。
