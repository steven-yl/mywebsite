---
title: "第十章 ASE 潜变量 z 设计深度解析"
date: 2026-07-19
draft: false
authors: [Steven]
tags: [mimickit, robots, ASE, 潜变量 z, 设计深度解析, VAE, DIAYN]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 10
series_weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第十章 ASE 潜变量 z 设计深度解析

> 本文档是 [第十章 ASE 算法](10-algorithm-ase.md) 的补充专题，系统说明 MimicKit 中技能潜变量$z$的设计动机、几何约束、失效模式，以及与 VAE / DIAYN 等方法的对比。  
> 对应代码：`mimickit/learning/ase_agent.py`、`mimickit/learning/ase_model.py`。

---

## 目录

1. [概述](#1-概述)
2. [z 在 ASE 中的角色与数据流](#2-z-在-ase-中的角色与数据流)
3. [球形约束：为何使用单位球面](#3-球形约束为何使用单位球面)
4. [相近 z 的影响分析](#4-相近-z-的影响分析)
5. [同一行为对应不同 z 的影响分析](#5-同一行为对应不同-z-的影响分析)
6. [非球形 + MSE 的替代方案](#6-非球形--mse-的替代方案)
7. [VAE ELBO 如何规范 z](#7-vae-elbo-如何规范-z)
8. [DIAYN 互信息如何规范 z](#8-diayn-互信息如何规范-z)
9. [ASE / VAE / DIAYN 综合对比](#9-ase--vae--diayn-综合对比)
10. [调参与诊断建议](#10-调参与诊断建议)
11. [代码索引](#11-代码索引)
12. [参考文献](#12-参考文献)

---

## 1. 概述

### 1.1 核心问题

ASE（Adversarial Skill Embeddings）在 AMP 对抗模仿框架上引入技能潜变量$z$，使策略变为条件策略$\pi(a|s,z)$，价值函数变为$V(s,z)$，并通过编码器$g(\cdot)$从运动特征反推$z$。

**无额外约束时**，$z$容易出现三类退化：

| 退化模式 | 表现 |
|---------|------|
| 忽略 z | 所有$z$产生相同行为，$z$成为冗余输入 |
| 一对多 | 同一 motion 对应多个$z$，编码器监督冲突 |
| 多对一 | 多个相近$z$塌缩为同一行为，技能空间不可区分 |

ASE 通过 **单位球面几何 + 编码器对齐 + 多样性损失 + AMP 判别器** 四者协同，规范$z$的语义与可复用性。

### 1.2 MimicKit 实现要点

| 组件 | 实现 |
|------|------|
| z 采样 | `normalize(N(0, I))`，落在单位超球面 |
| 编码器输出 | `normalize(linear(h))`，与 z 同几何 |
| 对齐度量 | 负点积$\text{err} = -z^\top \hat{z}$（等价于负余弦相似度） |
| enc reward |$r_{enc} = \max(0, z^\top \hat{z}) \in [0, 1]$|
| 多样性 |$\Delta_z = 0.5 - 0.5\, z^\top z'$，约束 z 距离与动作距离成比例 |

默认超参（`data/agents/ase_humanoid_agent.yaml`）：`latent_dim=64`，`enc_reward_weight=0.5`，`diversity_weight=0.01`，`diversity_tar=1.0`。

---

## 2. z 在 ASE 中的角色与数据流

### 2.1 训练时数据流

```
Rollout 阶段:
  z ~ normalize(N(0,I))          # 每 env 持有一个 z，定期重采样
  a ~ π(a|s, z)                  # Actor 输入 [obs; z]
  记录 (obs, disc_obs, z, a, r)  # _record_data_pre_step

奖励计算:
  r = w_task·r_task + w_disc·r_disc + w_enc·r_enc
  r_enc = max(0, z^T enc(disc_obs))

训练更新:
  PPO(actor, critic)             # 条件于 z
  编码器 L_enc = E[-z^T enc(disc_obs)]
  diversity_loss(actor)          # 不同 z → 不同 μ(a|s,z)
  判别器 (同 AMP)
```

### 2.2 关键代码路径

**采样与重采样**（`ase_agent.py`）：

```python
def _sample_latents(self, n):
    unorm_z = torch.normal(torch.zeros([n, z_dim], device=self._device))
    z = torch.nn.functional.normalize(unorm_z, dim=-1)
    return z
```

**Actor / Critic / Encoder**（`ase_model.py`）：

```python
def eval_actor(self, obs, z):
    in_data = torch.cat([obs, z], dim=-1)
    ...

def eval_enc(self, enc_obs):
    unorm_z = self._enc_out(h)
    z = torch.nn.functional.normalize(unorm_z, dim=-1)
    return z
```

**双向一致性要求**：

- **正向**：$z \to \pi(a|s,z) \to \text{motion}$（diversity loss 推动不同 z 不同动作）
- **逆向**：$\text{motion} \to g(\text{disc\_obs}) \to \hat{z} \approx z$（enc loss / enc reward）

---

## 3. 球形约束：为何使用单位球面

### 3.1 设计动机

ASE 将$z$定义为 **技能 ID**，而非带方差的生成式潜变量。单位球面$\|z\|_2 = 1$提供以下 inductive bias：

| 动机 | 说明 |
|------|------|
| 方向 = 技能 | 去掉模长冗余，变方向即换技能 |
| 均匀探索 | `normalize(N(0,I))` 在高维近似球面均匀分布 |
| 有界相似度 |$z^\top \hat{z} = \cos\theta \in [-1, 1]$，reward 尺度稳定 |
| 防止模长作弊 | 编码器无法靠放大$\|\hat{z}\|$提高点积 |
| 输入尺度匹配 | z 与 normalized obs 量级接近，直接 concat 进 MLP |
| 多样性公式成立 |$\Delta_z = 0.5 - 0.5\, z^\top z'$表示夹角，值域$[0,1]$|

### 3.2 数学形式

**采样**：

$$
\tilde{z} \sim \mathcal{N}(0, I_{d_z}), \quad z = \frac{\tilde{z}}{\|\tilde{z}\|_2}
$$

**编码器对齐**（$\|z\| = \|\hat{z}\| = 1$时）：

$$
\text{err}(z, \hat{z}) = -z^\top \hat{z} = -\cos\theta
$$

$$
r_{enc} = \max(0, z^\top \hat{z}) \in [0, 1]
$$

**多样性 z 距离**：

$$
\Delta_z = 0.5 - 0.5\, z^\top z' \in [0, 1]
$$

-$z = z' \Rightarrow \Delta_z = 0$
-$z \perp z' \Rightarrow \Delta_z = 0.5$

### 3.3 不用球形的后果

| 问题 | 后果 |
|------|------|
| 编码器模长作弊 |$\|\hat{z}\|$变大即可提高$z^\top \hat{z}$，方向未对齐也拿高 reward |
| 表示不唯一 |$z$与$2z$方向相同但 MSE/点积行为不同，技能 ID 无规范 |
| 采样分布偏 |$\mathcal{N}(0,I)$方向不均匀，$\|z\|$随机波动 |
| diversity 公式失效 |$\Delta_z$不再表示夹角 |
| reward 无界 | enc reward 与 disc reward 尺度失衡 |
| 下游复用困难 | 指定 z 时模长无标准，同一技能难以稳定复现 |

---

## 4. 相近 z 的影响分析

### 4.1 随机采样下「相近 z」的概率

在$d_z = 64$维，两个独立 `normalize(N(0,I))` 样本的期望内积$\mathbb{E}[z^\top z'] \approx 0$（近似正交）。**独立采样时相近 z 概率较低**，但在以下情况仍会出现：

- 训练后 z 空间**塌缩**（大量 z 行为相似）
- 人为指定相近 z 做下游测试
- 高维下小概率碰撞（并行 env 偶发）

### 4.2 对各组件的影响

| 组件 | 相近 z 的影响 | 严重程度 |
|------|--------------|---------|
| Actor | 相近输入 → 相近动作，并行 env 探索冗余 | 中 |
| 编码器 | 不同 z 对应相似 motion，$z \leftrightarrow \text{motion}$绑定弱化 | **高** |
| enc reward |$z^\top \hat{z}$对多个 z 同时偏高，技能不可区分 | **高** |
| diversity loss | **允许**相近 z 产生相近动作（$\Delta_z$小则$\Delta_a$应小） | 低（符合设计） |
| Critic |$V(s,z)$对 z 不敏感，条件价值退化 | 中 |
| 判别器 | 与 z 无关，无直接影响 | 无 |

### 4.3 根本原因

ASE 假设 **z 是技能空间中的可区分坐标**。相近 z 破坏$z \leftrightarrow \text{motion}$的一一对应，使 z 从「技能 ID」退化为「噪声条件变量」，下游无法通过切换 z 可靠切换技能。

### 4.4 典型表现（训练失败信号）

- `enc_reward_mean` 长期偏低
- 换 z 后行为变化极小
- `diversity_loss` 难以下降
- 多个 z 可视化后 motion 高度相似

---

## 5. 同一行为对应不同 z 的影响分析

### 5.1 问题定义

若 motion 特征 `disc_obs` 近似相同，但 rollout 记录的 target z 分别为$z_1, z_2, z_3 \ldots$，则编码器收到**矛盾监督**：

| 样本 | 监督信号 |
|------|---------|
| motion M, z=z₁ | enc(M) → z₁ |
| motion M, z=z₂ | enc(M) → z₂ |
| motion M, z=z₃ | enc(M) → z₃ |

### 5.2 对各组件的影响

#### 编码器（影响最大）

- 梯度互相抵消，enc(M) 预测不稳定或趋于折中方向
- 对任意真实 z，$z^\top \hat{z}$偏低
- `enc_reward_weight=0.5` 时，该 motion 在总奖励中持续吃亏

#### 策略 π(a|s,z)

- 若策略忽略 z，所有 z 产生相同行为 → **ASE 最不希望的状态**
- diversity loss 会惩罚：z 差大、动作差小 →$\rho = \Delta_a / \Delta_z$偏小 → loss 增大
- 形成 enc（要求 motion 可识别 z）与 diversity（要求不同 z 不同动作）的拉锯

#### Critic V(s,z)

- 同一行为、同一回报 →$V(s,z_1) \approx V(s,z_2)$
- Critic 学到 z 对价值无影响，条件价值函数退化为$V(s)$

#### 判别器

- 只看 disc_obs，与 z 无关；模仿质量不受影响
- **仅 enc reward + diversity loss 在强迫 z 与行为绑定**；若二者权重不足，可能出现「disc 好但 z 不可控」

### 5.3 两种场景区分

| 场景 | 描述 | 是否可接受 |
|------|------|-----------|
| A：同一 obs、不同 z、动作几乎相同 | 策略忽略 z | **不可接受** |
| B：宏观同类（如都是走路）、disc_obs 可区分 | 步态细节不同，特征空间可分 | **可接受** |

用户所说的「一个行为」若指 motion 流形上几乎相同的轨迹 → 场景 A；若指语义同类但特征可区分 → 场景 B，影响较小。

---

## 6. 非球形 + MSE 的替代方案

### 6.1 方案描述

将 ASE 改为：

- 采样：$z \sim \mathcal{N}(0, I)$，不 normalize
- 编码器：输出不 normalize
- 对齐：$L_{enc} = \|z - \hat{z}\|^2$（MSE）
- enc reward、diversity loss 需同步重写

### 6.2 优点

| 优点 | 说明 |
|------|------|
| 监督形式标准 | MSE 是常见回归损失，梯度$2(\hat{z}-z)$行为熟悉 |
| 保留模长信息 | 若技能需要「强度/幅度」，MSE 可同时约束方向与模长 |
| 实现更简单 | 无需 spherical projection |
| 线性插值直观 |$\alpha z_1 + (1-\alpha) z_2$混合技能（需模长有语义） |
| 软约束灵活 | 可加$\lambda \|z\|^2$替代硬投影 |

### 6.3 缺点

| 缺点 | 说明 |
|------|------|
| 表示不唯一 |$z$与$2z$方向相同但 MSE 不同，技能 ID 无规范 |
| 与方向对齐目标不一致 | 方向对、模长错仍被 MSE 惩罚 |
| enc reward 需重设计 | MSE 无天然有界 reward，与 disc reward 尺度难平衡 |
| diversity 公式失效 | 需改用$\|z - z'\|^2$并重标定 `diversity_tar` |
| 采样分布偏 | 方向不均匀，模长引入噪声维度 |
| 偏离 ASE 设计 | 缺少 ELBO/DIAYN 式额外约束时，z 易不可控 |
| 高维尺度 | MSE 随 `latent_dim` 增大，权重难调 |

### 6.4 改造需修改的代码位置

| 文件 | 方法 | 改动 |
|------|------|------|
| `ase_agent.py` | `_sample_latents` | 去掉 normalize |
| `ase_model.py` | `eval_enc` | 去掉 normalize |
| `ase_agent.py` | `_calc_enc_error` | 改为 MSE |
| `ase_agent.py` | `_calc_enc_rewards` | 改为 `-MSE` 或 `exp(-α·MSE)` |
| `ase_agent.py` | `_compute_diversity_loss` | 重写$\Delta_z$度量 |

### 6.5 实践建议

| 目标 | 建议 |
|------|------|
| ASE 论文同款可复用技能库 | **保留球形 + 点积** |
| 实验非球形 + MSE | 同步引入 KL 正则或 DIAYN 式互信息 reward |
| 折中方案 | 非球形采样 + 训练/推理时 normalize；或 MSE + cosine 混合损失 |

---

## 7. VAE ELBO 如何规范 z

### 7.1 基本设定

VAE 假设数据$x$（如 motion 片段）由潜变量$z$生成：

$$
p(x) = \int p(x|z)\, p(z)\, dz
$$

- **先验**$p(z)$：通常$\mathcal{N}(0, I)$
- **解码器**$p_\theta(x|z)$：给定 z 生成 x
- **编码器**$q_\phi(z|x)$：从 x 推断 z

### 7.2 ELBO 分解

$$
\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重构项}} - \underbrace{D_{\mathrm{KL}}\big(q_\phi(z|x)\,\|\, p(z)\big)}_{\text{KL 正则项}}
$$

### 7.3 两项各自如何规范 z

#### 重构项$\mathbb{E}[\log p_\theta(x|z)]$

- z 必须对生成 x **有信息**——无用维度会被解码器忽略
- 建立 **z → x** 因果：换 z 应改变生成结果
- 防止 z 被完全架空（与 KL 项配合）

#### KL 项$-D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$

| 效果 | 说明 |
|------|------|
| 有界、标准化 | z 被拉向$\mathcal{N}(0,I)$，模长/方差受控 |
| 平滑 latent 空间 | 相近 z → 相近 x，便于插值 |
| 防止后验坍缩 | 不让 encoder 输出极端确定的 z |
| 信息瓶颈 | 超出先验的信息需付 KL 代价 |

重参数化：$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon,\ \epsilon \sim \mathcal{N}(0,I)$

### 7.4 直觉

```
ELBO = 「用 z 重构 motion 的能力」−「z 偏离标准高斯的惩罚」
```

### 7.5 局限（相对 RL 技能发现）

- 主要是 **x ↔ z** 生成式约束，不直接保证不同 z 产生不同 RL 策略行为
- 可能出现 **posterior collapse**：z 几乎不被使用
- 通常离线学 latent，与在线 RL 策略分离
- 不保证每个 z 对应可区分、可控制的「技能」

---

## 8. DIAYN 互信息如何规范 z

### 8.1 核心目标

DIAYN（Diversity Is All You Need, Eysenbach et al. 2018）通过最大化互信息规范 z：

$$
\max I(Z; S') \quad \text{或} \quad \max I(Z; S' \mid S)
$$

**直觉**：轨迹后的状态$s'$应能**猜出**是哪个 z 在控制；同时不同 z 应产生**不同**的$s'$分布。

### 8.2 互信息分解

$$
I(Z; S') = H(Z) - H(Z|S')
$$

| 项 | 含义 | 优化方向 |
|----|------|---------|
|$H(Z)$| z 的熵 | 最大化 → 均匀使用各种 z |
|$H(Z|S')$| 给定 s' 后 z 的不确定性 | 最小化 → s' 能识别 z |

### 8.3 伪奖励（内在奖励）

用判别器$q_\phi(z|s')$近似：

$$
r_{\text{DIAYN}}(s, z, s') = \log q_\phi(z|s') - \log p(z)
$$

策略优化：

$$
J = \mathbb{E}\left[\sum_t r_{\text{DIAYN},t}\right] + \alpha H(\pi(\cdot|s,z))
$$

### 8.4 三项约束机制

| 机制 | 规范内容 |
|------|---------|
|$\log q_\phi(z|s')$高 | **可辨识性**：每个 z 产生独特 state 分布；同一 s' 不应对应多个 z |
|$\log p(z)$| **覆盖性**：防止只用单一 z，鼓励探索整个 z 空间 |
|$H(\pi)$| **技能内随机性**：固定 z 下仍需动作随机，避免 s' 分布退化 |

### 8.5 与 ASE 编码器的对应关系

| | DIAYN | ASE (MimicKit) |
|---|--------|----------------|
| 识别网络 |$q_\phi(z|s')$|$g(\text{disc\_obs}) \to \hat{z}$|
| 训练信号 | 内在 reward 驱动 RL | enc reward + enc loss |
| 多样性 | 互信息中隐含 | 显式 diversity loss |
| 运动先验 | 无（纯仿真探索） | AMP 判别器 |
| z 几何 | 离散或连续 + 先验 | 单位球面 + 点积 |

DIAYN 的规范是**信息论意义**的：z 必须与可观测 state 变化绑定，且被充分使用。

---

## 9. ASE / VAE / DIAYN 综合对比

### 9.1 架构对比

```
VAE (ELBO):
  z ~ N(0,I) ──→ 解码器 p(x|z) ──→ motion x
                    ↑
  x ──→ 编码器 q(z|x) ──→ KL 拉回先验

DIAYN (互信息):
  z ~ p(z) ──→ 策略 π(a|s,z) ──→ s'
                    ↑
  s' ──→ 判别器 q(z|s') ──→ 最大化 I(Z;S')

ASE (MimicKit):
  z ~ normalize(N(0,I)) ──→ π(a|s,z) ──→ disc_obs
       ↑                           ↓
  enc(disc_obs) ←── enc reward    disc reward (AMP)
       +
  diversity_loss (z 差 ↔ 动作差)
```

### 9.2 规范 z 的方式对比

| 维度 | VAE ELBO | DIAYN 互信息 | ASE |
|------|----------|--------------|-----|
| **规范什么** | z 分布形状 + 能否重构 x | z 是否控制可区分的 s' | z 方向 + motion 对齐 + 动作分化 |
| **先验角色** | 强：KL 硬拉向 N(0,I) | 中：log p(z) 防不用某些 z | 强：球面均匀采样 |
| **因果方向** | z → x 生成 + x → z 推断 | z → s' + s' → z 识别 | z → action → motion → enc → z |
| **防止 ignore z** | 重构 + KL | r = log q(z|s') - log p(z) | enc reward + diversity |
| **防止 z 塌缩** | KL 到先验 | H(Z) + 多 z 探索 | diversity + 球面采样 |
| **运动自然性** | 取决于数据/解码器 | 可能学到奇怪但可区分技能 | AMP 判别器保证 |
| **reward 有界性** | ELBO 各项有界 | log 项需注意尺度 | enc reward ∈ [0,1] |
| **MimicKit 实现** | 未实现 | 未实现 | `ASEAgent` |

### 9.3 公式速查

**VAE**：

$$
\max_{\phi,\theta}\ \mathbb{E}_{x}\Big[\mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x)\|p(z))\Big]
$$

**DIAYN**：

$$
\max_{\pi,\phi}\ I(Z;S') \approx \mathbb{E}_{z,s,s'}\big[\log q_\phi(z|s') - \log p(z)\big]
$$

**ASE (MimicKit)**：

$$
\max\ w_{\text{disc}}\, r_{\text{disc}} + w_{\text{enc}}\, \max(0, z^\top \hat{z}) + \text{PPO} + w_{\text{div}}\, L_{\text{div}}
$$

### 9.4 ASE 相对 VAE/DIAYN 的定位

ASE **没有** ELBO 或 DIAYN 式互信息，而用以下组件实现类似功能：

| ASE 组件 | 等效的规范作用 |
|---------|---------------|
| 球形 + 点积 | 类似 ELBO 中 KL 对 latent 几何的规范 |
| enc loss + enc reward | 类似 DIAYN 的$s' \to z$可辨识性（逆向） |
| diversity loss | 类似 DIAYN 的「不同 z 不同效果」 |
| AMP disc reward | VAE/DIAYN 不具备的运动自然性约束 |

若去掉球形且仅用 MSE，又缺少 ELBO/DIAYN 式约束，则 z 的语义规范会显著弱化。

---

## 10. 调参与诊断建议

### 10.1 关键超参

| 参数 | 作用 | 典型问题 |
|------|------|---------|
| `diversity_weight` | 不同 z 不同动作 | 过小 → 技能退化；过大 → 动作不稳定 |
| `enc_reward_weight` | z 与 motion 绑定强度 | 过小 → z 不可控；过大 → 牺牲模仿质量 |
| `latent_dim` | 技能空间容量上界 | 过小 → 技能混叠；过大 → 训练难、相近 z 更稀疏 |
| `latent_time_min/max` | 每个 z 持续时间 | 过短 → enc 难收敛；过长 → 覆盖 z 空间慢 |

### 10.2 训练健康指标

| 指标 | 健康 | 异常信号 |
|------|------|---------|
| `enc_reward_mean` | 稳步上升 | 长期偏低 → z-motion 绑定失败 |
| `enc_loss` | 下降并稳定 | 震荡或不降 → 编码器监督冲突 |
| `diversity_loss` | 收敛到较低值 | 长期高 → z 未分化动作 |
| `disc_reward_mean` | 与 AMP 相当 | 正常但换 z 无变化 → z 被忽略 |

### 10.3 故障排查

| 现象 | 可能原因 | 建议 |
|------|---------|------|
| 换 z 行为不变 | diversity_weight 过小 / 策略忽略 z | 增大 diversity_weight，检查 actor 是否收到 z |
| enc_loss 不降 | 同一 motion 多 z / z 空间塌缩 | 增大 enc_reward_weight，检查 diversity |
| 动作不自然 | disc 权重相对不足 | 增大 disc_reward_weight |
| 技能过于相似 | latent_dim 不足或 diversity 弱 | 增大 latent_dim / diversity_weight |

---

## 11. 代码索引

| 功能 | 文件 | 方法 |
|------|------|------|
| z 采样 | `ase_agent.py` | `_sample_latents` |
| z 重采样调度 | `ase_agent.py` | `_reset_latents`, `_update_latents` |
| rollout 记录 z | `ase_agent.py` | `_record_data_pre_step` |
| enc error / reward | `ase_agent.py` | `_calc_enc_error`, `_calc_enc_rewards` |
| enc 训练 | `ase_agent.py` | `_update_enc`, `_compute_enc_loss` |
| diversity loss | `ase_agent.py` | `_compute_diversity_loss` |
| Actor/Critic/Enc 网络 | `ase_model.py` | `eval_actor`, `eval_critic`, `eval_enc` |
| 默认超参 | `data/agents/ase_humanoid_agent.yaml` | — |

---

## 12. 参考文献

```bibtex
@article{2022-TOG-ASE,
  author = {Peng, Xue Bin and Guo, Yunrong and Halper, Lina and Levine, Sergey and Fidler, Sanja},
  title = {ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters},
  journal = {ACM Trans. Graph.},
  year = {2022}
}

@article{eysenbach2018diayn,
  author = {Eysenbach, Benjamin and Gupta, Abhishek and Ibarz, Julian and Levine, Sergey},
  title = {Diversity is All You Need: Learning Skills without a Reward Function},
  journal = {arXiv preprint arXiv:1802.06059},
  year = {2018}
}
```

**链接**：

- [ASE 项目页](https://xbpeng.github.io/projects/ASE/)
- [DIAYN 论文](https://arxiv.org/abs/1802.06059)
- [MimicKit ASE 主文档](10-algorithm-ase.md)

---

[← ASE 算法主文档](10-algorithm-ase.md) | [返回索引](TECHNICAL_INDEX.md)
