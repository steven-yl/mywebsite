---
title: "GQF（Q-Guided Flow）算法原理"
subtitle: ""
date: 2026-07-14T18:00:00+08:00
draft: false
authors: [Steven]
description: "梳理 QGF 算法原理、训练与测试时 Q 引导推理机制。"
summary: "QGF 算法原理与测试时引导解读。"
tags: [diffusion/flow, qgf]
categories: [docs qgf]
series: [qgf-docs]
weight: 11
series_weight: 11
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---


## QGF（Q-Guided Flow）算法原理详细技术文档

### 一、背景与问题设定

#### 1.1 什么是离线强化学习

在标准强化学习中，智能体通过与真实环境持续交互、收集经验来学习策略。但在许多实际场景中（如机器人操作、自动驾驶），与环境进行大量交互成本极高甚至危险。**离线强化学习**（Offline RL）的核心思想是：**仅使用一份预先收集好的固定数据集 $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ 来训练策略**，无需与环境实时交互。

（注：此处 $s$ 为 observation 观测，$a$ 为 action 动作，$r$ 为 reward 奖励，$s'$ 为 next observation 下一观测，本仓库动作空间被归一化到 $[-1, 1]$ 区间）

#### 1.2 离线 RL 的核心挑战

离线 RL 面临一个根本性矛盾：

- **Critic（Q 函数）** 需要评估某个动作的好坏：$Q(s, a)$
- 但 Critic 只在数据集中见过的 $(s, a)$ 对上训练得准确
- 传统的 Actor-Critic 方法会让 Actor 朝着 $Q(s, a)$ 高的方向更新，但 Actor 可能生成数据集**从未出现过**的动作（OOD, Out-of-Distribution）
- Critic 对这些 OOD 动作的 Q 值估计是**不可靠的**（往往被高估），导致策略崩溃

这就是 **OOD 动作过估计问题**——离线 RL 最核心的挑战。

#### 1.3 为什么用生成式策略（Flow Matching）

传统 Actor-Critic 方法通常使用**高斯策略**（Gaussian Actor）：给定状态 $s$，输出一个高斯分布的均值 $\mu(s)$ 和标准差 $\sigma(s)$，然后从该分布中采样动作。这种策略的最大问题是：**只能表达单峰分布**，无法捕捉真实数据集中复杂的多模态行为分布。

**Flow Matching**（流匹配）是一种生成式建模方法，它学习了从简单噪声分布到真实数据分布之间的**连续变换**。本仓库使用 Flow Matching 来构建行为策略，具体来说：

- 学习一个时间依赖的**速度场** $v_\theta(s, a_t, t)$，其中 $t \in [0, 1]$
- $t=0$ 时，动作 $a_0$ 从标准高斯噪声 $\mathcal{N}(0, I)$ 采样
- $t=1$ 时，动作 $a_1$ 是真实的干净动作
- 沿 ODE 积分：$\frac{da}{dt} = v_\theta(s, a_t, t)$，从 $t=0$ 积分到 $t=1$ 即可从噪声生成动作

Flow Matching 的**优势**：
- 表达力强，能天然建模多模态行为分布
- 采样过程稳定（确定性的 ODE 积分）
- 与 BC 目标天然契合，训练简单

---

### 二、方法分类总览

QGF 仓库实现了三类离线 RL 范式：

```
离线 RL 方法
├── 训练时 Actor-RL（联合优化）
│   ├── FQL：蒸馏多步去噪为一步策略 + 最大化 Q
│   ├── EDP：训练时单步近似最大化 Q + BC 正则
│   ├── QAM：Adjoint matching，无需 BPTT 的 Q 感知流训练
│   ├── CFGRL：Classifier-Free Guidance RL，条件流引导
│   ├── IQL：高斯 Actor + IQL（AWR / DDPG+BC 更新）
│   └── SAC：Soft Actor-Critic
│
├── 测试时 Guidance（推理时改进）
│   ├── ★ QGF：BC flow + IQL；推理时 Q 梯度引导，单步 Euler 近似干净动作
│   ├── QFQL：在噪声 a_t 上直接求梯度（有偏基线）
│   ├── QGF-Jacobian：对 Euler 近似做链式法则
│   ├── GradStep：先去噪得到干净动作，再在干净动作空间做梯度上升
│   ├── RobustQ：训练噪声条件 Q_robust(s, a_t, t)，推理用其梯度
│
└── 纯模仿 / 值函数（无策略改进）
    └── BC：纯 Flow Matching 行为克隆
```

**QGF 的核心创新**：将策略改进从"训练阶段"移到了"推理阶段"。Actor 永远只做 BC 训练（模仿数据），不接收任何 RL 梯度。Critic 独立训练（IQL 方式）。在推理时，通过 Q 函数的梯度来引导去噪过程，提升策略质量。

---

### 三、训练阶段详解

QGF 的训练分为三个**完全解耦**的模块，各自独立优化：

#### 3.1 Flow Matching BC 策略训练（`policy_loss`）

**数学原理**：

Flow Matching 使用**线性插值路径**（Rectified Flow / OT 直线路径）：

1. 从数据集中采样一个真实动作 $a_1$（即数据中的动作）
2. 从标准高斯分布采样噪声 $a_0 \sim \mathcal{N}(0, I)$
3. 采样离散时间 $t \in \{0, \frac{1}{N}, \frac{2}{N}, \ldots, 1\}$，其中 $N$ = `denoise_steps`（默认 10）
4. 线性插值得到中间状态：$a_t = (1-t) \cdot a_0 + t \cdot a_1$
5. 真值速度（目标）：$v^* = a_1 - a_0$（注意：这是一个常向量场，与 t 无关）

**损失函数**（均方误差）：

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{a_0, t}\big[\| v_\theta(s, a_t, t) - (a_1 - a_0) \|^2\big]
$$

**代码逻辑**（`QGFAgent.policy_loss`）：

```python
def policy_loss(self, batch, policy_params=None, rng=None):
    # 1. 获取干净动作 a1
    if self.config.get("action_chunking", False):
        a1 = jnp.reshape(batch["actions"], (B, -1))  # 展平 H 步动作
    else:
        a1 = batch["actions"][..., 0, :]  # 取第一步动作

    # 2. 采样噪声和离散时间
    a0 = jax.random.normal(eps_rng, a1.shape)         # ~ N(0, I)
    t = random.randint(0, denoise_steps+1) / denoise_steps  # 离散均匀

    # 3. 线性插值 + 目标速度
    a_t = a0 * (1 - t) + a1 * t   # 插值到中间状态
    vel = a1 - a0                  # 真值速度（常向量场）

    # 4. 网络预测 + MSE 损失
    pred_vel = self.policy(observations, a_t, t)
    bc_loss = jnp.mean((vel - pred_vel) ** 2)

    return bc_loss, {"bc_loss": bc_loss}
```

**关键设计**：
- 速度场是**常向量场**（不依赖 t），这是 Rectified Flow 的特性，使训练更稳定
- 时间 t 是离散均匀采样的，对应推理时的 Euler 离散步数
- 网络 `ActorFlowField` 是 4×512 GELU + LayerNorm 的 MLP，输入为 `[obs, a_t, time_embedding]`，输出为速度向量 $v_\theta$

#### 3.2 IQL Critic 训练（`critic_loss`）

**数学原理**：

IQL（Implicit Q-Learning）的核心思想是：**不在 OOD 动作上做 max-Q 操作**，而是通过 Expectile Regression 学习一个隐式的"最优"价值函数。

Critic 的 TD 目标使用 **Value 函数 V 而非 max Q** 来 bootstrap：

$$
y = r_{\text{cum}} + \gamma^H \cdot m \cdot V_\psi(s_{t+H})
$$

其中：
- $H$ = `horizon_length`（n-step 或 action chunk 步数）
- $r_{\text{cum}}$ 是 sample_sequence 计算的累积折扣回报
- $m$ 是 episode 终止掩码（Terminating 后为 0）

**损失函数**：

$$
\mathcal{L}_Q = \mathbb{E}\Big[\sum_{i=1}^{K} (Q_{\phi,i}(s, a) - y)^2 \cdot w_{\text{valid}}\Big]
$$

- $K$ = `num_qs`（默认 2，双 Q Ensemble）
- $w_{\text{valid}}$ 是 episode 截断后的权重（截断后为 0，不参与 loss）

**代码逻辑**（`QGFAgent.critic_loss`）：

```python
def critic_loss(self, batch, critic_params=None):
    H = self.config.get("horizon_length", 1)
    batch_actions, next_obs, rewards, masks, valid_w = self._get_flat_batch(batch)

    next_v = self.value(next_obs)
    target_q = rewards + (self.config["discount"] ** H) * masks * next_v

    qs = self.critic(batch["observations"], batch_actions, params=critic_params)
    critic_loss = (((qs - target_q[None]) ** 2) * valid_w).mean()

    return critic_loss, {"critic_loss": critic_loss, "q": qs[0].mean()}
```

**关键设计**：
- 使用 `target_critic`（软更新，`tau=0.005`）来评估 Value，但 **Critic 训练不使用 target Q 做 bootstrap**
- 双 Q Ensemble 降低过估计
- `valid_w` 确保 episode 截断后不参与训练

#### 3.3 IQL Value 训练（`value_loss`）

**数学原理**：

Value 函数 $V_\psi(s)$ 通过 **Expectile Regression**（期望分位回归）来拟合数据动作上的 Q 值上尾：

$$
\mathcal{L}_V = \mathbb{E}\big[\rho_\tau(Q_{\bar\phi}(s,a) - V_\psi(s)) \cdot w_{\text{valid}}\big]
$$

其中 Expectile 权重函数为：

$$
\rho_\tau(\delta) = \begin{cases}
\tau \cdot \delta^2 & \delta > 0 \\
(1-\tau) \cdot \delta^2 & \delta \le 0
\end{cases}
$$

默认 $\tau = 0.9$（`expectile`）。

**直观理解**：
- 当 $Q > V$ 时（TD 误差为正），权重 $\tau = 0.9$ 更大，V 向上拉
- 当 $Q < V$ 时（TD 误差为负），权重 $1-\tau = 0.1$ 更小，V 向下拉
- 结果：V 逼近 Q 在数据动作上的 **0.9 分位点**（即上尾，而非最大值）

**为什么这样做**：在离线数据中，直接对 Q 取 max 会选中 OOD 动作（Critic 没见过的动作，Q 估计不可靠）。Expectile 只向数据中**已有的好动作**的 Q 值靠拢，不会跑到 OOD 区域。

**代码逻辑**（`QGFAgent.value_loss`）：

```python
def value_loss(self, batch, value_params=None):
    batch_actions, _, _, _, valid_w = self._get_flat_batch(batch)
    qs = self.target_critic(batch["observations"], batch_actions)
    q = self._aggregate_q(qs)  # min 聚合，默认
    v = self.value(batch["observations"], params=value_params)
    value_loss = (expectile_loss(q - v, self.config["expectile"]) * valid_w).mean()
    return value_loss, {"value_loss": value_loss, "v": v.mean()}
```

#### 3.4 训练循环（`update`）

```python
@jax.jit
def update(self, batch):
    new_rng, policy_rng = jax.random.split(self.rng, 2)

    # 三个模块独立 Adam 更新（非联合反向传播）
    new_policy, policy_info = self.policy.apply_loss_fn(
        loss_fn=lambda p: self.policy_loss(batch, p, rng=policy_rng)
    )
    new_critic, critic_info = self.critic.apply_loss_fn(
        loss_fn=lambda p: self.critic_loss(batch, p)
    )
    new_target_critic = target_update(
        self.critic, self.target_critic, self.config["tau"]
    )
    new_value, value_info = self.value.apply_loss_fn(
        loss_fn=lambda p: self.value_loss(batch, p)
    )

    return self.replace(
        rng=new_rng,
        policy=new_policy,
        critic=new_critic,
        target_critic=new_target_critic,
        value=new_value,
    ), {**policy_info, **critic_info, **value_info}
```

**关键设计**：
- 三个模块（Policy、Critic、Value）**各自独立** `apply_loss_fn`，它们之间**不存在梯度流动**
- Policy 从不接收 Critic 或 Value 的梯度——这是 QGF 最核心的设计原则
- Target Critic 通过软更新：$\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{critic}} + (1-\tau) \cdot \theta_{\text{target}}$，其中 $\tau = 0.005$

---

### 四、推理阶段详解（QGF 核心）

推理是 QGF 最核心的创新所在。QGF 在去噪的每一步，用 Q 函数的梯度来引导速度场，使最终生成的动作不仅是"合理"的（BC 保证），还是"好"的（Q 保证）。

#### 4.1 推理算法伪代码

```
输入: 观测 s, 随机种子, 引导强度 w, 去噪步数 N=denoise_steps

a ← N(0, I)                             # 初始化：从标准高斯噪声开始
Δt ← 1/N                                 # 每步大小

for t_idx = 0 .. N-1:                    # N 步 Euler 积分
    t ← t_idx / N                         # 当前时间
    v_bc ← policy(s, a, t)               # ① BC 速度场

    # ② 计算干净动作近似 a'
    if denoised_action_approx == "one_euler_step_approx":
        a' ← clip(a + (1-t) * stop_grad(v_bc), -1, 1)   # QGF 默认
    elif denoised_action_approx == "noisy":
        a' ← a                                            # QFQL 基线

    # ③ 在 a' 处求 Q 梯度
    qgrad ← ∇_{a'} aggregate_q(target_critic(s, a')).sum()

    # ④ 可选：Jacobian 链式法则
    if apply_jacobian:
        J ← ∂a'/∂a                                       # 对 Euler 近似求导
        qgrad ← qgrad @ J                                 # 链式法则

    # ⑤ 引导速度 = BC 速度 + w × Q 梯度
    a ← a + (v_bc + w * qgrad) * Δt

a ← clip(a, -1, 1)                        # 最终裁剪

# 可选：Best-of-N 拒绝采样
if rejection_sampling > 1:
    采样 K 条轨迹，选 Q(s, a) 最大者
```

#### 4.2 关键设计决策详解

**决策一：为什么需要"干净动作近似"？**

在去噪过程中，中间状态 $a_t$ 是**噪声动作**（含有大量噪声），而 Critic 只在数据集的干净动作上训练过。如果在 $a_t$ 上直接求 Q 梯度，Critic 的估计是不可靠的（OOD 问题）。

**决策二：单步 Euler 近似（QGF）vs 噪声动作（QFQL）**

| 模式 | a' 计算方式 | Q 评估点 | 问题 |
|------|-----------|---------|------|
| QFQL | a' = a_t | 噪声动作 | Critic 在 OOD 区域估计不可靠，有偏 |
| QGF | a' = clip(a_t + (1-t)·v_bc, -1, 1) | 近似干净动作 | 在 Critic 训练分布内，更可靠 |

**QGF 的单步 Euler 近似**：用当前 BC 速度场做一步 Euler 积分，预测"如果继续沿当前方向去噪，最终会得到什么动作"。这个预测在 Critic 的训练分布内，因此 Q 梯度更可靠。

**决策三：丢弃 Jacobian（QGF 默认）vs 完整链式法则（QGF-Jacobian）**

严格来说，我们要的是 $\nabla_{a_t} Q$（对当前噪声动作的梯度），但我们计算的是 $\nabla_{a'} Q$（对近似干净动作的梯度）。根据链式法则：

$$
\nabla_{a_t} Q(s, f(a_t)) = \left(\frac{\partial f}{\partial a_t}\right)^\top \cdot \nabla_{a'} Q(s, a')
$$

其中 $f(a_t) = \text{clip}(a_t + (1-t) \cdot v_\theta(s, a_t, t), -1, 1)$ 是单步 Euler 映射。

- **QGF（默认）**：丢弃 Jacobian，直接使用 $\nabla_{a'} Q$ 作为 $\nabla_{a_t} Q$ 的近似。这是一个有偏近似，但**方差低**（因为不需要计算 Jacobian）。
- **QGF-Jacobian**：使用 `jax.vmap(jax.jacrev(map_single))` 计算完整的 $H \times H$ Jacobian 矩阵，然后通过 `einsum("bi,bij->bj", qgrad, jac)` 应用链式法则。更准确，但**方差高**（Jacobian 矩阵的估计本身有噪声）。

**QGF 选择丢弃 Jacobian 的原因**：在实验中，丢弃 Jacobian 带来的低方差优势超过了有偏近似带来的精度损失。

#### 4.3 引导速度的数学形式

QGF 的单步更新公式：

$$
a_{t+\Delta t} = a_t + \Big(v_\theta(s, a_t, t) + w \cdot \nabla_{a'} Q(s, a')\Big) \cdot \Delta t
$$

其中：
- $v_\theta$ 是 BC 速度场（让动作"像"数据中的动作）
- $\nabla_{a'} Q$ 是 Q 梯度（让动作"好"，即高回报）
- $w$ 是引导强度超参数，平衡"像"和"好"
- $a'$ 是单步 Euler 近似的干净动作

#### 4.4 与 Classifier Guidance 的类比

扩散模型中的 Classifier Guidance：

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t) - w \cdot \sigma_t \nabla_x \log p(c|x_t)
$$

QGF 将其中的：
- $\epsilon_\theta$（噪声预测）替换为 $v_\theta$（速度场）
- $\log p(c|x_t)$（分类器对数概率）替换为 $Q(s, a)$（价值函数）
- 在**动作空间**而非图像空间做逐步引导

#### 4.5 Rejection Sampling（Best-of-N）

`sample_actions(..., rejection_sampling=K)` 的额外机制：

1. 将同一观测 $s$ 复制 $K$ 份
2. 每份独立噪声初始化，完整去噪
3. 用 `aggregate_q(target_critic(s, a))` 评估每个动作的 Q 值
4. 选择 Q 值最大的动作

这是对 guidance 的正交增强，可同时使用 $w > 0$ 和 $K > 1$，代价是推理时间 × K。

---

### 五、网络结构详解

#### 5.1 ActorFlowField（流速度场）

**文件**：`utils/networks.py`

| 组件 | 描述 |
|------|------|
| 输入拼接 | `[obs, noised_action, time_embedding]` |
| 时间编码 | 正弦位置编码（`timestep_embedding`），将标量 t 映射到 16 维 |
| Trunk MLP | 4×512 GELU + LayerNorm（默认配置） |
| 输出层 | `Dense(action_dim)`，输出速度向量 v |

**前向传播**：
```python
def __call__(self, obs, noised_action, t=None):
    parts = [obs, noised_action]
    if t is not None:
        parts.append(embed_time(t, "sinusoidal"))  # 16维正弦编码
    concat_input = jnp.concatenate(parts, axis=-1)
    outputs = MLP(hidden_dims, activate_final=True)(concat_input)
    v = Dense(action_dim)(outputs)
    return v
```

#### 5.2 Value（Critic / Value 网络）

**文件**：`utils/networks.py`

| 组件 | 描述 |
|------|------|
| 网络类型 | 支持 `MLP`（默认 4×512）或 `BroNet`（残差块） |
| Ensemble | Critic: `num_ensembles=2`（双 Q），Value: `num_ensembles=1` |
| 输入 | Critic: `[obs, actions]`；Value: `[obs]` |
| 输出 | 标量 Q 值或 V 值 |

**Ensemble 实现**：通过 `nn.vmap` 在参数维度上并行化：
```python
if self.num_ensembles > 1:
    network = ensemblize(network, self.num_ensembles)
```
这使多个 Q head 共享网络结构但各自有独立参数，输出形状为 `(num_ensembles, batch_size)`。

#### 5.3 Q 聚合策略（`aggregate_q`）

```python
def aggregate_q(qs, config):
    aggregation_fn = getattr(jnp, config.get("q_aggregation", "min"))
    return aggregation_fn(qs, axis=0)
```

- `q_aggregation="min"`（默认）：取双 Q 的最小值，更保守，降低过估计（Clipped Double-Q）
- `q_aggregation="mean"`：取双 Q 的平均值

---

### 六、关键辅助模块

#### 6.1 `TrainState`（`utils/flax_utils.py`）

JAX 的不可变训练状态，包含：
- `params`：网络参数
- `opt_state`：优化器状态（Adam）
- `apply_fn`：前向传播函数
- `step`：训练步数计数器

**`apply_loss_fn`**：计算梯度并更新参数：
```python
def apply_loss_fn(self, loss_fn):
    grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
    # 记录梯度统计信息（max, min, norm）
    return self.apply_gradients(grads=grads), info
```

#### 6.2 `get_flat_batch`（`agents/common.py`）

从 `sample_sequence` 返回的序列 batch 中提取 Critic/Value 所需的标量/向量字段：

```python
def get_flat_batch(batch, config):
    if config.get("action_chunking", False):
        batch_actions = jnp.reshape(batch["actions"], (B, -1))  # (B, H*d_a)
    else:
        batch_actions = batch["actions"][..., 0, :]              # (B, d_a)
    next_obs = batch["next_observations"][..., -1, :]            # 序列末状态
    rewards = batch["rewards"][..., -1]                          # 累积 n-step 回报
    masks = batch["masks"][..., -1]                              # bootstrap 掩码
    valid_w = batch["valid"][..., -1]                            # 有效权重
    return batch_actions, next_obs, rewards, masks, valid_w
```

#### 6.3 `sample_sequence`（`utils/datasets.py`）

为 n-step TD 和 Action Chunking 采样连续序列：

```python
dataset.sample_sequence(batch_size=256, sequence_length=H, discount=0.99)
```

返回的 `rewards` 是累积折扣回报：$R_t^{(H)} = \sum_{i=0}^{H-1} \gamma^i r_{t+i} \cdot \prod_{j<i} \text{valid}_{t+j}$

---

### 七、超参数详解

| 超参数 | 默认值 | 含义 |
|--------|--------|------|
| `denoise_steps` | 10 | 训练/推理的 Euler 离散步数 N |
| `bc_lr` | 3e-4 | 策略学习率 |
| `critic_lr` | 3e-4 | Critic 学习率 |
| `value_lr` | 3e-4 | Value 学习率 |
| `discount` | 0.99 | 折扣因子 γ |
| `expectile` | 0.9 | 期望分位 τ（越大越乐观） |
| `tau` | 0.005 | Target 网络软更新系数 |
| `num_qs` | 2 | Q Ensemble 数量 |
| `q_aggregation` | "min" | Q 聚合方式（min 或 mean） |
| `denoised_action_approx` | "one_euler_step_approx" | 干净动作近似方式 |
| `apply_jacobian` | False | 是否应用 Jacobian 链式法则 |
| `horizon_length` | 1 | n-step 或 action chunk 步数 |
| `action_chunking` | False | 是否展平 H 步动作为一个大向量 |
| `guidance_weight` | 需要调参 | 推理时 Q 引导的强度 |

---

### 八、QGF 与其他方法的对比

| 维度 | QGF | QFQL | QGF-Jacobian | GradStep | FQL | EDP |
|------|-----|------|-------------|----------|-----|-----|
| 训练方式 | BC + IQL | BC + IQL | BC + IQL | BC + IQL | Flow + Q 联合 | Flow + Q 联合 |
| Actor 接收 RL 梯度 | 否 | 否 | 否 | 否 | 是 | 是 |
| 推理时 Q 评估点 | 近似干净 a' | 噪声 a_t | 近似干净 a' | 完全去噪后 | 一步 | 标准去噪 |
| 链式法则 | 丢弃 | 丢弃 | 完整 | 不需要 | 蒸馏 | 单步近似 |
| 方差 | 低 | 低 | 高 | 低 | 中 | 中 |
| 推理速度 | 慢（N 步 + Q 梯度） | 慢 | 很慢（+Jacobian） | 慢 | 快（1 步） | 中 |
| 调参复杂度 | 需调 w | 需调 w | 需调 w | 需调 w | 需调 α, bc_weight | 需调多超参 |

---

### 九、完整端到端流程

```
1. 数据准备
   └── Dataset.sample_sequence(batch_size, H, discount)
       └── 返回 (obs, actions, rewards, masks, valid, next_obs)

2. 训练循环（offline_steps 步）
   └── agent.update(batch)
       ├── policy_loss → Adam 更新 policy
       ├── critic_loss → Adam 更新 critic
       ├── target_update → 软更新 target_critic
       └── value_loss → Adam 更新 value

3. 评估（每 eval_interval 步）
   └── eval_with_test_time_guidance(agent, env, guidance_weights, ...)
       └── 对每个 guidance_weight w：
           └── run_episodes(agent, env, guidance_weight=w)
               └── 每步调用 agent.sample_actions(obs, guidance_weight=w)
                   ├── 初始化噪声 a ~ N(0,I)
                   ├── for t in 0..N-1:
                   │   ├── v_bc = policy(obs, a, t)
                   │   ├── a' = clip(a + (1-t)·v_bc, -1, 1)  # 干净动作近似
                   │   ├── qgrad = ∇Q(obs, a')               # Q 梯度
                   │   └── a += (v_bc + w·qgrad) · Δt        # 引导更新
                   └── 返回 clip(a, -1, 1)
```

---

### 十、核心设计动机总结

| 问题 | QGF 的解决方案 |
|------|---------------|
| Actor RL 梯度会穿过去噪链（BPTT 不稳定） | **训练时永远不给 Actor RL 梯度**，只用 BC 训练 |
| 在噪声动作 a_t 上求 Q 梯度不可靠（OOD） | **单步 Euler 近似干净动作**，在近干净动作上求 Q 梯度 |
| 完整 Jacobian 链式法则方差大 | **默认丢弃 Jacobian**，以低方差换取有偏近似 |
| 同一 checkpoint 要支持多种推理策略 | 通过 `denoised_action_approx` 和 `apply_jacobian` 配置切换 |
| 需要从离线数据中学习更好的策略 | **推理时用 Q 引导**，实现"一次训练，多种推理策略" |

---

## 总结

QGF（Q-Guided Flow）是一种**测试时 RL 方法**，其核心思想是将策略改进从训练阶段移到推理阶段。训练时，Actor 只用 BC 训练（模仿数据），Critic 和 Value 用 IQL 独立训练（不联动 Actor）。推理时，在去噪的每一步用 Q 函数的梯度引导速度场，使生成的动作既有 BC 的"合理性"，又有 Q 的"高质量"。

这种方式避免了传统 Actor-Critic 方法中 BPTT 穿过去噪链的稳定性问题，也避免了在 OOD 噪声动作上直接求 Q 梯度的不可靠性。通过单步 Euler 近似干净动作并丢弃 Jacobian，QGF 实现了低方差、高可靠性的测试时引导，是离线 RL 中一个优雅且实用的方案。