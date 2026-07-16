---
title: "QAM（Q-guided Action Matching）算法技术文档"
subtitle: ""
date: 2026-07-14T18:00:00+08:00
draft: false
authors: [Steven]
description: "QAM 算法背景、目标策略、训练与推理流程的技术文档。"
summary: "QAM（Q-guided Action Matching）算法技术文档。"
tags: [diffusion/flow, qgf]
categories: [docs qgf]
series: [qgf-docs]
weight: 12
series_weight: 12
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# QAM（Q-guided Action Matching）算法技术文档

## 目录

1. [背景与问题设定](#一背景与问题设定)  
2. [基础概念：强化学习与离线 RL](#二基础概念强化学习与离线-rl)  
3. [基础概念：Flow Matching 生成模型](#三基础概念flow-matching-生成模型)  
4. [为什么需要 KL 正则化？](#四为什么需要-kl-正则化)  
5. [KL 正则化最优策略的推导](#五kl-正则化最优策略的推导)  
6. [QAM 的核心思想](#六qam-的核心思想)  
7. [伴随状态（Adjoint State）的数学推导](#七伴随状态adjoint-state的数学推导)  
8. [参考动力学 $2v-\frac{x}{t}$ 的来源与推导](#八参考动力学-2v-fracxt-的来源与推导)  
9. [伴随匹配损失（Adjoint Matching Loss）](#九伴随匹配损失adjoint-matching-loss)  
10. [前向模拟：去噪链的构造](#十前向模拟去噪链的构造)  
11. [完整的训练流程](#十一完整的训练流程)  
12. [推理过程](#十二推理过程)  
13. [网络结构与超参数](#十三网络结构与超参数)  
14. [QAM 与相关方法的对比](#十四qam-与相关方法的对比)  
15. [实现要点与常见误区](#十五实现要点与常见误区)  
16. [总结](#十六总结)  

---

## 一、背景与问题设定

### 1.1 我们要解决什么问题？

我们希望训练一个智能体（例如机器人）完成任务，但又不能在线反复试错。原因包括：

- 真机试错成本高
- 可能损坏设备
- 实验周期长
- 数据只能来自已有日志

这就形成了 **离线强化学习（Offline RL）** 的问题：

> 仅利用一批固定的历史数据，学习一个高质量策略，甚至超越数据中的行为。

---

### 1.2 数据形式

离线 RL 数据集通常由大量转移样本构成：

$$
\mathcal D = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^N
$$

其中：

- $s$：当前状态
- $a$：执行的动作
- $r$：即时奖励
- $s'$：下一个状态

有时还会包含：

- `mask`：是否终止
- `valid`：是否有效
- 多步 horizon 信息

---

### 1.3 离线 RL 的核心难点：OOD 问题

离线 RL 最大的问题是 **分布外（Out-of-Distribution, OOD）** 动作。

Q 函数只能在数据覆盖区域内可靠估计。如果策略输出了训练数据中几乎没有出现过的动作，那么：

- critic 可能错误高估它们
- actor 会利用这些高估
- 训练进入恶性循环

因此，离线 RL 不能简单做：

$$
\max_\pi \mathbb E_{a\sim \pi(\cdot|s)}[Q(s,a)]
$$

必须加某种“别跑太远”的约束。

---

## 二、基础概念：强化学习与离线 RL

### 2.1 策略

策略 $\pi(a|s)$ 表示在状态 $s$ 下选择动作 $a$ 的条件分布。

- **确定性策略**：直接输出一个动作
- **随机策略**：输出一个分布

QAM 中使用的是**随机策略**，因为 Flow Matching 能自然表示复杂、多模态动作分布。

---

### 2.2 Q 函数

$$
Q(s,a)=\mathbb E\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0=s,a_0=a\right]
$$

表示从状态 $s$ 执行动作 $a$ 后，未来能获得的期望累计回报。

---

### 2.3 V 函数

$$
V(s)=\mathbb E_{a\sim \pi(\cdot|s)}[Q(s,a)]
$$

表示状态 $s$ 在当前策略下的期望价值。

---

### 2.4 优势函数

$$
A(s,a)=Q(s,a)-V(s)
$$

表示动作 $a$ 相比该状态下平均动作有多好。

---

### 2.5 贝尔曼方程

$$
Q(s,a)=r+\gamma \mathbb E_{s'}[V(s')]
$$

或者在 actor-critic 形式里：

$$
Q(s,a)=r+\gamma \mathbb E_{s',a'}[Q(s',a')]
$$

---

### 2.6 IQL 简介

QAM 中 critic 常采用 IQL 风格训练：

1. 用 expectile 回归学习 $V(s)$
2. 用 TD 学习 $Q(s,a)$
3. 不显式依赖当前 actor 的 log-prob

IQL 的优点是适配离线场景，避免了显式策略评估中的不稳定性。

---

## 三、基础概念：Flow Matching 生成模型

### 3.1 为什么用生成模型表示策略？

传统高斯策略往往只能表达单峰分布，而实际任务中可能有多个可行动作模式。

例如：

- 从左侧绕过去
- 从右侧绕过去

二者都可能是高质量动作。  
Flow Matching 可以学习复杂的多模态分布，因此非常适合作为策略。

---

### 3.2 Flow Matching 的基本思想

从一个简单噪声分布出发，学习一个速度场，把噪声逐步“推”到数据分布。

设动作变量为 $a_t$，速度场为 $v_\theta(s,a_t,t)$，则生成过程满足：

$$
\frac{d a_t}{dt}=v_\theta(s,a_t,t), \qquad t\in[0,1]
$$

其中：

- $t=0$：纯噪声
- $t=1$：干净动作

---

### 3.3 线性插值路径（Rectified Flow）

QAM 使用的路径为：

$$
a_t=(1-t)a_0+t a_1
$$

其中：

- $a_0\sim \mathcal N(0,I)$
- $a_1\sim p_{\text{data}}$

因此：

$$
\frac{d a_t}{dt}=a_1-a_0
$$

这说明最优速度场的监督目标是：

$$
v^*(s,a_t,t)=a_1-a_0
$$

---

### 3.4 Behavior Cloning 损失

训练 slow actor 的 BC 损失为：

$$
\mathcal L_{\text{BC}}
=
\mathbb E_{a_0,a_1,t}
\left[
\|v_{\text{slow}}(s,a_t,t)-(a_1-a_0)\|^2
\right]
$$

它让 slow actor 学会数据分布对应的流场。

---

### 3.5 推理时的 Euler 去噪

采样时从噪声动作 $a_0$ 出发，做 $N$ 步 Euler 更新：

$$
a_{t+h}=a_t+h\,v_\theta(s,a_t,t), \qquad h=\frac{1}{N}
$$

最终得到动作 $a_1$。

---

### 3.6 Action Chunking

如果一次要生成 $H$ 步动作，则把多个动作拼接成一个大向量统一生成：

- 输出维度变为 $H\times d_a$
- 一次采样得到连续动作块

这对机器人控制尤其有用。

---

## 四、为什么需要 KL 正则化？

### 4.1 纯 BC 的局限

BC 只能模仿数据，无法主动偏向高回报动作。  
如果数据中混杂很多普通甚至次优行为，BC 也只能学成平均水平。

---

### 4.2 直接最大化 Q 的风险

如果直接训练 actor：

$$
\max_\pi \mathbb E_{a\sim \pi(\cdot|s)}[Q(s,a)]
$$

actor 可能去找 critic 高估的 OOD 动作。  
因为 critic 在数据外不可靠，这会导致性能崩坏。

---

### 4.3 KL 正则化目标

一个经典解决方案是加 KL 约束：

$$
\max_\pi
\;
\mathbb E_{a\sim \pi}[Q(s,a)]
-
\frac{1}{\tau}\mathrm{KL}[\pi(\cdot|s)\|\pi_{\text{BC}}(\cdot|s)]
$$

解释：

- 第一项：追求高 Q
- 第二项：别离数据分布太远

其中 $\tau$ 是逆温度：

- $\tau$ 大：更激进
- $\tau$ 小：更保守

---

## 五、KL 正则化最优策略的推导

我们考虑固定状态 $s$，优化：

$$
\mathcal J[\pi]
=
\int \pi(a|s)Q(s,a)\,da
-\frac{1}{\tau}\int \pi(a|s)\log\frac{\pi(a|s)}{\pi_{\text{BC}}(a|s)}\,da
+\lambda\left(1-\int \pi(a|s)\,da\right)
$$

对 $\pi(a|s)$ 取变分导数：

$$
Q(s,a)-\frac{1}{\tau}\left(\log\frac{\pi(a|s)}{\pi_{\text{BC}}(a|s)}+1\right)-\lambda=0
$$

整理得：

$$
\log\frac{\pi(a|s)}{\pi_{\text{BC}}(a|s)}
=
\tau Q(s,a)-\tau\lambda-1
$$

取指数：

$$
\pi(a|s)=\pi_{\text{BC}}(a|s)\exp(\tau Q(s,a))\cdot C
$$

其中 $C$ 为归一化常数。于是：

$$
\boxed{
\pi^*(a|s)=\frac{1}{Z(s)}\pi_{\text{BC}}(a|s)e^{\tau Q(s,a)}
}
$$

其中：

$$
Z(s)=\int \pi_{\text{BC}}(a|s)e^{\tau Q(s,a)}da
$$

这说明最优策略就是：

> **BC 分布 × Q 的 Boltzmann 重加权**

---

## 六、QAM 的核心思想

### 6.1 要解决的问题

我们知道目标策略应为：

$$
\pi^*(a|s)\propto \pi_{\text{BC}}(a|s)e^{\tau Q(s,a)}
$$

但如何让一个 Flow 模型学到这个分布？

难点在于：

- 不能直接从 $\pi^*$ 采样
- 直接穿过整个去噪链做 BPTT 很不稳定
- 推理时在线算 Q 梯度会非常慢

---

### 6.2 QAM 的核心做法

QAM 采用两套 actor：

1. **slow actor**：学 BC 流场
2. **fast actor**：学 Q 引导修正

核心思想：

> 先用 slow actor 建立“数据分布流线”，再通过伴随状态把终端 Q 偏好反向传播到中间时刻，最后训练 fast actor 拟合这些局部修正。

也就是说：

- slow actor 负责“像数据”
- fast actor 负责“更高 Q”

---

### 6.3 伴随状态的意义

定义终端目标为：

$$
J(a_1)=\tau Q(s,a_1)
$$

伴随状态定义为：

$$
g_t=\frac{\partial J(a_1)}{\partial a_t}
$$

它表示：

> 在中间时刻 $t$，如果把动作变量 $a_t$ 微调一点，最终动作的 Q 会如何变化。

因此，$g_t$ 是中间时刻的“Q 敏感度”。

---

## 七、伴随状态（Adjoint State）的数学推导

### 7.1 去噪链视为离散动力系统

设时间步长为 $h=1/N$，离散时刻 $t_i=ih$。  
前向动力系统写为：

$$
a_{i+1}=F_i(a_i)=a_i+h f_i(a_i)
$$

QAM 中伴随传播采用的参考动力学是：

$$
f_i(a_i)=2v_{\text{slow}}(s,a_i,t_{i+1})-\frac{a_i}{t_{i+1}}
$$

---

### 7.2 终端目标

定义终端目标：

$$
J(a_N)=\tau Q(s,a_N)
$$

令伴随状态为：

$$
g_i:=\frac{\partial J}{\partial a_i}
$$

则终端条件为：

$$
g_N=\tau \nabla_{a_N}Q(s,a_N)
$$

在代码里常用带负号版本：

$$
g_N=-\tau \nabla_{a_N}Q(s,a_N)
$$

这是为了后续匹配损失的符号一致性。

---

### 7.3 伴随递推公式

由链式法则：

$$
g_i
=
\frac{\partial J}{\partial a_i}
=
\frac{\partial J}{\partial a_{i+1}}
\frac{\partial a_{i+1}}{\partial a_i}
=
g_{i+1}\frac{\partial F_i(a_i)}{\partial a_i}
$$

而：

$$
\frac{\partial F_i}{\partial a_i}
=
I+h\frac{\partial f_i}{\partial a_i}
$$

所以：

$$
g_i=g_{i+1}+h\,g_{i+1}\frac{\partial f_i}{\partial a_i}
$$

若写成列向量形式：

$$
g_i
=
g_{i+1}
+
h
\left(
\frac{\partial f_i}{\partial a_i}
\right)^\top
g_{i+1}
$$

这就是离散伴随传播方程。

---

### 7.4 用 VJP 高效计算

不显式构造 Jacobian，而用 JAX 的 VJP：

```python
def fn(xi):
    return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)

vjp = jax.vjp(fn, xs[i])[1](adj)[0]
adj = adj + h * vjp
```

对应数学式：

$$
g_i
=
g_{i+1}
+
h\left(\frac{\partial f_i}{\partial a_i}\right)^\top g_{i+1}
$$

---

### 7.5 为什么伴随状态不是简单的 $\nabla_{a_t}Q(s,a_t)$？

因为 QAM 的目标是终端动作 $a_1$ 的价值，而不是中间状态本身的价值。  
因此：

$$
g_t=\frac{\partial (\tau Q(s,a_1))}{\partial a_t}
$$

是**终端 Q 梯度经过整条去噪动力学反向传播后的结果**，不是直接在 $a_t$ 上评估的 Q 梯度。

---

## 八、参考动力学 $2v-\frac{x}{t}$ 的来源与推导

这是 QAM 中最关键也最容易混淆的部分。

---

### 8.1 从最优速度场定义出发

线性路径：

$$
x_t=(1-t)a_0+t a_1
$$

最优速度场定义为：

$$
v(x,t)=\mathbb E[a_1-a_0\mid x_t=x]
$$

---

### 8.2 用 $a_0$ 的条件均值改写

由路径方程：

$$
x_t=(1-t)a_0+t a_1
$$

解出 $a_1$：

$$
a_1=\frac{x_t-(1-t)a_0}{t}
$$

因此：

$$
a_1-a_0
=
\frac{x_t-a_0}{t}
$$

对条件 $x_t=x$ 取期望：

$$
\boxed{
v(x,t)=\frac{x-\mathbb E[a_0\mid x_t=x]}{t}
}
$$

也就是：

$$
\boxed{
v(x,t)=\frac{x}{t}-\frac{1}{t}\mathbb E[a_0\mid x_t=x]
}
$$

这是正确形式。  
注意：这里是 **减号**，不是加号。

---

### 8.3 条件 score 与 $\mathbb E[a_0\mid x_t=x]$ 的关系

若 $a_0\sim \mathcal N(0,I)$，则给定 $a_1$ 时：

$$
x_t\mid a_1 \sim \mathcal N(t a_1,(1-t)^2I)
$$

因此：

$$
\nabla_x \log p_t(x\mid a_1)
=
-\frac{x-t a_1}{(1-t)^2}
$$

由 Fisher identity：

$$
\nabla_x \log p_t(x)
=
\mathbb E[\nabla_x \log p_t(x\mid a_1)\mid x_t=x]
=
-\frac{x-t\mathbb E[a_1\mid x_t=x]}{(1-t)^2}
$$

又因为：

$$
x=(1-t)\mathbb E[a_0\mid x_t=x]+t\mathbb E[a_1\mid x_t=x]
$$

可得：

$$
x-t\mathbb E[a_1\mid x_t=x]=(1-t)\mathbb E[a_0\mid x_t=x]
$$

于是：

$$
\nabla_x \log p_t(x)
=
-\frac{\mathbb E[a_0\mid x_t=x]}{1-t}
$$

即：

$$
\boxed{
\mathbb E[a_0\mid x_t=x]=-(1-t)\nabla_x\log p_t(x)
}
$$

---

### 8.4 将速度场写成 drift + score 形式

把上式代回：

$$
v(x,t)
=
\frac{x}{t}-\frac{1}{t}\mathbb E[a_0\mid x_t=x]
$$

得到：

$$
v(x,t)
=
\frac{x}{t}+\frac{1-t}{t}\nabla_x\log p_t(x)
$$

所以：

$$
\boxed{
v(x,t)=\frac{x}{t}+\frac{1-t}{t}\nabla_x\log p_t(x)
}
$$

这说明：

- $\frac{x}{t}$：drift 项
- $\frac{1-t}{t}\nabla_x\log p_t(x)$：score 项

---

### 8.5 推导 QAM 的参考动力学

QAM 使用：

$$
f_{\text{ref}}(x,t)=2v(x,t)-\frac{x}{t}
$$

代入上式：

$$
2v(x,t)-\frac{x}{t}
=
2\left(
\frac{x}{t}+\frac{1-t}{t}\nabla_x\log p_t(x)
\right)-\frac{x}{t}
$$

化简：

$$
\boxed{
2v(x,t)-\frac{x}{t}
=
\frac{x}{t}+2\frac{1-t}{t}\nabla_x\log p_t(x)
}
$$

因此这个参考动力学等价于：

> 保留原始 drift 项，并将 score 项放大 2 倍。

---

### 8.6 直观解释

为什么这样设计？

因为 QAM 不是只想跟随 BC 生成流，而是想在 BC 分布几何上注入终端 Q 信息。

KL 正则化最优策略满足：

$$
\log \pi^*(a|s)=\log \pi_{\text{BC}}(a|s)+\tau Q(s,a)-\log Z(s)
$$

对动作求导：

$$
\nabla_a \log \pi^*(a|s)
=
\nabla_a \log \pi_{\text{BC}}(a|s)+\tau \nabla_a Q(s,a)
$$

这里第一项就是 BC 分布的 score，第二项是 Q 引导。  
因此，伴随传播应沿着 BC 的 score 几何来传播 Q 梯度。  
$2v-\frac{x}{t}$ 正是把 slow actor 的流场重写成“更显式的 score 传播形式”。

---

### 8.7 为什么不用 $v$ 而用 $2v-\frac{x}{t}$？

如果直接用 $v$，传播的是普通生成速度的局部敏感度。  
而 QAM 希望传播的是：

- BC 分布的几何结构
- 终端 Q 势函数诱导出的修正方向

因此更自然的是沿 score-form reference flow 来传播，而不是沿原始生成 ODE 直接传播。

---

## 九、伴随匹配损失（Adjoint Matching Loss）

### 9.1 核心思想

有了伴随状态 $g_t$，我们就知道每个时刻该往哪个方向修正速度场才能提高终端 Q。

于是训练 fast actor 去拟合这个局部修正。

---

### 9.2 非残差形式

若 fast actor 输出完整速度 $v_{\text{fine}}$，则修正量为：

$$
\delta v=v_{\text{fine}}-v_{\text{slow}}
$$

匹配损失为：

$$
\boxed{
\mathcal L_{\text{adj}}
=
\mathbb E_t
\left[
\left\|
\frac{2\delta v}{\sigma_t}+\sigma_t g_t
\right\|^2
\right]
}
$$

---

### 9.3 残差形式

若 fast actor 直接输出残差 $v_{\text{fast}}$，则：

$$
\delta v=v_{\text{fast}}
$$

损失为：

$$
\mathcal L_{\text{adj}}
=
\mathbb E_t
\left[
\left\|
\frac{2v_{\text{fast}}}{\sigma_t}+\sigma_t g_t
\right\|^2
\right]
$$

---

### 9.4 最优解的解析形式

令梯度为零：

$$
\frac{2\delta v}{\sigma_t}+\sigma_t g_t=0
$$

得到：

$$
\boxed{
\delta v=-\frac{\sigma_t^2}{2}g_t
}
$$

若代码中 $g_t=-\tau \nabla Q$，则：

$$
\delta v
=
\frac{\tau \sigma_t^2}{2}\nabla Q
$$

所以最优速度修正就是沿着提升 Q 的方向。

---

### 9.5 $\sigma_t$ 的含义

QAM 中常用：

$$
\sigma_t=\sqrt{\frac{2(1-t+h)}{t+h}}
$$

直观理解：

- $t$ 小时噪声大，允许更强修正
- $t$ 大时接近终点，修正应更保守

---

## 十、前向模拟：去噪链的构造

### 10.1 带噪声前向模拟

在训练阶段，QAM 前向模拟通常是：

$$
a_{t+h}
=
a_t
+
h\left(2v(a_t,t)-\frac{a_t}{t+h}\right)
+
\sqrt h\,\sigma_t\epsilon
$$

其中：

$$
\epsilon\sim\mathcal N(0,I)
$$

这相当于一个离散 SDE。

---

### 10.2 为什么加噪声？

加入噪声有三个作用：

1. 覆盖更多轨迹
2. 避免只在单一路径上估计伴随
3. 更接近 score-based stochastic flow 的几何结构

---

### 10.3 最后一步用纯 ODE

最后一步通常写成：

$$
a_{1}=a_{1-h}+h\,v_{\text{slow}}(s,a_{1-h},1-h)
$$

不再加噪声，以确保终点动作足够干净，便于计算 Q 梯度。

---

## 十一、完整的训练流程

### 11.1 总损失

QAM 总损失可写为：

$$
\boxed{
\mathcal L_{\text{QAM}}
=
\mathcal L_{\text{BC}}
+
\mathcal L_{\text{adj}}
+
\mathcal L_{\text{optional}}
}
$$

其中 $\mathcal L_{\text{optional}}$ 可以是 FQL 蒸馏或 edit policy 的额外项。

---

### 11.2 每个 batch 的训练步骤

1. 采样 batch  
   $$
   (s,a,r,s',mask,valid)
   $$

2. 更新 critic  
   - IQL 或 DDPG 风格

3. 更新 value  
   - 若启用 IQL

4. 更新 slow actor  
   - 最小化 BC flow loss

5. 前向模拟参考轨迹  
   - 存储 $\{a_t\}$

6. 终端计算 Q 梯度  
   - 初始化 $g_1$

7. 沿参考动力学反向传播伴随  
   - 得到 $\{g_t\}$

8. 用伴随匹配损失训练 fast actor

9. 软更新 target 网络

---

## 十二、推理过程

### 12.1 标准 flow 推理

从噪声出发，执行多步 Euler：

$$
a_{t+h}=a_t+h\,v_{\text{policy}}(s,a_t,t)
$$

其中 $v_{\text{policy}}$ 可能是：

- $v_{\text{fast}}+v_{\text{slow}}$（残差模式）
- 或 fast actor 的完整输出

---

### 12.2 Best-of-N

生成多个候选动作：

$$
a^{(1)},\dots,a^{(N)}
$$

再用 critic 评分：

$$
\hat a=\arg\max_i Q(s,a^{(i)})
$$

这种方法能显著提升采样质量。

---

### 12.3 推理和训练的区别

- 训练时：需要 Q 梯度、伴随传播
- 推理时：不需要 Q 梯度，只做前向生成

这是 QAM 相比 QGF 的关键优势。

---

## 十三、网络结构与超参数

### 13.1 网络模块

- `critic`：Q 网络 ensemble
- `target_critic`：目标 Q 网络
- `value`：V 网络
- `actor_slow`：BC 流场
- `target_actor_slow`：slow actor 的 EMA
- `actor_fast`：Q 引导修正流场
- `one_step_actor`：一步蒸馏可选
- `edit_actor`：编辑策略可选

---

### 13.2 核心超参数

- `inv_temp`：$\tau$，Q 引导强度
- `flow_steps`：去噪步数
- `residual`：是否残差参数化
- `target_actor`：是否用 target slow actor 传播伴随
- `clip_adj`：是否裁剪伴随
- `fql_alpha`：是否蒸馏一步策略
- `discount`：$\gamma$
- `expectile`：IQL 中 value 回归分位

---

## 十四、QAM 与相关方法的对比

### 14.1 与 BC

- BC：只学数据分布
- QAM：在 BC 基座上注入 Q 偏好

---

### 14.2 与 QGF

- QGF：推理时加 Q 梯度
- QAM：训练时把 Q 信息写进参数

结果：

- QGF 推理慢
- QAM 推理快

---

### 14.3 与 FBRAC / BPTT 方法

- FBRAC：直接穿过整条链反向传播
- QAM：先传播状态梯度，再局部拟合

因此 QAM 更稳定。

---

### 14.4 与 EDP

- EDP：通常单步近似 Q 修正
- QAM：整条链伴随传播

因此 QAM 的 Q 引导更精确。

---

## 十五、实现要点与常见误区

### 15.1 最常见的符号错误

正确公式是：

$$
v(x,t)=\frac{x-\mathbb E[a_0\mid x_t=x]}{t}
$$

不是

$$
\frac{x}{t}+\frac{1-t}{t}\mathbb E[a_0\mid x_t=x]
$$

---

### 15.2 score 符号要统一

如果定义：

$$
s_t(x)=\nabla_x\log p_t(x)
$$

则：

$$
v(x,t)=\frac{x}{t}+\frac{1-t}{t}s_t(x)
$$

若定义 $s_t(x)=-\nabla_x\log p_t(x)$，则符号会翻转。

---

### 15.3 伴随不是中间 Q 的梯度

伴随状态是：

$$
g_t=\frac{\partial (\tau Q(s,a_1))}{\partial a_t}
$$

不是：

$$
\nabla_{a_t}Q(s,a_t)
$$

这两者不能混淆。

---

### 15.4 为什么伴随传播沿 slow actor 而不是 fast actor？

因为：

1. slow actor 更稳定
2. 理论上它代表 BC 基座
3. 避免 fast actor 自己对自己产生病态反馈

---

## 十六、总结

QAM 的核心可以概括为一句话：

> 用 Flow Matching 学习数据分布的生成流，用伴随状态把终端 Q 偏好沿 BC 参考流反向传播，再用匹配损失把这种偏好蒸馏进 fast actor。

其核心公式包括：

### 1. KL 正则化最优策略
$$
\pi^*(a|s)\propto \pi_{\text{BC}}(a|s)e^{\tau Q(s,a)}
$$

### 2. BC flow 损失
$$
\mathcal L_{\text{BC}}
=
\mathbb E\left[\|v_{\text{slow}}-(a_1-a_0)\|^2\right]
$$

### 3. 伴随终端条件
$$
g_1=-\tau \nabla_{a_1}Q(s,a_1)
$$

### 4. 参考动力学
$$
f_{\text{ref}}(x,t)=2v_{\text{slow}}(x,t)-\frac{x}{t}
$$

### 5. 伴随传播
$$
g_t
=
g_{t+h}
+
h\,
\mathrm{vjp}\!\left(f_{\text{ref}},g_{t+h}\right)
$$

### 6. 伴随匹配损失
$$
\mathcal L_{\text{adj}}
=
\mathbb E_t
\left[
\left\|
\frac{2\delta v}{\sigma_t}+\sigma_t g_t
\right\|^2
\right]
$$

QAM 的优点在于：

- 避免整条链 BPTT
- 推理时无需 Q 梯度
- 能在多模态 flow 策略中注入 Q 引导
- 更适合高维连续动作离线 RL

代价是：

- 实现更复杂
- 计算量更大
- 超参数更多

但总体而言，QAM 提供了一条非常有代表性的路线：

> **在离线 RL 中，把“数据分布约束”和“价值提升偏好”统一到同一个生成流框架里。**

如果你愿意，我下一步可以继续把这份文档再升级成下面两种版本之一：

1. **论文风格版**：更严谨、公式更完整、适合写到 README / tech report  
2. **教学讲义版**：更通俗、配图思路更强、适合给新人培训使用