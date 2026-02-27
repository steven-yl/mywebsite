+++
date = '2026-02-27T08:40:58+08:00'
draft = false
title = 'ELBO'
+++



# ELBO 与变分方法详解

## 一、动机：为什么需要 ELBO？

在隐变量生成模型中，观测 $x$ 由隐变量 $z$ 通过解码器 $p_\phi(x|z)$ 生成，先验为 $p(z)$。**边际似然（证据）**为：

$$
\log p_\phi(x) = \log \int p_\phi(x|z)\, p(z)\, dz
$$

- **理想**：用最大似然估计（MLE）最大化 $\log p_\phi(x)$ 来学习解码器参数 $\phi$。
- **障碍**：积分对 $z$ 在高维、非线性解码器下**难以计算**（intractable），直接 MLE 不可行。
- **变分思路**：不直接算 $\log p_\phi(x)$，而是构造一个**可计算的下界**，通过最大化该下界间接增大 $\log p_\phi(x)$。这个下界就是 **ELBO**（Evidence Lower BOund，证据下界）。

---

## 二、变分方法核心：近似后验 $q(z|x)$

真实后验为：

$$
p_\phi(z|x) = \frac{p_\phi(x|z)\, p(z)}{p_\phi(x)}
$$

分母 $p_\phi(x)$ 含难算积分，故 $p_\phi(z|x)$ 也无法直接得到。

**变分方法**引入一族由参数 $\theta$ 控制的分布 $q_\theta(z|x)$（如高斯，由编码器输出 $\mu(x),\sigma(x)$），用 $q_\theta(z|x)$ **近似** $p_\phi(z|x)$。“变分”即在该族内调整 $\theta$，使 $q$ 尽量接近真实后验。

---

## 三、ELBO 的推导

下面从 $\log p_\phi(x)$ 出发，对**任意**与 $x$ 有关的分布 $q(z|x)$（不要求等于真实后验 $p_\phi(z|x)$），逐步推出 ELBO 与 KL 的分解。推导中只用到概率的乘法公式与期望的线性性，不涉及对 $z$ 的难算积分。

### 3.1 第一步：把常数放进对 $q$ 的期望里

$\log p_\phi(x)$ 与 $z$ 无关，故对任意分布 $q(z|x)$ 有：

$$
\log p_\phi(x)
= \int q(z|x)\, \log p_\phi(x)\, dz
= \mathbb{E}_{z \sim q(z|x)} \left[ \log p_\phi(x) \right].
$$

这样做的目的：后面会把被积函数换成含 $z$ 的式子，从而出现只依赖 $q$ 的期望，而不再出现 $p_\phi(x)$ 的积分。

### 3.2 第二步：用联合分布与后验表示

由条件概率 $p_\phi(z|x) = p_\phi(x,z)\,/\,p_\phi(x)$ 得 $p_\phi(x) = p_\phi(x,z)\,/\,p_\phi(z|x)$，代入上式：

$$
\log p_\phi(x)
= \mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{p_\phi(x,z)}{p_\phi(z|x)} \right].
$$

这里 $\log$ 里出现了联合 $p_\phi(x,z)$ 与后验 $p_\phi(z|x)$；后者仍然难算，下一步通过引入 $q$ 把 $p_\phi(z|x)$ 从 $\log$ 里“换掉”。

### 3.3 第三步：分子分母同乘 $q(z|x)$（乘 1 技巧）

在 $\log$ 内分子分母同乘 $q(z|x)$，不改变取值：

$$
\frac{p_\phi(x,z)}{p_\phi(z|x)}
= \frac{p_\phi(x,z)}{q(z|x)} \cdot \frac{q(z|x)}{p_\phi(z|x)}.
$$

于是：

$$
\log p_\phi(x)
= \mathbb{E}_{z \sim q(z|x)} \left[ \log \left( \frac{p_\phi(x,z)}{q(z|x)} \cdot \frac{q(z|x)}{p_\phi(z|x)} \right) \right].
$$

### 3.4 第四步：拆成两项（log 乘积 = 和）

$\log(ab) = \log a + \log b$，且期望线性，故：

$$
\log p_\phi(x)
= \mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{p_\phi(x,z)}{q(z|x)} \right]
+ \mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{q(z|x)}{p_\phi(z|x)} \right].
$$

### 3.5 第五步：认出 KL 散度

第二项正是 $q(z|x)$ 对 $p_\phi(z|x)$ 的 **KL 散度**：

$$
D_{\mathrm{KL}}\bigl( q(z|x) \,\|\, p_\phi(z|x) \bigr)
= \mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{q(z|x)}{p_\phi(z|x)} \right].
$$

由 Jensen 不等式或 Gibbs 不等式可知 KL 散度非负，且当且仅当 $q = p_\phi(z|x)$ 时为零，故：

$$
D_{\mathrm{KL}}\bigl( q(z|x) \,\|\, p_\phi(z|x) \bigr) \geq 0.
$$

### 3.6 结论：证据 = ELBO + KL

记第一项为 ELBO（证据下界），即得：

$$
\boxed{
\log p_\phi(x)
= \underbrace{\mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{p_\phi(x,z)}{q(z|x)} \right]}_{\mathrm{ELBO}}
+ D_{\mathrm{KL}}\bigl( q(z|x) \,\|\, p_\phi(z|x) \bigr)
}
$$

因此：

$$
\log p_\phi(x) \geq \mathrm{ELBO},
$$

等号成立当且仅当 $q(z|x) = p_\phi(z|x)$（几乎处处）。即 **ELBO 是 $\log p_\phi(x)$ 的下界**，且等号越紧说明 $q$ 越接近真实后验。ELBO 中只出现 $p_\phi(x,z)$ 与 $q(z|x)$，**不包含难算的 $p_\phi(x)$ 或 $p_\phi(z|x)$**。

---

## 四、ELBO 的两种等价形式

ELBO 的原始形式为 $\mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{p_\phi(x,z)}{q(z|x)} \right]$。下面通过分解 $p_\phi(x,z)$ 得到两种常用写法。

### 4.1 从联合分布分解开始

由联合概率分解 $p_\phi(x,z) = p_\phi(x|z)\, p(z)$（先验 × 似然），有：

$$
\log \frac{p_\phi(x,z)}{q(z|x)}
= \log p_\phi(x|z) + \log p(z) - \log q(z|x).
$$

对 $q(z|x)$ 取期望（期望的线性性），得：

$$
\mathrm{ELBO}
= \mathbb{E}_{z \sim q(z|x)} \left[ \log p_\phi(x|z) \right]
+ \mathbb{E}_{z \sim q(z|x)} \left[ \log p(z) - \log q(z|x) \right].
$$

### 4.2 形式一：重构项 + KL 正则项（VAE 常用）

第二项 $\mathbb{E}_{z \sim q(z|x)} \left[ \log p(z) - \log q(z|x) \right]$ 正是 **$-D_{\mathrm{KL}}(q(z|x) \| p(z))$**（$q$ 对先验 $p(z)$ 的 KL 的相反数）。因此：

$$
\boxed{
\mathrm{ELBO}
= \mathbb{E}_{z \sim q(z|x)} \left[ \log p_\phi(x|z) \right]
- D_{\mathrm{KL}}\bigl( q(z|x) \,\|\, p(z) \bigr)
}
$$

- **第一项** $\mathbb{E}_{z \sim q(z|x)} \left[ \log p_\phi(x|z) \right]$：**重构对数似然**。从 $q(z|x)$ 采样 $z$，看解码器 $p_\phi(x|z)$ 能否把 $z$ 还原成 $x$；越大重构越好。
- **第二项** $-D_{\mathrm{KL}}(q(z|x) \| p(z))$：**正则项**。约束 $q(z|x)$ 不要偏离先验 $p(z)$ 太远，避免后验塌缩或过于任意。

### 4.3 形式二：证据与后验近似误差

在第三节已得到 $\log p_\phi(x) = \mathrm{ELBO} + D_{\mathrm{KL}}(q \| p_\phi(z|x))$，移项即：

$$
\mathrm{ELBO}
= \log p_\phi(x) - D_{\mathrm{KL}}\bigl( q(z|x) \,\|\, p_\phi(z|x) \bigr).
$$

因此：
- 最大化 ELBO 等价于同时 **增大 $\log p_\phi(x)$** 与 **减小 $q$ 与真实后验的 KL**。
- $q$ 越接近 $p_\phi(z|x)$，KL 越小，ELBO 越紧（越接近 $\log p_\phi(x)$）。

---

## 五、为什么 ELBO 可计算、可优化？

| 对象 | 是否含难算积分 |
|------|----------------|
| $\log p_\phi(x)$ | 含 $\int p_\phi(x \mid z)p(z)\,dz$，难算 |
| ELBO | 仅含 $\mathbb{E}_{z \sim q(z \mid x)}[\cdots]$，无难解积分 |

ELBO 只涉及对 $q(z|x)$ 的期望：

1. **采样**：从 $q_\theta(z|x)$ 采样 $z$（VAE 中用重参数化 $z = \mu + \sigma \odot \varepsilon$，$\varepsilon \sim \mathcal{N}(0,I)$，便于对 $\theta$ 求导）。
2. **估计**：用蒙特卡洛 $\frac{1}{N}\sum_i \log p_\phi(x|z^{(i)})$ 估计第一项；当 $q$、$p$ 为高斯时，KL 项可解析写出，无需采样。
3. **梯度**：对 $\phi,\theta$ 可微，用反向传播即可训练解码器与编码器。

因此 **变分方法** 通过引入 $q(z|x)$，把“难算的边际积分”转化为“对 $q$ 的期望”，得到可优化的 ELBO。

---

## 六、与直接 MLE + 蒙特卡洛的对比

- **直接 MLE**：最大化 $\log p_\phi(x)$，梯度为 $\nabla_\phi \log p_\phi(x)$，仍含对 $z$ 的积分；若用先验 $p(z)$ 采样做蒙特卡洛，绝大多数 $z$ 对当前 $x$ 几乎无贡献，**梯度估计方差极大**，优化不稳定。
- **变分 + ELBO**：用 $q(z|x)$ 在“对当前 $x$ 重要的 $z$”附近采样，**有效样本多、方差小**，梯度估计可行，故 VAE 采用最大化 ELBO 而非直接 MLE。

---

## 七、小结

- **变分方法**：用可参数量 $q_\theta(z|x)$ 近似难算的后验 $p_\phi(z|x)$，并构造 $\log p_\phi(x)$ 的可算下界。
- **ELBO**：

  $$\mathcal{L}(\phi,\theta) = \mathbb{E}_{z \sim q_\theta(z|x)} \left[ \log p_\phi(x|z) \right] - D_{\mathrm{KL}}\bigl( q_\theta(z|x) \,\|\, p(z) \bigr)$$

- **训练**：最大化 ELBO，即同时近似 MLE（提高 $\log p_\phi(x)$）和学好近似后验 $q(z|x)$。VAE 即在此框架下，用神经网络表示 $p_\phi(x|z)$ 与 $q_\theta(z|x)$ 并优化上述目标。
