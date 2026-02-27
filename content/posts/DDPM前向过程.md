+++
date = '2026-02-27T08:40:58+08:00'
draft = false
title = 'DDPM前向过程'
+++


# DDPM 前向过程

前向过程将数据 $x^{(0)} \sim p_\text{data}$ 逐步加噪，得到 $x^{(1)}, \ldots, x^{(T)}$，最终 $x^{(T)}$ 近似标准高斯。下面给出定义与**单步转移**、**多步边际** $q(x^{(t)} \mid x^{(0)})$ 的闭式推导，以及**重参数化**形式。

---

## 1. 定义与记号

- **前向过程**是马尔可夫链：
  $$x^{(0)} \rightarrow x^{(1)} \rightarrow \cdots \rightarrow x^{(T)}.$$
- 固定方差序列 $\beta_1, \ldots, \beta_T \in (0,1)$，令
  $$\alpha_t = 1 - \beta_t, \qquad \bar\alpha_t = \prod_{s=1}^{t} \alpha_s.$$
  （约定 $\bar\alpha_0 = 1$。）

---

## 2. 单步转移（定义）

前向的**单步转移**取为均值缩小、方差固定的高斯：

$$
q(x^{(t)} \mid x^{(t-1)}) = \mathcal{N}\big(x^{(t)};\ \sqrt{1-\beta_t}\, x^{(t-1)},\ \beta_t \mathbf{I}\big) = \mathcal{N}\big(x^{(t)};\ \sqrt{\alpha_t}\, x^{(t-1)},\ \beta_t \mathbf{I}\big).
$$

等价地，可写成**重参数化**形式（便于采样与推导）：

$$
x^{(t)} = \sqrt{\alpha_t}\, x^{(t-1)} + \sqrt{\beta_t}\, \varepsilon_{t}, \qquad \varepsilon_t \sim \mathcal{N}(0, \mathbf{I}),\ \text{i.i.d.}
$$

---

## 3. 多步边际 $q(x^{(t)} \mid x^{(0)})$ 的闭式

我们希望对中间步积分，得到**从 $x^{(0)}$ 一步到 $x^{(t)}$** 的分布 $q(x^{(t)} \mid x^{(0)})$，并证明它仍是**单高斯**且**有闭式**。

### 3.1 递推：$x^{(t)}$ 用 $x^{(0)}$ 与噪声表示

由单步形式反复代入：

$$
\begin{aligned}
x^{(1)} &= \sqrt{\alpha_1}\, x^{(0)} + \sqrt{\beta_1}\, \varepsilon_1, \\
x^{(2)} &= \sqrt{\alpha_2}\, x^{(1)} + \sqrt{\beta_2}\, \varepsilon_2 = \sqrt{\alpha_2\alpha_1}\, x^{(0)} + \sqrt{\alpha_2\beta_1}\, \varepsilon_1 + \sqrt{\beta_2}\, \varepsilon_2, \\
&\vdots
\end{aligned}
$$

一般地，$x^{(t)}$ 可写成 $x^{(0)}$ 与 $\varepsilon_1,\ldots,\varepsilon_t$ 的线性组合。由于各 $\varepsilon_s$ 独立且与 $x^{(0)}$ 独立，该线性组合仍为**高斯**，只需求其均值与方差。下面推导中会自然出现 $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$。

### 3.2 均值与 $\sqrt{\bar\alpha_t}$ 的推导

记 $x^{(t)}$ 中 $x^{(0)}$ 的系数为 $c_t$。由递推：
- $x^{(1)} = \sqrt{\alpha_1}\, x^{(0)} + \cdots$，故 $c_1 = \sqrt{\alpha_1}$；
- $x^{(t)} = \sqrt{\alpha_t}\, x^{(t-1)} + \sqrt{\beta_t}\, \varepsilon_t$，若 $x^{(t-1)}$ 中 $x^{(0)}$ 的系数为 $c_{t-1}$，则 $x^{(t)}$ 中 $x^{(0)}$ 的系数为 $c_t = \sqrt{\alpha_t}\, c_{t-1}$。

因此
$$
c_t = \sqrt{\alpha_t}\, c_{t-1} = \sqrt{\alpha_t\,\alpha_{t-1}}\, c_{t-2} = \cdots = \sqrt{\alpha_t \cdots \alpha_1} = \sqrt{\prod_{s=1}^{t}\alpha_s} = \sqrt{\bar\alpha_t}.
$$

$\mathbb{E}[\varepsilon_s]=0$，故
$$
\mathbb{E}[x^{(t)} \mid x^{(0)}] = \sqrt{\bar\alpha_t}\, x^{(0)}.
$$

### 3.3 方差与 $(1-\bar\alpha_t)$ 的推导

记 $v_t = \mathrm{Var}(x^{(t)} \mid x^{(0)})$（标量方差，各维度独立且相同）。由 $x^{(t)} = \sqrt{\alpha_t}\, x^{(t-1)} + \sqrt{\beta_t}\, \varepsilon_t$，且 $x^{(t-1)}$ 与 $\varepsilon_t$ 在给定 $x^{(0)}$ 下独立，故
$$
v_t = \alpha_t\, v_{t-1} + \beta_t.
$$

利用 $\beta_t = 1 - \alpha_t$，代入得
$$
v_t = \alpha_t\, v_{t-1} + (1 - \alpha_t).
$$

**递推初值**：$x^{(0)}$ 给定无随机性，$v_0 = 0$。可验证 $v_1 = \beta_1 = 1 - \alpha_1 = 1 - \bar\alpha_1$。

**归纳**：设 $v_{t-1} = 1 - \bar\alpha_{t-1}$，则
$$
v_t = \alpha_t\, (1 - \bar\alpha_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t\bar\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\bar\alpha_{t-1} = 1 - \bar\alpha_t.
$$

因此
$$
\mathrm{Var}(x^{(t)} \mid x^{(0)}) = (1 - \bar\alpha_t)\, \mathbf{I}.
$$

于是：

$$
\boxed{
q(x^{(t)} \mid x^{(0)}) = \mathcal{N}\big(x^{(t)};\ \sqrt{\bar\alpha_t}\, x^{(0)},\ (1-\bar\alpha_t)\,\mathbf{I}\big).
}
$$

即：**给定 $x^{(0)}$ 时，$x^{(t)}$ 是单高斯，均值 $\sqrt{\bar\alpha_t}\, x^{(0)}$，方差 $(1-\bar\alpha_t)\mathbf{I}$**，与中间步无关，**有闭式、可采样、可求密度**。

---

## 4. 重参数化形式（采样与训练用）

将 $x^{(t)}$ 写成**仅依赖 $x^{(0)}$ 与一个标准高斯噪声 $\epsilon$** 的形式，便于实现采样与后续对 $\epsilon$ 的回归：

$$
x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \mathbf{I}).
$$

等价性：右边均值为 $\sqrt{\bar\alpha_t}\, x^{(0)}$，方差为 $(1-\bar\alpha_t)\mathbf{I}$，与 $q(x^{(t)} \mid x^{(0)})$ 一致；且**单步加噪**与**多步一次加噪**在分布上等价（给定 $x^{(0)}$），因此训练时可对 $(x^{(0)}, t)$ 随机采样，再按上式生成 $x^{(t)}$，让网络预测对应的 $\epsilon$（即 $\epsilon_\theta(x^{(t)}, t)$）。

---

## 5. 小结

| 量 | 形式 |
|----|------|
| 单步转移 | $q(x^{(t)} \mid x^{(t-1)}) = \mathcal{N}(\sqrt{\alpha_t}\, x^{(t-1)},\ \beta_t \mathbf{I})$ |
| 多步边际 | $q(x^{(t)} \mid x^{(0)}) = \mathcal{N}(\sqrt{\bar\alpha_t}\, x^{(0)},\ (1-\bar\alpha_t)\mathbf{I})$ |
| 重参数化 | $x^{(t)} = \sqrt{\bar\alpha_t}\, x^{(0)} + \sqrt{1-\bar\alpha_t}\, \epsilon,\ \epsilon\sim\mathcal{N}(0,\mathbf{I})$ |

- $\bar\alpha_t$ 随 $t$ 增大而减小，故 $\sqrt{\bar\alpha_t}$ 变小、$\sqrt{1-\bar\alpha_t}$ 变大，$x^{(t)}$ 中噪声占比增加；当 $t=T$ 且 $\bar\alpha_T \approx 0$ 时，$x^{(T)}$ 近似 $\mathcal{N}(0,\mathbf{I})$。
- 前向过程**不包含可学习参数**；反向过程才用神经网络拟合 $q(x^{(t-1)} \mid x^{(t)}, x^{(0)})$ 的近似 $p_\theta(x^{(t-1)} \mid x^{(t)})$。
