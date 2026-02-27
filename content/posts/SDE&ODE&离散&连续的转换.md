+++
date = '2026-02-27T08:40:58+08:00'
draft = false
title = 'SDE&ODE&离散&连续的转换'
+++



# SDE 与 ODE 公式总览

本文整理：**Flow（ODE）** 与 **Diffusion（SDE）** 的**前向 / 反向**动力学、**密度演化方程**，以及 **SDE 与 ODE 的相互转换**（保持相同边际分布 $p_t$）。约定：**前向** = 数据 → 噪声/先验（$t$ 增加），**反向** = 噪声/先验 → 数据（$t$ 减少或反向时间）。

---

## 一、基本形式（单时间方向）

### 1. Flow（确定性流，ODE）

**动力学**：仅漂移，无随机项
$$
\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t.
$$

**密度演化**（连续性方程）：$X_t \sim p_t$ 时
$$
\partial_t p_t = -\nabla \cdot (u_t\, p_t).
$$

---

### 2. Diffusion（随机扩散，SDE）

**动力学**：漂移 $\mu_t$ + 扩散系数 $\sigma_t$
$$
\mathrm{d}X_t = \mu_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t.
$$

**密度演化**（Fokker–Planck）：$X_t \sim p_t$ 时
$$
\partial_t p_t = -\nabla \cdot (\mu_t\, p_t) + \frac{\sigma_t^2}{2}\,\Delta p_t.
$$

---

## 二、Diffusion：前向 SDE 与反向 SDE

### 2.1 前向扩散 SDE（数据 → 噪声）

时间 $t \in [0,T]$，$X_0 \sim p_{\mathrm{data}}$，$X_T$ 近似噪声（如标准高斯）。通式：
$$
\mathrm{d}X_t = f_t(X_t)\,\mathrm{d}t + g_t\,\mathrm{d}W_t.
$$

常见取法（如 VP-SDE）：$f_t(x) = -\frac{\beta_t}{2}x$，$g_t = \sqrt{\beta_t}$，即
$$
\mathrm{d}X_t = -\frac{\beta_t}{2}\,X_t\,\mathrm{d}t + \sqrt{\beta_t}\,\mathrm{d}W_t.
$$

前向密度 $p_t$ 满足 Fokker–Planck（$f_t \equiv \mu_t$，$g_t \equiv \sigma_t$）：
$$
\partial_t p_t = -\nabla \cdot (f_t\, p_t) + \frac{g_t^2}{2}\,\Delta p_t.
$$

### 2.2 反向扩散 SDE（噪声 → 数据）

时间反向时，存在**反向 SDE**，其解在反向时间下边际分布与“前向 $p_t$ 在反向时间”一致，用于从 $X_T \sim \pi$ 采样得到 $X_0 \sim p_{\mathrm{data}}$。形式为（$\bar{W}_t$ 为反向时间布朗运动）：
$$
\boxed{
\mathrm{d}X_t = \left[ f_t(X_t) - g_t^2\,\nabla \log p_t(X_t) \right]\mathrm{d}t + g_t\,\mathrm{d}\bar{W}_t.
}
$$

- 漂移由 $f_t$ 改为 $f_t - g_t^2\,\nabla\log p_t$；扩散系数仍为 $g_t$。
- $\nabla\log p_t$ 为 **score**，通常用网络 $s_\theta(x,t)$ 拟合。
- 采样时从 $X_T$ 出发，沿**反向时间**积分上式得到 $X_0$。

---

## 三、Flow：前向 ODE 与反向 ODE

### 3.1 前向流 ODE（数据 → 先验）

$$
\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t, \qquad t \in [0,T] \text{ 或 } [0,1].
$$

$X_0 \sim p_{\mathrm{data}}$，在合适 $u_t$ 下 $X_T$ 服从先验（如高斯）。密度满足：
$$
\partial_t p_t = -\nabla \cdot (u_t\, p_t).
$$

### 3.2 反向流 ODE（先验 → 数据）

沿**同一动力学**反向积分即可：时间从 $T$ 到 $0$，等价于
$$
\boxed{
\mathrm{d}X_t = -u_t(X_t)\,\mathrm{d}t \quad \text{（$t$ 从 $T$ 减到 $0$）}.
}
$$

或令 $\tau = T - t$，则 $\frac{\mathrm{d}X_\tau}{\mathrm{d}\tau} = -u_{T-\tau}(X_\tau)$。无随机项，故**反向 = 前向速度场取反**。

---

## 四、SDE 与 ODE 的相互转换（保持相同 $p_t$）

以下均对**同一前向时间方向**，使转换后的 SDE/ODE 与原来的 ODE/SDE 有**相同的边际分布族** $p_t$。

### 4.1 ODE → SDE（在给定 ODE 的 $p_t$ 下构造 SDE）

已知**前向 ODE** $\mathrm{d}X_t = u_t^{\mathrm{target}}(X_t)\,\mathrm{d}t$ 及其边际 $p_t$。与之**同 $p_t$** 的 SDE 为：
$$
\boxed{
\mathrm{d}X_t = \left[ u_t^{\mathrm{target}}(X_t) + \frac{\sigma_t^2}{2}\,\nabla \log p_t(X_t) \right]\mathrm{d}t + \sigma_t\,\mathrm{d}W_t.
}
$$

- 任取 $\sigma_t \geq 0$；$\sigma_t \equiv 0$ 时退化为原 ODE。
- 即：在 Flow 的漂移上**加扩散** $\sigma_t \,\mathrm{d}W_t$，并**加修正漂移** $\frac{\sigma_t^2}{2}\nabla\log p_t$，使边际 $p_t$ 不变。

### 4.2 SDE → ODE（概率流 ODE，与给定 SDE 同 $p_t$）

已知**前向 SDE** $\mathrm{d}X_t = \mu_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$ 及其边际 $p_t$。与之**同 $p_t$** 的确定性 ODE（**概率流 ODE**）为：
$$
\boxed{
\mathrm{d}X_t = \left[ \mu_t(X_t) - \frac{\sigma_t^2}{2}\,\nabla \log p_t(X_t) \right]\mathrm{d}t.
}
$$

- 采样时可用该 ODE 替代 SDE，得到**相同 $p_t$** 但**无随机性**的轨迹。
- 对**前向扩散 SDE** $\mathrm{d}X_t = f_t\,\mathrm{d}t + g_t\,\mathrm{d}W_t$，对应概率流 ODE 为
  $$
  \mathrm{d}X_t = \left[ f_t(X_t) - \frac{g_t^2}{2}\,\nabla \log p_t(X_t) \right]\mathrm{d}t.
  $$

---

## 五、Diffusion 与 Flow 的对应关系（通过同 $p_t$）

| 出发点 | 目标 | 公式 |
|--------|------|------|
| **Flow 前向 ODE** 漂移 $u_t^{\mathrm{target}}$，边际 $p_t$ | 同 $p_t$ 的 **Diffusion 前向 SDE** | $\mathrm{d}X_t = \big[u_t^{\mathrm{target}} + \frac{\sigma_t^2}{2}\nabla\log p_t\big]\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$ |
| **Diffusion 前向 SDE** 漂移 $f_t$、扩散 $g_t$，边际 $p_t$ | 同 $p_t$ 的 **Flow 前向 ODE**（概率流） | $\mathrm{d}X_t = \big[f_t - \frac{g_t^2}{2}\nabla\log p_t\big]\mathrm{d}t$ |

- **反向**：Diffusion 反向 SDE 已在上文给出（$f_t - g_t^2\nabla\log p_t + g_t\,\mathrm{d}\bar{W}_t$）；若用概率流 ODE 做反向，则对上述概率流 ODE 做**时间反向**（漂移取反）即可，且与“反向 SDE 对应的概率流 ODE”一致（同一 $p_t$）。

---

## 六、公式总表

| 类别 | 方向 | 动力学 | 密度方程 |
|------|------|--------|----------|
| **Flow (ODE)** | 前向 | $\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t$ | $\partial_t p_t = -\nabla\cdot(u_t p_t)$ |
| **Flow (ODE)** | 反向 | $\mathrm{d}X_t = -u_t(X_t)\,\mathrm{d}t$ | 同上（时间反向） |
| **Diffusion (SDE)** | 前向 | $\mathrm{d}X_t = f_t(X_t)\,\mathrm{d}t + g_t\,\mathrm{d}W_t$ | $\partial_t p_t = -\nabla\cdot(f_t p_t) + \frac{g_t^2}{2}\Delta p_t$ |
| **Diffusion (SDE)** | 反向 | $\mathrm{d}X_t = \big[f_t - g_t^2\nabla\log p_t\big]\mathrm{d}t + g_t\,\mathrm{d}\bar{W}_t$ | 与前向 $p_t$ 在反向时间一致 |
| **ODE→SDE（同 $p_t$）** | — | $\mathrm{d}X_t = \big[u_t^{\mathrm{target}} + \frac{\sigma_t^2}{2}\nabla\log p_t\big]\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$ | — |
| **SDE→ODE（同 $p_t$）** | — | $\mathrm{d}X_t = \big[\mu_t - \frac{\sigma_t^2}{2}\nabla\log p_t\big]\mathrm{d}t$ | — |

**符号**：$u_t$ = Flow 速度场；$f_t,g_t$ = 扩散 SDE 漂移与扩散系数；$\mu_t,\sigma_t$ = 一般 SDE 漂移与扩散；$\nabla\log p_t$ = score，常用 $s_\theta(x,t)$ 拟合。

---

## 七、离散过程与连续过程

前文中的 **Flow（ODE）** 与 **Diffusion（SDE）** 均为**连续时间** $t \in [0,T]$。实际实现（如 DDPM）多为**离散时间** $t \in \{0,1,\ldots,T\}$。下面给出**离散**形式及其与**连续**形式的对应与极限关系。

### 7.1 离散与连续的区别

| 类型 | 时间 | 状态 | 动力学描述 |
|------|------|------|------------|
| **离散过程** | $t = 0,1,\ldots,T$（整数步） | $x^{(0)}, x^{(1)}, \ldots, x^{(T)}$ | 转移核 $q(x^{(t)} \mid x^{(t-1)})$ 或确定性映射 $x^{(t)} = F_t(x^{(t-1)})$ |
| **连续过程** | $t \in [0,T]$（实数） | $X_t$ | SDE $\mathrm{d}X_t = \mu_t\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$ 或 ODE $\mathrm{d}X_t = u_t\,\mathrm{d}t$ |

离散过程无“密度演化 PDE”，只有**转移概率/映射**与**多步边际**（如 $q(x^{(t)} \mid x^{(0)})$）；连续过程有 **Fokker–Planck / 连续性方程**。

---

### 7.2 离散 Diffusion（如 DDPM）

**前向**（数据 → 噪声）：马尔可夫链，单步转移为高斯
$$
q(x^{(t)} \mid x^{(t-1)}) = \mathcal{N}\big(x^{(t)};\ \sqrt{\alpha_t}\, x^{(t-1)},\ \beta_t \mathbf{I}\big), \qquad \alpha_t = 1 - \beta_t.
$$
多步边际（给定 $x^{(0)}$）：
$$
q(x^{(t)} \mid x^{(0)}) = \mathcal{N}\big(x^{(t)};\ \sqrt{\bar\alpha_t}\, x^{(0)},\ (1-\bar\alpha_t)\mathbf{I}\big), \qquad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s.
$$

**反向**（噪声 → 数据）：无闭式，用网络拟合
$$
p_\theta(x^{(t-1)} \mid x^{(t)}) = \mathcal{N}\big(x^{(t-1)};\ \mu_\theta(x^{(t)}, t),\ \tilde\beta_t \mathbf{I}\big),
$$
其中 $\mu_\theta$ 由 score / 噪声预测 $\epsilon_\theta$ 表出，$\tilde\beta_t = \beta_t(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$。

**与连续 SDE 的对应**：当步数 $T \to \infty$ 且 $\beta_t$ 按连续时间 $t/T$ 设计时，离散链在适当缩放下收敛到连续 VP-SDE 等形式（见 7.4）。

---

### 7.3 离散 Flow（确定性步）

**前向**：每步确定性映射
$$
x^{(t)} = F_t(x^{(t-1)}), \qquad t = 1,\ldots,T.
$$
例如欧拉步：$x^{(t)} = x^{(t-1)} + \Delta t\, u_t(x^{(t-1)})$，其中 $\Delta t = 1/T$，$u_t$ 为速度场。

**反向**：若 $F_t$ 可逆，则
$$
x^{(t-1)} = F_t^{-1}(x^{(t)}).
$$
若由 ODE 离散化得到（如 $x^{(t)} = x^{(t-1)} + \Delta t\, u_t(x^{(t-1)})$），则反向为
$$
x^{(t-1)} = x^{(t)} - \Delta t\, u_t(x^{(t)}) \quad \text{或用 } u_t(x^{(t-1)}) \text{ 的近似}.
$$

**与连续 ODE 的对应**：$T \to \infty$、$\Delta t \to 0$ 时，离散欧拉步收敛到 ODE $\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t$。

---

### 7.4 离散到连续的极限（详细推导）

将离散步 $k \in \{0,1,\ldots,T\}$ 对应到连续时间 $t \in [0,1]$：令 $t_k = k/T$，$\Delta t = 1/T$。记离散状态为 $x^{(k)}$，若存在连续时间过程 $X_t$，则约定在 $t = t_k$ 处 $X_{t_k}$ 与 $x^{(k)}$ 对应（可理解为对 $x^{(k)}$ 做常数或线性插值得到 $X_t^{(T)}$，再讨论 $T \to \infty$ 的极限）。

---

#### 7.4.1 离散 Diffusion → 连续 SDE（前向）

**离散前向**（DDPM）：
$$
x^{(k)} = \sqrt{1-\beta_k}\, x^{(k-1)} + \sqrt{\beta_k}\,\varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0, \mathbf{I}),\ \text{i.i.d.}
$$

单步增量可写为
$$
x^{(k)} - x^{(k-1)} = \big(\sqrt{1-\beta_k} - 1\big)\, x^{(k-1)} + \sqrt{\beta_k}\,\varepsilon_k.
$$

当 $\beta_k$ 较小时，$\sqrt{1-\beta_k} \approx 1 - \beta_k/2$，故
$$
x^{(k)} - x^{(k-1)} \approx -\frac{\beta_k}{2}\, x^{(k-1)} + \sqrt{\beta_k}\,\varepsilon_k.
$$

**时间与系数缩放**：为在 $T \to \infty$ 时得到非平凡连续极限，需要：
- 每步**漂移**量约为 $O(\Delta t)$，即 $-\frac{\beta_k}{2}x^{(k-1)}$ 与 $\Delta t$ 同阶；
- 每步**噪声**的方差约为 $O(\Delta t)$，这样在 $[0,1]$ 上累积方差为 $O(1)$，极限为布朗运动。

取**连续时间调度** $\beta(\tau)$，$\tau \in [0,1]$，令
$$
\beta_k = \beta(t_k)\,\Delta t = \frac{\beta(k/T)}{T}, \qquad t_k = \frac{k}{T}.
$$

则
- 漂移项：$-\frac{\beta_k}{2}\, x^{(k-1)} = -\frac{\beta(t_k)}{2T}\, x^{(k-1)}$，即每步漂移 $= -\frac{\beta(t_k)}{2}\, x^{(k-1)}\,\Delta t$；
- 噪声项：$\sqrt{\beta_k}\,\varepsilon_k = \sqrt{\beta(t_k)/T}\,\varepsilon_k$，单步方差 $= \beta(t_k)/T = \beta(t_k)\,\Delta t$。

对 $k = 1,\ldots,T$ 累加并令 $T \to \infty$（在适当正则性下）：
- 漂移和 $\sum_{k} -\frac{\beta(t_k)}{2}\, x^{(k-1)}\,\Delta t \to \int_0^t -\frac{\beta(s)}{2}\, X_s\,\mathrm{d}s$；
- 噪声和 $\sum_{k} \sqrt{\beta(t_k)\,\Delta t}\,\varepsilon_k$ 在分布意义下收敛到 $\int_0^t \sqrt{\beta(s)}\,\mathrm{d}W_s$（独立同分布、方差为 $\Delta t$ 的随机项在 $T\to\infty$ 下收敛到布朗运动的随机积分）。

因此**连续极限**为（VP-SDE，时间 $t \in [0,1]$）：
$$
\boxed{
\mathrm{d}X_t = -\frac{\beta(t)}{2}\,X_t\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t.
}
$$

即前文中的 $f_t(x) = -\frac{\beta(t)}{2}x$，$g_t = \sqrt{\beta(t)}$。若将时间区间取为 $[0,T]$，则可将 $\beta(t)$ 写为 $\beta(t/T)$ 或直接使用 $t$ 上的调度。

**反向的对应**：离散反向的均值 $\mu_\theta$ 由 score / $\epsilon_\theta$ 给出；在连续极限下，$p_t$ 的 score $\nabla\log p_t$ 与离散的 $\epsilon_\theta$ 通过 $x^{(t)} = \sqrt{\bar\alpha_t}\,x^{(0)} + \sqrt{1-\bar\alpha_t}\,\epsilon$ 相联系。连续反向 SDE 的漂移为 $f_t - g_t^2\,\nabla\log p_t$，与离散中“用 $\epsilon_\theta$ 估计 $\epsilon$ 再代入 $\tilde\mu_t$”在 $T\to\infty$ 下一致。

---

#### 7.4.2 离散 Flow → 连续 ODE

**离散前向**（欧拉离散化）：
$$
x^{(k)} = x^{(k-1)} + \Delta t\, u_{k}(x^{(k-1)}), \qquad \Delta t = \frac{1}{T}.
$$

这里 $u_k$ 为第 $k$ 步的速度场（可与连续时间 $t_k = k/T$ 对应，即 $u_k(x) = u(t_k, x)$）。等价于
$$
\frac{x^{(k)} - x^{(k-1)}}{\Delta t} = u_k(x^{(k-1)}).
$$

这正是 ODE $\frac{\mathrm{d}X_t}{\mathrm{d}t} = u_t(X_t)$ 的**欧拉格式**。在 $u_t(x)$ 关于 $x$ Lipschitz、关于 $t$ 连续等标准条件下，对 $x^{(0)} = X_0$ 做线性插值得到的 $X_t^{(T)}$（满足 $X_{k/T}^{(T)} = x^{(k)}$）在 $T \to \infty$ 时一致收敛到 ODE 的解 $X_t$，即
$$
\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t.
$$

**反向**：离散反向为 $x^{(k-1)} = x^{(k)} - \Delta t\, u_k(x^{(k)})$（或用 $x^{(k-1)}$ 的近似），即连续 ODE 反向 $\mathrm{d}X_t = -u_t(X_t)\,\mathrm{d}t$ 的欧拉离散化；$T \to \infty$ 时同样收敛到连续反向 ODE。

---

#### 7.4.3 小结

| 离散过程 | 缩放 / 条件 | 连续极限 |
|----------|-------------|----------|
| **DDPM 前向** $x^{(k)} = \sqrt{1-\beta_k}x^{(k-1)} + \sqrt{\beta_k}\varepsilon_k$ | $\beta_k = \beta(k/T)/T$，$T\to\infty$ | $\mathrm{d}X_t = -\frac{\beta(t)}{2}X_t\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W_t$ |
| **离散 Flow** $x^{(k)} = x^{(k-1)} + \Delta t\, u_k(x^{(k-1)})$ | $\Delta t = 1/T$，$u_k = u(k/T, \cdot)$，$T\to\infty$ | $\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t$ |

- 离散 Diffusion 的**漂移**来自 $\sqrt{1-\beta_k}-1 \approx -\beta_k/2$，**噪声**来自 $\sqrt{\beta_k}\,\varepsilon_k$；取 $\beta_k = O(1/T)$ 使单步漂移与噪声方差均为 $O(1/T)$，从而极限为连续 SDE。
- 离散 Flow 的**步长** $\Delta t = 1/T$ 直接对应 ODE 的 $\mathrm{d}t$，欧拉格式收敛性为标准数值分析结论。

---

### 7.5 离散 vs 连续 总表

| 类别 | 离散形式 | 连续形式 |
|------|----------|----------|
| **Diffusion 前向** | $q(x^{(t)} \mid x^{(t-1)}) = \mathcal{N}(\sqrt{\alpha_t}x^{(t-1)}, \beta_t \mathbf{I})$；$q(x^{(t)} \mid x^{(0)}) = \mathcal{N}(\sqrt{\bar\alpha_t}x^{(0)}, (1-\bar\alpha_t)\mathbf{I})$ | $\mathrm{d}X_t = f_t\,\mathrm{d}t + g_t\,\mathrm{d}W_t$；$\partial_t p_t = -\nabla\cdot(f_t p_t) + \frac{g_t^2}{2}\Delta p_t$ |
| **Diffusion 反向** | $p_\theta(x^{(t-1)} \mid x^{(t)})$，均值 $\mu_\theta$、方差 $\tilde\beta_t$ | $\mathrm{d}X_t = [f_t - g_t^2\nabla\log p_t]\,\mathrm{d}t + g_t\,\mathrm{d}\bar{W}_t$ |
| **Flow 前向** | $x^{(t)} = F_t(x^{(t-1)})$ 或 $x^{(t)} = x^{(t-1)} + \Delta t\, u_t(x^{(t-1)})$ | $\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t$；$\partial_t p_t = -\nabla\cdot(u_t p_t)$ |
| **Flow 反向** | $x^{(t-1)} = F_t^{-1}(x^{(t)})$ 或 $x^{(t-1)} = x^{(t)} - \Delta t\, u_t(\cdot)$ | $\mathrm{d}X_t = -u_t(X_t)\,\mathrm{d}t$ |

---

## 附录：SDE 与 Fokker–Planck 公式的推导

下面由**随机微分方程（SDE）**出发，推导其密度 $p_t$ 所满足的 **Fokker–Planck 方程**（也称 Kolmogorov 前向方程）。先在一维情形写出完整推导，再给出高维形式。

### A.1 设定的 SDE

考虑（可先设一维 $X_t \in \mathbb{R}$）：
$$
\mathrm{d}X_t = \mu_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t.
$$
其中 $\mu_t(x)$ 为漂移，$\sigma_t \geq 0$ 为扩散系数（可与 $x$ 有关，下面为书写简单取仅与 $t$ 有关），$W_t$ 为标准布朗运动。

设 $X_t$ 有密度 $p_t(x)$，目标是得到 $p_t$ 满足的 PDE。

### A.2 用 Itô 引理得到 $\varphi(X_t)$ 的演化

对任意光滑紧支撑函数 $\varphi(x)$，Itô 引理给出：
$$
\mathrm{d}\varphi(X_t) = \left( \mu_t(X_t)\,\varphi'(X_t) + \frac{\sigma_t^2}{2}\,\varphi''(X_t) \right)\mathrm{d}t + \sigma_t\,\varphi'(X_t)\,\mathrm{d}W_t.
$$
对时间取期望，$\mathrm{d}W_t$ 项均值为 0，故
$$
\frac{\mathrm{d}}{\mathrm{d}t}\,\mathbb{E}[\varphi(X_t)] = \mathbb{E}\left[ \mu_t(X_t)\,\varphi'(X_t) + \frac{\sigma_t^2}{2}\,\varphi''(X_t) \right].
$$

用密度写开：$\mathbb{E}[\varphi(X_t)] = \int \varphi(x)\,p_t(x)\,\mathrm{d}x$，且
$$
\mathbb{E}[\mu_t(X_t)\,\varphi'] = \int \mu_t(x)\,\varphi'(x)\,p_t(x)\,\mathrm{d}x, \qquad \mathbb{E}[\varphi''(X_t)] = \int \varphi''(x)\,p_t(x)\,\mathrm{d}x.
$$
于是
$$
\int \varphi(x)\,\partial_t p_t(x)\,\mathrm{d}x = \int \mu_t(x)\,\varphi'(x)\,p_t(x)\,\mathrm{d}x + \frac{\sigma_t^2}{2}\int \varphi''(x)\,p_t(x)\,\mathrm{d}x.
$$

### A.3 分部积分得到 Fokker–Planck（弱形式）

对右边两项做分部积分（边界项在紧支撑或无穷远处为 0）：
- $\int \mu_t\,\varphi'\,p_t\,\mathrm{d}x = -\int \varphi\,\partial_x(\mu_t\,p_t)\,\mathrm{d}x$；
- $\int \varphi''\,p_t\,\mathrm{d}x = \int \varphi\,\partial_{xx} p_t\,\mathrm{d}x$（即 $\int \varphi\,\Delta p_t\,\mathrm{d}x$ 在一维为 $\int \varphi\,p_t''\,\mathrm{d}x$）。

代入得
$$
\int \varphi\,\partial_t p_t\,\mathrm{d}x = -\int \varphi\,\partial_x(\mu_t\,p_t)\,\mathrm{d}x + \frac{\sigma_t^2}{2}\int \varphi\,\partial_{xx} p_t\,\mathrm{d}x.
$$
由 $\varphi$ 任意，得**弱形式**：对任意光滑紧支撑 $\varphi$，
$$
\int \varphi\,\left( \partial_t p_t + \partial_x(\mu_t\,p_t) - \frac{\sigma_t^2}{2}\,\partial_{xx} p_t \right)\mathrm{d}x = 0.
$$

### A.4 Fokker–Planck 方程（强形式）

由 $\varphi$ 任意，被积函数（在分布意义下）为 0，即
$$
\boxed{
\partial_t p_t = -\partial_x(\mu_t\,p_t) + \frac{\sigma_t^2}{2}\,\partial_{xx} p_t = -\nabla\cdot(\mu_t\,p_t) + \frac{\sigma_t^2}{2}\,\Delta p_t.
}
$$
一维时 $\nabla\cdot(\mu_t p_t) = \partial_x(\mu_t p_t)$，$\Delta p_t = \partial_{xx} p_t$；高维时 $\nabla\cdot$ 为散度，$\Delta$ 为拉普拉斯算子，推导相同（对 $\varphi(X_t)$ 用高维 Itô 公式，再分部积分）。

### A.5 高维 SDE 与 Fokker–Planck

对
$$
\mathrm{d}X_t = \mu_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t, \qquad X_t \in \mathbb{R}^d,
$$
其中 $W_t$ 为 $d$ 维布朗运动，$\sigma_t$ 为标量或矩阵，在标量扩散系数 $\sigma_t$ 且与 $x$ 无关时，密度 $p_t(x)$ 满足
$$
\partial_t p_t = -\nabla \cdot (\mu_t\, p_t) + \frac{\sigma_t^2}{2}\,\Delta p_t.
$$
若扩散系数为矩阵 $G_t$（$\mathrm{d}X_t = \mu_t\,\mathrm{d}t + G_t\,\mathrm{d}W_t$），则扩散项为 $\frac{1}{2}\sum_{i,j}(G_t G_t^\top)_{ij}\,\partial_{ij} p_t$；当 $G_t = \sigma_t \mathbf{I}$ 时仍为 $\frac{\sigma_t^2}{2}\Delta p_t$。

### A.6 小结

- **SDE** $\mathrm{d}X_t = \mu_t\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$ 与 **Fokker–Planck** $\partial_t p_t = -\nabla\cdot(\mu_t p_t) + \frac{\sigma_t^2}{2}\Delta p_t$ 一一对应：给定 SDE 可推出 $p_t$ 满足的 PDE；反之，若 $p_t$ 满足该 PDE 且与 SDE 的初始分布一致，则 $p_t$ 即为该 SDE 解的边际密度。
- 推导路径：**SDE → Itô 引理（对 $\varphi(X_t)$）→ 取期望并分部积分 → Fokker–Planck**。
