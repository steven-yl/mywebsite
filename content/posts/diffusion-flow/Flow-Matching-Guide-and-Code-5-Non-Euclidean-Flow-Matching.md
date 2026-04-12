---

title: "Flow Matching Guide and Code 第5章解读：Non-Euclidean Flow Matching"
subtitle: "黎曼流形上的流匹配"
date: 2026-03-05
draft: false
authors: [Steven]
description: "《Flow Matching Guide and Code》第5章 Non-Euclidean Flow Matching 的详细技术解读：黎曼流形上的概率路径、速度场、边际化技巧、测地线条件流与黎曼流匹配损失。"

tags: [diffusion/flow, tutorial, Riemannian, flow matching]
categories: [diffusion/flow]
series: [diffusion/flow系列]
weight: 5
series_weight: 5

hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""

summary: "第5章 Non-Euclidean Flow Matching 解读：从动机与黎曼流形设定出发，说明流形上的流、概率路径与连续性方程，边际化技巧（定理10）、RCFM 损失（定理11），以及测地线条件流与基于预度量的条件流；并对照欧氏 FM 与代码8（球面测地线 RCFM）做小结。"

---

## 1 为什么要做「非欧」Flow Matching？

前几章都在 **欧氏空间 $\mathbb{R}^d$** 里做 FM：样本 $x_t$ 在 $\mathbb{R}^d$ 里，速度场 $u_t(x)\in\mathbb{R}^d$，ODE 是 $\dot x_t = u_t(x_t)$。

但很多数据**本来就不在欧氏空间**，例如：

- **球面 $S^2$**：地球上的分布、方向、归一化向量；
- **矩阵李群**：旋转 $SO(3)$、蛋白质骨架等几何数据；
- **双曲空间、Stiefel 流形**等。

若强行把这类数据塞进 $\mathbb{R}^d$ 再做欧氏 FM，会破坏几何结构（例如把球面压平会扭曲距离和测地线）。因此需要一套在 **黎曼流形 $M$** 上定义的 Flow Matching，即 **Non-Euclidean / Riemannian Flow Matching**。

---

## 2 数学舞台：黎曼流形（5.1 节）

- **流形 $M$**：局部像 $\mathbb{R}^d$ 的弯曲空间，每点 $x$ 有一个**切空间** $T_x M$（该点的「速度」只能生活在这个切空间里）。
- **黎曼度量 $g$**：在每点 $x$ 的切空间 $T_x M$ 上定义一个内积 $\langle u,v\rangle_g$，从而有长度、角度、距离。
- **切丛 $TM$**：所有点的切空间并起来，$TM = \bigcup_{x\in M} x\times T_x M$。
- **向量场**：$u_t: [0,1]\times M \to TM$，且 $u_t(x)\in T_x M$（在 $x$ 处的速度必须在 $T_x M$ 里）。
- **黎曼散度 $\mathrm{div}_g$**：流形上的散度，替代欧氏里的 $\nabla\cdot$。
- **体积元 $\mathrm{dvol}_x$**：由度量 $g$ 决定的体积测度，积分写为 $\int_M f(x)\mathrm{dvol}_x$。

概率密度 $p: M\to\mathbb{R}_{\ge 0}$ 满足 $\int_M p(x)\mathrm{dvol}_x = 1$。这样就把「分布、速度场、ODE、守恒律」都搬到流形上了。

---

## 3 流形上的流与概率路径（5.2 节）

- **概率路径** $p_t$：一族随时间 $t\in[0,1]$ 变化的分布，$p_0=p$（源），$p_1=q$（目标）。
- **流** $\psi_t: M\to M$：时间依赖的微分同胚，$\psi_0=\mathrm{id}$，且由某个**流形上的 ODE** 生成。
- **速度场与流的关系**（定理 8）：在流形上，只要 $u_t$ 足够光滑（例如 $C^\infty([0,1]\times M, TM)$ 且局部 Lipschitz），就存在唯一的流 $\psi_t$ 满足

$$
\frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x) = u_t(\psi_t(x)),\quad \psi_0(x)=x.
$$

也就是说：**流由速度场唯一决定**，和欧氏情形一样。

**黎曼连续性方程**（质量守恒在流形上的版本）：

$$
\frac{\partial}{\partial t}p_t(x) + \mathrm{div}_g\bigl(p_t(x)u_t(x)\bigr) = 0.
$$

**定理 9（流形质量守恒）**：在适当可积性条件下，「$u_t$ 与 $p_t$ 满足上式」等价于「$u_t$ 生成 $p_t$」（即从 $p_0$ 出发沿 $u_t$ 的 ODE 演化，边际分布就是 $p_t$）。

**似然演化**（黎曼版瞬时变量替换）：

$$
\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(\psi_t(x)) = -\mathrm{div}_g(u_t)(\psi_t(x)).
$$

和欧氏情形形式一致，只是散度换成 $\mathrm{div}_g$。

---

## 4 流形上的概率路径怎么构造（5.3 节）

和欧氏 FM 一样，用 **条件路径 + 边际化**：

- **条件概率路径** $p_{t|1}(x|x_1)$：对每个「目标」$x_1$，一条从源到 $\delta_{x_1}$ 的路径，满足

$$
p_{0|1}(\cdot|x_1) = \pi_{0|1}(\cdot|x_1),\qquad p_{1|1}(\cdot|x_1) = \delta_{x_1}.
$$

- **边际概率路径**：

$$
p_t(x) = \int_M p_{t|1}(x|x_1)q(x_1)\mathrm{dvol}_{x_1}.
$$

这样自然有 $p_0=p$，$p_1=q$。

---

## 5 流形上的边际化技巧（5.4 节）

**定理 10（流形边际化技巧）**：若对每个 $x_1$，**条件速度场** $u_t(\cdot|x_1)$ 生成 **条件路径** $p_{t|1}(\cdot|x_1)$，则按后验加权的**边际速度场**

$$
u_t(x) = \int_M u_t(x|x_1)p_{1|t}(x_1|x)\,\mathrm{dvol}_{x_1}
= \mathbb{E}_{x_1\sim p_{1|t}(\cdot|x)}\bigl[u_t(x|x_1)\bigr],
$$

其中

$$
p_{1|t}(x_1|x) = \frac{p_{t|1}(x|x_1)q(x_1)}{p_t(x)},
$$

会生成**边际路径** $p_t$。

也就是说：**只要会设计并采样「条件路径 + 条件速度场」，就可以用条件期望得到边际速度场，而不用显式算 $p_t$ 或边际 ODE**。和欧氏 FM 的边际化完全平行，只是积分换成流形上的体积元。

---

## 6 黎曼 Flow Matching 损失（5.5 节）

- **黎曼 FM 损失**（理想但难算）：

$$
\mathcal{L}_{\mathrm{RFM}}(\theta)
= \mathbb{E}_{t,X_t\sim p_t}
D_{X_t}\bigl(u_t(X_t),u_t^\theta(X_t)\bigr).
$$

这里 $D_x$ 是在**切空间 $T_x M$** 上定义的 Bregman 散度，例如用黎曼内积的平方：$D_x(u,v) = \|u-v\|_g^2$。
- **黎曼条件 FM 损失（RCFM）**（实际用的）：

$$
\mathcal{L}_{\mathrm{RCFM}}(\theta)
= \mathbb{E}_{t,X_1,X_t\sim p_{t|1}(\cdot|X_1)}
D_{X_t}\bigl(u_t(X_t|X_1),u_t^\theta(X_t)\bigr).
$$

用**条件速度场** $u_t(X_t|X_1)$ 当监督信号。

**定理 11**：$\nabla_\theta \mathcal{L}_{\mathrm{RFM}}(\theta) = \nabla_\theta \mathcal{L}_{\mathrm{RCFM}}(\theta)$，且最小化 RCFM 得到的 $u_t^\theta$ 就是真实的边际速度场。因此训练时只需从 $q$ 抽 $X_1$、从 $p_{t|1}(\cdot|X_1)$ 抽 $X_t$，用 $u_t(X_t|X_1)$ 和 $u_t^\theta(X_t)$ 算 Bregman 散度即可。

---

## 7 流形上的条件流怎么取（5.6 节）

在欧氏空间我们用过**仿射条件流** $\psi_t(x_0|x_1) = (1-t)x_0 + t x_1$（直线）。在流形上「直线」没有定义，自然推广是**测地线**。

### 7.1 测地线条件流（最常用）

- **指数映射** $\exp_x: T_x M\to M$：从 $x$ 出发、初速度为 $v$ 的测地线在「时间 1」的终点。
- **对数映射** $\log_x: M\to T_x M$：$\exp_x$ 的逆（在存在且唯一的前提下）。

**测地线条件流**（式 5.17）：

$$
\psi_t(x_0|x_1)
= \exp_{x_0}\Bigl(\kappa(t)\log_{x_0}(x_1)\Bigr),\quad t\in[0,1],
$$

其中 $\kappa(t)$ 单调、$\kappa(0)=0$、$\kappa(1)=1$。在欧氏空间 $\exp_{x_0}(v)=x_0+v$，$\log_{x_0}(x_1)=x_1-x_0$，取 $\kappa(t)=t$ 就回到线性插值 $x_0+t(x_1-x_0)$。

对**有闭式 $\exp/\log$ 的流形**（如球面、$SO(3)$），这样就能**无模拟**地算 $\psi_t$ 和对应的条件速度场，训练和欧氏 FM 一样高效。

### 7.2 基于「预度量」的条件流（更一般）

若流形没有闭式测地线，或希望用非测地线的路径，可引入**预度量** $d(\cdot,\cdot): M\times M\to\mathbb{R}_{\ge 0}$（满足非负、正定、可微等），并要求条件流满足：

$$
d(\psi_t(x_0|x_1),x_1) = \bar\kappa(t)d(x_0,x_1),\quad \bar\kappa(t)=1-\kappa(t).
$$

这样 $t=1$ 时 $\psi_1(x_0|x_1)=x_1$。在**最小范数**意义下，可推出条件速度场的形式（式 5.19）：

$$
u_t(x|x_1)
= \frac{\mathrm{d}}{\mathrm{d}t}\ln\bar\kappa(t)\cdot d(x,x_1)
\frac{\nabla d(x,x_1)}{\|\nabla d(x,x_1)\|_g^2}.
$$

若取 $d$ 为测地线距离 $d_g$，就回到测地线流。若用**谱距离**等，可以在没有闭式测地线的流形上仍定义条件流，但通常就需要在训练时做**模拟**（从 $p_{t|1}(\cdot|x_1)$ 采样），计算会更贵。

### 7.3 奇异性与调度器

- 在**紧流形**（如球面）上，测地线距离在对径点等处不可微，$\nabla d$ 会退化。
- 文中提到可通过**增强调度器** $\bar\kappa(t,x,x_1)$ 或放宽「非退化」条件（允许零梯度集为零测度）来规避，实践中很多应用里这些奇点集很小，常可接受。

---

## 8 和欧氏 FM 的对照（小结）

| 项目      | 欧氏 FM ($\mathbb{R}^d$)                          | 非欧 FM (黎曼流形 $M$)                                     |
| ------- | --------------------------------------------- | -------------------------------------------------- |
| 状态空间    | $\mathbb{R}^d$                                  | 流形 $M$，切空间 $T_x M$                                     |
| 速度场     | $u_t(x)\in\mathbb{R}^d$                         | $u_t(x)\in T_x M$                                    |
| 守恒律     | 连续性方程 $\partial_t p_t + \nabla\cdot(p_t u_t)=0$ | 黎曼连续性方程 $\partial_t p_t + \mathrm{div}_g(p_t u_t)=0$ |
| 条件流典型选择 | 直线 $(1-t)x_0+tx_1$                              | 测地线 $\exp_{x_0}(\kappa(t)\log_{x_0}(x_1))$           |
| 损失      | CFM，Bregman 在 $\mathbb{R}^d$                    | RCFM，Bregman 在 $T_x M$（如 $|\cdot|_g^2$）                |
| 边际化     | 定理 3                                          | 定理 10（流形边际化技巧）                                     |


---

## 9 代码 8 在做什么（球面测地线 RCFM）

代码 8 演示的是在**球面**上、用**测地线条件流**做 RCFM 训练：

- **流形**：`Sphere()`。
- **调度器**：`CondOTScheduler()`（类似最优传输的线性 $\kappa(t)=t$）。
- **路径**：`GeodesicProbPath(scheduler, manifold)`，即上面说的 $\psi_t(x_0|x_1) = \exp_{x_0}(\kappa(t)\log_{x_0}(x_1))$。
- 每步：从 $\pi_{0,1}$ 抽 $(x_0,x_1)$，抽 $t\sim U[0,1]$，用 `path.sample(t, x_0, x_1)` 得到 $x_t$ 和条件速度 `sample.dx_t`；**关键**：网络输出要投影到切空间，即 `manifold.proju(sample.x_t, model_output)`，再和 `sample.dx_t` 算 MSE。

### 9.1 原理代码示例（球面 $S^2$ 测地线 + RCFM 损失）

下面用**纯公式**实现：球面 $\exp/\log$、测地线条件流 $\psi_t(x_0|x_1)$、条件速度场、切空间投影与 RCFM 的 MSE。约定 $x\in S^2$ 为单位向量，切空间 $T_x S^2 = \{v : v^\top x=0\}$。

**球面指数/对数映射**（$S^2$ 嵌入 $\mathbb{R}^3$，闭式）：

$$
\exp_x(v) = \cos(\|v\|)\,x + \sin(\|v\|)\,\frac{v}{\|v\|},\quad
\log_x(y) = \frac{\theta}{\sin\theta}(y - (\cos\theta)x),\ \ \theta=\arccos(x^\top y).
$$

**公式推导**（简要）：

- **指数映射 $\exp_x(v)$**  
  球面上的测地线是**大圆弧**。在 $x$ 点、沿切向量 $v\in T_x S^2$（故 $x^\top v=0$）出发的测地线落在由 $x$ 与 $v$ 张成的平面与球面的交线上，因此可设为
  $$
  \gamma(t) = \cos(\alpha t)\,x + \sin(\alpha t)\,\frac{v}{\|v\|},\quad t\ge 0.
  $$

  {{< admonition tip "公式推导" false >}}
  ### 1. 几何直观：大圆为什么是测地线？

  在球面上，连接两点间的最短路径是过这两点的大圆（以球心为圆心的圆）的劣弧。从微分几何的“内蕴”角度看，球面上的曲线是测地线，当且仅当它的**主法线与球面的法线重合**。对于大圆来说，这个条件完美满足：
  *   大圆所在平面过球心，因此在曲线上任意一点，指向曲率中心的**主法线**方向，恰好沿着半径指向球心。
  *   球面在该点的**法线**，也是沿着半径方向指向球心的。
  两者方向完全一致，所以大圆就是测地线。

  ### 2. 代数推导：从几何到参数方程

  理解了上述几何事实，我们就可以用一个统一的参数方程，来描述任意一条过球心平面所截出的大圆。

  **前提假设**：为简化推导，我们考虑**单位球面**（半径为1）。一般半径 $ R $ 的球面可在此基础上缩放得到。

  *   **步骤1：确定大圆所在的平面**
      一条大圆由**过球心的平面**唯一确定。这个平面可以由两个正交的单位向量完全张成：
      *   $ \mathbf{x} $：平面内的一个单位向量，我们可以把它视为大圆经过的某个起始点（在单位球面上，点坐标即为从球心指向该点的向量）。
      *   $ \mathbf{v} $：平面内的另一个单位向量，且 $ \mathbf{v} \perp \mathbf{x} $。它代表在起始点 $ \mathbf{x} $ 处的一个**切方向**。

  *   **步骤2：构造大圆的参数方程**
      有了这个平面的一组标准正交基 $\{\mathbf{x}, \mathbf{v}\}$，大圆上任意一点的位置向量，就是这个基的线性组合。因为曲线要在单位球面上，所以这个组合向量的长度必须恒为1。
      
      一个自然而优美的选择就是采用三角函数的参数化形式。设参数为 $ t $，当 $ t $ 从0变化时，向量 $ \mathbf{x} $ 和 $ \mathbf{v} $ 的系数应满足平方和为1，以确保点始终在球面上。因此，大圆上的点 $ \gamma(t) $ 可以表示为：
      $$
      \gamma(t) = \cos(\alpha t)\,\mathbf{x} + \sin(\alpha t)\,\mathbf{v}
      $$
      
      在这个表达式中：
      *   当 $ t = 0 $ 时，$ \gamma(0) = \mathbf{x} $，对应大圆的起始点。
      *   当 $ t $ 增加时，$ \gamma(t) $ 在由 $ \mathbf{x} $ 和 $ \mathbf{v} $ 张成的平面内匀速转动。
      *   $ \alpha $ 是一个常数，代表角速度。它决定了当参数 $ t $ 变化时，点在大圆上移动的快慢。你可以根据初始条件（如给定的终点位置）来确定它。

  *   **步骤3：对比你的公式**
      对比你给出的公式：
      $$
      \gamma(t) = \cos(\alpha t)\,x + \sin(\alpha t)\,\frac{v}{\|v\|},\quad t\ge 0
      $$
      可以发现：
      *   $ x $ 就是我们的起始单位向量 $ \mathbf{x} $。
      *   $ \frac{v}{\|v\|} $ 是将 $ v $ 单位化，它正是我们所需的、与 $ x $ 正交的单位切向量 $ \mathbf{v} $。这一步确保了 $ \frac{v}{\|v\|} $ 与 $ x $ 的点积为0（因为是切方向），且长度为1。

      因此，你的公式完美地描述了一条从点 $ x $ 出发，沿着切方向 $ v $，以角速度 $ \alpha $ 在单位球面的大圆上运动的测地线。

  ### 3. 验证：它为什么满足测地线方程？

  虽然几何直观已经足够，但我们可以从微分几何的测地线方程来验证。测地线的定义为：曲线的切向量在曲面上是**平行移动**的。

  1.  **求切向量**：对 $ \gamma(t) $ 求导，得到速度（切）向量：
      $$
      \dot{\gamma}(t) = -\alpha \sin(\alpha t)\,\mathbf{x} + \alpha \cos(\alpha t)\,\mathbf{v}
      $$
      可以验证，$ |\dot{\gamma}(t)| = \alpha $ 是一个常数，说明曲线是匀速参数化的。

  2.  **求加速度**：再求一次导，得到加速度向量：
      $$
      \ddot{\gamma}(t) = -\alpha^2 \cos(\alpha t)\,\mathbf{x} -\alpha^2 \sin(\alpha t)\,\mathbf{v} = -\alpha^2 \gamma(t)
      $$
      这个结果非常关键：**加速度 $ \ddot{\gamma}(t) $ 正好与位置向量 $ \gamma(t) $ 的方向相反，即指向球心**。

  3.  **分析平行移动条件**：
      *   对于球面上的曲线，切向量平行移动的条件等价于：**曲线的加速度向量（即曲率向量）没有切向分量，全部在法线方向（即沿球面法向）**。
      *   球面在 $ \gamma(t) $ 点处的单位法向量就是 $ \gamma(t) $ 本身（因为是单位球面）。
      *   而我们得到的加速度 $ \ddot{\gamma}(t) $ 完全平行于 $ \gamma(t) $，这意味着它的切向分量为0。因此，切向量 $ \dot{\gamma}(t) $ 沿曲线是平行移动的，满足测地线的定义。

  ### 总结

  你提供的公式 $ \gamma(t) = \cos(\alpha t)\,x + \sin(\alpha t)\,\frac{v}{\|v\|} $ 是单位球面上测地线的标准参数方程。它的核心思想是：**在由起始点 $ x $ 和单位切方向 $ \frac{v}{\|v\|} $ 张成的过球心的平面内，做匀速圆周运动**。这个公式几何意义明确，代数形式优美，且严格满足测地线的微分方程。

  对于半径为 $ R $ 的球面，只需将公式乘以 $ R $ 即可。
  {{< /admonition >}}


  由 $\gamma(0)=x$、$\dot\gamma(0)=\alpha\,v/\|v\|=v$ 得 $\alpha=\|v\|$。指数映射定义为「沿该测地线走单位时间」的终点，即
  $$
  \exp_x(v) = \gamma(1) = \cos(\|v\|)\,x + \sin(\|v\|)\,\frac{v}{\|v\|}.
  $$
  当 $v=0$ 时上式仍成立（$\cos 0=1,\,\sin 0=0$）；实现时 $\|v\|\to 0$ 用收缩 $\mathrm{projx}(x+v)$ 避免 $v/\|v\|$ 未定义。

- **对数映射 $\log_x(y)$**  
  设 $x,y\in S^2$，$\theta = \arccos(x^\top y)\in[0,\pi]$ 为两点夹角（即测地线距离）。要求 $u=\log_x(y)\in T_x S^2$ 满足 $\exp_x(u)=y$。  
  由指数公式，$y = \cos(\|u\|)\,x + \sin(\|u\|)\,u/\|u\|$。令 $\|u\|=\theta$（测地线长度），则
  $$
  y = \cos\theta\,x + \sin\theta\,\frac{u}{\|u\|}
  \quad\Rightarrow\quad
  \frac{u}{\|u\|} = \frac{y - \cos\theta\,x}{\sin\theta}.
  $$
  向量 $y - (\cos\theta)x$ 与 $u$ 同向，且 $\|y - (\cos\theta)x\|^2 = 1 - 2\cos\theta\,(x^\top y) + \cos^2\theta = \sin^2\theta$（因 $x^\top y=\cos\theta$），故 $(y - (\cos\theta)x)/\sin\theta$ 为单位向量。于是
  $$
  \log_x(y) = \theta\,\frac{y - (\cos\theta)x}{\sin\theta}
  = \frac{\theta}{\sin\theta}\,\bigl(y - (\cos\theta)x\bigr),\qquad \theta = \arccos(x^\top y).
  $$
  可验证 $x^\top \log_x(y)=0$（即落在 $T_x S^2$）：$x^\top(y - (\cos\theta)x) = x^\top y - \cos\theta = 0$。$\theta\to 0$ 时 $\theta/\sin\theta\to 1$，极限给出 $\log_x(x)=0$；实现时对小 $\theta$ 需单独分支避免除零。

  {{< admonition tip "公式推导" false >}}
  球面 $S^2$ 上**对数映射** $\log_x(y)$ 的推导过程。对数映射是指数映射的逆运算：给定球面上两点 $x$ 和 $y$，我们想要找到从 $x$ 出发的切向量 $u \in T_x S^2$，使得沿着测地线（大圆）走长度 $\|u\|$ 后恰好到达 $y$，即 $\exp_x(u) = y$。

  ### 1. 指数映射回顾

  首先回忆单位球面 $S^2 = \{ p \in \mathbb{R}^3 : \|p\| = 1 \}$ 上的指数映射。给定基点 $x \in S^2$ 和一个切向量 $u \in T_x S^2$（满足 $x \perp u$），指数映射 $\exp_x(u)$ 给出从 $x$ 出发沿测地线（大圆）走弧长为 $\|u\|$ 所到达的点。它的解析表达式为：
  $$
  \exp_x(u) = \cos(\|u\|)\, x + \sin(\|u\|)\, \frac{u}{\|u\|}.
  $$
  这里 $\|u\|$ 就是测地距离（单位球面上弧长等于圆心角）。这个公式的几何意义是：在过 $x$ 且切方向为 $u$ 的平面内，以角速度 $1$ 匀速旋转，经过时间 $\|u\|$ 到达新位置。验证：当 $\|u\| = 0$ 时，$\exp_x(0)=x$；当 $\|u\| = \pi$ 时，到达对径点 $-x$。

  ### 2. 对数映射的目标

  对数映射 $\log_x(y)$ 是上述过程的逆：给定终点 $y \in S^2$，我们要找到切向量 $u$ 使得 $\exp_x(u)=y$。通常要求 $y$ 不与 $x$ 对径（即 $y \neq -x$），否则对数映射不唯一（所有从 $x$ 出发的切向量沿测地线走 $\pi$ 都会到达 $-x$，但方向不确定）。我们假设 $x$ 与 $y$ 不互为对径点，且记它们之间的夹角为：
  $$
  \theta = \arccos(x^\top y) \in [0,\pi].
  $$
  这个 $\theta$ 正是球面上两点间的测地距离（即大圆弧长），因为单位球面上两点与球心连线所夹的圆心角等于球面距离。

  ### 3. 由指数公式反推 $u$

  设 $u$ 是对数映射的结果，记其长度为 $\|u\|$。由指数映射公式，有：
  $$
  y = \cos(\|u\|)\, x + \sin(\|u\|)\, \frac{u}{\|u\|}.
  $$
  这里隐含地要求 $\frac{u}{\|u\|}$ 是与 $x$ 正交的单位向量（因为切向量的单位化）。我们不知道 $\|u\|$ 和方向，但由几何直观，沿着测地线从 $x$ 走到 $y$ 的弧长就是 $\theta$，所以应该取 $\|u\| = \theta$。于是：
  $$
  y = \cos\theta\, x + \sin\theta\, \frac{u}{\|u\|}.
  $$
  移项可得：
  $$
  \sin\theta\, \frac{u}{\|u\|} = y - \cos\theta\, x.
  $$
  因此，向量 $y - \cos\theta\, x$ 的方向就是切向量 $u$ 的方向，而其长度应等于 $\sin\theta$（因为 $\frac{u}{\|u\|}$ 是单位向量）。我们来验证这一点。

  ### 4. 验证长度关系

  计算 $y - \cos\theta\, x$ 的模平方：
  $$
  \|y - \cos\theta\, x\|^2 = y^\top y - 2\cos\theta\, (x^\top y) + \cos^2\theta\, (x^\top x).
  $$
  由于 $x,y$ 都是单位向量，$x^\top x = y^\top y = 1$，且 $x^\top y = \cos\theta$，所以：
  $$
  \|y - \cos\theta\, x\|^2 = 1 - 2\cos\theta\cdot\cos\theta + \cos^2\theta = 1 - 2\cos^2\theta + \cos^2\theta = 1 - \cos^2\theta = \sin^2\theta.
  $$
  因为 $\theta \in [0,\pi]$，$\sin\theta \ge 0$，所以：
  $$
  \|y - \cos\theta\, x\| = \sin\theta.
  $$
  因此，当 $\theta \in (0,\pi)$ 时，$\sin\theta > 0$，我们可以写出：
  $$
  \frac{y - \cos\theta\, x}{\sin\theta} \quad\text{是一个单位向量，且与 }x\text{ 正交}.
  $$
  正交性验证：$x^\top (y - \cos\theta\, x) = x^\top y - \cos\theta\, (x^\top x) = \cos\theta - \cos\theta = 0$，所以它确实位于切平面 $T_x S^2$ 内。

  于是，切向量 $u$ 的方向已经确定，它的长度应为 $\theta$，所以：
  $$
  u = \theta \cdot \frac{y - \cos\theta\, x}{\sin\theta} = \frac{\theta}{\sin\theta}\, (y - \cos\theta\, x).
  $$
  这就是对数映射的显式公式：
  $$
  \log_x(y) = \frac{\theta}{\sin\theta}\, \bigl(y - (\cos\theta) x\bigr),\quad \theta = \arccos(x^\top y) \in (0,\pi).
  $$

  ### 5. 性质与极限情况

  - **切空间性**：容易验证 $x^\top \log_x(y) = 0$，因为 $x^\top (y - \cos\theta\, x) = 0$。
  - **当 $\theta \to 0$**（即 $y$ 趋近于 $x$）时，$\frac{\theta}{\sin\theta} \to 1$，而 $y - \cos\theta\, x$ 近似于 $y - x$（因为 $\cos\theta \approx 1 - \theta^2/2$），实际上更精确地，利用 $\cos\theta = 1 - \theta^2/2 + O(\theta^4)$，$y = x + \dot y \theta + \dots$，可以得到 $\log_x(y) \approx y - x$ 在切空间中的投影。极限情况 $\log_x(x) = 0$。
  - **当 $\theta = \pi$**（即 $y = -x$）时，$\sin\pi = 0$，公式失效，这是因为对径点没有唯一的对数映射（所有方向长度 $\pi$ 的切向量都映射到 $-x$）。在实际计算中需要单独处理这种情况，通常可以定义返回一个任意方向的切向量（例如随机方向），但更多时候我们避免处理对径点，或者通过其他方式处理。

  ### 6. 数值实现的注意事项

  在计算机实现中，对于很小的 $\theta$，直接计算 $\theta / \sin\theta$ 可能会遇到数值不稳定的问题（尽管 $\sin\theta \approx \theta$，但除法仍可能放大舍入误差）。通常采用泰勒展开来处理小角度：
  $$
  \frac{\theta}{\sin\theta} = 1 + \frac{\theta^2}{6} + O(\theta^4).
  $$
  当 $\theta$ 小于某个阈值（如 $10^{-6}$）时，直接用 $1$ 近似即可，或者用展开式以提高精度。此外，当 $\theta$ 接近 $\pi$ 时，$\sin\theta$ 很小，公式依然有效（因为分子 $\theta$ 非零，但若 $\theta$ 恰好为 $\pi$ 则需特殊处理）。对于 $\theta$ 接近 $\pi$ 但并非精确对径的情况，可以直接使用公式，因为 $\sin\theta$ 虽小但不为零，数值上通常可行。

  ### 7. 几何直观总结

  对数映射 $\log_x(y)$ 的几何意义是：从 $x$ 出发，沿着指向 $y$ 的大圆方向，取一个切向量，其长度等于球面距离 $\theta$。这个切向量由 $y$ 在切平面上的投影（减去法向分量）再缩放得到。公式中的因子 $\frac{\theta}{\sin\theta}$ 正是为了将投影向量的长度从 $\sin\theta$ 调整为 $\theta$。

  以上推导完整且严谨，它给出了球面上指数映射的逆运算的显式表达式，是黎曼几何中球面流形上处理切向量与流形点之间转换的重要工具。
  {{< /admonition >}}

**测地线条件流**：$\psi_t(x_0|x_1)=\exp_{x_0}(\kappa(t)\log_{x_0}(x_1))$，取 $\kappa(t)=t$。对 $t$ 求导得条件速度 $u_t(x_t|x_1)\in T_{x_t} S^2$。

**切空间投影**：$\mathrm{proj}_x(v)=v-(x^\top v)x$，把 $\mathbb{R}^3$ 中向量投影到 $T_x S^2$。

```python
import numpy as np

def _norm(x, axis=-1, keepdims=True):
    return np.linalg.norm(x, axis=axis, keepdims=keepdims)

# ---------- 球面 S^2：指数 / 对数 ----------
def sphere_log(x, y):
    """log_x(y): 从 x 指向 y 的切向量，y 在 S^2 上。"""
    cos_theta = np.clip(np.sum(x * y, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    # 避免 θ=0 处除零，用极限 θ/sinθ -> 1
    scale = np.where(theta < 1e-8, 1.0, theta / np.sin(theta))
    return scale * (y - cos_theta * x)

def sphere_exp(x, v):
    """exp_x(v): 从 x 沿切向量 v 走单位时间到达的点。"""
    n = _norm(v)
    n_safe = np.maximum(n, 1e-8)
    return np.cos(n) * x + np.sin(n) * (v / n_safe)

# ---------- 测地线条件流 ψ_t(x_0|x_1) = exp_{x_0}(κ(t) log_{x_0}(x_1))，κ(t)=t ----------
def geodesic_flow_t(x0, x1, t):
    """返回 x_t = ψ_t(x_0|x_1)，t 标量或与 batch 同形。"""
    v01 = sphere_log(x0, x1)
    return sphere_exp(x0, t * v01)

def geodesic_velocity_t(x0, x1, t, eps=1e-6):
    """条件速度 u_t(x_t|x_1) = d/dt ψ_t(x_0|x_1)，在 x_t 处的切向量。"""
    xt = geodesic_flow_t(x0, x1, t)
    # 数值求导: (ψ_{t+h} - ψ_{t-h}) / (2h)
    h = eps
    xt_plus = geodesic_flow_t(x0, x1, t + h)
    xt_minus = geodesic_flow_t(x0, x1, t - h)
    dxt = (xt_plus - xt_minus) / (2.0 * h)
    # 投影到 T_{x_t} S^2，保证是切向量
    return tangent_proj(xt, dxt)

def tangent_proj(x, v):
    """T_x S^2 投影: v - (x'v)x。"""
    return v - np.sum(x * v, axis=-1, keepdims=True) * x

# ---------- RCFM 单步：采样 (x_0,x_1), t -> x_t, u_t(x_t|x_1)，投影后 MSE ----------
def rcfm_loss_step(x0, x1, t, model_output):
    """
    x0, x1: (B,3) 单位向量; t: (B,1); model_output: (B,3) 网络原始输出。
    返回在切空间上与条件速度的 MSE（即 Bregman ∥u - u^θ∥_g^2）。
    """
    xt = geodesic_flow_t(x0, x1, t)
    u_true = geodesic_velocity_t(x0, x1, t)
    u_pred = tangent_proj(xt, model_output)
    return np.mean(np.sum((u_pred - u_true) ** 2, axis=-1))
```

- `geodesic_flow_t` 对应式 5.17 $\psi_t(x_0|x_1)=\exp_{x_0}(t\log_{x_0}(x_1))$。
- `geodesic_velocity_t` 给出条件速度场 $u_t(x_t|x_1)$，用作 RCFM 的监督；实际库中可由路径对 $t$ 的自动微分得到。
- `tangent_proj` 即 $\mathrm{proj}_x(v)$，保证损失在 $T_x M$ 上计算，与 5.5 节 Bregman $D_{x_t}(u,u^\theta)=\|u-u^\theta\|_g^2$ 一致。

---

## 10 小结与延伸阅读

第 5 章的核心是：**把欧氏 FM 的整套逻辑（概率路径、边际化、条件/边际损失、ODE 流）搬到黎曼流形上**，用测地线或预度量替代直线，用黎曼散度和切空间上的 Bregman 散度替代欧氏版本，从而在球面、李群等非欧数据上做可扩展的、无模拟（或低模拟）的 Flow Matching。

### 延伸阅读

- 原论文第 5 章：Riemannian Flow Matching 的完整假设与证明（假设 2、定理 10–11）。
- Chen & Lipman (2024)：基于测地线与预度量的条件流构造与无模拟训练。
- 代码库 `flow_matching.path.GeodesicProbPath`、`flow_matching.utils.manifolds.Sphere`：球面测地线 RCFM 实现。
- 第 4 章：欧氏 FM 的条件流与边际化（定理 3–4）作为对比。

