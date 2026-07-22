---
title: jl
date: 2025-07-23
tags: [.]
---

## 一、个人技术定位

### 1. 核心定位

建议将自己的职业定位概括为：

> 我是一名具备自动驾驶决策规划、车辆控制、机器人运动控制和学习型算法交叉背景的算法技术专家。我的核心优势不是只掌握某一种算法，而是能够从车辆或机器人动力学出发，在 Rule-based、Optimization-based 和 Learning-based 方法之间进行合理选型，并完成从算法原型、仿真验证、工程架构到大规模真实设备落地的完整闭环。

你的差异化能力主要体现在四点：

1. **算法跨度完整**：覆盖决策、规划、控制、机器人 IK/力控/WBC、模仿学习、强化学习和 VLA。
2. **量产经验扎实**：不仅做过论文复现或仿真，还承担过 2000 多台运营车辆的控制算法落地。
3. **软硬件理解深入**：理解车辆模型、执行器、传感器、求解器、实时线程和硬件延迟之间的联系。
4. **具备负责人能力**：能够定义技术路线、拆解指标、组织协作、管理风险，并推动算法从 0 到 1 落地。

---

## 二、简历中需要先校准的内容

### 1. 时间关系

### 2. 术语规范

建议统一为：

* Rule-based，而不是 RuleBase
* Learning-based，而不是 LearningBase
* Vibe Coding，而不是 VideCoding
* inverse kinematics（IK）
* whole-body control（WBC）
* imitation learning（IL）
* reinforcement learning（RL）
* π 系列，而不是 PI 系列
* cut-in、oncoming、lane change

### 3. 指标需要准备统计口径

所有简历指标都要能回答以下问题：

* 样本量是多少？
* 指标是在离线数据、开环仿真还是闭环仿真中统计？
* 对比基线是什么版本？
* 是否经过 A/B 实验？
* 指标是否具有统计显著性？
* 是否存在以舒适性换安全性、以召回率换精度的情况？
* 上线后指标是否发生回退？

例如“急刹降低 70%”不能只回答结果，要说明：

> 基于统一版本地图、预测、车辆参数和回放场景集，在场景识别召回基本一致的条件下，比较百公里纵向减速度低于某阈值的事件数；同时监控碰撞风险、通行效率和误减速，确认不是通过整体变保守获得指标提升。

---

# 三、自动驾驶 PNC 总体架构

## 1. 专家级回答框架

面对“你如何理解自动驾驶决策规划”时，可以这样回答：

> 决策规划本质上是在感知与预测存在不确定性的情况下，求解满足交通规则、碰撞约束、车辆动力学约束和舒适性要求的时空行为。工程上不能只看某个规划算法，而要同时考虑场景理解、交互预测、候选行为生成、轨迹优化、安全校验、控制可执行性和降级机制。

典型链路为：

$$
\text{Localization/Perception}
\rightarrow
\text{Prediction}
\rightarrow
\text{Scenario Understanding}
\rightarrow
\text{Behavior Decision}
\rightarrow
\text{Motion Planning}
\rightarrow
\text{Control}
$$

实际工程中还需要并行存在：

* ODD 判断
* 系统健康监控
* 安全仲裁
* Minimum Risk Maneuver
* 轨迹有效性检查
* 控制可执行性检查
* 远程守护或接管链路

## 2. 规划目标函数

轨迹规划通常可抽象为：

$$
\min_{\tau}
J(\tau)
=
w_s J_{\text{safety}}
+w_e J_{\text{efficiency}}
+w_c J_{\text{comfort}}
+w_r J_{\text{rule}}
+w_i J_{\text{interaction}}
+w_f J_{\text{feasibility}}
$$

其中：

* $J_{\text{safety}}$：碰撞、道路边界、风险距离。
* $J_{\text{efficiency}}$：速度、通过时间、进度。
* $J_{\text{comfort}}$：加速度、加加速度、横摆角速度。
* $J_{\text{rule}}$：车道线、限速、交通规则。
* $J_{\text{interaction}}$：对其他交通参与者行为的影响。
* $J_{\text{feasibility}}$：曲率、转角、转角速度和车辆动力学可执行性。

技术专家需要强调：

> 权重并不是单纯靠人工拍脑袋调参。需要先把硬安全要求转化为约束，再对软目标进行归一化和权衡；同时使用分场景权重、动态风险预算或多阶段优化，避免一个全局权重覆盖所有场景。

---

# 四、Rule-based 决策规划

## 1. Rule-based 方法的价值

Rule-based 方法的核心优势是：

* 可解释
* 可验证
* 安全边界清晰
* 长尾场景可定向修复
* 便于满足工程验收与监管要求

缺点包括：

* 规则组合容易膨胀
* 场景边界不连续
* 多车交互难以显式枚举
* 参数耦合严重
* 跨城市和跨 ODD 泛化成本高

专家回答不应简单说“规则方法稳定、学习方法泛化好”，而应回答：

> Rule-based 更适合承担安全边界、法规约束和确定性场景处理；Learning-based 更适合进行高维场景表征、多模意图建模和复杂交互策略学习。量产系统通常需要二者分层融合，而不是绝对替代。

## 2. 场景状态机

Cut-in、Oncoming、Lane Change 场景可以抽象成：

$$
\text{Inactive}
\rightarrow
\text{Potential}
\rightarrow
\text{Confirmed}
\rightarrow
\text{Interaction}
\rightarrow
\text{Recover}
$$

状态转移条件可以由以下信号构成：

* 横向距离及变化率
* 目标航向角
* 车道归属概率
* 预测轨迹跨线概率
* 相对速度
* TTC、THW、PET
* 目标历史轨迹稳定性
* 场景持续时间
* 预测多模分布

重点不是状态数量，而是：

1. 防止状态抖动。
2. 保持进入和退出条件的滞回。
3. 避免预测单帧跳变引起规划突变。
4. 区分风险确认与行为确认。
5. 保留场景上下文和历史决策。

## 3. 风险指标

常见指标包括：

$$
TTC = \frac{\Delta s}{-\Delta v}
$$

仅在目标距离正在收敛时有效。

$$
THW = \frac{\Delta s}{v_{\text{ego}}}
$$

还应结合：

* 预测碰撞时间
* 最近交汇时间
* 最小预测距离
* 轨迹重叠概率
* 制动需求减速度
* RSS 类安全距离
* 可达集或安全包络

专家回答应指出：

> TTC 只是一维相对运动近似，面对横穿、会车、切入和多模预测时容易失真。因此量产系统通常结合时空轨迹重叠、目标意图概率、车辆可制动能力以及场景上下文进行综合风险评估。

---

# 五、搜索、采样与优化规划

## 1. A* 与 Hybrid A*

A* 在离散图上使用：

$$
f(n)=g(n)+h(n)
$$

其中：

* $g(n)$：从起点到当前节点的累计代价。
* $h(n)$：当前节点到目标的启发式代价。

A* 的问题是普通栅格搜索通常不包含车辆航向和非完整约束。

Hybrid A* 将状态扩展为：

$$
x=[p_x,p_y,\theta]
$$

节点扩展通过车辆运动学模型完成，因此能够产生满足最小转弯半径和航向连续性的路径。工程上还会结合解析扩展、二维距离启发、非完整约束启发、Voronoi 代价和轨迹平滑。Hybrid A* 是自主驾驶搜索规划中的经典实用方案。([斯坦福人工智能实验室][1])

面试时应能够回答：

* 为什么 Hybrid A* 不直接输出控制轨迹？
* 离散步长如何影响精度和计算量？
* 如何设计启发函数保证搜索效率？
* 如何避免倒车与前进频繁切换？
* 如何进行曲率和曲率变化率平滑？
* 如何处理窄通道搜索失败？

## 2. State Lattice

State Lattice 不在连续控制空间任意扩展，而是预先构造满足动力学约束的运动基元。

典型流程：

1. 离散状态空间。
2. 定义相邻状态连接关系。
3. 求解两点边值问题。
4. 离线生成 motion primitives。
5. 在线搜索并进行碰撞检测。
6. 对结果进行平滑和速度规划。

其优势是：

* 运动基元本身动力学可行。
* 在线搜索速度较高。
* 可统一处理位置、航向和曲率。
* 适合低速机器人、泊车和窄空间规划。

不足是：

* 离散化导致分辨率限制。
* 基元库过大会增加搜索量。
* 基元库过小会降低可达性。
* 车辆模型和参数变化时可能需要重新生成。

State Lattice 的核心是用满足微分约束的运动基元构造离散状态图。([Wiley Online Library][2])

## 3. DP/QP 规划

典型 Frenet 规划可拆分为：

1. DP 搜索粗路径或粗速度。
2. QP 对轨迹进行连续优化。

路径 QP 可写为：

$$
\min_l
\sum_i
w_l l_i^2+
w_{dl}(l_i')^2+
w_{ddl}(l_i'')^2+
w_{dddl}(l_i''')^2
$$

约束包含：

$$
l_i^{\min}\le l_i\le l_i^{\max}
$$

速度规划则在 $s$-$t$ 空间优化速度、加速度和 jerk。

需要掌握的追问：

* DP 的节点代价和边代价如何设计？
* 障碍物边界如何投影到 Frenet 坐标系？
* 道路曲率较大时 Frenet 有什么问题？
* QP 无解时如何软化约束？
* 路径和速度分开规划会损失什么？
* 如何实现横纵向联合规划？

## 4. SQP

SQP 用二次规划近似非线性优化问题：

$$
\min_x f(x),\quad
g(x)=0,\quad h(x)\le0
$$

每轮在当前点对约束线性化，对拉格朗日函数进行二阶近似，求解 QP 子问题，再通过 line search 或 trust region 更新。

优势：

* 能处理一般非线性目标和约束。
* 局部收敛速度较好。
* 适合非线性轨迹优化和 NMPC。

问题：

* 依赖初值。
* 可能收敛到局部最优。
* 非凸障碍物约束可能导致不可行。
* 计算确定性弱于凸 QP。

## 5. iLQR/DDP

iLQR 通过不断执行以下步骤优化控制序列：

1. 使用当前控制序列前向 rollout。
2. 沿当前轨迹线性化动力学。
3. 对代价做二次近似。
4. 通过 Riccati 递推进行反向传播。
5. 得到前馈项 $k_t$ 和反馈增益 $K_t$。
6. 使用 line search 更新控制量。

$$
u_t^{new}
=
u_t+\alpha k_t+K_t(x_t^{new}-x_t)
$$

iLQR 适合非线性系统的局部轨迹优化，但原始形式不擅长一般不等式约束，需要通过 barrier、penalty、augmented Lagrangian 或 constrained iLQR 扩展。iLQR/DDP 类方法本质上利用动力学与目标函数的局部线性二次近似。([arXiv][3])

---

# 六、动态交互场景：Cut-in、Oncoming、Lane Change

## 1. Cut-in

Cut-in 的核心不是检测车辆是否跨线，而是估计：

* 是否存在切入意图？
* 什么时候开始切入？
* 切入后占据多少纵向空间？
* 自车减速是否会诱发后车风险？
* 目标是否可能放弃切入？
* 是否存在加速通过或横向避让方案？

完整方法可以包括：

1. 多模轨迹预测。
2. 车道级意图估计。
3. 风险区域构建。
4. 多行为候选生成。
5. 交互 rollout。
6. 纵向和横向联合决策。
7. 控制可执行性评估。

典型候选行为：

* 保持
* 温和减速让行
* 强减速避碰
* 加速通过
* 轻微横向避让
* 换道避让

专家需要强调：

> Cut-in 优化的难点不是让系统更早刹车，而是在安全、误减速和舒适性之间建立合理边界。过早制动会产生 ghost braking，过晚制动会造成急刹或碰撞风险。

## 2. Oncoming

Oncoming 场景通常涉及：

* 借道障碍
* 窄路会车
* 静态路权冲突
* 道路边界不规则
* 双方意图相互影响

“双 S 空间”可以理解为：

* 自车沿自身参考线的纵向进度 $s_e$
* 对向车沿其参考线的纵向进度 $s_o$

在 $s_e$-$s_o$ 空间中构建冲突区：

$$
\mathcal{C}=
\{(s_e,s_o)\mid
\text{两车几何包络重叠}\}
$$

优化的目标是选择从冲突区上方或下方绕行，对应：

* 自车先行
* 自车让行

该方法的优势是将复杂二维几何交互转化为进度空间中的避障问题。

需要准备的追问：

* 冲突区如何计算？
* 预测误差如何扩张边界？
* 双方都让行时如何打破僵局？
* 对方不按预测运动怎么办？
* 如何避免策略频繁切换？
* 如何定义先行权和让行权？

## 3. Lane Change

换道决策通常分为：

1. 必要性判断。
2. 目标车道收益评估。
3. 可行间隙判断。
4. 风险评估。
5. 换道轨迹生成。
6. 换道中持续重规划。
7. 中止或回退策略。

关键指标：

* 前后车间隙
* 相对速度
* 预测加速度需求
* 目标车道收益
* 换道完成时间
* 对后车造成的制动影响
* 换道过程碰撞概率
* 横向舒适性

---

# 七、Learning-based 决策规划

## 1. Learning-based 模型的输入输出

常见输入：

* 自车历史状态
* 目标历史轨迹
* 车道中心线和边界
* 路口拓扑
* 障碍物类型
* 信号灯
* 导航信息
* 场景标签
* 历史规划结果

常见输出：

* 行为分类
* 风险分数
* 目标速度
* 目标点
* 未来轨迹
* 多模轨迹及概率
* 控制序列

## 2. Query-based 模型

Query-based 架构使用一组可学习 query：

$$
Q=\{q_1,q_2,\ldots,q_N\}
$$

通过 cross-attention 与场景 token 交互。

不同 query 可以表示：

* 不同目标实例
* 不同轨迹模式
* 不同决策意图
* 不同时间段
* 不同行为候选

需要准备：

* Query 数量为什么这样设置？
* Query collapse 如何处理？
* 如何保证不同 query 的多样性？
* Hungarian matching 是否需要？
* 多模轨迹概率如何标定？
* Query 输出如何与规则安全层融合？

## 3. Collision Loss

简单的点距离碰撞损失：

$$
L_{\text{collision}}
=
\sum_{t,j}
\max(0,d_{\text{safe}}-d_{t,j})^2
$$

工程上应进一步考虑：

* 自车和目标车辆几何包络。
* 车身朝向。
* 随时间变化的安全距离。
* 目标预测概率。
* 场景类别。
* hard negative mining。
* 碰撞损失与轨迹回归损失的梯度平衡。

若碰撞 loss 权重过大，模型可能：

* 输出过度保守轨迹。
* 降低多模性。
* 牺牲行驶效率。
* 将轨迹推向道路边界。

因此需要同时监控安全、效率、舒适性和轨迹分布。

## 4. 数据闭环

专家应从以下层次回答：

### 场景发现

* 规则触发
* 模型不确定性
* 离线指标异常
* 人工接管
* 急刹事件
* 规划与控制偏差
* 仿真失败

### 数据生产

* 场景切片
* 时间对齐
* 坐标转换
* 障碍物过滤
* 地图匹配
* 标签生成
* 数据质量检查

### 数据采样

* 类别均衡
* 难例过采样
* 时间去重
* 地域去重
* 车辆去重
* 场景阶段均衡
* 长尾场景增强

### 评测闭环

* 离线开环
* Log replay
* Reactive simulation
* 闭环仿真
* Shadow mode
* 小流量灰度
* 全量发布

---

# 八、One-shot、Diffusion 与 Flow Matching 轨迹模型

## 1. One-shot 模型

One-shot 模型直接输出轨迹：

$$
\tau=f_\theta(c)
$$

其中 $c$ 是环境条件。

优势：

* 推理快
* 结构简单
* 易于实时部署

不足：

* 多模分布容易平均化
* 面对歧义场景可能输出折中轨迹
* 难以显式进行测试时引导
* 对长尾和分布外场景较敏感

## 2. Diffusion 轨迹模型

前向过程可写成：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,I)
$$

模型可以预测噪声、原始轨迹或 velocity：

$$
\epsilon_\theta(x_t,t,c),\quad
x_{0,\theta}(x_t,t,c),\quad
v_\theta(x_t,t,c)
$$

推理时从噪声或先验轨迹出发，逐步去噪获得轨迹。

扩散模型的主要价值：

* 建模多模轨迹分布。
* 可生成多个行为候选。
* 可通过 guidance 注入安全或目标条件。
* 相比确定性回归不容易直接输出均值轨迹。

主要问题：

* 多步采样影响实时性。
* 训练目标和闭环驾驶目标不完全一致。
* 低概率危险模式仍可能被采样。
* 轨迹多样性不等于行为质量。

DiffusionDrive 通过多模 anchor 和截断扩散，将传统多步去噪压缩为少量步骤，用于提高自动驾驶实时性。([arXiv][4])

Diffusion Planner 则将预测和规划放在统一模型中，并通过引导机制调整轨迹行为。([arXiv][5])

## 3. Flow Matching

Flow Matching 学习一个随时间变化的向量场：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c)
$$

最简单的线性概率路径：

$$
x_t=(1-t)x_0+t x_1
$$

对应目标速度：

$$
u_t=x_1-x_0
$$

训练目标：

$$
L_{\text{FM}}
=
\mathbb{E}
\left[
\|v_\theta(x_t,t,c)-u_t\|^2
\right]
$$

推理时通过 ODE 求解：

$$
x_1=x_0+\int_0^1 v_\theta(x_t,t,c)\,dt
$$

Flow Matching 可以看作直接学习从源分布到目标分布的概率流向量场，训练时不需要实际模拟完整生成过程。([arXiv][6])

## 4. Diffusion 与 Flow Matching 对比

| 维度       | Diffusion          | Flow Matching |
| -------- | ------------------ | ------------- |
| 学习目标     | 去噪或 score/velocity | 概率流向量场        |
| 推理形式     | SDE/ODE 逐步采样       | ODE 积分        |
| 路径       | 常用高斯扩散路径           | 可选择线性、OT 等路径  |
| 实时性      | 依赖采样步数             | 路径较直时可少步求解    |
| 多模能力     | 强                  | 强             |
| Guidance | 成熟                 | 同样可以加入        |
| 部署难点     | 多步时延、随机性           | ODE 离散误差、路径设计 |

不要回答“Flow Matching 一定比 Diffusion 快”。更准确的回答是：

> 实际速度取决于概率路径曲率、网络结构、ODE 求解方法和允许误差。如果 Flow Matching 学到的流比较直，可以用较少积分步；但复杂分布下仍可能需要更多函数评估。

## 5. 两段式生成轨迹架构

你简历中的“两段式”应准备成清晰架构：

### 第一阶段：意图或粗轨迹生成

输出：

* goal point
* behavior anchor
* 多模粗轨迹
* 车道级意图
* 速度 profile

### 第二阶段：条件生成或优化

以第一阶段输出为 prior：

$$
\tau_0\sim p_{\text{anchor}}(\tau|c)
$$

再通过 Diffusion 或 Flow Matching 精修：

$$
\tau^\star
=
\text{Generator}(\tau_0,c)
$$

优势：

* 缩小生成搜索空间。
* 提升收敛速度。
* 保留多模意图。
* 降低生成不合理模式的概率。
* 容易与规则候选或安全约束融合。

需要回答：

* 第一阶段错误是否会限制第二阶段？
* 如何保持 anchor 多样性？
* anchor 与生成模式如何对应？
* 如何防止 mode collapse？
* 训练和推理条件是否一致？
* 如何对生成轨迹做最终安全过滤？

---

# 九、闭环评测体系

## 1. 为什么不能只看 ADE/FDE

$$
ADE=\frac{1}{T}\sum_t\|\hat{p}_t-p_t\|
$$

$$
FDE=\|\hat{p}_T-p_T\|
$$

ADE/FDE 衡量与专家轨迹的距离，但不能充分反映：

* 是否碰撞
* 是否压线
* 是否舒适
* 是否满足动力学
* 是否造成死锁
* 是否能完成路线
* 是否存在闭环状态分布漂移

因此要建立四级指标：

### 模型指标

* ADE/FDE
* Precision/Recall
* NLL
* minADE/minFDE
* calibration
* mode coverage

### 轨迹质量指标

* 碰撞率
* 越界率
* jerk
* 曲率
* 动力学可行率
* 安全距离

### 闭环驾驶指标

* route completion
* intervention
* progress
* deadlock
* emergency braking
* 舒适性
* 通行效率

### 量产指标

* 百公里急刹
* 百公里接管
* 百公里异常停车
* 远程呼叫率
* 运营完成率
* 车辆可用率

## 2. 仿真闭环横向对比

为了保证公平性，需要固定：

* 相同场景集合
* 相同感知与预测输入
* 相同车辆模型
* 相同控制器
* 相同地图版本
* 相同随机种子或多随机种子统计
* 相同终止条件
* 相同安全裁决

并对 One-shot、Diffusion、Flow Matching 比较：

* 成功率
* 碰撞率
* 接管率
* 轨迹多样性
* 舒适性
* 推理时间
* P95/P99 时延
* 模型显存
* 长尾场景表现

---

# 十、最优控制与车辆控制

## 1. LQR

系统：

$$
x_{k+1}=Ax_k+Bu_k
$$

目标：

$$
J=
\sum_{k=0}^{N-1}
(x_k^\top Qx_k+u_k^\top Ru_k)
+x_N^\top Px_N
$$

通过 Riccati 方程得到：

$$
u_k=-Kx_k
$$

优点：

* 数学形式清晰。
* 计算量小。
* 局部稳定性容易分析。
* 适合线性化后的轨迹跟踪。

不足：

* 不直接处理输入与状态约束。
* 性能依赖线性化模型。
* 难以处理强非线性和复杂安全约束。

## 2. MPC

MPC 每个控制周期求解：

$$
\min_{x_k,u_k}
\sum_{k=0}^{N-1}
\|x_k-x_k^{ref}\|_{Q}^2+
\|u_k-u_k^{ref}\|_{R}^2
+\|x_N-x_N^{ref}\|_{P}^2
$$

满足：

$$
x_{k+1}=f(x_k,u_k)
$$

$$
x_k\in\mathcal{X},\quad
u_k\in\mathcal{U}
$$

只执行第一个控制量，然后滚动优化。

MPC 的核心优势不是“预测未来”，而是能够在有限时域内统一处理：

* 多变量耦合
* 状态约束
* 控制约束
* 前馈与反馈
* 未来参考轨迹
* 障碍物和安全边界

## 3. 运动学与动力学模型

### 运动学自行车模型

$$
\dot x=v\cos\psi
$$

$$
\dot y=v\sin\psi
$$

$$
\dot\psi=\frac{v}{L}\tan\delta
$$

适合：

* 低速
* 小侧偏
* 轮胎不接近极限
* 计算资源有限

### 动力学自行车模型

常见状态：

$$
x=[e_y,e_\psi,v_y,r]
$$

其中：

* $e_y$：横向误差
* $e_\psi$：航向误差
* $v_y$：横向速度
* (r)：横摆角速度

模型需要考虑：

* 前后轴侧偏角
* 轮胎侧偏刚度
* 质心位置
* 横摆惯量
* 纵向速度

适合：

* 高速
* 大横向加速度
* 对动态响应要求较高的场景

### 融合模型

低速时动力学模型容易出现与 $1/v_x$ 相关的数值问题，可使用速度相关权重：

$$
x_{next}
=
\lambda(v)x_{next}^{dyn}
+
(1-\lambda(v))x_{next}^{kin}
$$

$$
\lambda(v)=
\frac{1}{1+\exp[-a(v-v_0)]}
$$

面试重点应放在：

* 为什么低速使用运动学模型？
* 为什么高速需要动力学模型？
* 如何保证切换连续？
* 如何识别轮胎侧偏参数？
* 参数变化如何影响闭环稳定性？

## 4. 带挂车辆模型

带挂车辆核心状态通常包括：

* 牵引车位置和航向
* 挂车航向
* 铰接角
* 横向速度
* 横摆角速度
* 转角或转角速度

关键难点：

1. 非刚体多体耦合。
2. 铰接角可能造成 jackknife。
3. 高速下侧偏与横摆稳定性问题。
4. 倒车时系统呈现更强不稳定性。
5. 载重会改变惯量与轮胎特性。
6. 长预测时域带来计算压力。

专家回答：

> 对带挂重卡，不能把挂车简单当作几何尾部。挂车运动会通过铰接机构反作用于牵引车，必须在模型中显式描述牵引车、挂车横摆和铰接角动力学。同时需要对铰接角、横向加速度、横摆角速度和轮胎侧偏设置安全约束。

## 5. 安全边界 MPC

将障碍物投影约束加入 MPC：

$$
g(x_k)\ge d_{\text{safe}}
$$

若非线性约束实时求解困难，可以：

* 局部线性化障碍物边界。
* 使用 corridor 约束。
* 使用半空间约束。
* 引入 slack variable。
* 使用 sequential convexification。
* 使用 signed distance approximation。

软约束形式：

$$
g(x_k)+s_k\ge d_{\text{safe}},\quad s_k\ge0
$$

目标增加：

$$
J_{\text{slack}}=\rho\sum_ks_k^2
$$

需要说明：

> Slack 不是为了允许系统随意违反安全边界，而是防止数值上完全无解。真正的硬碰撞边界仍应由独立安全层兜底。

---

# 十一、优化求解器

## 1. OSQP

OSQP 求解凸二次规划：

$$
\min_x \frac12x^\top Px+q^\top x
$$

$$
l\le Ax\le u
$$

其特点包括：

* 基于 operator splitting/ADMM。
* 支持稀疏矩阵。
* 支持 warm start。
* 问题结构固定时可复用矩阵分解。
* 适合线性 MPC 和轨迹 QP。

OSQP 官方定义的标准问题即为带线性上下界约束的凸 QP。([OSQP][7])

工程优化点：

* 预分配内存。
* 固定稀疏结构。
* 更新数值而不是重建问题。
* 归一化状态和控制量。
* 合理设置 absolute/relative tolerance。
* 监控 primal/dual residual。
* 设置最大迭代次数。
* 准备求解失败回退方案。

## 2. Ipopt

Ipopt 用于一般大规模非线性规划，并采用内点线搜索过滤方法寻找局部解。([GitHub][8])

适合：

* NMPC
* 非线性轨迹优化
* 参数辨识
* 一般非线性等式和不等式约束

需要提供：

* 目标函数
* 梯度
* 约束
* Jacobian
* Hessian 或近似 Hessian

工程难点：

* 初值敏感。
* 非凸问题只能保证局部解。
* 自动微分和稀疏结构影响性能。
* 终止条件需要和实时系统匹配。
* 不能把单次平均耗时当成实时保证，要关注 P99 和最坏时间。

## 3. 求解器选型方法

| 问题              | 优先选择                  |
| --------------- | --------------------- |
| 凸 QP、固定结构、高频求解  | OSQP、qpOASES          |
| 一般 NLP          | Ipopt                 |
| 小规模高频 MPC       | 定制 Riccati/condensing |
| 强 warm-start 需求 | active-set 或 OSQP     |
| 非线性动力学、一般约束     | SQP/Ipopt             |
| 局部轨迹优化          | iLQR/DDP              |
| 嵌入式确定时延         | 代码生成或定制求解器            |

专家不应回答“某求解器最快”，而应回答：

> 求解器选择取决于问题凸性、变量规模、稀疏结构、约束类型、warm-start 效果、数值条件、最坏时延和硬件平台。平均耗时不是唯一指标，还需要关注收敛率、失败率、P99 时延和故障恢复。

---

# 十二、预控制与规划控制一体化

你“在决策中嵌入 MPC 预控制”的工作是非常重要的专家亮点。

## 1. 核心问题

规划模块通常根据几何、规则和舒适性选择轨迹，但最终轨迹可能：

* 控制器跟不上。
* 转角变化过快。
* 横向误差过大。
* 执行器延迟导致超调。
* 高速下动力学不可行。

## 2. 预控制架构

$$
\{\tau_1,\tau_2,\ldots,\tau_N\}
\rightarrow
\text{Parallel MPC Rollout}
\rightarrow
\{S_1,S_2,\ldots,S_N\}
\rightarrow
\text{Candidate Selection}
$$

评分可以包括：

$$
S_i=
w_1 E_{\text{tracking}}
+w_2 U_{\text{control}}
+w_3 R_{\text{saturation}}
+w_4 R_{\text{stability}}
+w_5 R_{\text{safety}}
$$

价值：

* 将控制可执行性提前注入决策。
* 避免选择几何可行但控制不可行的轨迹。
* 使规划和控制指标一致。
* 提升高速、大曲率和极限工况表现。

面试回答：

> 这不是简单把控制器复制到规划模块，而是用低成本前向闭环预测评估候选轨迹。需要解决线程资源、候选数量、模型一致性、初始控制状态同步以及超时降级问题。

---

# 十三、状态估计、参数辨识与底盘控制

## 1. EKF 转向零偏估计

系统状态可定义为：

$$
x=[b_\delta,\ldots]
$$

其中 $b_\delta$ 是转向零偏。

转向测量：

$$
\delta_{\text{real}}
=
\delta_{\text{sensor}}-b_\delta
$$

利用横摆角速度、车辆运动学模型或横向运动误差更新偏置。

EKF 步骤：

1. 状态预测。
2. 协方差预测。
3. 计算创新。
4. 计算 Kalman 增益。
5. 状态更新。
6. 协方差更新。

需要注意：

* 直线低速时可观测性较弱。
* 轮胎侧偏会污染零偏估计。
* 需要门限控制和工况筛选。
* 参数更新速度不能过快。
* 必须避免估计异常直接影响控制。

## 2. 载重辨识

纵向动力学：

$$
m\dot v=
F_{\text{drive}}
-F_{\text{brake}}
-F_{\text{roll}}
-F_{\text{air}}
-F_{\text{slope}}
$$

若能够估计驱动力、坡度和阻力，可反推出质量 (m)。

工程难点：

* 驱动扭矩存在误差。
* 制动压力与实际制动力不完全线性。
* 坡度估计误差显著。
* 低加速度时质量不可观。
* 风阻和滚阻参数随环境变化。

可使用：

* RLS
* EKF
* 滑动窗口最小二乘
* 多模型滤波
* 工况触发式辨识

## 3. 执行器模型

转向执行器可近似为：

$$
\tau\dot\delta+\delta
=
K\delta_{\text{cmd-delay}}
$$

还需考虑：

* 死区
* 饱和
* 回差
* 速率限制
* 通信延迟
* 温度影响
* 电压影响
* 不同车辆个体差异

控制算法专家需要说明：

> 很多控制误差不是控制律本身造成，而是执行器动态、传感器时间戳和通信链路没有建模。量产控制优化必须先拆分模型误差、定位误差、底盘误差和控制器误差。

---

# 十四、机械臂运动学与 IK

## 1. 正运动学

机械臂末端位姿：

$$
T_{0n}(q)
=
T_{01}(q_1)T_{12}(q_2)\cdots T_{n-1,n}(q_n)
$$

## 2. 微分运动学

$$
\dot x=J(q)\dot q
$$

其中 (J(q)) 为 Jacobian。

基本伪逆：

$$
\dot q=J^\dagger\dot x_d
$$

阻尼最小二乘：

$$
\dot q
=
J^\top(JJ^\top+\lambda^2I)^{-1}\dot x_d
$$

适合接近奇异位形时提升稳定性。

带零空间任务：

$$
\dot q
=
J^\dagger\dot x_d+
(I-J^\dagger J)\dot q_0
$$

零空间可用于：

* 远离关节限位
* 远离奇异点
* 减小关节速度
* 保持舒适姿态
* 避免自碰撞

## 3. IKFast、Pinocchio、Mink、Pink、PlaCo 对比

### IKFast

* 符号或解析 IK。
* 运行速度快。
* 适合固定机械结构。
* 可输出多组解析解。
* 对模型结构和生成过程有要求。
* 不适合动态增加复杂任务约束。

### Pinocchio

* 刚体动力学和运动学库。
* 提供 FK、Jacobian、逆动力学、质心动力学等。
* 适合作为优化 IK 和 WBC 的模型基础。

### Pink/Mink/PlaCo

通常基于任务优化或 QP：

$$
\min_{\dot q}
\sum_iw_i\|J_i\dot q-v_i^\star\|^2
$$

并可加入：

* 关节限位
* 速度约束
* 姿态任务
* 碰撞约束
* 多末端任务

专家回答要点：

> 在 PICO 遥操项目中选择 IKFast，不是认为解析 IK 在所有情况下都更好，而是当前机械臂结构固定、任务主要是单臂位姿跟踪、设备计算资源有限，并且要求低于 5 ms 的稳定延迟。若任务扩展到双臂协同、避碰、质心或全身约束，则会优先考虑基于 QP 的数值 IK 或 WBC。

---

# 十五、遥操系统

典型链路：

$$
\text{VR Pose}
\rightarrow
\text{Coordinate Calibration}
\rightarrow
\text{Motion Retargeting}
\rightarrow
\text{IK}
\rightarrow
\text{Joint Filtering}
\rightarrow
\text{Robot Command}
$$

需要处理：

* 坐标系标定
* 人臂和机械臂尺寸差异
* 末端位姿映射
* 关节限位
* 奇异点
* 网络延迟
* VR 跟踪丢失
* 指令平滑
* 安全速度限制
* 急停和恢复

常见滤波：

* 低通滤波
* One Euro Filter
* Savitzky–Golay
* 速度/加速度限幅
* 轨迹插值

专家回答：

> 遥操系统不能只追求末端位置误差，还需要平衡时延、抖动、关节连续性和操作者体感。滤波越强，抖动越小，但交互时延越大，因此需要根据位置、姿态和不同频段分别设计滤波参数。

---

# 十六、导纳与阻抗控制

## 1. 阻抗控制

目标是建立位姿误差与作用力之间的关系：

$$
F
=
M_d(\ddot x_d-\ddot x)
+
D_d(\dot x_d-\dot x)
+
K_d(x_d-x)
$$

输入通常是期望位置，输出为力或力矩命令。

适合：

* 力矩控制机械臂
* 动力学模型较准确
* 需要直接控制交互阻抗

## 2. 导纳控制

目标动力学：

$$
M_d\ddot x_r+
D_d\dot x_r+
K_dx_r
=
F_{\text{ext}}
$$

外力作为输入，生成位姿修正量，再交给位置控制器。

适合：

* 底层是高刚度位置控制接口
* 无法直接下发力矩
* 工业机械臂改造

## 3. 参数选型

* $M_d$ 大：响应慢，抗扰平稳。
* $D_d$ 大：振荡小，但操作沉重。
* $K_d$ 大：位置保持强，但柔顺性降低。

阻尼常参考：

$$
D_d=2\zeta\sqrt{M_dK_d}
$$

其中 $\zeta\approx 1$ 对应临界阻尼附近。

## 4. 实际工程问题

* 力传感器零偏
* 工具重力补偿
* 摩擦补偿
* 力信号噪声
* 碰撞瞬间冲击
* 离散系统稳定性
* 控制周期抖动
* 奇异位形
* 外力到关节力矩映射

$$
\tau_{\text{ext}}=J^\top F_{\text{ext}}
$$

专家回答：

> 导纳和阻抗的选择首先取决于底层硬件接口。位置型伺服系统更适合外层导纳，力矩可控且动力学模型可靠的平台更适合阻抗或操作空间力控。实际稳定性还受到控制周期、滤波延迟、结构柔性和通信延迟影响。

---

# 十七、WBC 全身控制

浮动基机器人动力学：

$$
M(q)\ddot q+h(q,\dot q)
=
S^\top\tau+
J_c^\top\lambda
$$

其中：

* $S$：驱动关节选择矩阵。
* $\tau$：关节力矩。
* $J_c$：接触 Jacobian。
* $\lambda$：接触力。

典型 WBC QP：

$$
\min_{\ddot q,\tau,\lambda}
\sum_i
w_i
\|J_i\ddot q+\dot J_i\dot q-\ddot x_i^\star\|^2
$$

满足：

* 浮动基动力学
* 接触约束
* 摩擦锥
* 关节力矩限制
* 关节位置和速度限制
* 自碰撞约束

任务可以包括：

* 质心跟踪
* 躯干姿态
* 足端位置
* 手部位置
* 动量控制
* 接触力分配
* 关节姿态

HQP 使用严格优先级：

1. 动力学和接触安全。
2. 平衡和质心。
3. 末端任务。
4. 姿态和能耗优化。

WBC 常通过操作空间控制或分层 QP 统一处理多任务、接触力和物理约束。([arXiv][9])

---

# 十八、DRL 全身运动控制

## 1. 基本建模

状态：

$$
s_t=
[q_t,\dot q_t,
R_t,\omega_t,
v_t,
c_t,
r_t]
$$

动作通常是：

* 目标关节位置
* 关节力矩
* PD residual
* 目标关节速度

常见形式：

$$
\tau=
K_p(q_{\text{target}}-q)
-K_d\dot q
$$

策略输出 $q_{\text{target}}$。

## 2. DeepMimic

DeepMimic 使用参考动作跟踪奖励和任务奖励：

$$
r_t=
w_p r_p+
w_v r_v+
w_e r_e+
w_c r_c+
w_t r_{\text{task}}
$$

其中分别对应：

* 关节姿态
* 关节速度
* 末端位置
* 质心
* 任务目标

DeepMimic 的特点是显式跟踪动作片段，能够训练动态动作，同时通过 RL 获得扰动恢复能力。([XbPeng][10])

问题：

* 奖励项多。
* 不同动作需要调权重。
* 对参考相位依赖较强。
* 大规模混合动作训练困难。

## 3. AMP

AMP 增加判别器，区分：

* 专家动作状态转移
* 策略动作状态转移

策略获得风格奖励：

$$
r_{\text{style}}
=
-\log(1-D(s_t,s_{t+1}))
$$

优势：

* 减少手工设计逐项跟踪奖励。
* 能学习动作分布而非严格逐帧跟踪。
* 可将动作风格与任务奖励结合。

不足：

* GAN 训练可能不稳定。
* 判别器过强会导致策略梯度不足。
* 仍可能发生 mode collapse。
* 训练新任务常需要重新联合训练。

AMP 将对抗模仿学习作为运动先验，与任务奖励共同训练控制策略。([arXiv][11])

## 4. ASE

ASE 在 AMP 基础上增加 latent skill：

$$
a_t=\pi(a|s_t,z)
$$

并通过互信息目标保证不同 (z) 产生不同技能。

优势：

* 从无标签动作集合学习技能空间。
* 一个底层策略覆盖多个运动技能。
* 下游任务只需要训练高层 latent policy。

ASE 将对抗模仿与无监督技能发现结合，形成可复用的技能嵌入空间。([arXiv][12])

## 5. ADD

ADD 使用 differential discriminator 学习不同动作维度和误差之间的自适应跟踪目标，从而减少针对不同角色和动作手工设计奖励的成本。([arXiv][13])

面试时可以总结：

| 方法        | 核心思想        | 主要问题       |
| --------- | ----------- | ---------- |
| DeepMimic | 显式参考动作跟踪    | 奖励设计复杂     |
| AMP       | 对抗式动作分布匹配   | 对抗训练不稳定    |
| ASE       | latent 技能空间 | 高层控制仍需训练   |
| ADD       | 自适应学习跟踪目标   | 系统复杂、训练调试难 |

## 6. BeyondMimic

BeyondMimic 将高质量动作跟踪与 guided diffusion 结合，希望从“跟踪已有动作”进一步扩展到测试时根据代价引导组合新动作。([arXiv][14])

面试时需要避免把“研读复现”表述为“真实机器人完整落地”。可以明确分为：

* 已完成论文与代码分析。
* 已完成仿真训练复现。
* 已分析 Sim2Real 问题。
* 尚未或已经完成实机验证。

---

# 十九、强化学习基础

## 1. PPO

PPO clipped objective：

$$
L^{CLIP}(\theta)
=
\mathbb{E}
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
$$

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
$$

意义：

* 限制单次策略更新幅度。
* 避免策略因为少量样本发生剧烈变化。
* 能对一批 on-policy 数据执行多轮 minibatch 更新。

PPO 通过 clipped surrogate objective 在实现复杂度、样本利用和训练稳定性之间取得折中。([arXiv][15])

需要掌握：

* GAE
* value loss
* entropy bonus
* advantage normalization
* KL early stop
* reward scaling
* observation normalization
* curriculum
* domain randomization

## 2. GRPO

GRPO 不依赖单独 critic，而是在同一个输入条件下采样一组输出，根据组内 reward 计算相对 advantage：

$$
A_i=
\frac{r_i-\operatorname{mean}(r)}
{\operatorname{std}(r)+\epsilon}
$$

GRPO 最初用于语言模型推理训练，通过组内相对奖励替代独立价值模型。([arXiv][16])

在轨迹生成中使用 GRPO 时，需要解决：

* 不同行为 mode 的 reward 不可直接比较。
* reward 尺度与场景相关。
* 稀疏碰撞奖励导致梯度方差大。
* 组内样本需要保持足够多样性。
* 训练可能牺牲生成多模性。

DiffusionDriveV2 使用组内和跨 anchor 的相对策略优化，在保留多模意图时约束低质量轨迹。([arXiv][17])

---

# 二十、VLA 具身智能

## 1. VLA 基础结构

输入：

* 图像或视频
* 语言指令
* 机器人本体状态
* 历史动作

输出：

* 离散 action token
* 连续关节动作
* 末端位姿
* action chunk
* flow/diffusion action trajectory

VLA 的核心挑战：

1. 跨机器人 action space 不统一。
2. 语言语义和低层控制频率差距巨大。
3. 数据质量和覆盖度有限。
4. 视觉延迟影响控制。
5. 长时任务需要高低层规划。
6. 安全性和闭环稳定性难以保证。

## 2. RT-2

RT-2 将机器人动作表示为 token，使视觉语言模型能够共同学习互联网视觉语言知识与机器人控制数据，从而提升语义泛化。([arXiv][18])

需要理解：

* Action tokenization。
* VLM co-fine-tuning。
* Web knowledge transfer。
* 语言推理到动作生成。
* 离散 action 的精度限制。

## 3. OpenVLA

OpenVLA 是开源的 7B VLA，使用大规模真实机器人示范进行训练，并支持参数高效微调。([arXiv][19])

面试中应关注：

* 如何对新机器人做 action normalization。
* 如何进行 LoRA 微调。
* 相机视角变化如何适配。
* action chunk 长度如何选择。
* 推理延迟和控制频率如何平衡。
* 基础模型的语义能力是否真正转化为操作成功率。

## 4. π0 与 π0.5

π0 在预训练 VLM 之上使用 Flow Matching 生成连续机器人动作，目标是处理高维、灵巧和跨机器人的操作任务。([arXiv][20])

π0.5 进一步引入更丰富的异构数据和高层任务推理，以提升开放环境和未见家庭场景中的泛化。([arXiv][21])

专家级判断：

> VLA 当前更适合作为高层语义理解、技能选择和动作先验，不应直接假设它可以替代全部底层控制。对于高频力控、碰撞安全和动态平衡，仍需要 IK、WBC、MPC、力控或安全控制层提供确定性保障。

---

# 二十一、软件工程与系统架构

## 1. 控制模块架构

建议按照以下层次描述：

### Interface 层

* ROS/CAN/UDP
* 车辆状态输入
* 轨迹输入
* 控制指令输出

### Preprocess 层

* 时间同步
* 坐标转换
* 信号滤波
* 异常值处理

### Model 层

* 车辆模型
* 执行器模型
* 参数管理
* 在线辨识

### Controller 层

* LQR
* MPC
* 纵向控制
* 横向控制
* 前馈补偿

### Safety 层

* 输入合法性
* 输出限幅
* 求解器状态
* 控制超时
* fallback
* emergency stop

### Monitoring 层

* 指标上报
* 实时日志
* 版本信息
* 故障码
* 性能统计

## 2. 实时性

需要关注：

* 控制周期
* P50/P95/P99 时延
* 最大时延
* 线程优先级
* CPU affinity
* 锁竞争
* 动态内存分配
* 日志阻塞
* 数据拷贝
* 缓存命中
* 求解器超时

面试回答：

> 实时系统不能只优化平均耗时，需要控制尾延迟。控制主线程内应避免动态内存分配、不可控锁等待和同步日志；耗时模块可以异步化，但必须明确数据版本、时间戳和超时回退。

## 3. 仿真与数据回放

完整链路：

* 单元测试
* 模型测试
* MIL
* SIL
* Log replay
* 闭环仿真
* HIL
* 封闭场测试
* 开放道路灰度

ROS 是机器人应用常用的软件库和工具生态，但真正量产时仍需在通信可靠性、实时性、进程隔离和安全机制上进行额外设计。([ROS][22])

---

# 二十二、技术负责人能力

## 1. 如何做技术规划

可以使用五步法：

### 第一步：定义业务问题

不要直接定义成“升级算法”，而是：

* Cut-in 急刹率过高。
* Oncoming 死锁率高。
* 路测调参周期过长。
* 控制器跨车型迁移成本高。

### 第二步：指标拆解

例如：

$$
\text{急刹率}
=
f(
\text{场景识别},
\text{预测误差},
\text{决策时机},
\text{规划轨迹},
\text{控制跟踪}
)
$$

### 第三步：建立技术假设

例如：

* 急刹主要来自风险确认过晚。
* 多模预测未进入决策。
* 纵向决策忽略横向避让能力。
* 模型训练集中高风险样本不足。

### 第四步：方案分层

* 短期：规则修复和监控。
* 中期：数据与模型优化。
* 长期：生成式决策规划架构。

### 第五步：发布闭环

* 离线指标
* 仿真
* 路测
* 灰度
* 运营监控
* 回滚机制

## 2. 如何管理五人团队

建议回答：

> 我会按照场景闭环而不是算法组件简单分工。每个成员都要对场景数据、算法、评测和问题定位形成一定闭环，同时设置公共架构负责人，避免每个场景独立发展成技术孤岛。

可以划分：

* 场景与问题分析
* Rule-based 规划
* Learning-based 模型
* 数据与评测
* 仿真与工程发布

管理机制：

* 周度指标 review
* 技术方案评审
* Code review
* 场景 case review
* 版本冻结
* 风险列表
* 发布 checklist
* 事故复盘

## 3. 技术冲突处理

专家回答模板：

> 技术争议首先转化为可验证假设，而不是靠资历决定。明确评价指标、数据集和资源预算，对候选方案做小规模对比。涉及安全边界的问题优先采用可解释和可回退方案；涉及长期架构的问题则同时评估维护成本和扩展性。

---

# 二十三、重点项目面试回答模板

## 项目一：Cut-in/Oncoming/Lane Change 场景负责人

### 背景

动态交互场景存在急刹、误减速、死锁和策略抖动问题。

### 难点

* 预测存在多模不确定性。
* 决策与规划耦合。
* 安全与效率冲突。
* 单点规则修改容易影响其他场景。
* 需要同时推进规则和学习方案。

### 方案

1. 建立场景数据闭环。
2. 对问题按识别、预测、决策、规划、控制归因。
3. 接入多模预测。
4. 增加风险意图联动和预决策。
5. 设计采样式交互推演。
6. 推进横纵向联合规划。
7. 使用双 S 空间处理会车。
8. 推进 One-shot/Diffusion/Flow Matching 对比。
9. 统一闭环仿真评测。

### 结果

结合实际简历指标回答，但必须说明统计口径和基线。

### 反思

> 规则优化见效快，但持续增加规则会导致系统复杂度提升。因此我在保留确定性安全边界的同时，逐步将高维交互模式迁移到学习模型中，并用闭环指标决定技术替代节奏。

---

## 项目二：Diffusion/Flow Matching 轨迹模型

### 回答重点

> 项目的目标不是单纯复现生成模型，而是验证生成式轨迹模型在闭环决策中的真实价值。我先构建 TrajEnv，并通过 MPC 生成可控专家数据，建立 One-shot 基线，再逐步加入 Diffusion 和 Flow Matching，比较多模性、轨迹质量、闭环成功率和实时性。

需要准备：

* 状态和 action 定义。
* MPC 专家是否存在偏差。
* 数据分布如何覆盖长尾。
* 条件编码方式。
* 轨迹归一化。
* noise schedule。
* 采样步数。
* ODE solver。
* inference latency。
* 闭环状态漂移。
* 安全过滤。

反思：

> 生成模型输出多样并不等于规划性能更好。真正困难的是如何定义高质量先验、如何对低质量 mode 施加约束，以及如何把闭环 reward 或安全代价反馈到训练中。

---

## 项目三：带挂重卡动力学 MPC

### 回答重点

> 该项目的核心难点是高速、非刚体、多体耦合和长预测时域。首先推导牵引车与挂车的横向动力学和铰接关系，再对模型进行离散化与局部线性化，在 MPC 中加入转角、转角速度、横摆角速度、横向加速度和铰接角约束。

追问准备：

* 八维状态具体是什么？
* 轮胎模型是什么？
* 为什么使用 ACADO？
* 模型如何验证？
* 如何进行参数辨识？
* 100 km/h 下如何保证数值稳定？
* 求解失败怎么办？
* 实车振荡如何定位？

---

## 项目四：乘用车融合 MPC

回答逻辑：

1. 低速动力学模型数值不稳定。
2. 纯运动学模型在高速下缺乏侧偏动态。
3. 设计速度相关 Sigmoid 权重。
4. 保证模型和控制输出连续。
5. 统一优化状态和约束。
6. 在低速、高速、大曲率分别测试。

不要只强调“提升 80%”，应说明：

* 相对于哪个基线。
* 使用什么横向误差指标。
* 在什么速度和曲率分布下。
* 是否牺牲舒适性。
* 是否存在个别场景回退。

---

## 项目五：MPC 求解器多线程优化

专家回答：

> 优化不是简单增加线程，而是分析多个候选轨迹之间哪些计算可以并行，哪些模型矩阵和工作空间可以复用。每个求解实例需要独立内存和状态，避免求解器内部非线程安全数据竞争；同时通过 CPU affinity 和固定候选数量控制尾延迟。

需要准备：

* 4.8 秒预测时域的离散步长。
* 状态和控制维度。
* 约束数量。
* 是否使用 condensing。
* 多线程数量。
* 矩阵分解是否复用。
* 2 ms 是均值还是 P99。
* 冷启动与 warm-start 的差别。

---

## 项目六：力控臂

回答逻辑：

1. 明确硬件接口和传感器。
2. 完成动力学参数或重力参数辨识。
3. 力传感器零偏和重力补偿。
4. 根据底层接口选择导纳或阻抗。
5. 设置质量、阻尼和刚度。
6. 加入力、速度和位置安全限制。
7. 在自由空间、接触和扰动场景测试。
8. 分析稳定性和控制延迟。

---

# 二十四、高频面试问题与建议答案

## 1. 你最核心的技术优势是什么？

> 我的优势是跨越决策、规划和控制建立完整闭环。我既做过 Rule-based 场景决策，也做过 Learning-based 和 Diffusion/Flow Matching 轨迹模型；在控制侧做过乘用车、物流车和带挂重卡 MPC，并深入到底盘执行器和参数辨识。因此在复杂问题中，我能够判断问题究竟来自数据、模型、规划、控制还是硬件，而不是只在某一个模块内调参数。

## 2. 为什么从控制转向决策规划？

> 控制工作让我建立了对车辆可执行性、实时性和硬件边界的深入理解。但自动驾驶系统很多性能问题并不是控制器能够单独解决的，例如交互意图、风险判断和候选行为质量。因此我逐步向上扩展到规划和决策，希望把控制可执行性前置到轨迹生成和行为选择中。

## 3. 规则和学习方法如何融合？

> 我倾向于采用分层融合。规则层负责 ODD、法规、安全硬约束和降级；学习模型负责复杂场景表征、多模意图和候选轨迹生成；最终由安全校验和控制可执行性模块进行仲裁。随着模型验证成熟，可以逐步扩大模型负责范围，但不能一次性移除安全边界。

## 4. Diffusion 相比 One-shot 真正解决什么问题？

> 最核心的是多模分布建模。One-shot 使用单一回归目标时容易产生平均轨迹，Diffusion 可以表示多个合理行为模式。但它带来采样时延、随机性和低质量 mode 等新问题，因此需要 anchor、少步采样、轨迹打分、安全过滤或 RL 后训练。

## 5. 如何判断一个算法可以量产？

> 我会从六个方面判断：性能收益、长尾稳定性、实时确定性、可解释与可诊断性、故障回退能力、工程维护成本。离线指标提升但闭环不稳定，或者平均耗时合格但 P99 超时，都不能认为具备量产条件。

## 6. 如何定位控制误差？

> 先分解为参考轨迹、定位、车辆状态、模型、执行器和控制器六类因素。通过离线重放固定部分输入，比较期望转角、实际转角、车辆响应和模型预测；再通过频域、时延和工况分析定位是增益问题、延迟问题、饱和问题还是模型偏差。

## 7. 为什么需要 MPC，而不是 LQR？

> LQR 在局部线性、无显式约束的问题上非常高效。MPC 的价值是统一处理多变量耦合、未来参考和输入/状态约束。是否使用 MPC 取决于场景复杂度和计算预算，而不是 MPC 一定优于 LQR。

## 8. 你如何处理跨部门协作？

> 首先统一问题定义和指标，避免算法、测试和车辆团队分别使用不同口径。然后建立可复现 case、模块归因和责任边界。对于硬件问题，用控制输入、总线反馈和车辆响应形成证据链；对于感知预测问题，用固定版本回放验证，减少口头争议。

## 9. 项目失败过吗？

建议准备一个真实案例：

> 曾经某次通过提高风险敏感性降低了高风险场景急刹，但闭环测试发现普通场景误减速增加。复盘发现模型训练集高风险样本过采样后，输出概率没有重新校准。后续通过分场景采样、概率标定和闭环效率约束解决。这个问题让我认识到单项安全指标不能脱离效率和分布校准独立优化。

## 10. 作为负责人，你如何避免自己成为瓶颈？

> 我会把技术判断沉淀为设计文档、指标标准、调试工具和评审机制。关键架构由我把关，但具体模块需要明确 owner，并通过接口、测试和 review 保证质量。负责人的价值不是亲自写完所有代码，而是提高团队整体决策质量和交付效率。

---

# 二十五、建议替换后的“专业技能”

1. 具备自动驾驶及机器人算法技术规划、架构设计和跨部门协同能力，拥有从 0 到 1 搭建决策、规划与控制模块并带领团队完成量产交付的经验。能够围绕业务目标建立数据、算法、仿真、测试、发布和运营监控闭环。

2. 深入掌握自动驾驶 Rule-based、Optimization-based 与 Learning-based 决策规划方法。熟悉 A*/Hybrid A*、State Lattice、动态规划、二次规划、SQP、iLQR/DDP 等搜索、采样和数值优化算法；具备 Cut-in、Oncoming、Lane Change 等复杂动态交互场景的建模和量产优化经验。

3. 熟悉端到端决策规划及生成式轨迹建模，掌握 Transformer、One-shot trajectory prediction、Diffusion、Flow Matching、多模轨迹生成、条件引导和生成模型后训练方法；具备数据链路建设、模型训练、开闭环评测和仿真验证经验。

4. 深入掌握最优控制理论及 LQR、MPC、NMPC 等控制器设计方法。具备乘用车、低速物流车、带挂重卡等多车型运动学和动力学建模、参数辨识及控制算法量产经验；熟悉 OSQP、qpOASES、Ipopt、ACADO 等求解器及实时优化方法。

5. 熟悉车辆状态估计和底盘控制，具备转向零偏、载重、执行器响应等在线辨识经验；理解转向、制动、驱动、换挡和车辆总线的硬件特性，能够对控制误差、执行器异常和系统时延进行系统性定位。

6. 熟悉机器人运动学、刚体动力学及 IK、导纳控制、阻抗控制、操作空间控制和 WBC/HQP；具备 VR 动捕遥操、机械臂 IK、七自由度机械臂柔顺力控方案设计与实现经验。

7. 熟悉基于模仿学习和强化学习的机器人全身运动控制，理解 DeepMimic、AMP、ASE、ADD、BeyondMimic 等方法的网络结构、奖励设计、动作先验、技能表示和 Sim2Real 问题；具备相关算法训练与复现经验。

8. 了解 VLA 机器人基础模型，熟悉 RT 系列、OpenVLA、π0/π0.5 等代表性架构，理解视觉语言模型、动作 token、action chunk、Diffusion/Flow action head、跨机器人数据和参数高效微调等关键技术。

9. 掌握深度学习、模仿学习和强化学习基础理论，熟练使用 PyTorch，理解 Transformer、GAIL、DQN、PPO、GRPO 等算法及训练稳定性、数据分布、奖励设计和离线/在线评测问题。

10. 熟悉 Linux、ROS、Docker、Git、C/C++、Python 和 MATLAB，了解 Isaac Sim、Isaac Gym、MuJoCo 等仿真平台。具备实时多线程、模块化架构、单元测试、仿真回放、性能分析和工程发布经验，拥有良好的软件工程和系统安全意识。

---

# 二十六、最终面试表达原则

每一个项目都按照以下顺序回答：

1. **业务问题是什么。**
2. **旧方案为什么不够。**
3. **问题是如何归因的。**
4. **为什么选择这个算法。**
5. **核心模型、目标和约束是什么。**
6. **工程上解决了哪些实时性和稳定性问题。**
7. **如何评测和证明有效。**
8. **如何上线及如何回退。**
9. **团队如何分工。**
10. **项目还有哪些局限和下一步方向。**

真正体现算法技术专家能力的，不是“知道多少算法”，而是：

> 能够识别问题本质，选择合适的技术边界，把算法转化为稳定系统，用可靠指标证明价值，并带领团队持续交付。

这套内容适合作为总知识框架；实际面试前应进一步压缩成“1 分钟自我介绍、6 个重点项目故事、30 个高频追问”三层材料。

[1]: https://ai.stanford.edu/~ddolgov/dolgov08gppSTAIR.html?utm_source=chatgpt.com "Practical Search Techniques in Path Planning for Autonomous Driving"
[2]: https://onlinelibrary.wiley.com/doi/pdf/10.1002/rob.20285?utm_source=chatgpt.com "Differentially constrained mobile robot motion planning in state lattices - Pivtoraiko - 2009 - Journal of Field Robotics - Wiley Online Library"
[3]: https://arxiv.org/abs/2207.06362?utm_source=chatgpt.com "Iterative Linear Quadratic Optimization for Nonlinear Control: Differentiable Programming Algorithmic Templates"
[4]: https://arxiv.org/html/2411.15139v1?utm_source=chatgpt.com "DiffusionDrive: Truncated Diffusion Model for End-to- ..."
[5]: https://arxiv.org/abs/2501.15564?utm_source=chatgpt.com "Diffusion-Based Planning for Autonomous Driving with Flexible Guidance"
[6]: https://arxiv.org/abs/2210.02747?utm_source=chatgpt.com "Flow Matching for Generative Modeling"
[7]: https://osqp.org/docs/?utm_source=chatgpt.com "OSQP solver documentation"
[8]: https://github.com/coin-or/ipopt?utm_source=chatgpt.com "COIN-OR Interior Point Optimizer IPOPT"
[9]: https://arxiv.org/abs/1506.01075?utm_source=chatgpt.com "ControlIt! - A Software Framework for Whole-Body Operational Space Control"
[10]: https://xbpeng.github.io/projects/DeepMimic/index.html?utm_source=chatgpt.com "DeepMimic: Example-Guided Deep Reinforcement ..."
[11]: https://arxiv.org/abs/2104.02180?utm_source=chatgpt.com "AMP: Adversarial Motion Priors for Stylized Physics-Based ..."
[12]: https://arxiv.org/abs/2205.01906?utm_source=chatgpt.com "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters"
[13]: https://arxiv.org/html/2510.13794v1?utm_source=chatgpt.com "MimicKit: A Reinforcement Learning Framework for Motion ..."
[14]: https://arxiv.org/abs/2508.08241?utm_source=chatgpt.com "BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion"
[15]: https://arxiv.org/abs/1707.06347?utm_source=chatgpt.com "Proximal Policy Optimization Algorithms"
[16]: https://arxiv.org/abs/2402.03300?utm_source=chatgpt.com "DeepSeekMath: Pushing the Limits of Mathematical ..."
[17]: https://arxiv.org/abs/2512.07745?utm_source=chatgpt.com "DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving"
[18]: https://arxiv.org/abs/2307.15818?utm_source=chatgpt.com "RT-2: Vision-Language-Action Models Transfer Web ..."
[19]: https://arxiv.org/abs/2406.09246?utm_source=chatgpt.com "OpenVLA: An Open-Source Vision-Language-Action Model"
[20]: https://arxiv.org/abs/2410.24164?utm_source=chatgpt.com "A Vision-Language-Action Flow Model for General Robot ..."
[21]: https://arxiv.org/abs/2504.16054?utm_source=chatgpt.com "[2504.16054] $π_{0.5}$: a Vision-Language-Action Model ..."
[22]: https://www.ros.org/?utm_source=chatgpt.com "ROS: Home"
