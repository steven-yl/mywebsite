---
title: "PyTorch 模型训练技术文档：求解器、参数配置与训练循环"
subtitle: ""
date: 2026-03-12T00:00:00+08:00
draft: false
authors: [Steven]
description: "系统解读 PyTorch 模型训练中的优化器（求解器）、学习率调度器、参数组、训练循环、损失选型与参数配置经验，覆盖完整 API 与可运行示例。"
summary: "从总览到各章节：Optimizer/SGD/Adam/AdamW 全解读、LRScheduler 族、param_groups、梯度累积与裁剪、损失选型及学习率与 batch 配置经验。"

tags: ["PyTorch", "Deep Learning", "优化器", "训练"]
categories: ["PyTorch"]
series: ["PyTorch 实践指南"]
weight: 4
series_weight: 4

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

## 文档索引

| 章节 | 主题 | 内容概要 |
|------|------|----------|
| [一、总览](#一总览整体架构与文档脉络) | 整体架构与文档脉络 | 训练闭环、优化器与调度器位置、知识结构、各部分职责与关联、SGD vs Adam / 调度器选型对比 |
| [二、优化器概念与基类](#二优化器概念与基类) | Optimizer 基类与通用接口 | 是什么、为什么需要、param_groups、state、zero_grad/step/state_dict/load_state_dict/add_param_group 完整说明 |
| [三、SGD 族](#三sgd-族) | SGD / Momentum / Nesterov | 公式、构造函数全部参数、适用场景、与 Adam 对比 |
| [四、自适应学习率优化器](#四自适应学习率优化器) | Adam / AdamW / RAdam / RMSprop 等 | 各算法思想、参数 betas/eps/weight_decay、Adam vs AdamW、适用场景 |
| [五、参数组与差异化配置](#五参数组与差异化配置) | 不同层不同 lr / weight_decay | param_groups 结构、bias 不加 weight_decay、backbone vs head、示例 |
| [六、学习率调度器](#六学习率调度器) | LRScheduler 族 | 基类 step/get_last_lr、StepLR/MultiStepLR/ExponentialLR/CosineAnnealing/OneCycleLR/ReduceLROnPlateau、每 step 与每 epoch 调用 |
| [七、训练循环与梯度](#七训练循环与梯度) | 闭环、累积、裁剪 | 最小闭环、gradient accumulation、gradient clipping、与 DDP 的衔接 |
| [八、损失函数选型](#八损失函数选型) | 常见 Loss 与使用场景 | MSE/CrossEntropy/NLL/BCE/L1/Huber、多任务与自定义 loss |
| [九、参数配置经验](#九参数配置经验) | 学习率、warmup、weight decay | 学习率与 batch 缩放、warmup、常见坑与调参建议 |
| [十、完整示例](#十完整示例) | 可运行脚本 | 从 DataLoader 到 optimizer + scheduler + 训练循环的端到端代码 |
| [十一、速查与小结](#十一速查与小结) | 组件对照与延伸 | 优化器/调度器速查表、与 [distributed_training_guide](./distributed_training_guide.md) / [dataloader_guide](./dataloader_guide.md) 的衔接 |

**阅读建议**：先读总览建立「数据 → 前向 → 损失 → 反向 → 优化器/调度器」的全局图景，再按需跳转优化器、调度器、参数组或配置经验；实现时按「构建 optimizer → 可选 scheduler → 循环内 zero_grad/backward/step/scheduler.step」顺序对照各章。

---

## 一、总览：整体架构与文档脉络

### 1.1 训练闭环在做什么

**目标**：在给定数据上，通过迭代更新模型参数 $\theta$，使损失 $L(\theta)$ 下降，从而拟合数据或任务目标。

**最小闭环**：每个 step 内依次执行：

1. **前向**：输入 → 模型 → 输出  
2. **损失**：输出与目标代入损失函数 → 标量 loss  
3. **反向**：`loss.backward()` → 计算梯度 $\nabla_\theta L$  
4. **更新**：优化器根据梯度更新参数，如 $\theta \leftarrow \theta - \eta \nabla_\theta L$（SGD 为例）  
5. **清空梯度**：`optimizer.zero_grad()`，为下一轮 backward 做准备  

其中「更新」由 **优化器（Optimizer / 求解器）** 完成；学习率 $\eta$ 可固定，也可由 **学习率调度器（LRScheduler）** 按 step 或 epoch 调整。因此：**优化器决定「如何用梯度更新参数」；调度器决定「学习率随时间如何变化」。**

### 1.2 整体架构与数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           训练准备（一次）                                         │
│  model = MyModel()                                                               │
│  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   ← 求解器            │
│  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)     │
│  loss_fn = nn.CrossEntropyLoss()                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           每个 step（或每 N 个 step 做一次 scheduler.step）       │
│  batch = next(dataloader)                                                        │
│  out = model(batch)          →  loss = loss_fn(out, target)  →  loss.backward()  │
│  optimizer.step()            →  optimizer.zero_grad()       →  [scheduler.step()] │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 知识结构（文档脉络）

- **基础层**：优化器是什么、param_groups、state、step/zero_grad（第二节）；损失函数与训练循环的角色（第七、八节）。
- **算法层**：SGD 族（第三节）、Adam/AdamW 等自适应算法（第四节）；参数组实现差异化 lr/weight_decay（第五节）。
- **调度层**：LRScheduler 基类与 step 时机（第六节）；与 optimizer 的绑定关系。
- **工程层**：梯度累积、梯度裁剪（第七节）；参数配置经验（第九节）；完整示例（第十节）。

各节关系：**优化器** 负责单步更新；**调度器** 在每 step 或每 epoch 后改 optimizer 的 lr；**参数组** 让同一 optimizer 内不同参数用不同 lr/weight_decay；**训练循环** 把 DataLoader、loss、optimizer、scheduler 串起来；**配置经验** 指导选 lr、warmup、weight_decay 等。

### 1.4 各部分职责与关联

| 组件 | 职责/主题 | 与其它部分的关系 |
|------|-----------|------------------|
| **Optimizer** | 持有参数引用与更新规则，根据梯度更新参数 | 接收 `model.parameters()`；`step()` 前需先 `backward()` 得到梯度 |
| **LRScheduler** | 按 step/epoch 修改 optimizer 中 param_groups 的 lr | 绑定一个 optimizer；通常在 `optimizer.step()` 之后调用 `scheduler.step()` |
| **param_groups** | 把参数分成多组，每组可有不同 lr、weight_decay 等 | 优化器构造时传入 list of dict；scheduler 通过改 `param_groups[i]['lr']` 生效 |
| **Loss** | 把模型输出与目标变成标量，供 backward | 输出必须是标量；选择与任务匹配的 loss（分类 CE、回归 MSE 等） |
| **训练循环** | 串联数据、前向、loss、反向、optimizer.step、zero_grad、可选 scheduler.step | 可与 [DataLoader](./dataloader_guide.md)、[DDP](./distributed_training_guide.md) 组合 |

### 1.5 优缺点与适用场景对比

#### 1.5.1 SGD vs Adam / AdamW

| 维度 | SGD (含 Momentum/Nesterov) | Adam / AdamW |
|------|---------------------------|--------------|
| **更新方式** | 一阶梯度 ± 动量，学习率需手调 | 自适应步长（一阶矩+二阶矩），对 lr 不敏感 |
| **泛化** | 大 batch 时常见更好泛化，调好 lr 后稳定 | 小 lr 下常用，大 lr 易欠拟合或不稳定 |
| **显存** | 仅动量缓存，显存占用小 | 需存一阶、二阶矩，显存略大 |
| **适用** | 大 batch 训练、追求极致精度、CV 预训练 | 快速实验、NLP/多任务、默认首选 |

**AdamW** 相对 Adam：weight decay 不进入动量与方差，等价于解耦的 L2 正则，通常比 Adam 更稳、泛化更好，推荐作默认。

#### 1.5.2 学习率调度器选型

| 调度器 | 特点 | 适用 |
|--------|------|------|
| **StepLR / MultiStepLR** | 固定 epoch 降 lr | 简单分段衰减 |
| **CosineAnnealingLR** | 余弦平滑降到 0 或 min_lr | 长训练、平滑收敛 |
| **OneCycleLR** | 先升后降、单周期 | 短 epoch、快速收敛 |
| **ReduceLROnPlateau** | 按验证指标触发的衰减 | 验证集驱动、早停配合 |
| **LambdaLR** | 自定义函数乘 lr | 完全自定义策略 |

---

## 二、优化器概念与基类

### 2.1 优化器是什么、为什么需要

**是什么**：`torch.optim.Optimizer` 是持有「待优化参数」和「更新规则」的对象；每次调用 `step()` 时，根据当前梯度按该规则更新参数。

**为什么需要**：梯度只给出下降方向，步长（学习率）、动量、自适应缩放等若手写易错且难以复用；抽象成 Optimizer 后，算法统一、接口一致，且便于实现参数组、状态保存与恢复。

**解决什么问题**：把「梯度 → 参数更新」标准化，支持 SGD、Adam、AdamW 等多种算法，并支持 per-parameter 或 per-group 的 lr、weight_decay 等。

### 2.2 参数组（param_groups）

**是什么**：优化器内部把参数分成若干 **参数组**；每个组是一个 dict，至少包含 `'params'`（该组参数列表），以及可选的 `'lr'`、`'weight_decay'` 等，与构造时传入的 defaults 合并。

**为什么需要**：不同层或不同模块往往需要不同学习率（如 backbone 小 lr、head 大 lr），或 bias 不加 weight_decay；参数组允许在同一 optimizer 里为不同子集设置不同超参。

**结构**：`optimizer.param_groups` 是 list of dict，例如：

```python
optimizer = torch.optim.SGD([
    {'params': model.base.parameters(), 'lr': 1e-2},
    {'params': model.head.parameters(), 'lr': 1e-1}
], momentum=0.9)
# param_groups[0]['params'] 为 base 参数，lr=1e-2
# param_groups[1]['params'] 为 head 参数，lr=1e-1
```

### 2.3 状态（state）

**是什么**：每个被优化参数在 optimizer 内有一个 id，对应 `state[id]` 的 dict，用于存该算法的内部状态（如 SGD 的 momentum buffer、Adam 的 exp_avg/exp_avg_sq）。

**为什么需要**：动量类、自适应类算法需要跨 step 记忆历史信息；state 在 `step()` 时被读入并写回。

**用户通常不直接改 state**；保存/恢复训练时用 `optimizer.state_dict()` 与 `optimizer.load_state_dict()` 即可。

### 2.4 基类方法完整说明

| 方法 | 作用 | 调用时机 |
|------|------|----------|
| **zero_grad(set_to_none=False)** | 把当前所有待优化参数的梯度置 0（或 None）；`set_to_none=True` 时置为 None，可省少量内存 | 每个 step 开始前，避免梯度累积时需在 step 前调用 |
| **step(closure=None)** | 根据当前梯度执行一次参数更新；若传 closure（无参、返回 loss），优化器可多次调用以重算 loss（如 LBFGS） | 在 `loss.backward()` 之后；若优化器需多次前向（如 LBFGS），则传入 closure |
| **state_dict()** | 返回 state + param_groups，用于 checkpoint | 保存 checkpoint 时 |
| **load_state_dict(state_dict)** | 从 state_dict 恢复 state 与 param_groups | 加载 checkpoint 后恢复训练 |
| **add_param_group(param_group)** | 增加一组参数（dict 需含 `'params'`，其余键同构造时的 defaults） | 动态扩展优化参数时（如微调加新层） |
| **register_step_pre_hook / register_step_post_hook** | 在 step 前/后执行自定义逻辑 | 调试或统计用 |
| **register_state_dict_pre_hook / register_load_state_dict_pre_hook** | 在 state_dict/load_state_dict 前后执行 | 序列化/反序列化定制 |

**注意**：`zero_grad()` 与 `step()` 的先后关系。常见写法是「先 zero_grad，再 forward/backward，再 step」；若做梯度累积，则多个 backward 后再一次 step，且每步开始 zero_grad。

### 2.5 示例：基类用法

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 模拟一步
x = torch.randn(4, 10)
y = model(x).sum()
optimizer.zero_grad()
y.backward()
optimizer.step()

# 查看 param_groups
assert len(optimizer.param_groups) == 1
assert optimizer.param_groups[0]['lr'] == 0.01

# 保存/恢复（仅示例，实际会连同 model）
state = optimizer.state_dict()
optimizer.load_state_dict(state)
```

---

## 三、SGD 族

### 3.1 结构说明

SGD 族包括：**SGD**（可选 momentum）、**Nesterov**。本节只讨论 `torch.optim.SGD` 的完整参数与含义；Momentum 与 Nesterov 均通过同一类实现。

### 3.2 关键概念与公式

- **SGD**：
  $$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$
  即用当前梯度乘以学习率更新。
- **SGD with Momentum**：先累积动量 $v_{t+1} = \mu v_t + \nabla L(\theta_t)$，再更新 $\theta_{t+1} = \theta_t - \eta v_{t+1}$，减轻震荡、加速收敛。
- **Nesterov**：在计算梯度前先做「临时更新」再算梯度，使动量更前瞻一步，公式上等价于对动量项做修正（PyTorch 中 `nesterov=True` 即启用）。

**为什么需要 Momentum**：纯 SGD 在病态曲面上震荡大、收敛慢；动量起到平滑梯度、利用历史方向的作用。

### 3.3 SGD 构造函数与参数

```python
torch.optim.SGD(params, lr=<required>, momentum=0, dampening=0,
                weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False)
```

| 参数 | 含义 | 常用值 |
|------|------|--------|
| **params** | 待优化参数（或 param_groups 的 list） | `model.parameters()` 或 list of dict |
| **lr** | 学习率 | 0.01～0.1 常见，大 batch 可线性缩放 |
| **momentum** | 动量系数 | 0.9 常用 |
| **dampening** | 对动量的阻尼，与 momentum 同用 | 一般 0 |
| **weight_decay** | L2 惩罚系数 | 1e-4～1e-2 |
| **nesterov** | 是否 Nesterov 动量 | True 可略加速收敛 |
| **maximize** | True 时做梯度上升 | 默认 False |

### 3.4 使用方式与示例

```python
# 纯 SGD
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# SGD + Momentum
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD + Nesterov
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# 带 weight_decay（L2 正则）
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### 3.5 适用场景

- 大 batch、长训练、追求泛化时常用 SGD + Momentum（或 Nesterov）。
- 小 batch、快速实验更常用 Adam/AdamW；若从 Adam 换到 SGD，需要重新调 lr（通常 SGD 的 lr 比 Adam 大一个数量级左右）。

---

## 四、自适应学习率优化器

### 4.1 结构说明

本节覆盖：**Adam**、**AdamW**、**RAdam**、**NAdam**、**RMSprop**、**Adagrad**、**Adadelta**。它们共同点是按「每个参数」维护标量或向量状态，用梯度的一阶/二阶信息自适应地缩放步长。

### 4.2 Adam

**是什么**：对每个参数维护一阶矩估计 $m_t$ 和二阶矩估计 $v_t$，做偏差修正后用 $\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$ 方向更新，步长为 lr。

**为什么需要**：不同参数梯度量纲和曲率差异大；固定 lr 要么对某些参数过大要么过小；自适应步长能减少对 lr 的敏感度，收敛快、调参简单。

**公式（简要）**：  
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$  
偏差修正：$\hat{m}_t = m_t/(1-\beta_1^t)$，$\hat{v}_t = v_t/(1-\beta_2^t)$。  
更新：$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$。

**构造函数**：

```python
torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, *, amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)
```

| 参数 | 含义 | 常用值 |
|------|------|--------|
| **lr** | 学习率 | 1e-3 最常用 |
| **betas** | 一阶、二阶矩的指数衰减率 | (0.9, 0.999) |
| **eps** | 数值稳定项，分母加在 $\sqrt{v}$ 上 | 1e-8 |
| **weight_decay** | L2 惩罚（在 Adam 中会进动量） | 0 或 1e-2 |
| **amsgrad** | 是否用 AMSGrad 变体（取 $v$ 的单调递增） | 一般 False |

### 4.3 AdamW

**是什么**：Adam 的变体，把 **weight decay** 从「加在梯度里」改为「直接对参数做衰减」，即每步先做 $\theta \leftarrow \theta - \eta\lambda\theta$ 再按 Adam 更新。这样 weight decay 不参与一阶/二阶矩计算，等价于解耦的 L2 正则。

**为什么需要**：原 Adam 里 weight decay 与梯度耦合，正则效果和收敛行为不如解耦形式；AdamW 在多数任务上更稳、泛化更好，**推荐作为默认选择**。

**构造函数**：与 Adam 相同，仅实现不同；`weight_decay` 默认 0，常用 0.01。

```python
torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, ...)
```

#### 4.3.1 Adam 与 AdamW 对比

- **Adam**：weight_decay 在实现上等价于 L2 正则（对梯度加 weight_decay·θ，再参与动量与二阶矩），与自适应步长耦合后，对泛化有时不利。
- **AdamW**：将 weight decay 从梯度更新中解耦，单独做 $\theta \leftarrow \theta - \eta\lambda\theta$，再按 Adam 公式更新；weight decay 不进入 $m_t/v_t$，更符合「参数衰减」语义，通常泛化更好，**推荐默认使用 AdamW**。

### 4.4 RAdam / NAdam

- **RAdam**：在训练早期二阶矩方差大时，用修正项避免步长过大，减少 warmup 需求。
- **NAdam**：把 Nesterov 思想融进 Adam，用「当前步的动量预测」更新，有时收敛略快。

两者接口与 Adam 类似，可作为替代尝试；默认仍推荐 AdamW。

### 4.5 RMSprop / Adagrad / Adadelta

- **RMSprop**：只维护二阶矩的指数平均，用 $\eta/\sqrt{v_t+\epsilon}$ 缩放梯度；适合 RNN、非平稳目标。
- **Adagrad**：累积平方梯度，步长单调降；适合稀疏梯度，但易过早变小。
- **Adadelta**：在 Adagrad 基础上用滑动平均替代累积，避免步长趋 0。

日常训练中 Adam/AdamW 使用更多；RMSprop 在部分 RL 或旧代码中仍见。

### 4.6 PyTorch 内置优化器一览（无遗漏）

以下为 `torch.optim` 中常见优化器及典型用途，便于按场景选择或查漏。

| 优化器 | 类名 | 典型用途 / 说明 |
|--------|------|------------------|
| SGD | `optim.SGD` | 通用；大 batch、需强泛化时常用 momentum + Nesterov |
| ASGD | `optim.ASGD` | 对强凸或近凸目标，对迭代做平均，稳定性好 |
| Adam | `optim.Adam` | 默认首选之一，收敛快、超参少 |
| AdamW | `optim.AdamW` | 带解耦 weight decay，NLP/Vision 常用，推荐默认 |
| Adamax | `optim.Adamax` | 基于无穷范数的 Adam 变体，个别任务更稳 |
| NAdam | `optim.NAdam` | Adam + Nesterov 动量，无需 warmup 时可用 |
| RAdam | `optim.RAdam` | 修正早期方差，冷启动更稳 |
| SparseAdam | `optim.SparseAdam` | 稀疏梯度（如 embedding）时省内存、省算力 |
| Rprop | `optim.Rprop` | 仅全批量、单机，一般不用于大数据 |
| RMSprop | `optim.RMSprop` | 按标量做自适应步长，可用于 RNN/强化学习 |
| Adadelta | `optim.Adadelta` | 无学习率，依赖梯度与参数更新量的移动平均 |
| Adagrad | `optim.Adagrad` | 稀疏特征/小学习率场景；易使有效学习率过小 |
| LBFGS | `optim.LBFGS` | 小批量二阶、需 closure、多次前向，适合小模型/全批量 |

### 4.7 使用示例

```python
# 默认推荐
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 仅 Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
```

---

## 五、参数组与差异化配置

### 5.1 结构说明

参数组即构造 optimizer 时传入的「list of dict」，每个 dict 一组参数 + 可选覆盖的 lr、weight_decay 等；优化器内部对所有组统一执行同一种算法，但每组用各自的超参。

### 5.2 常见用法

**1）Backbone 小 lr、Head 大 lr（微调/迁移学习）**

```python
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
], weight_decay=0.01)
```

**2）Bias 不加 weight_decay**

```python
bias_params = [p for n, p in model.named_parameters() if 'bias' in n]
other_params = [p for n, p in model.named_parameters() if 'bias' not in n]
optimizer = torch.optim.AdamW([
    {'params': other_params, 'weight_decay': 0.01},
    {'params': bias_params, 'weight_decay': 0}
], lr=1e-3)
```

**3）BatchNorm 等不 weight_decay（按需）**

```python
def get_param_groups(model, base_lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'bias' in n or 'norm' in n or 'bn' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'lr': base_lr, 'weight_decay': wd},
        {'params': no_decay, 'lr': base_lr, 'weight_decay': 0}
    ]
optimizer = torch.optim.AdamW(get_param_groups(model, 1e-3, 0.01))
```

### 5.3 add_param_group

微调中动态加入新模块时，可把新参数加进已有 optimizer：

```python
optimizer.add_param_group({'params': model.new_module.parameters(), 'lr': 1e-4})
```

注意：新组的其他键（如 weight_decay）若未写，会使用 optimizer 的 defaults。

---

## 六、学习率调度器

### 6.1 概念与基类

**是什么**：`torch.optim.lr_scheduler.LRScheduler` 绑定一个 optimizer，在每次 `scheduler.step()` 时根据策略更新 `optimizer.param_groups[*]['lr']`，从而在训练过程中改变学习率。

**为什么需要**：固定 lr 前期可能震荡、后期可能过大会影响收敛；按 step 或 epoch 衰减或先升后降，往往收敛更稳、最终效果更好。

**调用顺序**：必须先 `optimizer.step()`，再 `scheduler.step()`（若按 step 调）；若按 epoch 调，则每个 epoch 结束调一次 `scheduler.step()`。

### 6.2 基类方法

| 方法 | 作用 |
|------|------|
| **step(epoch=None)** | 前进一步；若按 epoch，可传 `epoch`（部分调度器已弃用该参数，建议不传） |
| **get_last_lr()** | 返回上一步之后各 param_group 的 lr 列表，用于打日志 |
| **state_dict() / load_state_dict()** | 保存/恢复调度器（含 last_epoch 等），断点续训时与 optimizer 一起保存 |

### 6.3 常用调度器一览

| 调度器 | 公式/行为 | 典型参数 |
|--------|-----------|----------|
| **StepLR** | 每 step_size 个 epoch 乘 gamma | step_size, gamma |
| **MultiStepLR** | 在指定 milestones（epoch 列表）乘 gamma | milestones, gamma |
| **ExponentialLR** | 每步 lr *= gamma | gamma |
| **CosineAnnealingLR** | 余弦从 lr 降到 0 或 eta_min | T_max, eta_min |
| **CosineAnnealingWarmRestarts** | 余弦周期并周期性地重启 | T_0, T_mult, eta_min |
| **OneCycleLR** | 先线性 warmup 再余弦/线性 衰减到 min_lr | max_lr, total_steps, pct_start 等 |
| **ReduceLROnPlateau** | 当指标不改善时 lr *= factor | mode, factor, patience |
| **LambdaLR** | lr = base_lr * lr_lambda(epoch) | lr_lambda |
| **SequentialLR / ChainedScheduler** | 多个 scheduler 按顺序或链式组合 | 见文档 |

### 6.4 按 step 与按 epoch

- **按 step**：每个 training step 后调用 `scheduler.step()`，如 OneCycleLR、部分 Cosine 实现。
- **按 epoch**：每个 epoch 结束调用一次 `scheduler.step()`，如 StepLR、MultiStepLR、ReduceLROnPlateau（plateau 的 step 传 validation 指标）。

示例（按 epoch）：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    train_one_epoch(...)
    scheduler.step()
```

示例（按 step，OneCycleLR）：

```python
total = len(dataloader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-2, total_steps=total, pct_start=0.1
)
for batch in dataloader:
    ...
    optimizer.step()
    scheduler.step()
```

### 6.5 学习率调度器完整列表（无遗漏）

以下为 `torch.optim.lr_scheduler` 中常见调度器及要点，便于速查与补全。

| 调度器 | 类名 | 主要参数 | 说明 |
|--------|------|----------|------|
| StepLR | `StepLR` | step_size, gamma | 每 step_size 个 epoch 乘 gamma |
| MultiStepLR | `MultiStepLR` | milestones, gamma | 在指定 epoch 乘 gamma |
| ExponentialLR | `ExponentialLR` | gamma | 每步/每 epoch 乘 gamma |
| CosineAnnealingLR | `CosineAnnealingLR` | T_max, eta_min | 余弦退火到 eta_min |
| CosineAnnealingWarmRestarts | `CosineAnnealingWarmRestarts` | T_0, T_mult, eta_min | 带周期重启的余弦 |
| OneCycleLR | `OneCycleLR` | max_lr, total_steps, ... | 单周期先升后降，可配 div_factor、pct_start |
| ReduceLROnPlateau | `ReduceLROnPlateau` | mode, factor, patience | 按验证指标 plateau 时乘 factor |
| LambdaLR | `LambdaLR` | lr_lambda | 自定义 lr = base_lr * lr_lambda(epoch) |
| SequentialLR | `SequentialLR` | schedulers, milestones | 按 milestone 顺序切换多个调度器 |
| ChainedScheduler | `ChainedScheduler` | schedulers | 链式组合（每步依次 step） |
| LinearLR / ConstantLR / PolynomialLR | `LinearLR` 等 | 见文档 | 线性/常数/多项式变化 |
| CyclicLR | `CyclicLR` | base_lr, max_lr, ... | 三角或指数周期变化 |

基类方法：`step()`（部分调度器需传入 `metrics`）、`get_last_lr()`、`state_dict()`、`load_state_dict()`；子类实现 `get_lr()` 返回当前各 param_group 的学习率列表。

---

## 七、训练循环与梯度

### 7.1 最小闭环

```python
model.train()
for batch in dataloader:
    inputs, targets = batch
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    # 若按 step 调 lr：scheduler.step()
```

### 7.2 梯度累积

当有效 batch = N * 单次 batch 时，每 N 步才 `optimizer.step()` 一次，每步只 `backward()` 不 `step()`，梯度累加；最后一步前 `zero_grad()`，然后 `backward()` 再 `step()`。

```python
accum_steps = 4
for i, batch in enumerate(dataloader):
    inputs, targets = batch
    if i % accum_steps == 0:
        optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
```

### 7.3 梯度裁剪

防止梯度爆炸，在 `backward()` 之后、`step()` 之前对梯度做范数裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 或按值裁剪
# torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
optimizer.step()
```

常用于 RNN、Transformer、大 lr 或长序列训练。

### 7.4 与 DDP 的衔接

在 分布式训练 中，每 rank 独立维护自己的 optimizer；DDP 在 backward 时已做梯度 AllReduce，因此各 rank 上梯度一致，各自 `optimizer.step()` 后参数保持同步。无需对 optimizer 做额外集体通信。

---

## 八、损失函数选型

### 8.1 常见 Loss 与适用任务

| Loss | 任务 | 输出与目标形状 | 注意 |
|------|------|----------------|------|
| **CrossEntropyLoss** | 多分类 | logits (N,C)，target (N,) long | 含 softmax，不要对 logits 再 softmax |
| **NLLLoss** | 多分类 | log_probs (N,C)，target (N,) | 需先 log_softmax |
| **BCELoss / BCEWithLogitsLoss** | 二分类 | 概率或 logits，(N,) 或 (N,1) | BCEWithLogits 更数值稳定 |
| **MSELoss** | 回归 | (N,*)，同形状 | 对异常值敏感 |
| **L1Loss** | 回归 | 同形状 | 更鲁棒 |
| **HuberLoss** | 回归 | 同形状 | 小误差 MSE、大误差 L1，折中 |

### 8.2 使用方式

```python
# 分类
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)

# 回归
loss_fn = nn.MSELoss()
loss = loss_fn(pred, target)

# 多任务：加权和
loss = ce_loss(logits_cls, y_cls) + 0.1 * mse_loss(pred_reg, y_reg)
loss.backward()
```

---

## 九、参数配置经验

### 9.1 学习率与 batch size

- **线性缩放**：batch 扩大 k 倍时，lr 常线性扩大 k 倍（如 32→256，lr 1e-3→8e-3）；再大 batch 可用 sqrt 缩放或 warmup 更长。
- **SGD**：常用 0.01～0.1；大 batch 可试 0.1 * batch/32。
- **Adam/AdamW**：1e-3 通用；2e-3～3e-3 也常见；再大易不稳定。

### 9.2 Warmup

前若干 step 或 epoch 从 0 或小 lr 线性升到目标 lr，减轻初期大梯度导致的震荡。Transformer、大 batch 训练常用。

```python
# 可用 LambdaLR 或 OneCycleLR 的 pct_start 实现
def warmup_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
```

### 9.3 Weight decay

- AdamW：0.01 常用；0.1 也有用。
- SGD：1e-4～1e-2；bias / BN 通常设为 0。

### 9.4 常见坑

- 忘记 `zero_grad()`：梯度会累积，等价于放大 lr。
- `scheduler.step()` 在 `optimizer.step()` 之前其实并不会出错，但大多数调度器（如 StepLR、CosineAnnealingLR）推荐在 `optimizer.step()` 之后调用，这样每 step 用的是当前 step 的 lr，和论文/官方实现一致。部分调度器（如 ReduceLROnPlateau）需根据文档操作。
- 恢复 checkpoint 时先恢复 scheduler 再恢复 optimizer，否则 scheduler 会覆盖加载进来的 lr。
- ReduceLROnPlateau 的 `step(metric)` 要传验证指标，且注意 mode（'min'/'max'）。

---

## 十、完整示例

下面是一段可独立运行的训练脚本（CPU 即可），串联 DataLoader、模型、损失、优化器、调度器与梯度裁剪。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 10
    lr = 1e-3
    weight_decay = 0.01
    max_grad_norm = 1.0

    # 数据
    X = torch.randn(1000, 20)
    y = (X.sum(dim=1, keepdim=True) > 0).long().squeeze(1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 模型、损失、优化器、调度器
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}  lr={current_lr:.2e}")

if __name__ == "__main__":
    main()
```

将上述代码保存为 `train_example.py`，在项目根目录执行 `python train_example.py` 即可运行（无 GPU 时自动用 CPU）。

---

## 十一、速查与小结

### 11.1 优化器速查

| 算法 | 构造示例 | 典型 lr |
|------|-----------|---------|
| SGD | `optim.SGD(params, lr=0.01, momentum=0.9)` | 0.01～0.1 |
| Adam | `optim.Adam(params, lr=1e-3)` | 1e-3 |
| AdamW | `optim.AdamW(params, lr=1e-3, weight_decay=0.01)` | 1e-3 |

### 11.2 调度器速查

| 调度器 | 构造示例 |
|--------|-----------|
| StepLR | `lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)` |
| CosineAnnealingLR | `lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)` |
| OneCycleLR | `lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=total)` |
| ReduceLROnPlateau | `lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)` |

### 11.3 与其它文档的衔接

- **数据流**：数据来自 [Dataset](./pytorch_dataset_guide.md) → [DataLoader](./dataloader_guide.md) → 本训练的 `for batch in dataloader`。
- **多卡**：本训练的 optimizer/scheduler 每 rank 一份，与 [DDP](./distributed_training_guide.md) 无冲突；checkpoint 保存/加载时需同时保存 optimizer、scheduler 的 state_dict。

以上为模型训练中求解器、参数配置与训练循环的完整技术解读；按文档索引可快速定位到优化器、调度器、参数组或配置经验等小节。
