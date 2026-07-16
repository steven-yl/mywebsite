---
title: "第六章：学习基础设施（mimickit/learning/）"
subtitle: ""
date: 2026-07-14T12:00:00+08:00
draft: false
authors: [Steven]
description: "第六章：学习基础设施（mimickit/learning/）。"
summary: "第六章：学习基础设施（mimickit/learning/）。"
tags: [mimickit, robots]
categories: [docs Mimickit, robots]
series: [mimickit-docs]
weight: 6
series_weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第六章：学习基础设施（`mimickit/learning/`）

## 6.1 模块结构

```
learning/
├── base_agent.py              # 训练循环抽象
├── base_model.py              # 模型基类、动作分布构建
├── ppo_agent.py / ppo_model.py
├── awr_agent.py / awr_model.py
├── amp_agent.py / amp_model.py
├── ase_agent.py / ase_model.py
├── add_agent.py / add_model.py
├── lcp_agent.py / lcp_model.py
├── smp_agent.py / smp_model.py
├── dummy_agent.py
├── agent_builder.py
├── experience_buffer.py
├── mp_optimizer.py
├── normalizer.py
├── diff_normalizer.py
├── distribution_gaussian_diag.py
├── distribution_categorical.py
├── return_tracker.py
├── rl_util.py
└── nets/
    ├── net_builder.py
    ├── fc_2layers_{128,256,512,1024}units.py
    ├── fc_3layers_1024units.py
    └── cnn_3conv_1fc_0.py
```

---

## 6.2 `BaseAgent` — 训练循环

### 6.2.1 为什么需要 BaseAgent

统一所有算法的**采样→构建数据→更新→日志**流程，子类仅重写模型构建、动作决策、损失计算。

### 6.2.2 枚举 `AgentMode`

```python
class AgentMode(enum.Enum):
    TRAIN = 0
    TEST = 1
```

### 6.2.3 初始化流程 `__init__`

1. `_load_params(config)` — 加载超参数
2. `_build_normalizers()` — 观测/动作归一化器
3. `_build_model(config)` — 抽象，子类实现
4. `_build_optimizer(config)` — 抽象
5. `_build_exp_buffer(config)` — 经验缓冲区
6. `_build_return_tracker()` — 回合回报追踪

### 6.2.4 公共方法

| 方法 | 说明 |
|------|------|
| `train_model(max_samples, out_dir, ...)` | 主训练循环 |
| `test_model(num_episodes)` | 评估模式 rollout |
| `get_action_size()` | 动作维度 |
| `set_mode(mode)` | 切换训练/测试 |
| `get_num_envs()` | 并行环境数 |
| `save(out_file)` / `load(in_file)` | 模型存取 |
| `calc_num_params()` | 可训练参数总数 |

### 6.2.5 训练迭代 `_train_iter`

```
_init_iter()
eval() + TRAIN mode
_rollout_train(steps_per_iter)     # 收集经验
build_train_data()                # 子类可扩展
update_model()                    # 子类实现
update_normalizers()              # 若 sample_count < normalizer_samples
返回 train_info 字典
```

### 6.2.6 Rollout 方法

| 方法 | 说明 |
|------|------|
| `_rollout_train(num_steps)` | 训练采样循环 |
| `_rollout_test(num_episodes)` | 测试采样至足够 episode |
| `_decide_action(obs, info)` | 抽象：策略 → 动作 |
| `_step_env(action)` | 调用 `env.step` |
| `_record_data_pre_step(...)` | 记录 obs, action |
| `_record_data_post_step(...)` | 记录 next_obs, reward, done |
| `_reset_done_envs(done)` | 终止环境重置 |
| `_reset_envs(env_ids)` | 环境重置 |

### 6.2.7 价值 Bootstrap

```python
def _compute_succ_val(self):
    return r_succ / (1.0 - discount)

def _compute_fail_val(self):
    return r_fail / (1.0 - discount)
```

当 episode 以 `SUCC`/`FAIL` 终止时，用终止奖励 bootstrap TD 目标。

### 6.2.8 动作边界损失 `_compute_action_bound_loss`

对 Box 动作空间，惩罚策略均值超出$[-1, 1]$（归一化后）：

$$
L_{bound} = \sum_i \max(0, \mu_i - 1)^2 + \max(0, -1 - \mu_i)^2
$$

### 6.2.9 抽象方法（子类必须实现）

- `_build_model(config)`
- `_build_optimizer(config)`
- `_get_exp_buffer_length()`
- `_sync_optimizer()`
- `_decide_action(obs, info)`
- `_update_model()`

---

## 6.3 `ExperienceBuffer`

### 6.3.1 数据结构

形状 `[buffer_length, num_envs, ...]`，按时间步环形写入。

### 6.3.2 方法清单

| 方法 | 说明 |
|------|------|
| `add_buffer(name, data_shape, dtype)` | 注册新数据字段 |
| `reset()` | 重置写入头 |
| `clear()` | 清空含样本计数 |
| `inc()` | 时间步前进 |
| `get_total_samples()` | 累计样本数 |
| `get_capacity()` | `buffer_length × num_envs` |
| `get_sample_count()` | 当前有效样本数 |
| `is_full()` | 是否填满 |
| `record(name, data)` | 记录当前步数据 |
| `get_data(name)` | 获取 `[T, B, ...]` 张量 |
| `get_data_flat(name)` | 获取 `[T×B, ...]` 展平张量 |
| `set_data(name, data)` | 整体设置 |
| `set_data_flat(name, data)` | 展平设置 |
| `sample(n)` | 随机采样 n 条 |
| `push(data_dict)` | 推入数据（用于判别器 replay） |

---

## 6.3 `Normalizer`

### 6.3.1 在线均值/方差归一化

维护 running mean/std，支持多进程合并（`mp_util.reduce_inplace_sum`）。

**归一化：**

$$
\hat{x} = \text{clip}\left(\frac{x - \mu}{\sigma}, -c, c\right)
$$

### 6.3.2 方法

| 方法 | 说明 |
|------|------|
| `record(x)` | 累积批次统计 |
| `update()` | 合并新统计到 running 值 |
| `normalize(x)` | 归一化 |
| `unnormalize(norm_x)` | 反归一化 |
| `get_mean()` / `get_std()` | 获取统计量 |
| `set_mean_std(mean, std)` | 手动设置（加载模型时用） |

动作归一化器初始化：Box 空间用$(high + low)/2$和$(high - low)/2$作为初始 mean/std。

---

## 6.4 `DiffNormalizer`

用于 **差分向量** 归一化（ADD、SMP SDS），接口与 `Normalizer` 相同，但统计的是差分而非绝对值。

---

## 6.5 `MPOptimizer`

封装 SGD/AdamW，支持：

- 梯度裁剪
- 多进程梯度平均
- 混合精度（`torch.amp.autocast`）

| 方法 | 说明 |
|------|------|
| `step(loss)` | 反向传播 + 优化 |
| `sync()` | 加载模型后同步优化器状态 |

---

## 6.6 `ReturnTracker`

追踪每个环境的回合回报与长度：

| 方法 | 说明 |
|------|------|
| `update(r, done)` | 累积奖励，done 时记录 episode |
| `get_mean_return()` | 平均回报 |
| `get_mean_ep_len()` | 平均 episode 长度 |
| `get_episodes()` | episode 总数 |
| `reset()` | 重置统计 |

---

## 6.7 `rl_util.compute_td_lambda_return`

### 6.7.1 TD(λ) 回报推导

定义 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

TD(λ) 回报（后向递归）：

$$
R_t^{(\lambda)} = r_t + \gamma \left[(1-\lambda) V(s_{t+1}) + \lambda R_{t+1}^{(\lambda)}\right]
$$

当$\lambda = 0$：退化为 1-step TD；$\lambda = 1$：退化为 Monte Carlo。

### 6.7.2 实现

```python
# rl_util.py — 后向遍历
for i in reversed(range(0, timesteps - 1)):
    curr_lambda = td_lambda * (1.0 - reset_mask[i])
    curr_val = r[i] + discount * ((1.0 - curr_lambda) * next_v[i] + curr_lambda * return_t[i+1])
    return_t[i] = curr_val
```

`reset_mask` 在 episode 终止时清零 λ 传播。

---

## 6.8 动作分布

### 6.8.1 `DistributionGaussianDiag`

对角高斯策略：

$$
\pi(a|s) = \mathcal{N}(\mu_\theta(s), \text{diag}(\sigma^2))
$$

| 方法 | 说明 |
|------|------|
| `sample()` |$a = \mu + \sigma \cdot \epsilon$，$\epsilon \sim \mathcal{N}(0,I)$|
| `mode()` | 返回均值$\mu$|
| `log_prob(a)` | 对数概率密度 |
| `entropy()` | 微分熵 |
| `param_reg()` | 参数正则化 |

**StdType：**

| 类型 | 说明 |
|------|------|
| `FIXED` | 固定标准差，不可学习 |
| `CONSTANT` | 全局可学习 log_std |
| `VARIABLE` | 状态依赖 log_std 网络 |

### 6.8.2 `DistributionCategorical`

离散动作空间用分类分布。

---

## 6.9 网络构建器 `net_builder`

| 网络名 | 结构 |
|--------|------|
| `fc_2layers_128units` | Linear→ReLU→128→ReLU→128 |
| `fc_2layers_256units` | 256→256 |
| `fc_2layers_512units` | 512→512 |
| `fc_2layers_1024units` | 1024→512 |
| `fc_3layers_1024units` | 1024→1024→512 |
| `cnn_3conv_1fc_0` | 3 层卷积 + 512 FC（图像观测） |

---

## 6.10 `BaseModel` / `PPOModel`

### 6.10.1 `BaseModel`

| 方法 | 说明 |
|------|------|
| `_build_action_distribution(config, env, input)` | 构建高斯/分类分布头 |

### 6.10.2 `PPOModel`

| 组件 | 说明 |
|------|------|
| `actor_net` | 输入 obs → 动作分布参数 |
| `critic_net` | 输入 obs → 标量价值 |
| `eval_actor(obs)` | 返回 `DistributionGaussianDiag` |
| `eval_critic(obs)` | 返回 value |
| `get_actor_params()` / `get_critic_params()` | 参数迭代器 |

### 6.10.3 `AMPModel`

增加 `disc_net`：输入 `disc_obs` → 标量 logit

### 6.10.4 `ASEModel`

增加 `enc_net`：输入 `disc_obs` → 潜变量 z；actor/critic 输入 `[obs, z]`

---

## 6.11 `DummyAgent`

空操作 Agent，用于环境测试，不做任何学习。

---

## 6.12 参考资料

| 论文/资源 | 链接 |
|----------|------|
| PPO (Schulman et al.) | https://arxiv.org/abs/1707.06347 |
| TD(λ) (Sutton, 1988) | http://incompleteideas.net/sutton/book/ebook/node96.html |
| GAE | https://arxiv.org/abs/1506.02438 |

---

[← 环境](05-environments.md) | [返回索引](TECHNICAL_INDEX.md) | [下一章：DeepMimic+PPO →](07-algorithm-deepmimic-ppo.md)
