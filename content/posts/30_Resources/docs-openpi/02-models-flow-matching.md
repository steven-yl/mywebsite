---
title: "02 流匹配模型：π₀ 与 π₀.₅"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "解读 π₀ / π₀.₅ 流匹配 VLA 的模型抽象、JAX 实现与配置，涵盖训练与推理算法。"
summary: "π₀ 与 π₀.₅ 流匹配模型原理与 JAX 实现详解。"
tags: [openpi, robots, diffusion/flow]
categories: [docs openpi]
series: [openpi-docs]
weight: 2
series_weight: 2
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 02 流匹配模型：π₀ 与 π₀.₅

> 本章解读 `src/openpi/models/model.py`（模型通用抽象）、`src/openpi/models/pi0.py`（π₀/π₀.₅ 的 JAX 实现）与 `src/openpi/models/pi0_config.py`（配置）。重点讲清：流匹配是什么、为什么用它、模型如何把多模态输入变成动作、训练与推理的算法步骤。

---

## 2.1 模块边界与文件职责

| 文件 | 职责 |
| --- | --- |
| `models/model.py` | 定义 `Observation`、`Actions`、`BaseModel`、`BaseModelConfig`、观测预处理、权重恢复 |
| `models/pi0.py` | `Pi0` 模型：前缀/后缀嵌入、`compute_loss`、`sample_actions` |
| `models/pi0_config.py` | `Pi0Config`：超参、模型类型判定、冻结过滤器（LoRA） |

π₀ 与 π₀.₅ **共用同一个 `Pi0` 类**，靠 `Pi0Config.pi05` 这个布尔开关切换两种行为差异（见 §2.7）。

---

## 2.2 通用抽象层（model.py）

### 2.2.1 `Observation` —— 模型输入的统一容器

```python
@struct.dataclass
class Observation(Generic[ArrayT]):
    images: dict[str, Float[ArrayT, "*b h w c"]]      # 多路图像，值域 [-1, 1]
    image_masks: dict[str, Bool[ArrayT, "*b"]]        # 每路图像是否有效
    state: Float[ArrayT, "*b s"]                      # 低维本体状态
    tokenized_prompt: Int[ArrayT, "*b l"] | None      # 分词后的语言指令
    tokenized_prompt_mask: Bool[ArrayT, "*b l"] | None
    token_ar_mask: Int[ArrayT, "*b l"] | None         # FAST 专用：自回归掩码
    token_loss_mask: Bool[ArrayT, "*b l"] | None      # FAST 专用：损失掩码
```

**为什么需要它**：不同机器人、不同模型的输入字段五花八门，用一个带类型标注的结构体统一承载，既方便类型检查（`jaxtyping`），又让模型代码与数据来源解耦。`ArrayT` 是泛型，可同时承载 JAX 数组、PyTorch 张量、numpy 数组。

**关键方法**：
- `from_dict(data)`：把数据变换产出的嵌套 dict 转成 `Observation`。其中处理了图像 dtype 转换——若图像是 `uint8`，自动转成 `[-1,1]` 的 float32（JAX 路径直接 `/255*2-1`；PyTorch 路径还会 `permute(0,3,1,2)` 调整通道顺序）。还校验 `tokenized_prompt` 与其 mask 必须成对出现。
- `to_dict()`：反向转换，把 `images`→`image`、`image_masks`→`image_mask`。

`Actions` 只是一个类型别名：`Float[ArrayT, "*b ah ad"]`（批 × 动作步数 × 动作维度）。

### 2.2.2 `preprocess_observation` —— 图像增强与补齐

```python
def preprocess_observation(rng, observation, *, train=False,
                           image_keys=IMAGE_KEYS, image_resolution=(224,224)) -> Observation:
```

职责：
1. 校验必需的图像键齐全（默认 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`）。
2. 把每路图像 resize 到 224×224（用 `image_tools.resize_with_pad`，保持长宽比、黑边填充）。
3. **训练时做数据增强**：非手腕相机做随机裁剪(95%)+缩放+小角度旋转(±5°)，所有相机做颜色抖动（亮度/对比度/饱和度）。增强前把 `[-1,1]` 转到 `[0,1]`，增强后转回。
4. 为缺失的图像 mask 填默认值（默认全 True）。

> 为什么手腕相机不做几何增强？手腕视角对位姿敏感，随机裁剪/旋转会破坏其几何含义，而第三人称相机做轻微几何扰动有助泛化。

### 2.2.3 `BaseModelConfig` 与 `BaseModel`

`BaseModelConfig`（frozen dataclass，抽象）共享三个字段并定义接口：
- `action_dim` / `action_horizon` / `max_token_len`
- `model_type`（抽象属性）、`create`（抽象，建模型）、`inputs_spec`（抽象，输入规格）
- `load(params)`：用 NNX 把恢复的参数注入模型（JAX）。
- `load_pytorch(train_config, weight_path)`：构造 `PI0Pytorch` 并用 safetensors 加载（PyTorch）。
- `fake_obs` / `fake_act`：按 `inputs_spec` 造全 1 的假数据，用于初始化与调试。

`BaseModel`（`nnx.Module`，抽象）：要求实现 `compute_loss` 与 `sample_actions`。

### 2.2.4 `restore_params` —— 检查点恢复

从 Orbax 检查点恢复非结构化参数 PyTree，支持 `gs://` 路径、可指定 dtype 与 sharding。它能处理两种检查点：openpi 训练保存的（键以 `value` 结尾，会被去掉）和官方发布的。

---

## 2.3 流匹配（Flow Matching）原理：是什么、为什么

### 什么是流匹配

流匹配是一种生成式建模范式。它的目标是学习一个**速度场（velocity field）** $v_\theta(x_t, t)$，使得从一个简单分布（高斯噪声）出发，沿着这个速度场做常微分方程（ODE）积分，最终落到目标数据分布上。

本项目采用**直线插值（linear/rectified flow）**的约定。定义：
- $t=1$ 为纯噪声，$t=0$ 为目标动作（注意：这与 π₀ 论文相反，代码注释里专门致歉了）。
- 噪声 $\epsilon \sim \mathcal{N}(0, I)$，真实动作 $a$。
- 插值轨迹：

$$x_t = t\cdot\epsilon + (1-t)\cdot a$$

- 真实速度（目标）：

$$u_t = \frac{dx_t}{dt} = \epsilon - a$$

模型学习预测这个速度，损失是预测速度与真实速度的均方误差：

$$\mathcal{L} = \mathbb{E}_{t,\epsilon}\;\lVert v_\theta(x_t, t) - (\epsilon - a)\rVert^2$$

### 为什么用流匹配（相比自回归/扩散）

- **相比自回归（π₀-FAST）**：流匹配直接输出整段连续动作，无需逐 token 解码，**推理步数固定（默认 10 步）且可并行**，因此推理更快、动作更平滑——适合高频连续控制。
- **相比传统 DDPM 扩散**：直线流匹配的目标速度是常向量 $\epsilon-a$，训练目标更简单、采样路径更直，少量积分步即可。

---

## 2.4 前缀嵌入 `embed_prefix`：图像 + 语言

```python
def embed_prefix(self, obs) -> (tokens, input_mask, ar_mask):
    # 1) 每路图像过 SigLIP 视觉塔 → 一串图像 token
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        input_mask.append(repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
        ar_mask += [False] * image_tokens.shape[1]      # 图像 token 之间互相可见
    # 2) 语言 token 过 Gemma 嵌入层
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]  # 图文之间全注意力
    return concat(tokens), concat(input_mask), array(ar_mask)
```

**要点**：
- `ar_mask`（autoregressive mask）全为 `False`，表示前缀内部是**双向注意力**（图像、语言彼此可见）。
- `input_mask` 标记哪些 token 是有效输入（用于屏蔽 padding）。

### 注意力掩码构造 `make_attn_mask`

```python
def make_attn_mask(input_mask, mask_ar):
    mask_ar = broadcast_to(mask_ar, input_mask.shape)
    cumsum = cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]   # 块状/因果可见性
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return logical_and(attn_mask, valid_mask)
```

`mask_ar` 用累积和的技巧表达三种注意力模式：
- `[0,0,0,1,1,1]`：前缀双向 + 后缀因果（prefix-LM）。
- `[1,1,1,...]`：纯因果。
- `[1,0,1,0,...]`：块间因果、块内双向。

直观理解：token $i$ 能看到 token $j$ 当且仅当 `cumsum[j] <= cumsum[i]` 且两者都是有效输入。

---

## 2.5 后缀嵌入 `embed_suffix`：状态 + 带噪动作 + 时间

```python
def embed_suffix(self, obs, noisy_actions, timestep):
    if not self.pi05:
        # π0：状态作为一个连续 token 进入后缀
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token); ar_mask += [True]   # 前缀不能注意到状态
    # 动作投影
    action_tokens = self.action_in_proj(noisy_actions)
    # 时间步用 sin-cos 位置编码（敏感区间 [0,1]）
    time_emb = posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)
    if self.pi05:
        # π0.5：时间走 MLP，作为 AdaRMS 的条件向量注入归一化
        time_emb = swish(self.time_mlp_in(time_emb)); time_emb = swish(self.time_mlp_out(time_emb))
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
    else:
        # π0：把时间嵌入广播后与动作拼接，过 MLP 融合
        time_tokens = repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time = concat([action_tokens, time_tokens], axis=-1)
        action_time = swish(self.action_time_mlp_in(action_time))
        action_expert_tokens = self.action_time_mlp_out(action_time)
        adarms_cond = None
    ar_mask += [True] + [False]*(self.action_horizon - 1)  # 动作首 token 起新块
    return concat(tokens), concat(input_mask), array(ar_mask), adarms_cond
```

**π₀ 与 π₀.₅ 在这里分叉**（两点核心差异）：

| | π₀ | π₀.₅ |
| --- | --- | --- |
| 状态如何进入 | 作为连续 `state_proj` token 放进后缀 | 作为离散数字进入**语言 token**（不在后缀） |
| 时间步如何注入 | 与动作拼接 → MLP 融合 | 经 `time_mlp` → 作为 **AdaRMS** 条件向量调制每层归一化 |

`ar_mask` 中状态 token 与动作首 token 标 `True`，意味着它们**开启新的注意力块**：前缀（图文）看不到状态/动作，但状态/动作能看到前缀。这正是流匹配 VLA 的关键——条件信息（图文状态）单向流向动作。

### 时间步的 sin-cos 编码 `posemb_sincos`

```python
def posemb_sincos(pos, embedding_dim, min_period, max_period):
    fraction = linspace(0, 1, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid = einsum("i,j->ij", pos, 1/period * 2π)
    return concat([sin(sinusoid), cos(sinusoid)], axis=-1)
```

把标量时间 $t\in[0,1]$ 映射成高维向量，让模型能区分不同去噪阶段。`min_period=4e-3, max_period=4.0` 决定了编码对 $[0,1]$ 区间的敏感度。

---

## 2.6 训练：`compute_loss`

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = preprocess_observation(preprocess_rng, observation, train=train)

    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001   # 采样时间步
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions             # 插值轨迹
    u_t = noise - actions                                                   # 目标速度

    # 一次前向：前缀 + 后缀 拼成一个长序列
    prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, time)
    attn_mask = make_attn_mask(concat([prefix_mask, suffix_mask]), concat([prefix_ar, suffix_ar]))
    positions = cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens],
                                            mask=attn_mask, positions=positions,
                                            adarms_cond=[None, adarms_cond])
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])        # 预测速度
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)                         # 逐步 MSE
```

**算法步骤**：
1. 预处理观测（含增强）。
2. 采样噪声 $\epsilon$ 与时间步 $t$（用 Beta(1.5,1) 偏向小 $t$，再缩放到 $[0.001,1]$）。
3. 构造插值点 $x_t$ 与目标速度 $u_t=\epsilon-a$。
4. 嵌入前缀+后缀，构造注意力掩码与位置，**一次性**前向（训练时前缀后缀同时算）。
5. 取后缀最后 `action_horizon` 个输出，过 `action_out_proj` 得预测速度 $v_t$。
6. 返回逐时间步的 MSE（外层训练循环会再对其求均值）。

> **为什么时间步用 Beta(1.5,1)**：该分布偏向较小的 $t$（更接近目标动作），让模型在"接近数据"的区域获得更多训练信号，经验上有助提升采样质量。

---

## 2.7 推理：`sample_actions`（去噪 ODE）

```python
def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
    observation = preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps                         # 从 t=1 往 t=0 走，dt 为负
    if noise is None:
        noise = jax.random.normal(rng, (batch, action_horizon, action_dim))

    # 1) 先用前缀做一次前向，填充 KV 缓存（图文只需算一次）
    prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None],
                                     mask=make_attn_mask(prefix_mask, prefix_ar),
                                     positions=cumsum(prefix_mask, 1) - 1)

    # 2) 欧拉法迭代去噪
    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = \
            self.embed_suffix(observation, x_t, broadcast(time, batch))
        # 后缀对 [前缀+后缀] 的注意力掩码
        full_attn_mask = concat([repeat(prefix_mask,...), make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
        positions = sum(prefix_mask, -1)[:,None] + cumsum(suffix_mask, -1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens],
                                                mask=full_attn_mask, positions=positions,
                                                kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        v_t = self.action_out_proj(suffix_out[:, -action_horizon:])
        return x_t + dt * v_t, time + dt          # 欧拉步

    x_0, _ = jax.lax.while_loop(cond=lambda c: c[1] >= -dt/2, body=step, init=(noise, 1.0))
    return x_0
```

**算法步骤**：
1. **前缀只算一次**：图文 token 不随去噪步变化，因此先前向一次把 KV 缓存填好，后续步骤复用——这是推理加速的关键。
2. **欧拉积分**：从 $t=1$（纯噪声）出发，每步 $x_{t+dt} = x_t + dt\cdot v_\theta(x_t,t)$，$dt=-1/\text{num\_steps}$，迭代 `num_steps`（默认 10）次到 $t=0$。
3. 每步只前向**后缀**（动作 token），通过 `kv_cache` 注意前缀。
4. 返回 $x_0$ 即去噪后的动作。`cond` 用 `time >= -dt/2` 做浮点鲁棒的终止判断。

```
t=1.0 ──step──► t=0.9 ──step──► ... ──► t=0.0
噪声 x_1                                 动作 x_0
 每步: x ← x + dt·v_θ(x,t)   （dt = -0.1）
```

---

## 2.8 `Pi0Config` 配置详解

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: Variant = "gemma_2b"          # 视觉-语言主干
    action_expert_variant: Variant = "gemma_300m"    # 动作专家
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None        # __post_init__: pi05→200, pi0→48
    pi05: bool = False               # 切换 π0 / π0.5
    discrete_state_input: bool = None  # __post_init__: 默认等于 pi05
    pytorch_compile_mode: str | None = "max-autotune"
```

**关键逻辑**：
- `__post_init__` 根据 `pi05` 设置 `max_token_len`（π₀.₅ 需要更长，因为状态进了语言流）与 `discrete_state_input`。
- `model_type` 属性：`pi05` 为真返回 `PI05`，否则 `PI0`。
- `create(rng)`：实例化 `Pi0` 模型。
- `inputs_spec(batch_size)`：声明输入的形状/dtype 规格（3 路图像、state、tokenized_prompt 等），供初始化与假数据生成。

### `get_freeze_filter` —— LoRA 微调的参数冻结

```python
def get_freeze_filter(self) -> nnx.filterlib.Filter:
    gemma_params_filter = PathRegex(".*llm.*")
    action_expert_params_filter = PathRegex(".*llm.*_1.*")
    # 若 paligemma 用 lora：冻结 gemma 主干（但排除 action expert）
    # 若 action expert 用 lora：冻结 action expert
    # 最后排除所有 lora 参数（lora 参数始终可训练）
```

它返回一个"应被冻结"的参数过滤器：使用 LoRA 时，冻结主干的全精度权重，只训练注入的低秩 LoRA 增量。`.*llm.*_1.*` 这个正则精准匹配第二个专家（动作专家）的权重——回顾 [01 章](01-architecture-overview.md) 的命名约定（`_1` 后缀）。LoRA 实现细节见 [04 章](04-backbone-tokenizers.md)。

---

## 2.9 小结

- π₀/π₀.₅ 是流匹配 VLA：学速度场 $v_\theta(x_t,t)\approx \epsilon-a$，推理时从噪声沿 ODE 积分回动作。
- 输入分前缀（图文，双向）与后缀（状态+动作，受控可见），由统一的混合专家 Transformer 处理。
- π₀ 与 π₀.₅ 的差异集中在两点：状态入口（连续 token vs 离散语言）与时间注入（MLP 拼接 vs AdaRMS）。
- 推理用 KV 缓存复用前缀、欧拉法 10 步去噪，兼顾速度与质量。

下一章 [03 π₀-FAST](03-models-pi0-fast.md) 讲另一条技术路线——自回归动作生成。
