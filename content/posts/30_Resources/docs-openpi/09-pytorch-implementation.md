---
title: "09 PyTorch 实现细节"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "解读 PI0Pytorch、PaliGemmaWithExpertModel、预处理与 transformers 补丁等 PyTorch 实现。"
summary: "openpi PyTorch 实现：PI0Pytorch 与 Gemma 双塔架构详解。"
tags: [openpi, robots, PyTorch]
categories: [docs openpi]
series: [openpi-docs]
weight: 9
series_weight: 9
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 09 PyTorch 实现细节

> 本章解读 `models_pytorch/pi0_pytorch.py`（`PI0Pytorch`）、`models_pytorch/gemma_pytorch.py`（`PaliGemmaWithExpertModel`）、`models_pytorch/preprocessing_pytorch.py`，以及 `transformers_replace` 补丁机制。本章是 [02 章](02-models-flow-matching.md) 流匹配模型的 PyTorch 镜像，对照阅读效果最佳。

---

## 9.1 PyTorch 实现的定位与限制

PyTorch 实现（2025-09 加入）让 π₀/π₀.₅ 能在 PyTorch 生态运行。它基于 HuggingFace `transformers` 的 `PaliGemmaForConditionalGeneration` 和 `GemmaForCausalLM`，但需要打补丁（见 §9.6）。

**支持**：π₀ / π₀.₅ 的推理与全参微调；单卡/多卡 DDP/多节点。
**不支持**：π₀-FAST、混合精度、FSDP、LoRA、EMA。

推理时与 JAX 行为对齐（多数权重 bf16，少数转 fp32 保稳定）；`torch.compile` 下推理速度与 JAX 相当。

---

## 9.2 `PI0Pytorch` 总览（pi0_pytorch.py）

`PI0Pytorch(nn.Module)` 是 [02 章](02-models-flow-matching.md) `Pi0` 的 PyTorch 对应。构造：

```python
class PI0Pytorch(nn.Module):
    def __init__(self, config):
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config, action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        if self.pi05:
            self.time_mlp_in = nn.Linear(width, width)
            self.time_mlp_out = nn.Linear(width, width)
        else:
            self.state_proj = nn.Linear(config.action_dim, width)
            self.action_time_mlp_in = nn.Linear(2*width, width)
            self.action_time_mlp_out = nn.Linear(width, width)
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)
        # 校验 transformers_replace 已正确安装
        from transformers.models.siglip import check
        if not check.check_whether_transformers_replace_is_installed_correctly(): raise ValueError(...)
```

层与 JAX 版一一对应：`action_in_proj`/`action_out_proj` 动作投影；π₀ 有 `state_proj` + `action_time_mlp_*`，π₀.₅ 有 `time_mlp_*`（AdaRMS 用）。构造末尾校验 transformers 补丁是否安装。

### 辅助方法
- `get_safe_dtype(target_dtype, device_type)`：CPU 不支持 bf16，回退 fp32。
- `create_sinusoidal_pos_embedding(time, dim, min_period, max_period, device)`：时间步 sin-cos 编码（对应 JAX `posemb_sincos`），方差/计算在 fp64/fp32 下做。
- `sample_beta(alpha, beta, bsize, device)`：Beta 分布采样（时间步）。
- `make_att_2d_masks(pad_masks, att_masks)`：构造 2D 注意力掩码（对应 JAX `make_attn_mask`，累积和技巧）。
- `sample_noise(shape, device)` / `sample_time(bsize, device)`：采样噪声与时间步（time = beta*0.999+0.001）。
- `_prepare_attention_masks_4d(att_2d_masks)`：把布尔掩码转成 `where(mask, 0.0, -2.38e38)` 的 4D 加性掩码。
- `_preprocess_observation(observation, train)`：调 `preprocess_observation_pytorch`，返回图像列表/掩码/token/state。

---

## 9.3 梯度检查点机制

PyTorch 版内置完整的梯度检查点（gradient checkpointing）支持以省显存：

- `gradient_checkpointing_enable()` / `gradient_checkpointing_disable()`：开关，递归设置 language_model、vision_tower、gemma_expert 的 `gradient_checkpointing` 标志。
- `is_gradient_checkpointing_enabled()`：查询。
- `_apply_checkpoint(func, *args, **kwargs)`：训练且启用时用 `torch.utils.checkpoint.checkpoint(func, ..., use_reentrant=False, preserve_rng_state=False)` 包装，否则直接调用。模型里几乎每个子计算（图像嵌入、语言嵌入、状态投影、MLP、主前向、输出投影）都过这个包装。

> 梯度检查点用"反向时重算激活"换显存，是大模型训练的标配。`train_pytorch.py`（[06 章](06-training-system.md)）默认启用它。

---

## 9.4 前缀/后缀嵌入与前向

### `embed_prefix(images, img_masks, lang_tokens, lang_masks)`
对应 [02 章](02-models-flow-matching.md) §2.4。逐路图像过 `paligemma_with_expert.embed_image`（SigLIP），语言 token 过 `embed_language_tokens` 并乘 `sqrt(dim)`，拼接成前缀，att_masks 全 0（双向）。每个子步用 `_apply_checkpoint` 包装。

### `embed_suffix(state, noisy_actions, timestep)`
对应 §2.5，含 π₀/π₀.₅ 分叉：
- π₀：`state_proj(state)` 作为状态 token（att_mask=1）；时间 sin-cos 编码后与动作拼接过 `action_time_mlp`；`adarms_cond=None`。
- π₀.₅：状态不进后缀；时间经 `time_mlp`（silu）作为 `adarms_cond`；动作直接作为 token。
- att_masks：状态/动作首 token 为 1（开启新块）。

### `forward(observation, actions, noise=None, time=None)` —— 训练
```python
def forward(self, observation, actions, noise=None, time=None):
    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
    if noise is None: noise = self.sample_noise(actions.shape, actions.device)
    if time is None: time = self.sample_time(actions.shape[0], actions.device)
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions     # 插值
    u_t = noise - actions                                            # 目标速度
    prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state, x_t, time)
    # bf16 对齐
    pad_masks = cat([prefix_pad, suffix_pad]); att_masks = cat([prefix_att, suffix_att])
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = cumsum(pad_masks, 1) - 1
    att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
    (_, suffix_out), _ = self.paligemma_with_expert.forward(
        attention_mask=att_2d_masks_4d, position_ids=position_ids, past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
    suffix_out = suffix_out[:, -self.config.action_horizon:].to(torch.float32)
    v_t = self.action_out_proj(suffix_out)
    return F.mse_loss(u_t, v_t, reduction="none")     # 逐元素 MSE
```

与 JAX `compute_loss`（[02 章](02-models-flow-matching.md) §2.6）算法完全一致：插值 → 目标速度 → 一次前向 → 取后缀末段 → 投影 → MSE。注意返回逐元素 MSE，由训练循环求均值。

### `sample_actions(device, observation, noise=None, num_steps=10)` —— 推理
对应 §2.7 的去噪 ODE：
1. `embed_prefix` 后用 `use_cache=True` 前向一次填 KV 缓存（设 `_attn_implementation="eager"`）。
2. 欧拉法循环：`dt=-1/num_steps`，从 `time=1.0` 起每步调 `denoise_step` 得 `v_t`，`x_t = x_t + dt*v_t`，直到 `time >= -dt/2`。
3. 返回 `x_t`。`@torch.no_grad()` 装饰，且 `sample_actions` 被 `torch.compile` 包装。

### `denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)`
单步去噪：`embed_suffix` → 构造后缀对 [前缀+后缀] 的注意力掩码 → 用 `past_key_values`（KV 缓存）只前向后缀 → 取末段过 `action_out_proj` 得 `v_t`。

---

## 9.5 `PaliGemmaWithExpertModel`（gemma_pytorch.py）

这是混合专家的 PyTorch 实现，把 HF 的 `PaliGemmaForConditionalGeneration`（含 SigLIP + Gemma-2B）和 `GemmaForCausalLM`（动作专家 300M）组合，让二者**逐层共享注意力**。

### 构造
```python
class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, vlm_config, action_expert_config, use_adarms=None, precision="bfloat16"):
        # 用 HF CONFIG_MAPPING 构造 paligemma 配置，按 vlm_config 覆盖 width/heads/depth/...
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        action_expert_config_hf = CONFIG_MAPPING["gemma"](..., use_adarms=use_adarms[1], ...)
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None     # 动作专家不需要词嵌入
        self.to_bfloat16_for_selected_params(precision)
```

### 方法
- `to_bfloat16_for_selected_params(precision)`：精度控制。`bfloat16` 时整体转 bf16，但**保留若干参数为 fp32**（patch_embedding、position_embedding、各种 layernorm、model.norm）以保数值稳定；`float32` 时全转 fp32。
- `embed_image(image)`：调 `paligemma.model.get_image_features`（SigLIP 视觉塔）。
- `embed_language_tokens(tokens)`：调 `paligemma.language_model.embed_tokens`。
- `forward(attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond)`：核心。三种模式：
  - 仅前缀（`inputs_embeds[1] is None`）：只跑 PaliGemma 语言模型，返回输出 + KV 缓存（推理 prefill）。
  - 仅后缀（`inputs_embeds[0] is None`）：只跑动作专家（推理去噪步）。
  - 两者都有（训练）：**逐层交错计算**——见下。

### 逐层混合专家前向（`compute_layer_complete`）
训练/联合前向时，对每一层：
```python
def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
    models = [paligemma.language_model, gemma_expert.model]
    # 1) 各专家各自做 input_layernorm（可能 AdaRMS，返回 gate）并算 Q/K/V
    for i, hidden in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden, gate = layer.input_layernorm(hidden, cond=adarms_cond[i])
        query/key/value = layer.self_attn.{q,k,v}_proj(hidden)...
    # 2) 拼接各专家的 Q/K/V，统一算注意力（RoPE + eager attention）
    query_states = cat(...); key_states = cat(...); value_states = cat(...)
    cos, sin = rotary_emb(...); apply_rotary_pos_emb(...)
    att_output = eager_attention_forward(...)
    # 3) 切回各专家，各自做 o_proj + 门控残差 + post_attention_layernorm + MLP + 门控残差
    for i, hidden in enumerate(inputs_embeds):
        out = layer.self_attn.o_proj(att_output[:, start:end])
        out = _gated_residual(hidden, out, gates[i])
        out, gate = layer.post_attention_layernorm(out, cond=adarms_cond[i])
        out = layer.mlp(out); out = _gated_residual(after_first_residual, out, gate)
    return outputs_embeds
```

这与 JAX `gemma.py` 的 `Block`（[04 章](04-backbone-tokenizers.md) §4.1.6）逻辑等价：各专家独立投影，拼接后共享注意力，再切回各专家做门控残差 + MLP。各层可选 `torch.utils.checkpoint` 包装。末尾对每个专家做 `model.norm`（`compute_final_norms`，同样可检查点）。

> 这里能看到 PyTorch 实现为对齐 JAX 行为付出的努力：注意力实现强制 `eager`、精度逐处控制、bf16/fp32 转换散布在各处、门控残差用 patch 进 transformers 的 `_gated_residual`。

---

## 9.6 transformers 补丁机制（transformers_replace）

PyTorch 实现依赖对 HF `transformers==4.53.2` 的修改，通过覆盖文件实现：

```bash
uv pip show transformers   # 确认 4.53.2
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

补丁提供三项能力（README 说明）：
1. **支持 AdaRMS**：让 Gemma 的 layernorm 接受 `cond` 参数并返回 gate（π₀.₅ 时间注入）。
2. **精确控制激活精度**：在关键处控制 bf16/fp32。
3. **允许 KV 缓存只读使用**：去噪步复用前缀缓存而不更新它。

构造 `PI0Pytorch` 时通过 `transformers.models.siglip.check.check_whether_transformers_replace_is_installed_correctly()` 校验补丁是否生效，否则报错提示安装步骤。

> ⚠️ **副作用**：默认 uv 用 hardlink，此操作会**永久修改 uv 缓存中的 transformers**，可能影响其它使用 transformers 的项目。撤销需 `uv cache clean transformers`。

---

## 9.7 PyTorch 预处理（preprocessing_pytorch.py）

`preprocess_observation_pytorch(observation, *, train, image_keys, image_resolution)`：[02 章](02-models-flow-matching.md) §2.2.2 的 PyTorch 版，且**兼容 torch.compile**（避免复杂类型标注）。

要点：
- 自动识别 `[B,C,H,W]` 与 `[B,H,W,C]` 两种图像布局并互转。
- resize 用 `image_tools.resize_with_pad_torch`。
- 训练增强：非手腕相机随机裁剪(95%)+插值缩放+小角度旋转（用 `grid_sample`），所有相机做亮度/对比度/饱和度抖动。**全程用张量操作（避免 `.item()`）以兼容 torch.compile**。
- 返回一个轻量 `SimpleProcessedObservation` 对象（而非复杂的 `Observation` 类，同样为 torch.compile 友好）。

> 与 JAX 版（用 `augmax` 做增强）相比，PyTorch 版手写了等价的增强逻辑，并特别注意 torch.compile 兼容性（不用 Python 标量、不用复杂 dataclass）。

---

## 9.8 JAX 与 PyTorch 模型实现对照表

| 概念 | JAX (`pi0.py` / `gemma.py`) | PyTorch (`pi0_pytorch.py` / `gemma_pytorch.py`) |
| --- | --- | --- |
| 主模型 | `Pi0` | `PI0Pytorch` |
| 混合专家骨干 | `gemma.Module`（NNX bridge） | `PaliGemmaWithExpertModel`（HF transformers） |
| 视觉塔 | `siglip._Module` | HF `PaliGemmaForConditionalGeneration` 内置 SigLIP |
| 时间编码 | `posemb_sincos` | `create_sinusoidal_pos_embedding` |
| 注意力掩码 | `make_attn_mask` | `make_att_2d_masks` + `_prepare_attention_masks_4d` |
| 门控残差 | `_gated_residual` | `_gated_residual`（patch 进 transformers） |
| 训练入口 | `compute_loss` | `forward` |
| 推理入口 | `sample_actions` | `sample_actions`（torch.compile） |
| 省显存 | `nn.remat` | `torch.utils.checkpoint` |
| 权重格式 | Orbax | safetensors |
| AdaRMS | RMSNorm(cond) 内建 | 需 transformers 补丁 |

---

## 9.9 小结

- `PI0Pytorch` 是 `Pi0` 的 PyTorch 镜像，算法（流匹配训练/推理）完全一致，层结构一一对应。
- `PaliGemmaWithExpertModel` 用 HF transformers 组合 PaliGemma + 动作专家，逐层交错计算实现共享注意力。
- 为对齐 JAX 行为，PyTorch 实现需打 transformers 补丁（AdaRMS/精度/KV 缓存），并在各处精细控制精度。
- 预处理与采样都为 torch.compile 兼容性做了专门处理；功能上不支持 FAST/LoRA/FSDP/EMA/混合精度。

至此，从模型、骨干、数据、训练、推理、客户端到 PyTorch 实现，openpi 的技术全貌已完整覆盖。回到 [文档索引](README.md) 可按需复查任意章节。
