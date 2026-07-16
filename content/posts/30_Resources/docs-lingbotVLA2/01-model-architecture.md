---
title: "1. 模型架构"
subtitle: ""
date: 2026-07-13T14:00:00+08:00
draft: false
authors: [Steven]
description: "本章详解 LingBot-VLA 2.0 的神经网络结构：Qwen3-VL 多模态骨干、MoE Action Expert、统一动作空间与联合注意力机制。"
summary: "本章详解 LingBot-VLA 2.0 的神经网络结构：Qwen3-VL 多模态骨干、MoE Action Expert、统一动作空间与联合注意力机制。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA2, robots]
series: [lingbotVLA2-docs]
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 1. 模型架构

本章详解 LingBot-VLA 2.0 的神经网络结构：Qwen3-VL 多模态骨干、MoE Action Expert、统一动作空间与联合注意力机制。

---

## 1.1 整体结构

LingBot-VLA 2.0 采用 **前缀-后缀（Prefix-Suffix）** 架构，源自 π₀（Physical Intelligence）与 OpenVLA 系列的设计范式：

| 部分 | 内容 | 模型 | 训练时是否更新 |
|------|------|------|----------------|
| **前缀 (Prefix)** | 图像 token + 语言 token + Query token | Qwen3-VL | 可选冻结 ViT |
| **后缀 (Suffix)** | 状态 token + 动作 chunk token | Qwen2 MoE Action Expert | 是 |

核心类层次：

```
LingbotVlaV2Policy          # HuggingFace PreTrainedModel 封装
  └── FlowMatchingV2          # Flow Matching 逻辑 + 投影层 + 对齐头
        └── QwenvlWithExpertV2Model
              ├── qwenvl: Qwen3VLForConditionalGeneration
              └── qwen_expert: Qwen2ForCausalLM (36层, MoE)
```

源码入口：`lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`

---

## 1.2 Qwen3-VL 骨干

### 1.2.1 是什么、为什么选它

**Qwen3-VL-4B-Instruct** 是阿里通义团队的多模态大模型，具备：

- 强大的视觉-语言对齐能力（预训练于海量图文对）
- 原生支持多图输入（适合多相机机器人场景）
- `grid_thw` 机制处理任意分辨率图像

VLA 需要理解「把红色杯子放到盘子里」这类指令，并关联到视觉场景；大 VLM 提供了现成的语义表征，避免从零训练视觉-语言模块。

### 1.2.2 组件

| 子模块 | 配置 | 作用 |
|--------|------|------|
| Vision Tower (ViT) | Qwen3-VL vision_config | 将图像编码为 visual tokens |
| Text Model | Qwen3-VL text_config | 编码语言指令，与 visual tokens 联合注意力 |
| Image Processor | `build_processor()` | 生成 `pixel_values` + `image_grid_thw` |

### 1.2.3 前缀嵌入流程

```python
# modeling_lingbot_vla_v2.py - embed_prefix 简化逻辑
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, image_grid_thw):
    # 1. 图像经 ViT → visual_embeds
    visual_embeds = self.qwenvl.visual(images, grid_thw=image_grid_thw)
    # 2. 语言 token embedding
    lang_embeds = self.qwenvl.model.embed_tokens(lang_tokens)
    # 3. 拼接 [visual | language | query_tokens]
    prefix_embs = torch.cat([visual_embeds, lang_embeds, query_token_embeds], dim=1)
    return prefix_embs, masks, position_ids, ...
```

**`image_grid_thw`**：Qwen3-VL 特有的 `(T, H, W)` 网格描述符，表示每张图被切分为多少 temporal/height/width patch。v2.0 **必须**提供此字段（与 v1.0 不同）。

### 1.2.4 因果 vs 双向注意力

配置项 `vlm_causal`：

- `false`（默认部分场景）：图像与语言 token 双向可见，利于全局场景理解
- `true`（真机配置常用）：语言对图像因果，更贴近自回归生成范式

---

## 1.3 统一动作表示（55 维）

### 1.3.1 设计动机

不同机器人本体（单臂、双臂、人形、移动底盘）的状态/动作维度各异。LingBot-VLA 2.0 定义 **55 维规范空间**，将异构数据映射到同一向量，使单一模型可跨本体预训练与微调。

### 1.3.2 维度分配

| 子空间 | 维度 | 含义 | 典型来源 |
|--------|------|------|----------|
| `arm.position` | 14 | 关节角位置 | 左臂 6 + 右臂 6 + padding |
| `end.position` | 14 | 末端执行器位姿 | 位置 3 + 旋转 4（×2 臂） |
| `effector.position` | 2 | 夹爪开合 | 左 1 + 右 1 |
| `hand.position` | 12 | 灵巧手关节 | 每只手 6 DoF |
| `waist.position` | 4 | 腰部关节 | 人形机器人 |
| `head.position` | 2 | 头部关节 | 人形/云台 |
| `base.position` | 3 | 移动底盘 | x, y, θ 或速度 |
| *reserved* | 4 | 预留 | 扩展用 |
| **合计** | **55** | | |

未使用的维度在 robot config 中保持零填充（padding），`joint_mask` 在损失计算时屏蔽无效维度。

### 1.3.3 动作 Chunk

模型一次预测 **连续多步动作**（默认 `chunk_size=50`），而非单步。好处：

- 时间一致性：避免逐步预测的抖动
- 计算效率：前缀只需编码一次
- 与 ACT（Action Chunking Transformer）思想一致

推理时可只执行 chunk 前 `use_length` 步（如 25 步），再重新观测规划。

---

## 1.4 MoE Action Expert

### 1.4.1 是什么、为什么需要

**Action Expert** 是专门生成机器人动作的解码器，与 VLM 分离：

- VLM 参数量大（~4B），主要做感知与语义理解
- Action Expert 较小（36 层 × 768 hidden），专注动作动力学
- 分离设计允许 **冻结 VLM、只微调 Expert**（`train_expert_only`）

v2.0 在 Expert 的 **全部 36 层 FFN** 使用 **Token-level Sparse MoE**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `token_num_experts` | 32 | 路由专家数 |
| `token_top_k` | 4 | 每 token 激活专家数 |
| `token_moe_intermediate_size` | 512 | 每专家 FFN 中间维 |
| `token_shared_intermediate_size` | 704 | 共享专家 FFN 中间维 |

### 1.4.2 MoE 前向（概念）

对每个 suffix token 的 hidden state \(h\)：

1. **Router**：\(g = \text{sigmoid}(W_g h)\)，取 top-4 专家
2. **Routed Experts**：\(y_{\text{routed}} = \sum_{i \in \text{top-k}} g_i \cdot \text{FFN}_i(h)\)
3. **Shared Expert**：\(y_{\text{shared}} = \text{FFN}_{\text{shared}}(h)\)（始终激活）
4. **输出**：\(y = y_{\text{routed}} + y_{\text{shared}}\)

实现：`Qwen2TokenMoeBlock` + `Qwen2FusedExperts`（Group GEMM 融合）

```python
# qwen2_action_expert.py - 融合专家权重形状
# gate_proj: [E, intermediate_size, hidden_size]
# up_proj:   [E, intermediate_size, hidden_size]
# down_proj: [E, hidden_size, intermediate_size]
```

### 1.4.3 负载均衡

MoE 常见问题：部分专家过载、部分「死亡」。本仓库采用：

1. **Loss-free bias**（`bias_update_speed=0.00025`）：根据各专家 token 计数动态调整 router bias，无需额外损失项
2. **可选 z-loss**（`router_z_loss_coeff=1e-4`）：惩罚 router logits 过大，稳定训练
3. **可选 sequence-wise balance loss**（`sequence_wise_loss_coeff=1e-3`）：DeepSeek-V3 风格序列内专家均衡

### 1.4.4 AdaNorm 与时间条件

Flow Matching 需要在不同噪声时间步 \(t\) 产生不同预测。`adanorm_time=true` 时：

- 正弦时间嵌入经 MLP 得到条件向量 `cond`
- 各层 RMSNorm 替换为 AdaRMSNorm：\(\hat{h} = (1+\gamma(\text{cond})) \cdot \text{RMSNorm}(h) + \beta(\text{cond})\)

```python
# modeling_lingbot_vla.py - AdaRMSNorm
gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
beta  = self.beta(cond).unsqueeze(1)
hidden_states = (1 + gamma) * hidden_states + beta
```

---

## 1.5 投影层与后缀嵌入

| 层 | 输入 → 输出 | 作用 |
|----|-------------|------|
| `state_proj` | state [B, 55] → [B, 768] | 本体状态嵌入 |
| `action_in_proj` | noisy action [B, T, 55] → [B, T, 768] | 噪声动作嵌入 |
| `action_time_mlp_in/out` | concat(action_emb, time_emb) → action_emb | 融合时间信息 |
| `action_out_proj` | hidden [B, T, 768] → [B, T, 55] | 预测速度场 v_t |

后缀序列结构：`[state_token | action_token_1 | ... | action_token_T]`

```python
# embed_suffix 核心逻辑
time_emb = create_sinusoidal_pos_embedding(timestep, proj_width, min_period=4e-3, max_period=4.0)
action_emb = action_in_proj(noisy_actions)
time_emb = time_emb[:, None, :].expand_as(action_emb)
action_time = cat([action_emb, time_emb], dim=-1)
action_time = action_time_mlp_out(silu(action_time_mlp_in(action_time)))
suffix_embs = cat([state_emb[:, None], action_time], dim=1)
```

---

## 1.6 联合注意力（Prefix-LM）

Action Expert 的 Query 来自 suffix，Key/Value 来自 **拼接后的 prefix + suffix**（或缓存的 prefix KV）。

### 1.6.1 注意力掩码

`make_att_2d_masks(pad_masks, att_masks)` 构建 `[B, L, L]` 布尔掩码：

- `pad_masks`：有效 token 位置
- `att_masks`：控制因果/块因果结构（如 state 可见全部 prefix，action tokens 因果）

### 1.6.2 Flex Attention

v2.0 默认 `attention_implementation: flex_cached`：

- 基于 PyTorch 2.5+ `flex_attention`
- 支持任意 2D block mask，高效处理 prefix+suffix 异构注意力
- 推理时 prefix KV **缓存复用**，每步去噪只计算 suffix

```python
# flex_attention.py
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
```

### 1.6.3 梯度阻断（Distillation 专用）

当启用深度/视频蒸馏时，可配置：

- `block_future_depth_to_action`：action suffix **不能 attend** 到 future depth query
- `block_suffix_to_future_video`：action suffix **不能 attend** 到 future video query

目的：迫使动作预测依赖当前几何/语义表征，而非直接「偷看」未来 query 的监督信号。

---

## 1.7 Dual-Query Tokens

在 prefix 末尾追加 **8 个可学习 query token**（`num_task_tokens=8`），分为：

| 类型 | 数量 | 监督来源 |
|------|------|----------|
| Current depth query | 部分 | LingBot-Depth 当前帧深度特征 |
| Future depth query | 部分 | LingBot-Depth 未来帧深度特征 |
| Future video query | 部分 | DINO-Video 未来帧 patch 特征 |

Query 经 Resampler 头投影到教师特征空间，计算 MSE 对齐损失。详见 [03-dual-query-distillation.md](./03-dual-query-distillation.md)。

---

## 1.8 LingBot-VLA 1.0 对比

| 项目 | v1.0 | v2.0 |
|------|------|------|
| 文件 | `modeling_lingbot_vla.py` | `modeling_lingbot_vla_v2.py` |
| 配置 | `LingbotVLAConfig` | `LingbotVLAV2Config` |
| VLM | Qwen2.5-VL | Qwen3-VL |
| `image_grid_thw` | 可选 | **必需** |
| MoE | 可选 | 默认全开 36 层 |
| 蒸馏 | 部分支持 | 完整 Dual-Query |

π₀ 基线（`lingbotvla/models/vla/pi0/`）使用 PaliGemma + Gemma Expert，训练入口为 `train_pi0.py`，用于对比实验。

---

## 1.9 参数量与显存

| 组件 | 约参数量 |
|------|----------|
| Qwen3-VL-4B | ~4B |
| Action Expert (MoE) | ~2B（含 32×36 专家） |
| 对齐头 + 投影层 | ~数十 M |
| **LingBot-VLA 2.0 总计** | **~6B** |

Post-training 推荐配置（4×A6000）：`micro_batch_size=1`，`enable_gradient_checkpointing=true`，约 **49GB/GPU**。

---

## 1.10 可运行示例：模型结构探查

```python
"""探查 LingBot-VLA 2.0 模型结构（需已下载权重）"""
import torch
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy

config = LingbotVLAV2Config.from_pretrained("robbyant/lingbot-vla-v2-6b")
config.tokenizer_path = "Qwen/Qwen3-VL-4B-Instruct"
config.max_action_dim = 55
config.max_state_dim = 55
config.n_action_steps = 50
config.num_steps = 10

model = LingbotVlaV2Policy(config)
print("VLM layers:", len(model.model.qwenvl_with_expert.qwenvl.model.layers))
print("Expert layers:", len(model.model.qwenvl_with_expert.qwen_expert.model.layers))
print("MoE layers:", sum(
    1 for l in model.model.qwenvl_with_expert.qwen_expert.model.layers
    if hasattr(l.mlp, "num_experts")
))
```
