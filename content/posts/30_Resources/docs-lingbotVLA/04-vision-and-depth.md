---
title: "4. 视觉编码与深度对齐"
subtitle: ""
date: 2026-07-13T12:00:00+08:00
draft: false
authors: [Steven]
description: "4. 视觉编码与深度对齐。"
summary: "4. 视觉编码与深度对齐。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA, robots]
series: [lingbotVLA-docs]
weight: 4
series_weight: 4
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 4. 视觉编码与深度对齐

---

## 4.1 视觉输入管线

### 4.1.1 数据侧预处理

`lingbotvla/data/vla_data/transform.py` → `prepare_images()`：

| 路径 | 处理 |
|------|------|
| 非 Qwen | `resize_with_pad` → $[-1, 1]$ 归一化 |
| Qwen (`'qwen' in tokenizer_path`) | `AutoProcessor` 处理，输出 flatten patch |

### 4.1.2 模型侧 `embed_image`

```python
# QwenvlWithExpertModel.embed_image
h = w = int(image.shape[1] ** 0.5)  # sqrt(num_patches)
image_grid_thw = [[1, h, w]] * batch
image_embeds = self.get_image_features(image, image_grid_thw)
# → (B, num_patches, 2048)
```

### 4.1.3 Qwen2.5-VL Vision Transformer

**结构（`qwenvl_in_vla.py`）：**

```
Input patches (Conv3d)
  → 32 × VisionBlock (window attention + full attention 交替)
  → Patch Merger (spatial_merge_size=2)
  → out_hidden_size = 2048
```

**Window Attention：** 降低 ViT 计算量；部分层 full attention 保持全局上下文。

**相关论文/项目：**

- [Qwen2-VL / Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)

---

## 4.2 深度对齐动机

RGB-only VLA 在深度模糊、遮挡、精细对齐任务上受限。**LingBot-VLA-Depth** 通过蒸馏冻结深度教师，使 VLM 隐空间包含几何信息，**无需推理时真实深度传感器**。

**教师栈：**

1. **MoGe-2** — 单目 metric depth
2. **MoRGBD (LingBot-Depth)** — RGB-D 特征编码器

```python
# vision_models/module_utils.py
def build_depth_model(moge_path, morgbd_path):
    # 加载并冻结 MoGe + MoRGBD

def get_depth_target(pil_images, depth_models, align_params):
    # MoGe 估计 depth → MoRGBD.infer_feat → target embeddings
    # 返回 (B*N_cam, num_tokens, dim_out)
```

---

## 4.3 对齐模式对比

| | Direct | Query |
|--|--------|-------|
| **config** | `align_params.mode: direct` | `align_params.mode: query` |
| **结构** | MLP on image tokens | 可学习 query + TaskTokenDepthHead |
| **Prefix 变化** | 无额外 token | 每相机 +num_task_tokens |
| **损失** | L1 + cosine matrix | Smooth L1 |
| **注意力** | 标准 prefix mask | `make_att_2d_masks_with_query` |
| **适用** | 简单蒸馏 | 默认推荐，Perceiver 聚合 |

---

## 4.4 Resampler / Perceiver 模块

**文件：** `align_heads/resampler.py`（源自 [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)）

### 4.4.1 `FeedForward(dim, mult=4)`

LayerNorm → Linear → GELU → Linear

### 4.4.2 `reshape_tensor(x, heads)`

`(B, L, D)` → `(B, heads, L, dim_head)` 用于 multi-head。

### 4.4.3 `PerceiverAttention`

**交叉注意力：** latents 作 Q，concat(image, latents) 作 K/V。

$$
\text{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
$$

使用 scaled dot-product with pre-scaling trick（f16 稳定）。

### 4.4.4 `AttentionPool2d`

CLIP 风格 attention pooling；可选 return all tokens。

### 4.4.5 `Resampler`

固定可学习 latents（`num_queries`）+ 多层 `(PerceiverAttention, FeedForward)`。

### 4.4.6 `TaskTokenResampler`

外部 task tokens 作 query（非固定 latent），用于 depth query 模式。

### 4.4.7 `ResamplerXL` / `ResamplerXLV2`

双分支投影输出版本（扩展用）。

---

## 4.5 Depth Head

**文件：** `align_heads/depth_head.py`

### `DepthHead`

```
llm_feats (B, L, llm_hidden)
  → Resampler(num_backbone_tokens)
  → (B, num_backbone_tokens, dim_out)
```

### `TaskTokenDepthHead`

```
llm_feats + external queries
  → TaskTokenResampler
  → depth token predictions
```

**proj_config 字段：**

| 键 | 含义 |
|----|------|
| `dim_head`, `num_heads` | 注意力头 |
| `num_layers` | Resampler 层数 |
| `num_backbone_tokens` | 输出 token 数（如 256） |
| `dim_out` | 与 MoRGBD 特征维对齐（1024） |
| `ff_mult` | FFN 倍数 |

---

## 4.6 深度损失详解

### Direct 模式 `_emb_loss`

1. **Token L1：** 预测 embedding vs 池化后的 target
2. **Cosine Matrix L1：** 
   $$
   S_{ij} = \langle \hat{e}_i, \hat{e}_j \rangle, \quad \hat{e} = e/\|e\|_2
   $$
   保持 token 间相似度结构与教师一致（对比几何）

总损失：`sim_loss.mean() + l1_loss.mean()`

可选 `contrastive_loss_weight` 在 config 中调节（Training_Config.md）。

### Query 模式

`F.smooth_l1_loss(pred, target)` — Huber 风格，对 outlier 更鲁棒。

**训练权重：** `depth_loss_weight`（典型 0.004）乘以 `loss_depth` 加入总 loss。

---

## 4.7 深度相关配置示例

```yaml
train:
  align_params:
    mode: 'query'
    num_task_tokens: 8
    use_image_tokens: True
    depth_loss_weight: 0.004
    contrastive_loss_weight: 0.3
    llm:
      dim_out: 2048
      image_token_size: 8      # 8×8 = 64 patches/相机（与 grid 相关）
      image_input_size: 224
    depth:
      model_type: MoRGBD
      moge_path: "Ruicheng/moge-2-vitb-normal"
      morgbd_path: "robbyant/lingbot-depth-pretrain-vitl-14"
      num_backbone_tokens: 256
      dim_out: 1024
```

**模型路径（训练脚本）：**

```python
# train_lingbotvla.py
if args.train.align_params != {}:
    depth_models = build_depth_model(moge_path, morgbd_path)
    depth_targets = get_depth_target(micro_batch['pil_images'], ...)
```

---

## 4.8 `module_utils.py` 辅助函数

| 函数 | 说明 |
|------|------|
| `build_depth_model(config)` | 实例化冻结 MoGe + MoRGBD |
| `get_depth_target(pil_images, ...)` | 批量计算教师 embedding |
| `log_depth(pred, gt, step, writer)` | TensorBoard 深度可视化 |

---

## 4.9 深度 Prefix 注意力示意

```
Camera0 patches | Camera1 patches | Camera2 patches |
  Query0 (8)    |   Query1 (8)    |   Query2 (8)    | Language tokens

规则：
- Query_i 仅 attend Camera_i patches + 自身
- Language 不 attend Query tokens
- Query 作为 depth 信息瓶颈
```

---

## 4.10 开源参考

| 项目 | 链接 |
|------|------|
| MoGe | [Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) |
| LingBot-Depth | [robbyant/lingbot-depth-pretrain-vitl-14](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14) |
| OpenFlamingo Perceiver | [mlfoundations/open_flamingo](https://github.com/mlfoundations/open_flamingo) |
| Depth Anything / 单目深度综述 | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) |

---

## 4.11 使用建议

| 场景 | 建议 |
|------|------|
| 仿真 RoboTwin | w/o depth 通常足够 |
| 真实双臂、遮挡 | w/ depth + query 模式 |
| 算力有限 | 关闭 depth；或减小 `num_task_tokens` |
| 微调 depth head | optimizer 对 `depth*` 参数 10× LR（`optimizer.py`） |
