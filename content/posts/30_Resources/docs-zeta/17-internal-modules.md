---
title: "第 16 章：内部与未导出模块附录"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "本文档覆盖 存在但未纳入公开 init.py 导出 的模块，避免知识遗漏。"
summary: "本文档覆盖 存在但未纳入公开 init.py 导出 的模块，避免知识遗漏。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 17
series_weight: 17
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 16 章：内部与未导出模块附录

本文档覆盖 **存在但未纳入公开 `__init__.py` 导出** 的模块，避免知识遗漏。

---

## 1. zeta.nn.attention（未导出）

| 文件 | 符号 | 说明 |
|------|------|------|
| `base.py` | `BaseAttention` | MQA 等注意力基类 |
| `cross_attention.py` | `CrossAttention` | 通用交叉注意力 |
| `dilated_attention.py` | `DilatedAttention`, `MultiheadDilatedAttention`, `ParallelWrapper` | 膨胀窗口注意力 |
| `multi_modal_cross_attn.py` | `MultiModalCrossAttention` | 与 cross_attn_images 重复实现 |
| `shaped_attention.py` | `ShapedAttention` | 形状约束注意力 |
| `xc_attention.py` | `XCAttention` | 跨通道注意力 |

---

## 2. zeta.nn.embeddings（未导出）

| 文件 | 符号 |
|------|------|
| `base.py` | `BaseEmbedding` |
| `patch_embedding.py` | `PatchEmbeddings` |

---

## 3. zeta.nn.masks（部分未导出）

| 文件 | 符号 |
|------|------|
| `block_diagonal.py` | `get_mask` |

`nn/__init__.py` 未 re-export masks 子包；使用：`from zeta.nn.masks.attn_masks import BlockDiagonalCausalMask`

---

## 4. zeta.nn.quant（未导出）

| 文件 | 符号 | 说明 |
|------|------|------|
| `qmoe.py` | `QMOEQuantizer`, `batch_gptq` | MoE GPTQ 量化 |
| `residual_vq.py` | `ResidualVectorQuantizer` | 残差 VQ |
| `random_proj_quan.py` | — | **空文件占位** |

---

## 5. zeta.nn.modules（未导出，216 中约 50+）

| 文件 | 主要符号 | 说明 |
|------|----------|------|
| `deepseek_moe.py` | `DeepSeekMoE` | DeepSeek 共享专家 |
| `g_shard_moe.py` | `GShardMoELayer`, `MOELayer`, `Top1Gate`, `Top2Gate` | GShard 分布式 MoE |
| `mixtral_expert.py` | `MixtralExpert` | Mixtral SWiGLU 专家 |
| `expert.py` | `Expert` | 单专家 FFN |
| `perceiver_resampler.py` | `PerceiverResampler`, `GatedCrossAttentionBlock` | Flamingo 风格重采样 |
| `s4.py` | `s4d_kernel` | S4 对角核 |
| `kv_cache.py` | `KVCache` | KV 缓存管理 |
| `diffusion.py` | `Diffuser` | 扩散过程封装 |
| `vision_mamba.py` | `VisionMambaBlock` | 视觉 Mamba |
| `sparq_attn.py` | `SparQAttention` | SparQ 稀疏注意力 |
| `monarch_mlp.py` | `MonarchMLP` | Monarch 结构化矩阵 MLP |
| `hebbian.py` | `BasicHebbianGRUModel` | Hebbian 学习 GRU |
| `swarmalator.py` | 多个仿真函数 | Swarmalator 动力学 |
| `triton_rmsnorm.py` | Triton RMSNorm | GPU 定制 kernel |
| `droppath.py` | DropPath | 随机深度路径 |
| `ssm_language.py` | SSM 语言模型 | 语言 SSM 完整栈 |
| `vss_block.py` | VSS Block | 视觉状态空间块 |
| `modality_adaptive_module.py` | 模态自适应 | 动态模态权重 |
| `omnimodal_fusion.py` | 全模态融合 | 实验性融合 |
| `pretrained_t_five.py` | T5 预训练封装 | T5 权重加载 |
| `decision_tree.py` | 决策树模块 | 可微决策树实验 |
| `batched_dp.py` | 动态规划批处理 | 序列 DP |
| `matrix.py` | 矩阵工具 | 张量矩阵运算 |
| `tensor.py` / `tensor_shape.py` / `tensor_to_int.py` | 张量工具 | 形状与转换 |
| `flatten_features.py` | 特征展平 | 视觉特征处理 |
| `multiclass_label.py` | 多类标签 | 标签编码 |
| `res_net.py` | ResNet 变体 | 与 resnet.py 并存 |
| `scalenorm.py` / `scaled_sinusoidal.py` | 缩放归一化 | 归一化变体 |
| `simple_rmsnorm.py` | 简化 RMSNorm | 轻量实现 |
| `shift_tokens.py` | Token 移位 | 局部上下文 |
| `skip_connect.py` | 跳跃连接 | U-Net 辅助 |
| `sparc_alignment.py` | SPARC 对齐 | 对齐损失模块 |
| `spacial_transformer.py` | 空间 Transformer | 拼写变体 spatial |
| `dyna_conv.py` | 动态卷积 | 条件卷积 |
| `alr_block.py` | ALR 块 | 自适应层 |
| `attn.py` | 注意力封装 | 内部 attn 辅助 |
| `cross_embed_layer.py` | CrossEmbed | 交叉嵌入 |
| `embedding_to_grid.py` | 嵌入→网格 | 空间恢复 |
| `expand_channels.py` | 通道扩展 | 维度变换 |
| `fast_text.py` | FastText 风格 | 文本卷积 |
| `film_efficient_metb3.py` | FiLM EfficientNet | 条件 EfficientNet |
| `flash_conv.py` | Flash 卷积 | 快速卷积实验 |
| `fractorial_net.py` | 分形网络 | FractalNet 风格 |
| `image_projector.py` | 图像投影 | 多模态投影 |
| `img_reshape.py` | 图像 reshape | 布局变换 |
| `lambda_mask.py` | Lambda 掩码 | 可学习掩码 |
| `log_ff.py` | 对数 FFN | 对数域 FFN |
| `mixtape.py` | Mixtape | 混合架构实验 |
| `nearest_upsample.py` | 最近邻上采样 | 上采样 |
| `norm_fractorals.py` | 分形归一化 | 见 NormalizationFractral 导出 |
| `peg.py` | PEG | 见第 7 章 |
| `pyro.py` | `hyper_optimize` | **已导出** |
| `recurrent_model.py` | 循环模型 | RNN 栈 |
| `scale.py` | Scale 层 | 可学习缩放 |

---

## 6. zeta.nn.modules.xmoe

| 文件 | 符号 | 说明 |
|------|------|------|
| `routing.py` | `Top1Gate`, `Top2Gate`, `top1gating`, `top2gating` | 分布式 MoE 路由 |
| `global_groups.py` | `get_moe_group`, `get_all2all_group` | 进程组 |

`xmoe/__init__.py` 为空，需直接导入。

---

## 7. zeta.models（未导出）

| 文件 | 符号 |
|------|------|
| `BEiT3.py` | `BEiT3` |
| `LongNet.py` | `LongNet`, `LongNetTokenizer` |
| `kosmos.py` | `Kosmos`, `KosmosTokenizer` |
| `mm_mamba.py` | `MultiModalMamba` |
| `Magneto.py` | **空文件** |

---

## 8. zeta.structs（未导出）

| 文件 | 符号 |
|------|------|
| `efficient_net.py` | `EfficientNet`, `MBConv`, `ConvBNReLU` |
| `hierarchical_transformer.py` | `HierarchicalTransformer`, `HierarchicalBlock`, `Compress` |

---

## 9. zeta.ops（未导出）

| 文件 | 符号 |
|------|------|
| `async_softmax.py` | `AsynchronizedAttention`, `asynchronized_softmax` |
| `expand.py` | `expand` |
| `laplace.py` | `laplace_solver`, `follow_gradient` |
| `mos.py` | `MixtureOfSoftmaxes` |
| `nonlinear.py` | `newtons_method`, `broydens_method` |
| `sparsemax.py` | `sparsemax` |

---

## 10. zeta.optim（未导出）

| 文件 | 符号 |
|------|------|
| `all_new_optimizer.py` | `FastAdaptiveOptimizer` |
| `parallel_gradient_descent.py` | `parallel_gradient_descent` |

---

## 11. zeta.rl（未导出）

| 文件 | 符号 |
|------|------|
| `ppo.py` | `ppo_step`（与 actor_critic.ppo 并存） |
| `reward_model.py` | `RewardModel` |
| `vision_model_rl.py` | `VisionRewardModel`, `ResidualBlock` |
| `priortized_replay_buffer.py` | `PrioritizedReplayBuffer` |
| `priortized_rps.py` | `PrioritizedSequenceReplayBuffer` |
| `sumtree.py` | `SumTree` |
| `rest.py` | **占位注释，无实现** |

---

## 12. zeta.training（未导出）

| 文件 | 符号 |
|------|------|
| `activation_checkpoint.py` | `activation_checkpointing` |
| `galore.py` | `GaloreOptimizer` |
| `hive_trainer.py` | `HiveTrainer` |

---

## 13. zeta.utils（未导出）

| 文件 | 符号 |
|------|------|
| `lazy_loader.py` | `LazyLoader`, `lazy_import` |

---

## 14. 不存在的包

| 路径 | 状态 |
|------|------|
| `zeta/tokenizers` | **不存在**（仅有 tests/examples/docs） |
| `zeta/cloud` | 注释掉，未启用 |

---

## 15. 已知 Bug / 技术债

| 位置 | 问题 |
|------|------|
| `rl/dpo.py:freeze_all_layers` | `reqires_grad` 拼写错误，冻结失效 |
| `structs/__init__.py` | `__all__` 含 `VideoTokenizer` 未导入 |
| `ops/__Init__.py` | 文件名大小写 |
| `Magneto.py`, `random_proj_quan.py`, `rl/rest.py` | 空/占位 |

---

上一章：[16-nn-modules-catalog.md](./16-nn-modules-catalog.md) | 返回：[README.md](./README.md)
