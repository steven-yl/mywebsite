---
title: "6. 训练与分布式系统"
subtitle: ""
date: 2026-07-13T12:00:00+08:00
draft: false
authors: [Steven]
description: "6. 训练与分布式系统。"
summary: "6. 训练与分布式系统。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA, robots]
series: [lingbotVLA-docs]
weight: 6
series_weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 6. 训练与分布式系统

---

## 6.1 训练入口

**脚本：** `tasks/vla/train_lingbotvla.py`  
**启动：** `bash train.sh tasks/vla/train_lingbotvla.py <yaml> [CLI overrides]`

`train.sh` 自动检测 GPU 数，设置 `torchrun` 环境变量。

### 参数 Dataclass（`utils/arguments.py`）

| 类 | 前缀 | 内容 |
|----|------|------|
| `ModelArguments` | `model.*` | model_path, tokenizer_path, vocab_size |
| `DataArguments` | `data.*` | train_path, joints, cameras, norm |
| `TrainingArguments` | `train.*` | LR, FSDP, batch, checkpoint, align_params |

`parse_args()` 读取 YAML 首个参数 + CLI `--section.key value` 覆盖。

---

## 6.2 训练循环

```python
for global_step in range(max_steps):
    micro_batches = next(data_iterator)  # list, len = grad_accum_steps
    for micro_batch in micro_batches:
        micro_batch = to_cuda(micro_batch)
        if depth_align:
            depth_targets = get_depth_target(...)
        loss, ... = model(**micro_batch, vlm_causal=..., depth_targets=...)
        (loss / len(micro_batches)).backward()
    clip_grad_norm_(...)
    optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
    all_reduce metrics
    if save_steps: Checkpointer.save(); export hf_ckpt
```

**Loss 聚合：**

- VLA：`losses.mean()` over action dims + valid joint mask
- Depth：`loss_depth * depth_loss_weight`
- 跨 micro-batch 平均后 backward

---

## 6.3 并行状态 `parallel_state.py`

### DeviceMesh 维度

`pp × dp_replicate × dp_shard × ulysses × cp × tp`

Flatten 子 mesh：

| 名称 | 用途 |
|------|------|
| `dp` | 数据并行 |
| `dp_shard_sp` | FSDP + Ulysses |
| `sp` | 序列并行组 |

**默认 VLA：** `ulysses_parallel_size=1`，纯 FSDP2。

### `init_parallel_state(...)`

训练开始时调用，构建全局 `ParallelState` 单例。

---

## 6.4 FSDP2 并行化

**文件：** `distributed/torch_parallelize.py` → `build_parallelize_model()`

### 策略

| 模式 | API |
|------|-----|
| `fsdp2` | `fully_shard` composable API |
| `fsdp1` | `FullyShardedDataParallel` |
| `fsdp2-vescale` | VeScale 后端 |
| `ddp` | DistributedDataParallel |

### FSDP2 分片粒度

- `Qwen2_5_VLDecoderLayer` × 36
- `Qwen2_5_VLVisionBlock` × 32
- Expert `Qwen2DecoderLayer` × 36
- 根模块 `fully_shard` 防止大 AllGather

**混合精度：** bf16 参数，fp32 reduce（`MixedPrecisionPolicy`）

### 权重加载

- `init_device=meta`：`distribute_tensor` 按 shard 加载
- `fsdp/initialize.py`：`parallel_load_safetensors` 分 rank 读盘

### Expert Parallel（MoE）

- `parallel_plan.py` — DTensor 分片计划
- `fsdp/extension.py` — EP 参数 state_dict 处理
- `fsdp/clip_grad_norm.py` — EP+FSDP 分群 clip

> VLA 4B 非 MoE，EP/group_gemm 为框架预留。

---

## 6.5 序列并行（Ulysses）

**目录：** `distributed/sequence_parallel/`

| 文件 | 功能 |
|------|------|
| `ulysses.py` | all-to-all：seq ↔ head 交换 |
| `async_ulysses.py` | 重叠 QKV 投影与通信 |
| `data.py` | 输入/label 切分，RoPE 切片 |
| `loss.py` | `reduce_sequence_parallel_loss` |
| `comm.py` | 进程组注册 |

**原理：** Pre-attn `gather_seq_scatter_heads` → 本地 full-seq partial-head attn → post `gather_heads_scatter_seq`。

**VLA 默认关闭**；长上下文文本/MoE 模型可启用。

---

## 6.6 注意力 Ops

| 文件 | 函数 | Ulysses |
|------|------|---------|
| `ops/attention.py` | `flash_attention_forward` | ✅ |
| `models/vla/pi0/flex_attention.py` | `flex_attention_forward` | ❌ |
| `models/vla/pi0/qwenvl_in_vla.py` | FA2 内部 | ❌ |

---

## 6.7 group_gemm / fused_moe

**目录：** `ops/group_gemm/`

Triton 实现 batched expert GEMM + scatter/gather，供 MoE 模型 Expert Parallel 使用。

`ops/fused_moe.py` → `FusedMoeExpertFunction`

**VLA PI0 路径不调用**；保留给 Qwen-MoE 等扩展。

---

## 6.8 优化器与学习率

### `optim/optimizer.py`

| 函数 | 说明 |
|------|------|
| `build_optimizer()` | AdamW (fused) 或 AnyPrecisionAdamW (Kahan) |
| depth 参数组 | 名称含 `depth` → **10× 学习率** |

### `optim/lr_scheduler.py`

| style | 说明 |
|-------|------|
| `constant` | 常数 LR |
| `linear` | 线性衰减 |
| `cosine` | 余弦衰减 |
| `two_stage` | 两阶段 |

---

## 6.9 Checkpoint

### 梯度 Checkpointing

`distributed/checkpoint.py` — 自定义 `CheckpointFunction`，修复 FSDP reentrant AllGather 问题。

### 分布式保存

**工厂：** `checkpoint/checkpointer.py` → `build_checkpointer()`

| manager | 实现 |
|---------|------|
| `dcp` | `DistributedCheckpointer` (PyTorch DCP) |
| `bytecheckpoint` | FSDP/DDP/VeScale checkpointer |

**目录结构：**

```
output/checkpoints/global_step_N/
  model/
  optimizer/
  ema/          # 可选
  extra_state/extra_state_rank_0.pt  # scheduler, dataloader, rng, step
```

**HF 导出：**

```python
state_dict = ckpt_to_state_dict(checkpoint_path)
save_model_weights(output_dir / "hf_ckpt", state_dict)
```

**转换脚本：** `scripts/mereg_dcp_to_hf.py`

### Resume

`enable_resume: true` → 自动找最新 checkpoint，恢复 model/optimizer/dataloader state。

---

## 6.10 Activation Offloading

`distributed/offloading.py`：

- `build_activation_offloading_context()` — forward 时 GPU→CPU 卸载激活
- 降低 VRAM，略增训练时间

---

## 6.11 日志与监控

| 后端 | 配置 |
|------|------|
| TensorBoard | 默认 SummaryWriter |
| W&B | 可选 wandb init |
| `loss.jsonl` | 本地 JSON 行日志 |

---

## 6.12 内存与吞吐调优

| 手段 | 配置 |
|------|------|
| 减小 micro_batch | `--train.micro_batch_size 4` |
| 梯度累积 | `--train.gradient_accumulation_steps 16` |
| FP32 action expert | `enable_fp32: true` |
| torch.compile | `use_compile: true` |
| 冻结 ViT | `freeze_vision_encoder: true` |
| 仅训 expert | `train_expert_only: true` |

**A6000×4 示例（Training_Config.md）：** micro=4, accum=16 → global_batch=256，约 47GB/GPU。

---

## 6.13 相关论文/项目

| 名称 | 链接 |
|------|------|
| FSDP | [PyTorch FSDP2](https://pytorch.org/docs/stable/distributed.fsdp.html) |
| VeOmni | [arXiv:2508.02317](https://arxiv.org/abs/2508.02317) |
| Ulysses SP | [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) |
| DCP | [torch.distributed.checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html) |

---

## 6.14 训练命令模板

```bash
# RoboTwin post-training (8 GPU)
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_load20000h.yaml \
    --data.train_path /path/to/mixed_robotwin_5tasks \
    --data.data_name robotwin \
    --data.norm_stats_file assets/norm_stats/robotwin_50.json \
    --train.output_dir output/

# 真实机器人
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/real_load20000h.yaml \
    --data.train_path /path/to/real_data \
    --data.data_name my_robot \
    --data.norm_stats_file assets/norm_stats/my_robot.json \
    --train.output_dir output/
```
