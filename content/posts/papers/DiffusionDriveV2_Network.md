---
title: "DiffusionDriveV2 网络结构图"
subtitle: ""
date: 2026-03-25T00:00:00+08:00
draft: false
authors: [Steven]
description: "DiffusionDrive V2（V2TransfuserModel）网络架构的结构化说明：包含 TransfuserBackbone、TrajectoryHead（扩散生成与级联评分）以及解码层与端到端数据流。"
summary: "本文以结构图的方式梳理 DiffusionDrive V2 的关键模块与连接关系：双流骨干特征融合、Transformer Decoder×3、扩散式 TrajectoryHead 的截断生成以及粗筛-精筛评分流水线。"
tags: [diffusion/flow, papers, DiffusionDriveV2, todo]
categories: [diffusion/flow, papers]
series: [diffusion/flow系列]
weight: 6
series_weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---


## 整体架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DiffusionDrive V2 (V2TransfuserModel)                     │
│                                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                               │
│  │ Camera Image  │  │  LiDAR BEV   │  │ Ego Status   │                               │
│  │ (B,3,256,1024)│  │ (B,1,256,256)│  │ (B, 8)       │                               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                               │
│         │                 │                 │                                        │
│         ▼                 ▼                 │                                        │
│  ┌──────────────────────────────────┐       │                                        │
│  │     TransfuserBackbone           │       │                                        │
│  │  (双流 ResNet34 + GPT Fusion)    │       │                                        │
│  └──────┬──────────────┬────────────┘       │                                        │
│         │              │                    │                                        │
│    bev_feature    bev_feature               │                                        │
│    _upscale       (fused)                   │                                        │
│         │              │                    │                                        │
│         │         ┌────▼─────┐         ┌────▼──────────┐                             │
│         │         │BEV Down  │         │Status Encoding│                             │
│         │         │Conv 1x1  │         │Linear(8→256)  │                             │
│         │         │512→256   │         └────┬──────────┘                             │
│         │         └────┬─────┘              │                                        │
│         │              │                    │                                        │
│         │         ┌────▼────────────────────▼───┐                                    │
│         │         │  KeyVal = [BEV_8x8; Status] │                                    │
│         │         │  + Positional Embedding      │                                    │
│         │         │  (B, 65, 256)                │                                    │
│         │         └────────────┬─────────────────┘                                   │
│         │                    │                                                       │
│         │              ┌─────▼──────────────────────┐                                │
│         │              │  Transformer Decoder (×3)   │                                │
│         │              │  Query: [traj_q; agent_q]   │                                │
│         │              │  KeyVal: [BEV; Status]      │                                │
│         │              │  d_model=256, nhead=8       │                                │
│         │              └──┬──────────────────┬───────┘                                │
│         │                 │                  │                                        │
│         │           trajectory_q        agents_q                                     │
│         │            (B,1,256)          (B,30,256)                                    │
│         │                 │                  │                                        │
│         │                 │            ┌─────▼──────┐                                 │
│         │                 │            │ AgentHead  │                                 │
│         │                 │            │ MLP→states │                                 │
│         │                 │            │ MLP→labels │                                 │
│         │                 │            └────────────┘                                 │
│         │                 │            agent_states (B,30,5)                          │
│         │                 │            agent_labels (B,30)                            │
│    ┌────▼─────┐           │                                                          │
│    │BEV Sem.  │           │                                                          │
│    │Head      │     ┌─────▼──────────────────────────────────────────────┐            │
│    │Conv→7cls │     │              TrajectoryHead                        │            │
│    └──────────┘     │  (扩散生成 + 粗筛 + 精筛 三阶段流水线)             │            │
│    bev_semantic     │                                                    │            │
│    (B,7,128,256)    │  详见下方 TrajectoryHead 展开图                    │            │
│                     └────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 骨干网络 TransfuserBackbone 详细结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TransfuserBackbone                             │
│                                                                     │
│   Camera Image                          LiDAR BEV                   │
│   (B,3,256,1024)                        (B,1,256,256)               │
│        │                                     │                      │
│        ▼                                     ▼                      │
│   ┌──────────┐                         ┌──────────┐                 │
│   │ ResNet34  │                         │ ResNet34  │                │
│   │ (Image)   │                         │ (LiDAR)   │                │
│   │ timm      │                         │ in_ch=1   │                │
│   └──────────┘                         └──────────┘                 │
│        │                                     │                      │
│   4 个 Stage, 每个 Stage 后进行双流融合:                             │
│                                                                     │
│   Stage i:  image_feat ──┐    ┌── lidar_feat                       │
│                          ▼    ▼                                     │
│                    ┌──────────────┐                                  │
│                    │  Channel     │                                  │
│                    │  Alignment   │  lidar_ch → img_ch (Conv1x1)    │
│                    │              │  img_ch → lidar_ch (Conv1x1)    │
│                    └──────┬───────┘                                  │
│                           ▼                                         │
│                    ┌──────────────┐                                  │
│                    │  GPT Fusion  │  (Self-Attention Transformer)    │
│                    │  Block ×2    │  concat [img_tokens; lidar_tokens]│
│                    │  + PosEmb    │  → Self-Attn → Split back        │
│                    └──────┬───────┘                                  │
│                     ┌─────┴─────┐                                   │
│                     ▼           ▼                                   │
│               image_feat   lidar_feat  (融合后, 进入下一 Stage)      │
│                                                                     │
│   最终输出:                                                          │
│   ┌─────────────────────────────────────────────────┐               │
│   │ lidar_feat (Stage4) ──→ FPN Top-Down            │               │
│   │   c5_conv(1x1) → up_conv5(3x3) → upsample      │               │
│   │   c4_conv(1x1) → up_conv4(3x3) → upsample      │               │
│   │                                                  │               │
│   │ → bev_feature_upscale (B, 64, H, W)  用于BEV语义│               │
│   │ → bev_feature (B, 512, 8, 8)         用于Decoder │               │
│   └─────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## TrajectoryHead 三阶段流水线

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              TrajectoryHead                                          │
│                                                                                      │
│  输入: ego_query(B,1,256), agents_query(B,30,256), bev_feature, status_encoding      │
│                                                                                      │
│  ╔══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  阶段 1: 截断扩散生成 (Truncated Diffusion)                                    ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                      │
│   Plan Anchors (20个K-Means聚类锚点)                                                │
│   (20, 8, 2) × num_groups 组                                                        │
│        │                                                                             │
│        ▼                                                                             │
│   ┌─────────────────────────────────┐                                                │
│   │ 加截断噪声 (t=8, 非完整1000步)  │                                                │
│   │ DDIM add_noise(anchor, noise, 8)│                                                │
│   └──────────────┬──────────────────┘                                                │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐        │
│   │  DDIM 去噪循环 (仅 2 步, 非完整 1000 步)                                │        │
│   │                                                                          │        │
│   │  for step in [10, 0]:  ← 截断时间步                                     │        │
│   │    ┌─────────────────────────────────────────────────────────────────┐   │        │
│   │    │ ① Sinusoidal Position Embedding (轨迹点 → 特征)                │   │        │
│   │    │    noisy_traj → sin/cos embed → plan_anchor_encoder → (B,G,256)│   │        │
│   │    │                                                                 │   │        │
│   │    │ ② Time Embedding                                               │   │        │
│   │    │    timestep → SinPosEmb → MLP(256→1024→256) → time_embed       │   │        │
│   │    │                                                                 │   │        │
│   │    │ ③ CustomTransformerDecoder (diff_decoder, 1层)                  │   │        │
│   │    │    详见下方 Diffusion Decoder Layer 结构                        │   │        │
│   │    │    → poses_reg (B, G*20, 8, 3)  [x, y, heading]               │   │        │
│   │    │                                                                 │   │        │
│   │    │ ④ DDIM Step: 用预测的 x_start 更新 diffusion_output            │   │        │
│   │    └─────────────────────────────────────────────────────────────────┘   │        │
│   └──────────────────────────────────────────────────────────────────────────┘        │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────┐                                               │
│   │ 多样性增强: add_mul_noise        │                                               │
│   │ 对去噪结果加微小高斯扰动          │                                               │
│   └──────────────┬───────────────────┘                                               │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────┐                                               │
│   │ Bezier 曲线拟合 → (x, y, yaw)   │                                               │
│   │ diffusion_output (B, G_all, 8, 3)│                                               │
│   └──────────────┬───────────────────┘                                               │
│                  │                                                                   │
│                  │  (可选) 拼接 Vocabulary 轨迹库中的候选                              │
│                  │                                                                   │
│  ╔══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  阶段 2: 粗筛评分 (Coarse Scorer)                                               ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════╝  │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐        │
│   │ Scorer 特征提取 (_get_scorer_inputs)                                     │        │
│   │   轨迹 xy → sin/cos pos embed (dim=64)                                  │        │
│   │   轨迹 heading → sin/cos 1D embed (dim=32)                              │        │
│   │   concat → plan_anchor_scorer_encoder → traj_feature (B, G_all, 512)    │        │
│   └──────────────┬───────────────────────────────────────────────────────────┘        │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐        │
│   │ ScorerTransformerDecoder (scorer_decoder, 1层)                           │        │
│   │   详见下方 Scorer Decoder Layer 结构                                     │        │
│   │   → traj_feature (B, G_all, 512)                                        │        │
│   └──────────────┬───────────────────────────────────────────────────────────┘        │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐        │
│   │ 5 个评分头 (MLP → sigmoid):                                              │        │
│   │   NC_head  → No Collision 分数                                           │        │
│   │   EP_head  → Ego Progress 分数                                           │        │
│   │   DAC_head → Drivable Area Compliance 分数                               │        │
│   │   TTC_head → Time-to-Collision 分数                                      │        │
│   │   C_head   → Comfort 分数                                                │        │
│   │                                                                          │        │
│   │ 综合评分: σ(NC)·σ(DAC)·(5·σ(TTC)+5·σ(EP)+2·σ(C))/12                    │        │
│   └──────────────┬───────────────────────────────────────────────────────────┘        │
│                  │                                                                   │
│            Top-K 筛选 (K=32)                                                         │
│                  │                                                                   │
│  ╔══════════════════════════════════════════════════════════════════════════════════╗  │
│  ║  阶段 3: 精筛评分 (Fine Scorer)                                                 ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════╝  │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐        │
│   │ ScorerTransformerDecoder (fine_scorer_decoder, 3层, 更深)                │        │
│   │   结构同粗筛, 但堆叠 3 层 ScorerTransformerDecoderLayer                 │        │
│   │   → 多层输出, 每层独立评分                                               │        │
│   └──────────────┬───────────────────────────────────────────────────────────┘        │
│                  │                                                                   │
│                  ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────────────────────┐        │
│   │ 5 个精筛评分头 (fine_NC/EP/DAC/TTC/C_head)                               │        │
│   │ 同样的综合评分公式                                                        │        │
│   │ 每层选出 best trajectory → 最终输出                                       │        │
│   └──────────────────────────────────────────────────────────────────────────┘        │
│                  │                                                                   │
│                  ▼                                                                   │
│          最终轨迹 trajectory (B, 8, 3)  [x, y, heading] × 8 个未来时间步             │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## Diffusion Decoder Layer (CustomTransformerDecoderLayer) 详细结构

```
┌──────────────────────────────────────────────────────────────────────────┐
│              CustomTransformerDecoderLayer (扩散去噪解码层)               │
│                                                                          │
│  输入: traj_feature (B, G*20, 256)                                       │
│        noisy_traj_points (B, G*20, 8, 2)                                │
│        bev_feature, agents_query, ego_query, time_embed                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ① GridSampleCrossBEVAttention                                  │      │
│  │    轨迹点坐标 → 归一化到 [-1,1]                                │      │
│  │    → grid_sample 从 BEV 特征图采样                             │      │
│  │    → 加权求和 (attention_weights softmax)                      │      │
│  │    → output_proj + residual                                    │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ② Cross-Attention with Agents                                  │      │
│  │    Q=traj_feature, K=V=agents_query                            │      │
│  │    MultiheadAttention(256, 8heads) + LayerNorm                 │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ③ Cross-Attention with Ego Query                               │      │
│  │    Q=traj_feature, K=V=ego_query                               │      │
│  │    MultiheadAttention(256, 8heads) + LayerNorm                 │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ④ FFN + LayerNorm                                              │      │
│  │    Linear(256→1024) → ReLU → Linear(1024→256) + LayerNorm     │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ⑤ Time Modulation (ModulationLayer)                            │      │
│  │    time_embed → MLP → (scale, shift)                           │      │
│  │    traj_feature = traj_feature * (1 + scale) + shift           │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ⑥ DiffMotionPlanningRefinementModule (任务解码)                │      │
│  │    plan_cls_branch: MLP → (B, 20) 分类分数                    │      │
│  │    plan_reg_branch: MLP → (B, 20, 8, 3) 轨迹回归              │      │
│  │    poses_reg[xy] += noisy_traj_points (残差预测)               │      │
│  │    poses_reg[heading] = tanh(·) × π                            │      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  输出: poses_reg (B, G*20, 8, 3), poses_cls (B, G, 20)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Scorer Decoder Layer (ScorerTransformerDecoderLayer) 详细结构

```
┌──────────────────────────────────────────────────────────────────────────┐
│            ScorerTransformerDecoderLayer (评分解码层, d=512)              │
│                                                                          │
│  输入: traj_feature (B, G, 512)                                          │
│        noisy_traj_points (B, G, 8, 2)                                   │
│        bev_feature, agents_query(256), ego_query(256)                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ① GridSampleCrossBEVAttentionScorer                            │      │
│  │    同 Diffusion 版本, 但 embed_dims=512                        │      │
│  │    轨迹点 → grid_sample BEV → 加权聚合 + residual              │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ② Cross-Attention with Agents                                  │      │
│  │    agents_query → Linear(256→512) 升维                         │      │
│  │    Q=traj, K=V=agents  MHA(512, 16heads) + LayerNorm           │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ③ Self-Attention (轨迹间交互)                                  │      │
│  │    MHA(512, 16heads) + LayerNorm                               │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ④ Cross-Attention with Ego Query                               │      │
│  │    ego_query → Linear(256→512) 升维                            │      │
│  │    Q=traj, K=V=ego  MHA(512, 16heads) + LayerNorm              │      │
│  └────────────────────┬───────────────────────────────────────────┘      │
│                       ▼                                                  │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │ ⑤ FFN + LayerNorm                                              │      │
│  │    Linear(512→2048) → ReLU → Linear(2048→512) + LayerNorm     │      │
│  └────────────────────────────────────────────────────────────────┘      │
│                                                                          │
│  输出: traj_feature (B, G, 512)                                          │
│  注意: Scorer 层没有 Time Modulation, 也没有轨迹回归                     │
│        仅提取特征, 由外部的 5 个评分头完成打分                            │
└──────────────────────────────────────────────────────────────────────────┘
```

## 端到端数据流总结

```
Camera(B,3,256,1024) ─┐
                      ├─→ TransfuserBackbone ─→ BEV Features + Fused Features
LiDAR(B,1,256,256) ───┘         │
                                │
                    ┌───────────┴───────────────────────────────┐
                    │                                           │
                    ▼                                           ▼
            BEV Semantic Head                     Transformer Decoder (×3)
            → 7类语义分割图                        Query=[traj_q; agent_q×30]
                                                  KeyVal=[BEV_8×8; status]
                                                        │
                                              ┌─────────┴─────────┐
                                              │                   │
                                              ▼                   ▼
                                        TrajectoryHead       AgentHead
                                              │              → 30个Agent
                                              │                bbox+label
                                              │
                              ┌────────────────┼────────────────┐
                              │                │                │
                              ▼                ▼                ▼
                     Stage 1: 扩散     Stage 2: 粗筛    Stage 3: 精筛
                     截断DDIM 2步      Scorer(1层)      Scorer(3层)
                     20锚点×G组        5维PDM评分        Top-32精排
                     → G*20条候选      → Top-K筛选       → 最优轨迹
                                                                │
                                                                ▼
                                                    最终输出: trajectory
                                                    (B, 8, 3) [x,y,yaw]
```

## 关键设计亮点

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. 截断扩散 (Truncated Diffusion)                                   │
│    - 不从纯噪声开始, 而是从 K-Means 锚点加少量噪声 (t=8) 开始      │
│    - 仅需 2 步 DDIM 去噪, 大幅降低推理延迟                          │
│                                                                     │
│ 2. 锚点初始化 (Anchor-based Initialization)                         │
│    - 20 个 K-Means 聚类轨迹锚点覆盖常见驾驶模式                    │
│    - 多组 (num_groups) 并行生成, 增加多样性                         │
│                                                                     │
│ 3. 粗筛-精筛级联评分 (Coarse-to-Fine Scoring)                       │
│    - 粗筛: 1层 Scorer, 快速过滤大量候选 → Top-32                    │
│    - 精筛: 3层 Scorer, 对少量候选精细评分                           │
│    - 5 维 PDM 子分数: NC, EP, DAC, TTC, Comfort                    │
│                                                                     │
│ 4. BEV 轨迹采样注意力 (GridSampleCrossBEVAttention)                 │
│    - 用轨迹点坐标直接在 BEV 特征图上 grid_sample                   │
│    - 让每条轨迹"看到"其经过区域的场景信息                           │
│                                                                     │
│ 5. 时间步调制 (Time Modulation)                                     │
│    - 扩散时间步通过 scale-shift 调制轨迹特征                        │
│    - 仅在扩散解码器中使用, 评分器不需要                             │
└─────────────────────────────────────────────────────────────────────┘
```
