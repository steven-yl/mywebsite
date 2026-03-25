---
title: "DiffusionDriveV2 代码结构图"
subtitle: ""
date: 2026-03-25T00:00:00+08:00
draft: false
authors: [Steven]
description: "DiffusionDrive 代码库的结构总览：安装/环境配置、数据下载、训练评估脚本，以及 `navsim` 核心代码（DiffusionDrive v1/v2、规划系统、仿真与打分）。"
summary: "通过目录树与模块依赖关系，梳理 DiffusionDrive v2（RL/Selection 相关模型及扩散模块）在工程中的位置与调用链路。"
tags: [diffusion/flow, papers, DiffusionDriveV2, todo]
categories: [diffusion/flow, papers]
series: [diffusion/flow系列]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---


```
DiffusionDrive/
│
├── 📄 setup.py                          # 项目安装配置
├── 📄 environment.yml                   # Conda 环境依赖
├── 📄 requirements.txt                  # pip 依赖
├── 📄 kmeans_navsim_traj_20.npy         # K-Means 轨迹聚类锚点
│
├── 📂 docs/                             # 文档
│   ├── DiffusionDrive_Architecture.md   # 架构说明
│   ├── install.md                       # 安装指南
│   └── train_eval.md                    # 训练与评估指南
│
├── 📂 assets/                           # 图片资源
│
├── 📂 gtrs_traj/                        # Ground-truth 参考轨迹
│   └── 16384.npy
│
├── 📂 download/                         # 数据集下载脚本
│   ├── download_maps.sh
│   ├── download_mini.sh
│   ├── download_navtrain.sh
│   ├── download_trainval.sh
│   ├── download_test.sh
│   ├── download_private_test_e2e.sh
│   └── super_download.sh
│
├── 📂 scripts/                          # 运行脚本
│   ├── 📂 training/                     # 训练脚本
│   │   ├── run_ego_mlp_agent_training.sh
│   │   └── run_transfuser_training.sh
│   ├── 📂 evaluation/                   # 评估脚本
│   │   ├── run_metric_caching.sh
│   │   ├── run_transfuser.sh
│   │   ├── run_cv_pdm_score_evaluation.sh
│   │   ├── run_ego_mlp_agent_pdm_score_evaluation.sh
│   │   └── run_human_agent_pdm_score_evaluation.sh
│   └── 📂 submission/                   # 提交脚本
│       ├── run_cv_create_submission_pickle.sh
│       └── run_merge_submission_pickles.sh
│
├── 📂 tutorial/                         # 教程
│   └── tutorial_visualization.ipynb
│
└── 📂 navsim/                           # ========== 核心代码 ==========
    │
    ├── 📂 agents/                       # ===== 智能体 (Agent) =====
    │   ├── abstract_agent.py            # 抽象基类 Agent
    │   ├── constant_velocity_agent.py   # 恒速 Baseline Agent
    │   ├── ego_status_mlp_agent.py      # MLP Baseline Agent
    │   ├── human_agent.py               # 人类驾驶 Agent
    │   │
    │   ├── 📂 transfuser/              # ---- TransFuser Agent ----
    │   │   ├── transfuser_agent.py      # Agent 入口
    │   │   ├── transfuser_config.py     # 配置
    │   │   ├── transfuser_backbone.py   # 骨干网络 (ResNet + Transformer)
    │   │   ├── transfuser_model.py      # 完整模型
    │   │   ├── transfuser_features.py   # 特征提取与目标构建
    │   │   ├── transfuser_loss.py       # 损失函数
    │   │   └── transfuser_callback.py   # 训练回调
    │   │
    │   ├── 📂 diffusiondrive/          # ---- DiffusionDrive v1 ----
    │   │   ├── transfuser_agent.py      # Agent 入口
    │   │   ├── transfuser_config.py     # 配置
    │   │   ├── transfuser_backbone.py   # 骨干网络
    │   │   ├── transfuser_model_v2.py   # 扩散模型 (核心)
    │   │   ├── transfuser_features.py   # 特征提取与目标构建
    │   │   ├── transfuser_loss.py       # 损失函数
    │   │   ├── transfuser_callback.py   # 训练回调
    │   │   └── 📂 modules/             # 扩散模块
    │   │       ├── blocks.py            # 基础网络块 (SinusoidalPosEmb, ResBlock 等)
    │   │       ├── conditional_unet1d.py# 条件 UNet1D 去噪网络
    │   │       ├── multimodal_loss.py   # 多模态损失
    │   │       └── scheduler.py         # 扩散调度器 (DDPM/DDIM)
    │   │
    │   └── 📂 diffusiondrivev2/        # ---- DiffusionDrive v2 ----
    │       ├── diffusiondrivev2_rl_agent.py   # RL Agent 入口
    │       ├── diffusiondrivev2_rl_config.py  # RL 配置
    │       ├── diffusiondrivev2_model_rl.py   # RL 模型
    │       ├── diffusiondrivev2_sel_agent.py  # Selection Agent 入口
    │       ├── diffusiondrivev2_sel_config.py # Selection 配置
    │       ├── diffusiondrivev2_model_sel.py  # Selection 模型
    │       ├── transfuser_backbone.py   # 骨干网络
    │       ├── transfuser_config.py     # 基础配置
    │       ├── transfuser_features.py   # 特征提取与目标构建
    │       ├── transfuser_loss.py       # 损失函数
    │       ├── transfuser_callback.py   # 训练回调
    │       └── 📂 modules/             # 扩散模块 (同 v1)
    │           ├── blocks.py
    │           ├── conditional_unet1d.py
    │           ├── multimodal_loss.py
    │           └── scheduler.py
    │
    ├── 📂 common/                       # ===== 公共数据结构 =====
    │   ├── dataclasses.py               # 核心数据类 (Scene, Frame, Trajectory 等)
    │   ├── dataloader.py                # 数据加载器
    │   └── enums.py                     # 枚举定义
    │
    ├── 📂 evaluate/                     # ===== 评估 =====
    │   └── pdm_score.py                 # PDM 评分计算
    │
    ├── 📂 visualization/               # ===== 可视化 =====
    │   ├── bev.py                       # 鸟瞰图 (BEV) 可视化
    │   ├── camera.py                    # 相机视图可视化
    │   ├── lidar.py                     # LiDAR 点云可视化
    │   ├── config.py                    # 可视化配置
    │   └── plots.py                     # 绘图工具
    │
    └── 📂 planning/                     # ===== 规划系统 =====
        │
        ├── 📂 training/                 # ---- 训练框架 ----
        │   ├── abstract_feature_target_builder.py  # 特征/目标构建抽象类
        │   ├── agent_lightning_module.py            # PyTorch Lightning 训练模块
        │   ├── dataset.py                           # 数据集定义
        │   └── 📂 callbacks/
        │       └── time_logging_callback.py         # 训练时间日志回调
        │
        ├── 📂 metric_caching/           # ---- 指标缓存 ----
        │   ├── caching.py               # 缓存逻辑
        │   ├── metric_cache.py          # 缓存数据结构
        │   ├── metric_cache_processor.py# 缓存处理器
        │   └── metric_caching_utils.py  # 缓存工具函数
        │
        ├── 📂 scenario_builder/         # ---- 场景构建 ----
        │   ├── navsim_scenario.py       # NavSim 场景定义
        │   └── navsim_scenario_utils.py # 场景工具函数
        │
        ├── 📂 simulation/              # ---- PDM 仿真规划器 ----
        │   └── 📂 planner/pdm_planner/
        │       ├── abstract_pdm_planner.py          # PDM 规划器抽象类
        │       ├── abstract_pdm_closed_planner.py   # 闭环 PDM 规划器抽象类
        │       ├── pdm_closed_planner.py            # 闭环 PDM 规划器实现
        │       ├── 📂 observation/                  # 观测处理
        │       │   ├── pdm_observation.py           # 观测数据
        │       │   ├── pdm_occupancy_map.py         # 占用栅格地图
        │       │   └── pdm_object_manager.py        # 目标物管理
        │       ├── 📂 proposal/                     # 轨迹提案生成
        │       │   ├── pdm_proposal.py              # 提案定义
        │       │   ├── pdm_generator.py             # 提案生成器
        │       │   └── batch_idm_policy.py          # 批量 IDM 策略
        │       ├── 📂 scoring/                      # 轨迹评分
        │       │   ├── pdm_scorer.py                # 评分器
        │       │   ├── pdm_scorer_utils.py          # 评分工具
        │       │   └── pdm_comfort_metrics.py       # 舒适度指标
        │       ├── 📂 simulation/                   # 运动学仿真
        │       │   ├── pdm_simulator.py             # 仿真器
        │       │   ├── batch_kinematic_bicycle.py   # 批量自行车运动学模型
        │       │   ├── batch_lqr.py                 # 批量 LQR 控制器
        │       │   └── batch_lqr_utils.py           # LQR 工具
        │       └── 📂 utils/                        # 工具函数
        │           ├── pdm_path.py                  # 路径处理
        │           ├── pdm_geometry_utils.py         # 几何工具
        │           ├── pdm_array_representation.py  # 数组表示
        │           ├── pdm_emergency_brake.py       # 紧急制动
        │           ├── pdm_enums.py                 # 枚举
        │           ├── route_utils.py               # 路线工具
        │           └── 📂 graph_search/             # 图搜索
        │
        ├── 📂 script/                   # ---- 运行入口脚本 ----
        │   ├── run_training.py                      # 训练入口
        │   ├── run_pdm_score.py                     # PDM 评分入口
        │   ├── run_pdm_score_fast.py                # 快速 PDM 评分
        │   ├── run_pdm_score_from_submission.py     # 从提交文件评分
        │   ├── run_metric_caching.py                # 指标缓存入口
        │   ├── run_dataset_caching.py               # 数据集缓存入口
        │   ├── run_create_submission_pickle.py      # 创建提交文件
        │   ├── run_merge_submission_pickles.py      # 合并提交文件
        │   ├── utils.py                             # 脚本工具
        │   ├── 📂 builders/             # 构建器
        │   │   ├── observation_builder.py           # 观测构建器
        │   │   ├── planner_builder.py               # 规划器构建器
        │   │   ├── simulation_builder.py            # 仿真构建器
        │   │   └── worker_pool_builder.py           # 工作池构建器
        │   └── 📂 config/              # Hydra 配置文件
        │       ├── 📂 common/
        │       │   ├── default_common.yaml
        │       │   ├── default_evaluation.yaml
        │       │   ├── 📂 agent/        # Agent 配置 (7种)
        │       │   ├── 📂 train_test_split/  # 数据集划分配置
        │       │   └── 📂 worker/       # 并行 Worker 配置
        │       ├── 📂 training/         # 训练配置
        │       ├── 📂 metric_caching/   # 指标缓存配置
        │       └── 📂 pdm_scoring/      # PDM 评分配置
        │
        └── 📂 utils/                    # ---- 工具 ----
            └── 📂 multithreading/
                └── worker_ray_no_torch.py  # Ray 分布式 Worker
```

## 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        scripts/ (Shell)                         │
│              训练 / 评估 / 提交 启动脚本                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 调用
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  navsim/planning/script/                         │
│         run_training.py  |  run_pdm_score.py  |  ...            │
│                    (Hydra 配置驱动)                               │
└──────┬──────────────┬──────────────┬───────────────┬────────────┘
       │              │              │               │
       ▼              ▼              ▼               ▼
┌────────────┐ ┌────────────┐ ┌───────────┐ ┌──────────────┐
│  training/ │ │  builders/ │ │ metric_   │ │ scenario_    │
│            │ │            │ │ caching/  │ │ builder/     │
│ Lightning  │ │ 构建观测/  │ │           │ │              │
│ 训练模块   │ │ 规划器/仿真│ │ 指标缓存  │ │ 场景构建     │
└─────┬──────┘ └─────┬──────┘ └─────┬─────┘ └──────┬───────┘
      │              │              │               │
      ▼              ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      navsim/agents/                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              abstract_agent.py (抽象基类)                 │   │
│  └──────┬──────────┬──────────────┬────────────┬────────────┘   │
│         │          │              │            │                 │
│         ▼          ▼              ▼            ▼                 │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐      │
│  │TransFuser│ │Diffusion │ │Diffusion  │ │ Baseline     │      │
│  │          │ │Drive v1  │ │Drive v2   │ │ Agents       │      │
│  │ model    │ │          │ │ (RL/Sel)  │ │ (CV/MLP/     │      │
│  │ backbone │ │ model_v2 │ │ model_rl  │ │  Human)      │      │
│  │ features │ │ backbone │ │ model_sel │ │              │      │
│  │ loss     │ │ features │ │ backbone  │ │              │      │
│  │ callback │ │ loss     │ │ features  │ │              │      │
│  └──────────┘ │ callback │ │ loss      │ └──────────────┘      │
│               │ modules/ │ │ callback  │                        │
│               │  ├ UNet1D│ │ modules/  │                        │
│               │  ├ blocks│ │  ├ UNet1D │                        │
│               │  ├ sched.│ │  ├ blocks │                        │
│               │  └ mm_lss│ │  ├ sched. │                        │
│               └──────────┘ │  └ mm_lss │                        │
│                            └───────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      navsim/common/                              │
│         dataclasses.py  |  dataloader.py  |  enums.py           │
│                   (公共数据结构与加载)                             │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              navsim/planning/simulation/                         │
│                    PDM Planner 仿真系统                           │
│  ┌────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐      │
│  │ observation │ │ proposal │ │ scoring │ │ simulation   │      │
│  │ 观测处理    │ │ 轨迹提案 │ │ 轨迹评分│ │ 运动学仿真   │      │
│  └────────────┘ └──────────┘ └─────────┘ └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    navsim/evaluate/                              │
│                     pdm_score.py                                 │
│                   (PDM 评分指标计算)                              │
└─────────────────────────────────────────────────────────────────┘
```
