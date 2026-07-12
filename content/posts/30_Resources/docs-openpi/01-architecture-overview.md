---
title: "01 架构总览"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "openpi 整体认知框架：问题定义、子系统构成、协作方式与设计取舍。"
summary: "openpi 架构总览：三类模型、五大子系统与数据流。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 01 架构总览

> 本章给出 openpi 的整体认知框架：它解决什么问题、由哪些部分构成、各部分如何协作、关键设计取舍与适用场景。读完本章你应当能把后续任意一章「挂」到全局图景中的正确位置。

---

## 1.1 openpi 是什么，解决什么问题

openpi 是一套面向机器人操控（manipulation）的**视觉-语言-动作（VLA）模型**开源实现。它要解决的核心问题是：

> 给定机器人当前的**多路摄像头图像** + **本体状态（关节角/夹爪等）** + **自然语言指令**，预测出未来一小段时间的**连续动作序列**（action chunk），驱动机器人完成任务。

它提供了在 **10000+ 小时真实机器人数据**上预训练好的基础模型（base model），以及在 ALOHA、DROID、LIBERO 等平台上微调好的专家模型。用户既可以**开箱即用地推理**，也可以**用自己的数据微调**。

本仓库包含三类模型，它们是理解整个项目的主线：

| 模型          | 范式                  | 动作表示               | 特点        | 适用场景           |
| ----------- | ------------------- | ------------------ | --------- | -------------- |
| **π₀**      | 流匹配（flow matching）  | 连续向量，去噪 ODE 积分     | 推理快、动作平滑  | 高频连续控制         |
| **π₀-FAST** | 自回归（autoregressive） | 离散 token（FAST 分词器） | 语言跟随好、可变长 | 强语言条件、对推理延迟不敏感 |
| **π₀.₅**    | 流匹配 + 知识隔离          | 连续向量，状态进语言流        | 开放世界泛化更好  | 新场景/新指令的零样本迁移  |

> 注意：在本仓库中，π₀.₅ 只支持**流匹配头**（flow matching head）进行训练与推理。

---

## 1.2 知识结构：五大子系统

整个代码库可以划分为五个子系统，外加一个独立的轻量客户端包。理解它们的职责边界，是理解全局的关键。

```
                         ┌──────────────────────────────────────────────┐
                         │                  ① 模型层                      │
                         │  models/  models_pytorch/                      │
                         │  π0 / π0.5 / π0-FAST + Gemma + SigLIP + LoRA   │
                         │  职责：定义网络结构、compute_loss、sample_actions │
                         └───────────────▲───────────────┬──────────────┘
                                         │ 提供 model     │ 被调用
                ┌────────────────────────┴───┐       ┌───▼─────────────────────┐
                │      ② 训练子系统           │       │     ③ 推理/服务子系统     │
                │  training/  scripts/train*  │       │  policies/  serving/     │
                │  配置/数据加载/优化器/检查点 │       │  Policy 封装 + WebSocket │
                └────────────▲───────────────┘       └───▲──────────────────────┘
                             │ 复用                       │ 复用
                ┌────────────┴───────────────────────────┴──────────────┐
                │              ④ 数据管线（Transform 体系）                │
                │  transforms.py  shared/normalize.py  机器人适配 policy   │
                │  职责：原始数据 ⇄ 模型输入/输出 的双向变换 + 归一化       │
                └────────────────────────▲──────────────────────────────┘
                                         │ 复用
                         ┌───────────────┴──────────────┐
                         │          ⑤ 通用基础设施        │
                         │  shared/ download/image/typing │
                         └───────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │  ⑥ 客户端运行时（独立包 openpi-client）                                │
   │  Runtime + Agent + Environment + Subscriber + WebSocket Client       │
   │  职责：在机器人侧采集观测→请求服务端→执行动作的闭环                     │
   └─────────────────────────────────────────────────────────────────────┘
```

### 各子系统职责与关联

1. **模型层（`models/`、`models_pytorch/`）**——定义"大脑"。
   - 提供两套实现：**JAX/Flax NNX**（训练与原始权重）与 **PyTorch**（新支持，便于 PyTorch 生态）。
   - 每个模型实现两个核心方法：`compute_loss`（训练）与 `sample_actions`（推理）。
   - 详见 [02](02-models-flow-matching.md)、[03](03-models-pi0-fast.md)、[04](04-backbone-tokenizers.md)、[09](09-pytorch-implementation.md)。

2. **训练子系统（`training/`、`scripts/train*.py`）**——把模型"练出来"。
   - 配置注册表（`config.py`）是中枢：一个 `TrainConfig` 完整描述了"用什么模型、什么数据、什么超参"。
   - 提供 JAX（FSDP + EMA + Orbax）与 PyTorch（DDP + safetensors）两条训练路径。
   - 详见 [06](06-training-system.md)。

3. **推理/服务子系统（`policies/`、`serving/`）**——把模型"用起来"。
   - `Policy` 把"模型 + 输入/输出变换"封装成 `infer(obs) -> actions` 的简单接口。
   - `WebsocketPolicyServer` 把 `Policy` 暴露成网络服务，支持远程推理。
   - 详见 [07](07-inference-policy-serving.md)。

4. **数据管线（`transforms.py`、机器人 policy 文件、`normalize.py`）**——训练与推理**共用**的数据"翻译层"。
   - 把不同机器人的原始数据格式翻译成模型期望的统一格式，再翻译回去。
   - 归一化（Normalize/Unnormalize）保证训练与推理使用相同统计量。
   - 详见 [05](05-data-pipeline.md)。

5. **通用基础设施（`shared/`）**——下载缓存、图像处理、运行时类型检查等横切能力。

6. **客户端运行时（`packages/openpi-client/`）**——刻意做成**零重依赖**的独立包，部署在机器人侧（可能是算力很弱的工控机），通过 WebSocket 调用远端 GPU 上的策略服务。详见 [08](08-client-runtime.md)。

---

## 1.3 端到端数据流（训练 vs 推理）

理解 openpi 最有效的方式，是跟踪数据从"原始格式"到"动作"的完整流动。两条路径**共享同一套数据变换**，这是设计上的关键一致性保证。

### 训练时（offline，数据来自数据集）

```
LeRobot/RLDS 数据集样本（原始 dict）
   │  repack_transforms.inputs        # 改键名，统一布局
   ▼
   │  data_transforms.inputs          # 机器人特定：AlohaInputs/LiberoInputs/...
   ▼
   │  Normalize                       # 用 norm_stats 归一化 state/actions
   ▼
   │  model_transforms.inputs         # ResizeImages + TokenizePrompt + Pad
   ▼
Observation + Actions  ──►  model.compute_loss(rng, obs, actions)  ──►  标量 loss
```

### 推理时（online，数据来自机器人）

```
机器人观测（原始 dict）
   │  repack.inputs → InjectDefaultPrompt → data_transforms.inputs
   ▼  → Normalize → model_transforms.inputs（同训练）
Observation
   │
   ▼  model.sample_actions(rng, obs)         # π0: 去噪ODE；FAST: 自回归解码
   │
模型原始输出（归一化空间的动作 / 动作 token）
   │  model_transforms.outputs               # FAST: ExtractFASTActions
   ▼  Unnormalize                            # 反归一化回物理空间
   │  data_transforms.outputs                # AlohaOutputs/LiberoOutputs（裁剪维度）
   ▼  repack.outputs
机器人可执行的动作序列
```

**核心洞察**：输入变换序列与输出变换序列是**镜像对称**的（输出按相反顺序"撤销"输入），这保证了模型始终工作在统一的归一化空间，而机器人始终看到自己的原生空间。`Group.push()` 方法刻意实现了这种非对称追加（inputs 追加到末尾、outputs 追加到开头），见 [05 章](05-data-pipeline.md)。

---

## 1.4 模型内部的统一抽象

三类模型尽管范式不同，但共享同一套抽象（`models/model.py`）：

- **`Observation`**（`@struct.dataclass`）：模型输入的统一容器。字段包括 `images`（多路图像，[-1,1] 浮点）、`image_masks`、`state`、`tokenized_prompt`、`tokenized_prompt_mask`，以及 FAST 专用的 `token_ar_mask`、`token_loss_mask`。
- **`BaseModelConfig`**（抽象基类）：所有模型配置共享 `action_dim`、`action_horizon`、`max_token_len`，并要求实现 `create`（建模型）、`inputs_spec`（输入规格）。
- **`BaseModel`**（抽象基类，`nnx.Module`）：要求实现 `compute_loss` 与 `sample_actions`。

这套抽象让训练循环、数据加载、策略封装都能以**模型无关**的方式编写——这是项目可扩展性的基础。

```python
# 三类模型都遵循同一接口（简化）
class BaseModel(nnx.Module, abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(self, rng, observation, actions, *, train=False) -> Array: ...

    @abc.abstractmethod
    def sample_actions(self, rng, observation, **kwargs) -> Actions: ...
```

---

## 1.5 双主干 + 混合专家架构（贯穿三类模型）

三类模型的骨架都是同一个思想：**PaliGemma 视觉-语言主干** + **动作专家（Action Expert）**，二者构成一个"混合专家 Transformer"。

```
       图像 ──► SigLIP 视觉塔 ──┐
                               ├──► [前缀 tokens]  ──┐
       语言 ──► Gemma 嵌入  ────┘                    │
                                          ┌──────────▼───────────────────────┐
                                          │  混合专家 Transformer（逐层）       │
       状态 ──► state_proj ─────┐         │  PaliGemma(2B) 与 Action(300M)     │
                               ├──► [后缀]│  共享 self-attention，但各自独立    │
   噪声动作+时间 ──► action_in ─┘         │  的 QKV/MLP/Norm 权重               │
                                          └──────────┬───────────────────────┘
                                                     │ 后缀输出
                                          action_out_proj ──► 预测速度场 v_t
```

- **前缀（prefix）**：图像 + 语言 token，使用**双向注意力**（彼此可见）。
- **后缀（suffix）**：状态 + 带噪动作 token（π₀），使用受控注意力（后缀可看前缀，前缀看不到后缀）。
- **两个专家共享注意力计算**（Q/K/V 拼接后一起算 attention），但各自有独立的投影权重。PaliGemma 专家加载预训练权重，Action 专家从头训练。命名约定：第一个专家权重无后缀（可无缝加载 PaliGemma 检查点），后续专家加后缀 `_1`。

这一架构的细节（含 RoPE、门控残差、AdaRMS）见 [04 章](04-backbone-tokenizers.md)。

---

## 1.6 JAX 与 PyTorch 双实现的取舍

| 维度 | JAX/Flax 实现 | PyTorch 实现 |
| --- | --- | --- |
| 定位 | 原始实现，训练主力，发布权重的来源 | 后加入（2025-09），便于 PyTorch 生态 |
| 训练并行 | 支持 FSDP（`fsdp_devices`） | 仅 DDP（数据并行） |
| 混合精度 | 支持（权重 fp32、计算 bf16） | 不支持，只能全 bf16 或全 fp32 |
| EMA | 支持 | 不支持 |
| LoRA 微调 | 支持 | 不支持 |
| π₀-FAST | 支持 | 不支持 |
| 权重格式 | Orbax checkpoint | safetensors |
| 依赖技巧 | NNX + linen bridge | 需 patch `transformers` 库 |

**结论**：
- 需要全功能（FSDP/LoRA/EMA/FAST）→ 用 JAX。
- 已有 PyTorch 技术栈、只做 π₀/π₀.₅ 的全参微调或推理 → 可用 PyTorch。
- 两者推理在 `torch.compile` 下速度相当。

PyTorch 实现需要把 `src/openpi/models_pytorch/transformers_replace/` 覆盖到 `transformers` 安装目录，以支持 ① AdaRMS ② 激活精度控制 ③ KV 缓存只读使用。详见 [09 章](09-pytorch-implementation.md)。

---

## 1.7 一个最小可运行示例（推理）

下面是官方 README 给出的最小推理示例，可作为理解全局调用链的"锚点"：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# 1) 取出名为 pi05_droid 的训练配置（含模型结构 + 数据变换定义）
config = _config.get_config("pi05_droid")

# 2) 下载（或命中缓存）预训练检查点
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 3) 用配置 + 检查点构造一个可推理的 Policy
#    内部完成：加载模型权重 + 组装输入/输出变换 + 加载归一化统计
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 4) 喂入一帧观测（DROID 格式），拿到动作块
example = {
    "observation/exterior_image_1_left": ...,  # uint8 HWC 图像
    "observation/wrist_image_left": ...,
    "observation/joint_position": ...,         # 7 维
    "observation/gripper_position": ...,       # 1 维
    "prompt": "pick up the fork",
}
action_chunk = policy.infer(example)["actions"]   # shape: [action_horizon, action_dim]
```

这段代码触及了几乎所有子系统：配置（训练）、下载（基础设施）、模型加载（模型层）、变换组装（数据管线）、推理（推理子系统）。后续每一章都是在放大其中一环。

---

## 1.8 设计优点与局限（总评）

**优点**
- **训练-推理一致性**：同一套 Transform 双向复用，从根本上避免了"训练/部署数据分布漂移"这一常见 bug 源。
- **模型无关的训练/数据框架**：新增模型只需实现 `compute_loss` / `sample_actions`。
- **配置即实验**：一个 `TrainConfig` 完整、可复现地描述一次实验；`tyro` 支持命令行覆盖。
- **关注点分离**：机器人侧客户端零重依赖，重算力放服务端，物理与策略环境解耦。
- **可扩展的归一化**：流式 `RunningStats` 支持在超大数据集上在线计算分位数。

**局限**
- 仅在 Ubuntu 22.04 + NVIDIA GPU 上测试；不支持多节点 JAX 训练（PyTorch 支持多节点）。
- PyTorch 实现功能子集（无 FAST/LoRA/FSDP/EMA/混合精度）。
- WebSocket 服务端**无鉴权、无 TLS**，仅适合可信内网；公网部署需自行加防护。
- `transformers_replace` 的 patch 会**永久修改 uv 缓存中的 transformers**，可能影响其它项目（需 `uv cache clean transformers` 撤销）。

---

## 1.9 如何继续阅读

- 想搞懂**模型怎么生成动作** → [02 流匹配](02-models-flow-matching.md) 与 [03 FAST](03-models-pi0-fast.md)。
- 想搞懂**网络的每一层** → [04 骨干与分词器](04-backbone-tokenizers.md)、[09 PyTorch 实现](09-pytorch-implementation.md)。
- 想搞懂**数据怎么喂进去** → [05 数据管线](05-data-pipeline.md)。
- 想**训练/微调自己的模型** → [06 训练系统](06-training-system.md)。
- 想**部署成服务并在机器人上跑** → [07 推理服务](07-inference-policy-serving.md) + [08 客户端运行时](08-client-runtime.md)。
