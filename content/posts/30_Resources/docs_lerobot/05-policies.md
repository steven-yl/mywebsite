# 05 — 策略模型（Policies）

## 1. 模块边界

```
policies/
├── pretrained.py          # PreTrainedPolicy 基类
├── factory.py             # make_policy, make_pre_post_processors
├── utils.py               # 视觉 feature 校验等
├── act/                   # Action Chunking Transformer
├── diffusion/             # Diffusion Policy
├── tdmpc/                 # TD-MPC
├── vqbet/                 # VQ-BeT
├── multi_task_dit/        # Multi-Task DiT
├── gaussian_actor/        # SAC 高斯策略
├── smolvla/               # SmolVLA
├── pi0/, pi05/, pi0_fast/ # Physical Intelligence π 系列
├── groot/                 # NVIDIA GR00T
├── xvla/, wall_x/, eo1/   # 其他 VLA
├── molmoact2/             # MolmoAct2
├── vla_jepa/              # VLA-JEPA 世界模型
└── rtc/                   # Real-Time Chunking（非独立 policy type）
```

每个策略子目录通常含：
- `configuration_*.py` — `*Config(PreTrainedConfig)`
- `modeling_*.py` — `*Policy(PreTrainedPolicy)`
- `processor_*.py` — 可选专用 processor 构建

---

## 2. PreTrainedPolicy 基类

**文件**：`policies/pretrained.py`

继承：`nn.Module` + `HubMixin` + `ABC`

### 2.1 必须实现的抽象方法

| 方法 | 签名/返回 | 用途 |
|------|-----------|------|
| `get_optim_params()` | `dict` 或 param groups | 优化器参数组（如 backbone 低 LR） |
| `reset()` | — | 清空 action queue、obs history |
| `forward(batch)` | `(loss: Tensor, log_dict: dict \| None)` | **训练**损失 |
| `predict_action_chunk(batch, **kwargs)` | `(B, chunk, action_dim)` | 一次预测多步动作 |
| `select_action(batch, **kwargs)` | `(B, action_dim)` | **推理**单步（内部可能 pop queue） |

### 2.2 已实现方法

| 方法 | 作用 |
|------|------|
| `save_pretrained` / `from_pretrained` | safetensors + config |
| `push_model_to_hub` | 上传权重、train config、model card |
| `wrap_with_peft` | LoRA 等 PEFT 包装 |
| `generate_model_card` | HF ModelCard |

### 2.3 类属性

- `config_class` — 对应 Config 类
- `name` — registry 字符串

---

## 3. 工厂 API（`factory.py`）

| 函数 | 作用 |
|------|------|
| `get_policy_class(name)` | 懒加载 Policy 类 |
| `make_policy_config(type, **kwargs)` | 构造 Config |
| `make_policy(cfg, ds_meta?, env_cfg?, rename_map?)` | 推断 features、加载权重、`.to(device)` |
| `make_pre_post_processors(...)` | 策略专用 pre/post pipeline |

**Feature 推断顺序**：

1. `dataset.meta.features` → `dataset_to_policy_features`
2. 或 `env_cfg` → `env_to_policy_features`
3. `validate_visual_features_consistency`

**Relative/Absolute 重连**：加载 postprocessor 后调用 `_reconnect_relative_absolute_steps()`。

---

## 4. 全部 16 种策略详解

### 4.1 模仿学习

#### ACT (`act`)

| 项 | 内容 |
|----|------|
| **架构** | ResNet 视觉 backbone → transformer encoder-decoder；可选 VAE 对动作序列建模 |
| **训练** | L1 重建 + 可选 KL |
| **推理** | Action chunk queue 或 temporal ensembling |
| **关键方法** | `get_optim_params()` 分离 backbone LR；`select_action()` 管理 queue |
| **适用** | 中等数据量双臂/单臂；真机常用 baseline |
| **extra** | base（轻量） |

#### Diffusion (`diffusion`)

| 项 | 内容 |
|----|------|
| **架构** | 视觉编码 + 1D U-Net 对动作轨迹去噪（DDPM/DDIM） |
| **训练** | 噪声预测 MSE |
| **推理** | 多步 denoise；obs/action 双端 queue |
| **关键方法** | `populate_queues()`；`forward()` → `compute_loss()` |
| **适用** | 多模态、需要表达多峰动作分布 |
| **extra** | `lerobot[diffusion]` |

#### VQ-BeT (`vqbet`)

| 项 | 内容 |
|----|------|
| **架构** | ResNet + spatial softmax keypoints；Residual VQ-VAE 离散化动作；GPT 预测 bin+offset |
| **训练** | 两阶段：先 VQ-VAE，再 BeT |
| **关键方法** | `forward()` 分支取决于 VQ 是否已 discretize |
| **适用** | 长 horizon、离散-连续混合动作 |

#### Multi-Task DiT (`multi_task_dit`)

| 项 | 内容 |
|----|------|
| **架构** | CLIP 视觉+文本+状态 → Diffusion Transformer |
| **训练** | diffusion 或 flow matching |
| **适用** | 多任务、语言条件 |

---

### 4.2 强化学习

#### TDMPC (`tdmpc`)

| 项 | 内容 |
|----|------|
| **架构** |  latent dynamics + reward + Q ensemble + policy π；MPC (CEM/MPPI) 或 direct π |
| **训练** | 多 term TD-MPC loss；target network EMA |
| **推理** | `plan()` / `select_action()` + action queue |
| **适用** | 在线/离线 RL，仿真较多 |

#### Gaussian Actor (`gaussian_actor`)

| 项 | 内容 |
|----|------|
| **架构** | CNN/预训练视觉编码 + 高斯策略（tanh squashing）；可选离散 gripper critic |
| **训练** | `forward()` 返回 log_prob 等（SAC 用，非 BC loss） |
| **适用** | HIL-SERL / `rl/` 模块 |

---

### 4.3 Vision-Language-Action (VLA)

#### PI0 (`pi0`)

| 项 | 内容 |
|----|------|
| **架构** | PaliGemma + Gemma action expert；flow matching 迭代 denoise |
| **推理** | `sample_actions()` + KV cache；支持 **RTC** |
| **关键方法** | `from_pretrained()` key remapping；`denoise_step()` |
| **extra** | `lerobot[pi]` |

#### PI05 (`pi05`)

π₀.₅ 变体，配置与权重与 PI0 同族，OpenPI 移植。

#### PI0-FAST (`pi0_fast`)

| 项 | 内容 |
|----|------|
| **架构** | PaliGemma + FAST action tokenizer；**自回归** token 生成 |
| **与 PI0 区别** | 离散 token 而非连续 flow matching |
| **processor** | 动态 `make_pi0_fast_pre_post_processors` |

#### SmolVLA (`smolvla`)

| 项 | 内容 |
|----|------|
| **架构** | SmolVLM + action expert；flow matching |
| **特点** | 较小 VLM；`NewLineTaskProcessorStep`；RTC |
| **extra** | `lerobot[smolvla]` |

#### GR00T (`groot`)

| 项 | 内容 |
|----|------|
| **架构** | Eagle 2.5 VL + CrossAttentionDiT flow-matching head |
| **实现** | 封装 Isaac-GR00T `GR00TN15` |
| **extra** | `lerobot[groot]`（flash-attn 等平台限制） |

#### XVLA (`xvla`)

Florence-2 编码器 + SoftPromptedTransformer；多 domain soft prompts。

#### WALL-X (`wall_x`)

Qwen2.5-VL MoE + flow-matching；cross-embodiment。

#### EO1 (`eo1`)

Qwen2.5-VL + flow matching action projector。

#### MolmoAct2 (`molmoact2`)

Molmo VLM + per-layer flow expert + 可选 discrete action head；modes: `continuous` / `discrete` / `both`。

#### VLA-JEPA (`vla_jepa`)

Qwen3-VL + DiT action head + 可选 V-JEPA 世界模型；联合 action + world model loss。

---

## 5. RTC（Real-Time Chunking）

**路径**：`policies/rtc/`

**不是**独立 `--policy.type`，而是 chunking 策略的**推理加速**模块：

| 组件 | 作用 |
|------|------|
| `ActionQueue` | 线程安全 chunk 队列 |
| `RTCProcessor` | prefix re-anchor，减少 chunk 边界 jitter |
| `LatencyTracker` | 跟踪推理延迟 |
| `ActionInterpolator` | 平滑执行 |

支持 RTC 的策略：PI0、PI05、PI0Fast、SmolVLA、MolmoAct2 等。

部署时使用 `--inference.type=rtc`（见 [09-rollout-inference.md](./09-rollout-inference.md)）。

---

## 6. 训练 vs 推理 API 对照

| 阶段 | 调用链 |
|------|--------|
| **Train** | `batch → preprocessor → forward(batch) → loss` |
| **Eval sim** | `obs → preprocessor → select_action → postprocessor → env.step` |
| **Rollout** | `robot.obs → preprocessor → InferenceEngine → postprocessor → send_action` |

**Episode 边界**：必须 `policy.reset()` 清空 queue。

---

## 7. PEFT 微调

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --peft.method_type=LORA \
  --peft.r=8 \
  ...
```

`wrap_with_peft()` 冻结 base，仅训练 adapter；Hub 可只上传 adapter 权重。

---

## 8. 策略选型对比表

| 策略 | 参数量级 | 语言 | Chunk | 真机 RTC | 典型数据 |
|------|----------|------|-------|----------|----------|
| ACT | 小 | ✗ | ✓ | sync | 50–200 demos |
| Diffusion | 中 | ✗ | ✓ | sync | 100+ demos |
| PI0 | 大 | ✓ | ✓ | rtc | 1k+ 或多任务 |
| SmolVLA | 中–大 | ✓ | ✓ | rtc | 中等 |
| TDMPC | 中 | ✗ | ✓ | sync | RL rollouts |

---

## 9. 示例

### 9.1 构造策略并 dummy forward

```python
"""本地初始化 ACT 并 fake batch forward（无需数据集）。"""
import torch
from lerobot.policies.factory import make_policy_config, get_policy_class

cfg = make_policy_config(
    "act",
    input_features={
        "observation.state": {"type": "STATE", "shape": (6,)},
        "observation.images.top": {"type": "VISUAL", "shape": (3, 96, 96)},
    },
    output_features={"action": {"type": "ACTION", "shape": (6,)}},
)
PolicyCls = get_policy_class("act")
policy = PolicyCls(cfg)
policy.eval()

B = 2
batch = {
    "observation.state": torch.randn(B, 6),
    "observation.images.top": torch.randn(B, 3, 96, 96),
    "action": torch.randn(B, 6),
}
loss, logs = policy.forward(batch)
print("loss", loss.item())
```

### 9.2 Hub 加载推理

```python
"""从 Hub 加载策略（需网络与权重）。"""
from lerobot.policies.factory import make_policy_config
from lerobot.policies.pretrained import PreTrainedPolicy

cfg = make_policy_config("act", pretrained_path="lerobot/act_aloha_sim_transfer_cube_human")
policy = PreTrainedPolicy.from_pretrained(cfg.pretrained_path, config=cfg)
policy.eval()
print(policy.name)
```

---

## 下一章

- 训练循环 → [06-training-evaluation.md](./06-training-evaluation.md)
- 部署 → [09-rollout-inference.md](./09-rollout-inference.md)
