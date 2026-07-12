---
title: "5. 动作表示与 Token 化"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "连续动作离散化为 Token、VLM 词表对齐、Co-Fine-Tuning 与输出约束策略。"
summary: "RT-2 动作 token 化与 Co-Fine-Tuning 详解。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 6
series_weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 5. 动作表示与 Token 化

本章详解 RT-2 如何将连续机器人动作离散化为 Token、如何与 VLM 词表对齐，以及 Co-Fine-Tuning 与输出约束策略。

---

## 5.1 为什么将动作表示为 Token

传统机器人策略输出连续向量 $\mathbf{a} \in \mathbb{R}^d$，与 VLM 的离散文本输出空间不兼容。RT-2 的核心创新是：

> **动作 = 另一种"语言"** —— 与 VQA 答案、图像描述使用相同的 Transformer 输出头与损失函数。

**好处**：
1. 无需新增动作专用网络层
2. Web 预训练权重完全共享
3. 语言与动作联合训练（Co-Fine-Tuning）自然可行
4. 支持 Chain-of-Thought（先 Plan 后 Action）

---

## 5.2 动作空间定义（RT-1 协议）

继承 [RT-1 (Brohan et al., 2022)](https://arxiv.org/abs/2212.06817)：

$$
\mathbf{a} = (a_0, a_1, \ldots, a_7)
$$

| 索引 | 符号 | 含义 | 类型 |
|------|------|------|------|
| 0 | $\tau$ | terminate（回合终止） | 离散 {0,1} |
| 1 | $\Delta x$ | 末端 X 位移 | 连续 |
| 2 | $\Delta y$ | 末端 Y 位移 | 连续 |
| 3 | $\Delta z$ | 末端 Z 位移 | 连续 |
| 4 | $\Delta r_x$ | 绕 X 轴旋转增量 | 连续 |
| 5 | $\Delta r_y$ | 绕 Y 轴旋转增量 | 连续 |
| 6 | $\Delta r_z$ | 绕 Z 轴旋转增量 | 连续 |
| 7 | $g$ | 夹爪开合 | 连续 |

---

## 5.3 均匀离散化算法

### 5.3.1 编码（连续 → bin）

对每个连续维度 $a_i$，给定范围 $[a_i^{\min}, a_i^{\max}]$：

$$
b_i = \text{clip}\left(\left\lfloor \frac{a_i - a_i^{\min}}{a_i^{\max} - a_i^{\min}} \times (B-1) \right\rfloor,\ 0,\ B-1\right)
$$

其中 $B = 256$（bin 数量）。

**terminate** 维度直接二值化：$b_0 \in \{0, 1\}$。

### 5.3.2 解码（bin → 连续）

$$
\hat{a}_i = a_i^{\min} + \frac{b_i}{B-1} (a_i^{\max} - a_i^{\min})
$$

**量化误差上界**（每维）：

$$
|\hat{a}_i - a_i| \leq \frac{a_i^{\max} - a_i^{\min}}{2(B-1)} \approx \frac{\Delta_i}{512}
$$

### 5.3.3 动作字符串

8 个 bin 索引空格拼接：

```
"{b0} {b1} {b2} {b3} {b4} {b5} {b6} {b7}"
```

示例：`"1 128 91 241 5 101 127 200"`

---

## 5.4 Token 映射

### 5.4.1 PaLI-X 策略

整数 0–1000 在 SentencePiece 词表中各有唯一 Token：

$$
\text{Tokenize}(b_i) = \text{SPM}(str(b_i)), \quad b_i \in [0, 255]
$$

### 5.4.2 PaLM-E 策略

选取 256 个**最少使用**的 SentencePiece Token 建立双射：

$$
\phi: \{0,\ldots,255\} \rightarrow \mathcal{V}_{\text{action}} \subset \mathcal{V}_{\text{SPM}}
$$

这是 **Symbol Tuning** 的一种实例（[Wei et al., 2023](https://arxiv.org/abs/2305.08298)）。

### 5.4.3 本仓库策略（建议实现）

本仓库 `num_tokens=20000`，可预留 Token ID `0–255` 作为动作 bin：

```python
ACTION_TOKEN_OFFSET = 0  # token_id = bin_index + OFFSET
```

---

## 5.5 训练数据格式

### 5.5.1 VQA 格式包装

**输入**（Prompt）：

```
Q: what action should the robot take to [task instruction]? A:
```

**输出**（Target）：

```
1 128 91 241 5 101 127 200
```

### 5.5.2 CoT 格式（PaLM-E）

```
Plan: Pick rxbar chocolate
Action: 132 114 128 5 25 156
```

### 5.5.3 完整训练样本

```json
{
  "image": "robot_cam_frame_042.jpg",
  "instruction": "pick up the apple",
  "action_bins": [0, 128, 91, 241, 5, 101, 127, 180],
  "prompt": "Q: what action should the robot take to pick up the apple? A:",
  "target": "0 128 91 241 5 101 127 180"
}
```

---

## 5.6 Co-Fine-Tuning

### 5.6.1 为什么需要

仅在机器人数据上微调 VLM 会导致 **灾难性遗忘**（catastrophic forgetting），丢失 Web 预训练的语义能力。

### 5.6.2 混合策略

每个 batch 混合两类数据：

| 数据类型 | 内容 | RT-2-PaLI-X 比例 | RT-2-PaLM-E 比例 |
|----------|------|------------------|------------------|
| Web VLM | VQA、Caption、推理 | ~50% | ~34% |
| Robot | 图像+指令+动作 Token | ~50% | ~66% |

通过 **提高机器人数据采样权重** 在 batch 内平衡。

### 5.6.3 消融结果

| 训练方式 | Unseen Avg |
|----------|------------|
| 5B from scratch | 9% |
| 5B fine-tune (robot only) | 42% |
| 5B co-fine-tune | 44% |
| 55B co-fine-tune | **63%** |

---

## 5.7 输出约束 (Output Constraint)

推理机器人动作任务时，**限制采样仅在 256 个有效动作 Token 上**：

$$
P(y_t \mid \cdot) \leftarrow \frac{P(y_t \mid \cdot) \cdot \mathbb{1}[y_t \in \mathcal{V}_{\text{action}}]}{\sum_{v \in \mathcal{V}_{\text{action}}} P(v \mid \cdot)}
$$

VQA 等视觉语言任务仍可使用完整词表。

---

## 5.8 可运行示例：ActionTokenizer

```python
import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ActionBounds:
    """各维动作范围，需根据机器人平台标定"""
    mins: Tuple[float, ...] = (-0.05, -0.05, -0.05, -30, -30, -30, 0)
    maxs: Tuple[float, ...] = (0.05, 0.05, 0.05, 30, 30, 30, 1)

class ActionTokenizer:
    NUM_BINS = 256
    NUM_DIMS = 8  # terminate + 6DoF + gripper

    def __init__(self, token_offset: int = 0):
        self.token_offset = token_offset
        self.bounds = ActionBounds()

    def encode_continuous(self, action: List[float]) -> List[int]:
        """连续动作 → bin 索引"""
        bins = [1 if action[0] > 0.5 else 0]  # terminate
        for i in range(1, 8):
            a_min = self.bounds.mins[i - 1] if i < 7 else self.bounds.mins[6]
            a_max = self.bounds.maxs[i - 1] if i < 7 else self.bounds.maxs[6]
            val = action[i] if i < len(action) else 0.0
            ratio = (val - a_min) / (a_max - a_min + 1e-8)
            bins.append(int(min(self.NUM_BINS - 1, ratio * (self.NUM_BINS - 1))))
        return bins

    def bins_to_token_ids(self, bins: List[int]) -> torch.Tensor:
        return torch.tensor([b + self.token_offset for b in bins], dtype=torch.long)

    def token_ids_to_bins(self, token_ids: torch.Tensor) -> List[int]:
        return [(t.item() - self.token_offset) for t in token_ids]

    def decode_bins(self, bins: List[int]) -> List[float]:
        """bin 索引 → 连续动作（近似）"""
        action = [float(bins[0])]
        for i in range(1, 8):
            b = bins[i]
            a_min = self.bounds.mins[i - 1] if i < 7 else self.bounds.mins[6]
            a_max = self.bounds.maxs[i - 1] if i < 7 else self.bounds.maxs[6]
            action.append(a_min + (b / (self.NUM_BINS - 1)) * (a_max - a_min))
        return action

    def format_target_string(self, bins: List[int]) -> str:
        return " ".join(str(b) for b in bins)


# 使用示例
tok = ActionTokenizer(token_offset=0)
continuous = [0, 0.01, -0.02, 0.0, 5.0, -10.0, 3.0, 0.8]
bins = tok.encode_continuous(continuous)
print("bins:", bins)
print("target:", tok.format_target_string(bins))

token_ids = tok.bins_to_token_ids(bins)
recovered = tok.decode_bins(tok.token_ids_to_bins(token_ids))
print("recovered:", recovered)
```

---

## 5.9 约束采样示例

```python
import torch
import torch.nn.functional as F

ACTION_VOCAB = torch.arange(0, 256)  # 256 个动作 token id

def constrained_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """仅在动作词表上采样"""
    action_logits = logits[ACTION_VOCAB]
    probs = F.softmax(action_logits / temperature, dim=-1)
    idx = torch.multinomial(probs, 1).item()
    return ACTION_VOCAB[idx].item()

# 模拟一步动作生成
fake_logits = torch.randn(20000)
action_token = constrained_sample(fake_logits)
print(f"sampled action token: {action_token}")
```

---

## 5.10 Language-Table 2D 动作变体

仿真环境使用简化 2D 动作：

$$
\text{action string} = \text{"\{X\} \{Y\}"}, \quad X, Y \in \{-10, -9, \ldots, 10\}
$$

共 $21 \times 21 = 441$ 种组合，无需 256-bin 均匀离散。

---

## 5.11 结构边界

```
┌─────────────────────────────────────────────────────────┐
│                    动作 Token 化管线                      │
├──────────────┬──────────────────────────────────────────┤
│ 机器人平台    │ 连续动作向量 (8-dim)                       │
├──────────────┼──────────────────────────────────────────┤
│ 离散化        │ 256-bin 均匀量化 → 8 个整数               │
├──────────────┼──────────────────────────────────────────┤
│ 字符串化      │ "b0 b1 ... b7"                           │
├──────────────┼──────────────────────────────────────────┤
│ Token 映射    │ SPM / 预留 ID → token_ids                │
├──────────────┼──────────────────────────────────────────┤
│ 模型训练      │ CrossEntropy(logits, token_ids)          │
├──────────────┼──────────────────────────────────────────┤
│ 推理          │ generate → 约束采样 → de-tokenize        │
├──────────────┼──────────────────────────────────────────┤
│ 机器人执行    │ 连续动作 → 底层控制器                     │
└──────────────┴──────────────────────────────────────────┘
```

**本仓库边界**：`rt2/model.py` 仅实现 Token 序列 → logits；**编解码器需用户自行实现**（见上文示例）。

---

## 5.12 参考文献

| 文献 | 链接 |
|------|------|
| RT-2 论文 §3.2 | https://arxiv.org/abs/2307.15818 |
| RT-1 动作空间 | https://arxiv.org/abs/2212.06817 |
| Symbol Tuning | https://arxiv.org/abs/2305.08298 |
| Language-Table | https://arxiv.org/abs/2206.04171 |

---

## 5.13 相关章节

- 训练数据混合 → [06-training-datasets.md](./06-training-datasets.md)
- 生成 API → [04-decoder-autoregression.md](./04-decoder-autoregression.md)
