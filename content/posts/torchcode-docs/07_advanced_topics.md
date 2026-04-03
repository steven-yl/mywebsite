---
title: "第七章：高级主题（TorchCode）"
date: 2026-04-01T10:00:00+08:00
draft: false
authors: [Steven]
description: "BPE、INT8 量化、DPO/GRPO/PPO 等对齐与工程向主题的实现解读。"
summary: "TorchCode 文档第七章：分词、量化与 RLHF 损失。"

tags: [PyTorch, TorchCode]
categories: [PyTorch]
series: [TorchCode 系列]
weight: 8
series_weight: 8
hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

# 第七章：高级主题

本章涵盖前沿技术：BPE 分词、INT8 量化、以及三种 RLHF 对齐损失函数（DPO、GRPO、PPO）。

---

## 7.1 Byte-Pair Encoding（BPE，字节对编码）

### 是什么
BPE 是 GPT、LLaMA 等模型使用的子词分词算法。它从字符级别开始，迭代地合并最频繁出现的相邻 token 对，构建一个从字符到子词的词表。

### 训练算法

```
1. 将语料中每个词拆分为字符序列，末尾加 </w> 标记
   "low" → ['l', 'o', 'w', '</w>']
2. 统计所有相邻 token 对的频率
3. 合并频率最高的对为新 token
   ('l', 'o') → 'lo'
4. 重复步骤 2-3，共 num_merges 次
```

### 编码算法

```
1. 将输入文本拆分为字符序列
2. 按训练时的合并顺序，依次应用每条合并规则
3. 返回最终的 token 序列
```

### 为什么需要子词分词
- 字符级：词表小但序列长，模型难以学习长距离依赖
- 词级：词表大（>100k），无法处理未见过的词（OOV）
- 子词级（BPE）：平衡词表大小和序列长度，能处理任意文本

### 代码示例

```python
from collections import Counter

class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        # 将每个词拆分为字符 + </w>
        words = {}
        for word in corpus:
            chars = tuple(list(word) + ['</w>'])
            words[chars] = words.get(chars, 0) + 1

        for _ in range(num_merges):
            # 统计相邻对频率
            pairs = Counter()
            for word, freq in words.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i+1])] += freq

            if not pairs:
                break

            # 合并最频繁的对
            best = pairs.most_common(1)[0][0]
            self.merges.append(best)

            # 更新词表
            new_words = {}
            for word, freq in words.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words[tuple(new_word)] = freq
            words = new_words

    def encode(self, text):
        # 按空格分词，每个词拆分为字符
        tokens = []
        for word in text.split():
            word_tokens = list(word) + ['</w>']
            # 按顺序应用合并规则
            for a, b in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and \
                       word_tokens[i] == a and word_tokens[i+1] == b:
                        new_tokens.append(a + b)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            tokens.extend(word_tokens)
        return tokens

# 测试
bpe = SimpleBPE()
bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
print("合并规则:", bpe.merges[:5])
print("编码结果:", bpe.encode('low lower'))
```

### 适用场景
- GPT-2/3/4 的分词器
- LLaMA 使用 SentencePiece（类似 BPE 的变体）
- 几乎所有现代 NLP 模型

---

## 7.2 INT8 Quantized Linear（INT8 量化线性层）

### 是什么
将浮点权重量化为 8 位整数存储，推理时反量化后计算。将模型体积压缩约 4 倍（FP32→INT8），同时保持接近原始的精度。

### 量化算法（Per-Channel）

```
1. 计算每个输出通道的缩放因子:
   scale[i] = max(|weight[i, :]|) / 127

2. 量化:
   weight_int8[i, j] = round(weight[i, j] / scale[i]).clamp(-128, 127)

3. 存储 weight_int8 (int8) 和 scale (float32)

4. 推理时反量化:
   weight_approx = weight_int8.float() * scale
   output = x @ weight_approx.T + bias
```

### Per-Channel vs Per-Tensor
- Per-Tensor：整个权重矩阵共享一个 scale，精度较低
- Per-Channel：每个输出通道独立 scale，精度更高（推荐）

### 代码示例

```python
import torch
import torch.nn as nn

class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        # Per-channel 量化
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp(min=1e-8)  # 防止除零
        weight_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('scale', scale)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        # 反量化后计算
        weight_fp = self.weight_int8.float() * self.scale
        out = x @ weight_fp.T
        if self.bias is not None:
            out = out + self.bias
        return out

# 测试
w = torch.randn(8, 4)
q = Int8Linear(w)
x = torch.randn(2, 4)
print("输出形状:", q(x).shape)
print("权重 dtype:", q.weight_int8.dtype)  # torch.int8
print("最大量化误差:", (w - q.weight_int8.float() * q.scale).abs().max().item())
# 内存节省：int8 = 1 byte vs float32 = 4 bytes → 4x 压缩
```

### 适用场景
- 模型部署（减少内存和带宽需求）
- 边缘设备推理
- 与 KV Cache 结合进一步减少推理内存

---

## 7.3 DPO Loss（Direct Preference Optimization）

### 是什么
DPO 是一种无需显式 reward model 的 RLHF 方法。直接从人类偏好数据（chosen/rejected 对）训练策略模型。

### 数学定义

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta \big[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big]\Big)$$

其中：
- $\pi_\theta$：当前策略模型
- $\pi_{\text{ref}}$：参考模型（冻结的 SFT 模型）
- $y_w$：人类偏好的回复（chosen）
- $y_l$：人类不偏好的回复（rejected）
- $\beta$：温度参数，控制偏离参考模型的程度

### 直觉理解
- 增大 chosen 回复相对于 ref 的概率
- 减小 rejected 回复相对于 ref 的概率
- $\beta$ 越大，对偏好差异越敏感

### 代码示例

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # 计算 chosen 和 rejected 的 reward 差
    chosen_reward = policy_chosen_logps - ref_chosen_logps
    rejected_reward = policy_rejected_logps - ref_rejected_logps
    # DPO loss = -log(sigmoid(beta * (chosen_reward - rejected_reward)))
    logits = beta * (chosen_reward - rejected_reward)
    return -F.logsigmoid(logits).mean()

# 测试
chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-5.0, -5.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
print("Loss:", dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())
# chosen 概率远高于 rejected → loss 较小
```

### 与 RLHF (PPO) 的区别
- DPO 不需要训练 reward model
- DPO 不需要在线采样（off-policy）
- DPO 训练更稳定，实现更简单
- 但 DPO 可能在分布外数据上表现不如 PPO

---

## 7.4 GRPO Loss（Group Relative Policy Optimization）

### 是什么
GRPO 是一种组内归一化的 REINFORCE 目标函数，常用于 RLAIF（AI 反馈的强化学习）。对同一 prompt 的多个回复，在组内计算归一化 advantage。

### 数学定义

$$A_i = \frac{r_i - \bar{r}_{g(i)}}{\text{std}_{g(i)} + \epsilon}$$

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_i\big[\text{stop\_grad}(A_i) \cdot \log \pi_\theta(y_i)\big]$$

其中 $\bar{r}_{g(i)}$ 和 $\text{std}_{g(i)}$ 是样本 $i$ 所在组的 reward 均值和标准差。

### 为什么需要组内归一化
- 不同 prompt 的 reward 尺度可能不同
- 组内归一化使得 advantage 在每个 prompt 内部是相对的
- 减少 reward 尺度差异对训练的影响

### 代码示例

```python
import torch

def grpo_loss(logps, rewards, group_ids, eps=1e-5):
    # 计算每个组的均值和标准差
    unique_groups = group_ids.unique()
    advantages = torch.zeros_like(rewards)

    for g in unique_groups:
        mask = (group_ids == g)
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        std = group_rewards.std()
        advantages[mask] = (group_rewards - mean) / (std + eps)

    # GRPO loss: -mean(detached_advantage * logps)
    return -(advantages.detach() * logps).mean()

# 测试
logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
print("Loss:", grpo_loss(logps, rewards, group_ids).item())
```

### 适用场景
- DeepSeek-R1 等使用 GRPO 进行推理能力训练
- 需要从 AI 反馈（而非人类反馈）学习的场景
- 每个 prompt 有多个采样回复的设置

---

## 7.5 PPO Loss（Proximal Policy Optimization）

### 是什么
PPO 是 RLHF 中最经典的策略优化算法。通过裁剪概率比率，限制每次更新的幅度，防止策略变化过大导致训练不稳定。

### 数学定义

$$r_i = \exp(\text{new\_logps}_i - \text{old\_logps}_i)$$

$$L_i^{\text{unclipped}} = r_i \cdot A_i$$

$$L_i^{\text{clipped}} = \text{clip}(r_i, 1-\epsilon, 1+\epsilon) \cdot A_i$$

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_i\big[\min(L_i^{\text{unclipped}}, L_i^{\text{clipped}})\big]$$

### 裁剪机制的直觉
- 当 advantage > 0（好动作）：ratio 被裁剪在 $[1-\epsilon, 1+\epsilon]$，防止过度增大该动作的概率
- 当 advantage < 0（坏动作）：ratio 被裁剪在 $[1-\epsilon, 1+\epsilon]$，防止过度减小该动作的概率
- 取 min 确保了"悲观"更新：只在两种估计都认为有利时才更新

### 代码示例

```python
import torch

def ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2):
    # 梯度只通过 new_logps 流动
    old_logps = old_logps.detach()
    advantages = advantages.detach()

    ratio = torch.exp(new_logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()

# 测试
new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6], requires_grad=True)
old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
loss = ppo_loss(new_logps, old_logps, advantages)
print("Loss:", loss.item())

# 验证梯度流动
loss.backward()
print("new_logps 有梯度:", new_logps.grad is not None)  # True
```

### 超参数
- `clip_ratio = 0.2`：裁剪范围（OpenAI 默认值）
- 通常配合 value function loss 和 entropy bonus 一起使用

---

## 7.6 三种 RLHF 损失函数对比

```
                    RLHF 对齐训练
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        DPO            GRPO           PPO
    (离线偏好)      (组内归一化)    (在线策略优化)
```

| 特性 | DPO | GRPO | PPO |
|------|-----|------|-----|
| 需要 reward model | ❌ | ✅ | ✅ |
| 需要在线采样 | ❌ | ✅ | ✅ |
| 输入数据 | chosen/rejected 对 | 组内多回复 + reward | new/old policy + advantage |
| 实现复杂度 | 低 | 中 | 高 |
| 训练稳定性 | 高 | 中 | 需要精心调参 |
| 典型应用 | Zephyr, Tulu | DeepSeek-R1 | ChatGPT, InstructGPT |
| 核心机制 | 隐式 reward 差 | 组内 advantage 归一化 | 概率比率裁剪 |
