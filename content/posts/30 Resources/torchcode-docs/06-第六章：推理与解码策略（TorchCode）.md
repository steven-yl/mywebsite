---
title: "第六章：推理与解码策略（TorchCode）"
date: 2026-04-01T10:00:00+08:00
draft: false
authors: [Steven]
description: "Top-k/Top-p 采样、Beam Search、Speculative Decoding 的原理与实现要点。"
summary: "TorchCode 文档第六章：推理与解码。"

tags: [PyTorch, TorchCode]
categories: [PyTorch]
series: [TorchCode 系列]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

# 第六章：推理与解码策略

本章介绍语言模型生成文本时的三种核心解码算法：采样策略、束搜索和推测解码。

---

## 6.1 Top-k / Top-p (Nucleus) Sampling

### 是什么
Top-k 和 Top-p 是控制语言模型生成多样性的采样策略。它们通过过滤低概率 token，在质量和多样性之间取得平衡。

### 算法步骤

```
1. Temperature 缩放: logits = logits / temperature
2. Top-k 过滤: 只保留概率最高的 k 个 token，其余设为 -inf
3. Top-p 过滤: 按概率降序排列，保留累积概率不超过 p 的 token
4. 从过滤后的分布中采样
```

### 各参数的作用

| 参数 | 效果 | 极端值 |
|------|------|--------|
| `temperature` | 控制分布的"尖锐度" | →0: 贪心（确定性）；→∞: 均匀随机 |
| `top_k` | 限制候选 token 数量 | k=1: 贪心；k=V: 无过滤 |
| `top_p` | 限制累积概率质量 | p→0: 只选最高概率；p=1: 无过滤 |

### Temperature 的直觉
- temperature < 1：分布更尖锐，高概率 token 更突出 → 更确定、更保守
- temperature > 1：分布更平坦，低概率 token 也有机会 → 更随机、更有创意
- temperature = 1：原始分布

### 代码示例

```python
import torch

def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    # 1. Temperature 缩放
    logits = logits / temperature

    # 2. Top-k 过滤
    if top_k > 0:
        topk_vals, _ = logits.topk(top_k)
        threshold = topk_vals[-1]
        logits[logits < threshold] = float('-inf')

    # 3. Top-p (nucleus) 过滤
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = probs.cumsum(dim=-1)
        # 移除累积概率超过 p 的 token（保留第一个超过的）
        mask = cumsum - probs > top_p
        sorted_logits[mask] = float('-inf')
        # 恢复原始顺序
        logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

    # 4. 采样
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# 测试
logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
print("top_k=1:", sample_top_k_top_p(logits.clone(), top_k=1))       # 总是 1
print("temp=0.01:", sample_top_k_top_p(logits.clone(), temperature=0.01))  # 几乎总是 1
```

### 实践建议
- 代码生成：temperature=0.2, top_p=0.95（低随机性）
- 创意写作：temperature=0.8, top_p=0.95（高多样性）
- 对话：temperature=0.7, top_k=50, top_p=0.9

---

## 6.2 Beam Search（束搜索）

### 是什么
Beam Search 是一种确定性的序列搜索算法，维护 `beam_width` 个最优候选序列，每步扩展所有候选并保留得分最高的。

### 算法

```
初始化: beams = [(0.0, [start_token])]

每一步:
  candidates = []
  对每个 beam (score, sequence):
    获取下一步的 log 概率分布
    对 top-beam_width 个 token:
      candidates.append((score + log_prob, sequence + [token]))
  beams = top beam_width candidates by score

终止条件:
  最优 beam 以 eos_token 结尾，或达到 max_len
```

### 与贪心搜索的区别
- 贪心搜索：每步只选最优 token（beam_width=1）
- Beam Search：保留多个候选，可能找到全局更优的序列
- 代价：计算量 × beam_width

### 代码示例

```python
import torch

def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    beams = [(0.0, [start_token])]

    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                candidates.append((score, seq))
                continue
            log_probs = log_prob_fn(seq)  # (V,)
            topk_vals, topk_idx = log_probs.topk(beam_width)
            for val, idx in zip(topk_vals, topk_idx):
                candidates.append((score + val.item(), seq + [idx.item()]))

        # 保留 top beam_width
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        # 最优 beam 已结束
        if beams[0][1][-1] == eos_token:
            break

    return beams[0][1]

# 测试
def simple_fn(tokens):
    lp = torch.full((5,), -10.0)
    lp[min(len(tokens), 4)] = 0.0
    return lp

seq = beam_search(simple_fn, start_token=0, max_len=5, beam_width=2, eos_token=4)
print("序列:", seq)  # [0, 1, 2, 3, 4]
```

### 适用场景
- 机器翻译（需要高质量输出）
- 语音识别
- 不适合开放式生成（倾向于生成重复、无聊的文本）

---

## 6.3 Speculative Decoding（推测解码）

### 是什么
利用一个小型"草稿模型"（draft model）快速生成多个候选 token，然后用大型"目标模型"（target model）并行验证。被接受的 token 无需重新计算，从而加速推理。

### 核心思想
- 大模型推理慢（受内存带宽限制），但一次验证 K 个 token 的成本与生成 1 个 token 相近
- 小模型推理快，可以快速"猜测"接下来的 token
- 如果猜对了，等于免费获得了多个 token

### 接受/拒绝算法

```
对每个位置 i = 0, ..., K-1:
  ratio = target_probs[i, token_i] / draft_probs[i, token_i]
  以概率 min(1, ratio) 接受
  如果拒绝:
    从 normalize(max(0, target - draft)) 采样一个修正 token
    追加到结果，停止
如果全部接受:
  返回所有 K 个 token
```

### 数学保证
这个接受/拒绝方案保证最终的 token 分布与直接从目标模型采样完全一致（不是近似）。

### 代码示例

```python
import torch

def speculative_decode(target_probs, draft_probs, draft_tokens):
    K = len(draft_tokens)
    accepted = []

    for i in range(K):
        token = draft_tokens[i].item()
        ratio = target_probs[i, token] / draft_probs[i, token]
        accept_prob = min(1.0, ratio.item())

        if torch.rand(1).item() < accept_prob:
            accepted.append(token)
        else:
            # 从修正分布采样
            residual = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            if residual.sum() > 0:
                residual = residual / residual.sum()
                correction = torch.multinomial(residual, 1).item()
            else:
                correction = torch.multinomial(target_probs[i], 1).item()
            accepted.append(correction)
            break

    return accepted

# 测试：完美草稿（target == draft）应全部接受
torch.manual_seed(0)
probs = torch.softmax(torch.randn(4, 10), dim=-1)
tokens = torch.tensor([2, 5, 1, 8])
result = speculative_decode(probs, probs, tokens)
print("完美草稿:", result)  # 应该是 [2, 5, 1, 8]（全部接受）

# 随机草稿：可能部分拒绝
target = torch.softmax(torch.randn(4, 10), dim=-1)
draft = torch.softmax(torch.randn(4, 10), dim=-1)
result = speculative_decode(target, draft, tokens)
print("随机草稿:", result)  # 长度 1-4
```

### 加速效果
- 草稿模型越接近目标模型，接受率越高，加速越明显
- 典型加速：2-3x（取决于草稿模型质量和 K 值）
- 无质量损失（输出分布与目标模型完全一致）

### 适用场景
- LLM 推理加速（如 Llama 70B + Llama 7B 作为草稿）
- 任何自回归生成场景
