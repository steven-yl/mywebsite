---
title: "Speculative Decoding 原理与实现"
date: 2026-04-13T10:00:00+08:00
draft: false
authors: [Steven]
description: "Speculative Decoding 原理与实现"
summary: "Speculative Decoding 原理与实现"

tags: [Deep Learning]
categories: [Deep Learning]
series: [Deep Learning系列]

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---

## Speculative Decoding 详细解释

Speculative Decoding（推测解码）是一种在**不改变输出分布**的前提下，显著加速大语言模型（LLM）自回归生成的方法。它通过引入一个**小模型**（draft model）来“猜测”后续多个 token，然后利用**大模型**（target model）一次性并行验证这些猜测，从而将多次串行的前向计算合并为一次，实现无损加速。

---

### 1. 问题背景：自回归生成的低效性

LLM 生成文本时，每一步只产生一个 token，且必须依赖之前所有 token 的结果：

\[
x_{t+1} \sim P(\cdot \mid x_1, x_2, \dots, x_t)
\]

这是一个**严格串行**的过程。每生成一个 token 都需要执行一次完整的前向传播（包括所有 Transformer 层）。对于大型模型（如 70B 参数），单次前向传播可能耗时数十毫秒甚至更多。生成一个 1000 token 的回答就需要 1000 次前向传播，导致推理延迟高、吞吐量低。

- **KV Cache** 虽然缓解了重复计算，但无法减少前向传播的次数。
- **批处理**（batching）可以提高吞吐，但对单条请求（在线服务）延迟改善有限。

Speculative Decoding 的目标就是**减少目标模型的前向调用次数**。

---

### 2. 核心思想

使用一个**更小、更快**的草稿模型（draft model）来预测接下来的 **K 个 token**（K 通常为 3~10）。然后用目标大模型**一次前向传播**验证这 K 个 token 是否“合理”。由于小模型生成速度快，且大模型一次验证多个 token 的时间远小于串行生成 K 个 token 的时间，因此总时间降低。

**关键点**：验证过程是并行的，且通过一个巧妙的接受/拒绝规则，保证最终生成的 token 分布与**直接使用大模型自回归**完全一致（无损）。

---

### 3. 算法详细步骤（以贪心解码为例，采样类似）

假设：
- 目标模型 \( M_p \)（大，慢，精确）
- 草稿模型 \( M_q \)（小，快，近似 \( M_p \)）
- 已生成序列 \( x_{1:t} \)
- 草稿长度 \( K \)（超参数）

#### 步骤 1：草稿生成
使用 \( M_q \) 以 \( x_{1:t} \) 为输入，**自回归**地生成 K 个 token：
\[
\hat{y}_1 = \arg\max_{y} M_q(y \mid x_{1:t})
\]
\[
\hat{y}_2 = \arg\max_{y} M_q(y \mid x_{1:t}, \hat{y}_1)
\]
\[
\dots
\]
\[
\hat{y}_K = \arg\max_{y} M_q(y \mid x_{1:t}, \hat{y}_1, \dots, \hat{y}_{K-1})
\]
得到草稿序列 \( \hat{y}_{1:K} \)。（采样模式下，这里是从分布中采样而非 argmax）

#### 步骤 2：并行验证
将扩展后的序列 \( x_{1:t}, \hat{y}_1, \hat{y}_2, \dots, \hat{y}_K \) 输入到 **目标模型 \( M_p \)**，执行**一次**前向传播。由于 Transformer 的因果注意力机制，我们可以同时获得所有位置的条件概率分布：
\[
p_1(\cdot) = M_p(\cdot \mid x_{1:t})
\]
\[
p_2(\cdot) = M_p(\cdot \mid x_{1:t}, \hat{y}_1)
\]
\[
\dots
\]
\[
p_{K+1}(\cdot) = M_p(\cdot \mid x_{1:t}, \hat{y}_1, \dots, \hat{y}_K)
\]
注意这里共 K+1 个分布，每个分布对应下一个 token 的预测。这些分布是通过一次前向计算得到的（利用因果掩码并行计算所有位置的输出）。

#### 步骤 3：接受/拒绝（贪心版本）
对于每个位置 \( i = 1 \dots K \)：
- 计算目标模型的 argmax token：\( y_i^* = \arg\max p_i(\cdot) \)
- 如果 \( y_i^* = \hat{y}_i \)，则**接受**该草稿 token，继续下一个位置。
- 如果 \( y_i^* \neq \hat{y}_i \)，则**拒绝** \( \hat{y}_i \)，并输出 \( y_i^* \) 作为当前步的真正 token，同时**丢弃**草稿中 \( \hat{y}_i, \dots, \hat{y}_K \)（包括未验证的后续 token）。然后退出循环。

如果所有 K 个草稿 token 都被接受，我们可以选择再输出一个额外的 token：\( y_{K+1}^* = \arg\max p_{K+1}(\cdot) \)，这样一次迭代最多可以生成 K+1 个 token。

#### 步骤 4：更新状态
将接受的 token 追加到 \( x_{1:t} \) 中，回到步骤 1 继续生成，直到遇到结束符或达到最大长度。

### 4. 为什么无损（数学保证）

**关键点**：在采样模式下（从分布中随机采样），上述接受/拒绝规则需要修正，以保证最终采样分布等于 \( M_p \) 的分布。这个修正使用了**重要性采样**的思想。

设目标模型分布为 \( p(x) = M_p(x \mid context) \)，草稿模型分布为 \( q(x) = M_q(x \mid context) \)。对于单步生成，我们可以：

- 从 \( q \) 中采样一个 token \( \hat{y} \)。
- 以概率 \( \min\left(1, \frac{p(\hat{y})}{q(\hat{y})}\right) \) 接受 \( \hat{y} \)。
- 如果拒绝，则从调整后的分布 \( p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y} \max(0, p(y) - q(y))} \) 中重新采样。

可以证明，这样最终采样到的 token 分布恰好等于 \( p(x) \)。这个结论可以推广到多步（序列）情况，只要在每一步独立应用上述规则，并且当某个草稿 token 被拒绝时，后续的草稿 token 全部丢弃。完整证明参见原始论文（Leviathan et al., 2022; Chen et al., 2023）。

**贪心解码**（取 argmax）是上述规则的特例：因为 argmax 相当于概率分布退化为 one‑hot，接受条件简化为“草稿 token 是否等于目标模型的 argmax”。

因此，Speculative Decoding 是**无损**的——它生成的 token 序列与直接使用目标模型自回归生成的序列在分布上完全相同。

---

### 5. 加速比分析

设草稿长度为 \( K \)，目标模型单次前向时间为 \( T_p \)，草稿模型生成一个 token 的时间为 \( T_q \)（通常 \( T_q \ll T_p \)）。每次迭代中：

- 草稿生成需要 K 次小模型前向：\( K \cdot T_q \)
- 一次大模型验证：\( T_p \)
- 假设平均接受长度为 \( \bar{a} \)（\( 1 \le \bar{a} \le K+1 \)），则每次迭代生成 \( \bar{a} \) 个 token。

平均每 token 耗时：
\[
\frac{K \cdot T_q + T_p}{\bar{a}}
\]
相比于直接使用大模型（每 token 耗时 \( T_p \)），加速比为：
\[
\text{Speedup} = \frac{T_p}{\frac{K T_q + T_p}{\bar{a}}} = \frac{\bar{a}}{1 + K \frac{T_q}{T_p}}
\]

**关键因素**：
- **接受长度 \( \bar{a} \)**：取决于草稿模型与目标模型分布的一致性。若完全一致，则 \( \bar{a} = K+1 \)，加速比接近 \( \frac{K+1}{1 + K \frac{T_q}{T_p}} \)。通常 \( T_q/T_p \) 很小（比如 0.05），因此加速比可接近 K+1。
- **实际中**，由于草稿模型无法完美预测，\( \bar{a} \) 通常在 2~5 之间（K 取 5~10）。结合 \( T_q \ll T_p \)，常见加速比为 2~3 倍。

---

### 6. 草稿模型的选择与训练

#### 6.1 草稿模型的要求
- **足够小**：推理速度要快，否则生成 K 个草稿的开销会抵消收益。典型草稿模型参数量为目标模型的 1/10 到 1/100。
- **分布接近目标模型**：接受率才会高。理想情况下，草稿模型是目标模型的蒸馏版本。

#### 6.2 草稿模型来源
- **同架构缩小版**：例如 LLaMA‑70B 配合 LLaMA‑7B。
- **独立训练的小模型**：用相同数据训练一个更小的模型。
- **蒸馏**：用目标模型的 soft label 训练草稿模型，提高分布对齐。
- **无需额外模型**：一些变体（如 Medusa）直接在目标模型上添加额外的输出头，利用目标模型自身的浅层表示作为草稿。

#### 6.3 不需要完全相同词表
草稿模型和目标模型可以使用不同的词表，但需要建立映射关系（例如通过子词对齐）。实际工程中通常使用相同的 tokenizer。

---

### 7. 主要变体

| 方法 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **标准推测解码** | 独立草稿模型 | 简单、通用 | 需要额外加载小模型，增加显存 |
| **Medusa** | 在目标模型最后一层添加多个解码头，并行预测后续 token | 无需额外模型，直接利用大模型特征 | 需要微调目标模型，增加参数量 |
| **Lookahead Decoding** | 使用 n‑gram 匹配或检索作为草稿 | 零训练开销 | 接受率可能较低，依赖缓存 |
| **Self-Speculative Decoding** | 利用目标模型浅层（早期层）的输出作为草稿，深层验证 | 无需额外模型，训练简单 | 浅层与深层分布差异可能较大 |
| **Eagle** | 使用目标模型的 hidden state 作为条件，训练一个轻量草稿网络 | 接受率高 | 需要额外训练 |

---

### 8. 实践中的注意事项

#### 8.1 采样与温度
在采样模式下（temperature > 0），草稿模型也应该使用相同的 temperature 进行采样，以保证分布对齐。如果使用不同的 temperature，接受率会下降。

#### 8.2 KV Cache 管理
草稿模型和目标模型各自维护 KV Cache。验证阶段，目标模型的 KV Cache 可以直接复用（因为输入的 token 是确定的）。草稿生成时，小模型的 KV Cache 独立更新。

#### 8.3 内存开销
加载两个模型会增加显存占用。一个常见技巧是**将草稿模型放在 CPU**（如果 CPU 推理足够快），或者使用量化后的草稿模型。

#### 8.4 动态调整 K
最优的草稿长度 K 取决于接受率。可以动态调整：当接受率低时减小 K，接受率高时增大 K。有些实现（如 vLLM）支持自适应 K。

#### 8.5 批处理中的推测解码
在批处理场景下（同时生成多个序列），推测解码可以并行应用：每个序列独立生成草稿，然后目标模型在一次前向中同时验证所有序列的草稿（batch 维度并行）。加速效果更好。

---

### 9. 与其他加速技术的对比

| 技术 | 原理 | 是否无损 | 加速比 | 额外开销 |
|------|------|----------|--------|----------|
| 量化 | 降低精度 | 有损（轻微） | 2-4x | 无额外模型 |
| 模型剪枝/蒸馏 | 减小模型尺寸 | 有损 | 2-10x | 需要训练 |
| KV Cache | 避免重复计算 | 无损 | 1.5-2x（长文本） | 显存增加 |
| Speculative Decoding | 小模型猜，大模型验 | 无损 | 2-3x | 额外模型或训练 |

推测解码可以**与量化、KV Cache 等方法正交组合**，进一步加速。

---

### 10. 伪代码示例（贪心解码）

```python
def speculative_decode_greedy(target_model, draft_model, prefix, K=5):
    input_ids = prefix
    while not done:
        # 1. Draft
        draft_ids = []
        cur = input_ids
        for _ in range(K):
            next_token = draft_model.generate(cur, temperature=0)  # argmax
            draft_ids.append(next_token)
            cur = cur + [next_token]
            if next_token == eos_token_id:
                break
        
        # 2. Verify with target model in one forward pass
        full_seq = input_ids + draft_ids
        logits = target_model.forward(full_seq)  # shape: (len(full_seq), vocab_size)
        
        # 3. Accept/Reject
        accept_count = 0
        for i, draft_token in enumerate(draft_ids):
            target_token = torch.argmax(logits[len(input_ids) + i])
            if target_token == draft_token:
                accept_count += 1
            else:
                # reject this draft token, use target's token instead
                output_token = target_token
                break
        else:
            # all K draft tokens accepted, optionally generate one more
            output_token = torch.argmax(logits[len(input_ids) + K])
            accept_count = K + 1
        
        # 4. Append accepted tokens
        input_ids.extend([output_token])
        if output_token == eos_token_id:
            break
```

---

### 11. 实验结果摘要（典型数据）

- **LLaMA‑70B + LLaMA‑7B**，K=5，贪心解码：加速比约 2.5x，接受率约 0.65。
- **GPT‑2 1.5B + GPT‑2 125M**，采样模式（temperature=1.0）：加速比 2.2x。
- **Medusa 在 Vicuna‑7B 上**：加速比 2x，几乎无损。

---

### 总结

Speculative Decoding 通过“小模型猜，大模型验”的巧妙设计，在保持输出分布不变的前提下，将大模型的自回归生成加速了 2-3 倍。它已经成为现代 LLM 推理系统（如 vLLM, TensorRT‑LLM, Hugging Face TGI）的标配技术之一，尤其适合在线低延迟场景。理解其原理对于部署高性能 LLM 服务非常有帮助。