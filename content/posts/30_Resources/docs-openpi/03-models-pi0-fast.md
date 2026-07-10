# 03 自回归模型：π₀-FAST

> 本章解读 `src/openpi/models/pi0_fast.py`。π₀-FAST 走的是与流匹配完全不同的路线：把连续动作用 **FAST 分词器**离散成 token，再像语言模型一样**自回归生成**。本章讲清它的动机、序列结构、训练（teacher forcing）与推理（KV 缓存逐 token 解码）。

---

## 3.1 为什么需要 π₀-FAST

流匹配的 π₀ 推理快、动作平滑，但动作长度固定、对语言条件的跟随有时不如自回归。π₀-FAST 把动作当成"另一种语言"：

- **统一建模**：动作和语言用同一个词表、同一套自回归损失，能更好地利用预训练 LLM 的语言能力 → **语言跟随更强**。
- **可变长输出**：自回归可生成不定长的 token 序列，遇到 EOS 即停。
- **代价**：逐 token 解码，推理比流匹配慢，适合对延迟不敏感、但要求强语言条件的场景（如 DROID 上的 `pi0_fast_droid`）。

> 注意：π₀-FAST **仅有 JAX 实现**，PyTorch 端不支持。

---

## 3.2 序列结构：前缀 + 后缀

π₀-FAST 把整个输入拼成一个 token 序列，分两段：

```
[图像 token][语言+状态 token]   |   [Action: <FAST动作token...> |]
└──────── 前缀（prefix）────────┘   └───────── 后缀（postfix）──────────┘
        ar_mask = 0（双向）                ar_mask = 1（因果，逐 token 生成）
        loss_mask = False                  loss_mask = True（只在动作上算损失）
```

- **前缀**：图像 token + 文本（含离散化的状态），双向注意力，不计损失。
- **后缀**：`Action: ` 引导词 + FAST 动作 token + 结束符 `|`，因果注意力，计算交叉熵损失。

这个序列由 `tokenizer.FASTTokenizer.tokenize` 构造（见 [04 章](04-backbone-tokenizers.md)），它同时产出 `tokenized_prompt`、`token_mask`、`token_ar_mask`、`token_loss_mask` 四个数组，恰好对应 `Observation` 里的字段。

---

## 3.3 配置 `Pi0FASTConfig`

```python
@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: Variant = "gemma_2b"        # 注意：FAST 只用单一 Gemma，无独立动作专家
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250
    fast_model_tokenizer: Any | None = None        # 可注入自定义分词器
    fast_model_tokenizer_kwargs: dict | None = None
```

**与 `Pi0Config` 的关键区别**：
- 没有 `action_expert_variant`——π₀-FAST 只用一个 Gemma（无混合专家），因为动作就是 token，直接走 LLM。
- `max_token_len` 默认更大（250），因为动作 token 拼在序列里。
- `model_type` 恒为 `PI0_FAST`。
- `get_freeze_filter`：若用 LoRA，冻结 `.*llm.*` 但排除 `.*lora.*`（比 π₀ 简单，因为只有一个专家）。
- `inputs_spec`：图像键是 `base_0_rgb` / `base_1_rgb` / `wrist_0_rgb`（与 π₀ 的三路不同），且包含 `token_ar_mask`、`token_loss_mask`。

---

## 3.4 输入嵌入 `embed_inputs`

```python
def embed_inputs(self, obs) -> (token_embeddings, input_mask, ar_mask):
    for name in obs.images:                       # 图像 token
        image_emb, _ = self.PaliGemma.img(obs.images[name], train=False)
        token_embeddings.append(image_emb)
        input_mask.append(repeat(obs.image_masks[name], "b -> b s", s=image_emb.shape[1]))
        ar_mask.append(0 * input_mask[-1])        # 图像双向
    # 文本/动作 token（已在 tokenizer 阶段拼好）
    tokenized_emb = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
    token_embeddings.append(tokenized_emb)
    input_mask.append(obs.tokenized_prompt_mask)
    ar_mask.append(obs.token_ar_mask)             # 前缀 0，后缀 1
    return concat(token_embeddings), concat(input_mask), concat(ar_mask)
```

与 π₀ 不同的是：这里直接用 `Observation` 里预先算好的 `token_ar_mask`（来自 FAST 分词器），而非在模型里硬编码块结构。

---

## 3.5 训练 `compute_loss`：next-token 预测

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    observation = preprocess_observation(rng, observation, train=train,
                                         image_keys=list(observation.images.keys()))
    input_emb, input_mask, ar_mask = self.embed_inputs(observation)
    attn_mask = make_attn_mask(input_mask, ar_mask)

    # 预测下一个 token：目标是输入右移一位
    targets = one_hot(observation.tokenized_prompt[:, 1:], vocab_size)

    # 前向：不输入最后一个 token（因为它没有"下一个"要预测）
    pre_logits, _, _ = self.PaliGemma.llm(embedded_prefix=input_emb[:, :-1],
                                          mask=attn_mask[:, :-1, :-1],
                                          return_prelogits=True)
    # 只对目标位置解码 logits（省显存：解码矩阵是 seq_len × vocab_size）
    logits, _ = self.PaliGemma.llm(pre_logits=pre_logits[:, -targets.shape[1]:])
    logp = log_softmax(logits, axis=-1)

    # 仅在动作 token（loss_mask=True）上计算交叉熵
    loss_mask = observation.token_loss_mask[:, 1:]
    token_pplx = sum(targets * logp, axis=-1)
    return -sum(token_pplx * loss_mask, axis=-1) / clip(sum(loss_mask, -1), 1)
```

**算法步骤**：
1. 预处理 + 嵌入，构造因果注意力掩码。
2. 目标 = 输入序列右移一位（标准 next-token）。
3. 前向时丢掉最后一个 token；用 `return_prelogits=True` 拿到倒数第二层表示。
4. **只对目标段解码 logits**（动作 token 段），避免对整段 250×257152 的大矩阵做 matmul，省显存。
5. 交叉熵损失只在 `token_loss_mask` 标记的动作 token 上累加，再除以有效 token 数归一化。

> **关键优化**：把"前向"和"解码 logits"拆成两步调用 LLM——前向返回 pre-logits，只对需要计损失的位置做 vocab 投影。这是处理大词表（257152）的常见省显存技巧。

---

## 3.6 推理 `sample_actions`：自回归解码

```python
def sample_actions(self, rng, observation, *, max_decoding_steps=256, temperature=0.0):
    observation = preprocess_observation(None, observation, train=False, image_keys=...)
    prefix_emb, prefix_mask, prefix_ar = self.embed_inputs(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar)

    # 1) 左对齐转右对齐（把有效 token 靠右，便于解码续写）
    prefix_emb, prefix_mask, prefix_attn_mask = left_to_right_align(prefix_emb, prefix_mask, prefix_attn_mask)
    # 为 KV 缓存预留 max_decoding_steps 的空间
    prefix_attn_mask = pad(prefix_attn_mask, ((0,0),(0,0),(0, max_decoding_steps)))
    prefix_logits, kv_cache, _ = self.PaliGemma.llm(embedded_prefix=prefix_emb,
                                                    mask=prefix_attn_mask,
                                                    positions=..., decode=True)
    last_logit = prefix_logits[:, -1:]            # 用最后一个位置预测第一个动作 token
    output_tokens = zeros((batch, max_decoding_steps))

    def step(carry):
        ... # 采样 token（temperature=0 时 argmax，否则 categorical）
        token = argmax(last_logit) if temperature == 0 else categorical(last_logit/temperature)
        output_tokens = put_along_last_axis(output_tokens, step, token)
        all_eos = all(any(token == PALIGEMMA_EOS_TOKEN))   # 全部到 EOS 就停
        token_emb = self.PaliGemma.llm(token, embed_only=True)
        last_logit, kv_cache, _ = self.PaliGemma.llm(embedded_prefix=token_emb,
                                                     mask=..., positions=..., decode=True, kv_cache=cache)
        return ..., step + 1

    # 用 while_loop 实现可 jit 的解码循环，遇 EOS 或步数上限停止
    _, _, output_tokens, _, _, _ = jax.lax.while_loop(cond, step, init)
    return output_tokens
```

**算法步骤**：
1. **右对齐**（`left_to_right_align`）：把有效 token 滚动到序列右端，使解码可以在末尾连续续写。
2. **prefill**：用 `decode=True` 一次前向前缀，建立 KV 缓存（预留 `max_decoding_steps` 容量），并用最后位置的 logit 预测第一个动作 token。
3. **逐 token 解码**：每步采样一个 token（贪心或带温度采样），写入 `output_tokens`，再把它嵌入、前向一步、更新 KV 缓存。
4. **提前停止**：当 batch 内所有样本都生成了 EOS（`PALIGEMMA_EOS_TOKEN=1`），或达到步数上限，循环终止。
5. 返回 token 序列——之后由 `ExtractFASTActions` 变换解码回连续动作（见 [04](04-backbone-tokenizers.md) / [05](05-data-pipeline.md)）。

> 注意：`sample_actions` 返回的是**动作 token**，不是连续动作。模型输出 token → 输出变换里的 `ExtractFASTActions` 调用 `FASTTokenizer.extract_actions` 才解码成真正的动作数组。这与 π₀ 直接输出连续动作不同。

---

## 3.7 辅助函数

| 函数 | 作用 |
| --- | --- |
| `make_attn_mask(input_mask, mask_ar)` | 同 [02 章](02-models-flow-matching.md)，用累积和构造块/因果注意力掩码 |
| `left_to_right_align(x, input_mask, attn_mask)` | `@jax.vmap`，把左对齐序列滚动成右对齐（解码前置步骤） |
| `put_along_last_axis(arr, indices, values)` | JAX 缺失的 `np.put_along_axis` 替代，用 one-hot 实现按索引写入 |

---

## 3.8 π₀ vs π₀-FAST 对比小结

| 维度 | π₀ / π₀.₅（流匹配） | π₀-FAST（自回归） |
| --- | --- | --- |
| 动作表示 | 连续向量 | 离散 FAST token |
| 网络 | PaliGemma + 独立动作专家（混合专家） | 单一 PaliGemma（无动作专家） |
| 训练目标 | 速度场 MSE | next-token 交叉熵 |
| 推理 | 欧拉法固定 N 步去噪（并行） | 逐 token 自回归（串行） |
| 输出 | 直接是动作 | token，需 `ExtractFASTActions` 解码 |
| 推理速度 | 快 | 慢 |
| 语言跟随 | 较好 | 更强 |
| 实现 | JAX + PyTorch | 仅 JAX |

下一章 [04 骨干与分词器](04-backbone-tokenizers.md) 深入两类模型共用的 Gemma/SigLIP/LoRA 与各种 Tokenizer。
