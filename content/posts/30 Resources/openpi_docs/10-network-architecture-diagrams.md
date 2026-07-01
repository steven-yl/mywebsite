---
title: "10 模型网络结构框架图（完整版）"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "逐层张量形状标注的 π₀ / π₀.₅ / π₀-FAST 三类模型网络结构框架图与维度对照。"
summary: "openpi 三类模型网络结构框架图与张量维度详解。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 10
series_weight: 10
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 10 模型网络结构框架图（完整版）

> 本章用逐层、带张量形状标注的框架图，完整呈现 π₀ / π₀.₅ / π₀-FAST 三类模型的网络结构。配合 [02](02-models-flow-matching.md)/[03](03-models-pi0-fast.md)/[04](04-backbone-tokenizers.md)/[09](09-pytorch-implementation.md) 章阅读。所有维度均来自仓库实际配置（`gemma.py`、`siglip.py`、`pi0_config.py`）。

---

## 10.1 关键维度速查表

下面所有图中用到的符号与数值（以 π₀.₅ LIBERO 配置为典型示例）：

| 符号 | 含义 | 典型值 |
| --- | --- | --- |
| `B` | batch size | 推理时 1 |
| `H,W` | 图像分辨率 | 224 × 224 |
| `P` | ViT patch 大小 | 14 |
| `N_img` | 每路图像 token 数 | (224/14)² = **256** |
| `N_cam` | 相机路数 | 3（base + 2 wrist） |
| `L_txt` | 语言 token 上限 (`max_token_len`) | π₀=48 / π₀.₅=200 |
| `D_vlm` | PaliGemma 主干宽度 | **2048** |
| `D_act` | 动作专家宽度 | **1024** |
| `D_img` | SigLIP 宽度 | **1152** |
| `AH` | action_horizon（动作步数） | 10 / 15 / 50 |
| `AD` | action_dim（pad 后） | **32** |
| `depth` | Transformer 层数 | VLM/Expert=18, SigLIP=27 |
| `heads/kv` | 注意力头数 / KV 头数 (MQA) | 8 / 1 |
| `head_dim` | 每头维度 | 256 |

### 各骨干网络参数表

| 网络 | variant | width | depth | mlp_dim | heads | kv_heads | head_dim |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SigLIP 视觉塔 | So400m/14 | 1152 | 27 | 4304 | 16 | - | 72 |
| PaliGemma 主干 | gemma_2b | 2048 | 18 | 16384 | 8 | 1 | 256 |
| 动作专家 | gemma_300m | 1024 | 18 | 4096 | 8 | 1 | 256 |

> 注意：PaliGemma 与动作专家**共享 depth(18)、heads(8)、kv_heads(1)、head_dim(256)**——这是它们能在每层共享自注意力计算的前提（见 [04 章](04-backbone-tokenizers.md) §4.1.4）。仅 `width` 与 `mlp_dim` 不同（专家更小）。

---

## 10.2 顶层框架图：π₀ / π₀.₅（流匹配）

```
                                       ┌─────────────────────── 输入 ───────────────────────┐
   3 路图像                          语言指令 prompt            本体状态 state         流匹配:噪声+时间
[B,224,224,3]×3                     "pick up cup"              [B, AD=32]            ε~N(0,I),t~Beta
      │                                  │                         │                      │
      ▼                                  ▼                         │                      │
┌──────────────┐                 ┌──────────────┐                 │                      │
│ SigLIP ViT   │                 │ Gemma Embed  │                 │  ┌───────────────────┴──────────┐
│ (So400m/14)  │                 │  + ×√D       │                 │  │  x_t=t·ε+(1-t)·a (插值轨迹)   │
│ patchify 14² │                 │              │                 │  │  目标速度 u_t = ε - a         │
│ 27 层 Encoder│                 │ [B,L_txt,    │                 │  └───────────────┬──────────────┘
│ pool=none    │                 │     2048]    │                 │                  │ x_t:[B,AH,AD]
└──────┬───────┘                 └──────┬───────┘                 │                  │
       │ [B,256,1152]                   │                         │                  │
       │ (每路图像)                      │                         │                  │
   ┌───▼────────────────────────────────▼───┐         ┌───────────▼──────────┐  ┌────▼─────────────┐
   │  embed_prefix                          │         │  embed_suffix(π0)     │  │ embed_suffix(π0.5)│
   │  拼接 N_cam 路图像token + 语言token      │         │ state_proj:           │  │ 状态进语言流(见左) │
   │  → 前缀序列                              │         │  Linear(32→1024)→token│  │ action_in_proj:   │
   │  prefix: [B, 256·N_cam + L_txt, 2048]   │         │ action_in_proj:       │  │  Linear(32→1024)  │
   │  注意力: 双向 (ar_mask=0)               │         │  Linear(32→1024)      │  │ time→time_mlp     │
   └──────────────────┬──────────────────────┘         │ 时间 sin-cos + 拼接   │  │  →AdaRMS cond     │
                      │                                 │  →action_time_mlp     │  │ suffix tokens     │
                      │                                 │ suffix:[B,1+AH,1024]  │  │ [B,AH,1024]       │
                      │                                 │ adarms_cond=None      │  │ adarms_cond=time  │
                      │                                 └───────────┬───────────┘  └────┬─────────────┘
                      │                                             └────────┬──────────┘
                      │ prefix_embs                                          │ suffix_embs + adarms_cond
                      ▼                                                      ▼
   ╔══════════════════════════════════════════════════════════════════════════════════════╗
   ║          混合专家 Transformer (PaliGemmaWithExpertModel) — 18 层逐层共享注意力           ║
   ║                                                                                        ║
   ║   每层: 前缀走 PaliGemma(2048) 权重, 后缀走 Action Expert(1024) 权重                     ║
   ║         Q/K/V 各自投影 → 拼接 → 统一 self-attention(RoPE,MQA) → 切回 → 各自 o_proj/MLP   ║
   ║         注意力掩码: 前缀双向; 后缀可见前缀, 前缀不可见后缀 (单向条件流)                   ║
   ║         π0.5: 每层 input/post-attn LayerNorm 为 AdaRMS(用 time 调制 scale/shift/gate)    ║
   ╚════════════════════════════════════════════════════╤═══════════════════════════════════╝
                                                         │ 后缀输出 suffix_out[:, -AH:]  [B,AH,1024]
                                                         ▼
                                              ┌────────────────────┐
                                              │ action_out_proj     │
                                              │ Linear(1024→32)     │
                                              └─────────┬──────────┘
                                                        │ v_t (预测速度)  [B,AH,AD=32]
              ┌─────────────────────────────────────────┴───────────────────────────────────┐
              │ 训练: loss = MSE(v_t, u_t)                推理: x ← x + dt·v_t, 欧拉迭代 N=10 步 │
              └────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10.3 SigLIP 视觉塔逐层展开（So400m/14）

```
输入图像  [B, 224, 224, 3]   (值域 [-1,1], float32)
   │
   │  ① Patchify: nn.Conv(out=1152, kernel=14, stride=14, padding=VALID)   ← 在 float32 下做
   ▼
   特征图 [B, 16, 16, 1152]
   │  reshape
   ▼
   patch 序列 [B, 256, 1152]                    (256 = 16×16 个 patch)
   │  ② + 2D sin-cos 位置编码 (posemb)            ← 在 float32 下做
   ▼  cast → dtype_mm (bf16)
   [B, 256, 1152]
   │
   ├─────────────────────────────────────────────┐
   │   ③ Encoder × 27 层 (Encoder1DBlock)          │
   │   ┌─────────────────────────────────────┐    │
   │   │ x                                    │    │
   │   │  → LayerNorm                         │    │
   │   │  → MultiHeadDotProductAttention      │    │  16 heads
   │   │     (self-attn, q=k=v=x)             │    │
   │   │  → + 残差                             │    │
   │   │  → LayerNorm                         │    │
   │   │  → MlpBlock: Dense(4304)→GELU→Dense  │    │
   │   │  → + 残差                             │    │
   │   └─────────────────────────────────────┘    │
   │   (用 nn.scan 折叠 27 层, nn.remat 检查点)     │
   └─────────────────────────────────────────────┘
   │  → LayerNorm (encoder_norm)
   ▼
   编码输出 [B, 256, 1152]
   │  ④ pool_type = "none"  → 不池化, 保留全部 256 个 token
   ▼
   图像 token [B, 256, 1152]   ──►  作为前缀 token 喂入 PaliGemma
```

> 关键：π₀ 用 `pool_type="none"`，每路相机产出 **256 个图像 token**（而非池化成 1 个向量）。3 路相机 = 768 个图像 token 进入前缀。

---

## 10.4 混合专家 Transformer 单层详解（gemma.py / gemma_pytorch.py）

这是整个模型最核心的结构。一层（共 18 层）的内部数据流：

```
   ┌──────────── 前缀专家 (PaliGemma) ────────────┐   ┌──────────── 后缀专家 (Action Expert) ───────────┐
   │  x_prefix  [B, S_pre, 2048]                  │   │  x_suffix  [B, S_suf, 1024]                       │
   └──────────────────┬───────────────────────────┘   └──────────────────┬──────────────────────────────┘
                      │                                                   │
        ① input_layernorm (RMSNorm / AdaRMS)                ① input_layernorm (RMSNorm / AdaRMS)
           π0.5: cond=time → 输出 (x, gate_pre)                 π0.5: cond=time → 输出 (x, gate_suf)
                      │                                                   │
        ② Q/K/V 投影 (PaliGemma 权重, MQA)                  ② Q/K/V 投影 (Expert 权重, MQA)
           q:[B,S_pre,8,256] k,v:[B,S_pre,1,256]               q:[B,S_suf,8,256] k,v:[B,S_suf,1,256]
                      │                                                   │
                      └──────────────┬────────────────────────────────────┘
                                     ▼  沿序列维拼接 Q/K/V
                          q:[B, S_pre+S_suf, 8, 256]   k,v:[B, S_pre+S_suf, 1, 256]
                                     │
                       ③ RoPE 旋转位置编码 (施加到 q,k, float32)
                       ④ q *= head_dim^-0.5 (缩放)
                       ⑤ (推理) 拼接 KV 缓存
                                     │
                       ⑥ MQA 自注意力:
                          logits = einsum(q,k) [float32]
                          masked = where(attn_mask, logits, -2.38e38)  ← 注意力掩码
                          probs = softmax(masked)
                          encoded = einsum(probs, v)
                                     │
                                     ▼  按 [S_pre | S_suf] 切回两个专家
                      ┌──────────────┴──────────────────┐
                      ▼                                  ▼
        ⑦ o_proj (PaliGemma)              ⑦ o_proj (Expert)
        ⑧ 门控残差: x + y·gate_pre        ⑧ 门控残差: x + y·gate_suf
                      │                                  │
        ⑨ post_attn_layernorm(RMSNorm/AdaRMS) → gate    ⑨ post_attn_layernorm → gate
        ⑩ MLP (GeGLU): Dense(16384) 门控 ×              ⑩ MLP (GeGLU): Dense(4096) ...
           Dense(16384) → Dense(2048)                      → Dense(1024)
        ⑪ 门控残差                                       ⑪ 门控残差
                      │                                  │
                      ▼                                  ▼
        x_prefix' [B,S_pre,2048]            x_suffix' [B,S_suf,1024]
   (循环 18 层后, 各自过 final_norm)
```

**注意力掩码可见性矩阵**（`make_attn_mask`，K 维 = 被注意对象，T 维 = 发起查询者）：

```
                    被注意 →  [图像][语言][状态][动作0..AH-1]
   查询 ↓
   [图像 ]                    ✓     ✓     ✗      ✗            前缀: 双向
   [语言 ]                    ✓     ✓     ✗      ✗            (彼此可见)
   [状态 ]                    ✓     ✓     ✓      ✗            后缀: 可见前缀
   [动作i]                    ✓     ✓     ✓   ✓(≤当前块)       前缀看不到后缀
```

含义：图文（前缀）作为条件信息**单向流向**状态/动作（后缀）；前缀内部双向；后缀能看到全部前缀 + 同块动作。这保证了"条件 → 生成"的信息流方向。

---

## 10.5 前缀 / 后缀序列布局（带长度标注）

以 π₀ LIBERO（3 路图像，但右手腕被 mask，AH=10）为例：

```
完整输入序列 = [───────── 前缀 prefix ─────────] [──── 后缀 suffix ────]

前缀 (双向注意力, ar_mask=0):
  ┌────────────┬────────────┬────────────┬──────────────┐
  │ base 图像   │ left_wrist  │ right_wrist │ 语言 token    │
  │ 256 token  │ 256 token  │ 256 token  │ ≤48 token     │
  │ (mask=1)   │ (mask=1)   │ (mask=0)   │              │
  └────────────┴────────────┴────────────┴──────────────┘
       └────────── 每个 token 维度 = 2048 (D_vlm) ──────────┘

后缀 (受控注意力, ar_mask: 状态=1, 动作首=1, 其余=0):
  ┌──────────┬──────────────────────────────┐
  │ 状态 token │ 动作 token × AH=10            │     ← π0 才有独立状态 token
  │ 1 token  │ 10 token                     │     ← π0.5 状态进语言流, 无此 token
  └──────────┴──────────────────────────────┘
       └────── 每个 token 维度 = 1024 (D_act) ──────┘

位置编码 positions = cumsum(input_mask) - 1   (padding 不占位置)
```

> π₀ 与 π₀.₅ 的后缀差异：π₀ 后缀 = `[状态] + [AH 个动作]`（长度 1+AH）；π₀.₅ 后缀 = `[AH 个动作]`（长度 AH），状态已离散化拼进前缀的语言 token。

---

## 10.6 时间步与动作的融合（π₀ vs π₀.₅）

```
═══ π₀: MLP 拼接融合 ═══════════════════════════════════════
  时间 t [B]
    │ create_sinusoidal_pos_embedding (min=4e-3, max=4.0)
    ▼
  time_emb [B, 1024]
    │ 广播到 AH 步: [B, AH, 1024]
    │
  noisy_action [B,AH,32] ─ action_in_proj(32→1024) → [B,AH,1024]
    │                                                    │
    └──────────────── concat ────────────────────────────┘
    ▼  [B, AH, 2048]
  action_time_mlp_in (2048→1024) → SiLU → action_time_mlp_out (1024→1024)
    ▼
  action_time tokens [B, AH, 1024]      adarms_cond = None


═══ π₀.₅: AdaRMS 条件注入 ══════════════════════════════════
  时间 t [B]
    │ create_sinusoidal_pos_embedding
    ▼
  time_emb [B, 1024]
    │ time_mlp_in(1024→1024)→SiLU→time_mlp_out(1024→1024)→SiLU
    ▼
  adarms_cond [B, 1024]  ──────────────► 注入每层 RMSNorm:
                                          modulation = Dense(cond, 3·width)
                                          scale,shift,gate = split(modulation)
                                          normed = normed·(1+scale)+shift
                                          残差用 gate 调制

  noisy_action [B,AH,32] ─ action_in_proj(32→1024) → action tokens [B,AH,1024]
                                          (动作 token 不与时间拼接)
```

---

## 10.7 推理时的去噪 ODE 数据流（π₀ / π₀.₅）

```
   ① 前缀只算一次 (图文不随去噪步变化) → 填 KV 缓存
   ┌────────────────────────────────────────────────────────────┐
   │ embed_prefix → PaliGemma 18 层前向 (use_cache=True)          │
   │ → kv_cache (供后续所有去噪步复用)                            │
   └────────────────────────────────────────────────────────────┘
                              │
   ② 欧拉法迭代 (默认 num_steps=10, dt = -1/10 = -0.1)
   ┌────────────────────────────────────────────────────────────┐
   │ x_t = noise (t=1.0)                                          │
   │ while t >= -dt/2:                                            │
   │   ┌──── denoise_step ──────────────────────────────────┐    │
   │   │ embed_suffix(state, x_t, t) → suffix tokens        │    │
   │   │ Action Expert 18 层前向 (注意 kv_cache 中的前缀)    │    │
   │   │ action_out_proj → v_t  [B,AH,32]                   │    │
   │   └────────────────────────────────────────────────────┘    │
   │   x_t = x_t + dt · v_t       (欧拉步)                        │
   │   t   = t + dt                                              │
   └────────────────────────────────────────────────────────────┘
                              │
   ③ x_0 = 去噪后的动作 [B, AH, 32]  →  输出变换 (Unnormalize + 裁剪维度)

   时间轴:  t=1.0 ──► 0.9 ──► 0.8 ──► ... ──► 0.1 ──► 0.0
            噪声                                      动作
```

---

## 10.8 顶层框架图：π₀-FAST（自回归）

```
   3 路图像                            prompt + 离散状态 + 动作 token (FAST 分词器拼好的序列)
[B,224,224,3]×3                       "Task:... State:<256桶离散> ; Action: <FAST tokens> |"
      │                                              │
      ▼                                              ▼
┌──────────────┐                          ┌────────────────────┐
│ SigLIP ViT   │                          │ Gemma Embedder      │
│ (So400m/14)  │                          │ encode(tokens)      │
│ → [B,256,    │                          │ → [B, L_txt, 2048]  │
│     1152]    │                          └─────────┬──────────┘
└──────┬───────┘                                    │
       │ 图像 token                                  │ 文本+状态+动作 token
       └──────────────────┬─────────────────────────┘
                          ▼  embed_inputs (拼接)
   ┌─────────────────────────────────────────────────────────────┐
   │  完整序列 [B, 256·N_cam + L_txt, 2048]                        │
   │                                                               │
   │  前缀段 (图像+文本+状态): ar_mask=0 双向, loss_mask=False      │
   │  后缀段 (Action: <动作 token> |): ar_mask=1 因果, loss_mask=T │
   └────────────────────────────┬──────────────────────────────────┘
                                ▼
   ╔══════════════════════════════════════════════════════════════╗
   ║   单一 Gemma-2B (gemma_fast.py) — 18 层, 无动作专家             ║
   ║   带增量 KV 缓存 (dynamic_update_slice), 支持逐 token 解码      ║
   ║   注意力: prefix-LM (前缀双向 + 后缀因果)                       ║
   ╚════════════════════════════════════╤═══════════════════════════╝
                                         ▼
                          ┌──────────────────────────────┐
                          │ Embedder.decode (与嵌入表共享) │
                          │ → logits [B, L, vocab=257152] │
                          └──────────────┬───────────────┘
            ┌─────────────────────────────┴────────────────────────────┐
            │ 训练: 仅动作 token 上算交叉熵 (next-token, loss_mask)       │
            │ 推理: 逐 token 采样(argmax/温度) → KV缓存 → 遇 EOS 停        │
            │       输出 token → ExtractFASTActions → 连续动作 [AH, AD]   │
            └────────────────────────────────────────────────────────────┘
```

> 与 π₀ 的根本差异：**单一 Gemma（无混合专家）**，动作是词表里的 token，用自回归生成；输出需经 `ExtractFASTActions` 解码回连续动作。

---

## 10.9 三类模型结构对照（汇总）

```
                  π₀ (流匹配)          π₀.₅ (流匹配)         π₀-FAST (自回归)
              ┌──────────────────┬──────────────────┬──────────────────┐
 视觉塔        │ SigLIP So400m/14 │ SigLIP So400m/14 │ SigLIP So400m/14 │
              ├──────────────────┼──────────────────┼──────────────────┤
 主干          │ PaliGemma 2B     │ PaliGemma 2B     │ PaliGemma 2B     │
 动作专家      │ Gemma 300M ✓     │ Gemma 300M ✓     │ ✗ (无)           │
              ├──────────────────┼──────────────────┼──────────────────┤
 状态入口      │ state_proj→后缀   │ 离散→语言token    │ 离散→语言token    │
 时间注入      │ MLP 拼接         │ AdaRMS           │ ✗ (无时间概念)    │
 动作表示      │ 连续向量         │ 连续向量         │ FAST 离散 token  │
 训练目标      │ 速度场 MSE       │ 速度场 MSE       │ next-token CE    │
 推理          │ 欧拉 10 步去噪    │ 欧拉 10 步去噪    │ 自回归逐 token    │
 输出投影      │ action_out_proj  │ action_out_proj  │ Embedder.decode  │
              └──────────────────┴──────────────────┴──────────────────┘
```

---

## 10.10 参数量与显存（README 数据）

| 组件 | 参数量(约) |
| --- | --- |
| SigLIP So400m | ~400M |
| PaliGemma 主干 (gemma_2b) | ~2B |
| 动作专家 (gemma_300m) | ~311M |

| 模式 | 显存需求 | 示例 GPU |
| --- | --- | --- |
| 推理 | > 8 GB | RTX 4090 |
| LoRA 微调 | > 22.5 GB | RTX 4090 |
| 全参微调 | > 70 GB | A100(80GB)/H100 |

---

## 10.11 小结

- 三类模型共享 SigLIP 视觉塔（256 token/路）+ PaliGemma 主干。
- π₀/π₀.₅ 用「PaliGemma + 动作专家」混合专家，逐层共享自注意力，靠注意力掩码实现单向条件流。
- π₀ 与 π₀.₅ 的结构差异集中在状态入口与时间注入两点。
- π₀-FAST 用单一 Gemma + 自回归，把动作当 token。

回到 [文档索引](README.md)。本章的层级图可与 [04 骨干](04-backbone-tokenizers.md)、[09 PyTorch 实现](09-pytorch-implementation.md) 的代码片段逐一对照。
