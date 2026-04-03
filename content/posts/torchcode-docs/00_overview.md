# 总览：TorchCode 知识架构与学习路径

## 1. 项目定位

TorchCode 是一个以"从零实现 PyTorch 算子"为核心的练习项目，涵盖 40 个深度学习关键组件。每个练习要求不使用高层 API，仅用基础张量操作实现对应算法，从而深入理解底层原理。

## 2. 知识图谱

```
                        ┌─────────────────────────────────────┐
                        │         深度学习核心知识体系          │
                        └──────────────┬──────────────────────┘
           ┌───────────────┬───────────┼───────────┬──────────────────┐
           ▼               ▼           ▼           ▼                  ▼
    ┌──────────┐   ┌──────────┐  ┌──────────┐ ┌──────────┐   ┌──────────┐
    │ 基础组件  │   │ 归一化   │  │ 注意力   │ │ 训练优化  │   │ 推理部署  │
    └────┬─────┘   └────┬─────┘  └────┬─────┘ └────┬─────┘   └────┬─────┘
         │              │             │             │              │
    ┌────┴────┐    ┌────┴────┐   ┌────┴────┐  ┌────┴────┐   ┌────┴────┐
    │ ReLU    │    │LayerNorm│   │ SDPA    │  │ Adam    │   │ Top-k   │
    │ GELU    │    │BatchNorm│   │ MHA     │  │CosLR   │   │ Beam    │
    │ Softmax │    │ RMSNorm │   │ Causal  │  │GradClip│   │ Spec.   │
    │ Linear  │    └─────────┘   │ Cross   │  │GradAcc │   │ Decode  │
    │ Embed   │                  │ GQA     │  │KaimInit│   └─────────┘
    │ Dropout │                  │ Sliding │  │LinReg  │
    │ CE Loss │                  │ Linear  │  └────────┘
    │ Conv2d  │                  │ KVCache │
    └─────────┘                  │ RoPE    │
                                 │ Flash   │
                                 └────┬────┘
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                        ┌──────────┐   ┌──────────┐
                        │ 架构组合  │   │ 高级主题  │
                        ├──────────┤   ├──────────┤
                        │GPT2 Block│   │ BPE      │
                        │SwiGLU MLP│   │ INT8量化  │
                        │ViT Patch │   │ DPO Loss │
                        │ LoRA     │   │ GRPO Loss│
                        │ MoE      │   │ PPO Loss │
                        └──────────┘   └──────────┘
```

## 3. 模块间依赖关系

以下展示了各算法之间的前置知识依赖：

```
ReLU / GELU ──────────────────────────────────────┐
Softmax ──────────────────┐                       │
Linear ──────────┐        │                       │
                 ▼        ▼                       ▼
              Attention (SDPA)              SwiGLU MLP
                 │                              │
        ┌────────┼────────┐                     │
        ▼        ▼        ▼                     │
      MHA    Causal    Cross-Attn               │
        │    Attention     │                    │
        │        │         │                    │
        │   ┌────┴────┐   │                    │
        │   ▼         ▼   │                    │
        │  GQA    Sliding  │                    │
        │  Window          │                    │
        │                  │                    │
        ▼                  ▼                    ▼
    KV Cache          GPT-2 Block ◄────── LayerNorm
        │                  │
        ▼                  ▼
  Speculative         完整 Transformer
   Decoding

RoPE ──► 现代 LLM 注意力（LLaMA 等）
Flash Attention ──► 高效注意力计算
Linear Attention ──► O(n) 复杂度注意力

LoRA ──► 参数高效微调
MoE ──► 稀疏专家模型
ViT Patch ──► 视觉 Transformer

Adam + Cosine LR + Gradient Clipping + Gradient Accumulation ──► 完整训练循环
DPO / GRPO / PPO ──► RLHF 对齐训练
BPE ──► 分词预处理
INT8 量化 ──► 模型压缩部署
```

## 4. 各模块作用总结

### 基础组件层
提供神经网络最底层的计算单元。激活函数（ReLU、GELU、Softmax）引入非线性；Linear 层实现仿射变换；Embedding 将离散 token 映射为连续向量；Dropout 提供正则化；Cross-Entropy 计算分类损失；Conv2d 处理空间特征。

### 归一化层
解决深层网络训练中的内部协变量偏移问题。LayerNorm 按特征维度归一化（Transformer 标配）；BatchNorm 按批次维度归一化（CNN 标配）；RMSNorm 是 LayerNorm 的简化版（LLaMA 等现代 LLM 使用）。

### 注意力机制层
Transformer 的核心。从基础 SDPA 出发，扩展到多头（MHA）、因果掩码（Causal）、交叉注意力（Cross）、分组查询（GQA）、滑动窗口（Sliding Window）、线性注意力（Linear）等变体。KV Cache 和 Flash Attention 分别从推理和训练角度优化效率。RoPE 提供相对位置编码。

### 架构组合层
将基础组件组装为完整模块。GPT-2 Block 是经典 Transformer 解码器块；SwiGLU MLP 是现代 LLM 的前馈网络；ViT Patch Embedding 将图像转为 token 序列；LoRA 实现参数高效微调；MoE 通过稀疏路由扩展模型容量。

### 训练优化层
覆盖训练全流程。Adam 是最常用的优化器；Cosine LR 提供学习率调度；梯度裁剪防止梯度爆炸；梯度累积模拟大 batch；Kaiming 初始化确保训练起步稳定。

### 推理解码层
模型部署时的关键算法。Top-k/Top-p 采样控制生成多样性；Beam Search 寻找高概率序列；Speculative Decoding 利用小模型加速大模型推理。

### 高级主题层
前沿技术。BPE 是主流分词算法；INT8 量化压缩模型体积；DPO/GRPO/PPO 是 RLHF 对齐训练的三种损失函数。

## 5. 归一化方法对比

| 方法 | 归一化维度 | 是否减均值 | 可学习参数 | 典型场景 |
|------|-----------|-----------|-----------|---------|
| LayerNorm | 最后一维（特征） | ✅ | γ, β | Transformer |
| BatchNorm | 第 0 维（批次） | ✅ | γ, β + running stats | CNN |
| RMSNorm | 最后一维（特征） | ❌ | weight | LLaMA, Gemma |

## 6. 注意力变体对比

| 变体 | 复杂度 | 掩码 | KV 头数 | 典型模型 |
|------|--------|------|---------|---------|
| SDPA | O(S²D) | 无 | — | 基础 Transformer |
| MHA | O(S²D) | 无 | = Q 头数 | BERT, GPT-2 |
| Causal | O(S²D) | 上三角 | = Q 头数 | GPT 系列 |
| GQA | O(S²D) | 可选 | < Q 头数 | LLaMA 2, Mistral |
| Sliding Window | O(S·W·D) | 带状 | — | Longformer, Mistral |
| Linear | O(S·D²) | 无 | — | Linear Transformer |
| Flash | O(S²D) 但省内存 | 可选 | — | 所有现代模型 |

## 7. RLHF 损失函数对比

| 方法 | 输入 | 核心思想 | 是否需要 reward model |
|------|------|---------|---------------------|
| DPO | chosen/rejected 对 | 直接从偏好学习，无需显式 reward | ❌ |
| GRPO | 组内多个回复 + reward | 组内归一化 advantage 的 REINFORCE | ✅ |
| PPO | new/old policy + advantage | 裁剪比率防止策略更新过大 | ✅ |

## 8. 推荐学习路径

### 入门路径（基础 → Transformer）
```
ReLU → Softmax → Linear → LayerNorm → Attention → MHA → Causal Attention → GPT-2 Block
```

### 现代 LLM 路径
```
RMSNorm → GQA → RoPE → KV Cache → SwiGLU MLP → Flash Attention → LoRA → MoE
```

### 训练工程路径
```
Cross-Entropy → Kaiming Init → Adam → Cosine LR → Gradient Clipping → Gradient Accumulation
```

### RLHF 对齐路径
```
DPO Loss → PPO Loss → GRPO Loss
```

### 推理部署路径
```
Top-k/Top-p Sampling → Beam Search → Speculative Decoding → INT8 Quantization → BPE
```
