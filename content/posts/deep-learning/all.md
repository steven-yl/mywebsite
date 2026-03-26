---
title: "技术要点"
subtitle: ""
date: 2026-03-23T00:00:00+08:00
draft: true
authors: [Steven]
description: "技术要点"
summary: "技术要点"

tags: []
categories: [Deep Learning]
series: [Deep Learning系列]
weight: 1
series_weight: 1

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""
---




### 一、 核心基础理论
1.  **数学与统计学基石**
    -   **线性代数**：矩阵分解（SVD, Eigendecomposition）、张量运算、低秩近似、特征空间理解（用于注意力机制、图神经网络）。
    -   **微积分与优化**：梯度下降（SGD, Adam, AdamW, Lion）、二阶优化（K-FAC）、梯度消失/爆炸的数学原理、Lipschitz常数约束。
    -   **概率图模型**：变分推断（VAEs）、最大似然估计、贝叶斯深度学习（MC Dropout, 贝叶斯神经网络）、扩散过程的随机微分方程（SDE）视角。
2.  **复杂度理论**
    -   模型参数量与计算量（FLOPs）的精确计算。
    -   时间/空间复杂度分析（Self-Attention的$ O(n^2) $瓶颈与线性注意力机制）。

### 二、 关键模型架构演化
1.  **Transformer 与 大语言模型（LLM）**
    -   **位置编码**：RoPE（旋转位置编码）、ALiBi、相对位置编码的数学原理与长文本外推能力。
    -   **注意力机制变体**：FlashAttention（IO感知的精确注意力）、Grouped-Query Attention (GQA)、Multi-Query Attention (MQA) 的显存优化。
    -   **MoE（混合专家模型）**：稀疏激活、负载均衡损失函数、专家并行策略（如DeepSpeed-MoE）。
    -   **主流架构**：LLaMA系（RMSNorm, SwiGLU）、Mamba（状态空间模型SSM与Transformer的融合趋势）。
2.  **生成式模型**
    -   **扩散模型**：DDPM, DDIM（加速采样）、Latent Diffusion（潜在空间压缩）、Classifier-Free Guidance (CFG) 引导机制、一致性模型（Consistency Models）的单步生成。
    -   **自回归与流模型**：Normalizing Flows、Masked Autoencoders (MAE)。
3.  **图神经网络（GNN）**
    -   消息传递机制（MPNN）、过平滑问题的解决、异构图神经网络（HAN）、图Transformer（Graphormer）。

### 三、 训练策略与优化技术
1.  **分布式训练**
    -   **并行策略**：数据并行（DP）、模型并行（张量并行/流水线并行）、**ZeRO（零冗余优化器）** 的三个阶段（ZeRO-1/2/3）、序列并行（SP）、上下文并行。
    -   **通信原语**：All-Reduce, All-to-All 的通信开销建模与拓扑感知调度。
2.  **微调与对齐**
    -   **参数高效微调（PEFT）**：LoRA及其变体（DoRA, LoRA+）、Prefix Tuning、Adapter 的数学秩分析。
    -   **人类对齐**：RLHF（强化学习+奖励模型+PPO）、DPO（直接偏好优化）的收敛性对比、KTO。
3.  **稳定训练技巧**
    -   梯度裁剪（Norm-based）、学习率预热（Warmup）、权重初始化（Xavier, Kaiming, 以及GPT-2/3使用的特殊的LayerNorm初始化）。
    -   混合精度训练（FP16, BF16, FP8）的动态范围管理与损失缩放（Loss Scaling）。

### 四、 高效推理与部署
1.  **量化与压缩**
    -   **量化技术**：GPTQ（基于Hessian的权重量化）、AWQ（激活感知量化）、FP8训练后量化、KV Cache 量化。
    -   **剪枝**：非结构化剪枝（SparseGPT）、结构化剪枝（移除注意力头或层级）、蒸馏（Logits蒸馏与特征蒸馏）。
2.  **推理加速**
    -   **KV Cache 优化**：PageAttention（vLLM）、PagedAttention 对内存碎片的管理。
    -   **投机性解码**：Speculative Decoding（使用小模型辅助大模型生成）。
    -   **服务框架**：TensorRT-LLM、TGI、SGLang 的调度机制。

### 五、 多模态与跨领域算法
1.  **视觉-语言模型（VLM）**
    -   架构融合：Q-Former（BLIP-2）、MLP Projector（LLaVA）的连接器设计。
    -   细粒度感知：视觉定位（Grounding）、区域级理解（RegionCLIP）。
2.  **具身智能与强化学习**
    -   离线强化学习（CQL, IQL）、模仿学习（BC）、基于模型的强化学习（MBRL）在机器人控制中的应用。
    -   决策Transformer（Decision Transformer）将序列建模引入决策。

### 六、 模型可解释性与安全性
1.  **机制可解释性**
    -   探针（Probing）技术、稀疏自编码器（SAE）提取大模型中的特征（“特征工程”的后现代版本）。
    -   注意力流（Attention Flow）与归因分析（Integrated Gradients, Shapley Value）。
2.  **红队与安全**
    -   **越狱攻击**：GCG（贪婪坐标梯度）等对抗性后缀生成。
    -   **对齐防守**：安全微调、护栏模型（Guardrails）、幻觉检测（如SelfCheckGPT）。

### 七、 工程化落地能力
1.  **框架精通**
    -   **PyTorch**：`torch.compile` 图优化、自定义算子（Cuda Extension, Triton语言）编写。
    -   **大规模框架**：DeepSpeed（ZeRO++, Ulysses）、Megatron-LM、PyTorch FSDP。
2.  **数据工程**
    -   数据合成（利用GPT-4生成高质量微调数据）、去重（MinHash）、毒性过滤与数据配比（Data Mixing）对模型能力的跷跷板效应研究。

### 八、 前沿研究与趋势
-   **长上下文**：1M-10M token 的上下文扩展技术（如位置插值、注意力窗格滑动）。
-   **Agent**：规划（Planning）、工具使用（Tool Use）、多智能体协作。
-   **世界模型**：Sora 类模型的物理模拟能力、Genie 等交互式环境生成。

这些技术要点构成了一个从 **“能够训练出模型”** 到 **“能够在资源受限下高效部署”** ，再到 **“理解模型内部机理并安全控制”** 的完整能力栈。