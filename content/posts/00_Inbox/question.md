---
title: question
date:
draft: false
authors:
  - Steven
description:
summary:
tags:
categories:
series:
weight: 0
series_weight: 0
hiddenFromHomePage: false
seriesNavigation: false
hiddenFromSearch: false
---
## 算法部分

### 1. 为什么要用 Actor-Critic 而不是纯 Critic？
纯值函数方法（pure value-based）难以处理大动作空间或连续动作空间，且只能给出贪婪策略。纯策略梯度方法（pure policy gradient）能处理这些问题，但更新方差很大。Actor-Critic 同时维护一个直接策略和一个 Critic，用 Critic 来降低方差，结合了二者的优点。

### 2. KL 散度和交叉熵、MLE 的关系？
交叉熵 CE(p,q) = KL(p,q) + 熵(p)。最小化其中任何一个都能得到相同的结果。MLE 本质上是数据分布与模型分布之间的 KL 散度最小化。

### 3. 不同 RL 场景应该如何设计 Reward？
主要取决于你构建的环境。可验证的任务（数学、代码）比不可验证的任务（写作）更容易设计奖励，原因显而易见。核心是环境不能容易受到奖励破解（reward hacking）的影响，不应该存在一条优化错误解的捷径。数据和奖励都必须仔细编写。

### 4. 如何理解 RL 中的 importance sampling / rejection sampling 等 Monte Carlo 方法？
重要性采样（importance sampling）用于异步 RL 中，复用为旧策略生成的 off-policy 样本。拒绝采样（rejection sampling）用于数据过滤，比如过滤 SFT 数据或丢弃太简单/太难的样本。广义上它们都是 Monte Carlo 方法——用样本估计期望而不是精确计算。

### 5. PPO / GRPO 的 advantage 是怎么算的，为什么要减去 baseline，这里一定要除以 std 吗？
PPO 有一个 Critic 和一个奖励模型。GRPO 两者都去掉了，它的“Critic”来自多次 rollout 并取均值作为 baseline。Baseline 用来降低方差，避免每个 rollout 都得到正向推动——本质上是在组内形成一种零和效果。标准化除以 std 不是必须的，可以参考 Dr.GRPO。

### 6. RL training 和 test-time scaling 各自是如何 explore 的？
RL 是将领域智能嵌入到权重中。Test-time scaling 是在已有智能的基础上，通过更多的推理 token 来寻找最优解路径并探索不同路径。前者塑造模型权重，后者增加推理预算。

### 7. PPO 是如何 clip 的，为什么要取 min，不 clip 会怎么样，CISPO 是怎么做的？
Clipping 是为了防止策略突变，因为不同样本上的比率（ratio）方差很大。取 min 来自 TRPO 的信任域思想，使更新保持悲观，只在有利时移动。CISPO 不同，它 clip 的是重要性权重而不是整个目标函数，因此梯度仍然流过所有 token，保留更多信号。

### 8. GRPO 为什么加上 KL 散度，KL 散度怎么计算，为什么 DAPO、GSPO 又去掉了 KL 散度？
加上 KL 散度是因为你已经有一个很好的基座模型，拥有不错的智能，你不想偏离太远。但你也想通过 RL 学习新领域。去掉 KL 是因为同样的约束会阻止你移动足够远来真正学习，这在 RLVR（基于规则验证的强化学习）场景中反而是你不希望的。所以移除 KL 有助于稳定性和训练。

### 9. 在 LLM 训练时，如果不小心多 All Reduce 了几次 loss，会发生什么？
（不确知这个问题要问什么）

### 10. DPO 的 reward 是什么，会不会 reward hacking，如何解决？
DPO 的奖励隐含在数据集中（样本标记为好/坏），因此不需要像 PPO 那样单独的奖励模型。奖励破解仍然会发生（如 @willccbb 指出的），因为存在通往目标解的“后门”。这是一个活跃的问题，没有单一的解决方案，必须对环境进行严格的压力测试。可以通过信号如奖励突然跳升或 rollout 长度爆炸来发现。

### 11. 有哪些解决 MoE 训推不一致问题的算法，各自是什么原理？
训练-推理不一致来自几个方面：不同的推理引擎、不同的专家选择逻辑、不同的内核、量化、舍入误差等（参考 Thinking Machines 博客）。一种解决方案是重放（replay）专家选择，使推理和训练选择相同的专家。广义上，你要让 rollout 策略和训练策略在数值上一致，否则重要性权重就会出错。

### 12. RL 训练时，group size / learning rate / ppo epoch / generation length 如何设置？
- Group size：更多的 rollout 意味着更多可学习的路径，常用 8 或 16。
- Learning rate：可管理，通常在 1e-6 量级。
- PPO epoch：通常为 1，更多 epoch 会使数据更 off-policy，破坏稳定性。
- Generation length：纯任务相关，没有任务细节很难量化。

### 13. 相比 GRPO，Dr.GRPO / DAPO / GSPO / CISPO / SAPO / DPPO / MaxRL / SimKO 是如何改进的，各自又有什么缺点？
所有这些算法都是 GRPO 的变体，主要是为了更好的稳定性。在小规模实验室场景下差异不大，但确实存在不同：
- Dr.GRPO：移除标准差归一化。
- DAPO：更高的 clip 范围、动态采样、token-level loss、过长过滤。
- GSPO：序列级重要性采样（而非 token 级），对 MoE 更友好。
- CISPO：clip 权重而不是目标函数。
- MaxRL：解决多样性崩溃（diversity collapse）。
- 关于 SAPO / DPPO 不熟悉（未读论文）。
- SimKO：未具体说明。

### 14. TRPO / DPPO / AReaL 是如何用 trust region 约束 RL objective 的？
TRPO 是约束优化，保持在 KL 边界内。PPO 通过 clip 近似做到同样的事。AReaL 在异步设置中通过限制 staleness / off-policy 程度来强制执行信任域，而不是使用硬 KL 约束。

### 15. RL 能否拓展 LLM 的能力边界？
相关论文表明，RL 主要是在基座模型中已有的能力基础上进行锐化，而不是产生全新的能力。这是一个值得推动的有趣方向，如果有足够多的探索和多样性，不完全确定 RL 不能拓展能力边界。

### 16. 结合 ProRL 等工作，谈谈如何 scale RL 训练边界？
（未阅读该论文）

### 17. OPD 相比于传统 RL / SFT 的改进，有哪些 OPD 的应用？
OPD（Online Preference DPO 或其他？）将 RL 的 on-policy 探索与 SFT 的稠密信号结合。RL 的主要弱点是奖励稀疏，OPD 通过让教师模型为学生自己的 on-policy rollout（重要部分）打分来解决这个问题。用于低成本迁移能力，最近是一个趋势。

### 18. LLM 推理能力是在哪一个训练阶段产生的？
不完全确定推理能力从何产生，但思维链（CoT）起了很大作用。另外，预训练提供基础智能，后训练（post-training）通过推理和测试时计算来放大智能。

### 19. DeepSeek R1 到 V3.2 / V4，RL 部分有哪些改进，MoE RL 有什么不同？
抱歉，不记得 V4 的相关内容。

---

## Infra 部分

### 1. 不考虑 CPU offload，GRPO 训练时显存里有几个模型，考虑了能省多少显存？
取决于算法和 KL 项。对于带 KL 的 GRPO，需要训练策略模型、用于计算 KL 的参考策略模型以及推理模型。内存优化方面：可以去除 KL 项从而省掉一个参考模型。还有一些优化如 NVFP4 量化节省显存；合理的分布式训练（FSDP、EP、TP 将模型和优化器状态分到多卡）帮助很大。还可以量化推理副本并分片优化器状态，使每张卡只持有自己的一部分。

### 2. 分布式推理：KV cache 传输优化、多卡通信优化策略
KV cache 传输在分离式预填充（disaggregated prefill）或解码阶段很重要。优化方式包括：使用 NVLink 或逐层重叠传输 KV/权重。多卡通信取决于是否在同一节点。一般来说，TP（张量并行）帮助很大，并尽量将大的 KV 传输限制在节点内部。

### 3. INT8 与 FP8 优劣对比，训推分别用什么精度
FP8 有更多指数位，更适合训练。INT8 在固定范围内精度更高，更适合推理权重。因此：训练用 FP8，推理服务用 INT8。

### 4. RL rollout 中的长尾问题是什么，有哪些解决方案？
批次内的长尾 rollout 可以通过 PipelineRL、连续批处理（continuous batching）或早期截断等方法解决。常见于异步设置中。

### 5. Continuous batching 在 RL 训练时会有什么问题，vLLM 和 SGLang 的区别？
Continuous batching 将处于不同解码阶段的 rollout 混合在一起进行 next-token 预测。vLLM 使用 PagedAttention，SGLang 使用 RadixAttention（前缀共享）。具体区别请读论文。RL 中需要处理的是：当序列在不同时间结束时，如何对齐完整的轨迹（trajectories）和 logprobs。

### 6. vLLM / SGLang 怎么看利用率，KV cache 在训练里的利用率怎么看？
利用率通过吞吐量（tok/s）、KV cache 利用率和 GPU 利用率衡量。KV cache 看占用率。同时注意内存是否受限。还要观察 rollout 是否被踢出。低 GPU 利用率通常意味着你在等待某些东西（rollout 同步或 CPU）而不是计算密集。两个引擎都暴露了调度器/缓存的相关指标。

### 7. 多机多卡 RL 训练时如何实现反向传播？
和普通反向传播类似，加上一些调整，主要是 FSDP、EP 和 TP。对于长上下文，使用 CP（上下文并行）。理想情况下没人用 PP（流水线并行），它很复杂，且在大多数规模下并非必要。

### 8. RL 训练有哪些异步框架，解决了同步训练的什么问题？
异步 RL 框架如 Prime-RL、VeRL、TRL 等。瓶颈归结为三件事：rollout 的陈旧度（staleness）及如何管理（重要性采样 + staleness 边界）、训练-推理不一致、以及 token-in token-out。解决不一致需要让 rollout 引擎和训练器数值一致、重放专家/路由决策，并在训练器侧而不是推理引擎侧计算 logprobs。

### 9. AReaL 或者其他 partially rollout 框架，在 rollout 时，会不会保存之前 policy 的 KV cache？
不太确定 AReaL，但有些方法会保留旧的 KV cache，有些不会。两者之间似乎没有真正的性能差异。

### 10. MoE 的 EP（专家并行）对 throughput 的影响
EP 对 MoE 模型非常重要：将一批专家放置在一个节点或一组 GPU 上，每个 GPU 持有专家子集。它通过并行化专家 FFN 提高吞吐量，但增加了 all-to-all 通信来将 token 路由到正确的专家。实际收益取决于 all-to-all 与计算的重叠程度以及专家负载是否均衡。

### 11. Long context 场景下的 compute-communication overlap，megatron 和 fsdp 各自的 parallelism
将通信与计算并行化是关键。FSDP 分片梯度/参数/优化器，按需收集，更简单。Megatron 使用 DP、PP 和 TP（3D 并行），通信更显式。对 Megatron 不太熟悉，用得不多。

### 12. 确定性模式怎么开，什么是 batch invariance，是什么导致的，有没有 atom add，atom add 能解决吗？
需要归约顺序、相同的内核和批处理不变的操作。参考 Thinky 博客了解详情。

### 13. AReaL 和 slime 对 RL rollout bottleneck 的理解有什么不同？
可以说 AReaL 的做法是完全异步加上 staleness 控制，将 rollout 生成与训练解耦，这样训练器永远不会等待最慢的 rollout，然后通过最大 off-policy 步数、重要性采样或移除超过策略阈值的样本来控制 staleness。不太确定 Slime，需要进一步阅读。

### 14. full async staleness 怎么看，训练时大概是多少？
Staleness 是生成 rollout 的策略与正在训练的策略之间的差距。通常实验中在 1-4 步落后。更大的 staleness 会导致重要性采样不稳定，因为比率会爆炸。

### 15. slime 里 data 怎么流，megatron 怎么结合，loss 怎么算？
（不确定 Slime）

### 16. VeRL / TRL / Unsloth / AReaL / slime 你会选哪个？
一个都不选，用 PrimeIntellect。
