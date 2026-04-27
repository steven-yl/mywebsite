---
title: "todo"
subtitle: ""
date: 2026-03-25T00:00:00+08:00
draft: false
authors: [Steven]
description: "待补充内容链接与后续待办说明。"
summary: "收集一个外部链接，后续用于整理与更新。"
tags: [todo]
categories: [todo]
series: [todo系列]
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

- https://zhuanlan.zhihu.com/p/686507361
- https://zhuanlan.zhihu.com/p/1912650149282447445
- https://mp.weixin.qq.com/s/7CbHuCTeDRhyY_CoDJtQzA
- https://mp.weixin.qq.com/s/RwYTdWl_3uC3EfT7tUEHjQ
- https://yuantianyuan01.github.io/FastWAM/
- https://analytic-diffusion.github.io/
- https://github.com/duoan/TorchCode


## GitHubDaily
@GitHub_Daily
想搞懂大语言模型内部到底怎么运作的，网上大部分资料过于学术或者太浅只讲概念，找一份既有数学推导又通俗易懂的教程真挺难的。

无独有偶，最近看到 LLM Internals 这个系列教程，从分词原理一路讲到注意力机制再到推理优化，每个知识点都配了逐步数值示例。

整个系列按照大模型的工作流程展开，先讲基础的分词和字节对编码，再深入注意力机制背后的数学原理，最后到实际的推理加速技术。

GitHub：http://github.com/amitshekhariit bhu/llm-internals

还有几篇关于注意力的相关教程，手把手带你算一遍 Q、K、V 矩阵运算，还讲了 KV 缓存、分页注意力、Flash Attention 这些优化手段怎么让模型跑得更快。

以及 Transformer 架构解析、混合专家模型、LoRA 微调等进阶内容也都覆盖了。

教程以博客和视频两种形式提供，内容还在持续更新，想从原理层面理解大模型的同学可以收藏跟着学。


## Yutong (Kelly) He
@electronickale
扩散规划器非常适合离线强化学习。但它们需要很多步骤才能有效工作！对于实时决策来说速度太慢了！
RACTD 在  #ICLR2026  中亮相 ：一种奖励感知型蒸馏方法，只需一步即可完成规划。
 🇧🇷 今天（4 月 23 日）P4-#4618 下午 3:15-5:45
 https://arxiv.org/abs/2506.07822 1/

## Minhyuk Sung
@MinhyukSung
 #ICLR2026  [1/2]
周五上午，请查看 𝗣𝗮𝗶𝗿𝗙𝗹𝗼𝘄，它能够在基于扩散/流的模型中实现更高质量的少步生成，而训练成本仅为原始训练成本的 0.2%–1.7%。

 📅 𝗙𝗿𝗶 𝗔𝗽𝗿 𝟮𝟰 𝗠𝗼𝗿𝗻𝗶𝗻𝗴, 𝗣𝟯-#𝟭𝟴𝟬𝟰
 🌐  网站： https://pair-flow.github.io 



## wesley hsieh
@chengyenhsieh
这位开发者已经重制了许多经典作品，包括 ViT、AlphaFold3、DDPM、Imagen 和 DALL·E。

每当我想要用代码来核对论文的细节时，我最终往往会去看他的实现代码。

一方面，从教育角度来看，他的工作令人印象深刻。另一方面，我很少见到有人做了这么多工作，却在社交媒体上如此沉默。

Lucidrains： https://github.com/lucidrains


## Ksenia_TuringPost
@TheTuringPost
你应该了解的 15 种以上的 LoRA（低秩自适应）变体

 ▪️ 原始 LoRA
 ▪️ QLoRA
 ▪️ DoRA
 ▪️ QDoRA
 ▪️ rsLoRA（排名稳定化）
 ▪️ VeRA（基于向量的随机自适应）
 ▪️ SingLoRA（单矩阵 LoRA）
 ▪️ 灵敏度-LoRA
 ▪️ ARD-LoRA（自适应排序动态）
 ▪️ LoRA 专家混合体
 ▪️ X-LoRA
 ▪️ AutoLoRA
 ▪️ LAG（LoRA 增强生成）
 ▪️ T-LoRA（时间步长相关）
 ▪️ 文本到 LoRA
 ▪️ Doc-to-LoRA
 ▪️ LoRA-Squeeze
 ▪️ 适配器混合物（MoA）

保存此列表，并查看以下资源以了解更多信息： https://turingpost.com/p/loraevolution 




## Sayak Paul
@RisingSayak
如果你在  #ICLR2026  中 ，并且有兴趣讨论扩散模型的有趣特性，那就来看看我们的海报吧！

P3-1710（下午 3:15 起）。

基于 https://cvlab-kaist.github.io/NoiseRefine/