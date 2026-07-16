---
title: "8. 参考文献与相关资源"
subtitle: ""
date: 2026-07-13T14:00:00+08:00
draft: false
authors: [Steven]
description: "本章汇总 LingBot-VLA 2.0 相关的论文、技术博客、开源项目与工具文档。"
summary: "本章汇总 LingBot-VLA 2.0 相关的论文、技术博客、开源项目与工具文档。"
tags: [lingbotVLA, VLA, robots]
categories: [docs lingbotVLA2, robots]
series: [lingbotVLA2-docs]
weight: 8
series_weight: 8
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 8. 参考文献与相关资源

本章汇总 LingBot-VLA 2.0 相关的论文、技术博客、开源项目与工具文档。

---

## 8.1 核心论文

| 标题 | 说明 | 链接 |
|------|------|------|
| **From Foundation to Application: Improving VLA Models in Practice** | LingBot-VLA 2.0 技术报告 | [arXiv:2607.06403](https://arxiv.org/pdf/2607.06403) |
| Flow Matching for Generative Modeling | 流匹配理论基础 | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) |
| VeOmni | 本仓库分布式训练基础设施 | [arXiv:2508.02317](https://arxiv.org/abs/2508.02317) |

---

## 8.2 VLA 与机器人学习

| 资源 | 说明 | 链接 |
|------|------|------|
| π₀ (Physical Intelligence) | Flow Matching VLA 先驱 | [Blog](https://www.physicalintelligence.company/blog/pi0) |
| OpenVLA | 开源 VLA 基线 | [GitHub](https://github.com/openvla/openvla) |
| RT-2 | Google 机器人 Transformer | [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) |
| Octo | 开源通用机器人策略 | [GitHub](https://github.com/octo-models/octo) |
| ACT | Action Chunking Transformer | [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) |
| Diffusion Policy | 扩散策略 | [GitHub](https://github.com/real-stanford/diffusion_policy) |

---

## 8.3 视觉-语言模型

| 资源 | 说明 | 链接 |
|------|------|------|
| Qwen3-VL | v2.0 骨干 VLM | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| Qwen2.5-VL | v1.0 骨干 | [HuggingFace](https://huggingface.co/Qwen) |
| Qwen-VL 技术报告 | 多模态架构 | [Qwen 文档](https://github.com/QwenLM/Qwen-VL) |

---

## 8.4 深度与几何

| 资源 | 说明 | 链接 |
|------|------|------|
| MoGe | 单目度量几何估计 | [arXiv:2410.19115](https://arxiv.org/abs/2410.19115), [GitHub](https://github.com/microsoft/MoGe) |
| MoGe-2 | v2.0 使用的 MoGe 变体 | [HF: moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) |
| LingBot-Depth (MoRGBD) | RGB-D 深度教师 | [HF: lingbot-vla-v2-6b/depth](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/depth) |
| DINOv2 | 自监督视觉表征 | [GitHub](https://github.com/facebookresearch/dinov2) |
| DINOv3 | 后续 DINO 版本 | [Meta AI](https://ai.meta.com/dinov3/) |

---

## 8.5 MoE 与高效训练

| 资源 | 说明 | 链接 |
|------|------|------|
| DeepSeek-V3 | MoE 负载均衡参考 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| Switch Transformer | 稀疏 MoE 奠基 | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) |
| PyTorch FSDP2 | 全分片数据并行 | [PyTorch Docs](https://pytorch.org/docs/stable/fsdp.html) |
| Flash Attention 2 | 高效注意力 | [GitHub](https://github.com/Dao-AILab/flash-attention) |
| Flex Attention | 可变掩码注意力 | [PyTorch Blog](https://pytorch.org/blog/flexattention/) |

---

## 8.6 数据与仿真

| 资源 | 说明 | 链接 |
|------|------|------|
| LeRobot | 机器人数据集标准 | [GitHub](https://github.com/huggingface/lerobot) |
| RoboTwin 2.0 | 仿真 benchmark | 见 `experiment/robotwin/README.md` |
| Open X-Embodiment | 大规模机器人数据 | [GitHub](https://github.com/google-deepmind/open_x_embodiment) |

---

## 8.7 优化器

| 资源 | 说明 | 链接 |
|------|------|------|
| Muon | 本仓库可选优化器 | 见 `lingbotvla/optim/muon.py` |
| AdamW | 默认优化器 | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |

---

## 8.8 模型权重下载

| 模型 | HuggingFace | ModelScope |
|------|-------------|------------|
| LingBot-VLA 2.0 (6B) | [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) | [Robbyant/LingBot-VLA-V2](https://modelscope.cn/collections/Robbyant/LingBot-VLA-V2) |
| Qwen3-VL-4B-Instruct | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | — |
| MoGe-2 | [Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) | — |

```bash
python scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir lingbot-vla
python scripts/download_hf_data.py   # 训练数据（若需要）
```

---

## 8.9 项目链接

| 类型 | 链接 |
|------|------|
| 论文 | [arXiv:2607.06403](https://arxiv.org/pdf/2607.06403) |
| 项目主页 | [technology.robbyant.com/lingbot-vla-v2](https://technology.robbyant.com/lingbot-vla-v2) |
| GitHub | [Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2) |
| HuggingFace Collection | [robbyant/lingbot-vla-v2](https://huggingface.co/collections/robbyant/lingbot-vla-v2) |
| 许可证 | Apache-2.0 |

---

## 8.10 引用

```bibtex
@article{lingbotvla2,
  title={From Foundation to Application: Improving VLA Models in Practice},
  author={Wei Wu and Fangjing Wang and Fan Lu and He Sun and Shi Liu and Yunnan Wang and Yibin Yan and Yong Wang and Shuailei Ma and Xinyang Wang and Yibin Liu and Shuai Yang and Tianxiang Zhou and Kejia Zhang and Lei Zhou and Cheng Su and Nan Xue and Bin Tan and Han Zhang and Youchao Zhang and Fei Liao and Xing Zhu and Yujun Shen and Kecheng Zheng},
  journal={arXiv preprint arXiv:2607.06403},
  year={2026}
}
```

---

## 8.11 本仓库文档索引

| 文档 | 路径 |
|------|------|
| 文档总索引 | [docs/README.md](./README.md) |
| 总体架构 | [docs/00-overview.md](./00-overview.md) |
| 模型架构 | [docs/01-model-architecture.md](./01-model-architecture.md) |
| Flow Matching | [docs/02-flow-matching.md](./02-flow-matching.md) |
| Dual-Query 蒸馏 | [docs/03-dual-query-distillation.md](./03-dual-query-distillation.md) |
| 数据流水线 | [docs/04-data-pipeline.md](./04-data-pipeline.md) |
| 训练系统 | [docs/05-training-system.md](./05-training-system.md) |
| 推理部署 | [docs/06-inference-deployment.md](./06-inference-deployment.md) |
| 配置参考 | [docs/07-configuration.md](./07-configuration.md) |
| 配置字段表 | [docs/config/lingbotvla_config_doc.md](./config/lingbotvla_config_doc.md) |
| 训练配置指南 | [configs/vla/Training_Config.md](../configs/vla/Training_Config.md) |
| 自定义数据 | [lingbotvla/data/vla_data/README.md](../lingbotvla/data/vla_data/README.md) |

---

## 8.12 推荐阅读路径（外部）

1. **VLA 入门**：RT-2 博客 → OpenVLA README → π₀ 博客
2. **Flow Matching**：arXiv:2210.02747 → LingBot-VLA 2.0 论文 §Action
3. **MoE**：Switch Transformer → DeepSeek-V3 §Load Balancing
4. **机器人数据**：LeRobot 文档 → 本仓库 `vla_data/README.md`
5. **深度估计**：MoGe README → LingBot-Depth 子模块
