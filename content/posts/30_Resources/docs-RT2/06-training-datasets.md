---
title: "6. 训练策略与数据集"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "RT-2 训练超参数、数据集组成、模型变体与 Co-Fine-Tuning 策略。"
summary: "RT-2 训练策略与数据集混合详解。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 6. 训练策略与数据集

本章整合 RT-2 论文与本仓库相关的训练超参数、数据集组成、模型变体与 Co-Fine-Tuning 策略。

---

## 6.1 训练范式总览

RT-2 的训练不是从头训练，而是 **Co-Fine-Tuning**：

```
预训练 VLM (Web 规模)
        │
        ▼
混合微调 (Web 数据 + 机器人演示)
        │
        ├── 保留 VLM 能力 (VQA, 推理, 语义)
        └── 习得动作 Token 输出 (低层控制)
```

---

## 6.2 模型变体与超参数

### 6.2.1 论文模型规格

| 模型 | 视觉骨干 | 语言骨干 | 学习率 | Batch Size | 梯度步数 |
|------|----------|----------|--------|------------|----------|
| RT-2-PaLI-X-55B | ViT-22B | UL2-32B (50层 Enc-Dec) | 1e-3 | 2048 | 80K |
| RT-2-PaLI-X-5B | ViT-22B | UL2-32B | 1e-3 | 2048 | 270K |
| RT-2-PaLM-E-12B | ViT-4B (ViT-22B-e) | PaLM-12B (Dec-Only) | 4e-4 | 512 | 1M |
| RT-2-PaLI-3B | ViT-G/14 (2B) | UL2-3B | 1e-3 | 128 | 300K |

超参数继承各自 VLM 原论文（PaLI-X [Chen et al., 2023](https://arxiv.org/abs/2305.18517)、PaLM-E [Driess et al., 2023](https://arxiv.org/abs/2303.03378)）。

### 6.2.2 本仓库默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `encoder_dim` | 512 | 远小于论文 ViT |
| `encoder_depth` | 6 | - |
| `decoder_dim` | 512 | - |
| `decoder_depth` | 6 | - |
| `num_tokens` | 20000 | 通用词表，非 SPM |
| `max_seq_len` | 1024 | - |

**本仓库未提供训练脚本**；以下训练流程描述适用于基于本架构自行实现训练循环的开发者。

---

## 6.3 数据集组成

### 6.3.1 数据集一览

| 数据集 | 描述 | 来源 | RT-2-PaLI-X 占比 | RT-2-PaLM-E 占比 |
|--------|------|------|------------------|------------------|
| **WebLI** | ~10B 图文对，109 语言，过滤 top 10% 得 ~1B 样本 | Chen et al. (2023b), Driess et al. (2023) | 预训练阶段 | 预训练阶段 |
| **Episodic WebLI** |  episodic 格式 Web 数据 | Chen et al. (2023a) | 未用于 PaLI-X co-fine-tune | - |
| **Robotics Dataset** | 移动操作机器人演示，7 技能自然语言标注 | Brohan et al. (2022) RT-1 | **50%** | **66%** |
| **Language-Table** | 仿真推块任务，多种预测任务 | Lynch et al. (2022) | - | - |

### 6.3.2 Robotics Dataset 详情

- **规模**：13 台机器人，17 个月，办公室厨房环境
- **标注**：每条轨迹有自然语言指令（动词 + 名词，如 "pick 7up can"）
- **技能类别**（7 种）：pick, knock, place upright, move, open/close drawer, pick from drawer, place into drawer
- **与 RT-1 相同**：RT-2 直接复用 RT-1 数据协议

### 6.3.3 WebLI 详情

- 来源：网络爬取图文对
- 过滤：跨模态相似度 top 10%
- 用途：VLM 预训练 + Co-Fine-Tuning 防止遗忘

### 6.3.4 Language-Table 详情

- 开源仿真环境：[Language-Table](https://github.com/google-research/language-table)
- RT-2-PaLI-3B 在此 co-fine-tune
- 动作格式：`"X Y"`，$X,Y \in [-10, 10]$
- 结果：90±10% vs RT-1 74±13%

---

## 6.4 Co-Fine-Tuning 实现要点

### 6.4.1 Batch 构造

```python
# 伪代码
def sample_batch(web_loader, robot_loader, robot_weight=0.5):
    if random.random() < robot_weight:
        return next(robot_loader)  # image, prompt, action_tokens
    else:
        return next(web_loader)    # image, vqa_prompt, text_answer
```

### 6.4.2 损失函数

两类样本统一使用交叉熵：

$$
\mathcal{L} = -\sum_t \log P_\theta(y_t \mid y_{<t}, \mathbf{I})
$$

### 6.4.3 数据增强

- 机器人数据：随机裁剪、颜色抖动（遵循 RT-1 协议）
- Web 数据：沿用 PaLI/PaLM-E 原论文增强

---

## 6.5 训练策略对比

| 策略 | 描述 | Unseen Avg (5B) |
|------|------|-----------------|
| **From Scratch** | 不用 VLM 预训练，仅机器人数据 | 9% |
| **Fine-Tuning** | 仅机器人数据微调 VLM | 42% |
| **Co-Fine-Tuning** | Web + Robot 混合 | **44%** (5B) / **63%** (55B) |

**结论**：
1. VLM 预训练至关重要（9% → 42%）
2. 混合 Web 数据进一步防止遗忘（42% → 44%）
3. 模型规模显著影响泛化（44% → 63%）

---

## 6.6 基于本仓库的训练循环示例

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rt2 import RT2

class RobotVLADataset(Dataset):
    """最小机器人 VLA 数据集示例"""
    def __init__(self, samples):
        self.samples = samples  # list of (img_tensor, token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, tokens = self.samples[idx]
        return img, tokens

def train_step(model, img, tokens, optimizer):
    model.train()
    encoded = model.encoder(img, return_embeddings=True)
    logits, loss = model.decoder(tokens, context=encoded, return_loss=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# 初始化
model = RT2()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 假数据
dummy = [(torch.randn(3,256,256), torch.randint(0,256,(16,))) for _ in range(10)]
loader = DataLoader(RobotVLADataset(dummy), batch_size=2)

for epoch in range(3):
    for img, tokens in loader:
        loss = train_step(model, img, tokens, optimizer)
    print(f"epoch {epoch}, loss={loss:.4f}")
```

---

## 6.7 依赖包与训练基础设施

| 包 | 用途 |
|----|------|
| `deepspeed` | 大模型分布式训练（论文规模） |
| `wandb` | 实验跟踪 |
| `transformers` | Tokenizer、预训练权重加载 |
| `palme` / `pali-torch` | 完整 VLM 骨干（非本仓库 core） |

---

## 6.8 推理部署

| 模型规模 | 频率 | 部署 |
|----------|------|------|
| 55B | 1-3 Hz | 多 TPU 云端，机器人通过网络查询 |
| 5B | ~5 Hz | 云端 |
| 3B | ~5 Hz | 可本地/仿真 |

本仓库小规模模型可在单 GPU 实时推理。

---

## 6.9 参考文献与数据资源

| 资源 | 链接 |
|------|------|
| RT-1 数据集说明 | https://arxiv.org/abs/2212.06817 |
| Open X-Embodiment | https://robotics-transformer-x.github.io |
| Language-Table | https://github.com/google-research/language-table |
| WebLI (PaLI) | https://arxiv.org/abs/2305.18517 |

---

## 6.10 相关章节

- 动作 Token 格式 → [05-action-tokenization.md](./05-action-tokenization.md)
- 评估指标 → [07-evaluation.md](./07-evaluation.md)
