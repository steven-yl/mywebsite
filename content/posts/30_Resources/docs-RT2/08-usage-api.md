---
title: "8. 使用指南与 API 参考"
subtitle: ""
date: 2026-07-12T20:00:00+08:00
draft: false
authors: [Steven]
description: "RT-2 安装配置、可运行示例、完整 API 参考与常见问题解答。"
summary: "RT-2 使用指南与 API 参考手册。"
tags: [rt2, robots]
categories: [docs RT2]
series: [rt2-docs]
weight: 9
series_weight: 9
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 8. 使用指南与 API 参考

本章提供安装、可运行示例、完整 API 参考与常见问题解答。

---

## 8.1 环境要求

| 项目 | 要求 |
|------|------|
| Python | ≥ 3.8（推荐 3.10+） |
| PyTorch | ≥ 2.0（Flash Attention 需要） |
| CUDA | 可选，GPU 加速推理/训练 |
| OS | Linux / macOS / Windows |

---

## 8.2 安装

### 8.2.1 从 PyPI 安装

```bash
pip install rt2
```

### 8.2.2 从源码安装

```bash
git clone https://github.com/kyegomez/RT-2.git
cd RT-2
pip install -e .
# 或
pip install -r requirements.txt
```

### 8.2.3 开发依赖

```bash
pip install pytest torch einops beartype zetascale
```

### 8.2.4 核心依赖说明

| 包 | 必需 | 说明 |
|----|------|------|
| `torch` | ✅ | 深度学习框架 |
| `einops` | ✅ | 张量操作（zetascale 依赖） |
| `beartype` | ✅ | 运行时类型检查（zetascale 依赖） |
| `zetascale` | ✅ | ViT + Transformer 组件 |
| `transformers` | ❌ | 可选，用于真实 Tokenizer |
| `deepspeed` | ❌ | 可选，大规模训练 |

---

## 8.3 快速开始

### 8.3.1 最小示例（与 `example.py` 一致）

```python
import torch
from rt2.model import RT2

# 输入
img = torch.randn(1, 3, 256, 256)          # 批次图像
caption = torch.randint(0, 20000, (1, 1024)) # Token ID 序列

# 模型
model = RT2()
model.eval()

# 前向
with torch.no_grad():
    output = model(img, caption)

print(type(output))   # 可能是 tuple (logits, loss) 或 Tensor
print(output[0].shape if isinstance(output, tuple) else output.shape)
# 期望 logits: (1, 1023, 20000) 或 (1, 1024, 20000)
```

### 8.3.2 推荐：显式获取 logits

```python
import torch
from rt2 import RT2

model = RT2()
model.eval()

img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 64))

with torch.no_grad():
    encoded = model.encoder(img, return_embeddings=True)
    logits = model.decoder(text, context=encoded, return_loss=False)

print(logits.shape)  # (1, 63, 20000)
```

### 8.3.3 自回归生成

```python
import torch
from rt2 import RT2
from zeta.utils.main import top_k

model = RT2()
model.eval()

img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 20000, (1, 8))

with torch.no_grad():
    encoded = model.encoder(img, return_embeddings=True)
    generated = model.decoder.generate(
        prompt,
        seq_len=8,
        context=encoded,
        temperature=0.8,
        filter_logits_fn=top_k,
        filter_thres=0.9,
    )

print(generated.shape)  # (1, 8)
```

---

## 8.4 API 参考

### 8.4.1 `RT2`

```python
class RT2(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        encoder_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        num_tokens: int = 20000,
        max_seq_len: int = 1024,
        decoder_dim: int = 512,
        decoder_depth: int = 6,
        decoder_heads: int = 8,
        attn_kv_heads: int = 2,
        use_abs_pos_emb: bool = False,
        cross_attend: bool = True,
        attn_flash: bool = True,
        qk_norm: bool = True,
    ): ...

    def forward(
        self,
        img: torch.Tensor,   # (B, 3, H, W)
        text: torch.Tensor,  # (B, seq_len) long
    ) -> torch.Tensor: ...
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `encoder` | `ViTransformerWrapper` | 视觉编码器 |
| `decoder` | `AutoregressiveWrapper` | 自回归解码器（包装 Transformer） |

#### forward 参数

| 参数 | 形状 | dtype | 必需 |
|------|------|-------|------|
| `img` | `(B, 3, image_size, image_size)` | float32 | ✅ |
| `text` | `(B, seq_len)` | int64 | ✅ |

#### forward 返回

- 默认：`(logits, loss)` 元组（来自 AutoregressiveWrapper）
- `logits`：`(B, seq_len-1, num_tokens)`
- `loss`：标量交叉熵

---

### 8.4.2 导出符号

```python
from rt2 import RT2          # 推荐
from rt2.model import RT2    # 等价
```

---

## 8.5 自定义配置示例

### 8.5.1 更小模型（CPU 友好）

```python
model = RT2(
    image_size=128,
    patch_size=16,
    encoder_dim=256,
    encoder_depth=3,
    encoder_heads=4,
    decoder_dim=256,
    decoder_depth=3,
    decoder_heads=4,
    num_tokens=1000,
    max_seq_len=256,
)
```

### 8.5.2 更大词表

```python
model = RT2(num_tokens=32000, max_seq_len=2048)
```

---

## 8.6 运行测试

```bash
cd RT-2
pytest tests/test.py -v
```

### 测试覆盖清单

| 测试 | 说明 |
|------|------|
| `test_init` | 模型实例化 |
| `test_forward` | 默认形状 |
| `test_forward_different_img_shape` | batch=2 |
| `test_forward_different_text_length` | seq_len=512 |
| `test_forward_different_num_tokens` | vocab=10000 |
| `test_forward_different_max_seq_len` | max_len=512 |
| `test_forward_exception` | 缺 text 报错 |
| `test_forward_no_return_embeddings` | encoder 模式错误 |
| `test_forward_different_*` | 运行时参数修改 |

---

## 8.7 端到端 VLA 推理示例

结合 [05-action-tokenization.md](./05-action-tokenization.md) 中的 `ActionTokenizer`：

```python
import torch
from rt2 import RT2

NUM_ACTION_TOKENS = 256

class RT2Policy:
    def __init__(self, model=None):
        self.model = model or RT2()
        self.model.eval()
        self.action_vocab = torch.arange(NUM_ACTION_TOKENS)

    @torch.no_grad()
    def predict_action_bins(self, img: torch.Tensor, prompt_ids: torch.Tensor, steps=8):
        encoded = self.model.encoder(img, return_embeddings=True)
        tokens = prompt_ids
        bins = []
        for _ in range(steps):
            logits = self.model.decoder.net(
                tokens, context=encoded, return_embeddings=False
            )[:, -1, :]  # (B, vocab)
            action_logits = logits[0, self.action_vocab]
            idx = action_logits.argmax().item()
            token_id = self.action_vocab[idx].item()
            bins.append(token_id)
            tokens = torch.cat([tokens, torch.tensor([[token_id]])], dim=1)
        return bins

# 使用
policy = RT2Policy()
img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 100, (1, 16))
bins = policy.predict_action_bins(img, prompt)
print("predicted bins:", bins)
```

---

## 8.8 常见问题 (FAQ)

### Q1: 输出是 tuple 还是 Tensor？

`AutoregressiveWrapper` 默认 `return_loss=True`，返回 `(logits, loss)`。推理时使用：

```python
logits = model.decoder(text, context=encoded, return_loss=False)
```

### Q2: 图像尺寸必须是 256 吗？

构造时 `image_size` 必须被 `patch_size` 整除。默认 256/32=8 patches per side。

### Q3: 如何加载预训练权重？

本仓库**不提供**预训练 checkpoint。可自行：

```python
state = torch.load("my_rt2_weights.pt")
model.load_state_dict(state)
```

### Q4: 与 PaLM-E 官方实现的关系？

README 称基于 PaLM-E 思想；代码使用 zetascale 的 ViT+Cross-Attn Decoder，是**简化教学/原型实现**，非 Google 官方权重。

### Q5: CUDA OOM 怎么办？

- 减小 `encoder_dim` / `decoder_dim`
- 减小 `max_seq_len` 或 batch size
- 设置 `attn_flash=True`（默认已开）

---

## 8.9 项目结构速查

```
RT-2/
├── rt2/
│   ├── __init__.py       # from rt2.model import RT2
│   └── model.py          # class RT2
├── example.py
├── tests/test.py
├── requirements.txt
├── pyproject.toml
└── docs/
    ├── README.md         # 文档索引
    ├── 00-overview.md
    ├── 01-vla-theory.md
    ├── 02-implementation.md
    ├── 03-vision-encoder.md
    ├── 04-decoder-autoregression.md
    ├── 05-action-tokenization.md
    ├── 06-training-datasets.md
    ├── 07-evaluation.md
    └── 08-usage-api.md   # 本文
```

---

## 8.10 引用

```bibtex
@inproceedings{rt2-2023,
  title={RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control},
  author={Brohan, Anthony and others},
  booktitle={CoRL},
  year={2023}
}
```

---

## 8.11 相关链接

| 资源 | URL |
|------|-----|
| GitHub 仓库 | https://github.com/kyegomez/RT-2 |
| RT-2 论文 | https://arxiv.org/abs/2307.15818 |
| Zeta 库 | https://github.com/kyegomez/zeta |
| 文档索引 | [README.md](./README.md) |
