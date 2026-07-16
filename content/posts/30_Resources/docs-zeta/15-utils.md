---
title: "第 14 章：工具函数（zeta.utils）"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "第 14 章：工具函数（zeta.utils）。"
summary: "第 14 章：工具函数（zeta.utils）。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 15
series_weight: 15
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 14 章：工具函数（zeta.utils）

## 1. 公开 API 概览

`utils/__init__.py` 导出 40+ 工具，核心来自 `main.py`。

---

## 2. 核心辅助函数（`main.py`）

| 函数 | 作用 |
|------|------|
| `exists` | 非 None 检查 |
| `default` | 默认值回退 |
| `maybe` | 条件执行 |
| `once` | 只执行一次的装饰器 |
| `l2norm` | L2 归一化 |
| `pad_at_dim` | 指定维填充 |
| `cast_tuple` | 标量转元组 |
| `group_by_key_prefix` / `group_dict_by_key` | 字典分组 |
| `pick_and_pop` | 提取并移除键 |
| `print_num_params` | 打印参数量 |
| `cosine_beta_schedule` | 扩散余弦 β 调度 |
| `get_sinusoid_encoding_table` | 正弦编码表 |
| `gumbel_noise` | Gumbel 噪声 |
| `top_p` / `top_k` / `top_a` | 核采样过滤 |
| `init_zero_` | 零初始化 |
| `cast_if_src_dtype` | dtype 转换 |
| `interpolate_pos_encoding_2d` | 2D 位置插值 |
| `ContrastiveTopK` | 对比 Top-K |
| `eval_decorator` | eval 模式装饰器 |
| `log` | 日志辅助 |
| `video_tensor_to_gift` | 视频转 GIF |
| `save_memory_snapshot` | 内存快照 |

---

## 3. 采样函数详解

### 3.1 Nucleus Sampling（top-p）

按概率质量截断：

$$C = \min\{V' \subseteq V : \sum_{w \in V'} p(w) \geq p\}$$

只从累积概率达 $p$ 的最小词集采样。

```python
import torch
from zeta.utils import top_p

logits = torch.randn(1, 1000)
filtered = top_p(logits, thres=0.9)
```

### 3.2 top-k / top-a

- **top-k**：只保留概率最高的 $k$ 个 token
- **top-a**：自适应阈值采样变体

---

## 4. CUDA 与性能

| 模块 | 符号 | 作用 |
|------|------|------|
| `benchmark.py` | `benchmark`, `ProfileConfig` | 微基准测试 |
| `cuda_memory_wrapper.py` | `track_cuda_memory_usage` | CUDA 内存追踪装饰器 |
| `cuda_wrapper.py` | CUDA 构建/检查 | 环境检测 |
| `verbose_execution.py` | `VerboseExecution`, `verbose_execution` | 逐层日志 |

```python
import torch
from torch import nn
from zeta.utils.verbose_execution import verbose_execution

@verbose_execution(log_params=True, log_gradients=True, log_memory=True)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

model = MyModel()
out = model(torch.randn(2, 10))
out.sum().backward()
```

---

## 5. I/O 与持久化

| 模块 | 符号 | 作用 |
|------|------|------|
| `img_to_tensor.py` | `img_to_tensor` | 图像路径→张量 |
| `text_to_tensor.py` | `text_to_tensor` | 文本→张量 |
| `save_load_wrapper.py` | `save_load` | 模型保存/加载装饰器 |
| `module_device.py` | `module_device` | 获取模块设备 |

---

## 6. 类型与日志

| 模块 | 符号 | 作用 |
|------|------|------|
| `enforce_types.py` | `enforce_types` | beartype 运行时类型检查 |
| `disable_logging.py` | `disable_warnings_and_logs` | 静默警告（根包导入时调用） |
| `params.py` | `print_num_params`, `print_main` | 参数统计 |
| `lazy_loader.py` | `LazyLoader`, `lazy_import` | 延迟导入 |

---

## 7. 扩散训练工具

### `cosine_beta_schedule`

噪声调度 $\beta_t$，余弦形式（Improved DDPM）：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

用于 `nn.modules` 中扩散/流匹配模块的训练。

---

## 8. 根包行为

`zeta/__init__.py` 首行调用 `disable_warnings_and_logs()`，导入 zeta 时自动抑制部分警告。若需调试日志，可在导入后手动恢复 logging 级别。

---

上一章：[14-training.md](./14-training.md) | 下一章：[16-nn-modules-catalog.md](./16-nn-modules-catalog.md)
