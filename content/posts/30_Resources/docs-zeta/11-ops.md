---
title: "第 10 章：底层算子（zeta.ops）"
subtitle: ""
date: 2026-07-14T16:00:00+08:00
draft: false
authors: [Steven]
description: "zeta/ops 提供纯函数级数学算子，被 nn 模块与高级用户直接调用。"
summary: "zeta/ops 提供纯函数级数学算子，被 nn 模块与高级用户直接调用。"
tags: [zeta, PyTorch]
categories: [docs zeta]
series: [zeta-docs]
weight: 11
series_weight: 11
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 第 10 章：底层算子（zeta.ops）

`zeta/ops` 提供纯函数级数学算子，被 `nn` 模块与高级用户直接调用。

---

## 1. Softmax 家族（`softmax.py`）

| 函数 | 公式/特点 | 用途 |
|------|-----------|------|
| `standard_softmax` | $\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | 默认 |
| `fast_softmax` | 数值稳定快速实现 | 性能 |
| `sparse_softmax` | 仅 Top-K 非零 | 稀疏注意力 |
| `sparsemax` | 欧氏投影到概率单纯形 | [1605.07704](https://arxiv.org/abs/1605.07704) |
| `temp_softmax` | 温度缩放 $\text{softmax}(x/T)$ | 采样/蒸馏 |
| `logit_scaled_softmax` | logit 缩放变体 | 校准 |
| `local_softmax` | 局部窗口 softmax | 局部归一化 |
| `norm_exp_softmax` | 归一化指数 | 稳定 |
| `selu_softmax` | SELU 后 softmax | 实验 |
| `gumbelmax` | Gumbel-Max 技巧 | 可微离散采样 |

### Gumbel-Max

$$y = \arg\max_i (x_i + g_i), \quad g_i \sim \text{Gumbel}(0,1)$$

用于 MoE 路由、离散选择的可微松弛。

**未导出**：`sparsemax.py` 的 `sparsemax`；`async_softmax.py` 的 `AsynchronizedAttention`。

---

## 2. 多模态 Reshape（`mm_rearranges.py`）

| 函数 | 作用 |
|------|------|
| `reshape_img_to_text` | 图像特征展平为文本序列格式 |
| `reshape_text_to_img` | 文本序列恢复为空间格式 |
| `reshape_video_to_text` | 视频时空→序列 |
| `reshape_audio_to_text` | 音频→序列 |

**为什么需要**：多模态 Transformer 常在统一序列空间操作，需在不同拓扑（BCHW、BLD）间转换。

```python
import torch
from zeta.ops import reshape_img_to_text, reshape_text_to_img

feat = torch.randn(2, 256, 14, 14)  # B, D, H, W
seq = reshape_img_to_text(feat)       # B, H*W, D
feat2 = reshape_text_to_img(seq, h=14, w=14)
```

---

## 3. 矩阵根与分解（`main.py`）

| 函数 | 作用 |
|------|------|
| `matrix_inverse_root` | 矩阵逆 p 次根 $A^{-1/p}$ |
| `matrix_root_diagonal` | 对角化求矩阵根 |
| `_matrix_root_eigen` | 特征分解法 |
| `_matrix_inverse_root_newton` | Newton 迭代法 |
| `compute_matrix_root_inverse_residuals` | 收敛残差 |
| `gram_matrix_new` | Gram 矩阵 |

**用途**：Shampoo/AdaGrad 类预条件优化器、协方差白化。

**Newton 迭代**（矩阵 p 次根）：

$$X_{k+1} = \frac{1}{p}\left[(p-1)X_k + A X_k^{-p+1}\right]$$

---

## 4. 图像布局操作（`main.py`）

| 函数 | 作用 |
|------|------|
| `img_transpose` / `img_transpose_2daxis` | 轴转置 |
| `img_compose_bw` / `img_decompose` | 黑白通道合成/分解 |
| `img_compose_decompose` | 组合操作 |
| `img_width_to_height` | 宽高变换 |
| `img_order_of_axes` | 轴顺序查询 |
| `channel_shuffle_new` | 通道混洗 |
| `squeeze_2d_new` / `unsqueeze_2d_new` | 2D squeeze/unsqueeze |
| `merge_small_dims` | 合并小维度 |
| `multi_dim_cat` / `multi_dim_split` | 多维 cat/split |

---

## 5. Einops 封装

| 模块 | 符号 | 作用 |
|------|------|------|
| `einops_from_to.py` | `EinopsToAndFrom` | 封装 rearrange 进/出 |
| `einops_poly.py` | `rearrange_many`, `reduce_many`, `repeat_many` | 批量 einops |

```python
from zeta.ops import EinopsToAndFrom

layer = EinopsToAndFrom("b (h w) c -> b c h w", h=14, w=14)(some_module)
```

---

## 6. 分布式算子（`dilated_attn_ops.py`）

| 函数 | 作用 |
|------|------|
| `get_rank` / `get_world_size` | 进程 rank/世界大小 |
| `get_data_parallel_group` | DP 进程组 |
| `get_data_parallel_rank` / `get_data_parallel_world_size` | DP rank/size |
| `Allgather` | All-Gather 通信 |
| `all_gather_func` | 函数式 all-gather |
| `padding_to_multiple_of` | 填充到倍数（分布式对齐） |

---

## 7. 其他算子

| 模块 | 符号 | 说明 |
|------|------|------|
| `absmax.py` | `absmax` | 绝对最大值 |
| `unitwise_norm.py` | `unitwise_norm` | 逐单元归一化 |
| `mm_softmax.py` | `mm_softmax` | 多模态 softmax |
| `misc_act.py` | `VPGELU`, `VPReLU` | 向量-参数化激活 |
| `mos.py` | `MixtureOfSoftmaxes` | 混合 softmax（未导出） |
| `laplace.py` | `laplace_solver`, `follow_gradient` | 拉普拉斯求解 |
| `nonlinear.py` | `newtons_method`, `broydens_method` | 非线性方程求解 |
| `expand.py` | `expand` | einops 风格 expand |

---

## 8. 内部模块 `async_softmax`

`AsynchronizedAttention`：异步 softmax 注意力，用于特定并行训练场景。

---

## 9. 可运行示例

```python
import torch
from zeta.ops import (
    standard_softmax,
    gumbelmax,
    unitwise_norm,
    gram_matrix_new,
)

x = torch.randn(4, 16)
p = standard_softmax(x, dim=-1)
g = gumbelmax(x, dim=-1)
n = unitwise_norm(torch.randn(8, 64, 128))
G = gram_matrix_new(torch.randn(32, 64))

print(p.shape, g.shape, n.shape, G.shape)
```

---

## 10. 注意事项

- 导入文件名为 `__Init__.py`（大写 I），在 Linux 等大小写敏感系统可能需修正为 `__init__.py`。
- 多数 ops 为 **无状态纯函数**，可安全用于 `torch.compile`。

---

上一章：[10-models.md](./10-models.md) | 下一章：[12-optim.md](./12-optim.md)
