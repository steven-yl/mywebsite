---
title: "PyTorch 激活函数"
subtitle: ""
date: 2026-03-24T00:00:00+08:00
draft: false
authors: [Steven]
description: "整理 PyTorch 常用激活函数的数学表达、代码实现与可视化对比，便于快速查阅。"
summary: "本文汇总 Sigmoid、Tanh、ReLU、GELU、Swish 等激活函数，并提供分组图与总览图。"
tags: ["PyTorch", "Deep Learning", "Activation Function"]
categories: ["PyTorch"]
series: ["PyTorch 实践指南"]
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: "/mywebsite/posts/images/activation_functions.webp"
---



```python
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Activation Functions
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_fn(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x):
    alpha, scale = 1.6732632423543772, 1.0507009873554805
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def softplus(x):
    return np.log(1 + np.exp(x))

def softsign(x):
    return x / (1 + np.abs(x))

def hardswish(x):
    return x * np.clip(x + 3, 0, 6) / 6

def hardsigmoid(x):
    return np.clip(x / 6 + 0.5, 0, 1)

def hardtanh(x):
    return np.clip(x, -1, 1)

def relu6(x):
    return np.clip(x, 0, 6)

x = np.linspace(-6, 6, 1000)

# ============================================================
# Individual plots (4x4 grid)
# ============================================================

all_funcs = [
    ('Sigmoid',             sigmoid(x),     '#e74c3c', 'output (0,1), classic but vanishing grad'),
    ('Tanh',                tanh_fn(x),     '#3498db', 'output (-1,1), zero-centered'),
    ('Hard Sigmoid',        hardsigmoid(x), '#e67e22', 'piecewise linear approx of Sigmoid'),
    ('Hard Tanh',           hardtanh(x),    '#1abc9c', 'piecewise linear approx of Tanh'),
    ('Softsign',            softsign(x),    '#9b59b6', 'like Tanh, slower tail decay'),
    ('ReLU',                relu(x),        '#2ecc71', 'most common, identity for x>0'),
    ('Leaky ReLU (a=0.01)', leaky_relu(x),  '#f39c12', 'small slope for x<0, avoids dying ReLU'),
    ('PReLU (a=0.25)',      prelu(x),       '#e91e63', 'learnable negative slope'),
    ('ELU',                 elu(x),         '#00bcd4', 'exp decay for x<0, near-zero mean'),
    ('SELU',                selu(x),        '#8bc34a', 'self-normalizing, use with LeCun init'),
    ('ReLU6',               relu6(x),       '#795548', 'ReLU clipped to [0,6], MobileNet'),
    ('GeLU',                gelu(x),        '#ff5722', 'Gaussian Error LU, Transformer default'),
    ('Swish / SiLU',        swish(x),       '#673ab7', 'x*sig(x), smooth, EfficientNet'),
    ('Mish',                mish(x),        '#009688', 'x*tanh(softplus(x)), YOLOv4'),
    ('Softplus',            softplus(x),    '#ff9800', 'smooth ReLU, ln(1+exp(x))'),
    ('Hard Swish',          hardswish(x),   '#607d8b', 'piecewise approx of Swish, MobileNetV3'),
]

n = len(all_funcs)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, (name, y, color, desc) in enumerate(all_funcs):
    ax = axes[i]
    ax.plot(x, y, color=color, linewidth=2.5)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.set_xlim(-6, 6)
    y_min, y_max = y.min(), y.max()
    margin = max((y_max - y_min) * 0.15, 0.5)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.03, desc, transform=ax.transAxes, fontsize=7.5,
            color='#555', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

for j in range(n, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Activation Functions Overview', fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('activation_functions_individual.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'{n} activation functions plotted.')

# ============================================================
# Grouped comparison (3 subplots)
# ============================================================

groups = {
    'Saturating': [
        ('Sigmoid',      sigmoid(x),        '#e74c3c', '-'),
        ('Tanh',         tanh_fn(x),        '#3498db', '-'),
        ('Hard Sigmoid', hardsigmoid(x),    '#e67e22', '--'),
        ('Hard Tanh',    hardtanh(x),       '#1abc9c', '--'),
        ('Softsign',     softsign(x),       '#9b59b6', '--'),
    ],
    'ReLU Family': [
        ('ReLU',           relu(x),           '#2ecc71', '-'),
        ('Leaky ReLU',     leaky_relu(x),     '#f39c12', '-'),
        ('PReLU (a=0.25)', prelu(x),          '#e91e63', '--'),
        ('ELU',            elu(x),            '#00bcd4', '-'),
        ('SELU',           selu(x),           '#8bc34a', '-'),
        ('ReLU6',          relu6(x),          '#795548', '--'),
    ],
    'Smooth': [
        ('GeLU',       gelu(x),       '#ff5722', '-'),
        ('Swish/SiLU', swish(x),      '#673ab7', '-'),
        ('Mish',       mish(x),       '#009688', '-'),
        ('Softplus',   softplus(x),   '#ff9800', '--'),
        ('Hard Swish', hardswish(x),  '#607d8b', '--'),
    ],
}

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for ax, (group_name, funcs) in zip(axes, groups.items()):
    for name, y, color, ls in funcs:
        ax.plot(x, y, label=name, color=color, linestyle=ls, linewidth=2)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_title(group_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Activation Functions by Category', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('activation_functions_grouped.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# All-in-one overlay
# ============================================================

all_flat = [item for g in groups.values() for item in g]

fig, ax = plt.subplots(figsize=(14, 8))
for name, y, color, ls in all_flat:
    ax.plot(x, y, label=name, color=color, linestyle=ls, linewidth=1.8, alpha=0.85)

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_title('All Activation Functions', fontsize=16, fontweight='bold')
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('f(x)', fontsize=13)
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.set_xlim(-6, 6)
ax.set_ylim(-3, 7)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions_all.png', dpi=150, bbox_inches='tight')
plt.show()

```

![Activation Functions](/mywebsite/posts/images/activation_functions.png)
![Activation Functions Category](/mywebsite/posts/images/activation_functions_category.png)
![All Activation Functions](/mywebsite/posts/images/activation_functions_all.png)
