---
title: "PyTorch lr曲线"
subtitle: ""
date: 2026-03-24T00:00:00+08:00
draft: false
authors: [Steven]
description: "lr曲线图"
summary: "lr曲线图"
tags: ["PyTorch", "Deep Learning", "LR"]
categories: ["PyTorch"]
series: ["PyTorch 实践指南"]
weight: 1
series_weight: 1
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---



```python
# ============================================================
# PyTorch Learning Rate Scheduler Visualization
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Helper: simulate scheduler over N steps, record lr
# ------------------------------------------------------------
def get_lr_curve(scheduler_fn, num_steps=100):
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = scheduler_fn(optimizer)
    lrs = []
    for _ in range(num_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    return lrs

def get_lr_curve_plateau(num_steps=100):
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    lrs = []
    for i in range(num_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        fake_loss = 1.0 / (1 + i * 0.05) + 0.05 * np.sin(i * 0.3)
        if 30 < i < 60:
            fake_loss = 0.4
        optimizer.step()
        scheduler.step(fake_loss)
    return lrs

NUM_EPOCHS = 100

# ------------------------------------------------------------
# Define all schedulers
# ------------------------------------------------------------
schedulers = {
    'StepLR\n(step_size=30, gamma=0.1)': lambda opt: optim.lr_scheduler.StepLR(
        opt, step_size=30, gamma=0.1
    ),
    'MultiStepLR\n(milestones=[30,60,80])': lambda opt: optim.lr_scheduler.MultiStepLR(
        opt, milestones=[30, 60, 80], gamma=0.1
    ),
    'ExponentialLR\n(gamma=0.95)': lambda opt: optim.lr_scheduler.ExponentialLR(
        opt, gamma=0.95
    ),
    'CosineAnnealingLR\n(T_max=100, eta_min=0)': lambda opt: optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=NUM_EPOCHS, eta_min=0
    ),
    'CosineAnnealingWarmRestarts\n(T_0=25, T_mult=2)': lambda opt: optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=25, T_mult=2, eta_min=0.001
    ),
    'OneCycleLR\n(max_lr=0.1, total_steps=100)': lambda opt: optim.lr_scheduler.OneCycleLR(
        opt, max_lr=0.1, total_steps=NUM_EPOCHS
    ),
    'LambdaLR\n(lr_lambda=0.95^epoch)': lambda opt: optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda epoch: 0.95 ** epoch
    ),
    'LinearLR\n(start_factor=0.1, total_iters=30)': lambda opt: optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=30
    ),
    'PolynomialLR\n(total_iters=100, power=2)': lambda opt: optim.lr_scheduler.PolynomialLR(
        opt, total_iters=NUM_EPOCHS, power=2
    ),
}

# Collect curves
curves = {}
for name, fn in schedulers.items():
    curves[name] = get_lr_curve(fn, NUM_EPOCHS)
curves['ReduceLROnPlateau\n(factor=0.5, patience=8)'] = get_lr_curve_plateau(NUM_EPOCHS)

epochs = np.arange(NUM_EPOCHS)

# ============================================================
# Figure 1: Individual plots (2x5 grid)
# ============================================================
colors = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
    '#1abc9c', '#f39c12', '#00bcd4', '#ff5722', '#607d8b',
]

fig, axes = plt.subplots(2, 5, figsize=(25, 8))
axes = axes.flatten()

for i, (name, lrs) in enumerate(curves.items()):
    ax = axes[i]
    ax.plot(epochs, lrs, color=colors[i % len(colors)], linewidth=2.2)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Learning Rate', fontsize=9)
    ax.set_xlim(0, NUM_EPOCHS)
    ax.set_ylim(-0.005, max(lrs) * 1.15)
    ax.grid(True, alpha=0.3)

plt.suptitle('PyTorch Learning Rate Schedulers', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('lr_schedulers_individual.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 2: Grouped comparison (3 panels)
# ============================================================
groups = {
    'Step-based Decay': [
        'StepLR\n(step_size=30, gamma=0.1)',
        'MultiStepLR\n(milestones=[30,60,80])',
        'ExponentialLR\n(gamma=0.95)',
        'ReduceLROnPlateau\n(factor=0.5, patience=8)',
    ],
    'Cosine & Cyclic': [
        'CosineAnnealingLR\n(T_max=100, eta_min=0)',
        'CosineAnnealingWarmRestarts\n(T_0=25, T_mult=2)',
        'OneCycleLR\n(max_lr=0.1, total_steps=100)',
    ],
    'Smooth & Custom': [
        'LambdaLR\n(lr_lambda=0.95^epoch)',
        'LinearLR\n(start_factor=0.1, total_iters=30)',
        'PolynomialLR\n(total_iters=100, power=2)',
    ],
}

line_styles = ['-', '--', '-.', ':', '-', '--']

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for ax, (group_name, members) in zip(axes, groups.items()):
    for j, name in enumerate(members):
        lrs = curves[name]
        label = name.split('\\n')[0]
        ax.plot(epochs, lrs, label=label,
                color=colors[j % len(colors)],
                linestyle=line_styles[j % len(line_styles)],
                linewidth=2)
    ax.set_title(group_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(0, NUM_EPOCHS)
    ax.grid(True, alpha=0.3)

plt.suptitle('LR Scheduler Comparison by Category', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('lr_schedulers_grouped.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Figure 3: All-in-one overlay
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

for i, (name, lrs) in enumerate(curves.items()):
    label = name.split('\\n')[0]
    ax.plot(epochs, lrs, label=label,
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=1.8, alpha=0.85)

ax.set_title('All LR Schedulers Overview', fontsize=16, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Learning Rate', fontsize=13)
ax.legend(fontsize=9, loc='upper right', ncol=2)
ax.set_xlim(0, NUM_EPOCHS)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr_schedulers_all.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'Plotted {len(curves)} LR schedulers')

```

![lr Functions](/mywebsite/posts/images/lr.png)
![lr Category](/mywebsite/posts/images/lr_category.png)
![All lr](/mywebsite/posts/images/lr_all.png)
