---
title: "Analytic Diffusion Studio — 评估指标与实验流程"
date: 2026-03-27T10:00:00+08:00
draft: false
authors: [Steven]
description: "Analytic Diffusion Studio — 评估指标与实验流程"
tags: [diffusion/flow, Analytic Diffusion Studio]
categories: [diffusion/flow, Analytic Diffusion Studio]
series: [Analytic Diffusion Studio系列]
weight: 12
hiddenFromHomePage: false
hiddenFromSearch: false

summary: "Analytic Diffusion Studio — 评估指标与实验流程"
---

# 12 — 评估指标与实验流程

文件：`src/local_diffusion/metrics.py`、`generate.py`

## 12.1 评估指标

### calculate_r2_score()

```python
def calculate_r2_score(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算 R² 决定系数。"""
    x_flat = x.detach().reshape(x.size(0), -1).cpu()  # [N, n]
    y_flat = y.detach().reshape(y.size(0), -1).cpu()

    var_y = torch.var(y_flat, dim=1)                    # [N]
    ss_res = torch.sum((x_flat - y_flat) ** 2, dim=1)   # [N]

    var_y = torch.where(var_y == 0, torch.ones_like(var_y), var_y)
    r2 = 1 - (ss_res / (var_y * x_flat.size(1)))
    return r2.mean().item()
```

**R² 决定系数**：衡量两组图像的相似程度。

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_j (x_j - y_j)^2}{\text{Var}(y) \cdot n}$$

- $R^2 = 1$：完全一致
- $R^2 = 0$：预测不比均值好
- $R^2 < 0$：预测比均值还差

**用途**：衡量解析方法的预测与 UNet 预测的一致性。

### calculate_mse()

```python
def calculate_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算均方误差。"""
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()
    mse = torch.mean((x_flat - y_flat) ** 2, dim=1)
    return mse.mean().item()
```

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{n} \sum_{j=1}^{n} (x_{ij} - y_{ij})^2$$

先对每个样本计算像素级 MSE，再对批次取平均。

### calculate_l2_distance()

```python
def calculate_l2_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """计算 L2 欧氏距离。"""
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()
    dist = torch.norm(x_flat - y_flat, p=2, dim=1)
    return dist.mean().item()
```

$$L_2 = \frac{1}{N} \sum_{i=1}^{N} \|x_i - y_i\|_2$$

**与 MSE 的区别**：L2 是距离（未平方、未除以维度），MSE 是归一化的平方误差。

## 12.2 实验评估流程

### evaluate_main_model()

主模型评估，在 `generate.py` 中调用：

```python
def evaluate_main_model(model, dataset_bundle, result, cfg, run_paths,
                        wandb_run, sampling_time_total):
```

执行步骤：

1. **计算采样时间指标**
   ```python
   metrics["main_sampling_time_total"] = sampling_time_total
   metrics["main_sampling_time_per_step"] = avg_step_time
   ```

2. **后处理图像**：`[-1, 1]` → `[0, 1]`
   ```python
   processed_images = dataset_bundle.postprocess(images_tensor).detach().cpu()
   ```

3. **保存最终图像**（如果 `save_final_images=True`）
   ```python
   for idx, image in enumerate(processed_images):
       save_image(image, run_paths.images / f"sample_{idx:04d}.png")
   ```

4. **保存图像网格**（如果 `save_image_grid=True`）
   ```python
   grid_tensor = make_grid(processed_images, nrow=4, normalize=False)
   save_image(grid_tensor, run_paths.run_dir / "grid.png")
   ```

5. **保存中间步骤图像**（如果 `save_intermediate_images=True`）
   - 调用 `_save_intermediates()`
   - 保存每个时间步的 $x_t$ 和 $\hat{x}_0$

6. **记录指标**：写入 `metrics.json` 和 WandB

### evaluate_comparison()

与 UNet 基线的对比评估：

```python
def evaluate_comparison(dataset_bundle, main_result, baseline_result,
                        baseline_model, cfg, run_paths, wandb_run):
```

对每个时间步执行：

1. **轨迹对比**：比较同一时间步下两种方法的 $\hat{x}_0$
   ```python
   r2 = calculate_r2_score(main_x0, base_x0)
   mse = calculate_mse(main_x0, base_x0)
   ```

2. **单步对比**：将解析方法的 $x_t$ 送入 UNet
   ```python
   xt = main_result.trajectory_xt[i].to(baseline_model.device)
   base_pred_single = baseline_model.denoise(xt, t)
   r2_single = calculate_r2_score(main_x0, base_pred_single)
   ```

3. **保存对比网格**
   ```python
   _save_comparison_step_grid(dataset_bundle,
       [main_x0, base_pred_single, base_x0], t, comp_dir)
   ```

4. **记录逐步指标**到 WandB 和 JSON

### 对比指标说明

| 指标 | 含义 |
|------|------|
| `r2_score_vs_unet` | 轨迹对比：解析方法 vs UNet（各自采样轨迹） |
| `mse_score_vs_unet` | 同上，MSE 版本 |
| `r2_score_vs_unet_single` | 单步对比：解析方法的 x̂₀ vs UNet 对同一 x_t 的预测 |
| `mse_score_vs_unet_single` | 同上，MSE 版本 |

## 12.3 辅助函数

### _save_intermediates()

```python
def _save_intermediates(dataset, result, run_paths):
    # 保存到 artifacts/intermediate_images/x_t/ 和 x0_pred/
    for i, (xt, x0) in enumerate(zip(result.trajectory_xt, result.trajectory_x0)):
        t_label = result.timesteps[i]
        for j, img in enumerate(dataset.postprocess(xt)):
            save_image(img, xt_dir / f"step_{t_label:04d}_sample_{idx:05d}.png")
```

### _save_comparison_step_grid()

```python
def _save_comparison_step_grid(dataset, trajectories, t, save_dir, filename_suffix):
    """保存对比网格图。trajectories 是多组图像张量的列表。"""
    n = min(8, trajectories[0].shape[0])
    combined_list = []
    for traj in trajectories:
        traj_img = dataset.postprocess(traj[:n])
        combined_list.extend([img for img in traj_img])
    grid = make_grid(combined_list, nrow=n, padding=2)
    save_image(grid, save_dir / f"step_{t:04d}_{filename_suffix}.png")
    return grid
```

### _log_metrics()

```python
def _log_metrics(metrics, run_paths, wandb_run):
    # 1. 读取已有 metrics.json（如果存在）
    # 2. 合并新指标
    # 3. 写回 metrics.json
    # 4. 打印摘要指标到控制台
```

分离摘要指标（标量）和详细指标（嵌套字典）：
- 摘要指标：打印到控制台 + WandB
- 详细指标：仅写入 JSON

## 12.4 WandB 集成

### 初始化

```python
def init_wandb(cfg, run_dir):
    if not cfg.metrics.wandb.enabled:
        return None

    run = wandb.init(
        project=cfg.metrics.wandb.project,
        entity=cfg.metrics.wandb.entity,
        config=config_to_dict(cfg),
        name=cfg.experiment.run_name,
        tags=all_tags,
    )

    # 记录 git 跟踪的代码
    run.log_code(root=project_root, include_fn=_include_only_tracked)
    return run
```

### 记录内容

| 内容 | 方式 |
|------|------|
| 配置 | `wandb.init(config=...)` |
| 图像网格 | `wandb.Image(grid_tensor)` |
| 对比网格 | `wandb.Image(grid)` per step |
| R²/MSE | `wandb_run.log({...}, step=diffusion_step)` |
| 代码快照 | `run.log_code(...)` |

## 12.5 输出目录结构

```
data/runs/{experiment_name}/{run_name}_{timestamp}/
├── config.yaml                    # 完整配置
├── grid.png                       # 生成样本网格
├── metrics.json                   # 所有指标
├── logs/
│   └── generate.log               # 运行日志
├── artifacts/
│   ├── images/                    # 最终样本
│   │   └── sample_0000.png
│   ├── tensors/                   # （预留）
│   └── intermediate_images/
│       ├── x_t/                   # 每步噪声图像
│       │   └── step_0999_sample_00000.png
│       └── x0_pred/               # 每步预测干净图像
│           └── step_0999_sample_00000.png
├── comparison/                    # 对比网格（如果有基线）
│   ├── step_0999_comparison_x0.png
│   └── step_0999_comparison_xt.png
└── code_snapshot.tar.gz           # 代码快照
```

## 12.6 随机种子管理

```python
def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

确保实验可复现。主模型和基线模型使用相同种子的独立 Generator：

```python
generator = torch.Generator(device=model.device)
generator.manual_seed(cfg.experiment.seed)

baseline_gen = torch.Generator(device=model.device)
baseline_gen.manual_seed(cfg.experiment.seed)
```

## 12.7 命令行接口

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*")  # 位置参数
    return parser.parse_args(argv)
```

用法：
```bash
uv run generate.py --config pca_locality/celeba_hq.yaml \
    sampling.num_samples=16 experiment.device=cpu
```
