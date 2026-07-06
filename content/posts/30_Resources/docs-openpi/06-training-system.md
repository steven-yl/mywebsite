---
title: "06 训练系统"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "解读 openpi 训练配置、JAX/PyTorch 双训练循环、FSDP、EMA、检查点与权重加载机制。"
summary: "openpi 训练系统：配置注册、训练循环与分布式策略详解。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 6
series_weight: 6
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 06 训练系统

> 本章解读 `training/config.py`（配置中枢）、`training/optimizer.py`、`training/utils.py`、`training/sharding.py`、`training/weight_loaders.py`、`training/checkpoints.py`、`scripts/train.py`（JAX 训练）与 `scripts/train_pytorch.py`（PyTorch DDP 训练）。重点讲清：配置如何驱动一切、两条训练循环的步骤、FSDP/EMA/检查点/权重加载等机制。

---

## 6.1 配置即一切：`TrainConfig`（config.py）

openpi 的设计哲学是"**一个配置完整描述一次实验**"。`TrainConfig` 是所有训练/推理的入口。

```python
@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str                                    # 配置名（唯一，用于引用）
    project_name: str = "openpi"
    exp_name: str = MISSING                       # 实验名（决定检查点目录）
    model: BaseModelConfig = Pi0Config()          # 模型结构
    weight_loader: WeightLoader = NoOpWeightLoader()  # 预训练权重加载器
    pytorch_weight_path: str | None = None        # PyTorch 微调权重路径
    pytorch_training_precision: Literal["bfloat16","float32"] = "bfloat16"
    lr_schedule: LRScheduleConfig = CosineDecaySchedule()
    optimizer: OptimizerConfig = AdamW()
    ema_decay: float | None = 0.99                # EMA 衰减（None 关闭）
    freeze_filter: Filter = nnx.Nothing()         # 冻结哪些参数（LoRA）
    data: DataConfigFactory = FakeDataConfig()    # 数据配置工厂
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000                # 保留 step%keep_period==0 的检查点
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict | None = None
    fsdp_devices: int = 1                          # >1 启用 FSDP
```

**派生属性**：
- `assets_dirs`：`assets_base_dir/name`。
- `checkpoint_dir`：`checkpoint_base_dir/name/exp_name`（无 exp_name 报错）。
- `trainable_filter`：`nnx.All(nnx.Param, nnx.Not(freeze_filter))`——可训练 = 是参数且未被冻结。
- `__post_init__`：禁止同时 `resume` 和 `overwrite`。

### 数据配置层级

`DataConfig`（最终配置）由 `DataConfigFactory`（工厂）生产：

- `DataConfig`：`repo_id`、`asset_id`、`norm_stats`、`repack_transforms`、`data_transforms`、`model_transforms`、`use_quantile_norm`、`action_sequence_keys`、`prompt_from_task`、`rlds_data_dir`、`action_space`、`datasets`。
- `AssetsConfig`：`assets_dir` + `asset_id`，决定从哪加载 norm stats（可指向基础模型检查点）。
- `DataConfigFactory`（抽象）：`create_base_config` 装配 repo_id/asset_id/norm_stats/use_quantile_norm；`_load_norm_stats` 从资产目录加载。
- 具体工厂：
  - `FakeDataConfig`：假数据。
  - `SimpleDataConfig`：可注入 data/model 变换工厂。
  - `LeRobotAlohaDataConfig`：ALOHA（含 repack、delta 动作、adapt_to_pi）。
  - `LeRobotLiberoDataConfig`：LIBERO（含 repack、可选 extra_delta_transform）。
  - `RLDSDroidDataConfig`：DROID 大规模 RLDS。
  - `LeRobotDROIDDataConfig`：自定义小规模 DROID（LeRobot 格式）。
- `ModelTransformFactory`：按模型类型产出 model_transforms（π₀：ResizeImages + TokenizePrompt + Pad；π₀.₅：额外传 discrete_state_input；FAST：TokenizeFASTInputs + ExtractFASTActions）。
- `GroupFactory`（Protocol）：`model_config -> Group` 的工厂接口。

### 配置注册与选择

```python
_CONFIGS = [ TrainConfig(name="pi0_aloha", ...), ..., *roboarena_config.get_roboarena_configs(), ... ]
if len({c.name for c in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {c.name: c for c in _CONFIGS}

def cli() -> TrainConfig:                  # 训练脚本用：tyro 可命令行覆盖
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def get_config(config_name: str) -> TrainConfig:   # 代码内按名取，含拼写纠错建议
    if config_name not in _CONFIGS_DICT: ...difflib 给最接近的名字...
    return _CONFIGS_DICT[config_name]
```

所有可用配置（`pi0_aloha`、`pi05_libero`、`pi0_fast_droid`、`debug` 等）都在 `_CONFIGS` 列表里注册，名字唯一。`cli()` 让命令行能选配置并覆盖任意字段，`get_config()` 供 `compute_norm_stats`、`serve_policy` 等按名取用。

---

## 6.2 优化器与学习率调度（optimizer.py）

### 学习率调度（Protocol `LRScheduleConfig`）
- `CosineDecaySchedule`：warmup + 余弦衰减。字段 `warmup_steps=1000`、`peak_lr=2.5e-5`、`decay_steps=30000`、`decay_lr=2.5e-6`。`create()` 返回 optax 调度。
- `RsqrtDecaySchedule`：warmup + 平方根倒数衰减。字段 `warmup_steps`、`peak_lr`、`timescale`。

### 优化器（Protocol `OptimizerConfig`）
- `AdamW`：`b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0`。`create()` 返回 `optax.chain(clip_by_global_norm, adamw)`（先裁剪梯度范数再更新）。
- `SGD`：`lr, momentum, nesterov`。
- `create_optimizer(optimizer, lr_schedule, weight_decay_mask)`：组合调度与优化器。

> `weight_decay` 设为极小的 `1e-10` 而非 0——注释说明设 0 会莫名 OOM。

---

## 6.3 训练状态与工具（utils.py）

- `TrainState`（`@struct.dataclass`）：训练状态容器。字段：`step`、`params`（nnx.State）、`model_def`（GraphDef）、`opt_state`、`tx`（优化器，非 pytree）、`ema_decay`、`ema_params`。
- `tree_to_info(tree, interp_func)`：把 PyTree 转成可读字符串（日志用）。
- `array_tree_to_info(tree)`：对数组 PyTree 打印 `shape@dtype`。

---

## 6.4 FSDP 分片（sharding.py）

FSDP（全分片数据并行）把参数/梯度/优化器状态切到多卡，用通信换显存。

- 轴定义：`BATCH_AXIS="batch"`、`FSDP_AXIS="fsdp"`、`DATA_AXIS=(batch, fsdp)`。
- `make_mesh(num_fsdp_devices)`：构造 2D 设备网格 `(device_count//fsdp, fsdp)`。
- `set_mesh(mesh)`（contextmanager）：维护全局 mesh 引用（用于激活分片约束）。
- `activation_sharding_constraint(pytree)`：给激活加分片约束。
- `fsdp_sharding(pytree, mesh, min_size_mbytes=4, log)`：核心。决定每个数组怎么分片：
  - fsdp 维为 1 → 全复制（不分片）。
  - 标量/向量（<2 维）→ 复制。
  - 小数组（<4MiB）→ 复制。
  - 大矩阵 → 沿**最大的、能被 fsdp 维整除**的轴分片。
  - 找不到合适轴 → 复制（并告警）。

```
4 张卡，fsdp_devices=2
  → mesh shape (2, 2)：batch 轴 2 组，每组 fsdp 切 2 片
  → 大权重沿某轴切成 2 片分到 2 卡，组间做数据并行
```

---

## 6.5 预训练权重加载（weight_loaders.py）

- `WeightLoader`（Protocol）：`load(params) -> params`，返回结构相同的参数（可加载子集后与参考合并）。
- `NoOpWeightLoader`：原样返回（从头训练）。
- `CheckpointWeightLoader(params_path)`：从检查点加载全部权重（支持训练检查点和发布检查点），并用 `_merge_params(..., missing_regex=".*lora.*")` 补齐缺失的 LoRA 权重（基础模型没有 LoRA 参数，从参考补）。
- `PaliGemmaWeightLoader`：从官方 PaliGemma 检查点加载，覆盖同名权重、保留额外权重（支持动作专家）。
- `_merge_params(loaded, params, missing_regex)`：先取 loaded 中属于参考的权重（按需转 dtype），再按正则补齐缺失键。

> **加载逻辑的精妙处**：基础模型只有 PaliGemma 权重，动作专家和 LoRA 参数是新增的。加载器用"取交集 + 按正则补缺失"的方式，让预训练权重无缝注入到更大的模型结构里。

---

## 6.6 检查点管理（checkpoints.py）

基于 Orbax。

- `initialize_checkpoint_dir(checkpoint_dir, *, keep_period, overwrite, resume)`：创建/复用检查点目录，返回 `(CheckpointManager, resuming)`。处理 overwrite（清空）、resume（标记续训）、冲突（报错）。特例：目录存在但无有效检查点时放弃 resume。
- `save_state(manager, state, data_loader, step)`：保存。把可推理参数单独存为 `params` 项，训练状态存 `train_state`，norm stats 通过 `CallbackHandler` 异步存到 `assets`。
- `restore_state(manager, state, data_loader, step)`：恢复训练状态。
- `load_norm_stats(assets_dir, asset_id)`：加载归一化统计。
- `CallbackHandler` / `CallbackSave` / `CallbackRestore`：异步保存任意函数（这里用于存 norm stats）。
- `_split_params(state)` / `_merge_params(...)`：**EMA 导出技巧**——若有 EMA 参数，导出时把 `ema_params` 作为 `params` 存（推理用 EMA 权重），训练状态里清空；恢复时反向还原。

> 为什么单独存 `params`：推理只需要参数（且优先用 EMA 版本），不需要优化器状态。拆开存让 `create_trained_policy` 能直接加载 `params` 子目录。

---

## 6.7 JAX 训练循环（scripts/train.py）

```python
def main(config):
    init_logging()
    # 1) 校验 batch_size 能被设备数整除
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = NamedSharding(mesh, PartitionSpec(DATA_AXIS))
    # 2) 初始化检查点目录 + wandb
    checkpoint_manager, resuming = initialize_checkpoint_dir(...)
    init_wandb(config, resuming=resuming, ...)
    # 3) 数据加载器
    data_loader = create_data_loader(config, sharding=data_sharding, shuffle=True)
    batch = next(iter(data_loader))
    # 4) 初始化训练状态（含 FSDP 分片、加载预训练权重）
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming: train_state = restore_state(checkpoint_manager, train_state, data_loader)
    # 5) jit 编译 train_step（带分片注解）
    ptrain_step = jax.jit(partial(train_step, config), in_shardings=..., out_shardings=..., donate_argnums=(1,))
    # 6) 主循环
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        if step % log_interval == 0: wandb.log(...)
        batch = next(data_iter)
        if step % save_interval == 0: save_state(...)
```

### `init_train_state`
- 创建优化器，定义 `init` 闭包：建模型 → 合并预训练参数 → 把冻结参数转 bf16 → 构造 `TrainState`（含 EMA 初始化）。
- 用 `jax.eval_shape` 算出状态形状，再用 `fsdp_sharding` 算分片。
- resume 时直接返回形状/分片；否则加载并校验权重，jit 初始化（donate buffer 省显存）。

### `train_step`（核心）
```python
def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params); model.train()
    def loss_fn(model, rng, obs, actions):
        return jnp.mean(model.compute_loss(rng, obs, actions, train=True))
    # 只对可训练参数求梯度（DiffState 过滤冻结参数）
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, obs, actions)
    # 优化器更新
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params); new_params = nnx.state(model)
    new_state = replace(state, step=step+1, params=new_params, opt_state=new_opt_state)
    # EMA 更新
    if state.ema_decay is not None:
        new_state = replace(new_state, ema_params=tree.map(
            lambda old, new: ema_decay*old + (1-ema_decay)*new, state.ema_params, new_params))
    info = {"loss": loss, "grad_norm": global_norm(grads), "param_norm": global_norm(kernel_params)}
    return new_state, info
```

要点：用 `nnx.DiffState` + `trainable_filter` **只对未冻结参数求梯度并更新**（LoRA/冻结主干的关键）；EMA 用指数滑动平均维护一份"平滑"权重。

### 辅助
- `init_logging`：自定义日志格式（级别缩写 D/I/W/E/C）。
- `init_wandb`：初始化 wandb，resume 时读 `wandb_id.txt`。
- `_load_weights_and_validate`：加载并校验权重结构（去掉 ShapeDtypeStruct）。

启动：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

---

## 6.8 PyTorch 训练循环（scripts/train_pytorch.py）

PyTorch 路径用 DDP（数据并行），**无 FSDP/EMA/混合精度/LoRA**。它复用同一套 config 与 data_loader（`framework="pytorch"`）。

### 关键函数
- `setup_ddp()` / `cleanup_ddp()`：初始化/销毁分布式进程组（nccl/gloo），返回 `(use_ddp, local_rank, device)`。
- `set_seed(seed, local_rank)`：各 rank 不同种子。
- `build_datasets(config)`：用统一加载器（`framework="pytorch"`）。
- `get_model_state_dict` / `get_model_parameters`：处理 DDP wrapper。
- `save_checkpoint(...)`：原子保存——临时目录写 `model.safetensors`（safetensors）、`optimizer.pt`、`metadata.pt`、norm stats，再 rename 到最终目录。仅主进程。
- `load_checkpoint(...)`：加载最新检查点（model + optimizer + metadata），含 OOM 处理与显存日志。
- `get_latest_checkpoint_step` / `log_memory_usage`：辅助。
- `train_loop(config)`：主循环（见下）。
- `init_logging` / `init_wandb`：日志与 wandb。

### `train_loop` 步骤
```python
def train_loop(config):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    # 1) 处理 resume/overwrite，建检查点目录，初始化 wandb（主进程）
    # 2) 数据加载器（按 world_size 切分批大小）
    loader, data_config = build_datasets(config)
    # 3) 构造模型（Pi0Config → PI0Pytorch），启用梯度检查点
    model = PI0Pytorch(model_cfg).to(device)
    model.gradient_checkpointing_enable()
    if world_size >= 8: torch.backends... tf32 等优化
    if use_ddp: model = DDP(model, find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=world_size>=8)
    # 4) 加载微调权重（pytorch_weight_path）
    # 5) 优化器 AdamW（参数来自 config.optimizer）+ 手写 cosine LR 调度（对齐 JAX）
    optim = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(b1,b2), eps=eps, weight_decay=wd)
    # 6) 主循环
    while global_step < num_train_steps:
        for observation, actions in loader:
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)
            for pg in optim.param_groups: pg["lr"] = lr_schedule(global_step)
            losses = model(observation, actions)          # 前向（compute_loss）
            loss = losses.mean(); loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.optimizer.clip_gradient_norm)
            optim.step(); optim.zero_grad(set_to_none=True)
            global_step += 1
            save_checkpoint(model, optim, global_step, config, is_main, data_config)
```

- `lr_schedule(step)`：手写的 warmup + 余弦衰减，刻意对齐 JAX 的 `CosineDecaySchedule`（包括 warmup 初值 `peak_lr/(warmup_steps+1)`）。
- 每步把梯度激进清零（`grad.detach_()` + 置 None）以省显存。
- 日志在 `log_interval` 聚合平均 loss/lr/grad_norm。

启动：
```bash
# 单卡
uv run scripts/train_pytorch.py pi05_libero --exp_name my_run --save_interval 1000
# 多卡（单节点）
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name ddp_test
```

---

## 6.9 JAX vs PyTorch 训练对比

| 维度 | JAX (`train.py`) | PyTorch (`train_pytorch.py`) |
| --- | --- | --- |
| 并行 | FSDP + 数据并行 | DDP（仅数据并行） |
| 混合精度 | ✅ 冻结参数 bf16 | ❌ 全 bf16 或全 fp32 |
| EMA | ✅ | ❌ |
| LoRA | ✅（DiffState 冻结） | ❌ |
| 梯度过滤 | nnx DiffState | 全参数训练 |
| 检查点 | Orbax（params/train_state/assets） | safetensors + pt |
| LR 调度 | optax | 手写对齐 JAX |
| 多节点 | ❌ | ✅（torchrun --nnodes） |

---

## 6.10 小结

- `TrainConfig` + 配置注册表是训练的中枢，一个名字唯一确定模型、数据、超参。
- JAX 训练用 FSDP 分片、EMA、Orbax、nnx 梯度过滤实现全功能训练。
- PyTorch 训练用 DDP + safetensors，功能子集，复用同一套配置与数据加载。
- 权重加载器用"交集 + 正则补缺"无缝注入预训练权重；检查点用"拆分 params/EMA"支撑推理。

下一章 [07 推理与服务](07-inference-policy-serving.md) 讲训练好的模型如何封装成可推理的服务。
