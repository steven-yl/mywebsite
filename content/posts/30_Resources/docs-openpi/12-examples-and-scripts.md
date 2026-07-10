# 12 示例、脚本与扩展指南

> 本章覆盖 `examples/`、`scripts/`、`training/misc/` 等入口代码，说明各文件的职责、调用关系与典型用法，补全主文档未单独展开的「工程落地」部分。

---

## 12.1 模块边界

```
scripts/          # CLI 入口（训练/服务/工具）
examples/         # 平台特定集成与演示
training/misc/    # 额外训练配置（PolaRiS、RoboArena 基线）
third_party/      # Git 子模块（aloha、libero 仿真环境）
packages/         # 独立客户端包 openpi-client
```

---

## 12.2 脚本入口（scripts/）

### `train.py` — JAX 训练主入口

**职责**：完整 JAX 训练循环（FSDP、EMA、Orbax、wandb）。

**调用链**：
```
tyro CLI → TrainConfig
  → create_data_loader(config, framework="jax")
  → init_train_state → train_step (jit)
  → save_state (Orbax)
```

**典型用法**：
```bash
# 提高 JAX GPU 显存占用上限
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
  --exp-name=my_experiment --overwrite

# 多卡 FSDP（2 卡示例）
uv run scripts/train.py pi05_libero --exp-name=fsdp_run --fsdp-devices 2
```

**关键函数**（`scripts/train.py`）：

| 函数 | 作用 |
| --- | --- |
| `init_logging()` | 自定义日志格式 |
| `init_wandb(config, resuming, ...)` | 初始化 Weights & Biases |
| `_load_weights_and_validate(loader, params_shape)` | 加载并校验预训练权重 |
| `init_train_state(config, init_rng, mesh, resume)` | 初始化分片训练状态 |
| `train_step(config, rng, state, batch)` | 单步训练（loss、梯度、EMA） |
| `main(config)` | 主循环 |

详见 [06 训练系统](06-training-system.md)。

---

### `train_pytorch.py` — PyTorch DDP 训练

**职责**：π₀/π₀.₅ 的 PyTorch 全参微调（单卡/多卡/多节点）。

**限制**：无 FAST、LoRA、FSDP、EMA、混合精度。

**典型用法**：
```bash
# 单卡
uv run scripts/train_pytorch.py pi05_libero --exp_name my_run

# 多卡 DDP
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi0_aloha_sim --exp_name ddp_test
```

**关键函数**：

| 函数 | 作用 |
| --- | --- |
| `setup_ddp()` / `cleanup_ddp()` | 分布式进程组 |
| `build_datasets(config)` | 创建 PyTorch 数据加载器 |
| `save_checkpoint(...)` | 原子写入 safetensors + optimizer |
| `load_checkpoint(...)` | 恢复训练 |
| `train_loop(config)` | 主训练循环 |
| `lr_schedule(step)` | 手写 cosine+warmup（对齐 JAX） |

---

### `serve_policy.py` — 策略 WebSocket 服务

**职责**：加载检查点并启动 `WebsocketPolicyServer`。

```bash
# 指定检查点
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# 使用环境默认预训练模型
uv run scripts/serve_policy.py --env DROID
```

**关键类型**：

| 类型/函数 | 作用 |
| --- | --- |
| `EnvMode` | `ALOHA` / `ALOHA_SIM` / `DROID` / `LIBERO` |
| `Checkpoint` | 自定义 config + checkpoint 路径 |
| `Default` | 使用 `DEFAULT_CHECKPOINT` 映射 |
| `create_default_policy(env, default_prompt)` | 按环境加载默认模型 |
| `create_policy(args)` | 分派 Checkpoint/Default |
| `main(args)` | 启动服务，可选 `PolicyRecorder` |

---

### `compute_norm_stats.py` — 归一化统计

**职责**：训练前遍历数据集，计算 `state`/`actions` 的 mean/std/q01/q99。

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

输出：`assets/<config_name>/<repo_id>/norm_stats.json`

**关键组件**：
- `RemoveStrings`：去掉字符串字段（JAX 不支持）
- `main(config_name, max_frames)`：遍历 + `RunningStats` 累计

详见 [norm_stats.md](../norm_stats.md) 与 [05 数据管线](05-data-pipeline.md) §5.9。

---

### `scripts/docker/`

| 文件 | 作用 |
| --- | --- |
| `compose.yml` | Docker Compose 编排 |
| `serve_policy/Dockerfile` | 策略服务镜像 |
| `install/install.sh` | 容器内依赖安装 |

详见 [docker.md](../docker.md)。

---

## 12.3 示例目录（examples/）

### 按平台分类

| 目录 | 平台 | 主要内容 |
| --- | --- | --- |
| `simple_client/` | 无机器人 | 随机观测测试远程/本地推理 |
| `aloha_sim/` | ALOHA 仿真 | 仿真环境 + Runtime 闭环 |
| `aloha_real/` | ALOHA 真机 | 真实双臂硬件集成 |
| `droid/` | DROID Franka | 推理 + 全量 RLDS 训练说明 |
| `libero/` | LIBERO benchmark | 数据转换、微调、Docker 评测 |
| `ur5/` | UR5 | 集成说明（笔记性质） |

### `examples/simple_client/` — 最小推理测试

无需机器人，向策略服务发送随机观测：

```bash
uv run examples/simple_client/main.py --host localhost --port 8000
```

适合验证：服务连通性、msgpack 协议、动作 shape。

### `examples/libero/` — 微调完整流程

| 文件 | 作用 |
| --- | --- |
| `convert_libero_data_to_lerobot.py` | RLDS → LeRobot 格式转换 |
| `main.py` | LIBERO 评测主脚本 |
| `README.md` | Docker 化评测 workflow |

典型流程（与 README 一致）：
1. `compute_norm_stats.py --config-name pi05_libero`
2. `train.py pi05_libero --exp-name=...`
3. `serve_policy.py` + LIBERO eval

### `examples/droid/` — DROID 大规模训练

| 文件 | 作用 |
| --- | --- |
| `README.md` | 真机推理步骤 |
| `README_train.md` | 全 DROID 数据集 RLDS 训练（需 `uv sync --group rlds`） |
| `main.py` | DROID 推理客户端 |

**空闲帧过滤**：`DroidRldsDataset` 使用过滤字典剔除静止帧，见 [05 章](05-data-pipeline.md) §5.8。

### `examples/inference.ipynb` — 交互式推理

Jupyter notebook，逐步演示 `create_trained_policy` 与 `infer`。

### `examples/convert_jax_model_to_pytorch.py` — 权重转换

JAX Orbax 检查点 → PyTorch safetensors：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /path/to/jax/checkpoint \
  --config_name pi05_libero \
  --output_path /path/to/pytorch/checkpoint
```

转换后在 config 中设置 `pytorch_weight_path` 或直接作为推理检查点目录（含 `model.safetensors`）。

---

## 12.4 额外训练配置（training/misc/）

### `polaris_config.py`

PolaRiS benchmark 基线配置，例如：
- `pi05_droid_jointpos_polaris`
- 关节位置空间下的 DROID 评测

### `roboarena_config.py`

RoboArena 基线配置，使用 `BinningTokenizer` / `FSQTokenizer`：
- `paligemma_binning_droid`
- FSQ 变体配置

这些配置通过 `config.py` 的 `_CONFIGS` 注册，与 π₀ 系列主模型独立。

---

## 12.5 第三方子模块（third_party/）

| 子模块 | 用途 | 初始化 |
| --- | --- | --- |
| `third_party/aloha` | ALOHA 仿真/硬件相关 | `git submodule update --init --recursive` |
| `third_party/libero` | LIBERO 仿真 benchmark | 同上 |

---

## 12.6 扩展新机器人平台 checklist

若要将 openpi 适配到新机器人，通常需要：

1. **定义 Policy 变换**（参考 `libero_policy.py`）：
   - `MyRobotInputs`：原始观测 → `{image, image_mask, state, prompt}`
   - `MyRobotOutputs`：模型输出 → 机器人动作维度

2. **注册 DataConfig**（`training/config.py`）：
   - `LeRobotMyRobotDataConfig`：repack + data_transforms + model_transforms

3. **注册 TrainConfig**：
   ```python
   TrainConfig(
       name="pi05_myrobot",
       model=Pi0Config(pi05=True, action_dim=32, action_horizon=...),
       data=LeRobotMyRobotDataConfig(...),
       weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
   )
   ```

4. **计算 norm stats**：`compute_norm_stats.py --config-name pi05_myrobot`

5. **训练与部署**：`train.py` → `serve_policy.py` → `openpi-client`

**关键原则**：训练与推理必须使用同一套 `Group` 变换（见 [05 章](05-data-pipeline.md)）。

---

## 12.7 测试与 CI

| 路径 | 覆盖 |
| --- | --- |
| `src/openpi/transforms_test.py` | Transform 组合与 `transform_dict` |
| `src/openpi/models/model_test.py` | 模型抽象 |
| `src/openpi/models/pi0_test.py` | Pi0 前向/采样 |
| `src/openpi/models/tokenizer_test.py` | 分词器 |
| `src/openpi/policies/policy_test.py` | Policy 推理 |
| `src/openpi/training/data_loader_test.py` | 数据加载 |
| `scripts/train_test.py` | 训练 smoke test |
| `.github/workflows/` | CI 自动化 |

---

## 12.8 小结

- `scripts/` 提供训练、服务、统计三条主路径；`examples/` 提供平台特定集成模板。
- 扩展新平台的核心工作是 **Policy 变换 + DataConfig + TrainConfig 注册**，无需修改模型核心。
- JAX 与 PyTorch 训练共用 config 与 data_loader，检查点格式不同（Orbax vs safetensors）。

返回 [文档索引](README.md)。
