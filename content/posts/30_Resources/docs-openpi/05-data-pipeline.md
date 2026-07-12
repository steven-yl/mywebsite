---
title: "05 数据管线：Transform 体系、归一化与数据加载"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "解读 Transform 体系、归一化统计、机器人 policy 适配与 LeRobot/RLDS 数据加载。"
summary: "openpi 数据管线：Transform、归一化与 DataLoader 详解。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 5
series_weight: 5
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---

# 05 数据管线：Transform 体系、归一化与数据加载

> 本章解读 `transforms.py`（变换核心）、`shared/normalize.py`（归一化统计）、机器人适配 policy（`aloha_policy.py`/`libero_policy.py`/`droid_policy.py`）、`training/data_loader.py`（数据加载）与 `training/droid_rlds_dataset.py`（RLDS）。这是训练与推理**共享**的数据"翻译层"。

---

## 5.1 为什么需要一套统一的 Transform 体系

不同机器人（ALOHA、DROID、LIBERO）的数据格式千差万别：键名不同、状态维度不同、夹爪表示不同、图像通道顺序不同。模型却要求统一的输入（`Observation`）。同时，训练时数据来自数据集，推理时来自机器人，二者必须用**完全相同**的变换，否则会产生训练/部署分布漂移。

openpi 的解法：把每一步数据处理抽象成 `DataTransformFn`，组合成可复用的管线，训练和推理共用。

```python
@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict: ...   # 输入/输出都是嵌套 dict
```

约定：每个 transform 处理**未批量化**的单样本，叶子是 numpy 数组。

---

## 5.2 变换的组合：`Group` 与 `CompositeTransform`

```python
@dataclasses.dataclass(frozen=True)
class Group:
    inputs: Sequence[DataTransformFn] = ()    # 输入侧变换
    outputs: Sequence[DataTransformFn] = ()   # 输出侧变换

    def push(self, *, inputs=(), outputs=()) -> "Group":
        # inputs 追加到末尾，outputs 追加到开头（非对称！）
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))
```

**`push` 的非对称设计是精髓**：输入变换从前往后加，输出变换从后往前加。这样无论叠加多少层变换，输出侧始终以"撤销"的顺序执行，保证输入/输出严格镜像。例如加一个 `DeltaActions`（输入侧转 delta）时，对应的 `AbsoluteActions`（输出侧转回 absolute）会被放到输出序列最前，先于其它输出变换执行。

- `CompositeTransform(transforms)`：把一组变换顺序串成一个。
- `compose(transforms)`：返回 `CompositeTransform` 的便捷函数。

---

## 5.3 完整变换管线（在 `create_trained_policy` 中组装）

推理时的输入/输出管线（训练时类似，见 §5.7）：

```
【输入侧】（顺序执行）
  repack.inputs              # RepackTransform：改键名/重组布局
  InjectDefaultPrompt        # 没有 prompt 时注入默认指令
  data_transforms.inputs     # 机器人特定：AlohaInputs/LiberoInputs/DroidInputs
  Normalize                  # 用 norm_stats 归一化 state/actions
  model_transforms.inputs    # ResizeImages + TokenizePrompt(+state) + PadStatesAndActions

【模型】 sample_actions

【输出侧】（顺序执行，与输入镜像）
  model_transforms.outputs   # FAST: ExtractFASTActions（token→动作）
  Unnormalize                # 反归一化回物理空间
  data_transforms.outputs    # AlohaOutputs/LiberoOutputs（裁剪到机器人维度）
  repack.outputs
```

---

## 5.4 内置变换逐一详解

| 变换 | 作用 | 关键点 |
| --- | --- | --- |
| `RepackTransform(structure)` | 按 `{新键: 旧的扁平路径}` 重组 dict | 用 '/' 分隔扁平化，再 `jax.tree.map` 取值 |
| `InjectDefaultPrompt(prompt)` | 缺 `prompt` 时注入默认值 | 已有则不覆盖 |
| `Normalize(norm_stats, use_quantiles, strict)` | 归一化 | z-score 或分位数→[-1,1] |
| `Unnormalize(norm_stats, use_quantiles)` | 反归一化 | 始终 `strict=True` |
| `ResizeImages(height, width)` | resize 所有图像 | 调 `image_tools.resize_with_pad` |
| `SubsampleActions(stride)` | 动作下采样 | `actions[::stride]` |
| `DeltaActions(mask)` | 绝对动作→相对（delta） | `actions -= where(mask, state, 0)` |
| `AbsoluteActions(mask)` | delta→绝对 | `actions += where(mask, state, 0)` |
| `TokenizePrompt(tokenizer, discrete_state_input)` | 文本（+状态）分词 | π₀/π₀.₅ 用 |
| `TokenizeFASTInputs(tokenizer)` | FAST 输入分词 | 产 ar_mask/loss_mask |
| `ExtractFASTActions(tokenizer, horizon, dim)` | FAST token→动作 | 输出侧 |
| `PromptFromLeRobotTask(tasks)` | 从 task_index 取指令 | 训练时用 |
| `PadStatesAndActions(model_action_dim)` | state/action 零填充到模型维度 | 如填到 32 维 |

### 归一化的两种方式

```python
# z-score（默认，pi0）
def _normalize(self, x, stats):
    return (x - stats.mean) / (stats.std + 1e-6)

# 分位数归一化（pi0.5/pi0-fast，映射到 [-1, 1]）
def _normalize_quantile(self, x, stats):
    return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
```

`use_quantile_norm` 由模型类型决定（`DataConfigFactory.create_base_config` 里设为 `model_type != PI0`，即 π₀.₅ 和 FAST 用分位数）。分位数归一化对离群值更鲁棒。

### DeltaActions 的意义

很多任务里"目标位姿相对当前状态的增量"比绝对位姿更易学。`DeltaActions` 把绝对动作减去当前状态变成 delta，`mask` 控制哪些维度转（如 ALOHA 的 6 个关节转、夹爪不转：`make_bool_mask(6, -1, 6, -1)`）。`AbsoluteActions` 在输出侧加回状态还原。

### 工具函数

- `flatten_dict` / `unflatten_dict`：'/' 分隔的嵌套字典扁平化/还原。
- `transform_dict(patterns, tree)`：正则键重命名（首个匹配生效，校验无别名冲突）。
- `apply_tree(tree, selector, fn, strict)`：对 selector 中存在的键应用 fn（归一化的底层）。
- `pad_to_dim(x, target_dim, axis, value)`：沿某轴 pad 到目标维度。
- `make_bool_mask(*dims)`：正数→若干 True，负数→若干 False。如 `make_bool_mask(2,-2,2)==(T,T,F,F,T,T)`。
- `_assert_quantile_stats`：校验启用分位数时 q01/q99 存在。

---

## 5.5 归一化统计（normalize.py）

### `NormStats`
```python
@pydantic.dataclasses.dataclass
class NormStats:
    mean: NDArray; std: NDArray
    q01: NDArray | None = None   # 1% 分位
    q99: NDArray | None = None   # 99% 分位
```

### `RunningStats`：流式统计

为什么需要"流式"：数据集可能有上千万帧，无法一次载入内存。`RunningStats` 边遍历边更新：

- `update(batch)`：首批初始化 mean/mean_of_squares/min/max + 每维直方图（5000 桶）；后续批次用 Welford 式增量更新均值与平方均值，并在 min/max 变化时重新分桶直方图。
- `get_statistics() → NormStats`：方差 = 平方均值 − 均值²，std = sqrt(max(0, 方差))，q01/q99 用直方图累积和估计。少于 2 个样本报错。
- `_adjust_histograms` / `_update_histograms` / `_compute_quantiles`：直方图维护与分位数估计的内部方法。

> **为什么用直方图估分位数**：精确分位数需要保存全部数据排序，内存不可行。5000 桶直方图在内存可控的前提下给出足够精度的近似（注释提醒 q01/q99 异常小的维度可能导致归一化后数值爆炸，见 README 排错表）。

### 持久化
- `serialize_json` / `deserialize_json`：通过 `_NormStatsDict`（pydantic）做 JSON 序列化。
- `save(directory, norm_stats)` / `load(directory)`：读写 `norm_stats.json`。

归一化统计随检查点保存在 `assets/<asset_id>/norm_stats.json`，推理时从检查点加载（而非 config 目录），确保与训练完全一致。

---

## 5.6 机器人适配变换（policies/*_policy.py）

这些是 `data_transforms`，把机器人原生格式 ⇄ 模型格式。每个机器人有 `Inputs`（双向：训练+推理）和 `Outputs`（仅推理）。

### ALOHA（双臂，14 维）`aloha_policy.py`

- `AlohaInputs`：4 路相机（cam_high/cam_low/cam_left_wrist/cam_right_wrist）映射到模型的 base/left_wrist/right_wrist；缺失相机填黑图 + mask False。处理状态/动作的"ALOHA↔pi 空间"转换。
- `AlohaOutputs`：取前 14 维动作，做 `_encode_actions`。
- 关键转换函数：
  - `_joint_flip_mask()`：关节符号翻转 `[1,-1,-1,1,...]`。
  - `_gripper_to_angular` / `_gripper_from_angular` / `_gripper_from_angular_inv`：夹爪在 ALOHA 线性空间与 pi 角度空间之间换算（含 Interbotix 机械几何常数 arm_length=0.036, horn_radius=0.022）。
  - `_decode_aloha` / `_decode_state` / `_encode_actions` / `_encode_actions_inv`：状态/动作/图像的编解码。
- `adapt_to_pi` 开关：标准 ALOHA 数据应设 True，把关节/夹爪转到 pi 内部运行时空间（基础模型就是在该空间预训练的）。

### LIBERO（单臂，状态 8 维，动作返回 7 维）`libero_policy.py`

- `LiberoInputs(model_type)`：`observation/image`→`base_0_rgb`，`observation/wrist_image`→`left_wrist_0_rgb`，右手腕填零。右手腕 mask 仅对 PI0_FAST 为 True（FAST 不屏蔽 padding 图），其余为 False。
- `LiberoOutputs`：返回前 7 维动作。
- `_parse_image`：float→uint8，CHW→HWC。

### DROID（单臂，状态 8 维 = 7 关节 + 1 夹爪）`droid_policy.py`

- `DroidInputs(model_type)`：拼接 joint_position(7) + gripper(1) 成 state(8)。按模型类型选图像布局：
  - PI0/PI05：`(base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb)`，masks `(T,T,F)`。
  - PI0_FAST：`(base_0_rgb, base_1_rgb, wrist_0_rgb)`，masks `(T,T,T)`。
- `DroidOutputs`：返回前 8 维动作。

> **统一模式**：所有机器人都把数据 pad 到模型动作维度（32），输出时裁剪回机器人真实维度（14/8/7）。这让一个模型能服务多种机器人。

---

## 5.7 数据加载（data_loader.py）

### 数据集抽象
- `Dataset` / `IterableDataset` / `DataLoader`（Protocol）：随机访问数据集、可迭代数据集、数据加载器接口。
- `TransformedDataset`：随机访问数据集 + 变换（`__getitem__` 时应用）。
- `IterableTransformedDataset`：可迭代数据集 + 变换，支持 `is_batched`（批数据拆成单样本变换再重组）。
- `FakeDataset`：按模型 `inputs_spec` 生成随机假数据（用于 `debug` 配置）。

### 数据集创建
- `create_torch_dataset(data_config, action_horizon, model_config)`：基于 LeRobot 创建数据集。`repo_id=="fake"` 返回 `FakeDataset`；否则用 `LeRobotDataset`（通过 `delta_timestamps` 取 action_horizon 步动作序列）。若 `prompt_from_task`，套一层 `PromptFromLeRobotTask`。
- `create_rlds_dataset(...)`：创建 DROID 的 `DroidRldsDataset`（见 §5.8）。
- `transform_dataset` / `transform_iterable_dataset`：套上 repack + data + Normalize + model 变换（训练用，注意此处也会校验 norm_stats 存在）。

### DataLoader 实现
- `create_data_loader(config, *, sharding, shuffle, num_batches, skip_norm_stats, framework)`：**统一入口**。根据 `rlds_data_dir` 是否设置，分派到 RLDS 或 torch 加载器；支持 `framework="jax"|"pytorch"`。
- `create_torch_data_loader(...)`：创建 `TorchDataLoader`。PyTorch DDP 时用 `DistributedSampler` 并按 world_size 切分批大小；JAX 时按 `jax.process_count()` 切分。
- `create_rlds_data_loader(...)`：创建 `RLDSDataLoader`（PyTorch 暂不支持 RLDS）。
- `TorchDataLoader`：包装 `torch.utils.data.DataLoader`，`_collate_fn` 把样本堆成 numpy 批，JAX 模式下转成分片数组、PyTorch 模式下转 torch 张量。`__iter__` 支持无限循环（耗尽后重建迭代器）。
- `RLDSDataLoader`：DROID 数据集本身已批量化，这里只做分片包装。
- `DataLoaderImpl`：把批 dict 转成 `(Observation, Actions)` 元组（`Observation.from_dict(batch), batch["actions"]`）。
- `_worker_init_fn`：在 worker 进程里设置 JAX 不预分配 GPU 显存。

```
LeRobotDataset / DroidRldsDataset
   │ TransformedDataset（套变换管线）
   ▼
TorchDataLoader / RLDSDataLoader   （批量化 + 分片）
   │ DataLoaderImpl
   ▼
(Observation, Actions)  ──►  训练循环
```

---

## 5.8 RLDS 数据集（droid_rlds_dataset.py）

LeRobot 加载器对超大数据集（DROID 约 2000 万帧）不够扩展，故用 TFDS/RLDS 格式。

- `DroidActionSpace`（Enum）：`JOINT_POSITION` / `JOINT_VELOCITY`（默认关节位置，便于仿真评测）。
- `RLDSDataset`（dataclass）：`name` / `version` / `weight` / `filter_dict_path`（多数据集混采的权重与过滤字典）。
- `DroidRldsDataset`：核心类。`__init__` 里构建 tf.data 管线：
  1. 仅用 CPU（`tf.config.set_visible_devices([], "GPU")`，避免和 PyTorch/JAX 抢显存）。
  2. 校验各数据集权重和为 1.0。
  3. `prepare_single_dataset`：从 RLDS 读取 → 过滤失败轨迹（文件名含 "success"）→ repeat → 加载过滤字典（哈希表，按 `episode--timestep` 决定保留哪些帧，即"空闲帧过滤"）。
  4. `restructure`：重排观测/动作键，**随机选两个外部相机之一**，**随机选三条语言指令之一**，拼接动作（关节位置/速度 + 夹爪），计算唯一 step_id 查过滤表。
  5. `chunk_actions`：把轨迹切成 action_horizon 长的动作块（末尾重复最后一个动作）。
  6. flatten → 应用过滤 → 移除辅助键 → 延迟解码图像（省 IO）。
  7. 多数据集按权重采样、shuffle（缓冲区 25 万）、batch、限内存。
- `__iter__`：`yield from self.dataset.as_numpy_iterator()`。
- `__len__`：硬编码约 2000 万（过滤后近似值，避免遍历计数）。

> **空闲帧过滤**（improved idle filter）：很多机器人轨迹有大段静止帧，过滤字典指定每个 episode 保留哪些时间步，提升数据质量。

---

## 5.9 计算归一化统计（compute_norm_stats.py）

训练前必须先算 norm stats。`scripts/compute_norm_stats.py`：

- `RemoveStrings`：去掉字符串字段（JAX 不支持，且算统计不需要）。
- `create_torch_dataloader` / `create_rlds_dataloader`：构建只含 repack + data 变换（**不含** Normalize/model 变换）的加载器。
- `main(config_name, max_frames)`：遍历数据，用 `RunningStats` 累计 `state` 与 `actions` 的统计，保存到 `config.assets_dirs / repo_id`。

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

> 注意：算统计用的是 repack + data 变换后、归一化**之前**的数据（即机器人空间转模型空间后的原始 state/actions），这正是 Normalize 将要作用的对象。

---

## 5.10 小结

- Transform 体系用统一协议 + `Group.push` 的镜像组合，保证训练/推理数据处理严格一致。
- 归一化用流式 `RunningStats` 在超大数据上算 mean/std/分位数，随检查点持久化。
- 机器人适配变换（ALOHA/LIBERO/DROID）处理键名、维度、夹爪空间差异，统一 pad 到 32 维。
- 数据加载统一支持 LeRobot（map 式）与 RLDS（iterable 式）、JAX 与 PyTorch 两套框架。

下一章 [06 训练系统](06-training-system.md) 讲这些数据如何驱动训练循环。
