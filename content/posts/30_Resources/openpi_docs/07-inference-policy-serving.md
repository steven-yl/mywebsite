---
title: "07 推理与策略服务"
subtitle: ""
date: 2026-07-01T21:10:00+08:00
draft: false
authors: [Steven]
description: "解读 Policy 封装、策略工厂、WebSocket 服务端与归一化往返等 openpi 推理部署流程。"
summary: "openpi 推理与策略服务：Policy 封装与 WebSocket 部署详解。"
tags: [openpi, robots]
categories: [docs openpi]
series: [openpi-docs]
weight: 7
series_weight: 7
hiddenFromHomePage: false
hiddenFromSearch: false
featuredImage: ""
featuredImagePreview: ""
---
# 07 推理与策略服务

> 本章解读 `policies/policy.py`（Policy 封装）、`policies/policy_config.py`（策略工厂）、`serving/websocket_policy_server.py`（WebSocket 服务端）、`scripts/serve_policy.py`（服务入口），以及 `shared/download.py`、`shared/image_tools.py`、`shared/array_typing.py` 等支撑。

---

## 7.1 推理子系统的边界

训练产出检查点后，推理子系统负责把"模型 + 数据变换 + 归一化"打包成一个简单接口，并暴露成网络服务：

```
检查点(gs:// 或本地)
   │ create_trained_policy        # 加载模型 + 组装变换 + 加载 norm stats
   ▼
Policy.infer(obs) -> {actions, state, policy_timing}
   │ 包进 WebsocketPolicyServer
   ▼
网络服务（机器人通过 WebSocket 客户端调用）
```

---

## 7.2 `Policy`：模型 + 变换的封装（policy.py）

```python
class Policy(BasePolicy):
    def __init__(self, model, *, rng=None, transforms=(), output_transforms=(),
                 sample_kwargs=None, metadata=None, pytorch_device="cpu", is_pytorch=False):
        self._input_transform = compose(transforms)
        self._output_transform = compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        if is_pytorch:
            self._model = model.to(pytorch_device); model.eval()
            self._sample_actions = model.sample_actions
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)  # JIT 编译
            self._rng = rng or jax.random.key(0)
```

### `infer(obs, *, noise=None)` —— 核心推理流程

```python
def infer(self, obs, *, noise=None):
    inputs = jax.tree.map(lambda x: x, obs)        # 拷贝（变换会就地改）
    inputs = self._input_transform(inputs)          # 1) 输入变换管线
    if not self._is_pytorch_model:
        inputs = tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)   # 2) 加 batch 维
        self._rng, sample_rng = jax.random.split(self._rng)
    else:
        inputs = tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None], inputs)
        sample_rng = self._pytorch_device
    observation = Observation.from_dict(inputs)     # 3) 转 Observation
    outputs = {
        "state": inputs["state"],
        "actions": self._sample_actions(sample_rng, observation, **sample_kwargs),  # 4) 采样
    }
    outputs = tree.map(lambda x: np.asarray(x[0, ...]...), outputs)  # 5) 去 batch 维（PyTorch 还 detach/cpu）
    outputs = self._output_transform(outputs)        # 6) 输出变换管线
    outputs["policy_timing"] = {"infer_ms": model_time * 1000}
    return outputs
```

**步骤**：拷贝 obs → 输入变换 → 加批维 → 构造 `Observation` → `sample_actions`（π₀ 去噪 / FAST 自回归）→ 去批维 → 输出变换 → 附带计时。JAX 与 PyTorch 路径在加批维、设备处理、rng 上有别，但流程一致。

- `metadata` 属性：返回 policy 元数据（如 ALOHA 的 `reset_pose`）。

### `PolicyRecorder`：调试录制

包装一个 policy，每次 `infer` 把 `{inputs, outputs}` 扁平化后 `np.save` 到 `step_{n}.npy`。用于离线分析策略行为（`serve_policy.py --record`）。

---

## 7.3 策略工厂 `create_trained_policy`（policy_config.py）

```python
def create_trained_policy(train_config, checkpoint_dir, *, repack_transforms=None,
                          sample_kwargs=None, default_prompt=None, norm_stats=None, pytorch_device=None):
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    # 自动检测 PyTorch（看是否有 model.safetensors）
    is_pytorch = os.path.exists(os.path.join(checkpoint_dir, "model.safetensors"))
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # 从检查点（而非 config 资产目录）加载，确保与训练一致
        norm_stats = load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs, metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch, pytorch_device=pytorch_device if is_pytorch else None,
    )
```

**要点**：
- **自动检测后端**：靠 `model.safetensors` 是否存在判断 PyTorch vs JAX，对调用者透明（同一 API）。
- **从检查点加载 norm stats**：刻意从 `checkpoint_dir/assets` 而非 config 资产目录加载，保证推理用的归一化统计与该检查点训练时完全一致。
- **变换管线镜像组装**：输入与输出严格对称（见 [05 章](05-data-pipeline.md) §5.3）。

---

## 7.4 WebSocket 服务端（websocket_policy_server.py）

```python
class WebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None): ...
    def serve_forever(self): asyncio.run(self.run())
    async def run(self):
        async with serve(self._handler, host, port, compression=None, max_size=None,
                          process_request=_health_check) as server:
            await server.serve_forever()
    async def _handler(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))   # 连接时先发元数据
        while True:
            obs = msgpack_numpy.unpackb(await websocket.recv())   # 收观测
            action = self._policy.infer(obs)                       # 推理
            action["server_timing"] = {"infer_ms": ..., "prev_total_ms": ...}
            await websocket.send(packer.pack(action))              # 回动作
```

**协议**：
1. 客户端连接后，服务端**首先发送一次 metadata**（如 reset_pose）。
2. 之后进入循环：收 obs（msgpack-numpy 编码的二进制帧）→ `policy.infer` → 回 action（含 `server_timing`）。
3. 异常时发送 traceback 字符串并以 `INTERNAL_ERROR` 关闭连接。

- `_health_check(connection, request)`：对 `/healthz` 路径返回 HTTP 200 "OK"（健康检查），其它路径走正常握手。

> **为什么用 msgpack-numpy**：见 [08 章](08-client-runtime.md) §8.5——安全（不像 pickle 可执行任意代码）、跨语言、无需 schema、对大数组比 pickle 快约 4 倍。

> ⚠️ **安全提示**：服务端绑定 `0.0.0.0` 且**无鉴权、无 TLS**。仅适合可信内网。公网暴露需自行加反向代理/鉴权/加密。

---

## 7.5 服务入口（serve_policy.py）

```python
class EnvMode(enum.Enum): ALOHA; ALOHA_SIM; DROID; LIBERO
@dataclasses.dataclass class Checkpoint: config: str; dir: str    # 从检查点加载
@dataclasses.dataclass class Default: ...                          # 用环境默认策略
@dataclasses.dataclass class Args:
    env: EnvMode = ALOHA_SIM; default_prompt: str | None = None
    port: int = 8000; record: bool = False
    policy: Checkpoint | Default = Default()

DEFAULT_CHECKPOINT = {  # 每个环境的默认检查点
    ALOHA: Checkpoint("pi05_aloha", "gs://openpi-assets/checkpoints/pi05_base"),
    DROID: Checkpoint("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid"), ...
}
```

- `create_default_policy(env, default_prompt)`：用环境默认检查点建策略。
- `create_policy(args)`：`match` 分派——`Checkpoint()` 从指定 config/dir 建，`Default()` 用环境默认。
- `main(args)`：建策略 → 取 metadata → 可选包 `PolicyRecorder` → 启动 `WebsocketPolicyServer`。

```bash
# 从指定检查点启动服务
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/my_experiment/20000
# 或用环境默认策略
uv run scripts/serve_policy.py --env DROID
```

---

## 7.6 支撑设施（shared/）

### 资产下载（download.py）

- `get_cache_dir()`：缓存目录（`$OPENPI_DATA_HOME` 或 `~/.cache/openpi`），设权限。
- `maybe_download(url, *, force_download, **kwargs)`：核心。本地路径直接返回；远程路径缓存到 `cache_dir/netloc/path`，用 `filelock` 保证多进程安全，先下到 `.partial` 再 `move`。`gs://openpi-assets` 走 gsutil，其它走 fsspec。
- `_download_gsutil` / `_download_fsspec`：两种下载后端（fsspec 带 tqdm 进度）。
- `_set_permission` / `_set_folder_permission` / `_ensure_permissions`：管理缓存目录权限（容器与训练脚本共享缓存）。
- `_INVALIDATE_CACHE_DIRS` / `_should_invalidate_cache` / `_get_mtime`：基于 mtime 的缓存失效（用于淘汰过期的旧检查点）。

### 图像处理（image_tools.py）

- `resize_with_pad(images, height, width, method)`（`@jax.jit`）：复刻 `tf.image.resize_with_pad`——保持长宽比缩放后黑边填充。uint8 四舍五入裁剪到 [0,255]，float32 裁剪到 [-1,1]（pad 值 -1.0）。
- `resize_with_pad_torch(images, height, width, mode)`：PyTorch 版，自动识别 channels-last/first，用 `F.interpolate` + `F.pad`。

### 运行时类型检查（array_typing.py）

- `Array = jax.Array | torch.Tensor`：统一数组类型。
- `typecheck(t)`：用 `jaxtyped + beartype` 做运行时类型检查的装饰器。
- `disable_typechecking()`（contextmanager）：临时关闭类型检查。
- `check_pytree_equality(*, expected, got, check_shapes, check_dtypes)`：比较两个 PyTree 结构（友好报错），可选查形状/dtype。
- monkey-patch `_check_dataclass_annotations`：绕过 JAX 树展开时的 jaxtyping 误报（issue #277）。

---

## 7.7 远程推理的完整链路

```
机器人侧（弱算力）                            服务端（GPU）
┌────────────────────┐   WebSocket(msgpack) ┌──────────────────────────┐
│ WebsocketClient    │ ───── obs ─────────► │ WebsocketPolicyServer    │
│ Policy             │                       │   policy.infer(obs)      │
│ (openpi-client 包) │ ◄──── action ──────── │   (变换 + 模型 + 反变换)  │
└────────────────────┘                       └──────────────────────────┘
```

服务端把 GPU 上的重型策略暴露成网络服务，机器人侧用零重依赖的客户端（见 [08 章](08-client-runtime.md)）调用，物理环境与策略环境彻底解耦。

---

## 7.8 小结

- `Policy` 把模型与输入/输出变换封装成 `infer(obs)->actions`，JAX/PyTorch 统一 API。
- `create_trained_policy` 自动检测后端、从检查点加载 norm stats、镜像组装变换管线。
- `WebsocketPolicyServer` 用 msgpack-numpy 协议暴露策略，先发 metadata 再循环响应；无鉴权需注意安全。
- `serve_policy.py` 提供环境默认策略与自定义检查点两种启动方式。

下一章 [08 客户端运行时](08-client-runtime.md) 讲机器人侧如何组织观测采集-推理-执行的闭环。
