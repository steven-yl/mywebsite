# 03 — 数据集系统（LeRobotDataset v3）

## 1. 模块边界

```
datasets/
├── lerobot_dataset.py      # LeRobotDataset 门面（读/写）
├── dataset_metadata.py     # LeRobotDatasetMetadata
├── dataset_reader.py       # 读路径：delta、视频解码
├── dataset_writer.py       # 写路径：buffer、编码
├── factory.py              # make_dataset, make_train_eval_datasets
├── io_utils.py             # parquet/json 读写
├── video_utils.py          # 编解码、VideoDecoderCache
├── compute_stats.py        # 均值/方差/分位数统计
├── dataset_tools.py        # 合并、拆分、重编码
├── sampler.py              # EpisodeAwareSampler
├── streaming_dataset.py    # 流式加载
├── pipeline_features.py    # 从 processor 推断 feature schema
└── utils.py                # DatasetInfo、路径模板、版本检查
```

**版本**：`CODEBASE_VERSION = "v3.0"`（与 v2.1 不兼容）

---

## 2. 为什么需要 LeRobotDataset？

| 问题 | 方案 |
|------|------|
| 各 lab 自定义格式 | 统一 **Parquet（表格）+ MP4（视觉）** |
| 大规模数据内存爆炸 | 分块 chunk、mmap、Streaming 模式 |
| 多步策略需要历史帧 | **delta_timestamps** 按 fps 对齐 |
| Hub 共享 | `push_to_hub` + dataset card |

---

## 3. 磁盘格式 v3

```
{root}/
├── meta/
│   ├── info.json              # DatasetInfo：features, fps, 计数器
│   ├── stats.json             # 归一化统计
│   ├── tasks.parquet          # task_index ↔ 任务字符串
│   └── episodes/
│       └── chunk-000/file-000.parquet   # 每 episode 元数据
├── data/
│   └── chunk-000/file-000.parquet       # 帧级数据
└── videos/                              # 可选
    └── observation.images.cam_high/
        └── chunk-000/file-000.mp4
```

### 3.1 自动注入列（DEFAULT_FEATURES）

`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`

### 3.2 DatasetInfo 关键字段

| 字段 | 说明 |
|------|------|
| `fps` | 控制频率，delta 时间换算基准 |
| `features` | 各 key 的 dtype、shape、names |
| `robot_type` | 机器人类型标识 |
| `chunks_size` | 每 chunk 文件数（默认 1000） |
| `data_path` / `video_path` | 路径模板 |

### 3.3 Feature 示例（info.json 概念）

```json
{
  "observation.state": {
    "dtype": "float32",
    "shape": [14],
    "names": ["joint_0", "..."]
  },
  "observation.images.top": {
    "dtype": "video",
    "shape": [480, 640, 3],
    "info": { "video.codec": "h264", ... }
  },
  "action": {
    "dtype": "float32",
    "shape": [14]
  }
}
```

---

## 4. LeRobotDataset 类

**文件**：`lerobot_dataset.py`

内部组合：
- **Metadata** — 只读 meta/
- **DatasetReader** — `__getitem__`、视频解码
- **DatasetWriter** — `add_frame` / `save_episode`（写模式）

### 4.1 构造模式

| 模式 | API | 用途 |
|------|-----|------|
| 读取 | `LeRobotDataset(repo_id, root=..., episodes=[...])` | 训练/可视化 |
| 创建 | `LeRobotDataset.create(repo_id, fps, features, ...)` | 新数据集 |
| 追加 | `LeRobotDataset.resume(repo_id, root=...)` | 继续录制 |

### 4.2 公共 API 完整列表

| 方法/属性 | 作用 |
|-----------|------|
| `__len__` | 当前选中 episode 的总帧数 |
| `__getitem__(idx)` | 完整样本：delta 窗口 + 视频解码 + transform + task 字符串 |
| `get_raw_item(idx)` | 原始 HF 行，不解码视频 |
| `add_frame(frame)` | 写模式：追加一帧到 episode buffer |
| `save_episode(...)` | 刷盘：parquet + 视频 + 更新 meta |
| `clear_episode_buffer()` | 丢弃未保存 episode |
| `has_pending_frames()` | buffer 是否非空 |
| `finalize()` | 关闭 writer（幂等） |
| `select_columns(names)` | 列子集 |
| `set_image_transforms` / `clear_image_transforms` | 解码后图像增强 |
| `push_to_hub(...)` | 上传 Hub，打 tag `v3.0` |
| `fps`, `num_frames`, `num_episodes`, `features` | 元数据快捷属性 |
| `hf_dataset` | 底层 `datasets.Dataset` |
| `absolute_to_relative_idx` | 全局 index → HF 行 index |

---

## 5. 读取路径详解

### 5.1 Episode 过滤

- `episodes=[0, 2, 5]` 只加载指定 episode
- `episode_filter: Callable` 按 meta 行谓词过滤（如按 task）

PyArrow **predicate pushdown** 在 `episode_index` 上过滤，避免全量扫描。

### 5.2 Delta Timestamps（多步输入）

策略配置 `observation_delta_indices = [0, -1, -2]` 经 `resolve_delta_timestamps()` 转为秒：

\[
\Delta t_i = \frac{\text{index}_i}{\text{fps}}
\]

Reader 在 episode 边界**裁剪**并添加 `{key}_is_pad` 标记 padding 帧。

### 5.3 视频解码流程

```
parquet 时间戳
  → 加上 episode videos/{key}/from_timestamp 偏移
  → decode_video_frames(path, timestamps, tolerance_s)
  → [torchcodec | pyav] 解码
  → 深度图：dequantize_depth()
  → image_transforms（训练增强）
  → 合并进 sample dict
```

| 函数 | 作用 |
|------|------|
| `decode_video_frames` | 后端路由 |
| `decode_video_frames_torchcodec` | 快速 seek（Linux/macOS ARM 等） |
| `decode_video_frames_pyav` | 通用 fallback；深度必选 |
| `VideoDecoderCache` | LRU 解码器缓存 |

环境变量：`LEROBOT_VIDEO_DECODER_CACHE_SIZE`

**容差**：`|t_frame - t_query| ≤ tolerance_s`（默认约 1/fps）

---

## 6. 写入路径详解

### 6.1 录制循环（与 record 脚本配合）

```python
frame = {
    "observation.state": state_tensor,
    "observation.images.cam": image_array,
    "action": action_tensor,
    "task": "pick the cube",
}
dataset.add_frame(frame)
# episode 结束：
dataset.save_episode()
```

### 6.2 save_episode 内部步骤

1. 分配全局 `index`、`episode_index`、`task_index`
2. 写 data chunk parquet
3. 编码 MP4（streaming 或 batch；多相机可并行）
4. `meta.save_episode()` 更新 episodes parquet、计数器、stats
5. 清空 episode buffer

### 6.3 视频编码

配置见 `configs/video.py`：`VideoEncoderConfig`、`RGBEncoderConfig`、`DepthEncoderConfig`

支持 **streaming encoding**（边录边编码）降低磁盘峰值。

---

## 7. LeRobotDatasetMetadata

| 方法 | 作用 |
|------|------|
| `create` | 初始化空数据集 |
| `save_episode` | 写入 episode 级元数据 |
| `save_episode_tasks` | 注册新 task 字符串 |
| `get_task_index(task)` | 字符串 → index |
| `filter_episodes(predicate)` | 按条件列 episode index |
| `update_video_info` | ffprobe 探测 MP4 写入 feature info |
| `update_chunk_settings` | 调整 chunk 大小限制 |
| `ensure_readable` | 写后切读模式时 reload |

属性：`video_keys`, `image_keys`, `depth_keys`, `camera_keys`, `has_language_columns`

---

## 8. 工厂函数（`factory.py`）

### 8.1 `make_dataset(cfg)`

根据 `TrainPipelineConfig.dataset`：

- 普通：`LeRobotDataset`
- `streaming=True`：`StreamingLeRobotDataset`
- 可选 ImageNet 统计覆盖视觉 normalization

### 8.2 `make_train_eval_datasets(cfg)`

- 按 task 分层划分 `eval_split` 比例
- 返回 `(train_dataset, eval_dataset)`

### 8.3 `resolve_delta_timestamps(cfg, ds_meta)`

将 policy 的 `*_delta_indices` 转为 `{feature_key: [seconds]}`

---

## 9. EpisodeAwareSampler

**文件**：`sampler.py`

训练 DataLoader 采样器：

- 仅在**同一 episode 内**连续或 shuffle 帧
- 可选 `drop_n_first_frames` / `drop_n_last_frames`（避免边界 padding 过多）
- 保证 BC 训练不跨 episode 非法拼接

---

## 10. 数据集工具（`dataset_tools.py`）

| 函数 | 作用 |
|------|------|
| `delete_episodes` | 删除 episode，重建数据集 |
| `split_dataset` | 按列表或比例拆分 |
| `merge_datasets` | 合并多个 repo |
| `modify_features` / `add_features` / `remove_feature` | 特征增删 |
| `modify_tasks` | 重标注 task |
| `recompute_stats` | 重算 stats.json |
| `convert_image_to_video_dataset` | 图像 → MP4 |
| `reencode_dataset` | 换编码器重编码 |

CLI：`lerobot-edit-dataset --operation.type=merge` 等（见 [10-cli-reference.md](./10-cli-reference.md)）

---

## 11. 统计与归一化（`compute_stats.py`）

| 函数 | 作用 |
|------|------|
| `get_feature_stats` | 单特征 mean/std/min/max/quantiles |
| `aggregate_stats` | 多 episode 聚合 |

stats 写入 `meta/stats.json`，供 `NormalizerProcessorStep` 使用。

---

## 12. 版本兼容

| 异常 | 含义 |
|------|------|
| `BackwardCompatibilityError` | 数据集版本 > 代码支持 |
| `ForwardCompatibilityError` | 数据集版本过旧 |

迁移：`python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=...`

---

## 13. 示例

### 13.1 加载与迭代

```python
"""加载 Hub 数据集并遍历前 3 帧。"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/pusht", episodes=[0])
for i in range(min(3, len(ds))):
    s = ds[i]
    print(i, s["action"].shape, s.get("task", ""))
```

### 13.2 创建本地数据集（写模式骨架）

```python
"""创建空数据集结构（需本地 root 可写）。"""
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_empty_dataset_info

root = Path("/tmp/my_lerobot_ds")
features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": None},
}
ds = LeRobotDataset.create(
    repo_id="user/local_test",
    root=root,
    fps=10,
    features=features,
    robot_type="so101",
    use_videos=False,
)
ds.finalize()
print("created at", root)
```

### 13.3 训练用 factory

```python
"""解析 TrainPipelineConfig 并构建 dataset（需完整 cfg）。"""
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from dataclasses import dataclass

@dataclass
class Cfg:
    dataset: DatasetConfig

cfg = Cfg(dataset=DatasetConfig(repo_id="lerobot/pusht"))
ds = make_dataset(cfg)
print(len(ds), ds.fps)
```

---

## 14. StreamingLeRobotDataset

**适用**：超大数据集、顺序预训练  
**限制**：随机 episode 访问弱；工厂中 `MultiLeRobotDataset` 仍实验性

Iterable 模式按 shard 拉取，降低 RAM。

---

## 下一章

- 处理器管道 → [04-processor-pipeline.md](./04-processor-pipeline.md)
- 训练集成 → [06-training-evaluation.md](./06-training-evaluation.md)
