基于 `ConditionalDiT1D` 的默认构造参数（`conditional_dit1d_model.py`），三个子模块的参数与维度如下。

---

## 1. 构造参数对照（传入值）

| 模块 | 构造参数 | 传入值 / 来源 | 默认值 |
|------|----------|---------------|--------|
| **TimeEncoder** | `time_embed_dim` | `time_embed_dim` | 128 |
| **StateEncoder** (`StateTokenEncoder`) | `history_state_dim` | `history_state_dim` | 7 |
| | `road_feature_dim` | `road_feature_dim` | 2 |
| | `hidden_dim` | `scene_embed_dim` | 128 |
| | `dropout` | `dropout` | 0.1 |
| | `history_len` | `int(history_len)` | 10 |
| | `road_points` | `int(road_points)` | 60 |
| | `num_heads` | `num_heads` | 4 |
| | `num_layers` | `encoder_layers` | 2 |
| **DiT** (`traj_decoder`) | `depth` | `depth` | 8 |
| | `output_dim` | `prediction_state_dim` | 4 |
| | `hidden_dim` | `scene_embed_dim` | 128 |
| | `heads` | `num_heads` | 4 |
| | `dropout` | `dropout` | 0.1 |
| | `mlp_ratio` | `mlp_ratio` | 4.0 |

---

## 2. 输入 / 输出维度（`forward` 接口）

| 模块 | 输入 | 输入形状 | 输出 | 输出形状 |
|------|------|----------|------|----------|
| **TimeEncoder** | `timestep` | `(B,)` 或 `(B, 1)` 标量时间步 | 时间嵌入 | `(B, 128)` |
| **StateEncoder** | `cond`（dict） | 见下表 | `memory`, `context_mask` | `(B, S, 128)`, `(B, S)` bool |
| **DiT** | `noisy_trajectory` | `(B, L, 4)` | 去噪轨迹 | `(B, L, 4)` |
| | `t_feature` | `(B, 128)` | | |
| | `context_feature` | `(B, S, 128)` | | |
| | `context_mask` | `(B, S)`，`True`=padding | | |

其中 `L = future_len = 25`（来自 `ConditionalDiT1D` 默认值，DiT 本身不硬编码）。

---

## 3. StateEncoder 输入 `cond` 各字段

| 字段 | 形状 | 说明 |
|------|------|------|
| `history` | `(B, T_h, 7)` | 历史轨迹，`T_h` 通常 ≤ `history_len` |
| `history_mask` | `(B, T_h)` | 1=有效，0=padding |
| `centerline` | `(B, 60, 2)` | 中心线点 |
| `centerline_mask` | `(B, 60)` | |
| `left_boundary` | `(B, 60, 2)` | 左边界 |
| `left_boundary_mask` | `(B, 60)` | |
| `right_boundary` | `(B, 60, 2)` | 右边界 |
| `right_boundary_mask` | `(B, 60)` | |
| `lane_dividers` | `(B, D, N, 2)` | 车道分隔线，`D`=分隔线数，`N`=每条点数 |
| `lane_dividers_mask` | `(B, D, N)` | |
| `max_v` | `(B, 60)` | 限速值 |
| `max_v_mask` | `(B, 60)` | |

**序列长度** \(S\)（拼接后 token 数）：

\[
S = T_h + 4 \times \text{road\_points} + D \times N
= T_h + 240 + D \times N
\]

（4 路道路 token：centerline + left + right + speed，各 `road_points=60`；再加 `D×N` 个 divider token。）

---

## 4. 内部关键维度（子模块结构）

| 模块 | 内部结构 | 关键维度 |
|------|----------|----------|
| **TimeEncoder** | Sinusoidal → Linear(128→128) → Mish → Linear(128→128) | 全程 `(B, 128)` |
| **StateEncoder** | 5× Linear 投影到 128 + type_emb(6,128) + TransformerEncoder ×2 | `d_model=128`, `nhead=4`, `dim_feedforward=512` |
| **DiT** | `preproj`: 4→512→128 | 中间 hidden=128 |
| | `DiTBlock` ×8 | self-attn + adaLN(时间) + cross-attn(场景) + MLP，MLP hidden=512 |
| | `FinalLayer` | 128→512→4，adaLN 由 `t_feature` 调制 |

---

## 5. 三者数据流（整体）

```
timestep (B,)           ──► TimeEncoder ──► t_feature (B, 128) ────────────────┐
                                                                                 │
cond (dict)             ──► StateEncoder ──► memory (B,S,128), mask (B,S) ──────┤
                                                                                 ▼
noisy_trajectory (B,L,4) ───────────────────────────────────────────────► DiT ──► (B,L,4)
```

如需按你实际训练 config 里的非默认值再出一版表，可以把 config 路径或参数贴过来。