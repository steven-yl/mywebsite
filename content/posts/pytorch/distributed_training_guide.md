---
title: "PyTorch 分布式训练与操作工具技术文档"
subtitle: ""
date: 2026-03-12T00:00:00+08:00
draft: false
authors: [Steven]
description: "系统梳理 PyTorch 原生分布式训练与常用工具的核心概念、API 以及端到端实践流程。"
summary: "从进程组初始化、DDP 封装、数据分片、集体通信到 Lightning 封装，全面讲解如何在单机多卡与多机多卡场景下正确使用 PyTorch 分布式训练。"

tags: ["PyTorch", "分布式训练", "Deep Learning"]
categories: ["PyTorch"]
series: ["PyTorch 实践指南"]
weight: 1
series_weight: 1

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: "/mywebsite/posts/images/distributed_training_guide.webp"
---

## 文档索引

| 章节 | 主题 | 内容概要 |
|------|------|----------|
| [一、概览](#一概览整体架构与文档脉络) | 整体架构与文档脉络 | 为何做分布式、整体数据流、知识结构图、各部分职责与关联、DP vs DDP / 单机 vs 多机 / 原生 vs Lightning 对比 |
| [二、核心概念](#二核心概念) | 进程、进程组、Rank、Backend、Collective | 每个概念：是什么、为什么需要、解决什么问题 |
| [三、进程组初始化](#三进程组初始化) | `init_process_group` 与 `destroy_process_group` | 初始化流程、参数完整说明、init_method、环境变量、单卡兼容、示例 |
| [四、DistributedDataParallel](#四distributeddataparallel-ddp) | DDP 包装与梯度同步 | 前向/反向/存储结构，构造函数全部参数，`model.module`、`find_unused_parameters`、bucket、示例 |
| [五、数据分片](#五数据分片distributedsampler) | DistributedSampler | 分片原理、构造函数、`set_epoch`、与 DataLoader 配合、验证集用法、示例 |
| [六、集体通信](#六集体通信-api) | barrier / all_reduce / broadcast / all_gather / gather / reduce | 各 API 签名、语义、ReduceOp、适用场景、本项目中的用法、示例 |
| [七、分布式启动](#七分布式启动) | torchrun 与 launch | 单机/多机命令、全部命令行参数、环境变量、与 init 的衔接、示例 |
| [八、Checkpoint 与日志](#八checkpoint-与日志) | 仅 rank 0 写盘与 barrier | 保存/加载模式、barrier 放置、日志只打 rank 0、示例 |
| [九、完整示例](#九完整示例) | 原生 DDP 端到端 | 与项目风格一致的可运行脚本与启动命令 |
| [十、PyTorch Lightning](#十使用-pytorch-lightning-做-ddp) | DDPStrategy | 封装内容、单机/多机用法、与本项目 pl_train 一致示例 |
| [十一、速查与小结](#十一速查与小结) | 组件对照表与延伸 | 组件/概念速查表、进阶方向 |

**阅读建议**：先读概览与核心概念建立全局图景，再按需跳转到对应章节查阅 API 与示例；做实现时可按「启动 → init → DDP + Sampler → 训练循环 → barrier/rank0 保存」顺序对照各章。

---

## 一、概览：整体架构与文档脉络

### 1.1 为什么要做分布式训练

**解决的问题**：

- **单卡显存/算力不足**：模型或 batch 过大，单 GPU 放不下或训练太慢。
- **提高吞吐**：多张卡同时算不同 batch，单位时间内处理的样本数成倍增加，缩短总训练时间。
- **多机扩展**：单机 GPU 数量有限时，通过多台机器进一步扩展总 GPU 数。

**本质**：把「一份模型 + 一份数据」拆成多份，让多个进程（每进程通常绑定 1 个 GPU）协同计算，在**梯度或参数**上做同步，使多卡/多机在数学上等价于「大 batch 的单卡训练」（数据并行时）。

### 1.2 整体架构与数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           启动阶段（仅执行一次）                                   │
│  ① 启动工具 (torchrun / torch.distributed.launch)                                 │
│       → 为每个 GPU 启动一个进程，注入 RANK / LOCAL_RANK / WORLD_SIZE 等            │
│  ② 进程内：torch.distributed.init_process_group(backend, ...)                     │
│       → 建立进程组，选定通信后端（NCCL/GLOO），各进程完成握手                       │
│  ③ 模型放到当前 GPU，再用 DistributedDataParallel(DDP) 包装                       │
│       → DDP 在 backward 时自动做梯度 AllReduce，保证各卡参数一致更新               │
│  ④ DataLoader + DistributedSampler                                                │
│       → 每个 rank 只看到数据集的不重叠子集，避免重复训练同一批数据                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           训练循环（每个 step）                                    │
│  各 rank：数据 → DataLoader（DistributedSampler 分片）→ 前向 → 反向 → DDP AllReduce │
│  同步点：dist.barrier() → 仅 rank 0 写 checkpoint / 打 log → 再 barrier（可选）    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 知识结构（文档脉络）

- **基础层**：进程、进程组、Rank/Local Rank/World Size、Node、Backend、Collective、梯度同步（第二节概念）。
- **搭建层**：进程组初始化（第三节）→ DDP 包装模型（第四节）→ DistributedSampler 分数据（第五节）→ 集体通信按需使用（第六节）。
- **入口层**：分布式启动（第七节）决定进程数与环境变量，训练脚本内再 init、DDP、Sampler。
- **工程层**：Checkpoint 与日志（第八节）、完整示例（第九节）、Lightning 封装（第十节）。

各节关系：**启动** 提供进程与环境 → **init** 建立通信 → **DDP** 与 **Sampler** 分别负责梯度同步与数据分片 → **集体通信** 用于用户级同步与聚合 → **Checkpoint/日志** 与 **示例/Lightning** 为落地用法。

### 1.4 各部分职责与关联

| 组件 | 职责/主题 | 与其它部分的关系 |
|------|-----------|------------------|
| **启动工具** | 生成多进程并注入 RANK/WORLD_SIZE/MASTER_* | 必须在 `init_process_group` 之前完成；不启动多进程就没有分布式 |
| **init_process_group** | 建立进程组、选定 backend | 所有分布式 API（DDP、collective、barrier）都依赖已初始化的进程组 |
| **DDP** | 包装模型，backward 时同步梯度 | 依赖已 init 的进程组；需配合 DistributedSampler 才能数据不重不漏 |
| **DistributedSampler** | 按 rank 切分数据索引 | 依赖 rank/world_size；DataLoader 用其替代默认 sampler |
| **集体通信 / barrier** | 同步、聚合张量 | 依赖已 init 的进程组；DDP 内部用 AllReduce；用户用 barrier 做「等齐再往下」 |
| **Rank 0 写盘与 log** | 单点持久化与日志 | 避免多进程写同一文件、日志刷屏；常与 barrier 配合 |

### 1.5 优缺点与适用场景对比

#### 1.5.1 DataParallel (DP) vs DistributedDataParallel (DDP)

| 维度 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|--------------------------------|
| **实现方式** | 单进程多线程，主卡聚合梯度再广播 | 多进程，每进程一卡，梯度 AllReduce |
| **通信** | 梯度汇总到主卡再广播；主卡瓶颈明显 | 各卡对等通信（如 Ring-AllReduce），主卡无瓶颈 |
| **速度** | 多卡扩展差，常比单卡还慢 | 多卡接近线性加速，推荐 |
| **使用难度** | `model = nn.DataParallel(model)` 即可 | 需多进程启动 + init + DistributedSampler |
| **适用场景** | 快速试验、卡数少（如 2 卡） | 正式训练、多卡/多机 |

**结论**：PyTorch 官方推荐用 **DDP** 做数据并行；DP 仅适合临时试验。

#### 1.5.2 单机多卡 vs 多机多卡

| 维度 | 单机多卡 | 多机多卡 |
|------|----------|----------|
| **通信** | 机内 NVLink/PCIe，延迟低、带宽高 | 跨机网络，延迟与带宽逊于机内 |
| **启动** | `torchrun --nproc_per_node=N train.py` | 每台机器各起一份 torchrun，需指定 nnodes、node_rank、master_addr、master_port |
| **环境变量** | 通常由 torchrun 自动设置 | 有时由调度系统（Slurm/K8s）设置 MASTER_*、RANK 等 |

#### 1.5.3 原生 torch.distributed vs PyTorch Lightning

| 维度 | 原生 torch.distributed + DDP | PyTorch Lightning (DDPStrategy) |
|------|------------------------------|----------------------------------|
| **控制力** | 完全手控 init、sampler、barrier、保存 | 框架封装，init/sampler/保存由 Trainer 处理 |
| **代码量** | 多：需写启动、rank 判断、sampler、barrier | 少：指定 strategy=DDPStrategy() 即可 |
| **适用** | 自定义训练循环、非标准流程 | 标准 train/val 循环、快速实验、与本项目 pl_train 一致 |

---

## 二、核心概念

### 2.1 进程与进程组

- **进程 (Process)**  
  **是什么**：操作系统中的一个独立执行单元；DDP 里通常「一个进程绑定一张 GPU」。  
  **为什么需要**：实现真正并行（多进程可跑在多核/多机上），避免 Python GIL 限制。  
  **解决什么问题**：单进程多线程无法充分利用多卡；多进程才能让每张卡在独立进程中运行，互不阻塞。

- **进程组 (Process Group)**  
  **是什么**：参与集体通信的一组进程的集合；默认用 `world` 表示「所有进程」。  
  **为什么需要**：为 collective（AllReduce、Barrier 等）划定「和谁通信」。  
  **解决什么问题**：多机多任务时可能只让部分节点参与一次训练，进程组用来区分「这一组」进程。

### 2.2 Rank、Local Rank、World Size

- **Rank（全局 rank）**  
  **是什么**：当前进程在整个分布式任务中的唯一编号，0 到 world_size-1。  
  **作用**：标识「我是谁」，用于划分数据（DistributedSampler）、决定谁写 checkpoint（如 rank 0）。  
  **解决什么问题**：多进程中需要唯一 ID，否则无法分片和选主。

- **Local Rank（本机 rank）**  
  **是什么**：当前进程在**本机**内的 GPU 编号，通常 0 到「本机 GPU 数-1」。  
  **作用**：`torch.cuda.set_device(local_rank)` 绑定当前进程到对应 GPU。  
  **解决什么问题**：多机时每台机器都有 rank 0；用 local_rank 才能正确绑定到本机某张卡。

- **World Size**  
  **是什么**：参与该次训练的**总进程数**（通常等于总 GPU 数）。  
  **作用**：AllReduce、DistributedSampler 等都需要「一共有多少参与方」。  
  **解决什么问题**：集体通信与数据分片都依赖「总数」这一信息。

### 2.3 Node（节点）

- **是什么**：一台物理或逻辑机器，上面有多张 GPU。  
- **作用**：多机训练时用 node_rank 区分机器，用 nnodes 表示机器数。  
- **解决什么问题**：启动多机任务时要指明「有多少台机器、当前是第几台」，以便正确建连（master_addr/master_port）。

### 2.4 Backend（通信后端）

- **是什么**：进程间做集体通信时使用的底层实现。  
- **常见选择**：  
  - **NCCL**：NVIDIA 的多卡/多机 GPU 通信库，**CUDA 训练默认推荐**。  
  - **GLOO**：CPU 或 GPU 都可用，多机无 NCCL 时可用；CPU 上调试时也常用。  
  - **MPI**：需单独安装 MPI 与 PyTorch MPI 后端，多用于 HPC。  
- **作用**：决定梯度、张量如何在不同进程间同步。  
- **解决什么问题**：不同硬件/环境需要不同通信实现，backend 提供统一 API、多种实现。

### 2.5 Collective（集体通信）

- **是什么**：一组进程共同参与的通信原语，如 AllReduce、Broadcast、Barrier、AllGather、Gather、ReduceScatter。  
- **作用**：  
  - **AllReduce**：各进程提供形状相同的张量，结果变为「所有进程得到同一份聚合后的张量」（如梯度求和/求平均）；DDP 的梯度同步即 AllReduce。  
  - **Barrier**：所有进程在此阻塞，直到都执行到 barrier，再一起继续。  
  - **Broadcast**：根进程把张量发到所有进程。  
  - **AllGather**：各进程提供一个张量，汇总后每个进程得到完整列表。  
- **解决什么问题**：多进程要「对齐状态」或「汇总结果」，必须依赖集体通信而不是各自为政。

### 2.6 梯度同步（DDP 中的 AllReduce）

- **是什么**：DDP 在 `backward()` 时，把各卡梯度做 AllReduce（通常求和再除以 world_size），使各卡用同一份梯度更新参数。  
- **作用**：数学上等价于「单卡大 batch」；每卡只算本地 batch 的梯度，通过同步得到全局梯度。  
- **解决什么问题**：数据并行下「多卡算多份小 batch，如何得到等价于大 batch 的更新」。

---

## 三、进程组初始化

### 3.1 结构说明与边界

- **职责**：建立默认进程组（world）、根据 backend 初始化通信库、让当前进程加入该组。  
- **调用前**：本进程已由启动工具（如 torchrun）启动，且环境变量或参数中已有正确的 rank、world_size；多机时还需 MASTER_ADDR、MASTER_PORT。  
- **调用后**：方可使用 DDP、DistributedSampler、`dist.barrier()`、`dist.all_reduce()` 等。

### 3.2 关键 API 与概念

#### 3.2.1 init_process_group

**完整签名（常用参数）**：

```text
torch.distributed.init_process_group(
    backend,           # str: "nccl" | "gloo" | "mpi"
    init_method=None,  # str: "env://" | "tcp://IP:PORT" | "file:///path"
    world_size=None,   # int，不设则从环境变量读取
    rank=None,         # int，不设则从环境变量读取
    timeout=datetime.timedelta(seconds=1800),  # 集体通信超时
    store=None,        # Store，用于 bootstrap，高级用法
)
```

- **backend**：通信后端；CUDA 训练用 `"nccl"`，CPU 或调试可用 `"gloo"`。  
- **init_method**：  
  - `"env://"`：从环境变量读取 RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT（**与 torchrun 配套，推荐**）。  
  - `"tcp://IP:PORT"`：指定 master 地址与端口，所有进程需能连到该地址。  
  - `"file:///path"`：通过共享文件系统做 rendezvous，适合无共享 IP 的环境。  
- **world_size / rank**：不传时从环境变量 WORLD_SIZE、RANK 读取（torchrun 会设置）。

**为什么需要**：多进程必须先「发现彼此」并约定通信方式，否则 DDP 与 collective 无法工作。  
**解决什么问题**：统一建立进程组与通信后端，为后续 DDP 与集体通信提供基础。

#### 3.2.2 destroy_process_group

- **作用**：销毁进程组，释放通信资源；训练结束后调用，便于干净退出。  
- **何时调用**：所有分布式训练与 collective 完成后；若不调用，进程退出时也会清理，但显式调用更规范。

#### 3.2.3 查询接口

- **dist.get_rank()**：当前进程的全局 rank。  
- **dist.get_world_size()**：当前进程组大小。  
- **dist.is_initialized()**：进程组是否已初始化（用于单卡/多卡分支判断）。

### 3.3 使用方式与适用条件

- 每个参与训练的进程**都必须**调用一次 `init_process_group`，且 backend、world_size 在所有进程中一致；rank 每进程不同。  
- 推荐用 **torchrun** 启动，由它注入 RANK、LOCAL_RANK、WORLD_SIZE 等，代码里用 `os.environ["RANK"]` 等读取后传入，或直接使用 `init_method="env://"` 不传 rank/world_size。  
- 单卡时可不调用 init（world_size==1），后续用 `dist.is_initialized()` 判断是否走分布式逻辑。

### 3.4 示例代码

```python
import os
import torch
import torch.distributed as dist

def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size == 1:
        return rank, world_size, local_rank  # 单卡不 init

    dist.init_process_group(
        backend="NCCL",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

if __name__ == "__main__":
    rank, world_size, local_rank = setup_distributed()
    print(f"rank={rank}, world_size={world_size}, local_rank={local_rank}")
    if dist.is_initialized():
        dist.destroy_process_group()
```

---

## 四、DistributedDataParallel (DDP)

### 4.1 结构说明与边界

- **角色**：`nn.Module` 的包装器；不改变单进程前向/反向的调用方式，在 backward 时插入梯度 AllReduce。  
- **前向**：每个进程用本进程的数据做一次前向，得到本地 loss。  
- **反向**：`loss.backward()` 时，DDP 注册的 hook 在各卡梯度计算完成后对梯度做 **AllReduce**（默认等价于求平均），保证各卡参数用同一梯度更新。  
- **存储**：原始模型在 `model.module`；保存/加载单卡权重时用 `model.module.state_dict()`。

**要点**：各进程模型结构必须一致；各 step 各进程应处理**不同**数据（由 DistributedSampler 保证），否则等价于重复算同一 batch。

### 4.2 构造函数参数（完整）

```text
torch.nn.parallel.DistributedDataParallel(
    module,                      # 要包装的 nn.Module，需已在目标 device 上
    device_ids=None,            # 单卡时 [local_rank]，多卡时通常 [local_rank]
    output_device=None,         # 默认与 device_ids[0] 一致
    dim=0,                       #  gather 的维度，一般用默认
    broadcast_buffers=True,      # 是否在 forward 前同步 BN 等 buffer
    process_group=None,         # 默认使用默认进程组
    bucket_cap_mb=25,           # 梯度桶大小（MB），影响通信/内存权衡
    find_unused_parameters=False,  # 若有参数未参与计算，须为 True
    gradient_as_bucket_view=False,  # True 可省部分内存，推荐与 Lightning 一致时开启
    static_graph=False,         # 若计算图固定可设为 True，利于优化
)
```

- **device_ids**：当前进程对应的 GPU 列表，单进程单卡时为 `[local_rank]`。  
- **find_unused_parameters**：若模型存在在 forward 中未参与计算的参数（如部分分支未走），必须设为 True，否则 backward 会报错。  
- **gradient_as_bucket_view**：梯度以 bucket 视图形式存在，可节省显存；本项目 pl_train 使用 True。  
- **broadcast_buffers**：每个 step 前将 BN 等 buffer 从 rank 0 广播到其它 rank，保证一致性。

### 4.3 关键概念

- **梯度桶 (bucket)**：DDP 将参数梯度按 bucket_cap_mb 打成若干桶，按桶做 AllReduce，以重叠通信与计算。  
- **model.module**：包装后的 DDP 实例的 `.module` 属性即原始模块；保存/加载时通常用 `model.module`。

### 4.4 使用方式与适用条件

- 必须在 `init_process_group` 之后、且模型已放到对应 GPU 上再包装：`model = model.cuda(local_rank)`，然后 `model = DDP(model, device_ids=[local_rank])`。  
- 优化器在 DDP 包装后创建，`model.parameters()` 会正确指向所有需要更新的参数。

### 4.5 示例代码

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    rank, world_size, local_rank = setup_distributed()
    model = MyModel().cuda(local_rank)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for batch in dataloader:
        out = model(batch)
        loss = out.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    if dist.is_initialized() and rank == 0:
        torch.save(model.module.state_dict(), "ckpt.pth")
```

---

## 五、数据分片：DistributedSampler

### 5.1 结构说明与边界

- **职责**：按 rank 和 world_size 把数据集索引划分成不重叠的子集，每个 rank 只拿到自己那一份索引；DataLoader 据此取数。  
- **与 DDP 的关系**：DDP 不关心数据从哪来，但若各进程用相同数据，梯度会重复；DistributedSampler 解决「数据分片」问题，保证不重不漏。

### 5.2 构造函数与 set_epoch

**构造函数**：

```text
torch.utils.data.distributed.DistributedSampler(
    dataset,        # Dataset
    num_replicas=None,  # 默认 dist.get_world_size()
    rank=None,      # 默认 dist.get_rank()
    shuffle=True,   # 是否打乱
    drop_last=False,  # 是否丢弃最后不完整 batch
    seed=0,
)
```

- **num_replicas / rank**：不传则从当前进程组读取；与 init 后的环境一致。  
- **shuffle**：为 True 时每个 epoch 内索引打乱，但各 rank 仍只看到自己的子集。  
- **drop_last**：为 True 时丢弃最后不足一 batch 的样本，保证各 rank 迭代次数一致（DDP 要求各进程 step 数一致，否则会挂起）。

**set_epoch(epoch)**：每个 epoch 开始时调用，内部用 epoch 作随机种子，使不同 epoch 的划分方式不同，提高数据利用与随机性。

### 5.3 使用方式与适用条件

- 构造 DataLoader 时传入 `sampler=train_sampler`，且**不要**再设 `shuffle=True`（Sampler 已决定顺序）。  
- 每个 epoch 开始时调用 `train_sampler.set_epoch(epoch)`。  
- 验证/测试若也要分布式跑，可用 `DistributedSampler(..., shuffle=False)`；若只在 rank 0 上验证，可不使用 DistributedSampler。

### 5.4 示例代码

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

dataset = MyDataset(...)
if dist.is_initialized():
    train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
else:
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    if dist.is_initialized():
        train_sampler.set_epoch(epoch)
    for batch in train_loader:
        ...
```

---

## 六、集体通信 API

### 6.1 结构说明与边界

- **职责**：协调多进程步调、聚合张量；DDP 内部已用 AllReduce 同步梯度，用户层更多用 **barrier** 做同步点，用 **all_reduce** / **all_gather** 等做指标汇总或自定义同步。  
- **约束**：所有参与同一 collective 的进程都必须调用该 collective，且张量形状/类型一致，否则会挂起或报错。

### 6.2 API 完整说明

#### 6.2.1 barrier

```python
torch.distributed.barrier(group=None)
```

- **作用**：所有进程阻塞直到都执行到这一行，再一起继续。  
- **应用**：等齐再写文件、再打印、再评估；本项目 eval 脚本中在 rank 0 写结果前常用 barrier。

#### 6.2.2 all_reduce

```python
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)
```

- **作用**：各进程提供形状相同的 tensor，按 op 聚合后写回各进程的 tensor（inplace）。  
- **op**：`ReduceOp.SUM`、`ReduceOp.AVG`、`ReduceOp.PRODUCT`、`ReduceOp.MIN`、`ReduceOp.MAX`。  
- **应用**：梯度或标量指标汇总；本项目 `code_train/utils/util.py` 中 `reduce_mean` 即 all_reduce(SUM) 再除以 nprocs。

#### 6.2.3 broadcast

```python
torch.distributed.broadcast(tensor, src, group=None, async_op=False)
```

- **作用**：src 进程的 tensor 广播到所有进程，覆盖各进程的 tensor。  
- **应用**：rank 0 读 checkpoint 后广播到各进程，或广播超参数。

#### 6.2.4 reduce

```python
torch.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False)
```

- **作用**：各进程的 tensor 按 op 聚合到 dst 进程。  
- **应用**：只在一个进程上得到聚合结果时使用。

#### 6.2.5 all_gather

```python
torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)
```

- **作用**：各进程提供一个 tensor，汇总后每个进程得到完整的 tensor_list（所有进程的 tensor）。  
- **约束**：各进程的 tensor 形状一致；tensor_list 长度为 world_size。  
- **应用**：收集各卡上的局部结果成完整列表；本项目 `code_train/utils/util.py` 中有对任意可 pickle 对象的 all_gather 封装（先序列化再 all_gather 张量）。

#### 6.2.6 gather

```python
torch.distributed.gather(tensor, gather_list, dst, group=None, async_op=False)
```

- **作用**：各进程的 tensor 收集到 dst 进程的 gather_list。  
- **应用**：只需在根进程汇总时使用。

#### 6.2.7 reduce_scatter

```python
torch.distributed.reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)
```

- **作用**：各进程提供 input_list，先按元素 reduce，再按 rank 切分，每进程得到 output 的一块。  
- **应用**：分布式优化器或特定通信模式。

### 6.3 使用方式与典型模式

- 写 checkpoint / 打 log：`dist.barrier()` → 仅 rank 0 写/打 log → 再 `dist.barrier()`（可选）。  
- 汇总标量 loss：每卡得到标量后转为 1 元素 tensor，`all_reduce(..., op=ReduceOp.AVG)`，再在 rank 0 打印。

### 6.4 示例代码

```python
import torch.distributed as dist

# 等所有进程到齐再保存
dist.barrier()
if rank == 0:
    torch.save(model.module.state_dict(), "ckpt.pth")
dist.barrier()

# 汇总各卡 loss 标量
loss_t = torch.tensor([loss_item], device=device)
dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
global_loss = loss_t.item()
if rank == 0:
    print(f"step loss avg: {global_loss}")
```

---

## 七、分布式启动

### 7.1 结构说明与边界

- **职责**：为每个 GPU 启动一个进程，并设置 RANK、LOCAL_RANK、WORLD_SIZE、MASTER_ADDR、MASTER_PORT 等环境变量；用户脚本只需读环境变量并 `init_process_group(init_method="env://")`。  
- **工具**：**torchrun**（推荐，PyTorch 1.9+）与 **torch.distributed.launch**（旧版，行为类似）。

### 7.2 命令行参数（torchrun）

| 参数 | 含义 | 示例 |
|------|------|------|
| --nproc_per_node | 每台机器上的进程数（通常=GPU 数） | 4 |
| --nnodes | 机器总数 | 2 |
| --node_rank | 当前机器编号（0 到 nnodes-1） | 0 或 1 |
| --master_addr | 主节点 IP | 192.168.1.1 |
| --master_port | 主节点端口 | 29500 |
| 脚本后参数 | 传给训练脚本 | --your_args ... |

### 7.3 环境变量（由 torchrun 设置）

- **RANK**：全局 rank。  
- **LOCAL_RANK**：本机 GPU 编号。  
- **WORLD_SIZE**：总进程数。  
- **MASTER_ADDR** / **MASTER_PORT**：主节点地址与端口（多机时必需）。

### 7.4 使用方式

- 单机 4 卡：`torchrun --nproc_per_node=4 train.py`。  
- 多机：每台机器执行一次 torchrun，指定 --nnodes、--node_rank、--master_addr、--master_port；首台为 master，其余 --master_addr 指向首台 IP。  
- 训练脚本入口放在 `if __name__ == "__main__":` 内，避免 spawn 时重复执行。

### 7.5 示例

**单机 4 卡：**

```bash
torchrun --nproc_per_node=4 train.py --your_args ...
```

**多机（2 机，每机 4 卡）：**

- 机器 0：`torchrun --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 --nproc_per_node=4 train.py`
- 机器 1：`torchrun --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=29500 --nproc_per_node=4 train.py`

---

## 八、Checkpoint 与日志

### 8.1 结构说明

- **保存**：通常只在 **rank 0** 保存一次，避免多进程写同一文件；保存前 `dist.barrier()` 保证所有进程已到保存点。  
- **加载**：可只在 rank 0 读文件再 broadcast，或每个进程各读一份（如共享 NFS）；若用 `model.module.load_state_dict(...)`，需在 DDP 包装后对 `model.module` 操作。  
- **日志**：仅 `rank == 0` 时 print 或写 TensorBoard，避免刷屏和重复。

### 8.2 使用方式

- 保存前 barrier，保存后可选再 barrier。  
- 使用 Lightning 时，Trainer 会在 rank 0 保存、写 log，一般无需手写 barrier。

### 8.3 示例代码

```python
def save_checkpoint(model, path):
    if not dist.is_initialized():
        torch.save(model.state_dict(), path)
        return
    dist.barrier()
    if dist.get_rank() == 0:
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, path)
    dist.barrier()

def log_only_rank0(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)
```

---

## 九、完整示例

与项目内 `torch2j6m`、`scripts` 风格接近的**原生 DDP + torchrun** 最小可运行示例。

```python
# train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="NCCL", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup()
    model = MyModel().cuda(local_rank)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    dataset = MyDataset(...)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(10):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            batch = batch.cuda(local_rank)
            out = model(batch)
            loss = out.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

        if rank == 0:
            ckpt = model.module.state_dict() if world_size > 1 else model.state_dict()
            torch.save(ckpt, f"ckpt_epoch_{epoch}.pth")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

**单机 4 卡启动：** `torchrun --nproc_per_node=4 train_ddp.py`

---

## 十、使用 PyTorch Lightning 做 DDP

### 10.1 结构说明

`Trainer(strategy=DDPStrategy(...))` 在内部完成：根据 devices 启动多进程（或检测已有分布式环境）、`init_process_group`、DDP 包装、为 DataLoader 自动注入 DistributedSampler、仅在 rank 0 写 checkpoint 与 log。用户只需构造普通 DataLoader 与 model，调用 `trainer.fit(model, train_dataloader, val_dataloader)`。

### 10.2 使用方式

- 单机多卡：`Trainer(devices=4, accelerator="gpu", strategy=DDPStrategy(...))`，Lightning 会 spawn 多进程。  
- 多机：仍用 **torchrun** 在每台机器上启动，设置 NNODES、NODE_RANK、MASTER_ADDR、MASTER_PORT；Lightning 检测到已有分布式环境会加入，不再重复 spawn。  
- `DDPStrategy(find_unused_parameters=True)` 与原生 DDP 含义相同。

### 10.3 示例（与本项目 pl_train.py 一致）

```python
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

trainer = pl.Trainer(
    max_epochs=100,
    devices=4,
    accelerator="gpu",
    strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
    logger=...,
    callbacks=...,
)
trainer.fit(model, train_loader, val_loader)
```

---

## 十一、速查与小结

### 11.1 组件/概念速查表

| 组件/概念 | 作用 | 解决的问题 |
|-----------|------|------------|
| **进程组 init** | 建立通信组、选定 backend | 多进程如何发现彼此、用什么后端通信 |
| **Rank / Local Rank / World Size** | 唯一标识与规模 | 谁存盘、谁打 log、数据如何分片 |
| **DDP** | 包装模型并同步梯度 | 多卡数据并行时梯度一致、等价大 batch |
| **DistributedSampler** | 按 rank 分数据索引 | 各卡数据不重不漏 |
| **Barrier / AllReduce / AllGather 等** | 同步与聚合 | 等齐再写文件、汇总指标、收集结果 |
| **torchrun** | 多进程启动与环境变量 | 免手写 spawn、统一 RANK/WORLD_SIZE 等 |
| **Rank 0 写盘与 log** | 单点持久化与日志 | 避免多进程写同一文件、日志刷屏 |
| **Lightning DDPStrategy** | 封装 init/DDP/Sampler/保存 | 标准训练循环下减少样板代码 |

### 11.2 主线串联

按「**启动 → init → DDP + Sampler → 训练循环 → barrier/rank0 保存**」这条线串联，即可覆盖 PyTorch 分布式训练与操作工具的主干用法。

### 11.3 进阶与延伸

- **梯度累积**：在 DDP 下每 N 个 step 再做一次 step/zero_grad，等效更大 batch。  
- **混合精度**：与 DDP 结合使用 AMP（如 torch.cuda.amp），需注意梯度缩放与同步顺序。  
- **多进程 DataLoader**：`num_workers > 0` 时每个训练进程会再起 worker，注意总进程数与资源。  
- **自定义进程组**：多任务时可为不同任务建不同进程组，使用 `process_group` 参数。  
- **弹性训练**：torchrun 支持 `--max_restarts` 等，节点失败时可重启；更复杂弹性可用 PyTorch Elastic。

---

*文档版本：基于 PyTorch 2.x 与 PyTorch Lightning 2.x；与项目 pl_train、torch2j6m、scripts 中的分布式用法保持一致。*
