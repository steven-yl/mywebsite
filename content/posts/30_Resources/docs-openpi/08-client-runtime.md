# 08 客户端运行时（openpi-client）

> 本章解读独立包 `packages/openpi-client/`：`websocket_client_policy.py`、`action_chunk_broker.py`、`base_policy.py`、`msgpack_numpy.py`、`image_tools.py`，以及 `runtime/` 下的 `Runtime`、`Agent`、`Environment`、`Subscriber`。这是部署在机器人侧的轻量库。

---

## 8.1 为什么有一个独立的客户端包

机器人侧的控制计算机往往算力弱、依赖环境受限，不适合安装 JAX/PyTorch/transformers 等重型依赖。openpi 的解法是把客户端逻辑抽到一个**零重依赖**的独立包 `openpi-client`：

- 它只依赖 `numpy`、`websockets`、`msgpack`、`pillow`、`dm-tree` 等轻量库。
- 它通过 WebSocket 调用远端 GPU 上的策略服务（见 [07 章](07-inference-policy-serving.md)）。
- 物理环境（机器人）与策略环境（模型）由此彻底解耦——可以在强力 GPU 上跑模型，机器人侧只管采集观测、执行动作。

```
packages/openpi-client/src/openpi_client/
├── base_policy.py              # BasePolicy 抽象接口
├── websocket_client_policy.py  # WebSocket 客户端策略
├── action_chunk_broker.py      # 动作块逐步分发
├── msgpack_numpy.py            # numpy-aware msgpack 编解码
├── image_tools.py              # 轻量图像 resize（PIL）
└── runtime/                    # 控制闭环编排
    ├── runtime.py              # Runtime 主循环
    ├── agent.py                # Agent 抽象
    ├── environment.py          # Environment 抽象
    └── subscriber.py           # Subscriber 抽象（数据记录/可视化）
```

---

## 8.2 策略接口 `BasePolicy`（base_policy.py）

```python
class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """从观测推断动作。"""
    def reset(self) -> None:
        """重置策略到初始状态。"""
```

服务端的 `Policy`（[07 章](07-inference-policy-serving.md)）和客户端的 `WebsocketClientPolicy`、`ActionChunkBroker` 都实现这个接口——这让本地策略和远程策略可以互换。

---

## 8.3 WebSocket 客户端策略（websocket_client_policy.py）

```python
class WebsocketClientPolicy(BasePolicy):
    def __init__(self, host="0.0.0.0", port=None, api_key=None):
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None: self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self):
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None,
                                                       additional_headers=headers)
                metadata = msgpack_numpy.unpackb(conn.recv())   # 接收服务端首帧 metadata
                return conn, metadata
            except ConnectionRefusedError:
                time.sleep(5)   # 服务端未就绪则重试

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))      # 发送观测
        response = self._ws.recv()
        if isinstance(response, str):              # 字符串响应 = 服务端报错
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)     # 返回动作

    def reset(self): pass
    def get_server_metadata(self) -> Dict: return self._server_metadata
```

**要点**：
- 构造时**阻塞等待服务端就绪**（连接被拒则每 5 秒重试），并接收服务端首帧 metadata。
- `infer`：msgpack 打包观测 → 发送 → 接收 → 解包。约定：服务端回二进制是正常动作，回字符串是错误（traceback）。
- 可选 `api_key` 通过 HTTP header 传递（服务端本身不强制鉴权，这是给前置网关用的）。

---

## 8.4 动作块分发 `ActionChunkBroker`（action_chunk_broker.py）

```python
class ActionChunkBroker(BasePolicy):
    """把策略返回的动作块逐步分发，块用完才再次推理。"""
    def __init__(self, policy, action_horizon):
        self._policy = policy; self._action_horizon = action_horizon
        self._cur_step = 0; self._last_results = None

    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)   # 真正请求一次，拿一整块
            self._cur_step = 0
        results = tree.map_structure(lambda x: x[self._cur_step, ...] if isinstance(x, np.ndarray) else x,
                                     self._last_results)    # 取当前步
        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None                       # 块用完，下次重新推理
        return results
```

**解决什么问题**：模型一次推理输出一整段动作（action_horizon 步），但机器人控制循环每个 tick 只执行一步。`ActionChunkBroker` 包装底层策略，把一块动作缓存下来逐 tick 吐出，块耗尽才发起新的推理请求——大幅减少推理调用次数（推理可能很慢/很贵），同时保持高频控制。

```
tick 1: infer(obs) → 真请求 → 缓存 [a0..a49] → 返回 a0
tick 2: 返回 a1（不请求）
...
tick 50: 返回 a49，块耗尽
tick 51: infer(obs) → 再次真请求 ...
```

---

## 8.5 msgpack-numpy 编解码（msgpack_numpy.py）

```python
def pack_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype.kind in ("V","O","c"): raise ValueError(...)
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def unpack_array(obj):
    if b"__ndarray__" in obj: return np.ndarray(buffer=obj[b"data"], dtype=..., shape=...)
    if b"__npgeneric__" in obj: return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
```

**为什么选 msgpack-numpy**（注释里给出的理由）：
- **安全**：不像 pickle/dill 那样可执行任意代码。
- **跨语言**：msgpack 有广泛的多语言支持。
- **无需 schema**：动态类型语言友好。
- **快**：对大数组比 pickle 约快 4 倍。

实现把 numpy 数组序列化为 `{data: bytes, dtype, shape}` 的 dict（改编自 msgpack-numpy 库，但不回退到 pickle，从而保持安全性）。客户端与服务端用同一套编解码，构成通信协议的基础。

---

## 8.6 轻量图像处理（image_tools.py）

客户端版图像工具（用 PIL，不依赖 JAX）：

- `convert_to_uint8(img)`：float 图像 → uint8（×255），减小网络传输体积。
- `resize_with_pad(images, height, width, method=BILINEAR)`：批量复刻 `tf.image.resize_with_pad`（PIL 实现），保持长宽比缩放后零填充。
- `_resize_with_pad_pil(image, height, width, method)`：单图实现（注意 PIL 用 (width, height) 顺序）。

> 机器人侧通常先把图像转 uint8、resize 到合适尺寸再发送，减少带宽占用。

---

## 8.7 控制闭环编排（runtime/）

`runtime/` 提供一套抽象，把"观测→决策→执行"组织成可复用的回合制循环。

### 抽象接口

- `Agent`（agent.py）：决策者。`get_action(observation) -> action`、`reset()`。通常内部持有一个 `BasePolicy`（如 `ActionChunkBroker(WebsocketClientPolicy(...))`）。
- `Environment`（environment.py）：机器人及其环境。`reset()`、`is_episode_complete() -> bool`、`get_observation() -> dict`、`apply_action(action)`。
- `Subscriber`（subscriber.py）：事件订阅者（数据记录、可视化）。`on_episode_start()`、`on_step(obs, action)`、`on_episode_end()`。

### `Runtime`（runtime.py）：主循环

```python
class Runtime:
    def __init__(self, environment, agent, subscribers, max_hz=0, num_episodes=1, max_episode_steps=0): ...

    def run(self):
        for _ in range(self._num_episodes):
            self._run_episode()
        self._environment.reset()   # 最后复位（真机回 home 位很重要）

    def _run_episode(self):
        self._environment.reset(); self._agent.reset()
        for s in self._subscribers: s.on_episode_start()
        self._in_episode = True; step_time = 1/self._max_hz if self._max_hz > 0 else 0
        while self._in_episode:
            self._step()
            # 按 max_hz 控制频率：不足则 sleep 补足
            ...
        for s in self._subscribers: s.on_episode_end()

    def _step(self):
        observation = self._environment.get_observation()
        action = self._agent.get_action(observation)    # 内部可能走 WebSocket → 服务端
        self._environment.apply_action(action)
        for s in self._subscribers: s.on_step(observation, action)
        if self._environment.is_episode_complete() or (max_episode_steps>0 and steps>=max_episode_steps):
            self.mark_episode_complete()
```

- `run_in_new_thread()`：在新线程跑主循环（非阻塞）。
- `mark_episode_complete()`：标记回合结束。
- `max_hz`：控制循环频率，每步若耗时不足则 `sleep` 补齐（保证恒定控制频率）。

### 闭环示意

```
Runtime._step（按 max_hz 循环）
   │ environment.get_observation()         # 读相机/关节
   ▼ agent.get_action(obs)                 # → ActionChunkBroker → WebsocketClient → 服务端推理
   │ environment.apply_action(action)      # 驱动机器人
   ▼ subscribers.on_step(obs, action)      # 记录/可视化
   └ 检查 episode 是否结束
```

> **职责分离**：`Runtime` 不关心 agent 是本地还是远程、environment 是真机还是仿真。具体实现（如 ALOHA/LIBERO 的 Environment、Agent）在 `examples/` 各目录下提供。

---

## 8.8 端到端组装示例

把本章组件拼起来，机器人侧的典型用法（伪代码，基于 examples 模式）：

```python
from openpi_client import websocket_client_policy, action_chunk_broker
from openpi_client.runtime import runtime

# 1) 连接远端策略服务
ws_policy = websocket_client_policy.WebsocketClientPolicy(host="192.168.1.10", port=8000)

# 2) 用动作块分发包装（一次推理，多步执行）
policy = action_chunk_broker.ActionChunkBroker(ws_policy, action_horizon=50)

# 3) 把 policy 包成 Agent（examples 中的具体实现）
agent = MyRobotAgent(policy)

# 4) 组装 Runtime 闭环
rt = runtime.Runtime(
    environment=MyRobotEnvironment(),     # 真机/仿真
    agent=agent,
    subscribers=[MyDataSaver()],          # 可选：记录数据
    max_hz=50,                            # 50Hz 控制
    num_episodes=10,
    max_episode_steps=1000,
)
rt.run()
```

---

## 8.9 小结

- `openpi-client` 是零重依赖的机器人侧库，通过 WebSocket + msgpack-numpy 调用远端策略服务。
- `WebsocketClientPolicy` 实现远程 `infer`；`ActionChunkBroker` 把一块动作逐步吐出以减少推理调用。
- `Runtime` + `Agent`/`Environment`/`Subscriber` 抽象出可复用的回合制控制闭环，对 agent/env 的具体实现无关。
- msgpack-numpy 选型兼顾安全、跨语言、无 schema 与高性能。

下一章 [09 PyTorch 实现](09-pytorch-implementation.md) 回到模型层，详解 PyTorch 版的 PI0Pytorch 与混合专家实现。
