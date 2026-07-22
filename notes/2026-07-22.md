<!--
知乎发布建议

标题：vLLM 的 P/D 解耦已经很省了，为什么多模态的 KV 还在反复搬？
话题：#vLLM #大模型推理 #多模态生成 #KVCache #分布式系统
-->

# vLLM 的 P/D 解耦已经很省了，为什么多模态的 KV 还在反复搬？

- **关注对象：** [RFC #5244](https://github.com/vllm-project/vllm-omni/issues/5244)
- **当前状态：** Open RFC
- **关注时间：** 2026-07-20 至 2026-07-22
- **适用场景：** AR/Prefill 与 DiT/Generation 解耦部署

---

## 写在前面

系统虽然能跑，但数据移动和调度方式还比较粗：同一请求可能反复传输完整 cache，排在
队首的请求等待传输时会挡住后面的可运行请求，cache 的分配、驻留和释放也缺少统一管理。

我顺着 AR → DiT 这条路径翻了一遍 vLLM-Omni 的代码，发现真正浪费的不只是一次 copy
的时间，而是 DiT 还不能像 vLLM Decode 那样，先查询本地 prefix cache，再决定哪些 KV
需要传输。对于共享稳定上下文的请求——例如同一 prompt 的多次 rollout——DiT 还缺少
“命中 prefix 就跳过”的能力。

在大语言模型里，Prefill/Decode 解耦已经有一套相对成熟的流程：接收端查询本地 cache，
只为缺失部分分配 block，再让发送端把缺失的 KV 写到目标地址。多模态生成的链路更长。
以 Hunyuan、BAGEL 一类 AR-DiT 架构为例，AR 阶段先理解文本和参考图像，DiT 阶段再在
多个 diffusion step 中复用这部分上下文。

[RFC #5244](https://github.com/vllm-project/vllm-omni/issues/5244) 想让 DiT 真正拥有这块
cache。本文重点回答四个问题：

1. 当前的 GPU 空等发生在哪里？
2. DiT 端应该怎样管理和调度 KV cache？
3. AR 到 DiT 的传输如何只搬缺失页面？
4. 哪些能力复用 vLLM，哪些语义留在 vLLM-Omni？

---

## 一、瓶颈不只是“传输慢”

我沿着当前实现看下来，GPU 空等主要来自三个能力缺口。

### 1.1 DiT 还不会先查本地 prefix

**当前路径：** `AR paged KV → gather/flatten → connector → DiT tensor → concat/clone`

AR 和 Generation 阶段已经使用 vLLM 风格的 paged KV 管理，DiT 则主要依赖各模型自己
维护的连续 tensor。缺少本地 prefix 查询、目标 block 预分配和 missing-only transfer，
就很难像 vLLM P/D 解耦那样，只请求本地尚未命中的 suffix。

这个差距在共享稳定上下文时尤其直观：vLLM Decode 会先查本地 prefix cache，再决定传不
传；DiT 还没有这一步，因此两个拥有相同稳定上下文的请求仍可能分别获取完整 context。

### 1.2 KV 等待发生在 forward 之前

[`DiffusionModelRunner._prepare_request_for_forward`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/worker/diffusion_model_runner.py)
会先消费预取结果，或同步接收分布式 KV：

```python
if use_prefetch and self._kv_prefetch_enabled:
    self.kv_transfer_manager.consume_and_distribute_kv_cache(...)
else:
    self.kv_transfer_manager.receive_multi_kv_cache_distributed(...)
```

请求进入 runner 后才等待远端传输或 H2D copy，“等待数据”和“可以计算”没有在调度层
被区分。排在前面的 remote miss 会占住执行路径，即使队列里还有 cache 已经 ready 的
请求。

### 1.3 DiT 缺少统一的 page lifecycle

连续 tensor 没有统一的 page、引用计数和驻留状态，很难同时表达 HBM 命中、CPU
offload、远端缺失、取消和超时。cache 的所有权仍藏在模型内部，而不是由 scheduler、
manager 和 runner 协作管理。

**这一阶段的核心矛盾不是带宽不够，而是 DiT 看不见 cache 状态，也就无法围绕 cache
做调度。**

---

## 二、RFC 的核心：让 DiT 真正拥有 cache

RFC 给出的方案可以浓缩成三个组件：

- **`DiTKVCacheManager`**：管理语义页面与物理 block，记录 HBM、CPU 和远端驻留状态，
  负责分配、引用计数、保留与释放。
- **cache-aware scheduler**：在请求进入 denoising batch 前先生成 cache plan，不再让
  runner 隐式等待。
- **显式 transfer session**：把发送方租约、接收方预留、传输完成、取消、超时和失败
  放进同一个生命周期。

**目标路径：** `AR KVCacheManager → session/connector → DiTKVCacheManager → BlockTable
→ paged attention`

这里最重要的变化，是 cache 的所有权从“模型拿到一个 tensor 再自行处理”，转移到
manager、scheduler 和 runner 共同维护的 page lifecycle。Orchestrator 只负责建立
session 和传递路由信息，页面计划与数据传输由两端直接完成。

**这不是给旧路径再包一层接口，而是把 KV 从模型私有数据提升为调度器能够管理的系统
资源。**

---

## 三、调度器要区分“没数据”和“没容量”

当 DiT scheduler 收到请求时，它不应该立刻把请求塞进 denoising batch，而是先回答三个
问题：需要哪些稳定页面、这些页面在哪里、当前还有多少传输和 HBM 预算。

RFC 设计了四类结果：

- **HBM 全命中：** acquire refs，立即进入 `READY`。
- **CPU 命中：** 预留 HBM，异步 H2D，进入 `WAITING_FOR_LOCAL_KVS`。
- **远端缺失：** 预留目标页面并接收数据，进入 `WAITING_FOR_REMOTE_KVS`。
- **容量不足：** 继续等待，或按策略驱逐、抢占。

只要还有 `READY` 请求，DiT 就能继续组 batch，不必被队首的远端 miss 拖住。

**这份 RFC 真正想消除的是 GPU bubble；missing-only transfer 只是其中一环。**

---

## 四、三个最关键的取舍

我看完 RFC 和后续讨论后，认为实现开始前最需要收敛的是下面三个选择。

### 4.1 推送方向：receiver-driven + sender-push

接收端先查询本地 cache，只为缺失页面预留目标 block，再把缺失页面与目标地址告诉 AR；
AR 保持 source-page lease，通过 Mooncake RDMA WRITE 写入目标地址；DiT commit 后返回终态
ACK，AR 才释放源页面。

后续代码审阅指出，Mooncake Transfer Engine 已经接近这种 receiver-driven plan 加
sender-push 的行为。真正缺少的是跨阶段的 lease、commit 和 terminal ACK 生命周期，
以及首期 transport path 的明确选择。

**结论：传输方向不是最大的悬念，完整的资源生命周期才是。**

### 4.2 Connector 边界：上层 Omni，下层复用 vLLM

RFC 后续讨论已经收敛出 hybrid 方向：scheduler 和 runner 面向
`OmniConnectorBase`，继续使用 Omni 的 session、stage DAG、CFG 和 artifact 语义；具体
connector 则继承 vLLM 的 `KVConnectorBase` / `ECConnectorBase`，复用 paged KV 传输、
completion ACK、source lease 和 delayed-free。

**结论：通用传输机制向 vLLM 靠拢，多阶段生成语义留在 Omni。**

### 4.3 Manager 复用层级：完整 KVCacheManager，还是只有 BlockPool

`experimental/ar_diffusion` 已经在 DiT engine 中复用了完整的 vLLM `KVCacheManager` 和
paged KV stack，并计划迁移到这份 RFC 的架构。它是一个已经存在的验证样例。

但 RFC 当前更倾向于新建只复用 `BlockPool` 的 `DiTKVCacheManager`。如果最终选择后者，
就需要明确完整 manager 在 DiT 的 stable context、dynamic latent 和 CFG 语义下究竟哪里
不适用；否则迁移反而会降低复用层级。

**结论：实验代码已经证明“完整复用”能跑，RFC 需要证明为什么还要退到 BlockPool。**

这三个取舍分别决定传输协议、抽象边界和 cache manager API。它们比继续增加接口更值得
优先确定。

---

## 五、方向合理，但性能收益仍待证明

RFC 预期 missing-only transfer、异步 copy 和 cache-aware scheduling 能减少传输量与
GPU 空闲时间。这个推断有清晰的机制支撑，但目前还没有 A/B 数据，不能写成“已经提速”。

后续至少应该固定一个 pilot model，并记录：

- AR → DiT 实际传输字节数；
- DiT GPU idle ratio 或 KV 等待时间；
- cache hit rate 与 HBM/CPU/remote 分布；
- 相同 workload 下的端到端吞吐与延迟；
- 取消、超时和 worker failure 时是否存在页面泄漏或 stale completion。

语义上也还有三个风险：CFG 分支是否真的能共享页面、dynamic latent 如何在 step 之间被
结构性地禁止复用，以及不同 DiT pipeline 从连续 tensor 迁移到 paged cache 的成本。

**性能方向值得验证，但在传输字节、GPU 空闲和端到端吞吐数据出现之前，所有收益都还是
预期。**

---

## 六、总结：它要优化的是“等待关系”，不只是 KV copy

我的判断是，这份 RFC 最有价值的地方，不是再造一个 KV 传输接口，而是把 AR → DiT 的
数据依赖从 runner 内部的同步动作，提升成 scheduler 能够理解和管理的状态。

理想状态下，DiT 只接收本地缺失的稳定页面；等待远端 KV 的请求不会挡住已经 ready 的
请求；source lease、target reservation 和 completion 由 session 串成完整生命周期；
通用 paged-cache 能力复用 vLLM，多模态与 diffusion 语义留在 Omni。

**读完这份 RFC 最值得记住的一句话是：要减少的不是某一次 KV copy，而是 GPU 因为看不见
数据依赖而产生的等待。**

接下来我会重点看三个证据：manager 复用边界能否确定、首个模型能否完成端到端迁移，
以及传输字节和 GPU bubble 是否真的下降。

---

## 参考资料

- [RFC #5244：vLLM-Omni multi-stage KV cache transfer manager](https://github.com/vllm-project/vllm-omni/issues/5244)
- [基于当前代码路径的设计审阅](https://github.com/vllm-project/vllm-omni/issues/5244#issuecomment-5042728136)
- [`experimental/ar_diffusion` 与 connector 分层的后续讨论](https://github.com/vllm-project/vllm-omni/issues/5244#issuecomment-5042757928)
