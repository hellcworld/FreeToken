# Fork 变更说明 / Fork Notes

本仓库是 [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) 的 fork,
在上游 `main`(`2757bb5`,v0.1.2 之后)之上叠加了功能提交,均在分支
`feat/qwen-vision` 上(线性堆叠):

| 本 fork 提交 | 内容 |
|---|---|
| `f1535a1` | **Qwen3.8-27B NVFP4 双方言支持**(修复 #139 一类加载崩溃) |
| `89d2189` | **Qwen3.5/3.6/3.8 视觉输入**(对齐 gemma4 的支持级别) |

## 一、Qwen3.8-27B NVFP4 支持(`f1535a1`)

Qwen3.8-27B 与已支持的 Qwen3.6-27B 张量布局完全一致,问题纯在量化方言:

- **modelopt 方言**(RadixArk 等):检测增加 `hf_quant_config.json` 兜底——
  导出流程若剥离了 config.json 中的 quantization_config 副本(Desktop catalog
  构建),旧代码会误判为未量化、落入"纯 NVFP4 默认循环"并在加载期逐张量走
  GPU 反量化,在 sm_120/WSL 上以误导性的 "device not ready" 崩溃(#139)。
- **compressed-tensors 混合方言**(unsloth:56 层 FP4 MLP + FP8 逐通道注意力 +
  末 8 层 FP8 MLP):FP8 投影走 per-tensor-FP8 W8A16 路径(逐通道尺度即逐行
  尺度),fp8 lm_head 反量化为 bf16,末 8 层 FP8 MLP 经**加载时 RTN 重量化**
  折入原生 NVFP4 缓冲(新模块 `models/nvfp4_quant.py`:CPU 参考反量化 + RTN
  量化器,与 triton 内核语义逐位一致)。
- 附带加固:`_try_fuse` 拒绝 dtype/形状不一致的融合组(精确报错到键名)、
  NVFP4 反量化 CPU 回退、`ShardReader.__contains__`。

**验证**:10 个 CPU 单测(`tests/models/test_qwen3_5_quant_loading.py`,合成
迷你检查点);真实检查点加载器级验证——nvidia/Qwen3.6-27B-NVFP4(金标)与
RadixArk/Qwen3.8-27B-NVFP4 发射清单逐键/逐形状/逐 dtype 一致(1173 张量、
零错误),unsloth/Qwen3.8-27B-NVFP4 从"第 7 个张量崩溃"修复为完整契约
(1043 张量、零错误)。

## 二、Qwen 家族视觉输入(`89d2189`)

与 gemma4 相同的支持级别(离线 `LLM.generate(mm_inputs=...)`;在线 server
仍为 text-only):

- **视觉塔 + merger**(新文件 `models/qwen3_5_moe/vision.py`):HF
  `Qwen3_5VisionModel` 的逐位复刻(Conv3d patch 嵌入、位置表双线性
  align_corners 重采样、融合 QKV 按帧双向注意力 + 2-D 旋转、tanh-GELU MLP、
  patch merger)。**数值对齐:与 transformers 参考实现同权重输出 rtol/atol
  2e-4 内一致**(`tests/models/test_qwen3_5_vision.py`,CPU fp32)。
- **交错 mRoPE**:全注意力层 64 旋转维按 `i%3` 选 T/H/W 轴(频率连续);
  多模态 prefill 携带 3 行位置走显式 cos/sin 路径;**文本路径三行相等时严格
  退化为原 partial rope,零回归**;`qwen3_5_rope_index()` 按参考
  `get_rope_index` 语义构建位置与 decode rope_delta。
- **加载与管道**:三条量化路径保留 `model.visual.*`(映射到共享
  `vision_tower.`/`embed_vision.` 前缀);`mm_positions`/`mm_rope_delta` 贯穿
  UserMsg/PendingReq/调度器;`llm.generate` 拿到 `grid_thw` 自动构建 mRoPE 行。

使用:`FREETOKEN_LOAD_VISION=1` 开启(约多占 0.9 GiB bf16;**FTW 转换也必须
带此环境变量,否则视觉权重不会写入 FTW**)。

**验证**:5 个新单测全绿;旧量化测试回归干净;真实检查点(RadixArk)
视觉开启后发射 1173 文本 + 333 视觉张量、零错误。

## 三、fp8 KV cache(`--kv-cache-dtype fp8_e4m3`,对应上游 issue #280)

opt-in 的 FP8 分页 KV 缓存:K/V 以 e4m3 存储,逐 (token, head) 动态 FP32
scale,在 Triton attention kernel 内反量化(解码 + scale 后仍走 bf16 dot +
fp32 累加)。GQA 模型(head_dim=128)KV 容量约 **1.94x**(4096 → 2112
字节/token/层,含 3% scale 开销);精度上量化是唯一误差源——单步 RNE 舍入,
逐元素相对误差 ≤ 6.25%(e4m3 格式上限),softmax 加权平均进一步压缩,
近无损。

- **写入路径**(`kvcache/mha_pool.py`):`store_kv` 逐 (token, head) 行
  amax → scale = amax/448,RNE 量化后复用现有 CUDA `store_cache`
  逐行拷贝;scale slab 随 `index_copy_` 写入。量化原语复用已在 H100 上
  位精确验证过的 e4m3 语义(权重 fp8 同一套)。
- **读取路径**(`kernel/triton/attention.py`):decode/extend/fallback 三个
  kernel 的缓存读取循环加 `KV_FP8` 分支,以 uint8 视图装载(单一代码路径,
  native/emulated 一致),`e4m3_u8_to_f32` 精确解码 × scale 反量化。
- **配置与校验**(`--kv-cache-dtype`):仅 CUDA + 全局 bfloat16 + Triton
  后端 + 纯 MHA/GQA 模型(排除 MLA/DSA/SWA/DSV4/BSA);开启后 auto 后端
  解析将 triton 提到首位(sm90 上不再默认 fa,fi),显式指定封闭厂商后端
  (fa/fi/trtllm)直接报错。内存核算(`spec_kv_bytes_per_token`)自动计入
  fp8 与 scale 字节。

使用:`ft serve --model ... --kv-cache-dtype fp8_e4m3`(显式
`--attention-backend triton` 可省略,auto 会解析到)。

**实测数据**(RTX 3060 12GB,`nvidia/Qwen3.6-35B-A3B-NVFP4` 端到端):

- **精度**:量化往返逐元素相对误差实测 ≤0.059(e4m3 格式上限 6.25%;
  bf16 为 ~0.4%),可精确表示的值零误差;kernel 输出对"反量化精确参考"
  偏差 ≤2e-2(纯累加顺序差异),对 bf16 原值参考 ≤1e-1 且 768 元素中仅
  1 个超出 5%——量化噪声被 softmax 平均进一步压缩,近无损。
- **显存(262K 上下文)**:bf16 KV = 5.00 GiB,fp8+scale = 2.54 GiB
  (启动日志实测一致),省 2.46 GiB(-49%,比值 1.97x)。12GB 卡上 bf16
  池只能按剩余显存缩到 ~131K token;**fp8 才能完整兑现该模型
  `max_position_embeddings=262144` 的原生上下文**。
- **端到端**:`--kv-cache-dtype fp8_e4m3` 冷启动 → auto 后端解析为
  triton → fp8 池分配 262309 token = 2.54 GiB → CUDA 图捕获 + warmup
  通过 → 真实补全正常返回。GDN 线性层的递归状态不经 KV 池,不受影响
  (该模型仅 10/40 层进池;全注意力密集模型同参数收益绝对值 ×4)。

**验证**:`tests/engine/test_kv_cache_dtype.py`(11 项配置矩阵,纯 CPU)、
`tests/kvcache/test_fp8_kv_quant.py`(量化往返精度、逐行 scale、池 slab
几何、内存核算,纯 CPU)、`tests/kernels/test_fp8_kv_attention.py`
(CUDA-gated:store 逐字节一致性 + decode/extend/fallback 对"反量化参考"
与"bf16 参考"双层容差比对)。

## 四、MTP 投机解码(`--speculative-algorithm MTP`)

Qwen3.5/3.6/3.8 checkpoint 都携带一个 MTP 草稿头(`mtp.*` 专用分片,全部三种
量化方言下均为未量化 bf16),上游在加载时直接丢弃。本 fork 将其加载并接入
**草稿-验证**投机解码:草稿头每步自回归草稿 K 个 token,目标模型一次 forward
验证 K+1 个位置。**greedy 输出与不开投机逐 token 一致(数学无损)**——这也是
e2e 测试的判据。

- **草稿头**(`models/qwen3_5_moe/mtp.py`,新文件):DeepSeek 式单块 MTP——
  `fc`(eh_proj)+ `pre_fc_norm_{embedding,hidden}` + 一个 bf16 全注意力解码层
  (复用 `Qwen3_5Attention`,新增 `quant=False` 开关)+ 末层 norm。层内 norm 为
  Gemma 风格(与目标层一致,加载时 +1 bake);头级 `mtp.norm`/`pre_fc_norm_*`
  依据 checkpoint 权重分布(中心 ~1 而非 ~0)判定为标准 RMSNorm。草稿头持有
  **共享页表下的独立 KV slab**(全局层号 `num_layers`),前缀缓存复用、页分配、
  拒绝回滚全部复用现有机制——被拒绝草稿的 target/draft KV 随一次
  `free_spec_tail` 同时回收。
- **加载**(`models/qwen3_5_moe/weight.py`):新增 `_iter_mtp_weights` 旁路
  (复用 q/k/v→`qkv_proj`、gate/up→`gate_up_proj` 融合与 Gemma bake);
  `iter_weights(include_mtp=)` 由 `load_weight` 经签名探测下发,默认路径行为
  不变(mtp.* 仍丢弃)。`_adjust_config` 校验后翻转 `model_config.enable_mtp`
  ——只有开启投机才构建草稿头/加载权重/多分配 KV slab,普通服务零开销。
- **执行**(`engine/spec.py`,新文件):验证 pass 以 `phase="decode"` +
  `extend_len=K+1` 提交——注意力元数据自然路由到 Triton 多 query extend 内核
  (块内因果 + 分页前缀,注意力内核零改动),两个 lm_head 仅在 prefill 做末
  token 聚合,故验证保留全部行。greedy 验收为纯张量前缀匹配;接受的 argmax
  行一次向量化写入 token_pool(接受的草稿 + 边界 bonus token 恰好落位)。
  非 greedy 请求自动降级为普通单 token 周期(整批统一)。**投机周期不与下一
  批重叠调度**(N+1 批的页分配/写回依赖 N 批的接受数,排空后调度)。
- **深挖审计修复(对照可运行原版逐路径对账发现)**:
  1. **[严重] verify 分支 q/k/v 非连续视图被内核误读**——`_conv_prefill`
     返回转置视图,`reshape` 生成语义正确但非 token 主序的视图(stride
     (2,1,64,2)),而 recurrent 内核仅按 stride(1) 步进 token、假定 head/dim
     连续 → 验证 pass 的 GDN 一直在读错位内存(表现为 token 第 27 位起分
     歧)。修复:verify 分支 q/k/v 强制 `.contiguous()`。逐层状态对账证实
     修复后 layer-0 状态链与 plain **逐位一致**(Δ=0.000)。
  2. **[严重] GDN 状态部分接受语义错误(原快照/恢复方案两个分支都不对)**:
     恢复快照 = 周期前状态(差 n+1 个 token),保留终态 = 全 span 后(差
     K-n 个)。重构为 vendored FLA 内核的 **target_verify 模式**
     (`disable_state_update` + `intermediate_states_buffer`):验证不写活跃
     槽,逐步状态入缓冲,验收后散写 step n(探针证实 buffer[s] 与顺序单步
     更新**逐位一致**)。
  3. conv 状态按提交边界重建(快照 + 捕获 conv_in 行 + 窗口回写,纯函数
     `_conv_window` 单测锁定);k=0 全 span=1 批次的零缓冲覆盖守卫;
     graph 重放下 note_decode 的 stale stash 守卫;DecodeManager
     inflight 预留按 spec 跨度;DraftMoE renormalize=True 对齐目标层。
  - 审计中的方法论发现:`causal_conv1d_decode` 会**原地覆盖输入缓冲**;
    triton autotune 配置随缓存冷暖跨进程漂移(对照必须背靠背)。
- **35B 实测排查记录(ftserve 自启干扰排除后)**:`wsl --shutdown` 后
  systemd 会自动拉回 ftserve(17GB 银行)——与测试进程双 35B 挤爆 RAM,
  早期"机器不稳"结论部分由此污染;`systemctl disable --now ftserve` 后
  干净复测仍崩溃,继而定位并修复两个真 bug:(1) **跨流页表竞态**——
  overlap 调度下前一批 drain 的 radix 提交会重指向页表行,而 spec 周期
  对页表是裸读(plain 路径有 captured-replay 快照保护),已改为 spec
  批两侧均不重叠;(2) **跨度快照**——"schedule→drain→launch"新顺序下
  speculator 会读到 drain 后推进的 Req lens,与按调度时 lens 构建的
  positions/out_loc 失配(store_cache 形状断言暴露),已将
  `(cached_len, device_len)` 在调度时快照进 `Batch.spec_req_lens`。
  另将 GDN target_verify 捕获缓冲改为持久化(消除每周期 ~280MB 分配
  扰动,亦是性能改进)。**遗留**:干净环境下 35B spec 仍出现异步非法
  访问(浮现点随机漂移:embedding/GDN GEMM/MoE silu;全部独立内核
  探针通过;同步点可将存活周期从 1-2 延长至 6-12 但不能消除;合成模型
  同代码路径逐位精确)——疑似 spec 周期加重异步负载触发该机 WSL dxg
  栈缺陷,或 35B 特有内核交互 bug,需 compute-sanitizer/原生 Linux
  复现根治。35B plain 基线(126 tok @ 6.17 tok/s,offload 512 槽)有效。
  - **已知良性残余**:bf16 下验证 pass(3 行 GEMM)与 plain(1 行)的
    浮点重结合差异可在近平局处翻转单个 token(合成模型 95 token 中第
    42 位,确定性、状态链精确)——所有 bf16 投机解码实现共有,非本实现
    缺陷。
- **GDN 混合架构支持**(Qwen3.5/3.6/3.8 实际均为 48 linear + 16 full 混合):
  验证 pass 以 `phase="decode"` + 多行 extend 提交,`build_fla_metadata` 与
  GDN 层的 decode 路由改为"每请求一行才走单步内核",多行自然落入 ragged
  chunk 路径(状态按请求连续推进);投机器在验证前快照活跃 GDN 槽位的
  conv+recurrent 状态,拒绝的请求在验收后整体恢复——草稿头本身是全注意力,
  草稿循环不触碰状态池。KV 池的 hybrid layer_ids 映射同步纳入 MTP slab。
- **配置**:`--speculative-algorithm MTP --speculative-num-draft-tokens K`
  (默认 2)。phase 1 限制:Qwen3.5 家族(混合 GDN 支持)、Triton 后端(强制)、
  TP=1、稠密模型(MTP 层为 routed experts 的 MoE checkpoint 报错)、非 SWA/
  DSV4/MLA/BSA、非多模态、非 FTW;CUDA graph 对投机周期关闭(普通周期不受
  影响),offload 后端的 CPU executor 缓冲按 (K+1)×并发放大。

**验证**:`tests/models/test_qwen3_5_mtp.py`(9 项,纯 CPU):mtp 权重
inclusion/exclusion、发射清单(融合/规范 bake)、草稿头 state-dict 键与形状
同加载器输出逐键一致、greedy 验收核(全对/部分对/预算收缩)、DecodeManager
投机跨度(仅全 greedy 批、预算钳制)、`free_spec_tail` 页对齐回滚。
`tests/e2e/test_mtp_lossless.py`(CUDA-gated):双子进程 plain vs spec 同提示
**逐 token 一致性断言** + 吞吐比报告,配 `dist/gen_synthetic_model.py` 生成的
合成混合架构小模型(12GB 卡可跑)。**GPU 实测通过**(RTX 3060 12GB + WSL2
Ubuntu + CUDA 13,greedy 127 token):plain 与 spec 输出**逐 token 完全一致
(TOKEN-IDENTICAL: True)**;spec 133.8 tok/s vs plain 104.8 tok/s(1.28x,
K=2,随机草稿头接受率≈0 下仍净赚——验证 pass 多行摊销)。多请求批量 e2e 在
该验证机受其 WSL dxg 驱动不稳定限制(WSL Service 持续意外终止,事件日志可查,
纯 torch 负载同样触发,与本改动无关);过渡/编译/生成中段均出现过 GPU 队列楔死
(py-spy 定位为 cudaEventRecord 内空转,CUDA_LAUNCH_BLOCKING 下无内核报错,
纯 torch 负载同样触发;在停用 ftserve、Windows 全新重启、缓存全热的最干净
条件下仍随机复现)——确认为该机 WSL dxg 驱动问题而非代码问题,建议换原生
Linux 或更新驱动后复测多请求批量 e2e。单提示对照(plain vs spec)已在
该机完成并通过。回归:tests/scheduler、
tests/engine、tests/kvcache、tests/models、tests/tokenizer、tests/server、
tests/daemon 与改动前基线完全一致(零回归)。

## 五、decode 输出速度优化批次（调度/度量/图尺寸）

面向 35B MoE offload 服务(qwen3.6-35b-a3b)的三项低风险代码改进 + 一份 spec
图捕获设计备忘。零行为变化项默认关闭或保持默认。

### 5.1 专家缓存命中率度量(`--moe-collect-stats` + `/v1/stats` moe 块)

offload 后端 decode 速度的第一决定因素是专家缓存命中率,但
`EngineConfig.moe_collect_stats` 此前只有字段没有 flag,命中率无从观测:

- **flag**:`--moe-collect-stats`(args.py)——把设备侧 miss 计数器捕获进 decode
  CUDA 图(机制上游已有,engine.py 在图捕获前设置)。
- **采样窗口**:调度器按 `--decode-log-interval`(默认 40)个 decode 批为一个
  窗口,窗口边界读一次计数器(设备同步摊销到 1/40)并 `reset_stats()` 重新定界
  ——报告的是**近期**命中率而非寿命平均,直接服务缓存尺寸 A/B。
- **通路**:`DetokenizeMsg`/`UserReply` 新增 4 个 float 字段(miss_rate/
  active_per_layer/missing_per_layer/fetch_rate,-1.0 = 未报告,0.0 是合法值)
  → detokenizer 透传 → `StatsTracker` 保持最近值(同 kv/mamba 语义)→
  `/v1/stats` 输出 `moe` 块(miss_rate + hit_rate 补数 + hybrid 的 fetch_rate,
  未报告时为 null)→ `ft ctl stats` 增加一行。
- 测试:`tests/scheduler/test_moe_stats_window.py`(窗口节拍/门控)、
  `tests/server/test_stats_moe.py`(observe/build_stats 语义)、
  `tests/server/test_message_wire.py` 增线上往返。

### 5.2 调度策略(`--schedule-policy prefill_first|decode_first|interleave`)

上游调度是严格 prefill 优先(scheduler.py 原 TODO):多请求并发时长 prompt 的
chunked prefill(8K/chunk)期间 decode 完全停摆,running 请求输出骤降(实测首
token ~10s 排队即此因)。`Scheduler._policy_pick_batch` 实现三策略,记录已发射
批的 phase 驱动交替:

- `prefill_first`(默认,行为不变):SGLang 式 prefill 优先,TTFT 最优;
- `decode_first`:running 请求优先,排队 prompt 等解码间隙(长输出场景可能饿
  等,显式选择);
- `interleave`:prefill 批间严格插入一个 decode 批——decode 在长 prefill 期间
  保持输出(约半速 prefill 为代价),**多客户端 agent 服务推荐**。
- spec 批语义保留:decode 分支仅在 `decode_manager.runnable` 为真时采纳,其
  `schedule_next_batch` 内的 spec 跨度展开始终落在被发射的批上。
- 测试:`tests/scheduler/test_schedule_policy.py`(三策略 + 单边回退 +
  decode 排空时退化)。

### 5.3 CUDA 图批尺寸(`--cuda-graph-bs` + 默认梯子补 3)

- 默认候选梯子 `[1,2,4]+range(8,...)` 意味着 bs=3 的 decode 批回放 bs=4 的图,
  每步浪费 1/4 行——梯子补 3 为 `[1,2,3,4]+...`(多一张小图,近零显存成本);
- 新增 `--cuda-graph-bs 1,2,3,4` 显式覆盖候选列表(与 DSV4 的 max_running_req
  钳制兼容,engine.py 已有的列表钳制直接生效)。
- 顺带:`--speculative-algorithm/--speculative-num-draft-tokens` 补进
  docs/cli.md(CHANGES 第四节曾记录它们未文档化)。

### 5.4 spec 图捕获 + overlap 恢复:设计备忘(未实现)

`docs/spec-graph-design.md`:uniform-span 门控捕获(图键 (bs,K))、静态
缓冲清单(staged/out_loc/positions/页表行快照/stash/gdn slots)、两处必须设备
化的算子(conv 窗口重建的 gather 化、draft 步 metadata 静态化)、页表快照如何
同时解锁 overlap 恢复,以及四步验收阶梯(合成模型逐 token 一致 → 混合跨度回退
→ 真实 checkpoint A/B → 原生 Linux 多请求 e2e)。盲写不可验证的图捕获代码风险
不可控(spec 路径历史上每轮深挖审计都修出严重 bug),留待有 GPU 的会话按备忘
实现。

### 5.5 RTX 3060 12GB + WSL2 实机 A/B(2026-09-15,116 盒子,nvidia/Qwen3.6-35B-A3B-NVFP4)

部署 `0.1.2+g77d40585b`(fp8 KV + max_running=2 + moe-cache-auto=1859 槽 + auto 18 层
CPU 锁定),3×255 token 流式取中位;双流 = 正在解码的请求 A vs +4s 到达的 16.5K
token 新请求 B:

| 配置 | 单流 decode | A 最大停顿 | B TTFT | 备注 |
|---|---|---|---|---|
| 基线(prefill_first, chunk 8192) | **27.2 tok/s** | **15.66s** | 29.9s | 命中率 91-96% 波动对 tok/s 无感 |
| --moe-backend hybrid | 23.8 tok/s(-12%) | - | - | 与 benchbw 预测一致(CPU/PCIe 比 1.045<2),否决 |
| --moe-cpu-layers 0 | 加载期崩溃 | - | - | bank 页注册路径在 WSL 上硬崩(无栈,疑 dxg 同族),否决 |
| interleave, chunk 8192 | 27.2 tok/s | 13.12s | 25.9s | 交替只能在 chunk 边界挤 1 token/chunk |
| interleave, chunk 2048 | 27.0 tok/s | **5.77s(-63%)** | 40.9s(+37%) | 多客户端 agent 推荐 |

- **瓶颈定位**:96% 命中率下 PCIe miss 仅 ~1.8ms/37ms 步(0.32 miss/层 × 1.78MB
  槽),单流 27 tok/s 的实际上限是 18 层 CPU 锁定 MoE 计算(WSL pin 配额 9.39GB <
  banks 16.93GB 的强制拆分);缓存再平衡、GDN 状态 bf16 对单流 decode 均无感。
- **MTP 在此机不可用**:116 即第四节记录的 WSL 验证机,35B spec 的异步非法 access
  未根治前不要在此开启;spec 图捕获/overlap 恢复见 docs/spec-graph-design.md。
- **运维根因**:WSL 在最后一个会话退出后随之关闭,ftserve 约 10 分钟的冷加载被
  反复 SIGTERM(9-14 晚 21 点的"崩溃循环"与"服务经常是死的"皆此因)。需 Windows
  侧常驻会话保活;临时手段:后台挂 `wsl -d Ubuntu -- bash -c 'while true; do
  sleep 30; done'`。
- 部署工具链:diagnostics/rexec.py(paramiko,FT116_PW 环境变量)+ box_*.sh/py
  A/B 脚本(改 /etc/default/ftserve → 重启 → 1-token 探针等就绪 → 测速)。

## 版本说明 / Versioning

- 本 fork 标签:**`qwen38-vision-v1.0.0`**(见 git tag)
- 基线:上游 `FlashML-org/FreeToken` main@`2757bb5`(2026-08-24,位于上游
  tag `v0.1.2` 之后);上游 API/内核兼容性随上游
- 上游更新合并策略:`git fetch origin && git rebase origin/main feat/qwen-vision`
  (两个提交按主题独立,便于向上游拆分提 PR)

## 已知边界 / Known limitations

- `RadixArk/Qwen3.8-27B-NVFP4` 已在 Alibaba Cloud Linux 4 LTS 的 NVIDIA A10
  24GB、驱动 r580.126.09、CUDA 13.0 上完成端到端 `ft serve` 冒烟。该 dense 27B
  checkpoint 需 `--memory-ratio 0.95 --max-running-requests 1 --kv-reserve-tokens 512`：
  实测约 23.2GB / 24GB 显存、约 28K 有效上下文，仅支持单并发
- 在线 OpenAI/Anthropic server 的图像输入已实现(Qwen3.5/3.6/3.8;gemma4 仍离线):
  需 `FREETOKEN_LOAD_VISION=1` + `freetoken[vision]` extra;tokenizer worker 做
  HF processor 预处理,调度器用常驻视觉塔编码;单请求至多 4 张图。同时修复了
  离线 mm 路径在真实 GPU 上的一串缺陷(dtype/device 错位、meta inv_freq、
  token-pool gather 被 rope delta 偏移等)——离线 `LLM.generate(mm_inputs=...)`
  与在线 API 现均已在 RTX 3060 12GB + WSL2 实机验证
- 视频输入(时间戳拆帧)未实现;MTP 投机解码已实现(phase 1:Qwen3.5 家族
  稠密模型含 GDN 混合 + Triton 后端 + TP=1 + greedy;随机采样验证、MoE 草稿
  专家、CUDA graph 捕获投机周期留待后续)。无损性已在 RTX 3060 + WSL2 实测
  通过(逐 token 一致;K=2 时 1.28x)
- 上游相关 issue:#139 / #67 / #106(Qwen3.8-27B)、#79(多模态路线图)、
  #421(MTP 投机解码)

## License / 许可证

本项目继承上游 **Apache License 2.0**(见 [LICENSE](LICENSE),版权归
FlashML-org/FreeToken 及其贡献者)。本 fork 的新增代码同样以 Apache-2.0
发布,无额外条款。所引用的模型检查点(Qwen、Gemma 等)版权归各自所有者,
遵循其原始许可。
