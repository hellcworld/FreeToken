# Fork 变更说明 / Fork Notes

本仓库是 [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) 的 fork,
在上游 `main`(`2757bb5`,v0.1.2 之后)之上叠加了两个功能提交,均在分支
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
- 视频输入(时间戳拆帧)未实现;MTP 投机解码权重仍按上游丢弃
- 上游相关 issue:#139 / #67 / #106(Qwen3.8-27B)、#79(多模态路线图)

## License / 许可证

本项目继承上游 **Apache License 2.0**(见 [LICENSE](LICENSE),版权归
FlashML-org/FreeToken 及其贡献者)。本 fork 的新增代码同样以 Apache-2.0
发布,无额外条款。所引用的模型检查点(Qwen、Gemma 等)版权归各自所有者,
遵循其原始许可。
