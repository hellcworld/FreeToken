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

## 版本说明 / Versioning

- 本 fork 标签:**`qwen38-vision-v1.0.0`**(见 git tag)
- 基线:上游 `FlashML-org/FreeToken` main@`2757bb5`(2026-08-24,位于上游
  tag `v0.1.2` 之后);上游 API/内核兼容性随上游
- 上游更新合并策略:`git fetch origin && git rebase origin/main feat/qwen-vision`
  (两个提交按主题独立,便于向上游拆分提 PR)

## 已知边界 / Known limitations

- 本 fork 的验证全部在 CPU 加载器层 + 数值对齐完成(开发机无 NVIDIA GPU);
  **GPU 端到端 `ft serve` 冒烟(≥24GB 显存)尚未执行**,稠密 27B NVFP4 需约
  22GB 显存全量驻留
- 在线 OpenAI/Anthropic server 的图像输入(image_url)未实现(gemma4 亦无)
- 视频输入(时间戳拆帧)未实现;MTP 投机解码权重仍按上游丢弃
- 上游相关 issue:#139 / #67 / #106(Qwen3.8-27B)、#79(多模态路线图)

## License / 许可证

本项目继承上游 **Apache License 2.0**(见 [LICENSE](LICENSE),版权归
FlashML-org/FreeToken 及其贡献者)。本 fork 的新增代码同样以 Apache-2.0
发布,无额外条款。所引用的模型检查点(Qwen、Gemma 等)版权归各自所有者,
遵循其原始许可。
