# 设计备忘：MTP spec 周期的 CUDA graph 捕获 + overlap 恢复（P1/P2）

> 状态：**设计完成，未实现**。两项都必须在真实 GPU 上按验证阶梯落地——本仓库的
> spec 路径（`engine/spec.py`）历史上每一步深挖审计都修出过严重 bug（页表竞态、
> stride 误读、GDN 部分接受语义），盲写图捕获代码的回归风险不可控。本文保留全部
> 分析，供有 GPU 的会话直接开工。

## 现状（为什么 spec 周期现在是 eager + 不 overlap）

- `engine/graph.py:189-197` `can_use_cuda_graph` 排除 `batch.spec_num_draft_tokens`：
  verify 批是 decode 相但每请求 K+1 行，bs×1 的 decode 图无法回放。
- `scheduler/scheduler.py` `overlap_loop`（"Spec cycles never overlap…" 注释块）：
  spec 周期两侧强制 drain。原因有二：
  1. **页表竞态**：spec 周期对页表是裸读（draft 步的 metadata gather 在引擎流上
     周期中途执行），而相邻批 drain 的 radix 提交会重指向页表行/释放页。plain
     decode 有 captured-replay 的页表快照保护，spec 没有。
  2. **跨度快照**：调度→drain→launch 顺序下 speculator 会读到 drain 后推进的
     Req lens——已用 `Batch.spec_req_lens` 调度时快照修复。

## P1：CUDA graph 捕获

### 核心障碍：周期内形状是数据依赖的

`MTPSpeculator.forward_batch`（spec.py:207-340）：

1. draft 循环的活跃集 `active = [i for i in range(bs) if k[i] > j]` 逐步缩小
   （K 被输出预算钳制的请求提前退出）→ 每步 batch 形状不同；
2. verify 行数 `total = sum(k_i + 1)` 可变；
3. GDN conv 窗口重建 `accepted_host = accepted.tolist()` 是周期内主机同步；
4. 每步 `ctx.attn_backend.prepare_metadata(d_batch)` 做主机侧构建。

### 捕获策略：uniform-span 门控

**只对全批 k_i == K 的周期捕获**（远端输出预算的常态；混合/钳制跨度回退
eager）。此时 (bs, K) 唯一决定所有形状：

- draft 步 j 的活跃集恒为 [0, bs)，`total = bs*(K+1)`；
- 图键 = (bs, K)，捕获集 = 常用 bs 列表 × {K}（K 由 flag 固定）。

### 需要搬进静态缓冲的输入（replay 前 copy_from）

| 张量 | 现状来源 | 图内消费点 |
|---|---|---|
| staged tokens [bs] | `batch.input_ids` | draft 步 0 输入 |
| draft KV out_loc [bs, K]（flat 布局） | `batch.out_loc` | 每步 `store_kv` |
| positions [bs] | lens 派生（CPU int 列表→张量） | 每步 rope/metadata |
| 页表行快照 | **新增**：cycle 开始时把每请求 [0, cached_len) 的页表行 copy 进静态 scratch | draft/verify 的 attention paged 前缀 gather |
| stash prev_hidden [bs, H] | `self._stash.index_select` | draft 步 0（改为图内 gather） |
| gdn slot ids [bs] | `r.linear_slot_idx` | verify 的状态 gather/scatter |

draft 链的 token 依赖（`cur_tokens.index_copy` / `prev_hidden.index_copy`）、
`greedy_accept`（纯张量）、SSM 散写（`ssm_buffer[:, arange, accepted]` gather +
index_copy）**都已是设备算子，可直接入图**。

### 必须重写为纯设备算子的两处

1. **conv 窗口重建**（spec.py:319-325）：现在是 per-layer×per-request 的 Python
   循环 + `accepted_host`。设备化：`joined = cat([snap_cols, conv_in.t()])` 后按
   每请求在列轴 `ker_m1 - 1 + n_i` 处 gather（n_i = accepted），一次向量化。
2. **draft 步 metadata**：`prepare_metadata` 的输出必须捕获期一次构建、replay
   期只 copy 输入张量。多 query extend（verify）路径的 cu_seqlens 为常量
   （uniform span 下 = arange 步进 K+1），可静态化——这正是 GraphCaptureBuffer
   对 decode GDN 的 `fla_cu_seqlens` 已采用的手法（graph.py:29-30）。

### 验收阶梯（顺序执行，任何一步失败即停）

1. `dist/gen_synthetic_model.py` 合成模型 + `tests/e2e/test_mtp_lossless.py`：
   图开/关 逐 token 一致；
2. 合成模型 K=2/K=3、bs∈{1,2,4}、混合跨度（预算钳制）回退路径正确；
3. 真实 checkpoint（NVFP4 35B）单请求 plain vs spec(图) tok/s 对比 +
   `bench_decode_moe.py`；
4. 多请求批量 e2e（WSL 验证机曾因 dxg 驱动不稳定无法跑——在原生 Linux 上补）。

env 门控：`FREETOKEN_SPEC_CUDA_GRAPH=1`（默认关），稳定后转正。

## P2：overlap 恢复

P1 的页表行快照同时解锁 P2：spec 周期改为从 scratch 快照读页表后，相邻批的
drain（radix 提交重指向/释放页）不再与裸读竞争，`overlap_loop` 中 spec 两侧的
强制 drain（scheduler.py 注释块 + `spec_num_draft_tokens` 两个条件分支）可删。

注意点：

- 快照必须在**调度时**（页分配完成、radix 提交之前） copy，与
  `Batch.spec_req_lens` 同一时点，保证与 positions/out_loc 一致；
- N+1 批的页分配仍然依赖 N 批的接受数（`free_spec_tail` 在 drain 侧），所以
  **调度**仍需等 drain，可 overlap 的只是"上一批 drain 处理 + 消息循环"与
  "本批 GPU 执行"（同 plain 路径的 overlap 语义）；
- 先落地 P1 图捕获再开 P2（图内读快照天然免疫中途重指向，eager 路径也要快照）。

## 参考线：实测收益预期

- 35B plain 基线 6.17 tok/s（3060 12GB，offload 512 槽）；合成模型 spec K=2
  零接受率下 1.28x（104.8→133.8 tok/s）——真实 MTP 头接受率显著更高；
- 图捕获的边际收益集中在小 bs 的 launch 开销（每周期 draft K 次 + verify 一次
  ≈ (K+1)×64 层的 kernel 链，eager 下 launch 占比可观）；
- overlap 恢复消除调度串行化（周期边界 CPU 工作与 GPU 不再互等）。
