# 盒子运维手册(Box Ops Runbook)

适用主机:**192.168.100.116**(Windows + WSL Ubuntu 26.04,RTX 3060 12GB,WSL 内存 23GB)
状态基准日:2026-10-10 | 变更史见 [CHANGES.md](../CHANGES.md) 九~十一章及其"〇"节速览
访问通道:SSH `admin@192.168.100.116`(密码经 `FT116_PW` 环境变量,工具
`diagnostics/rexec.py`);WSL 命令 `wsl -d Ubuntu -u root -- bash <script>`

---

## 一、LLM 后端:Bonsai-2-27B(`bonsai.service`)

### 1.1 是什么

PrismML **Ternary-Bonsai-2-27B**(Qwen3.8-27B 基底,~1.7bpw 三值 PQ2_0,整包 7.66GB)
的**去审查消融版**(BoldingBuilds Abliterated-v2,亦是 groxaxo 3060 榜单 96% 第一名
所用文件),单文件内嫁接 MTP 投机头。运行时为 **PrismML llama.cpp fork**(源码
`prism-b10743-adfffbe`,该 tag 已内置 MTP Hadamard 逆变换修复,无需补丁),源码在
`~/llama.cpp-prism`,二进制 `build133/bin/llama-server`。

> 编译备注:Ubuntu 26.04 新 glibc 与 CUDA 12.x 头文件冲突(noexcept 声明),12.8/
> 12.9/13.0 全灭;用 **CUDA 13.3 + gcc/g++-14**(host)编译通过,本机实测稳定
> (官方 KNOWN_ISSUES 有 13.3 崩溃报告,未复现——若日后异常优先怀疑此点)。

### 1.2 服务配置(`bonsai.service`,手动启/懒加载,不开机自启)

```ini
ExecStart=.../llama-server \
  -m .../models/bonsai2/Ternary-Bonsai-2-27B-Abliterated-v2-PQ2_0-MTP.gguf \
  --mmproj .../models/bonsai2/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf   # 官方视觉投影,与 abliterated 主模型兼容
  -ngl 999 -fa on -c 65536 -ctk q8_0 -ctv q8_0 \   # 全层进显存;64K 上下文;KV q8 量化(~1.5GB)
  --jinja --spec-type draft-mtp --spec-draft-n-max 2 \   # 模型聊天模板;MTP 投机(2 草稿)
  --image-min-tokens 1024 --host 127.0.0.1 --port 8012 \
  -np 1 --cache-ram 8192 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \    # 采样兜底=模型卡思考档(GGUF 元数据缺采样参数,必须显式设)
  --alias bonsai2-27b-abliterated
Environment=LD_LIBRARY_PATH=.../build133/bin       # 动态链接库
```

**不要开机自启**:由 gpu-gateway 按需唤醒(见二);同时 ftserve 与 bonsai 各占
~11GB 显存,**同跑必 OOM**——手动操作任何一个前先停另一个。

### 1.3 实测性能(RTX 3060 12GB,64K ctx)

| 指标 | 数值 |
|---|---|
| 冷加载 | 10-20s;显存 11.5-11.8 / 12.3GB |
| decode | 34-49 tok/s(短上下文);~39K ctx 降至 26-28(带宽瓶颈,预期行为);64K 满约 22-24 |
| prefill | ~500 tok/s 且平坦(4K/16K 实测一致)——对比旧 ftserve 8K 即掉 116 |
| MTP | 接受率 0.4-0.7 @temp1.0(贪心可到 0.92);关掉 MTP 会慢 ~40%,别关 |
| 能力 | 视觉(读图准确)、工具调用(JSON 规范)、中文长文连贯 |

### 1.4 文件与常用操作

- 模型目录 `/home/adminuser/models/bonsai2/`:abliterated-v2(现役)、原版
  `Bonsai-2-27B-PQ2_0-MTP.gguf`(备用,带审查版)、mmproj-Q8_0
- 手动启停:`systemctl stop ftserve && systemctl start bonsai`(反之亦然)
- **换回原版 GGUF**:unit 里 `-m` 路径改回原版文件 → `daemon-reload && restart bonsai`
- **换回 ftserve 引擎**:见 2.4

---

## 二、分时管理(`gpu-gateway`)

### 2.1 架构

```
LLM 客户端/Hermes ──> :8010 网关 ──> bonsai :8012(FT_UNIT 可切 ftserve :8011)
ComfyUI UI/caddy  ──> :8188 网关 ──> ComfyUI :8191
```

单进程纯标准库 asyncio(`scripts/gpu_gateway.py`,仓库内;部署于
`/usr/local/bin/gpu-gateway`),对外端口稳定,消费方零感知。

### 2.2 行为规则

- **监控不唤醒**:`/v1/stats`、`/health`、`/gpu-gateway/status` 网关本地应答
- **按需唤醒**:其余请求 single-flight 唤醒后端(systemctl start + 轮询 /health
  status=="ok"),请求扣住等待,就绪后原始字节透传(SSE/WS 天然支持);失败 503
- **空闲停车**:无在途请求且空闲 > `FT_IDLE` → `systemctl stop` 后端,释放 ~11GB
- **生图门禁**:ComfyUI `POST /prompt` 先等 LLM 排空(网关在途 0 + 上游 active 0,
  上限 `COMFY_WAIT_CAP`)再停车放行;超时 503
- **comfy 释放**:ComfyUI 队列空 > `COMFY_FREE_IDLE` → `POST /free unload_all`
- 统计后端无关:llama 走 `/slots`(active=is_processing 计数)+ `/props`(模型名),
  ftserve 走 `/v1/stats`,归一成同一形状

### 2.3 当前参数(`gpu-gateway.service` Environment)

| 参数 | 值 | 含义 |
|---|---|---|
| FT_UNIT | bonsai | 受管 LLM 单元(切 ftserve 即改此项) |
| FT_UP_PORT | 8012 | LLM 后端端口(ftserve 时代 8011) |
| FT_IDLE | **1800** | LLM 空闲 30 分钟停车(2026-10-10 从 600 调高) |
| COMFY_FREE_IDLE | 300 | ComfyUI 空闲 5 分钟释放 |
| COMFY_WAIT_CAP | 600 | 生图等 LLM 排空上限 |
| FT_WAKE_CAP | 300 | 唤醒超时 |

改参流程:编辑 unit `Environment=` → `daemon-reload && restart gpu-gateway`
(约 8 秒窗口,在途连接会断)。

### 2.4 运维与回滚

- 状态:`curl :8010/gpu-gateway/status`(ft_state/inflight/idle/model)或 `gpustat`
- 日志:`journalctl -u gpu-gateway -f`(唤醒/停车/门禁全量可读)
- **切回 ftserve**:unit env 改 `FT_UNIT=ftserve FT_UP_PORT=8011` → 重启网关;
  注意先 `systemctl stop bonsai`,且 ftserve 现在无人代管(用完手动停)

---

## 三、智能体(Hermes 栈)

### 3.1 组成

| 组件 | 运行方式 | 作用 |
|---|---|---|
| hermes-dashboard | 系统服务 :9119 | 网页版 UI + API(重启会断开所有页面 WS,数秒) |
| hermes-gateway | **linger 用户服务**(`~/.config/systemd/user/`) | 消息平台集成(微信),开机自启/崩溃拉起 |
| hermes CLI | `runuser -u adminuser -- hermes ...` | 配置/一次性任务(`-z`) |

### 3.2 关键配置(`~/.hermes/config.yaml`,改后重启对应服务)

| 键 | 现值 | 为什么 |
|---|---|---|
| model.default | bonsai2-27b-abliterated | 显示对齐(llama-server 不校验 model 字段) |
| model.base_url | http://127.0.0.1:8010/v1 | 走网关 |
| **model.context_length** | **65536** | **必设**:llama `/v1/models` 上报 GGUF 元数据 262K 而非 `-c` 实际值;不设则压缩 196K 才触发,必撞 64K 硬顶。设后压缩在 48K 输入触发(0.75 floor),保头 3 条+尾 20 条 |
| dashboard.public_url | https://hermes.infrafixed.com | **必设为公网域名**:WS 一次性票据校验绑定此值,错配=聊天无限重连(已修) |
| dashboard.basic_auth | admin / *(scrypt)* | 网页登录层 |

### 3.3 公网链路

```
浏览器 → CF 边缘 → Windows cloudflared 服务(token 隧道)
      → portproxy 0.0.0.0:8187 → WSL caddy(basic_auth)→ dashboard :9119
```

- **优选 IP**:本网络直连 CF 104.16 网段走 **HKG 27ms**(域名默认解析的 104.21 网段
  是 SEA 170ms;浏览器挂本机代理时走代理出口更差)。方案:hosts 钉
  `104.16.4.14 hermes.infrafixed.com` + 代理客户端加该域直连规则(缺一不可)
- LAN 内直连 `http://192.168.100.116:8187` 最稳,零断联
- 模型调用不经公网:dashboard 后端在 WSL 内部走 127.0.0.1:8010

---

## 四、自愈体系

### 4.1 GPU 楔死看门狗 v2(Windows 计划任务 `GPU-WSL-Watchdog`)

每 5 分钟(SYSTEM/最高权限,`C:\ProgramData\gpu-watchdog\watchdog.ps1`):

```
nvidia-smi 探测(45s)
 ├─ 正常 → 静默
 ├─ 挂起 → 30s 后二次探测 → 恢复=瞬时抖动,不动
 └─ 仍挂 → gateway ft_state=waking? → 冷加载豁免,不动
           否则计 1 分;连续第 2 个周期仍挂(≈10 分钟持续)才动手:
           强制重置 WslService(停不掉杀进程)→ 等 WSL 启动(≤240s)→
           6 条 portproxy 重指向当前 WSL IP → 20s 后复测
           ├─ 恢复 → 记 RECOVERED
           └─ 滚动 1 小时已重置 3 次 → 只记日志"建议手动重启整机"
```

日志:`C:\ProgramData\gpu-watchdog\watchdog.log`(+`state.txt` 连续计数/重置时间戳)。
脚本源:`diagnostics/gpu_watchdog_v2.ps1`。

### 4.2 开机自愈矩阵(WSL 重启后谁来拉起什么)

| 组件 | 启动方式 |
|---|---|
| gpu-gateway / comfyui / hermes-dashboard / caddy | 系统服务自启 |
| hermes-gateway(微信) | linger 用户服务,随 boot 拉起(实测开机 3 秒内) |
| bonsai | **不启动**,首个 LLM 请求唤醒(~12s) |
| ftserve | **不会**启动(ftproxy 已 disable 且 Wants=ftserve 已拆,2026-10-08 事故根治) |

### 4.3 战绩与边界

- 10-09:驱动楔死大半天无自愈(看门狗前身)→ 10-10:同病 4 次全部 2-5 分钟自愈
- 边界:驱动级救不回时看门狗放弃并提示手动重启;重置会断当时活跃会话(设计取舍)
- 病根未除:WSL 3.0.1.0(10-08 更新)为头号嫌疑,高频复发时考虑回退商店包或升级
  NVIDIA 驱动(616.92)

---

## 五、观测

### 5.1 工具

| 工具 | 位置 | 用途 |
|---|---|---|
| `gpustat` | 盒子 `/usr/local/bin/gpustat`(源 `diagnostics/gpustat.sh`) | SSH 一屏:服务/网关/模型/comfy 队列/GPU/RAM |
| `ftmon` | 开发机 `diagnostics/ftmon.py`(+`.cmd`,盒子侧 `ftmon-inner`) | Windows Terminal 实时面板(2s 刷新,CPU/RAM/GPU/SVC/GW/LLM/GEN 含 decode 速率与 MTP 接受率);`python diagnostics\ftmon.py [--interval N|--once]`,密码 `FT116_PW` |
| `rexec.py` | 开发机 `diagnostics/` | SSH 执行/上传(`--put`) |

两者**只读**,走本地应答与 journal,绝不唤醒已停车模型。

### 5.2 日志与指标速查

| 看什么 | 命令 |
|---|---|
| 网关事件(唤醒/停车/门禁) | `journalctl -u gpu-gateway -f` |
| 模型请求计时(decode/prefill/MTP) | `journalctl -u bonsai -f \| grep print_timing` |
| 网关状态 JSON | `curl :8010/gpu-gateway/status` |
| 模型健康/槽位 | `curl :8012/health`、`curl :8012/slots`(上下文深度/处理中/实际采样参数) |
| Hermes 内部状态 | `curl :9119/api/status`(登录后;含 gateway_running/平台连接) |
| 看门狗 | `type C:\ProgramData\gpu-watchdog\watchdog.log` |

`print_timing` 解读:`tg =` decode 均值 + `tg_3s` 3 秒瞬时;`draft acceptance =`
MTP 接受率;`prompt eval time = ... tokens per second` prefill 速率。

### 5.3 已知盲区

- WSL 下 nvidia-smi 不显示计算进程名(只显示数量)——面板 GPROC 行故意的
- llama `/slots` 无 completed 计数(gateway 状态里 completed 显示 `-`)

---

## 附:端口与路径速查

| 端口 | 归属 |
|---|---|
| 8010 / 8188 | 网关对外(LLM / ComfyUI) |
| 8012 / 8191 | bonsai / ComfyUI 直连(诊断用) |
| 8011 | ftserve(备用,手动) |
| 9119 / 8187 | hermes dashboard 直连 / caddy 认证入口 |
| 8000,8189,8190 | 旧 portproxy(8000=已退役 ftproxy,8189/8190=comfy 相关) |

| 路径 | 内容 |
|---|---|
| `~/models/bonsai2/` | GGUF ×2 + mmproj |
| `~/llama.cpp-prism/build133/` | llama-server 二进制 |
| `~/.hermes/config.yaml` | Hermes 全部配置 |
| `~/logs/bonsai_*.log`、`gov2_tests.log` | 部署/测试日志 |
| `diagnostics/box_*.sh`、`ft_*.py` | 全部一次性诊断脚本(不入库) |
