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
  -ngl 999 -fa on -c 98304 -ctk q4_0 -ctv q4_0 \   # 全层进显存;96K 上下文;KV q4 量化(~1.1GB;2026-10-27 起,显存 11.8/12.3GB)
  --jinja --spec-type draft-mtp --spec-draft-n-max 2 \   # MTP 投机(2 草稿)
  --chat-template-file /home/adminuser/models/bonsai2/chat_template_froggeric_v22.5.jinja \  # froggeric 固化模板(见 1.5)
  --image-min-tokens 1024 --host 127.0.0.1 --port 8012 \
  -np 1 --cache-ram 8192 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \    # 采样兜底=模型卡思考档(GGUF 元数据缺采样参数,必须显式设)
  --reasoning-budget 1600 \  # 思考 token 硬上限(~50s;2026-10-27 从 800 上调);防失控思考,见 1.5
  --alias bonsai2-27b-abliterated
Environment=LD_LIBRARY_PATH=.../build133/bin       # 动态链接库
```

**不要开机自启**:由 gpu-gateway 按需唤醒(见二);同时 ftserve 与 bonsai 各占
~11GB 显存,**同跑必 OOM**——手动操作任何一个前先停另一个。

### 1.3 实测性能(RTX 3060 12GB,64K ctx)

| 指标 | 数值 |
|---|---|
| 冷加载 | 10-20s;显存 11.8 / 12.3GB(96K q4 KV,2026-10-27 起;此前 64K q8 为 11.5-11.8) |
| decode | 34-49 tok/s(短上下文);~39K ctx 降至 26-28(带宽瓶颈,预期行为);64K 满约 22-24 |
| prefill | ~500 tok/s 且平坦(4K/16K 实测一致)——对比旧 ftserve 8K 即掉 116 |
| MTP | 接受率 0.4-0.7 @temp1.0(贪心可到 0.92);关掉 MTP 会慢 ~40%,别关 |
| **decode 天花板** | **已达硬件上限**:GPU 满频 1792MHz/95-96% util/160W 无降频;权重 7.66GB÷360GB/s≈47 t/s 纯解码上限,MTP 后实测 36-42 即 ~85-90% 理论值 |
| **草稿深度 n-max** | **2 为最优**(2026-10-13 全矩阵对照:n=1 平均 ~37.7 t/s(接受率 0.52-0.59/步长 1.5-1.6),n=2 ~39.2(0.49-0.65/2.0-2.3),n=3 ~26.5,n=4 ~31.5;≥3 接受率崩到 0.21-0.35 纯浪费)——别调 |
| **CPU 协同(-ngl 拆层)** | **有害,勿用**(2026-10-13 实测:-ngl 999 全 GPU ~33.9 t/s;63=2 层(~3% 权重)给 CPU → ~22.7(-33%);55=10 层(~15%)→ ~9.9(-71%,慢 3.4 倍)。R5 3600 双通道 DDR4 ~51GB/s 仅为显存 1/7,层间串行+PCIe 同步,拆多少亏多少。CPU 的正确角色:网关/系统/--cache-ram,不算模型) |
| **SM 频率上限 1500MHz(现役,2026-10-13 起)** | `nvidia-smi -lgc 210,1500`(下限 210 保空闲降频)。实测:decode 完全无损(5083 tok 长跑 37.7 t/s),负载功耗 172→~85W、温度 78→67°C;**代价:prefill -12%**(433→381 t/s,36K 全新提示 83→94s,有前缀缓存时无感)。持久化三重保障:开机计划任务 `GPU-Clock-Cap`(SYSTEM/onstart)+ 看门狗 v2 复位后自动重申(驱动级复位会清锁)+ 手动。调整:`schtasks /change`+`nvidia-smi -lgc 210,XXXX`;撤销:`schtasks /delete /tn GPU-Clock-Cap /f && nvidia-smi -rgc`。另:显存 7501 档为纸面值(实测恒 7301),功耗墙 180W 无收益,均已还原 |
| **KV q4_0** | 速度维度无收益(@36K 实测 26.9 vs 26.6 t/s,在 MTP 波动内);但**容量维度采用**(2026-10-27):96K 上下文靠 q4 才放得下(KV 1.1GB vs q8 的 2.25GB),与 hermes context_length=98304 联动(压缩触发 ~73.7K)。注意:hermes agent **无法**自行改 /etc/systemd 的 unit(工具策略拒绝敏感路径,需 sudo 批准)——运维改动走人工 |
| 能力 | 视觉(读图准确)、工具调用(JSON 规范)、中文长文连贯 |

### 1.4 文件与常用操作

- 模型目录 `/home/adminuser/models/bonsai2/`:abliterated-v2(现役)、原版
  `Bonsai-2-27B-PQ2_0-MTP.gguf`(备用,带审查版)、mmproj-Q8_0
- 手动启停:`systemctl stop ftserve && systemctl start bonsai`(反之亦然)
- **换回原版 GGUF**:unit 里 `-m` 路径改回原版文件 → `daemon-reload && restart bonsai`
- **换回 ftserve 引擎**:见 2.4

### 1.5 思考预算(`--reasoning-budget 800`,2026-10-13 实测定案)

背景:temp 1.0 下思维链长度方差极大(同一条文案请求 500~5000+ token 都出现过),
微信回复实测 105~558s。对照实验(同一提示,5 方案):

| 方案 | 耗时 | 结论 |
|---|---|---|
| 基线 | 23s~9min(方差) | 思考量不可控 |
| 请求级 `reasoning_effort=low` | 153s,**6000 token 打满、0 字回复** | **禁止使用**——该档位触发失控思考(官方 KNOWN_ISSUES 实锤) |
| 请求级 `reasoning_effort=none` | 6s | 思考全关,质量仍可 |
| 请求级 `enable_thinking=false` | 8s | 同上(Qwen 官方模板开关) |
| 提示词 `/no_think` 标签 | 26s | **无效**,模型照思考 |
| **服务端 `--reasoning-budget 1600`** | 12~60s | **采用**(2026-10-27 由 800 上调,思考上限 ≈50s):预算内自然收束,超限强制截断+中文重定向 |

强制性验证:budget=50 时思考被斩停在 83 字符(正中预算),正文照常输出 1261 字/25s。
预算按 token 计(英文思考 ~4 字符/token,故 3100 字符≈800 token 属正常)。

**2026-10-13 补丁(思考溢出事故)**:难题触顶预算后,默认预算消息拉不住该模型——
它把推理用英文**继续写进正文**(微信实测一条 13327 字的"回复"全是思考文本)。
修复:unit 增加 `--reasoning-budget-message "思考时间到。立即停止推理,直接用中文
输出最终答案,不要再写任何推理过程。"`——五房子难题触顶后正文直接以"**结论:…**"
开头输出规范答案,重定向验证通过。若换模型需重验此消息有效性。
**2026-10-15 模板更换**:unit 增加 `--chat-template-file .../chat_template_froggeric_v22.5.jinja`
(froggeric/Qwen-Fixed-Chat-Templates v22.5,单文件覆盖 Qwen3.5/3.6/3.8,vision 内建)。
相对原生模板:默认思考档 **xhigh→medium**(回复更利落)、修复工具调用/思考定界/流式
问题、思考档可用 low/medium/xhigh/none + 粘性标签切换。冒烟:文案 36s 干净;难题
89s 思考有界无泄漏;**视觉 15.8s 准确读图**(image placeholder 兼容)。回滚:unit 去掉
该旗标+restart(备份 `bonsai.service.bak-template`)。
注意:hermes 对自定义 OpenAI-compat 后端**不下发** `reasoning_effort`(仅 Kimi/TokenHub/
LM Studio 三类),所以只能服务端设,hermes 侧配置是空操作。回滚:unit 去掉这两个旗标,
备份 `/etc/systemd/system/bonsai.service.bak-reasoning`。

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
| hermes-gateway-refresh.timer | linger 用户定时器,**每日 04:05** | 预防性重启微信网关(见 3.2) |

### 3.2 微信连接静默假死(已复发 2 次,运维必读)

病症:hermes-gateway 进程 active、`/api/status` 微信平台仍显示 connected(陈旧
状态)、journal/errors.log **零报错**,但 inbound 完全停止——用户消息根本不达。
发作史:10-12 08:08 起 >5h、10-13 08:25 起 ~40min,两次均靠重启恢复。

- **诊断法**:看 `~/.hermes/logs/gateway.log` 尾部——最后一条 `[Weixin] inbound`
  的时间距现在多久;只有 housekeeping/mem_trim 在刷=假死实锤(`/api/status` 的
  weixin.state 不可信)
- **处置**:`runuser -u adminuser -- env XDG_RUNTIME_DIR=/run/user/1000
  systemctl --user restart hermes-gateway`(约 20s 重连)
- **预防**:每日 04:05 定时重启(已装);若复发周期 <1 天再考虑基于 gateway.log
  inbound 心跳的探测式重启

### 3.3 关键配置(`~/.hermes/config.yaml`,改后重启对应服务)

| 键 | 现值 | 为什么 |
|---|---|---|
| model.default | bonsai2-27b-abliterated | 显示对齐(llama-server 不校验 model 字段) |
| model.base_url | http://127.0.0.1:8010/v1 | 走网关 |
| **model.context_length** | **65536** | **必设**:llama `/v1/models` 上报 GGUF 元数据 262K 而非 `-c` 实际值;不设则压缩 196K 才触发,必撞 64K 硬顶。设后压缩在 48K 输入触发(0.75 floor),保头 3 条+尾 20 条 |
| dashboard.public_url | https://hermes.infrafixed.com | **必设为公网域名**:WS 一次性票据校验绑定此值,错配=聊天无限重连(已修) |
| dashboard.basic_auth | admin / *(scrypt)* | 网页登录层 |

### 3.3.1 Caddyfile 8187 块丢失事故(2026-10-28 修复)

10-13 14:09 有人改过 /etc/caddy/Caddyfile(非本文档流程):删掉了 :8187(hermes
basic_auth→9119)块,并把 :8189 上游从网关 8188 改为直连 ComfyUI 8191。后续还有
两处协调改动(portproxy 8188→8191、10-25 webui.service --comfy 8191),致全部生图
路径绕过生图门禁(bonsai 醒时生图必 OOM)。**2026-10-28 三处全部接回**:
portproxy 8188→8188、Caddyfile :8189 上游→8188、webui.service --comfy→8188;
功能实测:bonsai 醒着时提交生图,网关日志 "parking bonsai for comfy job" →
停车 → 放行 → 198s 出图 ✓。备份 *.bak-1028(b)。
caddy 某次重启加载新配置后 8187 消失,局域网/公网 hermes 入口断。2026-10-28:
恢复 :8187 块;发现旧 bcrypt 哈希与文档密码(hermes-box-9119)不匹配(疑似历史上
轮换过),**已重置为文档值** hermes/hermes-box-9119。备份 Caddyfile.bak-1028。

### 3.4 公网链路

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

### 4.2 开机自愈矩阵(WSL 2.7.14 时代,2026-10-15 重写)

| 组件 | 启动方式 |
|---|---|
| **WSL 本体** | 三重保险:计划任务 `WSL-Stack-Boot`(Windows 开机)+ `WSL-Stack-Keepalive`(每 10 分钟,执行 `gpu-watchdog/keepalive.ps1`:拉起 WSL+IP 变化重指 6 条 portproxy)+ `WSL-Stack-Hold`(常驻循环,持续持有一个 wsl 客户端会话,`holdtask2.ps1`) |
| gpu-gateway / comfyui / hermes-dashboard / caddy | 系统服务自启 |
| hermes-gateway(微信) | linger 用户服务,随 boot 拉起;错过的 04:05 刷新由 Persistent 定时器补跑 |
| bonsai | **不启动**,首个 LLM 请求唤醒(~12s) |
| ftserve | **不会**启动(ftproxy 已 disable 且 Wants=ftserve 已拆) |

**WSL 版本纪律(血泪教训)**:盒子必须停留在 **WSL 2.7.14 MSI**(已装,
安装包留存 `C:\Users\admin\wsl_2.7.14.msi`)。**WSL 3.0.1.0(10-08 Windows Update
自动升的商店版)是 10-08 以来全部怪病的总根**:①GPU 楔死病(10-09/10-10 共 7 次);
②**"无客户端会话 ~15s 即拆除整个用户空间"(设计行为,systemd 服务不算数)——
10-13 20:04 起 1.5 天黑栈与 10-15 早上的 30 秒级重启循环都是它**;③WSL vSwitch
抽风(Windows→WSL IP 路由反复断)。2.7.14 实测:9 分钟零客户端会话全栈存活。
`wsl --version` 应报 2.7.14.0;若 Windows Update 日后再度推送 WSL 升级,
第一怀疑对象就是它。另:.wslconfig 含 `vmIdleTimeout=-1`(VM 空闲关机禁用,双保险)。

**已部署的防升级三件套(2026-10-15)**:①注册表策略
`HKLM\SOFTWARE\Policies\Microsoft\WindowsStore AutoDownload=2`(禁用商店应用
自动更新——10-08 推 3.0.1.0 的主通道);②新版商店包 Microsoft.WSL 已被 MSI 安装
清除,仅剩 MSI 注册的 2.7.14;③keepalive.ps1 内置**版本哨兵**:每次运行检查
`wsl --version`,偏离 2.7.14 即在 keepalive.log 写 !!! 告警(含回退命令)。
手动升级/回退:`msiexec /i C:\Users\admin\wsl_2.7.14.msi /qn`(该 MSI 是 x64 2.7.14)。

### 4.3 战绩与边界

- 10-09:驱动楔死大半天无自愈(看门狗前身)→ 10-10:同病 4 次全部 2-5 分钟自愈
- 10-13~10-15:WSL 3.0.1.0 综合征全面爆发(黑栈 1.5 天、无会话拆用户空间、
  vSwitch 断路由)→ **10-15 回退 WSL 2.7.14 后清零**;回退前的 keepalive/hold
  三件套保留作保险
- 边界:驱动级救不回时看门狗放弃并提示手动重启;重置会断当时活跃会话(设计取舍)

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
