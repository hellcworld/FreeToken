# FreeToken(个人 Fork,中文版)

[English](README.md) | **简体中文**

> **Fork 声明** · 本仓库是
> [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)(Apache-2.0)的个人
> fork,在上游 `main@2757bb5` 之上新增了 **Qwen3.8-27B NVFP4 支持(两种量化方言)**
> 与 **Qwen3.5/3.6/3.8 视觉输入**(对齐 gemma4 的支持级别)。
> 变更详情、验证状态与版本说明见 **[CHANGES.md](CHANGES.md)** ·
> fork 标签 `qwen38-vision-v1.0.0`。

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| [论文](https://arxiv.org/abs/2608.16157) | [上游仓库](https://github.com/FlashML-org/FreeToken) |
</p>


在你已有的硬件上解锁数据中心级智能——在游戏 PC 上本地运行 290B+ 前沿 MoE 模型,获得极速交互体验。

## 简介

FreeToken 是一个边缘原生的混合专家(MoE)推理引擎,面向在个人与消费级硬件上运行前沿规模的开源权重模型。它将异构边缘资源——GPU、CPU、主机内存与互连——视为统一的弹性推理平台。核心特性包括:

- **高速边缘原生运行时**:带宽自适应的 CPU–GPU 协同执行($q^\star$ 策略)、全层双缓冲 prefill 流式处理、全局 LRU 专家缓存、兼容 CUDA Graph 的执行,以及 FTW 快速权重格式。
- **语义感知缓存**:为循环状态与 KV 缓存提供语义锚点检查点,让 Agent 场景的上下文编辑(如工具调用、思考块)避免重复重算上下文。
- **弹性显存管理**:运行时在专家缓存与 KV 内存之间动态重分配显存,无需重启引擎或重载权重。
- **广泛的 MoE 与生态支持**:支持各参数规模与量化格式(MXFP4、NVFP4、FP8、BF16)的前沿开源 MoE 模型(如 DeepSeek-V4-Flash、Qwen3.6-35B-A3B、GLM-5.2),提供 Anthropic/OpenAI 兼容 API,可与真实编码与工具调用 Agent(Codex、Claude Code、OpenCode、OpenClaw、DeepSeek Harness 等)无缝集成。
- **多样的消费级硬件**:覆盖消费级笔记本、游戏台式机与工作站 GPU,原生支持 NVIDIA RTX 30/40/50 系列。

## 快速开始

> 上游官方 **Desktop 桌面版**(flashml.ai)是独立的官方发行渠道,不属于本 fork——请见[上游仓库](https://github.com/FlashML-org/FreeToken)。

### CLI

使用 [uv](https://docs.astral.sh/uv/)(推荐)或 pip 安装:

```bash
uv pip install "freetoken[accel]"
```

或从源码构建(本 fork 的功能需从源码安装):

```bash
git clone https://github.com/hellcworld/FreeToken.git && cd FreeToken
git checkout feat/qwen-vision
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

本 fork 新增能力的使用要点:

- **Qwen3.8-27B**:直接 `ft serve --model <checkpoint>` 即可,支持
  [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)(modelopt)与
  [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)(compressed-tensors)两种方言
- **视觉输入**:`FREETOKEN_LOAD_VISION=1` 开启(FTW 转换同样需要该变量);离线
  `LLM.generate(mm_inputs=[{"pixel_values": ..., "grid_thw": ...}])` 传入图像,
  在线 server 暂为纯文本(与 gemma4 现状一致)

更多细节:

- [安装指南](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)(上游,英文)
- [快速上手](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)(上游,英文)
- [支持的模型](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)(上游,英文;本 fork 已扩充 Qwen3.8 条目)
- [CLI 参考](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)(上游,英文)

## 引用

如果在研究中使用了 FreeToken,请引用上游[论文](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## 致谢

**本 fork 站在上游项目的肩膀上。**引擎的全部核心工作归功于
[FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken)
及其作者(Song Han、Ion Stoica、Matei Zaharia 等团队)——本 fork 只是在他们的
成果之上叠加了上述两个功能。以下为上游自己的致谢,原样保留:

FreeToken 深受 [mini-sglang](https://github.com/sgl-project/mini-sglang) 启发,并借鉴设计与复用了以下项目的代码:
[SGLang](https://github.com/sgl-project/sglang)、
[vLLM](https://github.com/vllm-project/vllm)、
[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)、
[LightLLM](https://github.com/ModelTC/lightllm) 与 [llama.cpp](https://github.com/ggml-org/llama.cpp)。

## 许可证

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE),
本 fork 的新增代码同样以 Apache-2.0 发布。
