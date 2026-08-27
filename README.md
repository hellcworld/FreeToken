**English** | [简体中文](README.zh-CN.md)

> **Fork notice** · This is a personal fork of
> [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0)
> adding **Qwen3.8-27B NVFP4 support (both quant dialects)** and
> **Qwen3.5/3.6/3.8 vision input** (gemma4-parity) on top of upstream `main@2757bb5`.
> Changes, verification status and versioning: **[CHANGES.md](CHANGES.md)** ·
> fork tag `qwen38-vision-v1.0.0`.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| [Paper](https://arxiv.org/abs/2608.16157) | [Upstream](https://github.com/FlashML-org/FreeToken) |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.


## ✨ What's New in This Fork

Two features layered on upstream `main@2757bb5` (details & verification: [CHANGES.md](CHANGES.md)):

**1 · Qwen3.8-27B NVFP4 support — both quant dialects**

```bash
ft serve --model RadixArk/Qwen3.8-27B-NVFP4      # modelopt dialect
ft serve --model unsloth/Qwen3.8-27B-NVFP4       # compressed-tensors mixed FP8+FP4
```
Fixes the #139 load-failure class (stripped `hf_quant_config.json` mis-detection) and
serves the unsloth mixed-precision layout natively (FP8 W8A16 attention + load-time RTN
re-quant of its FP8 MLP layers). Dense 27B NVFP4 keeps all weights GPU-resident: budget
~22 GB VRAM.

**2 · Vision input for Qwen3.5/3.6/3.8 (offline + online)**

```python
llm = LLM(model="...", ...)          # serve with FREETOKEN_LOAD_VISION=1
out = llm.generate(
    [prompt_ids_with_image_placeholders],
    SamplingParams(...),
    mm_inputs=[{"pixel_values": px, "grid_thw": thw}],   # from the HF processor
)
```
The tower/merger are bit-faithful vs the transformers reference; interleaved-mRoPE
positions are built automatically from `grid_thw`. Text-only builds are bit-identical
to upstream. FTW conversion must also run with `FREETOKEN_LOAD_VISION=1`.

The same checkpoints accept images through the **online API** — OpenAI
`image_url` content parts (data: or http(s) URLs) on `/v1/chat/completions` and
Anthropic `image` blocks on `/v1/messages` — when the server runs with
`FREETOKEN_LOAD_VISION=1` and the `freetoken[vision]` extra (pillow +
torchvision) is installed: the tokenizer worker preprocesses images with the
checkpoint's HF processor, and the scheduler encodes them with the resident
vision tower (up to 4 images per request).

**Verification experiments** (CPU loader-level + numerical alignment; plus an
end-to-end `ft serve` smoke test of `RadixArk/Qwen3.8-27B-NVFP4` on an NVIDIA
A10 24 GB GPU with CUDA 13):

| Experiment 1 -- real-checkpoint loading | Experiment 2 -- numerical alignment |
|---|---|
| ![exp1](assets/fork/exp1_loading.png) | ![exp2](assets/fork/exp2_alignment.png) |

| Experiment 3 -- CPU test suites (15 passed) |
|---|
| ![exp3](assets/fork/exp3_tests.png) |

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

> The upstream **Desktop app** (flashml.ai) is a separate official distribution and not part of this fork -- see the [upstream repo](https://github.com/FlashML-org/FreeToken).

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)

## Citation

If you use FreeToken in your research, please cite the **upstream** [paper](https://arxiv.org/abs/2608.16157) -- the engine and its ideas are entirely the upstream team's work:

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

**This fork stands on the shoulders of the upstream project.** All engine credit
belongs to [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) and
its authors (Song Han, Ion Stoica, Matei Zaharia et al.) -- this fork only layers
the two features above onto their work. Upstream's own acknowledgments are
preserved below:

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
