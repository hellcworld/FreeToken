from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight).

    Loop models (SenseNova-Loop): layers inside the mHC range instead carry ``streams``
    parallel residual streams routed by the fp32 ``mhc_attn``/``mhc_ffn`` routers --
    :meth:`forward_mhc`. The standard :meth:`forward` is untouched."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # SenseNova-Loop mHC: this physical layer routes its residual through parallel
        # streams. Routers are fp32 (the mount keeps the module's declared dtype).
        ls = config.loop_settings
        self._mhc = ls if (ls is not None and ls.owns_mhc_layer(layer_id)) else None
        if self._mhc is not None:
            from .mhc import MHCRouter

            self.mhc_attn = MHCRouter(config.hidden_size, ls)
            self.mhc_ffn = MHCRouter(config.hidden_size, ls)

    def set_visit_layer_id(self, virtual: int) -> None:
        """Point the mixer's cache addressing at this VISIT's slot (loop layer re-entry).
        The physical layer object is shared across visits; the slot id is read at each
        kernel launch, so the mutation is safe under single-threaded forwards and bakes
        per-visit into CUDA graph captures."""
        if self._is_linear:
            self.linear_attn.layer_id = virtual
        else:
            self.self_attn.layer_id = virtual

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward_mhc(self, streams: torch.Tensor) -> torch.Tensor:
        """mHC form (HF MHCDecoderLayer): routers decide the pre/post stream weights and
        the Sinkhorn mixing matrix around each sublayer; the branch each sublayer sees is
        the router's pre-weighted stream sum, normed by the layer's own Gemma norms."""
        from .mhc import mhc_merge

        branch, post, mix = self.mhc_attn.route(streams)
        branch = self.input_layernorm.forward(branch)
        branch = self.linear_attn.forward(branch) if self._is_linear else self.self_attn.forward(branch)
        streams = mhc_merge(streams, branch, post, mix)
        branch, post, mix = self.mhc_ffn.route(streams)
        branch = self.mlp.forward(self.post_attention_layernorm.forward(branch))
        return mhc_merge(streams, branch, post, mix)


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._image_token_id = config.image_token_id
        # Loop execution plan ((virtual, physical) pairs); the identity plan for every
        # non-loop model. mhc settings cached for the boundary expand/reduce below.
        self._plan = config.loop_visit_pairs()
        self._mhc = config.loop_settings

    def _merge_multimodal(self, input_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Replace the ``image_token_id`` placeholder embeddings with the vision tower
        output gathered in ``batch.mm_embeds`` (request order == placeholder order)."""
        mm_embeds = get_global_ctx().batch.mm_embeds
        if mm_embeds is None:
            return x
        mask = input_ids == self._image_token_id
        n_slots = int(mask.sum())
        assert n_slots == mm_embeds.shape[0], (
            f"image placeholders ({n_slots}) != image embeddings ({mm_embeds.shape[0]}); "
            "image tokens must not be split across prefill chunks"
        )
        return x.masked_scatter(mask.unsqueeze(-1), mm_embeds.to(x.dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        x = self._merge_multimodal(input_ids, x)
        residual: torch.Tensor | None = None
        # Plan-driven walk (identity order for non-loop models): a looped physical layer
        # re-enters with its visit's virtual cache-slot id; the mHC range switches the
        # residual to [N, streams, D] between its boundaries (re-expanded and re-reduced
        # on EVERY visit -- the HF plan keeps mHC inside one loop phase).
        for virtual, physical in self._plan:
            layer = self.layers.op_list[physical]
            mhc = layer._mhc
            if mhc is not None:
                if physical == mhc.mhc_start:
                    # standard layer leaves the residual add deferred; the router reads
                    # the summed stream, so materialize it here (one add kernel)
                    h = x if residual is None else x + residual
                    streams = (
                        h.unsqueeze(-2)
                        .expand(*h.shape[:-1], mhc.streams, h.shape[-1])
                        .contiguous()
                    )
                    residual = None
                else:
                    streams = x
                layer.set_visit_layer_id(virtual)
                streams = layer.forward_mhc(streams)
                if physical == mhc.mhc_end - 1:
                    x = streams.mean(dim=-2)
                    residual = None
                else:
                    x = streams
            else:
                layer.set_visit_layer_id(virtual)
                x, residual = layer.forward(x, residual)
        if residual is None:
            x = self.norm.forward(x)
        else:
            x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        if config.is_multimodal:
            # FREETOKEN_LOAD_VISION=1 build: the Qwen-VL tower + patch merger (~0.9 GiB
            # bf16, never quantized in any dialect), mounted on the shared vision
            # prefixes so the FTW replay skip and weight loader line up (gemma4 pattern).
            from .vision import Qwen3_5PatchMerger, Qwen3_5VisionTower

            self.vision_tower = Qwen3_5VisionTower(config.vision_config)
            self.embed_vision = Qwen3_5PatchMerger(config.vision_config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        if getattr(config, "enable_mtp", False) and config.num_mtp_layers > 0:
            # MTP speculative decoding (--speculative-algorithm MTP): build the draft head
            # only when the engine enabled it (Engine._adjust_config flips enable_mtp), so
            # plain serving never pays its VRAM. Its attention layers own KV slabs
            # [num_layers, num_layers + num_mtp_layers) in the shared pool.
            from .mtp import Qwen3_5MTPHead

            assert config.num_mtp_layers == 1, "only the single-layer MTP head is supported"
            self.mtp = Qwen3_5MTPHead(config, layer_id=config.num_layers)
            self.mtp.mount_embedding(self.model.embed_tokens)
        # Last forward's pre-lm_head hidden states [T, H]; the MTP speculator reads the
        # per-request tail rows from it (draft input pairing / decode-time stash).
        self._last_hidden: torch.Tensor | None = None
        super().__init__()

    def forward(self) -> torch.Tensor:
        self._last_hidden = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(self._last_hidden)

    @torch.inference_mode()
    def encode_images(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Qwen-VL inputs: ``pixel_values`` [total_patches, C*T*P*P] + ``grid_thw``
        [n, 3] from the HF processor. Returns [num_image_tokens, hidden] in image
        order (tower patches are concatenated per image; the merger's 2x2 merge keeps
        that order). ``image_position_ids`` is the gemma4-style argument and unused."""
        assert grid_thw is not None, "qwen3_5 encode_images needs processor grid_thw"
        # HF processors hand back fp32 CPU pixels; the tower weights are
        # model-dtype on device
        weight = self.vision_tower.patch_embed.proj.weight
        pixel_values = pixel_values.to(device=weight.device, dtype=weight.dtype)
        grid_thw = grid_thw.to(device=weight.device)
        features = self.vision_tower.forward(pixel_values, grid_thw)
        return self.embed_vision.forward(features)


__all__ = ["Qwen3_5MoEForCausalLM"]
