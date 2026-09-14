"""Qwen3.5-family MTP (multi-token prediction) draft head.

DeepSeek-style single-block MTP that ships in every Qwen3.5/3.6/3.8 checkpoint under the
``mtp.*`` keys (a dedicated, always-unquantized shard): an eh_proj (``fc``) over the
concatenation ``[norm(embedding), norm(hidden)]``, one full-attention decoder layer, and a
final norm. The embedding and lm_head are shared with the target model. The draft head
keeps its OWN paged-KV slab (global layer id ``config.num_layers``) in the shared pool, so
prefix-cache reuse, page allocation and rejected-draft rollback all ride the existing
page-table machinery untouched.

The head runs in two modes:
* ``prefill`` — after each target prefill chunk, replay the chunk's tokens through the
  draft layer (attending to the target layer's shifted hidden states) to fill the draft
  KV slab, exactly like the target's own chunked prefill;
* ``draft`` — one token per step at decode time: ``[emb(t), h(prev)] -> fc -> layer ->
  norm -> shared lm_head -> argmax`` chains K draft tokens ahead of the verify pass.

Only draft QUALITY depends on this module; the target-model verify makes the output
lossless regardless of what it predicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaRMSNorm, OPList, RMSNorm
from freetoken.layers.moe import MoELayer
from freetoken.models.blocks import GatedMLP

from .attention import Qwen3_5Attention

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _DraftSharedExpert(BaseOP):
    """The draft head's shared SwiGLU expert with the checkpoint's SEPARATE gate/up
    projections (unlike the target layers' fused gate_up)."""

    def __init__(self, config: ModelConfig):
        from freetoken.layers import LinearReplicated

        self.gate_proj = LinearReplicated(
            config.hidden_size, config.shared_expert_intermediate_size, has_bias=False
        )
        self.up_proj = LinearReplicated(
            config.hidden_size, config.shared_expert_intermediate_size, has_bias=False
        )
        self.down_proj = LinearReplicated(
            config.shared_expert_intermediate_size, config.hidden_size, has_bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("consumed via Qwen3_5DraftMoE.forward pieces")



class Qwen3_5DraftMoE(BaseOP):
    """Resident bf16 routed MoE for the draft head of MoE checkpoints (the 35B-class
    MTP layer ships its experts as stacked bf16 ``mtp.layers.0.mlp.experts.
    {gate_up_proj,down_proj}`` tensors -- the exact layout of MoELayer's resident
    bf16 experts, so they load straight through the state dict).

    Routed compute goes through ``fused_topk`` + ``fused_experts_impl`` directly: the
    global offload backend's ``forward`` only serves OffloadMoELayer, and the draft
    layer is intentionally resident (~1.6 GiB at 35B -- charged to the weights budget
    by the engine's free-memory measurement like any other resident tensor). The
    shared expert keeps its checkpoint's SEPARATE gate/up projections (the head does
    not use the fused gate_up layout of the target layers)."""

    def __init__(self, config: ModelConfig):
        from freetoken.layers import LinearReplicated

        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,  # the target Qwen3_5MoE hardcodes True (HF semantics)
            activation=config.hidden_act,
            weight_format="bf16",
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _DraftSharedExpert(config)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from freetoken.moe.fused import fused_experts_impl, fused_topk

        num_tokens = hidden_states.shape[0]
        router_logits = self.gate.forward(hidden_states)
        # Shared expert BEFORE the routed experts: fused_experts_impl overwrites
        # ``hidden_states`` in place (Qwen3_5MoE.forward orders it the same way).
        shared = self.shared_expert.down_proj.forward(
            torch.nn.functional.silu(self.shared_expert.gate_proj.forward(hidden_states))
            * self.shared_expert.up_proj.forward(hidden_states)
        )
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.experts.top_k,
            renormalize=self.experts.renormalize,
        )
        routed = fused_experts_impl(
            hidden_states,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            topk_weights,
            topk_ids,
            activation=self.experts.activation,
        )
        return (routed + shared).view(num_tokens, hidden_states.shape[1])


class Qwen3_5MTPLayer(BaseOP):
    """The draft head's decoder layer: same pre-norm residual shape as the target layers,
    but built on plain bf16 linears (the ``mtp.*`` shard is un-quantized in every dialect).
    MoE checkpoints route the draft MLP through a RESIDENT bf16 MoELayer (the draft only
    ever computes bs x K rows, so the ~1.6 GiB of stacked bf16 experts at 35B is the
    right trade against streaming them). ``kv_only`` skips the MLP half during prefill
    replay -- the draft KV row is written inside the attention call, and the MLP output
    is unused there."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3_5Attention(config, layer_id, quant=False)
        if config.moe_enabled:
            self.mlp = Qwen3_5DraftMoE(config)
        else:
            self.mlp = GatedMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.self_attn.forward(hidden)
        if get_global_ctx()._mtp_kv_only:
            # Prefill replay: KV is stored; the MLP half would be dead compute.
            return hidden, residual
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5MTPHead(BaseOP):
    """``mtp.*`` state-dict owner. Key map (checkpoint -> module attribute path):
    ``mtp.fc.weight`` -> ``fc.weight``; ``mtp.pre_fc_norm_{embedding,hidden}.weight`` ->
    like-named RMSNorms; ``mtp.layers.0.*`` -> ``layers.0.*`` (one bf16 decoder layer);
    ``mtp.norm.weight`` -> ``norm.weight``. The layer norms are Gemma-style (1+w baked at
    load, matching the target layers); the head-level norms are standard RMSNorms (their
    checkpoint weights center near 1, not 0). MoE checkpoints ship the draft MLP's routed
    experts as pre-fused stacked tensors that slot directly into the resident MoELayer."""

    def __init__(self, config: ModelConfig, layer_id: int):
        H = config.hidden_size
        self.fc = _plain_linear(2 * H, H)
        self.pre_fc_norm_embedding = RMSNorm(H, config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(H, config.rms_norm_eps)
        self.layers = OPList([Qwen3_5MTPLayer(config, layer_id)])
        self.norm = RMSNorm(H, config.rms_norm_eps)

    def mount_embedding(self, embed_tokens) -> None:
        """Share the target's embedding table. Stored under an underscore name so the
        BaseOP state-dict walk never duplicates its keys under ``mtp.``."""
        self._embed = embed_tokens

    def _project_input(
        self, input_ids: torch.Tensor, prev_hidden: torch.Tensor
    ) -> torch.Tensor:
        """[emb(t), h(prev)] -> pre-norms -> eh_proj. ``prev_hidden`` rows pair 1:1 with
        ``input_ids`` (the hidden state of the position BEFORE each token; zeros for a
        sequence start)."""
        emb = self.pre_fc_norm_embedding.forward(self._embed.forward(input_ids))
        hid = self.pre_fc_norm_hidden.forward(prev_hidden)
        return self.fc.forward(torch.cat([emb, hid], dim=-1))

    def prefill(self, input_ids: torch.Tensor, prev_hidden: torch.Tensor) -> None:
        """Fill the draft KV slab over a prefill chunk (attention output unused)."""
        token = get_global_ctx()
        token._mtp_kv_only = True
        try:
            hidden, _residual = self.layers.op_list[0].forward(
                self._project_input(input_ids, prev_hidden), None
            )
        finally:
            token._mtp_kv_only = False
        del hidden

    def draft(self, input_ids: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """One draft step: returns the final-normed hidden state (consumed by the shared
        lm_head for the draft logits, and fed back as the next step's ``prev_hidden``)."""
        hidden, residual = self.layers.op_list[0].forward(
            self._project_input(input_ids, prev_hidden), None
        )
        # Standard RMSNorm (no fused-add op): the residual sum happens explicitly.
        return self.norm.forward(hidden + residual)


def _plain_linear(in_features: int, out_features: int) -> BaseOP:
    from freetoken.layers import LinearReplicated

    return LinearReplicated(in_features, out_features, has_bias=False)


__all__ = ["Qwen3_5MTPHead", "Qwen3_5MTPLayer"]
