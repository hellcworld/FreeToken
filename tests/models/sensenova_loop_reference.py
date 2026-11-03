"""VENDORED TEST REFERENCE -- SenseNova-Loop official remote-code modeling.

Verbatim copy of sensenova/SenseNova-Loop-9B-Preview@main modeling_sensenova_loop.py
(Apache-2.0, https://huggingface.co/sensenova/SenseNova-Loop-9B-Preview). It is the
numerical ground truth the FreeToken port is validated against; do not edit.
Loaded via importlib in test_sensenova_loop.py (never imported at collection time)."""
"""Hugging Face remote-code support for shared-weight SenseNova Loop and MHC.

This single file contains configuration, plan validation, MHC routing and the
model implementation. The original physical layer weights are reused on every
Loop visit; autoregressive cache states are independent per visit.
"""
import copy
import math
from contextvars import ContextVar

import torch
from torch import nn
from transformers import Qwen3_5Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf


_attention_capture = ContextVar("sensenova_attention_capture", default=None)


class SenseNovaLoopConfig(Qwen3_5Config):
    model_type = "sensenova_loop"


def execution_plan(config):
    """Return physical layer indices in the order they execute during inference."""
    loop = config.get("loop_settings")
    if not isinstance(loop, dict):
        raise ValueError("This implementation requires explicit loop_settings")

    text_config = config["text_config"]
    num_layers = text_config["num_hidden_layers"]
    start, end, count = loop.get("start"), loop.get("end"), loop.get("eval_count")
    if any(type(value) is not int for value in (num_layers, start, end, count)) or not (
        0 <= start < end <= num_layers and count >= 1
    ):
        raise ValueError("Invalid Loop interval or eval_count")
    if loop.get("counts") != [count]:
        raise ValueError("Only fixed-count Loop checkpoints are supported")

    layer_types = text_config["layer_types"]
    if len(layer_types) != num_layers or set(layer_types) - {"full_attention", "linear_attention"}:
        raise ValueError("Unexpected physical layer types")
    if "full_attention" not in layer_types:
        raise ValueError("At least one full-attention layer is required for sequence-length tracking")

    mhc = config.get("mhc")
    if mhc is not None:
        mhc_start, mhc_end, streams, iterations = (
            mhc.get(key) for key in ("start", "end", "streams", "iterations")
        )
        if (
            any(type(value) is not int for value in (mhc_start, mhc_end, streams, iterations))
            or not 0 <= mhc_start < mhc_end <= num_layers
            or streams not in (2, 4)
            or iterations < 1
        ):
            raise ValueError("Invalid MHC range, stream count, or iterations")
        if type(mhc.get("fixed_identity")) is not bool:
            raise ValueError("MHC fixed_identity must be a boolean")
        scale = mhc.get("pre_scale", 1.0)
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid MHC pre_scale")
        allowed = {"start", "end", "streams", "iterations", "fixed_identity", "pre_scale"}
        if set(mhc) - allowed:
            raise ValueError("Unknown MHC configuration fields")
        phase_boundaries = (0, start, end, num_layers)
        if not any(
            left <= mhc_start < mhc_end <= right
            for left, right in zip(phase_boundaries, phase_boundaries[1:])
        ):
            raise ValueError("MHC cannot cross a Loop phase boundary")

    return tuple(range(start)) + tuple(range(start, end)) * count + tuple(range(end, num_layers))


def sinkhorn(logits, iterations):
    """Normalize routing logits across rows and columns in log space."""
    logits = logits.float()
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


class MHCRouter(nn.Module):
    def __init__(self, dim, settings):
        super().__init__()
        self.streams = settings["streams"]
        self.iterations = settings["iterations"]
        self.fixed_identity = settings["fixed_identity"]
        self.pre_scale = settings.get("pre_scale", 1.0)
        streams = self.streams
        output_size = 2 * streams + (0 if self.fixed_identity else streams * streams)
        self.weight = nn.Parameter(torch.zeros(output_size, streams * dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.float32))
        with torch.no_grad():
            initial_probabilities = {2: (0.25, 0.75), 4: (0.0625, 0.1875, 0.25, 0.5)}[streams]
            self.bias[:streams].copy_(torch.logit(torch.tensor(initial_probabilities, device=self.bias.device)))
            if not self.fixed_identity:
                diagonal = math.log((1 - (streams - 1) / 16) * 16)
                self.bias[2 * streams:].copy_((torch.eye(streams, device=self.bias.device) * diagonal).flatten())

    def _apply(self, fn, recurse=True):
        # Keep the original FP32 values when the surrounding model is cast to BF16.
        def keep_fp32(tensor):
            result = fn(tensor)
            if tensor.is_floating_point() and result.dtype != torch.float32:
                if tensor.device.type == "meta" and result.device.type != "meta":
                    return torch.empty_like(result, dtype=torch.float32)
                return tensor.to(device=result.device, dtype=torch.float32)
            return result

        return super()._apply(keep_fp32, recurse=recurse)

    def forward(self, residual):
        if self.weight.dtype != torch.float32 or self.bias.dtype != torch.float32:
            raise RuntimeError("mHC routers require FP32 parameters; configure FSDP per-module precision")
        with torch.autocast(device_type=residual.device.type, enabled=False):
            flat = residual.flatten(-2).float()
            normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
            logits = torch.nn.functional.linear(normalized, self.weight, self.bias)
            streams = self.streams
            pre = logits[..., :streams].sigmoid()
            post = 2 * logits[..., streams:2 * streams].sigmoid()
            if self.fixed_identity:
                mix = torch.eye(streams, device=flat.device, dtype=torch.float32).expand(
                    *flat.shape[:-1], streams, streams
                )
            else:
                mix = sinkhorn(logits[..., 2 * streams:].unflatten(-1, (streams, streams)), self.iterations)
            branch = torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)
            if self.pre_scale != 1.0:
                branch = branch * self.pre_scale
        return branch, post, mix

    @staticmethod
    def merge(residual, branch, post, mix):
        """Mix residual streams and add the branch output to each stream."""
        with torch.autocast(device_type=residual.device.type, enabled=False):
            mixed_residual = torch.matmul(mix.to(residual.dtype), residual)
            branch_update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
            return mixed_residual + branch_update


def enter(hidden, physical, settings):
    """Expand to residual streams at the first mHC layer, or restore their shape."""
    if physical == settings["start"]:
        return hidden.unsqueeze(-2).expand(
            *hidden.shape[:-1], settings["streams"], hidden.shape[-1]
        ).contiguous()
    return hidden.unflatten(-1, (settings["streams"], -1))


def leave(residual, physical, settings):
    """Reduce streams at the last mHC layer; otherwise carry them forward."""
    return residual.mean(-2) if physical == settings["end"] - 1 else residual.flatten(-2)


class _CacheLayersView:
    """Expose the execution slot when a layer accesses its physical index."""
    def __init__(self, cache, physical, virtual):
        self.cache = cache
        self.physical = physical
        self.virtual = virtual

    def __getitem__(self, index):
        if index != self.physical:
            raise IndexError("A physical layer may only access its own virtual cache slot")
        return self.cache.layers[self.virtual]


class _VisitCache:
    """Map a physical attention module's layer_idx to one execution slot.

    Mapping is local to this forward call: shared modules and their layer_idx are
    never mutated. `layers` supports Transformers 5.12; state accessors support
    newer official backbone cache APIs as well.
    """
    def __init__(self, cache, physical, virtual):
        self.cache = cache
        self.virtual = virtual
        self.layers = _CacheLayersView(cache, physical, virtual)

    def update(self, key_states, value_states, layer_idx, **kwargs):
        return self.cache.update(key_states, value_states, self.virtual, **kwargs)

    def has_previous_state(self, layer_idx=None, **kwargs):
        return self.cache.has_previous_state(self.virtual, **kwargs)

    def update_conv_state(self, state, layer_idx, **kwargs):
        return self.cache.update_conv_state(state, self.virtual, **kwargs)

    def update_recurrent_state(self, state, layer_idx, **kwargs):
        return self.cache.update_recurrent_state(state, self.virtual, **kwargs)

    def get_conv_state(self, layer_idx, **kwargs):
        return self.cache.get_conv_state(self.virtual, **kwargs)

    def get_recurrent_state(self, layer_idx, **kwargs):
        return self.cache.get_recurrent_state(self.virtual, **kwargs)

    def is_recording_past(self, layer_idx):
        return self.cache.is_recording_past(self.virtual)


class SenseNovaLoopTextModel(hf.Qwen3_5TextModel):
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_hidden_states=None, output_attentions=None, **kwargs):
        """Run Loop visits, optionally returning their states and attention weights.

        Hidden states include the initial embedding and one state per visit.
        MHC states between its boundaries retain their flattened residual streams.
        Attention entries follow visit order, with None for linear-attention layers.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if output_hidden_states is None:
            output_hidden_states = getattr(self.config, "output_hidden_states", False)
        if output_attentions is None:
            output_attentions = getattr(self.config, "output_attentions", False)
        if not isinstance(output_hidden_states, bool):
            raise TypeError("output_hidden_states must be a boolean.")
        if output_attentions and self.config._attn_implementation != "eager":
            raise ValueError(
                "Attention weights require attn_implementation='eager' when loading the model."
            )
        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError("Disable use_cache when training with gradient checkpointing")
        if not use_cache and past_key_values is not None:
            raise ValueError("past_key_values requires use_cache=True")
        if past_key_values is not None and not isinstance(past_key_values, DynamicCache):
            raise TypeError("Loop generation currently supports DynamicCache only")
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self._loop_cache_config)
                past_key_values._loop_execution_plan = self._loop_plan
            elif getattr(past_key_values, "_loop_execution_plan", None) != self._loop_plan:
                # GenerationMixin may allocate an empty physical-depth cache.
                # A populated foreign cache cannot be reinterpreted safely.
                if past_key_values.get_seq_length() != 0 or past_key_values.has_previous_state():
                    raise ValueError("A populated cache from another execution plan cannot be reused")
                past_key_values = DynamicCache(config=self._loop_cache_config)
                past_key_values._loop_execution_plan = self._loop_plan

        if position_ids is None:
            seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(4, -1, -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_positions, position_ids = position_ids[0], position_ids[1:]
        else:
            text_positions = None

        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            mask_args = dict(
                config=self._loop_cache_config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_positions,
            )
            if hasattr(hf, "create_recurrent_attention_mask"):
                linear_mask = hf.create_recurrent_attention_mask(**mask_args)
            else:
                linear_mask = self._update_linear_attn_mask(attention_mask, past_key_values)
            masks = {
                "full_attention": hf.create_causal_mask(**mask_args),
                "linear_attention": linear_mask,
            }

        hidden = inputs_embeds
        position_embeddings = self.rotary_emb(hidden, position_ids)
        hidden_states = [hidden] if output_hidden_states else None
        attentions = [] if output_attentions else None
        hooks = []
        capture_token = None

        def record_attention(_module, _inputs, output):
            if _attention_capture.get() is attentions:
                if output[1] is None:
                    raise RuntimeError("The attention backend did not return attention weights.")
                attentions.append(output[1])

        # Repeated visits share weights, but each visit has its own cache slot.
        try:
            if attentions is not None:
                for layer in self.layers:
                    if hasattr(layer, "self_attn"):
                        hooks.append(layer.self_attn.register_forward_hook(record_attention))
                capture_token = _attention_capture.set(attentions)
            for execution_index, physical_index in enumerate(self._loop_plan):
                layer_type = self.config.layer_types[physical_index]
                if attentions is not None and layer_type == "linear_attention":
                    attentions.append(None)
                cache = (
                    _VisitCache(past_key_values, physical_index, execution_index)
                    if use_cache else None
                )
                hidden = self.layers[physical_index](
                    hidden, position_embeddings=position_embeddings,
                    attention_mask=masks[layer_type],
                    position_ids=text_positions, past_key_values=cache,
                    use_cache=use_cache, **kwargs,
                )
                if hidden_states is not None:
                    hidden_states.append(hidden)
        finally:
            if capture_token is not None:
                _attention_capture.reset(capture_token)
            for hook in hooks:
                hook.remove()

        hidden = self.norm(hidden)
        if hidden_states is not None:
            hidden_states[-1] = hidden
        return hf.Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden, past_key_values=past_key_values,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
            attentions=tuple(attentions) if attentions is not None else None,
        )


class MHCDecoderLayer(hf.Qwen3_5DecoderLayer):
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        residual = enter(hidden_states, self._mhc_physical, self._mhc_settings)
        branch, post, mix = self.mhc_attn(residual)
        branch = self.input_layernorm(branch)
        kind = self.layer_type if hasattr(self, "layer_type") else self.block_type
        if kind == "linear_attention":
            branch = self.linear_attn(hidden_states=branch, cache_params=past_key_values,
                                      attention_mask=attention_mask, **kwargs)
        else:
            branch, _ = self.self_attn(
                hidden_states=branch, position_embeddings=position_embeddings,
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, **kwargs)
        residual = MHCRouter.merge(residual, branch, post, mix)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(self.post_attention_layernorm(branch))
        return leave(MHCRouter.merge(residual, branch, post, mix),
                     self._mhc_physical, self._mhc_settings)


def _configure_text_model(text, config):
    raw = config.to_dict()
    plan = execution_plan(raw)
    # Change only these new model instances, preserving canonical state_dict
    # names and avoiding a second allocation of the full backbone.
    text.__class__ = SenseNovaLoopTextModel
    text._loop_plan = plan
    text._loop_cache_config = copy.deepcopy(text.config)
    text._loop_cache_config.num_hidden_layers = len(plan)
    text._loop_cache_config.layer_types = [text.config.layer_types[i] for i in plan]
    if raw.get("mhc") is not None:
        settings = raw["mhc"]
        for i in range(settings["start"], settings["end"]):
            layer = text.layers[i]
            layer.__class__ = MHCDecoderLayer
            layer._mhc_physical, layer._mhc_settings = i, dict(settings)
            layer.mhc_attn = MHCRouter(text.config.hidden_size, settings)
            layer.mhc_ffn = MHCRouter(text.config.hidden_size, settings)


class SenseNovaLoopModel(hf.Qwen3_5Model):
    config_class = SenseNovaLoopConfig
    _keep_in_fp32_modules = ["mhc_attn", "mhc_ffn"]
    _no_split_modules = ["Qwen3_5DecoderLayer", "MHCDecoderLayer", "Qwen3_5VisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    def __init__(self, config):
        execution_plan(config.to_dict())
        super().__init__(config)
        _configure_text_model(self.language_model, config)


class SenseNovaLoopForConditionalGeneration(hf.Qwen3_5ForConditionalGeneration):
    config_class = SenseNovaLoopConfig
    _keep_in_fp32_modules = ["mhc_attn", "mhc_ffn"]
    _no_split_modules = ["Qwen3_5DecoderLayer", "MHCDecoderLayer", "Qwen3_5VisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\."]

    def __init__(self, config):
        execution_plan(config.to_dict())
        super().__init__(config)
        _configure_text_model(self.model.language_model, config)
