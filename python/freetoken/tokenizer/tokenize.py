from __future__ import annotations

import importlib.util
import json
import os
import threading
from types import ModuleType
from typing import Any, List

import torch
from freetoken.message import TokenizeMsg
from freetoken.utils import init_logger
from transformers import PreTrainedTokenizerBase

from .effort import (
    EffortProfile,
    ThinkingProfile,
    probe_effort_profile,
    probe_thinking_profile,
    quantize_effort,
)
from .multimodal import (
    MAX_IMAGES_PER_REQUEST,
    expand_image_pad_ids,
    extract_image_parts,
    is_image_part,
    load_image_from_url,
)

logger = init_logger(__name__)


def resolve_thinking_mode(chat_template_kwargs: dict[str, Any] | None, tools: Any | None) -> str:
    """Resolve the thinking mode (``"thinking"`` or ``"chat"``) for a chat request.

    The single source of truth for this decision: the encode side
    (``_apply_dsv4_chat_encoder`` below) uses it to pick the prompt the model
    sees, and the frontend parse side (``server/openai_api.py``) imports it to
    decide whether the model's output begins inside a reasoning block. Keeping
    one implementation prevents the two sides from disagreeing. Thinking is on
    when tools are offered (dsv4 only emits well-formed tool calls in thinking
    mode) or when the caller requests it via ``chat_template_kwargs``.
    """
    ctk = chat_template_kwargs or {}
    mode = str(ctk.get("thinking_mode") or "chat")
    if tools or ctk.get("enable_thinking") or ctk.get("thinking"):
        mode = "thinking"
    if mode not in ("chat", "thinking"):
        mode = "chat"
    return mode


_EFFORT_PROBE_MESSAGES = [{"role": "user", "content": "ping"}]


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self._dsv4_encoder = _load_dsv4_encoder_if_needed(tokenizer)
        self._effort_profile: EffortProfile | None = None
        self._thinking_profile: ThinkingProfile | None = None
        self._effort_lock = threading.Lock()
        self._logged_effort_maps: set[tuple[Any, str | None]] = set()
        self._processor: Any | None = None
        self._processor_lock = threading.Lock()

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            input_ids, _, _ = self.tokenize_with_mm(msg)
            results.append(input_ids)
        return results

    def tokenize_with_mm(
        self, msg: TokenizeMsg
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Tokenize one message; for image-bearing prompts additionally return
        the processor's CPU ``pixel_values`` and ``grid_thw`` for the scheduler
        to encode with the resident vision tower (online multimodal path)."""
        if not (isinstance(msg.text, list) and self._has_image_parts(msg.text)):
            return self._tokenize_text(msg), None, None
        if self._dsv4_encoder is not None:
            raise ValueError("image inputs are not supported for this checkpoint")
        if len(msg.text) > 0:
            normalized, urls = extract_image_parts(msg.text)
        else:
            normalized, urls = msg.text, []
        if len(urls) > MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"a request may carry at most {MAX_IMAGES_PER_REQUEST} images"
            )
        processor = self._get_processor()
        images = [load_image_from_url(u) for u in urls]
        prep = processor(
            text=["placeholder"] * len(images), images=images, return_tensors="pt"
        )
        grid = prep.get("image_grid_thw", prep.get("grid_thw"))
        if grid is None:
            raise ValueError("the image processor returned no image grid")
        prompt = self._render(
            normalized, msg.tools, self._sanitize_effort(msg.chat_template_kwargs or {})
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        expanded = expand_image_pad_ids(
            ids, grid.tolist(), self._image_token_id()
        )
        input_ids = torch.tensor(expanded, dtype=torch.int32)
        return input_ids, prep["pixel_values"], grid

    def _tokenize_text(self, msg: TokenizeMsg) -> torch.Tensor:
        prompt = self.render_prompt(msg)
        # A jinja chat template owns every special token (HF's apply_chat_template
        # tokenizes with add_special_tokens=False for the same reason): tokenizers
        # that auto-add bos (muse-glimmer's, llama's) would otherwise double it --
        # the template already rendered one. Raw-string prompts and the dsv4
        # encoder path keep the default.
        templated = isinstance(msg.text, list) and self._dsv4_encoder is None
        input_ids: torch.Tensor = (  # type: ignore
            self.tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=not templated
            )
        )
        return input_ids.view(-1).to(torch.int32)

    @staticmethod
    def _has_image_parts(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(is_image_part(p) for p in content):
                return True
        return False

    def _get_processor(self) -> Any:
        """The HF processor of the served checkpoint, loaded lazily so text-only
        deployments never pay for it. Gated on FREETOKEN_LOAD_VISION matching
        the engine-side vision tower switch."""
        from freetoken.models.config import vision_load_enabled

        with self._processor_lock:
            if self._processor is None:
                if not vision_load_enabled():
                    raise ValueError(
                        "image inputs require the server to run with "
                        "FREETOKEN_LOAD_VISION=1"
                    )
                from transformers import AutoProcessor

                model_path = (
                    getattr(self.tokenizer, "name_or_path", None)
                    or getattr(self.tokenizer, "_name_or_path", "")
                    or ""
                )
                if not str(model_path):
                    raise ValueError("cannot locate the checkpoint for its image processor")
                self._processor = AutoProcessor.from_pretrained(str(model_path))
            return self._processor

    def _image_token_id(self) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if token_id is None or token_id < 0:
            raise ValueError(
                "the tokenizer has no <|image_pad|> token; this checkpoint "
                "does not support image inputs"
            )
        return int(token_id)

    def render_prompt(self, msg: TokenizeMsg) -> str:
        """The template/encoder half of ``tokenize``, exposed so the frontend can
        validate a request before committing an SSE stream. Sanitizes
        ``reasoning_effort`` first: every render path (worker, frontend
        validation, count_tokens) must quantize identically."""
        if not isinstance(msg.text, list):
            return msg.text
        return self._render(
            msg.text, msg.tools, self._sanitize_effort(msg.chat_template_kwargs or {})
        )

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: dict[str, Any],
    ) -> str:
        """Raw render, no effort sanitation — the probe needs unsupported values
        to actually reach the template so rejection is observable."""
        if self._dsv4_encoder is not None:
            return _apply_dsv4_chat_encoder(
                self._dsv4_encoder, messages, tools, chat_template_kwargs
            )
        # Broadcast the effort in every spelling the ecosystem's templates read
        # (muse-glimmer grades ``reasoning_strength``; Jinja ignores undeclared
        # variables) -- the same rule the thinking toggles use. An explicit
        # caller-provided spelling wins over the broadcast.
        if "reasoning_effort" in chat_template_kwargs:
            chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs.setdefault(
                "reasoning_strength", chat_template_kwargs["reasoning_effort"]
            )
        if tools is not None:
            chat_template_kwargs = {**chat_template_kwargs, "tools": tools}
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        assert isinstance(prompt, str)
        return prompt

    def effort_profile(self) -> EffortProfile:
        """The checkpoint's effort vocabulary, probed on first use and cached
        for the process lifetime."""
        with self._effort_lock:
            if self._effort_profile is None:
                self._effort_profile = probe_effort_profile(self._probe_render)
                logger.info(
                    "reasoning-effort profile: supported=%s default=%s",
                    sorted(self._effort_profile.supported) or "(none)",
                    self._effort_profile.default,
                )
            return self._effort_profile

    def thinking_profile(self) -> ThinkingProfile:
        """The checkpoint's thinking controls (toggle behavior + effort
        vocabulary), probed on first use and cached for the process lifetime.
        Feeds the /v1/cache/status gear derivation."""
        efforts = self.effort_profile()
        with self._effort_lock:
            if self._thinking_profile is None:
                self._thinking_profile = probe_thinking_profile(self._probe_render, efforts)
            return self._thinking_profile

    def _probe_render(
        self, kwargs: dict[str, Any], tools: list[dict[str, Any]] | None
    ) -> str:
        return self._render(_EFFORT_PROBE_MESSAGES, tools, kwargs)

    def _sanitize_effort(self, chat_template_kwargs: dict[str, Any]) -> dict[str, Any]:
        if "reasoning_effort" not in chat_template_kwargs:
            return chat_template_kwargs
        raw = chat_template_kwargs.get("reasoning_effort")
        mapped = quantize_effort(raw, self.effort_profile())
        if mapped == raw:
            return chat_template_kwargs
        # raw is client-controlled and may be unhashable (a JSON list/dict).
        key = (raw if isinstance(raw, str) else repr(raw), mapped)
        if key not in self._logged_effort_maps:
            self._logged_effort_maps.add(key)
            logger.info(
                "reasoning_effort %r is not supported by this checkpoint; using %s",
                raw,
                mapped if mapped is not None else "the template default",
            )
        sanitized = dict(chat_template_kwargs)
        if mapped is None:
            del sanitized["reasoning_effort"]
        else:
            sanitized["reasoning_effort"] = mapped
        return sanitized


def _load_dsv4_encoder_if_needed(tokenizer: PreTrainedTokenizerBase) -> ModuleType | None:
    if getattr(tokenizer, "chat_template", None):
        return None
    model_path = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "_name_or_path", "")
    if not model_path:
        return None
    encoder_path = os.path.join(str(model_path), "encoding", "encoding_dsv4.py")
    if not os.path.isfile(encoder_path):
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        return None
    return module


def _apply_dsv4_chat_encoder(
    encoder: ModuleType,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict,
) -> str:
    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("tool_calls"):
            message["tool_calls"] = _dsv4_tool_calls(message["tool_calls"])
    if tools:
        _attach_tools_to_dsv4_messages(rendered_messages, tools)

    # No effort filtering here: the caller sanitized already, and the probe
    # needs raw values to reach the encoder's own validation.
    return encoder.encode_messages(
        rendered_messages,
        thinking_mode=resolve_thinking_mode(chat_template_kwargs, tools),
        reasoning_effort=chat_template_kwargs.get("reasoning_effort"),
    )


def _dsv4_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """The dsv4 encoder's contract is ``function.arguments`` = JSON-object STRING
    (it json.loads then iterates .items()); a dict (what ``render_messages``
    produces for Jinja templates) trips its bare-except fallback, which wraps the
    whole payload in a bogus parameter literally named ``arguments``. Re-serialize
    here. Copies each tool-call dict: the outer message copy is shallow, so these
    are shared with the caller."""
    rendered = []
    for tc in tool_calls:
        tc = dict(tc)
        fn = dict(tc.get("function") or {})
        fn["arguments"] = _dsv4_arguments_str(fn.get("arguments"))
        tc["function"] = fn
        rendered.append(tc)
    return rendered


def _dsv4_arguments_str(arguments: Any) -> str:
    """Missing/empty means no arguments (vLLM parity); anything else that is not
    a JSON object is rejected -- ValueError becomes a per-request "could not
    encode request" error, never a worker crash -- matching sglang's
    validate-then-400. A non-object would otherwise raise uncaught in the
    encoder's .items() or be wrapped as garbage."""
    if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    shown = f"{arguments!r:.200}"
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"tool call function.arguments must be valid JSON, got {shown}"
            ) from err
        if isinstance(parsed, dict):
            return arguments
    raise ValueError(f"tool call function.arguments must be a JSON object, got {shown}")


def _attach_tools_to_dsv4_messages(messages: list[dict], tools: list[dict]) -> None:
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": tools})
