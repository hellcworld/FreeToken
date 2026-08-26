from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from freetoken.core import SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.message import (
    BaseBackendMsg,
    BaseTokenizerMsg,
    DetokenizeMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.mm_embeds_map: Dict[int, torch.Tensor] = {}
        self.counter = 0

    @torch.inference_mode()
    def encode_images(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the vision tower + projector on processor outputs, returning the
        ``[num_image_tokens, hidden]`` soft-token embeddings (on device). gemma4-style
        callers pass ``image_position_ids``; Qwen-VL callers pass ``grid_thw``."""
        model = self.engine.model
        if not hasattr(model, "encode_images"):
            raise RuntimeError(f"{type(model).__name__} does not support image inputs")
        return model.encode_images(
            pixel_values.to(self.device),
            None if image_position_ids is None else image_position_ids.to(self.device),
            None if grid_thw is None else grid_thw.to(self.device),
        )

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(
                UserMsg(
                    uid=uid,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    mm_embeds=self.mm_embeds_map.get(uid),
                    mm_positions=self.mm_positions_map.get(uid),
                    mm_rope_delta=self.mm_delta_map.get(uid, 0),
                )
            )
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[BaseTokenizerMsg]) -> None:
        for msg in reply:
            if isinstance(msg, PromptAdmittedMsg):
                # PromptAdmittedMsg feeds the online server's global accounting. Offline
                # generation already owns its inputs and has no FrontendManager stats sink.
                continue
            assert isinstance(msg, DetokenizeMsg)
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token in self.eos_token_ids):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
        mm_inputs: List[Dict[str, torch.Tensor] | None] | None = None,
    ) -> List[Dict[str, str | List[int]]]:
        """Offline generation.

        ``mm_inputs`` (optional) is aligned with ``prompts``; each entry is either
        ``None`` (text-only) or a model-specific dict from the HF processor: gemma4
        uses ``pixel_values`` + ``image_position_ids``, qwen3_5 vision uses
        ``pixel_values`` + ``grid_thw`` (the interleaved-mRoPE rows are built here
        from the token-id prompt). For multimodal prompts pass token-id ``prompts``
        containing the model's image-token placeholders.
        """
        self.pending_requests = []
        self.status_map = {}
        self.mm_embeds_map = {}
        self.mm_positions_map: Dict[int, torch.Tensor] = {}
        self.mm_delta_map: Dict[int, int] = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        if mm_inputs is not None:
            for uid, mm in enumerate(mm_inputs):
                if mm is not None:
                    self.mm_embeds_map[uid] = self.encode_images(
                        mm["pixel_values"],
                        mm.get("image_position_ids"),
                        mm.get("grid_thw"),
                    )
                    if mm.get("grid_thw") is not None and "mm_position_ids" not in mm:
                        # qwen3_5 interleaved mRoPE rows, built from the token prompt
                        from freetoken.models.qwen3_5_moe.vision import qwen3_5_rope_index

                        prompt = prompts[uid]
                        assert isinstance(prompt, list), (
                            "multimodal qwen prompts must be token ids with placeholders"
                        )
                        image_token_id = self.engine.model.model._image_token_id
                        rows, delta = qwen3_5_rope_index(
                            torch.tensor(prompt, dtype=torch.int64), mm["grid_thw"],
                            image_token_id,
                        )
                        self.mm_positions_map[uid] = rows
                        self.mm_delta_map[uid] = delta
            torch.cuda.synchronize(self.device)
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
