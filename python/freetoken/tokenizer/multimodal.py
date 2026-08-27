"""Multimodal (image) input helpers for the online tokenizer path.

The OpenAI wire format carries images as ``{"type": "image_url", "image_url":
{"url": ...}}`` content parts (data URLs or http(s) URLs). This module extracts
those parts, decodes them to PIL images, and provides the placeholder-expansion
splice that turns the chat template's single ``image_token_id`` marker per image
into the ``t*h*w // merge**2`` soft-token slots the engine's merge expects
(the same contract as the offline ``LLM.generate(mm_inputs=...)`` path).
"""
from __future__ import annotations

import base64
import binascii
import io
import urllib.request
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pillow is a freetoken[vision] extra; import lazily at use
    from PIL.Image import Image as PILImage

MAX_IMAGES_PER_REQUEST = 4
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MiB download cap for http(s) URLs
URL_TIMEOUT_SECONDS = 15.0


def is_image_part(part: Any) -> bool:
    """True for both the raw OpenAI shape (``type == "image_url"``) and the
    normalized template shape (``type == "image"``, carrying the raw url)."""
    if not isinstance(part, dict):
        return False
    return part.get("type") in ("image_url", "image")


def extract_image_parts(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split messages into (normalized_messages, image_urls).

    ``image_url`` parts (either spelling) become ``{"type": "image",
    "image_url": {"url": ...}}`` (the convention Qwen-VL chat templates
    render); text parts are kept as explicit ``{"type": "text", "text": ...}``
    dicts. Messages whose content was not a list are passed through untouched.
    Returns new structures; the input is not mutated.
    """
    urls: list[str] = []
    out_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not (isinstance(content, list) and any(is_image_part(p) for p in content)):
            out_messages.append(message)
            continue
        m = dict(message)
        parts_out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts_out.append({"type": "text", "text": part.get("text") or ""})
            elif is_image_part(part):
                holder = part.get("image_url")
                url = holder.get("url") if isinstance(holder, dict) else holder
                if not url or not isinstance(url, str):
                    raise ValueError("image_url part is missing image_url.url")
                urls.append(url)
                parts_out.append({"type": "image", "image_url": {"url": url}})
            else:
                ptype = part.get("type") if isinstance(part, dict) else None
                raise ValueError(
                    f"Unsupported content part type for text-only server: {ptype}"
                )
        m["content"] = parts_out
        out_messages.append(m)
    return out_messages, urls


def load_image_from_url(url: str) -> "PILImage":
    """Decode a data: URL (base64) or fetch an http(s) URL into a PIL image."""
    from PIL import Image

    if url.startswith("data:"):
        comma = url.find(",")
        if comma < 0:
            raise ValueError("malformed data URL: no ',' separator")
        try:
            payload = base64.b64decode(url[comma + 1 :], validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"invalid base64 image payload: {exc}") from exc
    elif url.startswith(("http://", "https://")):
        request = urllib.request.Request(url, headers={"User-Agent": "freetoken/0.1"})
        with urllib.request.urlopen(request, timeout=URL_TIMEOUT_SECONDS) as response:
            payload = response.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"image at {url[:120]} exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB cap"
            )
    else:
        raise ValueError(
            "image url must be a data: URL or an http(s) URL "
            f"(got {url[:60]!r}...)"
        )
    img = Image.open(io.BytesIO(payload))
    img.load()
    return img.convert("RGB")


def embeds_from_grid(grid_thw_row: list[int], spatial_merge_size: int) -> int:
    """Soft-token slots one image contributes: patches / merge_area."""
    t, h, w = (int(v) for v in grid_thw_row)
    merged = spatial_merge_size * spatial_merge_size
    return (t * h * w) // merged


def expand_image_pad_ids(
    input_ids: list[int],
    grid_thw: "list[list[int]]",
    image_token_id: int,
    spatial_merge_size: int = 2,
) -> list[int]:
    """Replace each ``image_token_id`` occurrence in ``input_ids`` with the
    per-image slot count derived from ``grid_thw`` (request order == image
    order). Raises when the counts cannot be matched 1:1."""
    per_image = [embeds_from_grid(row, spatial_merge_size) for row in grid_thw]
    out: list[int] = []
    idx = 0
    for tid in input_ids:
        if tid == image_token_id:
            if idx >= len(per_image):
                raise ValueError(
                    "prompt has more image placeholders than images "
                    f"({len(per_image)} image(s) supplied)"
                )
            out.extend([image_token_id] * per_image[idx])
            idx += 1
        else:
            out.append(tid)
    if idx != len(per_image):
        raise ValueError(
            "chat template rendered no image placeholder for every supplied image "
            f"({idx} of {len(per_image)} consumed); this checkpoint's template "
            "may not support image content parts"
        )
    return out
