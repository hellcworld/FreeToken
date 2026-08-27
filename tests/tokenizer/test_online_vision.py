"""Unit tests for the online multimodal (image) input path.

Covers the pure-Python pieces: content-part extraction/normalization (both the
raw OpenAI ``image_url`` shape and the frontend-normalized ``image`` shape),
the ``image_token_id`` placeholder expansion splice, and the 2D-tensor message
serialization the tokenizer worker -> scheduler mm payload relies on.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.message.utils import deserialize_type, serialize_type
from freetoken.tokenizer.multimodal import (
    embeds_from_grid,
    expand_image_pad_ids,
    extract_image_parts,
    is_image_part,
)


class TestExtractImageParts:
    def test_raw_openai_shape(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "look:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
        ]
        out, urls = extract_image_parts(messages)
        assert urls == ["data:image/png;base64,AAA"]
        assert out[0]["content"] == [
            {"type": "text", "text": "look:"},
            {"type": "image", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        # input untouched
        assert messages[0]["content"][1]["type"] == "image_url"

    def test_normalized_image_shape_round_trips(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image_url": {"url": "http://x/y.png"}},
            ]},
        ]
        out, urls = extract_image_parts(messages)
        assert urls == ["http://x/y.png"]
        assert out[0]["content"][0]["type"] == "image"

    def test_plain_content_untouched(self):
        messages = [
            {"role": "user", "content": "just text"},
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
        ]
        out, urls = extract_image_parts(messages)
        assert urls == []
        assert out == messages

    def test_missing_url_rejected(self):
        with pytest.raises(ValueError, match="missing image_url.url"):
            extract_image_parts([
                {"role": "user", "content": [{"type": "image_url", "image_url": {}}]},
            ])

    def test_other_part_types_still_rejected(self):
        with pytest.raises(ValueError, match="Unsupported content part"):
            extract_image_parts([
                {"role": "user", "content": [{"type": "audio_url", "audio_url": {}}]},
            ])

    def test_is_image_part(self):
        assert is_image_part({"type": "image_url", "image_url": {"url": "u"}})
        assert is_image_part({"type": "image", "image_url": {"url": "u"}})
        assert not is_image_part({"type": "text", "text": "t"})
        assert not is_image_part("text")


class TestExpandImagePadIds:
    PAD = 248056

    def test_single_image(self):
        grid = [[1, 30, 40]]  # 300 embeds
        ids = [1, 2, self.PAD, 3]
        out = expand_image_pad_ids(ids, grid, self.PAD)
        assert out == [1, 2] + [self.PAD] * 300 + [3]

    def test_two_images_in_order(self):
        grid = [[1, 4, 4], [1, 6, 6]]  # 4 and 9 embeds
        ids = [self.PAD, 9, self.PAD]
        out = expand_image_pad_ids(ids, grid, self.PAD)
        assert out == [self.PAD] * 4 + [9] + [self.PAD] * 9

    def test_more_placeholders_than_images(self):
        with pytest.raises(ValueError, match="more image placeholders"):
            expand_image_pad_ids([self.PAD, self.PAD], [[1, 4, 4]], self.PAD)

    def test_template_rendered_no_placeholder(self):
        with pytest.raises(ValueError, match="no image placeholder"):
            expand_image_pad_ids([1, 2, 3], [[1, 4, 4]], self.PAD)

    def test_embeds_from_grid(self):
        assert embeds_from_grid([1, 30, 40], 2) == 300
        assert embeds_from_grid([2, 8, 8], 2) == 64


class TestTensor2DSerialization:
    def test_round_trip_2d(self):
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        data = serialize_type(t)
        out = deserialize_type({}, data)
        assert out.shape == (3, 4)
        assert out.dtype == torch.float32
        assert torch.equal(out, t)

    def test_round_trip_1d_still_works(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        out = deserialize_type({}, serialize_type(t))
        assert out.shape == (3,)
        assert torch.equal(out, t)

    def test_3d_rejected(self):
        with pytest.raises(AssertionError):
            serialize_type(torch.zeros(2, 2, 2))
