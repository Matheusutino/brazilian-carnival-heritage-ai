"""Utility helpers for encoding images to base64 for LLM consumption."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Tuple

from PIL import Image


def encode_image_for_llm(
    image_path: Path,
    max_dim: int | None = 500,
    max_b64_length: int | None = 45000,
    min_dim: int = 192,
    initial_quality: int = 90,
    min_quality: int = 60,
) -> Tuple[str, Tuple[int, int]]:
    """Encode image as base64 while keeping size within token-friendly bounds.

    When ``max_dim`` is provided, the image is scaled so the largest dimension
    does not exceed that value. If ``max_b64_length`` is also set, the function
    iteratively reduces resolution/quality to stay within the limit.

    Args:
        image_path: Path to the image file.
        max_dim: Maximum allowed dimension (width or height).
        max_b64_length: Maximum acceptable length for the base64 string.
        min_dim: Minimum dimension threshold before giving up on further resizing.
        initial_quality: Starting JPEG quality.
        min_quality: Minimum JPEG quality allowed.

    Returns:
        Tuple containing the base64 string and the final image size (width, height).
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size

        if max(orig_w, orig_h) == 0:
            raise ValueError(f"Image has invalid dimensions: {image_path}")

        scale = 1.0
        if max_dim:
            scale = min(1.0, max_dim / max(orig_w, orig_h))
        quality = initial_quality

        while True:
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))

            if new_w == orig_w and new_h == orig_h:
                resized = img
            else:
                resized = img.resize((new_w, new_h), Image.LANCZOS)

            buffer = BytesIO()
            resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buffer.getvalue()).decode()

            if max_b64_length is None or len(b64) <= max_b64_length:
                return b64, resized.size

            if max_b64_length is None:
                return b64, resized.size

            if max(resized.size) <= min_dim and quality <= min_quality:
                # Cannot shrink further without sacrificing too much quality.
                return b64, resized.size

            # Reduce size/quality and try again
            if max(resized.size) > min_dim:
                scale *= 0.85
            elif quality > min_quality:
                quality = max(min_quality, quality - 5)
            else:
                return b64, resized.size
