"""Pre-flight check that a tile image is actually readable (FLT-01).

The mock tools never look at pixels, so a corrupted file would sail straight
through them; a real model endpoint would instead fail somewhere deep inside
inference. Checking up front turns both cases into one clean, explainable
outcome: the tile is escalated with a reason, and the job never crashes.

Stdlib-only structural checks (magic bytes + end-of-file markers) catch the
common failure — a truncated or garbage file. If Pillow happens to be
installed, a full decode is attempted as well (`decode=True`), which also
catches corruption mid-file.
"""
from __future__ import annotations

from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*")


class CorruptTileError(ValueError):
    """The tile image is missing, empty, truncated, or otherwise unreadable."""


def validate_tile_image(path: str | Path, decode: bool = True) -> None:
    """Raise CorruptTileError if `path` is not a readable image; else return None."""
    p = Path(path)
    if not p.is_file():
        raise CorruptTileError(f"tile file not found: {path}")

    data = p.read_bytes()
    if not data:
        raise CorruptTileError(f"tile file is empty: {path}")

    if data.startswith(PNG_SIGNATURE):
        if b"IEND" not in data[-32:]:
            raise CorruptTileError(f"PNG is truncated (no end chunk): {path}")
    elif data.startswith(JPEG_SIGNATURE):
        if not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
            raise CorruptTileError(f"JPEG is truncated (no end marker): {path}")
    elif data[:4] in TIFF_SIGNATURES:
        pass  # no cheap structural end-check for TIFF; the decode step covers it
    else:
        raise CorruptTileError(f"not a recognised image file: {path}")

    if decode:
        _try_full_decode(p)


def _try_full_decode(path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return  # Pillow not installed: structural checks above are all we can do
    try:
        with Image.open(path) as img:
            img.load()
    except Exception as exc:  # Pillow raises several different error types
        raise CorruptTileError(f"image failed to decode ({type(exc).__name__}): {path}") from exc
