"""Loading user-supplied images into ImageBlocks.

Providers take base64 payloads inline -- never file paths -- so the CLI
reads and encodes the bytes up front, before any request exists. The
constraints here mirror Anthropic's documented image limits (5 MB per
image; png/jpeg/gif/webp); the OpenAI dialect accepts the same four
types carried as ``data:`` URLs instead.

Validation is deliberately shallow: extension -> MIME mapping and a
size cap. We do NOT sniff magic bytes or decode pixels -- a wrong
extension produces a provider-side 400 the user sees verbatim, which
teaches more than silently second-guessing them.
"""

from __future__ import annotations

import base64
from pathlib import Path

from akshara.errors import ImageError
from akshara.types import ImageBlock

#: extension -> MIME type. Case-insensitive via .lower() at lookup.
MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

#: Anthropic's per-image cap (the stricter of the two dialects). Enforced
#: on RAW bytes -- base64 inflates by 4/3 on the wire, but providers size
#: their limits in decoded bytes.
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _media_type_for(name: str) -> str:
    """Extension -> MIME lookup shared by both loaders."""
    media_type = MEDIA_TYPES.get(Path(name).suffix.lower())
    if media_type is None:
        supported = ", ".join(sorted(MEDIA_TYPES))
        suffix = Path(name).suffix or "(none)"
        raise ImageError(
            f"unsupported image type '{suffix}' "
            f"for {Path(name).name} -- supported: {supported}"
        )
    return media_type


def _capped(raw: bytes, name: str) -> None:
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(
            f"{name} is {len(raw) / 1e6:.1f} MB -- over the "
            f"{MAX_IMAGE_BYTES / 1e6:.0f} MB per-image cap"
        )


def load_image_block(path: Path) -> ImageBlock:
    """Read an image file into a request-ready ImageBlock.

    Raises ImageError (a ValueError) on missing files, unsupported
    extensions, or oversize payloads -- before any turn starts, so bad
    input never pollutes history.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"image not found: {path}")

    media_type = _media_type_for(str(path))
    raw = path.read_bytes()
    _capped(raw, path.name)

    return ImageBlock(media_type=media_type,
                      data=base64.b64encode(raw).decode("ascii"))


def image_block_from_bytes(filename: str, raw: bytes) -> ImageBlock:
    """Same contract, for bytes that never touched a filesystem (web UI
    uploads arrive base64-decoded over HTTP). Same shallow philosophy:
    the extension decides the MIME type, size is capped, pixels un-sniffed.
    """
    media_type = _media_type_for(filename)
    _capped(raw, filename)
    return ImageBlock(media_type=media_type,
                      data=base64.b64encode(raw).decode("ascii"))
