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


def load_image_block(path: Path) -> ImageBlock:
    """Read an image file into a request-ready ImageBlock.

    Raises ImageError (a ValueError) on missing files, unsupported
    extensions, or oversize payloads -- before any turn starts, so bad
    input never pollutes history.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageError(f"image not found: {path}")

    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        supported = ", ".join(sorted(MEDIA_TYPES))
        suffix = path.suffix or "(none)"
        raise ImageError(
            f"unsupported image type '{suffix}' "
            f"for {path.name} -- supported: {supported}"
        )

    raw = path.read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(
            f"{path.name} is {len(raw) / 1e6:.1f} MB -- over the "
            f"{MAX_IMAGE_BYTES / 1e6:.0f} MB per-image cap"
        )

    return ImageBlock(media_type=media_type,
                      data=base64.b64encode(raw).decode("ascii"))
