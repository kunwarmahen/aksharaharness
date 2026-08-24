"""Hand-rolled Server-Sent Events parsing (the WHATWG SSE framing rules).

Two pure, provider-agnostic layers:

1. line splitting (bytes -> str lines)
   * incremental UTF-8 decoding -- a multibyte character CAN split
     across network chunks; decoding chunk-by-chunk corrupts it
   * splits on CRLF, LF, or lone CR (the spec allows all three)
   * a trailing CR is ambiguous (could be half of a CRLF spanning the
     chunk boundary), so we wait for more data before deciding

2. event grouping (lines -> (event_name | None, data) pairs)
   * ``field: value`` -- strip exactly ONE leading space from the value
   * multiple ``data:`` lines in one event join with ``\\n``
   * lines starting with ``:`` are comments/keep-alives -- ignore
   * a blank line dispatches the accumulated event
   * ``event:`` names the event (Anthropic sends names); OpenAI never does

Neither layer knows about provider payloads or the ``[DONE]`` sentinel --
sentinel handling belongs to the OpenAI adapter.

The logic lives in two incremental classes (``SSELineSplitter``,
``SSEEventGrouper``) with ``feed()/flush()`` methods; the sync functions
(``iter_sse_lines``, ``parse_events``) and their async twins
(``aiter_sse_lines``, ``aparse_events``) are thin skins over them. That
is how BOTH transports share 100% of the framing rules -- the same
trick the stream routers use in the adapters.
"""

from __future__ import annotations

import codecs
import re
from collections.abc import AsyncIterator, Iterable, Iterator

_LINE_TERMINATOR = re.compile(r"\r\n|\r|\n")


class SSELineSplitter:
    """Incremental bytes -> decoded lines. Feed chunks, collect lists."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""

    def feed(self, chunk: bytes) -> list[str]:
        self._pending += self._decoder.decode(chunk)
        return self._drain(final=False)

    def flush(self) -> list[str]:
        # Tolerate a missing final newline (spec says be liberal).
        self._pending += self._decoder.decode(b"", final=True)
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            match = _LINE_TERMINATOR.search(self._pending)
            if match is None:
                break
            if not final and match.group() == "\r" and match.end() == len(self._pending):
                break  # lone trailing \r may become \r\n with more data
            out.append(self._pending[: match.start()])
            self._pending = self._pending[match.end():]
        if final and self._pending:
            out.append(self._pending)
            self._pending = ""
        return out


class SSEEventGrouper:
    """Incremental lines -> (event_name | None, data) pairs."""

    def __init__(self) -> None:
        self._event_name: str | None = None
        self._data_lines: list[str] = []

    def feed(self, line: str) -> list[tuple[str | None, str]]:
        if line == "":
            # Blank line = event terminator. Dispatch only if there was
            # data; a name-only event carries nothing usable.
            dispatched = self._take()
            return dispatched
        if line.startswith(":"):
            return []  # comment / keep-alive ping
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]  # strip exactly one leading space
        if field == "data":
            self._data_lines.append(value)
        elif field == "event":
            self._event_name = value
        # other fields (id:, retry:) don't concern us
        return []

    def flush(self) -> list[tuple[str | None, str]]:
        # End of stream: tolerate a final event with no blank-line
        # terminator (some gateways truncate). Being liberal here costs
        # nothing -- a well-formed stream dispatches via the blank line.
        return self._take()

    def _take(self) -> list[tuple[str | None, str]]:
        if not self._data_lines:
            return []
        pair = (self._event_name, "\n".join(self._data_lines))
        self._event_name = None
        self._data_lines = []
        return [pair]


def iter_sse_lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield decoded text lines from raw response chunks."""
    splitter = SSELineSplitter()
    for chunk in chunks:
        yield from splitter.feed(chunk)
    yield from splitter.flush()


def parse_events(lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
    """Group SSE lines into ``(event_name, data)`` pairs.

    ``event_name`` is None when the stream never sent an ``event:``
    field (OpenAI's shape).
    """
    grouper = SSEEventGrouper()
    for line in lines:
        yield from grouper.feed(line)
    yield from grouper.flush()


# ---- async twins -----------------------------------------------------------
#
# The whole point of the feeder classes: these repeat NONE of the rules
# above -- they only re-shape the iteration (async for instead of for).


async def aiter_sse_lines(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Async twin of :func:`iter_sse_lines`."""
    splitter = SSELineSplitter()
    async for chunk in chunks:
        for line in splitter.feed(chunk):
            yield line
    for line in splitter.flush():
        yield line


async def aparse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str | None, str]]:
    """Async twin of :func:`parse_events`."""
    grouper = SSEEventGrouper()
    async for line in lines:
        for pair in grouper.feed(line):
            yield pair
    for pair in grouper.flush():
        yield pair
