"""The SSE framing parser against hand-built byte strings.

These pin the WHATWG SSE rules our adapters depend on. The regression
tests near the bottom (CR-at-chunk-boundary, multibyte-across-chunks)
are the ones that catch the bugs real networks cause.
"""

from __future__ import annotations

from akshara.providers.sse import iter_sse_lines, parse_events


def lines_of(*chunks: bytes) -> list[str]:
    return list(iter_sse_lines(chunks))


def events_of(raw: str) -> list[tuple[str | None, str]]:
    return list(parse_events(iter_sse_lines([raw.encode()])))


class TestLineSplitting:
    def test_lf(self):
        assert lines_of(b"a\nb\n") == ["a", "b"]

    def test_crlf(self):
        assert lines_of(b"a\r\nb\r\n") == ["a", "b"]

    def test_lone_cr(self):
        assert lines_of(b"a\rb\r") == ["a", "b"]

    def test_cr_at_chunk_boundary_is_not_two_lines(self):
        # b"...\r" + b"\n..." is ONE terminator spanning chunks.
        # Getting this wrong emits a spurious empty line, which would
        # prematurely terminate an SSE event mid-stream.
        got = lines_of(b"data: a\r", b"\ndata: b\n\n")
        assert got == ["data: a", "data: b", ""]

    def test_multibyte_char_split_across_chunks(self):
        # 中 = \xe4\xb8\xad -- split it in half and it must still decode.
        got = lines_of(b"data: \xe4", b"\xb8\xad\n")
        assert got == ["data: 中"]

    def test_missing_final_newline_is_flushed(self):
        assert lines_of(b"data: tail") == ["data: tail"]


class TestEventFraming:
    def test_named_event(self):
        assert events_of("event: foo\ndata: bar\n\n") == [("foo", "bar")]

    def test_unnamed_event(self):
        assert events_of("data: hello\n\n") == [(None, "hello")]

    def test_multiple_data_lines_join_with_newline(self):
        assert events_of("data: a\ndata: b\n\n") == [(None, "a\nb")]

    def test_strips_exactly_one_leading_space(self):
        assert events_of("data:  two spaces\n\n") == [(None, " two spaces")]

    def test_comment_lines_are_ignored(self):
        assert events_of(": keepalive\ndata: x\n\n") == [(None, "x")]

    def test_no_data_means_no_event_dispatched(self):
        assert events_of("event: x\n\n") == []

    def test_data_line_without_colon_is_empty_value(self):
        assert events_of("data\n\n") == [(None, "")]

    def test_anthropic_style_sequence(self):
        raw = (
            "event: message_start\n"
            'data: {"type":"message_start"}\n'
            "\n"
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n'
            "\n"
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n'
            "\n"
        )
        names = [name for name, _ in events_of(raw)]
        assert names == [
            "message_start",
            "content_block_delta",
            "message_stop",
        ]
