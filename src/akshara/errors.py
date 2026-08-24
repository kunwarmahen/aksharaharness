"""Exception taxonomy.

Two families, kept deliberately separate:

* Provider/network failures (``ProviderError`` and friends) are terminal
  for the current turn. They propagate out of the agent loop; the CLI
  catches and reports them.
* Tool failures (``ToolError``) NEVER propagate. The agent loop converts
  them into ``is_error`` ToolResults so the model can see what went
  wrong and recover. Errors are data.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Missing/invalid local configuration (e.g. no API key in env)."""


class ProviderError(Exception):
    """The provider returned a non-2xx status, or a stream broke mid-flight."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.retry_after = retry_after


class AuthError(ProviderError):
    """401/403 -- key missing, wrong, or unauthorized for that model."""


class RateLimitError(ProviderError):
    """429 -- quota exhausted. Check ``retry_after`` before backing off."""


class ContextOverflowError(ProviderError):
    """The request didn't fit the model's context window.

    The seam that will handle this (history compaction) is
    Agent._before_model_call -- see notes/05-agent-loop.md.
    """


class ToolError(Exception):
    """Raised by tools (bad arguments, sandbox escape, timeout...).

    Converted to an is_error ToolResult by the agent loop -- the loop
    itself never sees this exception.
    """


class ImageError(ValueError):
    """A user-supplied image failed validation before any request went out.

    Raised by akshara.images.load_image_block. A CLI-input problem --
    neither a provider failure nor tool data -- so it surfaces as a
    usage error (exit 2), never inside the loop.
    """
