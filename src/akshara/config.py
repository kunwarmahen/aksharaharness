"""Resolve provider credentials/models from environment variables.

Base-URL contract (matches the official SDKs so env files stay portable):

* ``ANTHROPIC_BASE_URL`` EXCLUDES the version segment -- the adapter
  appends ``/v1/messages``:
      https://api.anthropic.com     (direct)
      https://openrouter.ai/api     (OpenRouter)
* ``OPENAI_BASE_URL`` INCLUDES it -- the adapter appends ``/chat/completions``:
      https://api.openai.com/v1     (direct)
      https://openrouter.ai/api/v1  (OpenRouter)
* same for ``RESPONSES_BASE_URL`` -- the adapter appends ``/responses``
  (the Responses API lives on the same /v1 surface as chat-completions)
"""

from __future__ import annotations

import os
from pathlib import Path

from akshara.errors import ConfigError
from akshara.providers.base import ProviderSettings

_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-5",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    # OpenAI's Responses API: same /v1 surface, /responses endpoint.
    # Works against OpenAI directly, OpenRouter, or a local Ollama >= 0.13.3.
    "responses": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    # local server; model is a local tag you have pulled (ollama pull ...)
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3.8:latest",
    },
}

# Cloud windows are six figures; a typical local pull is not. Auto-compaction
# does the math against THIS number -- guessing 200k for an 8k local model
# would mean never compacting before the provider 400s.
_DEFAULT_CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic": 200_000,
    "openai": 200_000,
    "responses": 200_000,
    "ollama": 8192,
}

_dotenv_loaded = False


def _load_dotenv() -> None:
    """Import a .env file from the cwd into os.environ -- ONCE.

    ``uv run`` does NOT load .env for you (only ``--env-file`` does), and
    asking users to remember that flag is bad CLI manners. This loader is
    ~25 lines instead of a python-dotenv dependency: KEY=VALUE lines,
    full-line AND trailing '#' comments (``.env.example`` annotates every
    variable with one), optional quotes, split on the FIRST '=' so values
    may contain '='. Real environment variables ALWAYS win -- .env only
    fills gaps. Library callers get the same convenience via
    load_settings(); pass through if that's not what you want.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True

    path = Path(".env")
    if not path.is_file():
        return
    try:
        lines = path.read_text().splitlines()
    except OSError:
        # present but unreadable (root-owned, restrictive container
        # mount): treat as absent -- real env vars still work
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), _clean_value(value.strip())
        # a blank VALUE is template residue (``OLLAMA_API_KEY=   `` left
        # over from copying .env.example); importing it would shadow the
        # code's own fallbacks with ""
        if key and value and key not in os.environ:
            os.environ[key] = value


def _clean_value(raw: str) -> str:
    """One optional quote layer; then cut an unquoted `` # comment`` tail.

    A '#' inside quotes stays part of the value; so does one glued to an
    unquoted value (``abc#def``) -- only whitespace marks a comment.
    """
    if raw[:1] in ("'", '"'):
        close = raw.find(raw[0], 1)
        return raw[1:close] if close != -1 else raw[1:]
    for i, char in enumerate(raw):
        if char == "#" and (i == 0 or raw[i - 1].isspace()):
            return raw[:i].rstrip()
    return raw


def load_settings(name: str) -> ProviderSettings:
    """Read <PROVIDER>_API_KEY / <PROVIDER>_BASE_URL from the environment."""
    _load_dotenv()
    prefix = name.upper()
    if name == "ollama":
        # A local server needs no secret. Send a placeholder so the
        # Authorization header exists (proxies in front of Ollama may
        # require one; OLLAMA_API_KEY overrides).
        return ProviderSettings(
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url=os.environ.get(
                "OLLAMA_BASE_URL", _DEFAULTS["ollama"]["base_url"]
            ),
        )
    api_key = os.environ.get(f"{prefix}_API_KEY")
    if not api_key and name == "anthropic":
        # Claude Code's convention; some gateways set this instead.
        api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        raise ConfigError(
            f"No API key for provider {name!r}: set {prefix}_API_KEY "
            f"(see .env.example)"
        )
    return ProviderSettings(
        api_key=api_key,
        base_url=os.environ.get(
            f"{prefix}_BASE_URL", _DEFAULTS[name]["base_url"]
        ),
    )


def default_model(name: str) -> str:
    """Explicit env override wins; otherwise a per-provider default."""
    env_var = f"{name.upper()}_MODEL"
    return os.environ.get(env_var, _DEFAULTS[name]["model"])


def default_context_window(name: str) -> int:
    """Window assumption used when --context-window is not given."""
    env_var = f"{name.upper()}_CONTEXT_WINDOW"
    if value := os.environ.get(env_var):
        return int(value)
    return _DEFAULT_CONTEXT_WINDOWS[name]


def default_tool_select() -> int | None:
    """Width for dynamic tool loading from $AKSHARA_TOOLS_PER_TURN.

    Same semantics as --tool-select, which this backs when .env is more
    convenient than a flag: K forces that width on every session, 0
    forces selection off, and None (unset) leaves the auto-enable rule
    to decide. A non-integer is config corruption -- fail loudly rather
    than guess.
    """
    raw = os.environ.get("AKSHARA_TOOLS_PER_TURN")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(
            f"AKSHARA_TOOLS_PER_TURN must be an integer, got {raw!r}"
        ) from None


def browser_profile() -> Path | None:
    """Persistent Chromium profile dir from $AKSHARA_BROWSER_PROFILE.

    Set (a directory path) => every browser_* session -- and the headed
    one-shot opened by ``--browse-login`` -- shares this on-disk
    profile: log in once and every later session rides the saved
    cookies/localStorage. Unset => each session starts fresh, nothing
    persists (the safe default; a profile IS a plaintext credential
    store, so keeping it off until asked for is the honest default).
    A blank value counts as unset -- copying .env.example leaves empty
    templates around, and they must not shadow the code's own fallback.
    """
    raw = os.environ.get("AKSHARA_BROWSER_PROFILE", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def disabled_tool_patterns() -> list[str]:
    """Glob patterns from $AKSHARA_DISABLED_TOOLS (comma-separated).

    Each pattern is fnmatch-ed against registered tool names, so this
    hides one tool ('web_fetch'), a family ('browser_*'), or a whole
    MCP server ('mcp__slack__*'). Disabled tools are UNREGISTERED before
    catalog building, so they are never sent, never executed, and never
    suggested by list_available_tools either.
    """
    raw = os.environ.get("AKSHARA_DISABLED_TOOLS", "")
    return [p.strip() for p in raw.split(",") if p.strip()]
