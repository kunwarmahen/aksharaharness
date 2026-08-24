"""Local models via Ollama -- a profile of the OpenAI dialect.

Ollama serves an OpenAI-compatible chat-completions API on
``http://localhost:11434/v1``, so the whole wire half of this provider
is inherited from :class:`OpenAIProvider`. What actually differs:

* **No auth.** The local server ignores Authorization headers; we send
  ``Bearer ollama`` because *some* value is conventional (and proxies in
  front of Ollama may want a real one via ``OLLAMA_API_KEY``).
* **Local-scale context.** An 8k-window local model with the cloud
  default 200k context assumption would never trip auto-compaction
  before overflowing; config.default_context_window() answers 8192 for
  this name.
* **Model slugs are local tags** ("qwen3.8:latest", "llama3.2") -- set with
  OLLAMA_MODEL / --model, never guessed.

The distinct ``name`` matters beyond cosmetics: session checkpoints
store it so ``--resume`` rebuilds THIS provider, not plain openai.
"""

from __future__ import annotations

from akshara.providers.openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    name = "ollama"
