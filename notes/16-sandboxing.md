# 16 — Sandboxing: gates decide, sandboxes contain

*Companion to [notes/04](04-tools.md) (the bash tool) and
[notes/06](06-cli.md) (the permission gate). Book ch14's lesson, kept as
a protocol: containment is a LAYER, not a property of the bash tool.*

## The protocol

One seam, three methods (`akshara/sandbox.py`):

```python
class ToolSandbox(Protocol):
    def execute(self, command: list[str], *, cwd: Path,
                timeout: int) -> tuple[int, str]: ...  # (exit_code, combined output)
    @property
    def describe(self) -> str: ...                     # banner text
    @property
    def confined(self) -> bool: ...                    # is this REAL containment?
```

`Bash.run()` no longer spawns anything — it delegates to
`sandbox.execute(["bash", "-c", cmd], cwd=ctx.cwd, timeout=...)` and
translates `CommandTimedOut` (which carries the salvaged partial output)
into the same ToolError shape as before. Default `SubprocessSandbox()`,
so zero-wiring behavior is unchanged; the existing shell tests pass
untouched. That is the pluggability contract from the book: the tool
cannot tell which backend is under it.

## Two backends, two honesty levels

| | `SubprocessSandbox` | `BwrapSandbox` |
|---|---|---|
| mechanism | Popen + `start_new_session`, killpg on timeout/Ctrl-C, second `communicate()` to salvage partial output | bubblewrap: kernel-level namespaces |
| env | **allowlist**: `PATH LANG LC_ALL TERM SHELL`, plus `PAGER=cat GIT_PAGER=cat HOME→workspace`. Everything else — including every `*KEY* *TOKEN* *SECRET* AWS_ ANTHROPIC OPENAI` var — never reaches the child | `--clearenv` then the same allowlist set explicitly |
| network | full access | `--unshare-net` by default (`allow_network=True` drops it) |
| filesystem | **everything visible, everything writable** | only `/usr /bin /lib /lib64 /etc/ld.so.cache /dev /proc` (read-only) + the workspace (rw); `/home` and the rest of `/etc` simply do not exist in there |
| pid tree | children can orphan on a hard kill | `--unshare-pid`: bwrap is pid-1 init; when init dies the kernel reaps the whole tree — orphan escape is structurally dead |
| `confined` | **False** — convenience confinement, not security | True |

## What bwrap actually buys (verified live, not claimed)

The suite proves containment empirically wherever bwrap exists
(`tests/test_sandbox.py` skips elsewhere): a write to `/etc/` produces
**no file on the host**, the host home directory is invisible, an
`example.com:80` connect fails inside the namespace, a secret exported
in the parent's env prints empty, the workspace persists across calls,
and a timed-out `sleep 60 &` tree is gone in under 30s — not the
child's 60.

Two write-semantics subtleties worth knowing because they are easy to
overclaim:

* Unmounted paths (`/home/...`) → ENOENT. Clean.
* Paths that ARE mount anchors (`/etc/ld.so.cache`'s directory) accept
  writes that land on an ephemeral tmpfs and vanish with the sandbox.
  So "writes outside the workspace fail" is FALSE; "never reach the
  host" is the true invariant. The docstring says exactly that.

Unprivileged user namespaces make this work without root or
setuid helpers on stock kernels (`autodetect()` probes once with
`bwrap --unshare-pid … --ro-bind / / /bin/true` and falls back to
SubprocessSandbox if userns is disabled — never raises).

## The gate bridge: orthogonality preserved

`permissions.trust_sandbox(inner)` auto-approves `bash` ONLY while its
sandbox reports `confined`; every other tool delegates to the inner
gate. An unconfined sandbox earns nothing — even bash still asks. This
keeps the book's division clean: **gates decide** (should this run?),
**sandboxes contain** (what can it touch?). Neither substitutes for the
other, and neither knows the other's internals beyond the one boolean.

## Wiring

```
uv run akshara --sandbox              # autodetect: bwrap if usable, else subprocess
uv run akshara --sandbox none         # explicit legacy behavior
```

REPL inherits the sandbox into sub-agents and `/build` children;
`--build` uses `trust_sandbox(confirm_gate)` so a confined build runs
unattended while an unconfined one warns loudly and asks.

## What we deliberately did NOT defend against

No seccomp filter (bwrap's namespace wall is the threat model), no
resource limits beyond wall-clock timeout, no read-only hiding of the
workspace itself, and no protection against a model that convinces the
USER to run something outside the sandbox — gates still decide.
