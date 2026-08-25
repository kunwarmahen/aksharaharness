"""The permission gate -- how a harness asks a human before acting.

A gate is just ``Callable[[PermissionRequest], bool]``. Keeping it a
plain function (not a class hierarchy) is deliberate: the CLI supplies
one that renders a rich y/n/e prompt; tests supply lambdas.

Every dangerous call produces EXACTLY one gate invocation, and whatever
it answers becomes data: denied calls turn into ``is_error`` ToolResults
the model can read and react to ("the user said no, try another way"),
never into exceptions.

Approve-with-edits: a gate may REPLACE ``request.arguments`` before
answering True (amend the bash command, fix the path). The agent loop
notices the swap (identity check) and adopts the edited dict as the
call's arguments -- so what runs is what was approved, edits included.
``summarize`` lets an editing UI re-render the preview for the amended
args without knowing anything about the tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

#: The two runtime modes a session can sit in. "ask" defers to whatever
#: gate the frontend supplied (y/n/e terminal prompt, browser modal);
#: "yolo" approves everything.
MODES = ("ask", "yolo")


@dataclass(slots=True)
class PermissionRequest:
    """Everything a human needs to judge one tool call.

    Mutable ON PURPOSE, within one narrow contract: gates may rewrite
    ``arguments`` (and refresh ``summary``) while deciding. Everything
    else about the request is the loop's business.
    """

    tool_name: str
    arguments: dict[str, Any]
    summary: str  # built by the tool itself: the literal command / diff
    read_only: bool
    #: tool.summary(args, ctx) with the context pre-bound by the loop, so
    #: an edit-and-reapprove UI can re-render the preview for amended args.
    #: None => the UI falls back to showing raw JSON.
    summarize: Callable[[dict[str, Any]], str] | None = None


#: A gate receives one request per dangerous call; True => allow.
PermissionFn = Callable[[PermissionRequest], bool]


def allow_read_only(request: PermissionRequest) -> bool:
    """Auto-approve tools that declared read_only; deny everything else."""
    return request.read_only


def yolo(request: PermissionRequest) -> bool:
    """Allow everything (--yolo). You trust the model; you accept the risk."""
    return True


def deny_all(request: PermissionRequest) -> bool:
    """Deny everything. Useful for dry-runs and tests."""
    return False


def trust_sandbox(inner: PermissionFn, sandbox) -> PermissionFn:
    """Auto-approve bash ONLY while it runs in a kernel-confined sandbox.

    The bridge between the two orthogonal layers (book ch14): gates
    decide, sandboxes contain -- this decorator lets containment EARN
    approval for exactly one tool, delegating everything else to
    ``inner`` unchanged. Reads ``sandbox.confined`` (True only for
    BwrapSandbox), so a downgrade to SubprocessSandbox silently restores
    prompting: convenience confinement never counts as trust.
    """
    def gate(request: PermissionRequest) -> bool:
        if request.tool_name == "bash" and sandbox.confined:
            return True
        return inner(request)
    return gate


@dataclass
class SwitchableGate:
    """A gate whose policy flips between "ask" and "yolo" MID-SESSION.

    The CLI wires one of these around the frontend's real ask-gate
    (confirm prompt or browser modal), so /yolo in the REPL and the mode
    chip in the web UI are just ``set_mode`` calls -- the agent loop is
    untouched, it keeps calling this object like any other gate. A flip
    mid-turn simply applies to every not-yet-approved call afterwards;
    the plain attribute read needs no lock (one writer, GIL-atomic).

    Kept callable rather than adding a mode parameter to Agent on
    purpose: gates are plain functions by design here, and embedders who
    pass a bare function get today's fixed-gate behavior unchanged.
    """

    ask: PermissionFn  # what runs while mode == "ask"
    mode: str = "ask"

    def __call__(self, request: PermissionRequest) -> bool:
        if self.mode == "yolo":
            return yolo(request)
        return self.ask(request)

    def set_mode(self, mode: str) -> None:
        """Flip the policy; unknown names are a loud error, not a silent ask."""
        if mode not in MODES:
            raise ValueError(f"unknown permission mode {mode!r} "
                             f"(want one of {', '.join(MODES)})")
        self.mode = mode

    def toggle(self) -> str:
        """ask -> yolo -> ask ...; returns the mode now in force."""
        self.set_mode("ask" if self.mode == "yolo" else "yolo")
        return self.mode
