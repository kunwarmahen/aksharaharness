"""Exclusive, expiring holds on shared resources -- the ch17 hazard list,
kept as small as it promised.

Parallel tool batches made a race REAL: two ``edit_file`` calls aimed at
one path in the same batch can interleave read-modify-write and silently
eat an edit. The book's answer is leases; the principle we kept from its
guidance is *reads need no leases* -- only mutating tools ever hold one.

Semantics, each chosen for a reason:

* **TTL on every lease** -- a crashed/hung holder must not deadlock the
  resource forever. Expiry is checked at ACQUIRE time (lazy, no reaper
  thread): the next contender steals naturally.
* **Blocking acquire with deadline** -- most conflicts are siblings from
  one batch finishing in milliseconds; waiting converts them into
  serialization for free. Only past the deadline does it surface as
  ``LeaseBusy`` -- a ToolError, so errors-are-data applies: the model
  reads who held what and can retry.
* **Reentrancy by owner** -- the same owner re-acquiring refreshes the
  TTL instead of deadlocking against itself; nested holds count depth
  and the LAST release frees.
* **Only the holder releases** -- releasing someone else's lease is a
  bug, not a no-op.

Honest scope: guards ONE process's shared contexts (a tool batch's
workers all share the agent's ToolContext). Sub-agents build their own
ToolContext and their own manager -- their runs are already serialized
against the parent's batch, so there is nothing here for them to share
YET. And bash bypasses both the sandbox and this: consistency, since
bash was never confinable.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass

from akshara.errors import ToolError

_POLL_SECONDS = 0.01


class LeaseBusy(ToolError):
    """The resource stayed held past the acquisition deadline."""


@dataclass(slots=True)
class Lease:
    key: str
    owner: str
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at


def new_owner() -> str:
    """One owner identity per tool CALL (never reuse across calls)."""
    return uuid.uuid4().hex[:12]


class LeaseManager:
    """Key -> current Lease. One per shared ToolContext."""

    def __init__(self) -> None:
        self._held: dict[str, Lease] = {}
        self._depth: dict[tuple[str, str], int] = {}  # (key, owner) -> nests

    def holder(self, key: str) -> str | None:
        lease = self._held.get(key)
        if lease is None:
            return None
        if lease.expired:  # lazy expiry: no reaper thread needed
            del self._held[key]
            return None
        return lease.owner

    def acquire(self, key: str, owner: str, *, ttl: float = 60.0,
                timeout: float = 30.0) -> Lease:
        """Hold ``key`` exclusively, waiting up to ``timeout`` seconds.

        Same-owner re-acquire refreshes the TTL (nesting counts depth).
        """
        deadline = time.monotonic() + timeout
        while True:
            current = self.holder(key)  # also purges expired leases
            if current is None or current == owner:
                lease = self._held.get(key)
                if lease is None or lease.owner != owner:
                    lease = Lease(key=key, owner=owner,
                                  expires_at=time.monotonic() + ttl)
                    self._held[key] = lease
                    self._depth[(key, owner)] = 0
                else:  # reentrant: refresh TTL, count the nesting
                    lease.expires_at = time.monotonic() + ttl
                self._depth[(key, owner)] += 1
                return lease
            if time.monotonic() >= deadline:
                raise LeaseBusy(
                    f"resource {key!r} is busy: held by {current!r} "
                    f"(waited {timeout:.1f}s) -- retry once it finishes")
            time.sleep(_POLL_SECONDS)

    def release(self, key: str, owner: str) -> None:
        depth = self._depth.get((key, owner))
        if depth is None:
            raise ToolError(
                f"cannot release {key!r}: {owner!r} does not hold it")
        if depth > 1:
            self._depth[(key, owner)] = depth - 1
            return
        del self._depth[(key, owner)]
        lease = self._held.get(key)
        if lease is not None and lease.owner == owner:
            del self._held[key]

    @contextmanager
    def hold(self, key: str, *, ttl: float = 60.0, timeout: float = 30.0):
        """``with mgr.hold(key):`` -- always released, even on exception."""
        owner = new_owner()
        lease = self.acquire(key, owner, ttl=ttl, timeout=timeout)
        try:
            yield lease
        finally:
            self.release(key, owner)
