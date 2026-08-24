"""Leases: exclusive, expiring holds -- the ch17 write-conflict answer.

Deterministic mechanics here (contention, TTL expiry, reentrancy,
release discipline); the fs integration is pinned by pre-holding the
exact key format the tools use.
"""

from __future__ import annotations

import threading
import time

import pytest

from akshara.errors import ToolError
from akshara.leases import LeaseBusy, LeaseManager, new_owner
from akshara.tools.base import ToolContext
from akshara.tools.fs import EditFile, WriteFile, file_lease_key


class TestMechanics:
    def test_acquire_then_release_frees_the_key(self):
        mgr = LeaseManager()
        mgr.acquire("res", "a", ttl=10)
        assert mgr.holder("res") == "a"
        mgr.release("res", "a")
        assert mgr.holder("res") is None

    def test_second_owner_blocks_until_release(self):
        mgr = LeaseManager()
        mgr.acquire("res", "a", ttl=10)
        got = []

        def contender():
            with mgr.hold("res", ttl=10, timeout=5):
                got.append("b-inside")

        thread = threading.Thread(target=contender)
        thread.start()
        time.sleep(0.05)
        assert got == []          # still waiting on a's hold
        mgr.release("res", "a")
        thread.join(timeout=2)
        assert got == ["b-inside"]

    def test_timeout_surfaces_as_leasebusy_naming_the_holder(self):
        mgr = LeaseManager()
        mgr.acquire("res", "greedy-owner", ttl=10)
        with pytest.raises(LeaseBusy) as info:
            mgr.acquire("res", "b", timeout=0.05)
        assert "greedy-owner" in str(info.value)

    def test_ttl_expiry_reclaims_without_release(self):
        mgr = LeaseManager()
        mgr.acquire("res", "crashed-holder", ttl=0.05)  # holder never releases
        time.sleep(0.08)
        assert mgr.holder("res") is None  # lazy purge on next look
        mgr.acquire("res", "next", ttl=10)  # no waiting needed
        assert mgr.holder("res") == "next"

    def test_reentrant_acquire_refreshes_and_nests(self):
        mgr = LeaseManager()
        mgr.acquire("res", "a", ttl=10)
        mgr.acquire("res", "a", ttl=10)  # must not deadlock against itself
        mgr.release("res", "a")
        assert mgr.holder("res") == "a"  # outer hold survives inner release
        mgr.release("res", "a")
        assert mgr.holder("res") is None

    def test_only_the_holder_releases(self):
        mgr = LeaseManager()
        mgr.acquire("res", "a", ttl=10)
        with pytest.raises(ToolError, match="does not hold"):
            mgr.release("res", "b")
        assert mgr.holder("res") == "a"  # nothing was freed

    def test_hold_context_manager_releases_on_exception(self):
        mgr = LeaseManager()
        with pytest.raises(RuntimeError), mgr.hold("res"):
            raise RuntimeError("boom mid-critical-section")
        assert mgr.holder("res") is None

    def test_owner_ids_are_unique_per_call(self):
        assert new_owner() != new_owner()


class TestFsIntegration:
    def test_write_file_waits_for_a_pre_held_lease(self, tmp_path):
        # the exact key format the tool itself uses (shared helper)
        target = tmp_path / "f.txt"
        target.write_text("original")
        ctx = ToolContext(cwd=tmp_path)
        ctx.leases.acquire(file_lease_key(target.resolve()), "sibling",
                           ttl=10)

        result = {}

        def write():
            try:
                result["out"] = WriteFile().run(
                    {"path": "f.txt", "content": "new"}, ctx)
            except BaseException as exc:  # noqa: BLE001
                result["error"] = exc

        thread = threading.Thread(target=write)
        thread.start()
        time.sleep(0.05)
        assert "out" not in result  # blocked behind the sibling...
        ctx.leases.release(file_lease_key(target.resolve()), "sibling")
        thread.join(timeout=2)
        assert "edited" not in str(result) and result.get("out", "").startswith(("created", "overwrote"))
        assert target.read_text() == "new"

    def test_conflicting_edits_serialize_via_shared_context(self, tmp_path):
        # two threads, one ToolContext, same file: both edits land --
        # serialized by the lease rather than interleaved into loss
        target = tmp_path / "f.txt"
        target.write_text("AAA BBB CCC")
        ctx = ToolContext(cwd=tmp_path)
        edit = EditFile()

        def swap(frm: str, to: str) -> None:
            edit.run({"path": "f.txt", "old_string": frm, "to": to}
                     | {"new_string": to}, ctx)

        threads = [threading.Thread(target=swap, args=("BBB", "XXX")),
                   threading.Thread(target=swap, args=("CCC", "YYY"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        final = target.read_text()
        assert "XXX" in final and "YYY" in final  # neither edit was eaten

    def test_different_paths_never_block_each_other(self, tmp_path):
        # reads-and-writes-of-DIFFERENT resources stay fully parallel:
        # keys are per-path, so no global lock sneaks in
        ctx = ToolContext(cwd=tmp_path)
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        started = threading.Event()

        def slow_hold():
            with ctx.leases.hold(file_lease_key((tmp_path / "a.txt").resolve())):
                started.set()
                time.sleep(0.15)

        blocker = threading.Thread(target=slow_hold)
        blocker.start()
        started.wait(timeout=1)
        # while a.txt is held, b.txt writes proceed immediately
        out = WriteFile().run({"path": "b.txt", "content": "B"}, ctx)
        blocker.join(timeout=2)
        assert out.startswith("overwrote")
