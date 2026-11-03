"""Lost replies, queued dependents and host crashes must not replay robot effects."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pyrualean.arms import ArmResult, ArmToolkit
from pyrualean.libero import LiberoRobot
from tests.test_interaction_contracts import ProtocolBackend

MOVE = "robo.move_to([0.05, -0.10, 0.6])"
ONCE = "once('move-1', robo.move_to, [0.05, -0.10, 0.6])"


def toolkit(tmp_path, backend=None, *, durable=False, resume=False, **kwargs):
    backend = backend or ProtocolBackend()
    kit = ArmToolkit(LiberoRobot(backend), arm="cells", workspace=tmp_path / "workspace",
                     images="none", feedback="pure", resume=resume,
                     execution_state_path=tmp_path / "execution.sqlite3" if durable else None,
                     **kwargs)
    return backend, kit


def call(kit, code, identity, *, token=None, context=None):
    return kit.execute_tool("python", dict(code=code, request_id=identity, resume_token=token),
                            trace_context=context)


@pytest.mark.parametrize("tail", ["", "\nmissing_name"])
def test_lost_reply_is_cached_including_failure_without_budget_or_action_replay(tmp_path, tail):
    backend, kit = toolkit(tmp_path, max_turns=1)
    first = call(kit, MOVE + tail, "original")
    second = call(kit, MOVE + tail, "original")
    assert len(backend.contexts) == kit._cell_index == len(kit.turns) == 1
    assert first.is_error == second.is_error == bool(tail)
    assert second.result["execution"]["replayed"]
    assert second.result["execution"]["request_id"] == "original"
    assert second.result["execution"]["resume_token"] == first.result["execution"]["resume_token"]
    second.result["action_receipts"].clear()
    assert call(kit, MOVE + tail, "original").result["action_receipts"]


def test_provider_call_identity_works_without_tracing(tmp_path):
    backend, kit = toolkit(tmp_path)
    context = dict(provider_call_id="provider-one", decision_id="decision-one")
    for _ in range(2):
        kit.execute_tool("python", {"code": MOVE + "\n1/0"}, trace_context=context)
    assert len(backend.contexts) == 1


def test_same_id_with_modified_code_is_rejected(tmp_path):
    backend, kit = toolkit(tmp_path)
    call(kit, MOVE, "original")
    result = call(kit, "robo.release()", "original")
    assert result.is_error and result.result["skipped"]
    assert len(backend.contexts) == kit._cell_index == 1


def test_same_decision_and_prequeued_request_are_invalid_after_failure(tmp_path):
    backend, kit = toolkit(tmp_path)
    queued = kit.submission_context()
    first = call(kit, MOVE + "\n1/0", "first", context={"decision_id": "d1"})
    token = first.result["execution"]["resume_token"]
    assert call(kit, "robo.release()", "queued", token=token, context=queued).result["skipped"]
    assert call(kit, "robo.release()", "same-decision", token=token,
                context={"decision_id": "d1"}).result["skipped"]
    assert len(backend.contexts) == 1
    repaired = call(kit, "robo.release()", "repair", token=token, context={"decision_id": "d2"})
    assert not repaired.is_error and len(backend.contexts) == 2


def test_mcp_requires_identity_and_review_even_for_sequential_calls(tmp_path):
    backend, kit = toolkit(tmp_path)
    context = dict(transport="mcp")
    assert kit.execute_tool("python", {"code": MOVE}, trace_context=context).is_error
    assert not backend.contexts
    failed = call(kit, MOVE + "\n1/0", "first", context=context)
    token = failed.result["execution"]["resume_token"]
    for missing in (None, "not-the-token"):
        assert call(kit, "robo.release()", "repair", token=missing, context=context).result["skipped"]
    assert len(backend.contexts) == kit._cell_index == 1
    assert not call(kit, "robo.release()", "repair", token=token, context=context).is_error
    assert len(backend.contexts) == 2


def test_repaired_whole_cell_skips_named_completed_prefix(tmp_path):
    backend, kit = toolkit(tmp_path)
    failed = call(kit, ONCE + "\nmissing_name", "first")
    assert failed.is_error and len(backend.contexts) == 1
    repaired = call(kit, "missing_name = 7\nr = " + ONCE + "\nprint(r['value']['reached'])", "repair")
    assert not repaired.is_error and repaired.result["stdout"] == "True\n"
    assert len(backend.contexts) == 1
    assert repaired.result["operation"]["replayed"]


def test_operation_ids_distinguish_intentional_repetition_and_argument_conflict(tmp_path):
    backend, kit = toolkit(tmp_path)
    call(kit, ONCE, "one")
    assert call(kit, ONCE.replace("0.05", "0.06"), "conflict").is_error
    assert len(backend.contexts) == 1
    assert not call(kit, ONCE.replace("move-1", "move-2"), "intentional").is_error
    assert len(backend.contexts) == 2


def test_unknown_named_operation_is_not_replayed_after_reconciliation(tmp_path):
    backend, kit = toolkit(tmp_path)
    backend.next_status = "unknown"
    failed = call(kit, ONCE, "first")
    assert failed.result["execution_paused"]
    identity = failed.result["action_receipts"][0]["action_id"]
    assert call(kit, ONCE, "retry").result["execution_paused"]
    assert len(backend.contexts) == 1
    backend.receipts[identity].update(status="partial", effect_known=True, execution_complete=True)
    call(kit, "robo.reconcile()", "query")
    assert call(kit, ONCE, "after-query").result["execution_paused"]
    record = kit._once("move-1")
    assert record["receipts"][0]["status"] == "partial"
    assert len(backend.contexts) == 1


def test_restart_restores_requests_operations_and_budget_but_not_python(tmp_path):
    backend, kit = toolkit(tmp_path, durable=True, max_turns=4)
    first = call(kit, "saved = 7\n" + ONCE + "\n1/0", "first")
    kit.close()
    _, restored = toolkit(tmp_path, backend, durable=True, resume=True, max_turns=4)
    token = restored.resume_state()["resume_token"]
    assert token != first.result["execution"]["resume_token"]
    assert restored._cell_index == 1 and "saved" not in restored._globals
    cached = call(restored, "saved = 7\n" + ONCE + "\n1/0", "first")
    assert cached.result["execution"]["replayed"] and len(backend.contexts) == 1
    assert call(restored, ONCE, "repair").result["skipped"]
    assert not call(restored, ONCE, "repair", token=token).is_error
    assert restored._cell_index == 2 and len(backend.contexts) == 1
    assert call(restored, "saved", "check-variable", token=token).result["failure"]["type"] == "NameError"
    restored.close()


def test_restart_unknown_requires_exact_receipt_before_new_motion(tmp_path):
    backend, kit = toolkit(tmp_path, durable=True)
    backend.next_status = "unknown"
    first = call(kit, MOVE, "first")
    identity = first.result["action_receipts"][0]["action_id"]
    kit.close()
    _, restored = toolkit(tmp_path, backend, durable=True, resume=True)
    def resumed(code, identity):
        return call(restored, code, identity, token=restored.resume_state()["resume_token"])
    assert resumed(MOVE, "blocked").result["execution_paused"]
    assert len(backend.contexts) == 1
    backend.receipts[identity].update(status="partial", effect_known=True, execution_complete=True)
    assert not resumed("robo.reconcile()", "query").is_error
    assert not resumed(MOVE, "remaining").is_error
    assert len(backend.contexts) == 2
    restored.close()


def test_journal_refuses_second_writer_wrong_session_and_budget_changes(tmp_path):
    backend, kit = toolkit(tmp_path, durable=True, max_turns=2)
    with pytest.raises(BlockingIOError):
        toolkit(tmp_path, backend, durable=True, resume=True, max_turns=2)
    kit.close()
    other = ProtocolBackend()
    other.session_id = "different-scene"
    with pytest.raises(ValueError, match="different backend"):
        toolkit(tmp_path, other, durable=True, resume=True, max_turns=2)
    with pytest.raises(ValueError, match="different backend"):
        toolkit(tmp_path, backend, durable=True, resume=True, max_turns=3)
    _, restored = toolkit(tmp_path, backend, durable=True, resume=True, max_turns=2)
    restored.close()


def test_crash_after_effect_before_reply_leaves_durable_unknown_and_no_replay(tmp_path):
    script = '''
import os, sys
from pathlib import Path
from pyrualean.arms import ArmToolkit
from pyrualean.libero import LiberoRobot
from tests.test_interaction_contracts import ProtocolBackend
root = Path(sys.argv[1])
class Backend(ProtocolBackend):
    def execute_tool_with_context(self, name, arguments, context):
        (root / "effect.txt").write_text(context["action_id"])
        os._exit(77)
kit = ArmToolkit(LiberoRobot(Backend()), arm="cells", workspace=root/"workspace",
                 execution_state_path=root/"execution.sqlite3", images="none", feedback="pure",
                 max_turns=2)
kit.execute_tool("python", {"code": "once('move-1', robo.move_to, [0.05,-0.1,0.6])",
                            "request_id": "crashing"})
'''
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root)]))
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env,
                           capture_output=True, text=True, timeout=10)
    assert child.returncode == 77, child.stderr
    identity = (tmp_path / "effect.txt").read_text()
    backend, restored = toolkit(tmp_path, durable=True, resume=True, max_turns=2)
    report = restored.resume_state()
    assert report["cells_used"] == 1
    assert report["handoff"]["unresolved_actions"][0]["action_id"] == identity
    assert report["operations"][0]["action_ids"] == [identity]
    cached = call(restored, "once('move-1', robo.move_to, [0.05,-0.1,0.6])", "crashing")
    assert cached.is_error and cached.result["skipped"] and not backend.contexts
    assert call(restored, MOVE, "retry", token=report["resume_token"]).result["execution_paused"]
    assert not backend.contexts
    restored.close()


def test_feedback_compaction_preserves_review_token_and_request_identity():
    token = "t" * 32
    result = ArmResult(dict(stdout="x" * 80000, failure=dict(message="y" * 80000),
                            execution=dict(resume_token=token, request_id="request-42")))
    sent = json.loads(result.content_blocks[0]["text"])
    assert sent["execution"]["resume_token"] == token
    assert sent["execution"]["request_id"] == "request-42"


def test_restart_preserves_exhausted_call_budget(tmp_path):
    backend, kit = toolkit(tmp_path, durable=True, max_turns=1)
    call(kit, MOVE, "first")
    kit.close()
    _, restored = toolkit(tmp_path, backend, durable=True, resume=True, max_turns=1)
    result = call(restored, MOVE, "second", token=restored.resume_state()["resume_token"])
    assert result.is_error and len(backend.contexts) == 1 and restored._cell_index == 1
    restored.close()


def test_restart_keeps_wall_deadline_including_downtime(tmp_path, monkeypatch):
    import pyrualean.arms as arms

    monkeypatch.setattr(arms.time, "time", lambda: 1000.0)
    backend, kit = toolkit(tmp_path, durable=True, program_timeout_s=10)
    call(kit, MOVE, "first")
    kit.close()
    monkeypatch.setattr(arms.time, "time", lambda: 1011.0)
    _, restored = toolkit(tmp_path, backend, durable=True, resume=True, program_timeout_s=10)
    result = call(restored, MOVE, "second", token=restored.resume_state()["resume_token"])
    assert result.is_error and restored.robo._stop_reason == "timeout"
    assert restored.resume_state()["deadline"] == 1010.0 and len(backend.contexts) == 1
    restored.close()


def test_program_once_interruption_does_not_escape_or_replay(tmp_path):
    backend = ProtocolBackend()
    backend.next_status = "unknown"
    kit = ArmToolkit(LiberoRobot(backend), arm="program", workspace=tmp_path / "ws", images="none")
    code = "def run(robo):\n    " + ONCE
    for identity in ("first", "second"):
        result = kit.execute_tool("run_program", dict(code=code, request_id=identity))
        assert result.result["execution_paused"]
    assert len(backend.contexts) == 1


def test_cached_request_preserves_images_and_marks_them_historical(tmp_path):
    backend = ProtocolBackend()
    kit = ArmToolkit(LiberoRobot(backend), arm="cells", workspace=tmp_path / "ws",
                     images="on-motion")
    first = call(kit, MOVE, "first")
    second = call(kit, MOVE, "first")
    assert first.images and first.images == second.images
    assert first.result["observation"] == second.result["observation"]
    assert second.result["execution"]["replayed"] and len(backend.contexts) == 1


def test_restart_preserves_terminal_finish(tmp_path):
    backend, kit = toolkit(tmp_path, durable=True)
    kit.execute_tool("finish", dict(request_id="end", status="failure", summary="stop"))
    kit.close()
    _, restored = toolkit(tmp_path, backend, durable=True, resume=True)
    result = call(restored, MOVE, "late", token=restored.resume_state()["resume_token"])
    assert result.result["episode_finished"] and not backend.contexts
    restored.close()


def test_journal_write_failure_never_dispatches_unrecorded_motion(tmp_path, monkeypatch):
    backend, kit = toolkit(tmp_path, durable=True)
    save = kit._journal.put

    def fail_intent(kind, key, value):
        if kind == "action":
            kit._journal.failed = True
            raise OSError("simulated storage failure")
        return save(kind, key, value)

    monkeypatch.setattr(kit._journal, "put", fail_intent)
    with pytest.raises(RuntimeError, match="journal unavailable"):
        call(kit, MOVE, "first")
    assert not backend.contexts
    with pytest.raises(RuntimeError, match="journal unavailable"):
        call(kit, MOVE, "second")
    assert not backend.contexts
    kit.close()


def test_concurrent_retries_share_one_execution(tmp_path, monkeypatch):
    backend, kit = toolkit(tmp_path)
    started, release = threading.Event(), threading.Event()
    execute = backend.execute_tool_with_context

    def waiting(*args):
        started.set()
        assert release.wait(5)
        return execute(*args)

    monkeypatch.setattr(backend, "execute_tool_with_context", waiting)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(call, kit, MOVE, "same")
        try:
            assert started.wait(3)
            assert kit.close() is False  # shutdown must not hang or release a live writer's lease
            second = pool.submit(call, kit, MOVE, "same")
        finally:
            release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert all(not r.is_error for r in results)
    assert len(backend.contexts) == kit._cell_index == 1
    assert sum(r.result["execution"]["replayed"] for r in results) == 1
    assert kit.close() is True


def test_restart_does_not_accept_completion_flags_on_nonterminal_receipt(tmp_path):
    backend, kit = toolkit(tmp_path, durable=True)
    backend.next_status = "running"  # this fixture sets both flags true for non-unknown statuses
    first = call(kit, MOVE, "first")
    assert first.result["execution_paused"]
    receipt = first.result["action_receipts"][0]
    assert receipt["effect_known"] and receipt["execution_complete"]
    kit.close()
    _, restored = toolkit(tmp_path, backend, durable=True, resume=True)
    assert restored.robo._pending_actions == {receipt["action_id"]}
    blocked = call(restored, MOVE, "second", token=restored.resume_state()["resume_token"])
    assert blocked.result["execution_paused"] and len(backend.contexts) == 1
    restored.close()


def test_delayed_old_decision_stays_invalid_after_successful_repairs(tmp_path):
    backend, kit = toolkit(tmp_path)
    for decision in ("old-1", "old-2"):
        failed = call(kit, "1/0", decision, context={"decision_id": decision})
        token = failed.result["execution"]["resume_token"]
        assert not call(kit, "1+1", decision+"-repair", token=token,
                        context={"decision_id": decision+"-repair"}).is_error
    assert call(kit, MOVE, "delayed", token=token,
                context={"decision_id": "old-1"}).result["skipped"]
    assert call(kit, MOVE, "late-mcp", context={"transport": "mcp"}).result["skipped"]
    assert not backend.contexts
