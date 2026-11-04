"""Protocol behavior independent of robot geometry or a particular task."""

from copy import deepcopy

from pyrualean.operations import OperationAdapter, OperationRegistry, feedback


def registry():
    domain = dict(state="paused", sent=3, calls=0)

    def describe(operation_id, result=None):
        if operation_id != "upload":
            return None
        return result or {}, feedback(
            state=domain["state"], phase={"name": "send"}, completed=[], next_phase=None,
            used=domain["sent"], limit=5, unit="chunk", verification="remote file exists")

    def resume(operation_id, allowance):
        domain["calls"] += 1
        domain.update(sent=5, state="completed")
        return {"sent_now": 2}

    operations = OperationRegistry("session-a")
    operations.register("file_transfer", OperationAdapter(describe, resume))
    return domain, operations, operations.reference("file_transfer", "upload")


def test_non_robot_adapter_keeps_its_own_units_and_remaining_progress():
    domain, operations, ref = registry()
    initial = operations.status(ref)
    assert initial["operation"]["budget"]["unit"] == "chunk"
    result = operations.resume(ref)
    assert result["sent_now"] == 2 and domain["calls"] == 1
    assert result["operation"]["execution_complete"]
    assert result["operation"]["verification"]["status"] == "required"
    assert result["operation"]["continuation"]["action"] == "observe"
    operations.resume(ref)
    assert domain["calls"] == 1


def test_stale_or_unknown_reference_never_creates_or_dispatches():
    domain, operations, ref = registry()
    for bad in (None, {}, dict(ref, session_id="other"), dict(ref, id="missing"),
                dict(ref, kind="other"), dict(ref, id=[])):
        result = operations.resume(bad)
        assert result["status"] == "rejected"
        assert not result["operation"]["continuation"]["allowed"]
    assert domain["calls"] == 0


def test_unknown_receipt_blocks_resume_even_if_adapter_reports_a_known_endpoint():
    domain, operations, ref = registry()
    receipt = dict(action_id="dispatch-1", status="unknown", effect_known=False,
                   execution_complete=False, observation_id=None)
    original = deepcopy(receipt)
    operations.record_receipt("file_transfer", "upload", receipt)
    receipt["effect_known"] = True
    result = operations.resume(ref)["operation"]
    assert result["state"] == "unknown" and result["effect"] == "unknown"
    assert result["receipt"] == original and domain["calls"] == 0
    result["receipt"]["effect_known"] = True
    assert operations.status(ref)["operation"]["receipt"] == original


def test_ready_phase_requires_explicit_advance_instead_of_resume():
    domain, operations, ref = registry()
    domain["state"] = "ready"
    assert operations.resume(ref)["operation"]["continuation"]["action"] == "advance"
    assert domain["calls"] == 0
