"""Policy results survive the provenance-heavy feedback seen in organize_table."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from pyrualean.arms import MAX_TEXT_CHARS, ArmResult, ArmToolkit
from pyrualean.robodojo import RobodojoRobot
from pyrualean.robodojo_xpl_rgb import ObjectSight, RobodojoOfficialColourRobot
from tests.test_perception_api import CapturedBackend


@pytest.fixture
def recorded_locates():
    path = Path(__file__).with_name("fixtures") / "organize_table_locates.json"
    return json.loads(path.read_text())["cells"]


def _rich_calls(records):
    return [{key: record[key] for key in ("tool", "action_id", "observation_id", "kwargs",
                                          "ok", "error")} | deepcopy(record["summary"])
            for record in records]


@pytest.mark.parametrize("cell,count", [("1", 6), ("3", 4)])
def test_recorded_locates_retain_normal_stdout(recorded_locates, cell, count):
    records = recorded_locates[cell]
    assert len(records) == count
    stdout = "".join(f"{r['kwargs']['text_prompt']}: {r['summary'].get('center_xyz')}\n"
                     for r in records)
    result = dict(cell=int(cell), stdout=stdout, stderr="", primitive_calls=_rich_calls(records),
                  supervision=dict(mode="pause", paused=False, reason=None))
    assert len(json.dumps(result, indent=2)) > MAX_TEXT_CHARS
    text = ArmResult(result).content_blocks[0]["text"]
    sent = json.loads(text)
    assert len(text) <= MAX_TEXT_CHARS and sent["stdout"] == stdout
    assert len(sent["primitive_calls"]) == count
    for original, call in zip(records, sent["primitive_calls"], strict=True):
        assert call.get("center_xyz") == original["summary"].get("center_xyz")
        assert call["provenance"]["current_at_return"] is False
        assert call["provenance"]["aligned"] is False
        assert call["provenance"]["geometry"]["depth_is_measured"] is False
        assert call["provenance"]["geometry"]["assumptions"]
        assert "images" not in call["provenance"]
        assert "images" in original["summary"]["provenance"]


def test_six_printed_typed_results_fit_with_rich_feedback(recorded_locates):
    records = recorded_locates["1"]
    sightings = []
    for record in records:
        s = record["summary"]
        sightings.append(ObjectSight(
            found=s["found"], center_xyz=s.get("center_xyz"), top_z=s.get("top_z"), height_m=0.02,
            length_m=0.1, width_m=s.get("width_m"), long_axis_deg=0., grasp_yaw=s.get("grasp_yaw"),
            score=s["score"], centroid=s["centroid_rc"], area_px=100,
            camera="cam_head", note="height inferred from upright shape", views=s["views"],
            provenance=deepcopy(s["provenance"]),
        ))
    stdout = "\n".join(map(repr, sightings))
    result = ArmResult(dict(stdout=stdout, primitive_calls=_rich_calls(records),
                            supervision=dict(mode="pause", paused=False, reason=None)))
    sent = json.loads(result.content_blocks[0]["text"])
    assert sent["stdout"] == stdout
    assert "center_xyz=" in sent["stdout"] and "height inferred" in sent["stdout"]
    assert "sensor_artifact" not in sent["stdout"]
    assert all(item.provenance["images"]["cam_head"]["sensor_artifact"] == "sensors.npz"
               for item in sightings)


@pytest.mark.parametrize("robot_cls", [RobodojoRobot, RobodojoOfficialColourRobot])
def test_default_display_and_rich_feedback_preserve_full_audit_and_currency(robot_cls, tmp_path):
    robot = robot_cls(CapturedBackend(), primitives="no-vla")
    toolkit = ArmToolkit(robot, arm="cells", workspace=tmp_path, images="none")
    output = toolkit.execute_tool("python", {"code": "target = robo.locate('the green bottle')\n"
                                             "print(robo.require_current(target))"})
    target = toolkit._globals["target"]
    assert robot.require_current(target) is target
    call = next(c for c in output.result["primitive_calls"] if c["tool"] == "locate")
    assert "images" not in call["provenance"]
    audit = next(r for r in robot._ledger.records if r.tool == "locate")
    assert audit.summary["provenance"]["images"] == target.provenance["images"]
    assert "center_xyz=" in output.result["stdout"]
    assert "current_at_return" in output.result["stdout"]
    assert "calibration_source_id" not in output.result["stdout"]


@pytest.mark.parametrize("status", ["unknown", "partial"])
@pytest.mark.parametrize("noise", ["感知🙂", "\x01"])
def test_large_output_retains_normal_result_and_failure_handoff(status, noise):
    receipt = dict(action_id="action-7", status=status, effect_known=False,
                   execution_complete=False, stop_confirmed=False, error=noise * 20000)
    result = dict(
        cell=4, stdout="mouse: xyz=(0.34, -0.06, 0.77)\n", stderr=noise * 30000,
        exception="TimeoutError: motion outcome requires reconciliation",
        action_receipts=[receipt],
        handoff=dict(allowed_next="observe_or_reconcile", new_motion_allowed=False,
                     unresolved_actions=[receipt], last_action=receipt),
        supervision=dict(mode="pause", paused=True, reason="support remains unknown",
                         blocked={"supported": {"value": "unknown", "reason": "no fresh evidence"}},
                         reports=[{"large": noise * 20000}]),
        failure=dict(kind="tool", type="TimeoutError", message=noise * 20000),
        recovery=dict(next_action="observe_and_reconcile", automatic_replay=False,
                      scene_rolled_back=False, state={"large": noise * 20000}),
    )
    text = ArmResult(result).content_blocks[0]["text"]
    sent = json.loads(text)
    assert len(text) <= MAX_TEXT_CHARS and sent["output_truncated"]
    assert sent["stdout"] == result["stdout"] and sent["exception"] == result["exception"]
    assert sent["action_receipts"][0]["status"] == status
    assert sent["action_receipts"][0]["action_id"] == "action-7"
    assert sent["handoff"]["new_motion_allowed"] is False
    assert sent["handoff"]["allowed_next"] == "observe_or_reconcile"
    assert sent["supervision"]["reason"] == "support remains unknown"
    assert sent["supervision"]["blocked"]["supported"]["value"] == "unknown"
    assert sent["recovery"]["next_action"] == "observe_and_reconcile"
    assert sent["recovery"]["automatic_replay"] is False


def test_large_successful_audit_is_removed_before_short_stdout():
    result = dict(stdout="mouse is reachable\n", stderr="", state={"large": "state" * 10000},
                  primitive_calls=[dict(tool="locate", ok=True, details="audit" * 10000)],
                  supervision=dict(paused=False, reason=None))
    sent = json.loads(ArmResult(result).content_blocks[0]["text"])
    assert sent["stdout"] == result["stdout"]
    assert sent["primitive_call_count"] == 1


def test_long_action_history_retains_unresolved_counts_and_no_replay():
    receipts = [dict(action_id=f"action-{i}", status="unknown", error="x" * 1000)
                for i in range(80)]
    result = dict(stdout="inspect first\n", action_receipts=receipts,
                  handoff=dict(unresolved_actions=receipts, new_motion_allowed=False,
                               allowed_next="observe_or_reconcile", automatic_replay=False))
    text = ArmResult(result).content_blocks[0]["text"]
    sent = json.loads(text)
    assert len(text) <= MAX_TEXT_CHARS and sent["stdout"] == result["stdout"]
    assert sent["action_receipt_counts"]["unknown"] == 80
    assert sent["handoff"]["unresolved_action_count"] == 80
    assert sent["handoff"]["new_motion_allowed"] is False
    assert sent["handoff"]["automatic_replay"] is False


def test_extreme_diagnostic_keys_cannot_displace_controls_or_exceed_limit():
    junk = {("huge-key-" + str(i)) * 3000: {f"detail{j}": "x" * 2000 for j in range(40)}
            for i in range(40)}
    receipt = dict(action_id="known-action-identity", status="partial", effect_known=False)
    result = dict(junk, stdout="mouse target: (0.34, -0.06, 0.77)\n",
                  exception="TimeoutError: reconcile first", action_receipts=[receipt],
                  handoff=dict(junk, new_motion_allowed=False, allowed_next="observe_or_reconcile",
                               unresolved_actions=[receipt], automatic_replay=False),
                  failure=dict(junk, type="TimeoutError", kind="tool"),
                  supervision=dict(junk, paused=True, reason="support unknown"))
    text = ArmResult(result).content_blocks[0]["text"]
    sent = json.loads(text)
    assert len(text) <= MAX_TEXT_CHARS
    assert sent["stdout"] == result["stdout"] and sent["exception"] == result["exception"]
    assert sent["handoff"]["new_motion_allowed"] is False
    assert sent["handoff"]["automatic_replay"] is False
    assert sent["action_receipt_counts"]["partial"] == 1
    assert sent["action_receipts"][0]["action_id"] == "known-action-identity"
    assert sent["failure"]["type"] == "TimeoutError"
    assert sent["supervision"]["paused"] is True
    assert sent["supervision"]["reason"] == "support unknown"


def test_early_unknown_receipt_is_not_hidden_by_later_completed_calls():
    receipts = [dict(action_id="unresolved", status="unknown", error="x" * 30000)]
    receipts.extend(dict(action_id=f"later-{i}", status="completed") for i in range(100))
    text = ArmResult(dict(action_receipts=receipts)).content_blocks[0]["text"]
    sent = json.loads(text)
    assert len(text) <= MAX_TEXT_CHARS
    assert sent["action_receipts"][0]["action_id"] == "unresolved"
    assert sent["action_receipts"][0]["status"] == "unknown"
    assert sent["action_receipt_counts"]["unknown"] == 1


@pytest.mark.parametrize("extra_output", ["", "\x01感知" * 20000], ids=["normal", "oversized"])
def test_recorded_partial_motion_preserves_endpoint_and_query_identity(extra_output):
    fixture = json.loads((Path(__file__).with_name("fixtures") /
                          "organize_table_partial_motion.json").read_text())
    result = fixture["result"]
    result["stderr"] = extra_output
    original = deepcopy(result)
    text = ArmResult(result).content_blocks[0]["text"]
    sent = json.loads(text)
    assert len(text) <= MAX_TEXT_CHARS
    assert sent["stdout"] == original["stdout"]
    assert sent["supervision"]["reason"].startswith("motion_incomplete:")
    action = sent["supervision"]["last_action"]
    assert action["action_id"] == fixture["expected_action_id"]
    assert action["receipt"]["status"] == "partial"
    for key in ("reached", "dist_to_target_m", "orientation_error_deg", "final_xyz", "blocked"):
        assert action["summary"][key] == fixture["expected_endpoint"][key]
    assert "state" not in sent["supervision"]["observation"]
    assert "native_receipts" not in action["receipt"]
    assert result == original  # The full host audit was not mutated.


def test_oversized_feedback_never_emits_a_truncated_query_id():
    identity = "session-" + "s" * 32 + ":action:" + "a" * 32
    receipt = dict(error="x" * 30000, action_id=identity, status="unknown")
    result = dict(action_receipts=[receipt], handoff=dict(last_action=receipt,
                  unresolved_actions=[receipt], new_motion_allowed=False), stderr="x" * 30000)
    sent = json.loads(ArmResult(result).content_blocks[0]["text"])
    assert sent["action_receipts"][0]["action_id"] == identity
    assert sent["handoff"]["last_action"]["action_id"] == identity


@pytest.mark.parametrize("extra_output", ["", "\x01感知" * 20000], ids=["normal", "oversized"])
def test_recorded_assertion_feedback_does_not_reinject_native_audit(extra_output):
    """The real cell 30 used to pass 14,106 chars to a ~12k transport cut."""
    fixture = json.loads((Path(__file__).with_name("fixtures") /
                          "organize_table_assertion_feedback.json").read_text())
    result = fixture["result"]
    result["stderr"] = extra_output
    original = deepcopy(result)
    text = ArmResult(result).content_blocks[0]["text"]
    sent = json.loads(text)
    assert MAX_TEXT_CHARS < 12000 and len(text) <= MAX_TEXT_CHARS
    assert "native_receipts" not in text and "native_action_ids" not in text
    assert sent["stdout"] == original["stdout"]
    assert sent["failure"]["type"] == "AssertionError"
    motion = sent["recovery"]["last_motion"]
    expected = original["recovery"]["last_motion"]
    assert motion["action_id"] == expected["action_id"]
    assert motion["summary"] == expected["summary"]
    assert motion["receipt"]["status"] == "partial"
    assert motion["receipt"]["effect_known"] is True
    assert motion["receipt"]["execution_complete"] is True
    assert sent["recovery"]["automatic_replay"] is False
    assert sent["recovery"]["scene_rolled_back"] is False
    if extra_output:
        assert sent["output_truncated"] is True
        assert "chars omitted" in sent["stderr"]
    else:
        assert len(text) < 6000 < fixture["original_text_chars"]
        assert not sent.get("output_truncated", False)
    assert result == original


def test_normal_feedback_omits_repeated_audit_and_does_not_invent_permission():
    receipt = dict(session_id="session", action_id="session:action:one", tool="move_pose",
                   status="completed", effect_known=True, execution_complete=True,
                   observation_id="capture-2", last_native_action_ids=["native-1"])
    frame = dict(name="wrist.png", source="camera", source_id="camera-source-1",
                 source_observation_id="capture-2", captured_at=1234.5, aligned=True,
                 source_freshness="captured", step_id=4, state_version=1)
    result = dict(stdout="Move(reached=True, final_xyz=(0.1, 0.2, 0.3))\n",
                  action_receipts=[receipt],
                  handoff=dict(schema="pyrualean.handoff.v1", session_id="session",
                               observation_id="capture-2", last_action=receipt,
                               unresolved_actions=[], next_owner="program_or_planner",
                               allowed_next="check_prerequisites", new_motion_allowed=None,
                               stop_reason=None, automatic_replay=False, scene_rolled_back=False),
                  observation=dict(observation_id="host-turn-2", source_session_id="session",
                                   source_observation_id="capture-2", step_id=4, state_version=1,
                                   actions=dict(action_ids=[receipt["action_id"]], attempted=1),
                                   prior_targets_may_be_stale=True, images=[frame]))
    original = deepcopy(result)
    sent = json.loads(ArmResult(result).content_blocks[0]["text"])
    assert "handoff" not in sent and "new_motion_allowed" not in sent
    assert "actions" not in sent["observation"]
    assert "prior_targets_may_be_stale" not in sent["observation"]
    assert sent["observation"]["images"] == [frame]
    assert sent["observation"]["source_observation_id"] == "capture-2"
    assert sent["observation"]["state_version"] == 1
    assert sent["action_receipts"][0]["action_id"] == receipt["action_id"]
    assert sent["action_receipts"][0]["status"] == "completed"
    assert "last_native_action_ids" not in sent["action_receipts"][0]
    assert result == original


@pytest.mark.parametrize("stop_reason", [None, "truncated"])
def test_pending_or_stopped_feedback_preserves_execution_controls(stop_reason):
    receipt = dict(action_id="session:action:pending", status="unknown",
                   effect_known=False, execution_complete=False, stop_confirmed=False)
    handoff = dict(unresolved_actions=[] if stop_reason else [receipt], last_action=receipt,
                   allowed_next="finish" if stop_reason else "observe_or_reconcile",
                   new_motion_allowed=False, stop_reason=stop_reason,
                   automatic_replay=False, scene_rolled_back=False)
    result = dict(handoff=handoff, action_receipts=[receipt], stdout="inspect before retry\n")
    sent = json.loads(ArmResult(result).content_blocks[0]["text"])
    assert sent["handoff"] == handoff
    assert sent["action_receipts"] == [receipt]
