"""Execution guarantees of the primitive-level visual supervisor, without a GPU."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.supervision import (
    MonitorConfig,
    ValenClient,
    VisualSupervisor,
    decode_answers,
)

MOVE = "robo.move_to([0.05, -0.10, 0.6])"
WATCH = (
    "watch('transport', {'held': 'Is the black bowl held by the gripper?'}, "
    "alerts=[{'name': 'dropped', 'when': {'held': 'no'}}])\n"
)
CRITICAL_WATCH = WATCH.replace("alerts=", "critical=['held'], alerts=")


def answer(value="yes"):
    return {
        "answers": {
            "held": {
                "type": "choice",
                "probabilities": {k: 1.0 if k == value else 0.0 for k in ("yes", "no", "unknown")},
            }
        }
    }


class ScriptedValen:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.packets = []

    def predict(self, packet):
        self.packets.append(packet)
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return result


def setup(robo, tmp_path, outputs, **options):
    config = MonitorConfig(url="http://127.0.0.1:8001/predict", **{"mode": "pause", **options})
    client = ScriptedValen(outputs)
    supervisor = VisualSupervisor(config, tmp_path / "audit", client)
    toolkit = ArmToolkit(
        robo,
        arm="cells",
        workspace=tmp_path / "ws",
        supervisor=supervisor,
        images="none",
        feedback="pure",
    )
    return toolkit, supervisor, client


@pytest.fixture
def snapshot_robo(backend, monkeypatch):
    """A test sensor protocol: new read-only captures do not advance physics."""
    from pyrualean import LiberoRobot

    original_view = backend._view

    def metadata(step=-1):
        index = backend.steps[step]["step_idx"]
        source = f"test-sensor-frame-{index}"
        return dict(
            schema="pyrualean.observation.v1", session_id="test-sensor-session",
            observation_id=source, source_id=source, step=index,
            native_steps=backend.frames, captured_at=1000.0 + index,
            recorded_at=2000.0 + index, source_freshness="captured", aligned=True,
            images={camera: dict(source_id=source, artifact=f"{camera}_high.png", aligned=True)
                    for camera in LiberoRobot.CAMERAS},
        )

    def refresh():
        backend._record(command=None, result=None)
        return backend._view(-1)

    def view(step):
        return dict(original_view(step), observation=metadata(step))

    monkeypatch.setattr(backend, "session_id", "test-sensor-session", raising=False)
    monkeypatch.setattr(backend, "_view", view)
    monkeypatch.setattr(backend, "observation_metadata", metadata, raising=False)
    monkeypatch.setattr(backend, "refresh_observation", refresh, raising=False)
    return LiberoRobot(backend)


def test_alert_stops_remainder_preserves_episode_and_can_replan(robo, backend, tmp_path):
    tk, monitor, _ = setup(robo, tmp_path, [answer("no"), answer()])
    result = tk.execute_tool("python", {"code": WATCH + "saved = 42\n" + MOVE + "\n" + MOVE})
    assert result.result["execution_paused"]
    assert "episode_finished" not in result.result
    assert result.result["supervision"]["reports"][-1]["triggers"] == ["alert:dropped"]
    assert len(result.images) == len(robo.CAMERAS)
    assert len([r for r in robo._ledger if r.stateful]) == 1
    assert [r for r in robo._ledger if r.stateful][0].ok
    assert not backend.cancelled and not robo._stop_reason
    resumed = tk.execute_tool("python", {"code": WATCH + MOVE + "\nprint(saved)"})
    assert resumed.result["stdout"] == "42\n"
    assert not resumed.result["supervision"]["paused"]
    assert len([r for r in robo._ledger if r.stateful]) == 2
    events = [
        json.loads(line) for line in (monitor.directory / "events.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in events] == ["review", "clear"]


def test_normal_approach_needs_no_extra_pre_action_verification(snapshot_robo, backend, tmp_path):
    tk, monitor, client = setup(snapshot_robo, tmp_path, [answer()])
    result = tk.execute_tool("python", {"code": "obs = robo.snapshot()\n" + WATCH + MOVE})
    assert not result.result.get("execution_paused")
    assert not result.result.get("error")
    assert backend.frames > 0
    assert snapshot_robo._ledger.stateful_calls() == 1
    assert monitor.feedback()["verification_calls"] == 0
    assert len(client.packets) == 1  # Automatic endpoint review remains active.
    assert monitor.reports[-1]["action_id"] == snapshot_robo.action_receipt()["action_id"]


def test_plain_exception_handler_cannot_swallow_pause(robo, tmp_path):
    tk, _, client = setup(robo, tmp_path, [answer("no")])
    result = tk.execute_tool(
        "python", {"code": WATCH + f"try:\n    {MOVE}\nexcept Exception:\n    pass\n" + MOVE}
    )
    assert result.result["execution_paused"] and len(client.packets) == 1


def test_bare_except_cannot_resume_or_change_contract_in_same_cell(robo, tmp_path):
    tk, _, client = setup(robo, tmp_path, [answer("no")])
    result = tk.execute_tool(
        "python", {"code": WATCH + f"try:\n    {MOVE}\nexcept:\n    pass\n" + WATCH + MOVE}
    )
    assert result.result["execution_paused"] and len(client.packets) == 1


def test_pause_latch_blocks_motion_even_when_exception_was_caught(robo, tmp_path):
    tk, _, client = setup(robo, tmp_path, [answer("no")])
    result = tk.execute_tool(
        "python", {"code": WATCH + f"try:\n    {MOVE}\nexcept:\n    pass\n" + MOVE}
    )
    assert result.result["execution_paused"] and len(client.packets) == 1


def test_shadow_records_alert_but_executes_all_actions(robo, tmp_path):
    tk, monitor, _ = setup(robo, tmp_path, [answer("no"), answer()], mode="shadow")
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + MOVE})
    assert not result.result["supervision"]["paused"]
    assert [r["event"] for r in monitor.reports] == ["review", "clear"]
    assert len([r for r in robo._ledger if r.stateful]) == 2
    assert not result.images


def test_missing_contract_blocks_before_motion_and_resets_each_cell(robo, tmp_path):
    tk, _, client = setup(robo, tmp_path, [answer()])
    assert tk.execute_tool("python", {"code": MOVE}).result["execution_paused"]
    assert not client.packets and not robo._ledger.stateful_calls()
    tk.execute_tool("python", {"code": WATCH + MOVE})
    assert tk.execute_tool("python", {"code": MOVE}).result["execution_paused"]
    assert robo._ledger.stateful_calls() == 1


def test_phase_changes_do_not_reuse_old_evidence(robo, tmp_path):
    tk, monitor, client = setup(robo, tmp_path, [answer(), answer("no")])
    release = (
        "watch('released', {'held': 'Is the black bowl held by the gripper?'}, "
        "alerts=[{'name': 'still held', 'when': {'held': 'yes'}}])\n"
    )
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + release + MOVE})
    assert not result.result["supervision"]["paused"]
    assert [r["contract"]["phase"] for r in monitor.reports] == ["transport", "released"]
    context = json.loads(client.packets[-1]["state"]["messages"][0]["content"][0]["text"])
    assert len(context["frames"]) == 1


def test_critical_unknowns_pause_and_noncritical_unknowns_do_not(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [answer("unknown")] * 3)
    result = tk.execute_tool("python", {"code": CRITICAL_WATCH + "\n".join([MOVE] * 3)})
    assert result.result["supervision"]["actions"] == 2
    assert result.result["supervision"]["reports"][-1]["triggers"] == ["unknown:held"]
    result = tk.execute_tool(
        "python", {"code": "watch('approach', {'held': 'Is the bowl held?'}, critical=[])\n" + MOVE}
    )
    assert not result.result["supervision"]["paused"]


def test_critical_unknown_review_hands_back_without_forging_evidence(robo, tmp_path):
    tk, monitor, client = setup(robo, tmp_path, [answer("unknown")] * 4)
    first = tk.execute_tool("python", {"code": CRITICAL_WATCH + MOVE})
    assert not first.result["supervision"]["paused"]
    changed_phase = CRITICAL_WATCH.replace("'transport'", "'transport retry'")
    second = tk.execute_tool("python", {"code": changed_phase + MOVE + "\n" + MOVE})
    assert second.result["execution_paused"]
    assert second.result["supervision"]["reports"][-1]["triggers"] == ["unknown:held"]
    assert monitor.unknowns["held"] == 2
    assert robo._ledger.stateful_calls() == 2
    resumed = tk.execute_tool("python", {"code": CRITICAL_WATCH + MOVE + "\n" + MOVE})
    assert not resumed.result.get("execution_paused")
    assert len(client.packets) == robo._ledger.stateful_calls() == 4
    fact = monitor.read_evidence()["facts"]["held"]
    assert fact["value"] == "unknown" and fact["unknown_streak"] == 4
    assert fact["fresh"] and fact["observation"] is not None
    assert not monitor.reports[-1]["blocked"]


def test_ordinary_visual_questions_do_not_implicitly_make_unknown_critical(robo, tmp_path):
    tk, monitor, _ = setup(robo, tmp_path, [answer("unknown")] * 3)
    result = tk.execute_tool("python", {"code": WATCH + "\n".join([MOVE] * 3)})
    assert not result.result.get("execution_paused")
    assert monitor.contract["critical"] == []
    assert monitor.read_evidence()["facts"]["held"]["unknown_streak"] == 3
    assert robo._ledger.stateful_calls() == 3


@pytest.mark.parametrize("watch,value,trigger", [
    (CRITICAL_WATCH, "unknown", "unknown:held"),
    (WATCH, "no", "alert:dropped"),
    ("watch('inspect', {'held': 'Is the black bowl held by the gripper?'}, "
     "completion={'name': 'grasp', 'when': {'held': 'yes'}})\n",
     "yes", "completion_review:grasp"),
])
def test_new_question_version_does_not_inherit_review_notification(
    robo, tmp_path, watch, value, trigger
):
    tk, monitor, _ = setup(robo, tmp_path, [answer(value)] * 2, unknown_limit=1)
    first = tk.execute_tool("python", {"code": watch + MOVE})
    assert first.result["execution_paused"]
    revised = watch.replace("black bowl", "red cup")
    second = tk.execute_tool("python", {"code": revised + MOVE})
    assert second.result["execution_paused"]
    assert monitor.reports[-1]["triggers"] == [trigger]
    assert monitor.read_evidence()["facts"]["held"]["version"] == 2
    assert robo._ledger.stateful_calls() == 2


def test_matching_confirmation_window_survives_new_cell(robo, tmp_path):
    tk, monitor, _ = setup(robo, tmp_path, [answer(), answer()])
    watch = (
        "watch('verify', {'held': 'Is the bowl held?'}, completion="
        "{'name': 'grasp', 'when': {'held': 'yes'}, 'confirmations': 2})\n"
    )
    assert not tk.execute_tool("python", {"code": watch + MOVE}).result["supervision"]["paused"]
    result = tk.execute_tool("python", {"code": watch + MOVE})
    assert result.result["execution_paused"]
    assert monitor.reports[-1]["triggers"] == ["completion_review:grasp"]


@pytest.mark.parametrize("kind,trigger_value,opposite,trigger", [
    ("alerts", "no", "yes", "alert:condition"),
    ("completion", "yes", "no", "completion_review:condition"),
])
def test_review_rule_rearms_only_after_known_counterevidence(
    robo, tmp_path, kind, trigger_value, opposite, trigger
):
    rule = "{'name': 'condition', 'when': {'held': '" + trigger_value + "'}}"
    declaration = "[" + rule + "]" if kind == "alerts" else rule
    watch = (
        "watch('carry', {'held': 'Is the black bowl held by the gripper?'}, "
        + kind + "=" + declaration + ")\n"
    )
    tk, monitor, _ = setup(robo, tmp_path, [answer(value) for value in (
        trigger_value, "unknown", trigger_value, trigger_value, opposite, trigger_value,
    )])
    first = tk.execute_tool("python", {"code": watch + MOVE + "\n" + MOVE})
    assert first.result["execution_paused"]
    assert monitor.reports[-1]["triggers"] == [trigger]
    assert robo._ledger.stateful_calls() == 1

    resumed = tk.execute_tool("python", {"code": watch + "\n".join([MOVE] * 3)})
    assert not resumed.result.get("execution_paused")
    assert [report["event"] for report in monitor.reports] == ["clear"] * 3
    assert robo._ledger.stateful_calls() == 4

    recurred = tk.execute_tool("python", {"code": watch + "\n".join([MOVE] * 3)})
    assert recurred.result["execution_paused"]
    assert monitor.reports[-1]["triggers"] == [trigger]
    assert robo._ledger.stateful_calls() == 6


@pytest.mark.parametrize("kind,value,trigger", [
    ("alerts", "no", "alert:condition"),
    ("completion", "yes", "completion_review:condition"),
])
def test_repeated_source_and_new_matching_capture_do_not_repeat_review(
    snapshot_robo, backend, tmp_path, kind, value, trigger
):
    rule = "{'name': 'condition', 'when': {'held': '" + value + "'}}"
    declaration = "[" + rule + "]" if kind == "alerts" else rule
    watch = (
        "watch('inspect', {'held': 'Is the black bowl held by the gripper?'}, "
        + kind + "=" + declaration + ")\n"
    )
    tk, monitor, _ = setup(snapshot_robo, tmp_path, [answer(value)] * 3)
    first = tk.execute_tool("python", {"code": watch + "verify()"})
    assert first.result["execution_paused"]
    assert monitor.reports[-1]["triggers"] == [trigger]
    resumed = tk.execute_tool("python", {"code": watch +
        "verify()\nrobo.snapshot(refresh=True)\nverify()"})
    assert not resumed.result.get("execution_paused")
    assert [report["new_evidence"] for report in monitor.reports] == [
        {"held": False}, {"held": True},
    ]
    assert [report["event"] for report in monitor.reports] == ["clear", "clear"]
    assert backend.frames == 0 and snapshot_robo._ledger.stateful_calls() == 0


def test_completion_is_a_review_not_episode_success(robo, backend, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [answer(), answer()])
    code = (
        "watch('verify', {'held': 'Is the bowl held?'}, completion="
        "{'name': 'grasp', 'when': {'held': 'yes'}, 'confirmations': 2})\n"
    )
    result = tk.execute_tool("python", {"code": code + "\n".join([MOVE] * 3)})
    assert result.result["execution_paused"]
    assert result.result["supervision"]["actions"] == 2
    assert tk.finish is None and not backend.solved() and not backend.cancelled


def test_rule_requires_consecutive_evidence_and_unknown_is_neither_no_nor_yes(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [answer("no"), answer("unknown"), answer("no"), answer("no")])
    code = WATCH.replace("'when': {'held': 'no'}", "'when': {'held': 'no'}, 'confirmations': 2")
    result = tk.execute_tool("python", {"code": code + "\n".join([MOVE] * 5)})
    assert result.result["supervision"]["actions"] == 4


@pytest.mark.parametrize("output", [TimeoutError("slow"), {"answers": {}}, answer("bad-value")])
def test_monitor_errors_stop_next_motion(robo, tmp_path, output):
    tk, _, _ = setup(robo, tmp_path, [output])
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + MOVE})
    assert result.result["supervision"]["reports"][-1]["event"] == "monitor_error"
    assert result.result["execution_paused"] and robo._ledger.stateful_calls() == 1


def test_backend_failure_cannot_be_swallowed_to_continue_motion(robo, backend, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [])
    backend.error_next = {"error": "motion failed"}
    result = tk.execute_tool(
        "python", {"code": WATCH + f"try:\n    {MOVE}\nexcept Exception:\n    pass\n" + MOVE}
    )
    assert result.result["execution_paused"]
    assert result.result["supervision"]["reports"][-1]["event"] == "action_error"
    assert robo._ledger.stateful_calls() == 1


def test_zero_step_actions_do_not_count_as_new_evidence(robo, backend, tmp_path, monkeypatch):
    tk, _, client = setup(robo, tmp_path, [])
    monkeypatch.setattr(backend, "env_steps", lambda: 0)
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + MOVE})
    assert result.result["execution_paused"] and not client.packets
    assert "distinct executed" in result.result["supervision"]["reports"][-1]["error"]


def test_max_actions_survives_phase_changes(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [answer()] * 2, max_actions=2)
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + WATCH + MOVE + "\n" + MOVE})
    assert result.result["execution_paused"]
    assert result.result["supervision"]["reports"][-1]["triggers"] == ["action_budget_review"]


def test_input_is_images_and_questions_without_actions_or_truth(robo, tmp_path):
    tk, monitor, client = setup(robo, tmp_path, [answer()] * 2)
    tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + MOVE})
    packet = client.packets[-1]
    context = json.loads(packet["state"]["messages"][0]["content"][0]["text"])
    assert len(context["frames"]) == 2
    assert context["frames"][0]["step_id"] < context["frames"][1]["step_id"]
    assert set(packet) == {"state", "questions"}
    assert "alerts" not in json.dumps(packet) and "move_to" not in json.dumps(packet)
    assert len(list(monitor.directory.glob("*-request.json"))) == 2
    assert len(list(monitor.directory.glob("*-response.json"))) == 2


@pytest.mark.parametrize("change", [float("nan"), float("inf"), -0.1, True])
def test_reject_invalid_probabilities(change):
    raw = answer()
    raw["answers"]["held"]["probabilities"]["yes"] = change
    with pytest.raises(ValueError):
        decode_answers(raw, {"held": "question"}, 0.8)


def test_diffuse_prediction_is_unknown():
    raw = answer()
    raw["answers"]["held"]["probabilities"] = dict(yes=0.6, no=0.3, unknown=0.1)
    assert decode_answers(raw, {"held": "question"}, 0.8)[0] == {"held": "unknown"}


@pytest.mark.parametrize("steps,event", [(0, "motion_rejected"), (3, "motion_incomplete")])
def test_controller_refusal_can_return_locally_but_unclassified_partial_pauses(
    robo, backend, tmp_path, monkeypatch, steps, event
):
    tk, _, client = setup(robo, tmp_path, [])
    original = backend.execute_tool

    def reject(tool, args):
        if tool == "move_to":
            backend.frames += steps
            return {"reached": False, "ik_error": "arm intersects table", "steps_used": steps}
        return original(tool, args)

    # Exercise the common dispatch boundary, independent of adapter result parsers.
    monkeypatch.setattr(backend, "execute_tool", reject)
    monkeypatch.setattr(robo, "SUMMARY_KEYS", (*robo.SUMMARY_KEYS, "reached", "ik_error"))
    tk._globals["attempt"] = lambda: robo._stateful("move_to", {}, lambda p: p)
    result = tk.execute_tool("python", {"code": WATCH + "attempt()\nattempt()"})
    report = result.result["supervision"]
    assert not client.packets
    if steps:
        assert result.result["execution_paused"]
        assert event in report["reason"] and "arm intersects table" in report["reason"]
    else:
        assert not result.result.get("execution_paused")
        assert "arm intersects table" in report["reports"][-1]["error"]
    assert report["reports"][-1]["event"] == event
    assert report["reports"][-1]["executed_steps"] == steps
    assert robo._ledger.stateful_calls() == (1 if steps else 2)
    assert result.result["observation"]["actions"]["attempted"] == (1 if steps else 2)


def test_derived_arrival_failure_stops_before_more_actions(robo, backend, tmp_path, monkeypatch):
    tk, _, client = setup(robo, tmp_path, [])
    original = backend._do_move_to

    def incomplete(**kwargs):
        payload = original(**kwargs)
        payload["final_dist_m"] = 0.05
        assert "reached" not in payload
        return payload

    monkeypatch.setattr(backend, "_do_move_to", incomplete)
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + MOVE})
    assert result.result["execution_paused"] and not client.packets
    report = result.result["supervision"]["reports"][-1]
    assert report["event"] == "motion_incomplete"
    assert report["primitive_outcome"]["reached"] is False
    assert report["primitive_outcome"]["final_dist_m"] == 0.05
    assert report["executed_steps"] > 0
    assert robo._ledger.stateful_calls() == 1
    assert [r for r in robo._ledger if r.stateful][0].ok


def test_monitor_timeout_reports_original_cause(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [TimeoutError("Valen took too long")])
    result = tk.execute_tool("python", {"code": WATCH + MOVE})
    assert "TimeoutError: Valen took too long" in result.result["supervision"]["reason"]


def test_feedback_distinguishes_model_abstention_from_threshold(robo, tmp_path):
    raw = answer()
    raw["answers"]["held"]["probabilities"] = dict(yes=0.6, no=0.3, unknown=0.1)
    tk, _, _ = setup(robo, tmp_path, [raw, answer("unknown")])
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + MOVE})
    reports = result.result["supervision"]["reports"]
    assert reports[0]["values"]["held"] == "unknown"
    assert reports[0]["decisions"]["held"]["raw_winner"] == "yes"
    assert reports[0]["decisions"]["held"]["abstention_reason"] == "below_threshold"
    assert reports[1]["decisions"]["held"]["abstention_reason"] == "model_unknown"


def test_injected_grasp_helper_stops_transport_on_unknown_support(robo, tmp_path):
    tk, _, client = setup(robo, tmp_path, [answer(), answer("unknown")])
    code = WATCH + """
def observe_grasp():
    return strategy.GraspObservation.from_evidence(evidence(), 'held', holding=True)
r = strategy.verify_grasp(
    observe_grasp,
    lambda attempt: robo.move_to([0.05, -0.10, 0.6]),
    lambda before, attempt: robo.move_to([0.05, -0.10, 0.61]),
    lambda before, after: True,
)
print(r.status)
if r.status == 'verified':
    robo.move_to([0.05, 0.2, 0.7])
"""
    result = tk.execute_tool("python", {"code": code})
    assert not result.is_error and result.result["stdout"] == "uncertain\n"
    assert len(client.packets) == 2 and robo._ledger.stateful_calls() == 2


def test_large_policy_output_cannot_hide_review_reason(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [answer("no")])
    result = tk.execute_tool("python", {"code": WATCH + "print(chr(1) * 50000)\n" + MOVE})
    text = result.content_blocks[0]["text"]
    payload = json.loads(text)
    assert payload["supervision"]["paused"]
    assert "alert:dropped" in payload["supervision"]["reason"]
    assert payload["output_truncated"] and len(text) <= 20000


def test_malformed_replacement_invalidates_old_contract(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [])
    result = tk.execute_tool(
        "python",
        {"code": WATCH + "try:\n    watch('bad', {})\nexcept ValueError:\n    pass\n" + MOVE},
    )
    assert result.result["execution_paused"] and not robo._ledger.stateful_calls()


def test_http_protocol_against_local_valen_compatible_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(answer()).encode())

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ValenClient(
                MonitorConfig(mode="pause", url=f"http://127.0.0.1:{server.server_port}/predict")
            )
            assert client.predict({"questions": {}}) == answer()
            assert received == [("/predict", {"questions": {}})]
        finally:
            server.shutdown()
            thread.join()


def test_program_arm_rejected_before_execution(robo, tmp_path):
    _, monitor, _ = setup(robo, tmp_path, [])
    with pytest.raises(ValueError, match="cells"):
        ArmToolkit(robo, arm="program", workspace=tmp_path / "p", supervisor=monitor)


@pytest.mark.parametrize(
    "options",
    [
        dict(mode="pause"),
        dict(threshold=0.5),
        dict(timeout_s=float("nan")),
        dict(unknown_limit=True),
        dict(max_actions=0),
        dict(url="file:///tmp/model"),
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        MonitorConfig(**options)


def test_cli_forwards_monitor_configuration(tmp_path, monkeypatch):
    from pyrualean import cli

    commands = []
    out = tmp_path / "run"

    def host(argv, **kwargs):
        commands.append(argv)
        (out / "result.json").write_text(json.dumps({"status": "completed"}))
        return 0

    monkeypatch.setattr(cli, "run_host", host)
    assert (
        cli.main(
            [
                "play",
                "--arm",
                "cells",
                "--task",
                "0",
                "--out",
                str(out),
                "--monitor-mode",
                "pause",
                "--valen-url",
                "http://localhost:8001/predict",
                "--monitor-threshold",
                "0.9",
                "--monitor-max-actions",
                "3",
                "--monitor-timeout-s",
                "4",
                "--monitor-unknown-limit",
                "1",
            ]
        )
        == 0
    )
    argv = commands[0]
    assert argv[argv.index("--monitor-mode") + 1] == "pause"
    assert argv[argv.index("--monitor-threshold") + 1] == "0.9"
    assert argv[argv.index("--monitor-max-actions") + 1] == "3"
    assert argv[argv.index("--monitor-timeout-s") + 1] == "4.0"
    assert argv[argv.index("--monitor-unknown-limit") + 1] == "1"


def test_host_cli_receives_monitor_configuration(tmp_path, monkeypatch):
    from pyrualean import play

    configs = []

    def episode(*args, **kwargs):
        configs.append(kwargs["monitor"])
        return {"status": "completed"}

    monkeypatch.setattr(play, "play_episode", episode)
    assert (
        play.main(
            [
                "--arm",
                "cells",
                "--suite",
                "libero_spatial_swap",
                "--task",
                "0",
                "--out",
                str(tmp_path),
                "--monitor-mode",
                "shadow",
                "--valen-url",
                "http://localhost:8001/predict",
                "--monitor-unknown-limit",
                "1",
            ]
        )
        == 0
    )
    assert configs[0].mode == "shadow" and configs[0].unknown_limit == 1


def test_supervision_prompt_is_opt_in():
    from pyrualean.play import build_prompt
    from pyrualean.prompt import TaskCard

    card = TaskCard(suite="test", task=0, seed=0, task_language="Move the bowl", object_names=[])
    options = dict(budget_s=100, max_programs=None)
    original = build_prompt("cells", card, **options)
    disabled = build_prompt("cells", card, **options, monitor=MonitorConfig())
    enabled = build_prompt(
        "cells",
        card,
        **options,
        monitor=MonitorConfig(mode="pause", url="http://localhost:8001/predict"),
    )
    assert original == disabled and "watch(phase" not in original
    assert "watch(phase" in enabled and "interrupted assignment" in enabled
    from pyrualean.supervision import planning_instructions

    assert planning_instructions() in enabled
    assert "Astra–Valen planning skill" not in original


def test_missing_camera_pauses_with_partial_evidence(robo, backend, tmp_path, monkeypatch):
    tk, _, client = setup(robo, tmp_path, [])
    original = backend.artifact

    def artifact(name, step=-1):
        if "wrist" in name:
            raise FileNotFoundError(name)
        return original(name, step)

    monkeypatch.setattr(backend, "artifact", artifact)
    result = tk.execute_tool("python", {"code": WATCH + MOVE + "\n" + MOVE})
    assert result.result["execution_paused"] and not client.packets
    assert result.result["supervision"]["evidence_cameras"] == ["agentview"]
    assert len(result.images) == 1


def test_duplicate_critical_ids_are_rejected(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [])
    result = tk.execute_tool(
        "python",
        {
            "code": "watch('phase', {'held': 'Is the bowl held?'}, critical=['held', 'held'])\n"
            + MOVE
        },
    )
    assert "distinct" in result.result["exception"] and not robo._ledger.stateful_calls()


def test_replay_is_read_only_and_uses_native_control_steps(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "replay_supervision", Path(__file__).parents[1] / "examples/replay_supervision.py"
    )
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    image = tmp_path / "camera.png"
    image.write_bytes(b"recorded image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"step": 21, "images": {"head": str(image)}},
                {"step": 26, "images": {"head": str(image)}},
            ]
        )
    )
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"phase": "transport", "facts": {"held": "Is the bowl held?"}}))
    client = ScriptedValen([answer(), answer()])
    monkeypatch.setattr(ValenClient, "predict", lambda self, packet: client.predict(packet))
    out = tmp_path / "replay"
    assert (
        replay.main(
            [
                "--manifest",
                str(manifest),
                "--contract",
                str(contract),
                "--valen-url",
                "http://localhost:8001/predict",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    result = json.loads((out / "replay.json").read_text())
    assert result["robot_actions_executed"] == 0
    assert [row["step"] for row in result["reports"]] == [21, 26]
    with pytest.raises(ValueError, match="increasing"):
        replay.RecordedRobot([{"step": 1, "images": {"head": str(image)}}] * 2)


FACTS = "{'held': 'Is the black bowl held by the gripper?'}"
OBSERVE = f"watch('grasp', {FACTS})\n"
CRITICAL_OBSERVE = f"watch('grasp', {FACTS}, critical=['held'])\n"
REQUIRE = f"watch('transport', {FACTS}, requires={{'held': 'yes'}})\n"
DEFER = "defer_fact('held', until='transport', reason='Grasp must precede transport')\n"


def test_visual_review_handoff_does_not_clear_explicit_requirement(robo, tmp_path):
    tk, monitor, client = setup(robo, tmp_path, [answer("unknown")] * 2)
    first = tk.execute_tool("python", {"code": CRITICAL_OBSERVE + MOVE + "\n" + MOVE})
    assert first.result["execution_paused"]
    blocked = tk.execute_tool("python", {"code": REQUIRE + MOVE})
    assert blocked.result["execution_paused"]
    assert monitor.reports[-1]["event"] == "prerequisite_review"
    assert monitor.reports[-1]["blocked"]["held"]["expected"] == "yes"
    assert monitor.reports[-1]["blocked"]["held"]["value"] == "unknown"
    assert len(client.packets) == robo._ledger.stateful_calls() == 2


def test_visual_review_handoff_cannot_clear_unknown_execution(tmp_path):
    from pyrualean import LiberoRobot
    from tests.test_interaction_contracts import ProtocolBackend

    backend = ProtocolBackend()
    robo = LiberoRobot(backend)
    tk, monitor, client = setup(robo, tmp_path, [answer("unknown")] * 2)
    first = tk.execute_tool("python", {"code": CRITICAL_WATCH + MOVE + "\n" + MOVE})
    assert first.result["execution_paused"] and len(client.packets) == 2
    backend.next_status = "unknown"
    uncertain = tk.execute_tool("python", {"code": CRITICAL_WATCH + MOVE})
    assert uncertain.result["execution_paused"]
    action_id = uncertain.result["handoff"]["unresolved_actions"][0]["action_id"]
    assert robo.action_receipt(action_id)["status"] == "unknown"
    assert len(backend.contexts) == 3

    blocked = tk.execute_tool("python", {"code": CRITICAL_WATCH + MOVE})
    assert blocked.result["execution_paused"]
    assert blocked.result["handoff"]["unresolved_actions"][0]["action_id"] == action_id
    assert len(backend.contexts) == 3 and len(client.packets) == 2
    assert monitor.read_evidence()["facts"]["held"]["value"] == "unknown"


@pytest.mark.parametrize("value", ["unknown", "no"])
def test_single_unresolved_fact_blocks_dependency_before_motion(robo, tmp_path, value):
    tk, monitor, client = setup(robo, tmp_path, [answer(value)])
    first = tk.execute_tool("python", {"code": OBSERVE + MOVE})
    assert first.result["supervision"]["reports"][-1]["event"] == "clear"
    result = tk.execute_tool("python", {"code": REQUIRE + MOVE})
    assert result.result["execution_paused"] and len(client.packets) == 1
    assert robo._ledger.stateful_calls() == 1
    assert monitor.feedback()["blocked"]["held"]["value"] == value
    assert len(result.images) == len(robo.CAMERAS)
    assert result.result["supervision"]["observation"]["source"] == "host_review"


def test_unobserved_dependency_blocks_first_motion(robo, tmp_path):
    tk, monitor, _ = setup(robo, tmp_path, [])
    result = tk.execute_tool("python", {"code": REQUIRE + MOVE})
    assert result.result["execution_paused"] and not robo._ledger.stateful_calls()
    assert monitor.feedback()["blocked"]["held"]["invalidated_by"] == "not_observed"


def test_deferred_gate_survives_cells_contract_reset_and_then_recovers(robo, tmp_path):
    tk, monitor, client = setup(robo, tmp_path, [answer("unknown"), answer(), answer()])
    tk.execute_tool("python", {"code": OBSERVE + DEFER + MOVE})
    recorded = monitor.read_evidence()
    fact = recorded["facts"]["held"]
    assert fact["value"] == "unknown" and fact["fresh"]
    assert fact["observation"]["state_version"] == robo._action_revision == 1
    # Neither a new cell nor an empty requires argument can forget this gate.
    blocked = tk.execute_tool("python", {"code": WATCH + MOVE})
    assert blocked.result["execution_paused"] and len(client.packets) == 1
    blocked = tk.execute_tool("python", {"code": WATCH.replace(
        "alerts=", "requires={}, alerts=") + MOVE})
    assert blocked.result["execution_paused"] and len(client.packets) == 1
    repaired = tk.execute_tool("python", {"code": OBSERVE + MOVE})
    assert not repaired.result["supervision"]["paused"]
    resumed = tk.execute_tool("python", {"code": WATCH + MOVE})
    assert not resumed.result["supervision"]["paused"]
    assert robo._ledger.stateful_calls() == 3
    latest = monitor.read_evidence()
    assert latest["facts"]["held"]["observation"] != fact["observation"]
    assert latest["phase_requirements"] == recorded["phase_requirements"]
    assert json.loads((monitor.directory / "evidence.json").read_text()) == latest


def test_each_motion_requires_evidence_even_after_clear_unknown(robo, tmp_path):
    tk, _, client = setup(robo, tmp_path, [answer(), answer("unknown")])
    tk.execute_tool("python", {"code": OBSERVE + MOVE})
    result = tk.execute_tool("python", {"code": REQUIRE + MOVE + "\n" + MOVE})
    assert result.result["execution_paused"] and len(client.packets) == 2
    assert result.result["supervision"]["reports"][-2]["event"] == "clear"
    assert result.result["supervision"]["reports"][-1]["event"] == "prerequisite_review"


def test_stale_evidence_cannot_pass_even_when_last_value_was_yes(robo, tmp_path):
    other = {"answers": {"visible": answer()["answers"]["held"]}}
    tk, monitor, _ = setup(robo, tmp_path, [answer(), other])
    tk.execute_tool("python", {"code": OBSERVE + DEFER + MOVE})
    tk.execute_tool("python", {"code":
        "watch('adjust', {'visible': 'Is the plate visible?'})\n" + MOVE})
    stale = monitor.read_evidence()["facts"]["held"]
    assert stale["value"] == "yes" and not stale["fresh"]
    result = tk.execute_tool("python", {"code": WATCH + MOVE})
    assert result.result["execution_paused"] and robo._ledger.stateful_calls() == 2


def test_explicit_invalidators_preserve_only_unaffected_evidence(robo, tmp_path):
    other = {"answers": {"visible": answer()["answers"]["held"]}}
    tk, monitor, _ = setup(robo, tmp_path, [answer(), other, answer()])
    tk.execute_tool("python", {"code": OBSERVE + DEFER + MOVE})
    before = monitor.read_evidence()["facts"]["held"]
    tk.execute_tool("python", {"code":
        "watch('independent', {'visible': 'Is the plate visible?'}, "
        "invalidates=['visible'])\n" + MOVE})
    assert monitor.read_evidence()["facts"]["held"] == before
    result = tk.execute_tool("python", {"code": WATCH + MOVE})
    assert not result.result["supervision"]["paused"]


@pytest.mark.parametrize("kind", ["action", "monitor", "external"])
def test_failures_and_unobserved_state_changes_invalidate_preserved_facts(
    robo, backend, tmp_path, kind
):
    tk, monitor, _ = setup(robo, tmp_path, [answer(), TimeoutError("no evidence")])
    tk.execute_tool("python", {"code": OBSERVE + DEFER + MOVE})
    if kind == "external":
        backend.frames += 1
    else:
        if kind == "action":
            backend.error_next = {"error": "partially moved"}
        tk.execute_tool("python", {"code": OBSERVE.replace(
            ")\n", ", invalidates=[])\n") + MOVE})
    result = tk.execute_tool("python", {"code": WATCH + MOVE})
    assert result.result["execution_paused"]
    assert not monitor.read_evidence()["facts"]["held"]["fresh"]
    assert robo._ledger.stateful_calls() == (1 if kind == "external" else 2)


def test_question_change_invalidates_evidence_and_pinned_question_is_stable(robo, tmp_path):
    tk, monitor, _ = setup(robo, tmp_path, [answer()])
    tk.execute_tool("python", {"code": OBSERVE + MOVE})
    tk.execute_tool("python", {"code": OBSERVE.replace("black bowl", "red cup")})
    fact = monitor.read_evidence()["facts"]["held"]
    assert fact["version"] == 2 and not fact["fresh"] and fact["observation"] is None
    tk.execute_tool("python", {"code": DEFER})
    result = tk.execute_tool("python", {"code": OBSERVE + MOVE})
    assert "keep its question" in result.result["exception"]
    assert monitor.read_evidence()["facts"]["held"] == fact
    assert robo._ledger.stateful_calls() == 1


def test_evidence_snapshot_cannot_forge_a_gate_and_helpers_are_rebound(robo, tmp_path):
    tk, monitor, _ = setup(robo, tmp_path, [])
    tk.execute_tool("python", {"code": REQUIRE +
        "copy = evidence()\ncopy['facts']['held'].update(value='yes', fresh=True)\n"
        "copy['phase_requirements'].clear()\nevidence = None\ndefer_fact = None"})
    result = tk.execute_tool("python", {"code": WATCH + "assert callable(evidence)\n"
        "assert callable(defer_fact)\n" + MOVE})
    assert result.result["execution_paused"] and not robo._ledger.stateful_calls()
    assert monitor.read_evidence()["facts"]["held"]["value"] == "unknown"


def test_shadow_records_prerequisite_but_does_not_block(robo, tmp_path):
    tk, monitor, client = setup(robo, tmp_path, [answer()], mode="shadow")
    result = tk.execute_tool("python", {"code": REQUIRE + MOVE})
    assert not result.result["supervision"]["paused"]
    assert [r["event"] for r in monitor.reports] == ["prerequisite_review", "clear"]
    assert len(client.packets) == 1


def test_declared_negative_prerequisite_accepts_only_fresh_no(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [answer("no"), answer("no")])
    tk.execute_tool("python", {"code": OBSERVE + MOVE})
    result = tk.execute_tool("python", {"code": REQUIRE.replace("'yes'", "'no'") + MOVE})
    assert not result.result["supervision"]["paused"]


def test_scene_changes_during_prediction_do_not_establish_evidence(
    robo, backend, tmp_path, monkeypatch
):
    tk, monitor, client = setup(robo, tmp_path, [])

    def predict(packet):
        backend.frames += 1
        return answer()

    monkeypatch.setattr(client, "predict", predict)
    result = tk.execute_tool("python", {"code": OBSERVE + MOVE})
    assert result.result["execution_paused"]
    assert "evidence is stale" in monitor.reports[-1]["error"]
    assert not monitor.read_evidence()["facts"]["held"]["fresh"]


def test_large_output_retains_prerequisite_blockers(robo, tmp_path):
    tk, _, _ = setup(robo, tmp_path, [])
    result = tk.execute_tool("python", {"code": REQUIRE +
        "print(chr(1) * 8000)\n" + MOVE})
    text = result.content_blocks[0]["text"]
    payload = json.loads(text)
    assert len(text) <= 20000
    assert payload["supervision"]["paused"]
    assert payload["supervision"]["blocked"]["held"]["value"] == "unknown"


@pytest.mark.parametrize("where", ["action", "monitor"])
def test_interruption_invalidates_preservation_claims(robo, backend, tmp_path, monkeypatch, where):
    tk, monitor, client = setup(robo, tmp_path, [answer()])
    tk.execute_tool("python", {"code": OBSERVE + DEFER + MOVE})
    original = backend.execute_tool

    def interrupted_action(name, arguments):
        if name in robo.STATEFUL_TOOLS:
            raise SystemExit("interrupted")
        return original(name, arguments)

    def interrupted_monitor(packet):
        raise SystemExit("interrupted")

    if where == "action":
        monkeypatch.setattr(backend, "execute_tool", interrupted_action)
    else:
        monkeypatch.setattr(client, "predict", interrupted_monitor)
    tk.execute_tool("python", {"code": OBSERVE.replace(
        ")\n", ", invalidates=[])\n") + MOVE})
    assert not monitor.read_evidence()["facts"]["held"]["fresh"]
    result = tk.execute_tool("python", {"code": WATCH + MOVE})
    assert result.result["execution_paused"] and robo._ledger.stateful_calls() == 2


def test_supervision_skill_is_frozen_and_hashed_even_if_boot_fails(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from pyrualean import play, supervision
    from pyrualean.robots import get_adapter
    from pyrualean.supervision import planning_instructions

    frozen_core = planning_instructions()

    def boot(*args, **kwargs):
        # A source edit during boot must not change the frozen run's metadata/prompt.
        monkeypatch.setattr(supervision, "planning_instructions", lambda: "new revision")
        raise RuntimeError("no simulator")

    adapter = SimpleNamespace(name="libero", robot_cls=get_adapter("libero").robot_cls,
                              host_module=lambda: SimpleNamespace(boot=boot))
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    cell = SimpleNamespace(suite="test", task=0, seed=0, cuda_device=0, max_episode_steps=100)
    out = tmp_path / "run"
    result = play.play_episode(cell, out, arm="cells", model="m",
                              monitor=MonitorConfig(mode="pause", url="http://localhost:1"))
    assert result["status"] == "boot_error"
    frozen = (out / "supervision_skill.md").read_text()
    assert frozen == frozen_core
    assert result["monitor"]["planning_skill"]["sha256"] == hashlib.sha256(
        frozen.encode()).hexdigest()
    card = SimpleNamespace(max_env_steps=100, render=lambda **kwargs: "TASK")
    prompt = play.build_prompt("cells", card, budget_s=100, max_programs=None,
                               adapter=get_adapter("libero"),
                               monitor=MonitorConfig(mode="pause", url="http://localhost:1"),
                               supervision_instructions=frozen)
    assert frozen in prompt and "new revision" not in prompt


def test_verify_establishes_first_motion_prerequisite_without_moving(
    snapshot_robo, backend, tmp_path
):
    tk, monitor, client = setup(snapshot_robo, tmp_path, [answer(), answer()])
    observed = tk.execute_tool("python", {"code": REQUIRE + "checked = verify()"})
    assert not observed.is_error and not observed.result.get("execution_paused")
    assert backend.frames == 0 and snapshot_robo._ledger.stateful_calls() == 0
    assert monitor.read_evidence()["facts"]["held"]["fresh"]
    assert monitor.feedback()["verification_calls"] == 1
    assert len(client.packets) == 1
    resumed = tk.execute_tool("python", {"code": REQUIRE + MOVE})
    assert not resumed.result.get("execution_paused")
    assert snapshot_robo._ledger.stateful_calls() == 1
    assert monitor.reports[-1]["action_id"] == snapshot_robo.action_receipt()["action_id"]


def test_verify_repeated_source_cannot_clear_unknown_or_add_confirmation(
    snapshot_robo, backend, tmp_path
):
    tk, monitor, _ = setup(snapshot_robo, tmp_path, [answer("unknown"), answer(),
                                                    answer("unknown")])
    first = tk.execute_tool("python", {"code": CRITICAL_OBSERVE + "verify()\nverify()"})
    assert not first.result.get("execution_paused")
    assert monitor.unknowns["held"] == 1
    assert monitor.reports[-1]["values"]["held"] == "unknown"
    assert monitor.reports[-1]["new_evidence"] == {"held": False}
    second = tk.execute_tool("python", {"code": CRITICAL_OBSERVE +
        "new_view = robo.snapshot(refresh=True)\nverify(new_view['observation_id'])"})
    assert second.result["execution_paused"]
    assert monitor.unknowns["held"] == 2
    assert backend.frames == 0 and snapshot_robo._ledger.stateful_calls() == 0


def test_verify_completion_needs_two_distinct_captures(snapshot_robo, backend, tmp_path):
    tk, monitor, _ = setup(snapshot_robo, tmp_path, [answer()] * 3)
    watch = (
        "watch('verify', {'held': 'Is the bowl held?'}, completion="
        "{'name': 'grasp', 'when': {'held': 'yes'}, 'confirmations': 2})\n"
    )
    first = tk.execute_tool("python", {"code": watch + "verify()\nverify()"})
    assert not first.result.get("execution_paused")
    assert monitor.counts["completion"] == 1
    second = tk.execute_tool("python", {"code": watch + "robo.snapshot(refresh=True)\nverify()"})
    assert second.result["execution_paused"]
    assert monitor.reports[-1]["triggers"] == ["completion_review:grasp"]
    assert tk.finish is None and backend.frames == 0


def test_verify_rejects_expired_observation_before_predicting(snapshot_robo, tmp_path):
    tk, monitor, client = setup(snapshot_robo, tmp_path, [answer(), answer()])
    tk.execute_tool("python", {"code": OBSERVE + "old = robo.snapshot()\nverify()\n" + MOVE})
    result = tk.execute_tool("python", {"code": OBSERVE + "verify(old['observation_id'])"})
    assert result.result["execution_paused"]
    assert monitor.reports[-1]["event"] == "monitor_error"
    assert len(client.packets) == 2 and snapshot_robo._ledger.stateful_calls() == 1
    assert not monitor.read_evidence()["facts"]["held"]["fresh"]


def test_verify_unknown_id_cannot_establish_evidence(snapshot_robo, tmp_path):
    tk, monitor, client = setup(snapshot_robo, tmp_path, [])
    result = tk.execute_tool("python", {"code": REQUIRE + "verify('not-a-host-observation')"})
    assert result.result["execution_paused"] and not client.packets
    assert monitor.ledger.blockers("transport")
    assert snapshot_robo._ledger.stateful_calls() == 0


def test_verify_scope_rejects_motion_from_verifier_callback(
    snapshot_robo, backend, tmp_path, monkeypatch
):
    tk, monitor, client = setup(snapshot_robo, tmp_path, [])

    def bad_predict(packet):
        snapshot_robo.move_to([0.05, -0.1, 0.6])
        return answer()

    monkeypatch.setattr(client, "predict", bad_predict)
    result = tk.execute_tool("python", {"code": OBSERVE + "verify()"})
    assert result.result["execution_paused"]
    assert monitor.reports[-1]["event"] == "monitor_error"
    assert snapshot_robo._ledger.stateful_calls() == 0 and backend.frames == 0
    assert not monitor.read_evidence()["facts"]["held"]["fresh"]


def test_verification_failures_consume_separate_budget(snapshot_robo, tmp_path):
    tk, monitor, client = setup(snapshot_robo, tmp_path, [TimeoutError("no answer")],
                               max_actions=1, mode="shadow")
    result = tk.execute_tool("python", {"code": OBSERVE + "verify()\nverify()"})
    assert not result.is_error
    assert len(client.packets) == 1
    assert [report["event"] for report in monitor.reports] == [
        "monitor_error", "verification_budget_review",
    ]
    assert monitor.feedback()["verification_calls"] == 2
    assert monitor.actions == 0


def test_verify_unsupported_backend_is_explicitly_rejected(robo, tmp_path):
    tk, monitor, client = setup(robo, tmp_path, [])
    result = tk.execute_tool("python", {"code": OBSERVE + "verify()"})
    assert result.result["execution_paused"] and not client.packets
    assert "host-owned observation" in monitor.reports[-1]["error"]
    assert robo._ledger.stateful_calls() == 0


def test_source_refresh_without_verification_invalidates_previous_prerequisite(
    snapshot_robo, tmp_path
):
    tk, monitor, client = setup(snapshot_robo, tmp_path, [answer()])
    tk.execute_tool("python", {"code": REQUIRE + "verify()"})
    result = tk.execute_tool("python", {"code": REQUIRE +
        "robo.snapshot(refresh=True)\n" + MOVE})
    assert result.result["execution_paused"]
    assert monitor.ledger.blockers("transport")["held"]["invalidated_by"] == (
        "observation_source_changed"
    )
    assert len(client.packets) == 1 and snapshot_robo._ledger.stateful_calls() == 0


def test_unknown_review_rearms_after_new_known_capture(snapshot_robo, tmp_path):
    tk, monitor, _ = setup(snapshot_robo, tmp_path, [answer(value) for value in (
        "unknown", "unknown", "yes", "unknown", "unknown",
    )])
    paused = tk.execute_tool("python", {"code": CRITICAL_OBSERVE +
        "verify()\nrobo.snapshot(refresh=True)\nverify()"})
    assert paused.result["execution_paused"]
    known = tk.execute_tool("python", {"code": CRITICAL_OBSERVE +
        "robo.snapshot(refresh=True)\nverify()"})
    assert not known.result.get("execution_paused")
    assert monitor.unknowns["held"] == 0
    recurred = tk.execute_tool("python", {"code": CRITICAL_OBSERVE +
        "robo.snapshot(refresh=True)\nverify()\n"
        "robo.snapshot(refresh=True)\nverify()"})
    assert recurred.result["execution_paused"]
    assert monitor.reports[-1]["triggers"] == ["unknown:held"]
    assert monitor.unknowns["held"] == 2
    assert snapshot_robo._ledger.stateful_calls() == 0


def test_same_source_cannot_rearm_or_repeat_unknown_review(snapshot_robo, tmp_path):
    tk, monitor, _ = setup(snapshot_robo, tmp_path, [answer(value) for value in (
        "unknown", "unknown", "yes", "unknown",
    )])
    paused = tk.execute_tool("python", {"code": CRITICAL_OBSERVE +
        "verify()\nrobo.snapshot(refresh=True)\nverify()"})
    assert paused.result["execution_paused"]
    resumed = tk.execute_tool("python", {"code": CRITICAL_OBSERVE +
        "verify()\nrobo.snapshot(refresh=True)\nverify()"})
    assert not resumed.result.get("execution_paused")
    assert monitor.reports[0]["new_evidence"] == {"held": False}
    assert monitor.reports[0]["values"]["held"] == "unknown"
    assert monitor.unknowns["held"] == 3
    assert snapshot_robo._ledger.stateful_calls() == 0
