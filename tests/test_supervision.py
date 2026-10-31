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
    result = tk.execute_tool("python", {"code": WATCH + "\n".join([MOVE] * 3)})
    assert result.result["supervision"]["actions"] == 2
    assert result.result["supervision"]["reports"][-1]["triggers"] == ["unknown:held"]
    result = tk.execute_tool(
        "python", {"code": "watch('approach', {'held': 'Is the bowl held?'}, critical=[])\n" + MOVE}
    )
    assert not result.result["supervision"]["paused"]


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
