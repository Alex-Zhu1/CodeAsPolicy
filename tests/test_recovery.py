"""Generated-code failures must return evidence before the planner acts again."""

from __future__ import annotations

import json

import pytest

from pyrualean.arms import MAX_TEXT_CHARS, ArmToolkit

MOVE = "robo.move_to([0.05, -0.10, 0.6])"


@pytest.mark.parametrize("bad_tail", ["yield 1", "return 1", "def broken(:"])
def test_validation_rejects_entire_cell_before_motion(robo, backend, tmp_path, bad_tail):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, feedback="pure")
    result = kit.execute_tool("python", {"code": MOVE + "\n" + bad_tail})
    assert result.is_error and not kit.turns[-1]["ok"]
    assert result.result["failure"]["kind"] == "syntax"
    assert result.result["failure"]["location"]["line"] == 2
    assert result.result["recovery"]["motion_attempts"] == 0
    assert not result.result["recovery"]["namespace_may_have_changed"]
    assert backend.frames == 0 and not backend.cancelled


def test_partial_execution_is_repaired_without_replaying_motion(robo, backend, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, feedback="pure", images="none")
    failed = kit.execute_tool("python", {"code": "saved = 7\n" + MOVE + "\nmissing_name"})
    error, recovery = failed.result["failure"], failed.result["recovery"]
    assert error["type"] == "NameError" and error["location"]["line"] == 3
    assert error["location"]["file"] == str(tmp_path / "cell_001.py")
    assert recovery["next_action"] == "inspect_and_replan" and recovery["retry_allowed"]
    assert recovery["motion_attempts"] == recovery["completed_motion_calls"] == 1
    assert recovery["last_motion"]["tool"] == "move_to"
    assert recovery["state"]["eef_pos"] == [0.05, -0.10, 0.6]
    assert recovery["env_steps"] == backend.frames > 0
    assert not recovery["scene_rolled_back"] and not recovery["automatic_replay"]
    assert recovery["namespace_may_have_changed"] and not backend.cancelled
    frames = backend.frames
    repaired = kit.execute_tool("python", {"code": "missing_name = saved * 2\nmissing_name"})
    assert repaired.result["value"] == "14" and not repaired.is_error
    assert backend.frames == frames and kit.turns[-1]["ok"]
    assert "recovery" not in repaired.result


def test_failed_primitive_is_an_attempt_not_a_completed_action(robo, backend, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path)
    backend.error_next = {"error": "controller unavailable"}
    failed = kit.execute_tool("python", {"code": MOVE})
    assert failed.result["failure"]["kind"] == "primitive"
    assert failed.result["failure"]["location"]["line"] == 1
    assert failed.result["recovery"]["motion_attempts"] == 1
    assert failed.result["recovery"]["completed_motion_calls"] == 0
    assert failed.result["recovery"]["last_motion"]["error"]
    assert failed.result["recovery"]["next_action"] == "inspect_and_replan"


def test_failed_postcondition_stops_following_action(robo, backend, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="none")
    backend.error_next = {"final_dist_m": 0.1, "steps_used": 5}
    result = kit.execute_tool("python", {"code": (
        "m = " + MOVE + "\n"
        "assert m.reached, 'approach did not reach target'\n"
        "robo.release()"
    )})
    assert result.result["failure"]["kind"] == "verification"
    assert result.result["recovery"]["completed_motion_calls"] == 1
    assert not any(name == "release" for name, _ in backend.calls)


def test_repeated_errors_and_failed_attempts_consume_existing_budget(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, max_turns=4, images="none")
    for index in range(1, 5):
        result = kit.execute_tool("python", {"code": "undefined"})
        recovery = result.result["recovery"]
        assert recovery["same_failure_count"] == index
        assert recovery["calls_left"] == 4 - index
        if index == 3:
            assert recovery["next_action"] == "change_approach_or_finish"
    assert not recovery["retry_allowed"] and recovery["next_action"] == "finish"
    assert kit.execute_tool("python", {"code": MOVE}).is_error
    assert not robo._ledger.stateful_calls()


@pytest.mark.parametrize("code,kind,stage", [
    ("def run(:", "syntax", "validation"),
    ("import missing_module\ndef run(robo): pass", "import", "load"),
    ("print('loading')\nx = 1", "runtime", "load"),
    ("def run(robo):\n    raise SystemExit('stop')", "runtime", "execution"),
    ("raise SystemExit('stop')", "runtime", "load"),
])
def test_single_shot_program_failures_are_reported_without_extra_attempt(
    robo, tmp_path, code, kind, stage
):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path, max_programs=1)
    result = kit.execute_tool("run_program", {"code": code})
    assert result.is_error and not kit.turns[-1]["ok"]
    assert result.result["failure"]["kind"] == kind
    assert result.result["failure"]["stage"] == stage
    assert result.result["recovery"]["calls_left"] == 0
    assert result.result["recovery"]["next_action"] == "finish"
    assert not result.result["recovery"]["retry_allowed"]
    assert kit.execute_tool("run_program", {"code": "def run(robo): pass"}).is_error
    assert not robo._ledger.stateful_calls()
    if "loading" in code:
        assert result.result["stdout"] == "loading\n"


def test_program_runtime_error_keeps_original_location(robo, tmp_path):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path, max_programs=2)
    result = kit.execute_tool("run_program", {"code": (
        "def run(robo):\n    " + MOVE + "\n    1 / 0\n"
    )})
    assert result.result["failure"]["type"] == "ZeroDivisionError"
    assert result.result["failure"]["location"]["line"] == 3
    assert result.result["recovery"]["motion_attempts"] == 1
    assert result.result["recovery"]["calls_left"] == 1


def test_runtime_syntax_error_is_not_claimed_as_preflight_rejection(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path)
    result = kit.execute_tool("python", {"code": MOVE + "\nraise SyntaxError('late error')"})
    assert result.result["failure"]["kind"] == "syntax"
    assert result.result["failure"]["stage"] == "execution"
    assert result.result["recovery"]["namespace_may_have_changed"]
    assert result.result["recovery"]["motion_attempts"] == 1


def test_error_in_persistent_helper_points_to_its_definition(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path)
    kit.execute_tool("python", {"code": "def helper():\n    return missing\n"})
    result = kit.execute_tool("python", {"code": "helper()"})
    assert result.result["failure"]["location"] == {
        "file": str(tmp_path / "cell_001.py"), "line": 2,
    }


def test_long_output_keeps_parseable_failure_and_recovery(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path)
    result = kit.execute_tool("python", {"code": (
        "print(chr(1) * 50000)\n1 / 0"
    )})
    text = result.content_blocks[0]["text"]
    payload = json.loads(text)
    assert len(text) <= MAX_TEXT_CHARS and payload["output_truncated"]
    assert payload["failure"]["type"] == "ZeroDivisionError"
    assert payload["recovery"]["next_action"] == "repair_code"


def test_mcp_marks_execution_failures_but_not_normal_completion(robo, tmp_path):
    pytest.importorskip("mcp")
    from pyrualean.mcp_server import _result_to_mcp

    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path)
    failed = kit.execute_tool("python", {"code": "1 / 0"})
    assert _result_to_mcp(failed)[1]
    completed = kit.execute_tool("python", {"code": "2 + 2"})
    assert not _result_to_mcp(completed)[1]
