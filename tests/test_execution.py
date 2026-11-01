"""Execution boundaries must return control with the partially changed scene."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyrualean.arms import MAX_TEXT_CHARS, ArmToolkit
from pyrualean.execution import BoundedOutput, ExecutionLimits

MOVE = "robo.move_to([0.05, -0.1, 0.6])"


@pytest.mark.parametrize("arm", ["cells", "program"])
def test_finish_refuses_all_later_code_without_consuming_budget(robo, backend, tmp_path, arm):
    kit = ArmToolkit(robo, arm=arm, workspace=tmp_path)
    finished = kit.execute_tool("finish", {"status": "failure", "summary": "Cannot proceed"})
    before = len(robo._ledger)
    result = kit.execute_tool("python" if arm == "cells" else "run_program", {
        "code": MOVE if arm == "cells" else "def run(robo):\n    " + MOVE,
    })
    assert result.is_error and result.result["episode_finished"] is True
    assert backend.frames == 0 and len(robo._ledger) == before
    assert kit._cell_index == kit._program_index == 0
    assert kit.execute_tool("finish", {"status": "success"}).result["status"] == "failure"
    assert finished.result["observation"]["actions"]["attempted"] == 0


@pytest.mark.parametrize("mode", ["cell", "helper", "program", "load", "thread"])
def test_pure_python_loop_is_interrupted_in_a_bounded_subprocess(tmp_path, mode):
    # An outer process timeout prevents a regression in the guard from hanging pytest.
    script = r'''
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from conftest import FakeLiberoBackend
from pyrualean.arms import ArmToolkit
from pyrualean.execution import ExecutionLimits
from pyrualean.libero import LiberoRobot
mode, directory = sys.argv[1:]
backend = FakeLiberoBackend()
kit = ArmToolkit(LiberoRobot(backend), arm="program" if mode in ("program", "load") else "cells",
                 workspace=Path(directory), images="none", feedback="pure",
                 execution_limits=ExecutionLimits(timeout_s=0.5))
tool, code = "python", "saved = 7\nrobo.move_to([0.05,-0.1,0.6])\nwhile True: pass"
if mode == "helper":
    kit.execute_tool("python", {"code": "def spin():\n    while True: pass"})
    code = "saved = 7\nrobo.move_to([0.05,-0.1,0.6])\nspin()"
elif mode == "program":
    tool = "run_program"
    code = "def run(robo):\n    robo.move_to([0.05,-0.1,0.6])\n    while True: pass"
elif mode == "load":
    tool, code = "run_program", "while True: pass\ndef run(robo): pass"
if mode == "thread":
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(kit.execute_tool, tool, {"code": code}).result(timeout=3)
else:
    result = kit.execute_tool(tool, {"code": code})
assert result.result["failure"]["kind"] == "timeout", result.result
expected_stage = "load" if mode == "load" else "execution"
assert result.result["failure"]["stage"] == expected_stage, result.result
assert result.result["recovery"]["motion_attempts"] == (0 if mode == "load" else 1), result.result
frames = backend.frames
if tool == "python":
    repaired = kit.execute_tool("python", {"code": "saved + 1"})
    assert repaired.result["value"] == "8", repaired.result
else:
    repaired = kit.execute_tool("run_program", {"code": "def run(robo): print('repaired')"})
    assert repaired.result["stdout"] == "repaired\n", repaired.result
assert not repaired.is_error and backend.frames == frames and not backend.cancelled
print(json.dumps({"kind": result.result["failure"]["kind"], "frames": frames}))
'''
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "tests")]))
    run = subprocess.run([sys.executable, "-c", script, mode, str(tmp_path)],
                         env=env, capture_output=True, text=True, timeout=8)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["kind"] == "timeout"


def test_action_limit_stops_dependents_and_resets_only_the_local_budget(robo, backend, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, max_turns=2, images="none",
                     execution_limits=ExecutionLimits(max_actions=1))
    result = kit.execute_tool("python", {"code": MOVE + "\nrobo.release()"})
    assert result.result["failure"]["kind"] == "action_limit"
    assert result.result["recovery"]["motion_attempts"] == 1
    assert result.result["recovery"]["calls_left"] == 1
    assert not any(name == "release" for name, _ in backend.calls)
    assert not kit.execute_tool("python", {"code": "robo.release()"}).is_error
    assert kit.execute_tool("python", {"code": MOVE}).is_error


def test_episode_timeout_is_still_terminal_for_programs(robo, tmp_path):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path, program_timeout_s=0.01,
                     execution_limits=ExecutionLimits(timeout_s=1))
    result = kit.execute_tool("run_program", {"code": (
        "import time\ndef run(robo):\n    time.sleep(0.03)\n"
    )})
    assert result.is_error and result.result["failure"]["kind"] == "timeout"
    assert not result.result["recovery"]["retry_allowed"]
    assert result.result["recovery"]["stop_reason"] == "timeout"


def test_output_is_bounded_while_writing_and_keeps_parseable_failure(robo, tmp_path):
    stream = BoundedOutput(20)
    for _ in range(1000):
        assert stream.write("x" * 100) == 100
    assert stream.getvalue() == "x" * 20 and stream.omitted == 99980
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path,
                     execution_limits=ExecutionLimits(max_output_chars=100))
    result = kit.execute_tool("python", {"code": "print(chr(1) * 50000)\n1/0"})
    assert len(result.result["stdout"]) == 100
    assert result.result["output_omitted_chars"]["stdout"] == 49901
    payload = json.loads(result.content_blocks[0]["text"])
    assert payload["failure"]["type"] == "ZeroDivisionError"
    assert payload["observation"]["observation_id"]
    assert len(result.content_blocks[0]["text"]) <= MAX_TEXT_CHARS


def test_display_limit_and_old_image_provenance(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="on-demand",
                     execution_limits=ExecutionLimits(max_display_images=1))
    kit.execute_tool("python", {"code": "old = robo.image('wrist', resolution='low')"})
    result = kit.execute_tool("python", {"code": MOVE + "\ndisplay(old)\ndisplay(old)"})
    assert result.result["failure"]["kind"] == "output_limit"
    evidence = result.result["observation"]["images"]
    assert len(result.images) == len(evidence) == 1
    assert evidence[0]["source"] == "display" and evidence[0]["state_version"] is None
    assert not kit.execute_tool("python", {"code": "display(old)"}).is_error


def test_evidence_versions_include_failed_motion_and_correlate_images(robo, backend, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="on-demand", feedback="pure")
    first = kit.execute_tool("python", {"code": "robo.show('wrist')"}).result["observation"]
    backend.error_next = {"error": "controller connection lost"}
    failed = kit.execute_tool("python", {"code": MOVE}).result["observation"]
    assert first["state_version"] == 0 and failed["state_version"] == 1
    assert failed["actions"]["attempted"] == 1 and failed["actions"]["acknowledged"] == 0
    assert failed["prior_targets_may_be_stale"] and failed["step_id"] == first["step_id"]
    shown = kit.execute_tool("python", {"code": "robo.show('wrist')"}).result["observation"]
    assert shown["state_version"] == 1 and shown["actions"]["attempted"] == 0
    assert shown["images"][0]["state_version"] == 1
    assert shown["images"][0]["step_id"] == backend.frames
    assert len({first["observation_id"], failed["observation_id"], shown["observation_id"]}) == 3
    assert kit.turns[-1]["observation"] == shown


def test_image_capture_during_scene_change_is_not_marked_current(robo, backend, tmp_path):
    original = backend.artifact

    def changing_artifact(name, step=-1):
        backend.frames += 1
        return original(name, step)

    backend.artifact = changing_artifact
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="on-demand")
    result = kit.execute_tool("python", {"code": "robo.show('wrist')"})
    image = result.result["observation"]["images"][0]
    assert image["changed_during_read"] and image["step_id"] is None
    assert image["state_version"] is None
