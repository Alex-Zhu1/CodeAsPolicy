"""Confirmed stuck dispatches recovery without resetting the live episode."""

import json

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.stuck_recovery import StuckRecoveryConfig
from pyrualean.supervision import MonitorConfig, VisualSupervisor

SKILL = "Reassess causal prerequisites; defer later checks with a reactivation gate."


def toolkit(robo, tmp_path, **kwargs):
    return ArmToolkit(
        robo,
        arm="cells",
        workspace=tmp_path / "ws",
        stuck_recovery=StuckRecoveryConfig(SKILL, max_rounds=2),
        **kwargs,
    )


def stuck(kit):
    return kit.execute_tool("finish", {"status": "stuck", "summary": "Grasp missed twice."})


def test_dispatch_only_after_explicit_stuck_keeps_scene_namespace_and_budget(
    robo, backend, tmp_path
):
    kit = toolkit(robo, tmp_path, max_turns=4, images="none", feedback="pure")
    kit.execute_tool("python", {"code": "saved = 42\nrobo.move_to([0.05, -0.1, 0.6])"})
    frames, used = backend.frames, kit._cell_index
    assert not kit.stuck_recovery_events
    result = stuck(kit)
    assert kit.finish is None and result.result["episode_finished"] is False
    assert result.result["status"] == "stuck_confirmed" and not result.is_error
    assert result.result["recovery"]["instruction"] == SKILL
    assert result.result["recovery"]["calls_left"] == 3
    assert backend.frames == frames and kit._cell_index == used and not backend.cancelled
    assert len(result.images) == len(robo.CAMERAS)
    resumed = kit.execute_tool("python", {"code": "print(saved)"})
    assert resumed.result["stdout"] == "42\n" and kit._cell_index == used + 1
    assert kit.turns[1]["recovery"]["event"] == "recovery_dispatched"
    assert json.loads((tmp_path / "stuck_recovery.jsonl").read_text())["step"] == frames


def test_supervisor_pause_does_not_dispatch_skill_or_reset_supervisor(robo, tmp_path):
    supervisor = VisualSupervisor(
        MonitorConfig(mode="pause", url="http://localhost:1"), tmp_path / "monitor"
    )
    kit = toolkit(robo, tmp_path, supervisor=supervisor)
    result = kit.execute_tool("python", {"code": "robo.move_to([0.05,-0.1,0.6])"})
    assert result.result["execution_paused"] and not kit.stuck_recovery_events
    stuck(kit)
    assert supervisor.paused  # next Python cell, not finish, resets the cell latch
    resumed = kit.execute_tool("python", {"code": "robo.move_to([0.05,-0.1,0.6])"})
    assert resumed.result["execution_paused"]  # still requires watch
    assert len(kit.stuck_recovery_events) == 1


def test_recovery_dispatch_keeps_unresolved_dependencies(robo, tmp_path):
    supervisor = VisualSupervisor(
        MonitorConfig(mode="pause", url="http://localhost:1"), tmp_path / "monitor"
    )
    kit = toolkit(robo, tmp_path, supervisor=supervisor)
    kit.execute_tool("python", {"code":
        "watch('align', {'aligned': 'Is the key aligned with the opening?'})\n"
        "defer_fact('aligned', until='insert', reason='Insertion requires alignment')"})
    ledger = supervisor.read_evidence()
    assert not stuck(kit).result["episode_finished"]
    assert supervisor.read_evidence() == ledger
    result = kit.execute_tool("python", {"code":
        "watch('insert', {'aligned': 'Is the key aligned with the opening?'})\n"
        "robo.move_to([0.05, -0.1, 0.6])"})
    assert result.result["execution_paused"] and not robo._ledger.stateful_calls()
    assert result.result["supervision"]["blocked"]["aligned"]["value"] == "unknown"


def test_round_limit_ends_episode_and_is_audited(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    assert not stuck(kit).result["episode_finished"]
    assert not stuck(kit).result["episode_finished"]
    assert stuck(kit).result["episode_finished"]
    assert kit.finish["status"] == "stuck"
    assert kit.stuck_recovery_events[-1]["stop_reason"] == "recovery_round_limit"
    assert stuck(kit).result["episode_finished"] and len(kit.stuck_recovery_events) == 3


@pytest.mark.parametrize(
    "reason", ["truncated", "terminated", "call_budget", "wall_budget", "cancelled"]
)
def test_terminal_or_exhausted_episode_never_recovers(robo, tmp_path, reason):
    kit = toolkit(robo, tmp_path, max_turns=1, program_timeout_s=10)
    if reason == "truncated":
        robo._truncated = True
    elif reason == "terminated":
        robo._done = True
    elif reason == "call_budget":
        kit._cell_index = 1
    elif reason == "wall_budget":
        kit._recovery_started -= 11
    else:
        robo._stop_reason = reason
    assert stuck(kit).result["episode_finished"] and kit.finish["status"] == "stuck"
    assert kit.stuck_recovery_events[-1]["stop_reason"] == reason


def test_disabled_and_nonstuck_finish_remain_terminal(robo, tmp_path):
    disabled = ArmToolkit(robo, arm="cells", workspace=tmp_path / "disabled")
    assert "episode_finished" not in stuck(disabled).result
    assert disabled.finish["status"] == "stuck" and not disabled.stuck_recovery_events
    kit = toolkit(robo, tmp_path)
    assert kit.execute_tool("finish", {"status": "success", "summary": "Done"}).result[
        "episode_finished"
    ]
    assert not kit.stuck_recovery_events


def test_large_stuck_report_preserves_skill_and_valid_json(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    result = kit.execute_tool("finish", {"status": "stuck", "summary": "x" * 30000})
    payload = json.loads(result.content_blocks[0]["text"])
    assert payload["episode_finished"] is False
    assert payload["recovery"]["instruction"] == SKILL


def test_empty_stuck_summary_does_not_dispatch(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    assert kit.execute_tool("finish", {"status": "stuck", "summary": " "}).is_error
    assert kit.finish is None and not kit.stuck_recovery_events


def test_initial_prompt_contains_trigger_not_recovery_procedure():
    from types import SimpleNamespace

    from pyrualean.play import build_prompt

    prompt = build_prompt(
        "cells",
        SimpleNamespace(max_env_steps=300, render=lambda: "TASK"),
        budget_s=30,
        max_programs=None,
        stuck_recovery=StuckRecoveryConfig("PRIVATE_RECOVERY_PROCEDURE"),
    )
    assert "Confirmed-stuck recovery" in prompt
    assert "PRIVATE_RECOVERY_PROCEDURE" not in prompt


@pytest.mark.parametrize("instructions,rounds", [("", 1), ("x" * 12001, 1), ("ok", 0)])
def test_invalid_config_is_rejected(instructions, rounds):
    with pytest.raises(ValueError):
        StuckRecoveryConfig(instructions, rounds)
