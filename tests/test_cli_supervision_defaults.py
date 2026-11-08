"""Both CLI entry points keep visual supervision opt-in and preserve explicit modes."""

import json

import pytest

from pyrualean import cli, play
from pyrualean.arms import ArmToolkit
from pyrualean.stuck_recovery import recovery_instructions
from pyrualean.supervision import DEFAULT_VALEN_URL, planning_instructions


def capture_episode(monkeypatch):
    captured = []

    def episode(cell, out_dir, **kwargs):
        captured.append(dict(cell=cell, **kwargs))
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "completed"}
        (out_dir / "result.json").write_text(json.dumps(result))
        return result

    def run_host(argv, **kwargs):
        assert argv[1:3] == ["-m", "pyrualean.play"]
        return play.main(argv[3:])

    monkeypatch.setattr(play, "play_episode", episode)
    monkeypatch.setattr(cli, "run_host", run_host)
    return captured


@pytest.mark.parametrize("entry", ["outer", "host"])
@pytest.mark.parametrize("recovery_enabled", [False, True])
def test_play_defaults_and_explicit_bundled_recovery(
    tmp_path, monkeypatch, robo, entry, recovery_enabled
):
    monkeypatch.delenv("PYRUALEAN_VALEN_URL", raising=False)
    monkeypatch.delenv("PYRUALEAN_RUNTIME", raising=False)
    monkeypatch.delenv("PYRUALEAN_PRIMITIVES", raising=False)
    captured = capture_episode(monkeypatch)
    args = ["--robot", "robodojo_xpl_rgb", "--suite", "arx_x5",
            "--task", "insert_key", "--out", str(tmp_path / "run")]
    if recovery_enabled:
        args += ["--stuck-recovery"]
    assert (cli.main(["play", *args]) if entry == "outer" else play.main(args)) == 0
    config = captured[0]
    assert config["cell"].task == "insert_key" and config["arm"] == "cells"
    assert config["monitor"].mode == "off" and config["monitor"].url is None
    assert config["runtime"] == "codex" and config["model"] == "gpt-6-astra"
    assert config["reasoning_effort"] == "high"
    assert config["budget_s"] == 5400 and config["max_turns"] == 100
    assert config["images"] == "on-demand" and config["feedback"] == "pure"
    assert config["primitives"] == "no-vla"
    recovery = config["stuck_recovery"]
    assert config["code_limits"].timeout_s == 120

    from pyrualean.prompt import TaskCard

    card = TaskCard(suite="test", task=0, seed=0, task_language="Insert the key", object_names=[])
    prompt = play.build_prompt("cells", card, budget_s=100, max_programs=None,
                               monitor=config["monitor"], stuck_recovery=recovery)
    assert planning_instructions() not in prompt and "watch(phase" not in prompt
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", stuck_recovery=recovery)
    response = kit.execute_tool("finish", {"status": "stuck", "summary": "Grasp missed"})
    if recovery_enabled:
        assert recovery.instructions == recovery_instructions() and recovery.max_rounds == 2
        assert "Confirmed-stuck recovery" in prompt and recovery.instructions not in prompt
        assert response.result["episode_finished"] is False
        assert response.result["recovery"]["instruction"] == recovery.instructions
    else:
        assert recovery is None and "Confirmed-stuck recovery" not in prompt
        assert kit.finish["status"] == "stuck" and not kit.stuck_recovery_events


@pytest.mark.parametrize("entry", ["outer", "host"])
@pytest.mark.parametrize("options,mode,recovery_enabled", [
    (["--monitor-mode", "off", "--no-stuck-recovery"], "off", False),
    (["--monitor-mode", "off"], "off", False),
    (["--no-stuck-recovery"], "off", False),
    (["--monitor-mode", "pause"], "pause", False),
    (["--monitor-mode", "shadow"], "shadow", False),
    (["--monitor-mode", "pause", "--stuck-recovery"], "pause", True),
    (["--stuck-recovery-rounds", "4"], "off", False),
    (["--arm", "program"], "off", False),
])
def test_explicit_modes_survive_outer_to_host_forwarding(
    tmp_path, monkeypatch, entry, options, mode, recovery_enabled
):
    monkeypatch.delenv("PYRUALEAN_VALEN_URL", raising=False)
    captured = capture_episode(monkeypatch)
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--out", str(tmp_path), *options]
    assert (cli.main(["play", *args]) if entry == "outer" else play.main(args)) == 0
    config = captured[0]
    assert config["monitor"].mode == mode
    assert (config["stuck_recovery"] is not None) == recovery_enabled
    if mode == "off":
        assert config["monitor"].url is None
    else:
        assert config["monitor"].url == DEFAULT_VALEN_URL


@pytest.mark.parametrize("entry", ["outer", "host"])
@pytest.mark.parametrize("override", [False, True])
def test_valen_address_parameter_overrides_environment(tmp_path, monkeypatch, entry, override):
    monkeypatch.setenv("PYRUALEAN_VALEN_URL", "http://127.0.0.1:28001/predict")
    captured = capture_episode(monkeypatch)
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--out", str(tmp_path),
            "--monitor-mode", "pause"]
    if override:
        args += ["--valen-url", "http://127.0.0.1:38001/predict"]
    assert (cli.main(["play", *args]) if entry == "outer" else play.main(args)) == 0
    expected = 38001 if override else 28001
    assert captured[0]["monitor"].url == f"http://127.0.0.1:{expected}/predict"


@pytest.mark.parametrize("entry", ["outer", "host"])
def test_valen_address_alone_does_not_enable_supervision(tmp_path, monkeypatch, entry):
    monkeypatch.setenv("PYRUALEAN_VALEN_URL", "http://127.0.0.1:28001/predict")
    captured = capture_episode(monkeypatch)
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--out", str(tmp_path),
            "--valen-url", "http://127.0.0.1:38001/predict"]
    assert (cli.main(["play", *args]) if entry == "outer" else play.main(args)) == 0
    assert captured[0]["monitor"].mode == "off"
    assert captured[0]["monitor"].url is None


@pytest.mark.parametrize("entry", ["outer", "host"])
def test_custom_recovery_and_rounds_override_defaults(tmp_path, monkeypatch, entry):
    skill = tmp_path / "custom.md"
    skill.write_text("Custom recovery procedure")
    captured = capture_episode(monkeypatch)
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--out", str(tmp_path),
            "--stuck-recovery-skill", str(skill), "--stuck-recovery-rounds", "3"]
    assert (cli.main(["play", *args]) if entry == "outer" else play.main(args)) == 0
    recovery = captured[0]["stuck_recovery"]
    assert recovery.instructions == skill.read_text() and recovery.max_rounds == 3


def test_builtin_recovery_rounds_are_forwarded(tmp_path, monkeypatch):
    captured = capture_episode(monkeypatch)
    assert cli.main(["play", "--suite", "libero_spatial_swap", "--task", "0",
                     "--out", str(tmp_path), "--stuck-recovery",
                     "--stuck-recovery-rounds", "4"]) == 0
    recovery = captured[0]["stuck_recovery"]
    assert recovery.max_rounds == 4 and recovery.instructions == recovery_instructions()


@pytest.mark.parametrize("entry", ["outer", "host"])
@pytest.mark.parametrize("enable", [["--stuck-recovery"],
                                    ["--stuck-recovery-skill", "missing.md"]])
def test_conflicting_recovery_flags_are_rejected_before_launch(
    tmp_path, monkeypatch, entry, enable
):
    captured = capture_episode(monkeypatch)
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--out", str(tmp_path),
            "--no-stuck-recovery", *enable]
    with pytest.raises(SystemExit) as error:
        cli.main(["play", *args]) if entry == "outer" else play.main(args)
    assert error.value.code == 2 and not captured


@pytest.mark.parametrize("entry", ["outer", "host"])
@pytest.mark.parametrize("feature", ["monitor", "recovery", "bundled_recovery"])
def test_program_cannot_silently_accept_unsupported_features(
    tmp_path, monkeypatch, entry, feature
):
    captured = capture_episode(monkeypatch)
    skill = tmp_path / "custom.md"
    skill.write_text("Custom recovery")
    options = {"monitor": ["--monitor-mode", "pause"],
               "recovery": ["--stuck-recovery-skill", str(skill)],
               "bundled_recovery": ["--stuck-recovery"]}[feature]
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--arm", "program",
            "--out", str(tmp_path), *options]
    with pytest.raises(ValueError, match="only --arm cells"):
        cli.main(["play", *args]) if entry == "outer" else play.main(args)
    assert not captured


@pytest.mark.parametrize("entry", ["outer", "host"])
def test_explicit_launch_overrides_survive_environment_and_host_defaults(
    tmp_path, monkeypatch, entry
):
    monkeypatch.setenv("PYRUALEAN_RUNTIME", "openai")
    monkeypatch.setenv("PYRUALEAN_PRIMITIVES", "no-vla")
    captured = capture_episode(monkeypatch)
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--out", str(tmp_path),
            "--runtime", "codex",
            "--primitives", "full", "--reasoning", "xhigh", "--budget-s", "1200",
            "--max-turns", "0", "--images", "on-motion", "--feedback", "rich"]
    assert (cli.main(["play", *args]) if entry == "outer" else play.main(args)) == 0
    config = captured[0]
    assert config["runtime"] == "codex" and config["primitives"] == "full"
    assert config["max_turns"] is None and config["budget_s"] == 1200
    assert config["reasoning_effort"] == "xhigh"
    assert config["images"] == "on-motion" and config["feedback"] == "rich"


@pytest.mark.parametrize("entry", ["outer", "host"])
def test_negative_turn_budget_is_rejected_before_launch(tmp_path, monkeypatch, entry):
    captured = capture_episode(monkeypatch)
    args = ["--suite", "libero_spatial_swap", "--task", "0", "--out", str(tmp_path),
            "--max-turns", "-1"]
    with pytest.raises(SystemExit) as error:
        cli.main(["play", *args]) if entry == "outer" else play.main(args)
    assert error.value.code == 2 and not captured
