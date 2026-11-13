"""Tests for ``codeaspolicy.external_runtime`` and ``play_episode(..., runtime='external')``."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from codeaspolicy import external_runtime as erun
from codeaspolicy import play
from codeaspolicy.arms import ArmToolkit
from codeaspolicy.evaluation.latency import summarize_interaction
from codeaspolicy.interaction_trace import InteractionTrace
from codeaspolicy.prompt import TaskCard


@pytest.fixture
def card_images(tmp_path: Path) -> list[Path]:
    from PIL import Image

    paths: list[Path] = []
    for name in ("agentview_high.png", "wrist_high.png"):
        path = tmp_path / "card" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (10, 20, 30)).save(path)
        paths.append(path)
    return paths


def test_external_runtime_cli_and_images_end_to_end(robo, backend, tmp_path, card_images, capsys):
    out_dir = tmp_path / "ep"
    toolkit = ArmToolkit(
        robo,
        arm="cells",
        workspace=out_dir / "workspace",
        images="on-demand",
        feedback="pure",
        max_turns=10,
    )
    holder: dict[str, object] = {}

    def run_server() -> None:
        holder["generation"] = erun.run_external_agent(
            "PROMPT TEXT",
            card_images,
            toolkit,
            model="gemini-4-argon-high",
            max_turns=10,
            budget_s=30.0,
            out_dir=out_dir,
            arm="cells",
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    assert erun.main(["status", "--out", str(out_dir)]) == 0
    status_out = json.loads(capsys.readouterr().out)
    assert status_out["ready"] is True
    assert status_out["model"] == "gemini-4-argon-high"
    assert status_out["arm"] == "cells"
    assert status_out["max_turns"] == 10
    assert Path(status_out["prompt_path"]).is_file()
    assert Path(status_out["card_dir"]).is_dir()

    code_pick = (
        "bowl = robo.segment('the black bowl')\n"
        "robo.move_to([bowl.world_xyz[0], bowl.world_xyz[1], 0.60])\n"
        "pick = robo.pi0_pick('pick up the bowl', max_chunks=8)\n"
        "print('PICK_OK', pick.success)\n"
        "robo.show('agentview')\n"
    )
    code_file = tmp_path / "step1.py"
    code_file.write_text(code_pick, encoding="utf-8")

    assert erun.main(["python", "--out", str(out_dir), "--code-file", str(code_file)]) == 0
    step1 = json.loads(capsys.readouterr().out)
    assert step1["ok"] is True
    assert "PICK_OK True" in step1["stdout"]
    assert step1["steps_used"] > 0
    assert step1["finished"] is False
    assert len(step1["images"]) == 1
    assert Path(step1["images"][0]).is_file()
    assert Path(step1["images"][0]).parent == (out_dir / "external_images").resolve()

    code_place = (
        "plate = robo.segment('the white plate')\n"
        "robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.60], gripper=robo.CLOSE)\n"
        "robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.47], gripper=robo.CLOSE)\n"
        "print('RELEASED', robo.release())\n"
    )
    assert erun.main(["python", "--out", str(out_dir), "--code", code_place]) == 0
    step2 = json.loads(capsys.readouterr().out)
    assert step2["ok"] is True
    assert step2["terminated"] is True
    assert backend.solved() is True

    assert (
        erun.main(["finish", "--out", str(out_dir), "--summary", "Solved by Argon external CLI"])
        == 0
    )
    fin = json.loads(capsys.readouterr().out)
    assert fin["ok"] is True and fin["finished"] is True
    assert fin["finish"]["status"] == "success"
    assert fin["finish"]["summary"] == "Solved by Argon external CLI"
    assert fin["finish"]["environment_terminated"] is True

    thread.join(timeout=5.0)
    assert not thread.is_alive()
    gen = holder["generation"]
    assert isinstance(gen, dict)
    assert gen["status"] == "completed" and gen["returncode"] == 0
    assert gen["model"] == "gemini-4-argon-high"
    assert gen["turns"] == 3
    assert gen["tool_calls"] == {"python": 2, "finish": 1}
    assert (out_dir / "workspace" / "cell_001.py").is_file()
    assert (out_dir / "workspace" / "cell_002.py").is_file()
    assert (out_dir / "messages.json").is_file()
    assert (out_dir / "external_events.jsonl").is_file()
    assert "Solved by Argon external CLI" in (out_dir / "last_message.md").read_text()


def test_play_episode_with_external_runtime_end_to_end(
    backend, robo, tmp_path, card_images, monkeypatch
):
    import codeaspolicy.mcp_server as mcp

    monkeypatch.chdir(tmp_path)

    def dump_card(toolkit, cell, out_dir):
        card = TaskCard(
            suite="fake",
            task="insert_key",
            seed=0,
            task_language="insert the key",
            object_names=[],
            images={"agentview": str(card_images[0]), "wrist": str(card_images[1])},
        )
        (out_dir / "card.json").write_text(card.to_json() + "\n", encoding="utf-8")
        return card

    host = SimpleNamespace(
        boot=lambda *args, **kwargs: (backend, []),
        dump_card=dump_card,
    )
    adapter = SimpleNamespace(name="libero", robot_cls=type(robo), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "_make_backend", lambda *args: backend)
    monkeypatch.setattr(play, "build_prompt", lambda *args, **kwargs: "EXTERNAL PROMPT")
    monkeypatch.setattr(play, "stop_all", lambda *args: None)
    monkeypatch.setattr(
        mcp,
        "ArmMcpServer",
        lambda *args, **kwargs: SimpleNamespace(
            start=lambda: "http://127.0.0.1:1/mcp", stop=lambda: None
        ),
    )

    out_dir = tmp_path / "episode_ext"
    driver_errors: list[Exception] = []

    def drive_agent() -> None:
        try:
            session = erun.wait_for_session(out_dir, timeout_s=10.0)
            assert session["ready"] is True
            assert session["model"] == "gemini-4-argon-high"
            r1 = erun.request_python(
                out_dir,
                (
                    "bowl = robo.segment('the black bowl')\n"
                    "robo.move_to([bowl.world_xyz[0], bowl.world_xyz[1], 0.60])\n"
                    "robo.pi0_pick('pick up the bowl', max_chunks=8)\n"
                    "robo.show('agentview')\n"
                ),
            )
            assert r1["ok"] is True and len(r1["images"]) == 1
            assert Path(r1["images"][0]).is_file()

            r2 = erun.request_python(
                out_dir,
                (
                    "plate = robo.segment('the white plate')\n"
                    "robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.60], gripper=robo.CLOSE)\n"
                    "robo.move_to([plate.world_xyz[0], plate.world_xyz[1], 0.47], gripper=robo.CLOSE)\n"
                    "robo.release()\n"
                ),
            )
            assert r2["ok"] is True and r2["terminated"] is True

            r3 = erun.request_finish(out_dir, "Key inserted and verified", status="success")
            assert r3["finished"] is True
        except Exception as exc:  # pragma: no cover
            driver_errors.append(exc)

    driver = threading.Thread(target=drive_agent, daemon=True)
    driver.start()

    cell = SimpleNamespace(
        suite="fake",
        task="insert_key",
        seed=0,
        cuda_device=None,
        max_episode_steps=300,
    )
    result = play.play_episode(
        cell,
        out_dir,
        arm="cells",
        model="gemini-4-argon-high",
        runtime="external",
        budget_s=30.0,
        max_turns=25,
        images="on-demand",
        feedback="pure",
        primitives="full",
    )
    driver.join(timeout=5.0)
    assert not driver_errors, driver_errors

    assert result["status"] == "completed"
    assert result["environment_success"] is True
    assert result["model"] == "gemini-4-argon-high"
    assert result["host"]["runtime"] == "external"
    assert result["tool_turns"] == 3
    assert result["finish"]["status"] == "success"
    assert result["finish"]["summary"] == "Key inserted and verified"
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "calls.jsonl").is_file()
    assert (out_dir / "tool_turns.json").is_file()
    assert (out_dir / "workspace" / "cell_001.py").is_file()
    assert (out_dir / "workspace" / "cell_002.py").is_file()

    latency_report = summarize_interaction(out_dir / "interaction_events.jsonl")
    assert latency_report["gaps"] == []
    assert latency_report["counts"]["python_calls"] == 2
    assert latency_report["counts"]["finish_calls"] == 1

    post_status = erun.request_status(out_dir, wait_s=1.0)
    assert post_status["finished"] is True
    assert post_status["environment_success"] is True


def test_external_runtime_repairs_after_error_and_enforces_max_turns(robo, tmp_path, card_images):
    out_dir = tmp_path / "ep_turns"
    trace = InteractionTrace(out_dir / "interaction_events.jsonl")
    toolkit = ArmToolkit(
        robo,
        arm="cells",
        workspace=out_dir / "workspace",
        max_turns=2,
        trace=trace,
    )
    holder: dict[str, object] = {}

    def run_server() -> None:
        holder["generation"] = erun.run_external_agent(
            "PROMPT",
            card_images,
            toolkit,
            model="gemini-4-argon-high",
            max_turns=2,
            budget_s=20.0,
            out_dir=out_dir,
            trace=trace,
            post_budget_grace_s=2.0,
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Cell 1 raises ZeroDivisionError; cell 2 must still execute cleanly without manual token plumbing.
    r1 = erun.request_python(out_dir, "1 / 0")
    assert r1["ok"] is False and "ZeroDivisionError" in (r1["exception"] or "")
    r2 = erun.request_python(out_dir, "recovered = 99\nprint(recovered)")
    assert r2["ok"] is True and "99" in r2["stdout"]

    # Cell 3 exceeds max_turns=2 and is refused.
    r3 = erun.request_python(out_dir, "print('over budget')")
    assert r3["ok"] is False
    assert r3["turn_budget_exhausted"] is True
    assert "call budget exhausted" in (r3["error"] or "")

    r4 = erun.request_finish(out_dir, "budget exhausted", status="stuck")
    assert r4["finished"] is True

    thread.join(timeout=5.0)
    gen = holder["generation"]
    assert isinstance(gen, dict)
    assert gen["max_turns_hit"] is True
    assert gen["turns"] == 4
    assert toolkit.finish["status"] == "stuck"
    assert toolkit.finish["summary"] == "budget exhausted"


def test_external_runtime_enforces_wall_clock_timeout(robo, tmp_path, card_images):
    out_dir = tmp_path / "ep_timeout"
    toolkit = ArmToolkit(robo, arm="cells", workspace=out_dir / "workspace")
    stopped: list[str] = []
    gen = erun.run_external_agent(
        "PROMPT",
        card_images,
        toolkit,
        model="gemini-4-argon-high",
        max_turns=10,
        budget_s=0.05,
        out_dir=out_dir,
        on_timeout=lambda: stopped.append("timeout"),
    )
    assert gen["status"] == "timeout"
    assert gen["returncode"] is None
    assert stopped == ["timeout"]


def test_external_runtime_rejects_non_localhost_and_wrong_token(robo, tmp_path, card_images):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    with pytest.raises(ValueError, match="localhost"):
        erun.run_external_agent(
            "PROMPT",
            card_images,
            toolkit,
            out_dir=tmp_path / "bad",
            host="0.0.0.0",
        )

    out_dir = tmp_path / "auth_ep"
    thread = threading.Thread(
        target=lambda: erun.run_external_agent(
            "PROMPT",
            card_images,
            toolkit,
            out_dir=out_dir,
            budget_s=15.0,
        ),
        daemon=True,
    )
    thread.start()
    session = erun.wait_for_session(out_dir, timeout_s=5.0)
    req = urllib.request.Request(
        f"{session['url']}/status",
        headers={"Authorization": "Bearer wrong-token"},
        method="GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        opener.open(req, timeout=5.0)
    assert exc_info.value.code == 403
    erun.request_finish(out_dir, "done", status="failure")
    thread.join(timeout=5.0)
