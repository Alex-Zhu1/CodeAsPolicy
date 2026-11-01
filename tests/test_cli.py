"""Offline CLI commands."""

from __future__ import annotations

import json

from pyrualean.cli import main


def test_api_and_prompt_render_offline(capsys):
    assert main(["api"]) == 0
    assert "robo.move_to(" in capsys.readouterr().out
    assert main(["prompt", "--system-only"]) == 0
    out = capsys.readouterr().out
    assert "# Operating knowledge" in out and "def run(robo)" in out


def test_summarize_reads_result_dirs(tmp_path, capsys):
    directory = tmp_path / "ep"
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps(
            {
                "schema": "pyrualean-episode-v2",
                "suite": "libero_10",
                "task": 0,
                "seed": 1,
                "environment_success": False,
                "status": "error",
                "calls": 3,
                "env_steps": 40,
                "policy_wall_s": 12.5,
                "error": {"type": "ValueError", "message": "bad"},
            }
        )
    )
    assert main(["summarize", str(directory), "--json"]) == 0
    out = capsys.readouterr().out
    assert "libero_10:0" in out and '"native_sr": 0.0' in out


def test_provider_overrides_shape():
    from pyrualean.generate import provider_overrides

    assert provider_overrides(None, None) == []
    overrides = provider_overrides("https://gw.example.com", "MY_KEY")
    joined = " ".join(overrides)
    assert 'model_provider="pyrualean"' in joined
    assert 'model_providers.pyrualean.base_url="https://gw.example.com/v1"' in joined
    assert 'model_providers.pyrualean.wire_api="responses"' in joined
    assert 'model_providers.pyrualean.env_key="MY_KEY"' in joined


def test_execution_options_reach_the_host(tmp_path, monkeypatch):
    from pyrualean import cli

    captured = []

    def run_host(argv, **kwargs):
        captured.extend(argv)
        (tmp_path / "result.json").write_text(json.dumps({"status": "completed"}))
        return 0

    monkeypatch.setattr(cli, "run_host", run_host)
    main(["play", "--arm", "cells", "--task", "0", "--out", str(tmp_path),
          "--code-timeout-s", "7.5", "--code-max-actions", "4", "--code-output-chars", "300"])
    for key, value in (("--code-timeout-s", "7.5"), ("--code-max-actions", "4"),
                       ("--code-output-chars", "300")):
        assert captured[captured.index(key) + 1] == value
