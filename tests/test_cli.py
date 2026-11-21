"""Offline CLI commands."""

from __future__ import annotations

import json

from codeaspolicy.cli import main


def test_api_and_prompt_render_offline(capsys):
    assert main(["api"]) == 0
    assert "robo.move_to(" in capsys.readouterr().out
    assert main(["api", "--robot", "robodojo_xpl_rgb", "--primitives", "no-sam"]) == 0
    rd_out = capsys.readouterr().out
    assert "robo.execute_motion(" in rd_out and "robo.delta_grasp(" in rd_out
    assert main(["prompt", "--system-only"]) == 0
    out = capsys.readouterr().out
    assert "# Operating knowledge" in out and "def run(robo)" in out


def test_readme_robo_calls_cli_commands_and_paths_exist():
    import ast
    import re
    from pathlib import Path

    from codeaspolicy.robots import ROBOTS, get_adapter

    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    # 1. Every python fenced block in README must parse cleanly and only call real robo.* members.
    known_members: set[str] = set()
    for name in ROBOTS:
        cls = get_adapter(name).robot_cls
        known_members.update(m for m in dir(cls) if not m.startswith("_"))

    py_blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    assert len(py_blocks) >= 2
    for block in py_blocks:
        ast.parse(block)

    for attr in re.findall(r"\brobo\.([A-Za-z_][A-Za-z0-9_]*)", readme):
        assert attr in known_members, f"README references non-existent robo.{attr}"

    # 2. Every `codeaspolicy <subcommand>` named in README must be a real CLI subcommand.
    valid_cmds = {"api", "prompt", "card", "generate", "run", "play", "summarize", "compare", "valen-eval"}
    for cmd in re.findall(r"`codeaspolicy\s+([a-z][a-z0-9_-]*)`", readme):
        assert cmd in valid_cmds, f"README references unknown CLI command: codeaspolicy {cmd}"

    # 3. Every repository file/dir entry in the Repository Structure tree and markdown links must exist.
    for rel in re.findall(r"\]\(((?!https?://|#)[^)]+)\)", readme):
        assert (repo_root / rel).exists(), f"README broken relative link: {rel}"
    for tree_entry in re.findall(r"[├└]──\s+([A-Za-z0-9_./-]+)", readme):
        if tree_entry.startswith(("src/", "integrations/", "docs/", "tests/", "pyproject.toml")):
            assert (repo_root / tree_entry).exists(), f"README tree lists missing path: {tree_entry}"
    for sub_entry in re.findall(r"│\s+[├└]──\s+([A-Za-z0-9_./-]+)", readme):
        assert (repo_root / "src/codeaspolicy" / sub_entry).exists(), (
            f"README src/codeaspolicy tree lists missing path: {sub_entry}"
        )


def test_summarize_reads_result_dirs(tmp_path, capsys):
    directory = tmp_path / "ep"
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps(
            {
                "schema": "codeaspolicy-episode-v2",
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
    from codeaspolicy.generate import provider_overrides

    assert provider_overrides(None, None) == []
    overrides = provider_overrides("https://gw.example.com", "MY_KEY")
    joined = " ".join(overrides)
    assert 'model_provider="codeaspolicy"' in joined
    assert 'model_providers.codeaspolicy.base_url="https://gw.example.com/v1"' in joined
    assert 'model_providers.codeaspolicy.wire_api="responses"' in joined
    assert 'model_providers.codeaspolicy.env_key="MY_KEY"' in joined


def test_execution_options_reach_the_host(tmp_path, monkeypatch):
    from codeaspolicy import cli

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
