"""Offline checks of the Codex event audit in ``codeaspolicy.play``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from codeaspolicy.generate import CODEX_CATALOG, codex_catalog_flags
from codeaspolicy.hosts.common import RpentBackend
from codeaspolicy.play import (
    _agent_environment,
    _audit_events,
    _codex_argv,
    _codex_context_config,
    _make_backend,
    _usage_from_rollouts,
)


def test_codex_catalog_only_for_ids_codex_looks_up_as_gpt_6_astra():
    astra = next(
        entry
        for entry in json.loads(CODEX_CATALOG.read_text())["models"]
        if entry["slug"] == "gpt-6-astra"
    )
    assert "tool_mode" not in astra and astra["default_reasoning_summary"] == "auto"
    flag = f"model_catalog_json={json.dumps(str(CODEX_CATALOG))}"
    assert CODEX_CATALOG.is_absolute() and CODEX_CATALOG.is_file()
    for model in ("gpt-6-astra", "openai/gpt-6-astra"):
        assert codex_catalog_flags(model) == ["-c", flag]
    for model in ("openai/openai/gpt-6-astra", "gpt-5.5", "claude-opus-5-5"):
        assert codex_catalog_flags(model) == []
    common = dict(
        codex_bin="codex",
        reasoning_effort="high",
        mcp_url="http://127.0.0.1:1/mcp",
        workspace=Path("/w"),
        last_message=Path("/w/last.md"),
        images=[],
        base_url=None,
        api_key_env=None,
    )
    assert flag in _codex_argv(model="gpt-6-astra", **common)
    assert not any("model_catalog_json" in arg for arg in _codex_argv(model="gpt-5.5", **common))


def test_agent_environment_per_runtime(monkeypatch):
    for key in ("NODE_EXTRA_CA_CERTS", "NO_PROXY", "CODEX_HOME", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:3128")
    monkeypatch.setenv("no_proxy", "corp.internal")
    monkeypatch.setenv("ZDOTDIR", "/somewhere/.zdot")
    monkeypatch.setenv("CLAUDE_RELAY_BASE_URL", "https://relay.example/claude")
    monkeypatch.setenv("CLAUDE_RELAY_KEY_FILE", "/keys/relay")
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/certs/ca.pem")
    monkeypatch.setenv("MY_KEY", "secret")
    codex = _agent_environment(codex_home="/tmp/codex-home", api_key_env="MY_KEY")
    assert codex["SHELL"] == "/bin/bash" and codex["CODEX_HOME"] == "/tmp/codex-home"
    assert codex["MY_KEY"] == "secret" and codex["http_proxy"] == "http://127.0.0.1:3128"
    assert codex["NO_PROXY"] == codex["no_proxy"] == "corp.internal,localhost,127.0.0.1,::1"
    assert not {"ZDOTDIR", "CLAUDE_RELAY_BASE_URL", "NODE_EXTRA_CA_CERTS"} & set(codex)
    claude = _agent_environment(codex_home="/tmp/codex-home", api_key_env=None, runtime="claude")
    assert claude["CLAUDE_RELAY_BASE_URL"] == "https://relay.example/claude"
    assert claude["CLAUDE_RELAY_KEY_FILE"] == "/keys/relay"
    assert claude["NODE_EXTRA_CA_CERTS"] == "/certs/ca.pem"
    assert claude["http_proxy"] == "http://127.0.0.1:3128" and claude["PATH"] == "/usr/bin"
    assert claude["NO_PROXY"] == claude["no_proxy"] == "corp.internal,localhost,127.0.0.1,::1"
    assert not {"ZDOTDIR", "CODEX_HOME", "SHELL", "MY_KEY"} & set(claude)


def _events(path, items):
    lines = [{"type": "thread.started", "thread_id": "t1"}]
    lines += [{"type": "item.completed", "item": item} for item in items]
    lines.append({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}})
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_fallback_metadata_notice_is_a_warning_not_an_error(tmp_path):
    path = tmp_path / "codex_events.jsonl"
    _events(
        path,
        [
            {
                "type": "error",
                "message": "Model metadata for `openai/openai/gpt-6-astra` not found. "
                "Defaulting to fallback metadata; this can degrade performance",
            },
            {"type": "mcp_tool_call", "tool": "python"},
            {"type": "error", "message": "stream disconnected before completion"},
        ],
    )
    audit = _audit_events(path)
    assert audit["warnings"] and "fallback metadata" in audit["warnings"][0]
    assert audit["errors"] == ["stream disconnected before completion"]
    assert audit["tool_calls"] == {"python": 1}
    assert audit["input_tokens"] == 10 and audit["total_tokens"] == 12


def test_codex_context_config_reads_home_and_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        'model = "x"\nmodel_context_window = 272000\nmodel_auto_compact_token_limit = 945000\n'
    )
    monkeypatch.delenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", raising=False)
    assert _codex_context_config(None)["model_auto_compact_token_limit"] is None
    values = _codex_context_config(str(home))
    assert values["model_auto_compact_token_limit"] == 945000
    assert values["model_context_window"] == 272000
    assert values["source"] == "config.toml"
    monkeypatch.setenv("CODEX_AUTO_COMPACT_TOKEN_LIMIT", "80000")
    values = _codex_context_config(str(home))
    assert values["model_auto_compact_token_limit"] == 80000
    assert values["source"] == "CODEX_AUTO_COMPACT_TOKEN_LIMIT"


def test_usage_from_rollouts_counts_compactions(tmp_path):
    home = tmp_path / "home"
    rollout = home / "sessions" / "2026" / "09" / "22"
    rollout.mkdir(parents=True)
    lines = [
        {"type": "session_meta", "payload": {}},
        {"type": "compacted", "payload": {"message": "summary"}},
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 50,
                        "output_tokens": 7,
                        "reasoning_output_tokens": 3,
                        "total_tokens": 107,
                    }
                },
            },
        },
    ]
    (rollout / "rollout-2026-09-22T00-00-00-abc.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    usage = _usage_from_rollouts(str(home), ["abc"])
    assert usage["compactions"] == 1
    assert usage["input_tokens"] == 100 and usage["total_tokens"] == 107
    assert _usage_from_rollouts(str(home), ["missing"]) is None


def test_make_backend_prefers_the_hosts_factory():
    class _Mine(RpentBackend):
        pass

    toolkit = object()
    assert type(_make_backend(SimpleNamespace(make_backend=_Mine), toolkit)) is _Mine
    assert type(_make_backend(SimpleNamespace(), toolkit)) is RpentBackend


def test_external_and_argon_runtime_configuration(monkeypatch, tmp_path, backend, robo):
    import codeaspolicy.mcp_server as mcp
    from codeaspolicy import play
    from codeaspolicy.prompt import TaskCard

    assert "external" in play.RUNTIMES
    ext_cfg = play._interaction_config("external", "gemini-4-argon-high", "high")
    assert ext_cfg["protocol"] == "external_http"
    assert ext_cfg["submitted_reasoning_effort"] == "high"
    assert ext_cfg["reasoning_source"] == "external_agent"

    argon_openai = play._interaction_config(
        "openai",
        "gemini-4-argon-high",
        "xhigh",
        {"extra_body": {"reasoning_effort": "xhigh"}},
    )
    assert argon_openai["submitted_reasoning_effort"] == "high"

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_BASE_URL", "http://gemini-gateway.test/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

    def dump_card(toolkit, cell, out_dir):
        card = TaskCard(suite="fake", task="insert_key", seed=0, task_language="insert", object_names=[])
        (out_dir / "card.json").write_text(card.to_json())
        return card

    host = SimpleNamespace(boot=lambda *args, **kwargs: (backend, []), dump_card=dump_card)
    adapter = SimpleNamespace(name="libero", robot_cls=type(robo), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "_make_backend", lambda *args: backend)
    monkeypatch.setattr(play, "build_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(play, "stop_all", lambda *args: None)
    monkeypatch.setattr(
        mcp,
        "ArmMcpServer",
        lambda *args, **kwargs: SimpleNamespace(start=lambda: "http://127.0.0.1:1/mcp", stop=lambda: None),
    )

    def transport(url, body, timeout):
        assert url == "http://gemini-gateway.test/v1/chat/completions"
        assert body["model"] == "gemini-4-argon-high"
        return 200, {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "finish", "arguments": '{"summary": "ok"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": None},
        }

    cell = SimpleNamespace(suite="fake", task="insert_key", seed=0, cuda_device=None, max_episode_steps=300)
    res = play.play_episode(
        cell,
        tmp_path / "argon_openai",
        runtime="openai",
        model="gemini-4-argon-high",
        images="none",
        openai_options={"transport": transport},
    )
    assert res["status"] == "completed"
    assert res["host"]["base_url"] == "http://gemini-gateway.test/v1"
    assert res["finish"]["status"] == "success"
    assert res["finish"]["summary"] == "ok"

