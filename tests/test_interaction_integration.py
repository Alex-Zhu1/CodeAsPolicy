"""P0 host timing observes execution without changing the code/tool contract."""

import json
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.interaction_trace import InteractionTrace
from pyrualean.play import _interaction_config


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_cell_trace_distinguishes_validation_execution_and_feedback(robo, tmp_path):
    path = tmp_path / "interaction_events.jsonl"
    trace = InteractionTrace(path)
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", trace=trace)
    response = toolkit.execute_tool("python", {"code": "x = 21\nx * 2"})
    events = _events(path)
    assert [e["event"] for e in events] == [
        "tool_received", "tool_started", "validation_started", "code_validated",
        "execution_started", "execution_finished", "tool_result_ready",
    ]
    assert len({e["local_call_id"] for e in events}) == 1
    assert len({e["session_id"] for e in events}) == 1
    assert events[-1]["observation_id"] == response.result["observation"]["observation_id"]
    assert response.result["value"] == "42"
    assert "local_call_id" not in response.result  # host metadata stays outside model feedback
    assert "x = 21" not in path.read_text()
    assert "code_ready" not in {e["event"] for e in events}


def test_compile_rejection_never_emits_execution_or_moves_robot(robo, backend, tmp_path):
    path = tmp_path / "trace.jsonl"
    toolkit = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "ws", trace=InteractionTrace(path),
    )
    result = toolkit.execute_tool(
        "python", {"code": "robo.move_to([0.1, 0.2, 0.5])\nreturn"},
    )
    events = _events(path)
    names = [e["event"] for e in events]
    assert "validation_failed" in names
    assert "code_validated" not in names and "execution_started" not in names
    assert not any(name == "move_to" for name, _ in backend.calls)
    assert result.is_error
    assert events[-1]["outcome"] == "error"


def test_partial_execution_error_keeps_its_call_and_result(robo, tmp_path):
    path = tmp_path / "trace.jsonl"
    toolkit = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "ws", trace=InteractionTrace(path),
    )
    result = toolkit.execute_tool(
        "python", {"code": "robo.move_to([0.05, -0.1, 0.6])\n1 / 0"},
        trace_context={"local_call_id": "host-1", "provider_call_id": "provider-1"},
    )
    events = _events(path)
    assert all(e["local_call_id"] == "host-1" for e in events)
    assert all(e["provider_call_id"] == "provider-1" for e in events)
    assert next(e for e in events if e["event"] == "execution_finished")["outcome"] == "error"
    assert result.is_error and len(robo._ledger) >= 1
    toolkit.execute_tool("python", {"code": "2 + 2"})
    second = _events(path)[len(events):]
    assert all(e["local_call_id"] != "host-1" for e in second)
    assert not any("provider_call_id" in e for e in second)


def test_recorded_openai_settings_follow_submitted_overrides_without_secrets():
    config = _interaction_config("openai", "requested", "xhigh", {
        "max_tokens": 4096,
        "extra_body": {
            "model": "gateway-model", "reasoning_effort": "high", "max_tokens": 8192,
            "api_key": "DO-NOT-RECORD", "messages": ["DO-NOT-RECORD"],
        },
    })
    assert config["requested_reasoning_effort"] == "xhigh"
    assert config["submitted_reasoning_effort"] == "high"
    assert config["submitted_model"] == "gateway-model" and config["max_tokens"] == 8192
    assert config["provider_confirmed"] is False
    assert "DO-NOT-RECORD" not in json.dumps(config)
    default = _interaction_config("openai", "model", "xhigh")
    assert default["submitted_reasoning_effort"] is None
    assert default["reasoning_source"] == "provider_default"
    assert _interaction_config("codex", "model", "high")["submitted_reasoning_effort"] == "high"
    assert _interaction_config("openai", "model", "high", {"max_tokens": 0})["max_tokens"] is None
    assert _interaction_config(
        "openai", "model", "high", {"extra_body": {"max_tokens": 0}},
    )["max_tokens"] == 0


def test_uncaught_interrupt_is_not_reported_as_success(robo, tmp_path):
    path = tmp_path / "trace.jsonl"
    toolkit = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "ws", trace=InteractionTrace(path),
    )
    with pytest.raises(KeyboardInterrupt):
        toolkit.execute_tool("python", {"code": "raise KeyboardInterrupt()"})
    events = _events(path)
    assert events[-1]["event"] == "execution_finished"
    assert events[-1]["outcome"] == "interrupted"
    assert not any(e["event"] == "tool_result_ready" for e in events)
    assert toolkit._trace_context == {}


def test_play_writes_sidecar_and_actual_configuration_without_model(
    backend, robo, tmp_path, monkeypatch,
):
    import pyrualean.mcp_server as mcp
    import pyrualean.play as play
    from pyrualean.prompt import TaskCard

    monkeypatch.chdir(tmp_path)  # restore the process cwd after play enters its workspace

    def dump_card(toolkit, cell, out_dir):
        card = TaskCard(suite="fake", task="fake", seed=1, task_language="test", object_names=[])
        (out_dir / "card.json").write_text(card.to_json())
        return card

    host = SimpleNamespace(
        boot=lambda *args, **kwargs: (backend, []),
        dump_card=dump_card,
    )
    adapter = SimpleNamespace(name="libero", robot_cls=type(robo), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "_make_backend", lambda *args: backend)
    monkeypatch.setattr(play, "build_prompt", lambda *args, **kwargs: "test prompt")
    monkeypatch.setattr(play, "stop_all", lambda *args: None)
    monkeypatch.setattr(mcp, "ArmMcpServer", lambda *args, **kwargs: SimpleNamespace(
        start=lambda: "http://unused.invalid", stop=lambda: None,
    ))
    replies = iter([
        ("c1", "python", {"code": 'answer = 42\nplan(goal="Inspect", done_when="read image")'}),
        ("c2", "finish", {"status": "failed", "summary": "fixture complete"}),
    ])

    def transport(url, body, timeout):
        call_id, name, arguments = next(replies)
        return 200, {"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments),
            }}],
        }}]}

    out = tmp_path / "episode"
    cell = SimpleNamespace(
        suite="fake", task="fake", seed=1, cuda_device=None, max_episode_steps=20,
    )
    result = play.play_episode(
        cell, out, arm="cells", model="model", runtime="openai",
        base_url="http://unused.invalid/v1", images="none", reasoning_effort="xhigh",
        openai_options={"transport": transport, "extra_body": {"reasoning_effort": "high"}},
    )
    assert result["error"] is None
    assert result["tool_turns"] == 2
    plan = json.loads((out / "execution_plan.json").read_text())
    assert plan["task_card"]["path"] == str(out / "card.json")
    assert plan["goal"] == "Inspect" and plan["context"]["turn"] == 1
    record = json.loads((out / "result.json").read_text())["interaction"]
    assert record["configuration"]["submitted_reasoning_effort"] == "high"
    assert record["capabilities"]["provider_first_token"] is False
    events = _events(out / "interaction_events.jsonl")
    assert events[-1]["event"] == "episode_finished"
    assert any(e["event"] == "request_started" for e in events)
    assert len([e for e in events if e["event"] == "feedback_ready"]) == 2

    from pyrualean.evaluation.latency import summarize_interaction

    report = summarize_interaction(out / "interaction_events.jsonl")
    assert report["gaps"] == []
    assert report["counts"]["requests_with_unknown_tool_count"] == 0
    assert report["metrics"]["execution_s"]["n"] == 1
    assert report["metrics"]["request_to_tool_arguments_complete_s"]["n"] == 2
