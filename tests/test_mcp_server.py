"""MCP tracing preserves responses and distinguishes receipt from queued execution."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from pyrualean import mcp_server
from pyrualean.interaction_trace import InteractionTrace


def _run(coroutine):
    # Own the executor lifecycle instead of asyncio.run's asynchronous shutdown.
    loop = asyncio.new_event_loop()

    def wake():
        # Some sandboxes block the socketpair wakeup used by worker callbacks.
        # Keep real threads and locks, but ensure queued callbacks get serviced.
        loop.call_later(0.01, wake)

    wake()
    with ThreadPoolExecutor(max_workers=2) as executor:
        loop.set_default_executor(executor)
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()


def _handler(monkeypatch, toolkit):
    """Exercise the real SDK request handler without opening a network listener."""
    pytest.importorskip("mcp")
    from mcp import types
    from mcp.server import streamable_http_manager

    captured = []

    def capture_manager(*, app, **kwargs):
        captured.append(app)
        return SimpleNamespace()

    monkeypatch.setattr(streamable_http_manager, "StreamableHTTPSessionManager", capture_manager)
    mcp_server._build_asgi_app(toolkit, annotations=None)
    return captured[0].request_handlers[types.CallToolRequest]


async def _call(handler, marker, *, wire_id, name="python", arguments=None):
    from mcp import types
    from mcp.server.lowlevel.server import request_ctx
    from mcp.shared.context import RequestContext

    context = RequestContext(request_id=wire_id, meta=None, session=None, lifespan_context=None)
    token = request_ctx.set(context)
    try:
        response = await handler(types.CallToolRequest(
            method="tools/call", params=types.CallToolRequestParams(
                name=name, arguments=arguments if arguments is not None else {"marker": marker},
            ),
        ))
        return response.root
    finally:
        request_ctx.reset(token)


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _result(marker, *, failed=False):
    payload = {"observation": {"observation_id": f"observation-{marker}"}}
    if failed:
        payload["exception"] = "ValueError: generated code failed"
    return SimpleNamespace(result=payload, content_blocks=[
        {"type": "text", "text": f"原始结果 {marker}\n"},
        {"type": "image", "source": {"data": "aW1hZ2U=", "media_type": "image/png"}},
    ])


class _Toolkit:
    def __init__(self, trace=None, *, failed=False):
        self.trace = trace
        self.failed = failed
        self.calls = []

    def get_tools_spec(self):
        return [{"name": "python", "input_schema": {"type": "object"}}]

    def execute_tool(self, name, arguments, *, trace_context=None):
        self.calls.append((name, dict(arguments), dict(trace_context or {})))
        return _result(arguments["marker"], failed=self.failed)


@pytest.mark.parametrize("failed", [False, True])
def test_mcp_trace_preserves_content_and_error_flag(monkeypatch, tmp_path, failed):
    path = tmp_path / "interaction_trace.jsonl"
    toolkit = _Toolkit(InteractionTrace(path), failed=failed)
    handler = _handler(monkeypatch, toolkit)
    original = _result("one", failed=failed)
    expected_content, expected_error = mcp_server._result_to_mcp(original)

    response = _run(_call(handler, "one", wire_id="mcp-wire-request"))

    assert response.content == expected_content
    assert response.isError is expected_error is failed
    events = _events(path)
    received = [event for event in events if event["event"] == "tool_received"]
    ready = [event for event in events if event["event"] == "feedback_ready"]
    assert len(received) == len(ready) == 1
    local_id = received[0]["local_call_id"]
    assert local_id and local_id != "mcp-wire-request"
    assert ready[0]["local_call_id"] == local_id
    assert ready[0]["observation_id"] == "observation-one"
    assert ready[0]["outcome"] == ("error" if failed else "returned")
    assert received[0]["transport"] == ready[0]["transport"] == "mcp"
    assert received[0]["tool"] == ready[0]["tool"] == "python"
    assert len(toolkit.calls) == 1
    name, arguments, context = toolkit.calls[0]
    assert (name, arguments) == ("python", {"marker": "one"})
    assert context["local_call_id"] == local_id and context["transport"] == "mcp"
    assert context.get("provider_call_id") is None
    assert all(event.get("provider_call_id") is None for event in events)


def test_mcp_receives_queued_calls_before_execution_and_keeps_ids_separate(monkeypatch, tmp_path):
    path = tmp_path / "interaction_trace.jsonl"
    trace = InteractionTrace(path)
    release = threading.Event()
    order = []
    original_convert = mcp_server._result_to_mcp

    async def exercise():
        loop = asyncio.get_running_loop()
        first_started, second_received = asyncio.Event(), asyncio.Event()
        original_emit = trace.emit
        received_ids = []

        def emit(event, **fields):
            original_emit(event, **fields)
            order.append((event, fields.get("local_call_id")))
            if event == "tool_received":
                received_ids.append(fields["local_call_id"])
                if len(received_ids) == 2:
                    second_received.set()

        monkeypatch.setattr(trace, "emit", emit)

        class BlockingToolkit(_Toolkit):
            def execute_tool(self, name, arguments, *, trace_context=None):
                self.calls.append((name, dict(arguments), dict(trace_context or {})))
                if arguments["marker"] == "first":
                    loop.call_soon_threadsafe(first_started.set)
                    if not release.wait(timeout=5):
                        raise TimeoutError("test did not release the first call")
                return _result(arguments["marker"])

        toolkit = BlockingToolkit(trace)

        def convert(result):
            converted = original_convert(result)
            marker = result.result["observation"]["observation_id"].removeprefix("observation-")
            local_id = next(context["local_call_id"] for _, args, context in toolkit.calls
                            if args["marker"] == marker)
            order.append(("content_converted", local_id))
            return converted

        monkeypatch.setattr(mcp_server, "_result_to_mcp", convert)
        handler = _handler(monkeypatch, toolkit)
        tasks = [asyncio.create_task(_call(
            handler, "first", wire_id="mcp-first", name="mcp__pyrualean__python",
        ))]
        try:
            await asyncio.wait_for(first_started.wait(), timeout=3)
            tasks.append(asyncio.create_task(_call(handler, "second", wire_id="mcp-second")))
            await asyncio.wait_for(second_received.wait(), timeout=3)
            assert len(toolkit.calls) == 1
            assert len(received_ids) == len(set(received_ids)) == 2
            assert not any(event["event"] == "feedback_ready" for event in _events(path))
        finally:
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        return toolkit, responses

    toolkit, responses = _run(exercise())
    assert [response.content[0].text for response in responses] == [
        "原始结果 first\n", "原始结果 second\n",
    ]
    events = _events(path)
    received = [event for event in events if event["event"] == "tool_received"]
    ready = [event for event in events if event["event"] == "feedback_ready"]
    assert len(received) == len(ready) == len(toolkit.calls) == 2
    assert all(event.get("provider_call_id") is None for event in events)
    for (_, args, context), receipt in zip(toolkit.calls, received, strict=True):
        local_id = context["local_call_id"]
        assert local_id == receipt["local_call_id"]
        assert local_id not in ("mcp-first", "mcp-second")
        feedback = next(event for event in ready if event["local_call_id"] == local_id)
        assert feedback["observation_id"] == f"observation-{args['marker']}"
        assert receipt["tool"] == feedback["tool"] == "python"
        assert receipt["t"] <= feedback["t"]
        converted_at = order.index(("content_converted", local_id))
        assert converted_at < order.index(("feedback_ready", local_id))


@pytest.mark.parametrize("stage", ["execution", "conversion"])
def test_mcp_failure_has_no_ready_event_or_error_message_in_trace(monkeypatch, tmp_path, stage):
    path = tmp_path / "interaction_trace.jsonl"
    toolkit = _Toolkit(InteractionTrace(path))
    secret = "private exception details"

    def fail(*args, **kwargs):
        raise RuntimeError(secret)

    if stage == "execution":
        monkeypatch.setattr(toolkit, "execute_tool", fail)
    else:
        monkeypatch.setattr(mcp_server, "_result_to_mcp", fail)
    handler = _handler(monkeypatch, toolkit)
    response = _run(_call(handler, "one", wire_id="mcp-failed"))

    assert response.isError and response.content[0].text == secret
    events = _events(path)
    received = [event for event in events if event["event"] == "tool_received"]
    failed = [event for event in events if event["event"] == "feedback_failed"]
    assert len(received) == len(failed) == 1
    assert failed[0]["local_call_id"] == received[0]["local_call_id"]
    assert failed[0]["error_type"] == "RuntimeError"
    assert failed[0]["transport"] == "mcp" and failed[0]["tool"] == "python"
    assert not any(event["event"] == "feedback_ready" for event in events)
    assert secret not in path.read_text()


def test_mcp_without_trace_supports_existing_toolkit_signature(monkeypatch):
    class LegacyToolkit:
        get_tools_spec = _Toolkit.get_tools_spec

        def execute_tool(self, name, arguments):
            assert name == "python"
            return _result(arguments["marker"])

    handler = _handler(monkeypatch, LegacyToolkit())
    response = _run(_call(handler, "legacy", wire_id=42))
    content, is_error = mcp_server._result_to_mcp(_result("legacy"))
    assert response.content == content
    assert response.isError is is_error is False


def test_real_toolkit_mcp_blocks_prequeued_dependents_without_trace(monkeypatch, tmp_path):
    from pyrualean.arms import ArmToolkit
    from pyrualean.libero import LiberoRobot
    from tests.test_interaction_contracts import ProtocolBackend

    backend = ProtocolBackend()
    toolkit = ArmToolkit(LiberoRobot(backend), arm="cells", workspace=tmp_path / "ws",
                         images="none")
    execute = toolkit.execute_tool
    release = threading.Event()

    async def exercise():
        loop = asyncio.get_running_loop()
        started, queued = asyncio.Event(), asyncio.Event()
        submission = toolkit.submission_context
        submissions = []

        def received():
            context = submission()
            submissions.append(context)
            if len(submissions) == 2:
                queued.set()
            return context

        def blocking(*args, **kwargs):
            if args[1]["request_id"] == "first":
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5)
            return execute(*args, **kwargs)

        monkeypatch.setattr(toolkit, "submission_context", received)
        monkeypatch.setattr(toolkit, "execute_tool", blocking)
        handler = _handler(monkeypatch, toolkit)
        first = asyncio.create_task(_call(handler, None, wire_id=1, arguments={
            "request_id": "first", "code": "robo.move_to([0.05,-0.1,0.6])\n1/0",
        }))
        tasks = [first]
        try:
            await asyncio.wait_for(started.wait(), 3)
            tasks.append(asyncio.create_task(_call(handler, None, wire_id=2, arguments={
                "request_id": "second", "code": "robo.release()",
            })))
            await asyncio.wait_for(queued.wait(), 3)
        finally:
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert all(r.isError for r in responses)
        payload = json.loads(responses[1].content[0].text)
        assert payload["skipped"] and payload["execution"]["resume_token"]
        repaired = await _call(handler, None, wire_id=3, arguments={
            "request_id": "repair", "code": "robo.release()",
            "resume_token": payload["execution"]["resume_token"],
        })
        assert not repaired.isError

    _run(exercise())
    assert len(backend.contexts) == 2
    assert toolkit._cell_index == 2
