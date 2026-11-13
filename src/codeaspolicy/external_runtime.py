# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""External agent bridge runtime for the code arms (``--runtime external``).

Starts a lightweight localhost HTTP JSON server wrapping :class:`ArmToolkit`
(``python`` / ``run_program`` and ``finish``) inside ``out_dir`` so an external
agent worker (such as an Argon agent driving a live episode turn-by-turn) can
inspect the prompt and step-0 task card images, execute Python cells, receive
saved camera/display PNG paths under ``out_dir/external_images``, and conclude
the episode via ``finish`` with zero external dependencies.

CLI usage::

    python -m codeaspolicy.external_runtime status --out <dir>
    python -m codeaspolicy.external_runtime python --out <dir> --code-file <file>
    python -m codeaspolicy.external_runtime python --out <dir> --code "print(robo.state())"
    python -m codeaspolicy.external_runtime finish --out <dir> --summary "Completed task"
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .arms import ArmResult, ArmToolkit
from .interaction_trace import InteractionTrace
from .openai_runtime import IMAGE_NOTE, _execute_with_deadline

SESSION_SCHEMA = "codeaspolicy.external_session.v1"
SESSION_FILENAME = "external_session.json"
IMAGES_DIRNAME = "external_images"
DEFAULT_WAIT_S = 30.0
DEFAULT_REQUEST_TIMEOUT_S = 900.0
_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")


class ExternalRuntimeError(RuntimeError):
    """Raised when the external runtime session cannot be reached or fails."""


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a sibling temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    data = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    tmp_path.write_text(data, encoding="utf-8")
    os.replace(tmp_path, path)


def _safe_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    if not safe:
        safe = "image.png"
    if not safe.lower().endswith(".png"):
        safe += ".png"
    return safe


def _save_content_block_images(
    result: ArmResult,
    images_dir: Path,
    *,
    turn_index: int,
    counter_start: int = 0,
) -> list[str]:
    """Save ``type == 'image'`` blocks from ``result.content_blocks`` as PNGs."""

    images_dir.mkdir(parents=True, exist_ok=True)
    names = result.result.get("images") or []
    saved: list[str] = []
    image_pos = 0
    for block in result.content_blocks:
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        source = block.get("source") if isinstance(block.get("source"), dict) else {}
        b64_data = source.get("data") or block.get("data")
        png_bytes: bytes | None = None
        if isinstance(b64_data, str) and b64_data:
            with contextlib.suppress(Exception):
                png_bytes = base64.b64decode(b64_data)
        if png_bytes is None and image_pos < len(result.images):
            png_bytes = result.images[image_pos]
        if png_bytes is None:
            continue
        label = (
            str(names[image_pos])
            if image_pos < len(names) and names[image_pos]
            else f"image_{image_pos}.png"
        )
        image_pos += 1
        seq = counter_start + image_pos
        filename = f"turn_{turn_index:03d}_{seq:03d}_{_safe_filename(label)}"
        target = (images_dir / filename).resolve()
        target.write_bytes(png_bytes)
        saved.append(str(target))
    return saved


def _extract_text_block(result: ArmResult) -> str:
    for block in result.content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    return json.dumps(result.result, ensure_ascii=False, default=str)


def run_external_agent(
    prompt_text: str,
    image_paths: list[str | Path],
    arm_toolkit: ArmToolkit,
    *,
    model: str = "gemini-4-argon-high",
    max_turns: int | None = None,
    budget_s: float = 5400.0,
    out_dir: str | Path,
    arm: str = "cells",
    host: str = "127.0.0.1",
    port: int = 0,
    on_timeout: Callable[[], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    trace: InteractionTrace | None = None,
    poll_interval_s: float = 0.02,
    post_budget_grace_s: float = 0.5,
) -> dict[str, Any]:
    """Run an episode driven by an external agent over a localhost HTTP bridge.

    Writes ``out_dir/external_session.json`` atomically once the server is
    listening, saves any images returned in :attr:`ArmResult.content_blocks`
    under ``out_dir/external_images``, enforces ``max_turns`` and ``budget_s``,
    records ``messages.json`` and ``external_events.jsonl``, and returns the
    standard ``generation`` dictionary expected by :func:`codeaspolicy.play.play_episode`.
    """

    if host not in _ALLOWED_HOSTS:
        raise ValueError(f"external_runtime server must bind to localhost, got {host!r}")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "external_events.jsonl"
    session_path = out_dir / SESSION_FILENAME
    prompt_path = out_dir / "prompt.txt"
    if not prompt_path.is_file() and prompt_text:
        prompt_path.write_text(prompt_text, encoding="utf-8")

    resolved_initial_images: list[str] = []
    initial_image_refs: list[dict[str, str]] = []
    for idx, img_path in enumerate(image_paths):
        src = Path(img_path).resolve()
        resolved_initial_images.append(str(src))
        if src.is_file():
            dst = (images_dir / f"turn_000_{idx + 1:03d}_{_safe_filename(src.name)}").resolve()
            if src != dst:
                dst.write_bytes(src.read_bytes())
            initial_image_refs.append({"type": "image_ref", "image_ref": str(dst)})

    if image_paths:
        card_dir = Path(image_paths[0]).resolve().parent
    elif (out_dir / "card").is_dir():
        card_dir = (out_dir / "card").resolve()
    else:
        card_dir = out_dir

    started = time.perf_counter()
    session_id = trace.session_id if trace is not None else uuid.uuid4().hex
    token = secrets.token_hex(16)
    timeout_fired = False

    def stop_once() -> None:
        nonlocal timeout_fired
        if timeout_fired or on_timeout is None:
            return
        timeout_fired = True
        on_timeout()

    def remaining() -> float:
        return budget_s - (time.perf_counter() - started)

    def over() -> bool:
        return bool(should_stop is not None and should_stop())

    event_lock = threading.Lock()
    call_lock = threading.Lock()
    done_event = threading.Event()
    budget_exhausted_at: float | None = None

    recorded_messages: list[dict[str, Any]] = []
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    user_parts.extend(initial_image_refs)
    recorded_messages.append(
        {"role": "user", "content": user_parts if initial_image_refs else prompt_text}
    )

    tool_calls: dict[str, int] = {}
    tool_requests: list[dict[str, Any]] = []
    request_latency: list[float] = []
    errors: list[str] = []
    warnings: list[str] = []
    requests = 0
    image_counter = len(initial_image_refs)
    last_text = ""
    status = "completed"
    returncode: int | None = 0

    def emit_event(kind: str, **fields: Any) -> None:
        with event_lock, open(events_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"t": round(time.perf_counter() - started, 3), "type": kind, **fields},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def build_status_payload(server_url: str) -> dict[str, Any]:
        steps = arm_toolkit.robo._backend.env_steps()
        return {
            "ok": True,
            "schema": SESSION_SCHEMA,
            "ready": not done_event.is_set(),
            "finished": bool(arm_toolkit.finish is not None or done_event.is_set()),
            "finish": arm_toolkit.finish,
            "status": status,
            "session_id": session_id,
            "url": server_url,
            "model": model,
            "arm": arm,
            "prompt_path": str(prompt_path.resolve()),
            "card_path": (
                str((out_dir / "card.json").resolve())
                if (out_dir / "card.json").is_file()
                else None
            ),
            "card_dir": str(card_dir.resolve()),
            "initial_images": resolved_initial_images,
            "images_dir": str(images_dir.resolve()),
            "workspace": str(arm_toolkit.workspace.resolve()),
            "out_dir": str(out_dir),
            "budget_s": budget_s,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "remaining_s": round(max(0.0, remaining()), 3),
            "max_turns": max_turns,
            "turns_used": len(arm_toolkit.turns),
            "cells_used": arm_toolkit._cell_index,
            "programs_used": arm_toolkit._program_index,
            "turn_budget_exhausted": bool(arm_toolkit.turn_budget_exhausted),
            "steps_used": 0 if steps is None else int(steps),
            "env_steps": 0 if steps is None else int(steps),
            "terminated": bool(arm_toolkit.robo._reported_success()),
            "truncated": bool(getattr(arm_toolkit.robo, "_truncated", False)),
        }

    def execute_bridge_tool(
        tool_name: str, raw_args: dict[str, Any]
    ) -> tuple[int, dict[str, Any], bool]:
        nonlocal requests, image_counter, last_text, status, returncode, budget_exhausted_at
        with call_lock:
            left = remaining()
            if left <= 0:
                stop_once()
                status, returncode = "timeout", None
                return (
                    200,
                    {
                        "ok": False,
                        "is_error": True,
                        "error": "Episode wall-clock budget expired",
                        "timeout": True,
                        "finished": True,
                        "terminated": bool(arm_toolkit.robo._reported_success()),
                        "steps_used": int(arm_toolkit.robo._backend.env_steps() or 0),
                        "images": [],
                        "image_paths": [],
                    },
                    True,
                )

            was_budget_exhausted = bool(arm_toolkit.turn_budget_exhausted)
            arguments = dict(raw_args)
            arguments.pop("token", None)
            if tool_name == "finish":
                if not arguments.get("status"):
                    arguments["status"] = "success"
                arguments.setdefault("summary", "")
            elif tool_name == "python":
                if "code" not in arguments and "code_file" in arguments:
                    code_file = Path(str(arguments.pop("code_file"))).expanduser().resolve()
                    arguments["code"] = code_file.read_text(encoding="utf-8")
                if not isinstance(arguments.get("code"), str):
                    return 400, {"ok": False, "is_error": True, "error": "Missing 'code' string"}, False

            requests += 1
            call_id = str(arguments.get("request_id") or f"call_{requests}")
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1

            decision_id = trace.new_id("decision") if trace is not None else uuid.uuid4().hex
            attempt_id = trace.new_id("attempt") if trace is not None else uuid.uuid4().hex
            local_call_id = trace.new_id("tool") if trace is not None else uuid.uuid4().hex
            decision_context = dict(
                arm_toolkit.submission_context(),
                transport="external",
                resume_token=arm_toolkit.resume_state(details=False)["resume_token"],
            )
            if trace is not None:
                trace.emit(
                    "request_started",
                    decision_id=decision_id,
                    attempt_id=attempt_id,
                    runtime="external",
                    timeout_s=left,
                )
                trace.emit(
                    "request_finished",
                    decision_id=decision_id,
                    attempt_id=attempt_id,
                    status_code=200,
                    success=True,
                    elapsed_s=0.0,
                )
                trace.emit(
                    "response_received",
                    decision_id=decision_id,
                    attempt_id=attempt_id,
                    status_code=200,
                    success=True,
                )
                trace.emit(
                    "response_parsed",
                    decision_id=decision_id,
                    attempt_id=attempt_id,
                    tool_calls=1,
                )
                trace.emit(
                    "tool_arguments_complete",
                    local_call_id=local_call_id,
                    decision_id=decision_id,
                    attempt_id=attempt_id,
                    provider_call_id=call_id,
                    tool=tool_name,
                    arguments_valid=True,
                    completion_source="external_bridge",
                )

            execution_context = dict(
                decision_context,
                local_call_id=local_call_id,
                decision_id=decision_id,
                attempt_id=attempt_id,
                provider_call_id=call_id,
                agent_attempt_id=arm_toolkit.agent_attempt_id,
                invocation_id=call_id,
            )

            t0 = time.perf_counter()
            arm_result = _execute_with_deadline(
                arm_toolkit,
                tool_name,
                arguments,
                remaining(),
                stop_once,
                trace_context=execution_context,
            )
            elapsed = time.perf_counter() - t0
            request_latency.append(round(elapsed, 3))
            tool_requests.append(
                {"request": requests, "tool": tool_name, "elapsed_s": round(elapsed, 3)}
            )

            text_output = _extract_text_block(arm_result)
            turn_index = len(arm_toolkit.turns)
            saved_images = _save_content_block_images(
                arm_result,
                images_dir,
                turn_index=turn_index,
                counter_start=image_counter,
            )
            image_counter += len(saved_images)

            recorded_messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False, default=str),
                            },
                        }
                    ],
                }
            )
            recorded_messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": text_output}
            )
            if saved_images:
                image_names = ", ".join(Path(p).name for p in saved_images)
                image_parts: list[dict[str, Any]] = [
                    {
                        "type": "text",
                        "text": IMAGE_NOTE.format(sources=f"{call_id} ({image_names})"),
                    }
                ]
                image_parts.extend(
                    {"type": "image_ref", "image_ref": path} for path in saved_images
                )
                recorded_messages.append({"role": "user", "content": image_parts})

            if trace is not None:
                outcome = (
                    "skipped"
                    if arm_result.result.get("skipped")
                    else "error"
                    if arm_result.is_error
                    else "paused"
                    if (
                        arm_result.result.get("execution_paused")
                        or (arm_result.result.get("supervision") or {}).get("paused")
                    )
                    else "finished"
                    if (
                        arm_result.result.get("episode_finished")
                        or arm_toolkit.finish is not None
                    )
                    else "completed"
                )
                trace.emit(
                    "feedback_ready",
                    local_call_id=local_call_id,
                    decision_id=decision_id,
                    attempt_id=attempt_id,
                    provider_call_id=call_id,
                    tool=tool_name,
                    outcome=outcome,
                    images=len(saved_images),
                )

            if tool_name == "finish" and arm_toolkit.finish is not None:
                last_text = str(arm_toolkit.finish.get("summary") or "")
            elif isinstance(arm_result.result.get("stdout"), str) and arm_result.result["stdout"]:
                last_text = arm_result.result["stdout"]

            steps_used = arm_toolkit.robo._backend.env_steps()
            turn_record = arm_toolkit.turns[-1] if arm_toolkit.turns else {}
            terminated = bool(arm_toolkit.robo._reported_success())
            truncated = bool(getattr(arm_toolkit.robo, "_truncated", False))
            finished = bool(arm_toolkit.finish is not None)

            if arm_toolkit.turn_budget_exhausted and budget_exhausted_at is None:
                budget_exhausted_at = time.perf_counter()

            if timeout_fired or remaining() <= 0:
                status, returncode = "timeout", None
                should_close = True
            elif finished:
                should_close = True
            elif was_budget_exhausted and arm_toolkit.turn_budget_exhausted:
                should_close = True
            else:
                should_close = False

            emit_event(
                "tool",
                tool=tool_name,
                call_id=call_id,
                turn=turn_index,
                elapsed_s=round(elapsed, 3),
                ok=not arm_result.is_error,
                skipped=bool(arm_result.result.get("skipped")),
                images=len(saved_images),
                image_paths=saved_images,
                steps_used=0 if steps_used is None else int(steps_used),
                terminated=terminated,
                finished=finished,
            )

            payload: dict[str, Any] = {
                "ok": not arm_result.is_error,
                "is_error": bool(arm_result.is_error),
                "tool": tool_name,
                "call_id": call_id,
                "turn": turn_index,
                "turns_used": turn_index,
                "cells_used": arm_toolkit._cell_index,
                "programs_used": arm_toolkit._program_index,
                "max_turns": max_turns,
                "turn_budget_exhausted": bool(arm_toolkit.turn_budget_exhausted),
                "text": text_output,
                "stdout": arm_result.result.get("stdout", ""),
                "stderr": arm_result.result.get("stderr", ""),
                "value": arm_result.result.get("value"),
                "exception": arm_result.result.get("exception"),
                "error": arm_result.result.get("error"),
                "images": saved_images,
                "image_paths": saved_images,
                "moved": bool(turn_record.get("moved", arm_result.result.get("moved", False))),
                "step_delta": turn_record.get("env_steps", arm_result.result.get("env_steps")),
                "steps_used": 0 if steps_used is None else int(steps_used),
                "env_steps": 0 if steps_used is None else int(steps_used),
                "terminated": terminated,
                "truncated": truncated,
                "finished": finished,
                "finish": arm_toolkit.finish,
                "elapsed_s": round(elapsed, 3),
                "remaining_s": round(max(0.0, remaining()), 3),
                "execution": arm_result.result.get("execution"),
                "result": arm_result.result,
            }
            return 200, payload, should_close

    server_url_holder: list[str] = [""]

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return  # keep stderr clean

        def _check_auth(self, body: dict[str, Any] | None = None) -> bool:
            auth = self.headers.get("Authorization") or ""
            header_tok = self.headers.get("X-Session-Token") or ""
            supplied: str | None = None
            if auth.lower().startswith("bearer "):
                supplied = auth[7:].strip()
            elif header_tok:
                supplied = header_tok.strip()
            elif isinstance(body, dict) and isinstance(body.get("token"), str):
                supplied = body["token"]
            if supplied is not None and supplied != token:
                self._send_json(403, {"ok": False, "error": "Invalid session token"})
                return False
            return True

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            raw = (json.dumps(payload, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            if not self._check_auth():
                return
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path in ("/", "/status", "/health"):
                self._send_json(200, build_status_payload(server_url_holder[0]))
            else:
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"ok": False, "error": f"Invalid JSON body: {exc}"})
                return
            if not isinstance(body, dict):
                self._send_json(400, {"ok": False, "error": "JSON body must be an object"})
                return
            if not self._check_auth(body):
                return

            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/status":
                self._send_json(200, build_status_payload(server_url_holder[0]))
                return

            if path == "/python":
                tool_name, tool_args = "python", body
            elif path == "/run_program":
                tool_name, tool_args = "run_program", body
            elif path == "/finish":
                tool_name, tool_args = "finish", body
            elif path == "/tool":
                tool_name = str(body.get("name") or body.get("tool") or "")
                raw_sub = body.get("arguments")
                tool_args = dict(raw_sub) if isinstance(raw_sub, dict) else dict(body)
                tool_args.pop("name", None)
                tool_args.pop("tool", None)
                tool_args.pop("arguments", None)
            else:
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})
                return

            http_code, response_payload, should_close = execute_bridge_tool(tool_name, tool_args)
            self._send_json(http_code, response_payload)
            if should_close:
                done_event.set()

    http_server = ThreadingHTTPServer((host, port), _Handler)
    http_server.daemon_threads = True
    actual_port = int(http_server.server_port)
    server_url = f"http://{host}:{actual_port}"
    server_url_holder[0] = server_url

    server_thread = threading.Thread(
        target=http_server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
        name="external-runtime-http",
    )
    server_thread.start()

    session_info: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "ready": True,
        "finished": False,
        "url": server_url,
        "host": host,
        "port": actual_port,
        "token": token,
        "session_id": session_id,
        "prompt_path": str(prompt_path.resolve()),
        "card_path": (
            str((out_dir / "card.json").resolve())
            if (out_dir / "card.json").is_file()
            else None
        ),
        "card_dir": str(card_dir.resolve()),
        "initial_images": resolved_initial_images,
        "images_dir": str(images_dir.resolve()),
        "workspace": str(arm_toolkit.workspace.resolve()),
        "out_dir": str(out_dir),
        "model": model,
        "arm": arm,
        "budget_s": budget_s,
        "max_turns": max_turns,
        "pid": os.getpid(),
        "started_at": time.time(),
    }
    _write_json_atomic(session_path, session_info)
    emit_event(
        "session_ready",
        url=server_url,
        session_id=session_id,
        model=model,
        arm=arm,
        budget_s=budget_s,
        max_turns=max_turns,
        initial_images=len(resolved_initial_images),
    )

    try:
        if remaining() <= 0:
            stop_once()
            status, returncode = "timeout", None
            done_event.set()
        while not done_event.is_set():
            if over():
                break
            if remaining() <= 0:
                stop_once()
                status, returncode = "timeout", None
                break
            if (
                budget_exhausted_at is not None
                and arm_toolkit.finish is None
                and (time.perf_counter() - budget_exhausted_at) >= post_budget_grace_s
            ):
                break
            done_event.wait(timeout=min(poll_interval_s, max(0.001, remaining())))
    finally:
        if status == "timeout":
            stop_once()
        with contextlib.suppress(Exception):
            http_server.shutdown()
        with contextlib.suppress(Exception):
            http_server.server_close()
        server_thread.join(timeout=2.0)

        session_info.update(
            ready=False,
            finished=True,
            status=status,
            finish=arm_toolkit.finish,
            turns_used=len(arm_toolkit.turns),
            steps_used=int(arm_toolkit.robo._backend.env_steps() or 0),
            ended_at=time.time(),
        )
        with contextlib.suppress(Exception):
            _write_json_atomic(session_path, session_info)

        emit_event(
            "session_closed",
            status=status,
            returncode=returncode,
            turns=len(arm_toolkit.turns),
            steps_used=int(arm_toolkit.robo._backend.env_steps() or 0),
            finish=arm_toolkit.finish,
        )
        (out_dir / "messages.json").write_text(
            json.dumps(recorded_messages, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (out_dir / "last_message.md").write_text(last_text or "", encoding="utf-8")

    return {
        "runtime": "external",
        "model": model,
        "models": [model],
        "session_id": session_id,
        "url": server_url,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_output_tokens": None,
        "total_tokens": None,
        "requests": requests,
        "request_profile": [],
        "request_latency_s": request_latency,
        "turns": len(arm_toolkit.turns),
        "tool_calls": tool_calls,
        "tool_requests": tool_requests,
        "errors": errors,
        "warnings": warnings,
        "compactions": 0,
        "max_turns_hit": bool(arm_toolkit.turn_budget_exhausted),
        "status": status,
        "returncode": returncode,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "messages": len(recorded_messages),
        "images_saved": image_counter,
        "options": {
            "model": model,
            "arm": arm,
            "max_turns": max_turns,
            "budget_s": budget_s,
            "images": len(image_paths),
        },
    }


# -- Zero-dependency client & CLI ---------------------------------------------


def _local_opener() -> urllib.request.OpenerDirector:
    """Build a urllib opener that bypasses any environment proxy for localhost."""

    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_session(out_dir: str | Path) -> dict[str, Any]:
    """Read ``external_session.json`` from ``out_dir``."""

    path = Path(out_dir).expanduser().resolve() / SESSION_FILENAME
    if not path.is_file():
        raise ExternalRuntimeError(f"External session file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalRuntimeError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, dict) or not data.get("url"):
        raise ExternalRuntimeError(f"Malformed external session file: {path}")
    return data


def wait_for_session(
    out_dir: str | Path,
    *,
    timeout_s: float = DEFAULT_WAIT_S,
    poll_s: float = 0.05,
) -> dict[str, Any]:
    """Wait until ``out_dir/external_session.json`` exists and is valid JSON."""

    deadline = time.perf_counter() + max(0.0, timeout_s)
    last_error: Exception | None = None
    while True:
        try:
            session = load_session(out_dir)
            if session.get("ready") or session.get("finished"):
                return session
        except ExternalRuntimeError as exc:
            last_error = exc
        if time.perf_counter() >= deadline:
            raise ExternalRuntimeError(
                f"Timed out waiting {timeout_s}s for external session in {out_dir}"
            ) from last_error
        time.sleep(poll_s)


def _http_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Session-Token"] = token
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = _local_opener()
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        raise ExternalRuntimeError(f"HTTP {exc.code} from {url}: {raw[:500]}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ExternalRuntimeError(f"Request to {url} failed: {exc}") from exc


def request_status(
    out_dir: str | Path,
    *,
    wait_s: float = DEFAULT_WAIT_S,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Query the live status of an external session (or its final saved state)."""

    session = wait_for_session(out_dir, timeout_s=wait_s)
    if session.get("ready"):
        with contextlib.suppress(ExternalRuntimeError):
            live = _http_json(
                "GET",
                f"{session['url'].rstrip('/')}/status",
                token=session.get("token"),
                timeout_s=timeout_s,
            )
            return {**session, **live}
    result_path = Path(out_dir).expanduser().resolve() / "result.json"
    if result_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            res = json.loads(result_path.read_text(encoding="utf-8"))
            session = {
                **session,
                "episode_status": res.get("status"),
                "environment_success": res.get("environment_success"),
                "steps_used": res.get("env_steps", session.get("steps_used", 0)),
                "turns_used": res.get("tool_turns", session.get("turns_used", 0)),
                "finish": res.get("finish", session.get("finish")),
            }
    return session


def request_python(
    out_dir: str | Path,
    code: str | None = None,
    *,
    code_file: str | Path | None = None,
    request_id: str | None = None,
    resume_token: str | None = None,
    wait_s: float = DEFAULT_WAIT_S,
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute one ``python`` cell in the external session at ``out_dir``."""

    if code is None and code_file is not None:
        code = Path(code_file).expanduser().read_text(encoding="utf-8")
    if code is None:
        raise ValueError("request_python requires either `code` or `code_file`")
    session = wait_for_session(out_dir, timeout_s=wait_s)
    payload: dict[str, Any] = {"code": code}
    if request_id is not None:
        payload["request_id"] = request_id
    if resume_token is not None:
        payload["resume_token"] = resume_token
    return _http_json(
        "POST",
        f"{session['url'].rstrip('/')}/python",
        token=session.get("token"),
        payload=payload,
        timeout_s=timeout_s,
    )


def request_finish(
    out_dir: str | Path,
    summary: str = "",
    *,
    status: str = "success",
    request_id: str | None = None,
    resume_token: str | None = None,
    wait_s: float = DEFAULT_WAIT_S,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Conclude the external session at ``out_dir`` by calling ``finish``."""

    session = wait_for_session(out_dir, timeout_s=wait_s)
    payload: dict[str, Any] = {"status": status, "summary": summary}
    if request_id is not None:
        payload["request_id"] = request_id
    if resume_token is not None:
        payload["resume_token"] = resume_token
    return _http_json(
        "POST",
        f"{session['url'].rstrip('/')}/finish",
        token=session.get("token"),
        payload=payload,
        timeout_s=timeout_s,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"python -m {__package__ or 'codeaspolicy'}.external_runtime",
        description="Drive a live --runtime external episode turn-by-turn.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status_p = sub.add_parser("status", help="inspect session metadata and step/turn budgets")
    status_p.add_argument("--out", type=Path, required=True, help="episode output directory")
    status_p.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S)
    status_p.add_argument("--timeout-s", type=float, default=30.0)

    py_p = sub.add_parser("python", help="execute a Python cell in the live episode")
    py_p.add_argument("--out", type=Path, required=True, help="episode output directory")
    py_group = py_p.add_mutually_exclusive_group(required=False)
    py_group.add_argument("--code", default=None, help="inline Python source code (or '-' for stdin)")
    py_group.add_argument("--code-file", type=Path, default=None, help="path to a .py file to run")
    py_p.add_argument("--request-id", default=None)
    py_p.add_argument("--resume-token", default=None)
    py_p.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S)
    py_p.add_argument("--timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)

    fin_p = sub.add_parser("finish", help="call finish to conclude the episode")
    fin_p.add_argument("--out", type=Path, required=True, help="episode output directory")
    fin_p.add_argument("--summary", default="", help="one or two sentence summary")
    fin_p.add_argument(
        "--status",
        choices=("success", "failure", "stuck"),
        default="success",
        help="declared episode status (default: success)",
    )
    fin_p.add_argument("--request-id", default=None)
    fin_p.add_argument("--resume-token", default=None)
    fin_p.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S)
    fin_p.add_argument("--timeout-s", type=float, default=60.0)

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            response = request_status(args.out, wait_s=args.wait_s, timeout_s=args.timeout_s)
        elif args.command == "python":
            code = args.code
            if code == "-":
                code = sys.stdin.read()
            elif code is None and args.code_file is None:
                if not sys.stdin.isatty():
                    code = sys.stdin.read()
                else:
                    parser.error("python requires --code or --code-file")
            response = request_python(
                args.out,
                code,
                code_file=args.code_file,
                request_id=args.request_id,
                resume_token=args.resume_token,
                wait_s=args.wait_s,
                timeout_s=args.timeout_s,
            )
        elif args.command == "finish":
            response = request_finish(
                args.out,
                args.summary,
                status=args.status,
                request_id=args.request_id,
                resume_token=args.resume_token,
                wait_s=args.wait_s,
                timeout_s=args.timeout_s,
            )
        else:  # pragma: no cover
            parser.error(f"unknown command: {args.command}")
            return 2
    except ExternalRuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
    return 0


__all__ = [
    " DEFAULT_REQUEST_TIMEOUT_S".strip(),
    "DEFAULT_WAIT_S",
    "ExternalRuntimeError",
    "IMAGES_DIRNAME",
    "SESSION_FILENAME",
    "SESSION_SCHEMA",
    "load_session",
    "main",
    "request_finish",
    "request_python",
    "request_status",
    "run_external_agent",
    "wait_for_session",
]

if __name__ == "__main__":
    raise SystemExit(main())
