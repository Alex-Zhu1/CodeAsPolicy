"""Opt-in, synchronous visual supervision at robot primitive boundaries.

Valen answers visual questions; this host evaluates phase-specific rules. No
benchmark truth or proposed numerical actions are sent to the visual model.
The HTTP protocol matches GPT-as-Jev's local Valen /predict service.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class SupervisionPause(BaseException):
    """Unwind this cell, without cancelling or terminating the episode."""


@dataclass(frozen=True)
class MonitorConfig:
    mode: str = "off"
    url: str | None = None
    timeout_s: float = 30.0
    threshold: float = 0.8
    unknown_limit: int = 2
    max_actions: int = 8

    def __post_init__(self):
        if self.mode not in ("off", "shadow", "pause"):
            raise ValueError("monitor mode must be off, shadow or pause")
        if self.mode != "off" and not self.url:
            raise ValueError("monitoring requires --valen-url")
        if self.url:
            parsed = urlsplit(self.url)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.fragment
            ):
                raise ValueError("Valen URL must be HTTP(S), without credentials or fragments")
        for key in ("timeout_s", "threshold"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0.5 < self.threshold <= 1:
            raise ValueError("monitor threshold must be in (0.5, 1]")
        for key in ("unknown_limit", "max_actions"):
            if type(getattr(self, key)) is not int or not 1 <= getattr(self, key) <= 100:
                raise ValueError(f"{key} must be an integer in [1, 100]")


def add_monitor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--monitor-mode", choices=("off", "shadow", "pause"), default="off")
    parser.add_argument("--valen-url", help="Valen /predict URL (shared image filesystem required)")
    parser.add_argument("--monitor-timeout-s", type=float, default=30.0)
    parser.add_argument("--monitor-threshold", type=float, default=0.8)
    parser.add_argument("--monitor-unknown-limit", type=int, default=2)
    parser.add_argument("--monitor-max-actions", type=int, default=8)


def monitor_config(args: argparse.Namespace) -> MonitorConfig:
    return MonitorConfig(
        args.monitor_mode,
        args.valen_url,
        args.monitor_timeout_s,
        args.monitor_threshold,
        args.monitor_unknown_limit,
        args.monitor_max_actions,
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Valen redirects are not supported")


class ValenClient:
    def __init__(self, config: MonitorConfig):
        self.config = config
        # Local model traffic must not go through environment HTTP proxies.
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def predict(self, packet: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(packet, allow_nan=False).encode()
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("Valen request exceeds 4 MiB")
        request = Request(self.config.url, data=data, headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.config.timeout_s) as response:
            body = response.read(64 * 1024 + 1)
        if len(body) > 64 * 1024:
            raise ValueError("Valen response exceeds 64 KiB")
        return json.loads(body)


CRITERIA = {
    "yes": "The stated visual fact is visible in the latest observation.",
    "no": "The stated visual fact is visibly false in the latest observation.",
    "unknown": "The images do not establish this fact; visibility or identity is insufficient.",
}


def decode_answers(raw: Any, facts: dict, threshold: float) -> tuple[dict, dict]:
    answers = raw.get("answers") if isinstance(raw, dict) else None
    if not isinstance(answers, dict) or set(answers) != set(facts):
        raise ValueError("Valen must answer exactly the current phase's fact IDs")
    values, probabilities = {}, {}
    for key, answer in answers.items():
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Valen facts require choice probabilities")
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != set(CRITERIA):
            raise ValueError("Expected yes/no/unknown probabilities")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or not 0 <= v <= 1
            for v in probs.values()
        ):
            raise ValueError("Probabilities must be finite numbers in [0, 1]")
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Probabilities must sum to one")
        winner = max(probs, key=probs.get)
        values[key] = winner if probs[winner] >= threshold else "unknown"
        probabilities[key] = probs
    return values, probabilities


def _rule(value: Any, facts: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {"name", "when", "confirmations"}:
        raise ValueError("A rule has name, when, and optional confirmations")
    name, predicates = value.get("name"), value.get("when")
    if not isinstance(name, str) or not 1 <= len(name) <= 200:
        raise ValueError("Rule name must contain 1..200 characters")
    if (
        not isinstance(predicates, dict)
        or not predicates
        or set(predicates) - set(facts)
        or any(v not in ("yes", "no") for v in predicates.values())
    ):
        raise ValueError("Rule when must map known fact IDs to yes/no")
    count = value.get("confirmations", 1)
    if type(count) is not int or not 1 <= count <= 10:
        raise ValueError("Rule confirmations must be an integer in [1, 10]")
    return {"name": name, "when": dict(predicates), "confirmations": count}


class VisualSupervisor:
    def __init__(self, config: MonitorConfig, directory: Path, client=None):
        if config.mode == "off":
            raise ValueError("Do not instantiate a disabled supervisor")
        self.config, self.client = config, client or ValenClient(config)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.sequence = 0
        self.begin_cell(0)

    def begin_cell(self, cell: int) -> None:
        self.cell = cell
        self.contract = None
        self.paused = None
        self.reports = []
        self.evidence = []
        self.evidence_cameras = []
        self.previous = None
        self.actions = 0
        self.counts = {}
        self.unknowns = {}

    def watch(self, phase, facts, *, alerts=(), critical=None, completion=None):
        """Declare checks for the following primitives; changing phase clears evidence windows."""
        if self.paused:
            raise SupervisionPause(self.paused)
        self.contract = None  # malformed replacements cannot leave old rules active
        if not isinstance(phase, str) or not 1 <= len(phase) <= 200:
            raise ValueError("phase must contain 1..200 characters")
        if not isinstance(facts, dict) or not 1 <= len(facts) <= 12:
            raise ValueError("facts must map 1..12 IDs to atomic visual questions")
        if any(
            not isinstance(k, str)
            or not 1 <= len(k) <= 64
            or not isinstance(v, str)
            or not 1 <= len(v) <= 500
            for k, v in facts.items()
        ):
            raise ValueError("Fact IDs/questions must be bounded nonempty strings")
        if len({q.strip().casefold() for q in facts.values()}) != len(facts):
            raise ValueError("Visual questions must be distinct")
        if not isinstance(alerts, (list, tuple)) or len(alerts) > 8:
            raise ValueError("alerts must contain at most eight rules")
        rules = [_rule(rule, facts) for rule in alerts]
        if len({r["name"] for r in rules}) != len(rules):
            raise ValueError("Alert names must be distinct")
        critical = list(facts) if critical is None else critical
        if not isinstance(critical, (list, tuple)) or any(
            not isinstance(k, str) or k not in facts for k in critical
        ):
            raise ValueError("critical must list known fact IDs")
        if len(set(critical)) != len(critical):
            raise ValueError("critical fact IDs must be distinct")
        complete = _rule(completion, facts) if completion is not None else None
        self.contract = copy.deepcopy(
            dict(
                phase=phase, facts=facts, alerts=rules, critical=list(critical), completion=complete
            )
        )
        self.counts, self.unknowns, self.previous = {}, {}, None

    def _record(self, report: dict) -> None:
        self.reports.append(report)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")

    def _interrupt(self, reason: str) -> None:
        if self.config.mode == "pause":
            self.paused = reason
            raise SupervisionPause(reason)

    def before_action(self, tool: str) -> None:
        if self.paused:
            raise SupervisionPause(self.paused)
        if self.contract is None:
            if self.config.mode == "pause":
                self.paused = "missing_contract"
            self._record(dict(cell=self.cell, tool=tool, event="missing_contract"))
            self._interrupt("missing_contract: call watch before motion")

    def action_error(self, robo, tool: str, error: Exception) -> None:
        if self.config.mode == "pause":
            self.paused = "action_error"
        self.evidence = []
        self.evidence_cameras = []
        for camera in robo.CAMERAS:
            try:
                self.evidence.append(
                    robo._backend.artifact(robo._image_name(camera, robo.SHOW_RESOLUTION))
                )
                self.evidence_cameras.append(camera)
            except Exception:
                pass
        self._record(
            dict(
                cell=self.cell,
                tool=tool,
                event="action_error",
                error=f"{type(error).__name__}: {error}",
            )
        )
        self._interrupt("action_error: inspect current state before replanning")

    def after_action(self, robo, tool: str, steps_before: int | None) -> None:
        self.actions += 1
        if self.contract is None:  # shadow mode reports missing contracts but permits execution
            return
        if self.config.mode == "pause":
            self.paused = "monitor_pending"
        self.sequence += 1
        prefix = self.directory / f"{self.run_id}-{self.sequence:05d}"
        report = dict(
            cell=self.cell,
            sequence=self.sequence,
            tool=tool,
            contract=copy.deepcopy(self.contract),
            step=robo._backend.env_steps(),
            start_step=steps_before,
            mode=self.config.mode,
        )
        started = time.perf_counter()
        self.evidence = []
        self.evidence_cameras = []
        try:
            step = report["step"]
            if type(step) is not int or type(steps_before) is not int or step <= steps_before:
                raise ValueError("No distinct executed endpoint; cannot count visual evidence")
            current = dict(
                observation_id=f"{self.run_id}-{self.sequence}",
                step_id=step,
                captured_at=time.time(),
                images=[],
            )
            report["observation"] = current
            for index, camera in enumerate(robo.CAMERAS):
                data = robo._backend.artifact(robo._image_name(camera, robo.SHOW_RESOLUTION))
                path = Path(f"{prefix}-{index}.png")
                path.write_bytes(data)
                self.evidence.append(data)
                self.evidence_cameras.append(camera)
                current["images"].append(dict(camera=camera, path=str(path)))
            if not current["images"]:
                raise ValueError("Robot has no camera images")
            frames = [current]
            if self.previous is not None:
                if self.previous["step_id"] != steps_before:
                    raise ValueError("Non-contiguous observation interval")
                frames.insert(0, self.previous)
            content = [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "input_format": "pyrualean_visual_facts_v1",
                            "frames": frames,
                            "instruction": "Answer each question for the latest observation. "
                            "Earlier images are only local comparison, not current facts. "
                            "Use unknown for insufficient evidence. Do not infer task success, "
                            "future actions or execution quality.",
                        }
                    ),
                }
            ]
            for frame in frames:
                for image in frame["images"]:
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"Observation {frame['observation_id']}, "
                                f"step {frame['step_id']}: {image['camera']}",
                            },
                            {"type": "image_url", "image_url": {"url": image["path"]}},
                        ]
                    )
            packet = dict(
                state={"messages": [{"role": "user", "content": content}]},
                questions={
                    k: dict(type="choice", instructions=q, criteria=CRITERIA)
                    for k, q in self.contract["facts"].items()
                },
            )
            Path(f"{prefix}-request.json").write_text(json.dumps(packet), encoding="utf-8")
            raw = self.client.predict(packet)
            if time.perf_counter() - started > self.config.timeout_s:
                raise TimeoutError("Visual evidence exceeded the monitor time budget")
            Path(f"{prefix}-response.json").write_text(
                json.dumps(raw, allow_nan=False), encoding="utf-8"
            )
            values, probabilities = decode_answers(
                raw, self.contract["facts"], self.config.threshold
            )
            report.update(values=values, probabilities=probabilities, observation=current)
            self.previous = current
            triggers = []
            for index, rule in enumerate(self.contract["alerts"]):
                if self._matches(f"alert-{index}", rule, values):
                    triggers.append(f"alert:{rule['name']}")
            for fact in self.contract["critical"]:
                self.unknowns[fact] = (
                    self.unknowns.get(fact, 0) + 1 if values[fact] == "unknown" else 0
                )
                if self.unknowns[fact] >= self.config.unknown_limit:
                    triggers.append(f"unknown:{fact}")
            complete = self.contract["completion"]
            if complete and self._matches("completion", complete, values):
                triggers.append(f"completion_review:{complete['name']}")
            if self.actions >= self.config.max_actions:
                triggers.append("action_budget_review")
            report.update(event="review" if triggers else "clear", triggers=triggers)
        except Exception as exc:
            self.previous, self.counts, self.unknowns = None, {}, {}
            report.update(event="monitor_error", error=f"{type(exc).__name__}: {exc}")
        report["elapsed_s"] = round(time.perf_counter() - started, 4)
        report["artifact_prefix"] = str(prefix)
        self._record(report)
        self.paused = None
        if report["event"] != "clear":
            self._interrupt(report["event"] + ": " + ", ".join(report.get("triggers", [])))

    def _matches(self, key: str, rule: dict, values: dict) -> bool:
        matched = all(values[k] == v for k, v in rule["when"].items())
        self.counts[key] = self.counts.get(key, 0) + 1 if matched else 0
        return self.counts[key] >= rule["confirmations"]

    def feedback(self) -> dict:
        # Full contracts, probabilities and images remain in the audit. Keep
        # model feedback bounded so long shadow cells cannot truncate the pause
        # reason or the most recent evidence out of a tool response.
        recent = [
            {k: v for k, v in report.items() if k not in ("contract", "observation")}
            for report in self.reports[-8:]
        ]
        return dict(
            mode=self.config.mode,
            paused=self.paused is not None,
            reason=self.paused,
            actions=self.actions,
            reports=recent,
            report_count=len(self.reports),
            contract=self.contract,
            evidence_cameras=self.evidence_cameras,
            observation=self.reports[-1].get("observation") if self.reports else None,
            audit_dir=str(self.directory),
        )


def monitor_prompt(config: MonitorConfig) -> str:
    return f"""# Visual execution supervision

Supervision mode: {config.mode}. Valen checks images after EACH successful motion
primitive, independently of robo.show(). Its answers are fallible visual evidence.
Before motion in EVERY python cell, declare phase-specific checks using:
watch(phase, facts, alerts=[], critical=None, completion=None)
facts maps 1..12 IDs to distinct, concrete yes/no visual questions. Name objects
and arms in the questions. critical defaults to all fact IDs; pass a subset to
ignore uncertainty in irrelevant facts. An alert or completion rule has
{{"name": "...", "when": {{"fact_id": "yes"}}, "confirmations": 1}}.
All when predicates must match; values may be yes or no, never unknown.
Confirmations count consecutive distinct executed endpoints in this phase.
Call watch again BEFORE entering a different phase; this clears evidence windows.
Use checks appropriate after EVERY primitive in that phase. For example:
watch("transport mouse", {{"held": "Is the mouse held between the right gripper fingers?"}},
      alerts=[{{"name": "lost mouse", "when": {{"held": "no"}}}}])
# Only enter this phase AFTER grasp/lift has been verified, not while approaching.

In pause mode, an alert, completion candidate, {config.unknown_limit} consecutive
critical unknowns, a monitor error, or {config.max_actions} actions in one cell
unwinds the current cell and prevents its remaining motions. Missing watch also
blocks motion. Shadow mode records these events without stopping execution.
The tool returns a supervision report even with pure feedback, and pause evidence
images even with on-demand/none images. Inspect these before replanning.
The episode and Python variables persist, but an interrupted assignment may not
have completed. A reviewed action has already executed; an errored action may
have partially executed. Inspect last_action and do NOT replay the whole cell.
A new python call starts a fresh contract; re-read current state and
write a new plan. Never catch supervision interruptions or bypass watch.
Completion candidates request your review; they do not finish the task.
This checks primitive endpoints, not the intermediate trajectory inside move_to.
"""


def config_record(config: MonitorConfig) -> dict:
    return asdict(config)
