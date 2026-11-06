"""Synchronous visual supervision at robot primitive boundaries.

Valen answers visual questions; this host evaluates phase-specific rules. No
benchmark truth or proposed numerical actions are sent to the visual model.
The HTTP protocol matches GPT-as-Jev's local Valen /predict service.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from ._env import setting
from .evidence import EvidenceLedger

DEFAULT_VALEN_URL = "http://127.0.0.1:18001/predict"


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
    parser.add_argument(
        "--monitor-mode", choices=("off", "shadow", "pause"), default=None,
        help="optional visual supervision for cells (default: off)",
    )
    parser.add_argument(
        "--valen-url", default=setting("VALEN_URL", DEFAULT_VALEN_URL),
        help=f"Valen /predict service (default: $PYRUALEAN_VALEN_URL or {DEFAULT_VALEN_URL}; "
             "shared image filesystem required)",
    )
    parser.add_argument("--monitor-timeout-s", type=float, default=30.0)
    parser.add_argument("--monitor-threshold", type=float, default=0.8)
    parser.add_argument("--monitor-unknown-limit", type=int, default=2)
    parser.add_argument("--monitor-max-actions", type=int, default=8)


def monitor_config(args: argparse.Namespace) -> MonitorConfig:
    mode = args.monitor_mode if args.monitor_mode is not None else "off"
    if mode != "off" and args.arm != "cells":
        raise ValueError("Visual supervision currently supports only --arm cells")
    return MonitorConfig(
        mode,
        args.valen_url if mode != "off" else None,
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


def answer_diagnostics(probabilities: dict, threshold: float) -> dict:
    """Explain validated predictions without treating scores as calibrated accuracy."""
    result = {}
    for fact, probs in probabilities.items():
        winner = max(probs, key=probs.get)
        result[fact] = dict(
            raw_winner=winner, probability=probs[winner], threshold=threshold,
            abstention_reason=("model_unknown" if winner == "unknown" else
                               "below_threshold" if probs[winner] < threshold else None),
        )
    return result


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
        self.ledger = EvidenceLedger()
        self.last_endpoint = None
        self.last_source = None
        self.counts = {}
        self.unknowns = {}
        self._rule_sources = {}
        self._rule_scope = None
        # Review notifications survive cells without turning their observations
        # into hard prerequisites. New known evidence can rearm a notification.
        self._reviewed_unknowns = {}
        self._reviewed_rules = set()
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
        self.verifications = 0

    def watch(self, phase, facts, *, alerts=(), critical=None, completion=None,
              requires=None, invalidates=None):
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
        critical = [] if critical is None else critical
        if not isinstance(critical, (list, tuple)) or any(
            not isinstance(k, str) or k not in facts for k in critical
        ):
            raise ValueError("critical must list known fact IDs")
        if len(set(critical)) != len(critical):
            raise ValueError("critical fact IDs must be distinct")
        complete = _rule(completion, facts) if completion is not None else None
        self.ledger.configure(phase, facts, requires, invalidates)
        self.contract = copy.deepcopy(
            dict(
                phase=phase, facts=facts, alerts=rules, critical=list(critical),
                completion=complete,
                requires={key: gate["expected"] for key, gate in
                          self.ledger.requirements.get(phase, {}).items()},
                invalidates=invalidates,
            )
        )
        scope = json.dumps(self.contract, sort_keys=True)
        if scope != self._rule_scope:
            self.counts, self._rule_sources = {}, {}
            self._rule_scope = scope
        self.unknowns = {key: fact["unknown_streak"] for key, fact in self.ledger.facts.items()}
        self.previous = None
        self._save_evidence("watch")

    def read_evidence(self):
        """Return a detached snapshot; editing it cannot establish a visual fact."""
        return self.ledger.snapshot()

    def defer_fact(self, fact, *, until, reason, expected="yes"):
        if self.paused:
            raise SupervisionPause(self.paused)
        self.ledger.defer(fact, until=until, reason=reason, expected=expected)
        if self.contract and self.contract["phase"] == until:
            self.contract["requires"][fact] = expected
        self._save_evidence("defer_fact")

    def _save_evidence(self, event):
        snapshot = self.ledger.snapshot()
        data = json.dumps(snapshot, allow_nan=False)
        path = self.directory / "evidence.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(data, encoding="utf-8")
        pending.replace(path)
        with (self.directory / "evidence.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event=event, cell=self.cell, **snapshot)) + "\n")

    def _record(self, report: dict) -> None:
        self.reports.append(report)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")

    def _interrupt(self, reason: str) -> None:
        if self.config.mode == "pause":
            self.paused = reason
            raise SupervisionPause(reason)

    def _blockers(self):
        if self.contract is None:
            return {}
        # Only explicitly registered requirements confer execution authority.
        # Ordinary visual uncertainty asks the planner to review, not to obtain
        # Valen's permission before every motion in every later cell.
        return self.ledger.blockers(self.contract["phase"])

    @staticmethod
    def _source_protocol(robo):
        return callable(getattr(robo, "_resolve_observation", None)) and getattr(
            robo, "_source_verification_supported",
            getattr(robo, "_supports_observation_snapshots", True),
        )

    def before_action(self, tool: str, robo=None) -> None:
        if self.paused:
            raise SupervisionPause(self.paused)
        if self.contract is None:
            if self.config.mode == "pause":
                self.paused = "missing_contract"
            self._record(dict(cell=self.cell, tool=tool, event="missing_contract"))
            self._interrupt("missing_contract: call watch before motion")
            return
        if robo is not None and self.last_endpoint is not None:
            current = (robo._backend.env_steps(), robo._action_revision)
            if current != self.last_endpoint:
                self.ledger.invalidate(None, "state_changed_outside_observation")
                self._save_evidence("external_change")
        if robo is not None and self.last_source is not None:
            try:
                current = getattr(robo._backend, "current_observation_source", None)
                metadata = (current() if callable(current)
                            else robo._backend.observation_metadata())
                source = (metadata.get("session_id"), metadata.get("source_id"))
            except Exception:
                self.ledger.invalidate(None, "observation_source_unavailable")
                self._save_evidence("source_error")
                raise
            if (source != self.last_source or metadata.get("aligned", True) is not True
                    or metadata.get("native_steps") != self.last_endpoint[0]):
                self.ledger.invalidate(None, "observation_source_changed")
                self._save_evidence("source_changed")
        blocked = self._blockers()
        if blocked:
            observation = self._review_images(robo) if robo is not None else None
            self._record(dict(cell=self.cell, tool=tool, event="prerequisite_review",
                              phase=self.contract["phase"], blocked=blocked,
                              observation=observation))
            self._interrupt("prerequisite_review: establish fresh evidence for "
                            + ", ".join(blocked) + " before dependent motion")

    def begin_action(self, tool: str, *, action_id=None) -> None:
        """Invalidate before dispatch: a failing primitive may still change the scene."""
        keys = self.contract["invalidates"] if self.contract else None
        self.ledger.invalidate(keys, f"motion:{tool}", action_id=action_id)
        self._save_evidence("motion_started")

    def action_error(self, robo, tool: str, error: Exception) -> None:
        action_id = getattr(robo, "_last_action_id", None)
        self.action_interrupted(tool, action_id=action_id)
        if self.config.mode == "pause":
            self.paused = "action_error"
        observation = self._review_images(robo)
        self._record(
            dict(
                cell=self.cell,
                tool=tool,
                action_id=action_id,
                event="action_error",
                error=f"{type(error).__name__}: {error}",
                observation=observation,
            )
        )
        self._interrupt("action_error: inspect current state before replanning")

    def action_interrupted(self, tool: str, *, action_id=None) -> None:
        self.ledger.invalidate(None, f"action_error:{tool}", action_id=action_id)
        self._save_evidence("action_error")

    def _review_images(self, robo):
        """Host review images are context, not new Valen votes or established facts."""
        self.evidence = []
        self.evidence_cameras = []
        if self._source_protocol(robo):
            try:
                observation, images = robo._resolve_observation(None)
                for camera, data in images:
                    self.evidence.append(data)
                    self.evidence_cameras.append(camera)
                return dict(observation, source="host_review")
            except Exception as exc:
                return dict(source="host_review", observation_id=None,
                            error=f"{type(exc).__name__}: {exc}")
        observation = dict(
            observation_id=f"{self.run_id}-review-{uuid.uuid4().hex}",
            step_id=robo._backend.env_steps(),
            state_version=robo._action_revision,
            captured_at=None, recorded_at=time.time(), source="host_review",
        )
        for camera in robo.CAMERAS:
            try:
                self.evidence.append(
                    robo._backend.artifact(robo._image_name(camera, robo.SHOW_RESOLUTION))
                )
                self.evidence_cameras.append(camera)
            except Exception:
                pass
        return observation

    def after_action(self, robo, tool: str, steps_before: int | None, *, outcome=None) -> None:
        self.actions += 1
        self.last_endpoint = (robo._backend.env_steps(), getattr(robo, "_action_revision", None))
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
            action_id=(outcome or {}).get("action_id"),
        )
        step = report["step"]
        progressed = type(step) is int and type(steps_before) is int and step > steps_before
        missed = (outcome or {}).get("reached") is False
        no_motion = type(step) is int and type(steps_before) is int and step == steps_before
        recoverable = missed and (
            (outcome or {}).get("recoverable") is True
            or (no_motion and bool((outcome or {}).get("ik_error")))
        )
        if recoverable and progressed:
            # A partial endpoint is real evidence. Check it before returning the
            # result to a caller-written correction loop; visual pauses still win.
            report.update(motion_event="motion_incomplete",
                          primitive_outcome=copy.deepcopy(outcome),
                          executed_steps=step - steps_before)
        elif missed or not progressed:
            # A controller refusal is not a Valen failure. Never use an unchanged
            # endpoint as another visual vote. Expected zero-step refusals may
            # return to bounded Python logic; faults still hand off to the model.
            event = ("motion_incomplete" if progressed else "motion_rejected") if missed else (
                "no_progress"
            )
            reason = ((outcome or {}).get("ik_error") or (outcome or {}).get("note")
                      or ("Target not reached; inspect controller outcome" if missed else
                          "No distinct executed endpoint; cannot count visual evidence"))
            self.previous = None
            # No new endpoint means no new vote, including no evidence that a
            # preceding alert/unknown streak has cleared. A local retry must not
            # erase uncertainty merely by inserting an expected IK refusal.
            self.ledger.invalidate(None, event, action_id=(outcome or {}).get("action_id"))
            report.update(
                event=event, error=reason, primitive_outcome=copy.deepcopy(outcome or {}),
                executed_steps=(max(0, step - steps_before)
                                if type(step) is int and type(steps_before) is int else None),
                observation=self._review_images(robo),
            )
            self._save_evidence(event)
            self._record(report)
            self.paused = None
            if recoverable and self.actions < self.config.max_actions:
                return
            self._interrupt(f"{event}: {reason}; inspect current state and revise the motion")
            return
        self._evaluate(robo, report, prefix, steps_before=steps_before)

    def verify(self, robo, observation_id=None) -> dict:
        """Read-only verification of the declared facts, independent of motion.

        A snapshot must belong to this robot's current scene. This operation
        cannot reconcile an unresolved action or generate a new sensor frame;
        request an explicit read-only snapshot refresh when more evidence is needed.
        """
        if self.paused:
            raise SupervisionPause(self.paused)
        if self.contract is None:
            raise ValueError("verify requires watch(phase, facts, ...) first")
        self.verifications += 1
        if self.verifications > self.config.max_actions:
            report = dict(cell=self.cell, tool="verify", event="verification_budget_review",
                          verification_calls=self.verifications,
                          verification_limit=self.config.max_actions)
            self._record(report)
            self._interrupt("verification_budget_review: return for a new judgment")
            return copy.deepcopy(report)
        self.sequence += 1
        prefix = self.directory / f"{self.run_id}-{self.sequence:05d}"
        report = dict(cell=self.cell, sequence=self.sequence, tool="verify",
                      contract=copy.deepcopy(self.contract), mode=self.config.mode,
                      verification_calls=self.verifications, read_only=True)
        self._evaluate(robo, report, prefix, observation_id=observation_id, strict=True)
        return copy.deepcopy(report)

    def _capture(self, robo, prefix, observation_id, *, strict):
        """Resolve host-owned image bytes, preserving the producer's source identity."""
        trusted = None
        if self._source_protocol(robo):
            trusted, images = robo._resolve_observation(observation_id)
            if not robo._observation_is_current(trusted):
                raise ValueError("Observation is stale or its image/state sources are unaligned")
            current = copy.deepcopy(trusted)
            current.setdefault("step_id", robo._backend.env_steps())
            current.setdefault("state_version", current.get("action_revision"))
        else:
            if strict:
                raise ValueError(
                    "Read-only verification requires a host-owned observation snapshot")
            # Recorded replay adapters predate the snapshot protocol. Only the
            # post-action compatibility path accepts these declared endpoints.
            step = robo._backend.env_steps()
            revision = getattr(robo, "_action_revision", None)
            current = dict(observation_id=f"{self.run_id}-{self.sequence}",
                           source_id=f"{self.run_id}-endpoint-{step}-{revision}",
                           step_id=step, state_version=revision, captured_at=None,
                           source_freshness="unverified")
            images = ((camera, robo._backend.artifact(
                robo._image_name(camera, robo.SHOW_RESOLUTION))) for camera in robo.CAMERAS)
        image_metadata = current.get("images", {})
        if not isinstance(image_metadata, dict):
            image_metadata = {image["camera"]: image for image in image_metadata}
        current["images"] = []
        for index, (camera, data) in enumerate(images):
            path = Path(f"{prefix}-{index}.png")
            path.write_bytes(data)
            self.evidence.append(data)
            self.evidence_cameras.append(camera)
            current["images"].append(dict(image_metadata.get(camera, {}),
                                           camera=camera, path=str(path),
                                           source_id=image_metadata.get(camera, {}).get(
                                               "source_id", current.get("source_id"))))
        if not current["images"]:
            raise ValueError("Robot has no camera images")
        return current, trusted

    def _evaluate(self, robo, report, prefix, *, steps_before=None,
                  observation_id=None, strict=False):
        started = time.perf_counter()
        self.evidence = []
        self.evidence_cameras = []
        if self.config.mode == "pause":
            self.paused = "monitor_pending"
        try:
            scope = (robo._verification_scope() if hasattr(robo, "_verification_scope")
                     else nullcontext())
            if strict and not hasattr(robo, "_verification_scope"):
                raise ValueError("Robot does not provide a read-only verification scope")
            with scope:
                current, trusted = self._capture(robo, prefix, observation_id, strict=strict)
                report["observation"] = current
                endpoint = (robo._backend.env_steps(), getattr(robo, "_action_revision", None))
                frames = [current]
                if self.previous is not None and steps_before is not None:
                    if self.previous["step_id"] != steps_before:
                        raise ValueError("Non-contiguous observation interval")
                    if self.previous.get("source_id") != current.get("source_id"):
                        frames.insert(0, self.previous)
                content = [{
                    "type": "text",
                    "text": json.dumps({
                        "input_format": "pyrualean_visual_facts_v1", "frames": frames,
                        "instruction": "Answer each question for the latest observation. "
                        "Earlier images are only local comparison, not current facts. "
                        "Use unknown for insufficient evidence. Do not infer task success, "
                        "future actions or execution quality.",
                    }),
                }]
                for frame in frames:
                    for image in frame["images"]:
                        content.extend([
                            {"type": "text", "text": f"Observation {frame['observation_id']}, "
                             f"step {frame['step_id']}: {image['camera']}"},
                            {"type": "image_url", "image_url": {"url": image["path"]}},
                        ])
                packet = dict(
                    state={"messages": [{"role": "user", "content": content}]},
                    questions={k: dict(type="choice", instructions=q, criteria=CRITERIA)
                               for k, q in self.contract["facts"].items()},
                )
                Path(f"{prefix}-request.json").write_text(json.dumps(packet), encoding="utf-8")
                raw = self.client.predict(packet)
                if time.perf_counter() - started > self.config.timeout_s:
                    raise TimeoutError("Visual evidence exceeded the monitor time budget")
                Path(f"{prefix}-response.json").write_text(
                    json.dumps(raw, allow_nan=False), encoding="utf-8")
                values, probabilities = decode_answers(raw, self.contract["facts"],
                                                       self.config.threshold)
                if (robo._backend.env_steps(), getattr(robo, "_action_revision", None)) != endpoint:
                    raise ValueError("Scene changed during visual observation; evidence is stale")
                if trusted is not None and not robo._observation_is_current(trusted):
                    raise ValueError("Scene changed during visual observation; evidence is stale")
            novel = self.ledger.observe(values, probabilities, current)
            # A repeated inference cannot flip the vote or clear uncertainty.
            # Raw model responses remain available in the immutable audit files.
            effective_values = {key: self.ledger.facts[key]["value"] for key in values}
            effective_probs = {key: self.ledger.facts[key]["probabilities"] for key in values}
            self._rearm_reviews(effective_values, novel)
            self.unknowns = {key: fact["unknown_streak"] for key, fact in self.ledger.facts.items()}
            self.last_endpoint = endpoint
            if trusted is not None:
                self.last_source = (trusted.get("session_id"), trusted.get("source_id"))
            self.previous = current
            report.update(values=effective_values, probabilities=effective_probs,
                          decisions=answer_diagnostics(effective_probs, self.config.threshold),
                          new_evidence=novel, blocked=self._blockers())
            triggers = []
            for index, rule in enumerate(self.contract["alerts"]):
                if self._matches(f"alert-{index}", rule, effective_values, current):
                    triggers.append(f"alert:{rule['name']}")
            for fact in self.contract["critical"]:
                version = self.ledger.facts[fact]["version"]
                if (self.unknowns[fact] >= self.config.unknown_limit
                        and self._reviewed_unknowns.get(fact) != version):
                    self._reviewed_unknowns[fact] = version
                    triggers.append(f"unknown:{fact}")
            complete = self.contract["completion"]
            if complete and self._matches("completion", complete, effective_values, current):
                triggers.append(f"completion_review:{complete['name']}")
            if not strict and self.actions >= self.config.max_actions:
                triggers.append("action_budget_review")
            report.update(event="review" if triggers else "clear", triggers=triggers)
        except BaseException as exc:
            self.previous = None
            self.ledger.invalidate(None, "monitor_error", action_id=report.get("action_id"))
            if not isinstance(exc, Exception):
                self._save_evidence("monitor_interrupted")
                raise
            report.update(event="monitor_error", error=f"{type(exc).__name__}: {exc}")
        report["elapsed_s"] = round(time.perf_counter() - started, 4)
        report["artifact_prefix"] = str(prefix)
        self._save_evidence(report["event"])
        self._record(report)
        self.paused = None
        if report["event"] != "clear":
            detail = report.get("error") or ", ".join(report.get("triggers", []))
            self._interrupt(report["event"] + ": " + detail)

    def _rearm_reviews(self, values: dict, novel: dict) -> None:
        """Acknowledge new evidence, never relabel a visual answer as planner approval."""
        for key, value in values.items():
            if novel[key] and value != "unknown":
                self._reviewed_unknowns.pop(key, None)
        for signature in tuple(self._reviewed_rules):
            _, predicates = signature
            if any(
                self.ledger.facts[key]["version"] != version
                or (novel.get(key) and values[key] != "unknown" and values[key] != expected)
                for key, version, expected in predicates
            ):
                self._reviewed_rules.remove(signature)

    def _matches(self, key: str, rule: dict, values: dict, observation: dict) -> bool:
        """Notify once per sustained condition; uncertainty is not evidence it cleared."""
        source_id = observation.get("source_id") or observation["observation_id"]
        if self._rule_sources.get(key) != source_id:
            matched = all(values[k] == v for k, v in rule["when"].items())
            self.counts[key] = self.counts.get(key, 0) + 1 if matched else 0
            self._rule_sources[key] = source_id
        signature = (key.split("-", 1)[0], tuple(
            (fact, self.ledger.facts[fact]["version"], expected)
            for fact, expected in sorted(rule["when"].items())
        ))
        if (self.counts.get(key, 0) >= rule["confirmations"]
                and signature not in self._reviewed_rules):
            self._reviewed_rules.add(signature)
            return True
        return False

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
            verification_calls=self.verifications,
            verification_limit=self.config.max_actions,
            unknown_streaks=dict(self.unknowns),
            reports=recent,
            report_count=len(self.reports),
            contract=self.contract,
            evidence_cameras=self.evidence_cameras,
            observation=self.reports[-1].get("observation") if self.reports else None,
            evidence=self.ledger.summary(),
            blocked=self.reports[-1].get("blocked", {}) if self.reports else {},
            audit_dir=str(self.directory),
        )


def monitor_prompt(config: MonitorConfig, *, instructions: str | None = None) -> str:
    return f"""# Optional visual review

Supervision mode: {config.mode}. Valen checks images after EACH successful motion
primitive, independently of robo.show(). Its answers are fallible visual evidence.
Before motion in EVERY python cell, declare phase-specific checks using:
watch(phase, facts, alerts=[], critical=None, completion=None, requires=None, invalidates=None)
facts maps 1..12 IDs to distinct, concrete yes/no visual questions. Name objects
and arms in the questions. critical defaults to []; opt in only when sustained
uncertainty warrants returning to you for judgment. An alert or completion rule has
{{"name": "...", "when": {{"fact_id": "yes"}}, "confirmations": 1}}.
All when predicates must match; values may be yes or no, never unknown.
Confirmations count distinct observation sources, not repeated requests for one image.
Call watch again BEFORE entering a different phase. Repeating the same watch in a
new cell preserves its confirmation window; changing its declaration starts a new
rule window. Fact uncertainty and explicitly registered prerequisites persist.
Each continuous critical-unknown episode requests review once. An alert or
completion condition also requests review once while it persists; a new known
contradiction rearms it. Repeating an image or changing cells does not rearm review.
After a review, inspect the evidence and write the next cell using your judgment:
continue, observe, correct or change strategy. Ordinary unknown does not require
Valen to approve your new cell. Its recorded answer remains unknown. A new known
observation resets that fact's unknown streak and rearms its uncertainty review.
critical handles consecutive unknowns; a confident no needs an alert. clear means
no pause rule fired, NOT that all facts are established.
Use verify(observation_id=None) when a declared prerequisite lacks fresh evidence,
or a dependent action needs an unresolved fact checked. A normal approach with no
such prerequisite needs no extra pre-action visual call; endpoint monitoring still
runs automatically. Reuse its evidence when it is current and sufficient.
verify reads a host-owned current snapshot and cannot move the robot or reconcile
an unresolved action. Call robo.snapshot(refresh=True) only when another sensor
capture is needed; repeating verify on the same source cannot add confirmations
or clear uncertainty. verify returns values, blocked prerequisites, and the
observation reference. At most {config.max_actions} explicit verification calls
(including failures) are allowed per cell, separately from the motion allowance.
Use checks appropriate after EVERY primitive in that phase. For example:
watch("transport mouse", {{"held": "Is the mouse held between the right gripper fingers?"}},
      alerts=[{{"name": "lost mouse", "when": {{"held": "no"}}}}])
# Only enter this phase AFTER grasp/lift has been verified, not while approaching.
# Assess grasp from the available images, motion and gripper results together.
# This watch requests review of a suspected loss; it does not require Valen yes.

requires and defer_fact are separate, explicit HARD visual prerequisites. Use
them only when that fresh visual answer must be established before every motion
in the named phase. Do not add requires={{"held": "yes"}} to routine transport.
New cells and ordinary review do not erase these gates. Device constraints and
unresolved execution receipts remain enforced independently of Valen.

In pause mode, an alert, completion candidate, {config.unknown_limit} consecutive
critical unknowns (once per episode of uncertainty), a monitor error, or
{config.max_actions} actions in one cell
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
""" + "\n\n" + (planning_instructions() if instructions is None else instructions)


def planning_instructions() -> str:
    """Canonical, packaged runtime core of the Astra–Valen planning skill."""
    return files(__package__ or "robouse").joinpath("knowledge/astra_valen_supervision.md").read_text(
        encoding="utf-8"
    )


def config_record(config: MonitorConfig, *, instructions: str | None = None) -> dict:
    record = asdict(config)
    if config.mode != "off":
        instructions = planning_instructions() if instructions is None else instructions
        record["planning_skill"] = dict(
            name="astra-valen-supervision", file="supervision_skill.md",
            sha256=hashlib.sha256(instructions.encode()).hexdigest(),
        )
    return record
