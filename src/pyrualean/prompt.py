# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Model-facing prompt built from the live library.

The API reference is generated from :class:`pyrualean.libero.LiberoRobot`'s
signatures and docstrings (``inspect``), so the model always sees exactly the
functions the host will inject.  The operating knowledge is a hand-written
markdown file that carries the same manipulation lessons RPent gives its
tool-calling agent, rewritten for someone who writes a program instead of
issuing one call per turn.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from ._robot import PRIMITIVE_SETS, RobotBase, effective_primitives
from .libero import RESULT_TYPES, LiberoRobot
from .task_acceptance import normalize_acceptance, render_acceptance

_HOST_ONLY = frozenset({"ledger", "backend"})
_LINE = 88


def _annotation(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, type):
        return value.__name__
    return str(value).replace("typing.", "")


def _format_parameters(sig: inspect.Signature) -> list[str]:
    parts: list[str] = []
    seen_kw_only = False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY and not seen_kw_only:
            parts.append("*")
            seen_kw_only = True
        text = name
        annotation = _annotation(param.annotation)
        if annotation:
            text += f": {annotation}"
        if param.default is not inspect.Parameter.empty:
            text += f" = {param.default!r}" if annotation else f"={param.default!r}"
        parts.append(text)
    return parts


def _render_callable(owner: str, name: str, fn: Any) -> str:
    sig = inspect.signature(fn)
    params = _format_parameters(sig)
    ret = _annotation(sig.return_annotation)
    suffix = f" -> {ret}" if ret else ""
    head = f"{owner}.{name}({', '.join(params)}){suffix}"
    if len(head) > _LINE:
        joined = ",\n    ".join(params)
        head = f"{owner}.{name}(\n    {joined},\n){suffix}"
    doc = inspect.getdoc(fn) or ""
    return head + "\n" + textwrap.indent(doc, "    ") if doc else head


def _render_property(owner: str, name: str, prop: property) -> str:
    fn = prop.fget
    ret = ""
    if fn is not None:
        ret = _annotation(inspect.signature(fn).return_annotation)
    head = f"{owner}.{name}" + (f" -> {ret}" if ret else "") + "  (read-only attribute)"
    doc = inspect.getdoc(fn) if fn is not None else ""
    return head + ("\n" + textwrap.indent(doc, "    ") if doc else "")


def _render_result_type(cls: type) -> str:
    parts = []
    for item in fields(cls):
        parts.append(f"{item.name}: {_annotation(item.type)}")
    head = f"class {cls.__name__}({', '.join(parts)})"
    if len(head) > _LINE:
        head = f"class {cls.__name__}(\n    " + ",\n    ".join(parts) + ",\n)"
    extra = []
    for name, member in vars(cls).items():
        if isinstance(member, property) and not name.startswith("_"):
            ret = _annotation(inspect.signature(member.fget).return_annotation)
            doc = inspect.getdoc(member.fget) or ""
            extra.append(f"    .{name} -> {ret}: {doc}")
    doc = inspect.getdoc(cls) or ""
    body = textwrap.indent(doc, "    ") if doc else ""
    return "\n".join(part for part in [head, body, *extra] if part)


def _public_members(robot_cls: type) -> dict[str, Any]:
    """Public methods and properties in definition order, base classes first.

    An override keeps the position of the member it overrides, so a robot
    that re-documents ``image`` for its own cameras does not reorder the
    reference.  ``RobotBase``'s configuration attributes never show, and
    neither do the members the robot declares absent (``ABSENT_MEMBERS``).
    """

    absent = frozenset(getattr(robot_cls, "ABSENT_MEMBERS", ()))
    members: dict[str, Any] = {}
    for klass in reversed(robot_cls.__mro__):
        if klass is object:
            continue
        for name, member in vars(klass).items():
            if name.startswith("_") or name in _HOST_ONLY or name in absent:
                continue
            if isinstance(member, property) or inspect.isfunction(member):
                members[name] = member
    return members


def _return_type_name(member: Any) -> str | None:
    fn = member.fget if isinstance(member, property) else member
    if fn is None:
        return None
    annotation = inspect.signature(fn).return_annotation
    if annotation is inspect.Signature.empty:
        return None
    return annotation if isinstance(annotation, str) else getattr(annotation, "__name__", None)


def vla_hidden(robot_cls: type, primitives: str = "full") -> tuple[frozenset[str], frozenset[str]]:
    """What the ``no-vla`` primitive set hides: ``(member names, result type names)``.

    The members are ``robot_cls.VLA_PRIMITIVES`` (methods, or constants of
    that name); the result types are those returned by a hidden method and
    by no remaining public member. ``full`` is intersected with host capabilities.
    """

    primitives = effective_primitives(robot_cls, primitives)
    if primitives == "full":
        return frozenset(), frozenset()
    names = frozenset(getattr(robot_cls, "VLA_PRIMITIVES", ()))
    members = _public_members(robot_cls)
    returned_by_vla = {_return_type_name(members[n]) for n in names if n in members}
    returned_by_rest = {_return_type_name(m) for n, m in members.items() if n not in names}
    return names, frozenset(t for t in returned_by_vla - returned_by_rest if t)


def _apply_edits(text: str, edits: tuple[tuple[str, str], ...], what: str) -> str:
    for old, new in edits:
        found = text.count(old)
        if found != 1:
            raise RuntimeError(f"{what} drifted: {old[:60]!r} found {found} times, expected once")
        text = text.replace(old, new)
    return text


def api_reference(
    robot_cls: type = LiberoRobot, *, var: str = "robo", primitives: str = "full",
    intent_only: bool | None = None,
) -> str:
    """Render the policy-facing API of ``robot_cls`` from its live definition.

    ``primitives="no-vla"`` leaves out the robot's ``VLA_PRIMITIVES``, the
    result types only they return, and (through ``VLA_REFERENCE_EDITS``) the
    sentences of the other docstrings that describe the VLA.
    Intent-capable robots default to the shared operation interface, omitting
    legacy controller tuning. ``intent_only=False`` explicitly renders the full
    library reference for existing scripts and controller development.
    """

    primitives = effective_primitives(robot_cls, primitives)
    vla_members, vla_types = vla_hidden(robot_cls, primitives)
    sections: list[str] = []
    intent_only = intent_only is not False and getattr(robot_cls, "INTENT_API", False)
    legacy_members = (set(getattr(robot_cls, "LEGACY_MOTION_METHODS", ())) | {"try_grasp"}
                      if intent_only else ())
    class_doc = inspect.getdoc(robot_cls) or ""
    sections.append(f"class {robot_cls.__name__}\n{textwrap.indent(class_doc, '    ')}")
    # Base-class configuration, ``RESULT_TYPES`` (rendered as the result-type
    # section below) and ``ClassVar``-annotated configuration are not
    # constants the policy needs.
    hidden = set(vars(RobotBase)) | {"RESULT_TYPES"} | set(vla_members)
    # A robot class may build on another robot's (the same robot on another host): what it
    # inherits from that one is as much its API as what it defines itself.
    lineage = [cls for cls in reversed(robot_cls.__mro__) if not issubclass(RobotBase, cls)]
    defined: dict[str, Any] = {}
    for cls in lineage:
        hidden |= {
            name
            for name, annotation in inspect.get_annotations(cls).items()
            if str(annotation).replace("typing.", "").startswith("ClassVar")
        }
        defined.update(vars(cls))
    constants = [
        f"{var}.{name} = {value!r}"
        for name, value in defined.items()
        if name.isupper() and not name.startswith("_") and name not in hidden
    ]
    if constants:
        sections.append("Constants:\n" + "\n".join(f"    {c}" for c in constants))
    members: list[str] = []
    for name, member in _public_members(robot_cls).items():
        if name in vla_members or name in legacy_members:
            continue
        if isinstance(member, property):
            members.append(_render_property(var, name, member))
        else:
            members.append(_render_callable(var, name, member))
    sections.append("Methods and attributes of `robo`:\n\n" + "\n\n".join(members))
    result_types = tuple(getattr(robot_cls, "RESULT_TYPES", ()) or RESULT_TYPES)
    results = [
        _render_result_type(cls)
        for cls in result_types
        if is_dataclass(cls) and cls.__name__ not in vla_types
        and not (intent_only and cls.__name__ == "Move")
    ]
    sections.append(
        "Result types (frozen dataclasses; access fields as attributes):\n\n" + "\n\n".join(results)
    )
    if getattr(robot_cls, "COORDINATION_API", False):
        from .coordination import RESULT_TYPES as MOTION_TYPES
        from .coordination import Pose

        sections.append("Coordination types:\n\n" + "\n\n".join(
            _render_result_type(cls) for cls in MOTION_TYPES
            if not intent_only or cls.__name__ in {"Pose", "Observation"}
        ))
        sections.append(_render_callable("pose", "shifted", Pose.shifted))
        sections.append(
            "Write cooperation strategy in Python: choose arm roles, targets, direction, "
            "holding requirements, completion evidence and budgets from current observations. "
            "Use execute_motion(id, left=..., right=..., interaction=..., directions=..., "
            "holding=..., completion=...) for routine motion. "
            "The executor selects spacing and speed, checks constraints, and retains progress. "
            "Motion, alignment and grasp results share result['operation']: a session-scoped "
            "ref, state, phase, cumulative budget, effect certainty, receipt and continuation. "
            "Use operation_status(ref) to inspect and resume_operation(ref) to continue only "
            "the already-started phase. ready/advance requires choosing the next grasp_step; "
            "completed means execution finished, never verified object/task success. "
            "unknown/reconcile forbids blind retry; needs_replan requires a revised decision. "
            "Completed IDs never move again. After reconciling an interruption, resume_motion(id) "
            "continues only known remaining work, retaining step and correction budgets. "
            "Use begin_grasp plus grasp_step for open/approach/align/close/probe, and align_local "
            "for explicit observed adjustments inside a fixed envelope. Do not wrap these or "
            "intent calls in once(); retain once() for set_gripper and home. Default action "
            "allowance is the remaining native episode budget, with no future-task reserve. "
            "A costly route can return paused/segment_complete after useful resting progress; "
            "resume the same ID. An explicit cap can be increased cumulatively with max_steps. "
            "Do not abandon a task or begin cleanup solely because a cost estimate suggests "
            "the whole remaining sequence may not fit. Continue useful constrained progress "
            "until completion, actual episode termination or a physical/execution blocker. "
            "A preview is not a simulation of later stages. A grasp, direction or route change "
            "needs a new intent based on "
            "the actual scene. Check reached, needs_model and needs_observation before dependent "
            "actions. Semantic completion is never inferred from pose or obstructed closure. "
            "Choose grasps backwards from the object's remaining use: its working region "
            "must be free when used and the receiver grasp region accessible during transfer. "
            "A donor may temporarily hold a future working region if transfer frees it. "
            "For a handoff, choose both "
            "grasp regions and a mutually reachable exchange pose before pickup. A changed "
            "grasp or slip requires rechecking transfer and subsequent use, even if the "
            "new hold is stable. Account for finger clearance, not just TCP separation; "
            "verify support before release. "
            "On snapshot backends, object_ref(name, description) preserves a logical object "
            "identity across grasps. Pass object_ref=ref to begin_grasp while the target is "
            "visible: preparation/open/approach/align return object_tracking. The executor "
            "tracks material features through native RGB frames and closure, without extra "
            "actions. If preclosure features are missing, observe the visible object or "
            "supply named current landmarks via observe_object; do not wait until occlusion. "
            "Only actual post-closure positions form the fixed hold baseline. close/probe "
            "return object_evidence. Preclosure tracking is not hold evidence. Inspect "
            "baseline_ready, baseline_quality.space and recovery before dependent actions. "
            "An empty capture is not a baseline; repair its exact capture_observation_id "
            "using visible matching landmarks, or revise visibility/grasp. Do not repeatedly "
            "probe to manufacture evidence. observe_hold compares against the fixed capture "
            "and reassociates original features after lost tracks without rebasing. "
            "Repeated reset=True cannot erase drift within one grasp. following/relative_motion "
            "are 3D evidence; image_consistent/image_relative_motion are explicitly limited "
            "wrist-image 2D evidence and cannot verify depth or a stable 3D hold. "
            "Inspect state/reason/features; unknown is not empty. Evidence changes require "
            "fresh observation and strategy review. Do not infer a held object's pose from "
            "table-height geometry. "
            "Read motion_capabilities: object/inter-arm collision checking and force control "
            "are not provided by current adapters. Split at "
            "required observation, contact or release decisions. Timed trajectories are "
            "continuous within uninterrupted execution, not across interruptions or model waits. "
            "CoordinationPause(BaseException) and SupervisionPause stop code for review; do not "
            "catch these interruptions. Already executed actions remain done. Use watch when "
            "monitoring is enabled; only declared prerequisites are enforced. Skills provide "
            "guidance, not a prescribed task sequence or object-support verdict."
        )
    # A robot without ``done`` never ends an episode as "terminated".
    absent = frozenset(getattr(robot_cls, "ABSENT_MEMBERS", ()))
    reasons = "truncated | timeout | cancelled"
    if "done" not in absent:
        reasons = "terminated | " + reasons
    sections.append(
        "Exceptions:\n"
        "    ToolError(RuntimeError): a call was rejected or a service failed; "
        "catch it if you have a fallback.\n"
        "    ValueError: an argument was invalid (e.g. a planar move over "
        "0.30 m) - nothing was executed.\n"
        "    EpisodeFinished(BaseException): the episode is over "
        f"(reason: {reasons}); return."
    )
    text = "\n\n".join(sections)
    class_section = sections[0]
    if primitives == "no-vla":
        edits = tuple(getattr(robot_cls, "VLA_REFERENCE_EDITS", ()))
        text = _apply_edits(text, edits, f"{robot_cls.__name__} docstrings")
        for old, new in edits:
            class_section = class_section.replace(old, new)
    if intent_only:
        text = text.replace(class_section, f"class {robot_cls.__name__}\n"
                            "    Two-arm robot. Declare motion through execute_motion; the "
                            "executor owns spacing, timing, corrections and continuation. "
                            "Positions are world TCP metres; angles follow robot conventions "
                            "below. Low-level motion methods are disabled in this session.", 1)
    return text


def knowledge_stem(name: str, primitives: str = "full") -> str:
    """Knowledge file stem: ``<name>`` for the full primitive set, ``<name>-novla`` otherwise."""

    if primitives not in PRIMITIVE_SETS:
        raise ValueError(f"primitives must be one of {PRIMITIVE_SETS}, got {primitives!r}")
    return name if primitives == "full" else f"{name}-novla"


def knowledge_text(name: str = "libero", primitives: str = "full", *,
                   intent_only: bool | None = None) -> str:
    """Return the packaged operating-knowledge markdown for a backend.

    ``primitives="no-vla"`` reads ``knowledge/<name>-novla.md``: the same
    document with the sections and sentences about the VLA removed.
    RoboDojo defaults to the shared operation protocol; ``intent_only=False``
    explicitly includes legacy controller guidance.
    """

    stem = knowledge_stem(name, primitives)
    package = resources.files("pyrualean")
    text = package.joinpath(f"knowledge/{stem}.md").read_text(encoding="utf-8")
    if intent_only is not False and name in {"robodojo", "robodojo_xpl_rgb"}:
        # Share sensing/acceptance facts with the adapter-specific knowledge;
        # legacy controller tuning instructions do not enter the policy prompt.
        sections = {part.split("\n", 1)[0]: "## " + part for part in text.split("\n## ")[1:]}
        motion = package.joinpath("knowledge/robodojo-intent.md").read_text(encoding="utf-8")
        return "\n\n".join([motion.rstrip()] + [sections[key].rstrip() for key in (
            "The VLA", "Perception",
            "Judging progress", "Safety and recovery",
        ) if key in sections]) + "\n"
    return text


@dataclass
class TaskCard:
    """What the policy author is told about one evaluation cell.

    The host assembles benchmark instructions, public observations and optional
    executor capabilities. Sources can identify native data, derived values and
    configuration fallbacks. The planner does not author the card; no privileged
    object poses or evaluator verdicts are included.
    """

    suite: str
    task: int | str
    seed: int
    task_language: str
    object_names: list[str]
    #: Initial gripper pose of a single-arm robot (LIBERO).  Robots with
    #: several arms leave these ``None`` and describe each arm in ``facts``.
    eef_pos: list[float] | None = None
    eef_quat: list[float] | None = None
    gripper_opening: float | None = None
    images: dict[str, str] = field(default_factory=dict)
    max_env_steps: int = 10000
    #: ``False``: the program runs on exactly this cell (per-cell single shot).
    #: ``True``: the program will be run on other seeds of the same task, so
    #: the images are only an example layout (transfer mode).
    transfer: bool = False
    #: Which robot the card describes (one of :data:`pyrualean.robots.ROBOTS`).
    robot: str = "libero"
    #: Robot-specific initial facts rendered as extra bullets (non-LIBERO
    #: robots put their identifiers and initial pose here).
    facts: dict[str, Any] = field(default_factory=dict)
    #: Origin of host-assembled fields; optional for older cards and other robots.
    sources: dict[str, str] = field(default_factory=dict)
    #: Unrounded public robot measurements used to derive the displayed initial facts.
    initial_state: dict[str, Any] = field(default_factory=dict)
    #: Source identifiers, clock domains and artifact digests retained in card.json.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Supplemental public requirements and their coverage/source, never hidden scoring rules.
    #: The original task_language remains authoritative; an empty list means unspecified.
    acceptance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.acceptance = normalize_acceptance(self.acceptance)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> TaskCard:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def render(self, *, compact: bool = False) -> str:
        """Render full audit detail, or a concise startup view without changing stored data."""
        if self.robot != "libero" or self.eef_pos is None or self.gripper_opening is None:
            return self._render_generic(compact=compact and self.robot.startswith("robodojo"))
        eef = ", ".join(f"{v:.3f}" for v in self.eef_pos)
        lines = [
            "# Task card",
            f"- suite: {self.suite}",
            f"- task: {self.task}",
            f"- seed: {self.seed}",
            f"- task instruction (authoritative): {self.task_language!r}",
            f"- movable objects in the scene: {', '.join(self.object_names)}",
            f"- initial gripper position [x, y, z]: [{eef}] m, opening "
            f"{self.gripper_opening:.3f} (open, pointing down)",
            f"- simulator step budget: {self.max_env_steps}",
        ]
        if self.images:
            names = ", ".join(f"{k} ({Path(v).name})" for k, v in self.images.items())
            if self.transfer:
                lines.append(
                    f"- attached images: {names}. They show ONE example layout "
                    f"(seed {self.seed}) of this task.  Your program will be run "
                    "on OTHER seeds of the same task, where the objects and the "
                    "container are placed differently, so the images only tell "
                    "you what the objects look like and what the task means.  "
                    "Do not hard-code pixel coordinates or world positions read "
                    "off these images; localise everything at runtime with "
                    "`robo.segment`, `robo.back_project`, `robo.region_center` "
                    "and `robo.world_map`."
                )
            else:
                lines.append(
                    f"- attached images: {names}. They are step 0 of your run, "
                    "exactly what `robo.image('agentview')` / "
                    "`robo.image('wrist')` return before any motion; a pixel "
                    "(row, col) you read off the 1024x1024 agentview image can be "
                    "turned into world coordinates at runtime with "
                    "`robo.back_project(row, col, step=0)`."
                )
        lines.extend(render_acceptance(self.acceptance))
        return "\n".join(lines)

    def _render_generic(self, *, compact: bool = False) -> str:
        """Card of a non-LIBERO robot: identifiers, instruction, facts, images."""

        lines = [
            "# Task card",
            f"- robot: {self.robot}",
            f"- task: {self.task}",
        ]
        if self.suite:
            lines.append(f"- variant: {self.suite}")
        lines += [
            f"- seed: {self.seed}",
            f"- task instruction (authoritative): {self.task_language!r}",
        ]
        if self.object_names:
            lines.append(f"- objects named by the benchmark: {', '.join(self.object_names)}")
        if not compact:
            for key, value in self.sources.items():
                lines.append(f"- source of {key}: {value}")
        for key, value in self.facts.items():
            if compact and key == "motion capabilities at episode start":
                lines.extend(self._motion_summary(value))
                continue
            if isinstance(value, dict):
                lines.append(f"- {key}:\n\n```json\n"
                             + json.dumps(value, indent=2, ensure_ascii=False) + "\n```\n")
            else:
                lines.append(f"- {key}: {value}")
        if "native action budget" not in self.facts:
            lines.append(f"- simulator step budget: {self.max_env_steps}")
        if compact:
            lines.extend(self._startup_summary())
        elif self.provenance:
            lines.append("\nTask-card provenance:\n\n```json\n"
                         + json.dumps(self.provenance, indent=2, ensure_ascii=False) + "\n```\n")
        if self.images:
            names = ", ".join(f"{k} ({Path(v).name})" for k, v in self.images.items())
            cameras = ", ".join(f"`robo.image({k!r})`" for k in self.images)
            if self.transfer:
                lines.append(
                    f"- attached images: {names}. They show ONE example layout "
                    f"(seed {self.seed}) of this task; your program will run on "
                    "other seeds where objects are placed differently, so do "
                    "not hard-code pixels or world positions read off them."
                )
            elif compact:
                lines.append(f"- initial images: {names}; copied from recorded step 0.")
            else:
                lines.append(
                    f"- attached images: {names}. They are step 0 of your run, "
                    f"exactly what {cameras} return before any motion."
                )
        lines.extend(render_acceptance(self.acceptance))
        return "\n".join(lines)

    @staticmethod
    def _motion_summary(caps: dict) -> list[str]:
        def flag(name):
            value = caps.get(name)
            return "yes" if value is True else "no" if value is False else "unknown"

        lines = [f"- timed joint commands: {flag('speed_control')}; "
                 f"continuous via trajectory: {flag('continuous_trajectory')}"]
        if caps.get("motion_intents"):
            lines.extend([
                "- execute_motion(id, ...): supply goals, interaction, allowed directions, "
                "holding requirements, completion conditions and step allocation",
                "- executor chooses spacing/speed in simulation time and bounded arrival "
                "corrections; resume_motion(id) continues known remaining work",
                "- a completed intent never dispatches again; direction/grasp/path changes "
                "require a new intent; semantic completion still needs observation",
                "- preview=True estimates without moving; low-level motion profiles remain "
                "available for explicit control requirements",
            ])
        elif caps.get("speed_control") is True:
            defaults = caps.get("trajectory_defaults") or {}
            values = ", ".join(f"{name}={defaults.get(name, 'unknown')} {unit}"
                               for name, unit in (("velocity", "rad/s"),
                                                  ("acceleration", "rad/s²"), ("jerk", "rad/s³")))
            lines.append(f"- default joint-command limits: {values}; "
                         "simulation time, no measured-speed guarantee")
            if caps.get("motion_phases"):
                lines.append("- motion phases: transit / approach / carry / precision; "
                             "choose explicitly from task intent and observed clearance")
            if caps.get("motion_preview"):
                lines.append("- move_to/move_both(preview=True): read-only step/time estimate; "
                             "execution replans from current state")
            if caps.get("motion_window"):
                lines.append("- motion_window(far, near, distance_m=0.02, at='end'/'start'): "
                             "slow only a nominal endpoint region in one continuous move")
        lines.append("- force capabilities: " + "; ".join(
            f"{name}={flag(name)}" for name in
            ("force_control", "force_feedback", "gripper_force_control", "impedance_control")))
        checks = ", ".join(caps.get("collision_checks", [])) or "unknown"
        lines.append(f"- collision checks: {checks}; "
                     f"inter-arm clearance={flag('inter_arm_clearance')}; "
                     f"object collision={flag('object_collision')}")
        return lines

    def _startup_summary(self) -> list[str]:
        source = self.provenance.get("initial_observation") or {}
        identity = source.get("observation_id") or "unavailable"
        native = source.get("native_steps")
        alignment = "reported aligned" if source.get("aligned") is True else "unverified"
        images = self.provenance.get("images") or {}
        if any(item.get("aligned") is not True for item in images.values()):
            alignment = "unverified"
        if any(item.get("image_source_id") and any(
                   state_id and state_id != item["image_source_id"]
                   for state_id in (source.get("source_id"), item.get("image_state_source_id")))
               for item in images.values()):
            alignment = "mismatched"
        lines = [f"- initial observation: {identity}; recorded step 0; native actions "
                 f"{native if native is not None else 'unknown'}; "
                 f"image/state alignment {alignment}"]
        budget = self.provenance.get("budget_at_card_creation") or {}
        if budget.get("remaining") is not None:
            lines.append(f"- native actions remaining at card creation: {budget['remaining']}")
        if "fallback" in self.sources.get("native action budget", ""):
            lines.append("- budget source: configured fallback; native limit unavailable")
        if not self.task_language:
            lines.append("- task instruction unavailable; do not infer it from the task identifier")
        lines.append("- startup snapshot only; use robo.state() and robo.motion_capabilities() "
                     "for current data. Full source records are saved in card.json.")
        return lines


@dataclass(frozen=True)
class PromptBundle:
    """System text, task text and image attachments for one generation."""

    system: str
    user: str
    images: tuple[str, ...] = ()

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def text(self) -> str:
        return self.system + "\n\n" + self.user


CONTRACT = """\
# What you are asked to do

You control {robot_blurb} by WRITING A
PROGRAM, not by chatting with the robot.  Reply with one complete Python file
that defines

    def run(robo):

The host imports the file, constructs `robo` (a `{cls}` bound to the
live simulator{services}) and calls
`run(robo)` exactly once.  No language model runs while the program executes:
every decision has to be made in code, from the return values and images
documented below.

# How the episode is scored

{scoring_head}
- The return value of `run` is ignored.
{scoring_limits}
{scoring_tail}

# Output format

Reply with the complete source of `policy.py` in a single ```python fenced
block and nothing else.  The program may import `math`, `time` and `numpy`
(as `np`); do not import anything else, read files or start threads.  Keep it
self-contained: helper functions live in the same file.  Print short progress
lines with `print(...)`; stdout is saved with the episode.
"""


#: The scoring bullets of :data:`CONTRACT` (first, limits, last): ``True`` for
#: a robot whose benchmark reports success while ``run`` executes, ``False``
#: for one judged from the scene the program leaves behind
#: (``RobotAdapter.live_success``), where a limit is not itself a failure.
SCORING_NOTES = {
    True: (
        "- Success = the benchmark's own task predicate fires (`terminated`) at any\n"
        "  moment while `run` executes.  The simulator checks it, not you.",
        "- The episode counts as a failure when `run` raises an uncaught exception,\n"
        "  when the host's wall-clock budget of {budget_s} s runs out, or when the\n"
        "  simulator step budget ({max_env_steps} steps) is exhausted.",
        "- Once the task terminates, every further motion call raises\n"
        "  `EpisodeFinished` (a BaseException): just return.",
    ),
    False: (
        "- Success is judged by the benchmark from the scene `run` leaves behind.\n"
        "  Nothing reports it while `run` executes: no result carries a task-success\n"
        "  flag, so the program must decide from its own observations when the\n"
        "  instruction is satisfied, and return.",
        "- `run` is stopped by an uncaught exception, by the host's wall-clock budget\n"
        "  of {budget_s} s or by the simulator step budget ({max_env_steps} steps); the\n"
        "  scene is judged as it is then.",
        "- Once the simulator step budget is exhausted, every further motion call\n"
        "  raises `EpisodeFinished` (a BaseException): just return.",
    ),
}


def render_prompt(
    card: TaskCard | None = None,
    *,
    budget_s: int = 1200,
    knowledge: str | None = None,
    robot_cls: type | None = None,
    primitives: str = "full",
) -> PromptBundle:
    """Assemble the generation prompt for one cell (or a cell-less preview).

    The robot comes from ``card.robot`` (``libero`` when there is no card):
    its class supplies the API reference, its knowledge file the operating
    knowledge and its adapter the wording of the contract.  ``primitives``
    is the primitive set (:data:`pyrualean._robot.PRIMITIVE_SETS`).
    """

    from .robots import get_adapter

    adapter = get_adapter(card.robot if card is not None else "libero")
    primitives = effective_primitives(robot_cls or adapter.robot_cls, primitives)
    max_env_steps = card.max_env_steps if card is not None else 10000
    services = adapter.services if primitives == "full" else adapter.services_novla
    scoring_head, scoring_limits, scoring_tail = SCORING_NOTES[bool(adapter.live_success)]
    contract = CONTRACT.format(
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        robot_blurb=adapter.blurb,
        cls=(robot_cls or adapter.robot_cls).__name__,
        services=f", {services}" if services else "",
        scoring_head=scoring_head,
        scoring_limits=scoring_limits.format(budget_s=budget_s, max_env_steps=max_env_steps),
        scoring_tail=scoring_tail,
    )
    reference = api_reference(robot_cls or adapter.robot_cls, primitives=primitives)
    know = (
        knowledge
        if knowledge is not None
        else knowledge_text(adapter.knowledge, primitives=primitives)
    )
    system = "\n\n".join(
        [
            contract.rstrip(),
            "# API reference (generated from the library the host injects)\n\n" + reference,
            know.strip(),
        ]
    )
    if card is None:
        user = "# Task card\n(no cell selected; render with a TaskCard for a real run)"
        images: tuple[str, ...] = ()
    else:
        user = card.render(compact=True)
        images = tuple(card.images.values())
    return PromptBundle(system=system, user=user, images=images)


__all__ = [
    "CONTRACT",
    "PromptBundle",
    "TaskCard",
    "api_reference",
    "knowledge_stem",
    "knowledge_text",
    "render_prompt",
    "vla_hidden",
]
