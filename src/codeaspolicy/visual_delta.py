# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""PEAR Visual-Delta protocol v1: parsing, units, prompts and pluggable policies.

The top-level Code planner does the global work (which object, which arm, where to
carry it).  The last few centimetres of contact/alignment are delegated to a closed
loop that asks a visual delta policy (a PEAR fine-tuned VLM or any local policy) for
one small relative move at a time, executes it with the guarded ``nudge`` executor
and reports the physical outcome back.  This module is the robot-independent half:

* the action JSON (``parse_visual_delta_response`` / ``validate_delta_action``);
* unit, frame and gripper-sign converters shared with the distillation tooling;
* observation and execution-feedback JSON, system and per-turn prompts;
* ``VisualDeltaPolicy`` with an OpenAI-compatible client, scripted and callable
  policies;
* ``run_visual_delta_loop``, the bounded multi-turn loop used both by the host tool
  and by the in-process fallback of ``robo.visual_align_and_insert``.

Protocol v1 action JSON (relative deltas; ``0`` for unchanged fields)::

    single arm: {"arm": {"x": cm, "y": cm, "z": cm, "roll": rad, "pitch": rad,
                         "yaw": rad},
                 "gripper": {"width": g}, "done": false}
    both arms:  {"left_arm": {...}, "right_arm": {...},
                 "left_gripper": {"width": g}, "right_gripper": {"width": g}}

``x, y, z`` are centimetres, ``roll, pitch, yaw`` radians.  In the world frame the
position delta moves the grasp point (TCP, 0.145 m out along link6's x axis) and the
angle deltas are added to RoboDojo's gripper angles (``robodojo_kinematics``:
``pitch`` / ``yaw`` / ``roll`` in degrees); in the ``tcp`` frame the position delta
is along the grasp frame's axes (+x pointing, +y finger-closing axis, +z normal) and
the angle deltas rotate about those axes.  Action gripper ``width`` is a directional
switch: ``> 0`` opens, ``< 0`` closes, ``0`` keeps (RoboDojo's own commands are the
opposite sign: ``OPEN = -1``, ``CLOSE = 1``, ``KEEP = 0``).  Observation gripper
``width`` is the measured jaw span in centimetres.  The assistant answers with an
optional ``<think>...</think>`` followed by one fenced ``json`` block.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

PROTOCOL = "pear.visual_delta.v1"
ARM_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
POSITION_FIELDS = ("x", "y", "z")
ANGLE_FIELDS = ("roll", "pitch", "yaw")
FRAMES = ("world", "tcp")
ARM_MODES = ("left", "right", "both")
SINGLE_KEYS = ("arm", "gripper")
BIMANUAL_KEYS = ("left_arm", "right_arm", "left_gripper", "right_gripper")
#: Width of what the fingers span at a measured opening (``robodojo_tools``).
GRIP_ZERO_M = 0.0023
GRIP_SPAN_M = 0.1091
#: RoboDojo's gripper commands (``robo.OPEN`` / ``robo.CLOSE`` / ``robo.KEEP``).
ROBODOJO_OPEN = -1
ROBODOJO_CLOSE = 1
ROBODOJO_KEEP = 0
#: PEAR's directional gripper switch in action JSON.
PEAR_OPEN = 1
PEAR_CLOSE = -1
PEAR_KEEP = 0
#: A delta whose every component is below these is "no motion" (convergence).
ZERO_CM = 0.05
ZERO_RAD = 0.005
#: Beyond this pitch (degrees) yaw and roll are folded into one heading.
NEAR_VERTICAL_DEG = 85.0
#: Hard sanity bounds on a single parsed delta (a malformed answer, not a move).
MAX_ABS_CM = 50.0
MAX_ABS_RAD = math.pi
FEEDBACK_STATUSES = ("reached", "stopped", "out_of_reach")


class VisualDeltaError(ValueError):
    """A response or action that does not follow the PEAR Visual-Delta protocol."""


# ---------------------------------------------------------------------------
# units, frames and gripper signs
# ---------------------------------------------------------------------------


def cm_to_m(value: float) -> float:
    return float(value) / 100.0


def m_to_cm(value: float) -> float:
    return float(value) * 100.0


def rad_to_deg(value: float) -> float:
    return math.degrees(float(value))


def deg_to_rad(value: float) -> float:
    return math.radians(float(value))


def wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def width_cm(opening: float) -> float:
    """Observed jaw span in cm of a normalised RoboDojo opening (1.0 open, 0.0 closed)."""

    return round((GRIP_ZERO_M + GRIP_SPAN_M * float(opening)) * 100.0, 1)


def pear_gripper_to_robodojo(width: float) -> int:
    """PEAR action switch (> 0 open, < 0 close, 0 keep) -> RoboDojo command (-1/1/0)."""

    value = float(width)
    if not math.isfinite(value):
        raise VisualDeltaError("gripper width must be finite")
    if value > 0:
        return ROBODOJO_OPEN
    if value < 0:
        return ROBODOJO_CLOSE
    return ROBODOJO_KEEP


def robodojo_gripper_to_pear(command: float | None) -> int:
    """RoboDojo command (-1 open, 1 close, 0/None keep) -> PEAR switch (+1/-1/0)."""

    if command is None:
        return PEAR_KEEP
    value = float(command)
    if value < 0:
        return PEAR_OPEN
    if value > 0:
        return PEAR_CLOSE
    return PEAR_KEEP


def ee_command_to_pear(ee_command: float | None) -> int:
    """A native ``<arm>_ee_joint_state`` command (1.0 open .. 0.0 closed) -> PEAR switch.

    ``None`` (no change commanded) keeps; anything at or below 0.05 is a close (the
    host's own threshold) and anything above an open.
    """

    if ee_command is None:
        return PEAR_KEEP
    return PEAR_CLOSE if float(ee_command) <= 0.05 else PEAR_OPEN


def zero_arm() -> dict[str, float]:
    return {name: 0.0 for name in ARM_FIELDS}


def arm_delta_to_motion(delta: Mapping[str, float]) -> dict[str, Any]:
    """PEAR arm delta (cm / rad) -> executor units: ``dxyz`` metres, angles degrees."""

    return {
        "dxyz": [cm_to_m(delta.get(k, 0.0)) for k in POSITION_FIELDS],
        "dpitch": rad_to_deg(delta.get("pitch", 0.0)),
        "dyaw": rad_to_deg(delta.get("yaw", 0.0)),
        "droll": rad_to_deg(delta.get("roll", 0.0)),
    }


def motion_to_arm_delta(dxyz_m: Sequence[float], dpitch_deg: float = 0.0,
                        dyaw_deg: float = 0.0, droll_deg: float = 0.0,
                        *, digits: int | None = None) -> dict[str, float]:
    """Executor units (metres / degrees) -> PEAR arm delta (cm / rad)."""

    out = {
        "x": m_to_cm(dxyz_m[0]), "y": m_to_cm(dxyz_m[1]), "z": m_to_cm(dxyz_m[2]),
        "roll": deg_to_rad(droll_deg), "pitch": deg_to_rad(dpitch_deg),
        "yaw": deg_to_rad(dyaw_deg),
    }
    if digits is not None:
        out = {k: round(v, digits if k in POSITION_FIELDS else digits + 1) for k, v in out.items()}
    return out


def canonical_angles(pitch: float, yaw: float, roll: float) -> tuple[float, float, float]:
    """Gripper angles (degrees) with the near-vertical yaw/roll ambiguity folded.

    Pointing within ``90 - NEAR_VERTICAL_DEG`` degrees of straight down (up), yaw and
    roll turn about almost the same axis and only ``yaw - roll`` (``yaw + roll``) is
    well defined: report that as the heading and roll 0.
    """

    pitch, yaw, roll = float(pitch), float(yaw), float(roll)
    if abs(pitch) > NEAR_VERTICAL_DEG:
        return pitch, wrap_deg(yaw - math.copysign(1.0, pitch) * roll), 0.0
    return pitch, wrap_deg(yaw), wrap_deg(roll)


def angle_delta(before: Sequence[float], after: Sequence[float]) -> tuple[float, float, float]:
    """(dpitch, dyaw, droll) degrees between two (pitch, yaw, roll) gripper angles.

    Plain wrapped differences, except when BOTH poses are near vertical: then the
    heading change is taken from the folded angles and droll is 0 (a tiny turn near
    vertical must not read as two opposite ~90 degree yaw and roll changes).
    """

    p0, y0, r0 = (float(v) for v in before)
    p1, y1, r1 = (float(v) for v in after)
    if abs(p0) > NEAR_VERTICAL_DEG and abs(p1) > NEAR_VERTICAL_DEG:
        c0, c1 = canonical_angles(p0, y0, r0), canonical_angles(p1, y1, r1)
        return p1 - p0, wrap_deg(c1[1] - c0[1]), 0.0
    return wrap_deg(p1 - p0), wrap_deg(y1 - y0), wrap_deg(r1 - r0)


def pose_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    """World-frame PEAR arm delta between two pose summaries.

    A pose summary has ``xyz`` (grasp point, metres) and ``pitch`` / ``yaw`` /
    ``roll`` (degrees, ``robodojo_kinematics.angles``).  Angles follow
    :func:`angle_delta` (wrapped; folded near vertical).
    """

    d = [float(a) - float(b) for a, b in zip(after["xyz"], before["xyz"], strict=True)]
    dpitch, dyaw, droll = angle_delta(
        (before["pitch"], before["yaw"], before["roll"]),
        (after["pitch"], after["yaw"], after["roll"]))
    return motion_to_arm_delta(d, dpitch, dyaw, droll)


def is_zero_delta(delta: Mapping[str, float], *, zero_cm: float = ZERO_CM,
                  zero_rad: float = ZERO_RAD) -> bool:
    return (all(abs(float(delta.get(k, 0.0))) < zero_cm for k in POSITION_FIELDS)
            and all(abs(float(delta.get(k, 0.0))) < zero_rad for k in ANGLE_FIELDS))


def is_zero_action(action: Mapping[str, Any], **kwargs: float) -> bool:
    """No arm motion and no gripper switch in a validated action."""

    for key, value in action.items():
        if key.endswith("arm") and not is_zero_delta(value, **kwargs):
            return False
        if key.endswith("gripper") and float(value.get("width", 0.0)) != 0.0:
            return False
    return True


# ---------------------------------------------------------------------------
# parsing and validation
# ---------------------------------------------------------------------------

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisualDeltaError(f"{where} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise VisualDeltaError(f"{where} must be finite, got {value!r}")
    return out


def _arm_block(value: Any, where: str, strict: bool) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise VisualDeltaError(f"{where} must be an object with fields {ARM_FIELDS}")
    extra = sorted(set(value) - set(ARM_FIELDS))
    if extra:
        raise VisualDeltaError(f"{where} has unknown fields {extra}")
    missing = [k for k in ARM_FIELDS if k not in value]
    if missing and strict:
        raise VisualDeltaError(f"{where} is missing fields {missing}; use 0 for unchanged")
    out = {}
    for key in ARM_FIELDS:
        number = _number(value.get(key, 0.0), f"{where}.{key}")
        bound = MAX_ABS_CM if key in POSITION_FIELDS else MAX_ABS_RAD
        if abs(number) > bound:
            raise VisualDeltaError(f"{where}.{key}={number} is outside +-{bound}")
        out[key] = number
    return out


def _gripper_block(value: Any, where: str, strict: bool) -> dict[str, float]:
    if value is None:
        if strict:
            raise VisualDeltaError(f"{where} is required; width 0 keeps the gripper")
        return {"width": 0.0}
    if not isinstance(value, Mapping):
        raise VisualDeltaError(f"{where} must be an object with a width field")
    extra = sorted(set(value) - {"width"})
    if extra:
        raise VisualDeltaError(f"{where} has unknown fields {extra}")
    if "width" not in value:
        if strict:
            raise VisualDeltaError(f"{where}.width is required; 0 keeps the gripper")
        return {"width": 0.0}
    width = _number(value["width"], f"{where}.width")
    # normalised to the switch it means: +1 open, -1 close, 0 keep
    return {"width": float((width > 0) - (width < 0))}


def validate_delta_action(action: Any, arm: str = "right", *, strict: bool = True
                          ) -> dict[str, Any]:
    """Check one PEAR v1 action and return it normalised.

    ``arm`` selects the mode: ``"left"`` / ``"right"`` expect ``arm`` + ``gripper``,
    ``"both"`` expects ``left_arm`` / ``right_arm`` / ``left_gripper`` /
    ``right_gripper``.  With ``strict`` every field is required (``0`` for unchanged)
    and unknown keys are refused; without it missing fields read as ``0``.  The
    result has float fields, gripper widths normalised to the switch they mean
    (+1 open, -1 close, 0 keep) and a boolean ``done``.

    Raises:
        VisualDeltaError: the action does not follow the protocol.
    """

    if arm not in ARM_MODES:
        raise VisualDeltaError(f"arm mode must be one of {ARM_MODES}, got {arm!r}")
    if not isinstance(action, Mapping):
        raise VisualDeltaError("the action must be a JSON object")
    done = action.get("done", False)
    if not isinstance(done, bool):
        raise VisualDeltaError(f"done must be a boolean, got {done!r}")
    keys = SINGLE_KEYS if arm != "both" else BIMANUAL_KEYS
    extra = sorted(set(action) - set(keys) - {"done"})
    if extra:
        if set(extra) & set(BIMANUAL_KEYS):
            hint = "; bimanual keys need arm mode 'both'"
        elif set(extra) & set(SINGLE_KEYS):
            hint = "; single-arm keys need arm mode 'left' or 'right'"
        else:
            hint = ""
        raise VisualDeltaError(f"unknown keys {extra}{hint}")
    out: dict[str, Any] = {}
    for key in keys:
        value = action.get(key)
        if key.endswith("arm"):
            if value is None:
                if strict:
                    raise VisualDeltaError(f"{key} is required; all-zero deltas keep the arm")
                out[key] = zero_arm()
            else:
                out[key] = _arm_block(value, key, strict)
        else:
            out[key] = _gripper_block(value, key, strict)
    out["done"] = done
    return out


def split_response(text: str) -> tuple[str | None, str]:
    """(think text or None, the JSON text) of an assistant answer.

    The JSON is the content of the last fenced block; without a fence, the outermost
    ``{...}`` after any think block is taken.
    """

    if not isinstance(text, str):
        raise VisualDeltaError("the response must be text")
    think_match = _THINK.search(text)
    think = think_match.group(1).strip() if think_match else None
    rest = text[think_match.end():] if think_match else text
    fences = _FENCE.findall(rest)
    if fences:
        return think, fences[-1].strip()
    start, end = rest.find("{"), rest.rfind("}")
    if start < 0 or end <= start:
        raise VisualDeltaError("no fenced json block or JSON object in the response")
    return think, rest[start:end + 1]


def parse_visual_delta_response(text: str, arm: str = "right", *, strict: bool = True
                                ) -> dict[str, Any]:
    """Parse an assistant answer: optional think block, then one fenced json block.

    Returns ``{"think": str | None, "action": <validated action>, "done": bool,
    "raw_json": str}``.

    Raises:
        VisualDeltaError: no JSON, invalid JSON or an action off the protocol.
    """

    think, raw = split_response(text)
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisualDeltaError(f"invalid JSON: {exc.msg} at char {exc.pos}") from exc
    normalised = validate_delta_action(action, arm, strict=strict)
    return {"think": think, "action": normalised, "done": normalised["done"], "raw_json": raw}


def action_arms(action: Mapping[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    """Per-arm executor specs of a validated action.

    Each spec has ``dxyz`` (metres), ``dpitch`` / ``dyaw`` / ``droll`` (degrees) and
    ``gripper`` in RoboDojo's command convention (-1 open, 1 close, 0 keep).
    """

    names = ("left", "right") if arm == "both" else (arm,)
    out = {}
    for name in names:
        delta = action["arm"] if arm != "both" else action[f"{name}_arm"]
        grip = action["gripper"] if arm != "both" else action[f"{name}_gripper"]
        spec = arm_delta_to_motion(delta)
        spec["gripper"] = pear_gripper_to_robodojo(grip.get("width", 0.0))
        out[name] = spec
    return out


def format_action(arms: Mapping[str, Mapping[str, Any]], arm: str, *, done: bool = False,
                  digits: int = 1) -> dict[str, Any]:
    """Executor specs (metres / degrees / RoboDojo gripper) -> PEAR action JSON."""

    def block(spec):
        return motion_to_arm_delta(spec.get("dxyz", (0, 0, 0)), spec.get("dpitch", 0.0),
                                   spec.get("dyaw", 0.0), spec.get("droll", 0.0),
                                   digits=digits)

    def grip(spec):
        return {"width": robodojo_gripper_to_pear(spec.get("gripper", 0))}

    if arm == "both":
        out: dict[str, Any] = {}
        for name in ("left", "right"):
            spec = arms.get(name) or {}
            out[f"{name}_arm"] = block(spec)
            out[f"{name}_gripper"] = grip(spec)
    else:
        spec = arms.get(arm) or {}
        out = {"arm": block(spec), "gripper": grip(spec)}
    if done:
        out["done"] = True
    return out


# ---------------------------------------------------------------------------
# observation and execution feedback
# ---------------------------------------------------------------------------


def arm_observation(pose: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """(arm block, gripper block) of one arm's pose summary for the state JSON.

    ``pose`` has ``xyz`` (grasp point, metres), ``pitch`` / ``yaw`` / ``roll``
    (degrees) and ``opening`` (normalised).  Positions are cm with 1 decimal, angles
    rad with 2 decimals, the gripper the measured jaw span in cm.
    """

    xyz = [float(v) for v in pose["xyz"]]
    pitch, yaw, roll = canonical_angles(pose.get("pitch", 0.0), pose.get("yaw", 0.0),
                                        pose.get("roll", 0.0))
    block = {
        "x": round(m_to_cm(xyz[0]), 1), "y": round(m_to_cm(xyz[1]), 1),
        "z": round(m_to_cm(xyz[2]), 1),
        "roll": round(deg_to_rad(roll), 2),
        "pitch": round(deg_to_rad(pitch), 2),
        "yaw": round(deg_to_rad(yaw), 2),
    }
    return block, {"width": width_cm(pose.get("opening", 0.0))}


def format_state(poses: Mapping[str, Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """The proprioceptive state JSON in the action's own shape."""

    if arm == "both":
        out: dict[str, Any] = {}
        for name in ("left", "right"):
            block, grip = arm_observation(poses[name])
            out[f"{name}_arm"], out[f"{name}_gripper"] = block, grip
        return out
    block, grip = arm_observation(poses[arm])
    return {"arm": block, "gripper": grip}


_OUT_OF_REACH = ("ik_path_rejected", "table_clearance", "joint_limits", "out_of_reach")


def feedback_status(result: Mapping[str, Any]) -> str:
    """``reached`` / ``stopped`` / ``out_of_reach`` of one executed delta."""

    reason = str(result.get("stop_reason") or "")
    if result.get("reached"):
        return "reached"
    if result.get("status") == "rejected" and (
            reason in _OUT_OF_REACH or result.get("ik_error")):
        return "out_of_reach"
    return "stopped"


def format_feedback(result: Mapping[str, Any], arm: str, frame: str = "world"
                    ) -> dict[str, Any]:
    """Execution feedback JSON for the next turn from a ``nudge`` result.

    ``achieved`` is the measured motion in the requested frame, in the action's own
    shape and units (cm / rad), so the policy can compare it with what it asked.
    """

    arms = result.get("arms") or {}
    achieved_specs = {}
    for name, info in arms.items():
        got = (info.get("achieved_delta") or {}).get(frame) or {}
        achieved_specs[name] = dict(got, gripper=0)
    achieved = format_action(achieved_specs, arm, digits=1)
    for key in [k for k in achieved if k.endswith("gripper")]:
        achieved.pop(key)
    out: dict[str, Any] = {
        "status": feedback_status(result),
        "reason": result.get("stop_reason"),
        "steps_used": int(result.get("steps_used", 0) or 0),
        "achieved": achieved,
    }
    if result.get("stalled"):
        out["stalled"] = True
    if result.get("clamped"):
        out["clamped"] = True
    if result.get("ik_error"):
        out["detail"] = str(result["ik_error"])[:200]
    holding = {name: bool(info.get("holding")) for name, info in arms.items()
               if "holding" in info}
    if holding:
        out["holding"] = holding if arm == "both" else holding.get(arm, False)
    return out


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You control the last few centimetres of a robot manipulation step.
Each turn you see labelled camera views and the current proprioceptive state, and you
answer with ONE small relative motion of the grasp point (PEAR Visual-Delta v1).

Action JSON (relative deltas; write 0 for every unchanged field):
- single arm: {"arm": {"x": dx, "y": dy, "z": dz, "roll": dr, "pitch": dp, "yaw": dw},
  "gripper": {"width": g}}
- both arms: {"left_arm": {...}, "right_arm": {...}, "left_gripper": {"width": g},
  "right_gripper": {"width": g}}
- optional "done": true when the last-mile sub-goal is complete (tabletop/in-air fine
  grasp, hook hanging, connector plugging, pouring tilt, button press, or slot/aperture
  insertion).
Units: x, y, z in centimetres; roll, pitch, yaw in radians. Gripper width in the
action is a switch: +1 opens, -1 closes, 0 keeps. In the state, gripper width is the
measured jaw span in centimetres.
Frames: "world" = table frame (x across the table to the robot's right, y away from
the robot, z up; angle deltas add to the gripper's pitch / yaw / roll). "tcp" = grasp
frame (+x pointing direction, +y finger-closing axis, +z normal).
Feedback after each move: "reached" (the move completed), "stopped" (contact or stall
before the target; the arm is in contact or blocked), "out_of_reach" (nothing moved).
Prefer small moves (0.2 - 2 cm, <= 0.15 rad) near contact.

Answer format: optionally <think>brief reasoning</think>, then exactly one fenced
json code block containing the action object and nothing else."""

ENV_PROMPT_TEMPLATE = """Goal: {goal}
Arm mode: {arm}   Frame: {frame}   Round: {round}/{max_rounds}   Steps left: {steps_left}
Limits per move: {max_linear_cm} cm, {max_angular_rad} rad
Images: {images}
State: {state}
Last execution: {feedback}
Reply with the next delta action."""


def build_user_prompt(*, goal: str, arm: str, frame: str, state: Mapping[str, Any],
                      feedback: Mapping[str, Any] | None, round_index: int,
                      max_rounds: int, steps_left: int, image_labels: Sequence[str] = (),
                      max_linear_m: float = 0.03, max_angular_deg: float = 15.0) -> str:
    """The per-turn environment prompt (text part; images are attached separately)."""

    return ENV_PROMPT_TEMPLATE.format(
        goal=goal.strip() or "(none)", arm=arm, frame=frame, round=round_index + 1,
        max_rounds=max_rounds, steps_left=steps_left,
        max_linear_cm=round(m_to_cm(max_linear_m), 1),
        max_angular_rad=round(deg_to_rad(max_angular_deg), 2),
        images=" ".join(f"[{label}]" for label in image_labels) or "(none)",
        state=json.dumps(state, separators=(", ", ": ")),
        feedback=json.dumps(feedback, separators=(", ", ": ")) if feedback else "none (first turn)",
    )


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


@runtime_checkable
class VisualDeltaPolicy(Protocol):
    """Anything that proposes the next delta for one turn.

    ``request`` carries ``protocol``, ``goal``, ``arm``, ``frame``, ``round``,
    ``max_rounds``, ``system`` / ``user`` prompt text, ``images`` (a list of
    ``{"label": "cam_head", "png": bytes}``), ``state`` (the state JSON),
    ``feedback`` (the last execution feedback or None) and ``history`` (earlier
    turns).  The answer is either the assistant text (think block + fenced json) or
    an action dict; both are validated by the loop.
    """

    def predict(self, request: dict[str, Any]) -> str | Mapping[str, Any]: ...


class CallablePolicy:
    """A plain function ``f(request) -> text | action dict`` as a policy."""

    def __init__(self, fn: Callable[[dict[str, Any]], Any], name: str | None = None) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "callable")

    def predict(self, request: dict[str, Any]) -> Any:
        return self.fn(request)


class ScriptedPolicy:
    """Replays fixed answers in order (deterministic tests, scripted demos).

    After the script is exhausted it answers ``{"done": true}`` with zero deltas.
    Every request is kept in ``requests``.
    """

    name = "scripted"

    def __init__(self, responses: Sequence[Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def predict(self, request: dict[str, Any]) -> Any:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index < len(self.responses):
            answer = self.responses[index]
            if isinstance(answer, BaseException):
                raise answer
            return answer
        return format_action({}, request.get("arm", "right"), done=True)


def as_policy(policy: Any) -> VisualDeltaPolicy:
    """A policy object, a callable, or a list of scripted answers -> a policy."""

    if policy is None:
        raise ValueError("no visual delta policy given")
    if hasattr(policy, "predict") and callable(policy.predict):
        return policy
    if callable(policy):
        return CallablePolicy(policy)
    if isinstance(policy, (list, tuple)):
        return ScriptedPolicy(policy)
    raise TypeError(f"not a visual delta policy: {type(policy).__name__}")


def policy_name(policy: Any) -> str:
    name = getattr(policy, "name", None) or getattr(policy, "model", None)
    return str(name or type(policy).__name__)


def has_configured_policy(
    primitives: Any = None,
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether a visual-delta VLM policy is configured on ``primitives`` or in ``env``."""

    if primitives is not None and getattr(primitives, "_visual_delta_policy", None) is not None:
        return True
    environ = os.environ if env is None else env
    for key in ("CODEASPOLICY_DELTA_BASE_URL", "CODEASPOLICY_DELTA_MODEL"):
        value = environ.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def encode_png(image: Any) -> bytes:
    """An RGB(A) uint8 array as PNG bytes (PNG bytes pass through)."""

    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    import io

    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(array[..., :3].astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


Transport = Callable[[str, dict[str, Any], float], tuple[int, Any]]


def urllib_transport(api_key: str | None) -> Transport:
    """POST JSON with the standard library; returns (status code, decoded body)."""

    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def post(url: str, body: dict[str, Any], timeout_s: float) -> tuple[int, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                text = response.read().decode("utf-8", errors="replace")
                status = response.status
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        try:
            return status, json.loads(text)
        except ValueError:
            return status, text

    return post


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "pear-visual-delta"


class OpenAIVisualDeltaClient:
    """A PEAR Visual-Delta model behind an OpenAI-compatible chat completions API.

    Configuration (arguments win over the environment): ``base_url`` or
    ``CODEASPOLICY_DELTA_BASE_URL`` (default ``http://127.0.0.1:8000/v1``, e.g. a
    vLLM server of a fine-tuned Qwen-VL), ``model`` or ``CODEASPOLICY_DELTA_MODEL``,
    the API key from ``api_key``, else the variable named by ``api_key_env`` /
    ``CODEASPOLICY_DELTA_API_KEY_ENV`` (default ``OPENAI_API_KEY``).  Each image
    is sent as a text label ``[cam_head]`` followed by a PNG data URL.
    ``transport(url, body, timeout_s) -> (status, payload)`` may be injected.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, *, api_key_env: str | None = None,
                 timeout_s: float = 60.0, temperature: float = 0.0, max_tokens: int = 512,
                 history_turns: int = 0, transport: Transport | None = None) -> None:
        self.base_url = (base_url or os.environ.get("CODEASPOLICY_DELTA_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("CODEASPOLICY_DELTA_MODEL") or DEFAULT_MODEL
        key_env = (api_key_env or os.environ.get("CODEASPOLICY_DELTA_API_KEY_ENV")
                   or "OPENAI_API_KEY")
        self.api_key = api_key if api_key is not None else os.environ.get(key_env)
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.history_turns = max(0, int(history_turns))
        self._transport = transport
        self.name = self.model

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def messages(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system",
                                      "content": request.get("system") or SYSTEM_PROMPT}]
        history = list(request.get("history") or [])
        for turn in history[-self.history_turns:] if self.history_turns else []:
            if turn.get("user") and turn.get("assistant"):
                out.append({"role": "user", "content": turn["user"]})
                out.append({"role": "assistant", "content": turn["assistant"]})
        content: list[dict[str, Any]] = []
        for image in request.get("images") or []:
            data = base64.b64encode(encode_png(image["png"])).decode("ascii")
            content.append({"type": "text", "text": f"[{image['label']}]"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{data}"}})
        content.append({"type": "text", "text": request.get("user") or ""})
        out.append({"role": "user", "content": content})
        return out

    def predict(self, request: dict[str, Any]) -> str:
        body = {"model": request.get("model") or self.model,
                "messages": self.messages(request),
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        post = self._transport or urllib_transport(self.api_key)
        status, payload = post(self.url, body, self.timeout_s)
        if status != 200 or not isinstance(payload, Mapping):
            raise RuntimeError(f"visual delta server answered {status}: {str(payload)[:300]}")
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("visual delta server answer has no message content") from exc
        if isinstance(content, list):  # content parts
            content = "".join(str(part.get("text", "")) for part in content
                              if isinstance(part, Mapping))
        return str(content)


# ---------------------------------------------------------------------------
# the closed loop
# ---------------------------------------------------------------------------

SUCCESS_REASONS = ("done", "converged", "contact_complete")


def clamp_spec(spec: Mapping[str, Any], max_linear_m: float, max_angular_deg: float
               ) -> tuple[dict[str, Any], bool]:
    """Scale a per-arm executor spec into the per-move limits; (spec, whether clamped)."""

    out = dict(spec)
    dxyz = [float(v) for v in spec.get("dxyz", (0.0, 0.0, 0.0))]
    norm = math.sqrt(sum(v * v for v in dxyz))
    clamped = False
    if norm > max_linear_m > 0:
        dxyz = [v * max_linear_m / norm for v in dxyz]
        clamped = True
    out["dxyz"] = dxyz
    for key in ("dpitch", "dyaw", "droll"):
        value = float(spec.get(key, 0.0))
        if abs(value) > max_angular_deg:
            value = math.copysign(max_angular_deg, value)
            clamped = True
        out[key] = value
    return out, clamped


def run_visual_delta_loop(
    *,
    goal: str,
    arm: str,
    frame: str,
    policy: Any,
    observe: Callable[..., dict[str, Any]],
    execute: Callable[[dict[str, dict[str, Any]], int], dict[str, Any]],
    step_allowance: int,
    max_rounds: int = 8,
    max_linear_m: float = 0.03,
    max_angular_deg: float = 15.0,
    stop_on_contact_complete: bool = False,
    max_invalid: int = 2,
    max_idle_rounds: int = 3,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the bounded observe -> ask -> nudge -> feedback loop.

    ``observe(images=True)`` returns ``{"poses": {arm: pose summary}, "images":
    [{"label", "png"}], "remaining_steps": int}``; ``execute(specs, max_steps)``
    runs one guarded nudge and returns its result dict.  Policy errors and invalid
    answers end or skip a round without motion; executor exceptions (cancellation,
    a finished episode) propagate.  At most ``step_allowance`` native steps are
    spent in total and at most ``max_rounds`` policy turns are taken.
    """

    policy = as_policy(policy)
    rounds: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    feedback: dict[str, Any] | None = None
    used = invalid = idle = 0
    stop_reason = "max_rounds"
    error: str | None = None
    policy_done = converged = contact_complete = False
    for index in range(max_rounds):
        left = step_allowance - used
        if left <= 0:
            stop_reason = "step_budget"
            break
        seen = observe(images=True)
        if seen.get("remaining_steps") is not None and int(seen["remaining_steps"]) <= 0:
            stop_reason = "episode_budget"
            break
        state = format_state(seen["poses"], arm)
        images = list(seen.get("images") or [])
        user = build_user_prompt(goal=goal, arm=arm, frame=frame, state=state,
                                 feedback=feedback, round_index=index, max_rounds=max_rounds,
                                 steps_left=left, image_labels=[i["label"] for i in images],
                                 max_linear_m=max_linear_m, max_angular_deg=max_angular_deg)
        request = {"protocol": PROTOCOL, "goal": goal, "arm": arm, "frame": frame,
                   "round": index, "max_rounds": max_rounds, "system": SYSTEM_PROMPT,
                   "user": user, "images": images, "state": state, "feedback": feedback,
                   "history": list(history), "steps_left": left, "model": model}
        record: dict[str, Any] = {"round": index, "state": state, "steps_used": 0}
        rounds.append(record)
        try:
            answer = policy.predict(request)
        except Exception as exc:  # noqa: BLE001 - a policy failure ends the loop cleanly
            error = f"{type(exc).__name__}: {exc}"
            record["error"] = error
            stop_reason = "policy_error"
            break
        try:
            if isinstance(answer, Mapping):
                parsed = {"think": None, "action": validate_delta_action(answer, arm)}
                text = json.dumps(dict(answer))
            else:
                parsed = parse_visual_delta_response(answer, arm)
                text = str(answer)
        except VisualDeltaError as exc:
            invalid += 1
            record["error"] = f"invalid response: {exc}"
            feedback = {"status": "invalid_response", "error": str(exc)[:300], "steps_used": 0}
            record["feedback"] = feedback
            history.append({"user": user, "assistant": str(answer)[:2000]})
            if invalid >= max_invalid:
                stop_reason = "invalid_policy_response"
                break
            continue
        invalid = 0
        history.append({"user": user, "assistant": text[:2000]})
        action = parsed["action"]
        record["action"] = action
        if parsed.get("think"):
            record["think"] = parsed["think"][:500]
        policy_done = bool(action.get("done"))
        if is_zero_action(action):
            if policy_done:
                stop_reason = "done"
            else:
                converged = True
                stop_reason = "converged"
            break
        specs, clamped = {}, False
        for name, spec in action_arms(action, arm).items():
            specs[name], was = clamp_spec(spec, max_linear_m, max_angular_deg)
            clamped = clamped or was
        result = dict(execute(specs, left))
        if clamped:
            result["clamped"] = True
        steps = int(result.get("steps_used", 0) or 0)
        used += steps
        feedback = format_feedback(result, arm, frame)
        record.update(executed=specs, clamped=clamped, steps_used=steps, feedback=feedback,
                      status=result.get("status"), stop_reason=result.get("stop_reason"))
        idle = idle + 1 if steps == 0 else 0
        reason = str(result.get("stop_reason") or "")
        if reason == "reserve_steps_exhausted":
            stop_reason = "step_budget"
            break
        if policy_done:
            stop_reason = "done"
            break
        if result.get("stalled") and stop_on_contact_complete:
            contact_complete = True
            stop_reason = "contact_complete"
            break
        if reason.startswith("episode"):
            stop_reason = "episode_budget"
            break
        if idle >= max_idle_rounds:
            stop_reason = "no_progress"
            break
    final = observe(images=False)
    success = stop_reason in SUCCESS_REASONS
    out: dict[str, Any] = {
        "protocol": PROTOCOL, "goal": goal, "arm": arm, "frame": frame,
        "status": "completed" if success else ("partial" if used else "rejected"),
        "reached": success, "done": policy_done and stop_reason == "done",
        "converged": converged, "contact_complete": contact_complete,
        "stop_reason": stop_reason, "rounds_used": len(rounds), "max_rounds": max_rounds,
        "steps_used": used, "step_allowance": step_allowance, "policy": policy_name(policy),
        "rounds": rounds, "last_feedback": feedback,
        "final_state": format_state(final["poses"], arm),
    }
    if model:
        out["model"] = model
    if error:
        out["policy_error"] = error
    return out


__all__ = [
    "ANGLE_FIELDS",
    "ARM_FIELDS",
    "BIMANUAL_KEYS",
    "ENV_PROMPT_TEMPLATE",
    "FEEDBACK_STATUSES",
    "FRAMES",
    "GRIP_SPAN_M",
    "GRIP_ZERO_M",
    "NEAR_VERTICAL_DEG",
    "POSITION_FIELDS",
    "PROTOCOL",
    "SINGLE_KEYS",
    "SYSTEM_PROMPT",
    "CallablePolicy",
    "OpenAIVisualDeltaClient",
    "ScriptedPolicy",
    "VisualDeltaError",
    "VisualDeltaPolicy",
    "action_arms",
    "angle_delta",
    "arm_delta_to_motion",
    "arm_observation",
    "as_policy",
    "build_user_prompt",
    "canonical_angles",
    "clamp_spec",
    "cm_to_m",
    "deg_to_rad",
    "ee_command_to_pear",
    "encode_png",
    "feedback_status",
    "format_action",
    "format_feedback",
    "format_state",
    "has_configured_policy",
    "is_zero_action",
    "is_zero_delta",
    "m_to_cm",
    "motion_to_arm_delta",
    "parse_visual_delta_response",
    "pear_gripper_to_robodojo",
    "pose_delta",
    "rad_to_deg",
    "robodojo_gripper_to_pear",
    "run_visual_delta_loop",
    "split_response",
    "validate_delta_action",
    "width_cm",
]
