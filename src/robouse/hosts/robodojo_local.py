"""Observed-frame local alignment: explicit targets, no hidden contact search."""

from __future__ import annotations

import copy
import json

import numpy as np

from . import robodojo_kinematics as kin
from .robodojo_coordination import _matrix, _positive


def vector(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(x) not in (int, float) or not np.isfinite(x) for x in value)):
        raise ValueError(f"{name} must be three finite numbers")
    return np.array(value, float)


def rotation(vector_deg):
    v = np.deg2rad(vector_deg)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def violation(local, pose):
    reference = _matrix(local["reference"])
    basis = reference[:3, :3] if local["frame"] == "tcp" else np.eye(3)
    delta = basis.T @ (pose[:3, 3] - reference[:3, 3])
    if np.any(np.abs(delta) > np.asarray(local["translation_limits"]) + 1e-6):
        return f"{local['arm']}: local_translation_constraint"
    if kin.rotation_error(reference[:3, :3], pose[:3, :3]) > local["rotation_limit_deg"] + 1e-5:
        return f"{local['arm']}: local_rotation_constraint"
    return None


def prepare_local(controller, intent_id, *, arm, offset, reference=None, frame="world",
                  translation_limits=(.01, .01, .01), rotation_deg=(0., 0., 0.),
                  rotation_limit_deg=5., holding="keep", completion=None,
                  max_steps=None, preview=False):
    if arm not in kin.ARMS or frame not in ("world", "tcp"):
        raise ValueError("alignment needs left/right arm and world/tcp frame")
    offset = vector(offset, "offset")
    bounds = vector(translation_limits, "translation_limits")
    turn = vector(rotation_deg, "rotation_deg")
    if np.any(bounds < 0) or np.any(bounds > .05) or not np.any(bounds):
        raise ValueError("translation limits must be in [0, .05] m and not all zero")
    _positive(rotation_limit_deg, "rotation_limit_deg", 15.)
    if np.any(np.abs(offset) > bounds) or np.linalg.norm(turn) > rotation_limit_deg:
        raise ValueError("requested alignment exceeds the declared local envelope")
    request = dict(arm=arm, offset=offset.tolist(), reference=reference, frame=frame,
                   translation_limits=bounds.tolist(), rotation_deg=turn.tolist(),
                   rotation_limit_deg=rotation_limit_deg, holding=holding,
                   completion=completion, max_steps=max_steps)
    request = json.loads(json.dumps(request, allow_nan=False))
    previous = controller.records.get(intent_id)
    if previous is not None and not preview:
        if previous["contract"].get("local", {}).get("request") != request:
            return dict(status="rejected", intent_id=intent_id, reached=False, steps_used=0,
                        needs_model=True, stop_reason="intent_identity_conflict")
        return controller._result(previous)
    snap = controller.motion._snapshot()
    anchor = (snap["poses"][arm] if reference is None else
              controller._goal(reference, snap["poses"][arm]))
    basis = anchor[:3, :3] if frame == "tcp" else np.eye(3)
    goal = anchor.copy()
    goal[:3, 3] += basis @ offset
    goal[:3, :3] = basis @ rotation(turn) @ basis.T @ anchor[:3, :3]
    local = dict(arm=arm, reference=controller._pose(anchor), frame=frame,
                 translation_limits=bounds.tolist(), rotation_limit_deg=rotation_limit_deg,
                 request=request)
    return controller.prepare(
        intent_id, targets={arm: controller._pose(goal)}, interaction="contact",
        directions={}, holding={arm: holding}, completion=copy.deepcopy(completion or
            {"position_m": .0015, "orientation_deg": 1.}), max_steps=max_steps,
        reserve_steps=0, keep_relative=False, local=local, preview=preview,
    )
