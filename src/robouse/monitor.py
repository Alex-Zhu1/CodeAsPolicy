# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Code-as-Monitor (CaM): programmatic geometric and perceptual constraint checking.

Strict epistemic honesty: all checks operate exclusively on public observations
(proprioceptive State/ArmState, ObjectPose, WorldPoint, Segment, and
observe_object / observe_hold evidence dicts), preserving provenance and
observation_id without simulator ground-truth leakage.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contract_evidence import check_fact, check_hold


@dataclass(frozen=True)
class ConstraintElement:
    """A compact geometric abstraction derived from robot perception or proprioception."""

    name: str
    kind: str  # "point", "axis", "box", "plane"
    xyz: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] | None = None
    extent_m: tuple[float, float, float] | None = None
    yaw_deg: float | None = None
    observation_id: str | None = None
    step: int | None = None
    uncertainty_m: float | None = None
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        if self.kind not in ("point", "axis", "box", "plane"):
            raise ValueError(f"kind must be point, axis, box, or plane; got {self.kind!r}")
        if self.xyz is not None:
            xyz_vals = tuple(float(v) for v in self.xyz)
            if len(xyz_vals) != 3 or not all(math.isfinite(v) for v in xyz_vals):
                raise ValueError("xyz must contain 3 finite numbers")
            object.__setattr__(self, "xyz", xyz_vals)
        if self.axis is not None:
            axis_vals = tuple(float(v) for v in self.axis)
            if len(axis_vals) != 3 or not all(math.isfinite(v) for v in axis_vals):
                raise ValueError("axis must contain 3 finite numbers")
            norm = math.sqrt(sum(v * v for v in axis_vals))
            if norm < 1e-9:
                raise ValueError("axis must be a non-zero vector")
            object.__setattr__(self, "axis", tuple(v / norm for v in axis_vals))
        if self.extent_m is not None:
            extent_vals = tuple(float(v) for v in self.extent_m)
            if len(extent_vals) != 3 or not all(math.isfinite(v) and v >= 0 for v in extent_vals):
                raise ValueError("extent_m must contain 3 non-negative numbers")
            object.__setattr__(self, "extent_m", extent_vals)
        if self.yaw_deg is not None:
            object.__setattr__(self, "yaw_deg", float(self.yaw_deg))
        if self.step is not None:
            object.__setattr__(self, "step", int(self.step))
        if self.uncertainty_m is not None:
            object.__setattr__(self, "uncertainty_m", float(self.uncertainty_m))

    def is_current(self, robo: Any) -> bool:
        """Check if this element's perception/state origin is still current with the robot's state."""
        if self.provenance:
            if "perception_id" in self.provenance:
                return bool(
                    hasattr(robo, "perception_is_current")
                    and robo.perception_is_current({"provenance": self.provenance})
                )
            if (self.provenance.get("binding") == "unsupported"
                    or self.provenance.get("source_freshness") == "unverified"
                    or self.provenance.get("current_at_return") is False):
                return False
        if self.step is not None and hasattr(robo, "_backend"):
            return bool(robo._backend.env_steps() == self.step)
        return True

    def require_current(self, robo: Any) -> ConstraintElement:
        """Assert that this element is current with the robot's live scene; raise ValueError if stale."""
        if not self.is_current(robo):
            current_step = getattr(robo._backend, "env_steps", lambda: None)() if hasattr(robo, "_backend") else None
            raise ValueError(
                f"ConstraintElement '{self.name}' is stale (recorded at step {self.step}, "
                f"current step {current_step}). "
                f"Re-observe the scene before evaluating constraints."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def from_pose(name: str, obj_pose_or_point: Any) -> ConstraintElement:
    """Convert an ObjectPose, WorldPoint, landmark dict, or coordinates into a ConstraintElement."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be non-empty text")

    # If already a ConstraintElement
    if isinstance(obj_pose_or_point, ConstraintElement):
        return obj_pose_or_point

    xyz: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] | None = None
    extent_m: tuple[float, float, float] | None = None
    yaw_deg: float | None = None
    observation_id: str | None = None
    step: int | None = None
    uncertainty_m: float | None = None
    prov: dict[str, Any] | None = None
    kind = "point"

    # 1. ObjectPose (from robo.locate)
    if hasattr(obj_pose_or_point, "center_xyz"):
        center = obj_pose_or_point.center_xyz
        if center is not None:
            xyz = tuple(float(v) for v in center)
        length_m = getattr(obj_pose_or_point, "length_m", None)
        width_m = getattr(obj_pose_or_point, "width_m", None)
        height_m = getattr(obj_pose_or_point, "height_m", None)
        if length_m is not None and width_m is not None and height_m is not None:
            extent_m = (float(length_m), float(width_m), float(height_m))
            kind = "box"
            axis = (0.0, 0.0, 1.0)  # upright object axis
        yaw_deg = getattr(obj_pose_or_point, "long_axis_deg", None)
        if yaw_deg is None:
            yaw_deg = getattr(obj_pose_or_point, "grasp_yaw", None)
        prov = getattr(obj_pose_or_point, "provenance", None)
        if isinstance(prov, dict):
            observation_id = prov.get("observation_id")
            step = prov.get("step")

    # 2. WorldPoint (from robo.back_project)
    elif hasattr(obj_pose_or_point, "pixel") and hasattr(obj_pose_or_point, "xyz"):
        xyz = tuple(float(v) for v in obj_pose_or_point.xyz)
        kind = "point"
        prov = getattr(obj_pose_or_point, "provenance", None)
        if isinstance(prov, dict):
            observation_id = prov.get("observation_id")
            step = prov.get("step")

    # 3. Dict / landmark evidence
    elif isinstance(obj_pose_or_point, dict):
        raw_xyz = obj_pose_or_point.get("xyz") or obj_pose_or_point.get("center_xyz")
        if raw_xyz is not None:
            xyz = tuple(float(v) for v in raw_xyz)
        raw_extent = obj_pose_or_point.get("extent_m")
        if raw_extent is not None:
            extent_m = tuple(float(v) for v in raw_extent)
        elif all(k in obj_pose_or_point for k in ("length_m", "width_m", "height_m")):
            extent_m = (
                float(obj_pose_or_point["length_m"]),
                float(obj_pose_or_point["width_m"]),
                float(obj_pose_or_point["height_m"]),
            )
        yaw_deg = (
            obj_pose_or_point.get("yaw_deg")
            or obj_pose_or_point.get("long_axis_deg")
            or obj_pose_or_point.get("grasp_yaw")
            or obj_pose_or_point.get("yaw")
        )
        uncertainty_m = obj_pose_or_point.get("uncertainty_m")
        raw_axis = obj_pose_or_point.get("axis")
        if raw_axis is not None:
            axis = tuple(float(v) for v in raw_axis)
        elif extent_m is not None:
            axis = (0.0, 0.0, 1.0)
        prov = obj_pose_or_point.get("provenance")
        if isinstance(prov, dict):
            observation_id = prov.get("observation_id")
            step = prov.get("step")
        if observation_id is None:
            observation_id = obj_pose_or_point.get("observation_id")
        if step is None:
            step = obj_pose_or_point.get("step")
        kind = obj_pose_or_point.get("kind", "box" if extent_m is not None else "point")

    # 4. Raw list/tuple coordinates [x, y, z]
    elif isinstance(obj_pose_or_point, (list, tuple)) and len(obj_pose_or_point) == 3:
        xyz = tuple(float(v) for v in obj_pose_or_point)
        kind = "point"

    else:
        raise TypeError(f"Cannot create ConstraintElement from {type(obj_pose_or_point).__name__}")

    return ConstraintElement(
        name=name,
        kind=kind,
        xyz=xyz,
        axis=axis,
        extent_m=extent_m,
        yaw_deg=yaw_deg,
        observation_id=observation_id,
        step=step,
        uncertainty_m=uncertainty_m,
        provenance=prov if isinstance(prov, dict) else None,
    )


def from_arm(state: Any, arm: str = "right") -> ConstraintElement:
    """Extract a ConstraintElement from State / ArmState or dict with tcp_pos and orientation.

    Matches RoboDojo's pointing convention:
    pitch=90 points straight down [0, 0, -1]
    pitch=0, yaw=0 points [0, 1, 0]
    pitch=0, yaw=90 points [-1, 0, 0]
    """
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")

    arm_obj = state
    step = getattr(state, "step", None)
    obs_id = getattr(state, "observation_id", None)
    if hasattr(state, "arm") and callable(state.arm):
        arm_obj = state.arm(arm)
    elif hasattr(state, arm):
        arm_obj = getattr(state, arm)
    elif isinstance(state, dict):
        if arm in state:
            arm_obj = state[arm]
        step = step if step is not None else state.get("step")
        obs_id = obs_id if obs_id is not None else state.get("observation_id")

    prov = getattr(state, "provenance", None)
    if isinstance(prov, dict):
        obs_id = obs_id or prov.get("observation_id")
        step = step if step is not None else prov.get("step")

    # Extract TCP position
    tcp = getattr(arm_obj, "tcp_pos", None)
    if tcp is None and isinstance(arm_obj, dict):
        tcp = arm_obj.get("tcp_pos") or arm_obj.get("xyz")
    elif tcp is None and hasattr(arm_obj, "xyz"):
        tcp = getattr(arm_obj, "xyz")

    xyz = tuple(float(v) for v in tcp) if tcp is not None else None

    # Compute unit pointing axis
    pitch = getattr(arm_obj, "pitch", None)
    yaw = getattr(arm_obj, "yaw", None)
    if pitch is None and isinstance(arm_obj, dict):
        pitch = arm_obj.get("pitch")
        yaw = arm_obj.get("yaw")

    axis: tuple[float, float, float] | None = None
    yaw_deg: float | None = None

    if pitch is not None and yaw is not None:
        p = math.radians(float(pitch))
        y = math.radians(float(yaw))
        axis = (-math.cos(p) * math.sin(y), math.cos(p) * math.cos(y), -math.sin(p))
        yaw_deg = float(yaw)
    else:
        quat = getattr(arm_obj, "eef_quat", None)
        if quat is None and isinstance(arm_obj, dict):
            quat = arm_obj.get("eef_quat")
        if quat is not None:
            try:
                from .hosts.robodojo_kinematics import angles, rot_from_quat

                rotation = rot_from_quat(quat)
                _, y_deg, _ = angles(rotation)
                axis = tuple(float(v) for v in rotation[:, 0])
                yaw_deg = float(y_deg)
            except Exception:
                axis = None
                yaw_deg = None

    arm_prov = (
        prov
        if isinstance(prov, dict)
        else ({"step": step, "observation_id": obs_id} if step is not None or obs_id is not None else None)
    )
    return ConstraintElement(
        name=arm,
        kind="axis" if axis is not None else "point",
        xyz=xyz,
        axis=axis,
        yaw_deg=yaw_deg,
        observation_id=obs_id,
        step=step,
        provenance=arm_prov,
    )


@dataclass(frozen=True)
class ConstraintCheck:
    """The result of evaluating a spatio-temporal geometric or evidence constraint."""

    name: str
    kind: str  # "reactive" or "proactive"
    passed: bool
    metric: str
    value: float | None
    bound: float | tuple[float, ...] | None
    delta: float | None  # signed margin or excess over threshold
    elements: tuple[str, ...]
    reason: str | None = None
    remedy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": self.passed,
            "metric": self.metric,
            "value": self.value,
            "bound": self.bound,
            "delta": self.delta,
            "elements": list(self.elements),
            "reason": self.reason,
            "remedy": self.remedy,
        }


def check_distance(
    elem_a: ConstraintElement | Sequence[float],
    elem_b: ConstraintElement | Sequence[float],
    *,
    min_m: float | None = None,
    max_m: float | None = None,
    xy_only: bool = False,
    name: str = "distance",
    kind: str = "proactive",
) -> ConstraintCheck:
    """Check Euclidean 3D (or XY horizontal) distance between two elements/points."""
    pos_a = elem_a.xyz if isinstance(elem_a, ConstraintElement) else tuple(float(v) for v in elem_a)
    pos_b = elem_b.xyz if isinstance(elem_b, ConstraintElement) else tuple(float(v) for v in elem_b)
    name_a = getattr(elem_a, "name", "a")
    name_b = getattr(elem_b, "name", "b")
    elements = (name_a, name_b)

    if pos_a is None or pos_b is None:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="distance_xy_m" if xy_only else "distance_m",
            value=None,
            bound=(min_m, max_m) if min_m is not None and max_m is not None else (min_m or max_m),
            delta=None,
            elements=elements,
            reason="missing_position: one or both elements have no 3D coordinates",
            remedy="Verify element coordinates from perception or proprioception.",
        )

    if xy_only:
        dist = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])
        metric = "distance_xy_m"
    else:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_a, pos_b)))
        metric = "distance_m"

    if min_m is not None and dist < min_m:
        delta = dist - min_m  # negative margin
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric=metric,
            value=dist,
            bound=min_m,
            delta=delta,
            elements=elements,
            reason=f"clearance_violation: distance {dist:.4f}m is below minimum {min_m:.4f}m",
            remedy=f"Retract or adjust trajectory to maintain at least {min_m:.4f}m clearance.",
        )
    if max_m is not None and dist > max_m:
        delta = dist - max_m  # positive excess
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric=metric,
            value=dist,
            bound=max_m,
            delta=delta,
            elements=elements,
            reason=f"reach_violation: distance {dist:.4f}m exceeds maximum bound {max_m:.4f}m",
            remedy=f"Move closer to target within {max_m:.4f}m.",
        )

    bound = (min_m, max_m) if min_m is not None and max_m is not None else (min_m or max_m)
    margin = (
        min(
            dist - min_m if min_m is not None else float("inf"),
            max_m - dist if max_m is not None else float("inf"),
        )
        if (min_m is not None or max_m is not None)
        else 0.0
    )
    return ConstraintCheck(
        name=name,
        kind=kind,
        passed=True,
        metric=metric,
        value=dist,
        bound=bound,
        delta=margin,
        elements=elements,
        reason=None,
        remedy=None,
    )


def check_tilt(
    elem: ConstraintElement | Sequence[float],
    *,
    reference_axis: Sequence[float] = (0.0, 0.0, -1.0),
    max_deg: float = 15.0,
    name: str = "tilt",
    kind: str = "proactive",
) -> ConstraintCheck:
    """Check angular deviation (in degrees) between elem.axis and reference_axis."""
    axis = elem.axis if isinstance(elem, ConstraintElement) else tuple(float(v) for v in elem)
    elem_name = getattr(elem, "name", "element")
    elements = (elem_name,)

    if axis is None:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="tilt_deg",
            value=None,
            bound=max_deg,
            delta=None,
            elements=elements,
            reason="missing_axis: element has no orientation axis",
            remedy="Provide element with pointing or normal axis.",
        )

    norm_a = math.sqrt(sum(v * v for v in axis))
    norm_ref = math.sqrt(sum(v * v for v in reference_axis))
    if norm_a < 1e-9 or norm_ref < 1e-9:
        raise ValueError("axis vectors must be non-zero")

    dot = sum((a / norm_a) * (b / norm_ref) for a, b in zip(axis, reference_axis))
    dot = max(-1.0, min(1.0, dot))
    ang_deg = math.degrees(math.acos(dot))
    delta = ang_deg - max_deg

    if ang_deg > max_deg:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="tilt_deg",
            value=ang_deg,
            bound=max_deg,
            delta=delta,
            elements=elements,
            reason=f"tilt_constraint: tilt {ang_deg:.1f}deg exceeds maximum allowable {max_deg:.1f}deg",
            remedy=f"Align axis with reference direction within {max_deg:.1f}deg.",
        )

    return ConstraintCheck(
        name=name,
        kind=kind,
        passed=True,
        metric="tilt_deg",
        value=ang_deg,
        bound=max_deg,
        delta=-delta,  # positive margin to limit
        elements=elements,
        reason=None,
        remedy=None,
    )


def check_in_bounds(
    elem: ConstraintElement | Sequence[float],
    *,
    min_xyz: Sequence[float | None] | None = None,
    max_xyz: Sequence[float | None] | None = None,
    margin_m: float = 0.0,
    name: str = "bounds",
    kind: str = "proactive",
) -> ConstraintCheck:
    """Check if elem.xyz is within an axis-aligned bounding volume."""
    pos = elem.xyz if isinstance(elem, ConstraintElement) else tuple(float(v) for v in elem)
    elem_name = getattr(elem, "name", "element")
    elements = (elem_name,)

    if pos is None:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="bounds_violation_m",
            value=None,
            bound=None,
            delta=None,
            elements=elements,
            reason="missing_position: element has no 3D coordinates",
            remedy="Verify element coordinates before evaluating bounds.",
        )

    excess = 0.0
    violation_dim = None

    if min_xyz is not None:
        for i, bound in enumerate(min_xyz):
            if bound is not None and pos[i] < bound - margin_m:
                diff = (bound - margin_m) - pos[i]
                if diff > excess:
                    excess = diff
                    violation_dim = ("x", "y", "z")[i]

    if max_xyz is not None:
        for i, bound in enumerate(max_xyz):
            if bound is not None and pos[i] > bound + margin_m:
                diff = pos[i] - (bound + margin_m)
                if diff > excess:
                    excess = diff
                    violation_dim = ("x", "y", "z")[i]

    if excess > 0.0:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="bounds_violation_m",
            value=excess,
            bound=0.0,
            delta=excess,
            elements=elements,
            reason=f"bounds_violation: coordinate {violation_dim} exceeds boundary by {excess:.4f}m",
            remedy=f"Adjust {violation_dim} position to stay within allowed workspace bounds.",
        )

    return ConstraintCheck(
        name=name,
        kind=kind,
        passed=True,
        metric="bounds_violation_m",
        value=0.0,
        bound=0.0,
        delta=0.0,
        elements=elements,
        reason=None,
        remedy=None,
    )


def check_above(
    elem: ConstraintElement | Sequence[float],
    target: ConstraintElement | Any,
    *,
    min_dz_m: float = 0.02,
    max_xy_m: float = 0.05,
    name: str = "above",
    kind: str = "reactive",
) -> ConstraintCheck:
    """Check if elem is horizontally aligned within max_xy_m and at least min_dz_m above target."""
    pos_elem = elem.xyz if isinstance(elem, ConstraintElement) else tuple(float(v) for v in elem)
    elem_name = getattr(elem, "name", "element")

    target_name = getattr(target, "name", "target")
    elements = (elem_name, target_name)

    if pos_elem is None:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="alignment",
            value=None,
            bound=None,
            delta=None,
            elements=elements,
            reason="missing_position: element has no 3D position",
            remedy="Establish element 3D position before checking alignment.",
        )

    pos_target = None
    top_z = None

    if isinstance(target, ConstraintElement):
        pos_target = target.xyz
        if target.xyz is not None:
            top_z = target.xyz[2] + (target.extent_m[2] / 2.0 if target.extent_m else 0.0)
    elif hasattr(target, "center_xyz") and target.center_xyz is not None:
        pos_target = tuple(float(v) for v in target.center_xyz)
        top_z = getattr(target, "top_z", None)
        if top_z is None:
            top_z = pos_target[2]
    elif isinstance(target, (list, tuple)) and len(target) == 3:
        pos_target = tuple(float(v) for v in target)
        top_z = pos_target[2]
    elif isinstance(target, dict):
        raw = target.get("xyz") or target.get("center_xyz")
        if raw is not None:
            pos_target = tuple(float(v) for v in raw)
            top_z = target.get("top_z", pos_target[2])

    if pos_target is None or top_z is None:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="alignment",
            value=None,
            bound=None,
            delta=None,
            elements=elements,
            reason="missing_position: target has no 3D position",
            remedy="Locate target position before checking alignment.",
        )

    xy_dist = math.hypot(pos_elem[0] - pos_target[0], pos_elem[1] - pos_target[1])
    dz = pos_elem[2] - top_z

    if xy_dist > max_xy_m:
        delta = xy_dist - max_xy_m
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="xy_offset_m",
            value=xy_dist,
            bound=max_xy_m,
            delta=delta,
            elements=elements,
            reason=f"horizontal_alignment_violation: offset {xy_dist:.4f}m exceeds tolerance {max_xy_m:.4f}m",
            remedy=f"Align horizontally over target (max offset {max_xy_m:.4f}m) before descending.",
        )

    if dz < min_dz_m:
        delta = dz - min_dz_m
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="clearance_z_m",
            value=dz,
            bound=min_dz_m,
            delta=delta,
            elements=elements,
            reason=f"vertical_clearance_violation: height {dz:.4f}m above target is less than clearance {min_dz_m:.4f}m",
            remedy=f"Raise gripper to at least {min_dz_m:.4f}m above target top ({top_z:.4f}m).",
        )

    return ConstraintCheck(
        name=name,
        kind=kind,
        passed=True,
        metric="clearance_z_m",
        value=dz,
        bound=min_dz_m,
        delta=dz - min_dz_m,
        elements=elements,
        reason=None,
        remedy=None,
    )


def check_hold_evidence(
    evidence_dict: dict[str, Any] | None,
    *,
    max_drift_m: float = 0.015,
    name: str = "hold_retention",
    kind: str = "reactive",
) -> ConstraintCheck:
    """Evaluate an observe_hold evidence dict for retention and relative motion."""
    elements = ("gripper",)

    if not isinstance(evidence_dict, dict):
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="hold_evidence",
            value=None,
            bound=None,
            delta=None,
            elements=elements,
            reason="missing_evidence: observe_hold evidence dictionary is None or invalid",
            remedy="Call robo.observe_hold(ref, arm=arm) to obtain fresh hold evidence.",
        )

    arm = evidence_dict.get("arm", "arm")
    elements = (f"{arm}_gripper",)
    state = evidence_dict.get("state")
    gripper_info = evidence_dict.get("gripper", {})
    obstructed = (
        gripper_info.get("obstructed_closure", False)
        if isinstance(gripper_info, dict)
        else False
    )
    features = evidence_dict.get("features", {})

    drifts = [
        f["relative_change_m"]
        for f in features.values()
        if isinstance(f, dict) and "relative_change_m" in f
    ]
    max_drift = max(drifts) if drifts else 0.0

    if not obstructed:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="gripper_obstructed_closure",
            value=0.0,
            bound=1.0,
            delta=-1.0,
            elements=elements,
            reason="gripper_not_obstructed: gripper closed fully without grasping object or opened",
            remedy="Re-attempt grasp alignment and ensure gripper fingers contact the object before lifting.",
        )

    if (state in ("image_relative_motion", "image_consistent")
            or evidence_dict.get("evidence_space") == "wrist_image_2d"):
        reason = (
            "image_relative_motion: observed features shifted relative to the wrist camera"
            if state == "image_relative_motion"
            else "image_evidence_only: image consistency does not establish a 3D hold"
        )
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="feature_drift_m",
            value=None,
            bound=max_drift_m,
            delta=None,
            elements=elements,
            reason=reason + "; metre drift cannot be measured from image evidence alone",
            remedy="Obtain fresh 3D feature evidence and review observed motion before continuing.",
        )

    if state == "relative_motion" or max_drift > max_drift_m:
        delta = max_drift - max_drift_m
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="feature_drift_m",
            value=max_drift,
            bound=max_drift_m,
            delta=delta,
            elements=elements,
            reason=f"relative_motion: object slipped or moved relative to gripper (drift {max_drift:.4f}m > {max_drift_m:.4f}m)",
            remedy="Object slipped in grasp. Regrip the object firmly or re-acquire baseline before continuing motion.",
        )

    if state == "unknown":
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="hold_confidence",
            value=0.0,
            bound=1.0,
            delta=-1.0,
            elements=elements,
            reason=f"unknown: hold evidence cannot be established ({evidence_dict.get('reason')})",
            remedy="Observe object from unoccluded camera angle or re-establish visual feature baseline.",
        )

    if state == "conflicting_evidence":
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="hold_confidence",
            value=0.0,
            bound=1.0,
            delta=-1.0,
            elements=elements,
            reason=f"conflicting_evidence: {evidence_dict.get('reason')}",
            remedy="Resolve conflicting evidence with fresh observation before moving.",
        )

    if state != "following" or not drifts:
        return ConstraintCheck(
            name=name,
            kind=kind,
            passed=False,
            metric="feature_drift_m",
            value=None,
            bound=max_drift_m,
            delta=None,
            elements=elements,
            reason="missing_hold_evidence: following state and measured 3D feature drift are required",
            remedy="Call robo.observe_hold(ref, arm=arm) to obtain usable 3D hold evidence.",
        )

    # Observed 3D features followed the gripper within the requested drift bound.
    return ConstraintCheck(
        name=name,
        kind=kind,
        passed=True,
        metric="feature_drift_m",
        value=max_drift,
        bound=max_drift_m,
        delta=max_drift_m - max_drift,  # positive margin to threshold
        elements=elements,
        reason=None,
        remedy=None,
    )


def evaluate_constraints(
    checks: Sequence[ConstraintCheck | Callable[[], ConstraintCheck]],
) -> dict[str, Any]:
    """Evaluate a sequence of ConstraintChecks or check callables and produce a summary report."""
    evaluated: list[dict[str, Any]] = []
    violated: list[dict[str, Any]] = []

    for item in checks:
        check = item() if callable(item) else item
        if not isinstance(check, ConstraintCheck):
            raise TypeError(f"Expected ConstraintCheck, got {type(check).__name__}")
        d = check.to_dict()
        evaluated.append(d)
        if not check.passed:
            violated.append(d)

    passed = len(violated) == 0
    total = len(evaluated)
    if passed:
        summary = f"All {total} constraint(s) passed"
    else:
        fails = "; ".join(f"{v['name']}: {v['reason'] or v['metric']}" for v in violated)
        summary = f"{total - len(violated)}/{total} constraints passed; {len(violated)} violated: {fails}"

    return {
        "passed": passed,
        "violated": violated,
        "checks": evaluated,
        "summary": summary,
    }


class MonitorSuite:
    """A collection of constraint predicates to evaluate together."""

    def __init__(
        self, checks: Sequence[ConstraintCheck | Callable[[], ConstraintCheck]] = ()
    ) -> None:
        self.checks: list[ConstraintCheck | Callable[[], ConstraintCheck]] = list(checks)

    def add(self, check: ConstraintCheck | Callable[[], ConstraintCheck]) -> MonitorSuite:
        self.checks.append(check)
        return self

    def evaluate(self) -> dict[str, Any]:
        return evaluate_constraints(self.checks)


def proactive_constraints(
    *,
    max_tilt_deg: float | dict[str, float] | None = None,
    z_bounds: Sequence[float] | dict[str, Sequence[float]] | None = None,
    keep_out: list[dict[str, Any]] | None = None,
    corridor_m: float | dict[str, float] | None = None,
    reference_axis: Sequence[float] | None = None,
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated, serializable proactive constraint dict for robo.execute_motion."""
    constraints: dict[str, Any] = {}

    if reference_axis is not None:
        axis = tuple(float(v) for v in reference_axis)
        if (len(axis) != 3 or not all(math.isfinite(v) for v in axis)
                or math.hypot(*axis) < 1e-9):
            raise ValueError("reference_axis must contain 3 finite numbers and be non-zero")
        constraints["reference_axis"] = list(axis)

    if max_tilt_deg is not None:
        if isinstance(max_tilt_deg, (int, float)):
            val = float(max_tilt_deg)
            if not (0 < val <= 180):
                raise ValueError("max_tilt_deg must be in (0, 180]")
            constraints["max_tilt_deg"] = val
        elif isinstance(max_tilt_deg, dict):
            for k, v in max_tilt_deg.items():
                val = float(v)
                if not (0 < val <= 180):
                    raise ValueError(f"max_tilt_deg[{k}] must be in (0, 180]")
            constraints["max_tilt_deg"] = {k: float(v) for k, v in max_tilt_deg.items()}
        else:
            raise TypeError("max_tilt_deg must be float or dict")

    if z_bounds is not None:
        if isinstance(z_bounds, (list, tuple)):
            if len(z_bounds) != 2 or z_bounds[0] > z_bounds[1]:
                raise ValueError("z_bounds must be [min_z, max_z] with min_z <= max_z")
            constraints["z_bounds"] = [float(z_bounds[0]), float(z_bounds[1])]
        elif isinstance(z_bounds, dict):
            for k, v in z_bounds.items():
                if len(v) != 2 or v[0] > v[1]:
                    raise ValueError(f"z_bounds[{k}] must be [min_z, max_z] with min_z <= max_z")
            constraints["z_bounds"] = {k: [float(v[0]), float(v[1])] for k, v in z_bounds.items()}
        else:
            raise TypeError("z_bounds must be list/tuple or dict")

    if keep_out is not None:
        if not isinstance(keep_out, (list, tuple)) or len(keep_out) > 16:
            raise ValueError("keep_out must be a list of at most 16 obstacle specs")
        cleaned_keep_out = []
        for item in keep_out:
            # 1. Direct perception result or ConstraintElement (e.g. ObjectPose from robo.locate)
            if hasattr(item, "center_xyz") or hasattr(item, "pixel") or isinstance(item, ConstraintElement):
                if getattr(item, "found", True) is False:
                    raise ValueError("keep_out obstacle was not found; re-observe before moving")
                elem = item if isinstance(item, ConstraintElement) else from_pose("obstacle", item)
                if elem.xyz is None:
                    raise ValueError("keep_out obstacle has no 3D position; re-observe before moving")
                margin = 0.02
                xy_only = elem.kind == "box"
                if elem.extent_m is not None:
                    extent = elem.extent_m[:2] if xy_only else elem.extent_m
                    r = math.hypot(*(v / 2.0 for v in extent)) + margin
                else:
                    r = 0.05 + margin
                cleaned_keep_out.append(
                    {"xyz": list(elem.xyz), "radius_m": r, "xy_only": xy_only}
                )
            # 2. Dictionary specification (either element wrapper or raw xyz/radius_m)
            elif isinstance(item, dict):
                if "element" in item:
                    elem_raw = item["element"]
                    found = (elem_raw.get("found", True) if isinstance(elem_raw, dict)
                             else getattr(elem_raw, "found", True))
                    if found is False:
                        raise ValueError("keep_out obstacle was not found; re-observe before moving")
                    elem = elem_raw if isinstance(elem_raw, ConstraintElement) else from_pose("obstacle", elem_raw)
                    if elem.xyz is None:
                        raise ValueError("keep_out obstacle has no 3D position; re-observe before moving")
                    margin = float(item.get("margin_m", 0.02))
                    if not math.isfinite(margin) or margin < 0:
                        raise ValueError("keep_out margin_m must be finite and non-negative")
                    xy_only = bool(item.get("xy_only", elem.kind == "box"))
                    if elem.extent_m is not None:
                        extent = elem.extent_m[:2] if xy_only else elem.extent_m
                        r = math.hypot(*(v / 2.0 for v in extent)) + margin
                    else:
                        r = 0.05 + margin
                    if r <= 0:
                        raise ValueError("keep_out radius_m must be positive")
                    cleaned_keep_out.append(
                        {"xyz": list(elem.xyz), "radius_m": r, "xy_only": xy_only}
                    )
                else:
                    if "xyz" not in item or "radius_m" not in item:
                        raise ValueError("Each keep_out item must contain 'xyz' and 'radius_m'")
                    xyz = tuple(float(v) for v in item["xyz"])
                    if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                        raise ValueError("keep_out xyz must contain 3 finite numbers")
                    r = float(item["radius_m"])
                    if not (math.isfinite(r) and r > 0):
                        raise ValueError("keep_out radius_m must be positive")
                    spec = {"xyz": list(xyz), "radius_m": r}
                    if "xy_only" in item:
                        spec["xy_only"] = bool(item["xy_only"])
                    cleaned_keep_out.append(spec)
            else:
                raise TypeError(f"Unsupported keep_out item type: {type(item).__name__}")
        constraints["keep_out"] = cleaned_keep_out

    if corridor_m is not None:
        if isinstance(corridor_m, (int, float)):
            val = float(corridor_m)
            if not (0 < val <= 0.2):
                raise ValueError("corridor_m must be in (0, 0.2]")
            constraints["corridor_m"] = val
        elif isinstance(corridor_m, dict):
            for k, v in corridor_m.items():
                val = float(v)
                if not (0 < val <= 0.2):
                    raise ValueError(f"corridor_m[{k}] must be in (0, 0.2]")
            constraints["corridor_m"] = {k: float(v) for k, v in corridor_m.items()}
        else:
            raise TypeError("corridor_m must be float or dict")

    if geometry is not None:
        from .hosts.robodojo_geometry import validate_geometry

        constraints["geometry"] = validate_geometry({**constraints, "geometry": geometry})

    return constraints


def _parse_2d_point(pt: Any) -> tuple[float, float] | None:
    if pt is None:
        return None
    if isinstance(pt, (tuple, list)) and len(pt) == 2:
        try:
            return (float(pt[0]), float(pt[1]))
        except (ValueError, TypeError):
            return None
    if isinstance(pt, dict):
        for keys in (("u", "v"), ("x", "y"), ("col", "row"), ("px", "py")):
            if keys[0] in pt and keys[1] in pt:
                try:
                    return (float(pt[keys[0]]), float(pt[keys[1]]))
                except (ValueError, TypeError):
                    return None
    return None


def _parse_2d_box(box: Any) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    if isinstance(box, (tuple, list)) and len(box) == 4:
        try:
            return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        except (ValueError, TypeError):
            return None
    if isinstance(box, dict):
        for keys in (("u_min", "v_min", "u_max", "v_max"), ("x_min", "y_min", "x_max", "y_max"),
                     ("xmin", "ymin", "xmax", "ymax"), ("x1", "y1", "x2", "y2")):
            if all(k in box for k in keys):
                try:
                    return tuple(float(box[k]) for k in keys)
                except (ValueError, TypeError):
                    return None
    return None


def _extract_center(obj: Any, project: Callable[[Any], tuple[float, float] | None] | None) -> tuple[float, float] | None:
    if obj is None:
        return None
    for attr in ("center_px", "centroid_px", "centroid", "pixel", "xy_px"):
        val = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
        pt = _parse_2d_point(val)
        if pt is not None:
            return pt
    if project is not None:
        for attr in ("xyz", "center_xyz", "point_xyz"):
            val = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
            if val is not None:
                try:
                    pt = project(val)
                    if pt is not None:
                        return (float(pt[0]), float(pt[1]))
                except Exception:
                    pass
    return None


def _draw_line(
    img: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    h, w = img.shape[:2]
    u0, v0 = int(round(p0[0])), int(round(p0[1]))
    u1, v1 = int(round(p1[0])), int(round(p1[1]))
    num_steps = max(abs(u1 - u0), abs(v1 - v0), 1) * 2
    us = np.round(np.linspace(u0, u1, num_steps + 1)).astype(np.int32)
    vs = np.round(np.linspace(v0, v1, num_steps + 1)).astype(np.int32)
    radius = max(1, thickness) // 2
    color_arr = np.array(color, dtype=np.uint8)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            cur_u = us + du
            cur_v = vs + dv
            valid = (cur_u >= 0) & (cur_u < w) & (cur_v >= 0) & (cur_v < h)
            img[cur_v[valid], cur_u[valid]] = color_arr


def _draw_bbox(
    img: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    u0, v0, u1, v1 = bbox
    u_min, u_max = min(u0, u1), max(u0, u1)
    v_min, v_max = min(v0, v1), max(v0, v1)
    _draw_line(img, (u_min, v_min), (u_max, v_min), color, thickness)
    _draw_line(img, (u_max, v_min), (u_max, v_max), color, thickness)
    _draw_line(img, (u_max, v_max), (u_min, v_max), color, thickness)
    _draw_line(img, (u_min, v_max), (u_min, v_min), color, thickness)


def _draw_crosshair(
    img: np.ndarray,
    center: tuple[float, float],
    color: tuple[int, int, int],
    size: int = 6,
    thickness: int = 2,
) -> None:
    u, v = center
    _draw_line(img, (u - size, v), (u + size, v), color, thickness)
    _draw_line(img, (u, v - size), (u, v + size), color, thickness)


def _blend_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 128),
    alpha: float = 0.35,
) -> None:
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        return
    bool_mask = mask.astype(bool) if mask.dtype != bool else mask
    if not np.any(bool_mask):
        return
    color_arr = np.array(color, dtype=np.float32)
    img[bool_mask] = np.clip(
        img[bool_mask].astype(np.float32) * (1.0 - alpha) + color_arr * alpha,
        0,
        255,
    ).astype(np.uint8)


def _get_3d_box_corners(
    center_xyz: tuple[float, float, float],
    extent_m: tuple[float, float, float],
    yaw_deg: float = 0.0,
) -> list[tuple[float, float, float]]:
    cx, cy, cz = center_xyz
    dx, dy, dz = extent_m[0] / 2.0, extent_m[1] / 2.0, extent_m[2] / 2.0
    yaw_rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    local = [
        (-dx, -dy, -dz),
        ( dx, -dy, -dz),
        ( dx,  dy, -dz),
        (-dx,  dy, -dz),
        (-dx, -dy,  dz),
        ( dx, -dy,  dz),
        ( dx,  dy,  dz),
        (-dx,  dy,  dz),
    ]
    corners = []
    for lx, ly, lz in local:
        gx = cx + (lx * cos_y - ly * sin_y)
        gy = cy + (lx * sin_y + ly * cos_y)
        gz = cz + lz
        corners.append((gx, gy, gz))
    return corners


def _draw_3d_wireframe(
    img: np.ndarray,
    projected: list[tuple[float, float] | None],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    if len(projected) != 8:
        return
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i0, i1 in edges:
        p0 = projected[i0]
        p1 = projected[i1]
        if p0 is not None and p1 is not None:
            _draw_line(img, p0, p1, color, thickness)


def paint_elements(
    image: np.ndarray,
    elements: Sequence[Any],
    *,
    project: Callable[[Any], tuple[float, float] | None] | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Project and paint 2D/3D geometric constraint elements onto a camera image.

    Pure numpy implementation operating directly on uint8 (H, W, 3) RGB arrays.
    Renders ConstraintElements (points, axes, boxes), ObjectObservations (bounding boxes,
    centers, masks), HoldObservations (reference vs current drift vectors and masks),
    and ObjectPose / WorldPoint 3D/2D geometry.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must be an (H, W, 3) uint8 numpy array, got {getattr(image, 'shape', None)}"
        )
    img = image.copy() if copy else image

    for elem in elements:
        if elem is None:
            continue

        # 1. ConstraintElement
        if isinstance(elem, ConstraintElement):
            if elem.kind == "point":
                pt = None
                if project is not None and elem.xyz is not None:
                    pt = project(elem.xyz)
                if pt is None:
                    pt = getattr(elem, "xy_px", getattr(elem, "pixel", None))
                if pt is not None:
                    _draw_crosshair(img, pt, color=(0, 255, 255), size=6, thickness=2)
                if elem.extent_m is not None and elem.xyz is not None and project is not None:
                    corners = _get_3d_box_corners(elem.xyz, elem.extent_m, elem.yaw_deg or 0.0)
                    proj_corners = [project(c) for c in corners]
                    _draw_3d_wireframe(img, proj_corners, color=(255, 200, 0), thickness=1)
            elif elem.kind in ("axis", "vector", "line"):
                if elem.xyz is not None and elem.axis is not None and project is not None:
                    length = elem.extent_m[0] if elem.extent_m else 0.1
                    p0 = project(elem.xyz)
                    p1_3d = (elem.xyz[0] + elem.axis[0] * length,
                             elem.xyz[1] + elem.axis[1] * length,
                             elem.xyz[2] + elem.axis[2] * length)
                    p1 = project(p1_3d)
                    if p0 is not None and p1 is not None:
                        _draw_line(img, p0, p1, color=(0, 255, 120), thickness=2)
                        _draw_crosshair(img, p0, color=(0, 255, 255), size=4, thickness=1)
            elif elem.kind in ("box", "region", "plane"):
                if elem.xyz is not None and elem.extent_m is not None and project is not None:
                    corners = _get_3d_box_corners(elem.xyz, elem.extent_m, elem.yaw_deg or 0.0)
                    proj_corners = [project(c) for c in corners]
                    _draw_3d_wireframe(img, proj_corners, color=(255, 120, 0), thickness=2)
                elif hasattr(elem, "bbox_2d"):
                    bbox = _parse_2d_box(elem.bbox_2d)
                    if bbox:
                        _draw_bbox(img, bbox, color=(255, 120, 0), thickness=2)
            continue

        # 2. HoldObservation (has reference and current, or observe_hold output)
        has_ref = "reference" in elem if isinstance(elem, dict) else hasattr(elem, "reference")
        has_cur = "current" in elem if isinstance(elem, dict) else hasattr(elem, "current")
        if has_ref and has_cur:
            ref = elem["reference"] if isinstance(elem, dict) else elem.reference
            cur = elem["current"] if isinstance(elem, dict) else elem.current
            ref_pt = _extract_center(ref, project)
            cur_pt = _extract_center(cur, project)
            if ref_pt is not None:
                _draw_crosshair(img, ref_pt, color=(0, 200, 255), size=6, thickness=2)
            if cur_pt is not None:
                _draw_crosshair(img, cur_pt, color=(255, 165, 0), size=6, thickness=2)
            if ref_pt is not None and cur_pt is not None:
                _draw_line(img, ref_pt, cur_pt, color=(255, 40, 40), thickness=2)
            cur_mask = cur.get("mask") if isinstance(cur, dict) else getattr(cur, "mask", None)
            if cur_mask is not None and isinstance(cur_mask, np.ndarray):
                _blend_mask(img, cur_mask, color=(255, 165, 0), alpha=0.3)
            continue

        # 3. ObjectPose / WorldPoint / Perception results
        if hasattr(elem, "center_xyz") or hasattr(elem, "length_m"):  # ObjectPose-like
            c3d = getattr(elem, "center_xyz", None)
            if c3d is not None and project is not None:
                c2d = project(c3d)
                if c2d is not None:
                    _draw_crosshair(img, c2d, color=(255, 255, 0), size=6, thickness=2)
                if getattr(elem, "length_m", None) and getattr(elem, "width_m", None):
                    hm = getattr(elem, "height_m", 0.05) or 0.05
                    yaw = getattr(elem, "long_axis_deg", 0.0) or 0.0
                    corners = _get_3d_box_corners(c3d, (elem.length_m, elem.width_m, hm), yaw)
                    proj_corners = [project(c) for c in corners]
                    _draw_3d_wireframe(img, proj_corners, color=(0, 220, 255), thickness=2)
            elif hasattr(elem, "centroid") and elem.centroid is not None:
                c2d = _parse_2d_point(elem.centroid)
                if c2d is not None:
                    _draw_crosshair(img, c2d, color=(255, 255, 0), size=6, thickness=2)
            continue

        if hasattr(elem, "pixel") and hasattr(elem, "depth_m"):  # WorldPoint-like
            pt = None
            if project is not None and getattr(elem, "xyz", None) is not None:
                pt = project(elem.xyz)
            if pt is None and getattr(elem, "pixel", None) is not None:
                pt = _parse_2d_point(elem.pixel)
            if pt is not None:
                _draw_crosshair(img, pt, color=(0, 255, 255), size=6, thickness=2)
            continue

        # 4. ObjectObservation or dict from observe_object
        bbox_raw = elem.get("bbox_2d") if isinstance(elem, dict) else getattr(elem, "bbox_2d", None)
        if bbox_raw is None:
            bbox_raw = elem.get("box") if isinstance(elem, dict) else getattr(elem, "box", None)
        if bbox_raw is None:
            bbox_raw = elem.get("bbox") if isinstance(elem, dict) else getattr(elem, "bbox", None)
        bbox = _parse_2d_box(bbox_raw)
        if bbox is not None:
            _draw_bbox(img, bbox, color=(0, 255, 0), thickness=2)

        center_raw = elem.get("center_px") if isinstance(elem, dict) else getattr(elem, "center_px", None)
        if center_raw is None:
            center_raw = elem.get("centroid_px") if isinstance(elem, dict) else getattr(elem, "centroid_px", None)
        if center_raw is None:
            center_raw = elem.get("centroid") if isinstance(elem, dict) else getattr(elem, "centroid", None)
        center = _parse_2d_point(center_raw)
        if center is None and project is not None:
            xyz = elem.get("xyz") if isinstance(elem, dict) else getattr(elem, "xyz", None)
            if xyz is not None:
                center = project(xyz)
        if center is not None:
            _draw_crosshair(img, center, color=(0, 255, 0), size=6, thickness=2)

        mask = elem.get("mask") if isinstance(elem, dict) else getattr(elem, "mask", None)
        if mask is not None and isinstance(mask, np.ndarray):
            _blend_mask(img, mask, color=(0, 255, 128), alpha=0.35)

        # 5. Direct raw 2D primitives: tuples / lists / dicts
        if bbox is None and center is None:
            if isinstance(elem, (tuple, list)):
                if len(elem) == 4:
                    box = _parse_2d_box(elem)
                    if box:
                        _draw_bbox(img, box, color=(255, 255, 0), thickness=2)
                elif len(elem) == 2:
                    pt = _parse_2d_point(elem)
                    if pt:
                        _draw_crosshair(img, pt, color=(0, 255, 255), size=6, thickness=2)
            elif isinstance(elem, dict):
                col = elem.get("color", (255, 255, 0))
                if "line" in elem or "vector" in elem:
                    pts = elem.get("line") or elem.get("vector")
                    if isinstance(pts, (tuple, list)) and len(pts) == 2:
                        p0 = _parse_2d_point(pts[0]) or (project(pts[0]) if project else None)
                        p1 = _parse_2d_point(pts[1]) or (project(pts[1]) if project else None)
                        if p0 and p1:
                            _draw_line(img, p0, p1, col, thickness=elem.get("thickness", 2))
                elif "xyz" in elem and project is not None:
                    p = project(elem["xyz"])
                    if p:
                        _draw_crosshair(img, p, col, size=6, thickness=2)
                        if "extent_m" in elem:
                            corners = _get_3d_box_corners(elem["xyz"], elem["extent_m"], elem.get("yaw_deg", 0.0))
                            proj_corners = [project(c) for c in corners]
                            _draw_3d_wireframe(img, proj_corners, col, thickness=1)

    return img


__all__ = [
    "ConstraintCheck",
    "ConstraintElement",
    "check_above",
    "check_distance",
    "check_fact",
    "check_hold",
    "check_hold_evidence",
    "check_in_bounds",
    "check_tilt",
    "evaluate_constraints",
    "from_arm",
    "from_pose",
    "paint_elements",
    "proactive_constraints",
]

