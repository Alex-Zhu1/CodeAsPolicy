# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Kinematics of RoboDojo's two ARX-X5 arms, computed in the host: full pose, no RPC.

RPent's RoboDojo environment exposes one IK call, ``solve_ik_position``: a target position, a
few candidate orientations (the current one, the starting one, 25 degree tilts of the current
one about world x and y), the one needing the least joint motion.  Nothing can ask for an
orientation, and the tilts accumulate: an arm sent away and back to its starting position by
``move_to`` came back rotated by 47 to 52 degrees.  The simulator's own full-pose action path
(``<arm>_ee_pose``) swings the arm when its IK diverges.

So the host solves IK itself.  The chain is the benchmark's URDF (``Assets/Robots/x5/X5A.urdf``,
joints ``joint1`` .. ``joint6``, ending at ``link6``), and each arm's base pose was fitted to
113 (joint angles, reported end-effector pose) pairs recorded in four episodes: the
reported ``<arm>_ee_pose`` is exactly the pose of ``link6`` in the world (0.00 mm and 0.000
degrees of residual), with the bases on the table edge at x = -0.30 / +0.30, y = -0.45,
z = 0.765, turned 90 degrees about z.  The solution is commanded as joint angles, the action
path RPent's ``move_to`` uses for its own IK solutions.

Frames.  ``link6``'s x axis is the gripper's pointing direction (wrist to fingertips) and the
two fingers close along its y axis.  At the start both grippers point away from the robot
(+y in the world) with the fingers side by side (closing along world x).  :data:`TCP_OFFSET`
places the grasp point, between the fingertips, on the pointing axis.

Orientation for people: ``pitch`` is how far the pointing direction is below the horizontal
(0 level, 90 straight down), ``yaw`` its heading on the table (0 away from the robot, +90 to
the robot's left, i.e. world -x), ``roll`` the turn of the fingers about the pointing direction
(0 as at the start).  The rotation is ``Rz(90 + yaw) @ Ry(pitch) @ Rx(roll)``; pointing straight
down, ``yaw`` and ``roll`` both turn about the vertical, and :func:`angles` reports roll 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..robodojo import TCP_OFFSET as _TCP_OFFSET

ARMS = ("left", "right")

#: Joint origins (xyz, rpy) and axes of joint1 .. joint6 of X5A.urdf.
_CHAIN = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1.0, 0.0, 0.0)),
)
#: Joint limits the simulator enforces, measured (each joint alone driven to +-3 rad from the
#: start, examples/robodojo_joint_limits.py), with a small margin; X5A.urdf says +-10 (joint6
#: +-3.14), and an IK solution with joint5 at -pi stopped at -1.55.  joint5 stops at +-1.56,
#: joint3 at -0.04 and joint2 at -0.23 below the start (0); joint1, joint3 (up) and joint6 turned
#: the full 3 rad; joint2 and joint4 were stopped by the table / the arm itself before a limit
#: (joint4 reached -1.68 and +1.78 in other runs).
JOINT_LIMITS = (
    (-3.1, 3.1),
    (-0.2, 3.1),
    (0.0, 3.1),
    (-1.75, 1.75),
    (-1.52, 1.52),
    (-3.1, 3.1),
)
#: Fitted base of each arm: (x, y, z) and yaw about the world z axis.
BASES = {
    "left": ((-0.3, -0.45, 0.765), 1.5710982518),
    "right": ((0.3, -0.45, 0.765), 1.5710982518),
}
#: Table surface height (the arm bases stand on it).
TABLE_Z = 0.765
#: The table top's extent [x], [y] (RoboDojo's ``env_cfg/scene/default.yml``: a 1.4 x 1.1 m slab
#: centred at y = -0.05).  The arms stand on it at y = -0.45; an elbow folded back past its
#: near edge (y = -0.60) may go lower than the top.
TABLE_X = (-0.70, 0.70)
TABLE_Y = (-0.60, 0.50)
_EDGE_MARGIN = 0.03
#: The grasp point on link6's x axis, metres: the middle of the finger pads, which span 0.13 to
#: 0.155 (finger meshes); RoboDojo's own robot config calls it ``gripper_bias``.
TCP_OFFSET = _TCP_OFFSET
#: The fingertips, for the table check.
TIP_OFFSET = 0.158
#: Far-range cutoff (metres) above which housing_exposure_penalty returns 0.0.
_HOUSING_EXPOSURE_CUTOFF_M = 0.38
#: Canonical deterministic ordering for inter-arm collision labels.
_VIOLATION_ORDER = (
    "camera_housing_collision",
    "crossbar_collision",
    "finger_collision",
    "forearm_collision",
)
#: Local link6 box specifications: (part, (cx, cy, cz), (hx, hy, hz), bound_r).
_CAMERA_HOUSING_LOCAL = (
    "camera_housing",
    (0.0375, 0.0, 0.021),
    (0.0475, 0.032, 0.051),
    math.sqrt(0.0475**2 + 0.032**2 + 0.051**2),
)
_CROSSBAR_LOCAL = (
    "crossbar",
    (0.084, 0.0, 0.0),
    (0.014, 0.050, 0.020),
    math.sqrt(0.014**2 + 0.050**2 + 0.020**2),
)
_FINGER_HALF_EXTENTS = (0.0315, 0.006, 0.014)
_FINGER_BOUND_R = math.sqrt(0.0315**2 + 0.006**2 + 0.014**2)
_FOREARM_RADII = (0.030, 0.026, 0.024)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _homogeneous(rot: np.ndarray, pos: Sequence[float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


_ORIGINS = tuple(
    _homogeneous(_rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0]), xyz) for xyz, rpy, _ in _CHAIN
)
_AXES = tuple(np.asarray(axis) for _, _, axis in _CHAIN)
_BASE_T = {arm: _homogeneous(_rot_z(yaw), xyz) for arm, (xyz, yaw) in BASES.items()}


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")


def frames(arm: str, q: Sequence[float]) -> list[np.ndarray]:
    """World poses of each joint frame (after its rotation) and of link6, in chain order."""

    _check_arm(arm)
    if len(q) != 6:
        raise ValueError("an X5 arm has 6 joints")
    t = _BASE_T[arm]
    out = []
    for origin, axis, angle in zip(_ORIGINS, _AXES, q, strict=True):
        t = t @ origin @ _homogeneous(_axis_angle(axis, float(angle)), (0.0, 0.0, 0.0))
        out.append(t)
    return out


def fk(arm: str, q: Sequence[float]) -> np.ndarray:
    """World pose (4x4) of link6, the frame RoboDojo reports as ``<arm>_ee_pose``."""

    return frames(arm, q)[-1]


def tcp(pose: np.ndarray, offset: float = TCP_OFFSET) -> np.ndarray:
    """The grasp point of a link6 pose."""

    return pose[:3, 3] + pose[:3, 0] * offset


def quat_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    """A rotation matrix as a unit quaternion [w, x, y, z] (w >= 0)."""

    m = rot
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(v * v for v in q))
    q = tuple(v / n for v in q)
    return q if q[0] >= 0 else tuple(-v for v in q)  # type: ignore[return-value]


def rot_from_quat(q: Sequence[float]) -> np.ndarray:
    """[w, x, y, z] to a rotation matrix."""

    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """The gripper rotation for (pitch, yaw, roll) in degrees (module docstring)."""

    return (
        _rot_z(math.radians(90.0 + yaw)) @ _rot_y(math.radians(pitch)) @ _rot_x(math.radians(roll))
    )


_VERTICAL = math.sin(math.radians(1.0))


def angles(rot: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of a gripper rotation; roll 0 when pointing straight down."""

    point = rot[:, 0]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    horizontal = math.hypot(point[0], point[1])
    # within a degree of straight up or down, heading and roll turn about (almost) the same
    # axis and only their difference is well defined: report it as the heading, roll 0
    if horizontal < _VERTICAL:
        jaw = rot[:, 1]
        # with roll 0 the jaw axis is Rz(90 + yaw) applied to +y, i.e. (-cos yaw, -sin yaw, 0)
        yaw = math.degrees(math.atan2(-jaw[1], -jaw[0]))
        return pitch, _wrap(yaw), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    base = orientation(pitch, yaw, 0.0)
    rest = base.T @ rot  # a rotation about x by roll
    roll = math.degrees(math.atan2(rest[2, 1], rest[1, 1]))
    return pitch, _wrap(yaw), _wrap(roll)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two rotations."""

    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _rotvec(rot: np.ndarray) -> np.ndarray:
    c = (np.trace(rot) - 1.0) / 2.0
    angle = math.acos(max(-1.0, min(1.0, c)))
    if angle < 1e-9:
        return np.zeros(3)
    w = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:  # angle near pi: take the axis from the symmetric part
        axis = np.sqrt(np.maximum(np.diag(rot) + 1.0, 0.0) / 2.0)
        axis = axis * np.sign(w + (w == 0))
        return axis / np.linalg.norm(axis) * angle
    return w / (2 * s) * angle


def jacobian(arm: str, q: Sequence[float], offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(6x6 world-frame geometric Jacobian of the point ``offset`` along link6's x, its pose)."""

    fr = frames(arm, q)
    end = fr[-1]
    point = end[:3, 3] + end[:3, 0] * offset
    jac = np.zeros((6, 6))
    for i, (t, axis) in enumerate(zip(fr, _AXES, strict=True)):
        a = t[:3, :3] @ axis
        jac[:3, i] = np.cross(a, point - t[:3, 3])
        jac[3:, i] = a
    return jac, end


@dataclass(frozen=True)
class IKResult:
    q: tuple[float, ...]
    pos_err_m: float
    rot_err_deg: float
    ok: bool
    collides: bool


def clearance_violations(
    arm: str,
    q: Sequence[float],
    offset: float = TCP_OFFSET,
    *,
    allow_table_touch: bool = False,
) -> list[str]:
    """Parts of the arm in the table: below its top, over its extent.

    With ``allow_table_touch=True``, only the fingertip floor is relaxed from
    ``TABLE_Z + 0.002`` down to ``TABLE_Z - 0.004`` so a guarded stall-stopped
    descent can seat the flat finger pads past the bevel onto the tabletop;
    ``joint2``..``joint6`` remain strictly above ``TABLE_Z + 0.015``.
    """

    def over(p: np.ndarray) -> bool:
        return (
            TABLE_X[0] - _EDGE_MARGIN <= p[0] <= TABLE_X[1] + _EDGE_MARGIN
            and TABLE_Y[0] - _EDGE_MARGIN <= p[1] <= TABLE_Y[1] + _EDGE_MARGIN
        )

    fr = frames(arm, q)
    bad = []
    for i, t in enumerate(fr[1:], start=2):  # joint1's frame is at the base
        if t[2, 3] < TABLE_Z + 0.015 and over(t[:3, 3]):
            bad.append(f"joint{i}")
    tip = fr[-1][:3, 3] + fr[-1][:3, 0] * max(offset, TIP_OFFSET)
    tip_floor = TABLE_Z - 0.004 if allow_table_touch else TABLE_Z + 0.002
    if tip[2] < tip_floor and over(tip):
        bad.append("fingertips")
    return bad


def _seg_seg_dist_sq(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    q0: tuple[float, float, float],
    q1: tuple[float, float, float],
) -> float:
    """Exact squared Euclidean distance between 3D line segments [p0, p1] and [q0, q1]."""

    d1x, d1y, d1z = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    d2x, d2y, d2z = q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]
    rx, ry, rz = p0[0] - q0[0], p0[1] - q0[1], p0[2] - q0[2]
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    eps = 1e-18
    if a <= eps and e <= eps:
        return rx * rx + ry * ry + rz * rz
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            denom = a * e - b * b
            if denom > eps:
                s = max(0.0, min(1.0, (b * f - c * e) / denom))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))
    dx = rx + d1x * s - d2x * t
    dy = ry + d1y * s - d2y * t
    dz = rz + d1z * s - d2z * t
    return dx * dx + dy * dy + dz * dz


def _support(prim: tuple, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    """Support point of an OBB or capsule's core line segment along direction (dx, dy, dz)."""

    if prim[0] == 0:  # OBB: (0, part, center, rot_flat, half_extents, bound_r)
        cx, cy, cz = prim[2]
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = prim[3]
        hx, hy, hz = prim[4]
        sx = hx if (r00 * dx + r10 * dy + r20 * dz) >= 0.0 else -hx
        sy = hy if (r01 * dx + r11 * dy + r21 * dz) >= 0.0 else -hy
        sz = hz if (r02 * dx + r12 * dy + r22 * dz) >= 0.0 else -hz
        return (
            cx + r00 * sx + r01 * sy + r02 * sz,
            cy + r10 * sx + r11 * sy + r12 * sz,
            cz + r20 * sx + r21 * sy + r22 * sz,
        )
    # Capsule: (1, part, center, p0, p1, radius, bound_r)
    p0, p1 = prim[3], prim[4]
    if p0[0] * dx + p0[1] * dy + p0[2] * dz >= p1[0] * dx + p1[1] * dy + p1[2] * dz:
        return p0
    return p1


def _closest_seg(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[tuple[float, float, float], list[tuple[float, float, float]]]:
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    ab_sq = abx * abx + aby * aby + abz * abz
    if ab_sq <= 1e-24:
        return a, [a]
    t = -(a[0] * abx + a[1] * aby + a[2] * abz) / ab_sq
    if t <= 0.0:
        return a, [a]
    if t >= 1.0:
        return b, [b]
    return (a[0] + t * abx, a[1] + t * aby, a[2] + t * abz), [a, b]


def _closest_tri(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[tuple[float, float, float], list[tuple[float, float, float]]]:
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    d1 = -(abx * a[0] + aby * a[1] + abz * a[2])
    d2 = -(acx * a[0] + acy * a[1] + acz * a[2])
    if d1 <= 0.0 and d2 <= 0.0:
        return a, [a]
    d3 = -(abx * b[0] + aby * b[1] + abz * b[2])
    d4 = -(acx * b[0] + acy * b[1] + acz * b[2])
    if d3 >= 0.0 and d4 <= d3:
        return b, [b]
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        if denom <= 1e-24:
            return a, [a]
        v = d1 / denom
        return (a[0] + v * abx, a[1] + v * aby, a[2] + v * abz), [a, b]
    d5 = -(abx * c[0] + aby * c[1] + abz * c[2])
    d6 = -(acx * c[0] + acy * c[1] + acz * c[2])
    if d6 >= 0.0 and d5 <= d6:
        return c, [c]
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        if denom <= 1e-24:
            return a, [a]
        w = d2 / denom
        return (a[0] + w * acx, a[1] + w * acy, a[2] + w * acz), [a, c]
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        if denom <= 1e-24:
            return b, [b]
        w = (d4 - d3) / denom
        return (
            b[0] + w * (c[0] - b[0]),
            b[1] + w * (c[1] - b[1]),
            b[2] + w * (c[2] - b[2]),
        ), [b, c]
    total = va + vb + vc
    if total <= 1e-24:
        best_v, best_s = _closest_seg(a, b)
        best_d2 = best_v[0] * best_v[0] + best_v[1] * best_v[1] + best_v[2] * best_v[2]
        for p, q in ((a, c), (b, c)):
            v_e, s_e = _closest_seg(p, q)
            d2 = v_e[0] * v_e[0] + v_e[1] * v_e[1] + v_e[2] * v_e[2]
            if d2 < best_d2:
                best_d2, best_v, best_s = d2, v_e, s_e
        return best_v, best_s
    inv = 1.0 / total
    v = vb * inv
    w = vc * inv
    return (
        a[0] + v * abx + w * acx,
        a[1] + v * aby + w * acy,
        a[2] + v * abz + w * acz,
    ), [a, b, c]


def _closest_on_simplex(
    simplex: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], list[tuple[float, float, float]]]:
    n = len(simplex)
    if n == 1:
        return simplex[0], simplex
    if n == 2:
        return _closest_seg(simplex[0], simplex[1])
    if n == 3:
        return _closest_tri(simplex[0], simplex[1], simplex[2])
    a, b, c, d = simplex
    faces = ((a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a))
    best_v: tuple[float, float, float] | None = None
    best_s: list[tuple[float, float, float]] | None = None
    best_d2 = math.inf
    degenerate = False
    for p, q, r, s in faces:
        pqx, pqy, pqz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
        prx, pry, prz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
        nx = pqy * prz - pqz * pry
        ny = pqz * prx - pqx * prz
        nz = pqx * pry - pqy * prx
        sign_s = (s[0] - p[0]) * nx + (s[1] - p[1]) * ny + (s[2] - p[2]) * nz
        if abs(sign_s) <= 1e-18:
            degenerate = True
            break
        sign_o = -(p[0] * nx + p[1] * ny + p[2] * nz)
        if sign_o * sign_s < 0.0:
            v_f, s_f = _closest_tri(p, q, r)
            d2 = v_f[0] * v_f[0] + v_f[1] * v_f[1] + v_f[2] * v_f[2]
            if d2 < best_d2:
                best_d2, best_v, best_s = d2, v_f, s_f
    if degenerate:
        best_d2 = math.inf
        for p, q, r, _ in faces:
            v_f, s_f = _closest_tri(p, q, r)
            d2 = v_f[0] * v_f[0] + v_f[1] * v_f[1] + v_f[2] * v_f[2]
            if d2 < best_d2:
                best_d2, best_v, best_s = d2, v_f, s_f
        return best_v, best_s  # type: ignore[return-value]
    if best_v is None:
        return (0.0, 0.0, 0.0), simplex
    return best_v, best_s  # type: ignore[return-value]


def _gjk_core_distance(
    prim_a: tuple,
    prim_b: tuple,
    *,
    early_clear_sq: float | None = None,
    early_hit_sq: float | None = None,
) -> float:
    """Exact GJK Euclidean distance between core shapes (OBB or segment); 0.0 when overlapping."""

    ca, cb = prim_a[2], prim_b[2]
    dx, dy, dz = cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]
    if dx * dx + dy * dy + dz * dz <= 1e-18:
        dx, dy, dz = 1.0, 0.0, 0.0
    sa = _support(prim_a, dx, dy, dz)
    sb = _support(prim_b, -dx, -dy, -dz)
    w = (sa[0] - sb[0], sa[1] - sb[1], sa[2] - sb[2])
    simplex = [w]
    v = w
    for _ in range(32):
        v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if v_sq <= 1e-18:
            return 0.0
        if early_hit_sq is not None and v_sq < early_hit_sq:
            return math.sqrt(v_sq)
        dx, dy, dz = -v[0], -v[1], -v[2]
        sa = _support(prim_a, dx, dy, dz)
        sb = _support(prim_b, -dx, -dy, -dz)
        w = (sa[0] - sb[0], sa[1] - sb[1], sa[2] - sb[2])
        vw = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
        if early_clear_sq is not None and vw > 0.0 and vw * vw >= early_clear_sq * v_sq:
            return vw / math.sqrt(v_sq)
        if v_sq - vw <= 1e-12 * max(1.0, v_sq):
            return math.sqrt(v_sq)
        if any(
            (w[0] - p[0]) ** 2 + (w[1] - p[1]) ** 2 + (w[2] - p[2]) ** 2 <= 1e-18
            for p in simplex
        ):
            return math.sqrt(v_sq)
        simplex.append(w)
        v_new, simplex_new = _closest_on_simplex(simplex)
        v_new_sq = v_new[0] * v_new[0] + v_new[1] * v_new[1] + v_new[2] * v_new[2]
        if v_new_sq <= 1e-18:
            return 0.0
        if v_new_sq >= v_sq - 1e-15:
            return math.sqrt(v_sq)
        v, simplex = v_new, simplex_new
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _primitive_distance(
    prim_a: tuple,
    prim_b: tuple,
    *,
    threshold: float | None = None,
) -> float:
    """Exact 3D distance between two collision primitives (OBB or capsule)."""

    kind_a, kind_b = prim_a[0], prim_b[0]
    if kind_a == 1 and kind_b == 1:
        d_sq = _seg_seg_dist_sq(prim_a[3], prim_a[4], prim_b[3], prim_b[4])
        return math.sqrt(d_sq) - (prim_a[5] + prim_b[5])
    rad = (prim_a[5] if kind_a == 1 else 0.0) + (prim_b[5] if kind_b == 1 else 0.0)
    if threshold is not None:
        core_thresh = threshold + rad
        if core_thresh > 0.0:
            sq = core_thresh * core_thresh
            raw = _gjk_core_distance(prim_a, prim_b, early_clear_sq=sq, early_hit_sq=sq)
        else:
            raw = _gjk_core_distance(prim_a, prim_b, early_clear_sq=0.0)
    else:
        raw = _gjk_core_distance(prim_a, prim_b)
    return raw - rad


def _arm_envelopes(arm: str, q: Sequence[float], opening: float) -> list[tuple]:
    """Physical ARX-X5 collision primitives (capsules + OBBs) for one arm."""

    fr = frames(arm, q)
    op = max(0.0, min(1.0, float(opening)))
    t6 = fr[5]
    r00, r01, r02 = float(t6[0, 0]), float(t6[0, 1]), float(t6[0, 2])
    r10, r11, r12 = float(t6[1, 0]), float(t6[1, 1]), float(t6[1, 2])
    r20, r21, r22 = float(t6[2, 0]), float(t6[2, 1]), float(t6[2, 2])
    rot_flat = (r00, r01, r02, r10, r11, r12, r20, r21, r22)
    p6x, p6y, p6z = float(t6[0, 3]), float(t6[1, 3]), float(t6[2, 3])

    def _obb(
        part: str,
        lc: tuple[float, float, float],
        half: tuple[float, float, float],
        bound_r: float,
    ) -> tuple:
        cx = p6x + r00 * lc[0] + r01 * lc[1] + r02 * lc[2]
        cy = p6y + r10 * lc[0] + r11 * lc[1] + r12 * lc[2]
        cz = p6z + r20 * lc[0] + r21 * lc[1] + r22 * lc[2]
        return (0, part, (cx, cy, cz), rot_flat, half, bound_r)

    y_finger = 0.012 + 0.0545 * op
    prims: list[tuple] = [
        _obb(*_CAMERA_HOUSING_LOCAL),
        _obb(*_CROSSBAR_LOCAL),
        _obb("finger", (0.1265, y_finger, 0.0), _FINGER_HALF_EXTENTS, _FINGER_BOUND_R),
        _obb("finger", (0.1265, -y_finger, 0.0), _FINGER_HALF_EXTENTS, _FINGER_BOUND_R),
    ]
    pts = [
        (float(fr[i][0, 3]), float(fr[i][1, 3]), float(fr[i][2, 3]))
        for i in (2, 3, 4, 5)
    ]
    for (p0, p1), rad in zip(zip(pts[:-1], pts[1:], strict=True), _FOREARM_RADII, strict=True):
        cx = 0.5 * (p0[0] + p1[0])
        cy = 0.5 * (p0[1] + p1[1])
        cz = 0.5 * (p0[2] + p1[2])
        half_len = 0.5 * math.sqrt(
            (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
        )
        prims.append((1, "forearm", (cx, cy, cz), p0, p1, rad, half_len + rad))
    return prims


def _pair_label(part_a: str, part_b: str) -> str:
    if part_a == "camera_housing" or part_b == "camera_housing":
        return "camera_housing_collision"
    if part_a == "crossbar" or part_b == "crossbar":
        return "crossbar_collision"
    if part_a == "finger" or part_b == "finger":
        return "finger_collision"
    return "forearm_collision"


def inter_arm_violations(
    left_q: Sequence[float],
    right_q: Sequence[float],
    *,
    left_opening: float = 0.5,
    right_opening: float = 0.5,
    min_clearance_m: float = 0.010,
) -> list[str]:
    """Deterministic inter-arm collision labels when 3D envelope clearance < ``min_clearance_m``."""

    left_prims = _arm_envelopes("left", left_q, left_opening)
    right_prims = _arm_envelopes("right", right_q, right_opening)
    margin = float(min_clearance_m)
    found: set[str] = set()
    for pa in left_prims:
        ca = pa[2]
        ra = pa[-1]
        for pb in right_prims:
            label = _pair_label(pa[1], pb[1])
            if label in found:
                continue
            cb = pb[2]
            r_sum = ra + pb[-1] + max(margin, 0.0)
            dx = ca[0] - cb[0]
            dy = ca[1] - cb[1]
            dz = ca[2] - cb[2]
            if dx * dx + dy * dy + dz * dz >= r_sum * r_sum + 1e-12:
                continue
            dist = _primitive_distance(pa, pb, threshold=margin)
            if (dist < margin) if margin > 0.0 else (dist <= 1e-9):
                found.add(label)
    return [label for label in _VIOLATION_ORDER if label in found]


def _inter_arm_min_distance(
    left_q: Sequence[float],
    right_q: Sequence[float],
    *,
    left_opening: float = 0.5,
    right_opening: float = 0.5,
) -> float:
    """Minimum exact 3D Euclidean clearance (metres) between the two arms' envelopes."""

    left_prims = _arm_envelopes("left", left_q, left_opening)
    right_prims = _arm_envelopes("right", right_q, right_opening)
    best = math.inf
    for pa in left_prims:
        ca = pa[2]
        ra = pa[-1]
        for pb in right_prims:
            cb = pb[2]
            dx = ca[0] - cb[0]
            dy = ca[1] - cb[1]
            dz = ca[2] - cb[2]
            lower_bound = math.sqrt(dx * dx + dy * dy + dz * dz) - (ra + pb[-1])
            if lower_bound >= best:
                continue
            dist = _primitive_distance(pa, pb)
            if dist < best:
                best = dist
    return best


def housing_exposure_penalty(
    arm: str,
    q: Sequence[float],
    other_q: Sequence[float] | None = None,
) -> float:
    """Smooth penalty when ``arm``'s ``+z_link6`` camera housing faces the other arm at close range.

    Returns ``0.0`` when ``other_q is None`` or TCPs are farther than ``0.38 m``.
    """

    _check_arm(arm)
    if other_q is None:
        return 0.0
    other_arm = "right" if arm == "left" else "left"
    pose = fk(arm, q)
    other_pose = fk(other_arm, other_q)
    tcp_self = tcp(pose)
    tcp_other = tcp(other_pose)
    tcp_dist = float(np.linalg.norm(tcp_other - tcp_self))
    if tcp_dist > _HOUSING_EXPOSURE_CUTOFF_M:
        return 0.0
    proximity = (_HOUSING_EXPOSURE_CUTOFF_M - tcp_dist) / _HOUSING_EXPOSURE_CUTOFF_M
    z_axis = pose[:3, 2]
    housing_origin = pose[:3, 3] + pose[:3, 0] * 0.0375
    align = 0.0
    for target_pt in (tcp_other, other_pose[:3, 3]):
        vec = target_pt - housing_origin
        norm = float(np.linalg.norm(vec))
        cos_theta = 1.0 if norm < 1e-9 else float(np.dot(z_axis, vec) / norm)
        if cos_theta > align:
            align = cos_theta
    return float(proximity * align)


def twist_margin_rad(q: Sequence[float], twist_rad: float = 0.0) -> float:
    """Headroom of ``joint6`` against ``JOINT_LIMITS[5]`` before and after a ``twist_rad`` roll."""

    if len(q) != 6:
        raise ValueError("an X5 arm has 6 joints")
    lo, hi = JOINT_LIMITS[5]
    q6 = float(q[5])
    margin = min(q6 - lo, hi - q6)
    if twist_rad != 0.0:
        q6_after = q6 + float(twist_rad)
        margin = min(margin, q6_after - lo, hi - q6_after)
    return float(margin)


def solve(
    arm: str,
    target: np.ndarray,
    seed: Sequence[float],
    *,
    offset: float = TCP_OFFSET,
    rot_weight: float = 0.2,
    pos_tol: float = 0.002,
    rot_tol_deg: float = 1.0,
    iterations: int = 150,
    allow_table_touch: bool = False,
) -> IKResult:
    """Damped least squares from ``seed`` for a 4x4 target of the point ``offset`` along link6's x.

    ``rot_weight`` is metres per radian in the error: it only sets how the solver trades the two
    while it converges, and what it settles on when the pose is out of reach.
    """

    q = np.array(seed, dtype=float)
    lo = np.array([lim[0] for lim in JOINT_LIMITS])
    hi = np.array([lim[1] for lim in JOINT_LIMITS])
    damping = 0.02
    pos_err = rot_err = math.inf
    for _ in range(iterations):
        jac, end = jacobian(arm, q, offset)
        point = end[:3, 3] + end[:3, 0] * offset
        dp = target[:3, 3] - point
        dr = _rotvec(target[:3, :3] @ end[:3, :3].T)
        pos_err, rot_err = float(np.linalg.norm(dp)), math.degrees(float(np.linalg.norm(dr)))
        if pos_err < pos_tol * 0.25 and rot_err < rot_tol_deg * 0.25:
            break
        err = np.concatenate([dp, rot_weight * dr])
        j = jac.copy()
        j[3:] *= rot_weight
        step = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), err)
        norm = float(np.linalg.norm(step))
        if norm > 0.25:
            step *= 0.25 / norm
        q = np.clip(q + step, lo, hi)
    end = fk(arm, q)
    point = end[:3, 3] + end[:3, 0] * offset
    pos_err = float(np.linalg.norm(target[:3, 3] - point))
    rot_err = rotation_error(target[:3, :3], end[:3, :3])
    ok = pos_err <= pos_tol and rot_err <= rot_tol_deg
    return IKResult(
        tuple(float(v) for v in q),
        pos_err,
        rot_err,
        ok,
        bool(clearance_violations(arm, q, offset, allow_table_touch=allow_table_touch)),
    )


def best_solution(
    arm: str,
    target: np.ndarray,
    current: Sequence[float],
    *,
    offset: float = TCP_OFFSET,
    extra_seeds: Sequence[Sequence[float]] = (),
    allow_table_touch: bool = False,
    jaw_symmetry: bool = False,
    other_arm_q: Sequence[float] | None = None,
    opening: float = 0.5,
    other_opening: float = 0.5,
    twist_rad: float = 0.0,
    **kwargs,
) -> IKResult:
    """The solution nearest ``current`` that reaches the target and clears table/other arm.

    Seeds: the current joints, the same with the wrist turned half a turn either way, the extra
    seeds given (the starting joints), and a few fixed perturbations of the current joints.  When
    ``jaw_symmetry=True``, both ``target`` and ``target @ Rx(pi)`` are evaluated.  When
    ``other_arm_q`` is provided, inter-arm envelope violations mark candidates as colliding and
    valid candidates are penalized for pointing the ``+z_link6`` camera housing toward the other
    arm.  When ``twist_rad != 0``, candidates with sufficient ``joint6`` twist headroom are
    preferred.  When no seed reaches the target collision-free, the closest miss is returned.
    """

    _check_arm(arm)
    cur = np.array(current, dtype=float)
    seeds = [cur]
    for d in (math.pi, -math.pi):
        s = cur.copy()
        s[5] = s[5] + d
        if JOINT_LIMITS[5][0] <= s[5] <= JOINT_LIMITS[5][1]:
            seeds.append(s)
    if jaw_symmetry or twist_rad != 0.0:
        for q6_seed in (-2.5, 0.0, 2.5):
            if abs(q6_seed - cur[5]) > 0.4:
                s = cur.copy()
                s[5] = q6_seed
                seeds.append(s)
    seeds.extend(np.array(s, dtype=float) for s in extra_seeds)
    rng = np.random.default_rng(0)
    seeds.extend(cur + rng.normal(0.0, 0.6, size=6) for _ in range(6))

    targets = [target]
    if jaw_symmetry:
        flipped = target.copy()
        flipped[:3, :3] = target[:3, :3] @ _rot_x(math.pi)
        targets.append(flipped)

    good, misses = [], []
    for cand_target in targets:
        for s in seeds:
            res = solve(
                arm,
                cand_target,
                s,
                offset=offset,
                allow_table_touch=allow_table_touch,
                **kwargs,
            )
            if not res.collides and other_arm_q is not None:
                if arm == "left":
                    bad = inter_arm_violations(
                        res.q,
                        other_arm_q,
                        left_opening=opening,
                        right_opening=other_opening,
                    )
                else:
                    bad = inter_arm_violations(
                        other_arm_q,
                        res.q,
                        left_opening=other_opening,
                        right_opening=opening,
                    )
                if bad:
                    res = IKResult(res.q, res.pos_err_m, res.rot_err_deg, res.ok, True)
            (good if res.ok and not res.collides else misses).append(res)
    if good:
        weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])

        def _score(r: IKResult) -> tuple[bool, bool, float]:
            travel = float(np.sum(weights * np.abs(np.array(r.q) - cur)))
            exposure = (
                housing_exposure_penalty(arm, r.q, other_arm_q)
                if other_arm_q is not None
                else 0.0
            )
            if twist_rad != 0.0:
                margin = twist_margin_rad(r.q, twist_rad)
                infeasible = margin < 0.0
                tight = margin < 0.15
                twist_pen = max(0.0, 0.5 - margin) * 4.0
            else:
                infeasible = False
                tight = False
                twist_pen = 0.0
            return (infeasible, tight, travel + 10.0 * exposure + twist_pen)

        return min(good, key=_score)
    return min(
        misses,
        key=lambda r: (not r.ok, r.collides, r.pos_err_m + 0.1 * math.radians(r.rot_err_deg)),
    )


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Rotation a fraction ``t`` of the way from ``a`` to ``b``."""

    rv = _rotvec(a.T @ b)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-9:
        return a.copy()
    return a @ _axis_angle(rv / angle, angle * t)


def line_path(
    arm: str,
    start_q: Sequence[float],
    target: np.ndarray,
    *,
    offset: float = TCP_OFFSET,
    step_m: float = 0.02,
    step_deg: float = 8.0,
    pos_tol: float = 0.002,
    rot_tol_deg: float = 1.0,
    allow_table_touch: bool = False,
) -> tuple[list[tuple[float, ...]], str | None]:
    """Joint waypoints that move the grasp point along a straight line to ``target``.

    Every waypoint is solved from the previous one, so the arm cannot jump between IK branches
    on the way; returns the waypoints and, when the line cannot be followed, why not.
    """

    start = fk(arm, start_q)
    p0 = start[:3, 3] + start[:3, 0] * offset
    dist = float(np.linalg.norm(target[:3, 3] - p0))
    turn = rotation_error(start[:3, :3], target[:3, :3])
    n = max(1, math.ceil(dist / step_m), math.ceil(turn / step_deg))
    q = tuple(float(v) for v in start_q)
    out = []
    for k in range(1, n + 1):
        t = k / n
        way = np.eye(4)
        way[:3, :3] = slerp(start[:3, :3], target[:3, :3], t)
        way[:3, 3] = p0 + (target[:3, 3] - p0) * t
        res = solve(
            arm,
            way,
            q,
            offset=offset,
            pos_tol=pos_tol,
            rot_tol_deg=rot_tol_deg,
            allow_table_touch=allow_table_touch,
        )
        if not res.ok:
            return out, f"the line leaves the arm's reach {t * dist:.3f} m along it"
        if res.collides:
            return out, f"the line takes the arm into the table {t * dist:.3f} m along it"
        if max(abs(a - b) for a, b in zip(res.q, q, strict=True)) > 0.6:
            return out, "the line passes a singular pose (the wrist would flip)"
        q = res.q
        out.append(q)
    return out, None


__all__ = [
    "ARMS",
    "BASES",
    "IKResult",
    "JOINT_LIMITS",
    "TABLE_X",
    "TABLE_Y",
    "TABLE_Z",
    "TCP_OFFSET",
    "TIP_OFFSET",
    "angles",
    "best_solution",
    "clearance_violations",
    "fk",
    "frames",
    "housing_exposure_penalty",
    "inter_arm_violations",
    "jacobian",
    "line_path",
    "orientation",
    "quat_wxyz",
    "rot_from_quat",
    "rotation_error",
    "slerp",
    "solve",
    "tcp",
    "twist_margin_rad",
]
