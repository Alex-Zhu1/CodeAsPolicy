"""Explicit TCP-local sphere envelopes over public pose observations.

No environment or object truth is read here. Callers supply spheres enclosing
the finger, wrist, or held-object geometry they actually know. Unrepresented
links, changing grasps, and unknown objects remain outside this check's scope.
Fixed envelopes must enclose the entire planned gripper opening range; this
module does not infer changing finger geometry from the gripper command.

A swept check certifies linear TCP translation plus shortest-path SLERP between
the supplied poses. Attached sphere centers follow curved paths: their chord
error is bounded by ``norm(local_center) * rotation_angle**2 / 8`` on each
interval. Adaptive subdivision either proves the required clearance, finds a
violation, or rejects an unresolved interval conservatively. This is not a
certificate of the unsampled FK curve or arbitrary physical servo motion.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from . import robodojo_kinematics as kin

_ARMS = ("left", "right")
_MAX_DEPTH = 20
_DEFAULT_TOLERANCE = 1e-4


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return float(value)


def _xyz(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError(f"{name} must contain three finite numbers")
    return [float(v) for v in value]


def _obstacles(constraints):
    obstacles = constraints.get("keep_out", [])
    if not isinstance(obstacles, (list, tuple)) or len(obstacles) > 16:
        raise ValueError("keep_out must contain at most 16 obstacles")
    out = []
    for index, item in enumerate(obstacles):
        if not isinstance(item, dict):
            raise ValueError("keep_out obstacles must be dictionaries")
        xy_only = item.get("xy_only", False)
        if type(xy_only) is not bool:
            raise ValueError("keep_out xy_only must be bool")
        out.append(dict(
            name=f"keep_out:{index}", xyz=_xyz(item.get("xyz"), "keep_out xyz"),
            radius_m=_number(item.get("radius_m"), "keep_out radius_m", positive=True),
            xy_only=xy_only,
        ))
    return out


def validate_geometry(constraints):
    """Return normalized ``constraints['geometry']``, or None when absent.

    Schema: ``arms`` maps left/right to up to 32 named spheres, with TCP-local
    ``xyz``, positive ``radius_m``, optional non-negative ``uncertainty_m`` and
    optional textual ``source``. ``clearance_m`` is the minimum surface gap.
    Each allowed contact specifies exactly two IDs in ``pair`` and an explicit
    ``max_penetration_m``. IDs are ``left:name``, ``right:name``, or ``keep_out:N``.
    Only represented cross-arm or sphere/obstacle pairs can receive exceptions.
    """
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a dict")
    if "geometry" not in constraints:
        return None
    geometry = constraints["geometry"]
    allowed = {"arms", "clearance_m", "allowed_contacts", "sweep_tolerance_m"}
    if not isinstance(geometry, dict) or set(geometry) - allowed:
        raise ValueError(f"geometry supports only {sorted(allowed)}")
    arms = geometry.get("arms")
    if not isinstance(arms, dict) or not arms or set(arms) - set(_ARMS):
        raise ValueError("geometry arms must map left/right to nonempty sphere lists")
    clean, radii = {}, {}
    for arm, spheres in arms.items():
        if not isinstance(spheres, (list, tuple)) or not 1 <= len(spheres) <= 32:
            raise ValueError("each geometry arm needs 1..32 spheres")
        clean[arm] = []
        for sphere in spheres:
            allowed_sphere = {"name", "xyz", "radius_m", "uncertainty_m", "source"}
            if not isinstance(sphere, dict) or set(sphere) - allowed_sphere:
                raise ValueError(f"geometry spheres support only {sorted(allowed_sphere)}")
            name = sphere.get("name")
            if (not isinstance(name, str) or not name.strip() or ":" in name
                    or len(name) > 80):
                raise ValueError("sphere name must be nonempty text without ':' (at most 80 chars)")
            identity = f"{arm}:{name}"
            if identity in radii:
                raise ValueError(f"duplicate sphere name: {identity}")
            item = dict(name=name, xyz=_xyz(sphere.get("xyz"), "sphere xyz"),
                        radius_m=_number(sphere.get("radius_m"), "sphere radius_m", positive=True),
                        uncertainty_m=_number(sphere.get("uncertainty_m", 0.), "uncertainty_m"))
            if "source" in sphere:
                if not isinstance(sphere["source"], str) or not sphere["source"].strip():
                    raise ValueError("sphere source must be nonempty text")
                item["source"] = sphere["source"]
            radii[identity] = item["radius_m"] + item["uncertainty_m"]
            if not math.isfinite(radii[identity]):
                raise ValueError("effective sphere radius must be finite")
            clean[arm].append(item)
    obstacles = _obstacles(constraints)
    radii.update({item["name"]: item["radius_m"] for item in obstacles})
    contacts = geometry.get("allowed_contacts", [])
    if not isinstance(contacts, (list, tuple)) or len(contacts) > 256:
        raise ValueError("allowed_contacts must contain at most 256 explicit pairs")
    cleaned_contacts, seen = [], set()
    for contact in contacts:
        if not isinstance(contact, dict) or set(contact) != {"pair", "max_penetration_m"}:
            raise ValueError("each allowed contact needs only pair and max_penetration_m")
        pair = contact["pair"]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(not isinstance(name, str) or name not in radii for name in pair)):
            raise ValueError("allowed contact pair must name two represented sphere/obstacle IDs")
        groups = [name.split(":", 1)[0] for name in pair]
        if groups[0] == groups[1]:
            raise ValueError("allowed contacts must be cross-arm or sphere/obstacle pairs")
        pair = tuple(sorted(pair))
        if pair in seen:
            raise ValueError("duplicate allowed contact pair")
        seen.add(pair)
        penetration = _number(contact["max_penetration_m"], "max_penetration_m")
        if penetration > radii[pair[0]] + radii[pair[1]]:
            raise ValueError("max_penetration_m cannot exceed the pair's summed radii")
        cleaned_contacts.append(dict(pair=list(pair), max_penetration_m=penetration))
    return dict(
        arms=clean,
        clearance_m=_number(geometry.get("clearance_m", 0.), "clearance_m"),
        allowed_contacts=cleaned_contacts,
        sweep_tolerance_m=_number(geometry.get("sweep_tolerance_m", _DEFAULT_TOLERANCE),
                                  "sweep_tolerance_m", positive=True),
    )


def geometry_coverage(constraints):
    """Describe modeled geometry honestly, including missing arm coverage."""
    geometry = validate_geometry(constraints)
    arms = geometry["arms"] if geometry else {}
    return dict(
        geometry_supplied=geometry is not None,
        scope="caller_supplied_tcp_spheres_only" if geometry else "no_volume_geometry",
        represented_arms=sorted(arms),
        sphere_counts={arm: len(spheres) for arm, spheres in arms.items()},
        obstacle_checks=bool(geometry and constraints.get("keep_out")),
        inter_arm_checks=all(arm in arms for arm in _ARMS),
        same_arm_checks=False,
        full_robot_collision=False,
        unrepresented_geometry="not_checked",
        sweep_model="linear_tcp_translation_shortest_slerp" if geometry else None,
        actual_motion_between_observations="not_certified",
        gripper_shape="fixed_supplied_envelope; opening_range_must_be_enclosed",
        allowed_contact_pairs=[item["pair"] for item in geometry["allowed_contacts"]]
        if geometry else [],
    )


def _pose(value, arm):
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arm} pose must be a finite rigid 4x4 transform") from exc
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0.)
            or not math.isclose(float(np.linalg.det(pose[:3, :3])), 1., abs_tol=1e-6)):
        raise ValueError(f"{arm} pose must be a finite rigid 4x4 transform")
    return pose


def _segment_distance(start, end):
    difference = end - start
    squared = float(np.dot(difference, difference))
    fraction = float(np.clip(-np.dot(start, difference) / squared, 0., 1.)) if squared else 0.
    distance = float(np.linalg.norm(start + fraction * difference))
    if not math.isfinite(distance):
        raise ValueError("geometry numeric range exceeded")
    return distance


def geometry_violation(poses, constraints, previous_poses=None):
    """Return first volume violation/uncertified-clearance diagnostic, or None.

    ``previous_poses`` enables the explicitly documented interpolated-pose sweep;
    without it, only the current spheres are checked. Contact permissions apply
    only to these named spheres, never to other pairs. The host uses supplied
    envelopes in place of TCP-point keep-out checks for represented arms.
    A None result covers only the pairs described by ``geometry_coverage``.
    """
    geometry = validate_geometry(constraints)
    if geometry is None:
        return None
    if not isinstance(poses, dict) or (previous_poses is not None
                                       and not isinstance(previous_poses, dict)):
        raise ValueError("geometry poses must map arm names to rigid 4x4 transforms")
    coverage = geometry_coverage(constraints)
    previous = poses if previous_poses is None else previous_poses
    transforms = {}
    for arm in geometry["arms"]:
        if arm not in poses or arm not in previous:
            raise ValueError(f"geometry requires current and previous {arm} poses")
        transforms[arm] = (_pose(previous[arm], arm), _pose(poses[arm], arm))
    rotations, angles = {}, {}
    for arm, (start, end) in transforms.items():
        # Quaternion extraction retains the relative signs of a mixed-axis
        # exact half-turn, where the rotation matrix's skew part is zero.
        quaternion = np.asarray(kin.quat_wxyz(start[:3, :3].T @ end[:3, :3]))
        sine = float(np.linalg.norm(quaternion[1:]))
        angle = 2. * math.atan2(sine, float(quaternion[0]))
        rotations[arm] = (quaternion[1:] / sine if sine else np.zeros(3), angle)
        angles[arm] = angle
    spheres = {}
    for arm, specs in geometry["arms"].items():
        spheres[arm] = [dict(spec, arm=arm, identity=f"{arm}:{spec['name']}") for spec in specs]
    obstacles = [dict(item, arm=None, identity=item["name"], uncertainty_m=0.)
                 for item in _obstacles(constraints)]
    pairs = [(sphere, obstacle) for arm in spheres
             for sphere, obstacle in product(spheres[arm], obstacles)]
    pairs.extend(product(spheres.get("left", []), spheres.get("right", [])))
    permissions = {tuple(item["pair"]): item["max_penetration_m"]
                   for item in geometry["allowed_contacts"]}

    def center(sphere, fraction):
        local = np.asarray(sphere["xyz"], float)
        if sphere["arm"] is None:
            return local
        start, end = transforms[sphere["arm"]]
        position = start[:3, 3] + fraction * (end[:3, 3] - start[:3, 3])
        axis, angle = rotations[sphere["arm"]]
        half_angle = angle * fraction / 2.
        rotation = start[:3, :3] @ kin.rot_from_quat([
            math.cos(half_angle), *(axis * math.sin(half_angle)),
        ])
        return position + rotation @ local

    for first, second in pairs:
        pair = tuple(sorted((first["identity"], second["identity"])))
        minimum_gap = -permissions[pair] if pair in permissions else geometry["clearance_m"]
        radii = sum(item["radius_m"] + item["uncertainty_m"] for item in (first, second))
        dimensions = 2 if second.get("xy_only") else 3
        # For f(t)=R(t)c, ||f''|| <= theta**2*||c||. Linear interpolation
        # error is at most max||f''|| * interval_width**2 / 8. Translation
        # is exactly linear; relative sphere motion adds both error bounds.
        curvature = sum(angles[item["arm"]] ** 2 * math.hypot(*item["xyz"])
                        for item in (first, second) if item["arm"] is not None)
        if not math.isfinite(radii) or not math.isfinite(curvature):
            raise ValueError("geometry numeric range exceeded")

        def relative(fraction):
            value = (center(first, fraction) - center(second, fraction))[:dimensions]
            if not np.isfinite(value).all():
                raise ValueError("geometry numeric range exceeded")
            return value

        intervals = [(0., 1., relative(0.), relative(1.), 0)]
        while intervals:
            lo, hi, start, end, depth = intervals.pop()
            error_bound = curvature * (hi - lo) ** 2 / 8.
            lower_gap = _segment_distance(start, end) - error_bound - radii
            if lower_gap >= minimum_gap:
                continue
            midpoint = (lo + hi) / 2.
            middle = relative(midpoint)
            observed_gap = min(float(np.linalg.norm(point)) for point in (start, middle, end)) - radii
            exact_line = error_bound == 0.
            collision = observed_gap < minimum_gap or (exact_line and lower_gap < minimum_gap)
            unresolved = error_bound <= geometry["sweep_tolerance_m"] or depth >= _MAX_DEPTH
            if collision or unresolved:
                gap = min(observed_gap, lower_gap) if exact_line else (
                    observed_gap if collision else lower_gap)
                return dict(
                    name="geometry:" + "|".join(pair), kind="constraint",
                    metric="geometry_clearance_m", value=gap, bound=minimum_gap,
                    delta=minimum_gap - gap,
                    remedy="Increase separation or revise the represented geometry/contact pair "
                           "using fresh public evidence before replanning.",
                    reason="geometry_collision" if collision else "geometry_clearance_unverified",
                    pair=list(pair), interval=[lo, hi],
                    value_basis="interpolated_pose_geometry" if collision else "conservative_lower_bound",
                    chord_error_bound_m=error_bound, coverage=coverage,
                )
            intervals.append((midpoint, hi, middle, end, depth + 1))
            intervals.append((lo, midpoint, start, middle, depth + 1))
    return None
