"""C2 joint splines, retimed against analytic Bernstein derivative bounds.

The client owns the simulation clock. Each native action carries one block of
position/velocity samples; the client replaces its arm-only 80/20 interpolation.
No wall-clock sleeps, force estimates, or continuity across separate calls.
"""

from __future__ import annotations

import math

import numpy as np

from ..motion import PHASES, MotionProfile, MotionWindow, motion_from_payload
from . import robodojo_kinematics as kin

TRAJECTORY_KEY = "__pyrualean_trajectory__"
BACKEND = "robodojo_joint_substeps_v1"


def backend(primitives, *, status=None):
    if status is None:
        getter = getattr(primitives.env, "get_status", None)
        status = getter() if callable(getter) else {}
    raw = (status or {}).get("motion_backend") or {}
    if not isinstance(raw, dict) or raw.get("schema") != BACKEND:
        return None
    dt, n = raw.get("physics_dt_s"), raw.get("substeps")
    if (type(dt) not in (int, float) or not math.isfinite(dt) or not 0 < dt <= 1
            or type(n) is not int or not 1 <= n <= 1000
            or raw.get("time_domain") != "simulation"
            or raw.get("position_velocity_targets") is not True):
        return None
    return dict(schema=BACKEND, physics_dt_s=dt, substeps=n,
                time_domain="simulation", position_velocity_targets=True)


def capabilities(primitives, *, status=None):
    clock = backend(primitives, status=status)
    return dict(
        speed_control=clock is not None,
        speed_control_scope="planned_joint_position_velocity_targets" if clock else None,
        continuous_trajectory=clock is not None,
        trajectory_clock=clock,
        trajectory_defaults=MotionProfile().payload(),
        motion_phases={name: dict(values) for name, values in PHASES.items()},
        motion_preview=True,
        motion_window=True,
        native_motion=True,
        bounded_grasp=True,
        grasp_visual_confirmation_required=True,
        trajectory_units=dict(velocity="rad/s", acceleration="rad/s^2", jerk="rad/s^3"),
        trajectory_continuity="C2 within one call; rest at ends; interruptions end continuity",
        measured_speed_guarantee=False,
        force_control=False,
        force_feedback=False,
        gripper_force_control=False,
        impedance_control=False,
        force_control_reason="current action/observation adapter exposes no effort or force loop",
    )


def _evaluate(controls, u):
    """Evaluate Bernstein polynomials, vectorized over samples and joints."""
    degree = controls.shape[1] - 1
    weights = np.stack([
        math.comb(degree, i) * u**i * (1 - u)**(degree - i)
        for i in range(degree + 1)
    ], axis=1)
    return np.einsum("ni,nij->nj", weights, controls)


def hold(joints, clock):
    """Hold a completed spline without reintroducing native measured-start interpolation."""
    values = [*joints["left"], *joints["right"]]
    return dict(schema=BACKEND, physics_dt_s=clock["physics_dt_s"],
                positions=[list(values) for _ in range(clock["substeps"])],
                velocities=[[0.] * 12 for _ in range(clock["substeps"])])


class TrajectoryBudgetError(ValueError):
    """Keep the estimate when a feasible trajectory does not fit the call budget."""

    def __init__(self, estimate):
        super().__init__("timed_path_exceeds_native_step_budget")
        self.trajectory = estimate


def _simplify(q, preserve=(), tolerance=1e-4):
    """Bound joint-path simplification error, including all removed IK samples.

    Exact collinearity makes timing depend on tiny IK residuals. A 0.0001 rad
    per-joint tolerance removes that chatter; Cartesian/relative-pose/table
    checks still validate every emitted physics substep. Keep limit boundaries
    and turns larger than the tolerance. Use a stack to avoid recursion limits
    on long, curved paths.
    """
    kept = {0, len(q) - 1, *preserve}
    boundaries = sorted(kept)
    pending = list(zip(boundaries[:-1], boundaries[1:]))
    while pending:
        a, b = pending.pop()
        if b - a <= 1:
            continue
        chord = q[b] - q[a]
        fraction = np.clip((q[a+1:b] - q[a]) @ chord
                           / max(np.dot(chord, chord), 1e-30), 0., 1.)
        errors = np.max(abs(q[a+1:b] - q[a] - fraction[:, None] * chord), axis=1)
        i = a + 1 + int(np.argmax(errors))
        reversals = np.flatnonzero(np.diff(np.r_[0., fraction, 1.]) < -1e-12)
        if reversals.size:
            # Collinear backtracking is still a real path, even when its
            # interior extrema lie between the overall endpoints.
            i = a + int(reversals[0])
        if reversals.size or errors[i-a-1] > tolerance:
            kept.add(i)
            pending.extend(((a, i), (i, b)))
    return np.asarray(sorted(kept))


def _controls(q, durations, blend=None):
    """Shared knot derivatives give C2 joins without requiring interior acceleration=0."""
    delta = np.diff(q, axis=0)
    slopes = delta / durations[:, None]
    velocity, acceleration = np.zeros_like(q), np.zeros_like(q)
    if len(q) > 2:
        a, b = slopes[:-1], slopes[1:]
        velocity[1:-1] = np.where(a * b > 0, np.sign(a) * np.minimum(abs(a), abs(b)), 0.)
        acceleration[1:-1] = 2 * (b - a) / (durations[:-1] + durations[1:])[:, None]
    # A constant joint segment must have zero derivatives at both ends. Set
    # this exactly instead of repeatedly shrinking an impossible nonzero blend.
    flat = abs(delta) < 1e-12
    acceleration[:-1][flat] = 0.
    acceleration[1:][flat] = 0.
    if blend is not None:
        velocity *= blend[:, None]
        acceleration *= blend[:, None]

    def build():
        h = durations[:, None]
        return np.stack((q[:-1], q[:-1] + velocity[:-1] * h / 5,
            q[:-1] + 2 * velocity[:-1] * h / 5 + acceleration[:-1] * h**2 / 20,
            q[1:] - 2 * velocity[1:] * h / 5 + acceleration[1:] * h**2 / 20,
            q[1:] - velocity[1:] * h / 5, q[1:]), axis=1)

    # Limit shared accelerations until each joint's Bernstein controls are
    # monotone within its segment. This prevents overshoot, including reversals.
    for _ in range(20):
        controls = build()
        differences = np.diff(controls, axis=1)
        bad = np.any(differences * np.sign(delta)[:, None, :] < -1e-12, axis=1)
        bad |= (abs(delta) < 1e-12) & np.any(abs(differences) > 1e-12, axis=1)
        if not bad.any():
            return controls
        nodes = np.zeros_like(acceleration, dtype=bool)
        nodes[:-1] |= bad
        nodes[1:] |= bad
        acceleration[nodes] *= .5
    # Zero acceleration is a conservative, still C2 fallback for degenerate knots.
    acceleration[:] = 0.
    return build()


def _bounds(controls, durations):
    """Certified derivative hulls, tightened by de Casteljau subdivision."""
    derivatives, bounds = [], []
    d = controls
    for degree in (5, 4, 3):
        d = degree * np.diff(d, axis=1) / durations[:, None, None]
        derivatives.append(d)
        pieces = d[:, None]
        for _ in range(3):
            work = pieces
            left, right = [work[:, :, 0]], [work[:, :, -1]]
            for _ in range(work.shape[2] - 1):
                work = (work[:, :, :-1] + work[:, :, 1:]) / 2
                left.append(work[:, :, 0])
                right.append(work[:, :, -1])
            pieces = np.concatenate((np.stack(left, axis=2),
                                     np.stack(right[::-1], axis=2)), axis=1)
        bounds.append(np.max(abs(pieces), axis=(1, 2, 3)))
    return derivatives, np.asarray(bounds).T


def _path_limits(q, motion):
    """Conservative per-IK-segment limits in nominal TCP path distance.

    A crossing segment is slow in full. Neither smoothing nor simplification
    may erase its limit boundary. Pure orientation moves remain slow throughout.
    """
    names = ("velocity", "acceleration", "jerk")
    if isinstance(motion, MotionProfile):
        return np.tile([getattr(motion, k) for k in names], (len(q)-1, 1)), None, None
    xyz = np.array([[kin.tcp(kin.fk(a, row[sl])) for a, sl in
                    (("left", slice(0, 6)), ("right", slice(6, 12)))] for row in q])
    travel = np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=2), axis=1)
    progress = np.r_[0., np.cumsum(travel)]
    distance = (progress[:-1] if motion.at == "start" else progress[-1]-progress[1:])
    near = distance < motion.distance_m + 1e-10
    limits = np.array([[getattr(motion.near if slow else motion.far, k) for k in names]
                       for slow in near])
    return limits, near, float(progress[-1])


def _retime(q, profile, native_dt, blend=None, limits=None):
    """Lengthen only constrained segments; neighboring knot derivatives are refitted."""
    if limits is None:
        limits = np.tile([profile.velocity, profile.acceleration, profile.jerk], (len(q)-1, 1))
    limits = np.array(limits, dtype=float, copy=True)
    limits[:, 0] = np.minimum(limits[:, 0], .15 / native_dt)
    durations = np.maximum(np.max(abs(np.diff(q, axis=0)), axis=1) / limits[:, 0], native_dt)
    for _ in range(32):
        controls = _controls(q, durations, blend)
        _, bounds = _bounds(controls, durations)
        factors = np.max((bounds / limits) ** np.array([1., .5, 1/3]), axis=1)
        if np.max(factors) <= 1 + 1e-9:
            break
        durations *= np.maximum(1., factors * 1.001)
    else:
        # A uniform final dilation is exact for these homogeneous derivative
        # rules. Never return an unconverged local fit with unverified bounds.
        durations *= max(1., float(np.max(factors))) * 1.001
    count = max(1, math.ceil(float(sum(durations)) / native_dt))
    durations *= count * native_dt / sum(durations)
    controls = _controls(q, durations, blend)
    derivatives, bounds = _bounds(controls, durations)
    if np.any(bounds > limits * (1 + 1e-8)):
        raise ValueError("trajectory retiming did not satisfy derivative limits")
    return controls, derivatives, durations, bounds, count


def parameterize(start, path, clock, profile=None, *, max_actions=10000, check=None,
                 line_tolerance_m=None, _blend=None, _attempt=0,
                 _simplify_tolerance=1e-4):
    """Return native blocks and an auditable timing summary, before any dispatch.

    Monotone joint controls keep each spline in its adjacent joint ranges.
    Shared velocity/acceleration gives C2 continuity. Segment-local retiming and
    subdivided Bernstein bounds certify derivatives between emitted samples.
    Geometry must still be checked on every emitted substep by the caller.
    """
    profile = motion_from_payload(profile if profile is not None else {})
    dt, block = clock["physics_dt_s"], clock["substeps"]
    native_dt = dt * block
    q = np.asarray([[*p["left"], *p["right"]] for p in [start, *path]], dtype=float)
    if q.ndim != 2 or q.shape[1] != 12 or not np.isfinite(q).all():
        raise ValueError("trajectory needs finite six-joint targets for both arms")
    q = q[np.r_[True, np.max(np.abs(np.diff(q, axis=0)), axis=1) > 1e-12]]
    if len(q) == 1:
        q = np.repeat(q, 2, axis=0)
    original = q
    original_limits, near, travel = _path_limits(q, profile)
    boundaries = set(np.flatnonzero(np.any(np.diff(original_limits, axis=0), axis=1)) + 1)
    knots = _simplify(q, preserve=boundaries, tolerance=_simplify_tolerance)
    q = q[knots]
    limits = np.array([original_limits[a:b].min(axis=0) for a, b in zip(knots[:-1], knots[1:])])
    p, derivatives, durations, bounds, count = _retime(q, profile, native_dt, _blend, limits)
    maxima = np.max(bounds, axis=0)
    ratios = np.max(bounds / limits, axis=0)
    estimate = dict(profile=profile.payload(), time_domain="simulation",
        physics_dt_s=dt, native_dt_s=native_dt, planned_duration_s=count * native_dt,
        planned_steps=count, continuity="C2", endpoint_velocity="zero",
        derivative_bounds=dict(zip(("velocity", "acceleration", "jerk"), maxima.tolist(), strict=True)),
        limiting_factor=("velocity", "acceleration", "jerk")[int(np.argmax(ratios))],
        budget_steps=max_actions, fits_budget=count <= max_actions,
        input_segments=len(original)-1, timed_segments=len(q)-1,
        timing="local_derivative_bounds", geometry_refits=_attempt,
        joint_path_simplification_tolerance_rad=_simplify_tolerance)
    if isinstance(profile, MotionWindow):
        slow = near[knots[:-1]]
        estimate["window"] = dict(at=profile.at, distance_m=profile.distance_m,
            nominal_path_length_m=travel, metric="sum_max_arm_tcp_displacement",
            crossing_segment="near_limits_for_entire_segment", regions={})
        for name, mask in (("near", slow), ("far", ~slow)):
            if np.any(mask):
                estimate["window"]["regions"][name] = dict(
                    profile=getattr(profile, name).payload(), segments=int(mask.sum()),
                    planned_duration_s=float(durations[mask].sum()),
                    derivative_bounds=dict(zip(("velocity", "acceleration", "jerk"),
                                               bounds[mask].max(axis=0).tolist(), strict=True)))
    if count > max_actions or count * block > 200000:
        raise TrajectoryBudgetError(estimate)
    times = np.cumsum(durations)
    sample_t = np.arange(1, count * block + 1) * dt
    indices = np.minimum(np.searchsorted(times, sample_t, side="left"), len(durations) - 1)
    begins = np.r_[0.0, times[:-1]]
    u = np.clip((sample_t - begins[indices]) / durations[indices], 0, 1)
    positions = _evaluate(p[indices], u)
    velocities = _evaluate(derivatives[0][indices], u)
    positions[-1], velocities[-1] = q[-1], 0.0
    references = {
        arm: [kin.fk(arm, values[sl]) for values in original]
        for arm, sl in (("left", slice(0, 6)), ("right", slice(6, 12)))
    } if line_tolerance_m is not None else {}
    original_s = np.r_[0., np.cumsum(np.max(abs(np.diff(original, axis=0)), axis=1))]
    # Validate actual substeps, including blends between the original IK knots.
    def validate(i, values):
        if line_tolerance_m is not None:
            a, b = knots[indices[i]:indices[i]+2]
            chord = original[b] - original[a]
            fraction = np.clip(np.dot(values-original[a], chord)
                               / max(np.dot(chord, chord), 1e-30), 0, 1)
            distance = original_s[a] + fraction * (original_s[b] - original_s[a])
            reference = min(b-1, max(a, np.searchsorted(original_s, distance, side="left")-1))
            for arm, sl in (("left", slice(0, 6)), ("right", slice(6, 12))):
                first, last = references[arm][reference:reference + 2]
                pose = kin.fk(arm, values[sl])
                begin, end, now = kin.tcp(first), kin.tcp(last), kin.tcp(pose)
                direction = end - begin
                fraction = np.clip(np.dot(now - begin, direction)
                                   / max(np.dot(direction, direction), 1e-12), 0, 1)
                if np.linalg.norm(now - begin - fraction * direction) > line_tolerance_m:
                    raise ValueError(f"{arm}: timed blending leaves Cartesian path tolerance")
                angle = kin.rotation_error(first[:3, :3], last[:3, :3])
                turn_fraction = (kin.rotation_error(first[:3, :3], pose[:3, :3]) / angle
                                 if angle > 1e-6 else 0.)
                expected_rotation = kin.slerp(first[:3, :3], last[:3, :3],
                                             np.clip(turn_fraction, 0, 1))
                if kin.rotation_error(pose[:3, :3], expected_rotation) > 1.0:
                    raise ValueError(f"{arm}: timed blending leaves orientation path tolerance")
        if check is not None:
            check({"left": values[:6], "right": values[6:]})

    bad_segments = set()
    first_error = None
    for i, values in enumerate(positions):
        try:
            validate(i, values)
        except ValueError as exc:
            # Refit only blending errors. Cancellation, deadlines, table/joint
            # violations and requested relative-pose constraints still reject.
            if not any(text in str(exc) for text in (
                "smoothed trajectory leaves Cartesian line tolerance",
                "timed blending leaves Cartesian path tolerance",
                "timed blending leaves orientation path tolerance",
            )):
                raise
            bad_segments.add(indices[i])
            first_error = first_error or exc
    if bad_segments:
        if _attempt >= 6:
            if _simplify_tolerance > 1e-10:
                # Preserve the original path if even a tiny simplification
                # prevents geometry validation; never widen its tolerance.
                return parameterize(start, path, clock, profile.payload(),
                                    max_actions=max_actions, check=check,
                                    line_tolerance_m=line_tolerance_m,
                                    _simplify_tolerance=1e-10)
            raise first_error
        blend = np.ones(len(q)) if _blend is None else _blend.copy()
        for segment in bad_segments:
            blend[segment:segment+2] *= .5
            if _attempt >= 3:
                blend[segment:segment+2] = 0.
        # Slow shared knot derivatives locally, then recompute time and certify
        # the entire curve again. Never relax the caller's geometry tolerance.
        return parameterize(start, path, clock, profile.payload(), max_actions=max_actions,
                            check=check, line_tolerance_m=line_tolerance_m,
                            _blend=blend, _attempt=_attempt+1,
                            _simplify_tolerance=_simplify_tolerance)
    ways = []
    for i in range(count):
        sl = slice(i * block, (i + 1) * block)
        points, speeds = positions[sl], velocities[sl]
        ways.append(dict(
            joints={"left": points[-1, :6].tolist(), "right": points[-1, 6:].tolist()},
            trajectory=dict(schema=BACKEND, physics_dt_s=dt,
                            positions=points.tolist(), velocities=speeds.tolist()),
        ))
    return ways, estimate


def check_joints(joints, *, start=None):
    """Allow tiny measured start errors to hold/recover, never to grow further out."""
    for arm in kin.ARMS:
        if kin.clearance_violations(arm, joints[arm]):
            raise ValueError(f"{arm}: timed trajectory intersects table")
        for i, (value, (lo, hi)) in enumerate(zip(joints[arm], kin.JOINT_LIMITS, strict=True)):
            initial = start[arm][i] if start is not None else value
            if start is not None and lo - .002 <= initial <= hi + .002:
                lo, hi = min(lo, initial), max(hi, initial)
            if not lo - 1e-9 <= value <= hi + 1e-9:
                raise ValueError(f"{arm}: timed trajectory exceeds joint limits")
