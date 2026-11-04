"""Small code-policy API for synchronized motion; task strategy stays in Python."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import ClassVar

from .motion import MotionProfile, MotionWindow, resolve_motion


class CoordinationPause(BaseException):
    """Execution cannot continue locally; return the actual scene to the model."""


@dataclass(frozen=True)
class Pose:
    """World TCP pose: metres, unit quaternion [w,x,y,z], no implicit EEF offset."""

    xyz: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    frame: str = "world"
    point: str = "tcp"

    def __post_init__(self):
        xyz, quat = tuple(float(v) for v in self.xyz), tuple(float(v) for v in self.quat)
        if len(xyz) != 3 or len(quat) != 4 or not all(map(math.isfinite, (*xyz, *quat))):
            raise ValueError("Pose needs three finite metres and four finite quaternion values")
        if self.frame != "world" or self.point != "tcp":
            raise ValueError("Only explicit world-frame TCP poses are supported")
        norm = math.sqrt(sum(v * v for v in quat))
        if abs(norm - 1) > 0.001:
            raise ValueError("quat must be a unit quaternion in [w,x,y,z] order")
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "quat", tuple(v / norm for v in quat))

    def shifted(self, dx=0.0, dy=0.0, dz=0.0) -> Pose:
        """Translate in world axes, preserving orientation; fixed to this snapshot."""
        return replace(self, xyz=tuple(a + b for a, b in zip(self.xyz, (dx, dy, dz), strict=True)))


@dataclass(frozen=True)
class Observation:
    """Both TCP poses at one endpoint; images use image(camera, step=step).

    state_version matches the runtime/Valen action revision; native_revision is
    the host's native action revision. native_steps is the simulator counter,
    step the recorded-image index. Openings do not establish object support.
    """

    id: str
    state_version: int
    native_revision: int
    step: int
    native_steps: int | None
    left: Pose
    right: Pose
    left_opening: float
    right_opening: float


@dataclass(frozen=True)
class ArmMotion:
    """Measured endpoint and error against this arm's final target, not task success."""

    reached: bool
    pose: Pose
    opening: float
    position_error_m: float
    orientation_error_deg: float


@dataclass(frozen=True)
class DualMove:
    """Result of move_both. status is completed, rejected (zero steps), or partial.

    Inspect reached, left/right errors and stop_reason before dependent actions.
    recoverable permits a bounded, explicitly programmed correction; it does not
    certify support or prescribe a retry. Observe the actual state before repair.
    steps_used counts native actions for this call; remaining_steps is its local
    allowance, not the episode budget. checked names actual geometric coverage.
    Critical faults and visual reviews instead unwind the current code block;
    their physical endpoint remains in the tool's coordination feedback.
    tracking reports shared measured-progress decisions and corrective holds.
    Corrections retain the current target across checkpoints, consume the same
    max_steps allowance and never bypass table or relative-pose constraints.
    """

    status: str
    left: ArmMotion
    right: ArmMotion
    steps_used: int
    stop_reason: str | None
    recoverable: bool
    observation: Observation
    remaining_steps: int
    checked: tuple[str, ...]
    trajectory: dict | None = None
    tracking: dict | None = None

    @property
    def reached(self) -> bool:
        """Both final motion targets reached; no claim about objects or task completion."""
        return self.status == "completed" and self.left.reached and self.right.reached


def _observation(raw, robo, step):
    return Observation(
        raw.get("id") or raw["observation_id"],
        robo._action_revision,
        raw["native_revision"],
        step,
        raw.get("native_steps"),
        Pose(**raw["left"]["pose"]),
        Pose(**raw["right"]["pose"]),
        raw["left"]["opening"],
        raw["right"]["opening"],
    )


def _result(raw, robo):
    obs = _observation(raw["observation"], robo, raw["observation"]["step"])
    arms = {
        a: ArmMotion(**{**raw["arms"][a], "pose": Pose(**raw["arms"][a]["pose"])})
        for a in ("left", "right")
    }
    return DualMove(
        raw["status"],
        arms["left"],
        arms["right"],
        raw["executed_steps"],
        raw.get("stop_reason"),
        raw["recoverable"],
        obs,
        raw["remaining_steps"],
        tuple(raw["checked"]),
        raw.get("trajectory"),
        raw.get("tracking"),
    )


class CoordinationMixin:
    """Adapter-neutral Python surface over the host's synchronized executor."""

    COORDINATION_API: ClassVar[bool] = True

    def _legacy_motion_guard(self):
        if getattr(self, "_intent_motion_only", False):
            raise ValueError(
                "This policy session uses execute_motion(intent_id, ...): declare the goal, "
                "interaction, direction/holding constraints, completion and budget. "
                "The executor owns motion parameters; low-level motion is disabled."
            )

    def motion_capabilities(self) -> dict:
        """Read live motion support before choosing constraints.

        speed_control/continuous_trajectory require trajectory_clock with a
        verified simulation timestep. Limits apply to commanded joint splines;
        measured_speed_guarantee is false. force_control, force_feedback,
        gripper_force_control and impedance_control describe separate abilities.
        Current RoboDojo adapters provide no force loop. Legacy hosts without
        timing support retain geometric moves and reject explicit motion/via.
        """
        return self._readonly("motion_capabilities", {})

    def motion_profile(self, velocity=1.0, acceleration=4.0, jerk=40.0) -> MotionProfile:
        """Joint-command limits: rad/s, rad/s², rad/s³, applied to every moving joint.

        Pass as motion to move_to or move_both. Defaults are planning choices,
        not hardware ratings. No TCP speed or force guarantee. Nonpositive,
        nonfinite and boolean values reject before any movement.
        """
        self._legacy_motion_guard()
        return MotionProfile(velocity, acceleration, jerk)

    def pose(self, xyz, quat, *, frame="world", point="tcp") -> Pose:
        """Construct a world TCP Pose: metres, unit quaternion [w,x,y,z]."""
        return Pose(xyz, quat, frame, point)

    def motion_window(self, far="transit", near="approach", *,
                      distance_m=0.02, at="end") -> MotionWindow:
        """Use slower limits near a path endpoint without an intermediate stop.

        far/near are phase names or motion_profile() values; near must be no
        faster in any derivative. at='end' slows the last distance_m of nominal
        TCP travel; at='start' moves away slowly before speeding up. The whole
        IK segment crossing the window is slow. Choose distance from observed
        geometry; this neither detects contact nor provides collision checking.
        For dual motion the distance is a shared path metric, not per-arm clearance.
        Use far='carry', near='precision' for supported placement; verify a
        grasp with a SHORT lift before longer carry. Preview to check the cost.
        """
        self._legacy_motion_guard()
        return MotionWindow(resolve_motion(far), resolve_motion(near), distance_m, at)

    def observe(self) -> Observation:
        """Read both arms together; use the returned step for matching camera images."""
        state = self.state()
        return _observation(self._readonly("motion_observe", {}), self, state.step)

    def move_both(
        self,
        left=None,
        right=None,
        *,
        grippers: dict[str, str] | None = None,
        keep_relative: bool = False,
        max_steps: int = 60,
        timeout_s: float = 120.0,
        position_tolerance_m: float = 0.01,
        orientation_tolerance_deg: float = 5.0,
        max_step_m: float = 0.01,
        max_step_deg: float = 5.0,
        checkpoint_steps: int = 4,
        require_inter_arm_clearance: bool = False,
        motion: MotionProfile | MotionWindow | str | None = None,
        via: list[dict] | None = None,
        preview: bool = False,
    ) -> DualMove | dict:
        """Move both arms on one native action clock, returning their actual outcomes.

        left/right: world TCP xyz (metres, retain current orientation), a Pose,
        or None (hold that arm). grippers maps left/right to 'keep'/'open'/'close';
        omitted means keep the last COMMAND, not the measured gap. Commands take
        effect from the start of this motion, including open/close during travel.
        Choose release timing and support checks in your Python/watch contract.

        Both paths are checked before either moves. No implicit sequential or
        near-pose fallback. keep_relative additionally checks the initial relative
        TCP transform using the supplied position/orientation tolerances, along
        sampled paths and after each native action; this is not force control.
        max_step_m/max_step_deg bound nominal Cartesian spacing (IK allowance:
        1 mm / 0.5 deg). Unsupported requested collision checks reject the move.

        max_steps/timeout_s bound this call, including planning/checkpoint waits.
        Episode and code budgets still apply across calls. checkpoint_steps (1..8)
        bounds segments before Valen checks when enabled. Test result.reached;
        a recoverable rejection/partial result may be handled by a bounded Python
        loop. Critical faults, expired budgets or visual reviews stop this code
        block with current-state feedback. Already executed actions remain done.

        Compose freely with move_to/set_gripper/home. For sequential cooperation,
        write the sequence and branches in Python; no operation object is needed.

        On a timed backend, motion selects joint velocity/acceleration/jerk
        limits (default motion_profile()). via is up to 128 intermediate
        {left: pose_or_xyz, right: pose_or_xyz} mappings, followed by left/right
        final targets. An omitted arm holds its preceding waypoint. xyz keeps
        the preceding orientation. The entire route is checked before dispatch
        and follows one C2 joint spline, with zero endpoint velocity/acceleration.
        Joint blending is checked at every physical substep. No stop is inserted
        at intermediate waypoints; reversals may require zero joint velocity.
        Gripper commands apply from the first tick throughout the route. Split
        calls where observation, contact or release decisions must occur.
        Continuity is in simulation time within this call, not across model/RPC
        waits, interruptions or separate calls. trajectory reports timing and
        planned derivative bounds. Explicit motion/via require backend support.
        motion also accepts 'native' (original waypoint execution, no C2 or
        derivative guarantees), 'transit' (smooth empty-hand travel), 'approach'
        (final approach), 'carry' (held-object transport; default limits), or
        'precision' (placement/insertion). Phase choice is explicit; no force
        or obstacle knowledge is inferred from the grippers.
        preview=True returns a read-only planning dict with accepted, trajectory,
        planned_steps and remaining_episode_steps, without creating an executable
        plan or moving either gripper. Over-budget results retain their timing
        estimate. Execution replans from current state; preview is not permission.
        """
        self._legacy_motion_guard()
        motion = resolve_motion(motion)
        if type(preview) is not bool:
            raise ValueError("preview must be bool")
        if via is not None and (not isinstance(via, list) or len(via) > 128
                                or any(not isinstance(v, dict) or set(v) - {"left", "right"}
                                       for v in via)):
            raise ValueError("via must contain at most 128 left/right waypoint mappings")
        self._guard()
        obs = self.observe()
        previous = {a: getattr(obs, a) for a in ("left", "right")}
        waypoints = []
        for waypoint in via or []:
            converted = {}
            for arm, value in waypoint.items():
                if value is None:
                    continue
                target = value if isinstance(value, Pose) else Pose(value, previous[arm].quat)
                converted[arm] = dict(xyz=target.xyz, quat=target.quat)
                previous[arm] = target
            waypoints.append(converted)
        targets = {}
        for arm, value in (("left", left), ("right", right)):
            if value is not None:
                target = value if isinstance(value, Pose) else Pose(value, previous[arm].quat)
                targets[arm] = dict(xyz=target.xyz, quat=target.quat)
        grips = {"left": "keep", "right": "keep", **(grippers or {})}
        if set(grips) != {"left", "right"} or any(
            v not in ("keep", "open", "close") for v in grips.values()
        ):
            raise ValueError("grippers must map left/right to 'keep', 'open' or 'close'")
        raw = self._readonly(
            "motion_plan",
            dict(
                observation_id=obs.id,
                targets=targets,
                grippers=grips,
                keep_relative=keep_relative,
                max_steps=max_steps,
                timeout_s=timeout_s,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_deg=orientation_tolerance_deg,
                max_step_m=max_step_m,
                max_step_deg=max_step_deg,
                checkpoint_steps=checkpoint_steps,
                require_inter_arm_clearance=require_inter_arm_clearance,
                motion=motion.payload() if motion is not None else None,
                via=waypoints,
                **({"preview": True} if preview else {}),
            ),
        )

        if preview:
            return raw

        def record(payload):
            payload["observation"]["step"] = self._last_state.step
            payload["observation"]["state_version"] = self._action_revision
            self._coordination_feedback = payload
            return payload

        if not raw["accepted"]:
            record(raw)
            if not raw["recoverable"]:
                raise CoordinationPause(raw["stop_reason"])
            return _result(raw, self)
        plan_id = raw["id"]
        try:
            while True:
                raw = self._stateful("motion_execute", {"plan_id": plan_id}, record)
                if raw["status"] == "checkpoint":
                    continue
                if not raw["recoverable"] and raw["status"] != "completed":
                    raise CoordinationPause(raw["stop_reason"])
                return _result(raw, self)
        except BaseException as exc:
            previous = self._coordination_feedback or {}
            stopped = self._readonly("motion_stop", {"plan_id": plan_id})
            if isinstance(exc, CoordinationPause):
                stopped.update(
                    status=previous.get("status", "paused"),
                    stop_reason=previous.get("stop_reason", str(exc)),
                )
            else:
                stopped.update(status="paused", interruption=f"{type(exc).__name__}: {exc}")
            record(stopped)
            raise
        finally:
            self._readonly("motion_discard", {"plan_id": plan_id})


RESULT_TYPES = (Pose, Observation, ArmMotion, DualMove, MotionProfile, MotionWindow)
