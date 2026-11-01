"""Small code-policy API for synchronized motion; task strategy stays in Python."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import ClassVar


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
    )


class CoordinationMixin:
    """Adapter-neutral Python surface over the host's synchronized executor."""

    COORDINATION_API: ClassVar[bool] = True

    def motion_capabilities(self) -> dict:
        """Return actual frame, synchronization, feedback and collision-check coverage."""
        return self._readonly("motion_capabilities", {})

    def pose(self, xyz, quat, *, frame="world", point="tcp") -> Pose:
        """Construct a world TCP Pose: metres, unit quaternion [w,x,y,z]."""
        return Pose(xyz, quat, frame, point)

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
    ) -> DualMove:
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
        """
        self._guard()
        obs = self.observe()
        targets = {}
        for arm, value in (("left", left), ("right", right)):
            if value is not None:
                target = value if isinstance(value, Pose) else Pose(value, getattr(obs, arm).quat)
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
            ),
        )

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


RESULT_TYPES = (Pose, Observation, ArmMotion, DualMove)
