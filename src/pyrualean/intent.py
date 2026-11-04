"""Policy-facing motion intent; timing and continuation belong to the executor."""

from __future__ import annotations

from dataclasses import asdict
from typing import ClassVar


def _argument(value):
    """Normalize public pose/array inputs without coercing booleans or NaNs."""
    from dataclasses import is_dataclass

    if is_dataclass(value):
        value = asdict(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _argument(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_argument(v) for v in value]
    return value


class IntentMixin:
    INTENT_API: ClassVar[bool] = True
    LEGACY_MOTION_METHODS: ClassVar[frozenset[str]] = frozenset({
        "move_to", "move_both", "motion_profile", "motion_window",
    })

    def execute_motion(
        self, intent_id: str, *, left=None, right=None, interaction="free",
        directions=None, holding=None, completion=None, max_steps=None,
        reserve_steps=0, keep_relative=False, preview=False,
    ) -> dict:
        """Execute an immutable, bounded motion intent for one or both arms.

        intent_id identifies this intent for this live environment session. Use
        the same ID to retrieve/continue it; changed arguments with that ID are
        rejected. A completed ID never moves again. Do not wrap this method in
        once(): the executor owns its identity and remaining motion.

        left/right are absolute world TCP xyz (keep initial orientation), Pose,
        or {"xyz": [...], "angles": [pitch, yaw, roll]} in move_to's degrees.
        Omitted arms hold their initial poses. Each route is a Cartesian line;
        the executor never chooses a new grasp, orientation or detour.
        interaction describes intent: 'free' (observed clear route), 'approach'
        (contact only near the endpoint), or 'contact' (interaction throughout).
        directions optionally maps arms to allowed positive world translation
        axes, e.g. {'right': [0, 0, -1]}. Goals and measured motion must remain
        on that axis within the completion tolerance. No sideways search.

        holding maps arms to 'keep', 'empty', or 'retain'. All retain the current
        gripper command: this method never opens, closes or regrips. 'empty' is
        your observed empty-hand assertion; 'retain' additionally requires a
        recorded close command and obstructed closure before/after each action.
        This check cannot prove object identity, stability or detect all slips.

        completion optionally supplies position_m, orientation_deg and verify
        (a short visual/semantic condition). Pose tolerance applies at arrival;
        the executor separately bounds tracking error during travel according
        to interaction. Direction and relative-pose constraints stay strict.
        verify is returned as needs_observation, never silently certified from
        TCP arrival. Without it, reached still means motion completion only.
        The executor chooses spacing, speed and at most two arrival corrections.
        By default all remaining native actions are available; there is no
        required future-task or return reserve. Optional max_steps caps the
        whole intent across continuations; reserve_steps is explicit only.
        If the whole route does not fit, the executor attempts a shorter part
        with a resting endpoint and returns paused/segment_complete. This is
        progress, not arrival: inspect and resume the SAME ID. Cost estimates
        alone are not a reason to abandon a task. Constraints and speed stay
        unchanged. The native episode limit remains a hard limit.

        Returns status, reached, actual poses, steps_used (this dispatch),
        intent_steps_used (cumulative), remaining_steps, tracking, needs_model
        and needs_observation. Inspect stop_reason when needs_model is true:
        change the grasp/direction/path or allocation with a new intent only
        after observing the actual state. Normal checkpoints continue locally.
        After a known interruption, reconcile then resume_motion(intent_id);
        the executor replans only the remainder from the measured endpoint.
        Unknown execution, external movement or a lost host session cannot
        automatically resume. Process-restart recovery is not advertised.
        preview=True estimates this intent without reserving its ID or moving.
        """
        from .coordination import Pose

        self._interaction_guard()

        def target(value):
            if isinstance(value, Pose):
                return asdict(value)
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, dict):
                return {k: target(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [target(v) for v in value]
            return value

        raw = self._readonly("intent_prepare", dict(
            intent_id=intent_id,
            targets={a: target(v) for a, v in (("left", left), ("right", right)) if v is not None},
            interaction=interaction, directions={} if directions is None else target(directions),
            holding={} if holding is None else holding,
            completion={} if completion is None else completion,
            max_steps=max_steps, reserve_steps=reserve_steps,
            keep_relative=keep_relative, preview=preview,
        ))
        if raw.get("interrupted"):
            from .coordination import CoordinationPause

            self._coordination_feedback = raw if preview else self.motion_status(intent_id)
            raise CoordinationPause("intent preparation interrupted; resume the saved intent")
        if preview or raw["status"] not in ("checkpoint", "paused"):
            return raw
        return self.resume_motion(intent_id)

    def motion_status(self, intent_id: str) -> dict:
        """Read a retained motion intent without dispatching or refreshing sensors.

        Completed results describe their recorded endpoint, not a claim that the
        scene remains unchanged. Unknown IDs reject; they never recreate motion.
        intent contains a copy of the frozen goal and constraints for inspection.
        """
        return self._readonly("intent_status", {"intent_id": intent_id})

    def resume_motion(self, intent_id: str, *, max_steps=None) -> dict:
        """Continue only this intent's remaining motion after receipt reconciliation.

        Same frozen targets, direction/holding constraints and cumulative step
        and correction counts. max_steps optionally raises the TOTAL cumulative
        allowance (not an extra allowance); repeating it never resets spending.
        Known stops replan from actual joint state;
        completed intents return their saved result without dispatch. Unknown
        or externally changed execution requires review, never whole-call replay.
        """
        self._interaction_guard()
        raw = self.motion_status(intent_id)
        if "operation" not in raw:
            return raw
        return self.resume_operation(raw["operation"]["ref"], max_steps=max_steps)

    def operation_status(self, operation: dict) -> dict:
        """Read a live operation by the ref returned in result['operation']['ref'].

        Common feedback: state, current phase, completed phases, cumulative budget,
        last effectful receipt, continuation and verification. Domain fields remain
        available. No movement or new visual observation. Recorded completion is
        historical; it does not establish that the scene or evidence is still current.
        References include kind and session; a stale session cannot bind to a new ID.
        """
        return self._readonly("operation_status", {"operation": _argument(operation)})

    def resume_operation(self, operation: dict, *, max_steps=None) -> dict:
        """Continue only an already-started motion or grasp phase in this live session.

        Pass result['operation']['ref']. Reconcile unknown action receipts first.
        The executor retains targets, adjustment identity, counters and completed
        effects. A ready grasp returns without starting its next phase: choose that
        phase explicitly with grasp_step after checking evidence. Completed and
        stopped operations never replay. Unknown effects or a changed endpoint
        prohibit continuation. max_steps raises the TOTAL allowance, not a new budget.
        This protocol does not provide recovery across host process restarts.
        """
        self._interaction_guard()
        raw = self.operation_status(operation)
        common = raw["operation"]
        if not common["continuation"]["allowed"]:
            return raw
        phase = common.get("phase") or {}
        grasp_id = operation["id"] if operation["kind"] == "grasp" else None
        if grasp_id is not None and phase.get("name") == "close":
            self._grasp_tracking_feedback(grasp_id, {}, seed=False)

        def record(payload):
            if "observation" in payload and self._last_state is not None:
                payload["observation"]["step"] = self._last_state.step
                payload["observation"]["state_version"] = self._action_revision
            self._coordination_feedback = payload
            return payload

        steps = 0
        try:
            while True:
                raw = self._stateful("operation_resume", {
                    "operation": _argument(operation), "allowance": max_steps}, record)
                steps += raw.get("steps_used", 0)
                if raw["operation"]["state"] != "running":
                    raw["steps_used"] = steps
                    if grasp_id is not None:
                        raw = self._grasp_object_feedback(grasp_id, phase.get("name"), raw)
                    self._coordination_feedback = raw
                    return raw
        except BaseException:
            # Pausing is metadata only. The host may still be finishing a native
            # action; it will refuse uncertain continuation rather than replay it.
            try:
                if operation["kind"] == "motion":
                    self._readonly("intent_pause", {"intent_id": operation["id"]})
                record(self.operation_status(operation))
            except Exception:
                pass  # Preserve the original failure; receipt gates still block unknown work.
            raise

    def align_local(
        self, intent_id: str, offset, *, arm="right", reference=None, frame="world",
        translation_limits=(.01, .01, .01), rotation_deg=(0., 0., 0.),
        rotation_limit_deg=5., holding="keep", completion=None, max_steps=None, preview=False,
    ) -> dict:
        """Align to an explicit offset inside a frozen observed-pose envelope.

        reference is a world TCP Pose or xyz/angles target; omitted means the
        current measured pose, frozen on first use of this ID. offset is metres
        FROM that reference, not an incremental nudge. frame='world' uses world
        axes; 'tcp' uses the reference's fixed axes. rotation_deg is an axis-angle
        vector in those axes, NOT pitch/yaw/roll. Zero preserves orientation.
        translation_limits gives the maximum absolute offset on each axis
        (0..5 cm); rotation_limit_deg bounds the total rotation (up to 15 deg).
        Start, goal, planned substeps and measured endpoints must stay inside.
        Bounds do not detect object collisions or force. No automatic search.

        Choose the offset from fresh images. Reuse the SAME reference for a
        series of adjustments so their envelope does not drift. A new target
        needs a new ID; repeated IDs and resume_motion never repeat completed
        alignment. holding is keep/empty/retain. Default arrival is 1.5 mm/1 deg.
        Returns the same motion feedback and partial-progress semantics as
        execute_motion; reached is TCP arrival, not grasp or contact proof.
        """
        self._interaction_guard()
        raw = self._readonly("local_prepare", dict(
            intent_id=intent_id, arm=arm, offset=_argument(offset),
            reference=_argument(reference),
            frame=frame, translation_limits=_argument(translation_limits),
            rotation_deg=_argument(rotation_deg), rotation_limit_deg=rotation_limit_deg,
            holding=holding, completion=completion, max_steps=max_steps, preview=preview,
        ))
        if preview or raw["status"] not in ("checkpoint", "paused"):
            return raw
        return self.resume_motion(intent_id)

    def begin_grasp(
        self, grasp_id: str, target, *, arm="right", approach=(0., 0., .04),
        pregrasp=None, probe=(0., 0., .025), opening=1.,
        completion=None, translation_limits=(.01, .01, .01), rotation_limit_deg=5.,
        max_steps=None, object_ref=None,
    ) -> dict:
        """Register a staged grasp without moving, opening or preflighting later phases.

        target is world TCP xyz, Pose or xyz/angles at closure. pregrasp can
        specify a separate pose/orientation; otherwise target+approach (world
        metres) is used. probe is a short world displacement after closure.
        opening is normalized finger separation. completion controls align
        accuracy (default 1.5 mm/1 deg). Local refinement is bounded relative to
        the ORIGINAL target by translation_limits (world axes, max 5 cm each)
        and rotation_limit_deg (max 15 deg). No object-specific recipe or search.

        Call grasp_step for open, approach, align, close, probe in that order.
        Each stage returns feedback before the next. Only the CURRENT stage is
        planned: an expensive future probe or task is not a pickup veto.
        Default budget is remaining native actions, with no return reserve.
        Explicit max_steps covers all stages and refinements cumulatively.
        The ID freezes the plan. Use it to continue, never wrap in once().

        On snapshot backends, pass object_ref=robo.object_ref(name, description)
        to bind the intended object while it is visible. Preparation and open/
        approach/align return object_tracking; native RGB frames continue these
        material features through closure, without extra physical actions.
        Only the measured POST-closure positions become the hold baseline.
        close/probe return object_evidence. Inspect
        baseline_ready and baseline_quality.space before dependent actions.
        A missing baseline gives an exact capture ID and recovery instruction;
        do not treat unknown as retention or keep probing to manufacture a
        baseline. Repeated completed stages return saved evidence without
        closing/probing again or shifting the baseline. Evidence is read-only.
        """
        self._interaction_guard()
        self._object_store()
        binding = (dict(reference=self._object_ref(object_ref), arm=arm)
                   if object_ref is not None else None)
        if grasp_id in self._grasp_objects and self._grasp_objects[grasp_id] != binding:
            result = self.grasp_status(grasp_id)
            reason = "grasp_object_identity_conflict"
            result.update(status="rejected", reached=False, steps_used=0,
                          needs_model=True, stop_reason=reason)
            result["operation"].update(state="rejected", execution_complete=False,
                                       reason=reason,
                                       continuation=dict(action="replan", allowed=False))
            return result
        result = self._readonly("grasp_prepare", dict(
            grasp_id=grasp_id, target=_argument(target), arm=arm, approach=_argument(approach),
            pregrasp=_argument(pregrasp), probe=_argument(probe), opening=opening,
            completion=completion, translation_limits=_argument(translation_limits),
            rotation_limit_deg=rotation_limit_deg,
            max_steps=max_steps,
        ))
        if result.get("status") != "rejected":
            from copy import deepcopy
            new = grasp_id not in self._grasp_objects
            self._grasp_objects[grasp_id] = deepcopy(binding)
            if new and binding is not None:
                result = self._grasp_tracking_feedback(grasp_id, result, seed=True)
        return result

    def grasp_step(self, grasp_id: str, stage: str, *, adjustment_id=None,
                   offset=(0., 0., 0.), rotation_deg=(0., 0., 0.), max_steps=None) -> dict:
        """Execute/continue one grasp phase: open, approach, align, close, probe.

        A completed phase never dispatches again. Paused phases continue from
        the measured endpoint. max_steps can raise the total grasp allowance
        without resetting spending. Unknown effects/external motion reject.
        reached refers to this stage; grasp_complete only follows the probe.
        Inspect stage_result, next_stage, actual_pose and stop_reason.

        After approach, align goes to the nominal closure target. To refine an
        alignment after observing error, use stage='align', a NEW adjustment_id,
        offset FROM the original target (world metres) and optional axis-angle
        rotation_deg in world axes. All refinements share the original envelope
        and budget. No refinement is allowed after closure. A failed approach
        must finish before alignment; a failed align may be refined inside its
        bounds. No automatic regrasp/opening, no blind retries.

        close requires measured alignment and reports obstructed closure only.
        probe requires retained closure, keeps squeezing, and stops on loss.
        retained/grasp_complete do NOT verify object identity or stability:
        inspect fresh images and object_evidence (when bound via object_ref)
        before carrying, turning or releasing a donor. image_consistent is
        explicitly 2D projection evidence, not a 3D grasp certificate.
        """
        self._interaction_guard()
        if stage == 'close' and (grasp_id, stage) not in getattr(self, '_grasp_evidence', {}):
            # Observe before dispatch; never close first and then rediscover a
            # partly hidden target from its text description.
            self._grasp_tracking_feedback(grasp_id, {}, seed=False)
        result = self._stateful("grasp_step", dict(
            grasp_id=grasp_id, stage=stage, adjustment_id=adjustment_id,
            offset=_argument(offset), rotation_deg=_argument(rotation_deg), max_steps=max_steps,
        ), lambda payload: payload)
        result = self._grasp_object_feedback(grasp_id, stage, result)
        self._coordination_feedback = result
        return result

    def grasp_status(self, grasp_id: str) -> dict:
        """Read recorded grasp phases, budget and last feedback without executing."""
        return self._readonly("grasp_status", {"grasp_id": grasp_id})
