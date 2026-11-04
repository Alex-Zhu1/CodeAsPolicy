"""Intent contracts through the public API, with actual dispatch/fault accounting."""

import numpy as np
import pytest

from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts import robodojo_tools as motion
from pyrualean.hosts import robodojo_trajectory as trajectory
from pyrualean.interaction import InteractionPause
from tests.test_robodojo_trajectory import TimedEnvironment, station


def empty_move(robo, intent_id="travel", **kwargs):
    return robo.execute_motion(intent_id, holding={"right": "empty"}, **kwargs)


def test_intent_policy_rejects_legacy_bypass_before_any_dispatch(tmp_path):
    env, robo, _ = station(tmp_path)
    goal = robo.observe().right.shifted(dz=.06)
    robo._intent_motion_only = True
    for call in (
        lambda: robo.move_to(goal.xyz, motion="native"),
        lambda: robo.move_to(goal.xyz, preview=True),
        lambda: robo.move_both(right=goal, motion="native"),
        lambda: robo.motion_profile(),
        lambda: robo.motion_window(),
    ):
        with pytest.raises(ValueError, match="executor owns motion parameters"):
            call()
    assert not env.actions
    assert empty_move(robo, right=goal)["reached"]
    # Direct library users can continue to use the existing low-level API.
    robo._intent_motion_only = False
    assert robo.move_to(goal.shifted(dz=.02).xyz, motion="native").reached


def test_executor_selects_mechanics_and_preview_does_not_reserve_identity(tmp_path):
    env, robo, _ = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06)
    free = empty_move(robo, right=target, preview=True)
    contact = empty_move(robo, right=target, interaction="contact", preview=True, max_steps=120)
    assert free["planned_steps"] < contact["planned_steps"]
    assert free["policy"]["motion"] == {"mode": "native"}
    assert contact["trajectory"]["continuity"] == "C2"
    assert not env.actions
    assert robo.motion_status("travel")["stop_reason"] == "unknown_intent_in_this_session"
    result = empty_move(robo, right=target)
    assert result["reached"] and result["intent_steps_used"] == len(env.actions)


def test_completed_identity_cannot_replay_or_silently_change_goal(tmp_path):
    env, robo, _ = station(tmp_path)
    goal = robo.observe().right.shifted(dz=.06)
    first = empty_move(robo, right=goal)
    count = len(env.actions)
    assert first["reached"]
    assert empty_move(robo, right=goal)["steps_used"] == 0
    assert robo.resume_motion("travel")["steps_used"] == 0
    conflict = empty_move(robo, right=goal.shifted(dz=.03))
    assert conflict["stop_reason"] == "intent_identity_conflict"
    assert len(env.actions) == count


def test_direction_constraint_rejects_the_goal_before_any_motion(tmp_path):
    env, robo, _ = station(tmp_path)
    target = robo.observe().right.shifted(dx=.03, dz=.04)
    result = empty_move(robo, right=target, directions={"right": [0, 0, 1]})
    assert result["status"] == "rejected" and result["needs_model"]
    assert result["stop_reason"] == "right: direction_constraint"
    assert not env.actions


def test_direction_is_checked_on_measured_motion_not_only_requested_goal(tmp_path):
    env, robo, _ = station(tmp_path)
    step = env.step

    def drift(action):
        value = step(action)
        env.joints["right"][0] += .1
        return value

    env.step = drift
    result = empty_move(robo, right=robo.observe().right.shifted(dz=.06),
                        directions={"right": [0, 0, 1]})
    assert result["stop_reason"] == "right: direction_constraint"
    assert result["needs_model"] and len(env.actions) == 1


def test_contact_budget_estimate_executes_a_resting_prefix_without_speeding_up(tmp_path):
    env, robo, _ = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06)
    result = empty_move(robo, right=target, interaction="contact", max_steps=20,
                        completion={"position_m": .0025})
    assert result["status"] == "paused" and result["stop_reason"] == "segment_complete"
    assert not result["reached"] and 0 < len(env.actions) <= 20
    assert result["policy"]["motion"]["phase"] == "precision"
    assert all(trajectory.TRAJECTORY_KEY in a for a in env.wire_actions)
    assert np.max(np.abs(env.wire_actions[-1][trajectory.TRAJECTORY_KEY]["velocities"][-1])) < 1e-9
    before = len(env.actions)
    result = robo.resume_motion("travel", max_steps=150)
    assert result["reached"] and result["intent_steps_used"] == len(env.actions) > before
    assert result["step_limit"] == 150
    assert robo.resume_motion("travel", max_steps=150)["steps_used"] == 0


def test_reserved_episode_budget_is_preserved(tmp_path):
    env, robo, _ = station(tmp_path)
    env.step_limit = 7
    result = empty_move(robo, right=robo.observe().right.shifted(dz=.10),
                        max_steps=20, reserve_steps=4)
    assert result["status"] == "paused" and 0 < len(env.actions) <= 3
    assert env.step_limit - len(env.actions) >= 4
    # A fresh scene isolates the second allocation from the first prefix.
    env, robo, _ = station(tmp_path / "second")
    env.step_limit = 7
    okay = empty_move(robo, "short", right=robo.observe().right.shifted(dz=.04),
                      max_steps=20, reserve_steps=4)
    assert okay["reached"] and len(env.actions) <= 3


def test_retention_requirement_stops_on_loss_without_opening_or_regrasping(tmp_path):
    env, robo, host = station(tmp_path)
    host._primitives._grip_commands = {"right": 0.0}
    step = env.step

    def drop(action):
        value = step(action)
        env.grips["right"] = 0.
        return value

    env.step = drop
    result = robo.execute_motion("carry", right=robo.observe().right.shifted(dz=.06),
                                 holding={"right": "retain"}, max_steps=100)
    assert result["stop_reason"] == "right: holding_requirement"
    assert result["needs_model"] and len(env.actions) == 1
    assert env.actions[0]["right_ee_joint_state"] == [0.]
    assert result["policy"]["motion"]["phase"] == "carry"


def test_known_cancel_resumes_remaining_path_and_never_resets_step_allowance(tmp_path):
    env, robo, host = station(tmp_path)
    step = env.step

    def interrupt(action):
        value = step(action)
        if len(env.actions) == 2:
            host._primitives.cancelled.set()
        return value

    env.step = interrupt
    goal = robo.observe().right.shifted(dz=.12)
    paused = empty_move(robo, right=goal, max_steps=8)
    assert paused["status"] == "paused" and paused["intent_steps_used"] == 2
    prefix = [a["right_arm_joint_state"] for a in env.actions]
    host._primitives.cancelled.clear()
    result = robo.resume_motion("travel")
    assert result["reached"] and result["intent_steps_used"] == len(env.actions) <= 8
    assert result["remaining_steps"] == 8 - len(env.actions)
    assert env.actions[2]["right_arm_joint_state"] not in prefix
    count = len(env.actions)
    assert robo.resume_motion("travel")["reached"] and len(env.actions) == count


def test_corrections_remain_bounded_across_replanning_after_cancel(tmp_path):
    env, robo, host = station(tmp_path)
    step = env.step

    def lag(action):
        before = np.array(env.joints["right"])
        value = step(action)
        if len(env.actions) % 2:
            env.joints["right"] = (before + .4 * (np.array(env.joints["right"]) - before)).tolist()
        if len(env.actions) == 2:
            host._primitives.cancelled.set()
        return value

    env.step = lag
    first = empty_move(robo, right=robo.observe().right.shifted(dz=.12), max_steps=20)
    assert first["status"] == "paused" and first["tracking"]["correction_steps"] == 1
    host._primitives.cancelled.clear()
    result = robo.resume_motion("travel")
    assert result["tracking"]["correction_steps"] == 2
    assert result["needs_model"] and result["tracking"]["cause"] == "correction_limit"
    assert result["intent_steps_used"] == len(env.actions)


def test_timed_cancel_replans_from_measured_endpoint_with_a_new_resting_start(tmp_path):
    env, robo, host = station(tmp_path)
    step = env.step

    def interrupt(action):
        value = step(action)
        if len(env.actions) == 3:
            host._primitives.cancelled.set()
        return value

    env.step = interrupt
    first = empty_move(robo, right=robo.observe().right.shifted(dz=.05),
                       interaction="approach", max_steps=100)
    assert first["status"] == "paused"
    start = np.r_[env.joints["left"], env.joints["right"]]
    host._primitives.cancelled.clear()
    result = robo.resume_motion("travel")
    assert result["reached"]
    block = env.wire_actions[3][trajectory.TRAJECTORY_KEY]
    assert np.max(abs(np.array(block["positions"])[0] - start)) < .001
    assert np.max(abs(np.array(block["velocities"])[0])) < .01
    assert result["intent_steps_used"] == len(env.actions)


def test_external_motion_after_pause_requires_review(tmp_path):
    env, robo, host = station(tmp_path)
    step = env.step

    def interrupt(action):
        value = step(action)
        host._primitives.cancelled.set()
        return value

    env.step = interrupt
    assert empty_move(robo, right=robo.observe().right.shifted(dz=.08))["status"] == "paused"
    host._primitives.cancelled.clear()
    env.joints["left"][0] += .01
    count = len(env.actions)
    result = robo.resume_motion("travel")
    assert result["stop_reason"] == "state_changed_outside_intent"
    assert result["needs_model"] and len(env.actions) == count


def test_transport_uncertainty_never_becomes_automatic_resume(tmp_path):
    env, robo, host = station(tmp_path)
    env.raise_after_step = True
    with pytest.raises(InteractionPause, match="unknown|reconcile"):
        empty_move(robo, right=robo.observe().right.shifted(dz=.06))
    record = robo.motion_status("travel")
    assert record["needs_model"]
    with pytest.raises(InteractionPause, match="unknown|reconcile"):
        robo.resume_motion("travel")
    assert len(env.actions) == 1


def test_visual_completion_is_returned_for_observation_not_claimed_as_verified(tmp_path):
    _, robo, _ = station(tmp_path)
    result = empty_move(robo, right=robo.observe().right.shifted(dz=.04),
                        completion={"verify": "The intended object followed the gripper"})
    assert result["reached"] and result["needs_observation"]
    assert result["verification"] == "The intended object followed the gripper"
    assert not result.get("task_success")


@pytest.mark.parametrize("distance,expected", [(.10, "paused"), (.04, "completed")])
def test_python_deadline_after_checkpoint_preserves_remaining_or_completed_intent(
        tmp_path, monkeypatch, distance, expected):
    from pyrualean.execution import ExecutionLimit

    env, robo, _ = station(tmp_path)
    original = robo._stateful

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise ExecutionLimit("timeout", "code deadline")

    monkeypatch.setattr(robo, "_stateful", interrupted)
    with pytest.raises(ExecutionLimit):
        empty_move(robo, right=robo.observe().right.shifted(dz=distance), max_steps=8)
    assert robo.motion_status("travel")["status"] == expected
    before = len(env.actions)
    monkeypatch.setattr(robo, "_stateful", original)
    result = robo.resume_motion("travel")
    assert result["reached"] and result["intent_steps_used"] == len(env.actions) <= 8
    if expected == "completed":
        assert len(env.actions) == before


def test_preview_does_not_supersede_a_prepared_execution(tmp_path):
    from dataclasses import asdict

    env, robo, _ = station(tmp_path)
    goal = robo.observe().right.shifted(dz=.04)
    prepared = robo._readonly("intent_prepare", dict(
        intent_id="ready", targets={"right": asdict(goal)}, interaction="free",
        directions={}, holding={"right": "empty"}, completion={}, max_steps=8,
        reserve_steps=0, keep_relative=False,
    ))
    assert prepared["status"] == "checkpoint"
    preview = empty_move(robo, "estimate", right=goal.shifted(dz=.03), preview=True)
    assert preview["accepted"] and not env.actions
    assert robo.resume_motion("ready")["reached"]


def test_accuracy_requirement_selects_precision_without_a_task_recipe(tmp_path):
    _, robo, _ = station(tmp_path)
    target = robo.observe().right.shifted(dz=.05)
    coarse = empty_move(robo, right=target, interaction="contact", preview=True, max_steps=120)
    fine = empty_move(robo, right=target, interaction="contact", preview=True, max_steps=120,
                      completion={"position_m": .0025})
    assert fine["policy"]["motion"]["phase"] == "precision"
    assert coarse["policy"]["motion"]["phase"] == "approach"
    assert fine["planned_steps"] > coarse["planned_steps"]


def _precision_station(tmp_path):
    env = TimedEnvironment()
    # Start away from the synthetic environment's straight-arm joint limit so
    # both positive and negative millimetre tracking errors are reachable.
    for arm in kin.ARMS:
        pose = motion._pose_of(arm, env.joints[arm])
        pose[2, 3] += .05
        solved = kin.solve(arm, pose, env.joints[arm], pos_tol=1e-5, rot_tol_deg=.001)
        assert solved.ok
        env.joints[arm] = list(solved.q)
    return station(tmp_path, env)


def _offset_measured_pose(env, *, dz=0., angle=0.):
    """Inject measured joint tracking error without changing commanded targets."""
    joints = env.joints["right"]
    pose = motion._pose_of("right", joints)
    pose[2, 3] += dz
    radians = np.deg2rad(angle)
    c, s = np.cos(radians), np.sin(radians)
    pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ pose[:3, :3]
    solved = kin.solve("right", pose, joints, pos_tol=1e-5, rot_tol_deg=.001)
    assert solved.ok and not solved.collides
    env.joints["right"] = list(solved.q)


@pytest.mark.parametrize("offset", [dict(dz=-.0015), dict(angle=1.5)])
def test_precision_arrival_does_not_abort_on_bounded_moving_lag(tmp_path, offset):
    env, robo, _ = _precision_station(tmp_path)
    goal = robo.observe().right.shifted(dz=.02)
    args = dict(right=goal, interaction="contact", max_steps=100,
                completion={"position_m": .001, "orientation_deg": 1.})
    preview = empty_move(robo, preview=True, **args)
    assert preview["accepted"]
    step, injected = env.step, []

    def lag(action):
        value = step(action)
        if len(env.actions) < preview["planned_steps"]:
            _offset_measured_pose(env, **offset)
            injected.append(len(env.actions))
        return value

    env.step = lag
    result = empty_move(robo, **args)
    assert injected and result["reached"], result
    assert len(env.actions) == preview["planned_steps"]
    assert result["tracking"]["phase"] == "arrival"
    assert result["tracking"]["correction_steps"] == 0
    assert result["arms"]["right"]["position_error_m"] <= .001
    assert result["arms"]["right"]["orientation_error_deg"] <= 1.


@pytest.mark.parametrize("offset", [dict(dz=-.0015), dict(angle=1.5)])
def test_tracking_envelope_never_certifies_an_inaccurate_arrival(tmp_path, offset):
    env, robo, _ = _precision_station(tmp_path)
    step = env.step

    def persistent_lag(action):
        value = step(action)
        _offset_measured_pose(env, **offset)
        return value

    env.step = persistent_lag
    result = empty_move(robo, right=robo.observe().right.shifted(dz=.02),
                        interaction="contact", max_steps=100,
                        completion={"position_m": .001, "orientation_deg": 1.})
    assert result["needs_model"] and not result["reached"], result
    assert result["tracking"]["phase"] == "arrival"
    assert result["tracking"]["correction_steps"] <= 2
    assert not result["arms"]["right"]["reached"]


@pytest.mark.parametrize("spare_steps", [0, 2])
def test_arrival_holds_only_the_final_target_within_the_original_budget(tmp_path, spare_steps):
    env, robo, _ = _precision_station(tmp_path)
    args = dict(right=robo.observe().right.shifted(dz=.02), interaction="contact",
                completion={"position_m": .001, "orientation_deg": 1.})
    preview = empty_move(robo, preview=True, max_steps=100, **args)
    count = preview["planned_steps"]
    step = env.step

    def settling_lag(action):
        value = step(action)
        n = len(env.actions)
        _offset_measured_pose(env, dz=-.0024 if n < count else -.0018 if n == count else 0.)
        return value

    env.step = settling_lag
    result = empty_move(robo, max_steps=count + spare_steps, **args)
    assert result["tracking"]["phase"] == "arrival", result
    if spare_steps:
        assert result["reached"], result
        assert result["tracking"]["correction_steps"] == 1
        assert len(env.actions) == count + 1
        assert env.actions[-1] == env.actions[-2]
        hold = env.wire_actions[-1][trajectory.TRAJECTORY_KEY]
        assert np.max(np.abs(hold["velocities"])) == 0.
        assert np.max(np.ptp(hold["positions"], axis=0)) == 0.
        before = len(env.actions)
        assert robo.resume_motion("travel")["steps_used"] == 0
        assert len(env.actions) == before
    else:
        assert not result["reached"] and result["needs_model"]
        assert result["tracking"]["cause"] == "insufficient_correction_budget"
        assert len(env.actions) == count


def test_excessive_moving_lag_still_stops_before_the_endpoint(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    args = dict(right=robo.observe().right.shifted(dz=.02), interaction="contact",
                completion={"position_m": .001, "orientation_deg": 1.}, max_steps=100)
    count = empty_move(robo, preview=True, **args)["planned_steps"]
    step = env.step

    def lag(action):
        value = step(action)
        _offset_measured_pose(env, dz=-.006)
        return value

    env.step = lag
    result = empty_move(robo, **args)
    assert result["needs_model"] and not result["reached"]
    assert result["tracking"]["phase"] == "path_tracking"
    assert result["tracking"]["correction_steps"] == 0
    assert len(env.actions) == 1 < count


@pytest.mark.parametrize("constraint", ["direction", "relative"])
def test_tracking_tolerance_does_not_weaken_hard_pose_constraints(tmp_path, constraint):
    env, robo, _ = _precision_station(tmp_path)
    initial = robo.observe()
    args = dict(right=initial.right.shifted(dz=.02), interaction="contact",
                completion={"position_m": .001, "orientation_deg": 1.}, max_steps=100)
    if constraint == "direction":
        args["directions"] = {"right": [0, 0, 1]}
    else:
        args.update(left=initial.left.shifted(dz=.02), keep_relative=True)
    step = env.step

    def drift(action):
        value = step(action)
        # This drift fits the 2.5 mm contact tracking envelope, but not either
        # explicit 1 mm geometric constraint (backwards past the axis start).
        _offset_measured_pose(env, dz=-.002)
        return value

    env.step = drift
    result = empty_move(robo, **args)
    assert result["needs_model"] and len(env.actions) == 1, result
    expected = ("right: direction_constraint" if constraint == "direction"
                else "relative_pose_constraint")
    assert result["stop_reason"] == expected


def test_editing_status_metadata_cannot_change_a_paused_intent(tmp_path):
    env, robo, host = station(tmp_path)
    step = env.step

    def interrupt(action):
        value = step(action)
        if len(env.actions) == 1:
            host._primitives.cancelled.set()
        return value

    env.step = interrupt
    first = empty_move(robo, right=robo.observe().right.shifted(dz=.05),
                       interaction="approach", max_steps=100)
    assert first["status"] == "paused"
    status = robo.motion_status("travel")
    status["policy"]["motion"] = {"mode": "native"}
    status["intent"]["max_steps"] = 1000
    host._primitives.cancelled.clear()
    result = robo.resume_motion("travel")
    assert result["reached"] and "window" in result["policy"]["motion"]
    assert result["remaining_steps"] == 100 - result["intent_steps_used"]
    assert all(trajectory.TRAJECTORY_KEY in a for a in env.wire_actions)


def test_cancel_during_preflight_retains_a_known_zero_action_intent(tmp_path):
    from pyrualean.coordination import CoordinationPause

    env, robo, host = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06)
    host._primitives.cancelled.set()
    with pytest.raises(CoordinationPause, match="preparation interrupted"):
        empty_move(robo, right=target, max_steps=8)
    result = robo.motion_status("travel")
    assert result["status"] == "paused" and result["intent_steps_used"] == 0
    assert not env.actions
    host._primitives.cancelled.clear()
    completed = robo.resume_motion("travel")
    assert completed["reached"] and completed["intent_steps_used"] == len(env.actions) <= 8


def test_dual_intent_keeps_relative_pose_and_same_targets_each_tick(tmp_path):
    env, robo, _ = station(tmp_path)
    initial = robo.observe()
    result = robo.execute_motion("lift_both", left=initial.left.shifted(dz=.04),
                                 right=initial.right.shifted(dz=.04),
                                 holding={"left": "empty", "right": "empty"}, keep_relative=True)
    assert result["reached"]
    for action in env.actions:
        poses = {a: kin.fk(a, action[f"{a}_arm_joint_state"]) for a in kin.ARMS}
        separation = np.linalg.norm(poses["right"][:3, 3] - poses["left"][:3, 3])
        assert separation == pytest.approx(.6, abs=.001)


@pytest.mark.parametrize("argument", [
    {"directions": {"right": [0, 0, 0]}}, {"completion": {"force": 5}},
    {"completion": {"position_m": True}}, {"max_steps": True}, {"reserve_steps": -1},
    {"holding": []}, {"interaction": "insert_key"},
])
def test_invalid_or_unsupported_intent_cannot_dispatch(tmp_path, argument):
    env, robo, _ = station(tmp_path)
    result = robo.execute_motion("invalid", right=robo.observe().right.shifted(dz=.06), **argument)
    assert result["status"] == "rejected" and not env.actions
