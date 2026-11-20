"""Intent contracts through the public API, with actual dispatch/fault accounting."""

import numpy as np
import pytest

from codeaspolicy.hosts import robodojo_kinematics as kin
from codeaspolicy.hosts import robodojo_tools as motion
from codeaspolicy.hosts import robodojo_trajectory as trajectory
from codeaspolicy.interaction import InteractionPause
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


def test_live_constraint_stop_preserves_metric_diagnostics(tmp_path):
    env, robo, _ = station(tmp_path)
    step = env.step

    def drift(action):
        value = step(action)
        env.joints["right"][0] += .1
        return value

    env.step = drift
    result = empty_move(robo, right=robo.observe().right.shifted(dz=.06),
                        constraints={"corridor_m": .005})
    assert result["stop_reason"] == "right: corridor_constraint"
    assert result["needs_model"] and len(env.actions) == result["steps_used"] == 1
    diagnostic = result["constraint_violation"]
    assert diagnostic["metric"] == "corridor_deviation_m"
    assert diagnostic["value"] > diagnostic["bound"] == .005
    assert diagnostic["delta"] > 0 and diagnostic["remedy"]
    assert robo.motion_status("travel")["constraint_violation"] == diagnostic


@pytest.mark.parametrize("offset, radius, xy_only, shift", [
    ([0., 0., .01], .004, False, {"dz": .06}),
    ([.0698, 0., .01], .07, False, {"dz": .06}),
    ([-.01, 0., 1.], .002, True, {"dx": -.04, "dz": .04}),
])
def test_keep_out_crossing_between_native_waypoints_rejects_without_moving(
    tmp_path, offset, radius, xy_only, shift,
):
    env, robo, _ = station(tmp_path)
    start = robo.observe().right
    obstacle = np.asarray(start.xyz) + offset
    goal = start.shifted(**shift)
    dimensions = 2 if xy_only else 3
    # Both the initial state and final target are clear; the route intersects.
    assert np.linalg.norm(np.asarray(start.xyz)[:dimensions] - obstacle[:dimensions]) > radius
    assert np.linalg.norm(np.asarray(goal.xyz)[:dimensions] - obstacle[:dimensions]) > radius
    result = empty_move(robo, right=goal, constraints={"keep_out": [{
        "xyz": obstacle.tolist(), "radius_m": radius, "xy_only": xy_only,
    }]})
    assert result["status"] == "rejected" and result["steps_used"] == 0
    assert result["stop_reason"] == "right: keep_out_constraint"
    diagnostic = result["constraint_violation"]
    assert diagnostic["metric"] == "obstacle_distance_m"
    assert diagnostic["value"] < diagnostic["bound"] == radius
    assert diagnostic["delta"] < 0 and diagnostic["remedy"]
    assert not env.actions


@pytest.mark.parametrize("xy_only", [False, True])
@pytest.mark.parametrize("interaction", ["free", "approach"])
def test_keep_out_clear_route_remains_executable(tmp_path, xy_only, interaction):
    env, robo, _ = station(tmp_path)
    start = robo.observe().right
    obstacle = np.asarray(start.xyz) + [.03, 0., .01]
    result = empty_move(robo, right=start.shifted(dz=.06), interaction=interaction,
                        constraints={"keep_out": [{
                            "xyz": obstacle.tolist(), "radius_m": .004, "xy_only": xy_only,
                        }]})
    assert result["reached"] and env.actions, result
    assert "constraint_violation" not in result


@pytest.mark.parametrize("constraints", [
    {"max_tilt_deg": {"right": 5.}},
    {"z_bounds": {"right": [0., 2.]}},
    {"corridor_m": {"right": .01}},
])
def test_arm_specific_constraints_allow_bilateral_motion(tmp_path, constraints):
    env, robo, _ = station(tmp_path)
    initial = robo.observe()
    result = robo.execute_motion(
        "lift_both", left=initial.left.shifted(dz=.04), right=initial.right.shifted(dz=.04),
        holding={"left": "empty", "right": "empty"}, constraints=constraints,
    )
    assert result["reached"] and env.actions, result


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
    assert result["reserve_steps"] == 4
    assert robo.motion_status("travel")["reserve_steps"] == 4
    assert env.step_limit - len(env.actions) >= 4
    # A fresh scene isolates the second allocation from the first prefix.
    env, robo, _ = station(tmp_path / "second")
    env.step_limit = 7
    okay = empty_move(robo, "short", right=robo.observe().right.shifted(dz=.04),
                      max_steps=20, reserve_steps=4)
    assert okay["reached"] and okay["reserve_steps"] == 4 and len(env.actions) <= 3


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


def test_retention_permits_thin_object_full_closure_and_sub_millimetre_settle(tmp_path):
    env, robo, host = station(tmp_path)
    host._primitives._grip_commands = {"right": 0.0}
    env.grips["right"] = .01015
    step = env.step

    def settle(action):
        value = step(action)
        env.grips["right"] = .00961
        return value

    env.step = settle
    first = robo.execute_motion("lift_thin_shaft", right=robo.observe().right.shifted(dz=.04),
                                holding={"right": "retain"}, max_steps=100)
    assert first["reached"], first
    env.grips["right"] = 0.0
    second = robo.execute_motion("carry_thin_blade", right=robo.observe().right.shifted(dz=.04),
                                 holding={"right": "retain"}, max_steps=100)
    assert second["reached"], second
    assert second["policy"]["motion"]["phase"] == "carry"
    assert second["policy"]["motion"]["velocity"] == pytest.approx(2.0)
    assert second["policy"]["motion"]["acceleration"] == pytest.approx(16.0)
    assert second["policy"]["motion"]["jerk"] == pytest.approx(900.0)
    windowed = robo.execute_motion(
        "approach_thin_blade", right=robo.observe().right.shifted(dz=.04),
        interaction="approach", holding={"right": "retain"}, preview=True,
    )
    assert windowed["accepted"]
    assert windowed["policy"]["motion"]["window"]["far"]["velocity"] == pytest.approx(2.0)
    assert windowed["policy"]["motion"]["window"]["near"]["velocity"] == pytest.approx(1.6)


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


@pytest.mark.parametrize("distance,expected", [(.18, "paused"), (.04, "completed")])
def test_python_deadline_after_checkpoint_preserves_remaining_or_completed_intent(
        tmp_path, monkeypatch, distance, expected):
    from codeaspolicy.execution import ExecutionLimit

    env, robo, _ = station(tmp_path)
    original = robo._stateful

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise ExecutionLimit("timeout", "code deadline")

    monkeypatch.setattr(robo, "_stateful", interrupted)
    with pytest.raises(ExecutionLimit):
        empty_move(robo, right=robo.observe().right.shifted(dz=distance), max_steps=16)
    assert robo.motion_status("travel")["status"] == expected
    before = len(env.actions)
    if expected == "paused":
        assert before == 8
    monkeypatch.setattr(robo, "_stateful", original)
    result = robo.resume_motion("travel")
    assert result["reached"] and result["intent_steps_used"] == len(env.actions) <= 16
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
    from codeaspolicy.coordination import CoordinationPause

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


def test_post_contact_retreat_clearance_blocks_low_lateral_sweep(tmp_path):
    env = TimedEnvironment()
    pose = np.eye(4)
    pose[:3, :3] = kin.orientation(35., 0., 0.)
    pose[:3, 3] = [.20, -.18, .815]
    solved = kin.solve("right", pose, env.joints["right"], pos_tol=1e-4, rot_tol_deg=.1)
    assert solved.ok
    env.joints["right"] = list(solved.q)
    env, robo, _ = station(tmp_path, env)
    start = robo.observe().right
    stroke = empty_move(robo, "contact_push", right=start.shifted(dy=.03), interaction="contact")
    assert stroke["reached"], stroke
    before = len(env.actions)
    after_contact = robo.observe().right
    # A diagonal lateral free sweep from the low contact endpoint is rejected before moving.
    sweep = empty_move(
        robo, "diagonal_sweep",
        right=after_contact.shifted(dx=-.06, dy=-.03, dz=.04),
        interaction="free",
    )
    assert sweep["status"] == "rejected"
    assert sweep["stop_reason"] == "right: post_contact_retreat_clearance"
    assert len(env.actions) == before
    # Lifting vertically clears the post-contact guard, allowing subsequent lateral travel.
    lift = empty_move(robo, "vertical_clear", right=after_contact.shifted(dz=.05),
                      interaction="free")
    assert lift["reached"], lift
    sideways = empty_move(robo, "lateral_after_lift",
                          right=robo.observe().right.shifted(dx=-.04), interaction="free")
    assert sideways["reached"], sideways


def test_free_space_carry_reorientation_uses_coarser_angular_spacing_than_approach(tmp_path):
    env, robo, host = station(tmp_path)
    host._primitives._grip_commands = {"right": 0.0}
    start = robo.observe().right
    rotated = {"xyz": list(start.xyz), "angles": [0., 0., 40.]}
    free_carry = robo.execute_motion(
        "carry_free", right=rotated, interaction="free",
        holding={"right": "retain"}, preview=True,
    )
    precise_free = robo.execute_motion(
        "carry_precise", right=rotated, interaction="free",
        holding={"right": "retain"}, completion={"orientation_deg": 1.5}, preview=True,
    )
    approach_carry = robo.execute_motion(
        "carry_approach", right=rotated, interaction="approach",
        holding={"right": "retain"}, preview=True,
    )
    assert free_carry["accepted"] and approach_carry["accepted"] and precise_free["accepted"]
    assert free_carry["policy"]["max_step_deg"] == 8.
    assert precise_free["policy"]["max_step_deg"] == 5.
    assert approach_carry["policy"]["max_step_deg"] == 5.
    assert free_carry["trajectory"]["input_segments"] < approach_carry["trajectory"]["input_segments"]
    executed = robo.execute_motion(
        "carry_free", right=rotated, interaction="free", holding={"right": "retain"},
    )
    assert executed["reached"] and executed["reserve_steps"] == 0


def test_precision_approach_window_and_joint_fallback_keep_step_budget_compact(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    start = robo.observe().right
    descent = empty_move(
        robo, "precise_descent", right=start.shifted(dz=.065),
        interaction="approach", completion={"position_m": .0015, "orientation_deg": 1.},
        preview=True,
    )
    assert descent["accepted"]
    assert descent["policy"]["motion"]["window"]["distance_m"] == pytest.approx(.01)
    assert descent["policy"]["motion"]["window"]["near"]["phase"] == "precision"
    assert descent["planned_steps"] <= 12

    swing_target = {
        "xyz": [-0.1136718057940872, -0.21590945046771678, 0.845],
        "quat": [0.4473469441437785, 0.5476136517337958, 0.4473469441437785, -0.547613651733796],
    }
    swing = robo.execute_motion(
        "free_swing_compact", left=swing_target, interaction="free",
        holding={"left": "empty"}, preview=True,
    )
    assert swing["accepted"] and swing["planned_path"] == "joint"
    assert swing["planned_steps"] <= 23
    assert not env.actions


def test_thin_retain_carry_timing_and_checkpoint_steps_8_cut_steps_and_chunks(tmp_path):
    import math

    # 1. Thin-retain carry (opening=0.02): fast smooth C2 carry + checkpoint_steps=8
    env, robo, host = station(tmp_path / "thin")
    host._primitives._grip_commands = {"right": 0.0}
    env.grips["right"] = 0.02
    chunks = []
    original = robo._stateful

    def count_chunks(name, *args, **kwargs):
        if name in ("intent_execute", "operation_resume"):
            chunks.append(name)
        return original(name, *args, **kwargs)

    robo._stateful = count_chunks
    start = robo.observe().right
    goal = start.shifted(dx=-0.10, dy=-0.05, dz=0.12).rotated([0, 0, 1], 45)
    thin_res = robo.execute_motion(
        "thin_carry_fast", right=goal, holding={"right": "retain"}, max_steps=200,
    )
    assert thin_res["reached"], thin_res
    assert thin_res["policy"]["motion"] == {
        "velocity": 2.0, "acceleration": 16.0, "jerk": 900.0, "phase": "carry",
    }
    # Cut nearly in half from ~49 steps under legacy 1.1/6.0 thin_retain profile
    assert thin_res["intent_steps_used"] <= 28
    # checkpoint_steps=8 halves resume chunks compared to checkpoint_steps=4
    assert len(chunks) == math.ceil(thin_res["intent_steps_used"] / 8)

    # 2. Normal held carry (opening=0.08) + non-precise vs precise approach near_profile
    env2, robo2, host2 = station(tmp_path / "normal")
    host2._primitives._grip_commands = {"right": 0.0}
    env2.grips["right"] = 0.08
    start2 = robo2.observe().right
    norm_preview = robo2.execute_motion(
        "norm_carry", right=start2.shifted(dz=0.10),
        holding={"right": "retain"}, preview=True,
    )
    assert norm_preview["accepted"]
    assert norm_preview["policy"]["motion"] == {
        "velocity": 2.2, "acceleration": 20.0, "jerk": 950.0, "phase": "carry",
    }
    app_normal = empty_move(
        robo2, "app_normal", right=start2.shifted(dz=0.06),
        interaction="approach", preview=True,
    )
    app_precise = empty_move(
        robo2, "app_precise", right=start2.shifted(dz=0.06),
        interaction="approach", completion={"position_m": 0.002}, preview=True,
    )
    assert app_normal["policy"]["motion"]["window"]["near"]["velocity"] == pytest.approx(1.6)
    assert app_precise["policy"]["motion"]["window"]["near"]["velocity"] == pytest.approx(0.9)


def test_holding_string_shorthand_and_open_gripper_empty_inference_keep_free_moves_compact(tmp_path):
    def open_station(name):
        env, robo, host = station(tmp_path / name)
        env.grips = {"left": 1.0, "right": 1.0}
        host._primitives._grip_commands = {"left": 1.0, "right": 1.0}
        return env, robo, host

    # 1. Invalid holding values still reject cleanly before any dispatch.
    env_bad, robo_bad, _ = open_station("bad_holding")
    for bad in ("bogus", 123, []):
        rej = robo_bad.execute_motion(
            f"bad_{type(bad).__name__}", right=robo_bad.observe().right.shifted(dz=0.04),
            holding=bad,
        )
        assert rej["status"] == "rejected" and not env_bad.actions

    # 2. String shorthand "empty" and omitted holding={} on physically open gripper both
    #    select fast native free-space motion without mutating contract["holding"].
    env_l, robo_l, _ = open_station("left_hover")
    left_target = {"xyz": [-0.14, -0.15, 0.85], "angles": [90.0, 0.0, 0.0]}
    prev_str = robo_l.execute_motion(
        "left_key_hover_str", left=left_target, holding="empty", max_steps=30, preview=True,
    )
    assert prev_str["accepted"] and prev_str["partial_plan"] is False
    assert prev_str["policy"]["motion"] == {"mode": "native"}
    assert prev_str["planned_steps"] <= 22

    exec_omitted = robo_l.execute_motion("left_key_hover_v2", left=left_target, max_steps=30)
    assert exec_omitted["reached"] and exec_omitted.get("partial_plan") is not True
    assert exec_omitted["policy"]["motion"] == {"mode": "native"}
    assert exec_omitted["intent_steps_used"] <= 22
    assert robo_l.motion_status("left_key_hover_v2")["intent"]["holding"] == {}

    # 3. right_key_hover (~35 cm + >100 deg) and right_view_handoff finish in 1 call <= 30 steps.
    _, robo_r, _ = open_station("right_hover")
    right_target = {"xyz": [0.02, -0.02, 0.84], "angles": [70.0, 80.0, 0.0]}
    res_r = robo_r.execute_motion("right_key_hover", right=right_target, max_steps=30)
    assert res_r["reached"] and res_r.get("partial_plan") is not True
    assert res_r["intent_steps_used"] <= 25

    _, robo_vh, _ = open_station("right_handoff")
    handoff_target = {"xyz": [0.06, -0.08, 0.90], "angles": [45.0, 90.0, 0.0]}
    res_vh = robo_vh.execute_motion(
        "right_view_handoff", right=handoff_target, holding="empty", max_steps=30,
    )
    assert res_vh["reached"] and res_vh.get("partial_plan") is not True
    assert res_vh["intent_steps_used"] <= 26

    # 4. Omitted holding={} on a closed/holding gripper safely stays in carry phase,
    #    and holding="retain" shorthand expands to active arms and enforces retention.
    env_c, robo_c, host_c = station(tmp_path / "closed")
    host_c._primitives._grip_commands = {"right": 0.0}
    env_c.grips["right"] = 0.08
    carry_omitted = robo_c.execute_motion(
        "carry_closed_omitted", right=robo_c.observe().right.shifted(dz=0.05), preview=True,
    )
    assert carry_omitted["accepted"]
    assert carry_omitted["policy"]["motion"]["phase"] == "carry"
    retain_str = robo_c.execute_motion(
        "carry_retain_str", right=robo_c.observe().right.shifted(dz=0.05), holding="retain",
    )
    assert retain_str["reached"]
    assert robo_c.motion_status("carry_retain_str")["intent"]["holding"] == {"right": "retain"}


def test_continuous_line_chord_coalescing_eliminates_per_chord_ceil_inflation(tmp_path):
    import math

    env, robo, host = station(tmp_path)
    env.grips = {"left": 1.0, "right": 1.0}
    host._primitives._grip_commands = {"left": 1.0, "right": 1.0}

    target = {"xyz": [0.02, -0.02, 0.84], "angles": [70.0, 80.0, 0.0]}
    t_mat = np.eye(4)
    t_mat[:3, 3] = target["xyz"]
    t_mat[:3, :3] = kin.orientation(*target["angles"])
    p0 = motion._pose_of("right", env.joints["right"])
    d = float(np.linalg.norm(t_mat[:3, 3] - p0[:3, 3]))
    rot = kin.rotation_error(p0[:3, :3], t_mat[:3, :3])
    count = max(1, math.ceil(d / 0.02), math.ceil(rot / 8.0))

    # Measure what legacy per-chord ceil(dq / JOINT_STEP_RAD) would have emitted on these knots.
    q_cur = np.asarray(env.joints["right"], float)
    naive_steps = 0
    for i in range(1, count + 1):
        des = np.eye(4)
        des[:3, 3] = p0[:3, 3] + (t_mat[:3, 3] - p0[:3, 3]) * i / count
        des[:3, :3] = kin.slerp(p0[:3, :3], t_mat[:3, :3], i / count)
        sol = kin.solve("right", des, q_cur)
        assert sol.ok and not sol.collides
        dq = float(np.max(np.abs(np.asarray(sol.q, float) - q_cur)))
        naive_steps += max(1, math.ceil(dq / motion.JOINT_STEP_RAD))
        q_cur = np.asarray(sol.q, float)
    assert naive_steps > count + 4

    start_q = np.asarray(env.joints["right"], float)
    res = robo.execute_motion("coalesced_line", right=target, holding="empty", max_steps=30)
    assert res["reached"] and res["planned_path"] == "line"
    assert res["intent_steps_used"] < naive_steps - 4
    assert res["intent_steps_used"] <= 25

    prev_q = start_q
    prev_pose = motion._pose_of("right", prev_q)
    for act in env.actions:
        cur_q = np.asarray(act["right_arm_joint_state"], float)
        assert float(np.max(np.abs(cur_q - prev_q))) <= motion.JOINT_STEP_RAD + 1e-6
        assert not kin.clearance_violations("right", cur_q)
        cur_pose = motion._pose_of("right", cur_q)
        step_m = float(np.linalg.norm(cur_pose[:3, 3] - prev_pose[:3, 3]))
        step_deg = kin.rotation_error(prev_pose[:3, :3], cur_pose[:3, :3])
        assert step_m <= 0.02 + 0.001 + 1e-6
        assert step_deg <= 8.0 + 0.5 + 1e-6
        prev_q, prev_pose = cur_q, cur_pose


def test_pre_budget_simplify_tolerance_compresses_c2_carry_before_budget_error(tmp_path):
    env, robo, host = station(tmp_path)
    p_start = np.eye(4)
    p_start[:3, 3] = [-0.18, -0.15, 0.86]
    p_start[:3, :3] = kin.orientation(75.0, 0.0, 0.0)
    s_start = kin.solve("left", p_start, env.joints["left"])
    assert s_start.ok
    env.joints["left"] = list(s_start.q)
    host._primitives._grip_commands = {"left": 0.0, "right": 1.0}
    env.grips = {"left": 0.08, "right": 1.0}

    target = {"xyz": [0.00, -0.12, 0.90], "angles": [60.0, -45.0, 0.0]}
    # Without pre-budget simplify_tolerance=0.015, default 1e-4 C2 timing requires far more steps.
    ctrl = host._primitives._operations.intents.motion
    obs = ctrl.observe()
    t_mat = np.eye(4)
    t_mat[:3, 3] = target["xyz"]
    t_mat[:3, :3] = kin.orientation(*target["angles"])
    unsimplified = ctrl.plan(
        observation_id=obs["id"],
        targets={"left": {"xyz": t_mat[:3, 3].tolist(), "quat": list(kin.quat_wxyz(t_mat[:3, :3]))}},
        grippers={"left": "keep", "right": "keep"},
        keep_relative=False, max_steps=16, timeout_s=60.0,
        position_tolerance_m=0.01, orientation_tolerance_deg=5.0,
        max_step_m=0.02, max_step_deg=8.0, checkpoint_steps=8,
        require_inter_arm_clearance=False,
        motion={"velocity": 2.2, "acceleration": 20.0, "jerk": 950.0, "phase": "carry"},
        preview=True, simplify_tolerance=1e-4,
    )
    assert not unsimplified["accepted"] and "budget" in unsimplified["stop_reason"]
    assert unsimplified["trajectory"]["planned_steps"] > 16

    # With pre-budget simplify_tolerance=0.015, execute_motion succeeds cleanly in 1 call <= 16 steps.
    res = robo.execute_motion(
        "supported_transfer_hover", left=target, holding={"left": "retain"}, max_steps=16,
    )
    assert res["reached"] and res.get("partial_plan") is not True
    assert res["intent_steps_used"] <= 16
    assert res["trajectory"]["joint_path_simplification_tolerance_rad"] == pytest.approx(0.015)


def test_bracketed_prefix_search_packs_budget_and_reports_exact_resume_recommendation(tmp_path):
    # 1. C2 contact precision move with tight max_steps=20
    env_c2, robo_c2, _ = station(tmp_path / "c2")
    goal_c2 = robo_c2.observe().right.shifted(dz=0.06)
    first_c2 = robo_c2.execute_motion(
        "prefix_c2", right=goal_c2, interaction="contact", holding="empty",
        completion={"position_m": 0.0025}, max_steps=20,
    )
    assert first_c2["status"] == "paused" and first_c2["stop_reason"] == "segment_complete"
    assert 16 <= first_c2["intent_steps_used"] <= 20  # >= 80% packed, never > max_steps
    assert first_c2["full_path_estimate"] is not None and first_c2["full_path_estimate"] > 20
    assert first_c2["recommended_max_steps"] is not None
    assert first_c2["recommended_max_steps"] > first_c2["intent_steps_used"]
    assert robo_c2.motion_status("prefix_c2")["recommended_max_steps"] == first_c2["recommended_max_steps"]

    done_c2 = robo_c2.resume_motion("prefix_c2", max_steps=first_c2["recommended_max_steps"])
    assert done_c2["reached"] and done_c2["status"] == "completed"
    assert done_c2["intent_steps_used"] == len(env_c2.actions) <= first_c2["recommended_max_steps"]

    # 2. Native free-space move with tight max_steps=12 and max_steps=18 on a ~21-step move
    target_r = {"xyz": [0.02, -0.02, 0.84], "angles": [70.0, 80.0, 0.0]}
    for cap in (12, 18):
        env_n, robo_n, host_n = station(tmp_path / f"nat_{cap}")
        env_n.grips = {"left": 1.0, "right": 1.0}
        host_n._primitives._grip_commands = {"left": 1.0, "right": 1.0}
        first_n = robo_n.execute_motion(f"prefix_nat_{cap}", right=target_r, max_steps=cap)
        assert first_n["status"] == "paused" and first_n["stop_reason"] == "segment_complete"
        assert int(0.8 * cap) <= first_n["intent_steps_used"] <= cap
        assert first_n["full_path_estimate"] is not None and first_n["full_path_estimate"] > cap
        assert first_n["recommended_max_steps"] is not None
        done_n = robo_n.resume_motion(f"prefix_nat_{cap}", max_steps=first_n["recommended_max_steps"])
        assert done_n["reached"] and done_n["status"] == "completed"
        assert done_n["intent_steps_used"] == len(env_n.actions) <= first_n["recommended_max_steps"]

