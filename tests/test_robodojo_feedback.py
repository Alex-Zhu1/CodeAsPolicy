"""Shared feedback behavior through public APIs, including bounded continuation."""

import numpy as np
import pytest

from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from tests.test_coordination import JointEnvironment


def station(tmp_path, *, lag_arm="right", lag_actions=1):
    env = JointEnvironment()
    original = env.step

    def lag(action):
        before = np.asarray(env.joints[lag_arm])
        original(action)
        if len(env.actions) <= lag_actions:
            goal = np.asarray(env.joints[lag_arm])
            env.joints[lag_arm] = (before + .4 * (goal - before)).tolist()
        return env.get_obs(), 0., False, {"status": env.get_status()}

    env.step = lag
    host = NativeToolkit(env, None, tmp_path / "native")
    return env, RobodojoOfficialColourRobot(NativeBackend(host)), host


def test_native_lag_holds_current_target_then_advances(tmp_path):
    env, robo, _ = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06)
    result = robo.move_to(target.xyz, path="line", motion="native")
    assert result.reached, result
    assert result.tracking["correction_steps"] == 1
    assert env.actions[0] == env.actions[1]
    assert env.actions[2] != env.actions[1]
    assert result.steps_used == len(env.actions)


def test_correction_never_borrows_budget_for_unexecuted_path(tmp_path):
    env, robo, _ = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06)
    preview = robo.move_to(target.xyz, path="line", motion="native", preview=True)
    env.step_limit = preview["planned_steps"]
    result = robo.move_to(target.xyz, path="line", motion="native")
    assert result.status == "partial" and result.stop_reason == "tracking_error"
    assert result.tracking["cause"] == "insufficient_correction_budget"
    assert result.steps_used == len(env.actions) == 1


def test_dual_correction_survives_checkpoint_without_replaying_prior_path(tmp_path):
    env, robo, _ = station(tmp_path)
    o = robo.observe()
    result = robo.move_both(o.left.shifted(dz=.06), o.right.shifted(dz=.06),
                            motion="native", checkpoint_steps=1, max_step_m=.02)
    assert result.reached, result
    assert result.tracking["correction_steps"] == 1
    assert env.actions[0] == env.actions[1]
    assert all(a["left_ee_joint_state"] == [.3] for a in env.actions)
    assert result.steps_used == len(env.actions)
    # The corrective checkpoint is a new native action, not a replayed dispatch.
    assert len({r.action_id for r in robo._ledger if r.stateful}) == len(env.actions)


def test_progressing_lag_cannot_consume_unbounded_corrections(tmp_path):
    env, robo, _ = station(tmp_path, lag_actions=100)
    result = robo.move_to(robo.observe().right.shifted(dz=.12).xyz,
                          path="line", motion="native")
    assert not result.reached and result.stop_reason == "tracking_error"
    assert result.tracking["correction_steps"] <= 2
    assert result.steps_used < 10


def test_no_progress_stops_without_a_corrective_dispatch(tmp_path):
    env, robo, _ = station(tmp_path, lag_actions=0)
    env.frozen = "right"
    result = robo.move_to(robo.observe().right.shifted(dz=.06).xyz,
                          path="line", motion="native")
    assert not result.reached
    assert result.tracking["correction_steps"] == 0
    assert result.steps_used == 1
    assert result.tracking["contact_inferred"] is False


def test_cancel_during_correction_keeps_partial_receipt_and_refuses_replay(tmp_path):
    env, robo, host = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06).xyz
    original = env.step

    def cancel(action):
        value = original(action)
        if len(env.actions) == 2:
            host._primitives.cancelled.set()
        return value

    env.step = cancel
    args = dict(xyz=target, arm="right", path="line", motion={"mode": "native"})
    context = dict(session_id=host.session_id, action_id="corrective-interruption")
    result = host.execute_tool_with_context("move_pose", args, context)
    assert result["action_receipt"]["status"] == "partial"
    assert result["action_receipt"]["effect_known"]
    assert len(env.actions) == 2 and env.actions[0] == env.actions[1]
    host._primitives.cancelled.clear()
    with pytest.raises(ValueError, match="already used"):
        host.execute_tool_with_context("move_pose", args, context)
    assert len(env.actions) == 2


def test_timed_endpoint_corrections_use_resting_blocks_without_native_fallback(tmp_path):
    from pyrualean.hosts import robodojo_trajectory as trajectory
    from tests.test_robodojo_trajectory import TimedEnvironment

    env = TimedEnvironment()
    original = env.step

    def lag_until_rest(action):
        before = np.asarray(env.joints["right"])
        original(action)
        block = action[trajectory.TRAJECTORY_KEY]
        goal = np.asarray(env.joints["right"])
        gain = .4 if np.max(np.abs(block["velocities"][-1])) < 1e-10 else 0.
        env.joints["right"] = (before + gain * (goal - before)).tolist()
        return env.get_obs(), 0., False, {"status": env.get_status()}

    env.step = lag_until_rest
    host = NativeToolkit(env, None, tmp_path / "native")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))
    result = robo.move_to(robo.observe().right.xyz, pitch=0, yaw=0, roll=2,
                          path="line", motion="approach")
    assert result.reached, result
    assert result.tracking["correction_steps"] == result.trajectory["settle_steps"] == 2
    for action in env.wire_actions[-2:]:
        block = action[trajectory.TRAJECTORY_KEY]
        assert np.max(np.abs(block["velocities"])) == 0.
        assert np.max(np.ptp(block["positions"], axis=0)) == 0.


@pytest.mark.parametrize("angle,can_correct,expected", [
    (4., True, "advance"), (9., True, "correct"), (9., False, "stop")])
def test_rotation_progress_and_trajectory_continuity(angle, can_correct, expected):
    from pyrualean.hosts import robodojo_kinematics as kin
    from pyrualean.hosts.robodojo_feedback import ProgressTracker, TrackingLimits

    goal = np.eye(4)
    before, actual = goal.copy(), goal.copy()
    before[:3, :3] = kin.orientation(0, 12., 0)
    actual[:3, :3] = kin.orientation(0, angle, 0)
    # Match the target's rotation convention; translation is constant.
    goal[:3, :3] = kin.orientation(0, 0, 0)
    tracker = ProgressTracker()
    decision = tracker.assess({"left": before}, {"left": actual}, {"left": goal},
                              TrackingLimits(.01, 5.), spare_steps=1, can_correct=can_correct)
    assert decision == expected
    if not can_correct:
        assert tracker.last["cause"] == "continuous_trajectory"
