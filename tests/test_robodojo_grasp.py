"""Bounded grasp effects, evidence limits and interruption behavior."""

import copy

import numpy as np
import pytest

from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from tests.test_coordination import JointEnvironment

TARGET = [.25, -.15, .815]
POSE = dict(arm="right", pitch=90, yaw=0, roll=0)


class GraspEnvironment(JointEnvironment):
    def __init__(self, *, empty=False, slip=False, stall=False, approach_lag=False,
                 approach_gain=.8):
        super().__init__()
        self.grips = {arm: 1. for arm in kin.ARMS}
        self.empty, self.slip, self.stall = empty, slip, stall
        self.approach_lag = approach_lag
        self.approach_gain = approach_gain

    def step(self, action):
        before = copy.deepcopy(self.joints)
        super().step(action)
        tcp = kin.tcp(kin.fk("right", self.joints["right"]))
        if self.approach_lag and np.linalg.norm(tcp - (np.array(TARGET) + [0, 0, .02])) < .001:
            # A one-off servo lag at the transition from coarse to fine motion.
            old_q = np.asarray(before["right"])
            self.joints["right"] = (
                old_q + self.approach_gain * (np.asarray(self.joints["right"]) - old_q))
            self.approach_lag = False
        if self.stall and tcp[2] < .845:
            self.joints = before
        if action["right_ee_joint_state"][0] == 0:
            if self.empty or self.slip and tcp[2] > TARGET[2] + .006:
                self.grips["right"] = 0.
        return self.get_obs(), 0., False, {"status": self.get_status()}


def station(tmp_path, **kwargs):
    env = GraspEnvironment(**kwargs)
    host = NativeToolkit(env, None, tmp_path / "native")
    return env, RobodojoOfficialColourRobot(NativeBackend(host)), host


def test_preview_checks_whole_attempt_without_moving_or_claiming_grasp(tmp_path):
    env, robo, _ = station(tmp_path)
    result = robo.try_grasp(TARGET, **POSE, preview=True)
    assert result["accepted"], result
    assert not env.actions
    assert result["planned_step_bound"] <= 64
    assert not result["retained"] and not result["grasp_verified"]
    assert {p["phase"] for p in result["phases"]} == {"transit", "approach", "contact", "probe"}


def test_probe_preserves_squeeze_and_requires_visual_confirmation(tmp_path):
    env, robo, _ = station(tmp_path)
    result = robo.try_grasp(TARGET, **POSE)
    assert result["status"] == "completed", result
    assert result["retained"] and result["needs_visual_confirmation"]
    assert not result["grasp_verified"]
    assert result["steps_used"] == len(env.actions) <= result["planned_step_bound"] <= 64
    close = next(i for i, a in enumerate(env.actions) if a["right_ee_joint_state"] == [0.])
    # Closure only begins at the grasp pose; every probe action preserves squeeze.
    tcp = kin.tcp(kin.fk("right", env.actions[close]["right_arm_joint_state"]))
    assert tcp == pytest.approx(TARGET, abs=.0025)
    assert all(a["right_ee_joint_state"] == [0.] for a in env.actions[close:])
    assert all(a["left_ee_joint_state"] == [1.] for a in env.actions)
    for before, after in zip(env.actions[close:], env.actions[close+1:], strict=False):
        delta = (kin.tcp(kin.fk("right", after["right_arm_joint_state"]))
                 - kin.tcp(kin.fk("right", before["right_arm_joint_state"])))
        assert np.linalg.norm(delta) <= .0045


@pytest.mark.parametrize("kwargs", [{"max_steps": 1}, {"xyz": [.25, -.15, .7]}])
def test_rejected_attempt_never_opens_or_moves(tmp_path, kwargs):
    env, robo, _ = station(tmp_path)
    result = robo.try_grasp(**dict(xyz=TARGET, **POSE) | kwargs)
    assert result["status"] == "rejected" and not result["retained"]
    assert result["steps_used"] == 0 and not env.actions


def test_empty_close_does_not_lift_or_search(tmp_path):
    env, robo, _ = station(tmp_path, empty=True)
    result = robo.try_grasp(TARGET, **POSE)
    assert result["stop_reason"] == "empty_grasp"
    assert not result["retained"] and result["phase"] == "close"
    assert next(p for p in result["phases"] if p["phase"] == "probe")["steps_used"] == 0
    assert kin.tcp(kin.fk("right", env.joints["right"])) == pytest.approx(TARGET, abs=.0025)


def test_slipping_probe_stops_and_does_not_open_or_regrasp(tmp_path):
    env, robo, _ = station(tmp_path, slip=True)
    result = robo.try_grasp(TARGET, **POSE)
    assert result["stop_reason"] == "grasp_lost_during_probe", result
    assert not result["retained"] and not result["retry_performed"]
    assert kin.tcp(kin.fk("right", env.joints["right"]))[2] < TARGET[2] + .012
    assert env.actions[-1]["right_ee_joint_state"] == [0.]


def test_stalled_descent_stops_before_closure(tmp_path):
    env, robo, _ = station(tmp_path, stall=True)
    result = robo.try_grasp(TARGET, **POSE)
    assert result["stop_reason"] == "tracking_error", result
    assert not result["retained"]
    assert all(a["right_ee_joint_state"] != [0.] for a in env.actions)
    assert result.get("local_corrections", 0) <= 2


def test_small_approach_endpoint_lag_settles_within_correction_budget(tmp_path):
    env, robo, _ = station(tmp_path, approach_lag=True)
    result = robo.try_grasp(TARGET, **POSE)
    assert result["status"] == "completed", result
    assert 1 <= result["local_corrections"] <= 2
    approach = next(p for p in result["phases"] if p["phase"] == "approach")
    assert approach["steps_used"] == approach["planned_steps"] + 1
    assert result["steps_used"] == len(env.actions) <= result["planned_step_bound"]


def test_phase_arrival_includes_ik_residual_in_the_same_error_check(tmp_path, monkeypatch):
    line_path = kin.line_path

    def biased_near_point(arm, q, target, **kwargs):
        if np.linalg.norm(target[:3, 3] - (np.array(TARGET) + [0, 0, .02])) < .0001:
            target = target.copy()
            target[2, 3] += .0007
        return line_path(arm, q, target, **kwargs)

    monkeypatch.setattr(kin, "line_path", biased_near_point)
    env, robo, _ = station(tmp_path, approach_lag=True, approach_gain=.85)
    result = robo.try_grasp(TARGET, **POSE)
    assert result["status"] == "completed", result
    assert result["local_corrections"] == 1
    assert result["tracking"]["decision"] == "advance"


def test_existing_grasp_is_not_opened_on_reentry(tmp_path):
    env, robo, _ = station(tmp_path)
    assert robo.try_grasp(TARGET, **POSE)["retained"]
    count = len(env.actions)
    result = robo.try_grasp(TARGET, **POSE)
    assert result["stop_reason"] == "gripper_may_hold_object"
    assert len(env.actions) == count


def test_cancelled_attempt_has_partial_receipt_and_cannot_replay(tmp_path):
    env, _, host = station(tmp_path)
    step = env.step

    def cancel_after_effect(action):
        result = step(action)
        if len(env.actions) == 2:
            host._primitives.cancelled.set()
        return result

    env.step = cancel_after_effect
    context = dict(session_id=host.session_id, action_id="grasp-interrupted")
    result = host.execute_tool_with_context("try_grasp", dict(xyz=TARGET, **POSE), context)
    assert len(env.actions) == 2
    assert result["action_receipt"]["status"] == "partial"
    assert result["action_receipt"]["effect_known"]
    host._primitives.cancelled.clear()
    with pytest.raises(ValueError, match="already used"):
        host.execute_tool_with_context("try_grasp", dict(xyz=TARGET, **POSE), context)
    assert len(env.actions) == 2


def test_episode_ending_mid_attempt_stops_at_last_native_effect(tmp_path):
    env, robo, _ = station(tmp_path)
    original = env.step

    def end_early(action):
        result = original(action)
        if len(env.actions) == 2:
            env.step_limit = 2
            result = (*result[:3], {"status": env.get_status()})
        return result

    env.step = end_early
    result = robo.try_grasp(TARGET, **POSE)
    assert result["stop_reason"] == "episode_native_step_budget"
    assert result["steps_used"] == len(env.actions) == 2
    assert not result["retained"]
