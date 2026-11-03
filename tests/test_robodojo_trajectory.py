"""Timed commands through the public API and the real standalone client adapter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hosts import robodojo_trajectory as trajectory
from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit
from pyrualean.motion import MotionProfile, MotionWindow, motion_from_payload, resolve_motion
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from tests.test_coordination import JointEnvironment
from tests.test_observation_transport import deploy

CLOCK = dict(schema=trajectory.BACKEND, physics_dt_s=.004, substeps=10,
             time_domain="simulation", position_velocity_targets=True)


def joints(value):
    return {"left": [0.] * 6, "right": [value, 0., 0., 0., 0., 0.]}


def samples(ways, key):
    return np.array([row for way in ways for row in way["trajectory"][key]])


def test_spline_limits_hold_between_native_ticks_and_across_knots():
    limits = dict(velocity=.5, acceleration=1.2, jerk=6.)
    ways, info = trajectory.parameterize(joints(0), [joints(.1), joints(.3), joints(.4)],
                                         CLOCK, limits)
    q, v = samples(ways, "positions"), samples(ways, "velocities")
    assert q[-1] == pytest.approx([0.] * 6 + [.4, 0., 0., 0., 0., 0.])
    assert v[-1] == pytest.approx([0.] * 12)
    assert np.all(np.diff(q[:, 6]) >= -1e-12)
    assert np.all(v[(q[:, 6] > .05) & (q[:, 6] < .35), 6] > 0)
    # Independent finite differences, with rest before/after the curve, include
    # every segment junction and the start/end acceleration transitions.
    padded = np.vstack([np.zeros((3, 12)), q, np.repeat(q[-1:], 3, axis=0)])
    for order, (name, bound) in enumerate(limits.items(), 1):
        measured = np.max(abs(np.diff(padded, n=order, axis=0))) / .004**order
        assert measured <= bound * 1.001
        assert info["derivative_bounds"][name] <= bound * (1 + 1e-9)
    assert info["planned_duration_s"] == pytest.approx(len(ways) * .04)
    slower, _ = trajectory.parameterize(joints(0), [joints(.1), joints(.3), joints(.4)],
                                        CLOCK, dict(velocity=.05, acceleration=.1, jerk=.5))
    assert len(slower) > len(ways)


def test_reversal_and_repeated_points_stay_within_joint_ranges():
    ways, _ = trajectory.parameterize(joints(0), [joints(.2), joints(.2), joints(-.1)], CLOCK)
    q = samples(ways, "positions")[:, 6]
    assert q.min() >= -.1 - 1e-12 and q.max() <= .2 + 1e-12
    assert q[-1] == pytest.approx(-.1)
    holds, _ = trajectory.parameterize(joints(0), [joints(0)], CLOCK)
    assert len(holds) == 1 and not np.any(samples(holds, "velocities"))
    with pytest.raises(ValueError, match="budget"):
        trajectory.parameterize(joints(0), [joints(.5)], CLOCK, max_actions=1)


def test_redundant_geometric_samples_do_not_change_timing_or_motion():
    reference, _ = trajectory.parameterize(joints(0), [joints(1.)], CLOCK)
    for count in (5, 10, 20, 50):
        ways, info = trajectory.parameterize(
            joints(0), [joints(float(v)) for v in np.linspace(0, 1, count+1)[1:]], CLOCK)
        assert len(ways) == len(reference)
        assert info["timed_segments"] == 1
        assert samples(ways, "positions") == pytest.approx(samples(reference, "positions"))


def test_small_ik_residuals_do_not_multiply_duration_or_erase_reversals():
    x = np.linspace(0., 1.8, 31)
    q = np.zeros((len(x), 12))
    q[:, 0], q[:, 1] = x, .001 * np.sin(x / 1.8 * np.pi)
    knots = trajectory._simplify(q)
    for a, b in zip(knots[:-1], knots[1:]):
        chord = q[b] - q[a]
        fraction = np.clip((q[a:b+1] - q[a]) @ chord / (chord @ chord), 0., 1.)
        assert np.max(abs(q[a:b+1] - q[a] - fraction[:, None] * chord)) <= 1e-4
    initial = dict(left=q[0, :6], right=q[0, 6:])
    path = [dict(left=row[:6], right=row[6:]) for row in q[1:]]
    limits = dict(velocity=1.8, acceleration=10., jerk=120.)
    ways, info = trajectory.parameterize(initial, path, CLOCK, limits)
    original, _ = trajectory.parameterize(initial, path, CLOCK, limits,
                                          _simplify_tolerance=1e-10)
    assert len(ways) < len(original) / 2
    assert info["timed_segments"] < info["input_segments"]
    padded = np.vstack([np.zeros((3, 12)), samples(ways, "positions"),
                        np.repeat(q[-1:], 3, axis=0)])
    for order, bound in enumerate(limits.values(), 1):
        assert np.max(abs(np.diff(padded, n=order, axis=0))) / .004**order <= bound * 1.001
    # A turn inside the endpoint interval must survive even though all its
    # points are exactly collinear. Phase boundaries must also survive.
    assert trajectory._simplify(np.array([[0.], [.8], [.2], [1.]])).tolist() == [0, 1, 2, 3]
    assert 11 in trajectory._simplify(q, preserve={11})


def test_simplification_falls_back_when_geometry_needs_the_original_knots():
    initial = joints(0.)
    peak, end = joints(.2), joints(.4)
    peak["right"][1] = 5e-5

    def check(values):
        x, y = values["right"][:2]
        if abs(y - .00025 * min(x, .4 - x)) > 1e-7:
            raise ValueError("timed blending leaves Cartesian path tolerance")

    ways, info = trajectory.parameterize(initial, [peak, end], CLOCK, check=check)
    assert info["joint_path_simplification_tolerance_rad"] == 1e-10
    assert info["timed_segments"] == 2
    for row in samples(ways, "positions"):
        check(dict(left=row[:6], right=row[6:]))


def test_local_retiming_has_continuous_knot_derivatives_and_bounded_motion():
    q = np.array([[0.]*12, [.2, .08] + [0.]*10, [.25, .3] + [0.]*10,
                  [.1, .5] + [0.]*10, [.1, .6] + [0.]*10])
    p, derivatives, durations, bounds, count = trajectory._retime(q, MotionProfile(), .04)
    for d in derivatives[:2]:
        assert d[:-1, -1] == pytest.approx(d[1:, 0], abs=1e-10)
        assert d[[0, -1], [0, -1]] == pytest.approx(np.zeros((2, 12)), abs=1e-10)
    assert np.any(abs(derivatives[1][:-1, -1]) > .01)  # no forced acceleration reset
    assert np.all(bounds <= np.array([1., 4., 40.]) * (1 + 1e-8))
    assert count * .04 == pytest.approx(sum(durations))
    # Independently evaluate densely; hull certification must dominate actual derivatives.
    for i, d in enumerate(derivatives):
        for segment in range(len(p)):
            u = np.linspace(0, 1, 501)
            values = trajectory._evaluate(np.repeat(d[segment:segment+1], len(u), axis=0), u)
            assert np.max(abs(values)) <= bounds[segment, i] + 1e-10


def test_budget_rejection_keeps_estimate_and_limiting_factor():
    with pytest.raises(trajectory.TrajectoryBudgetError) as error:
        trajectory.parameterize(joints(0), [joints(1.)], CLOCK, max_actions=2)
    estimate = error.value.trajectory
    assert estimate["planned_steps"] > 2 and not estimate["fits_budget"]
    assert estimate["limiting_factor"] in {"velocity", "acceleration", "jerk"}


@pytest.mark.parametrize("at", ["start", "end"])
def test_window_preserves_local_bounds_and_continuity_without_boundary_stop(at):
    motion = MotionWindow(resolve_motion("transit"), resolve_motion("precision"), .03, at)
    q = np.array([[*joints(v)["left"], *joints(v)["right"]]
                  for v in np.linspace(0, .8, 41)])
    limits, near, length = trajectory._path_limits(q, motion)
    assert near.any() and (~near).any() and length > .03
    boundary = np.flatnonzero(np.diff(near))[0] + 1
    knots = trajectory._simplify(q, preserve={boundary})
    assert boundary in knots
    local = np.array([limits[a:b].min(axis=0) for a,b in zip(knots[:-1], knots[1:])])
    _, derivatives, _, bounds, _ = trajectory._retime(q[knots], motion, .04, limits=local)
    assert np.all(bounds <= local * (1 + 1e-8))
    for derivative in derivatives[:2]:
        assert derivative[:-1, -1] == pytest.approx(derivative[1:, 0], abs=1e-10)
    assert np.max(abs(derivatives[0][0, -1])) > .01
    path = [joints(v) for v in np.linspace(0, .8, 41)[1:]]
    ways, info = trajectory.parameterize(joints(0), path, CLOCK, motion.payload())
    slow, _ = trajectory.parameterize(joints(0), path, CLOCK, motion.near.payload())
    assert len(ways) < len(slow)
    for region in info["window"]["regions"].values():
        for name, bound in region["derivative_bounds"].items():
            assert bound <= region["profile"][name] * (1 + 1e-8)
    assert samples(ways, "velocities")[-1] == pytest.approx(np.zeros(12))
    # Independently differentiate emitted positions; check stencils wholly
    # within the slow region against its own tighter bounds, not the far caps.
    positions = samples(ways, "positions")
    padded = np.vstack([np.repeat(q[:1], 3, axis=0), positions,
                        np.repeat(positions[-1:], 3, axis=0)])
    in_near = (padded[:, 6] <= q[boundary, 6] if at == "start"
               else padded[:, 6] >= q[boundary, 6])
    for order, name in enumerate(("velocity", "acceleration", "jerk"), 1):
        mask = np.all(np.lib.stride_tricks.sliding_window_view(in_near, order+1), axis=1)
        measured = abs(np.diff(padded, n=order, axis=0)[mask]).max() / .004**order
        assert measured <= getattr(motion.near, name) * 1.002


def test_window_crossing_segment_and_zero_translation_use_near_limits():
    motion = MotionWindow(resolve_motion("transit"), resolve_motion("precision"), .0001)
    q = np.array([[*joints(v)["left"], *joints(v)["right"]] for v in [0, .3, .8]])
    _, near, _ = trajectory._path_limits(q, motion)
    assert near.tolist() == [False, True]  # whole segment crossing the tiny window
    _, near, _ = trajectory._path_limits(np.zeros((2, 12)), motion)
    assert near.tolist() == [True]


def test_invalid_motion_windows_reject_before_dispatch(tmp_path):
    env, robo, _ = station(tmp_path)
    for value in [0, -1, True, float("nan"), float("inf"), 1.1]:
        with pytest.raises(ValueError):
            robo.motion_window(distance_m=value)
    for kwargs in [dict(at="middle"), dict(far="precision", near="transit"), dict(far=None),
                   dict(near=robo.motion_window())]:
        with pytest.raises(ValueError):
            robo.motion_window(**kwargs)
    payload = robo.motion_window().payload()
    assert motion_from_payload(payload).payload() == payload
    with pytest.raises(ValueError):
        motion_from_payload({**payload, "velocity": 2})
    assert not env.actions


def test_window_preview_execution_and_partial_stop(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    motion = robo.motion_window(far="carry", near="precision", distance_m=.015)
    target = obs.right.shifted(dz=.10)
    preview = robo.move_to(target.xyz, motion=motion, preview=True)
    assert preview["accepted"] and not env.actions
    assert "near" in preview["trajectory"]["window"]["regions"]
    actual = robo.move_to(target.xyz, motion=motion)
    assert actual.reached and actual.steps_used == preview["planned_steps"]
    env.frozen = "right"
    partial = robo.move_to(obs.right.xyz, motion=motion)
    assert partial.status == "partial" and partial.steps_used < partial.trajectory["planned_steps"]


def test_dual_window_preview_and_budget_rejection_do_not_execute(tmp_path):
    env, robo, toolkit = station(tmp_path)
    target = robo.observe().right.shifted(dz=.1)
    motion = robo.motion_window(at="start")
    preview = robo.move_both(right=target, motion=motion, max_steps=200, preview=True)
    assert preview["accepted"] and not env.actions
    result = robo.move_both(right=target, motion=motion, max_steps=5)
    assert result.status == "rejected" and not env.actions


def test_start_limit_tolerance_only_allows_holding_or_recovering_measured_error():
    start = joints(0)
    start["right"][2] = -.0003155
    trajectory.check_joints(start, start=start)
    recovered = copy.deepcopy(start)
    recovered["right"][2] = 0.
    trajectory.check_joints(recovered, start=start)
    for value in (-.0004, -.003):
        worse = copy.deepcopy(start)
        worse["right"][2] = value
        with pytest.raises(ValueError, match="joint limits"):
            trajectory.check_joints(worse, start=start)
    far = copy.deepcopy(start)
    far["right"][2] = -.003
    with pytest.raises(ValueError, match="joint limits"):
        trajectory.check_joints(far, start=far)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), "slow"])
def test_invalid_limits_reject(value):
    for name in ("velocity", "acceleration", "jerk"):
        with pytest.raises(ValueError):
            MotionProfile(**{name: value})


class TimedEnvironment(JointEnvironment):
    """Execute real adapter expansion, then ideal joint tracking, without Isaac."""

    def __init__(self, adapter="PyRUA_Lean"):
        super().__init__()
        self.adapter = deploy(adapter)
        self.obs_manager = SimpleNamespace(dt=.004, collect_interval=10.)
        self.sim = SimpleNamespace(cfg=SimpleNamespace(decimation=1))
        self.robot_manager = SimpleNamespace(
            robot_list=[SimpleNamespace(type="target", arm_name=f"{a}_arm_joint_state",
                                        arm_joint_indices=list(range(6)))
                        for a in ("left", "right")],
            process_name=lambda name: name, control_robot=lambda: None,
        )
        self.wire_actions, self.expanded = [], []

    def get_status(self):
        result = super().get_status()
        capability = self.adapter._motion_backend(self)
        if capability:
            result["motion_backend"] = capability
        return result

    def process_control_info(self, info, env_idx):
        return [copy.deepcopy(info) for _ in range(10)]

    def take_action(self, action):
        info = {key: {"position": value} for key, value in action.items()}
        info["support_arm"] = {"position": [42]}
        self.expanded.extend(self.process_control_info(info, 0))
        self.last_result = super().step(action)

    def step(self, action):
        self.wire_actions.append(copy.deepcopy(action))
        self.adapter._take_action(self, action)
        return self.last_result


def station(tmp_path, env=None):
    env = env or TimedEnvironment()
    host = NativeToolkit(env, None, tmp_path)
    return env, RobodojoOfficialColourRobot(NativeBackend(host)), host


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
def test_timed_dual_route_uses_substeps_and_preserves_gripper_support_arm(tmp_path, adapter):
    env, robo, host = station(tmp_path, TimedEnvironment(adapter))
    caps = robo.motion_capabilities()
    assert caps["speed_control"] and caps["continuous_trajectory"]
    assert not any(caps[k] for k in ("force_control", "force_feedback", "impedance_control",
                                     "gripper_force_control", "measured_speed_guarantee"))
    obs = robo.observe()
    host._primitives._grip_commands = {"left": 0., "right": 0.}
    result = robo.move_both(obs.left.shifted(dz=.06), obs.right.shifted(dz=.06),
                            via=[{"left": obs.left.shifted(dz=.03),
                                  "right": obs.right.shifted(dz=.03)}],
                            motion=robo.motion_profile(.8, 3., 30.), max_steps=250,
                            keep_relative=True)
    assert result.reached, result
    assert result.trajectory["planned_steps"] == result.steps_used == len(env.actions)
    assert "substep_arm_table" in result.checked
    assert len(env.expanded) == 10 * result.steps_used
    for action, chunk in zip(env.wire_actions,
                             [env.expanded[i:i+10] for i in range(0, len(env.expanded), 10)],
                             strict=True):
        payload = action[trajectory.TRAJECTORY_KEY]
        for i, point in enumerate(chunk):
            assert point["right_arm_joint_state"]["position"] == payload["positions"][i][6:]
            assert point["right_arm_joint_state"]["velocity"] == payload["velocities"][i][6:]
            assert point["right_ee_joint_state"]["position"] == [0.]
            assert point["support_arm"]["position"] == [42]
    # No per-action 20% arm hold, and no forced stop at the via waypoint.
    mid = env.wire_actions[len(env.wire_actions) // 2][trajectory.TRAJECTORY_KEY]
    assert mid["positions"][-1] != mid["positions"][-2]
    assert np.linalg.norm(mid["velocities"][-1]) > .001
    assert "process_control_info" not in vars(env)  # temporary override restored


def test_single_arm_automatically_uses_live_timing(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    result = robo.move_to(obs.right.shifted(dz=.04).xyz, path="line")
    assert result.reached and result.trajectory, result
    assert all(a["left_arm_joint_state"] == [0.] * 6 for a in env.actions)
    assert env.wire_actions[0].get(trajectory.TRAJECTORY_KEY)


def test_phase_preview_never_moves_and_estimates_match_execution(tmp_path):
    env, robo, host = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06).xyz
    native_before, records_before = env.get_status()["step"], len(host.state.records)
    fast = robo.move_to(target, motion="transit", preview=True)
    slow = robo.move_to(target, motion="precision", preview=True)
    assert fast["accepted"] and slow["accepted"] and fast["preview"]
    assert fast["planned_steps"] < slow["planned_steps"]
    assert env.get_status()["step"] == native_before and len(host.state.records) == records_before
    assert not env.actions and not robo._receipts
    result = robo.move_to(target, motion="transit")
    assert result.reached and result.steps_used == fast["planned_steps"]
    assert result.trajectory["profile"]["phase"] == "transit"
    with pytest.raises(ValueError, match="motion"):
        robo.move_to(target, motion="fastest")


def test_dual_preview_creates_no_executable_plan_or_actions(tmp_path):
    env, robo, host = station(tmp_path)
    target = robo.observe().right.shifted(dz=.04)
    estimate = robo.move_both(right=target, motion="transit", max_steps=200, preview=True)
    assert estimate["accepted"] and estimate["preview"]
    assert not env.actions and not robo._receipts
    assert "id" not in estimate and estimate["replan_before_dispatch"]
    actual = robo.move_both(right=target, motion="transit", max_steps=200)
    assert actual.reached and actual.steps_used == estimate["planned_steps"]


def test_native_mode_uses_original_waypoints_and_keeps_squeeze(tmp_path):
    env, robo, host = station(tmp_path)
    host._primitives._grip_commands = {"left": 0., "right": 1.}
    env.grips["left"] = .4
    target = robo.observe().right.shifted(dz=.04)
    native = robo.move_to(target.xyz, motion="native", preview=True)
    smooth = robo.move_to(target.xyz, motion="approach", preview=True)
    assert native["accepted"] and native["planned_steps"] < smooth["planned_steps"]
    assert not env.wire_actions
    result = robo.move_to(target.xyz, motion="native")
    assert result.reached and result.trajectory is None
    assert all("__pyrualean_trajectory__" not in a for a in env.wire_actions)
    assert all(a["left_ee_joint_state"] == [0.] for a in env.wire_actions)
    target = robo.observe().right.shifted(dz=.02)
    assert robo.move_both(right=target, motion="native").reached


def test_native_preview_rejects_over_budget_without_dispatch(tmp_path):
    env, robo, _ = station(tmp_path)
    env.step_limit = 1
    result = robo.move_to(robo.observe().right.shifted(dz=.06).xyz,
                          motion="native", preview=True)
    assert not result["accepted"] and result["planned_steps"] > 1
    assert not env.wire_actions


def test_native_tracking_miss_stops_before_later_waypoints_and_settling(tmp_path):
    env, robo, _ = station(tmp_path)
    target = robo.observe().right.shifted(dz=.06).xyz
    estimate = robo.move_to(target, path="line", motion="native", preview=True)
    assert estimate["planned_steps"] >= 3
    env.frozen = "right"
    result = robo.move_to(target, path="line", motion="native")
    assert not result.reached and result.status == "partial"
    assert result.stop_reason == "tracking_error"
    assert result.steps_used == len(env.wire_actions) == 1
    receipt = robo.action_receipt()
    assert receipt["status"] == "partial" and receipt["execution_complete"]


def test_live_early_end_stops_the_trajectory_without_claiming_budget_exhaustion(tmp_path):
    env, robo, _ = station(tmp_path)
    original = env.get_status

    def status():
        value = original()
        if value["step"] >= 2:
            value.update(over=True, native_step_limit=value["step_limit"], step_limit=value["step"])
        return value

    env.get_status = status
    result = robo.move_to(robo.observe().right.shifted(dz=.06).xyz, motion="approach")
    assert result.status == "partial" and result.stop_reason == "episode_ended"
    assert result.truncated and result.steps_used == len(env.wire_actions) == 2
    assert not hasattr(result, "success")


def test_single_preview_budget_rejection_keeps_estimate_without_dispatch(tmp_path):
    env, robo, _ = station(tmp_path)
    env.step_limit = 1
    estimate = robo.move_to(robo.observe().right.shifted(dz=.06).xyz,
                            motion="transit", preview=True)
    assert estimate["preview"] and not estimate["accepted"]
    assert estimate["trajectory"]["planned_steps"] > 1
    assert not env.actions


def test_local_blending_refits_without_relaxing_cartesian_tolerance(tmp_path):
    from pyrualean.hosts import robodojo_tools as tools

    env, robo, _ = station(tmp_path)
    start = np.array(robo.state().right.tcp_pos)
    target = np.array([.3418, -.0644, .8481])
    args = dict(arm="right", pitch=75, yaw=0, gripper=robo.OPEN, path="line", motion="transit")
    estimate = robo.move_to(target, **args, preview=True)
    assert estimate["accepted"] and not env.actions
    assert estimate["trajectory"]["geometry_refits"] > 0
    result = robo.move_to(target, **args)
    assert result.reached and result.steps_used < 100
    direction = target-start
    for action in env.wire_actions:
        for q in action[trajectory.TRAJECTORY_KEY]["positions"]:
            tcp = tools._pose_of("right", q[6:])[:3, 3]
            fraction = np.clip(np.dot(tcp-start, direction)/np.dot(direction, direction), 0, 1)
            assert np.linalg.norm(tcp-start-fraction*direction) <= .003 + 1e-10


def test_geometry_refitting_never_masks_joint_or_deadline_failures():
    for reason in ["motion_timeout", "right: timed trajectory exceeds joint limits"]:
        calls = []

        def check(_):
            calls.append(True)
            raise ValueError(reason)

        with pytest.raises(ValueError, match=reason):
            trajectory.parameterize(joints(0), [joints(.2)], CLOCK, check=check)
        assert len(calls) == 1


def test_unreachable_later_waypoint_and_insufficient_budget_move_nothing(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    bad = robo.move_both(right=obs.right.shifted(dx=10),
                         via=[{"right": obs.right.shifted(dz=.02)}])
    assert bad.status == "rejected" and not env.actions
    result = robo.move_both(right=obs.right.shifted(dz=.03), max_steps=3,
                            motion=robo.motion_profile(.1, .2, 1))
    assert result.status == "rejected" and "budget" in result.stop_reason
    assert not env.actions
    env.step_limit = 1
    result = robo.move_to(obs.right.shifted(dz=.03).xyz)
    assert result.status == "rejected" and not result.reached and not env.actions
    assert result.stop_reason == "timed_path_exceeds_native_step_budget"
    assert result.ik_error is None and result.path is None and result.planned_path == "line"
    assert result.steps_used == 0 and "budget" in result.note


def test_tracking_failure_does_not_execute_remaining_trajectory(tmp_path):
    env, robo, _ = station(tmp_path)
    env.frozen = "right"
    obs = robo.observe()
    result = robo.move_both(right=obs.right.shifted(dz=.04), max_steps=200,
                            position_tolerance_m=.002)
    assert result.status == "partial" and "tracking_error" in result.stop_reason
    assert result.steps_used < result.trajectory["planned_steps"]
    assert not result.reached


def test_legacy_host_rejects_explicit_timing_and_continuous_routes(tmp_path):
    env, robo, _ = station(tmp_path, JointEnvironment())
    obs = robo.observe()
    assert not robo.motion_capabilities()["speed_control"]
    result = robo.move_both(right=obs.right.shifted(dz=.02), motion=robo.motion_profile())
    assert result.status == "rejected" and "unsupported" in result.stop_reason
    single = robo.move_to(obs.right.shifted(dz=.02).xyz, motion=robo.motion_profile())
    assert not single.reached and single.ik_error is None
    assert single.status == "rejected" and single.stop_reason == "unsupported_timed_trajectory"
    assert single.path is None and single.planned_path == "line" and single.steps_used == 0
    via = robo.move_both(right=obs.right.shifted(dz=.02), via=[{"right": obs.right}])
    assert via.status == "rejected" and not env.actions
    assert robo.move_to(obs.right.shifted(dz=.02).xyz).reached  # legacy path still works


def test_adapter_rejects_clock_and_payload_mismatch_before_any_action():
    env = TimedEnvironment()
    ways, _ = trajectory.parameterize(joints(0), [joints(.1)], CLOCK)
    way = ways[0]
    action = {f"{a}_arm_joint_state": way["joints"][a] for a in ("left", "right")}
    action.update({f"{a}_ee_joint_state": [0.] for a in ("left", "right")})
    action[trajectory.TRAJECTORY_KEY] = copy.deepcopy(way["trajectory"])
    action[trajectory.TRAJECTORY_KEY]["physics_dt_s"] = .01
    with pytest.raises(ValueError, match="clock"):
        env.adapter._take_action(env, action)
    action[trajectory.TRAJECTORY_KEY] = copy.deepcopy(way["trajectory"])
    action[trajectory.TRAJECTORY_KEY]["positions"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        env.adapter._take_action(env, action)
    assert not env.actions and not env.expanded
    env.sim.cfg.decimation = 2
    assert env.adapter._motion_backend(env) is None


def test_gripper_completion_holds_timed_endpoint_without_native_reinterpolation(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    result = robo.move_both(right=obs.right.shifted(dz=.02),
                            grippers={"right": "open"}, max_steps=200)
    assert result.reached and result.right.opening >= .95
    assert result.steps_used == result.trajectory["planned_steps"]
    holds = env.wire_actions[-result.trajectory["gripper_hold_steps"]:]
    for command in holds:
        payload = command[trajectory.TRAJECTORY_KEY]
        assert not np.any(payload["velocities"])
        assert all(p == payload["positions"][0] for p in payload["positions"])


def test_substep_geometry_checks_can_reject_between_ik_knots(monkeypatch):
    from pyrualean.hosts import robodojo_kinematics as kin

    def clearance(arm, q):
        return ["table"] if arm == "right" and .03 < q[0] < .04 else []

    monkeypatch.setattr(kin, "clearance_violations", clearance)
    with pytest.raises(ValueError, match="intersects table"):
        trajectory.parameterize(joints(0), [joints(.1)], CLOCK, check=trajectory.check_joints)


def test_adapter_restores_native_method_after_execution_exception():
    env = TimedEnvironment()
    ways, _ = trajectory.parameterize(joints(0), [joints(.1)], CLOCK)
    action = {f"{a}_arm_joint_state": ways[0]["joints"][a] for a in ("left", "right")}
    action[trajectory.TRAJECTORY_KEY] = ways[0]["trajectory"]

    def fail(action):
        raise RuntimeError("injected native failure")

    env.take_action = fail
    with pytest.raises(RuntimeError, match="injected"):
        env.adapter._take_action(env, action)
    assert "process_control_info" not in vars(env)


def test_startup_card_and_generated_api_describe_live_support(tmp_path):
    from pyrualean.hosts.rpent_robodojo import Cell
    from pyrualean.hosts.xpolicylab_robodojo import dump_card
    from pyrualean.prompt import api_reference, knowledge_text

    env, robo, host = station(tmp_path)
    card = dump_card(host, Cell("arx_x5", "example", 0), tmp_path)
    assert card.facts["motion capabilities at episode start"] == robo.motion_capabilities()
    reference = api_reference(type(robo), primitives="no-vla")
    assert "robo.motion_profile(" in reference and "via: list[dict]" in reference
    assert "planned_path" in reference and "stop_reason" in reference
    mechanics = []
    for name in ("robodojo", "robodojo_xpl_rgb"):
        for primitives in ("full", "no-vla"):
            manual = knowledge_text(name, primitives)
            assert "force_feedback=False" in manual and "rad/s³" in manual
            assert "move_both(left=final_left" in manual
            boundary = "## The VLA" if primitives == "full" else "## Perception"
            mechanics.append(manual.split(boundary)[0])
    assert all(text == mechanics[0] for text in mechanics)


def test_single_arm_outcomes_distinguish_geometry_completion_and_partial(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    rejected = robo.move_to(obs.right.shifted(dx=10).xyz)
    assert rejected.status == "rejected" and rejected.stop_reason == "ik_path_rejected"
    assert rejected.ik_error and rejected.path is None and rejected.planned_path is None
    assert rejected.steps_used == 0 and not env.actions
    completed = robo.move_to(obs.right.shifted(dz=.02).xyz, path="line")
    assert completed.reached and completed.status == "completed" and completed.steps_used > 0
    assert completed.path == completed.planned_path == "line"
    assert completed.stop_reason is None and completed.ik_error is None
    env.frozen = "right"
    partial = robo.move_to(robo.observe().right.shifted(dz=.04).xyz, path="line")
    assert not partial.reached and partial.status == "partial" and partial.steps_used > 0
    assert partial.stop_reason == "tracking_error" and partial.ik_error is None
    assert partial.path == partial.planned_path == "line"
    assert partial.steps_used < partial.trajectory["planned_steps"]


def test_timing_refusal_does_not_claim_a_selected_near_pose_was_executed(tmp_path, monkeypatch):
    from pyrualean.hosts import robodojo_tools

    env, robo, _ = station(tmp_path)
    env.step_limit = 1
    original = robodojo_tools._plan

    def near_plan(*args, **kwargs):
        ways, taken, why, _ = original(*args, **kwargs)
        return ways, taken, why, "selected the closest one, 0.012 m from the request"

    monkeypatch.setattr(robodojo_tools, "_plan", near_plan)
    result = robo.move_to(robo.observe().right.shifted(dz=.03).xyz)
    assert result.status == "rejected" and result.steps_used == 0
    assert "closest one" not in result.note and "budget" in result.note
    assert not env.actions


def test_monitored_timing_refusal_returns_without_a_visual_vote(tmp_path):
    from tests.test_coordination import WATCH, monitored

    env, robo, _ = station(tmp_path)
    env.step_limit = 1
    cells, monitor, packets = monitored(robo, tmp_path)
    result = cells.execute_tool("python", {"code": WATCH + """
r = robo.move_to(robo.observe().right.shifted(dz=.03).xyz)
print(r.status, r.stop_reason, r.steps_used)
"""}).result
    assert result["stdout"].strip() == "rejected timed_path_exceeds_native_step_budget 0"
    assert not result.get("execution_paused")
    assert not packets and not env.actions
    assert monitor.previous is None


def test_clock_change_invalidates_plan_before_dispatch(tmp_path):
    from pyrualean.coordination import CoordinationPause

    env, robo, host = station(tmp_path)
    execute = host._tools["motion_execute"]

    def changed_clock(**kwargs):
        env.obs_manager.dt = .008
        return execute(**kwargs)

    host._tools["motion_execute"] = changed_clock
    obs = robo.observe()
    with pytest.raises(CoordinationPause, match="clock"):
        robo.move_both(right=obs.right.shifted(dz=.02))
    assert not env.actions
    assert robo._coordination_feedback["executed_steps"] == 0
