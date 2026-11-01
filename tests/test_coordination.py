"""Public code API through the real native host, with joint-level fault injection."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.coordination import CoordinationPause, Pose
from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit
from pyrualean.prompt import api_reference
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot


class JointEnvironment:
    def __init__(self):
        self.joints = {a: [0.0] * 6 for a in kin.ARMS}
        self.grips = {a: 0.3 for a in kin.ARMS}
        self.actions = []
        self.frozen = None
        self.raise_after_step = False
        self.step_limit = 1000
        self.captures = 0

    def get_status(self):
        return {"step": len(self.actions), "step_limit": self.step_limit}

    def get_obs(self):
        source_id = f"joint-test-capture-{self.captures}"
        state = {}
        for a in kin.ARMS:
            pose = kin.fk(a, self.joints[a])
            state[f"{a}_arm_joint_state"] = list(self.joints[a])
            state[f"{a}_ee_pose"] = [*pose[:3, 3], *kin.quat_wxyz(pose[:3, :3])]
            state[f"{a}_ee_joint_state"] = [self.grips[a]]
        return {
            "state": state,
            "pyrualean_observation": {
                "observation_id": source_id, "source_id": source_id,
                "image_source_id": source_id, "source_freshness": "captured",
                "aligned": True, "captured_at": float(self.captures),
                "capture_clock": "test_sensor",
            },
            "vision": {
                camera: {"color": np.full((8, 8, 3), len(self.actions), dtype=np.uint8)}
                for camera in RobodojoOfficialColourRobot.CAMERAS
            },
        }

    def step(self, action):
        self.actions.append(copy.deepcopy(action))
        self.captures += 1
        for a in kin.ARMS:
            if a != self.frozen:
                self.joints[a] = list(action[f"{a}_arm_joint_state"])
            goal = action[f"{a}_ee_joint_state"][0]
            # Object stops a closing gripper at .3, opening advances .3 per tick.
            self.grips[a] = max(0.3, self.grips[a] + np.clip(goal - self.grips[a], -0.3, 0.3))
        if self.raise_after_step:
            raise RuntimeError("transport failed after physical step")
        return self.get_obs(), 0.0, False, {"status": self.get_status()}

    def refresh_observation(self):
        self.captures += 1
        return self.get_obs()


@pytest.fixture
def station(tmp_path):
    env = JointEnvironment()
    host = NativeToolkit(env, None, tmp_path / "native")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))
    return env, robo, host


def lift(robo, **kwargs):
    obs = robo.observe()
    return robo.move_both(obs.left.shifted(dz=0.04), obs.right.shifted(dz=0.04), **kwargs)


def monitored(robo, tmp_path, prediction="yes", **config):
    from pyrualean.supervision import MonitorConfig, VisualSupervisor

    packets = []

    def predict(packet):
        packets.append(packet)
        return {
            "answers": {
                key: {
                    "type": "choice",
                    "probabilities": {
                        value: float(value == prediction) for value in ("yes", "no", "unknown")
                    },
                }
                for key in packet["questions"]
            }
        }

    supervisor = VisualSupervisor(
        MonitorConfig(mode="pause", url="http://localhost:8001/predict", **config),
        tmp_path / "audit",
        client=SimpleNamespace(predict=predict),
    )
    cells = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "cells", supervisor=supervisor, images=False
    )
    return cells, supervisor, packets


WATCH = "watch('move', {'held': 'Is the object supported by the right gripper?'})\n"
CRITICAL_WATCH = (
    "watch('move', {'held': 'Is the object supported by the right gripper?'}, "
    "critical=['held'])\n"
)


def test_sync_moves_both_each_tick_and_keeps_squeeze(station):
    env, robo, host = station
    host._primitives._grip_commands = {a: 0.0 for a in kin.ARMS}
    obs = robo.observe()
    result = lift(robo, keep_relative=True)
    assert result.reached and result.left.reached and result.right.reached
    assert result.steps_used == len(env.actions)
    assert all(
        action[f"{arm}_arm_joint_state"] != [0.0] * 6 for action in env.actions for arm in kin.ARMS
    )
    assert all(
        action[f"{arm}_ee_joint_state"] == [0.0] for action in env.actions for arm in kin.ARMS
    )
    assert robo.image("cam_head", step=obs.step)[0, 0, 0] == 0
    assert robo.image("cam_head", step=result.observation.step)[0, 0, 0] == len(env.actions)
    assert result.observation.state_version == robo._action_revision
    assert result.observation.native_revision == len(env.actions)


def test_xyz_retains_orientation_and_omitted_arm_holds(station):
    env, robo, _ = station
    obs = robo.observe()
    result = robo.move_both(right=obs.right.shifted(dz=0.03).xyz)
    assert result.reached
    assert result.right.pose.quat == pytest.approx(obs.right.quat, abs=0.01)
    assert all(v["left_arm_joint_state"] == [0.0] * 6 for v in env.actions)


def test_python_freely_composes_single_arm_grip_and_both(station, tmp_path):
    env, robo, _ = station
    cells = ArmToolkit(robo, arm="cells", workspace=tmp_path / "cells", images=False)
    result = cells.execute_tool(
        "python",
        {
            "code": """
o = robo.observe()
a = robo.move_to(o.left.shifted(dz=.02).xyz, arm='left')
if a.reached:
    b = robo.move_both(right=o.right.shifted(dz=.02))
    if b.reached:
        robo.set_gripper('right', robo.CLOSE)
        print('composed')
"""
        },
    ).result
    assert "composed" in result.get("stdout", ""), result
    assert [r.tool for r in robo._ledger if r.stateful] == ["move_pose", "motion_execute", "grip"]
    assert len(env.actions) > 0


def test_preflight_refusal_returns_both_errors_without_actions_or_visual_vote(station, tmp_path):
    env, robo, _ = station
    cells, _, packets = monitored(robo, tmp_path)
    result = cells.execute_tool(
        "python",
        {
            "code": WATCH
            + """
o = robo.observe()
r = robo.move_both(o.left.shifted(dz=.03), o.right.shifted(dx=10))
print(r.status, r.recoverable, r.steps_used)
"""
        },
    ).result
    assert result["stdout"].strip() == "rejected True 0", result
    assert not env.actions and not packets
    assert not result.get("execution_paused")
    assert result["coordination"]["arms"]["right"]["position_error_m"] > 9


def test_bounded_generated_loop_changes_rejected_target_in_same_cell(station, tmp_path):
    env, robo, _ = station
    cells, _, packets = monitored(robo, tmp_path)
    result = cells.execute_tool(
        "python",
        {
            "code": WATCH
            + """
o = robo.observe()
for dz in (10, .02):
    r = robo.move_both(o.left.shifted(dz=dz), o.right.shifted(dz=dz))
    print(r.status)
    if r.reached:
        break
"""
        },
    ).result
    assert result["stdout"].split() == ["rejected", "completed"], result
    assert env.actions and len(packets) == 1


def test_partial_result_stops_both_but_returns_to_python(station, tmp_path):
    env, robo, _ = station
    env.frozen = "left"
    cells, monitor, packets = monitored(robo, tmp_path)
    result = cells.execute_tool(
        "python",
        {
            "code": WATCH
            + """
o = robo.observe()
r = robo.move_both(o.left.shifted(dz=.04), o.right.shifted(dz=.04),
                   position_tolerance_m=.002, checkpoint_steps=1)
print(r.status, r.recoverable, r.steps_used)
"""
        },
    ).result
    assert not result.get("execution_paused"), result
    assert result["stdout"].strip() == "partial True 1"
    assert len(env.actions) == len(packets) == 1
    assert monitor.reports[-1]["motion_event"] == "motion_incomplete"
    assert monitor.reports[-1]["event"] == "clear"
    assert not result["coordination"]["arms"]["left"]["reached"]


def test_generated_correction_after_partial_uses_actual_endpoint(station, tmp_path):
    env, robo, _ = station
    env.frozen = "left"
    step = env.step

    def briefly_stalled(action):
        value = step(action)
        env.frozen = None
        return value

    env.step = briefly_stalled
    cells, _, _ = monitored(robo, tmp_path, max_actions=20)
    result = cells.execute_tool(
        "python",
        {
            "code": WATCH
            + """
o = robo.observe()
left, right = o.left.shifted(dz=.03), o.right.shifted(dz=.03)
for attempt in range(2):
    r = robo.move_both(left, right, position_tolerance_m=.002)
    print(r.status)
    if r.reached or not r.recoverable:
        break
"""
        },
    ).result
    assert result["stdout"].split() == ["partial", "completed"], result
    assert sum(r.summary["executed_steps"] for r in robo._ledger if r.stateful) == len(env.actions)
    heights = [kin.tcp(kin.fk("right", v["right_arm_joint_state"]))[2] for v in env.actions]
    assert all(
        b >= a - 1e-5 for a, b in zip(heights, heights[1:], strict=False)
    )  # no replay from start


def test_visual_unknown_stops_even_a_recoverable_partial_move(station, tmp_path):
    env, robo, _ = station
    env.frozen = "left"
    cells, monitor, _ = monitored(robo, tmp_path, prediction="unknown", unknown_limit=1)
    result = cells.execute_tool(
        "python",
        {
            "code": CRITICAL_WATCH
            + """
o = robo.observe()
r = robo.move_both(o.left.shifted(dz=.04), o.right.shifted(dz=.04),
                   position_tolerance_m=.002)
print('MUST NOT RUN')
"""
        },
    ).result
    assert result["execution_paused"] and len(env.actions) == 1
    assert result["coordination"]["status"] == "paused"
    assert result["coordination"]["executed_steps"] == 1
    assert result["coordination"]["observation"]["state_version"] == monitor.last_endpoint[1]
    assert "MUST NOT RUN" not in result.get("stdout", "")


def test_gripper_timing_is_chosen_by_python_not_a_fixed_release_workflow(station):
    env, robo, _ = station
    r = lift(robo, grippers={"left": "open", "right": "close"})
    assert r.reached
    assert env.actions[0]["left_ee_joint_state"] == [1.0]
    assert env.actions[0]["right_ee_joint_state"] == [0.0]
    assert env.grips == {"left": 1.0, "right": 0.3}
    assert robo.move_both(grippers={"left": "open", "right": "open"}).reached


@pytest.mark.parametrize("method", ["single", "both"])
def test_declared_phase_gate_applies_to_all_motion(station, tmp_path, method):
    env, robo, _ = station
    cells, _, packets = monitored(robo, tmp_path)
    action = (
        "robo.set_gripper('left', robo.OPEN)"
        if method == "single"
        else ("robo.move_both(grippers={'left': 'open'})")
    )
    result = cells.execute_tool(
        "python",
        {
            "code": (
                "watch('release', {'held': 'Is the object supported by the right gripper?'}, "
                "requires={'held': 'yes'})\n" + action
            )
        },
    ).result
    assert result["execution_paused"] and not env.actions and not packets
    assert "prerequisite_review" in result["supervision"]["reason"]


def test_declared_gate_with_fresh_evidence_allows_release(station, tmp_path):
    env, robo, _ = station
    cells, _, _ = monitored(robo, tmp_path)
    first = cells.execute_tool("python", {"code": WATCH + "robo.set_gripper('right', robo.CLOSE)"})
    assert not first.result.get("execution_paused")
    result = cells.execute_tool(
        "python",
        {
            "code": """
watch('release', {'held': 'Is the object supported by the right gripper?'}, requires={'held':'yes'})
robo.set_gripper('left', robo.OPEN)
"""
        },
    ).result
    assert not result.get("execution_paused"), result
    assert env.grips["left"] == 1.0


def test_unsupported_geometry_and_relative_preflight_return_rejection(station):
    env, robo, _ = station
    assert not robo.motion_capabilities()["inter_arm_clearance"]
    assert lift(robo, require_inter_arm_clearance=True).status == "rejected"
    obs = robo.observe()
    r = robo.move_both(
        left=obs.left.shifted(dz=0.04), keep_relative=True, position_tolerance_m=0.002
    )
    assert r.status == "rejected" and "relative_pose_constraint" in r.stop_reason
    assert not env.actions


def test_relative_violation_in_execution_hands_back_to_model(station):
    env, robo, _ = station
    step = env.step

    def drift(action):
        value = step(action)
        env.joints["left"][0] += 0.02
        env.joints["right"][0] -= 0.02
        return value

    env.step = drift
    with pytest.raises(CoordinationPause, match="relative_pose_constraint"):
        lift(robo, keep_relative=True, orientation_tolerance_deg=1.5)
    assert len(env.actions) == 1
    assert robo._coordination_feedback["status"] == "partial"


def test_recoverable_tracking_miss_cannot_hide_relative_violation(station):
    env, robo, _ = station
    env.frozen = "left"
    with pytest.raises(CoordinationPause, match="relative_pose_constraint"):
        lift(robo, keep_relative=True, position_tolerance_m=.002)
    assert len(env.actions) == 1
    assert not robo._coordination_feedback["recoverable"]


def test_native_error_after_physical_step_is_charged_and_hands_back(station, tmp_path):
    env, robo, _ = station
    env.raise_after_step = True
    cells = ArmToolkit(robo, arm="cells", workspace=tmp_path / "cells", images=False)
    result = cells.execute_tool(
        "python",
        {
            "code": """
o = robo.observe()
robo.move_both(o.left.shifted(dz=.03), o.right.shifted(dz=.03))
print('MUST NOT RUN')
"""
        },
    ).result
    assert result["execution_paused"]
    assert result["coordination"]["executed_steps"] == len(env.actions) == 1
    assert "transport failed" in result["coordination"]["stop_reason"]
    assert result["coordination"]["remaining_steps"] == 59
    assert robo.state().step == 1
    assert "MUST NOT RUN" not in result.get("stdout", "")


def test_call_deadline_stops_inside_segment(station, monkeypatch):
    from pyrualean.hosts import robodojo_coordination as controller

    env, robo, _ = station
    clock = [1000.0]
    monkeypatch.setattr(controller, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    step = env.step

    def slow(action):
        value = step(action)
        clock[0] += 1
        return value

    env.step = slow
    with pytest.raises(CoordinationPause, match="motion_timeout"):
        lift(robo, timeout_s=0.5, checkpoint_steps=8)
    assert len(env.actions) == 1
    assert robo._coordination_feedback["executed_steps"] == 1


@pytest.mark.parametrize("change", ["timeout", "drift"])
def test_interruption_between_checkpoints_reports_existing_partial_motion(
    station, monkeypatch, change
):
    from pyrualean.hosts import robodojo_coordination as controller

    env, robo, host = station
    clock = [1000.0]
    monkeypatch.setattr(controller, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    execute = host._tools["motion_execute"]

    def changed_after_checkpoint(**kwargs):
        result = execute(**kwargs)
        if result["status"] == "checkpoint":
            if change == "timeout":
                clock[0] += 1
            else:
                env.joints["left"][0] += .01
        return result

    monkeypatch.setitem(host._tools, "motion_execute", changed_after_checkpoint)
    if change == "timeout":
        with pytest.raises(CoordinationPause, match="motion_timeout"):
            lift(robo, timeout_s=.5, checkpoint_steps=1)
        result = robo._coordination_feedback
        assert result["status"] == "partial" and result["executed_steps"] == 1
    else:
        result = lift(robo, checkpoint_steps=1)
        assert result.status == "partial" and result.steps_used == 1
    assert len(env.actions) == 1


def test_limits_preflight_and_episode_budget_survive_new_calls(station):
    from pyrualean._backend import EpisodeFinished

    env, robo, _ = station
    assert lift(robo, max_steps=1).status == "rejected"
    assert not env.actions
    env.step_limit = 1
    with pytest.raises(CoordinationPause, match="episode_native_step_budget"):
        lift(robo)
    assert len(env.actions) == 1
    with pytest.raises(EpisodeFinished):
        lift(robo)
    assert len(env.actions) == 1


def test_host_refuses_plan_after_external_drift(station, monkeypatch):
    env, robo, host = station
    execute = host._tools["motion_execute"]

    def drifted(**kwargs):
        env.joints["left"][0] += 0.01
        return execute(**kwargs)

    monkeypatch.setitem(host._tools, "motion_execute", drifted)
    r = lift(robo)
    assert r.status == "rejected" and r.recoverable
    assert not env.actions


def test_keep_retains_last_close_even_if_measured_open(station):
    env, robo, host = station
    env.grips = {a: 1.0 for a in kin.ARMS}
    host._primitives._grip_commands = {a: 0.0 for a in kin.ARMS}
    assert lift(robo).reached
    assert all(v[f"{a}_ee_joint_state"] == [0.0] for v in env.actions for a in kin.ARMS)


def test_program_can_branch_on_rejected_motion(station, tmp_path):
    env, robo, _ = station
    program = ArmToolkit(robo, arm="program", workspace=tmp_path / "program", images=False)
    result = program.execute_tool(
        "run_program",
        {
            "code": """
def run(robo):
    r = robo.move_both(require_inter_arm_clearance=True)
    if not r.reached:
        print(r.status)
"""
        },
    ).result
    assert result["status"] == "completed" and result["stdout"].strip() == "rejected"
    assert not env.actions


def test_pose_and_generated_api_expose_the_simple_interface():
    with pytest.raises(ValueError):
        Pose([1, 2, 3], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        Pose([1, 2, 3], [1, 0, 0, 0], point="eef")
    values = [1, 2, 3]
    p = Pose(values, [1, 0, 0, 0])
    values[0] = 100
    assert p.xyz == (1, 2, 3)
    text = api_reference(RobodojoOfficialColourRobot)
    assert "robo.move_both(" in text and "DualMove" in text
    assert "robo.coordination(" not in text and "op.plan(" not in text


@pytest.mark.parametrize("insert_refusal", [False, True])
def test_unknown_streak_survives_recoverable_partial_retries(
    station, tmp_path, monkeypatch, insert_refusal
):
    env, robo, host = station
    env.frozen = "left"
    cells, monitor, packets = monitored(robo, tmp_path, prediction="unknown", unknown_limit=2)
    monkeypatch.setitem(
        host._tools, "move_pose",
        lambda **kw: {"reached": False, "ik_error": "no reachable solution", "steps_used": 0},
    )
    retry = "    robo.move_to([0, 0, 1.0])\n" if insert_refusal else ""
    result = cells.execute_tool(
        "python",
        {
            "code": CRITICAL_WATCH
            + """
o = robo.observe()
for attempt in range(3):
    r = robo.move_both(o.left.shifted(dz=.04), o.right.shifted(dz=.04),
                       position_tolerance_m=.002)
    print(r.status)
"""
            + retry
        },
    ).result
    assert result["execution_paused"] and len(env.actions) == 2
    assert len(packets) == 2 and result["stdout"].split() == ["partial"]
    assert monitor.unknowns["held"] == 2
    assert monitor.actions == 2 + int(insert_refusal)


def test_program_critical_fault_is_paused_with_physical_feedback(station, tmp_path):
    env, robo, _ = station
    env.raise_after_step = True
    programs = ArmToolkit(robo, arm="program", workspace=tmp_path / "programs", images=False)
    result = programs.execute_tool(
        "run_program",
        {
            "code": """
def run(robo):
    o = robo.observe()
    robo.move_both(o.left.shifted(dz=.02), o.right.shifted(dz=.02))
    print('MUST NOT RUN')
"""
        },
    ).result
    assert result["status"] == "paused" and result["execution_paused"]
    assert result["coordination"]["executed_steps"] == len(env.actions) == 1
    assert "MUST NOT RUN" not in result.get("stdout", "")


def test_zero_step_ik_retries_still_hit_supervisor_action_budget(station, tmp_path, monkeypatch):
    env, robo, host = station
    cells, monitor, packets = monitored(robo, tmp_path, max_actions=2)
    monkeypatch.setitem(
        host._tools,
        "move_pose",
        lambda **kw: {
            "reached": False,
            "ik_error": "no reachable solution",
            "steps_used": 0,
        },
    )
    result = cells.execute_tool(
        "python",
        {
            "code": WATCH
            + """
for attempt in range(5):
    robo.move_to([0, 0, 1.0])
print('MUST NOT RUN')
"""
        },
    ).result
    assert result["execution_paused"] and not env.actions and not packets
    assert monitor.actions == 2
    assert "MUST NOT RUN" not in result.get("stdout", "")
