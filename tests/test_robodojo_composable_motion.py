"""Composable pose axes, local FK/IK, executable waypoints, and free-space joint fallback."""

from __future__ import annotations

import math

import numpy as np
import pytest

from codeaspolicy.hosts import robodojo_kinematics as kin
from codeaspolicy.hosts import robodojo_trajectory as trajectory
from tests.test_robodojo_intent import _precision_station, station


def test_pose_from_axes_and_rigid_pivot_rotation_preserve_local_offset(tmp_path):
    env, robo, _ = station(tmp_path)
    down = robo.pose_from_axes([0.18, -0.05, 0.84], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0])
    x_ax, y_ax, z_ax = down.axes()
    assert x_ax == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert y_ax == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert z_ax == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)

    flipped = robo.pose_from_axes(
        [0.18, -0.05, 0.84], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], roll_deg=180.0
    )
    fx, fy, _ = flipped.axes()
    assert fx == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert fy == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)

    # Rigid screw rotation around an off-axis shaft pivot keeps the shaft fixed in TCP frame.
    pivot = np.array([0.195, -0.05, 0.80])
    r0 = kin.rot_from_quat(down.quat)
    local_pivot = r0.T @ (pivot - np.asarray(down.xyz))
    for deg in (-60.0, -30.0, 45.0, 90.0):
        turned = down.rotated([0.0, 0.0, 1.0], deg, pivot=pivot)
        rt = kin.rot_from_quat(turned.quat)
        world_pivot_after = np.asarray(turned.xyz) + rt @ local_pivot
        assert world_pivot_after == pytest.approx(pivot, abs=1e-6)
    assert not env.actions


def test_local_fk_ik_jaw_symmetry_and_continuous_line_check_zero_actions(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    obs = robo.observe()
    fk_pose = robo.fk("right")
    assert fk_pose.xyz == pytest.approx(obs.right.xyz, abs=1e-5)
    assert fk_pose.quat == pytest.approx(obs.right.quat, abs=1e-5)

    # Seed right wrist near +joint6 limit so one jaw branch is crowded (<0.35 rad margin)
    # while the 180-deg symmetric jaw branch sits near central joint6 with wide margin.
    q_crowded = [0.0, 0.9, 1.1, -0.43, 0.0, 2.95]
    assert not kin.clearance_violations("right", q_crowded)
    crowded_pose = robo.fk("right", q_crowded)
    direct = robo.ik("right", crowded_pose, seed=q_crowded, jaw_symmetry=False)
    sym = robo.ik("right", crowded_pose, seed=q_crowded, jaw_symmetry=True)
    assert direct.ok and not direct.flipped_jaw
    assert direct.min_margin_rad == pytest.approx(3.1 - 2.95, abs=0.02)
    assert sym.ok and sym.flipped_jaw
    assert sym.min_margin_rad > 0.8

    # Turning further past +3.1 rad from q_crowded fails continuous=True (wrist branch jump),
    # whereas turning the same physical direction from sym.joints succeeds continuously!
    further_crowded = crowded_pose.rotated([0.0, 0.0, 1.0], -35.0)
    blocked_turn = robo.ik("right", further_crowded, seed=q_crowded, continuous=True)
    assert not blocked_turn.ok

    further_sym = sym.pose.rotated([0.0, 0.0, 1.0], -35.0)
    clean_turn = robo.ik("right", further_sym, seed=sym.joints, continuous=True)
    assert clean_turn.ok and not clean_turn.collides
    assert not env.actions


def test_executable_waypoints_arc_and_closed_loop_run_continuous_c2_spline(tmp_path):
    env, robo, host = _precision_station(tmp_path)
    host._primitives._grip_commands = {"right": 0.0}
    env.grips["right"] = 0.02

    start = robo.observe().right
    pivot = np.asarray(start.xyz) + np.array([0.015, 0.0, 0.0])
    arc = [{"right": start.rotated([0.0, 0.0, 1.0], deg, pivot=pivot)} for deg in (10.0, 20.0, 30.0)]

    prev = robo.execute_motion(
        "pivot_arc", waypoints=arc, interaction="contact",
        holding={"right": "retain"}, preview=True,
    )
    assert prev["accepted"] and prev["preview"] and not env.actions

    res = robo.execute_motion(
        "pivot_arc", waypoints=arc, interaction="contact",
        holding={"right": "retain"},
    )
    assert res["reached"], res
    assert res["intent_steps_used"] == len(env.actions) == prev["planned_steps"]
    mid = env.wire_actions[len(env.actions) // 2][trajectory.TRAJECTORY_KEY]
    assert np.linalg.norm(mid["velocities"][-1]) > 1e-3

    # Closed loop returning to current pose still executes all waypoints instead of short-circuiting.
    now = robo.observe().right
    before_loop = len(env.actions)
    loop = robo.execute_motion(
        "closed_loop",
        waypoints=[{"right": now.shifted(dz=0.03)}, {"right": now}],
        interaction="free",
        holding={"right": "retain"},
    )
    assert loop["reached"] and len(env.actions) > before_loop + 4


def test_waypoints_partial_budget_resumes_remaining_legs_without_replaying(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    start = robo.observe().right
    pts = [
        {"right": start.shifted(dz=0.03)},
        {"right": start.shifted(dx=-0.03, dz=0.03)},
        {"right": start.shifted(dx=-0.03, dy=0.03, dz=0.03)},
    ]
    full_preview = robo.execute_motion(
        "staged_curve", waypoints=pts, interaction="approach",
        holding={"right": "empty"}, preview=True,
    )
    assert full_preview["accepted"]
    half_cap = max(4, full_preview["planned_steps"] // 2)
    first = robo.execute_motion(
        "staged_curve", waypoints=pts, interaction="approach",
        holding={"right": "empty"}, max_steps=half_cap,
    )
    assert first["status"] == "paused" and first["stop_reason"] == "segment_complete"
    assert not first["reached"] and 0 < len(env.actions) <= half_cap

    done = robo.resume_motion("staged_curve", max_steps=full_preview["planned_steps"] + 30)
    assert done["reached"], done
    assert robo.observe().right.xyz == pytest.approx(pts[-1]["right"].xyz, abs=0.005)


@pytest.mark.parametrize("bad_kwargs", [
    {"following": [{"right": [0.2, -0.1, 0.85]}], "preview": True},
    {"directions": {"right": [0.0, 0.0, 1.0]}},
    {"constraints": {"corridor_m": 0.02}},
])
def test_waypoints_reject_incompatible_options_before_dispatch(tmp_path, bad_kwargs):
    env, robo, _ = _precision_station(tmp_path)
    start = robo.observe().right
    res = robo.execute_motion(
        "bad_wp", waypoints=[{"right": start.shifted(dz=0.02)}],
        holding={"right": "empty"}, **bad_kwargs,
    )
    assert res["status"] == "rejected" and not env.actions


def test_free_space_falls_back_to_joint_swing_on_wrist_singularity_while_approach_stays_strict(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    # Place right arm on positive joint6 near +2.85 rad and target an orientation whose
    # shortest SLERP crosses +3.1 rad (blocking local straight-line IK at path fraction 0.5),
    # while best_solution easily reaches the target via the negative joint6 branch.
    q_start = [0.0, 0.9, 1.1, -0.43, 0.15, 2.85]
    q_goal = [0.0, 0.9, 1.1, -0.43, 0.15, 2.85 + math.radians(40.0) - 2.0 * math.pi]
    assert not kin.clearance_violations("right", q_start)
    assert not kin.clearance_violations("right", q_goal)
    env.joints["right"] = list(q_start)
    target_pose = robo.fk("right", q_goal)

    # Confirm approach/contact MUST reject straight-line wrist-limit / singularity crossings!
    strict = robo.execute_motion(
        "strict_approach", right=target_pose, interaction="approach",
        holding={"right": "empty"},
    )
    assert strict["status"] == "rejected" and not env.actions
    assert any(
        msg in strict["stop_reason"]
        for msg in ("singularity/IK branch jump", "unreachable or table collision at path fraction")
    )

    # Free-space preview and execution succeed cleanly via joint_fallback_swing!
    free_prev = robo.execute_motion(
        "free_swing", right=target_pose, interaction="free",
        holding={"right": "empty"}, preview=True,
    )
    assert free_prev["accepted"] and free_prev["planned_path"] == "joint"
    assert "joint_fallback_swing" in free_prev["checked"]
    assert not env.actions

    # Keep-out constraints still inspect every substep of the joint swing!
    mid_q = 0.5 * (np.asarray(q_start) + np.asarray(q_goal))
    mid_xyz = robo.fk("right", mid_q).xyz
    blocked = robo.execute_motion(
        "blocked_swing", right=target_pose, interaction="free",
        holding={"right": "empty"},
        constraints={"keep_out": [{"xyz": list(mid_xyz), "radius_m": 0.02}]},
    )
    assert blocked["status"] == "rejected" and blocked["stop_reason"] == "right: keep_out_constraint"
    assert not env.actions

    executed = robo.execute_motion(
        "free_swing", right=target_pose, interaction="free",
        holding={"right": "empty"},
    )
    assert executed["reached"], executed
    assert len(env.actions) == free_prev["planned_steps"]


def test_free_space_joint_fallback_without_trajectory_clock_bounds_per_step_tcp_motion(tmp_path):
    env, robo, _ = station(tmp_path)
    q_start = [0.0, 0.9, 1.1, -0.43, 0.15, 2.85]
    q_goal = [0.0, 0.9, 1.1, -0.43, 0.15, 2.85 + math.radians(40.0) - 2.0 * math.pi]
    env.joints["right"] = list(q_start)
    target_pose = robo.fk("right", q_goal)

    res = robo.execute_motion(
        "clockless_swing", right=target_pose, interaction="free",
        holding={"right": "empty"},
    )
    assert res["reached"] and res["planned_path"] == "joint"
    prev_pose = kin.fk("right", q_start)
    for act in env.actions:
        q_step = act["right_arm_joint_state"]
        cur_pose = kin.fk("right", q_step)
        d_m = float(np.linalg.norm(cur_pose[:3, 3] - prev_pose[:3, 3]))
        cos_ang = float(np.clip((np.trace(prev_pose[:3, :3].T @ cur_pose[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
        ang_deg = math.degrees(math.acos(cos_ang))
        assert d_m <= 0.02 + 1e-3
        assert ang_deg <= 8.0 + 0.5
        prev_pose = cur_pose


