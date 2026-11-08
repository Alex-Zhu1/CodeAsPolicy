# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for Code-as-Monitor (CaM) constraint elements, predicates, and guards."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pyrualean import monitor
from pyrualean.arms import ArmToolkit
from pyrualean.libero import LiberoRobot
from pyrualean.robodojo import ArmState, ObjectPose, State, WorldPoint
from pyrualean.supervision import MonitorConfig, SupervisionPause, VisualSupervisor
from tests.test_robodojo_trajectory import station


# ---------------------------------------------------------------------------
# Task 1: ConstraintElement & Predicate Unit Tests
# ---------------------------------------------------------------------------


def test_constraint_element_from_object_pose():
    pose = ObjectPose(
        found=True,
        center_xyz=(0.15, -0.10, 0.80),
        top_z=0.85,
        height_m=0.10,
        length_m=0.08,
        width_m=0.06,
        long_axis_deg=45.0,
        grasp_yaw=45.0,
        score=0.92,
        centroid=(120, 150),
        area_px=650,
        camera="cam_head",
        note=None,
        provenance={"observation_id": "obs-pose-1", "step": 3},
    )
    elem = monitor.from_pose("sauce_bottle", pose)

    assert elem.name == "sauce_bottle"
    assert elem.kind == "box"
    assert elem.xyz == (0.15, -0.10, 0.80)
    assert elem.extent_m == (0.08, 0.06, 0.10)
    assert elem.yaw_deg == 45.0
    assert elem.axis == (0.0, 0.0, 1.0)
    assert elem.observation_id == "obs-pose-1"
    assert elem.step == 3
    assert elem.to_dict()["name"] == "sauce_bottle"


def test_constraint_element_from_world_point():
    point = WorldPoint(
        pixel=(100, 200),
        xyz=(0.02, 0.12, 0.77),
        depth_m=0.60,
        camera="cam_head",
        provenance={"observation_id": "obs-pt-2", "step": 1},
    )
    elem = monitor.from_pose("place_location", point)

    assert elem.name == "place_location"
    assert elem.kind == "point"
    assert elem.xyz == (0.02, 0.12, 0.77)
    assert elem.extent_m is None
    assert elem.observation_id == "obs-pt-2"
    assert elem.step == 1


def test_constraint_element_from_landmark_dict_and_coords():
    landmark = {
        "xyz": [0.05, -0.05, 0.79],
        "uncertainty_m": 0.008,
        "provenance": {"observation_id": "obs-lm-3", "step": 2},
    }
    elem_lm = monitor.from_pose("rim_landmark", landmark)
    assert elem_lm.name == "rim_landmark"
    assert elem_lm.xyz == (0.05, -0.05, 0.79)
    assert elem_lm.uncertainty_m == 0.008
    assert elem_lm.observation_id == "obs-lm-3"
    assert elem_lm.step == 2

    elem_raw = monitor.from_pose("raw_coord", [0.1, 0.2, 0.3])
    assert elem_raw.xyz == (0.1, 0.2, 0.3)
    assert elem_raw.kind == "point"


def test_constraint_element_from_arm_state_pointing_convention():
    # pitch=90 pointing straight down -> [0, 0, -1]
    arm_down = ArmState(
        tcp_pos=(0.10, -0.20, 0.85),
        pitch=90.0,
        yaw=0.0,
        roll=0.0,
        eef_pos=(0.10, -0.20, 0.995),
        eef_quat=(1.0, 0.0, 0.0, 0.0),
        gripper=0.8,
        joints=(0.0,) * 6,
    )
    state = State(step=4, task="test_arm", left=arm_down, right=arm_down, truncated=False)
    elem_down = monitor.from_arm(state, arm="right")
    assert elem_down.name == "right"
    assert elem_down.kind == "axis"
    assert elem_down.step == 4
    assert elem_down.axis[0] == pytest.approx(0.0, abs=1e-5)
    assert elem_down.axis[1] == pytest.approx(0.0, abs=1e-5)
    assert elem_down.axis[2] == pytest.approx(-1.0, abs=1e-5)

    # pitch=0, yaw=0 points along world +Y.
    arm_fwd = ArmState(
        tcp_pos=(0.10, -0.20, 0.85),
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        eef_pos=(0.10, -0.345, 0.85),
        eef_quat=(1.0, 0.0, 0.0, 0.0),
        gripper=0.8,
        joints=(0.0,) * 6,
    )
    elem_fwd = monitor.from_arm(arm_fwd, arm="right")
    assert elem_fwd.axis[0] == pytest.approx(0.0, abs=1e-5)
    assert elem_fwd.axis[1] == pytest.approx(1.0, abs=1e-5)
    assert elem_fwd.axis[2] == pytest.approx(0.0, abs=1e-5)

    # pitch=0, yaw=90 points along world -X.
    arm_left = ArmState(
        tcp_pos=(0.10, -0.20, 0.85),
        pitch=0.0,
        yaw=90.0,
        roll=0.0,
        eef_pos=(0.245, -0.20, 0.85),
        eef_quat=(1.0, 0.0, 0.0, 0.0),
        gripper=0.8,
        joints=(0.0,) * 6,
    )
    elem_left = monitor.from_arm(arm_left, arm="left")
    assert elem_left.axis[0] == pytest.approx(-1.0, abs=1e-5)
    assert elem_left.axis[1] == pytest.approx(0.0, abs=1e-5)
    assert elem_left.axis[2] == pytest.approx(0.0, abs=1e-5)


def test_check_distance_clearance_and_reach():
    a = monitor.ConstraintElement("gripper", "point", xyz=(0.0, 0.0, 0.80))
    b = monitor.ConstraintElement("obstacle", "point", xyz=(0.0, 0.05, 0.80))

    # Min distance clearance violation
    check_min = monitor.check_distance(a, b, min_m=0.10)
    assert check_min.passed is False
    assert check_min.delta == pytest.approx(0.05 - 0.10)
    assert "clearance" in check_min.reason
    assert "Retract" in check_min.remedy

    # Max distance reach violation
    check_max = monitor.check_distance(a, b, max_m=0.03)
    assert check_max.passed is False
    assert check_max.delta == pytest.approx(0.05 - 0.03)
    assert "reach" in check_max.reason
    assert "Move closer" in check_max.remedy

    # Within clearance and reach
    check_ok = monitor.check_distance(a, b, min_m=0.02, max_m=0.08)
    assert check_ok.passed is True
    assert check_ok.reason is None

    # Horizontal XY only
    c = monitor.ConstraintElement("high_point", "point", xyz=(0.0, 0.05, 1.20))
    check_xy = monitor.check_distance(a, c, min_m=0.02, max_m=0.08, xy_only=True)
    assert check_xy.passed is True
    assert check_xy.metric == "distance_xy_m"
    assert check_xy.value == pytest.approx(0.05)


def test_check_tilt():
    # Pointing straight down [0, 0, -1] vs reference [0, 0, -1]
    elem_down = monitor.ConstraintElement("down_tool", "axis", xyz=(0, 0, 0), axis=(0.0, 0.0, -1.0))
    check_down = monitor.check_tilt(elem_down, reference_axis=(0.0, 0.0, -1.0), max_deg=10.0)
    assert check_down.passed is True
    assert check_down.value == pytest.approx(0.0)

    # Tilted 30 degrees
    rad30 = math.radians(30.0)
    tilted_axis = (math.sin(rad30), 0.0, -math.cos(rad30))
    elem_tilted = monitor.ConstraintElement("tilted_tool", "axis", xyz=(0, 0, 0), axis=tilted_axis)
    check_tilt = monitor.check_tilt(elem_tilted, reference_axis=(0.0, 0.0, -1.0), max_deg=15.0)
    assert check_tilt.passed is False
    assert check_tilt.value == pytest.approx(30.0, abs=1e-2)
    assert check_tilt.delta == pytest.approx(15.0, abs=1e-2)
    assert "tilt_constraint" in check_tilt.reason
    assert "Align axis" in check_tilt.remedy


def test_check_in_bounds():
    elem = monitor.ConstraintElement("tcp", "point", xyz=(0.10, 0.20, 0.75))

    # In bounds
    check_ok = monitor.check_in_bounds(
        elem, min_xyz=(0.0, 0.0, 0.70), max_xyz=(0.5, 0.5, 0.90)
    )
    assert check_ok.passed is True

    # Z too low
    check_low = monitor.check_in_bounds(
        elem, min_xyz=(0.0, 0.0, 0.78), max_xyz=(0.5, 0.5, 0.90)
    )
    assert check_low.passed is False
    assert check_low.delta == pytest.approx(0.03, abs=1e-4)
    assert "z" in check_low.reason
    assert "Adjust" in check_low.remedy


def test_check_above():
    gripper = monitor.ConstraintElement("gripper", "point", xyz=(0.10, 0.20, 0.90))
    receptacle = monitor.ConstraintElement(
        "bin", "box", xyz=(0.10, 0.20, 0.75), extent_m=(0.1, 0.1, 0.1)
    )
    # top_z of receptacle is 0.75 + 0.05 = 0.80. dz = 0.90 - 0.80 = 0.10 m.
    check_ok = monitor.check_above(gripper, receptacle, min_dz_m=0.05, max_xy_m=0.03)
    assert check_ok.passed is True

    # Misaligned horizontally (xy offset = 0.08 > 0.03)
    gripper_offset = monitor.ConstraintElement("gripper", "point", xyz=(0.18, 0.20, 0.90))
    check_xy = monitor.check_above(gripper_offset, receptacle, min_dz_m=0.05, max_xy_m=0.03)
    assert check_xy.passed is False
    assert "horizontal_alignment" in check_xy.reason

    # Too low vertically (dz = 0.82 - 0.80 = 0.02 < 0.05)
    gripper_low = monitor.ConstraintElement("gripper", "point", xyz=(0.10, 0.20, 0.82))
    check_dz = monitor.check_above(gripper_low, receptacle, min_dz_m=0.05, max_xy_m=0.03)
    assert check_dz.passed is False
    assert "vertical_clearance" in check_dz.reason


def test_check_hold_evidence():
    # 1. Gripper not obstructed
    ev_unobstructed = {
        "arm": "right",
        "state": "following",
        "gripper": {"obstructed_closure": False},
        "features": {},
    }
    check_unob = monitor.check_hold_evidence(ev_unobstructed)
    assert check_unob.passed is False
    assert "gripper_not_obstructed" in check_unob.reason
    assert "Re-attempt grasp alignment" in check_unob.remedy

    # 2. Relative motion / slip
    ev_slip = {
        "arm": "right",
        "state": "relative_motion",
        "gripper": {"obstructed_closure": True},
        "features": {"f1": {"relative_change_m": 0.025}},
    }
    check_slip = monitor.check_hold_evidence(ev_slip, max_drift_m=0.015)
    assert check_slip.passed is False
    assert "relative_motion" in check_slip.reason
    assert "Regrip the object" in check_slip.remedy

    # 3. Unknown state
    ev_unk = {
        "arm": "right",
        "state": "unknown",
        "reason": "camera_occluded",
        "gripper": {"obstructed_closure": True},
        "features": {},
    }
    check_unk = monitor.check_hold_evidence(ev_unk)
    assert check_unk.passed is False
    assert "unknown" in check_unk.reason
    assert "Observe object" in check_unk.remedy

    # 4. Verified hold
    ev_good = {
        "arm": "right",
        "state": "following",
        "gripper": {"obstructed_closure": True},
        "features": {"f1": {"relative_change_m": 0.004}},
    }
    check_good = monitor.check_hold_evidence(ev_good, max_drift_m=0.015)
    assert check_good.passed is True
    assert check_good.reason is None


def test_monitor_suite_and_evaluate_constraints():
    c1 = monitor.ConstraintCheck(
        name="clearance", kind="proactive", passed=True, metric="distance_m",
        value=0.1, bound=0.05, delta=0.05, elements=("arm", "table")
    )
    c2 = monitor.ConstraintCheck(
        name="tilt", kind="proactive", passed=False, metric="tilt_deg",
        value=25.0, bound=15.0, delta=10.0, elements=("arm",),
        reason="tilt exceeds limit", remedy="Align axis"
    )

    suite = monitor.MonitorSuite([c1]).add(c2)
    report = suite.evaluate()
    assert report["passed"] is False
    assert len(report["violated"]) == 1
    assert report["violated"][0]["name"] == "tilt"
    assert "1/2 constraints passed; 1 violated: tilt" in report["summary"]

    report_all_good = monitor.evaluate_constraints([c1])
    assert report_all_good["passed"] is True
    assert "All 1 constraint(s) passed" in report_all_good["summary"]


def test_proactive_constraints_builder():
    constraints = monitor.proactive_constraints(
        max_tilt_deg=20.0,
        z_bounds=[0.75, 1.10],
        keep_out=[{"xyz": [0.1, 0.2, 0.8], "radius_m": 0.05, "xy_only": True}],
        corridor_m=0.05,
    )
    assert constraints["max_tilt_deg"] == 20.0
    assert constraints["z_bounds"] == [0.75, 1.10]
    assert len(constraints["keep_out"]) == 1
    assert constraints["corridor_m"] == 0.05

    with pytest.raises(ValueError, match="max_tilt_deg must be in"):
        monitor.proactive_constraints(max_tilt_deg=200.0)

    with pytest.raises(ValueError, match="z_bounds must be"):
        monitor.proactive_constraints(z_bounds=[1.0, 0.5])


# ---------------------------------------------------------------------------
# Task 2: Proactive Kinematic / Geometric Guards in robodojo_intent
# ---------------------------------------------------------------------------


def test_robodojo_intent_preflight_rejection_keep_out(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    goal = obs.right.shifted(dz=0.06)

    # Obstacle directly at the midpoint
    midpoint = [
        (obs.right.xyz[0] + goal.xyz[0]) / 2.0,
        (obs.right.xyz[1] + goal.xyz[1]) / 2.0,
        (obs.right.xyz[2] + goal.xyz[2]) / 2.0,
    ]
    constraints = {"keep_out": [{"xyz": midpoint, "radius_m": 0.08}]}

    result = robo.execute_motion(
        "keep_out_test",
        right=goal,
        holding={"right": "empty"},
        constraints=constraints,
    )
    assert result["accepted"] is False
    assert result["status"] == "rejected"
    assert result["stop_reason"] == "right: keep_out_constraint"
    # Zero physical environment actions dispatched
    assert not env.actions


def test_robodojo_intent_preflight_rejection_z_bounds(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    goal = obs.right.shifted(dz=0.06)

    # Set z_bounds below goal height
    max_allowed_z = obs.right.xyz[2] + 0.02
    constraints = {"z_bounds": [0.70, max_allowed_z]}

    result = robo.execute_motion(
        "z_bounds_test",
        right=goal,
        holding={"right": "empty"},
        constraints=constraints,
    )
    assert result["accepted"] is False
    assert result["status"] == "rejected"
    assert result["stop_reason"] == "right: z_bounds_constraint"
    assert not env.actions


def test_robodojo_intent_preflight_rejection_max_tilt_deg(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    state = robo.state()
    # Goal with pitch rotated by 30 degrees
    tilted_goal = {
        "xyz": list(obs.right.shifted(dz=0.04).xyz),
        "angles": [state.right.pitch + 30.0, state.right.yaw, state.right.roll],
    }
    constraints = {"max_tilt_deg": 10.0}

    result = robo.execute_motion(
        "tilt_test",
        right=tilted_goal,
        holding={"right": "empty"},
        constraints=constraints,
    )
    assert result["accepted"] is False
    assert result["status"] == "rejected"
    assert result["stop_reason"] == "right: tilt_constraint"
    assert not env.actions


def test_robodojo_intent_corridor_constraint_accepted_and_rejected(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    goal = obs.right.shifted(dz=0.05)

    # Generous corridor: accepted and reaches goal
    ok_result = robo.execute_motion(
        "corridor_ok",
        right=goal,
        holding={"right": "empty"},
        constraints={"corridor_m": 0.05},
    )
    assert ok_result["accepted"] is True
    assert ok_result["reached"] is True

    # Ultra-tight corridor (0.0001m): rejected if interpolation or start departs by even 0.1mm
    obs2 = robo.observe()
    goal2 = obs2.right.shifted(dz=0.06)
    tight_result = robo.execute_motion(
        "corridor_tight",
        right=goal2,
        holding={"right": "empty"},
        constraints={"corridor_m": 0.00001},
    )
    # Rejection occurs either at preflight compile or step guard
    assert tight_result["stop_reason"] == "right: corridor_constraint"


# ---------------------------------------------------------------------------
# Task 3: Integration with arms.py and supervision.py
# ---------------------------------------------------------------------------


def test_arms_cells_has_monitor_global(robo, tmp_path):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")

    code = (
        "p1 = monitor.from_pose('pt1', [0.1, 0.2, 0.8])\n"
        "p2 = monitor.from_pose('pt2', [0.1, 0.2, 0.85])\n"
        "chk = monitor.check_distance(p1, p2, min_m=0.02, max_m=0.10)\n"
        "suite = monitor.MonitorSuite([chk])\n"
        "res = suite.evaluate()\n"
        "print(f'MONITOR_PASSED={res[\"passed\"]}')\n"
    )
    result = toolkit.execute_tool("python", {"code": code})
    assert not result.is_error
    assert "exception" not in result.result
    assert "MONITOR_PASSED=True" in result.result["stdout"]


def test_visual_supervisor_check_constraints_pause_and_events(tmp_path):
    audit_dir = tmp_path / "audit"
    config = MonitorConfig(url="http://127.0.0.1:18001/predict", mode="pause")
    supervisor = VisualSupervisor(config, audit_dir)

    pass_check = monitor.ConstraintCheck(
        name="clearance", kind="proactive", passed=True, metric="distance_m",
        value=0.10, bound=0.05, delta=0.05, elements=("gripper", "table")
    )
    res_pass = supervisor.check_constraints(pass_check)
    assert res_pass["passed"] is True

    fail_check = monitor.ConstraintCheck(
        name="tilt", kind="proactive", passed=False, metric="tilt_deg",
        value=35.0, bound=15.0, delta=20.0, elements=("right_arm",),
        reason="tilt exceeds limit", remedy="Align tool vertical"
    )

    with pytest.raises(SupervisionPause) as exc_info:
        supervisor.check_constraints(fail_check)
    assert "tilt exceeds limit" in str(exc_info.value)

    # Verify event recorded in events.jsonl
    events_file = audit_dir / "events.jsonl"
    assert events_file.exists()
    lines = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    violation_events = [e for e in lines if e.get("event") == "constraint_violation"]
    assert len(violation_events) == 1
    v = violation_events[0]
    assert v["name"] == "tilt"


def test_constraint_element_currency_and_stale_detection(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    elem = monitor.from_arm(robo.state(), arm="right")
    assert elem.step == 0
    assert elem.is_current(robo) is True
    elem.require_current(robo)

    # Robot moves -> step increments to > 0
    robo.execute_motion("travel", right=obs.right.shifted(dz=0.03), holding={"right": "empty"})
    assert robo._backend.env_steps() > 0

    # Old elem is now stale
    assert elem.is_current(robo) is False
    with pytest.raises(ValueError, match="is stale"):
        elem.require_current(robo)


def test_proactive_constraints_direct_perception_object(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    goal = obs.right.shifted(dz=0.06)

    # Obstacle as an ObjectPose directly
    midpoint = (
        (obs.right.xyz[0] + goal.xyz[0]) / 2.0,
        (obs.right.xyz[1] + goal.xyz[1]) / 2.0,
        (obs.right.xyz[2] + goal.xyz[2]) / 2.0,
    )
    obstacle_pose = ObjectPose(
        found=True,
        center_xyz=midpoint,
        top_z=midpoint[2] + 0.04,
        height_m=0.08,
        length_m=0.06,
        width_m=0.06,
        long_axis_deg=0.0,
        grasp_yaw=0.0,
        score=0.95,
        centroid=(100, 100),
        area_px=400,
        camera="cam_head",
        note=None,
    )

    # Directly pass ObjectPose into keep_out list!
    constraints = {"keep_out": [obstacle_pose]}
    result = robo.execute_motion(
        "direct_perception_keep_out",
        right=goal,
        holding={"right": "empty"},
        constraints=constraints,
    )
    assert result["accepted"] is False
    assert result["stop_reason"] == "right: keep_out_constraint"
    assert "constraint_violation" in result
    cv = result["constraint_violation"]
    assert cv["metric"] == "obstacle_distance_m"
    assert "Route right around obstacle" in cv["remedy"]


def test_dual_engine_supervisor_watch_with_code_invariants(tmp_path):
    audit_dir = tmp_path / "audit"
    config = MonitorConfig(url="http://127.0.0.1:18001/predict", mode="pause")
    supervisor = VisualSupervisor(config, audit_dir)

    # Watch with pure code invariants (no Valen HTTP needed!)
    good_inv = lambda r: monitor.check_in_bounds(
        monitor.from_arm(r.state(), "right"), max_xyz=(None, None, 1.5)
    )
    supervisor.watch("test_phase", invariants=[good_inv])
    assert supervisor.contract["phase"] == "test_phase"

    # Action within bounds passes
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    supervisor.before_action("execute_motion", robo)
    robo.execute_motion("reach", right=obs.right.shifted(dz=0.03), holding={"right": "empty"})
    supervisor.after_action(robo, "execute_motion", steps_before=0)
    assert supervisor.paused is None

    # Register invariant that fails: max z = 0.5m (current robot is at ~0.8m)
    strict_inv = lambda r: monitor.check_in_bounds(
        monitor.from_arm(r.state(), "right"), max_xyz=(None, None, 0.5)
    )
    supervisor.watch("strict_phase", invariants=[strict_inv])

    with pytest.raises(SupervisionPause) as exc:
        supervisor.before_action("move_to", robo)
    assert "constraint_violation" in str(exc.value)

    feedback = supervisor.feedback()
    assert feedback["paused"] is True
    assert "constraint_violation" in feedback
    assert feedback["constraint_violation"]["metric"] == "bounds_violation_m"


def test_recovery_constraint_violation_feedback_in_cells(tmp_path):
    env, robo, _ = station(tmp_path)
    audit_dir = tmp_path / "audit"
    config = MonitorConfig(url="http://127.0.0.1:18001/predict", mode="pause")
    supervisor = VisualSupervisor(config, audit_dir)
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", supervisor=supervisor)

    # Run cell that triggers a keep_out constraint violation in execute_motion
    code = (
        "obs = robo.observe()\n"
        "goal = obs.right.shifted(dz=0.06)\n"
        "mid = [(obs.right.xyz[i] + goal.xyz[i]) / 2.0 for i in range(3)]\n"
        "res = robo.execute_motion('fail_move', right=goal, holding={'right': 'empty'}, "
        "constraints={'keep_out': [{'xyz': mid, 'radius_m': 0.1}]})\n"
    )
    result = toolkit.execute_tool("python", {"code": code})
    assert result.result["moved"] is False
    assert "recovery" in result.result
    assert "constraint_violation" in result.result["recovery"]
    cv = result.result["recovery"]["constraint_violation"]
    assert cv["metric"] == "obstacle_distance_m"
    assert "Route right around obstacle" in cv["remedy"]


@pytest.mark.parametrize("state", ["image_relative_motion", "image_consistent"])
def test_hold_image_evidence_does_not_claim_measured_3d_drift(state):
    evidence = {
        "arm": "right",
        "state": state,
        "evidence_space": "wrist_image_2d",
        "gripper": {"obstructed_closure": True},
        "features": {},
        "image_features": {
            "f1": {"displacement_px": 10.0 if state == "image_relative_motion" else 0.0,
                   "comparison_bound_px": 4.0},
        },
    }
    result = monitor.check_hold_evidence(evidence)
    assert result.passed is False
    assert result.value is None
    assert result.delta is None
    assert "image" in result.reason


@pytest.mark.parametrize("state", [None, "following", "unsupported_state"])
def test_hold_missing_metric_evidence_is_not_zero_drift(state):
    result = monitor.check_hold_evidence({
        "state": state, "gripper": {"obstructed_closure": True}, "features": {},
    })
    assert result.passed is False
    assert result.value is None
    assert result.delta is None


@pytest.mark.parametrize("provenance", [
    {"binding": "unsupported", "step": 0},
    {"source_freshness": "unverified", "step": 0},
    {"current_at_return": False, "step": 0},
    {"binding": "unsupported", "source_freshness": "unverified", "current_at_return": False},
])
def test_constraint_element_rejects_explicitly_unverified_provenance(tmp_path, provenance):
    _, robo, _ = station(tmp_path)
    point = WorldPoint(pixel=(10, 20), xyz=(0.1, 0.2, 0.8), depth_m=0.6,
                       camera="cam_head", provenance=provenance)
    elem = monitor.from_pose("unverified", point)
    assert elem.is_current(robo) is False
    with pytest.raises(ValueError, match="stale"):
        elem.require_current(robo)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("found,center", [(False, None), (False, (0.1, 0.2, 0.8)), (True, None)])
def test_keep_out_missing_perception_rejects_before_dispatch(tmp_path, wrapped, found, center):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    pose = ObjectPose(
        found=found, center_xyz=center, top_z=None, height_m=None, length_m=None,
        width_m=None, long_axis_deg=None, grasp_yaw=None, score=None, centroid=None,
        area_px=0, camera="cam_head", note="no usable geometry",
    )
    obstacle = {"element": pose} if wrapped else pose
    with pytest.raises(ValueError, match="keep_out obstacle"):
        robo.execute_motion(
            "missing_obstacle", right=obs.right.shifted(dz=0.03),
            holding={"right": "empty"}, constraints={"keep_out": [obstacle]},
        )
    assert not env.actions


def test_keep_out_sphere_contains_height_and_rejects_intersection(tmp_path):
    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    x, y, z = obs.right.xyz
    obstacle = monitor.ConstraintElement(
        "tall_obstacle", "box", xyz=(x, y, z + 0.2), extent_m=(0.02, 0.02, 0.5),
    )
    constraints = {"keep_out": [{"element": obstacle, "xy_only": False}]}
    sphere = monitor.proactive_constraints(**constraints)["keep_out"][0]
    assert sphere["radius_m"] == pytest.approx(math.hypot(0.01, 0.01, 0.25) + 0.02)
    result = robo.execute_motion(
        "tall_obstacle", right=obs.right.shifted(dz=0.03),
        holding={"right": "empty"}, constraints=constraints,
    )
    assert result["accepted"] is False
    assert result["stop_reason"] == "right: keep_out_constraint"
    assert not env.actions


@pytest.mark.parametrize("margin", [-0.01, float("nan"), float("inf")])
def test_keep_out_element_margin_must_not_shrink_or_invalidate_geometry(margin):
    obstacle = monitor.ConstraintElement("obstacle", "point", xyz=(0.1, 0.2, 0.8))
    with pytest.raises(ValueError, match="margin_m"):
        monitor.proactive_constraints(keep_out=[{"element": obstacle, "margin_m": margin}])


@pytest.mark.parametrize("axis", [(0, 0, 0), (0, 1), (0, 0, float("nan")), (0, 0, float("inf"))])
def test_proactive_constraints_reference_axis_requires_finite_nonzero_vector(axis):
    with pytest.raises(ValueError, match="reference_axis"):
        monitor.proactive_constraints(reference_axis=axis)


def test_execute_motion_preserves_custom_constraint_reference_axis(tmp_path):
    from pyrualean.hosts.robodojo_kinematics import rot_from_quat

    env, robo, _ = station(tmp_path)
    obs = robo.observe()
    axis = rot_from_quat(robo.state().right.eef_quat)[:, 0].tolist()
    constraints = {"max_tilt_deg": 10.0, "reference_axis": axis}
    assert monitor.proactive_constraints(**constraints)["reference_axis"] == list(axis)
    result = robo.execute_motion(
        "reference_axis", right=obs.right.shifted(dz=0.03),
        holding={"right": "empty"}, constraints=constraints,
    )
    assert result["accepted"] is True
    assert result["reached"] is True
    assert env.actions


@pytest.mark.parametrize("pitch,yaw", [
    (0.0, 0.0), (0.0, 90.0), (30.0, -45.0), (89.5, 35.0), (90.0, 0.0),
])
def test_from_arm_pointing_axis_matches_robodojo_quaternion(pitch, yaw):
    from pyrualean.hosts.robodojo_kinematics import orientation, quat_wxyz

    rotation = orientation(pitch, yaw, 20.0)
    states = (
        {"tcp_pos": [0.1, 0.2, 0.9], "pitch": pitch, "yaw": yaw},
        {"tcp_pos": [0.1, 0.2, 0.9], "eef_quat": quat_wxyz(rotation)},
    )
    for state in states:
        assert monitor.from_arm(state).axis == pytest.approx(rotation[:, 0], abs=1e-8)


def test_from_arm_live_state_agrees_with_public_quaternion(tmp_path):
    from pyrualean.hosts.robodojo_kinematics import rot_from_quat

    _, robo, _ = station(tmp_path)
    state = robo.state()
    expected = rot_from_quat(state.right.eef_quat)[:, 0]
    # Public pitch/yaw values are rounded to hundredths of a degree.
    assert monitor.from_arm(state, "right").axis == pytest.approx(expected, abs=1e-4)


def test_paint_elements_copy_and_validation():
    import numpy as np

    with pytest.raises(ValueError, match="uint8 numpy array"):
        monitor.paint_elements(np.zeros((10, 10)), [])

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res_copy = monitor.paint_elements(img, [(50, 50)], copy=True)
    assert not np.array_equal(res_copy, img)
    assert np.all(img == 0)

    res_inplace = monitor.paint_elements(img, [(50, 50)], copy=False)
    assert res_inplace is img
    assert np.any(img > 0)


def test_paint_elements_2d_points_and_bboxes():
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    elem_point = (50, 50)
    elem_box = (10, 10, 40, 40)
    painted = monitor.paint_elements(img, [elem_point, elem_box])

    # Check point crosshair at (50, 50)
    assert np.any(painted[50, 45:55] > 0)
    assert np.any(painted[45:55, 50] > 0)

    # Check box borders: top edge at v=10, left edge at u=10
    assert np.any(painted[10, 10:40] > 0)
    assert np.any(painted[10:40, 10] > 0)
    # Interior of box should still be zeros
    assert np.all(painted[25, 25] == 0)


def test_paint_elements_3d_projected_elements():
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    def simple_project(xyz):
        u = float(xyz[0] * 100)
        v = float(xyz[1] * 100)
        if 0 <= u < 100 and 0 <= v < 100:
            return (u, v)
        return None

    elem_pt = monitor.ConstraintElement("pt", "point", xyz=(0.5, 0.5, 0.2), extent_m=(0.1, 0.1, 0.1))
    elem_axis = monitor.ConstraintElement("ax", "axis", xyz=(0.2, 0.2, 0.2), axis=(1, 0, 0), extent_m=(0.2, 0.2, 0.2))
    elem_box = monitor.ConstraintElement("bx", "box", xyz=(0.7, 0.7, 0.2), extent_m=(0.1, 0.1, 0.1))

    painted = monitor.paint_elements(img, [elem_pt, elem_axis, elem_box], project=simple_project)
    assert np.any(painted > 0)
    # Check projected point center at u=50, v=50
    assert np.any(painted[50, 50] > 0)


def test_paint_elements_object_observation():
    import numpy as np

    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True

    obs = {
        "bbox_2d": (20, 20, 80, 80),
        "center_px": (50, 50),
        "mask": mask,
    }

    painted = monitor.paint_elements(img, [obs])
    # Check mask tinting applied
    assert not np.array_equal(painted[40, 40], [50, 50, 50])
    # Check box border drawn
    assert np.any(painted[20, 20:80] != 50)
    # Check center crosshair drawn
    assert np.any(painted[50, 45:55] != 50)


def test_paint_elements_hold_observation_drift():
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[50:60, 50:60] = True

    hold_obs = {
        "reference": {"center_px": (30, 30)},
        "current": {"center_px": (60, 60), "mask": mask},
    }

    painted = monitor.paint_elements(img, [hold_obs])
    # Reference marker at (30, 30)
    assert np.any(painted[30, 30] > 0)
    # Current marker at (60, 60)
    assert np.any(painted[60, 60] > 0)
    # Drift vector midpoint at (45, 45)
    assert np.any(painted[45, 45] > 0)
    # Mask area tinted
    assert np.any(painted[55, 55] > 0)


def test_paint_elements_object_pose_and_world_point():
    from types import SimpleNamespace
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    def simple_project(xyz):
        return (float(xyz[0] * 100), float(xyz[1] * 100))

    pose = SimpleNamespace(
        center_xyz=(0.5, 0.5, 0.2),
        length_m=0.1,
        width_m=0.1,
        height_m=0.1,
        long_axis_deg=0.0,
    )
    wp = SimpleNamespace(
        xyz=(0.3, 0.3, 0.1),
        pixel=(30, 30),
        depth_m=0.5,
    )

    painted = monitor.paint_elements(img, [pose, wp], project=simple_project)
    # Center crosshair of pose at (50, 50)
    assert np.any(painted[50, 50] > 0)
    # World point crosshair at (30, 30)
    assert np.any(painted[30, 30] > 0)
