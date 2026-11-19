"""3D inter-arm collision envelopes, smart IK branch selection, and intent preflight guards."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from codeaspolicy.hosts import robodojo_kinematics as kin
from tests.test_robodojo_trajectory import TimedEnvironment, station


def _pose(xyz, pitch: float, yaw: float, roll: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = kin.orientation(pitch, yaw, roll)
    t[:3, 3] = xyz
    return t


def _solve_pose(arm: str, xyz, pitch: float, yaw: float, roll: float, seed=None) -> tuple[float, ...]:
    q0 = [0.0] * 6 if seed is None else list(seed)
    res = kin.best_solution(arm, _pose(xyz, pitch, yaw, roll), q0, pos_tol=1e-3, rot_tol_deg=0.5)
    assert res.ok and not res.collides, f"{arm} IK failed: {res}"
    return res.q


def test_default_pose_clear_and_submillisecond():
    q_zero = (0.0,) * 6
    assert kin.inter_arm_violations(q_zero, q_zero) == []
    assert kin.housing_exposure_penalty("left", q_zero, None) == 0.0
    assert kin.housing_exposure_penalty("left", q_zero, q_zero) == 0.0
    assert kin.housing_exposure_penalty("right", q_zero, q_zero) == 0.0

    # Pure-NumPy check stays fast (< 0.5 ms typical; generous 3 ms CI bound)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        kin.inter_arm_violations(q_zero, q_zero)
    avg_ms = (time.perf_counter() - t0) * 1000.0 / n
    assert avg_ms < 3.0, f"inter_arm_violations too slow: {avg_ms:.3f} ms/call"


def test_camera_housing_collision_and_180deg_camera_away_symmetry_selection():
    # Orthogonal handoff: donor (left) pitch=90 at (-0.03, -0.25, 0.90),
    # receiver (right) horizontal pitch=0, yaw=+90 (pointing -x), roll=+90 (vertical jaws)
    # pre-staged 5 cm back at (+0.05, -0.25, 0.90) with open fingers (opening=1.0).
    q_rec_pre = _solve_pose("right", (0.05, -0.25, 0.90), 0.0, 90.0, 90.0)

    # Donor yaw=-90 points +z_link6 = (+1, 0, 0) TOWARD receiver;
    # Donor yaw=+90 (= target @ Rx(pi)) points +z_link6 = (-1, 0, 0) AWAY from receiver.
    t_don_toward = _pose((-0.03, -0.25, 0.90), 90.0, -90.0, 0.0)
    t_don_away = _pose((-0.03, -0.25, 0.90), 90.0, 90.0, 0.0)
    q_don_toward = _solve_pose("left", (-0.03, -0.25, 0.90), 90.0, -90.0, 0.0)
    q_don_away = _solve_pose("left", (-0.03, -0.25, 0.90), 90.0, 90.0, 0.0)

    bad = kin.inter_arm_violations(q_don_toward, q_rec_pre, left_opening=0.1, right_opening=1.0)
    assert "camera_housing_collision" in bad
    assert kin.housing_exposure_penalty("left", q_don_toward, q_rec_pre) > 0.3

    clear = kin.inter_arm_violations(q_don_away, q_rec_pre, left_opening=0.1, right_opening=1.0)
    assert clear == []
    assert kin.housing_exposure_penalty("left", q_don_away, q_rec_pre) == 0.0

    # Without jaw_symmetry, best_solution marks the colliding camera-toward pose collides=True.
    no_sym = kin.best_solution(
        "left",
        t_don_toward,
        q_don_toward,
        jaw_symmetry=False,
        other_arm_q=q_rec_pre,
        opening=0.1,
        other_opening=1.0,
    )
    assert no_sym.collides

    # With jaw_symmetry=True, best_solution flips 180 deg even when seeded at q_don_toward!
    with_sym = kin.best_solution(
        "left",
        t_don_toward,
        q_don_toward,
        jaw_symmetry=True,
        other_arm_q=q_rec_pre,
        opening=0.1,
        other_opening=1.0,
    )
    assert with_sym.ok and not with_sym.collides
    z_axis = kin.fk("left", with_sym.q)[:3, 2]
    assert z_axis[0] < -0.95, f"expected +z_link6 away (-x), got {z_axis}"

    # Even when both branches are collision-free at mid-close range (~0.19 m),
    # housing_exposure_penalty still steers jaw_symmetry=True to turn camera away.
    q_rec_mid = _solve_pose("right", (0.14, -0.18, 0.90), 0.0, 45.0, 90.0)
    assert kin.inter_arm_violations(q_don_toward, q_rec_mid, left_opening=0.1, right_opening=0.38) == []
    assert kin.housing_exposure_penalty("left", q_don_toward, q_rec_mid) > 0.2
    assert kin.housing_exposure_penalty("left", q_don_away, q_rec_mid) == 0.0

    mid_sym = kin.best_solution(
        "left",
        t_don_toward,
        q_don_toward,
        jaw_symmetry=True,
        other_arm_q=q_rec_mid,
        opening=0.1,
        other_opening=0.38,
    )
    assert mid_sym.ok and not mid_sym.collides
    assert kin.fk("left", mid_sym.q)[:3, 2][0] < -0.95
    assert kin.housing_exposure_penalty("left", mid_sym.q, q_rec_mid) == 0.0

    # Verify flipped target matches target @ Rx(pi)
    assert np.allclose(t_don_toward[:3, :3] @ kin._rot_x(math.pi), t_don_away[:3, :3], atol=1e-9)


def test_open_finger_wedging_vs_partial_opening_and_standoff_in_orthogonal_handoff():
    # Donor (left) holds stem 3.0 cm past P_head=(0.0, -0.25, 0.90) with camera housing away.
    q_don = _solve_pose("left", (-0.03, -0.25, 0.90), 90.0, 90.0, 0.0)

    # Bad: Receiver drives open fingers (opening=1.0) all the way to P_head (0 standoff)
    q_rec_wedge = _solve_pose("right", (0.0, -0.25, 0.90), 0.0, 90.0, 90.0)
    wedge_bad = kin.inter_arm_violations(q_don, q_rec_wedge, left_opening=0.1, right_opening=1.0)
    assert wedge_bad, "open-finger wedging at P_head must be flagged"
    assert any(label in wedge_bad for label in ("crossbar_collision", "camera_housing_collision", "finger_collision"))

    # Also bad: Receiver at 10 mm standoff STILL collides when opening=1.0 (tall upper finger hits crossbar)
    q_rec_standoff = _solve_pose("right", (0.010, -0.25, 0.90), 0.0, 90.0, 90.0)
    wide_at_standoff = kin.inter_arm_violations(
        q_don, q_rec_standoff, left_opening=0.1, right_opening=1.0
    )
    assert wide_at_standoff, "opening=1.0 upper finger should still violate clearance against donor crossbar"

    # Good: partial opening (opening=0.38) + 10 mm outer-chord standoff clears all envelopes (>= 10 mm)
    clean = kin.inter_arm_violations(q_don, q_rec_standoff, left_opening=0.1, right_opening=0.38)
    assert clean == [], f"expected clear orthogonal handoff, got {clean}"


def test_distinct_violation_labels_finger_crossbar_and_forearm():
    # 1. Pure finger collision: fingertips pointing toward each other at x = -0.015 and +0.015
    # (TCPs 30 mm apart -> tips are 13 mm past TCP -> tips only 4 mm apart < 10 mm;
    # crossbars/housings are ~130 mm apart).
    q_left_tip = _solve_pose("left", (-0.015, -0.20, 0.95), 0.0, -90.0, 0.0)
    q_right_tip = _solve_pose("right", (0.015, -0.20, 0.95), 0.0, 90.0, 0.0)
    tip_bad = kin.inter_arm_violations(q_left_tip, q_right_tip, left_opening=0.2, right_opening=0.2)
    assert tip_bad == ["finger_collision"]

    # Backing off 15 mm each makes fingertips 34 mm apart -> clear!
    q_left_far = _solve_pose("left", (-0.030, -0.20, 0.95), 0.0, -90.0, 0.0)
    q_right_far = _solve_pose("right", (0.030, -0.20, 0.95), 0.0, 90.0, 0.0)
    assert kin.inter_arm_violations(q_left_far, q_right_far, left_opening=0.2, right_opening=0.2) == []

    # 2. Crossbar collision even when BOTH grippers are fully closed (opening=0.0):
    # Two horizontal arms pointing forward (+y, pitch=0, yaw=0, roll=0) side-by-side at
    # x = -0.050 and +0.050 (10.0 cm separation):
    #   - camera housing to crossbar gap = 0.100 - (0.032 + 0.050) = 1.8 cm (clear!)
    #   - closed fingers (opening=0.0) gap = 0.100 - 0.036 = 6.4 cm (clear!)
    #   - fixed crossbars span [-0.050, +0.050] -> gap = 0.0 cm < 1.0 cm -> pure crossbar_collision!
    q_left_cb = _solve_pose("left", (-0.050, -0.10, 0.92), 0.0, 0.0, 0.0)
    q_right_cb = _solve_pose("right", (0.050, -0.10, 0.92), 0.0, 0.0, 0.0)
    cb_bad = kin.inter_arm_violations(q_left_cb, q_right_cb, left_opening=0.0, right_opening=0.0)
    assert cb_bad == ["crossbar_collision"]

    # Separating by 1.6 cm total (x = +-0.058 -> 11.6 cm apart) clears the +-5.0 cm crossbars!
    q_left_cb_clear = _solve_pose("left", (-0.058, -0.10, 0.92), 0.0, 0.0, 0.0)
    q_right_cb_clear = _solve_pose("right", (0.058, -0.10, 0.92), 0.0, 0.0, 0.0)
    assert kin.inter_arm_violations(
        q_left_cb_clear, q_right_cb_clear, left_opening=0.0, right_opening=0.0
    ) == []

    # 3. Pure forearm collision (wrists/TCPs ~19 cm apart while forearm links collide)
    q_left_fa = (-1.2, 2.0, 0.5, 0.0, 0.0, 0.0)
    q_right_fa = (1.2, 2.0, 0.5, 0.0, 0.0, 0.0)
    assert kin.inter_arm_violations(q_left_fa, q_right_fa, left_opening=0.0, right_opening=0.0) == [
        "forearm_collision"
    ]

    # 4. Canonical deterministic ordering with zero duplicates when multiple parts collide
    q_left_all = (-1.2, 2.2, 0.8, 0.0, 0.0, 0.0)
    q_right_all = (1.2, 2.2, 0.8, 0.0, 0.0, 0.0)
    assert kin.inter_arm_violations(q_left_all, q_right_all, left_opening=0.0, right_opening=0.0) == [
        "camera_housing_collision",
        "crossbar_collision",
        "finger_collision",
        "forearm_collision",
    ]


def test_twist_margin_rad_and_branch_selection_for_plus_minus_90_deg_twists():
    # Direct unit tests of twist_margin_rad against JOINT_LIMITS[5] = (-3.1, 3.1)
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, 0.0]) == pytest.approx(3.1)
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, 2.0]) == pytest.approx(1.1)
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, -2.5]) == pytest.approx(0.6)

    half_pi = math.pi / 2.0
    # +90 deg twist from q6=+2.0 exceeds +3.1 -> negative margin
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, 2.0], +half_pi) == pytest.approx(3.1 - (2.0 + half_pi))
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, 2.0], +half_pi) < 0.0
    # +90 deg twist from 180-deg branch q6=2.0-pi=-1.1416 -> ample positive margin
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, 2.0 - math.pi], +half_pi) > 1.9

    # -90 deg twist from q6=-2.0 exceeds -3.1 -> negative margin
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, -2.0], -half_pi) == pytest.approx((-2.0 - half_pi) + 3.1)
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, -2.0], -half_pi) < 0.0
    assert kin.twist_margin_rad([0, 0, 0, 0, 0, -2.0 + math.pi], -half_pi) > 1.9

    # IK branch selection for +90 deg twist:
    # Construct a target pose directly from a known configuration with q6 = +2.3 rad.
    # Unflipped branch has q6=+2.3 (twist_margin(+pi/2) = -0.77 rad < 0);
    # 180-deg jaw_symmetry branch has q6 = +2.3 - pi = -0.84 rad (twist_margin(+pi/2) > 2.2 rad!).
    q_base_pos = list(_solve_pose("right", (0.20, -0.20, 0.95), 90.0, 0.0, 0.0))
    q_base_pos[5] = 2.3
    fk_pos = kin.fk("right", q_base_pos)
    t_pos = np.eye(4)
    t_pos[:3, :3] = fk_pos[:3, :3]
    t_pos[:3, 3] = kin.tcp(fk_pos)

    # Even seeded right at q_base_pos (zero travel for unflipped), jaw_symmetry + twist_rad=+pi/2
    # must pick the 180-deg branch with positive twist margin!
    sol_pos = kin.best_solution(
        "right", t_pos, q_base_pos, jaw_symmetry=True, twist_rad=+half_pi
    )
    assert sol_pos.ok and not sol_pos.collides
    assert kin.twist_margin_rad(sol_pos.q, +half_pi) > 1.0
    assert sol_pos.q[5] < 0.0

    # Mirror case for -90 deg twist from q6 = -2.3 rad:
    q_base_neg = list(q_base_pos)
    q_base_neg[5] = -2.3
    fk_neg = kin.fk("right", q_base_neg)
    t_neg = np.eye(4)
    t_neg[:3, :3] = fk_neg[:3, :3]
    t_neg[:3, 3] = kin.tcp(fk_neg)

    sol_neg = kin.best_solution(
        "right", t_neg, q_base_neg, jaw_symmetry=True, twist_rad=-half_pi
    )
    assert sol_neg.ok and not sol_neg.collides
    assert kin.twist_margin_rad(sol_neg.q, -half_pi) > 1.0
    assert sol_neg.q[5] > 0.0


def test_best_solution_backward_compatible_defaults():
    q_zero = [0.0] * 6
    target = _pose((0.20, -0.22, 0.92), 45.0, 0.0, 0.0)
    res_default = kin.best_solution("right", target, q_zero)
    res_explicit = kin.best_solution(
        "right",
        target,
        q_zero,
        jaw_symmetry=False,
        other_arm_q=None,
        opening=0.5,
        other_opening=0.5,
        twist_rad=0.0,
    )
    assert res_default == res_explicit


def test_intent_inter_arm_preflight_rejects_colliding_motion_in_zero_steps(tmp_path):
    # Set up orthogonal handoff start:
    # Left (donor) parked at (-0.03, -0.25, 0.90), pitch=90, yaw=90 (camera away), opening=0.1
    # Right (receiver) pre-staged at (+0.07, -0.25, 0.90), pitch=0, yaw=90, roll=90, opening=1.0
    env = TimedEnvironment()
    env.joints["left"] = list(_solve_pose("left", (-0.03, -0.25, 0.90), 90.0, 90.0, 0.0))
    env.joints["right"] = list(_solve_pose("right", (0.07, -0.25, 0.90), 0.0, 90.0, 90.0))
    env.grips["left"] = 0.1
    env.grips["right"] = 1.0
    assert kin.inter_arm_violations(
        env.joints["left"], env.joints["right"], left_opening=0.1, right_opening=1.0
    ) == []

    env, robo, _ = station(tmp_path, env)

    # 1. Driving open receiver (opening=1.0) into P_head=(0.0, -0.25, 0.90) collides with stationary donor
    # -> rejected in 0 native steps (both preview and live execution)!
    wedge_goal = {"xyz": [0.0, -0.25, 0.90], "angles": [0.0, 90.0, 90.0]}
    preview = robo.execute_motion(
        "wedge_preview",
        right=wedge_goal,
        interaction="approach",
        holding={"right": "empty"},
        preview=True,
    )
    assert preview["status"] == "rejected"
    assert not preview["accepted"]
    assert preview["steps_used"] == 0
    assert preview["stop_reason"].startswith("inter_arm_collision:")
    assert "note" in preview and "inter-arm" in preview["note"].lower()
    assert preview["inter_arm_violations"]
    assert not env.actions

    live = robo.execute_motion(
        "wedge_live",
        right=wedge_goal,
        interaction="approach",
        holding={"right": "empty"},
    )
    assert live["status"] == "rejected"
    assert live["needs_model"]
    assert live["steps_used"] == 0
    assert live["stop_reason"].startswith("inter_arm_collision:")
    assert not env.actions

    # 2. Valid cooperative handoff: reduce receiver opening to 0.38 and target 10 mm standoff
    env.grips["right"] = 0.38
    standoff_goal = {"xyz": [0.010, -0.25, 0.90], "angles": [0.0, 90.0, 90.0]}
    clean = robo.execute_motion(
        "clean_handoff",
        right=standoff_goal,
        interaction="approach",
        holding={"right": "empty"},
    )
    assert clean["reached"], clean
    assert len(env.actions) > 0


def test_intent_inter_arm_preflight_rejects_synchronized_dual_arm_crossing(tmp_path):
    env = TimedEnvironment()
    env.joints["left"] = list(_solve_pose("left", (-0.08, -0.18, 0.94), 0.0, -90.0, 0.0))
    env.joints["right"] = list(_solve_pose("right", (0.08, -0.18, 0.94), 0.0, 90.0, 0.0))
    env.grips["left"] = 0.4
    env.grips["right"] = 0.4
    env, robo, _ = station(tmp_path, env)

    # Both arms converge simultaneously toward center -> colliding waypoints rejected in 0 steps
    result = robo.execute_motion(
        "dual_collide",
        left={"xyz": [-0.01, -0.18, 0.94], "angles": [0.0, -90.0, 0.0]},
        right={"xyz": [0.01, -0.18, 0.94], "angles": [0.0, 90.0, 0.0]},
        interaction="free",
        holding={"left": "empty", "right": "empty"},
    )
    assert result["status"] == "rejected"
    assert result["steps_used"] == 0
    assert result["stop_reason"].startswith("inter_arm_collision:")
    assert "finger_collision" in result["inter_arm_violations"]
    assert not env.actions


def test_intent_inter_arm_preflight_allows_retreat_from_tight_start_blocks_deeper_push(tmp_path):
    env = TimedEnvironment()
    # Start already slightly inside 10 mm finger clearance (~4 mm apart)
    env.joints["left"] = list(_solve_pose("left", (-0.015, -0.20, 0.95), 0.0, -90.0, 0.0))
    env.joints["right"] = list(_solve_pose("right", (0.015, -0.20, 0.95), 0.0, 90.0, 0.0))
    env.grips["left"] = 0.2
    env.grips["right"] = 0.2
    assert kin.inter_arm_violations(
        env.joints["left"], env.joints["right"], left_opening=0.2, right_opening=0.2
    ) == ["finger_collision"]
    env, robo, _ = station(tmp_path, env)

    # Pushing even closer is rejected in 0 steps
    push = robo.execute_motion(
        "push_deeper",
        right={"xyz": [0.005, -0.20, 0.95], "angles": [0.0, 90.0, 0.0]},
        interaction="approach",
        holding={"right": "empty"},
    )
    assert push["status"] == "rejected"
    assert push["steps_used"] == 0
    assert push["stop_reason"].startswith("inter_arm_collision:")
    assert not env.actions

    # Retreating away to clear separation is permitted so the robot is never deadlocked
    retreat = robo.execute_motion(
        "retreat_clear",
        right={"xyz": [0.08, -0.20, 0.95], "angles": [0.0, 90.0, 0.0]},
        interaction="free",
        holding={"right": "empty"},
    )
    assert retreat["reached"], retreat
    assert len(env.actions) > 0


def test_min_clearance_sensitivity_and_input_validation():
    # Two fingertips pointing at each other with ~4 mm tip-to-tip clearance
    q_l = _solve_pose("left", (-0.015, -0.20, 0.95), 0.0, -90.0, 0.0)
    q_r = _solve_pose("right", (0.015, -0.20, 0.95), 0.0, 90.0, 0.0)
    assert kin.inter_arm_violations(q_l, q_r, left_opening=0.2, right_opening=0.2, min_clearance_m=0.010) == [
        "finger_collision"
    ]
    assert kin.inter_arm_violations(q_l, q_r, left_opening=0.2, right_opening=0.2, min_clearance_m=0.002) == []

    # housing_exposure_penalty cutoff strictly > 0.38 m returns 0.0
    q_far_l = _solve_pose("left", (-0.20, -0.20, 0.92), 90.0, -90.0, 0.0)
    q_far_r = _solve_pose("right", (0.20, -0.20, 0.92), 90.0, 90.0, 0.0)
    assert np.linalg.norm(kin.tcp(kin.fk("left", q_far_l)) - kin.tcp(kin.fk("right", q_far_r))) > 0.38
    assert kin.housing_exposure_penalty("left", q_far_l, q_far_r) == 0.0

    # Input validation
    with pytest.raises(ValueError, match="arm"):
        kin.housing_exposure_penalty("middle", [0.0] * 6, [0.0] * 6)
    with pytest.raises(ValueError, match="6 joints"):
        kin.twist_margin_rad([0.0] * 5)
    with pytest.raises(ValueError, match="6 joints"):
        kin.inter_arm_violations([0.0] * 5, [0.0] * 6)


def test_inter_arm_min_distance_public_export_and_alias():
    assert "inter_arm_min_distance" in kin.__all__
    assert kin._inter_arm_min_distance is kin.inter_arm_min_distance
    q_l = _solve_pose("left", (-0.015, -0.20, 0.95), 0.0, -90.0, 0.0)
    q_r = _solve_pose("right", (0.015, -0.20, 0.95), 0.0, 90.0, 0.0)
    d_tight = kin.inter_arm_min_distance(q_l, q_r, left_opening=0.2, right_opening=0.2)
    assert 0.0 < d_tight < 0.010
    q_r_far = _solve_pose("right", (0.055, -0.20, 0.95), 0.0, 90.0, 0.0)
    d_clear = kin.inter_arm_min_distance(q_l, q_r_far, left_opening=0.2, right_opening=0.2)
    assert d_clear > d_tight + 0.03


def test_robodojo_ik_rejects_insufficient_twist_margin_and_flips_with_jaw_symmetry(tmp_path):
    from codeaspolicy.coordination import Pose

    env = TimedEnvironment()
    q_base = list(_solve_pose("right", (0.20, -0.20, 0.95), 90.0, 0.0, 0.0))
    q_base[5] = 2.3  # q6 + 1.57 = 3.87 > 3.1
    env.joints["right"] = list(q_base)
    _, robo, _ = station(tmp_path, env)

    fk_pos = kin.fk("right", q_base)
    tcp_pos = kin.tcp(fk_pos)
    pose = Pose(
        (float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])),
        kin.quat_wxyz(fk_pos[:3, :3]),
    )

    # Without jaw_symmetry, q6 + 1.57 > 3.1 must fail with reason="insufficient_twist_margin"
    sol_no_sym = robo.ik(
        "right",
        pose,
        seed=q_base,
        jaw_symmetry=False,
        twist_rad=1.57,
        check_other_arm=False,
    )
    assert sol_no_sym.ok is False
    assert sol_no_sym.collides is False
    assert sol_no_sym.reason == "insufficient_twist_margin"

    # Also enforced when continuous=True
    sol_cont = robo.ik(
        "right",
        pose,
        seed=q_base,
        jaw_symmetry=False,
        continuous=True,
        twist_rad=1.57,
        check_other_arm=False,
    )
    assert sol_cont.ok is False
    assert sol_cont.collides is False
    assert sol_cont.reason == "insufficient_twist_margin"

    # With jaw_symmetry=True, IK flips 180 deg into negative q6 and succeeds!
    sol_sym = robo.ik(
        "right",
        pose,
        seed=q_base,
        jaw_symmetry=True,
        twist_rad=1.57,
        check_other_arm=False,
    )
    assert sol_sym.ok is True
    assert sol_sym.reason is None
    assert sol_sym.flipped_jaw is True
    assert sol_sym.joints[5] + 1.57 <= kin.JOINT_LIMITS[5][1]

    # When neither jaw branch has sufficient twist headroom (e.g. twist_rad=6.5 > joint6 span),
    # ik returns ok=False with reason="insufficient_twist_margin".
    sol_impossible = robo.ik(
        "right",
        pose,
        seed=q_base,
        jaw_symmetry=True,
        twist_rad=6.5,
        check_other_arm=False,
    )
    assert sol_impossible.ok is False
    assert sol_impossible.collides is False
    assert sol_impossible.reason == "insufficient_twist_margin"


