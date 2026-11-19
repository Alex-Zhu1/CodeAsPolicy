"""PEAR Visual-Delta protocol, guarded compact nudges and the closed visual-delta loop."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from codeaspolicy import visual_delta as vd
from codeaspolicy._backend import ToolError
from codeaspolicy.hosts import robodojo_kinematics as kin
from codeaspolicy.hosts import robodojo_tools as motion
from codeaspolicy.hosts import robodojo_visual_delta as rvd
from codeaspolicy.hosts.robodojo_native import STATEFUL, NativeBackend, NativeToolkit
from codeaspolicy.prompt import api_reference
from codeaspolicy.robodojo import (
    STATEFUL_TOOLS,
    DeltaGrasp,
    FeatureAlignment,
    Nudge,
    RobodojoRobot,
    VisualAlign,
)
from codeaspolicy.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from tests.test_coordination import JointEnvironment

START = {"right": [.25, -.15, .86], "left": [-.25, -.15, .86]}


def _place(env, z=.86):
    """Both grippers pointing straight down above the table (clear of joint limits)."""
    for arm, xyz in START.items():
        pose = np.eye(4)
        pose[:3, :3] = kin.orientation(90, 0, 0)
        pose[:3, 3] = [xyz[0], xyz[1], z]
        solved = kin.best_solution(arm, pose, env.joints[arm])
        assert solved.ok and not solved.collides
        env.joints[arm] = list(solved.q)
    return env


class StallEnvironment(JointEnvironment):
    """A rigid surface: the right grasp point cannot go below ``floor`` (no progress)."""

    def __init__(self, floor=.845):
        super().__init__()
        self.floor = floor

    def step(self, action):
        before = copy.deepcopy(self.joints)
        result = super().step(action)
        if kin.tcp(kin.fk("right", self.joints["right"]))[2] < self.floor:
            self.joints["right"] = before["right"]
            return self.get_obs(), 0.0, False, {"status": self.get_status()}
        return result


def station(tmp_path, env=None):
    env = _place(env or JointEnvironment())
    host = NativeToolkit(env, None, tmp_path / "native")
    return env, RobodojoOfficialColourRobot(NativeBackend(host)), host


def tcp(env, arm="right"):
    return kin.tcp(kin.fk(arm, env.joints[arm]))


def answer(dz_cm=0.0, *, dx=0.0, dy=0.0, yaw=0.0, grip=0, done=None, think="align"):
    action = {"arm": {"x": dx, "y": dy, "z": dz_cm, "roll": 0, "pitch": 0, "yaw": yaw},
              "gripper": {"width": grip}}
    if done is not None:
        action["done"] = done
    return f"<think>{think}</think>\n```json\n{json.dumps(action)}\n```"


# -- protocol -------------------------------------------------------------------


def test_parse_think_and_fenced_json():
    parsed = vd.parse_visual_delta_response(answer(-1.5, dx=0.3, yaw=0.1, grip=-1))
    assert parsed["think"] == "align" and parsed["done"] is False
    assert parsed["action"]["arm"] == {"x": 0.3, "y": 0.0, "z": -1.5, "roll": 0.0,
                                       "pitch": 0.0, "yaw": 0.1}
    assert parsed["action"]["gripper"] == {"width": -1.0}
    bare = vd.parse_visual_delta_response(
        'ok {"arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, '
        '"gripper": {"width": 0}, "done": true}')
    assert bare["done"] is True and bare["think"] is None


@pytest.mark.parametrize("text, match", [
    ("no json here", "no fenced json"),
    ("```json\n{\"arm\": \n```", "invalid JSON"),
    (answer().replace('"yaw": 0.0', '"yaw": true'), "must be a number"),
    (answer().replace('"roll": 0, ', ''), "missing fields"),
    (answer().replace('"gripper"', '"grip"'), "unknown keys"),
    (answer(done="yes"), "done must be a boolean"),
    (answer().replace('"x": 0.0', '"x": 99'), "outside"),
    (answer().replace('"width": 0', '"width": 0, "force": 1'), "unknown fields"),
])
def test_strict_validation_rejects_off_protocol_answers(text, match):
    with pytest.raises(vd.VisualDeltaError, match=match):
        vd.parse_visual_delta_response(text)


def test_non_strict_fills_zeros_and_bimanual_mode():
    loose = vd.validate_delta_action({"arm": {"z": -1}}, "left", strict=False)
    assert loose["arm"] == {**vd.zero_arm(), "z": -1.0} and loose["gripper"] == {"width": 0.0}
    zero = {k: 0 for k in vd.ARM_FIELDS}
    both = vd.validate_delta_action({"left_arm": dict(zero, x=1), "right_arm": zero,
                                     "left_gripper": {"width": 1},
                                     "right_gripper": {"width": -2}}, "both")
    assert both["left_arm"]["x"] == 1.0 and both["right_gripper"] == {"width": -1.0}
    with pytest.raises(vd.VisualDeltaError, match="arm mode 'both'"):
        vd.validate_delta_action({"left_arm": zero}, "right")
    with pytest.raises(vd.VisualDeltaError, match="single-arm keys"):
        vd.validate_delta_action({"arm": zero, "gripper": {"width": 0}}, "both")


def test_gripper_sign_translation_never_mixes_conventions():
    assert vd.GRIP_ZERO_M == motion.GRIP_ZERO_M and vd.GRIP_SPAN_M == motion.GRIP_SPAN_M
    assert [vd.pear_gripper_to_robodojo(w) for w in (1, 0.5, 0, -1, -3)] == [
        RobodojoRobot.OPEN, RobodojoRobot.OPEN, RobodojoRobot.KEEP, RobodojoRobot.CLOSE,
        RobodojoRobot.CLOSE]
    assert [vd.robodojo_gripper_to_pear(c) for c in (-1, 0, 1, None)] == [1, 0, -1, 0]
    # Native ee command: 1.0 opens, 0.0 closes (robodojo_tools._command)
    assert vd.ee_command_to_pear(motion._command(RobodojoRobot.OPEN)) == vd.PEAR_OPEN
    assert vd.ee_command_to_pear(motion._command(RobodojoRobot.CLOSE)) == vd.PEAR_CLOSE
    assert vd.ee_command_to_pear(motion._command(RobodojoRobot.KEEP)) == vd.PEAR_KEEP
    action = vd.parse_visual_delta_response(answer(grip=1))["action"]
    assert vd.action_arms(action, "right")["right"]["gripper"] == RobodojoRobot.OPEN
    assert vd.width_cm(1.0) == 11.1 and vd.width_cm(0.0) == 0.2


def test_units_frames_and_round_trip():
    spec = vd.arm_delta_to_motion({"x": 1.0, "y": -2.0, "z": 0.5, "roll": 0.1,
                                   "pitch": -0.2, "yaw": 0.05})
    assert spec["dxyz"] == pytest.approx([0.01, -0.02, 0.005])
    assert spec["dpitch"] == pytest.approx(math.degrees(-0.2))
    back = vd.format_action({"right": dict(spec, gripper=RobodojoRobot.CLOSE)}, "right",
                            digits=2)
    assert back["arm"]["x"] == 1.0 and back["arm"]["roll"] == pytest.approx(0.1)
    assert back["gripper"] == {"width": -1}
    before = {"xyz": [0.1, 0.2, 0.9], "pitch": 90.0, "yaw": 179.0, "roll": 0.0}
    after = {"xyz": [0.11, 0.2, 0.88], "pitch": 85.0, "yaw": -179.0, "roll": 0.0}
    delta = vd.pose_delta(before, after)
    assert delta["x"] == pytest.approx(1.0) and delta["z"] == pytest.approx(-2.0)
    assert delta["yaw"] == pytest.approx(math.radians(2.0))  # wrapped, not -358 deg
    state = vd.format_state({"right": dict(after, opening=0.3)}, "right")
    assert state == {"arm": {"x": 11.0, "y": 20.0, "z": 88.0, "roll": 0.0,
                             "pitch": round(math.radians(85), 2),
                             "yaw": round(math.radians(-179), 2)},
                     "gripper": {"width": vd.width_cm(0.3)}}


def test_near_vertical_yaw_roll_ambiguity_is_folded_into_heading():
    # Two nearly identical downward poses whose raw angles differ by ~90 deg each.
    before = {"xyz": [0, 0, 0.9], "pitch": 88.5, "yaw": 10.0, "roll": 80.0}
    after = {"xyz": [0, 0, 0.9], "pitch": 88.5, "yaw": -80.0, "roll": -10.0}
    r0 = kin.orientation(before["pitch"], before["yaw"], before["roll"])
    r1 = kin.orientation(after["pitch"], after["yaw"], after["roll"])
    assert kin.rotation_error(r0, r1) < 3.0
    delta = vd.pose_delta(before, after)
    assert delta["roll"] == 0.0 and abs(delta["yaw"]) < 1e-9
    assert vd.canonical_angles(88.5, 10.0, 80.0) == (88.5, -70.0, 0.0)
    assert vd.canonical_angles(-88.5, 10.0, 80.0) == (-88.5, 90.0, 0.0)  # y + r upward
    assert vd.canonical_angles(60.0, 10.0, 80.0) == (60.0, 10.0, 80.0)
    start, end = np.eye(4), np.eye(4)
    start[:3, :3], end[:3, :3] = r0, kin._rot_z(math.radians(0.5)) @ r0
    achieved = rvd.achieved_delta(start, end)["world"]
    assert achieved["droll"] == 0.0 and achieved["dyaw"] == pytest.approx(0.5, abs=0.1)
    state = vd.format_state({"right": dict(before, opening=1.0)}, "right")
    assert state["arm"]["roll"] == 0.0 and state["arm"]["yaw"] == round(math.radians(-70), 2)
    # Away from vertical the raw differences are kept (exactly reproducible).
    assert vd.angle_delta((45, 10, 80), (46, -80, -10)) == (1.0, 270.0 - 360.0, -90.0)


def test_feedback_status_mapping():
    assert vd.feedback_status({"reached": True}) == "reached"
    assert vd.feedback_status({"status": "stopped", "stalled": True}) == "stopped"
    assert vd.feedback_status({"status": "rejected", "stop_reason": "ik_path_rejected",
                               "ik_error": "x"}) == "out_of_reach"
    assert vd.feedback_status({"status": "partial", "stop_reason": "step_budget"}) == "stopped"


def test_openai_client_sends_labelled_views_and_reads_content(monkeypatch):
    monkeypatch.setenv("CODEASPOLICY_DELTA_BASE_URL", "http://delta.test/v1/")
    monkeypatch.setenv("CODEASPOLICY_DELTA_MODEL", "qwen-pear")
    monkeypatch.setenv("CODEASPOLICY_DELTA_API_KEY_ENV", "PEAR_KEY")
    monkeypatch.setenv("PEAR_KEY", "secret")
    sent = []

    def transport(url, body, timeout_s):
        sent.append((url, body))
        return 200, {"choices": [{"message": {"content": answer(-0.5)}}]}

    client = vd.OpenAIVisualDeltaClient(transport=transport)
    assert client.api_key == "secret" and client.model == "qwen-pear"
    png = vd.encode_png(np.zeros((4, 4, 3), np.uint8))
    request = {"system": vd.SYSTEM_PROMPT, "user": "Goal: x",
               "images": [{"label": "cam_head", "png": png},
                          {"label": "cam_right_wrist_crop", "png": png}]}
    text = client.predict(request)
    assert vd.parse_visual_delta_response(text)["action"]["arm"]["z"] == -0.5
    url, body = sent[0]
    assert url == "http://delta.test/v1/chat/completions" and body["model"] == "qwen-pear"
    parts = body["messages"][-1]["content"]
    assert [p["text"] for p in parts if p["type"] == "text"][:2] == [
        "[cam_head]", "[cam_right_wrist_crop]"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["messages"][0] == {"role": "system", "content": vd.SYSTEM_PROMPT}
    failing = vd.OpenAIVisualDeltaClient(transport=lambda *a: (503, "busy"))
    with pytest.raises(RuntimeError, match="503"):
        failing.predict(request)


# -- nudge ----------------------------------------------------------------------


def test_nudge_world_frame_is_compact_and_holds_the_other_arm(tmp_path):
    env, robo, _ = station(tmp_path)
    left = list(env.joints["left"])
    start = tcp(env)
    result = robo.nudge((0.0, 0.0, -0.01))
    assert isinstance(result, Nudge)
    assert result.status == "completed" and result.reached and not result.stalled
    assert result.steps_used == len(env.actions) <= 4  # 1 cm: ~2-3 actions, no ramp
    assert np.allclose(tcp(env) - start, [0, 0, -0.01], atol=0.002)
    assert result.achieved["dxyz"] == pytest.approx([0, 0, -0.01], abs=0.002)
    assert all(a["left_arm_joint_state"] == left for a in env.actions)
    assert result.final_xyz == pytest.approx(tuple(tcp(env)), abs=1e-3)
    json.dumps(result.arms)  # plain JSON for envelopes and journals


def test_nudge_tcp_frame_pushes_along_the_fingers(tmp_path):
    env, robo, _ = station(tmp_path)
    start = tcp(env)
    result = robo.nudge((0.01, 0.0, 0.0), frame="tcp")  # pointing straight down
    assert result.reached, result
    assert np.allclose(tcp(env) - start, [0, 0, -0.01], atol=0.002)
    assert result.achieved["dxyz"][0] == pytest.approx(0.01, abs=0.002)
    turned = robo.nudge(dyaw=10.0)
    assert turned.reached and turned.achieved["dyaw"] == pytest.approx(10.0, abs=1.0)


def test_compact_nudge_avoids_timed_spline_step_inflation(tmp_path):
    from tests.test_robodojo_trajectory import TimedEnvironment

    env, robo, _ = station(tmp_path, TimedEnvironment())
    goal = list(tcp(env))
    goal[2] -= 0.01
    preview = robo.move_to(goal, preview=True)
    assert preview["accepted"], preview
    result = robo.nudge((0.0, 0.0, -0.01))
    assert result.reached, result
    assert result.steps_used <= 4 < preview["planned_steps"]
    assert not any("trajectory" in str(key) for a in env.wire_actions for key in a)


def test_stop_on_stall_ends_cleanly_at_contact(tmp_path):
    env, robo, host = station(tmp_path, StallEnvironment(floor=.845))
    result = robo.nudge((0.0, 0.0, -0.04), stall_steps=2)
    assert result.status == "stopped" and result.stalled
    assert result.stop_reason == "contact_or_stall" and not result.reached
    assert result.steps_used < result.planned_steps
    assert tcp(env)[2] == pytest.approx(.845, abs=0.006)
    assert result.final_xyz[2] == pytest.approx(tcp(env)[2], abs=1e-3)
    receipt = robo._ledger.records[-1].receipt
    assert receipt["status"] == "partial" and receipt["effect_known"]
    pushed = robo.nudge((0.0, 0.0, -0.02), stop_on_stall=False)
    assert pushed.status == "partial" and not pushed.stalled
    assert pushed.steps_used >= pushed.planned_steps
    assert pushed.arms["right"]["dist_to_target_m"] > 0.01


def test_preflight_rejections_use_zero_actions(tmp_path):
    env, robo, _ = station(tmp_path)
    too_far = robo.nudge((0.0, 0.0, -0.08))
    assert too_far.status == "rejected" and too_far.stop_reason == "delta_exceeds_limit"
    into_table = robo.nudge((0.0, 0.0, -0.11), max_linear_m=0.12)
    assert into_table.status == "rejected" and into_table.stop_reason == "ik_path_rejected"
    assert "table" in into_table.ik_error
    assert not env.actions
    with pytest.raises(ValueError):
        robo.nudge((0, 0), arm="right")
    with pytest.raises(ValueError):
        robo.nudge(arm="both", dxyz=(0, 0, 0.01))


def test_budget_and_reserve_limits(tmp_path):
    env, robo, _ = station(tmp_path)
    env.step_limit = 10
    reserved = robo.nudge((0.0, 0.0, 0.01), reserve_steps=10)
    assert reserved.status == "rejected" and reserved.stop_reason == "reserve_steps_exhausted"
    assert not env.actions
    capped = robo.nudge((0.0, 0.0, 0.03), max_steps=2)
    assert capped.steps_used == 2 == len(env.actions)
    assert capped.status == "partial" and capped.stop_reason == "step_budget"
    rest = robo.nudge((0.0, 0.0, 0.03), reserve_steps=5)
    assert len(env.actions) <= 5 and rest.stop_reason == "step_budget"


def test_gripper_switch_and_both_arms(tmp_path):
    env, robo, _ = station(tmp_path)
    closed = robo.nudge((0.0, 0.0, -0.005), gripper=robo.CLOSE)
    assert all(a["right_ee_joint_state"] == [0.0] for a in env.actions)
    assert closed.holding and closed.reached
    start = {a: tcp(env, a) for a in kin.ARMS}
    both = robo.nudge(arm="both", left={"dxyz": [0, 0, 0.01]},
                      right={"dxyz": [0.01, 0, 0], "gripper": robo.KEEP})
    assert both.reached and set(both.arms) == {"left", "right"}
    assert np.allclose(tcp(env, "left") - start["left"], [0, 0, 0.01], atol=0.002)
    assert np.allclose(tcp(env, "right") - start["right"], [0.01, 0, 0], atol=0.002)
    assert env.actions[-1]["right_ee_joint_state"] == [0.0]  # KEEP keeps the squeeze


# -- the closed visual-delta loop ------------------------------------------------


def test_host_loop_feeds_physical_outcome_back_and_stops_on_done(tmp_path):
    env, robo, host = station(tmp_path)
    policy = vd.ScriptedPolicy([answer(-1.0), answer(0.5, dx=0.5),
                                answer(0.0, done=True)])
    rvd.set_policy(host._primitives, policy)
    start = tcp(env)
    result = robo.visual_align_and_insert("seat the held plug in the socket below")
    assert isinstance(result, VisualAlign)
    assert result.status == "completed" and result.done and result.stop_reason == "done"
    assert result.rounds_used == 3 and result.steps_used == len(env.actions) > 0
    assert np.allclose(tcp(env) - start, [0.005, 0, -0.005], atol=0.003)
    first, second, third = policy.requests
    assert [i["label"] for i in first["images"]] == [
        "cam_head", "cam_right_wrist", "cam_right_wrist_crop"]
    assert first["images"][0]["png"].startswith(b"\x89PNG")
    assert first["feedback"] is None and "first turn" in first["user"]
    assert second["feedback"]["status"] == "reached"
    assert second["feedback"]["achieved"]["arm"]["z"] == pytest.approx(-1.0, abs=0.2)
    assert '"status": "reached"' in second["user"]
    assert third["state"]["arm"]["z"] == pytest.approx(start[2] * 100 - 0.5, abs=0.3)
    record = robo._ledger.records[-1]
    assert record.tool == "visual_align_and_insert" and record.stateful
    assert record.receipt["status"] == "completed"
    json.dumps({"rounds": result.rounds, "feedback": result.last_feedback})  # plain JSON


def test_contact_stall_is_reported_and_can_complete_the_loop(tmp_path):
    env, robo, host = station(tmp_path, StallEnvironment(floor=.845))
    policy = vd.ScriptedPolicy([answer(-3.0), answer(-1.0)])
    rvd.set_policy(host._primitives, policy)
    result = robo.visual_align_and_insert("press the button below", max_rounds=4)
    # two stalled presses, then the exhausted script answers done
    assert result.stop_reason == "done" and result.stalled and result.rounds_used == 3
    assert policy.requests[1]["feedback"]["status"] == "stopped"
    assert policy.requests[1]["feedback"]["reason"] == "contact_or_stall"
    assert policy.requests[2]["feedback"]["stalled"] is True
    env2, robo2, host2 = station(tmp_path / "b", StallEnvironment(floor=.845))
    rvd.set_policy(host2._primitives, vd.ScriptedPolicy([answer(-3.0), answer(-1.0)]))
    done = robo2.visual_align_and_insert("press the button below",
                                         stop_on_contact_complete=True)
    assert done.status == "completed" and done.contact_complete
    assert done.stop_reason == "contact_complete" and done.rounds_used == 1


def test_local_callable_policy_runs_in_process_one_nudge_per_move(tmp_path):
    env, robo, _ = station(tmp_path)
    seen = []

    def policy(request):
        seen.append(request)
        if len(seen) < 3:
            return {"arm": {"x": 0, "y": 0, "z": -0.5, "roll": 0, "pitch": 0, "yaw": 0},
                    "gripper": {"width": 0}}
        return {"arm": {k: 0 for k in vd.ARM_FIELDS}, "gripper": {"width": 0}}

    result = robo.visual_align_and_insert("align", policy=policy, crop_wrist=False)
    assert result.status == "completed" and result.stop_reason == "converged"
    assert result.rounds_used == 3 and result.steps_used == len(env.actions)
    tools = [r.tool for r in robo._ledger.records]
    assert tools.count("nudge") == 2 and "visual_align_and_insert" not in tools
    assert [i["label"] for i in seen[0]["images"]] == ["cam_head", "cam_right_wrist"]
    assert seen[1]["feedback"]["status"] == "reached"


def test_invalid_answers_policy_errors_and_budgets_stop_without_crashing(tmp_path):
    env, robo, host = station(tmp_path)
    rvd.set_policy(host._primitives, vd.ScriptedPolicy(["nonsense", "still nonsense"]))
    bad = robo.visual_align_and_insert("align")
    assert bad.stop_reason == "invalid_policy_response" and bad.status == "rejected"
    assert bad.rounds[0]["feedback"]["status"] == "invalid_response" and not env.actions
    rvd.set_policy(host._primitives, vd.ScriptedPolicy([RuntimeError("server down")]))
    failed = robo.visual_align_and_insert("align")
    assert failed.stop_reason == "policy_error" and "server down" in failed.policy_error
    rvd.set_policy(host._primitives, vd.ScriptedPolicy([answer(-10.0)] * 8))
    capped = robo.visual_align_and_insert("align", max_steps=3)
    assert capped.steps_used <= 3 == capped.step_allowance
    assert capped.stop_reason == "step_budget" and capped.rounds[0]["clamped"]
    assert capped.rounds[0]["executed"]["right"]["dxyz"][2] == pytest.approx(-0.03)
    env.step_limit = len(env.actions) + 10
    calls = vd.ScriptedPolicy([answer(-1.0)])
    rvd.set_policy(host._primitives, calls)
    reserved = robo.visual_align_and_insert("align", reserve_steps=20)
    assert reserved.stop_reason == "reserve_steps_exhausted" and not calls.requests


def test_out_of_reach_feedback_and_no_progress(tmp_path):
    env, robo, host = station(tmp_path)
    policy = vd.ScriptedPolicy([answer(-30.0)] * 5)  # clamped to 12 cm: below the table
    rvd.set_policy(host._primitives, policy)
    result = robo.visual_align_and_insert("align", max_linear_m=0.12, max_rounds=5)
    assert result.stop_reason == "no_progress" and result.steps_used == 0
    assert policy.requests[1]["feedback"]["status"] == "out_of_reach"
    assert result.status == "rejected" and not env.actions


# -- align_feature ---------------------------------------------------------------


def test_align_feature_vertical_transverse_only_keeps_axial_clearance(tmp_path):
    env, robo, _ = station(tmp_path)
    start = tcp(env)
    feature = start + np.array([0.008, -0.006, -0.04])  # held distal tip 4 cm below TCP
    target = feature + np.array([-0.012, 0.009, -0.015])  # 15 mm transverse, 15 mm below tip
    res = robo.align_feature(feature.tolist(), target.tolist(), arm="right")
    assert isinstance(res, FeatureAlignment)
    assert res.status == "completed" and res.aligned and res.reached and not res.stalled
    assert res.preview is False and res.accepted is True
    assert res.axial_gap_m == pytest.approx(0.015, abs=1e-4)
    assert res.transverse_delta == pytest.approx((-0.012, 0.009, 0.0), abs=1e-4)
    assert res.planned_dxyz == pytest.approx((-0.012, 0.009, 0.0), abs=1e-4)
    assert res.transverse_error_before_m == pytest.approx(0.015, abs=1e-4)
    assert res.transverse_error_after_m <= 0.0025
    assert res.stroke_status is None and res.stroke_steps_used == 0
    assert res.align_steps_used == res.steps_used == len(env.actions) > 0
    shift = tcp(env) - start
    assert shift == pytest.approx([-0.012, 0.009, 0.0], abs=0.0025)
    assert abs(shift[2]) < 0.002  # vertical standoff preserved


def test_align_feature_horizontal_axis_standoff_and_guarded_stroke(tmp_path):
    env, robo, _ = station(tmp_path)
    start = tcp(env)
    # Horizontal engagement along +y (e.g., side plugging / hook hanging);
    # transverse plane is world xz.
    feature = start + np.array([0.004, 0.05, -0.005])
    target = feature + np.array([-0.010, 0.025, 0.008])
    res = robo.align_feature(
        feature.tolist(),
        target.tolist(),
        arm="right",
        axis=(0.0, 2.0, 0.0),  # non-unit vector normalized automatically
        standoff_m=0.010,
        stroke_m=0.020,
        max_steps=25,
    )
    assert res.status == "completed" and res.aligned and res.reached and not res.stalled
    assert res.axis == pytest.approx((0.0, 1.0, 0.0), abs=1e-5)
    assert res.axial_gap_m == pytest.approx(0.025, abs=1e-4)
    assert res.transverse_delta == pytest.approx((-0.010, 0.0, 0.008), abs=1e-4)
    # standoff_m=0.010 from 0.025 gap => +0.015 m axial approach during alignment
    assert res.planned_dxyz == pytest.approx((-0.010, 0.015, 0.008), abs=1e-4)
    assert res.standoff_m == pytest.approx(0.010)
    assert res.transverse_error_after_m <= 0.0025
    assert res.stroke_status == "completed"
    assert res.requested_stroke_m == pytest.approx(0.020)
    assert res.achieved_stroke_m == pytest.approx(0.020, abs=0.0025)
    assert res.align_steps_used > 0 and res.stroke_steps_used > 0
    assert res.steps_used == res.align_steps_used + res.stroke_steps_used == len(env.actions)
    # Net shift = planned_dxyz + stroke along +y = (-0.010, +0.035, +0.008)
    assert tcp(env) - start == pytest.approx([-0.010, 0.035, 0.008], abs=0.003)


def test_align_feature_guarded_stroke_stalls_cleanly_at_contact(tmp_path):
    env, robo, _ = station(tmp_path, StallEnvironment(floor=.845))
    start = tcp(env)
    feature = start + np.array([0.006, 0.0, -0.02])
    target = feature + np.array([-0.008, 0.004, -0.02])
    res = robo.align_feature(
        feature.tolist(), target.tolist(), arm="right", stroke_m=0.035, max_steps=25
    )
    assert res.aligned is True and res.transverse_error_after_m <= 0.0025
    assert res.status == "stopped" and res.stalled is True and res.reached is False
    assert res.stop_reason == "contact_or_stall"
    assert res.stroke_status == "stopped"
    assert res.stroke_steps_used > 0
    assert res.achieved_stroke_m is not None and 0.005 <= res.achieved_stroke_m < 0.025
    assert res.final_xyz[2] == pytest.approx(.845, abs=0.006)
    assert res.steps_used == res.align_steps_used + res.stroke_steps_used == len(env.actions)


def test_align_feature_skips_stroke_when_alignment_stalls(tmp_path):
    class SideWallEnv(JointEnvironment):
        def __init__(self, x_wall):
            super().__init__()
            self.x_wall = x_wall

        def step(self, action):
            before = copy.deepcopy(self.joints)
            out = super().step(action)
            if kin.tcp(kin.fk("right", self.joints["right"]))[0] < self.x_wall:
                self.joints["right"] = before["right"]
                return self.get_obs(), 0.0, False, {"status": self.get_status()}
            return out

    start_x = START["right"][0]
    env, robo, _ = station(tmp_path, SideWallEnv(x_wall=start_x - 0.003))
    start = tcp(env)
    feature = start.copy()
    target = start + np.array([-0.020, 0.0, -0.015])
    res = robo.align_feature(
        feature.tolist(), target.tolist(), arm="right", stroke_m=0.020, max_steps=20
    )
    assert res.aligned is False
    assert res.transverse_error_after_m > 0.010
    assert res.status == "stopped" and res.stalled is True
    assert res.stroke_status == "skipped" and res.stroke_steps_used == 0
    assert res.achieved_stroke_m == 0.0
    assert abs(tcp(env)[2] - start[2]) < 0.002  # never descended


def test_align_feature_zero_delta_skips_alignment_in_zero_steps(tmp_path):
    env, robo, _ = station(tmp_path)
    start = tcp(env)
    feature = [0.25, -0.15, 0.82]
    target = [0.2502, -0.1502, 0.80]  # 0.28 mm transverse <= 0.5 mm threshold
    res = robo.align_feature(feature, target, arm="right")
    assert res.status == "completed" and res.aligned and res.reached
    assert res.align_steps_used == 0 and res.steps_used == 0 and not env.actions
    assert res.transverse_error_after_m == pytest.approx(res.transverse_error_before_m)

    # With stroke_dxyz: skips alignment (0 steps) and immediately runs stroke
    with_stroke = robo.align_feature(
        feature, target, arm="right", stroke_dxyz=(0.0, 0.0, -0.015)
    )
    assert with_stroke.status == "completed" and with_stroke.aligned and with_stroke.reached
    assert with_stroke.align_steps_used == 0 and with_stroke.stroke_steps_used > 0
    assert with_stroke.achieved_stroke_m == pytest.approx(0.015, abs=0.0025)
    assert tcp(env)[2] == pytest.approx(start[2] - 0.015, abs=0.0025)


def test_align_feature_preview_is_readonly_and_zero_steps(tmp_path):
    env, robo, _ = station(tmp_path)
    start = tcp(env)
    feature = start.copy()
    target = start + np.array([0.010, -0.005, -0.015])
    plan = robo.align_feature(
        feature.tolist(), target.tolist(), arm="right", stroke_m=0.020, preview=True
    )
    assert isinstance(plan, FeatureAlignment)
    assert plan.preview is True and plan.accepted is True and plan.status == "planned"
    assert plan.steps_used == 0 and plan.planned_steps > 0 and not env.actions
    assert plan.planned_dxyz == pytest.approx((0.010, -0.005, 0.0), abs=1e-4)
    assert not robo._ledger.stateful_calls()

    # Refusal in preview: stroke below table costs 0 steps and reports ik_error
    bad_table = robo.align_feature(
        feature.tolist(),
        target.tolist(),
        arm="right",
        stroke_m=0.11,
        max_linear_m=0.12,
        preview=True,
    )
    assert bad_table.preview is True and bad_table.accepted is False
    assert bad_table.status == "rejected" and bad_table.stop_reason == "ik_path_rejected"
    assert bad_table.ik_error and "table" in bad_table.ik_error
    assert not env.actions

    # Refusal in preview: alignment delta exceeds max_linear_m
    too_far = robo.align_feature(
        feature.tolist(),
        (start + np.array([0.08, 0.0, -0.01])).tolist(),
        arm="right",
        preview=True,
    )
    assert too_far.accepted is False and too_far.stop_reason == "delta_exceeds_limit"
    assert not env.actions

    # Refusal in preview: planned steps exceed max_steps
    tight = robo.align_feature(
        feature.tolist(),
        (start + np.array([0.025, 0.0, -0.01])).tolist(),
        arm="right",
        stroke_m=0.025,
        max_steps=2,
        preview=True,
    )
    assert tight.accepted is False and tight.stop_reason == "step_budget"
    assert not env.actions

    # Zero-step no-op preview succeeds even when remaining < reserve_steps (consistent with live)
    env.step_limit = 5
    noop_preview = robo.align_feature(
        [0.25, -0.15, 0.82],
        [0.2502, -0.1502, 0.80],
        arm="right",
        reserve_steps=20,
        preview=True,
    )
    assert noop_preview.preview is True and noop_preview.accepted is True
    assert noop_preview.status == "planned" and noop_preview.aligned is True
    assert noop_preview.planned_steps == 0 and noop_preview.steps_used == 0
    assert not env.actions


def test_align_feature_provenance_rejects_stale_feature_but_allows_static_target(tmp_path):
    env, robo, _ = station(tmp_path)
    # Record a static receptor measurement at step 0 (unoccluded observation).
    receptor_world = (0.242, -0.145, 0.810)
    px_h = robo.project_point(receptor_world, "cam_head")
    px_r = robo.project_point(receptor_world, "cam_right_wrist")
    assert px_h is not None and px_r is not None
    static_target = robo.triangulate(
        "cam_head", (px_h.row, px_h.col), "cam_right_wrist", (px_r.row, px_r.col)
    )
    old_feature = robo.ray_at_z(px_h.row, px_h.col, z=0.830, camera="cam_head")

    # Move arm so step 0 measurements become stale.
    robo.nudge((0.0, 0.0, 0.005))
    steps_before = len(env.actions)

    # Stale held feature is rejected in 0 steps by default (require_current_feature=True).
    with pytest.raises(ToolError, match="stale"):
        robo.align_feature(old_feature, static_target, arm="right")
    assert len(env.actions) == steps_before

    # Fresh held feature paired with earlier static target succeeds by default!
    held_world = (0.250, -0.150, 0.830)
    fh = robo.project_point(held_world, "cam_head")
    fr = robo.project_point(held_world, "cam_right_wrist")
    assert fh is not None and fr is not None
    fresh_feature = robo.triangulate(
        "cam_head", (fh.row, fh.col), "cam_right_wrist", (fr.row, fr.col)
    )
    ok = robo.align_feature(fresh_feature, static_target, arm="right")
    assert ok.status == "completed" and ok.aligned and ok.reached
    assert ok.transverse_delta == pytest.approx((-0.008, 0.005, 0.0), abs=5e-4)

    # require_current_target=True rejects stale static target in 0 steps.
    steps_after = len(env.actions)
    fresh_feature2 = robo.ray_at_z(fh.row, fh.col, z=0.830, camera="cam_head")
    with pytest.raises(ToolError, match="stale"):
        robo.align_feature(
            fresh_feature2, static_target, arm="right", require_current_target=True
        )
    assert len(env.actions) == steps_after

    # require_current_feature=False allows explicit override with mapping input too.
    mapping_target = {"xyz": list(receptor_world)}
    override = robo.align_feature(
        old_feature, mapping_target, arm="right", require_current_feature=False, preview=True
    )
    assert override.preview is True and override.accepted is True

    # Invalid arguments raise ValueError without stepping.
    with pytest.raises(ValueError, match="axis"):
        robo.align_feature([0, 0, 0.85], [0, 0, 0.83], axis=(0, 0, 0))
    with pytest.raises(ValueError, match="at most one"):
        robo.align_feature([0, 0, 0.85], [0, 0, 0.83], stroke_m=0.01, stroke_dxyz=(0, 0, -0.01))
    with pytest.raises(ValueError, match="standoff_m"):
        robo.align_feature([0, 0, 0.85], [0, 0, 0.83], standoff_m=-0.01)
    with pytest.raises(ValueError, match="xyz"):
        robo.align_feature({"bad": 1}, [0, 0, 0.83])


# -- table-touch kinematics, nudge & delta_grasp ---------------------------------


class FlexibleGripEnvironment(JointEnvironment):
    """Joint environment with configurable tabletop stop, close width, and optional slip."""

    def __init__(
        self,
        *,
        initial_opening: float = 1.0,
        close_stop: float = 0.025,
        floor: float | None = None,
        x_wall: float | None = None,
        slip_after_lift_z: float | None = None,
        slip_to: float = 0.004,
    ):
        super().__init__()
        self.grips = {a: float(initial_opening) for a in kin.ARMS}
        self.close_stop = float(close_stop)
        self.floor = floor
        self.x_wall = x_wall
        self.slip_after_lift_z = slip_after_lift_z
        self.slip_to = float(slip_to)

    def step(self, action):
        before = copy.deepcopy(self.joints)
        self.actions.append(copy.deepcopy(action))
        self.captures += 1
        for a in kin.ARMS:
            if a != self.frozen:
                self.joints[a] = list(action[f"{a}_arm_joint_state"])
            goal = float(action[f"{a}_ee_joint_state"][0])
            target_open = self.close_stop if goal <= 0.05 else goal
            self.grips[a] = self.grips[a] + float(np.clip(target_open - self.grips[a], -0.35, 0.35))
        tcp_r = kin.tcp(kin.fk("right", self.joints["right"]))
        if self.floor is not None and tcp_r[2] < self.floor:
            self.joints["right"] = before["right"]
        if self.x_wall is not None and tcp_r[0] < self.x_wall:
            self.joints["right"] = before["right"]
        if self.slip_after_lift_z is not None:
            tcp_now = kin.tcp(kin.fk("right", self.joints["right"]))
            if tcp_now[2] >= self.slip_after_lift_z and action["right_ee_joint_state"][0] <= 0.05:
                self.grips["right"] = self.slip_to
        return self.get_obs(), 0.0, False, {"status": self.get_status()}


def test_kinematics_allow_table_touch_relaxes_only_fingertips_not_links():
    # Vertical pitch=90 at tcp_z = 0.776 => tip_z = 0.763 (2 mm below TABLE_Z=0.765).
    seed = [0.0] * 6
    pose_low = np.eye(4)
    pose_low[:3, :3] = kin.orientation(90, 0, 0)
    pose_low[:3, 3] = [0.25, -0.15, 0.776]
    # Default allow_table_touch=False rejects fingertips below TABLE_Z + 0.002 = 0.767.
    sol_strict = kin.best_solution("right", pose_low, seed)
    assert sol_strict.collides is True
    assert kin.clearance_violations("right", sol_strict.q) == ["fingertips"]

    # allow_table_touch=True permits fingertips down to TABLE_Z - 0.004 = 0.761.
    sol_touch = kin.best_solution("right", pose_low, seed, allow_table_touch=True)
    assert sol_touch.ok and not sol_touch.collides
    assert kin.clearance_violations("right", sol_touch.q, allow_table_touch=True) == []

    # Overdrive past 4 mm (tcp_z = 0.772 => tip_z = 0.759 < 0.761) is STILL rejected.
    pose_too_deep = pose_low.copy()
    pose_too_deep[2, 3] = 0.772
    sol_deep = kin.best_solution("right", pose_too_deep, sol_touch.q, allow_table_touch=True)
    assert sol_deep.collides is True
    assert "fingertips" in kin.clearance_violations("right", sol_deep.q, allow_table_touch=True)

    # line_path respects allow_table_touch=False vs True.
    pose_hover = pose_low.copy()
    pose_hover[2, 3] = 0.805
    q_hover = kin.best_solution("right", pose_hover, seed).q
    _, why_strict = kin.line_path("right", q_hover, pose_low, step_m=0.003)
    assert why_strict is not None and "table" in why_strict
    ways_touch, why_touch = kin.line_path(
        "right", q_hover, pose_low, step_m=0.003, allow_table_touch=True
    )
    assert why_touch is None and len(ways_touch) >= 9

    # Links (joint2..joint6) are NEVER relaxed even with allow_table_touch=True!
    # Horizontal pitch=0 at z = 0.772 puts joint6 at z = 0.772 < TABLE_Z + 0.015 = 0.780.
    pose_horiz_low = np.eye(4)
    pose_horiz_low[:3, :3] = kin.orientation(0, 0, 0)
    pose_horiz_low[:3, 3] = [0.25, -0.15, 0.772]
    sol_horiz = kin.solve("right", pose_horiz_low, seed, allow_table_touch=True)
    assert sol_horiz.collides is True
    assert "joint6" in kin.clearance_violations("right", sol_horiz.q, allow_table_touch=True)


def test_nudge_allow_table_contact_and_thin_held_microsteps(tmp_path):
    # Physical tabletop halts fingertips at tip_z = 0.765 => tcp_z = 0.778.
    env = _place(FlexibleGripEnvironment(initial_opening=1.0, close_stop=0.025, floor=0.778), z=0.798)
    host = NativeToolkit(env, None, tmp_path / "native")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))

    # Without allow_table_contact, descending to tcp_z = 0.775 is rejected in 0 steps.
    rejected = robo.nudge((0.0, 0.0, -0.023), arm="right")
    assert rejected.status == "rejected" and rejected.stop_reason == "ik_path_rejected"
    assert not env.actions

    # allow_table_contact=True with compact=False raises ValueError.
    with pytest.raises(ValueError, match="compact=True"):
        robo.nudge((0.0, 0.0, -0.023), arm="right", compact=False, allow_table_contact=True)

    # With allow_table_contact=True, uses 3 mm steps (ceil(0.023/0.003)=8) and stalls at tabletop!
    seated = robo.nudge((0.0, 0.0, -0.023), arm="right", allow_table_contact=True)
    assert seated.status == "stopped" and seated.stalled is True
    assert seated.stop_reason == "contact_or_stall"
    assert seated.planned_steps >= 8
    assert tcp(env)[2] == pytest.approx(0.778, abs=0.003)

    # Close on a thin part (opening=0.025 <= 0.045), then lift 2.4 cm: automatically uses 3 mm steps!
    robo.set_gripper("right", robo.CLOSE)
    lift_thin = robo.nudge((0.0, 0.0, 0.024), arm="right", allow_table_contact=True)
    assert lift_thin.reached is True
    assert lift_thin.planned_steps >= 8  # 24 mm / 3 mm = 8

    # Contrast with thick held object (opening=0.30 > 0.045): uses normal 5 mm steps.
    env2 = _place(FlexibleGripEnvironment(initial_opening=0.30, close_stop=0.30), z=0.84)
    host2 = NativeToolkit(env2, None, tmp_path / "native2")
    robo2 = RobodojoOfficialColourRobot(NativeBackend(host2))
    robo2.set_gripper("right", robo2.CLOSE)
    lift_thick = robo2.nudge((0.0, 0.0, 0.024), arm="right")
    assert lift_thick.reached is True and lift_thick.planned_steps <= 5  # 24 mm / 5 mm = 5
    assert lift_thin.planned_steps > lift_thick.planned_steps


def test_delta_grasp_tabletop_pickup_seats_closes_and_lifts_gently(tmp_path):
    # Stage right arm 3 cm above a 5 mm flat part (surface z = 0.770) with 6 mm transverse offset.
    # Physical tabletop stops fingertips at tip_z = 0.765 => tcp_z = 0.778.
    env = _place(FlexibleGripEnvironment(initial_opening=1.0, close_stop=0.025, floor=0.778), z=0.805)
    host = NativeToolkit(env, None, tmp_path / "native")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))

    part_surface_pt = [0.256, -0.146, 0.770]  # 5 mm flat part on tabletop
    # Verify default max_steps (=36) completes canonical tabletop pickup + lift without override!
    plan = robo.delta_grasp(
        part_surface_pt,
        arm="right",
        contact_table=True,
        overdrive_m=0.005,
        lift_dxyz=(0.0, 0.0, 0.025),
        preview=True,
    )
    assert plan.accepted is True and plan.planned_steps >= 22  # 3 mm align/approach/lift + close
    res = robo.delta_grasp(
        part_surface_pt,
        arm="right",
        contact_table=True,
        overdrive_m=0.005,
        lift_dxyz=(0.0, 0.0, 0.025),
    )
    assert isinstance(res, DeltaGrasp)
    assert res.status == "completed" and res.reached is True and res.aligned is True
    assert res.stalled is False and res.table_contact_stalled is True
    assert res.transverse_error_after_m <= 0.0025
    assert res.closed is True and res.closed_opening == pytest.approx(0.025, abs=0.002)
    assert res.holding is True and res.width_m is not None and res.width_m > 0.004
    assert res.lift_status == "completed" and res.lift_steps_used >= 8  # 25 mm at 3 mm gentle steps!
    assert res.opening_after_lift == pytest.approx(0.025, abs=0.002)
    assert res.retained_closure is True
    assert res.final_xyz[0] == pytest.approx(0.256, abs=0.003)
    assert res.final_xyz[1] == pytest.approx(-0.146, abs=0.003)
    assert res.final_xyz[2] == pytest.approx(0.778 + 0.025, abs=0.004)
    assert res.steps_used == len(env.actions)
    assert robo._ledger.records[-1].receipt["status"] == "completed"

    # Also verify contact_table=True on a thicker part (opening=0.25 > 0.045) reports lift_step_m=0.003
    # matching the 3 mm lift executed by nudge(..., allow_table_contact=True).
    env_thick = _place(FlexibleGripEnvironment(initial_opening=1.0, close_stop=0.25, floor=0.778), z=0.805)
    host_thick = NativeToolkit(env_thick, None, tmp_path / "thick_table")
    raw = rvd.delta_grasp(
        host_thick._primitives,
        {a: list(env_thick.joints[a]) for a in kin.ARMS},
        part_surface_pt,
        arm="right",
        contact_table=True,
        overdrive_m=0.005,
        lift_dxyz=(0.0, 0.0, 0.025),
    )
    assert raw["status"] == "completed" and raw["reached"] is True
    assert raw["lift_step_m"] == pytest.approx(rvd.FINE_STEP_M)
    assert raw["lift_steps_used"] >= 8


def test_delta_grasp_thin_sub3mm_part_still_lifts_and_slip_sets_retained_false(tmp_path):
    # Ultra-thin part (< 3.5 mm) settles at closed_opening=0.004 (<= HOLD_MIN_OPENING=0.01):
    # delta_grasp MUST still execute lift_dxyz in 3 mm steps and report retained_closure=True.
    env_thin = _place(
        FlexibleGripEnvironment(initial_opening=1.0, close_stop=0.004, floor=0.778), z=0.800
    )
    host_thin = NativeToolkit(env_thin, None, tmp_path / "thin")
    robo_thin = RobodojoOfficialColourRobot(NativeBackend(host_thin))
    res_thin = robo_thin.delta_grasp(
        [0.25, -0.15, 0.778],
        arm="right",
        contact_table=True,
        overdrive_m=0.004,
        lift_dxyz=(0.0, 0.0, 0.021),
    )
    assert res_thin.status == "completed" and res_thin.reached is True
    assert res_thin.closed_opening == pytest.approx(0.004, abs=0.001)
    assert res_thin.lift_status == "completed" and res_thin.lift_steps_used >= 7  # 3 mm steps
    assert res_thin.opening_after_lift == pytest.approx(0.004, abs=0.001)
    assert res_thin.retained_closure is True

    # Slip during liftoff: wider initial grip (0.040 > 0.025) collapses to 0.004 on lift -> False.
    env_slip = _place(
        FlexibleGripEnvironment(
            initial_opening=1.0,
            close_stop=0.040,
            floor=0.778,
            slip_after_lift_z=0.790,
            slip_to=0.004,
        ),
        z=0.800,
    )
    host_slip = NativeToolkit(env_slip, None, tmp_path / "slip")
    robo_slip = RobodojoOfficialColourRobot(NativeBackend(host_slip))
    res_slip = robo_slip.delta_grasp(
        [0.25, -0.15, 0.778],
        arm="right",
        contact_table=True,
        overdrive_m=0.004,
        lift_dxyz=(0.0, 0.0, 0.021),
    )
    assert res_slip.status == "completed" and res_slip.closed_opening == pytest.approx(0.040, abs=0.002)
    assert res_slip.opening_after_lift == pytest.approx(0.004, abs=0.002)
    assert res_slip.retained_closure is False


def test_delta_grasp_horizontal_handoff_narrows_preopen_and_respects_standoff(tmp_path):
    # Horizontal receiver along its radial approach heading from base (0.30, -0.45):
    # pitch=0, yaw=45 (pointing toward (-0.707, +0.707, 0)), roll=90, opening=1.0.
    env = FlexibleGripEnvironment(initial_opening=1.0, close_stop=0.12)
    pose_r = np.eye(4)
    pose_r[:3, :3] = kin.orientation(0, 45, 90)
    pose_r[:3, 3] = [0.017, -0.167, 0.86]
    sol_r = kin.best_solution("right", pose_r, env.joints["right"])
    assert sol_r.ok and not sol_r.collides
    env.joints["right"] = list(sol_r.q)
    host = NativeToolkit(env, None, tmp_path / "handoff")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))

    u_pointing = pose_r[:3, 0]
    u_perp = np.array([-u_pointing[1], u_pointing[0], 0.0])
    head_pt = pose_r[:3, 3] + u_pointing * 0.045 + u_perp * 0.006 + np.array([0.0, 0.0, -0.005])
    res = robo.delta_grasp(
        head_pt.tolist(),
        arm="right",
        axis=u_pointing.tolist(),
        standoff_m=0.008,
        opening=0.38,
        max_steps=24,
    )
    assert res.status == "completed" and res.reached is True and res.aligned is True
    # Narrowed from 1.0 -> 0.38 before advancing!
    assert res.preopen_steps_used > 0
    assert res.opening_after_preopen == pytest.approx(0.38, abs=0.02)
    assert res.align_steps_used > 0 and res.approach_steps_used > 0 and res.close_steps_used > 0
    expected_tcp = head_pt - u_pointing * 0.008
    assert np.allclose(tcp(env), expected_tcp, atol=0.003)
    assert res.closed is True and res.holding is True and res.closed_opening == pytest.approx(0.12, abs=0.01)
    assert res.steps_used == len(env.actions)

    # Second call when already at opening=0.38 skips pre-open in 0 steps!
    env2 = FlexibleGripEnvironment(initial_opening=0.38, close_stop=0.12)
    env2.joints["right"] = list(sol_r.q)
    host2 = NativeToolkit(env2, None, tmp_path / "handoff2")
    robo2 = RobodojoOfficialColourRobot(NativeBackend(host2))
    res2 = robo2.delta_grasp(
        head_pt.tolist(),
        arm="right",
        axis=u_pointing.tolist(),
        standoff_m=0.008,
        opening=0.38,
        max_steps=24,
    )
    assert res2.status == "completed" and res2.preopen_steps_used == 0
    assert res2.steps_used < res.steps_used


def test_delta_grasp_aborts_before_closing_on_unexpected_stall(tmp_path):
    # Unexpected wall blocks transverse alignment: must abort before approaching or closing.
    start_x = START["right"][0]
    env = _place(FlexibleGripEnvironment(initial_opening=1.0, x_wall=start_x - 0.003), z=0.86)
    host = NativeToolkit(env, None, tmp_path / "wall")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))
    res = robo.delta_grasp(
        [start_x - 0.020, -0.15, 0.84], arm="right", contact_table=False, max_steps=20
    )
    assert res.aligned is False and res.closed is False
    assert res.status == "stopped" and res.stalled is True
    assert res.stop_reason == "contact_or_stall"
    assert res.approach_steps_used == 0 and res.close_steps_used == 0

    # Non-table approach stalls unexpectedly: must abort before closing.
    env2 = _place(FlexibleGripEnvironment(initial_opening=1.0, floor=0.845), z=0.86)
    host2 = NativeToolkit(env2, None, tmp_path / "stall_app")
    robo2 = RobodojoOfficialColourRobot(NativeBackend(host2))
    res2 = robo2.delta_grasp(
        [start_x, -0.15, 0.825], arm="right", contact_table=False, max_steps=20
    )
    assert res2.aligned is True and res2.closed is False
    assert res2.status == "stopped" and res2.stalled is True
    assert res2.table_contact_stalled is False and res2.close_steps_used == 0


def test_delta_grasp_preview_provenance_and_budget(tmp_path):
    from codeaspolicy.execution import ExecutionBudget, ExecutionLimit, ExecutionLimits

    env = _place(FlexibleGripEnvironment(initial_opening=1.0, close_stop=0.03), z=0.82)
    host = NativeToolkit(env, None, tmp_path / "prev")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))

    # Read-only preview costs 0 actions and returns DeltaGrasp with status="planned".
    plan = robo.delta_grasp(
        [0.255, -0.15, 0.79],
        arm="right",
        contact_table=True,
        overdrive_m=0.005,
        lift_dxyz=(0.0, 0.0, 0.02),
        preview=True,
    )
    assert isinstance(plan, DeltaGrasp)
    assert plan.preview is True and plan.accepted is True and plan.status == "planned"
    assert plan.steps_used == 0 and plan.planned_steps > 0 and not env.actions
    assert not robo._ledger.stateful_calls()

    # Over-limit preview rejects in 0 steps.
    too_far = robo.delta_grasp([0.35, -0.15, 0.79], arm="right", preview=True)
    assert too_far.accepted is False and too_far.stop_reason == "delta_exceeds_limit"

    # Tight budget preview rejects in 0 steps.
    tight = robo.delta_grasp([0.255, -0.15, 0.79], arm="right", max_steps=2, preview=True)
    assert tight.accepted is False and tight.stop_reason == "step_budget"

    # Provenance check when require_current_target=True.
    pt = robo.ray_at_z(240, 320, z=0.79, camera="cam_head")
    robo.nudge((0.0, 0.0, 0.005))
    steps_before = len(env.actions)
    with pytest.raises(ToolError, match="stale"):
        robo.delta_grasp(pt, arm="right", require_current_target=True)
    assert len(env.actions) == steps_before

    # ExecutionBudget: preview=True counts 0 actions, stateful delta_grasp counts 1 action.
    with ExecutionBudget(ExecutionLimits(max_actions=1), tmp_path / "ws", robo) as budget:
        p = robo.delta_grasp([0.25, -0.15, 0.80], arm="right", preview=True)
        assert p.accepted is True and budget.actions == 0
        g = robo.delta_grasp([0.25, -0.15, 0.80], arm="right")
        assert g.reached is True and budget.actions == 1
        with pytest.raises(ExecutionLimit):
            robo.delta_grasp([0.25, -0.15, 0.80], arm="right")


# -- registration ----------------------------------------------------------------


def test_execution_budget_counts_each_stateful_nudge(tmp_path):
    from codeaspolicy.execution import ExecutionBudget, ExecutionLimit, ExecutionLimits

    env, robo, _ = station(tmp_path)
    with ExecutionBudget(ExecutionLimits(max_actions=3), tmp_path / "ws", robo) as budget:
        robo.nudge((0, 0, 0.005))
        robo.visual_align_and_insert("align", policy=[answer(0.5)])  # one nudge inside
        assert budget.actions == 2
        # Read-only preview uses 0 actions; stateful align_feature counts as 1 action.
        preview = robo.align_feature([0.25, -0.15, 0.82], [0.255, -0.15, 0.80], preview=True)
        assert preview.preview is True and budget.actions == 2
        aligned = robo.align_feature([0.25, -0.15, 0.82], [0.255, -0.15, 0.80])
        assert aligned.aligned is True and budget.actions == 3
        with pytest.raises(ExecutionLimit):
            robo.nudge((0, 0, 0.005))


def test_registration_and_reference():
    assert {"nudge", "visual_align_and_insert", "align_feature", "delta_grasp"} <= STATEFUL_TOOLS
    assert {"nudge", "visual_align_and_insert", "align_feature", "delta_grasp"} <= STATEFUL
    assert "preview_align_feature" not in STATEFUL_TOOLS
    assert "preview_align_feature" not in STATEFUL
    assert "preview_delta_grasp" not in STATEFUL_TOOLS
    assert "preview_delta_grasp" not in STATEFUL
    assert set(rvd.SPECS) == {
        "nudge",
        "visual_align_and_insert",
        "align_feature",
        "preview_align_feature",
        "delta_grasp",
        "preview_delta_grasp",
    }
    assert rvd.preview_align_feature._readonly is True
    assert rvd.preview_delta_grasp._readonly is True
    for cls in (RobodojoRobot, RobodojoOfficialColourRobot):
        assert Nudge in cls.RESULT_TYPES
        assert DeltaGrasp in cls.RESULT_TYPES
        assert VisualAlign in cls.RESULT_TYPES
        assert FeatureAlignment in cls.RESULT_TYPES
        text = api_reference(cls)
        assert "robo.nudge(" in text and "robo.visual_align_and_insert(" in text
        assert "robo.align_feature(" in text and "robo.delta_grasp(" in text
        assert "class Nudge(" in text and "class VisualAlign(" in text
        assert "class FeatureAlignment(" in text and "class DeltaGrasp(" in text
        for prims in ("no-vla", "no-sam"):
            filtered = api_reference(cls, primitives=prims)
            assert "robo.nudge(" in filtered and "robo.align_feature(" in filtered
            assert "robo.delta_grasp(" in filtered
            assert "class FeatureAlignment(" in filtered and "class DeltaGrasp(" in filtered
        for word in ("insert_key", "keyhole", "make_kong", "mahjong"):
            assert word not in text


def test_rpent_toolkit_registration_adds_both_tools():
    added = {}

    class Toolkit:
        def __init__(self, env):
            self._primitives = type("P", (), {"env": env})()

        def add_tool(self, name, spec, handler):
            added[name] = (spec, handler)

    motion.register(Toolkit(_place(JointEnvironment())))
    assert added["nudge"][0]["name"] == "nudge"
    assert added["visual_align_and_insert"][0]["input_schema"]["required"] == ["goal"]
    assert added["align_feature"][0]["input_schema"]["required"] == ["feature_xyz", "target_xyz"]
    assert added["preview_align_feature"][0]["name"] == "preview_align_feature"
    assert added["preview_align_feature"][1].func._readonly is True
    assert added["delta_grasp"][0]["input_schema"]["required"] == ["target_xyz"]
    assert added["preview_delta_grasp"][0]["name"] == "preview_delta_grasp"
    assert added["preview_delta_grasp"][1].func._readonly is True
