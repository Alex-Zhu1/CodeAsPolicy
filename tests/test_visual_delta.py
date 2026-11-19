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
    assert {"nudge", "visual_align_and_insert", "align_feature"} <= STATEFUL_TOOLS
    assert {"nudge", "visual_align_and_insert", "align_feature"} <= STATEFUL
    assert "preview_align_feature" not in STATEFUL_TOOLS
    assert "preview_align_feature" not in STATEFUL
    assert set(rvd.SPECS) == {
        "nudge",
        "visual_align_and_insert",
        "align_feature",
        "preview_align_feature",
    }
    assert rvd.preview_align_feature._readonly is True
    for cls in (RobodojoRobot, RobodojoOfficialColourRobot):
        assert Nudge in cls.RESULT_TYPES
        assert VisualAlign in cls.RESULT_TYPES
        assert FeatureAlignment in cls.RESULT_TYPES
        text = api_reference(cls)
        assert "robo.nudge(" in text and "robo.visual_align_and_insert(" in text
        assert "robo.align_feature(" in text
        assert "class Nudge(" in text and "class VisualAlign(" in text
        assert "class FeatureAlignment(" in text
        for prims in ("no-vla", "no-sam"):
            filtered = api_reference(cls, primitives=prims)
            assert "robo.nudge(" in filtered and "robo.align_feature(" in filtered
            assert "class FeatureAlignment(" in filtered
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
