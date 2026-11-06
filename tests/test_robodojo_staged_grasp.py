"""Staged grasp and local alignment through public APIs and native receipts."""

import numpy as np
import pytest

from pyrualean._backend import EpisodeFinished
from pyrualean.hosts import robodojo_tools as tools
from pyrualean.interaction import InteractionPause
from tests.test_robodojo_intent import _offset_measured_pose, _precision_station


def setup(tmp_path, **kwargs):
    env, robo, host = _precision_station(tmp_path)
    goal = robo.observe().right.shifted(dz=.015)
    result = robo.begin_grasp("pick", goal, opening=.7, approach=[0, 0, .025],
                              probe=[0, 0, .015], **kwargs)
    assert result["status"] == "ready", result
    return env, robo, host, goal


def through(robo, stages=("open", "approach", "align", "close", "probe")):
    for stage in stages:
        result = robo.grasp_step("pick", stage)
        assert result["reached"], result
    return result


def test_grasp_registers_without_actions_and_requires_explicit_stage_order(tmp_path):
    env, robo, _, goal = setup(tmp_path)
    assert not env.actions
    assert robo.grasp_step("pick", "close")["stop_reason"] == "grasp_stage_order"
    assert not env.actions
    conflict = robo.begin_grasp("pick", goal.shifted(dz=.01))
    assert conflict["stop_reason"] == "grasp_identity_conflict" and not env.actions


def test_stages_keep_support_and_repeated_completed_stages_never_move(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    through(robo, ("open", "approach", "align"))
    before = len(env.actions)
    assert robo.grasp_step("pick", "open")["steps_used"] == 0
    assert len(env.actions) == before
    result = through(robo, ("close", "probe"))
    assert result["grasp_complete"] and result["retained"]
    assert not result["grasp_verified"] and result["needs_visual_confirmation"]
    assert result["grasp_steps_used"] == len(env.actions)
    assert all(a["right_ee_joint_state"] == [0.] for a in env.actions[before:])
    assert all(a["left_ee_joint_state"] == [.3] for a in env.actions)
    count = len(env.actions)
    for stage in ("open", "approach", "align", "close", "probe"):
        assert robo.grasp_step("pick", stage)["steps_used"] == 0
    assert len(env.actions) == count
    assert robo.grasp_status("pick")["grasp_complete"]


def test_refinement_uses_original_target_and_a_new_identity(tmp_path):
    env, robo, _, goal = setup(tmp_path)
    through(robo, ("open", "approach", "align"))
    first = robo.grasp_step("pick", "align", adjustment_id="observed_a", offset=[0, 0, .002])
    assert first["reached"], first
    second = robo.grasp_step("pick", "align", adjustment_id="observed_b", offset=[0, 0, .004])
    assert second["reached"], second
    assert second["alignment_target"]["xyz"][2] == pytest.approx(goal.xyz[2] + .004)
    assert second["actual_pose"]["xyz"][2] == pytest.approx(goal.xyz[2] + .004, abs=.0015)
    count = len(env.actions)
    assert robo.grasp_step("pick", "align", adjustment_id="observed_b",
                           offset=[0, 0, .004])["steps_used"] == 0
    changed = robo.grasp_step("pick", "align", adjustment_id="observed_b", offset=[0, 0, .005])
    assert changed["stop_reason"] == "grasp_stage_identity_conflict"
    assert len(env.actions) == count
    through(robo, ("close",))
    count = len(env.actions)
    rejected = robo.grasp_step("pick", "align", adjustment_id="too_late", offset=[0, 0, .002])
    assert rejected["stop_reason"] == "cannot_refine_closed_grasp"
    assert len(env.actions) == count


def test_refinement_cannot_escape_declared_envelope(tmp_path):
    env, robo, _, _ = setup(tmp_path, translation_limits=[.003, .003, .003])
    through(robo, ("open", "approach", "align"))
    before = len(env.actions)
    result = robo.grasp_step("pick", "align", adjustment_id="outside", offset=[.004, 0, 0])
    assert result["stop_reason"] == "alignment_outside_original_envelope"
    assert len(env.actions) == before


def test_failed_alignment_does_not_close_and_can_receive_an_observed_refinement(tmp_path):
    env, robo, _, goal = setup(tmp_path)
    through(robo, ("open", "approach"))
    env.frozen = "right"
    failed = robo.grasp_step("pick", "align")
    assert not failed["reached"] and "align" not in failed["completed_stages"]
    count = len(env.actions)
    assert robo.grasp_step("pick", "close")["stop_reason"] == "grasp_stage_order"
    assert len(env.actions) == count
    assert all(a["right_ee_joint_state"] == [.7] for a in env.actions[3:])
    # A fresh observed plan may still use the nominal target. The stopped
    # operation is not replayed; this is a separately identified adjustment.
    env.frozen = None
    # Current pose is outside the local envelope after this early stall.
    out = robo.grasp_step("pick", "align", adjustment_id="reobserve", offset=[0, 0, .001])
    assert not out["reached"]
    assert "local_translation_constraint" in out["stage_result"]["stop_reason"]
    assert goal.xyz[2] < robo.observe().right.xyz[2]


def test_gripper_interruption_continues_remaining_stroke_without_reopening(tmp_path):
    env, robo, host, _ = setup(tmp_path)
    through(robo, ("open", "approach", "align"))
    original = env.step
    count = len(env.actions)

    def cancel(action):
        value = original(action)
        if len(env.actions) == count + 1:
            host._primitives.cancelled.set()
        return value

    env.step = cancel
    with pytest.raises(EpisodeFinished, match="cancelled"):
        robo.grasp_step("pick", "close")
    assert robo.grasp_status("pick")["status"] == "paused"
    host._primitives.cancelled.clear()
    result = robo.grasp_step("pick", "close")
    assert result["reached"] and result["grasp_steps_used"] == len(env.actions)
    assert all(a["right_ee_joint_state"] == [0.] for a in env.actions[count:])
    assert result["steps_used"] == 2  # remaining close/settle ticks, not a fresh stroke
    assert len(env.actions) == count + 3


def test_empty_close_is_terminal_for_that_stage_and_never_probes(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    through(robo, ("open", "approach", "align"))
    original = env.step

    def empty(action):
        value = original(action)
        if action["right_ee_joint_state"] == [0.]:
            env.grips["right"] = 0.
        return value

    env.step = empty
    failed = robo.grasp_step("pick", "close")
    assert failed["stop_reason"] == "empty_grasp"
    count = len(env.actions)
    assert robo.grasp_step("pick", "close")["steps_used"] == 0
    assert robo.grasp_step("pick", "probe")["stop_reason"] == "grasp_stage_order"
    assert len(env.actions) == count


def test_probe_stops_on_lost_closure_without_opening_or_regrasping(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    through(robo, ("open", "approach", "align", "close"))
    original = env.step

    def slip(action):
        value = original(action)
        env.grips["right"] = 0.
        return value

    env.step = slip
    count = len(env.actions)
    failed = robo.grasp_step("pick", "probe")
    assert not failed["retained"] and failed["stop_reason"] == "right: holding_requirement"
    assert len(env.actions) == count + 1
    assert env.actions[-1]["right_ee_joint_state"] == [0.]


def test_unknown_physical_effect_blocks_stage_continuation(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    env.raise_after_step = True
    with pytest.raises((InteractionPause, RuntimeError)):
        robo.grasp_step("pick", "open")
    assert robo.grasp_status("pick")["status"] == "needs_model"
    assert len(env.actions) == 1


def test_stage_budget_can_extend_cumulatively_without_replaying_open(tmp_path):
    env, robo, _, _ = setup(tmp_path, max_steps=1)
    first = robo.grasp_step("pick", "open")
    assert first["status"] == "paused" and len(env.actions) == 1
    done = robo.grasp_step("pick", "open", max_steps=5)
    assert done["reached"] and done["grasp_steps_used"] == len(env.actions) == 3
    assert robo.grasp_step("pick", "open", max_steps=5)["steps_used"] == 0


def test_motion_stage_resumes_only_remaining_path_after_known_interrupt(tmp_path):
    env, robo, host, _ = setup(tmp_path)
    through(robo, ("open",))
    original, count = env.step, len(env.actions)

    def cancel(action):
        value = original(action)
        if len(env.actions) == count + 1:
            host._primitives.cancelled.set()
        return value

    env.step = cancel
    paused = robo.grasp_step("pick", "approach")
    assert paused["status"] == "paused" and not paused["reached"]
    prefix = [a["right_arm_joint_state"] for a in env.actions[count:]]
    host._primitives.cancelled.clear()
    result = robo.grasp_step("pick", "approach")
    assert result["reached"] and result["grasp_steps_used"] == len(env.actions)
    assert env.actions[count + 1]["right_arm_joint_state"] not in prefix
    assert all(a["right_ee_joint_state"] == [.7] for a in env.actions[count:])


def test_default_motion_budget_is_not_the_old_sixty_action_cap(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    target = robo.observe().right.shifted(dz=.06)
    preview = robo.execute_motion("full", right=target, interaction="contact",
                                  completion={"position_m": .001}, preview=True)
    assert preview["accepted"] and preview["step_limit"] == env.step_limit
    assert not env.actions


def test_too_small_allocation_can_extend_the_same_unexecuted_identity(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    goal = robo.observe().right.shifted(dz=.03)
    preview = robo.execute_motion("small", right=goal, interaction="contact", max_steps=1,
                                  completion={"position_m": .0015}, preview=True)
    assert not preview["accepted"] and preview["preview"] and not env.actions
    first = robo.execute_motion("small", right=goal, interaction="contact", max_steps=1,
                                completion={"position_m": .0015})
    assert first["status"] == "paused" and not first["reached"] and not env.actions
    assert first["stop_reason"] == "allocation_too_small_for_progress"
    result = robo.resume_motion("small", max_steps=100)
    assert result["reached"] and result["intent_steps_used"] == len(env.actions) > 0
    count = len(env.actions)
    assert robo.resume_motion("small", max_steps=100)["steps_used"] == 0
    assert len(env.actions) == count


def test_external_movement_requires_replanning_an_unfinished_grasp(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    through(robo, ("open",))
    assert robo.execute_motion("elsewhere", right=robo.observe().right.shifted(dz=.02),
                               holding={"right": "empty"})["reached"]
    count = len(env.actions)
    result = robo.grasp_step("pick", "approach")
    assert result["stop_reason"] == "grasp_execution_uncertain_or_external_motion"
    assert len(env.actions) == count


@pytest.mark.parametrize("frame", ["world", "tcp"])
def test_local_alignment_freezes_reference_and_repeated_id_never_reanchors(tmp_path, frame):
    env, robo, _ = _precision_station(tmp_path)
    before = tools._pose_of("right", env.joints["right"])
    offset = np.array([0., 0., .004]) if frame == "world" else np.array([.004, 0., 0.])
    result = robo.align_local("local", offset, frame=frame, holding="empty")
    assert result["reached"], result
    expected = before[:3, 3] + (offset if frame == "world" else before[:3, :3] @ offset)
    assert robo.observe().right.xyz == pytest.approx(expected, abs=.0015)
    count = len(env.actions)
    assert robo.align_local("local", offset, frame=frame, holding="empty")["steps_used"] == 0
    assert len(env.actions) == count


def test_local_preview_and_goal_bounds_do_not_dispatch(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    preview = robo.align_local("local", [0, 0, .004], preview=True)
    assert preview["accepted"] and not env.actions
    for kwargs in (dict(offset=[0, 0, .011]), dict(offset=[0, 0, .001], rotation_deg=[0, 0, 6])):
        result = robo.align_local("bad", **kwargs)
        assert result["status"] == "rejected" and not env.actions


def test_local_envelope_is_checked_on_actual_feedback(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    original = env.step

    def escape(action):
        value = original(action)
        _offset_measured_pose(env, dz=.012)
        return value

    env.step = escape
    result = robo.align_local("local", [0, 0, .004])
    assert result["stop_reason"] == "right: local_translation_constraint"
    assert len(env.actions) == 1 and not result["reached"]


def test_handoff_stage_reacquires_state_after_donor_release_and_withdraw(tmp_path):
    env, robo, host, _ = setup(tmp_path)
    tools._remember(host._primitives, "left", 0.0)
    env.grips["left"] = .25
    host._primitives._operations.grasps.records["pick"]["expected"] = (
        host._primitives._operations.intents.motion._snapshot()
    )
    through(robo, ("open", "approach", "align", "close"))
    # Donor has not released yet: handoff reacquisition must refuse.
    clamped = robo.grasp_step("pick", "handoff")
    assert clamped["stop_reason"] == "handoff_donor_not_released"
    # Donor opens and retreats while receiver holds stationary.
    robo.set_gripper("left", robo.OPEN)
    retreat = robo.execute_motion(
        "donor_retreat", left=robo.observe().left.shifted(dz=.03),
        holding={"left": "empty", "right": "retain"},
    )
    assert retreat["reached"], retreat
    status = robo.grasp_status("pick")
    assert status["handoff_reacquisition_required"] and status["next_stage"] == "handoff"
    before = len(env.actions)
    handoff = robo.grasp_step("pick", "handoff")
    assert handoff["reached"] and handoff["steps_used"] == 0
    assert handoff["handoff_reacquired"] and handoff["next_stage"] == "probe"
    assert "handoff" in handoff["completed_stages"]
    assert len(env.actions) == before
    probed = robo.grasp_step("pick", "probe")
    assert probed["reached"] and probed["grasp_complete"] and probed["retained"]


def test_handoff_rejects_when_receiver_moved_after_close(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    through(robo, ("open", "approach", "align", "close"))
    moved = robo.execute_motion(
        "receiver_shift", right=robo.observe().right.shifted(dz=.02),
        holding={"right": "retain"},
    )
    assert moved["reached"]
    count = len(env.actions)
    assert robo.grasp_step("pick", "handoff")["stop_reason"] == "handoff_receiver_pose_changed"
    assert (
        robo.grasp_step("pick", "probe")["stop_reason"]
        == "grasp_execution_uncertain_or_external_motion"
    )
    assert len(env.actions) == count


def test_rejected_begin_grasp_surfaces_prepare_error_on_follow_up_step(tmp_path):
    env, robo, _, goal = setup(tmp_path)
    rejected = robo.begin_grasp("zero_probe", goal, probe=[0., 0., 0.])
    assert rejected["status"] == "rejected"
    assert "probe must be 0.5..6 cm" in rejected["stop_reason"]
    step = robo.grasp_step("zero_probe", "open")
    assert step["stop_reason"] == "unknown_grasp_in_this_session"
    assert "probe must be 0.5..6 cm" in step["prepare_error"]
    assert not env.actions
