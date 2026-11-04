"""Shared operation contracts through real robot dispatch/receipt boundaries."""

import pytest

from pyrualean._backend import EpisodeFinished
from pyrualean.interaction import InteractionPause
from tests.test_robodojo_continuous_grasp import visual_station
from tests.test_robodojo_intent import _precision_station
from tests.test_robodojo_staged_grasp import setup, through


def ref(result):
    return result["operation"]["ref"]


def test_grasp_phase_completion_is_distinct_from_operation_and_object_verification(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    operation = ref(robo.grasp_status("pick"))
    assert robo.resume_operation(operation)["operation"]["state"] == "ready"
    assert not env.actions
    opened = robo.grasp_step("pick", "open")
    common = opened["operation"]
    assert common["phase_complete"] and not common["execution_complete"]
    assert common["progress"]["next_phase"] == "approach"
    assert common["receipt"]["effect_known"]
    count = len(env.actions)
    assert robo.resume_operation(operation)["operation"]["continuation"]["action"] == "advance"
    assert len(env.actions) == count
    complete = through(robo, ("approach", "align", "close", "probe"))
    assert complete["operation"]["execution_complete"]
    assert complete["operation"]["verification"]["status"] == "required"
    count = len(env.actions)
    assert robo.resume_operation(operation)["operation"]["continuation"]["action"] == "observe"
    assert len(env.actions) == count


def test_generic_resume_continues_a_partial_close_without_advancing_probe(tmp_path):
    env, robo, host, _ = setup(tmp_path)
    through(robo, ("open", "approach", "align"))
    operation = ref(robo.grasp_status("pick"))
    original, count = env.step, len(env.actions)

    def cancel(action):
        result = original(action)
        if len(env.actions) == count + 1:
            host._primitives.cancelled.set()
        return result

    env.step = cancel
    with pytest.raises(EpisodeFinished):
        robo.grasp_step("pick", "close")
    status = robo.operation_status(operation)["operation"]
    assert status["state"] == "paused" and status["effect"] == "known"
    assert status["phase"]["name"] == "close"
    host._primitives.cancelled.clear()
    result = robo.resume_operation(operation)
    assert result["steps_used"] == 2
    assert result["operation"]["progress"]["next_phase"] == "probe"
    assert len(env.actions) == count + 3
    assert all(a["right_ee_joint_state"] == [0.] for a in env.actions[count:])
    robo.resume_operation(operation)
    assert len(env.actions) == count + 3


def test_grasp_allowance_extension_and_cached_phase_report_live_total(tmp_path):
    env, robo, _, _ = setup(tmp_path, max_steps=1)
    partial = robo.grasp_step("pick", "open")
    complete = robo.resume_operation(ref(partial), max_steps=200)
    assert complete["operation"]["budget"]["used"] == len(env.actions) == 3
    assert complete["operation"]["budget"]["limit"] == 200
    through(robo, ("approach", "align", "close", "probe"))
    receipt = robo.operation_status(ref(partial))["operation"]["receipt"]
    old = robo.grasp_step("pick", "open")
    assert old["steps_used"] == 0
    assert old["operation"]["state"] == "completed"
    assert old["operation"]["budget"]["used"] == len(env.actions)
    assert old["operation"]["receipt"] == receipt


def test_motion_and_grasp_share_protocol_but_namespaces_do_not_collide(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    grasp = ref(robo.grasp_status("pick"))
    motion = robo.execute_motion("pick", right=robo.observe().right.shifted(dz=.02),
                                 holding={"right": "empty"})
    assert ref(motion)["id"] == grasp["id"] and ref(motion)["kind"] != grasp["kind"]
    assert motion["operation"]["budget"]["used"] == len(env.actions)
    before = len(env.actions)
    changed = robo.resume_operation(grasp)["operation"]
    assert changed["state"] == "needs_replan" and not changed["continuation"]["allowed"]
    assert len(env.actions) == before


def test_old_session_reference_cannot_execute_same_named_new_operation(tmp_path):
    _, robo, _, _ = setup(tmp_path / "old")
    operation = ref(robo.grasp_status("pick"))
    env, new, _, _ = setup(tmp_path / "new")
    result = new.resume_operation(operation)
    assert result["stop_reason"] == "operation_session_mismatch"
    assert not env.actions


def test_changed_object_binding_rejects_with_common_feedback_without_rebinding(tmp_path):
    env, robo, _, goal = setup(tmp_path)
    conflict = robo.begin_grasp("pick", goal, object_ref=robo.object_ref("part", "a part"))
    assert conflict["operation"]["state"] == "rejected"
    assert conflict["operation"]["reason"] == "grasp_object_identity_conflict"
    assert robo.operation_status(ref(conflict))["operation"]["state"] == "ready"
    assert not env.actions


@pytest.mark.parametrize("kind", ["grasp", "motion"])
def test_unknown_dispatch_stays_unknown_in_common_feedback(tmp_path, kind):
    env, robo, _, _ = setup(tmp_path)
    env.raise_after_step = True
    with pytest.raises((InteractionPause, RuntimeError)):
        if kind == "grasp":
            robo.grasp_step("pick", "open")
        else:
            robo.execute_motion("travel", right=robo.observe().right.shifted(dz=.06),
                                holding={"right": "empty"})
    result = robo.grasp_status("pick") if kind == "grasp" else robo.motion_status("travel")
    common = robo.operation_status(ref(result))["operation"]
    assert common["state"] == "unknown" and common["effect"] == "unknown"
    assert not common["receipt"]["effect_known"]
    assert common["continuation"]["action"] == "reconcile"
    with pytest.raises(InteractionPause):
        robo.resume_operation(ref(result))
    assert len(env.actions) == 1


def test_partial_motion_continuation_and_preview_identity(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    goal = robo.observe().right.shifted(dz=.06)
    preview = robo.execute_motion("travel", right=goal, interaction="contact", preview=True)
    assert preview["operation"]["state"] == "preview"
    assert robo.resume_operation(ref(preview))["status"] == "rejected"
    assert not env.actions
    first = robo.execute_motion("travel", right=goal, interaction="contact", max_steps=20,
                                completion={"position_m": .0025})
    assert first["operation"]["state"] == "paused"
    count = len(env.actions)
    done = robo.resume_operation(ref(first), max_steps=150)
    assert done["operation"]["state"] == "completed"
    assert done["steps_used"] == len(env.actions) - count
    assert done["operation"]["budget"]["used"] == len(env.actions)
    count = len(env.actions)
    robo.resume_motion("travel")
    robo.resume_operation(ref(first))
    assert len(env.actions) == count


def test_generic_close_resume_preserves_visual_baseline_and_uncertainty(tmp_path):
    env, robo, host, _, _, _ = visual_station(tmp_path)
    before = robo.grasp_status("tracked")["grasp_steps_used"]
    # Set the operation allowance to one more tick, forcing a partial closure.
    host_registry = host._primitives._operations
    host_registry.grasps.records["tracked"]["limit"] = before + 1
    partial = robo.grasp_step("tracked", "close")
    assert partial["status"] == "paused"
    closed = robo.resume_operation(ref(partial), max_steps=200)
    assert closed["object_evidence"]["baseline_ready"]
    assert closed["operation"]["verification"]["status"] == "required"
    probed = robo.grasp_step("tracked", "probe")
    assert probed["object_evidence"]["state"] == "image_consistent"
    assert probed["operation"]["verification"]["status"] == "required"
    assert probed["operation"]["budget"]["used"] == len(env.actions)


def test_refinement_resume_keeps_adjustment_identity_and_original_offset(tmp_path):
    env, robo, host, goal = setup(tmp_path)
    through(robo, ("open", "approach", "align"))
    host._primitives._operations.grasps.records["pick"]["limit"] = len(env.actions) + 1
    partial = robo.grasp_step("pick", "align", adjustment_id="observed-offset",
                              offset=[0, 0, .008])
    assert partial["operation"]["state"] == "paused"
    assert partial["operation"]["phase"]["adjustment_id"] == "observed-offset"
    done = robo.resume_operation(ref(partial), max_steps=200)
    assert done["operation"]["phase_complete"]
    assert done["alignment_target"]["xyz"][2] == pytest.approx(goal.xyz[2] + .008)
    assert done["operation"]["progress"]["next_phase"] == "close"
    count = len(env.actions)
    robo.resume_operation(ref(partial))
    assert len(env.actions) == count


def test_empty_close_requires_replanning_instead_of_resuming_or_advancing(tmp_path):
    env, robo, _, _ = setup(tmp_path)
    through(robo, ("open", "approach", "align"))
    original = env.step

    def empty(action):
        result = original(action)
        if action["right_ee_joint_state"] == [0.]:
            env.grips["right"] = 0.
        return result

    env.step = empty
    result = robo.grasp_step("pick", "close")
    assert result["operation"]["state"] == "needs_replan"
    assert result["operation"]["effect"] == "known"
    assert result["operation"]["continuation"]["action"] == "replan"
    count = len(env.actions)
    robo.resume_operation(ref(result))
    assert len(env.actions) == count
