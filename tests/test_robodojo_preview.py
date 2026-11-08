"""Hypothetical pose sequences share one contract and never simulate contact."""

import copy

import pytest

from tests.test_robodojo_intent import _precision_station


def preview(robo, first, following, **kwargs):
    return robo.execute_motion("candidate", right=first, preview=True,
                               holding={"left": "empty", "right": "empty"},
                               following=following, **kwargs)


def test_sequence_preview_rolls_hypothetical_starts_without_effects_or_saved_ids(tmp_path):
    env, robo, host = _precision_station(tmp_path)
    start = robo.observe()
    first = start.right.shifted(dz=.04)
    last = first.shifted(dx=.02)
    before_joints = copy.deepcopy(env.joints)
    before_grips = copy.deepcopy(env.grips)
    result = preview(robo, first, [{"right": last}])
    assert result["accepted"] and result["status"] == "planned", result
    assert result["planned_steps"] > 1 and result["steps_used"] == 0
    assert result["preview_scope"] == "measured_start_then_hypothetical_pose_sequence"
    assert result["sequence_joint_planning"] == "rolling_ik_state_across_legs"
    stages = result["sequence"]
    assert len(stages) == 2
    assert stages[0]["start"]["right"]["xyz"] == pytest.approx(start.right.xyz)
    assert stages[0]["hypothetical_start"] is False
    assert stages[1]["start"]["right"]["xyz"] == pytest.approx(first.xyz)
    assert stages[1]["targets"]["right"]["xyz"] == pytest.approx(last.xyz)
    assert stages[1]["hypothetical_start"] is True
    assert result["simulated_effects"] == []
    assert "grasp_changes" in result["unverified"]
    assert result["replan_before_dispatch"] is True
    assert not env.actions and env.joints == before_joints and env.grips == before_grips
    assert not host._primitives._operations.intents.motion.plans
    assert robo.motion_status("candidate")["stop_reason"] == "unknown_intent_in_this_session"


def test_second_leg_is_checked_from_previous_goal_instead_of_restarting_at_measurement(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    start = robo.observe().right
    first, second = start.shifted(dx=.04, dz=.06), start.shifted(dx=-.04, dz=.06)
    specs = {"keep_out": [{"xyz": list(start.shifted(dz=.06).xyz), "radius_m": .008}]}
    # Each independent spoke from the measured start misses the obstacle.
    for goal in (first, second):
        standalone = robo.execute_motion("spoke", right=goal, holding={"right": "empty"},
                                          preview=True, constraints=specs)
        assert standalone["accepted"], standalone
    # The horizontal second leg between those two targets crosses it.
    result = preview(robo, first, [{"right": second}], constraints=specs)
    assert result["status"] == "rejected" and not result["accepted"], result
    assert result["stop_reason"] == "right: keep_out_constraint"
    assert result["constraint_violation"]["delta"] < 0
    assert result["sequence"][1]["start"]["right"]["xyz"] == pytest.approx(first.xyz)
    assert not env.actions


def test_return_to_start_still_plans_every_leg_and_does_not_truncate_for_budget(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    start = robo.observe().right
    first = start.shifted(dz=.06)
    full = preview(robo, first, [{"right": start}])
    assert full["accepted"] and full["planned_steps"] >= 6, full
    assert full["partial_plan"] is False
    too_short = preview(robo, first, [{"right": start}], max_steps=1)
    assert not too_short["accepted"] and too_short["status"] == "rejected", too_short
    assert "budget" in too_short["stop_reason"]
    assert too_short.get("partial_plan") is not True
    assert len(too_short["sequence"]) == 2 and not env.actions


def test_following_arm_is_included_in_constraints_even_when_not_in_first_target(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    measured = robo.observe()
    result = preview(robo, measured.right.shifted(dz=.03),
                     [{"left": measured.left.shifted(dz=.04)}],
                     constraints={"z_bounds": {"left": [0., measured.left.xyz[2] + .02]}})
    assert result["stop_reason"] == "left: z_bounds_constraint", result
    assert result["status"] == "rejected" and not env.actions


@pytest.mark.parametrize("kwargs", [
    {"directions": {"right": [0., 0., 1.]}},
    {"constraints": {"corridor_m": .02}},
])
def test_sequence_rejects_constraints_whose_meaning_requires_per_leg_contracts(tmp_path, kwargs):
    env, robo, _ = _precision_station(tmp_path)
    start = robo.observe().right
    result = preview(robo, start.shifted(dz=.02), [{"right": start}], **kwargs)
    assert result["status"] == "rejected" and "following does not support" in result["stop_reason"]
    assert not env.actions


@pytest.mark.parametrize("following", [
    "not-a-sequence", [{}], [{"holding": {"right": "retain"}}],
    [{"constraints": {}}], [{"right": [0., 0., 0.]}] * 33,
])
def test_invalid_sequence_schema_rejects_before_actions(tmp_path, following):
    env, robo, _ = _precision_station(tmp_path)
    result = preview(robo, robo.observe().right, following)
    assert result["status"] == "rejected" and "following" in result["stop_reason"]
    assert not env.actions


def test_following_never_turns_into_an_executable_route(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    start = robo.observe().right
    result = robo.execute_motion("no-sequence-execution", right=start.shifted(dz=.02),
                                 following=[{"right": start}], preview=False)
    assert result["status"] == "rejected" and "only with preview=True" in result["stop_reason"]
    assert not env.actions
