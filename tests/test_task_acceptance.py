"""Public requirements survive prompt rendering without becoming task verdicts."""

import copy
import json

import pytest

from pyrualean.prompt import TaskCard, render_prompt


def card(**kwargs):
    return TaskCard(suite="test", task="example", seed=0,
                    task_language="  Move the block, then close the lid.\nKeep the tray upright.  ",
                    object_names=[], **kwargs)


def requirements():
    return [
        dict(id="move", kind="outcome", condition="The block is inside the box.",
             source="public task sheet §1"),
        dict(id="order", kind="order", condition="Close the lid after placing the block.",
             source="task_language"),
        dict(id="tray", kind="invariant", condition="Keep the tray upright throughout.",
             source="task_language"),
        dict(id="lid", kind="final_state", condition="The lid is closed at the end.",
             source="public task sheet §2"),
    ]


@pytest.mark.parametrize("robot", ["libero", "robodojo_xpl_rgb"])
def test_legacy_card_does_not_invent_requirements_or_assume_complete_coverage(tmp_path, robot):
    original = card(robot=robot)
    legacy = json.loads(original.to_json())
    legacy.pop("acceptance")
    # Even a task identifier whose scorer we have investigated conveys no public rules.
    legacy["task"] = "make_kong"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    loaded = TaskCard.from_json(path)
    assert loaded.task_language == original.task_language
    assert loaded.acceptance["requirements"] == []
    assert loaded.acceptance["coverage"] == "unknown"
    assert loaded.acceptance["coverage_source"] is None
    for text in (loaded.render(), loaded.render(compact=True)):
        assert "no supplemental public acceptance requirements supplied" in text
        assert "no verdict is supplied" in text
        assert "mahjong9" not in text and "0.319" not in text


@pytest.mark.parametrize("robot", ["libero", "robodojo_xpl_rgb"])
def test_all_requirement_kinds_and_sources_survive_round_trip_and_prompts(tmp_path, robot):
    from pyrualean.play import build_prompt

    supplied = dict(requirements=requirements(), coverage="complete",
                    coverage_source="public task sheet revision 2")
    original = card(robot=robot, acceptance=supplied,
                    **(dict(eef_pos=[0., 0., .5], gripper_opening=.08)
                       if robot == "libero" else {}))
    preserved = copy.deepcopy(original.acceptance)
    supplied["requirements"][0]["condition"] = "Changed only in the caller"
    path = tmp_path / "card.json"
    path.write_text(original.to_json())
    loaded = TaskCard.from_json(path)
    assert loaded.acceptance == preserved
    assert loaded.task_language == original.task_language
    texts = [loaded.render(), loaded.render(compact=True), render_prompt(loaded).text()]
    texts += [build_prompt(arm, loaded, budget_s=100, max_programs=None)
              for arm in ("cells", "program")]
    for text in texts:
        for requirement in preserved["requirements"]:
            assert requirement["condition"] in text
            assert requirement["source"] in text
        assert "public task sheet revision 2" in text
        assert "no verdict is supplied" in text  # complete coverage is not satisfied criteria


@pytest.mark.parametrize("acceptance", [
    {"requirements": [{"id": "missing_source", "kind": "outcome", "condition": "done"}]},
    {"requirements": [dict(id="x", kind="outcome", condition="done", source="")]},
    {"requirements": [requirements()[0], requirements()[0]]},
    {"requirements": [dict(id="x", kind="model_claim", condition="done", source="Astra")]},
    {"coverage": "complete", "requirements": requirements()},
    {"coverage": "complete", "coverage_source": "public task sheet", "requirements": []},
    {"environment_success": True},
])
def test_ungrounded_coverage_and_verdicts_are_not_accepted_as_public_requirements(acceptance):
    with pytest.raises(ValueError):
        card(acceptance=acceptance)


def test_supplied_requirements_alone_do_not_certify_coverage():
    example = card(acceptance={"requirements": requirements()[:1]})
    assert example.acceptance["coverage"] == "unknown"
    assert "Unknown coverage alone is no reason to stop useful work" in example.render()
