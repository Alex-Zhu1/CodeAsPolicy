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
    rendered = example.render()
    assert "Unknown coverage alone is no reason to stop useful work" in rendered
    assert "do not invent hidden checks" not in rendered
    assert "never waives an unperformed task phase" in rendered
    assert "supply stacks" in rendered


def test_complete_coverage_forbids_assumption_waivers_without_unknown_coverage_warning():
    supplied = dict(requirements=requirements(), coverage="complete",
                    coverage_source="public task sheet revision 2")
    rendered = card(acceptance=supplied).render()
    assert "never waives a supplied requirement" in rendered
    assert "Unknown coverage alone is no reason to stop useful work" not in rendered
    assert "do not invent hidden checks" not in rendered


def test_load_acceptance_supports_single_file_and_task_catalog_with_provenance(tmp_path):
    from pyrualean.task_acceptance import CATALOG_SCHEMA, acceptance_sha256, load_acceptance

    single_payload = dict(
        schema="pyrualean.task_acceptance.v1",
        requirements=requirements(),
        coverage="complete",
        coverage_source="public benchmark manual",
    )
    single_file = tmp_path / "single.json"
    single_file.write_text(json.dumps(single_payload), encoding="utf-8")
    updated = card(robot="robodojo_xpl_rgb").with_acceptance(single_file)
    assert updated.acceptance["coverage"] == "complete"
    assert updated.provenance["acceptance_sha256"] == acceptance_sha256(updated.acceptance)
    assert updated.provenance["acceptance_artifact"]["path"] == str(single_file.resolve())
    assert updated.provenance["acceptance_artifact"]["matched_key"] is None
    assert "public benchmark manual" in updated.sources["task acceptance"]

    catalog_payload = {
        "schema": CATALOG_SCHEMA,
        "tasks": {
            "arx_x5/make_kong": {
                "requirements": [
                    dict(
                        id="wait_discard",
                        kind="order",
                        condition="Wait until the opponent discards a tile before acting.",
                        source="mahjong rules §1",
                    ),
                    dict(
                        id="expose_triplet",
                        kind="outcome",
                        condition="Expose the three matching hand tiles face up.",
                        source="mahjong rules §2",
                    ),
                    dict(
                        id="supplemental_tile",
                        kind="outcome",
                        condition=(
                            "Draw the supplemental tile from the supply stack and stand it "
                            "upright at the end of the hand row."
                        ),
                        source="mahjong rules §3",
                    ),
                ],
                "coverage": "complete",
                "coverage_source": "public mahjong task sheet",
            }
        },
    }
    catalog_file = tmp_path / "catalog.json"
    catalog_file.write_text(json.dumps(catalog_payload), encoding="utf-8")
    matched = TaskCard(
        suite="arx_x5", task="make_kong", seed=0,
        task_language="Wait for the opponent to discard a tile, then declare a kong.",
        object_names=[], robot="robodojo_xpl_rgb",
    ).with_acceptance(catalog_file)
    assert matched.acceptance["coverage"] == "complete"
    assert len(matched.acceptance["requirements"]) == 3
    assert matched.provenance["acceptance_artifact"]["matched_key"] == "arx_x5/make_kong"
    assert "supplemental tile from the supply stack" in matched.render(compact=True)

    unmatched, meta = load_acceptance(catalog_file, suite="arx_x5", task="other_task")
    assert unmatched["requirements"] == [] and unmatched["coverage"] == "unknown"
    assert meta["matched_key"] is None
