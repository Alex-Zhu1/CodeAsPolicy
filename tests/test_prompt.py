"""The prompt is generated from the live library and stays within budget."""

from __future__ import annotations

import inspect

import pytest

from pyrualean import LiberoRobot, TaskCard, api_reference, render_prompt
from pyrualean.libero import RESULT_TYPES


def _public_methods():
    return [
        name
        for name, member in vars(LiberoRobot).items()
        if not name.startswith("_")
        and (inspect.isfunction(member) or isinstance(member, property))
        and name not in ("ledger", "backend")
    ]


def test_api_reference_lists_every_public_method_and_result_type():
    text = api_reference()
    for name in _public_methods():
        assert f"robo.{name}" in text, name
    for cls in RESULT_TYPES:
        assert f"class {cls.__name__}(" in text
    assert "robo.OPEN = -1.0" in text and "robo.CLOSE = 1.0" in text
    assert "EpisodeFinished" in text
    assert "robo.ledger" not in text  # host-only helpers stay hidden


def test_signatures_render_defaults_and_keyword_markers():
    text = api_reference()
    assert "robo.move_to(" in text
    assert "gripper: float = -1.0" in text
    assert "step_clip: float = 0.025" in text
    assert "robo.pi0_pick(\n    prompt: str,\n    *,\n    max_chunks: int = 24," in text


@pytest.mark.parametrize("robot", ["robodojo", "robodojo_xpl_rgb"])
@pytest.mark.parametrize("primitives", ["full", "no-vla"])
def test_generated_motion_policy_declares_intents_without_controller_tuning(robot, primitives):
    from pyrualean.play import build_prompt
    from pyrualean.robots import get_adapter

    card = TaskCard(suite="arx_x5", task="test", seed=0, task_language="Move the object",
                    object_names=[], robot=robot)
    text = build_prompt("cells", card, budget_s=900, max_programs=None, primitives=primitives)
    for name in ("execute_motion", "resume_motion", "motion_status", "set_gripper",
                 "operation_status", "resume_operation",
                 "begin_grasp", "grasp_step", "grasp_status", "align_local",
                 "object_ref", "observe_object", "observe_hold"):
        assert f"robo.{name}(" in text
    for name in ("move_to", "move_both", "motion_profile", "motion_window", "try_grasp"):
        assert f"robo.{name}(\n" not in text
    assert 'motion="native"' not in text and 'motion="transit"' not in text
    assert "Low-level movement and speed selection are disabled" in text
    assert "## Perception" in text and "## Judging progress" in text
    assert "reserve steps for required cleanup" not in text
    assert "Do not abandon a task" in text
    assert "unknown is not empty" in text
    assert "table-height geometry" in text
    assert "robo.move_to(\n" in api_reference(get_adapter(robot).robot_cls, intent_only=False)


@pytest.mark.parametrize("robot", ["robodojo", "robodojo_xpl", "robodojo_xpl_rgb"])
def test_offline_generation_defaults_to_shared_operations(robot):
    from pyrualean.robots import get_adapter

    card = TaskCard(suite="arx_x5", task="test", seed=0, task_language="Move the object",
                    object_names=[], robot=robot)
    cls = get_adapter(robot).robot_cls
    for text in (api_reference(cls), render_prompt(card).system):
        for name in ("execute_motion", "begin_grasp", "align_local", "operation_status",
                     "resume_operation"):
            assert f"robo.{name}(" in text
        for name in ("move_to", "move_both", "motion_profile", "motion_window", "try_grasp"):
            assert f"robo.{name}(\n" not in text
    assert "pyrualean.operation.v1" in render_prompt(card).system


def test_render_prompt_contains_contract_reference_knowledge_and_card(tmp_path):
    img = tmp_path / "agentview_high.png"
    img.write_bytes(b"png")
    card = TaskCard(
        suite="libero_object_swap",
        task=2,
        seed=1,
        task_language="Pick the salad dressing and place it in the basket",
        object_names=["basket_1", "salad_dressing_1"],
        eef_pos=[-0.148, 0.0, 0.261],
        eef_quat=[1, 0, 0, 0],
        gripper_opening=0.078,
        images={"agentview": str(img)},
    )
    bundle = render_prompt(card, budget_s=900)
    assert "def run(robo)" in bundle.system
    assert "900 s" in bundle.system
    assert "# API reference" in bundle.system
    assert "# Operating knowledge" in bundle.system
    assert "gripper=+1" in bundle.system
    assert "Pick the salad dressing" in bundle.user
    assert "basket_1, salad_dressing_1" in bundle.user
    assert bundle.images == (str(img),)
    assert bundle.messages()[0]["role"] == "system"
    assert len(bundle.system) < 40_000


def test_default_libero_cells_prompt_keeps_contract_within_budget():
    from pyrualean.play import build_prompt

    card = TaskCard(suite="libero_object_swap", task=2, seed=0,
                    task_language="Move the object", object_names=[])
    text = build_prompt("cells", card, budget_s=900, max_programs=None)
    assert "contract={...}" in text and "public invariant IDs" in text
    assert len(text) < 40_000


def test_task_card_round_trips_through_json(tmp_path):
    card = TaskCard(
        suite="s",
        task=1,
        seed=2,
        task_language="x",
        object_names=["a"],
        eef_pos=[0, 0, 0],
        eef_quat=[0, 0, 0, 1],
        gripper_opening=0.08,
    )
    path = tmp_path / "card.json"
    path.write_text(card.to_json())
    assert TaskCard.from_json(path) == card


def test_transfer_card_forbids_hard_coded_pixels(tmp_path):
    img = tmp_path / "agentview_high.png"
    img.write_bytes(b"png")
    card = TaskCard(
        suite="libero_object_swap",
        task=2,
        seed=0,
        task_language="x",
        object_names=["a"],
        eef_pos=[0, 0, 0],
        eef_quat=[0, 0, 0, 1],
        gripper_opening=0.08,
        images={"agentview": str(img)},
        transfer=True,
    )
    text = card.render()
    assert "OTHER seeds" in text and "Do not hard-code" in text
    assert "step 0 of your run" not in text
    assert "step 0 of your run" in TaskCard.from_json(_dump(tmp_path, card)).render()


def _dump(tmp_path, card):
    # from_json ignores the transfer flag only if it is absent; here it is
    # present and False after the round trip below.
    card.transfer = False
    path = tmp_path / "card.json"
    path.write_text(card.to_json())
    return path


@pytest.mark.parametrize("mode", [None, "off", "pause"])
def test_code_prompt_only_describes_visual_apis_when_enabled(mode):
    from pyrualean.play import build_prompt
    from pyrualean.supervision import MonitorConfig

    card = TaskCard(
        suite="arx_x5", task="organize_table", seed=0,
        task_language="Organize the table", object_names=[], robot="robodojo_xpl_rgb",
    )
    text = build_prompt(
        "cells", card, budget_s=5400, max_programs=None, primitives="no-vla",
        images="on-demand", feedback="pure", budget_mode="calls",
        monitor=None if mode is None else MonitorConfig(
            mode=mode, url="http://localhost:1" if mode == "pause" else None,
        ),
    )
    # The optional helper used to reference an unbound evidence() and require
    # watch even in the default-off prompt. Keep monitored APIs scoped to opt-in.
    assert ("evidence()" in text) is (mode == "pause")
    assert ("watch(phase, facts" in text) is (mode == "pause")
    assert "All callbacks still need watch contracts" not in text
    assert "Only verified permits dependent work" not in text
    assert "strategy.verify_grasp" in text
    if mode == "pause":
        assert "explicit HARD visual prerequisites" in text
        assert "unresolved execution receipts remain enforced" in text
