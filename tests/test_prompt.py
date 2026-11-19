"""The prompt is generated from the live library and stays within budget."""

from __future__ import annotations

import inspect

import pytest

from codeaspolicy import LiberoRobot, TaskCard, api_reference, render_prompt
from codeaspolicy.libero import RESULT_TYPES


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
    from codeaspolicy.play import build_prompt
    from codeaspolicy.robots import get_adapter

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
    assert "Contingency & Cleanup Reserve" in text
    assert "never justifies higher-order multi-stage detours" in text
    assert "wider than the object's full diagonal footprint" in text
    assert "Do not abandon a task" in text
    assert "unknown is not empty" in text
    assert "table-height geometry" in text
    assert "rotated_end_pose" in text
    assert "vertical-axis twist" in text
    assert "sequential single-arm clearance" in text
    assert "place-and-regrasp" in text
    assert "object_bottom_z" in text and "L_below + 0.03" in text
    assert "fixed jaw-rail crossbar" in text and "±4.5–5.0 cm" in text
    assert "tangent-line contact" in text and "z = z_support + h" in text
    for forbidden in ("insert_key", "keyhole", "make_kong", "mahjong"):
        assert forbidden not in text
    assert "robo.move_to(\n" in api_reference(get_adapter(robot).robot_cls, intent_only=False)


@pytest.mark.parametrize("robot", ["robodojo", "robodojo_xpl", "robodojo_xpl_rgb"])
def test_offline_generation_defaults_to_shared_operations(robot):
    from codeaspolicy.robots import get_adapter

    card = TaskCard(suite="arx_x5", task="test", seed=0, task_language="Move the object",
                    object_names=[], robot=robot)
    cls = get_adapter(robot).robot_cls
    for text in (api_reference(cls), render_prompt(card).system):
        for name in ("execute_motion", "begin_grasp", "align_local", "operation_status",
                     "resume_operation"):
            assert f"robo.{name}(" in text
        for name in ("move_to", "move_both", "motion_profile", "motion_window", "try_grasp"):
            assert f"robo.{name}(\n" not in text
    assert "codeaspolicy.operation.v1" in render_prompt(card).system


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
    from codeaspolicy.play import build_prompt

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
    from codeaspolicy.play import build_prompt
    from codeaspolicy.supervision import MonitorConfig

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


@pytest.mark.parametrize("robot", ["robodojo", "robodojo_xpl", "robodojo_xpl_rgb"])
def test_no_sam_prompt_exposes_geometry_without_sam3_and_keeps_intent_invariants(robot):
    import re

    from codeaspolicy.play import build_prompt
    from codeaspolicy.robots import get_adapter

    card = TaskCard(
        suite="arx_x5",
        task="swap_blocks",
        seed=0,
        task_language="Swap the two blocks",
        object_names=[],
        robot=robot,
    )
    cls = get_adapter(robot).robot_cls
    ref = api_reference(cls, primitives="no-sam")
    prompt_text = build_prompt(
        "cells",
        card,
        budget_s=900,
        max_programs=None,
        images="on-demand",
    )
    assert prompt_text == build_prompt(
        "cells",
        card,
        budget_s=900,
        max_programs=None,
        images="on-demand",
        primitives="no-sam",
    )
    bundle = render_prompt(card, budget_s=900, primitives="no-sam")

    for doc in (ref, prompt_text, bundle.system):
        for geom_fn in (
            "camera_ray",
            "ray_at_z",
            "triangulate",
            "project_point",
            "crop_grid",
            "pose_from_axes",
            "fk",
            "ik",
        ):
            assert f"robo.{geom_fn}(" in doc
        for pose_fn in ("shifted", "axes", "rotated"):
            assert f"pose.{pose_fn}(" in doc
        for geom_cls in ("CameraRay", "PlanePoint", "TriangulatedPoint", "PixelPoint", "IKSolution"):
            assert f"class {geom_cls}(" in doc
        for hidden_fn in ("locate", "segment", "back_project", "pi0_pick"):
            assert f"robo.{hidden_fn}(" not in doc
        for hidden_cls in ("ObjectPose", "ObjectSight", "Segment", "WorldPoint", "SeenPoint", "Pick"):
            assert f"class {hidden_cls}(" not in doc
        assert not re.search(r"pi0|lingbot|rldx|vla|sam3", doc, re.IGNORECASE)
        assert not re.search(r"\brobo\.(locate|segment|back_project)\b", doc)

    # All robodojo-intent.md and coordination invariants remain intact under no-sam.
    assert "Low-level movement and speed selection are disabled" in prompt_text
    assert "## Perception" in prompt_text and "## Judging progress" in prompt_text
    assert "Contingency & Cleanup Reserve" in prompt_text
    assert "never justifies higher-order multi-stage detours" in prompt_text
    assert "wider than the object's full diagonal footprint" in prompt_text
    assert "Do not abandon a task" in prompt_text
    assert "unknown is not empty" in prompt_text
    assert "table-height geometry" in prompt_text
    assert "rotated_end_pose" in prompt_text
    assert "vertical-axis twist" in prompt_text
    assert "sequential single-arm clearance" in prompt_text
    assert "place-and-regrasp" in prompt_text
    assert "object_bottom_z" in prompt_text and "L_below + 0.03" in prompt_text
    assert "fixed jaw-rail crossbar" in prompt_text and "±4.5–5.0 cm" in prompt_text
    assert "tangent-line contact" in prompt_text and "z = z_support + h" in prompt_text
    for forbidden in ("insert_key", "keyhole", "make_kong", "mahjong"):
        assert forbidden not in prompt_text
    assert "crop_grid" in prompt_text and "observe_object" in prompt_text and "observe_hold" in prompt_text


@pytest.mark.parametrize(
    "filename",
    [
        "robodojo-intent.md",
        "robodojo.md",
        "robodojo-nosam.md",
        "robodojo_xpl_rgb.md",
        "robodojo_xpl_rgb-nosam.md",
    ],
)
def test_robodojo_knowledge_encodes_compact_transfer_and_bottom_clearance(filename):
    from importlib import resources

    path = resources.files("codeaspolicy").joinpath(f"knowledge/{filename}")
    doc = path.read_text(encoding="utf-8")
    assert "place-and-regrasp" in doc
    assert "±1.2 cm" in doc and "0.145 m" in doc
    assert ">= 5 cm" in doc
    assert "object_bottom_z" in doc and "L_below + 0.03" in doc
    assert "and **direct\n    in-air handoff**" not in doc
    for forbidden in ("insert_key", "keyhole", "make_kong", "mahjong"):
        assert forbidden not in doc.lower()


@pytest.mark.parametrize(
    "filename",
    [
        "robodojo-intent.md",
        "robodojo.md",
        "robodojo-nosam.md",
        "robodojo_xpl_rgb.md",
        "robodojo_xpl_rgb-nosam.md",
    ],
)
def test_robodojo_knowledge_encodes_crowded_contact_and_regrasp_polarity(filename):
    import re
    from importlib import resources

    path = resources.files("codeaspolicy").joinpath(f"knowledge/{filename}")
    doc = path.read_text(encoding="utf-8")
    # ARX-X5 fixed jaw-rail crossbar & crowded-row selective contact invariants
    assert "fixed jaw-rail crossbar" in doc
    assert "±4.5–5.0 cm" in doc and "opening=0.0" in doc
    assert "align `close_dir` along the contact stroke" in doc
    assert "3–4.5 cm" in doc
    # Minimum closed-jaw gap, gravity-hinge reorientation, and curved-rim polarity invariants
    assert "GRIP_ZERO_M" in doc and "~2.3 mm" in doc
    assert "z_support + 0.018" in doc
    assert "tangent-line contact" in doc
    assert "<= 45°–60°" in doc and ">= 120°–180°" in doc
    assert "roll_deg=180" in doc
    assert "z = z_support + h" in doc
    for forbidden in ("insert_key", "keyhole", "make_kong", "mahjong"):
        assert forbidden not in doc.lower()
    # Verify newly added crowded-contact and curved-rim clauses stay free of task nouns
    for marker in (
        "selective contact / tipping inside a tightly packed row or array",
        "curved-rim",
    ):
        idx = doc.lower().index(marker)
        end = doc.find("\n- ", idx + 1)
        snippet = doc[idx : None if end == -1 else end]
        assert not re.search(r"\b(key|keys|lock|locks|kong|mahjong)\b", snippet, re.IGNORECASE)


def test_robodojo_knowledge_encodes_pre_contact_feature_alignment_with_parity():
    import re
    from importlib import resources

    filenames = (
        "robodojo-intent.md",
        "robodojo.md",
        "robodojo-novla.md",
        "robodojo-nosam.md",
        "robodojo_xpl_rgb.md",
        "robodojo_xpl_rgb-novla.md",
        "robodojo_xpl_rgb-nosam.md",
    )
    marker = "- **Pre-contact feature-to-target visual alignment (`robo.align_feature`)**:"
    snippets: list[str] = []
    for filename in filenames:
        doc = resources.files("codeaspolicy").joinpath(f"knowledge/{filename}").read_text(
            encoding="utf-8"
        )
        idx = doc.index(marker)
        end = doc.find("\n- ", idx + 1)
        snippet = doc[idx : None if end == -1 else end]
        snippets.append(snippet)
        assert "distal working feature" in snippet
        assert "crop_grid" in snippet and "triangulate" in snippet and "ray_at_z" in snippet
        assert "z_feature" in snippet and "oblique ray parallax" in snippet
        assert "achieved_stroke_m << stroke_m" in snippet
        assert "preview=True" in snippet
        assert not re.search(
            r"\b(insert_key|key|keys|keyhole|lock|locks|make_kong|kong|mahjong)\b",
            snippet,
            re.IGNORECASE,
        )
        assert not re.search(r"pi0|lingbot|rldx|vla|sam3", snippet, re.IGNORECASE)
        assert not re.search(r"\brobo\.(locate|segment|back_project)\b", snippet)
    assert all(s == snippets[0] for s in snippets), "feature alignment bullet must match across all variants"
