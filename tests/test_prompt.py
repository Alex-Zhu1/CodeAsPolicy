"""The prompt is generated from the live library and stays within budget."""

from __future__ import annotations

import inspect
import re
from collections import Counter

import pytest

from codeaspolicy import LiberoRobot, TaskCard, api_reference, knowledge_text, render_prompt
from codeaspolicy.libero import RESULT_TYPES


def _public_methods():
    return [
        name
        for name, member in vars(LiberoRobot).items()
        if not name.startswith("_")
        and (inspect.isfunction(member) or isinstance(member, property))
        and name not in ("ledger", "backend")
    ]


def _duplicate_paragraphs(text: str, min_chars: int = 80) -> list[str]:
    paras = [
        re.sub(r"\s+", " ", p).strip()
        for p in re.split(r"\n\s*\n", text)
        if len(p.strip()) >= min_chars
    ]
    return [p for p, count in Counter(paras).items() if count > 1]


SINGLE_CASE_FORBIDDEN = (
    "opening=0.38",
    "opening=0.35–0.42",
    "standoff_m=0.008",
    "z ≈ 0.85–0.87",
    "y ≈ -0.28",
    "101 * opening",
    "insert_key",
    "keyhole",
    "make_kong",
    "mahjong",
)


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
@pytest.mark.parametrize("primitives", ["full", "no-vla", "no-sam"])
def test_robodojo_prompts_stay_within_52kb_budget_and_have_no_duplicates_or_recipes(
    monkeypatch, robot, primitives
):
    from codeaspolicy.play import build_prompt
    from codeaspolicy.robots import get_adapter

    monkeypatch.delenv("CODEASPOLICY_DELTA_MODEL", raising=False)
    monkeypatch.delenv("CODEASPOLICY_DELTA_BASE_URL", raising=False)

    card = TaskCard(
        suite="arx_x5",
        task="test",
        seed=0,
        task_language="Move the object",
        object_names=[],
        robot=robot,
    )
    cls = get_adapter(robot).robot_cls
    cells_prompt = build_prompt(
        "cells", card, budget_s=900, max_programs=None, primitives=primitives
    )
    sys_prompt = render_prompt(card, primitives=primitives).system
    ref = api_reference(cls, primitives=primitives)
    know = knowledge_text(robot, primitives=primitives)

    assert len(cells_prompt) <= 52_000, f"{robot}/{primitives} cells={len(cells_prompt)}"
    assert len(sys_prompt) <= 52_000, f"{robot}/{primitives} sys={len(sys_prompt)}"
    assert len(know) <= 20_000, f"{robot}/{primitives} knowledge={len(know)}"
    if primitives == "no-sam":
        assert len(ref) <= 28_000, f"{robot}/no-sam api={len(ref)}"

    assert _duplicate_paragraphs(cells_prompt) == []
    assert _duplicate_paragraphs(sys_prompt) == []
    assert _duplicate_paragraphs(know) == []

    for doc in (cells_prompt, sys_prompt, ref, know):
        for forbidden in SINGLE_CASE_FORBIDDEN:
            assert forbidden not in doc.lower(), f"{forbidden!r} leaked into {robot}/{primitives}"


@pytest.mark.parametrize("robot", ["robodojo", "robodojo_xpl", "robodojo_xpl_rgb"])
@pytest.mark.parametrize("primitives", ["full", "no-vla", "no-sam"])
def test_orthogonal_coarse_delta_api_prunes_ceremony_while_preserving_runtime(
    monkeypatch, robot, primitives
):
    from codeaspolicy.play import build_prompt
    from codeaspolicy.robots import get_adapter

    monkeypatch.delenv("CODEASPOLICY_DELTA_MODEL", raising=False)
    monkeypatch.delenv("CODEASPOLICY_DELTA_BASE_URL", raising=False)

    card = TaskCard(
        suite="arx_x5",
        task="test",
        seed=0,
        task_language="Move the object",
        object_names=[],
        robot=robot,
    )
    cls = get_adapter(robot).robot_cls
    cells_text = build_prompt("cells", card, budget_s=900, max_programs=None, primitives=primitives)
    sys_text = render_prompt(card, primitives=primitives).system
    ref = api_reference(cls, primitives=primitives)

    exposed_core = (
        "execute_motion",
        "resume_motion",
        "motion_status",
        "set_gripper",
        "home",
        "delta_grasp",
        "align_feature",
        "nudge",
        "ik",
        "fk",
        "pose_from_axes",
        "crop_grid",
        "triangulate",
        "ray_at_z",
        "object_ref",
        "observe_object",
        "observe_hold",
    )
    hidden_ceremony = (
        "begin_grasp",
        "grasp_step",
        "grasp_status",
        "align_local",
        "preview_align_feature",
        "preview_delta_grasp",
        "operation_status",
        "resume_operation",
        "move_to",
        "move_both",
        "motion_profile",
        "motion_window",
        "try_grasp",
    )

    for doc in (ref, sys_text, cells_text):
        for name in exposed_core:
            assert f"robo.{name}(" in doc, f"missing robo.{name}("
        for name in hidden_ceremony:
            assert f"robo.{name}(" not in doc, f"should hide robo.{name}("
        assert "class Move(" not in doc

    # Runtime backward compatibility: every hidden robot method remains callable on the class
    # and appears when intent_only=False.
    full_ref = api_reference(cls, primitives=primitives, intent_only=False)
    for name in (
        "begin_grasp",
        "grasp_step",
        "grasp_status",
        "align_local",
        "operation_status",
        "resume_operation",
        "move_to",
        "move_both",
        "try_grasp",
        "visual_align_and_insert",
        "visual_servo",
    ):
        assert callable(getattr(cls, name)), f"{cls.__name__}.{name} must remain callable"
        assert f"robo.{name}(" in full_ref, f"intent_only=False missing robo.{name}("


@pytest.mark.parametrize("robot", ["robodojo", "robodojo_xpl_rgb"])
def test_visual_servo_and_visual_align_are_capability_gated(monkeypatch, robot):
    from codeaspolicy.play import build_prompt
    from codeaspolicy.robots import get_adapter

    card = TaskCard(
        suite="arx_x5",
        task="test",
        seed=0,
        task_language="Insert the part",
        object_names=[],
        robot=robot,
    )
    cls = get_adapter(robot).robot_cls

    # Unset env vars -> hidden in default intent_only=True prompt.
    monkeypatch.delenv("CODEASPOLICY_DELTA_MODEL", raising=False)
    monkeypatch.delenv("CODEASPOLICY_DELTA_BASE_URL", raising=False)
    for doc in (
        api_reference(cls, primitives="no-sam"),
        render_prompt(card, primitives="no-sam").system,
        build_prompt("cells", card, budget_s=900, max_programs=None, primitives="no-sam"),
    ):
        assert "robo.visual_align_and_insert(" not in doc
        assert "robo.visual_servo(" not in doc
        assert "class VisualAlign(" not in doc

    # Explicit delta_vla=True exposes them even without env vars.
    explicit = api_reference(cls, primitives="no-sam", delta_vla=True)
    assert "robo.visual_align_and_insert(" in explicit
    assert "robo.visual_servo(" in explicit
    assert "class VisualAlign(" in explicit

    # Setting CODEASPOLICY_DELTA_MODEL or CODEASPOLICY_DELTA_BASE_URL exposes them automatically.
    for env_var, val in (
        ("CODEASPOLICY_DELTA_MODEL", "pear-delta-v1"),
        ("CODEASPOLICY_DELTA_BASE_URL", "http://127.0.0.1:8000/v1"),
    ):
        monkeypatch.delenv("CODEASPOLICY_DELTA_MODEL", raising=False)
        monkeypatch.delenv("CODEASPOLICY_DELTA_BASE_URL", raising=False)
        monkeypatch.setenv(env_var, val)
        for doc in (
            api_reference(cls, primitives="no-sam"),
            render_prompt(card, primitives="no-sam").system,
            build_prompt("cells", card, budget_s=900, max_programs=None, primitives="no-sam"),
        ):
            assert "robo.visual_align_and_insert(" in doc
            assert "robo.visual_servo(" in doc
            assert "class VisualAlign(" in doc


def test_visual_servo_runtime_dispatch_and_fallback():
    from codeaspolicy.robodojo import RobodojoRobot, VisualAlign

    class _FakeBackend:
        def __init__(self, tools):
            self._tools = set(tools)
            self.calls = []
            self.steps = 0

        def execute_tool(self, name, kwargs):
            self.calls.append((name, kwargs))
            if name == "view_env_state":
                return {
                    "step": 0,
                    "task_language": "test",
                    "truncated": False,
                    "state": {
                        "state": {
                            "left_ee_pose": [-0.3, -0.3, 0.9, 1, 0, 0, 0],
                            "right_ee_pose": [0.3, -0.3, 0.9, 1, 0, 0, 0],
                            "left_ee_joint_state": [1.0],
                            "right_ee_joint_state": [1.0],
                            "left_arm_joint_state": [0.0] * 6,
                            "right_arm_joint_state": [0.0] * 6,
                        }
                    },
                }
            if name not in self._tools:
                return {"error": f"unknown tool {name!r}"}
            self.steps += 2
            return {
                "goal": kwargs["goal"],
                "arm": kwargs["arm"],
                "status": "completed",
                "reached": True,
                "done": True,
                "converged": False,
                "contact_complete": False,
                "stalled": False,
                "stop_reason": "done",
                "rounds_used": 1,
                "steps_used": 2,
                "step_allowance": kwargs["max_steps"],
                "final_xyz": [0.3, -0.15, 0.9],
                "holding": False,
                "policy": "fake",
                "rounds": [],
                "last_feedback": None,
                "policy_error": None,
            }

        def solved(self):
            return False

        def artifact(self, name, step=-1):
            return b""

        def env_steps(self):
            return self.steps

        def cancel(self):
            pass

    # Backend supporting visual_servo dispatches visual_servo directly.
    b_servo = _FakeBackend({"visual_servo", "visual_align_and_insert"})
    robo_servo = RobodojoRobot(b_servo)
    res_servo = robo_servo.visual_servo("hang mug on hook")
    assert isinstance(res_servo, VisualAlign) and res_servo.reached
    assert any(name == "visual_servo" for name, _ in b_servo.calls)

    # Backend supporting only visual_align_and_insert falls back cleanly.
    b_legacy = _FakeBackend({"visual_align_and_insert"})
    robo_legacy = RobodojoRobot(b_legacy)
    res_legacy = robo_legacy.visual_servo("insert plug")
    assert isinstance(res_legacy, VisualAlign) and res_legacy.reached
    assert any(name == "visual_align_and_insert" for name, _ in b_legacy.calls)
    assert all(name != "visual_servo" for name, _ in b_legacy.calls)


def test_robodojo_ik_binds_loosely_to_kinematics_envelopes_and_twist(monkeypatch):
    from codeaspolicy.hosts import robodojo_kinematics as kin
    from codeaspolicy.robodojo import RobodojoRobot

    class _MinBackend:
        def execute_tool(self, name, kwargs):
            return {
                "step": 0,
                "task_language": "test",
                "truncated": False,
                "state": {
                    "state": {
                        "left_ee_pose": [-0.3, -0.3, 0.9, 1, 0, 0, 0],
                        "right_ee_pose": [0.3, -0.3, 0.9, 1, 0, 0, 0],
                        "left_ee_joint_state": [0.5],
                        "right_ee_joint_state": [0.5],
                        "left_arm_joint_state": [0.0] * 6,
                        "right_arm_joint_state": [0.0] * 6,
                    }
                },
            }

        def solved(self):
            return False

        def artifact(self, name, step=-1):
            return b""

        def env_steps(self):
            return 0

        def cancel(self):
            pass

    robo = RobodojoRobot(_MinBackend())
    target = robo.pose_from_axes((0.20, -0.15, 0.85), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0))

    # 1. Twist headroom selection: with seed joint6 near +2.4 rad, twist_rad=+1.4 exceeds +3.1
    #    unless jaw_symmetry flips 180 deg into negative joint6.
    seed_high_j6 = (0.0, 0.6, 0.8, -0.3, 0.0, 2.4)
    sol_twist = robo.ik(
        "right",
        target,
        seed=seed_high_j6,
        jaw_symmetry=True,
        twist_rad=1.4,
        check_other_arm=False,
    )
    assert sol_twist.ok
    assert sol_twist.joints[5] + 1.4 <= kin.JOINT_LIMITS[5][1]

    # 2. Loose binding to inter_arm_violations: rejects when colliding, passes with check_other_arm=False.
    monkeypatch.setattr(
        kin,
        "inter_arm_violations",
        lambda lq, rq, **kw: ["camera_housing_collision"],
        raising=False,
    )
    blocked = robo.ik(
        "right",
        target,
        seed=(0.0,) * 6,
        other_joints=(0.0,) * 6,
        check_other_arm=True,
        opening=0.4,
    )
    assert not blocked.ok and blocked.collides
    assert blocked.reason is not None and "camera_housing_collision" in blocked.reason

    unchecked = robo.ik(
        "right",
        target,
        seed=(0.0,) * 6,
        other_joints=(0.0,) * 6,
        check_other_arm=False,
    )
    assert unchecked.ok and not unchecked.collides

    # 3. Loose binding to housing_exposure_penalty: jaw_symmetry picks the branch with lower penalty.
    monkeypatch.setattr(kin, "inter_arm_violations", lambda lq, rq, **kw: [], raising=False)
    base_sol = robo.ik("right", target, seed=(0.0,) * 6, jaw_symmetry=False, check_other_arm=False)
    base_z = kin.fk("right", base_sol.joints)[:3, 2]
    monkeypatch.setattr(
        kin,
        "housing_exposure_penalty",
        lambda arm, q, other_q=None: 1.0 if float(kin.fk(arm, q)[:3, 2] @ base_z) > 0.0 else 0.0,
        raising=False,
    )
    away = robo.ik(
        "right",
        target,
        seed=(0.0,) * 6,
        other_joints=(0.0,) * 6,
        jaw_symmetry=True,
        check_other_arm=True,
    )
    assert away.ok and away.flipped_jaw


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

    card = TaskCard(
        suite="libero_object_swap",
        task=2,
        seed=0,
        task_language="Move the object",
        object_names=[],
    )
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
    card.transfer = False
    path = tmp_path / "card.json"
    path.write_text(card.to_json())
    return path


@pytest.mark.parametrize("mode", [None, "off", "pause"])
def test_code_prompt_only_describes_visual_apis_when_enabled(mode):
    from codeaspolicy.play import build_prompt
    from codeaspolicy.supervision import MonitorConfig

    card = TaskCard(
        suite="arx_x5",
        task="organize_table",
        seed=0,
        task_language="Organize the table",
        object_names=[],
        robot="robodojo_xpl_rgb",
    )
    text = build_prompt(
        "cells",
        card,
        budget_s=5400,
        max_programs=None,
        primitives="no-vla",
        images="on-demand",
        feedback="pure",
        budget_mode="calls",
        monitor=None
        if mode is None
        else MonitorConfig(
            mode=mode,
            url="http://localhost:1" if mode == "pause" else None,
        ),
    )
    assert ("evidence()" in text) is (mode == "pause")
    assert ("watch(phase, facts" in text) is (mode == "pause")
    assert "All callbacks still need watch contracts" not in text
    assert "Only verified permits dependent work" not in text
    assert "strategy.verify_grasp" in text
    if mode == "pause":
        assert "explicit HARD visual prerequisites" in text
        assert "unresolved execution receipts remain enforced" in text


@pytest.mark.parametrize("robot", ["robodojo", "robodojo_xpl", "robodojo_xpl_rgb"])
def test_no_sam_prompt_exposes_geometry_without_sam3_and_keeps_core_knowledge(monkeypatch, robot):
    from codeaspolicy.play import build_prompt
    from codeaspolicy.robots import get_adapter

    monkeypatch.delenv("CODEASPOLICY_DELTA_MODEL", raising=False)
    monkeypatch.delenv("CODEASPOLICY_DELTA_BASE_URL", raising=False)

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

    # Core task-agnostic Coarse + Delta knowledge remains intact under no-sam.
    for marker in (
        "## Perception",
        "## Judging progress",
        "GRIP_ZERO_M",
        "place-and-regrasp",
        "object_bottom_z",
        "robo.delta_grasp(",
        "contact_table=True",
        "robo.align_feature(",
        "robo.nudge(",
        "stop_on_stall=True",
        "crop_grid",
        "triangulate",
        "ray_at_z",
        "z_support + h",
        "observe_object",
        "observe_hold",
    ):
        assert marker in prompt_text, f"missing {marker!r} in no-sam prompt"
    for forbidden in SINGLE_CASE_FORBIDDEN:
        assert forbidden not in prompt_text.lower()


@pytest.mark.parametrize(
    "filename",
    [
        "robodojo-intent.md",
        "robodojo.md",
        "robodojo-novla.md",
        "robodojo-nosam.md",
        "robodojo_xpl_rgb.md",
        "robodojo_xpl_rgb-novla.md",
        "robodojo_xpl_rgb-nosam.md",
    ],
)
def test_all_robodojo_knowledge_files_are_lean_deduplicated_and_task_agnostic(filename):
    from importlib import resources

    doc = resources.files("codeaspolicy").joinpath(f"knowledge/{filename}").read_text(
        encoding="utf-8"
    )
    assert len(doc) <= 10_000, f"{filename} unexpectedly large: {len(doc)}"
    assert _duplicate_paragraphs(doc) == []
    for marker in ("tcp_pos", "GRIP_ZERO_M", "place-and-regrasp", "object_bottom_z"):
        assert marker in doc, f"{filename} missing {marker!r}"
    for forbidden in SINGLE_CASE_FORBIDDEN:
        assert forbidden not in doc.lower(), f"{filename} contains single-case token {forbidden!r}"
    if filename.endswith("-nosam.md") or filename == "robodojo-intent.md":
        assert not re.search(r"pi0|lingbot|rldx|vla|sam3", doc, re.IGNORECASE)
        assert not re.search(r"\brobo\.(locate|segment|back_project)\b", doc)
