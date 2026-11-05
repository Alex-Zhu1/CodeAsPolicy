"""Task cards preserve native instructions and distinguish facts from missing data."""

import hashlib
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrualean.hosts.robodojo_card import build_card
from pyrualean.hosts.rpent_robodojo import Cell
from pyrualean.prompt import TaskCard
from tests.test_robodojo_trajectory import station


@pytest.mark.parametrize("module", [
    "rpent_robodojo", "xpolicylab_robodojo", "xpolicylab_robodojo_rgb",
])
def test_hosts_preserve_native_instruction_and_full_startup_capabilities(tmp_path, module):
    from pyrualean.play import build_prompt

    env, robo, host = station(tmp_path)
    instruction = '  把红块移到左边。\nDo not lift the tray. "left" means robot left.  '
    host.state.records[0]["instruction"] = instruction
    initial = host.state.records[0]["state"]
    # Current state differs; the card must still describe recorded step 0.
    env.grips["right"] = .8
    original = env.get_status
    env.get_status = lambda: {**original(), "success": True, "hidden_object_poses": [123]}
    toolkit = host
    if module == "rpent_robodojo":
        # RPent wraps the same public observation envelope in a tool result.
        toolkit = SimpleNamespace(
            state=host.state, _primitives=host._primitives,
            execute_tool=lambda name, args: SimpleNamespace(result=host.execute_tool(name, args)),
        )
    card = import_module(f"pyrualean.hosts.{module}").dump_card(
        toolkit, Cell("arx_x5", "example", 0), tmp_path)
    assert card.task_language == instruction and card.object_names == []
    assert card.acceptance["requirements"] == []
    assert card.acceptance["coverage"] == "unknown"
    assert card.acceptance["verdict_authority"] == "environment"
    assert card.facts["right gripper opening"].startswith(
        f'{initial["right_ee_joint_state"][0]:.2f} ')
    caps = card.facts["motion capabilities at episode start"]
    assert caps == robo.motion_capabilities()
    assert caps["frames"] == ["world"] and caps["collision_checks"] == ["sampled_arm_table"]
    assert caps["inter_arm_clearance"] is False and caps["force_control"] is False
    assert TaskCard.from_json(tmp_path / "card.json") == card
    provenance = card.provenance
    observation = host.state.records[0]["observation"]
    assert provenance["initial_observation"]["observation_id"] == observation["observation_id"]
    assert provenance["initial_observation"]["captured_at"] == observation["captured_at"]
    assert provenance["initial_observation"]["capture_clock"] == "test_sensor"
    assert provenance["instruction_sha256"] == hashlib.sha256(instruction.encode()).hexdigest()
    assert card.initial_state == initial
    assert provenance["initial_state_sha256"] == hashlib.sha256(
        json.dumps(initial, sort_keys=True, allow_nan=False).encode()).hexdigest()
    for camera, image in provenance["images"].items():
        data = Path(card.images[camera]).read_bytes()
        assert image["sha256"] == hashlib.sha256(data).hexdigest()
        assert image["size_bytes"] == len(data)
        assert Path(image["source_path"]).read_bytes() == data
        assert image["observation_id"] == observation["observation_id"]
        assert image["image_source_id"] == observation["source_id"]
    artifact = provenance["observation_artifact"]
    if module != "rpent_robodojo":
        assert artifact["sha256"] == hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
    else:
        assert artifact is None
    text = card.render()
    assert "copied verbatim" in text and "no Astra-generated task or plan" in text
    assert '"force_control": false' in text and '"speed_control": true' in text
    assert "simulator step budget" not in text
    assert "hidden_object_poses" not in card.to_json() and "success" not in card.to_json()
    prompt = build_prompt("cells", card, budget_s=1200, max_programs=None, primitives="no-vla")
    assert card.render(compact=True) in prompt and "robo.execute_motion(" in prompt
    assert "Task acceptance:" in prompt and "whole original instruction" in prompt
    assert "robo.motion_profile(" not in prompt
    assert text not in prompt and "# Execution plan" in prompt
    compact = card.render(compact=True)
    assert "force_control=no" in compact and "inter-arm clearance=no" in compact
    assert "reported aligned" in compact and "simulation time" in compact
    assert provenance["instruction_sha256"] not in compact
    assert "trajectory_clock" not in compact
    # Compare against the old full capability/source rendering, not the newly
    # enlarged provenance block, to ensure actual prompt savings.
    from dataclasses import replace
    assert len(compact) < len(replace(card, provenance={}).render()) * .7
    # This is a frozen startup description, not a silently refreshed live report.
    env.sim.cfg.decimation = 2
    assert caps["speed_control"] is True and robo.motion_capabilities()["speed_control"] is False
    assert not env.actions


@pytest.mark.parametrize("limit", [None, -1, True, float("inf"), 2.5])
def test_missing_measurements_are_unknown_and_budget_fallback_is_labelled(tmp_path, limit):
    _, _, host = station(tmp_path)
    card = build_card(
        cell=Cell("arx_x5", "example", 0, max_episode_steps=77), raw={}, instruction="",
        images={}, status={"step_limit": limit}, primitives=host._primitives, robot="robodojo",
    )
    assert card.task_language == ""  # never invent a task from the task identifier
    assert card.max_env_steps == 77
    assert "fallback" in card.sources["native action budget"]
    for arm in ("left", "right"):
        assert card.facts[f"{arm} arm initial grasp point [x, y, z]"].startswith("unknown")
        assert card.facts[f"{arm} gripper initial pitch / yaw / roll"].startswith("unknown")
        assert card.facts[f"{arm} gripper opening"].startswith("unknown")
    assert "0.00 (1 = open" not in card.render() and "[0.145" not in card.render()


def test_invalid_measurements_do_not_become_plausible_starting_poses(tmp_path):
    _, _, host = station(tmp_path)
    card = build_card(
        cell=Cell("arx_x5", "example", 0), instruction="Move the block", images={},
        raw={"left_ee_pose": [0, 0, 1, 0, 0, 0, 0],
             "right_ee_pose": [float("nan"), 0, 1, 1, 0, 0, 0],
             "left_ee_joint_state": [float("nan")], "right_ee_joint_state": [2]},
        status={"step_limit": 123}, primitives=host._primitives, robot="robodojo",
    )
    assert card.max_env_steps == 123 and "fallback" not in card.sources["native action budget"]
    for key, value in card.facts.items():
        if key.startswith(("left", "right")):
            assert value.startswith("unknown")
    assert "NaN" not in card.to_json()


def test_old_task_cards_without_sources_still_load(tmp_path):
    path = tmp_path / "old-card.json"
    path.write_text('{"suite":"arx_x5","task":"example","seed":0,'
                    '"task_language":"Move the block","object_names":[],"robot":"robodojo"}')
    card = TaskCard.from_json(path)
    assert card.sources == {} and card.provenance == {} and card.initial_state == {}
    assert "Move the block" in card.render()
    assert "native actions unknown" in card.render(compact=True)


def test_unavailable_native_status_uses_labelled_fallback_without_inventing_timing(tmp_path):
    from pyrualean.hosts.xpolicylab_robodojo import dump_card

    env, _, host = station(tmp_path)

    def unavailable():
        raise OSError("status unavailable")

    env.get_status = unavailable
    card = dump_card(host, Cell("arx_x5", "example", 0, max_episode_steps=91), tmp_path)
    assert card.max_env_steps == 91 and "fallback" in card.sources["native action budget"]
    caps = card.facts["motion capabilities at episode start"]
    assert caps["speed_control"] is False and caps["trajectory_clock"] is None


def test_zero_native_budget_is_not_replaced_with_configured_budget(tmp_path):
    from pyrualean.hosts.xpolicylab_robodojo import dump_card

    env, _, host = station(tmp_path)
    env.step_limit = 0
    card = dump_card(host, Cell("arx_x5", "example", 0, max_episode_steps=91), tmp_path)
    assert card.max_env_steps == 0
    assert "fallback" not in card.sources["native action budget"]


def test_initial_native_steps_are_distinct_from_recorded_index_and_live_budget(tmp_path):
    _, _, host = station(tmp_path)
    initial = host.state.records[0]
    metadata = {**initial["observation"], "native_steps": 7, "captured_at": 1.25,
                "alignment": {"state_frame": 7, "image_frame": 6}}
    card = build_card(
        cell=Cell("arx_x5", "example", 0), raw={**initial["state"], "hidden_object_poses": [9]},
        instruction="Move", images={}, status={"step": 11, "step_limit": 100},
        primitives=host._primitives, robot="robodojo", observation=metadata,
    )
    assert card.provenance["initial_observation"]["step"] == 0
    assert card.provenance["initial_observation"]["native_steps"] == 7
    assert card.provenance["initial_observation"]["alignment"] == metadata["alignment"]
    assert card.provenance["budget_at_card_creation"]["remaining"] == 89
    assert card.provenance["capability_snapshot"]["native_steps"] == 11
    assert "hidden_object_poses" not in card.to_json()
    compact = card.render(compact=True)
    assert "recorded step 0; native actions 7" in compact
    assert "remaining at card creation: 89" in compact


def test_missing_provenance_is_unknown_and_mismatched_sources_are_visible(tmp_path):
    _, _, host = station(tmp_path)
    image = tmp_path / "image.png"
    image.write_bytes(b"copied image bytes")
    kwargs = dict(cell=Cell("arx_x5", "example", 0), raw={}, instruction="",
                  images={"cam_head": str(image)}, status={"step": 13},
                  primitives=host._primitives, robot="robodojo")
    missing = build_card(**kwargs)
    assert missing.provenance["initial_observation"]["native_steps"] is None
    assert missing.provenance["initial_observation"]["observation_id"] is None
    compact = missing.render(compact=True)
    assert "alignment unverified" in compact and "native actions unknown" in compact
    assert "configured fallback" in compact and "instruction unavailable" in compact
    mismatched = build_card(**kwargs, observation={
        "observation_id": "state-now", "source_id": "state-now", "aligned": True,
        "images": {"cam_head": {"image_source_id": "image-before",
                                "image_state_source_id": "image-before", "aligned": True}},
    })
    assert "alignment mismatched" in mismatched.render(compact=True)


def test_offline_prompt_is_compact_without_advertising_unavailable_plan_helper(tmp_path):
    from pyrualean.prompt import render_prompt

    _, _, host = station(tmp_path)
    card = build_card(cell=Cell("arx_x5", "example", 0), raw={}, instruction="Move",
                      images={}, status={}, primitives=host._primitives, robot="robodojo")
    prompt = render_prompt(card, primitives="no-vla").text()
    assert card.render(compact=True) in prompt and card.render() not in prompt
    assert "# Execution plan" not in prompt
