"""A motion failure must not lose its evidence when the final show is skipped."""

import copy

import pytest

from pyrualean import LiberoRobot
from pyrualean.arms import ArmToolkit
from tests.test_interaction_contracts import ProtocolBackend

MOVE = "m = robo.move_to([0.05, -0.1, 1.5])"
FAILED_APPROACH = MOVE + "\nassert m.reached\nrobo.show('wrist')"


def setup(tmp_path, *, arm="cells", images="on-demand"):
    backend = ProtocolBackend()
    robo = LiberoRobot(backend)
    kit = ArmToolkit(robo, arm=arm, workspace=tmp_path, images=images, feedback="pure")
    reads = []
    original = backend.artifact

    def artifact(name, step=-1):
        reads.append((name, step))
        return original(name, step)

    backend.artifact = artifact
    return backend, robo, kit, reads


@pytest.mark.parametrize("arm", ["cells", "program"])
def test_assert_before_show_returns_source_bound_images_without_extra_motion(tmp_path, arm):
    backend, robo, kit, reads = setup(tmp_path, arm=arm)
    code = FAILED_APPROACH
    if arm == "program":
        code = "def run(robo):\n    " + code.replace("\n", "\n    ")
    result = kit.execute_tool("python" if arm == "cells" else "run_program", {"code": code})
    assert result.result["failure"]["type"] == "AssertionError"
    assert len(result.images) == 2
    assert reads == [("agentview_high.png", 1), ("wrist_high.png", 1)]
    for item in result.result["observation"]["images"]:
        assert item["source_observation_id"] == item["source_id"] == "capture-1"
        assert item["recorded_step"] == 1
        assert item["step_id"] == backend.frames > 0
        assert item["state_version"] == robo._action_revision == 1
        assert item["captured_at"] == 1001 and item["aligned"]
    assert len(backend.contexts) == 1 and len(backend.steps) == 2
    assert robo._ledger.stateful_calls() == 1
    assert result.result["recovery"]["env_steps"] == backend.frames
    assert not result.result["recovery"]["automatic_replay"]


@pytest.mark.parametrize("code", [
    MOVE + "\ndef broken(:", "missing_name", "robo.state()\nassert False",
])
def test_validation_and_readonly_failures_do_not_request_default_images(tmp_path, code):
    backend, _, kit, reads = setup(tmp_path)
    result = kit.execute_tool("python", {"code": code})
    assert result.is_error and result.images == [] and reads == []
    assert backend.frames == 0 and not backend.contexts


def test_normal_on_demand_motion_still_does_not_attach_images(tmp_path):
    _, _, kit, reads = setup(tmp_path)
    result = kit.execute_tool("python", {"code": MOVE})
    assert not result.is_error and result.images == [] and reads == []


def test_failure_uses_requested_camera_once_instead_of_adding_default_views(tmp_path):
    _, _, kit, reads = setup(tmp_path)
    result = kit.execute_tool("python", {"code": (
        "robo.show('wrist')\nrobo.show('wrist')\n" + FAILED_APPROACH
    )})
    assert result.is_error and len(result.images) == 1
    assert reads == [("wrist_high.png", 1)]
    assert kit.execute_tool("python", {"code": "1"}).images == []


def test_none_preserves_disabled_camera_images_on_failure(tmp_path):
    _, _, kit, reads = setup(tmp_path, images="none")
    result = kit.execute_tool("python", {"code": FAILED_APPROACH})
    assert result.is_error and result.images == [] and reads == []


def test_unknown_action_images_do_not_reconcile_or_allow_another_motion(tmp_path):
    backend, robo, kit, _ = setup(tmp_path)
    backend.next_status = "unknown"
    result = kit.execute_tool("python", {"code": FAILED_APPROACH})
    assert result.result["execution_paused"] and len(result.images) == 2
    action = result.result["handoff"]["unresolved_actions"][0]
    assert action["status"] == "unknown"
    assert robo._pending_actions == {action["action_id"]}
    assert all(item["state_version"] is None for item in result.result["observation"]["images"])
    frames = backend.frames
    again = kit.execute_tool("python", {"code": MOVE})
    assert again.result["execution_paused"]
    assert len(backend.contexts) == 1 and backend.frames == frames


def test_missing_camera_artifacts_preserve_failure_without_fabricated_images(tmp_path):
    _, robo, kit, _ = setup(tmp_path)

    def missing(*args, **kwargs):
        raise FileNotFoundError("camera artifact unavailable")

    robo._backend.artifact = missing
    result = kit.execute_tool("python", {"code": FAILED_APPROACH})
    assert result.result["failure"]["type"] == "AssertionError"
    assert result.images == [] and result.result["observation"]["images"] == []


@pytest.mark.parametrize("invalid", ["old", "unaligned", "unverified"])
def test_invalid_source_images_are_not_relabelled_current(tmp_path, invalid):
    backend, _, kit, reads = setup(tmp_path)
    original = backend.observation_metadata
    old = original()

    def metadata(step=-1):
        source = copy.deepcopy(old if invalid == "old" else original(step))
        if invalid == "unaligned":
            source["aligned"] = False
            for item in source["images"].values():
                item["aligned"] = False
        elif invalid == "unverified":
            source["source_freshness"] = "unverified"
        return source

    backend.observation_metadata = metadata
    result = kit.execute_tool("python", {"code": FAILED_APPROACH})
    assert result.is_error and len(result.images) == 2
    for item in result.result["observation"]["images"]:
        assert item["state_version"] is None
        if invalid == "old":
            assert item["source_observation_id"] == "capture-0"
            assert item["recorded_step"] == item["step_id"] == 0 < backend.frames
    expected_step = 0 if invalid == "old" else 1
    assert reads == [("agentview_high.png", expected_step), ("wrist_high.png", expected_step)]
