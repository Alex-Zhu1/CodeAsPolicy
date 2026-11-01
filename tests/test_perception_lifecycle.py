"""Public perception keeps historical geometry and refuses expired applicability."""

import copy

import pytest

from pyrualean._backend import ToolError
from pyrualean.arms import ArmToolkit
from pyrualean.robodojo import RobodojoRobot
from tests.test_perception_snapshots import Segmenter, packet, setup


def test_historical_pixels_keep_their_geometry_but_cannot_be_current_targets(tmp_path):
    env, host, backend = setup(tmp_path)
    robo = RobodojoRobot(backend)
    old = robo.snapshot()
    point = robo.back_project(4, 4, observation_id=old["observation_id"])
    assert point.xyz == pytest.approx([.2, .4, 1.])
    assert robo.require_current(point) is point
    env.packet = packet(2, depth=.5, translation=-.2)
    # Even before a new record is saved, the live source invalidates the old result.
    assert not robo.perception_is_current(point)
    with pytest.raises(ToolError, match="stale"):
        robo.require_current(point)
    current = robo.snapshot(refresh=True)
    historic = robo.back_project(4, 4, observation_id=old["observation_id"])
    fresh = robo.back_project(4, 4, observation_id=current["observation_id"])
    assert historic.xyz == point.xyz
    assert historic.provenance["current_at_return"] is False
    assert fresh.xyz == pytest.approx([-.2, .4, 1.5])
    assert robo.perception_is_current(fresh)
    assert not env.actions and len(host.state.records) == 2


def test_inference_finishing_after_source_change_keeps_original_sources(tmp_path):
    segmenter = Segmenter()
    env, _, backend = setup(tmp_path, segmenter)
    robo = RobodojoRobot(backend)
    source = robo.snapshot()
    segmenter.callback = lambda: setattr(env, "packet", packet(2, depth=.5))
    result = robo.locate("object", observation_id=source["observation_id"])
    assert result.found and result.top_z == pytest.approx(1.)
    assert result.provenance["source_id"] == "capture:1"
    assert result.provenance["current_at_return"] is False
    with pytest.raises(ToolError):
        robo.require_current(result)
    assert not env.actions


def test_same_native_step_new_capture_invalidates_result(tmp_path):
    env, _, backend = setup(tmp_path)
    robo = RobodojoRobot(backend)
    result = robo.locate("object")
    assert result.provenance["current_at_return"]
    env.packet = packet(2)
    env.packet["pyrualean_observation"]["native_steps"] = 0
    assert backend.env_steps() == 0
    assert not robo.perception_is_current(result)


def test_action_attempt_expires_geometry_even_with_same_camera_frame(tmp_path):
    env, host, backend = setup(tmp_path)
    robo = RobodojoRobot(backend)
    result = robo.locate("object")
    # A controller may complete a zero-distance/no-op request without a new frame.
    host._tools["grip"] = lambda **kwargs: {"reached": True, "steps_used": 0}
    robo.set_gripper("right", robo.OPEN)
    assert not env.actions
    assert not robo.perception_is_current(result)
    with pytest.raises(ToolError):
        robo.require_current(result)


def test_forged_or_cross_session_provenance_never_passes_current_check(tmp_path):
    _, _, backend = setup(tmp_path / "first")
    robo = RobodojoRobot(backend)
    point = robo.back_project(4, 4)
    _, _, second_backend = setup(tmp_path / "second")
    other = RobodojoRobot(second_backend)
    assert not other.perception_is_current(point)
    before = copy.deepcopy(point.provenance)
    point.provenance["source_id"] = "invented"
    assert not robo.perception_is_current(point)
    point.provenance.clear()
    point.provenance.update(before)
    assert robo.perception_is_current(point)  # original host record was untouched


def test_wrong_result_binding_fails_without_replacing_current_state(tmp_path, monkeypatch):
    _, _, backend = setup(tmp_path)
    robo = RobodojoRobot(backend)
    source = robo.snapshot()
    execute = backend.execute_tool

    def corrupt(tool, arguments):
        payload = execute(tool, arguments)
        if tool == "back_project":
            payload["provenance"]["observation_id"] = "another-capture"
        return payload

    monkeypatch.setattr(backend, "execute_tool", corrupt)
    with pytest.raises(ToolError, match="does not match"):
        robo.back_project(4, 4, observation_id=source["observation_id"])
    assert not robo._ledger.records[-1].ok
    assert robo._latest_observation_id == source["observation_id"]


def test_perception_callback_cannot_dispatch_motion(tmp_path):
    segmenter = Segmenter()
    env, host, backend = setup(tmp_path, segmenter)
    robo = RobodojoRobot(backend)
    segmenter.callback = lambda: robo.set_gripper("right", robo.OPEN)
    with pytest.raises(RuntimeError, match="read-only"):
        robo.locate("object")
    assert not env.actions and not host._receipts


def test_new_cell_checks_stale_result_before_issuing_motion(tmp_path):
    env, host, backend = setup(tmp_path / "native")
    robo = RobodojoRobot(backend)
    cells = ArmToolkit(robo, arm="cells", workspace=tmp_path / "cells", images=False)
    initial = cells.execute_tool("python", {"code": "target = robo.locate('object')"})
    assert not initial.is_error
    env.packet = packet(2)
    blocked = cells.execute_tool("python", {"code":
        "robo.require_current(target)\nrobo.set_gripper('right', robo.OPEN)"})
    assert blocked.is_error and "stale" in blocked.result["exception"]
    assert not env.actions and not host._receipts
    repaired = cells.execute_tool("python", {"code":
        "view = robo.snapshot(refresh=True)\n"
        "target = robo.locate('object', observation_id=view['observation_id'])\n"
        "robo.require_current(target)"})
    assert not repaired.is_error
    assert cells._globals["target"].provenance["source_id"] == "capture:2"


def test_live_source_query_failure_does_not_grant_applicability(tmp_path, monkeypatch):
    _, _, backend = setup(tmp_path)
    robo = RobodojoRobot(backend)
    point = robo.back_project(4, 4)

    def unavailable():
        raise TimeoutError("source unavailable")

    monkeypatch.setattr(backend, "current_observation_source", unavailable)
    assert not robo.perception_is_current(point)
