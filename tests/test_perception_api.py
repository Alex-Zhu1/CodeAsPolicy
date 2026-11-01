"""Typed perception results and calibration retain the capture that produced them."""

from copy import deepcopy

import numpy as np
import pytest

from pyrualean import ToolError
from pyrualean.hosts import robodojo_sight as sight
from pyrualean.robodojo import ObjectPose, RobodojoRobot, Segment, WorldPoint
from pyrualean.robodojo_xpl_rgb import ObjectSight, RobodojoOfficialColourRobot, SeenPoint
from tests.test_robodojo import FakeRobodojoBackend


class CapturedBackend(FakeRobodojoBackend):
    """A recorded source protocol over the typed API's existing two-arm fake."""

    session_id = "perception-api-session"
    supports_snapshot_perception = True

    def _record(self, command, result):
        super()._record(command, result)
        self.steps[-1]["native_steps"] = self.step_count
        self.steps[-1]["state"]["image_state"] = deepcopy(self.steps[-1]["state"]["state"])

    def observation_metadata(self, step=-1):
        index = step if step >= 0 else len(self.steps) + step
        source = f"capture-{index}"
        return dict(
            schema="pyrualean.observation.v1", session_id=self.session_id,
            observation_id=source, source_id=source, step=index,
            native_steps=self.steps[index]["native_steps"],
            source_freshness="captured", aligned=True,
            images={camera: dict(source_id=source, image_state_source_id=source,
                                 calibration_source_id=source, calibration_method="packet_matrices",
                                 aligned=True, artifact=f"{camera}.png")
                    for camera in RobodojoRobot.CAMERAS},
        )

    def _view(self, step):
        return dict(super()._view(step), observation=self.observation_metadata(step))

    def refresh_observation(self):
        self._record(None, None)
        return self._view(-1)

    def execute_tool(self, name, arguments):
        if name not in ("segment", "locate", "back_project"):
            return super().execute_tool(name, arguments)
        arguments = dict(arguments)
        step = arguments.pop("observation_step")
        assert arguments.pop("session_id") == self.session_id
        assert arguments.pop("observation_id") == self.observation_metadata(step)["observation_id"]
        result = super().execute_tool(name, arguments)
        result.result["provenance"] = dict(
            self.observation_metadata(step), schema="pyrualean.perception.v1",
            operation=name, frozen=True,
        )
        self.last_payload = result.result
        return result


@pytest.mark.parametrize("robot_cls", [RobodojoRobot, RobodojoOfficialColourRobot])
def test_typed_perception_preserves_detached_capture_sources(robot_cls):
    backend = CapturedBackend()
    robo = robot_cls(backend, primitives="no-vla")
    snapshot = robo.snapshot()
    segment = robo.segment("the green bottle", observation_id=snapshot["observation_id"])
    point = robo.back_project(*segment.centroid, segment.camera,
                              observation_id=segment.provenance["observation_id"])
    pose = robo.locate("the green bottle", observation_id=snapshot["observation_id"])
    for result in (segment, point, pose):
        assert result.provenance["observation_id"] == snapshot["observation_id"]
        assert result.provenance["images"]["cam_head"]["source_id"] == snapshot["source_id"]
        assert robo.require_current(result) is result
    backend.last_payload["provenance"]["images"]["cam_head"]["source_id"] = "wrong-frame"
    assert robo.perception_is_current(pose)
    pose.provenance["images"]["cam_head"]["source_id"] = "edited-by-policy"
    assert not robo.perception_is_current(pose)
    assert robo.perception_is_current(segment)  # another result cannot mutate this record


def test_missing_object_still_has_a_capture_identity():
    robo = RobodojoRobot(CapturedBackend(), primitives="no-vla")
    snapshot = robo.snapshot()
    absent = robo.segment("the red mug", observation_id=snapshot["observation_id"])
    assert not absent.found and absent.centroid is None
    assert absent.provenance["observation_id"] == snapshot["observation_id"]
    assert robo.perception_is_current(absent)  # currency does not mean the object was found


@pytest.mark.parametrize("robot_cls", [RobodojoRobot, RobodojoOfficialColourRobot])
def test_historical_packet_calibration_stays_with_the_image(robot_cls):
    backend = CapturedBackend()
    robo = robot_cls(backend, primitives="no-vla")
    old = robo.snapshot()
    first = robo.camera_meta("cam_left_wrist", observation_id=old["observation_id"])
    backend.refresh_observation()
    backend.steps[-1]["state"]["cameras"]["cam_left_wrist"] = deepcopy(
        backend.steps[-1]["state"]["cameras"]["cam_left_wrist"]
    )
    backend.steps[-1]["state"]["cameras"]["cam_left_wrist"]["extrinsic_matrix"][0][3] = 4.0
    latest = robo.snapshot()
    current = robo.camera_meta("cam_left_wrist")
    historical = robo.camera_meta("cam_left_wrist", observation_id=old["observation_id"])
    assert historical == first
    assert current["extrinsic_matrix"][0][3] == 4.0
    assert historical["extrinsic_matrix"][0][3] == 0.0
    assert historical["provenance"]["image_source_id"] == old["source_id"]
    assert current["provenance"]["observation_id"] == latest["observation_id"]
    assert robo.camera_meta("cam_left_wrist", step=0) == historical
    with pytest.raises(ValueError, match="different observations"):
        robo.camera_meta("cam_left_wrist", step=latest["step"],
                         observation_id=old["observation_id"])


def test_historical_rgb_calibration_uses_image_time_arm_state():
    backend = CapturedBackend()
    record = backend.steps[0]["state"]
    record["cameras"] = {}
    captured_pose = deepcopy(record["image_state"]["left_ee_pose"])
    record["state"]["left_ee_pose"][0] += 1.0  # telemetry is later than the image
    robo = RobodojoOfficialColourRobot(backend, primitives="no-vla")
    snapshot = robo.snapshot()
    meta = robo.camera_meta("cam_left_wrist", observation_id=snapshot["observation_id"])
    assert np.allclose(meta["extrinsic_matrix"], sight.wrist_view(captured_pose).pose)
    assert meta["provenance"]["calibration_source_id"] == snapshot["source_id"]
    assert meta["provenance"]["calibration_method"] == "robot_model"
    assert meta["provenance"]["model"] == "robodojo_sight.wrist_view:v1"


def test_rgb_calibration_refuses_missing_image_time_pose():
    backend = CapturedBackend()
    backend.steps[0]["state"]["cameras"] = {}
    backend.steps[0]["state"]["image_state"] = {}
    robo = RobodojoOfficialColourRobot(backend, primitives="no-vla")
    with pytest.raises(ToolError, match="image-time arm pose"):
        robo.camera_meta("cam_left_wrist")


@pytest.mark.parametrize("invalid", [
    {"intrinsic_matrix": np.eye(3).tolist()},
    {"extrinsic_matrix": np.eye(4).tolist()},
    {"intrinsic_matrix": [[float("nan")]], "extrinsic_matrix": np.eye(4).tolist()},
])
def test_rgb_calibration_refuses_invalid_packet_instead_of_falling_back_to_model(invalid):
    backend = CapturedBackend()
    backend.steps[0]["state"]["cameras"]["cam_left_wrist"] = invalid
    robo = RobodojoOfficialColourRobot(backend, primitives="no-vla")
    with pytest.raises(ToolError, match="invalid calibration"):
        robo.camera_meta("cam_left_wrist")


def test_explicit_record_step_keeps_its_calibration_when_source_id_is_repeated():
    class RepeatedSource(CapturedBackend):
        def observation_metadata(self, step=-1):
            return dict(super().observation_metadata(step), observation_id="repeated",
                        source_id="repeated")

    backend = RepeatedSource()
    robo = RobodojoOfficialColourRobot(backend, primitives="no-vla")
    first = robo.camera_meta("cam_left_wrist", step=0)
    backend.refresh_observation()
    backend.steps[-1]["state"]["cameras"]["cam_left_wrist"]["extrinsic_matrix"][0][3] = 2.0
    latest = robo.camera_meta("cam_left_wrist")
    assert latest["extrinsic_matrix"][0][3] == 2.0
    assert robo.camera_meta("cam_left_wrist", step=0) == first
    assert robo.camera_meta("cam_left_wrist", step=0, observation_id="repeated") == first


def test_legacy_backend_refuses_explicit_binding_without_changing_its_arguments():
    backend = FakeRobodojoBackend()
    robo = RobodojoRobot(backend, primitives="no-vla")
    old = robo.locate("the green bottle")
    assert old.provenance["binding"] == "unsupported"
    assert not robo.perception_is_current(old)
    assert backend.calls[-1][1] == {
        "text_prompt": "the green bottle", "camera": "cam_head", "min_score": 0.2,
    }
    for method, args in ((robo.locate, ("bottle",)), (robo.segment, ("bottle",)),
                         (robo.back_project, (200, 300)), (robo.camera_meta, ())):
        with pytest.raises(NotImplementedError, match="observation_id"):
            method(*args, observation_id="foreign-capture")


def test_result_provenance_addition_preserves_positional_constructor_compatibility():
    pose_fields = (True, (0., 0., 1.), 1., .1, .2, .1, 0., 90., .9, (2, 3), 5, "cam_head", None)
    assert ObjectPose(*pose_fields).provenance is None
    sight_result = ObjectSight(*pose_fields, 2)
    assert sight_result.views == 2 and sight_result.provenance is None
    point = SeenPoint((2, 3), (0., 0., 1.), 1., "cam_head", "object")
    assert point.surface == "object" and point.provenance is None
    assert WorldPoint((2, 3), (0., 0., 1.), 1., "cam_head").provenance is None
    assert Segment(False, None, 0, None, None, "cam_head").provenance is None
