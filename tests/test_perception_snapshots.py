"""Native perception uses the exact recorded RGB/depth/calibration packet."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit


def packet(sequence=1, *, depth=1., translation=.2):
    pose = np.eye(4)
    pose[:3, 3] = [translation, .4, 2.]
    intrinsic = np.array([[100., 0, 4.], [0, 100., 4.], [0, 0, 1.]])
    state = {}
    for arm in ("left", "right"):
        state[f"{arm}_arm_joint_state"] = [0.] * 6
        state[f"{arm}_ee_joint_state"] = [1.]
        state[f"{arm}_ee_pose"] = [translation, .4, 2., 1., 0., 0., 0.]
    return dict(
        state=state, image_state={**state, "captured_sequence": sequence},
        instruction="inspect the object",
        vision={name: dict(color=np.full((8, 8, 3), sequence, np.uint8),
                           depth=np.full((8, 8), depth, np.float32),
                           intrinsic_matrix=intrinsic.copy(), extrinsic_matrix=pose.copy(),
                           shape=[8, 8, 3])
                for name in ("cam_head", "cam_left_wrist")},
        pyrualean_observation=dict(source_id=f"capture:{sequence}",
                                  native_steps=sequence - 1, aligned=True,
                                  captured_at=100. + sequence,
                                  capture_clock="client_unix_utc", source_freshness="captured"),
    )


class Environment:
    def __init__(self):
        self.packet = packet()
        self.reads = 0
        self.actions = []

    def get_obs(self):
        self.reads += 1
        return self.packet

    def get_status(self):
        return {"step": self.packet["pyrualean_observation"]["native_steps"], "step_limit": 100}

    def step(self, action):
        self.actions.append(action)
        raise AssertionError("perception must not dispatch motion")


class Segmenter:
    def __init__(self, callback=None):
        self.callback = callback
        self.images = []

    def segment(self, image, **kwargs):
        self.images.append(np.asarray(image).copy())
        if self.callback:
            self.callback()
        return SimpleNamespace(found=True, mask=np.ones((8, 8), bool), score=.9,
                               box=[0, 0, 7, 7])


def setup(tmp_path, segmenter=None):
    env = Environment()
    toolkit = NativeToolkit(env, segmenter or Segmenter(), tmp_path)
    return env, toolkit, NativeBackend(toolkit)


def reference(backend, step=-1):
    meta = backend.observation_metadata(step)
    return dict(session_id=meta["session_id"], observation_id=meta["observation_id"],
                observation_step=meta["step"])


@pytest.mark.parametrize("camera", ["cam_head", "cam_left_wrist"])
def test_historical_projection_keeps_original_depth_and_camera_pose(tmp_path, camera):
    env, toolkit, backend = setup(tmp_path)
    old = reference(backend)
    first = backend.execute_tool("back_project", dict(row=4, col=4, camera=camera, **old))
    assert first["world_xyz"] == pytest.approx([.2, .4, 1.])
    # Mutating the live packet, including its arrays, cannot rewrite the disk snapshot.
    env.packet["vision"][camera]["depth"][:] = 3.
    env.packet["vision"][camera]["extrinsic_matrix"][0, 3] = 99.
    env.packet = packet(2, depth=.5, translation=-.2)
    backend.refresh_observation()
    reads = env.reads
    previous = backend.execute_tool("back_project", dict(row=4, col=4, camera=camera, **old))
    current = backend.execute_tool("back_project", dict(row=4, col=4, camera=camera,
                                                        **reference(backend)))
    assert previous["world_xyz"] == first["world_xyz"]
    assert current["world_xyz"] == pytest.approx([-.2, .4, 1.5])
    assert env.reads == reads and not env.actions
    source = previous["provenance"]
    assert source["observation_id"] == "capture:1" and source["step"] == 0
    assert source["is_current_record"] is False and source["frozen"] is True
    image = source["images"][camera]
    assert image["source_id"] == image["calibration_source_id"] == "capture:1"
    assert image["depth_artifact"] == "sensors.npz"
    assert toolkit.state.load_observation(0)["image_state"]["captured_sequence"] == 1


def test_segmentation_and_geometry_remain_on_one_packet_when_environment_changes(tmp_path):
    env = Environment()
    segmenter = Segmenter(lambda: setattr(env, "packet", packet(2, depth=.5)))
    backend = NativeBackend(NativeToolkit(env, segmenter, tmp_path))
    reads = env.reads
    out = backend.execute_tool("locate", dict(text_prompt="the object", **reference(backend)))
    assert out["found"] and out["top_z"] == pytest.approx(1.)
    assert out["provenance"]["source_id"] == "capture:1"
    assert np.all(segmenter.images[0] == 1)
    assert env.packet["pyrualean_observation"]["source_id"] == "capture:2"
    assert env.reads == reads and not env.actions


@pytest.mark.parametrize("changes, message", [
    ({"observation_id": "missing"}, "unknown observation_id"),
    ({"session_id": "another-session"}, "another session"),
    ({"session_id": None}, "session_id is required"),
    ({"observation_step": 99}, "not recorded"),
    ({"observation_step": True}, "nonnegative integer"),
    ({"observation_id": "capture:2", "observation_step": 0}, "does not match"),
])
def test_invalid_observation_never_falls_back_to_current(tmp_path, changes, message):
    env, toolkit, backend = setup(tmp_path)
    kwargs = reference(backend)
    if changes.get("observation_id") == "missing":
        kwargs.pop("observation_step")
    kwargs.update(changes)
    reads = env.reads
    result = backend.execute_tool("locate", dict(text_prompt="object", **kwargs))
    assert result["code"] == "invalid_observation" and message in result["error"]
    assert not toolkit._primitives.sam3_client.images and env.reads == reads


def test_missing_historical_sensor_file_does_not_use_live_depth(tmp_path):
    env, toolkit, backend = setup(tmp_path)
    old = reference(backend)
    toolkit.state.artifact_path("sensors.npz", 0).unlink()
    result = backend.execute_tool("back_project", dict(row=4, col=4, **old))
    assert result["code"] == "invalid_observation"
    assert "world_xyz" not in result and env.reads == 1


def test_repeated_source_exact_step_keeps_each_record_and_raw_depth_dtype(tmp_path):
    env, toolkit, backend = setup(tmp_path)
    old = reference(backend)
    env.packet["vision"]["cam_head"]["depth"] = np.full((8, 8), 500, np.uint16)
    backend.refresh_observation()
    latest = reference(backend)
    assert latest["observation_id"] == old["observation_id"]
    first = backend.execute_tool("back_project", dict(row=4, col=4, **old))
    second = backend.execute_tool("back_project", dict(row=4, col=4, **latest))
    assert first["depth_m"] == 1. and second["depth_m"] == .5
    loaded = toolkit.state.load_observation(1)["vision"]["cam_head"]["depth"]
    assert loaded.dtype == np.uint16 and not loaded.flags.writeable


def test_legacy_source_is_frozen_without_fabricated_capture_evidence(tmp_path):
    env = Environment()
    env.packet.pop("pyrualean_observation")
    env.get_status = lambda: {"step": 0}
    backend = NativeBackend(NativeToolkit(env, Segmenter(), tmp_path))
    result = backend.execute_tool("segment", {"text_prompt": "object"})
    provenance = result["provenance"]
    assert provenance["frozen"] and provenance["source_id"] is None
    assert provenance["source_freshness"] == "unverified" and not provenance["aligned"]
    assert provenance["images"]["cam_head"]["calibration_source_id"] is None


def test_segment_returns_same_image_source_as_projection(tmp_path):
    env, toolkit, backend = setup(tmp_path)
    pinned = reference(backend)
    segmentation = backend.execute_tool("segment", dict(text_prompt="object", **pinned))
    row, col = segmentation["centroid_rc"]
    projection = backend.execute_tool("back_project", dict(row=row, col=col, **pinned))
    assert segmentation["provenance"]["source_id"] == projection["provenance"]["source_id"]
    assert segmentation["provenance"]["images"] == projection["provenance"]["images"]
    assert len(toolkit.state.records) == 1 and env.reads == 1


def test_returned_metadata_cannot_rewrite_stored_image_or_calibration_sources(tmp_path):
    env, toolkit, backend = setup(tmp_path)
    source = backend.observation_metadata()
    source["images"]["cam_head"]["calibration_source_id"] = "forged"
    state = backend.execute_tool("view_env_state", {"step": 0})
    state["state"]["cameras"]["cam_head"]["extrinsic_matrix"][0][3] = 99.
    state["observation"]["images"]["cam_head"]["source_id"] = "forged"
    result = backend.execute_tool("back_project", dict(row=4, col=4, **reference(backend)))
    assert result["world_xyz"] == pytest.approx([.2, .4, 1.])
    image = result["provenance"]["images"]["cam_head"]
    assert image["source_id"] == image["calibration_source_id"] == "capture:1"
    # The in-memory index contains metadata, not an unbounded raw-frame history.
    json.dumps(toolkit.state.records)


def test_partial_packet_calibration_is_not_labelled_as_robot_model(tmp_path):
    env = Environment()
    env.packet["vision"]["cam_head"].pop("extrinsic_matrix")
    backend = NativeBackend(NativeToolkit(env, Segmenter(), tmp_path, colour_only=True))
    result = backend.execute_tool("back_project", dict(row=4, col=4, **reference(backend)))
    assert "error" in result and "world_xyz" not in result
    image = result["provenance"]["images"]["cam_head"]
    assert image["calibration_method"] == "packet_matrices"
    assert image["calibration_model"] is None
