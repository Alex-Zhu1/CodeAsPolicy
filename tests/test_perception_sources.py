"""Perception reuses geometry only within its image and calibration source."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hosts import robodojo_sight as sight
from pyrualean.hosts import robodojo_tools as tools


def _world():
    view = sight.head_view()
    obs = {
        "state": {"left_arm_joint_state": [0.0] * 6},
        "image_state": {"left_arm_joint_state": [0.0] * 6},
        "vision": {
            "cam_head": {
                "color": np.zeros((480, 640, 3), dtype=np.uint8),
                "intrinsic_matrix": np.array(
                    [[view.fx, 0, view.cx], [0, view.fy, view.cy], [0, 0, 1]]
                ),
                "extrinsic_matrix": view.pose.copy(),
            }
        },
    }
    context = {"session_id": "session-a", "observation_id": "obs-a", "source_id": "frame-a"}
    prim = SimpleNamespace(
        env=SimpleNamespace(get_obs=lambda: obs),
        _snapshot_context=context,
        sam3_client=None,
    )
    return prim, obs


@pytest.mark.parametrize(
    "changed", ["pixel", "intrinsic", "extrinsic", "image_state", "source", "session"]
)
def test_equal_joints_do_not_reuse_geometry_from_another_capture_or_calibration(changed):
    prim, obs = _world()
    tools._scene(prim, obs)["regions"].append(
        {"camera": "cam_head", "rows": np.array([300]), "cols": np.array([320]),
         "points": np.array([[0.0, 0.0, 0.9]])}
    )
    assert tools.back_project(prim, 300, 320)["surface"] == "object"
    cam = obs["vision"]["cam_head"]
    if changed == "pixel":
        cam["color"][1, 1, 0] = 1  # outside the old sampled eight-pixel grid
    elif changed == "intrinsic":
        cam["intrinsic_matrix"][0, 0] += 1
    elif changed == "extrinsic":
        cam["extrinsic_matrix"][0, 3] += 0.01
    elif changed == "image_state":
        obs["image_state"]["left_arm_joint_state"][0] = 0.1
    elif changed == "source":
        prim._snapshot_context["source_id"] = "frame-b"
    elif changed == "session":
        prim._snapshot_context["session_id"] = "session-b"
    out = tools.back_project(prim, 300, 320)
    assert out["surface"] == "table"
    assert out["geometry_provenance"]["method"] == "ray_table_intersection"
    assert obs["state"]["left_arm_joint_state"] == [0.0] * 6


def test_rgb_back_projection_reads_one_packet_and_uses_its_calibration():
    prim, obs = _world()
    later = deepcopy(obs)
    later["vision"]["cam_head"]["extrinsic_matrix"][0, 3] += 0.2
    calls = []

    def changing_observation():
        calls.append(None)
        return obs if len(calls) == 1 else later

    prim.env.get_obs = changing_observation
    first = tools.back_project(prim, 300, 320)
    assert len(calls) == 1
    second = tools.back_project(prim, 300, 320)
    assert len(calls) == 2
    assert second["world_xyz"][0] - first["world_xyz"][0] == pytest.approx(0.2)
    provenance = second["geometry_provenance"]
    assert provenance["basis"] == "inferred_rgb"
    assert provenance["depth_is_measured"] is False
    assert provenance["calibration_basis"] == {"cam_head": "observation_camera_matrices"}


def test_depth_and_inferred_rgb_do_not_claim_the_same_measurement_basis():
    prim, obs = _world()
    inferred = tools.back_project(prim, 300, 320)
    obs["vision"]["cam_head"]["depth"] = np.ones((480, 640))
    depth = tools.back_project(prim, 300, 320)
    assert inferred["geometry_provenance"]["method"] == "ray_table_intersection"
    assert depth["geometry_provenance"]["method"] == "depth_back_projection"
    assert depth["geometry_provenance"]["basis"] == "depth_image"


@pytest.mark.parametrize("depth", [False, True])
@pytest.mark.parametrize("focal", [0, float("nan"), float("inf")])
def test_invalid_snapshot_camera_matrix_does_not_silently_use_model_calibration(depth, focal):
    prim, obs = _world()
    obs["vision"]["cam_head"]["intrinsic_matrix"][0, 0] = focal
    if depth:
        obs["vision"]["cam_head"]["depth"] = np.ones((480, 640))
    assert "error" in tools.back_project(prim, 300, 320)


def test_snapshot_height_measurements_never_consume_or_poison_historical_priors(monkeypatch):
    prim, obs = _world()
    prim._measured_heights = {"cube": (0.19, "prism", 0.99)}
    priors = []
    mask = np.zeros((480, 640), dtype=bool)
    mask[290:310, 310:330] = True

    def measure(first, others, known):
        priors.append(known)
        return {
            "views": 2,
            "remembered": False,
            "match": 0.95,
            "height_m": 0.04,
            "kind": "prism",
            "rows": np.array([300]),
            "cols": np.array([320]),
            "points": np.array([[0.0, 0.0, 0.79]]),
        }

    monkeypatch.setattr(sight, "measure", measure)
    monkeypatch.setattr(sight, "seeds", lambda first: np.empty((0, 3)))
    monkeypatch.setattr(tools, "object_geometry", lambda points: {"height_m": 0.04})
    for source in ("current", "current", "historical", "current"):
        prim._snapshot_context.update(observation_id=source, source_id=source)
        tools._measure_in_colour(prim, obs, "cam_head", mask, "cube", 0.2)
    assert priors == [None, (0.04, "prism", 0.95), None, None]
    assert prim._measured_heights == {"cube": (0.19, "prism", 0.99)}


def test_inferred_surface_keeps_multiview_basis_when_reused_by_back_projection():
    prim, obs = _world()
    tools._scene(prim, obs)["regions"].append(
        {"camera": "cam_head", "rows": np.array([300]), "cols": np.array([320]),
         "points": np.array([[0.0, 0.0, 0.9]]),
         "seen": {"views": 2, "used_cameras": ["cam_head", "cam_left_wrist"],
                  "remembered": False, "kind": "prism", "match": 0.93}}
    )
    result = tools.back_project(prim, 300, 320)["geometry_provenance"]
    assert result["method"] == "rgb_inferred_surface"
    assert result["agreement_view_count"] == 2
    assert result["used_cameras"] == ["cam_head", "cam_left_wrist"]
    assert result["prior_scope"] == "same_snapshot"
    assert result["multiview_match"] == 0.93
