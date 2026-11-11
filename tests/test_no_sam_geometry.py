"""Unit tests for SAM-free camera geometry, annotated crops, and the ``no-sam`` primitive set."""

from __future__ import annotations

import math
import re
import sys

import numpy as np
import pytest

from pyrualean import TaskCard, ToolError, knowledge_text, render_prompt
from pyrualean import monitor as monitor_mod
from pyrualean._robot import NO_SAM_MESSAGE, PRIMITIVE_SETS, effective_primitives
from pyrualean.action_contract import ActionContract
from pyrualean.cli import main as cli_main
from pyrualean.hosts import robodojo_sight as sight
from pyrualean.hosts import sam3_local, xpolicylab_robodojo
from pyrualean.hosts import xpolicylab_bridge as xb
from pyrualean.play import build_prompt
from pyrualean.robodojo import (
    CameraRay,
    PixelPoint,
    PlanePoint,
    RobodojoRobot,
    TriangulatedPoint,
    WorldPoint,
)
from pyrualean.robodojo_xpl import RobodojoOfficialRobot
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
from tests.test_perception_api import CapturedBackend


def _rgb_model_robot(*, primitives: str = "no-sam") -> tuple[CapturedBackend, RobodojoOfficialColourRobot]:
    """Return a snapshot-capable colour-only RoboDojo robot using derived URDF camera poses."""
    backend = CapturedBackend()
    backend.steps[0]["state"]["cameras"] = {}
    robo = RobodojoOfficialColourRobot(backend, primitives=primitives)
    return backend, robo


@pytest.mark.parametrize("camera", ["cam_head", "cam_left_wrist", "cam_right_wrist"])
@pytest.mark.parametrize("use_packet_matrices", [False, True])
def test_project_ray_and_plane_round_trip_across_all_cameras(camera: str, use_packet_matrices: bool):
    backend = CapturedBackend()
    if not use_packet_matrices:
        backend.steps[0]["state"]["cameras"] = {}
        robo: RobodojoRobot = RobodojoOfficialColourRobot(backend, primitives="no-sam")
    else:
        # Populate exact View matrices for all three cameras so packet and derived paths agree.
        state = backend.steps[0]["state"]["image_state"]
        views = {
            "cam_head": sight.head_view(),
            "cam_left_wrist": sight.wrist_view(state["left_ee_pose"]),
            "cam_right_wrist": sight.wrist_view(state["right_ee_pose"]),
        }
        backend.steps[0]["state"]["cameras"] = {
            cam: {
                "shape": [v.shape[0], v.shape[1], 3],
                "intrinsic_matrix": [[v.fx, 0.0, v.cx], [0.0, v.fy, v.cy], [0.0, 0.0, 1.0]],
                "extrinsic_matrix": v.pose.tolist(),
                "has_depth": True,
            }
            for cam, v in views.items()
        }
        robo = RobodojoOfficialColourRobot(backend, primitives="no-sam")

    target_xyz = np.array([0.04, 0.12, 0.815], dtype=np.float64)
    px = robo.project_point(target_xyz, camera=camera)
    assert isinstance(px, PixelPoint)
    assert px.camera == camera
    assert px.depth_m > 0.0
    assert px.in_bounds is True

    ray = robo.camera_ray(px.row, px.col, camera=camera)
    assert isinstance(ray, CameraRay)
    assert ray.camera == camera
    assert ray.pixel == pytest.approx((px.row, px.col))
    u = np.asarray(ray.direction, dtype=np.float64)
    o = np.asarray(ray.origin, dtype=np.float64)
    assert float(np.linalg.norm(u)) == pytest.approx(1.0, abs=1e-12)

    # Point must lie directly along origin + s * direction with error < 1e-6 m.
    s = float(np.dot(target_xyz - o, u))
    assert s > 0.0
    assert float(np.linalg.norm((o + s * u) - target_xyz)) < 1e-7

    # Plane intersection at target_xyz[2] recovers exact 3D point and optical depth.
    plane_pt = robo.ray_at_z(px.row, px.col, z=float(target_xyz[2]), camera=camera)
    assert isinstance(plane_pt, PlanePoint)
    assert isinstance(plane_pt, WorldPoint)
    assert plane_pt.z_plane == pytest.approx(float(target_xyz[2]))
    assert plane_pt.depth_m == pytest.approx(px.depth_m, abs=1e-7)
    assert float(np.linalg.norm(np.asarray(plane_pt.xyz) - target_xyz)) < 1e-7
    assert plane_pt.provenance["geometry"]["basis"] == "calibrated_plane_intersection"
    assert plane_pt.provenance["geometry"]["depth_is_measured"] is False
    assert plane_pt.provenance["geometry"]["height_prior_used"] is True
    assert plane_pt.provenance["geometry"]["table_z_m"] == pytest.approx(float(target_xyz[2]))


@pytest.mark.parametrize(
    ("cam1", "cam2", "xyz"),
    [
        ("cam_head", "cam_left_wrist", (-0.08, 0.10, 0.83)),
        ("cam_head", "cam_right_wrist", (0.10, 0.08, 0.89)),
        ("cam_left_wrist", "cam_right_wrist", (0.00, 0.18, 0.85)),
    ],
)
def test_triangulate_recovers_exact_3d_point_across_camera_pairs(
    cam1: str, cam2: str, xyz: tuple[float, float, float]
):
    _, robo = _rgb_model_robot()
    p1 = robo.project_point(xyz, camera=cam1)
    p2 = robo.project_point(xyz, camera=cam2)
    assert p1 is not None and p2 is not None

    tri = robo.triangulate(cam1, (p1.row, p1.col), cam2, (p2.row, p2.col))
    assert isinstance(tri, TriangulatedPoint)
    assert tri.cameras == (cam1, cam2)
    assert tri.miss_m < 1e-9
    assert float(np.linalg.norm(np.asarray(tri.xyz) - np.asarray(xyz))) < 1e-6
    assert tri.depths_m[0] == pytest.approx(p1.depth_m, abs=1e-6)
    assert tri.depths_m[1] == pytest.approx(p2.depth_m, abs=1e-6)
    assert tri.provenance["geometry"]["basis"] == "calibrated_stereo_rays"
    assert tri.provenance["geometry"]["depth_is_measured"] is True
    assert tri.provenance["geometry"]["height_prior_used"] is False
    assert tri.provenance["geometry"]["agreement_view_count"] == 2


def test_geometry_rejections_and_behind_camera_handling():
    backend, robo = _rgb_model_robot()

    # Plane behind cam_head (cam_head sits at z ~ 1.47 looking down towards table z = 0.765).
    with pytest.raises(ToolError, match="behind camera"):
        robo.ray_at_z(240.0, 320.0, z=2.5, camera="cam_head")

    # Ray parallel to horizontal plane: wrist camera at initial level pose (pitch=0) has optical
    # centre row looking horizontally when local y == 0 -> world z component == 0!
    meta = robo.camera_meta("cam_right_wrist")
    k = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
    t = np.asarray(meta["extrinsic_matrix"], dtype=np.float64)
    view = sight.View(k, t, (480, 640))
    # Solve for pixel whose world ray has dz == 0: world_d = local_d @ R^T => world_dz = local_d . R[2, :]
    r2 = t[2, :3]
    # Pick col = cx so local_x = 0 => local_y * r2[1] - 1.0 * r2[2] = 0 => local_y = r2[2] / r2[1]
    assert abs(float(r2[1])) > 0.1
    local_y = float(r2[2] / r2[1])
    parallel_row = float(view.cy - view.fy * local_y)
    with pytest.raises(ToolError, match="parallel"):
        robo.ray_at_z(parallel_row, float(view.cx), z=0.765, camera="cam_right_wrist")

    # Same camera rejected in triangulate.
    with pytest.raises(ToolError, match="distinct cameras"):
        robo.triangulate("cam_head", (240, 320), "cam_head", (240, 320))

    # Mismatched pixels (large epipolar skew > max_miss_m) rejected with clear mm error.
    p_head = robo.project_point((0.0, 0.10, 0.80), camera="cam_head")
    p_right = robo.project_point((0.0, 0.10, 0.80), camera="cam_right_wrist")
    assert p_head is not None and p_right is not None
    with pytest.raises(ToolError, match=r"Rays miss by .* mm"):
        robo.triangulate(
            "cam_head",
            (p_head.row - 120.0, p_head.col),
            "cam_right_wrist",
            (p_right.row + 120.0, p_right.col),
            max_miss_m=0.015,
        )

    # Parallel rays across two cameras rejected.
    ray_head = robo.camera_ray(240.0, 320.0, camera="cam_head")
    # Construct a custom camera matrix identical in orientation to cam_head to test near-parallel rays.
    backend.steps[0]["state"]["cameras"] = {
        "cam_head": robo.camera_meta("cam_head"),
        "cam_left_wrist": {
            **robo.camera_meta("cam_head"),
            "extrinsic_matrix": [
                [*robo.camera_meta("cam_head")["extrinsic_matrix"][0][:3], -0.25],
                robo.camera_meta("cam_head")["extrinsic_matrix"][1],
                robo.camera_meta("cam_head")["extrinsic_matrix"][2],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "cam_right_wrist": robo.camera_meta("cam_right_wrist"),
    }
    robo_parallel = RobodojoOfficialColourRobot(backend, primitives="no-sam")
    with pytest.raises(ToolError, match="parallel"):
        robo_parallel.triangulate("cam_head", (240.0, 320.0), "cam_left_wrist", (240.0, 320.0))

    # Rays intersecting behind one camera (s1 = -0.3 <= 0, s2 = +0.5 > 0) rejected.
    p_behind = np.asarray(ray_head.origin) - 0.3 * np.asarray(ray_head.direction)
    cam2_pos = p_behind + np.array([0.5, 0.0, 0.0], dtype=np.float64)
    # Isaac camera convention: local -z is forward (-x in world => col 2 is +x = [1, 0, 0]).
    backend.steps[0]["state"]["cameras"]["cam_left_wrist"] = {
        "shape": [480, 640, 3],
        "intrinsic_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "extrinsic_matrix": [
            [0.0, 0.0, 1.0, float(cam2_pos[0])],
            [1.0, 0.0, 0.0, float(cam2_pos[1])],
            [0.0, 1.0, 0.0, float(cam2_pos[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "has_depth": False,
    }
    robo_behind = RobodojoOfficialColourRobot(backend, primitives="no-sam")
    with pytest.raises(ToolError, match="behind"):
        robo_behind.triangulate("cam_head", (240.0, 320.0), "cam_left_wrist", (240.0, 320.0))

    # project_point returns None when world point lies behind camera, and in_bounds=False off-sensor.
    backend.steps[0]["state"]["cameras"] = {}
    behind_head = np.asarray(ray_head.origin) - 0.3 * np.asarray(ray_head.direction)
    assert robo.project_point(behind_head, camera="cam_head") is None
    off_screen = robo.project_point((3.0, 0.1, 0.8), camera="cam_head")
    assert off_screen is not None and off_screen.in_bounds is False


def test_snapshot_provenance_currency_lifecycle_and_contract_compatibility():
    backend, robo = _rgb_model_robot()
    snap0 = robo.snapshot()

    target = (0.05, 0.12, 0.765)
    px_h = robo.project_point(target, "cam_head", observation_id=snap0["observation_id"])
    px_r = robo.project_point(target, "cam_right_wrist", observation_id=snap0["observation_id"])
    assert px_h is not None and px_r is not None

    ray = robo.camera_ray(px_h.row, px_h.col, "cam_head")
    plane_pt = robo.ray_at_z(px_h.row, px_h.col, z=0.765, camera="cam_head")
    tri_pt = robo.triangulate("cam_head", (px_h.row, px_h.col), "cam_right_wrist", (px_r.row, px_r.col))

    for res in (ray, plane_pt, tri_pt):
        assert res.provenance["observation_id"] == snap0["observation_id"]
        assert res.provenance["frozen"] is True
        assert robo.perception_is_current(res) is True
        assert robo.require_current(res) is res

    ray_rec = next(r for r in robo._ledger.records if r.tool == "camera_ray")
    assert "origin" in ray_rec.summary and "direction" in ray_rec.summary
    tri_rec = next(r for r in robo._ledger.records if r.tool == "triangulate")
    assert "depths_m" in tri_rec.summary and "cameras" in tri_rec.summary and "pixels" in tri_rec.summary

    # Compatibility with monitor.from_pose and ActionContract keep_out envelopes.
    elem_plane = monitor_mod.from_pose("plane_target", plane_pt)
    elem_tri = monitor_mod.from_pose("tri_target", tri_pt)
    assert elem_plane.is_current(robo) is True
    assert elem_tri.is_current(robo) is True

    contract = ActionContract(robo)
    summary = contract.configure({"constraints": {"keep_out": [plane_pt, tri_pt]}})
    assert len(summary["constraints"]["keep_out"]) == 2

    # Executing a stateful action increments _action_revision and invalidates currency.
    robo.set_gripper("right", robo.OPEN)
    for res in (ray, plane_pt, tri_pt):
        assert robo.perception_is_current(res) is False
        with pytest.raises(ToolError, match="stale"):
            robo.require_current(res)
    assert elem_plane.is_current(robo) is False
    assert elem_tri.is_current(robo) is False

    # Historical observation_id still resolves the exact historical wrist pose consistently.
    backend.steps[-1]["state"]["image_state"]["right_ee_pose"][0] += 0.20
    robo.snapshot()
    hist_tri = robo.triangulate(
        "cam_head",
        (px_h.row, px_h.col),
        "cam_right_wrist",
        (px_r.row, px_r.col),
        observation_id=snap0["observation_id"],
    )
    assert hist_tri.xyz == pytest.approx(tri_pt.xyz, abs=1e-6)
    assert robo.perception_is_current(hist_tri) is False


def test_crop_grid_renders_without_cv2_and_validates_bounds(monkeypatch):
    _, robo = _rgb_model_robot()
    # Simulate minimal environment without cv2 installed.
    monkeypatch.setitem(sys.modules, "cv2", None)

    full = robo.crop_grid("cam_head", step_px=40, scale=1)
    assert isinstance(full, np.ndarray)
    assert full.dtype == np.uint8
    assert full.shape == (26 + 480, 36 + 640, 3)

    base_crop = robo.crop_grid("cam_head", box=(100, 150, 160, 230), step_px=20, scale=2)
    marked_crop = robo.crop_grid(
        "cam_head",
        box=(100, 150, 160, 230),
        step_px=20,
        points=[(125.0, 185.0), (140.0, 210.0)],
        scale=2,
    )
    assert base_crop.shape == (26 + 60 * 2, 36 + 80 * 2, 3)
    assert marked_crop.shape == base_crop.shape
    assert marked_crop.dtype == np.uint8
    # Drawing crosshair markers changes pixels around the target point inside the crop region.
    assert not np.array_equal(marked_crop[26:, 36:], base_crop[26:, 36:])
    # Exclusive upper boundary points (r1, c1) are omitted, not clamped onto the crop edge.
    edge_crop = robo.crop_grid(
        "cam_head", box=(100, 150, 160, 230), step_px=20, points=[(160.0, 200.0), (120.0, 230.0)], scale=2
    )
    assert np.array_equal(edge_crop, base_crop)
    # Odd scale factors (scale=1 and scale=3) with odd step_px=5 produce exact uniform tick spacing.
    for odd_scale in (1, 3):
        odd_crop = robo.crop_grid("cam_head", box=(0, 0, 20, 20), step_px=5, scale=odd_scale)
        tick_rows = np.where(odd_crop[26:, 33, 0] == 210)[0].tolist()
        expected_rows = [r * odd_scale + (odd_scale // 2) for r in (0, 5, 10, 15)]
        assert tick_rows == expected_rows

    for bad_box in ((-1, 0, 50, 50), (50, 0, 50, 50), (0, 60, 50, 50), (0, 0, 481, 100), (0, 0, 100, 641), (0, 0, 50)):
        with pytest.raises(ValueError):
            robo.crop_grid("cam_head", box=bad_box)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scale"):
        robo.crop_grid("cam_head", scale=0)
    with pytest.raises(ValueError, match="step_px"):
        robo.crop_grid("cam_head", step_px=4)


def test_no_sam_primitive_set_refusal_and_host_sam3_bypass(monkeypatch, tmp_path):
    assert PRIMITIVE_SETS == ("full", "no-vla", "no-sam")
    assert effective_primitives(RobodojoRobot, "no-sam") == "no-sam"
    assert effective_primitives(RobodojoOfficialRobot, "no-sam") == "no-sam"
    assert effective_primitives(RobodojoOfficialColourRobot, "no-sam") == "no-sam"
    assert effective_primitives(RobodojoOfficialColourRobot, "full") == "no-vla"

    backend, robo = _rgb_model_robot(primitives="no-sam")
    backend.calls.clear()
    for hidden_sam in ("locate", "segment", "back_project"):
        expected = NO_SAM_MESSAGE.format(name=hidden_sam)
        with pytest.raises(RuntimeError, match=re.escape(expected)):
            getattr(robo, hidden_sam)("any")
    with pytest.raises(RuntimeError, match="pi0_pick is not available in this run"):
        robo.pi0_pick()
    assert backend.calls == []

    # Host SAM3 bypass: xpolicylab_robodojo.boot with primitives="no-sam" never loads SAM3.
    sam_calls: list[object] = []
    monkeypatch.setattr(sam3_local, "shared", lambda **kw: sam_calls.append(kw))

    bridge = xb.Bridge(poll_s=0.05)
    bridge.update_obs(
        {
            "state": {
                "left_ee_pose": [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0],
                "right_ee_pose": [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0],
                "left_ee_joint_state": [1.0],
                "right_ee_joint_state": [1.0],
                "left_arm_joint_state": [0.0] * 6,
                "right_arm_joint_state": [0.0] * 6,
            },
            "vision": {
                cam: {
                    "color": np.zeros((480, 640, 3), dtype=np.uint8),
                    "shape": [480, 640, 3],
                }
                for cam in RobodojoRobot.CAMERAS
            },
            "instruction": "Move the block",
            "additional_info": {"pyrualean": {"step": 0, "max_steps": 1500, "over": False}},
            "pyrualean_observation": {
                "protocol": 1,
                "source_session_id": "test-src",
                "source_id": "test-src:1",
                "sequence": 1,
                "captured_at": 100.0,
                "capture_clock": "monotonic",
                "source_freshness": "captured",
                "alignment": {"status": "aligned", "basis": "synchronized_capture"},
            },
        }
    )
    cell = xpolicylab_robodojo.Cell(suite="arx_x5", task="swap_blocks", seed=0, cuda_device=0)
    toolkit, daemons = xpolicylab_robodojo.boot(
        cell,
        tmp_path / "episode",
        bridge=bridge,
        depth=False,
        primitives="no-sam",
    )
    assert daemons == []
    assert toolkit._primitives.sam3_client is None
    assert sam_calls == []
    live_robo = RobodojoOfficialColourRobot(
        xpolicylab_robodojo.make_backend(toolkit), primitives="no-sam"
    )
    live_pt = live_robo.ray_at_z(240.0, 320.0, z=0.765, camera="cam_head")
    assert isinstance(live_pt, PlanePoint)
    assert live_robo.perception_is_current(live_pt) is True

    # Verify CLI --help advertises no-sam ("no external VLA or SAM3 service").
    capsys_out: list[str] = []
    monkeypatch.setattr(sys.stdout, "write", lambda s: capsys_out.append(s))
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["play", "--help"])
    assert exc_info.value.code == 0
    assert "no external VLA or SAM3 service" in " ".join("".join(capsys_out).split())

    # Verify knowledge_text and render_prompt under no-sam have zero SAM3/VLA mentions.
    card = TaskCard(
        suite="arx_x5",
        task="swap_blocks",
        seed=0,
        task_language="Swap the blocks",
        object_names=[],
        robot="robodojo_xpl_rgb",
    )
    kt = knowledge_text("robodojo_xpl_rgb", primitives="no-sam")
    assert "crop_grid" in kt and "triangulate" in kt and "ray_at_z" in kt
    assert "SAM3" not in kt and "robo.locate" not in kt
    kt_xpl = knowledge_text("robodojo_xpl", primitives="no-sam")
    assert "crop_grid" in kt_xpl and "SAM3" not in kt_xpl
    prompt = build_prompt("cells", card, budget_s=900, max_programs=None, primitives="no-sam")
    assert "robo.triangulate(" in prompt and "robo.crop_grid(" in prompt
    assert "robo.locate(" not in prompt and "SAM3" not in prompt
    _ = render_prompt(card, budget_s=900, primitives="no-sam")
