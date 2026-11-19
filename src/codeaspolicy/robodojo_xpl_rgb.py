# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo under its own evaluation client as that client is configured: no depth image.

``codeaspolicy.robodojo_xpl`` is the RoboDojo robot under RoboDojo's own evaluation client with
the depth image and the camera matrices its simulator can give.  Those are switched off in
the client's configuration (``env_cfg/arx_x5.yml``: ``approximate_depth``, ``depth``,
``intrinsic_matrix``, ``extrinsic_matrix``; RoboDojo PR 2), a policy cannot switch them on
from its side, and no adapter on the leaderboard reads them.  The robot here is the same
robot with perception that needs no more than that default: ``locate`` measures an object
from its masks in the cameras that see it (:mod:`codeaspolicy.hosts.robodojo_sight`),
``back_project`` answers for the object under a pixel, ``camera_meta`` uses recorded camera
matrices when available or derives them from the state captured with the image.  Motion and
the gripper are unchanged.

The host is :mod:`codeaspolicy.hosts.xpolicylab_robodojo_rgb`.  There is no VLA in that host:
capability filtering selects ``no-vla`` automatically.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar

from ._backend import ToolError
from .robodojo import (
    ArmState,
    CameraRay,
    DeltaGrasp,
    FeatureAlignment,
    Gripper,
    Home,
    IKSolution,
    Move,
    Nudge,
    ObjectPose,
    Pick,
    PixelPoint,
    PlanePoint,
    RobodojoRobot,
    Segment,
    State,
    TriangulatedPoint,
    VisualAlign,
    WorldPoint,
    _arm_state,
    _calibration_provenance,
    _camera,
    _floats,
    _tuple3,
)
from .robots import RobotAdapter


@dataclass(frozen=True, repr=False)
class ObjectSight(ObjectPose):
    """Outcome of ``locate`` (RGB-only): ``ObjectPose`` fields plus ``views`` (cameras agreeing on 3D height)."""

    views: int = 1


@dataclass(frozen=True, repr=False)
class SeenPoint(WorldPoint):
    """Outcome of ``back_project`` (RGB-only): ``WorldPoint`` plus ``surface`` (``"object"`` or ``"table"``)."""

    surface: str = "table"


class RobodojoOfficialColourRobot(RobodojoRobot):
    """The RoboDojo dual-arm station: two 6-DoF ARX-X5 arms with parallel grippers at a table.

    World frame: x runs across the table (the LEFT arm is at negative x and
    appears on the LEFT of the head image), y points away from the robot
    towards the far edge of the table, z is up; metres. State and ``Pose``
    orientations are unit quaternions [w, x, y, z]; ``move_to`` uses the
    gripper angles below, not generic XYZ Euler angles. The arms are
    ``"left"`` and ``"right"``
    (``robo.LEFT`` / ``robo.RIGHT``). ``move_to`` names one arm and
    holds the other; ``move_both`` moves both on one native action clock.
    Choose roles, ordering and recovery in your Python program.

    ``move_to`` drives an arm's GRASP POINT, the middle of the finger pads
    (``state().left.tcp_pos``), to a world position, with an orientation given
    in degrees: ``pitch`` is how far the gripper points below the horizontal
    (0 level, 90 straight down), ``yaw`` the heading of the pointing direction
    on the table (0 away from the robot, +90 towards the robot's left, -90
    towards its right) and ``roll`` the turn of the fingers about the pointing
    direction (0: side by side, as at the start).  Pointing straight down, the
    fingers close along the line at ``yaw`` degrees from +x (counter-clockwise
    seen from above): ``locate``'s ``grasp_yaw`` is that line across an
    object's narrow side.  The table
    surface is at z = 0.765; the arms stand on it at x = -0.30 and +0.30,
    y = -0.45.

    Two gripper conventions, do not mix them up.  COMMANDS are signed:
    ``robo.CLOSE`` (1) closes, ``robo.OPEN`` (-1) opens, ``robo.KEEP`` (0)
    retains the last command. ``move_both`` uses 'close'/'open'/'keep'.
    Gripper commands in either motion take effect from its first native action,
    not on arrival; use a separate close/open call when timing requires it.
    MEASUREMENTS (``state().left.gripper``,
    ``Gripper.opening``) are openings: 1.0 = open, 0.0 = closed on nothing.

    Every native action (one step of ``move_to``, ``set_gripper`` or
    ``home``, one VLA action) counts against the episode's
    native-action budget; when it is spent the state reports ``truncated``
    and motion calls raise ``EpisodeFinished``.  Motion calls block until the
    primitive finishes and return a typed result; nothing is queued.

    Perception is from colour images alone; there is no depth image.  A fixed
    head camera (``cam_head``, the whole table) and one wrist camera per arm
    (``cam_left_wrist``, ``cam_right_wrist``), 640x480 each, and where each
    one is, is known from the robot's state.  ``locate`` finds an object by a
    text prompt and measures it in 3D (centre, top, footprint, heading) from
    the cameras that see it: it takes two of them to measure a height.
    ``segment`` / ``back_project`` give a mask centroid and the world point at
    one pixel.  Object poses are never given.

    The frozen Pi0.5 VLA (``pi0_pick``) is a learned dual-arm policy trained on
    demonstrations of this benchmark's tasks.
    """

    NAME: ClassVar[str] = "robodojo_xpl_rgb"
    SUPPORTS_VLA: ClassVar[bool] = False
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Home,
        ObjectSight,
        Pick,
        Nudge,
        DeltaGrasp,
        FeatureAlignment,
        VisualAlign,
        Segment,
        SeenPoint,
        CameraRay,
        PlanePoint,
        TriangulatedPoint,
        PixelPoint,
        IKSolution,
    )
    SAM_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = (
        RobodojoRobot.SAM_REFERENCE_EDITS[0],
        (
            "``locate`` finds an object by a\n"
            "    text prompt and measures it in 3D (centre, top, footprint, heading) from\n"
            "    the cameras that see it: it takes two of them to measure a height.\n"
            "    ``segment`` / ``back_project`` give a mask centroid and the world point at\n"
            "    one pixel.",
            "``camera_meta`` derives calibrated intrinsics and extrinsics; inspect\n"
            "    ``image`` / ``crop_grid`` to find ``(row, col)`` features, then map them\n"
            "    into 3D with ``ray_at_z`` on a known plane or ``triangulate`` /\n"
            "    ``observe_object`` across two views: it takes two views to measure an\n"
            "    unknown height.",
        ),
        *RobodojoRobot.SAM_REFERENCE_EDITS[2:],
    )
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (*RobodojoRobot.SUMMARY_KEYS, "views", "surface")

    def _reported_success(self) -> bool:
        # RoboDojo's client ends the episode itself the moment the task is solved, and reports
        # that verdict; by then no action is taken any more, so the hosts may stop there too
        return bool(self._backend.solved())

    def camera_meta(
        self, camera: str = "cam_head", *, step: int = -1,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        """Return camera calibration dict (``shape``, ``intrinsic_matrix``, ``extrinsic_matrix``, ``provenance``), derived from image-time state when unrecorded."""
        import numpy as np

        from .hosts import robodojo_sight as sight

        camera = _camera(camera, self.CAMERAS)
        snapshot, step = self._camera_snapshot(step, observation_id)
        if snapshot is not None:
            cameras = snapshot.get("cameras") or (snapshot.get("state") or {}).get("cameras") or {}
            calibration = cameras.get(camera) or {}
            intrinsic, extrinsic = (calibration.get("intrinsic_matrix"),
                                    calibration.get("extrinsic_matrix"))
            if intrinsic is not None or extrinsic is not None:
                try:
                    k, t = np.asarray(intrinsic, dtype=float), np.asarray(extrinsic, dtype=float)
                    valid = (k.shape == (3, 3) and t.shape == (4, 4)
                             and np.isfinite(k).all() and np.isfinite(t).all()
                             and k[0, 0] > 0 and k[1, 1] > 0)
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ToolError("camera_meta", "Snapshot has incomplete or invalid calibration")
                result = {
                    key: deepcopy(calibration[key])
                    for key in ("shape", "intrinsic_matrix", "extrinsic_matrix")
                    if key in calibration
                }
                result["provenance"] = _calibration_provenance(snapshot, camera)
                return result
        if camera == "cam_head":
            view = sight.head_view()
        else:
            name = "left" if camera == "cam_left_wrist" else "right"
            if snapshot is None:
                arm = self.state(step).arm(name)
            else:
                state = snapshot.get("state") or {}
                image_state = snapshot.get("image_state")
                if image_state is None:
                    image_state = state.get("image_state")
                if not isinstance(image_state, dict):
                    raise ToolError("camera_meta", "Snapshot has no image-time arm pose")
                try:
                    _floats(image_state.get(f"{name}_ee_pose"), 7, "image-time arm pose")
                except ValueError as exc:
                    raise ToolError(
                        "camera_meta", "Snapshot has no valid image-time arm pose"
                    ) from exc
                arm = _arm_state(image_state, name)
            view = sight.wrist_view([*arm.eef_pos, *arm.eef_quat])
        k = [[view.fx, 0.0, view.cx], [0.0, view.fy, view.cy], [0.0, 0.0, 1.0]]
        result = {
            "shape": [view.shape[0], view.shape[1], 3],
            "intrinsic_matrix": k,
            "extrinsic_matrix": [[float(v) for v in row] for row in view.pose],
        }
        if snapshot is not None:
            provenance = _calibration_provenance(snapshot, camera)
            image = (snapshot.get("images") or {}).get(camera) or {}
            model = ("robodojo_sight.head_view:v1" if camera == "cam_head"
                     else "robodojo_sight.wrist_view:v1")
            provenance.update(
                calibration_method="robot_model", model=model,
                calibration_source_id=(model if camera == "cam_head"
                                       else image.get("image_state_source_id")),
            )
            result["provenance"] = provenance
        return result

    def back_project(
        self, row: int, col: int, camera: str = "cam_head", *,
        observation_id: str | None = None,
    ) -> SeenPoint:
        """Return world ``xyz`` at pixel ``(row, col)`` (RGB-only; SAM3 segments the object under the pixel or intersects table ``z=0.765``)."""
        camera = _camera(camera, self.CAMERAS)
        kwargs: dict[str, Any] = {"row": int(row), "col": int(col), "camera": camera}
        p = self._perception_call("back_project", kwargs, observation_id=observation_id)
        pixel = p.get("pixel") or (int(row), int(col))
        return SeenPoint(
            pixel=(int(pixel[0]), int(pixel[1])),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
            surface=str(p.get("surface") or "table"),
            provenance=deepcopy(p.get("provenance")),
        )

    def locate(
        self,
        text_prompt: str,
        camera: str = "cam_head",
        min_score: float = 0.2,
        *,
        observation_id: str | None = None,
    ) -> ObjectSight:
        """Segment ``text_prompt`` (SAM3) and measure 3D geometry from colour views (there is no depth image; check ``views`` and ``note``; ``views >= 2`` measures height accurately)."""
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        p = self._perception_call("locate", kwargs, observation_id=observation_id)
        centroid = p.get("centroid_rc") or ()

        def num(key: str) -> float | None:
            return float(p[key]) if p.get(key) is not None else None

        center = p.get("center_xyz")
        note = p.get("geometry_error") or p.get("height_guess")
        return ObjectSight(
            found=bool(p.get("found")),
            center_xyz=_tuple3(center) if center is not None else None,
            top_z=num("top_z"),
            height_m=num("height_m"),
            length_m=num("length_m"),
            width_m=num("width_m"),
            long_axis_deg=num("long_axis_deg"),
            grasp_yaw=num("grasp_yaw"),
            score=num("score"),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            camera=str(p.get("camera", camera)),
            note=str(note) if note else None,
            views=int(p.get("views") or 1),
            provenance=deepcopy(p.get("provenance")),
        )


ADAPTER = RobotAdapter(
    name="robodojo_xpl_rgb",
    robot_cls=RobodojoOfficialColourRobot,
    host=f"{__package__ or 'codeaspolicy'}.hosts.xpolicylab_robodojo_rgb",
    knowledge="robodojo_xpl_rgb",
    blurb="a dual-arm ARX-X5 robot in the RoboDojo benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
    services_nosam="",
)


__all__ = [
    "ADAPTER",
    "CameraRay",
    "DeltaGrasp",
    "FeatureAlignment",
    "IKSolution",
    "ObjectSight",
    "PixelPoint",
    "PlanePoint",
    "RobodojoOfficialColourRobot",
    "SeenPoint",
    "TriangulatedPoint",
]
