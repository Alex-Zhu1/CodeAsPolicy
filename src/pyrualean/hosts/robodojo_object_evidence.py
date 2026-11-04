"""Observed object landmarks, without table-height or rigid-grasp assumptions.

Correspondences are declared by the planner, or estimated by checked optical
flow. Neither text segmentation nor a reference ID certifies object identity.
Only recorded images/calibration/robot state are read; no simulator object state.
"""

from __future__ import annotations

import itertools
import math
from numbers import Real

import numpy as np

from . import robodojo_features as feature_tools
from . import robodojo_tools as tools


def _pixels(landmarks, obs):
    if not isinstance(landmarks, dict) or len(landmarks) > 8:
        raise ValueError("landmarks must map at most eight feature names to camera pixels")
    result = {}
    for feature, cameras in landmarks.items():
        if not isinstance(feature, str) or not 1 <= len(feature) <= 80:
            raise ValueError("feature names must be nonempty text of at most 80 characters")
        if not isinstance(cameras, dict) or not 1 <= len(cameras) <= 3:
            raise ValueError("each feature needs pixels in one to three cameras")
        result[feature] = {}
        for camera, pixel in cameras.items():
            cam = (obs.get("vision") or {}).get(camera, {})
            image = cam.get("color")
            if image is None:
                raise ValueError(f"no recorded image for {camera}")
            if (not isinstance(pixel, (list, tuple)) or len(pixel) != 2
                    or any(not isinstance(x, Real) or isinstance(x, bool)
                           or not math.isfinite(x) for x in pixel)):
                raise ValueError("pixels must be finite [row, col] numbers")
            if not all(0 <= x < size for x, size in zip(pixel, image.shape[:2], strict=True)):
                raise ValueError("pixel outside recorded image")
            result[feature][camera] = [float(x) for x in pixel]
    return result


def _track(previous, old, obs):
    """Track named pixels with forward/backward and appearance checks; fail closed."""
    try:
        import cv2
    except ImportError:
        return {}, {"tracking": "opencv_unavailable"}
    points, reasons = {}, {}
    for name, feature in previous.get("landmarks", {}).items():
        for camera, pixel in feature.get("pixels", {}).items():
            key = f"{name}/{camera}"
            before = (old.get("vision", {}).get(camera) or {}).get("color")
            after = (obs.get("vision", {}).get(camera) or {}).get("color")
            if before is None or after is None or before.shape != after.shape:
                reasons[key] = "camera_unavailable_or_changed"
                continue
            if (before.ndim != 3 or before.shape[2] < 3 or before.dtype != np.uint8
                    or after.dtype != np.uint8):
                reasons[key] = "unsupported_image_format"
                continue
            a = cv2.cvtColor(np.asarray(before)[..., :3], cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(np.asarray(after)[..., :3], cv2.COLOR_RGB2GRAY)
            row, col = map(int, pixel)
            patch = a[max(0, row - 5):row + 6, max(0, col - 5):col + 6]
            if patch.size < 64 or float(patch.std()) < 5.:
                reasons[key] = "insufficient_texture"
                continue
            start = np.array([[[pixel[1], pixel[0]]]], np.float32)
            end, ok, error = cv2.calcOpticalFlowPyrLK(a, b, start, None,
                                                   winSize=(21, 21), maxLevel=3)
            if end is None or not ok[0, 0] or not np.isfinite(end).all():
                reasons[key] = "tracking_lost"
                continue
            back, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, end, None,
                                                     winSize=(21, 21), maxLevel=3)
            c, r = map(float, end[0, 0])
            if (back is None or not back_ok[0, 0] or not np.isfinite(back).all()
                    or np.linalg.norm(back - start) > 1.5 or float(error[0, 0]) > 15.
                    or not (0 <= r < b.shape[0] and 0 <= c < b.shape[1])):
                reasons[key] = "tracking_inconsistent"
                continue
            points.setdefault(name, {})[camera] = [r, c]
    return points, reasons


def _triangulate(primitives, obs, pixels, pixel_error):
    rays = []
    for camera, pixel in pixels.items():
        view = tools._perception_view(primitives, obs, camera)
        if view is None:
            continue
        if (not np.isfinite(view.rot).all() or not np.isfinite(view.pos).all()
                or not np.allclose(view.rot.T @ view.rot, np.eye(3), atol=1e-3)
                or not math.isclose(float(np.linalg.det(view.rot)), 1., abs_tol=1e-3)):
            continue  # a finite 4x4 array alone is not a calibrated camera pose
        direction = view.rot @ np.array([(pixel[1] - view.cx) / view.fx,
                                         -(pixel[0] - view.cy) / view.fy, -1.])
        direction /= np.linalg.norm(direction)
        rays.append((view, direction, pixel))
    if len(rays) < 2:
        return dict(xyz=None, reason="need_two_calibrated_views", uncertainty_m=None)
    angle = max(math.acos(float(np.clip(abs(a[1] @ b[1]), 0., 1.)))
                for a, b in itertools.combinations(rays, 2))
    if angle < math.radians(5):
        return dict(xyz=None, reason="insufficient_parallax", uncertainty_m=None)
    projections = [np.eye(3) - np.outer(d, d) for _, d, _ in rays]
    try:
        point = np.linalg.solve(sum(projections), sum(
            p @ view.pos for p, (view, _, _) in zip(projections, rays, strict=True)))
    except np.linalg.LinAlgError:
        return dict(xyz=None, reason="degenerate_camera_geometry", uncertainty_m=None)
    residuals, depths = [], []
    for view, _, pixel in rays:
        row, col, depth = view.project(point)
        residuals.append(float(np.linalg.norm(np.array([row, col]) - pixel)))
        depths.append(float(depth))
    if (not np.isfinite(point).all() or not np.isfinite(depths + residuals).all()
            or min(depths) <= 0 or max(residuals) > 3.):
        return dict(xyz=None, reason="inconsistent_correspondence", uncertainty_m=None,
                    reprojection_error_px=max(residuals))
    # A diagnostic error scale, not a calibrated confidence interval. Includes
    # pixel association, parallax and a model/calibration floor.
    uncertainty = .002 + max(depths) * (pixel_error + max(residuals)) / (
        min(min(view.fx, view.fy) for view, _, _ in rays) * math.sin(angle))
    return dict(xyz=point.tolist(), reason=None, uncertainty_m=uncertainty,
                reprojection_error_px=max(residuals), parallax_deg=math.degrees(angle))


def _masks(primitives, obs, description):
    masks, views = {}, {}
    sam = getattr(primitives, "sam3_client", None)
    for camera, cam in obs.get("vision", {}).items():
        if cam.get("color") is None:
            continue
        views[camera] = dict(detection="unavailable", identity_verified=False)
        if sam is None:
            continue
        try:
            detected = sam.segment(cam["color"], text_prompt=description, min_score=.3)
        except (RuntimeError, ValueError) as exc:
            views[camera].update(detection="failed", reason=str(exc))
            continue
        if not detected.found:
            views[camera]["detection"] = "not_found_or_occluded"
            continue
        mask = np.asarray(detected.mask, dtype=bool)
        if mask.shape != np.asarray(cam["color"]).shape[:2] or not mask.any():
            views[camera]["detection"] = "invalid_mask"
            continue
        rows, cols = np.nonzero(mask)
        masks[camera] = mask
        views[camera].update(detection="text_candidate",
            score=float(detected.score) if detected.score is not None else None,
            centroid_rc=[float(rows.mean()), float(cols.mean())],
            box_rc=[int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())])
    return masks, views


def observe(primitives, *, description, landmarks=None, previous=None, previous_packet=None,
            anchor=None, anchor_packet=None, capture=False, arm=None, tracked=None, seed=False):
    obs = tools._obs(primitives)
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 300:
        raise ValueError("description must be nonempty text of at most 300 characters")
    if type(capture) is not bool or type(seed) is not bool or arm not in (None, "left", "right"):
        raise ValueError("capture must be bool; arm must be left/right")
    context = primitives._snapshot_context
    association, reasons, details, shapes, views = "declared_correspondence", {}, {}, {}, {}
    # Empty intermediate observations never destroy the independently retained
    # anchor. Reassociation searches image appearance, never a rigid-grasp prediction.
    origin = anchor if anchor is not None else previous
    packet = anchor_packet if anchor is not None else previous_packet
    same_source = origin and origin["provenance"].get("source_id") == context.get("source_id")
    if landmarks is None and capture:
        origin = None
    reseeded, continuity = False, {}
    if landmarks is None and tracked is not None:
        landmarks, reasons, details = tracked['points'], tracked['failures'], tracked['methods']
        continuity = tracked['continuity']
        association = 'continuous_visual_tracking'
    elif landmarks is None and origin and packet is not None:
        if same_source:
            landmarks = {n: f["pixels"] for n, f in origin["landmarks"].items()}
            shapes = {n: f.get("appearance", {}) for n, f in origin["landmarks"].items()}
            association = origin["association"]
        elif anchor is not None:
            landmarks, reasons, details = feature_tools.reassociate(anchor, packet, obs)
            association = "fixed_keyframe_reassociation"
        else:
            landmarks, reasons = _track(origin, packet, obs)
            association = "optical_flow_estimate"
    elif landmarks is None:
        masks, views = _masks(primitives, obs, description)
        found, reasons = feature_tools.capture(obs, masks, arm=arm,
                                              identity=str(context.get("step", 0)))
        landmarks = {n: f["pixels"] for n, f in found.items()}
        shapes = {n: f["appearance"] for n, f in found.items()}
        association = "masked_material_features"
    # Before closure only, new visible views may add independently detected
    # material points. Closure and holding never replace lost IDs by detection.
    if seed and tracked is not None and sum(
            f'cam_{arm}_wrist' in p for p in (landmarks or {}).values()) < 2:
        masks, views = _masks(primitives, obs, description)
        found, capture_reasons = feature_tools.capture(obs, masks, arm=arm,
                                                      identity=str(context.get('step', 0)))
        reasons.update(capture_reasons)
        if found and any(f'cam_{arm}_wrist' in f['pixels'] for f in found.values()):
            landmarks = {n: f['pixels'] for n, f in found.items()}
            shapes = {n: f['appearance'] for n, f in found.items()}
            association, reseeded = 'preclosure_material_features', True
    points = _pixels({} if landmarks is None else landmarks, obs)
    features = {}
    for name, pixels in points.items():
        appearance = feature_tools.appearance(obs, pixels)
        for camera, values in shapes.get(name, {}).items():
            appearance.setdefault(camera, {}).update(values)
        features[name] = dict(pixels=pixels, appearance=appearance, **_triangulate(
            primitives, obs, pixels, 3. if association == "optical_flow_estimate" else 2.))
    if not points and not views and tracked is None:
        _, views = _masks(primitives, obs, description)
    robot = {}
    state = obs.get("state", {})
    for side in tools.ARMS:
        pose = tools._pose_of(side, tools._joints(state, side))
        robot[side] = dict(tcp_matrix=pose.tolist(), opening=tools._opening(state, side),
            command=context.get("gripper_commands", {}).get(side),
            command_epoch=context.get("gripper_command_epochs", {}).get(side))
    for camera, cam in obs.get("vision", {}).items():
        if cam.get("color") is None:
            continue
        item = views.setdefault(camera, dict(detection="not_requested"))
        item["landmarks"] = {n: f["pixels"][camera] for n, f in features.items()
                             if camera in f["pixels"]}
        item["shape"] = list(np.asarray(cam["color"]).shape[:2])
        view = tools._perception_view(primitives, obs, camera)
        for side in tools.ARMS:
            if (view is not None and camera == f"cam_{side}_wrist"
                    and np.isfinite(view.rot).all() and np.isfinite(view.pos).all()
                    and np.allclose(view.rot.T @ view.rot, np.eye(3), atol=1e-3)
                    and math.isclose(float(np.linalg.det(view.rot)), 1., abs_tol=1e-3)):
                world = np.eye(4)
                world[:3, :3], world[:3, 3] = view.rot, view.pos
                item["tcp_from_camera"] = (np.linalg.inv(
                    np.asarray(robot[side]["tcp_matrix"])) @ world).tolist()
                item["intrinsics"] = [float(x) for x in (view.fx, view.fy, view.cx, view.cy)]
    return dict(landmarks=features, views=views, robot=robot, association=association,
                tracking_failures=reasons, association_details=details,
                tracking_reseeded=reseeded, continuity=continuity,
                anchor_observation_id=(anchor or {}).get("object_observation_id"),
                identity_verified=False, object_pose=None,
                limitations=["Correspondence/identity is declared or estimated, not certified.",
                             "No table-height prior, full object pose, force or contact inference.",
                             "Occlusion or failed matching is unknown, not proof of object loss."],
                geometry_provenance=dict(method="observed_landmark_rays", basis="inferred_rgb",
                    depth_is_measured=False, height_prior_used=False, used_cameras=list(views)))
