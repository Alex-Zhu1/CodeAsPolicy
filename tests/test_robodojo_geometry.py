"""Swept, explicitly supplied geometry; no simulator state or task geometry."""

import copy
import json
import math

import numpy as np
import pytest

from pyrualean.hosts.robodojo_geometry import (
    geometry_coverage, geometry_violation, validate_geometry,
)
from pyrualean.monitor import proactive_constraints


def pose(xyz=(0., 0., 0.), yaw=0.):
    result = np.eye(4)
    angle = math.radians(yaw)
    cosine, sine = math.cos(angle), math.sin(angle)
    result[:3, :3] = [[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]]
    result[:3, 3] = xyz
    return result


def sphere(name="pad", xyz=(0., 0., 0.), radius=.01, **kwargs):
    return dict(name=name, xyz=[float(v) for v in xyz], radius_m=radius, **kwargs)


def constraints(arms=None, obstacles=(), **kwargs):
    return dict(geometry=dict(arms=arms or {"left": [sphere()]}, **kwargs),
                keep_out=list(obstacles))


def obstacle(xyz=(0., 0., 0.), radius=.01, **kwargs):
    return dict(xyz=[float(v) for v in xyz], radius_m=radius, **kwargs)


@pytest.mark.parametrize("xy_only", [False, True])
def test_pure_tcp_rotation_detects_arc_collision_between_endpoints_and_midpoint(xy_only):
    # The endpoints, midpoint, and TCP chord all miss. An attached sphere hits
    # this obstacle at 60 degrees, requiring the curved sweep to be examined.
    angle = math.pi / 3.
    center = [.1 * math.cos(angle), .1 * math.sin(angle), 5. if xy_only else 0.]
    specs = constraints({"left": [sphere(xyz=(.1, 0., 0.))]},
                        [obstacle(center, xy_only=xy_only)])
    previous, current = {"left": pose()}, {"left": pose(yaw=180.)}
    assert geometry_violation(previous, specs) is None
    assert geometry_violation(current, specs) is None
    assert geometry_violation({"left": pose(yaw=90.)}, specs) is None

    violation = geometry_violation(current, specs, previous)
    assert violation["metric"] == "geometry_clearance_m"
    assert violation["reason"] == "geometry_collision"
    assert violation["value"] < violation["bound"] == 0.
    assert violation["delta"] > 0
    assert violation["interval"] != [0., 1.]
    assert violation["pair"] == ["keep_out:0", "left:pad"]
    assert violation["coverage"]["scope"] == "caller_supplied_tcp_spheres_only"
    assert violation["coverage"]["full_robot_collision"] is False
    json.dumps(violation, allow_nan=False)


def test_clear_rotating_arc_is_accepted_even_when_its_endpoint_chord_hits_obstacle():
    specs = constraints({"left": [sphere(xyz=(.1, 0., 0.))]},
                        [obstacle(radius=.03)])
    assert geometry_violation({"left": pose(yaw=180.)}, specs,
                              {"left": pose()}) is None


def test_exact_mixed_axis_half_turn_preserves_rotation_axis_signs():
    # At exactly pi the skew part is zero; reconstructing each axis sign from
    # that skew would follow the wrong semicircle and even miss the endpoint.
    end = pose()
    end[:3, :3] = [[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]
    specs = constraints({"left": [sphere(xyz=(.1, 0., 0.))]},
                        [obstacle((0., -.1, 0.))])
    assert geometry_violation({"left": pose()}, specs) is None
    violation = geometry_violation({"left": end}, specs, {"left": pose()})
    assert violation["reason"] == "geometry_collision"


@pytest.mark.parametrize("xy_only", [False, True])
def test_sphere_translation_checks_entire_segment_against_sphere_or_cylinder(xy_only):
    specs = constraints(obstacles=[obstacle((0., 0., 10. if xy_only else 0.),
                                            xy_only=xy_only)])
    previous = {"left": pose((-.3, 0., 0.))}
    current = {"left": pose((.7, 0., 0.))}
    assert geometry_violation(previous, specs) is None
    assert geometry_violation(current, specs) is None
    violation = geometry_violation(current, specs, previous)
    assert violation["reason"] == "geometry_collision"
    assert violation["value"] == pytest.approx(-.02)
    assert violation["chord_error_bound_m"] == 0.


def test_declared_clearance_and_uncertainty_expand_envelope():
    specs = constraints({"left": [sphere(uncertainty_m=.02)]},
                        [obstacle((.06, 0., 0.))], clearance_m=.03)
    violation = geometry_violation({"left": pose()}, specs)
    assert violation["value"] == pytest.approx(.02)
    assert violation["bound"] == .03
    assert violation["delta"] == pytest.approx(.01)
    specs["geometry"]["clearance_m"] = .01
    assert geometry_violation({"left": pose()}, specs) is None


def test_uncertified_narrow_clearance_rejects_conservatively_with_honest_reason():
    # A coarse requested bound may reject a geometrically clear curved path;
    # its diagnostic must not invent an observed collision.
    specs = constraints({"left": [sphere(xyz=(.1, 0., 0.))]},
                        [obstacle(radius=.03)], sweep_tolerance_m=1.)
    violation = geometry_violation({"left": pose(yaw=180.)}, specs,
                                   {"left": pose()})
    assert violation["reason"] == "geometry_clearance_unverified"
    assert violation["value_basis"] == "conservative_lower_bound"
    assert violation["chord_error_bound_m"] > .1
    assert violation["value"] < 0.


def test_two_arms_cross_between_clear_endpoints():
    specs = constraints({"left": [sphere()], "right": [sphere()]})
    previous = {"left": pose((-.3, 0., 0.)), "right": pose((.3, 0., 0.))}
    current = {"left": pose((.3, 0., 0.)), "right": pose((-.3, 0., 0.))}
    assert geometry_violation(previous, specs) is None
    assert geometry_violation(current, specs) is None
    violation = geometry_violation(current, specs, previous)
    assert violation["pair"] == ["left:pad", "right:pad"]
    assert violation["coverage"]["inter_arm_checks"] is True


def test_rotating_envelope_hits_stationary_other_arm():
    specs = constraints({"left": [sphere(xyz=(.1, 0., 0.))], "right": [sphere()]})
    previous = {"left": pose(), "right": pose((0., .1, 0.))}
    current = {"left": pose(yaw=180.), "right": pose((0., .1, 0.))}
    violation = geometry_violation(current, specs, previous)
    assert violation["pair"] == ["left:pad", "right:pad"]
    assert violation["reason"] == "geometry_collision"


def test_contact_permission_is_limited_to_named_pair_and_bounded_penetration():
    specs = constraints({"left": [sphere(radius=.03)], "right": [sphere(radius=.03)]},
                        clearance_m=.01,
                        allowed_contacts=[dict(pair=["right:pad", "left:pad"],
                                               max_penetration_m=.02)])
    current = {"left": pose(), "right": pose((.045, 0., 0.))}
    assert geometry_violation(current, specs) is None
    current["right"] = pose((.03, 0., 0.))
    violation = geometry_violation(current, specs)
    assert violation["bound"] == -.02
    assert violation["value"] == pytest.approx(-.03)

    # Permission for the pad pair does not permit another pair of spheres.
    current["right"] = pose((.045, 0., 0.))
    specs["geometry"]["arms"]["left"].append(sphere("wrist", (.2, 0., 0.), .03))
    specs["geometry"]["arms"]["right"].append(sphere("wrist", (.155, 0., 0.), .03))
    violation = geometry_violation(current, specs)
    assert violation["pair"] == ["left:wrist", "right:wrist"]
    assert violation["bound"] == .01


def test_obstacle_contact_permission_does_not_disable_other_obstacles():
    specs = constraints(obstacles=[obstacle(), obstacle()],
                        allowed_contacts=[dict(pair=["left:pad", "keep_out:0"],
                                               max_penetration_m=.02)])
    violation = geometry_violation({"left": pose()}, specs)
    assert violation["pair"] == ["keep_out:1", "left:pad"]


def test_absent_geometry_keeps_legacy_scope_explicit():
    assert validate_geometry({}) is None
    assert geometry_violation({}, {"keep_out": [obstacle()]}) is None
    coverage = geometry_coverage({"keep_out": [obstacle()]})
    assert coverage["geometry_supplied"] is False
    assert coverage["scope"] == "no_volume_geometry"
    assert coverage["represented_arms"] == []
    assert coverage["obstacle_checks"] is False
    assert coverage["inter_arm_checks"] is False
    assert coverage["full_robot_collision"] is False
    assert coverage["actual_motion_between_observations"] == "not_certified"
    assert geometry_coverage(constraints())["inter_arm_checks"] is False


def test_normalization_is_serializable_does_not_mutate_input_and_preserves_source():
    specs = constraints({"left": [sphere(source="public gripper model")],
                         "right": [sphere()]},
                        allowed_contacts=[dict(pair=["right:pad", "left:pad"],
                                               max_penetration_m=0.)])
    original = copy.deepcopy(specs)
    normalized = validate_geometry(specs)
    assert specs == original
    assert normalized["allowed_contacts"][0]["pair"] == ["left:pad", "right:pad"]
    assert normalized["arms"]["left"][0]["source"] == "public gripper model"
    assert normalized["arms"]["left"][0]["uncertainty_m"] == 0.
    assert normalized["sweep_tolerance_m"] > 0.
    json.dumps(normalized, allow_nan=False)


def test_proactive_constraints_accepts_geometry_after_obstacle_normalization():
    geometry = dict(arms={"left": [sphere()]},
                    allowed_contacts=[dict(pair=["left:pad", "keep_out:0"],
                                           max_penetration_m=0.)])
    result = proactive_constraints(geometry=geometry, keep_out=[obstacle()],
                                   z_bounds=[0., 1.])
    assert result["geometry"] == validate_geometry(dict(geometry=geometry,
                                                        keep_out=[obstacle()]))
    assert result["z_bounds"] == [0., 1.]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("xyz", [float("nan"), 0., 0.]),
    ("xyz", [0., float("inf"), 0.]),
    ("xyz", [True, 0., 0.]),
    ("radius_m", float("nan")),
    ("radius_m", float("inf")),
    ("radius_m", 0.),
    ("radius_m", True),
    ("uncertainty_m", float("nan")),
    ("uncertainty_m", -.01),
    ("name", ""),
    ("name", "left:pad"),
    ("source", ""),
    ("unknown", 1.),
])
def test_invalid_sphere_geometry_is_rejected(field, value):
    specs = constraints()
    specs["geometry"]["arms"]["left"][0][field] = value
    with pytest.raises(ValueError):
        validate_geometry(specs)


@pytest.mark.parametrize("geometry", [
    None, {}, {"arms": {}}, {"arms": {"front": [sphere()]}},
    {"arms": {"left": []}}, {"arms": {"left": [sphere(), sphere()]}},
    {"arms": {"left": [sphere()]}, "clearance_m": float("nan")},
    {"arms": {"left": [sphere()]}, "clearance_m": -.1},
    {"arms": {"left": [sphere()]}, "sweep_tolerance_m": 0.},
    {"arms": {"left": [sphere()]}, "sweep_tolerance_m": float("inf")},
    {"arms": {"left": [sphere()]}, "allow_all_contacts": True},
])
def test_invalid_geometry_configuration_is_rejected(geometry):
    with pytest.raises(ValueError):
        validate_geometry(dict(geometry=geometry))


@pytest.mark.parametrize("contacts", [
    [dict(pair=["left:pad", "right:missing"], max_penetration_m=0.)],
    [dict(pair=["left:pad", "left:pad"], max_penetration_m=0.)],
    [dict(pair=["left:pad", "right:pad"], max_penetration_m=float("nan"))],
    [dict(pair=["left:pad", "right:pad"], max_penetration_m=.03)],
    [dict(pair=["left:pad", "right:pad"])],
    [dict(pair=["left:pad", "right:pad"], max_penetration_m=0., disabled=True)],
    [dict(pair=["left:pad", "right:pad"], max_penetration_m=0.),
     dict(pair=["right:pad", "left:pad"], max_penetration_m=0.)],
])
def test_invalid_or_unbounded_contact_permissions_are_rejected(contacts):
    specs = constraints({"left": [sphere()], "right": [sphere()]},
                        allowed_contacts=contacts)
    with pytest.raises(ValueError):
        validate_geometry(specs)


@pytest.mark.parametrize("bad", [np.full((4, 4), float("nan")), np.eye(3),
                                  np.diag([1., 1., -1., 1.]), np.eye(4) * 2.])
def test_invalid_public_pose_is_rejected(bad):
    with pytest.raises(ValueError, match="rigid 4x4"):
        geometry_violation({"left": bad}, constraints())


def test_declared_geometry_requires_pose_for_every_represented_arm():
    specs = constraints({"left": [sphere()], "right": [sphere()]})
    with pytest.raises(ValueError, match="right poses"):
        geometry_violation({"left": pose()}, specs)
    with pytest.raises(ValueError, match="right poses"):
        geometry_violation({"left": pose(), "right": pose()}, specs, {"left": pose()})


def test_public_preview_rejects_rotation_only_volume_crossing_with_clear_endpoints(tmp_path):
    from pyrualean.coordination import Pose
    from pyrualean.hosts import robodojo_kinematics as kin
    from pyrualean.hosts import robodojo_tools as motion
    from tests.test_robodojo_intent import _precision_station

    env, robo, _ = _precision_station(tmp_path)
    start = motion._pose_of("right", env.joints["right"])
    end_joints = list(env.joints["right"])
    end_joints[5] += math.radians(120.)
    end = motion._pose_of("right", end_joints)
    assert end[:3, 3] == pytest.approx(start[:3, 3], abs=1e-12)
    local = np.array([0., .08, 0.])
    middle = start[:3, 3] + kin.slerp(start[:3, :3], end[:3, :3], .5) @ local
    specs = constraints({"right": [sphere(xyz=tuple(local))]},
                        [obstacle(tuple(middle))])
    assert geometry_violation({"right": start}, specs) is None
    assert geometry_violation({"right": end}, specs) is None
    result = robo.execute_motion(
        "rotation-preview", right=Pose(tuple(end[:3, 3]), kin.quat_wxyz(end[:3, :3])),
        holding={"right": "empty"}, preview=True, constraints=specs,
    )
    assert result["status"] == "rejected", result
    assert result["stop_reason"] == "geometry_constraint"
    assert result["constraint_violation"]["pair"] == ["keep_out:0", "right:pad"]
    assert result["steps_used"] == 0 and not env.actions
    assert robo.motion_status("rotation-preview")["stop_reason"] == "unknown_intent_in_this_session"


def test_public_preview_checks_uncommanded_arm_volume_and_accepts_clear_sequence(tmp_path):
    from pyrualean.hosts import robodojo_tools as motion
    from tests.test_robodojo_intent import _precision_station

    env, robo, _ = _precision_station(tmp_path)
    measured = robo.observe()
    left = motion._pose_of("left", env.joints["left"])
    right = motion._pose_of("right", env.joints["right"])
    common = left[:3, 3] + [0., 0., .03]
    offset = right[:3, :3].T @ (common - right[:3, 3])
    specs = constraints({"left": [sphere()], "right": [sphere(xyz=tuple(offset))]})
    previous = {"left": left, "right": right}
    final = {"left": left.copy(), "right": right}
    final["left"][2, 3] += .06
    assert geometry_violation(previous, specs) is None
    assert geometry_violation(final, specs) is None
    result = robo.execute_motion("both-preview", left=measured.left.shifted(dz=.06),
                                 holding={"left": "empty"}, preview=True, constraints=specs)
    assert result["stop_reason"] == "geometry_constraint", result
    assert result["constraint_violation"]["pair"] == ["left:pad", "right:pad"]
    assert result["geometry_coverage"]["inter_arm_checks"] is True
    specs["geometry"]["arms"]["right"][0]["xyz"][0] += .2
    clear = robo.execute_motion("both-clear", left=measured.left.shifted(dz=.06),
                                holding={"left": "empty"}, preview=True, constraints=specs)
    assert clear["accepted"] and clear["status"] == "planned", clear
    assert not env.actions


def test_live_cross_arm_guard_preserves_metrics_on_first_measured_departure(tmp_path):
    from pyrualean.hosts import robodojo_tools as motion
    from tests.test_robodojo_intent import _offset_measured_pose, _precision_station

    env, robo, _ = _precision_station(tmp_path)
    measured = robo.observe()
    right = motion._pose_of("right", env.joints["right"])
    left = motion._pose_of("left", env.joints["left"])
    obstacle_center = right[:3, 3] - [0., 0., .02]
    offset = left[:3, :3].T @ (obstacle_center - left[:3, 3])
    specs = constraints({"left": [sphere(xyz=tuple(offset), radius=.004)],
                         "right": [sphere(radius=.004)]})
    step = env.step

    def drift(action):
        result = step(action)
        _offset_measured_pose(env, dz=-.05)
        return result

    env.step = drift
    result = robo.execute_motion("guarded", right=measured.right.shifted(dz=.06),
                                 holding={"right": "empty"}, constraints=specs)
    assert result["stop_reason"] == "geometry_constraint", result
    assert result["steps_used"] == len(env.actions) == 1
    assert result["constraint_violation"]["pair"] == ["left:pad", "right:pad"]
    assert result["constraint_violation"]["delta"] > 0
    assert robo.motion_status("guarded")["constraint_violation"] == result["constraint_violation"]
