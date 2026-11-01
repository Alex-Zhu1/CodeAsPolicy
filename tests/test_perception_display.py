"""Printing task geometry must not exhaust stdout on repeated provenance."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from pyrualean import ToolError
from pyrualean.robodojo_xpl_rgb import ObjectSight, RobodojoOfficialColourRobot
from tests.test_perception_api import CapturedBackend


def test_recorded_scene_display_fits_output_without_dropping_geometry_or_source():
    records = json.loads((Path(__file__).with_name("fixtures") /
                          "organize_table_locates.json").read_text())["cells"]["1"]
    sightings = []
    for record in [*records, records[0]]:
        s = record["summary"]
        sightings.append(ObjectSight(
            found=s["found"], center_xyz=s.get("center_xyz"), top_z=s.get("top_z"),
            height_m=0.02, length_m=0.1, width_m=s.get("width_m"), long_axis_deg=0.,
            grasp_yaw=s.get("grasp_yaw"), score=s.get("score"), centroid=s.get("centroid_rc"),
            area_px=100, camera="cam_head", note="height inferred; check intended object",
            views=s.get("views", 0), provenance=deepcopy(s["provenance"]),
        ))
    before = [deepcopy(item.provenance) for item in sightings]
    stdout = "\n".join(map(repr, sightings))
    assert len(stdout) < 8000  # The real cell stream limit that truncated the old repr.
    assert stdout.count("center_xyz=") == 7
    assert stdout.count("height inferred; check intended object") == 7
    for item in sightings:
        assert item.provenance["observation_id"] in repr(item)
        assert "current_at_return" in repr(item)
        if item.provenance.get("geometry"):
            assert item.provenance["geometry"]["assumptions"]
    assert [item.provenance for item in sightings] == before
    assert "assumptions" not in stdout  # Still accessible explicitly, never mutated.


def test_compact_repr_does_not_refresh_or_weaken_perception_checks():
    robot = RobodojoOfficialColourRobot(CapturedBackend(), primitives="no-vla")
    target = robot.locate("the green bottle")
    before = deepcopy(target.provenance)
    repr(target)
    assert target.provenance == before
    assert robot.require_current(target) is target
    robot.snapshot(refresh=True)
    repr(target)
    assert target.provenance == before
    with pytest.raises(ToolError, match="stale"):
        robot.require_current(target)
