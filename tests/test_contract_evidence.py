"""Live hold checks preserve provenance and image-versus-spatial limitations."""

import copy
import json

import pytest

from pyrualean.action_contract import ActionContract, ActionContractPause
from pyrualean.contract_evidence import check_fact, check_hold


def evidence(space="image"):
    result = dict(
        arm="left", state="image_consistent", evidence_space="wrist_image_2d",
        baseline_ready=True, current=True, needs_observation=False,
        provenance={"perception_id": "host-owned-current"},
        gripper={"obstructed_closure": True},
        image_features={name: dict(camera="cam_left_wrist", displacement_px=drift,
                                  comparison_bound_px=4.) for name, drift in (("a", 1.), ("b", 2.))},
        features={},
    )
    if space == "spatial":
        result.update(state="following", evidence_space="3d", image_features={},
                      features={"a": {"relative_change_m": .002},
                                "b": {"relative_change_m": .003}})
    return result


class Robot:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.validated = []
        self.current = True
        self._action_revision = 0

    def observe_hold(self, reference, *, arm):
        self.calls.append((reference, arm))
        return copy.deepcopy(self.result)

    def perception_is_current(self, result):
        self.validated.append(result)
        return self.current and result.get("provenance") == {
            "perception_id": "host-owned-current",
        }


def test_image_consistency_never_satisfies_spatial_requirement():
    robot = Robot(evidence())
    image = check_hold(robot, "ref", "left", space="image")
    spatial = check_hold(robot, "ref", "left", space="spatial")
    assert image.passed and image.metric == "feature_drift_px"
    assert not spatial.passed and "image evidence" in spatial.reason
    assert len(robot.calls) == len(robot.validated) == 2
    json.dumps(image.to_dict(), allow_nan=False)


@pytest.mark.parametrize("space", ["image", "spatial"])
def test_observed_spatial_following_satisfies_both_coverage_levels(space):
    robot = Robot(evidence("spatial"))
    result = check_hold(robot, "ref", "left", space=space)
    assert result.passed
    assert result.metric == "feature_drift_m" and result.value == .003


@pytest.mark.parametrize("change", ["source", "missing_provenance", "forged_provenance"])
def test_current_flag_cannot_replace_host_source_validation(change):
    robot = Robot(evidence())
    if change == "source":
        robot.current = False
    elif change == "missing_provenance":
        robot.result.pop("provenance")
    else:
        robot.result["provenance"]["perception_id"] = "invented"
    result = check_hold(robot, "ref", "left", space="image")
    assert not result.passed and result.reason.startswith("stale:")
    assert robot.validated


@pytest.mark.parametrize("state", ["unknown", "relative_motion", "image_relative_motion",
                                    "conflicting_evidence"])
def test_nonpassing_observations_never_authorize_motion(state):
    robot = Robot(evidence())
    robot.result.update(state=state, reason="public observation did not establish support")
    result = check_hold(robot, "ref", "left", space="image")
    assert not result.passed and result.reason.startswith(state)


def test_check_reobserves_at_each_action_boundary_and_blocks_changed_source():
    robot = Robot(evidence())
    contract = ActionContract(robot)
    contract.configure({"preserves": {
        "visible_support": lambda: check_hold(robot, "ref", "left", space="image"),
    }})
    contract.before_action("move")
    robot._action_revision += 1
    robot.current = False
    with pytest.raises(ActionContractPause) as exc:
        contract.after_action("move")
    assert len(robot.calls) == 2
    assert exc.value.report["failed_assumption"] == "visible_support"
    assert exc.value.report["evidence_state"] == "stale"
    assert "stale:" in exc.value.report["reason"]


@pytest.mark.parametrize("invalid", [None, -1., float("nan"), float("inf"), True])
def test_invalid_spatial_measurements_cannot_pass(invalid):
    robot = Robot(evidence("spatial"))
    robot.result["features"]["a"]["relative_change_m"] = invalid
    result = check_hold(robot, "ref", "left")
    assert not result.passed
    json.dumps(result.to_dict(), allow_nan=False)


def test_spatial_and_image_numeric_bounds_remain_enforced():
    robot = Robot(evidence("spatial"))
    robot.result["features"]["a"]["relative_change_m"] = .02
    result = check_hold(robot, "ref", "left", max_drift_m=.01)
    assert not result.passed and result.value == .02 and result.delta == -.01
    robot.result = evidence()
    robot.result["image_features"]["a"]["displacement_px"] = 5.
    result = check_hold(robot, "ref", "left", space="image")
    assert not result.passed and result.value == 5. and result.delta == -1.


@pytest.mark.parametrize("change", ["baseline", "arm", "gripper", "camera", "features"])
def test_incomplete_or_conflicting_evidence_stays_blocked(change):
    robot = Robot(evidence())
    if change == "baseline":
        robot.result["baseline_ready"] = False
    elif change == "arm":
        robot.result["arm"] = "right"
    elif change == "gripper":
        robot.result["gripper"]["obstructed_closure"] = False
    elif change == "camera":
        robot.result["image_features"]["a"]["camera"] = "cam_head"
    else:
        robot.result["image_features"].pop("a")
    assert not check_hold(robot, "ref", "left", space="image").passed


def test_observation_errors_return_a_diagnostic_instead_of_motion_authority():
    robot = Robot(evidence())

    def unavailable(*args, **kwargs):
        raise RuntimeError("camera unavailable")

    robot.observe_hold = unavailable
    result = check_hold(robot, "ref", "left")
    assert not result.passed and "camera unavailable" in result.reason
    assert result.remedy


def test_hold_bridge_reads_real_grasp_and_snapshot_evidence(tmp_path):
    from tests.test_robodojo_continuous_grasp import visual_station

    env, robot, _, ref, _, _ = visual_station(tmp_path)
    closed = robot.grasp_step("tracked", "close")
    assert closed["object_evidence"]["baseline_ready"]
    assert not check_hold(robot, ref, "right", space="image").passed
    probed = robot.grasp_step("tracked", "probe")
    assert probed["object_evidence"]["state"] == "image_consistent"
    actions = len(env.actions)
    image = check_hold(robot, ref, "right", space="image")
    spatial = check_hold(robot, ref, "right", space="spatial")
    assert image.passed, image
    assert image.metric == "feature_drift_px"
    assert not spatial.passed and "image evidence" in spatial.reason
    assert len(env.actions) == actions  # Reading/comparing evidence cannot create motion.
    recorded = robot._ledger.records[-1].summary
    assert recorded["evidence_space"] == "wrist_image_2d"
    assert robot.perception_is_current(recorded)
    assert recorded["provenance"]["observation_id"] == robot.snapshot()["observation_id"]
    json.dumps(image.to_dict(), allow_nan=False)


@pytest.fixture
def fact_station(tmp_path):
    from pyrualean.supervision import MonitorConfig, VisualSupervisor
    from tests.test_robodojo_trajectory import station
    from tests.test_supervision import ScriptedValen, answer

    env, robot, host = station(tmp_path)
    client = ScriptedValen([answer()])
    supervisor = VisualSupervisor(
        MonitorConfig(mode="pause", url="http://127.0.0.1:1"), tmp_path / "audit", client,
    )
    robot._supervisor = supervisor
    supervisor.watch("inspect", {"held": "Is the object held?"})
    supervisor.verify(robot)
    assert len(client.packets) == 1
    return env, robot, host, supervisor, client


def test_fact_bridge_reuses_actual_verified_ledger_without_extra_vqa(fact_station):
    env, robot, _, supervisor, client = fact_station
    assert check_fact(robot, "held").passed
    assert check_fact(robot, "held", expected="yes").passed
    opposite = check_fact(robot, "held", expected="no")
    assert not opposite.passed and opposite.reason.startswith("visual_fact_mismatch")
    contract = ActionContract(robot, {"requirements": [
        dict(id="preserve_support", kind="invariant", condition="Keep the object supported.",
             source="public instruction"),
    ]})
    contract.configure({"preserves": {"preserve_support": lambda: check_fact(robot, "held")}})
    assert contract.before_action("move")["passed"]
    assert supervisor.ledger.facts["held"]["fresh"]
    assert len(client.packets) == 1 and not env.actions
    json.dumps(contract.summary(), allow_nan=False)


@pytest.mark.parametrize("change", ["action", "source", "pending", "identity", "invalidated", "unknown"])
def test_fact_fresh_flag_cannot_override_current_source_and_action(fact_station, change):
    env, robot, _, supervisor, client = fact_station
    if change == "action":
        robot._action_revision += 1
    elif change == "source":
        robot.snapshot(refresh=True)
    elif change == "pending":
        robot._pending_actions.add("unresolved-action")
    elif change == "identity":
        supervisor.ledger.facts["held"]["observation"]["source_id"] = "not-the-host-source"
    elif change == "invalidated":
        supervisor.ledger.invalidate(None, "action_started")
    else:
        supervisor.ledger.facts["held"]["value"] = "unknown"
    check = check_fact(robot, "held")
    assert not check.passed and check.reason.startswith("unknown:"), check
    assert len(client.packets) == 1 and not env.actions
    json.dumps(check.to_dict(), allow_nan=False)


def test_fact_without_supervisor_or_observed_binding_remains_unknown(fact_station):
    _, robot, _, supervisor, client = fact_station
    missing = check_fact(robot, "undeclared")
    assert not missing.passed and missing.reason.startswith("unknown:")
    supervisor.watch("new", {"new_fact": "Is the destination clear?"})
    unobserved = check_fact(robot, "new_fact")
    assert not unobserved.passed and unobserved.value is None
    robot._supervisor = None
    disabled = check_fact(robot, "held")
    assert not disabled.passed and disabled.reason.startswith("unknown:")
    assert len(client.packets) == 1
