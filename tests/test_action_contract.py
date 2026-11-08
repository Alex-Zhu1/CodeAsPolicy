"""Action admission from explicit checks, independent of optional visual review."""

import json
import sys
from types import SimpleNamespace

import pytest

from pyrualean.action_contract import ActionContract, ActionContractPause
from pyrualean.monitor import ConstraintCheck, check_distance


@pytest.fixture
def robot():
    return SimpleNamespace(_action_revision=0)


def acceptance():
    return {"requirements": [
        {"id": "neighbors", "kind": "invariant", "condition": "Keep neighbors upright.",
         "source": "public instruction"},
        {"id": "placed", "kind": "final_state", "condition": "Place the object on its base.",
         "source": "public instruction"},
    ]}


def test_no_contract_or_public_invariant_preserves_legacy_action_flow(robot):
    contract = ActionContract(robot)
    assert contract.before_action("move")["passed"]
    robot._action_revision += 1
    assert contract.after_action("move")["passed"]
    assert not contract.summary()["configured"]


def test_public_invariant_requires_named_binding_before_first_action(robot):
    contract = ActionContract(robot, acceptance())
    assert contract.summary()["uncovered_invariants"] == ["neighbors"]
    contract.configure({"requires": {"neighbors": lambda: True}})
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("move")
    assert robot._action_revision == 0
    assert exc.value.report["failed_assumption"] == "neighbors"
    assert exc.value.report["evidence_state"] == "missing"
    contract.configure({"preserves": {"neighbors": lambda: True}})
    summary = contract.summary()
    assert not summary["uncovered_invariants"]
    assert summary["declared"]["preserves"] == ["neighbors"]
    assert summary["checked"] == []  # Binding is not an observation or safety verdict.
    assert contract.before_action("move")["passed"]
    assert contract.summary()["checked"][0]["evidence_state"] == "passed"


def test_phase_replacement_preserves_invariants_across_empty_declarations(robot):
    state = {"neighbors": True, "held": True}
    contract = ActionContract(robot, acceptance())
    contract.configure({
        "requires": {"held": lambda: state["held"]},
        "preserves": {"neighbors": lambda: state["neighbors"]},
        "constraints": {"z_bounds": [0.1, 1.0]},
    })
    contract.before_action("carry")
    contract.configure({})
    assert contract.constraints == {}
    assert contract.summary()["requires"] == []
    assert contract.summary()["preserves"] == ["neighbors"]
    state["held"] = None
    assert contract.before_action("independent_work")["passed"]
    state["neighbors"] = False
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("independent_work")
    assert exc.value.report["failed_assumption"] == "neighbors"
    contract.configure({"preserves": {}})
    assert contract.summary()["preserves"] == ["neighbors"]


def test_requires_only_before_preserves_before_and_after(robot):
    calls = []
    state = {"invariant": True}
    contract = ActionContract(robot)

    def required():
        calls.append("requires")
        return True

    def preserved():
        calls.append("preserves")
        return state["invariant"]

    contract.configure({"requires": {"ready": required}, "preserves": {"stable": preserved}})
    contract.before_action("move")
    robot._action_revision += 1
    state["invariant"] = False
    with pytest.raises(ActionContractPause) as exc:
        contract.after_action("move")
    assert calls == ["requires", "preserves", "preserves"]
    assert exc.value.report["stage"] == "after_action"
    assert exc.value.report["failed_assumption"] == "stable"


def test_unknown_blocks_dependent_motion_but_can_be_observed_and_replaced(robot):
    observed = {"held": None}
    contract = ActionContract(robot)
    contract.configure({"requires": {"held": lambda: observed["held"]}})
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("carry")
    assert exc.value.report["evidence_state"] == "unknown"
    # The contract does not latch the whole episode or forbid new observations.
    observed["held"] = True
    assert contract.before_action("carry")["passed"]
    observed["held"] = None
    contract.configure({"requires": {"clear_work_area": lambda: True}})
    assert contract.before_action("independent_work")["passed"]


def test_static_result_expires_with_action_revision_and_can_be_refreshed(robot):
    contract = ActionContract(robot)
    result = check_distance((0, 0, 0), (0, 0, 1), max_m=2)
    contract.configure({"requires": {"clearance": result}})
    assert contract.before_action("move")["passed"]
    robot._action_revision += 1
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("next_move")
    failure = exc.value.report["failures"][0]
    assert failure["evidence_state"] == "stale"
    assert failure["value"] == 1 and failure["bound"] == 0
    contract.configure({"requires": {"clearance": result}})
    with pytest.raises(ActionContractPause):
        contract.before_action("next_move")
    contract.configure({"requires": {
        "clearance": check_distance((0, 0, 0), (0, 0, 1), max_m=2),
    }})
    assert contract.before_action("next_move")["passed"]


def test_static_preservation_cannot_certify_a_changed_endpoint(robot):
    contract = ActionContract(robot)
    contract.configure({"preserves": {"clearance": check_distance((0, 0, 0), (0, 0, 1), max_m=2)}})
    contract.before_action("move")
    robot._action_revision += 1
    with pytest.raises(ActionContractPause) as exc:
        contract.after_action("move")
    assert exc.value.report["evidence_state"] == "stale"
    assert not contract.summary()["checked"][0]["passed"]


@pytest.mark.parametrize("bad_return", [None, "yes", {"passed": True}, 1])
def test_ambiguous_check_returns_never_become_truth(robot, bad_return):
    contract = ActionContract(robot)
    contract.configure({"requires": {"ready": lambda: bad_return}})
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("move")
    assert exc.value.report["evidence_state"] == "unknown"
    json.dumps(contract.summary(), allow_nan=False)


def test_exception_becomes_structured_failure_and_releases_evaluation_guard(robot):
    contract = ActionContract(robot)

    def broken():
        raise ValueError("sensor unavailable")

    contract.configure({"requires": {"observed": broken}})
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("move")
    assert exc.value.report["evidence_state"] == "error"
    assert "sensor unavailable" in exc.value.report["reason"]
    assert not contract.evaluating
    contract.configure({})
    assert contract.before_action("independent_work")["passed"]


def test_metric_failure_preserves_remedy_and_json_feedback(robot):
    contract = ActionContract(robot)
    contract.configure({"requires": {"route_clear": lambda: check_distance(
        (0, 0, 0), (0, 0, 1), min_m=2)}})
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("carry")
    report = exc.value.report
    failure = report["failures"][0]
    assert report["failed_assumption"] == "route_clear"
    assert failure["metric"] == "distance_m"
    assert failure["value"] == 1 and failure["bound"] == 2
    assert failure["delta"] == -1 and failure["remedy"]
    encoded = json.dumps(contract.summary(), allow_nan=False)
    assert "function" not in encoded and "lambda" not in encoded
    report["failures"].clear()
    assert contract.last_report["failures"]


def test_constraints_are_normalized_and_detached(robot):
    contract = ActionContract(robot)
    supplied = {"z_bounds": (0.1, 1), "keep_out": [{"xyz": (1, 2, 3), "radius_m": .1}]}
    contract.configure({"constraints": supplied})
    supplied["keep_out"].clear()
    returned = contract.constraints
    assert returned["z_bounds"] == [.1, 1.]
    assert returned["keep_out"][0]["xyz"] == [1., 2., 3.]
    returned["keep_out"].clear()
    assert contract.constraints["keep_out"]


@pytest.mark.parametrize("bad_spec", [
    None, {"invented": True}, {"requires": [lambda: True]},
    {"preserves": {"": lambda: True}}, {"requires": {"ready": True}},
    {"requires": {"ready": lambda robot: True}}, {"constraints": []},
])
def test_invalid_replacement_cannot_fall_back_to_previous_authority(robot, bad_spec):
    contract = ActionContract(robot)
    contract.configure({"preserves": {"stable": lambda: True}})
    with pytest.raises((ValueError, TypeError)):
        contract.configure(bad_spec)
    assert contract.summary()["preserves"] == ["stable"]
    with pytest.raises(ActionContractPause):
        contract.before_action("move")
    contract.configure({})
    assert contract.before_action("move")["passed"]


@pytest.mark.parametrize("attempt", ["dispatch", "configure", "recursive_check"])
def test_checks_cannot_authorize_side_effects_even_if_they_catch_pause(robot, attempt):
    contract = ActionContract(robot)
    dispatched = []

    def predicate():
        assert contract.evaluating
        try:
            if attempt == "dispatch":
                # This is the RobotBase integration protocol: reject before dispatch.
                contract.reject("move", "reentrant_action", "Keep checks read-only.")
                dispatched.append(True)
            elif attempt == "configure":
                contract.configure({})
            else:
                contract.before_action("move")
        except BaseException:
            pass
        return True

    contract.configure({"preserves": {"stable": predicate}})
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("move")
    assert exc.value.report["failed_assumption"] == "stable"
    assert not dispatched
    assert contract.summary()["preserves"] == ["stable"]
    assert not contract.evaluating


def test_reject_exposes_uniform_report_to_dispatch_integration(robot):
    contract = ActionContract(robot)
    with pytest.raises(ActionContractPause) as exc:
        contract.reject("move", "unsupported_constraint_dispatch", "Use execute_motion.")
    assert exc.value.report == contract.last_report
    assert exc.value.report["reason"] == "unsupported_constraint_dispatch"
    json.dumps(exc.value.report, allow_nan=False)


def test_malformed_constraint_result_cannot_pass_on_truthiness(robot):
    contract = ActionContract(robot)
    result = ConstraintCheck("ambiguous", "reactive", None, "unknown", None, None, None, ())
    contract.configure({"requires": {"ready": lambda: result}})
    with pytest.raises(ActionContractPause) as exc:
        contract.before_action("move")
    assert exc.value.report["evidence_state"] == "unknown"


def test_restart_restores_names_and_geometry_but_never_passed_evidence(robot):
    contract = ActionContract(robot, acceptance())
    contract.configure({
        "requires": {"held": lambda: True},
        "preserves": {"neighbors": lambda: True, "custom_clearance": lambda: True},
        "constraints": {"z_bounds": [.1, 1.]},
    })
    contract.before_action("move")
    saved = json.loads(json.dumps(contract.summary(), allow_nan=False))
    restored = ActionContract(robot, acceptance())
    restored.restore(saved)
    assert restored.summary()["preserves"] == ["neighbors", "custom_clearance"]
    assert restored.summary()["requires"] == ["held"]
    assert restored.summary()["checked"] == []
    assert restored.constraints == {"z_bounds": [.1, 1.]}
    with pytest.raises(ActionContractPause) as exc:
        restored.before_action("move")
    assert exc.value.report["evidence_state"] == "unknown"
    # Omitting a saved custom invariant cannot erase it during rebinding.
    restored.configure({"preserves": {"neighbors": lambda: True}})
    with pytest.raises(ActionContractPause) as exc:
        restored.before_action("move")
    assert exc.value.report["failed_assumption"] == "custom_clearance"
    restored.configure({"preserves": {"custom_clearance": lambda: True}})
    assert restored.before_action("independent_work")["passed"]


def test_restored_geometry_only_contract_needs_explicit_reconfiguration(robot):
    original = ActionContract(robot)
    original.configure({"constraints": {"z_bounds": [.1, 1.]}})
    restored = ActionContract(robot)
    restored.restore(original.summary())
    with pytest.raises(ActionContractPause):
        restored.before_action("move")
    restored.configure({"constraints": {"z_bounds": [.1, 1.]}})
    assert restored.before_action("move")["passed"]


def test_legacy_empty_summary_does_not_activate_contract_on_restart(robot):
    restored = ActionContract(robot)
    restored.restore(ActionContract(robot).summary())
    assert not restored.summary()["configured"]
    assert restored.before_action("move")["passed"]


def test_completion_checks_ensures_only_at_explicit_completion_boundary(robot):
    state = {"placed": False, "held": True, "neighbors": True}
    calls = []
    contract = ActionContract(robot)

    def placed():
        calls.append("ensures")
        return state["placed"]

    contract.configure({"requires": {"held": lambda: state["held"]},
                        "ensures": {"placed": placed},
                        "preserves": {"neighbors": lambda: state["neighbors"]}})
    contract.before_action("move")
    robot._action_revision += 1
    contract.after_action("move")
    assert not calls
    with pytest.raises(ActionContractPause) as exc:
        contract.verify_completion()
    assert exc.value.report["failed_assumption"] == "placed"
    assert exc.value.report["stage"] == "verify_completion"
    state.update(placed=True, held=False)
    assert contract.verify_completion("finish")["passed"]
    state["neighbors"] = False
    with pytest.raises(ActionContractPause) as exc:
        contract.verify_completion("finish")
    assert exc.value.report["failed_assumption"] == "neighbors"


def test_completion_declarations_replace_without_claiming_old_outcome(robot):
    contract = ActionContract(robot)
    contract.configure({"ensures": {"previous_goal": lambda: False}})
    contract.configure({"ensures": {"new_goal": lambda: True}})
    report = contract.verify_completion()
    assert report["passed"]
    assert [check["assumption"] for check in report["checks"]] == ["new_goal"]
    assert "previous_goal" not in json.dumps(contract.summary())


def test_restored_completion_checks_remain_unbound_until_redeclared(robot):
    original = ActionContract(robot)
    original.configure({"ensures": {"placed": lambda: True}})
    original.verify_completion()
    restored = ActionContract(robot)
    restored.restore(original.summary())
    assert restored.summary()["ensures"] == ["placed"]
    with pytest.raises(ActionContractPause):
        restored.verify_completion()
    restored.configure({"ensures": {"placed": lambda: True}})
    assert restored.verify_completion()["passed"]


def test_completion_cannot_skip_missing_public_invariant_binding(robot):
    contract = ActionContract(robot, acceptance())
    with pytest.raises(ActionContractPause) as exc:
        contract.verify_completion("finish")
    assert exc.value.report["failed_assumption"] == "neighbors"


def test_load_policy_cleans_sys_modules_on_pause(tmp_path):
    from pyrualean.runner import load_policy
    from pyrualean.supervision import SupervisionPause

    code_contract = (
        "from pyrualean.action_contract import ActionContractPause\n"
        "raise ActionContractPause({'reason': 'top_level_contract_pause', 'passed': False})\n"
        "def run(robo):\n    pass\n"
    )
    path_contract = tmp_path / "top_level_contract.py"
    path_contract.write_text(code_contract, encoding="utf-8")

    modules_before = set(sys.modules.keys())
    with pytest.raises(ActionContractPause):
        load_policy(path_contract)
    modules_after = set(sys.modules.keys())
    assert not [m for m in modules_after - modules_before if m.startswith("pyrualean_policy_")]

    code_supervision = (
        "from pyrualean.supervision import SupervisionPause\n"
        "raise SupervisionPause('top_level_supervision_pause')\n"
        "def run(robo):\n    pass\n"
    )
    path_supervision = tmp_path / "top_level_supervision.py"
    path_supervision.write_text(code_supervision, encoding="utf-8")

    modules_before = set(sys.modules.keys())
    with pytest.raises(SupervisionPause):
        load_policy(path_supervision)
    modules_after = set(sys.modules.keys())
    assert not [m for m in modules_after - modules_before if m.startswith("pyrualean_policy_")]


def test_run_policy_catches_action_contract_and_supervision_pause(robo):
    from pyrualean.runner import run_policy
    from pyrualean.supervision import SupervisionPause

    def contract_policy(r):
        raise ActionContractPause({"reason": "precondition_failed", "passed": False})

    outcome = run_policy(contract_policy, robo)
    assert outcome.status == "paused"
    assert outcome.reason == "precondition_failed"
    assert isinstance(outcome.pause, ActionContractPause)
    assert "ActionContractPause" in (outcome.error or "")
    # Verify PolicyOutcome.to_dict() is JSON-serializable when paused
    serialized = json.dumps(outcome.to_dict())
    assert "pause" not in outcome.to_dict()
    assert json.loads(serialized)["status"] == "paused"

    def supervision_policy(r):
        raise SupervisionPause("visual_safety_trigger")

    outcome = run_policy(supervision_policy, robo)
    assert outcome.status == "paused"
    assert outcome.reason == "visual_safety_trigger"
    assert isinstance(outcome.pause, SupervisionPause)
    assert "SupervisionPause" in (outcome.error or "")
    serialized = json.dumps(outcome.to_dict())
    assert "pause" not in outcome.to_dict()
    assert json.loads(serialized)["status"] == "paused"


def test_arm_toolkit_run_program_with_action_contract_pause(robo, tmp_path):
    from pyrualean.arms import ArmToolkit

    kit = ArmToolkit(robo, arm="program", workspace=tmp_path / "ws_contract")

    # Top-level module ActionContractPause via pre-bound plan(...)
    top_level_code = (
        "plan(goal='Init', done_when='ready', status='done', "
        "contract={'ensures': {'init_ready': lambda: False}})\n"
        "def run(robo):\n    pass\n"
    )
    res_top = kit.execute_tool("run_program", {"code": top_level_code})
    assert res_top.result["execution_paused"] is True
    assert res_top.result["status"] == "paused"
    assert "action_contract" in res_top.result

    # Inside run(robo) ActionContractPause
    inside_code = (
        "def run(robo):\n"
        "    plan(goal='Grasp', done_when='reached', contract={'requires': {'ready': lambda: False}})\n"
        "    robo.move_to([0.05, -0.10, 0.6])\n"
    )
    res_inside = kit.execute_tool("run_program", {"code": inside_code})
    assert res_inside.result["execution_paused"] is True
    assert res_inside.result["status"] == "paused"
    assert "action_contract" in res_inside.result
    assert "recovery" in res_inside.result
    assert res_inside.result["recovery"]["failed_assumption"] == "ready"


def test_arm_toolkit_run_program_with_supervision_pause(robo, tmp_path):
    from pyrualean.arms import ArmToolkit
    from pyrualean.supervision import SupervisionPause

    kit = ArmToolkit(robo, arm="program", workspace=tmp_path / "ws_supervision")

    # Top-level SupervisionPause via workspace module helper
    (kit.workspace / "alarm.py").write_text(
        "from pyrualean.supervision import SupervisionPause\n"
        "def trigger():\n"
        "    raise SupervisionPause('top_level_alarm')\n",
        encoding="utf-8",
    )
    top_level_code = (
        "import alarm\n"
        "alarm.trigger()\n"
        "def run(robo):\n    pass\n"
    )
    res_top = kit.execute_tool("run_program", {"code": top_level_code})
    assert res_top.result["execution_paused"] is True
    assert res_top.result["status"] == "paused"
    assert "supervision" in res_top.result

    # Inside run(robo) SupervisionPause via robot supervisor
    class MockSupervisor:
        def before_action(self, tool, r):
            raise SupervisionPause("inside_safety_alarm")

        def feedback(self):
            return {"paused": True, "reason": "inside_safety_alarm"}

        @property
        def paused(self):
            return True

    robo._supervisor = MockSupervisor()
    try:
        inside_code = (
            "def run(robo):\n"
            "    robo.move_to([0.05, -0.10, 0.6])\n"
        )
        res_inside = kit.execute_tool("run_program", {"code": inside_code})
        assert res_inside.result["execution_paused"] is True
        assert res_inside.result["status"] == "paused"
        assert "supervision" in res_inside.result
    finally:
        robo._supervisor = None
