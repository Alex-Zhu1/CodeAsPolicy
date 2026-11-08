"""Task contracts at real dispatch boundaries, across different robot adapters."""

import json

import pytest

from pyrualean.action_contract import ActionContractPause
from pyrualean.arms import ArmToolkit
from tests.test_robodojo_intent import _precision_station


def cell(toolkit, code):
    return toolkit.execute_tool("python", {"code": code})


def test_public_invariant_needs_executable_binding_even_with_monitor_off(robo, backend, tmp_path):
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"acceptance": {"requirements": [dict(
        id="preserve_support", kind="invariant", condition="Keep the support unchanged.",
        source="public instruction",
    )]}}))
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", task_card_path=card)
    result = cell(tk, 'plan(goal="Move", done_when="arrived", evidence="support is unchanged")\n'
                      'robo.move_to([0, 0, .6])')
    assert result.result["execution_paused"]
    assert result.result["action_contract"]["uncovered_invariants"] == ["preserve_support"]
    assert not any(r.stateful for r in robo._ledger.records)
    assert tk.finish is None
    # A changed phase name cannot make the public invariant disappear.
    second = cell(tk, 'plan(goal="Different stage", done_when="arrived", contract={})\n'
                      'robo.move_to([0, 0, .6])')
    assert second.result["execution_paused"] and not any(r.stateful for r in robo._ledger.records)


def test_post_action_invariant_pauses_remaining_cell_without_losing_receipt(robo, tmp_path):
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", feedback="pure")
    result = cell(tk, '''
plan(goal="Transport", done_when="arrived", contract={
    "preserves": {"height": lambda: robo.state().eef_pos[2] >= .60}})
robo.move_to([0, 0, .55])
robo.move_to([0, 0, .7])
''')
    assert result.result["execution_paused"]
    assert result.result["recovery"]["failed_assumption"] == "height"
    actions = [r for r in robo._ledger.records if r.stateful]
    assert len(actions) == 1 and actions[0].ok
    assert actions[0].receipt["execution_complete"] is True
    assert not robo._pending_actions
    assert robo.state().eef_pos[2] == pytest.approx(.55)
    # Reading and changing the note are possible, but persistent protection remains.
    observed = cell(tk, 'print(robo.state().eef_pos)')
    assert not observed.is_error
    blocked = cell(tk, 'plan(goal="New stage", done_when="arrived", contract={})\n'
                       'robo.move_to([0, 0, .7])')
    assert blocked.result["execution_paused"]
    assert len([r for r in robo._ledger.records if r.stateful]) == 1


def test_unknown_prerequisite_blocks_only_work_that_depends_on_it(robo, tmp_path):
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    blocked = cell(tk, 'plan(goal="Release", done_when="supported", '
                       'contract={"requires": {"support": lambda: None}})\n'
                       'robo.move_to([0, 0, .6])')
    assert blocked.result["execution_paused"]
    assert blocked.result["recovery"]["evidence_state"] == "unknown"
    assert not any(r.stateful for r in robo._ledger.records)
    independent = cell(tk, 'print(robo.state().eef_pos)\n'
                           'plan(goal="Independent clear move", done_when="arrived", contract={})\n'
                           'robo.move_to([0, 0, .65])')
    assert not independent.is_error and not independent.result.get("execution_paused")


def test_check_callback_cannot_dispatch_or_reconfigure(robo, tmp_path):
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    result = cell(tk, 'plan(goal="Move", done_when="arrived", contract={"requires": '
                      '{"bad": lambda: robo.move_to([0, 0, .6])}})\n'
                      'robo.move_to([0, 0, .65])')
    assert result.result["execution_paused"]
    assert not any(r.stateful for r in robo._ledger.records)


def test_current_failed_assumption_takes_priority_over_retained_geometry(robo, tmp_path):
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    robo._coordination_feedback = {"constraint_violation": {
        "metric": "old_clearance", "value": .01, "bound": .02, "delta": -.01}}
    result = cell(tk, 'plan(goal="Release", done_when="supported", '
                      'contract={"requires": {"support": lambda: None}})\n'
                      'robo.move_to([0, 0, .6])')
    assert result.result["recovery"]["kind"] == "contract"
    assert result.result["recovery"]["failed_assumption"] == "support"
    assert result.result["recovery"]["next_action"] == "reobserve_or_replan"
    assert not any(r.stateful for r in robo._ledger.records)


def test_arrival_does_not_satisfy_semantic_completion_or_finish(robo, tmp_path):
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    moved = cell(tk, 'plan(goal="Place", done_when="supported", '
                     'contract={"ensures": {"supported": lambda: None}})\n'
                     'robo.move_to([0, 0, .6])')
    assert not moved.is_error and not moved.result.get("execution_paused")
    done = cell(tk, 'plan(goal="Place", done_when="supported", status="done")')
    assert done.result["execution_paused"]
    assert tk._plan()["status"] == "in_progress"
    finished = tk.execute_tool("finish", {"status": "success", "summary": "arrived"})
    assert finished.result["status"] == "completion_not_verified"
    assert finished.result["episode_finished"] is False and tk.finish is None
    stopped = tk.execute_tool("finish", {"status": "failure", "summary": "support remains unknown"})
    assert stopped.result["ok"]


def test_restart_requires_rebinding_live_protections(tmp_path):
    from pyrualean.libero import LiberoRobot
    from tests.test_interaction_contracts import ProtocolBackend

    backend = ProtocolBackend()
    kwargs = dict(arm="cells", workspace=tmp_path / "ws", images="none",
                  execution_state_path=tmp_path / "execution.sqlite3")
    tk = ArmToolkit(LiberoRobot(backend), **kwargs)
    first = cell(tk, 'plan(goal="Transport", done_when="arrived", contract={"preserves": '
                     '{"height": lambda: robo.state().eef_pos[2] > .4}})')
    assert not first.is_error
    assert tk.close()
    resumed = ArmToolkit(LiberoRobot(backend), resume=True, **kwargs)
    # Call the robot directly to isolate contract restoration from code-review tokens.
    with pytest.raises(ActionContractPause):
        resumed.robo.move_to([0, 0, .6])
    assert not backend.contexts
    resumed._plan(goal="Inspect and continue", done_when="arrived", contract={
        "preserves": {"height": lambda: resumed.robo.state().eef_pos[2] > .4}})
    assert resumed.robo.move_to([0, 0, .6]).reached
    resumed.close()


@pytest.mark.parametrize("method", ["execute_motion", "align_local", "begin_grasp"])
def test_stage_constraints_inherit_through_all_preparation_routes(tmp_path, method):
    env, robo, host = _precision_station(tmp_path)
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    pose = robo.observe().right
    ceiling = pose.xyz[2] + .005
    tk._plan(goal="Bounded operation", done_when="arrived",
             contract={"constraints": {"z_bounds": {"right": [0, ceiling]}}})
    if method == "execute_motion":
        result = robo.execute_motion("move", right=pose.shifted(dz=.02))
    elif method == "align_local":
        result = robo.align_local("align", [0, 0, .02], arm="right", reference=pose,
                                  translation_limits=[.03, .03, .03])
    else:
        robo.begin_grasp("pick", pose, arm="right", approach=[0, 0, .02], opening=.7)
        result = robo.grasp_step("pick", "open")
        assert result["reached"]
        count = len(env.actions)
        result = robo.grasp_step("pick", "approach")
        assert len(env.actions) == count
    assert not result["reached"]
    if method != "begin_grasp":
        assert not env.actions
    assert "z_bounds" in result["stop_reason"]


def test_geometry_constraints_cannot_be_bypassed_by_raw_primitive(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    tk._plan(goal="Bounded", done_when="arrived", contract={"constraints": {"z_bounds": [0, 2]}})
    with pytest.raises(ActionContractPause, match="unsupported_constraint_dispatch"):
        robo.set_gripper("right", robo.OPEN)
    assert not env.actions


def test_old_prepared_operation_cannot_adopt_a_new_contract(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    target = robo.observe().right.shifted(dz=.025)
    robo.begin_grasp("pick", target, arm="right")
    tk._plan(goal="Bounded", done_when="arrived", contract={"constraints": {"z_bounds": [0, 2]}})
    with pytest.raises(ActionContractPause, match="prepared_constraints_changed"):
        robo.grasp_step("pick", "open")
    assert not env.actions


def test_explicit_constraint_cannot_relax_inherited_bounds(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    pose = robo.observe().right
    tk._plan(goal="Bounded", done_when="arrived", contract={"constraints": {"z_bounds": [0, 1]}})
    with pytest.raises(ActionContractPause, match="conflicting inherited constraint"):
        robo.execute_motion("move", right=pose, constraints={"z_bounds": [0, 2]})
    assert not env.actions


def test_obstacle_contact_by_volume_is_rejected_when_tcp_is_clear(tmp_path):
    import numpy as np
    from pyrualean.hosts import robodojo_kinematics as kin

    env, robo, _ = _precision_station(tmp_path)
    pose = robo.observe().right
    center = np.asarray(pose.xyz) + kin.rot_from_quat(pose.quat) @ np.array([.06, 0., 0.])
    result = robo.execute_motion("volume", right=pose.shifted(dz=.01), constraints={
        "keep_out": [{"xyz": center.tolist(), "radius_m": .01}],
        "geometry": {"arms": {"right": [{"name": "held", "xyz": [.06, 0, 0], "radius_m": .015}]}}
    })
    assert not result["reached"] and result["constraint_violation"]
    assert result["geometry_coverage"]["full_robot_collision"] is False
    assert not env.actions


def test_inherited_obstacles_cannot_redirect_an_explicit_contact_exception(tmp_path):
    env, robo, _ = _precision_station(tmp_path)
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    obstacle = {"xyz": [0, 0, .5], "radius_m": .02}
    tk._plan(goal="Protect surroundings", done_when="arrived", contract={
        "constraints": {"keep_out": [obstacle]}})
    explicit = {"keep_out": [{"xyz": [0, .2, .5], "radius_m": .02}], "geometry": {
        "arms": {"right": [{"name": "pad", "xyz": [0, 0, 0], "radius_m": .01}]},
        "allowed_contacts": [{"pair": ["right:pad", "keep_out:0"], "max_penetration_m": .005}],
    }}
    with pytest.raises(ActionContractPause, match="would change an allowed contact"):
        robo.execute_motion("contact", right=robo.observe().right, constraints=explicit)
    assert not env.actions
