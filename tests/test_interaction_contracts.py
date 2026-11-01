"""Code-cell recovery against a source/receipt backend with controllable faults."""

import copy
import json
from types import SimpleNamespace

import pytest

from tests.conftest import FakeLiberoBackend
from pyrualean import LiberoRobot
from pyrualean.arms import ArmResult, ArmToolkit
from pyrualean.supervision import MonitorConfig, VisualSupervisor


class ProtocolBackend(FakeLiberoBackend):
    session_id = "sensor-session"

    def __init__(self):
        super().__init__()
        self.receipts = {}
        self.contexts = []
        self.next_status = "completed"

    def observation_metadata(self, step=-1):
        record = self.steps[step]
        index = record["step_idx"]
        source = f"capture-{index}"
        return dict(
            schema="pyrualean.observation.v1", session_id=self.session_id,
            observation_id=source, source_id=source, step=index, native_steps=self.frames,
            captured_at=1000 + index, source_freshness="captured", aligned=True,
            images={c: dict(source_id=source, artifact=f"{c}_high.png", aligned=True)
                    for c in LiberoRobot.CAMERAS},
        )

    def _view(self, step):
        return dict(super()._view(step), observation=self.observation_metadata(step))

    def refresh_observation(self):
        self._record(command=None, result=None)
        return self._view(-1)

    def execute_tool_with_context(self, name, arguments, context):
        self.contexts.append(copy.deepcopy(context))
        # Simulate movement followed by a lost acknowledgement, never a retry.
        result = super().execute_tool(name, arguments)
        status, self.next_status = self.next_status, "completed"
        receipt = dict(context, status=status, effect_known=status != "unknown",
                       execution_complete=status != "unknown",
                       observation_id=self.observation_metadata()["observation_id"])
        self.receipts[context["action_id"]] = receipt
        result.result["action_receipt"] = copy.deepcopy(receipt)
        return result

    def reconcile(self, action_id):
        return copy.deepcopy(self.receipts[action_id])


@pytest.fixture
def session(tmp_path):
    backend = ProtocolBackend()
    robo = LiberoRobot(backend)
    predictions = []

    def predict(packet):
        predictions.append(packet)
        return {"answers": {key: {"type": "choice", "probabilities": {
            "yes": 1., "no": 0., "unknown": 0.,
        }} for key in packet["questions"]}}

    supervisor = VisualSupervisor(
        MonitorConfig(mode="pause", url="http://unused.invalid"), tmp_path / "audit",
        client=SimpleNamespace(predict=predict),
    )
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "cells", images=False,
                         feedback="pure", supervisor=supervisor)
    return backend, robo, toolkit, predictions


WATCH = "watch('move', {'clear': 'Is the target area clear?'}, requires={'clear': 'yes'})\n"
MOVE = "robo.move_to([0.05, -0.10, 0.6])\n"


def cell(toolkit, code):
    return toolkit.execute_tool("python", {"code": code})


def test_verify_execute_unknown_reconcile_new_cell_chain(session):
    backend, robo, toolkit, predictions = session
    before = cell(toolkit, WATCH + "source = robo.snapshot()\nverify(source['observation_id'])")
    assert not before.is_error and not backend.contexts and backend.frames == 0
    backend.next_status = "unknown"
    stopped = cell(toolkit, WATCH + MOVE + "should_not_run = True").result
    assert stopped["execution_paused"] and "should_not_run" not in toolkit._globals
    receipt = stopped["handoff"]["unresolved_actions"][0]
    action_id = receipt["action_id"]
    assert len(backend.contexts) == 1
    assert backend.contexts[0]["observation_id"] == toolkit._globals["source"]["observation_id"]
    assert stopped["observation"]["actions"]["action_ids"] == [action_id]
    assert stopped["action_receipts"][0]["action_id"] == action_id
    assert receipt["effect_known"] is False
    calls = len(predictions)
    still_unknown = cell(toolkit, WATCH + "robo.snapshot(refresh=True)\nverify()")
    assert still_unknown.result["execution_paused"]
    assert len(predictions) == calls  # images cannot stand in for an execution receipt
    assert cell(toolkit, WATCH + MOVE).result["execution_paused"]
    assert len(backend.contexts) == 1

    backend.receipts[action_id].update(status="partial", effect_known=True, execution_complete=True)
    reconciled = cell(toolkit, "r = robo.reconcile()\nassert r['status'] == 'partial'")
    assert not reconciled.is_error and not reconciled.result["handoff"]["unresolved_actions"]
    # Completion of the interrupted action is insufficient to authorize the next one.
    assert cell(toolkit, WATCH + MOVE).result["execution_paused"]
    resumed = cell(toolkit, WATCH + "robo.snapshot(refresh=True)\nverify()\n" + MOVE)
    assert not resumed.is_error and not resumed.result.get("execution_paused")
    assert len(backend.contexts) == 2 and backend.contexts[1]["action_id"] != action_id
    assert backend.contexts[0]["turn"] != backend.contexts[1]["turn"]
    assert toolkit.turns[-1]["handoff"]["last_action"]["status"] == "completed"


def test_wrong_receipt_cannot_clear_blocker_and_success_is_refused(session):
    backend, robo, toolkit, _ = session
    cell(toolkit, WATCH + "verify()")
    backend.next_status = "unknown"
    cell(toolkit, WATCH + MOVE)
    action_id = robo.action_receipt()["action_id"]
    backend.receipts[action_id].update(action_id="different", effect_known=True,
                                       execution_complete=True, status="completed")
    assert cell(toolkit, "robo.reconcile()").is_error
    assert robo._pending_actions == {action_id}
    assert toolkit.execute_tool("finish", {"status": "success"}).is_error
    assert toolkit.finish is None
    assert cell(toolkit, MOVE).result["execution_paused"]
    assert len(backend.contexts) == 1


def test_historical_read_does_not_replace_current_state(session):
    _, robo, toolkit, _ = session
    first = robo.state()
    cell(toolkit, WATCH + "verify()\n" + MOVE)
    current = robo._current_state()
    old = robo.state(step=0)
    assert old == first and current != first
    assert robo._current_state() == current


def test_unknown_handoff_survives_large_output_and_large_native_receipts(session):
    backend, robo, toolkit, _ = session
    cell(toolkit, WATCH + "verify()")
    backend.next_status = "unknown"
    response = cell(toolkit, WATCH + "print(chr(1) * 8000)\n" + MOVE)
    action_id = robo.action_receipt()["action_id"]
    robo._receipts[action_id]["native_receipts"] = [{"details": "x" * 1000}] * 200
    response.result["handoff"] = robo._interaction_handoff()
    block = response.content_blocks[0]["text"]
    assert len(block) <= 20000
    parsed = json.loads(block)
    assert parsed["handoff"]["unresolved_actions"][0]["action_id"] == action_id
    assert parsed["handoff"]["allowed_next"] == "observe_or_reconcile"


def test_program_unknown_pauses_instead_of_escaping(tmp_path):
    backend = ProtocolBackend()
    backend.next_status = "unknown"
    toolkit = ArmToolkit(LiberoRobot(backend), arm="program", workspace=tmp_path, images=False)
    result = toolkit.execute_tool("run_program", {"code": "def run(robo):\n    " + MOVE.strip()})
    assert result.result["status"] == "paused"
    assert result.result["handoff"]["unresolved_actions"]
    assert len(backend.contexts) == 1


def test_compressed_feedback_retains_handoff_after_many_actions():
    ids = ["s" * 32 + ":action:" + str(i).zfill(32) for i in range(300)]
    result = ArmResult(dict(observation={"actions": {"action_ids": ids, "attempted": 300}},
                            handoff={"unresolved_actions": [{"action_id": ids[-1]}]}))
    text = result.content_blocks[0]["text"]
    assert len(text) <= 20000
    report = json.loads(text)
    assert report["observation"]["actions"]["action_id_count"] == 300
    assert report["handoff"]["unresolved_actions"][0]["action_id"] == ids[-1]


def test_native_bridge_receipt_reaches_cells_and_late_recovery(tmp_path, monkeypatch):
    """Exercise the real native adapter, transport and supervision in one chain."""
    from pyrualean.hosts import robodojo_tools as native_tools
    from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit
    from pyrualean.hosts.xpolicylab_bridge import REQUEST_KEY, Bridge, BridgeEnv
    from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
    from tests.test_observation_transport import Environment, acknowledge, observation, start
    from tests.test_supervision import setup

    bridge = Bridge(poll_s=.02)

    def source(sequence, receipt=None):
        raw = observation(sequence, receipt)
        raw.update(Environment().get_obs())
        bridge.update_obs(raw)

    source(1)
    host = NativeToolkit(BridgeEnv(bridge), None, tmp_path / "native")
    robo = RobodojoOfficialColourRobot(NativeBackend(host))
    step = bridge.step
    monkeypatch.setattr(bridge, "step", lambda action, **kw: step(action, timeout_s=.1, **kw))

    def grip(**kwargs):
        native_tools._act(host._primitives, native_tools._obs(host._primitives)["state"], {}, {})
        return {"reached": True, "steps_used": 1}

    host._tools["grip"] = grip
    yes = {"answers": {"clear": {"type": "choice", "probabilities": {
        "yes": 1., "no": 0., "unknown": 0.,
    }}}}
    toolkit, _, _ = setup(robo, tmp_path, [yes, yes, yes])

    def dispatched(thread, results):
        for _ in range(50):
            actions = bridge.get_action()
            if actions and REQUEST_KEY in actions[0]:
                return actions[0][REQUEST_KEY]
            if not thread.is_alive():
                break
        pytest.fail(f"Expected native dispatch; cell returned {results}")

    assert not cell(toolkit, WATCH + "verify()").is_error
    move = "robo.set_gripper('right', robo.CLOSE)"
    thread, results = start(lambda: cell(toolkit, WATCH + move))
    request = dispatched(thread, results)
    thread.join(1)
    assert not thread.is_alive() and results[0].result["execution_paused"]
    action_id = results[0].result["handoff"]["unresolved_actions"][0]["action_id"]
    assert request["action_id"] == action_id
    assert cell(toolkit, WATCH + move).result["execution_paused"]
    source(2, acknowledge(request))
    repaired = cell(toolkit, WATCH + "receipt = robo.reconcile()\nverify()")
    assert not repaired.is_error and not repaired.result.get("execution_paused")
    assert robo.action_receipt()["status"] == "partial"
    assert robo.snapshot()["source_id"] == "client:2"
    thread, results = start(lambda: cell(toolkit, WATCH + move))
    second = dispatched(thread, results)
    assert second["action_id"] != action_id
    source(3, acknowledge(second))
    thread.join(1)
    assert not thread.is_alive() and not results[0].is_error
    assert not results[0].result.get("execution_paused")
    assert results[0].result["handoff"]["last_action"]["status"] == "completed"
