"""Causal identity across the native client, bridge, and toolkit, without a simulator."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from pyrualean.hosts import robodojo_tools as tools
from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit
from pyrualean.hosts.xpolicylab_bridge import (
    OBSERVATION_KEY,
    REQUEST_KEY,
    WAIT,
    Bridge,
    BridgeEnv,
    PolicyModel,
)


def observation(sequence, receipt=None, source_session="client", behind=0):
    return {
        "state": {"sample": sequence}, "vision": {},
        "additional_info": {"pyrualean": {"step": sequence - 1, "pictures_behind": behind}},
        OBSERVATION_KEY: dict(protocol=1, source_session_id=source_session,
                              source_id=f"{source_session}:{sequence}", sequence=sequence,
                              captured_at=123.0 + sequence, source_freshness="captured",
                              capture_clock="client_unix_utc",
                              action_receipt=receipt),
    }


def start(call):
    results = []

    def run():
        try:
            results.append(call())
        except BaseException as exc:
            results.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, results


def acknowledge(request):
    return {**request, "status": "completed"}


def test_late_receipt_cannot_complete_another_action_or_replace_session():
    bridge = Bridge(poll_s=.05)
    bridge.update_obs(observation(1))
    thread, results = start(lambda: bridge.step({"target": 1}, timeout_s=.03,
                                              context={"action_id": "first"}))
    request = bridge.get_action()[0][REQUEST_KEY]
    thread.join(1)
    assert isinstance(results[0], TimeoutError)
    assert bridge.receipts("first")[0]["status"] == "unknown"
    with pytest.raises(RuntimeError, match="unresolved"):
        bridge.step({"target": 2})
    wrong = {**acknowledge(request), "native_action_id": "someone-else"}
    bridge.update_obs(observation(2, wrong))
    bridge.update_obs(observation(99, acknowledge(request), source_session="old-client"))
    assert bridge.get_obs()["state"]["sample"] == 2
    assert not bridge.receipts("first")[0]["effect_known"]
    bridge.update_obs(observation(3, acknowledge(request)))
    assert bridge.receipts("first")[0]["status"] == "completed"

    thread, results = start(lambda: bridge.step({"target": 2}, timeout_s=1,
                                              context={"action_id": "second"}))
    second = bridge.get_action()[0][REQUEST_KEY]
    bridge.update_obs(observation(4, acknowledge(request)))
    thread.join(.01)
    assert thread.is_alive()  # first action's repeated ack does not release the second
    bridge.update_obs(observation(5, acknowledge(second)))
    thread.join(1)
    assert results[0]["state"]["sample"] == 5


def test_queued_timeout_withdraws_action_and_cancellation_is_not_hardware_stop():
    bridge = Bridge(poll_s=.001)
    bridge.update_obs(observation(1))
    with pytest.raises(TimeoutError):
        bridge.step({"target": 1}, timeout_s=.001, context={"action_id": "queued"})
    assert bridge.get_action() == WAIT
    receipt = bridge.receipts("queued")[0]
    assert receipt["status"] == "rejected" and receipt["execution_complete"]
    thread, results = start(lambda: bridge.step({}, timeout_s=30,
                                              context={"action_id": "dispatched"}))
    request = bridge.get_action()[0][REQUEST_KEY]
    cancellation = bridge.cancel()
    assert cancellation == dict(requested=True, confirmed=False, scope="native_dispatch",
                                physical_stop_confirmed=None)
    thread.join(.5)  # cancellation must not wait for the native request's 30 s deadline
    assert not thread.is_alive()
    assert isinstance(results[0], TimeoutError)
    assert "cancellation requested" in str(results[0])
    assert bridge.receipts("dispatched")[0]["status"] == "unknown"
    with pytest.raises(RuntimeError, match="unresolved"):
        bridge.step({"next": 1})
    bridge.update_obs(observation(2, acknowledge(request)))
    assert bridge.cancel()["confirmed"]


def test_old_client_is_readable_but_cannot_supply_guessed_action_ack():
    bridge = Bridge()
    bridge.update_obs({"state": {}, "vision": {}})
    assert bridge.get_obs()[OBSERVATION_KEY].get("source_id") is None
    assert not bridge.get_obs()[OBSERVATION_KEY]["aligned"]
    with pytest.raises(RuntimeError, match="identity protocol"):
        bridge.step({})
    with pytest.raises(RuntimeError, match="capture protocol"):
        bridge.refresh()


@pytest.mark.parametrize("changes", [
    {"source_session_id": ""}, {"source_id": ""}, {"sequence": True}, {"sequence": 0},
    {"sequence": 1.5}, {"captured_at": None}, {"captured_at": float("nan")},
    {"captured_at": float("inf")}, {"capture_clock": ""}, {"source_freshness": "cached"},
    {"protocol": True},
])
def test_incomplete_protocol_identity_never_enables_dispatch_or_refresh(changes):
    bridge = Bridge()
    obs = observation(1)
    obs[OBSERVATION_KEY].update(changes)
    bridge.update_obs(obs)
    source = bridge.get_obs()[OBSERVATION_KEY]
    assert source["source_id"] is None and source["source_freshness"] == "unverified"
    assert not source["aligned"]
    with pytest.raises(RuntimeError, match="identity protocol"):
        bridge.step({})
    with pytest.raises(RuntimeError, match="capture protocol"):
        bridge.refresh()


def test_protocol_number_alone_is_not_source_identity_and_cannot_downgrade_valid_source():
    bridge = Bridge()
    malformed = {"state": {"bad": True}, "vision": {}, OBSERVATION_KEY: {"protocol": 1}}
    bridge.update_obs(malformed)
    assert not bridge.get_obs()[OBSERVATION_KEY]["aligned"]
    assert bridge.get_obs()[OBSERVATION_KEY]["source_id"] is None
    bridge.update_obs(observation(1))
    bridge.update_obs(malformed)
    assert bridge.get_obs()[OBSERVATION_KEY]["source_id"] == "client:1"


def test_old_client_readonly_refresh_keeps_image_provenance_and_pose():
    bridge = Bridge()
    first = observation(1, behind=1)
    first["vision"] = {"wrist": {"extrinsic_matrix": "first"}}
    bridge.update_obs(first)
    moved = observation(2, behind=1)
    moved["vision"] = {"wrist": {"extrinsic_matrix": "second"}}
    bridge.update_obs(moved)
    assert bridge.get_obs()["image_state"] == {"sample": 1}
    refresh = observation(3, behind=1)
    refresh["additional_info"]["pyrualean"]["step"] = 1  # no native action
    refresh["vision"] = {"wrist": {"extrinsic_matrix": "second"}}
    bridge.update_obs(refresh)
    obs = bridge.get_obs()
    assert obs["image_state"] == {"sample": 1}
    assert obs["vision"]["wrist"]["extrinsic_matrix"] == "first"
    assert obs[OBSERVATION_KEY]["image_source_id"] == "client:1"
    assert not obs[OBSERVATION_KEY]["aligned"]


class Environment:
    def __init__(self):
        self.actions = []
        self.step_limit = 100

    def get_obs(self):
        state = {}
        for arm in ("left", "right"):
            state[f"{arm}_arm_joint_state"] = [0.] * 6
            state[f"{arm}_ee_joint_state"] = [1.]
        return {
            "state": state,
            "vision": {"cam_head": {"color": np.zeros((2, 2, 3), dtype=np.uint8)}},
        }

    def get_status(self):
        return {"step": len(self.actions), "step_limit": self.step_limit}

    def take_action(self, action):
        assert REQUEST_KEY not in action
        self.actions.append(action)

    def step(self, action):
        self.take_action(action)
        return self.get_obs(), 0., False, {"status": self.get_status()}

    @property
    def episode_step(self):
        return len(self.actions)

    @property
    def episode_step_limit(self):
        return self.step_limit

    def is_episode_end(self):
        return len(self.actions) >= self.step_limit


class IsaacEnvironment(Environment):
    """A stale renderer whose pixels only catch up at the synchronous render fence."""

    def __init__(self):
        super().__init__()
        self.rendered_step = -1
        self.forward_calls = 0
        self.render_calls = []
        self.sim = SimpleNamespace(sim=SimpleNamespace(
            current_time=2.0, current_time_step_index=120, forward=self.forward, render=self.render,
        ))
        self.capture_manager = SimpleNamespace(tiled_cameras=[
            SimpleNamespace(_render_product_path="/Render/head"),
            SimpleNamespace(_render_product_path="/Render/wrist"),
        ])
        self.frame = 100
        self.frame_lag = 0
        self.missing_products = set()
        self.emit = lambda *args: None
        self.annotator_syncs = 0

    def forward(self):
        self.forward_calls += 1

    def render(self):
        self.render_calls.append({})
        self.frame += 1
        self.emit("pre_update", {"SWHFrameNumber": self.frame})
        for camera in self.capture_manager.tiled_cameras:
            path = camera._render_product_path
            if path not in self.missing_products:
                self.emit("new_frame", {"product_path_handle": path, "results": {},
                                         "swh_frame_number": self.frame - self.frame_lag})
        self.rendered_step = len(self.actions)

    def get_obs(self):
        obs = super().get_obs()
        obs["vision"]["cam_head"]["color"][:] = max(0, self.rendered_step)
        # The image and the independently read calibration must come from one state.
        obs["vision"]["cam_head"]["extrinsic_matrix"] = np.eye(4)
        obs["vision"]["cam_head"]["extrinsic_matrix"][0, 3] = len(self.actions)
        return obs

    def take_action(self, action):
        super().take_action(action)
        self.sim.sim.current_time += 0.1
        self.sim.sim.current_time_step_index += 1


def install_renderer(monkeypatch, env):
    """Use the actual deployment capture algorithm with deterministic SDK events."""
    names = (
        "carb", "carb.eventdispatcher", "carb.settings", "omni", "omni.graph",
        "omni.graph.core", "omni.kit", "omni.kit.app", "omni.kit.renderer_capture",
        "omni.usd", "omni.syntheticdata", "omni.syntheticdata.scripts",
        "omni.syntheticdata.scripts.helpers",
    )
    modules = {}
    for name in names:
        module = modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    callbacks = {}

    class Subscription:
        def __init__(self, event_name, on_event, **kwargs):
            self.name, self.callback = event_name, on_event
            callbacks.setdefault(event_name, []).append(on_event)

        def __del__(self):
            callbacks[self.name].remove(self.callback)

    def emit(event_name, payload):
        for callback in list(callbacks.get(event_name, [])):
            callback(payload)

    env.emit = emit
    env.capture_settings = {"/app/asyncRendering": True, "/app/asyncRenderingLowLatency": True,
                            "/rtx/ecoMode/enabled": True}
    env.capture_sync = False
    env.sim.sim.render = env.render

    def set_capture_sync(value):
        previous, env.capture_sync = env.capture_sync, value
        return previous

    def evaluate_sync(graph_id):
        assert graph_id == "/Render/PostProcess/SDGPipeline"
        env.annotator_syncs += 1

    modules["carb.eventdispatcher"].get_eventdispatcher = lambda: SimpleNamespace(
        observe_event=Subscription,
    )
    modules["carb.settings"].get_settings = lambda: SimpleNamespace(
        get=env.capture_settings.get, set=env.capture_settings.__setitem__,
        destroy_item=lambda name: env.capture_settings.pop(name, None),
    )
    modules["omni.graph.core"].Controller = lambda: SimpleNamespace(
        graph=lambda path: path, evaluate_sync=evaluate_sync,
    )
    modules["omni.kit.app"].GLOBAL_EVENT_PRE_UPDATE = "pre_update"
    modules["omni.kit.renderer_capture"].acquire_renderer_capture_interface = lambda: (
        SimpleNamespace(set_capture_sync=set_capture_sync)
    )
    modules["omni.usd"].get_context = lambda: SimpleNamespace(
        stage_rendering_event_name=lambda kind: kind,
    )
    modules["omni.usd"].StageRenderingEventType = SimpleNamespace(NEW_FRAME="new_frame")
    modules["omni.syntheticdata.scripts.helpers"]._get_syntheticdata_iface = lambda: (
        SimpleNamespace(parse_rendered_simulation_event=lambda path, results: (path, 1, 1))
    )


def deploy(name):
    path = Path(__file__).resolve().parents[1] / "integrations/xpolicylab" / name / "deploy.py"
    spec = importlib.util.spec_from_file_location(f"transport_deploy_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Client:
    def __init__(self, bridge):
        self.model = PolicyModel(bridge)

    def call(self, func_name, obs=None):
        method = getattr(self.model, "on_trial_end" if func_name == "trial_end" else func_name)
        return method() if obs is None else method(obs)


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
def test_readonly_capture_and_action_receipts_through_real_deploy(adapter, tmp_path, monkeypatch):
    bridge = Bridge(poll_s=.02)
    env = IsaacEnvironment()
    install_renderer(monkeypatch, env)
    client = deploy(adapter)
    thread, results = start(lambda: client.eval_one_episode(env, Client(bridge)))
    bridge.first_obs(1)
    host = NativeToolkit(BridgeEnv(bridge), None, tmp_path)
    backend = NativeBackend(host)
    initial = backend.observation_metadata()
    refreshed = backend.refresh_observation()["observation"]
    assert refreshed["source_id"] != initial["source_id"]
    assert refreshed["native_steps"] == initial["native_steps"] == 0
    assert refreshed["aligned"] and not env.actions
    assert refreshed["captured_at"] is not None
    alignment = bridge.get_obs()[OBSERVATION_KEY]["alignment"]
    assert alignment["basis"] == "isaac_render_product_fence"
    assert alignment["physics_step_before"] == alignment["physics_step_after"] == 120
    assert env.forward_calls == 2 and len(env.render_calls) == 2
    assert (host.state.root / "0001/observation.json").exists()

    def one_step():
        tools._act(host._primitives, tools._obs(host._primitives)["state"], {}, {})
        return {"reached": True}

    host._tools["grip"] = one_step
    result = backend.execute_tool_with_context("grip", {}, dict(
        session_id=backend.session_id, action_id="move",
        observation_id=refreshed["observation_id"],
    ))
    receipt = result["action_receipt"]
    assert receipt["status"] == "completed" and receipt["effect_known"]
    assert receipt["execution_complete"] and len(receipt["native_action_ids"]) == 1
    assert result["observation"]["action_id"] == "move"
    assert backend.reconcile("move")["status"] == "completed"
    with pytest.raises(ValueError, match="reconcile"):
        backend.execute_tool_with_context("grip", {}, {"action_id": "move"})
    assert len(env.actions) == 1
    assert result["observation"]["aligned"]
    assert bridge.get_obs()["vision"]["cam_head"]["color"][0, 0, 0] == 1
    assert bridge.get_obs()["vision"]["cam_head"]["extrinsic_matrix"][0, 3] == 1
    bridge.finish()
    thread.join(1)
    assert results == [None]


def test_unverified_native_observations_do_not_invent_independent_sources(tmp_path):
    host = NativeToolkit(Environment(), None, tmp_path)
    backend = NativeBackend(host)
    first = backend.observation_metadata()
    second = backend.refresh_observation()["observation"]
    assert first["observation_id"] == second["observation_id"]
    assert first["source_id"] is None and second["captured_at"] is None
    assert second["source_freshness"] == "unverified"
    assert second["step"] == first["step"] + 1


def test_stale_source_is_rejected_before_native_dispatch(tmp_path):
    bridge = Bridge()
    raw = Environment().get_obs()
    raw.update({key: value for key, value in observation(1).items()
                if key not in ("state", "vision")})
    bridge.update_obs(raw)
    backend = NativeBackend(NativeToolkit(BridgeEnv(bridge), None, tmp_path))
    newer = Environment().get_obs()
    newer.update({key: value for key, value in observation(2).items()
                  if key not in ("state", "vision")})
    bridge.update_obs(newer)
    result = backend.execute_tool_with_context("grip", {"arm": "right", "gripper": -1}, {
        "action_id": "stale", "source_id": "client:1", "source_native_steps": 0,
    })
    receipt = result["action_receipt"]
    assert receipt["status"] == "rejected" and receipt["effect_known"]
    assert receipt["execution_complete"] and not receipt["native_action_ids"]
    assert bridge.receipts("stale") == []
    assert result["observation"]["source_id"] == "client:2"
    assert result["log"]["result"]["steps_used"] == 0


def test_live_source_is_separate_from_recorded_artifacts_at_same_native_step(tmp_path):
    bridge = Bridge()
    raw = Environment().get_obs()
    raw.update({key: value for key, value in observation(1).items()
                if key not in ("state", "vision")})
    bridge.update_obs(raw)
    backend = NativeBackend(NativeToolkit(BridgeEnv(bridge), None, tmp_path))
    recorded = backend.observation_metadata()
    newer = Environment().get_obs()
    newer.update({key: value for key, value in observation(2).items()
                  if key not in ("state", "vision")})
    newer["additional_info"]["pyrualean"]["step"] = 0
    bridge.update_obs(newer)
    live = backend.current_observation_source()
    assert live["session_id"] == recorded["session_id"]
    assert live["native_steps"] == recorded["native_steps"] == 0
    assert live["source_id"] == "client:2" and recorded["source_id"] == "client:1"
    assert backend.observation_metadata() == recorded  # no artifact relabeling
    assert len(backend.toolkit.state.records) == 1


def test_partial_motion_receipt_and_unknown_after_native_error(tmp_path):
    host = NativeToolkit(Environment(), None, tmp_path)
    backend = NativeBackend(host)

    def partial():
        tools._act(host._primitives, tools._obs(host._primitives)["state"], {}, {})
        return {"status": "partial", "reached": False}

    host._tools["grip"] = partial
    receipt = backend.execute_tool_with_context("grip", {}, {"action_id": "partial"})[
        "action_receipt"]
    assert receipt["status"] == "partial" and receipt["effect_known"]
    assert receipt["execution_complete"] and receipt["stop_confirmed"] is None

    def fail():
        partial()
        raise RuntimeError("result lost after physical execution")

    host._tools["grip"] = fail
    with pytest.raises(RuntimeError, match="result lost"):
        backend.execute_tool_with_context("grip", {}, {"action_id": "lost"})
    unknown = backend.reconcile("lost")
    assert unknown["status"] == "unknown" and not unknown["effect_known"]
    assert not unknown["execution_complete"]
    assert len(host._primitives.env.actions) == 2  # reconciliation did not replay anything


def test_native_reconciliation_requires_exact_late_completion(tmp_path, monkeypatch):
    bridge = Bridge(poll_s=.02)
    initial = Environment().get_obs()
    initial.update({key: value for key, value in observation(1).items()
                    if key not in ("state", "vision")})
    bridge.update_obs(initial)
    host = NativeToolkit(BridgeEnv(bridge), None, tmp_path)
    backend = NativeBackend(host)
    step = bridge.step
    monkeypatch.setattr(bridge, "step", lambda action, **kw: step(action, timeout_s=.03, **kw))

    def move():
        tools._act(host._primitives, tools._obs(host._primitives)["state"], {}, {})
        return {"reached": True}

    host._tools["grip"] = move
    thread, results = start(lambda: backend.execute_tool_with_context(
        "grip", {}, {"action_id": "lost", "session_id": backend.session_id}))
    request = bridge.get_action()[0][REQUEST_KEY]
    thread.join(1)
    assert isinstance(results[0], TimeoutError)
    assert backend.reconcile("lost")["status"] == "unknown"
    final = Environment().get_obs()
    final["vision"]["cam_head"]["color"][:] = 91
    final.update({key: value for key, value in observation(2, acknowledge(request)).items()
                  if key not in ("state", "vision")})
    bridge.update_obs(final)
    receipt = backend.reconcile("lost")
    assert receipt["status"] == "partial"  # target arrival was never established
    assert receipt["effect_known"] and receipt["execution_complete"]
    assert receipt["observation_id"] == "client:2"
    current = backend.observation_metadata()
    assert current["source_id"] == receipt["current_observation_id"] == "client:2"
    assert current["step"] == receipt["current_observation_step"] == 1
    from PIL import Image
    with Image.open(host.state.artifact_path("cam_head.png")) as image:
        assert np.asarray(image)[0, 0, 0] == 91
    backend.reconcile("lost")
    assert len(host.state.records) == 2  # polling the resolved receipt does not recapture
    assert bridge.get_action() == WAIT


def test_terminal_action_ack_and_observation_are_published_atomically(monkeypatch):
    bridge = Bridge(poll_s=.02)
    env = IsaacEnvironment()
    env.step_limit = 1
    install_renderer(monkeypatch, env)
    client = deploy("PyRUA_Lean")
    thread, results = start(lambda: client.eval_one_episode(env, Client(bridge)))
    initial = bridge.first_obs(1)[OBSERVATION_KEY]["source_id"]
    obs, _, done, _ = BridgeEnv(bridge).step({"joint": [0.]})
    thread.join(1)
    assert results == [None] and done
    assert obs[OBSERVATION_KEY]["source_id"] != initial
    assert obs[OBSERVATION_KEY]["action_receipt"]["status"] == "completed"


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
def test_capture_synchronizes_stale_renderer_without_physics_or_native_steps(adapter, monkeypatch):
    env = IsaacEnvironment()
    install_renderer(monkeypatch, env)
    client = deploy(adapter)
    env.take_action({"target": 1})
    assert env.rendered_step == -1
    for _ in range(2):
        obs, evidence = client._capture_observation(env)
        assert evidence["status"] == "aligned"
        assert evidence["native_step_before"] == evidence["native_step_after"] == 1
        assert evidence["physics_step_before"] == evidence["physics_step_after"] == 121
        assert evidence["physics_time_before"] == evidence["physics_time_after"] == 2.1
        assert obs["vision"]["cam_head"]["color"][0, 0, 0] == 1
        assert obs["vision"]["cam_head"]["extrinsic_matrix"][0, 3] == 1
    assert len(env.actions) == 1
    assert env.forward_calls == len(env.render_calls) == 2


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
def test_capture_detects_physics_advance_instead_of_asserting_alignment(adapter, monkeypatch):
    env = IsaacEnvironment()
    original = env.render

    def unsafe_render(**kwargs):
        original(**kwargs)
        env.sim.sim.current_time_step_index += 1

    env.render = unsafe_render
    install_renderer(monkeypatch, env)
    _, evidence = deploy(adapter)._capture_observation(env)
    assert evidence["status"] == "mismatched"
    assert "advanced" in evidence["reason"]
    assert not env.actions


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
def test_unsupported_capture_reports_unknown_not_guessed_picture_lag(adapter):
    env = Environment()
    _, evidence = deploy(adapter)._capture_observation(env)
    assert evidence["status"] == "unknown"
    obs = observation(1)
    obs[OBSERVATION_KEY]["alignment"] = evidence
    bridge = Bridge()
    bridge.update_obs(obs)
    source = bridge.get_obs()[OBSERVATION_KEY]
    assert source["source_id"] == "client:1"  # packet identity remains useful
    assert source["image_source_id"] is None and not source["aligned"]


def test_explicit_alignment_overrides_module_based_legacy_lag_hint():
    obs = observation(1, behind=1)
    obs[OBSERVATION_KEY]["alignment"] = {
        "status": "aligned", "basis": "isaac_render_product_fence",
    }
    bridge = Bridge()
    bridge.update_obs(obs)
    source = bridge.get_obs()[OBSERVATION_KEY]
    assert source["aligned"] and source["image_source_id"] == "client:1"
    assert bridge.get_obs()["image_state"] == obs["state"]


@pytest.mark.parametrize("alignment", [None, {}, {"status": "aligned"},
                                       {"status": "unknown", "basis": "unavailable"}])
def test_explicit_alignment_requires_nonempty_basis(alignment):
    obs = observation(1)
    obs[OBSERVATION_KEY]["alignment"] = alignment
    bridge = Bridge()
    bridge.update_obs(obs)
    assert not bridge.get_obs()[OBSERVATION_KEY]["aligned"]


def test_standalone_adapter_capture_implementations_stay_identical():
    root = Path(__file__).resolve().parents[1] / "integrations/xpolicylab"
    assert (root / "PyRUA_Lean/deploy.py").read_text() == (
        root / "PyRUA_RSI/deploy.py"
    ).read_text()


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
def test_skipped_render_is_not_successful_capture(adapter, monkeypatch):
    env = IsaacEnvironment()
    install_renderer(monkeypatch, env)
    env.missing_products = {"/Render/head", "/Render/wrist"}
    _, evidence = deploy(adapter)._capture_observation(env)
    assert evidence["status"] == "unknown" and evidence["render_status"] == "incomplete"
    assert "completion" in evidence["reason"]
    assert len(env.render_calls) == 8 and not env.actions
    assert env.annotator_syncs == 0


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
@pytest.mark.parametrize("lag,missing", [(20, set()), (0, {"/Render/wrist"})])
def test_capture_requires_new_completion_for_every_camera(adapter, monkeypatch, lag, missing):
    env = IsaacEnvironment()
    env.frame_lag = lag
    env.missing_products = missing
    install_renderer(monkeypatch, env)
    _, evidence = deploy(adapter)._capture_observation(env)
    assert evidence["status"] == "unknown"
    assert evidence["render_status"] == "incomplete"
    assert len(env.render_calls) == 8
    assert not env.capture_sync
    assert all(env.capture_settings.values())  # original render settings restored
    assert not env.actions


@pytest.mark.parametrize("adapter", ["PyRUA_Lean", "PyRUA_RSI"])
def test_capture_completes_annotators_and_restores_settings(adapter, monkeypatch):
    env = IsaacEnvironment()
    install_renderer(monkeypatch, env)
    _, evidence = deploy(adapter)._capture_observation(env)
    assert evidence["status"] == "aligned"
    assert evidence["annotators_synchronized"] and env.annotator_syncs == 1
    assert evidence["first_render_frame"] == 101
    assert evidence["render_frames"] == {"/Render/head": 101, "/Render/wrist": 101}
    assert not env.capture_sync and all(env.capture_settings.values())
