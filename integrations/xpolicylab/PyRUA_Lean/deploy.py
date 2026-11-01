"""The evaluation loop of the PyRUA_Lean adapter.

The policy is a program a language model writes while the episode runs: between two actions it
may think for minutes, and it decides itself when it has finished.  The loop is demo_policy's
with three differences, each because of that:

* ``get_action`` is asked again while it answers "wait" (the policy server never holds one
  request longer than a few seconds, so the client's request timeout stays what it is);
* an empty answer means the program has finished: the episode ends there and RoboDojo judges
  the scene as it stands;
* every observation carries the episode's step count and limit plus explicit capture
  alignment evidence; the benchmark's verdict is reported to the policy server once
  the episode is over (the protocol's ``trial_end``).

The observation is the one RoboDojo's client builds. Capture first synchronizes the
renderer without advancing physics, and records the alignment evidence. Actions
remain joint targets.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any


def _capture_observation(task_env: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Acquire current camera buffers with a bounded, read-only render fence.

    RoboDojo's tiled cameras attach annotators without a Replicator writer graph;
    orchestrator.step can wait forever for a frame that graph never schedules.
    Instead observe completed render events for every actual camera product, after
    this capture's Fabric update. Isaac's sim.render disables physics updates.
    This helper is duplicated in each independently installable adapter on purpose.
    """
    sim = getattr(getattr(task_env, "sim", None), "sim", None)
    evidence: dict[str, Any] = {"status": "unknown", "basis": "unavailable"}
    cameras = getattr(getattr(task_env, "capture_manager", None), "tiled_cameras", ())
    products = [getattr(camera, "_render_product_path", None) for camera in cameras]
    if (sim is None or not callable(getattr(sim, "forward", None))
            or not callable(getattr(sim, "render", None)) or not products or not all(products)):
        evidence["reason"] = "environment has no supported synchronous capture capability"
        return task_env.get_obs(), evidence
    try:
        import carb.eventdispatcher
        import carb.settings
        import omni.graph.core as og
        import omni.kit.app as kit_app
        import omni.kit.renderer_capture as renderer_capture
        import omni.usd as usd
        from omni.syntheticdata.scripts import helpers
    except ImportError:
        evidence["reason"] = "Isaac render-completion observation is unavailable"
        return task_env.get_obs(), evidence

    def counters():
        return {
            "physics_time": float(sim.current_time),
            "physics_step": int(sim.current_time_step_index),
            "native_step": _status(task_env).get("step"),
        }

    try:
        before = counters()
        if not math.isfinite(before["physics_time"]):
            raise ValueError("nonfinite physics clock")
    except (AttributeError, TypeError, ValueError, OverflowError):
        evidence["reason"] = "simulation does not expose valid physics capture counters"
        return task_env.get_obs(), evidence
    evidence.update(basis="isaac_render_product_fence", **{
        f"{key}_before": value for key, value in before.items()
    })
    completed: dict[str, int] = {}
    first_frame: int | None = None
    iface = helpers._get_syntheticdata_iface()

    def on_pre_update(event):
        nonlocal first_frame
        if first_frame is None:
            frame = event["SWHFrameNumber"] or event["frameNumber"]
            if frame is not None:
                first_frame = int(frame)

    def on_render(event):
        try:
            path = iface.parse_rendered_simulation_event(
                event["product_path_handle"], event["results"]
            )[0]
            frame = int(event["swh_frame_number"])
        except (KeyError, TypeError, ValueError):
            return  # unrelated viewport event or unsupported event payload
        if path in products and first_frame is not None and frame >= first_frame:
            completed[path] = frame

    settings = carb.settings.get_settings()
    changes = {"/app/asyncRendering": False, "/app/asyncRenderingLowLatency": False,
               "/rtx/ecoMode/enabled": False}
    previous = {key: settings.get(key) for key in changes}
    renderer = renderer_capture.acquire_renderer_capture_interface()
    previous_sync = renderer.set_capture_sync(True)
    dispatcher = carb.eventdispatcher.get_eventdispatcher()
    subscriptions = []
    try:
        for key, value in changes.items():
            settings.set(key, value)
        subscriptions.append(dispatcher.observe_event(
            event_name=kit_app.GLOBAL_EVENT_PRE_UPDATE, on_event=on_pre_update,
            observer_name="pyrualean.capture.start",
        ))
        subscriptions.append(dispatcher.observe_event(
            event_name=usd.get_context().stage_rendering_event_name(
                usd.StageRenderingEventType.NEW_FRAME
            ), on_event=on_render, observer_name="pyrualean.capture.complete",
        ))
        sim.forward()
        deadline = time.monotonic() + 5.0
        for _ in range(8):
            sim.render()  # flush Fabric + app update with playSimulations=False
            if len(completed) == len(products) or time.monotonic() >= deadline:
                break
        evidence.update(first_render_frame=first_frame, render_frames=dict(completed),
                        render_products=products)
        if len(completed) == len(products):
            evidence["render_status"] = "completed"
        else:
            evidence.update(
                render_status="incomplete",
                reason="camera render completion was not observed within capture budget",
            )
        if evidence["render_status"] == "completed":
            # Match Isaac Camera's acquisition callback: finish the SDG annotator
            # graph before reading its GPU buffers, not only the renderer itself.
            controller = og.Controller()
            controller.evaluate_sync(
                graph_id=controller.graph("/Render/PostProcess/SDGPipeline")
            )
            evidence["annotators_synchronized"] = True
        # The original observation builder reads its normal images/state/calibration;
        # any extra render it does still uses this fixed physical state.
        obs = task_env.get_obs()
    finally:
        subscriptions.clear()  # unsubscribe before the next acquisition
        renderer.set_capture_sync(previous_sync)
        for key, value in previous.items():
            if value is None:
                settings.destroy_item(key)
            else:
                settings.set(key, value)
    after = counters()
    evidence.update({f"{key}_after": value for key, value in after.items()})
    if before != after:
        evidence.update(status="mismatched", reason="physical state advanced during capture")
    elif evidence["render_status"] == "completed":
        evidence["status"] = "aligned"
    return obs, evidence


def _status(task_env: Any, env_idx: int = 0) -> dict[str, Any]:
    info: dict[str, Any] = {}
    # RoboDojo's environment counts per environment; XPolicyLab's debug environment has one
    # counter and its own names
    taken = getattr(task_env, "take_action_cnt", None)
    try:
        info["step"] = int(taken[env_idx])
    except (TypeError, IndexError, ValueError):
        if isinstance(getattr(task_env, "episode_step", None), int):
            info["step"] = task_env.episode_step
    for name in ("step_lim", "episode_step_limit"):
        try:
            info["step_limit"] = int(getattr(task_env, name))
            break
        except (TypeError, ValueError, AttributeError):
            continue
    for name in ("task_name", "eval_seed"):
        value = getattr(task_env, name, None)
        if value is not None:
            info[name] = value
    seeds = getattr(task_env, "env_seeds", None)
    try:
        info["layout"] = int(seeds[env_idx])
    except (TypeError, IndexError, ValueError):
        pass
    return info


def _waiting(actions: Any) -> bool:
    return (
        isinstance(actions, (list, tuple))
        and len(actions) == 1
        and isinstance(actions[0], dict)
        and actions[0].get("pyrualean") == "wait"
    )


def _judge_as_it_stands(task_env: Any) -> None:
    """End an episode the program has finished: RoboDojo runs its final checks now."""

    if task_env.is_episode_end() or not hasattr(task_env, "success"):
        return  # over already, or an environment with nothing to judge (the debug one)
    running = (
        task_env.get_running_env_idx_list()
        if hasattr(task_env, "get_running_env_idx_list")
        else [0]
    )
    for env_idx in running:
        task_env.success[env_idx] = False
    task_env.is_episode_end()


def eval_one_episode(TASK_ENV: Any, model_client: Any) -> None:
    model_client.call(func_name="reset")
    source_session = uuid.uuid4().hex
    sequence = 0
    receipt = None
    refresh_id = None

    def capture():
        nonlocal sequence
        obs, alignment = _capture_observation(TASK_ENV)
        sequence += 1
        info = dict(obs.get("additional_info") or {})
        info["pyrualean"] = _status(TASK_ENV)
        obs["additional_info"] = info
        obs["pyrualean_observation"] = {
            "protocol": 1,
            "source_session_id": source_session,
            "source_id": f"{source_session}:{sequence}",
            "sequence": sequence,
            # Acquisition completion in the client's clock; never subtract a host clock.
            "captured_at": time.time(),
            "capture_clock": "client_unix_utc",
            "source_freshness": "captured",
            "alignment": alignment,
            "action_receipt": receipt,
            "refresh_id": refresh_id,
        }
        return obs

    try:
        while not TASK_ENV.is_episode_end():
            obs = capture()
            model_client.call(func_name="update_obs", obs=obs)
            actions = model_client.call(func_name="get_action")
            while _waiting(actions):
                actions = model_client.call(func_name="get_action")
            if not actions:
                break  # the program has finished
            if len(actions) == 1 and actions[0].get("pyrualean") == "observe":
                refresh_id = actions[0].get("refresh_id")
                continue  # a read-only capture: no hold/motion/native budget consumed
            refresh_id = None
            for action in actions:
                action = dict(action)
                request = action.pop("__pyrualean_request__", None)
                receipt = None
                if isinstance(request, dict):
                    receipt = {key: request.get(key) for key in
                               ("session_id", "action_id", "native_action_id")}
                    receipt["status"] = "unknown"
                TASK_ENV.take_action(action)
                if receipt is not None:
                    receipt["status"] = "completed"
                if TASK_ENV.is_episode_end():
                    break
    finally:
        _judge_as_it_stands(TASK_ENV)
        verdict = dict(_status(TASK_ENV))
        verdict["source_session_id"] = source_session
        try:
            verdict["success"] = bool(TASK_ENV.success[0])
        except (TypeError, IndexError, AttributeError):
            verdict["success"] = False
        try:
            verdict["pyrualean_final_observation"] = capture()
        except Exception:  # an ended/dead simulator may no longer permit capture
            pass  # no invented terminal observation or execution acknowledgment
        try:
            model_client.call(func_name="trial_end", obs=verdict)
        except Exception as exc:  # noqa: BLE001 - the trial stands whether or not it was heard
            print(f"[PyRUA_Lean] the verdict did not reach the policy server: {exc}", flush=True)


def eval_one_episode_batch(TASK_ENV: Any, model_client: Any) -> None:
    num_envs = int(getattr(TASK_ENV, "num_envs", 1) or 1)
    if num_envs > 1:
        raise RuntimeError(
            f"RoboDojo started {num_envs} environments; PyRUA_Lean drives one conversation per "
            "environment.  Keep eval_batch: false in deploy.yml."
        )
    eval_one_episode(TASK_ENV, model_client)
