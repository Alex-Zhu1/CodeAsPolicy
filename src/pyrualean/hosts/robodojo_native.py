# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""A RoboDojo toolkit that needs nothing but an environment.

RPent's ``RoboDojoToolkit`` wraps RPent's simulator bridge and records RPent's artifacts.  Under
RoboDojo's own evaluation client there is no RPent: there is an observation (three colour
images and the robot's state, with a depth image and the cameras' matrices when the client's
configuration switches them on) and an action that takes joint targets.
:class:`NativeToolkit` is what the robot class needs on top of that and no more: the tools of
:mod:`.robodojo_tools` (through the depth image where the observation has one, from colour
alone where it has not), ``segment``, the recorded steps (``view_env_state``, the images and
the cameras' calibration of each step) and the envelope RPent's toolkit wraps a
state-changing tool's result in, which :func:`pyrualean._backend.unwrap` reads.

``env`` is anything with ``get_obs() -> dict`` (``vision`` / ``state`` / ``instruction``),
``step(action) -> (obs, reward, done, info)`` with ``info["status"]`` holding ``step`` and
``step_limit``, and ``get_status()``.  ``sam3_client`` segments an image by words or by a pixel
(RPent's ``Sam3Client`` interface).
"""

from __future__ import annotations

import copy
import io
import json
import threading
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from . import robodojo_tools as tools

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATEFUL = frozenset({"move_pose", "grip", "home", "motion_execute"})
PERCEPTION = frozenset({"locate", "back_project", "segment"})


class Cancelled(Exception):
    """The episode was stopped while a tool was running."""


class _Primitives:
    """What the tools of :mod:`.robodojo_tools` are bound to."""

    def __init__(self, env: Any, sam3_client: Any, colour_only: bool = True) -> None:
        self.env = env
        self.sam3_client = sam3_client
        #: no depth image is read, even where the observation has one
        self.colour_only = colour_only
        self._last_obs: dict[str, Any] | None = None
        self.cancelled = threading.Event()

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled()


class _FrozenEnvironment:
    """A recorded sensor packet; perception cannot reacquire or move the live scene."""

    def __init__(self, obs: dict[str, Any], status: dict[str, Any]) -> None:
        self._obs = obs
        self._status = status

    def get_obs(self) -> dict[str, Any]:
        return self._obs

    def get_status(self) -> dict[str, Any]:
        return dict(self._status)

    def step(self, *args, **kwargs):
        raise RuntimeError("A perception snapshot cannot execute physical actions")


class _SnapshotPrimitives:
    """Share source-keyed perception caches, but never the live environment."""

    _LOCAL = frozenset({"_parent", "env", "_last_obs", "_snapshot_context"})

    def __init__(self, parent, obs, metadata) -> None:
        object.__setattr__(self, "_parent", parent)
        self.env = _FrozenEnvironment(obs, {"step": metadata.get("native_steps")})
        self._last_obs = obs
        self._snapshot_context = copy.deepcopy(metadata)

    def __getattr__(self, name):
        return getattr(self._parent, name)

    def __setattr__(self, name, value):
        if name in self._LOCAL:
            object.__setattr__(self, name, value)
        else:
            setattr(self._parent, name, value)


def _save_packet(folder: Path, obs: dict[str, Any]) -> None:
    """Persist arrays losslessly without pickle or retaining a sensor history in RAM."""
    arrays: dict[str, np.ndarray] = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("Sensor packet arrays cannot contain Python objects")
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, dict):
            return {"mapping": {str(key): encode(item) for key, item in value.items()}}
        if isinstance(value, (tuple, list)):
            return {"sequence": [encode(item) for item in value]}
        return {"value": _plain(value)}

    packet = encode(obs)
    np.savez_compressed(folder / "sensors.npz", **arrays)
    (folder / "sensors.json").write_text(json.dumps(packet) + "\n")


def _load_packet(folder: Path) -> dict[str, Any]:
    packet = json.loads((folder / "sensors.json").read_text())
    with np.load(folder / "sensors.npz", allow_pickle=False) as arrays:
        def decode(value):
            if "array" in value:
                array = arrays[value["array"]].copy()
                array.setflags(write=False)
                return array
            if "mapping" in value:
                return {key: decode(item) for key, item in value["mapping"].items()}
            if "sequence" in value:
                return [decode(item) for item in value["sequence"]]
            return value["value"]

        return decode(packet)


def _png(image: Any) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image)[..., :3].astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _plain(value: Any) -> Any:
    """A state entry as plain lists and numbers (it is written to JSON and read back)."""

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Steps:
    """Recorded endpoints, with state, source metadata, calibration and images on disk."""

    def __init__(self, out_dir: Path, session_id: str | None = None) -> None:
        self.root = Path(out_dir) / "steps"
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.session_id = session_id or uuid.uuid4().hex

    def record(
        self, obs: dict[str, Any], status: dict[str, Any], action_id: str | None = None
    ) -> int:
        # Keep PNGs, state JSON and the lossless sensor packet on one detached value,
        # even if a native environment reuses and mutates its camera buffers later.
        obs = copy.deepcopy(obs)
        step = len(self.records)
        folder = self.root / f"{step:04d}"
        folder.mkdir(exist_ok=True)
        _save_packet(folder, obs)
        for name, cam in (obs.get("vision") or {}).items():
            color = (cam or {}).get("color")
            if color is not None:
                (folder / f"{name}.png").write_bytes(_png(color))
        try:
            spent = int(status.get("step", 0)) >= int(status.get("step_limit", 1 << 30))
        except (TypeError, ValueError):
            spent = False
        cameras = {}
        for name, cam in (obs.get("vision") or {}).items():
            cam = cam or {}
            pose = tools.extrinsic_of(cam)
            cameras[name] = {
                "shape": _plain(cam.get("shape")),
                "has_depth": tools.depth_image(cam) is not None,
                "intrinsic_matrix": _plain(cam.get("intrinsic_matrix")),
                "extrinsic_matrix": _plain(pose),
            }
        source = dict(obs.get("pyrualean_observation") or {})
        source_id = source.get("source_id")
        native_steps = source.get("native_steps", status.get("step"))
        observation_id = (
            source.get("observation_id") or source_id
            or f"{self.session_id}:unverified:{native_steps}"
        )
        aligned = source.get("aligned", False)
        image_source = source.get("image_source_id", source_id)
        receipt = source.get("action_receipt") or {}
        images = {}
        for name, cam in (obs.get("vision") or {}).items():
            cam = cam or {}
            if cam.get("color") is None:
                continue
            # An incomplete supplied calibration is an invalid packet calibration,
            # never permission to relabel it as the built-in robot model.
            has_matrices = (cam.get("intrinsic_matrix") is not None
                            or tools.extrinsic_of(cam) is not None)
            depth_key = next((key for key in (
                "distance_to_image_plane", "depth", "approximate_depth"
            ) if cam.get(key) is not None), None)
            model = ("robodojo_sight.head_view:v1" if name == "cam_head" else
                     "robodojo_sight.wrist_view:v1" if name in CAMERAS else None)
            calibration_source = (image_source if has_matrices or name != "cam_head"
                                  else model)
            images[name] = dict(
                source_id=image_source, image_source_id=image_source,
                artifact=f"{name}.png", aligned=aligned,
                sensor_artifact="sensors.npz", sensor_manifest="sensors.json",
                sensor_path=f"vision.{name}",
                calibration_source_id=calibration_source,
                calibration_method="packet_matrices" if has_matrices else "robot_model",
                calibration_model=None if has_matrices else model,
                calibration_artifact="observation.json",
                calibration_path=f"cameras.{name}",
                depth_artifact="sensors.npz" if depth_key else None,
                depth_path=f"vision.{name}.{depth_key}" if depth_key else None,
                depth_source_id=image_source if depth_key else None,
                image_state_source_id=image_source,
                image_state_artifact="observation.json", image_state_path="image_state",
            )
        metadata = dict(
            schema="pyrualean.observation.v1", session_id=self.session_id,
            observation_id=observation_id, source_id=source_id,
            step=step, native_steps=native_steps, action_id=action_id or receipt.get("action_id"),
            captured_at=source.get("captured_at"), capture_clock=source.get("capture_clock"),
            recorded_at=time.time(), source_freshness=source.get("source_freshness", "unverified"),
            aligned=aligned,
            images=images,
        )
        if "alignment" in source:
            metadata["alignment"] = copy.deepcopy(source["alignment"])
        record = {
            "state": _plain(obs.get("state") or {}),
            "image_state": _plain(obs.get("image_state") or obs.get("state") or {}),
            "instruction": str(obs.get("instruction") or obs.get("instructions") or ""),
            "truncated": bool(spent or status.get("over")),
            "cameras": cameras,
            "observation": metadata,
        }
        self.records.append(record)
        (folder / "observation.json").write_text(json.dumps(record, indent=2) + "\n")
        return step

    def index(self, step: int) -> int:
        if not self.records:
            raise IndexError("no step recorded")
        index = len(self.records) + step if step < 0 else step
        if not 0 <= index < len(self.records):
            raise IndexError(f"step {step} is not recorded (0 .. {len(self.records) - 1})")
        return index

    def artifact_path(self, name: str, step: int = -1) -> Path:
        return self.root / f"{self.index(step):04d}" / name

    def load_bytes(self, name: str, step: int = -1) -> bytes:
        return self.artifact_path(name, step).read_bytes()

    def resolve_observation(
        self, observation_id: str | None, session_id: str | None,
        observation_step: int | None = None,
    ) -> int:
        """Resolve exactly the requested recorded packet; never substitute the latest."""
        if session_id is not None and session_id != self.session_id:
            raise ValueError("observation session_id belongs to another session")
        if observation_id is not None and (not isinstance(observation_id, str)
                                           or not observation_id):
            raise ValueError("observation_id must be a non-empty string")
        if observation_id is not None and session_id is None:
            raise ValueError("session_id is required with observation_id")
        if observation_step is not None:
            if (not isinstance(observation_step, int) or isinstance(observation_step, bool)
                    or observation_step < 0):
                raise ValueError("observation_step must be a recorded nonnegative integer")
            index = self.index(observation_step)
            if observation_id is None:
                raise ValueError("observation_id is required with observation_step")
            if self.records[index]["observation"]["observation_id"] != observation_id:
                raise ValueError("observation_id does not match observation_step")
            return index
        if observation_id is None:
            return self.index(-1)
        for index in reversed(range(len(self.records))):
            if self.records[index]["observation"]["observation_id"] == observation_id:
                return index
        raise ValueError("unknown observation_id; no recorded sensor packet matches it")

    def load_observation(self, step: int) -> dict[str, Any]:
        return _load_packet(self.root / f"{self.index(step):04d}")


def segment(
    primitives: Any, text_prompt: str, camera: str = "cam_head", min_score: float = 0.2
) -> dict[str, Any]:
    """Segment an object by words in the current image of one camera (RPent's tool, as is)."""

    obs = tools._obs(primitives)
    cam = (obs.get("vision") or {}).get(camera)
    if cam is None:
        return {"error": f"camera {camera!r} not in observation"}
    color = cam.get("color")
    if color is None:
        return {"error": "camera color image missing"}
    sam3 = getattr(primitives, "sam3_client", None)
    if sam3 is None:
        return {"error": "sam3_client not configured"}
    res = sam3.segment(color, text_prompt=text_prompt, min_score=float(min_score))
    if not res.found:
        return {"camera": camera, "found": False, "text_prompt": text_prompt}
    mask = np.asarray(res.mask, dtype=bool)
    rows, cols = np.nonzero(mask)
    count = int(rows.size)
    return {
        "camera": camera,
        "found": True,
        "score": float(res.score) if res.score is not None else None,
        "box_px": [float(v) for v in res.box] if res.box is not None else None,
        "mask_shape": list(mask.shape),
        "mask_moments": [count, int(rows.sum()), int(cols.sum())],
        "centroid_rc": [int(rows.sum()) // count, int(cols.sum()) // count] if count else None,
    }


class NativeToolkit:
    """The robot's tools over a bare environment, with the steps they record."""

    def __init__(
        self,
        env: Any,
        sam3_client: Any,
        out_dir: Path,
        verdict: Callable[[], bool] | None = None,
        colour_only: bool | None = None,
    ) -> None:
        obs = env.get_obs()
        if colour_only is None:
            # what the observation offers decides: depth where the client sends it
            head = (obs.get("vision") or {}).get("cam_head") or {}
            colour_only = tools.depth_image(head) is None
        self._primitives = _Primitives(env, sam3_client, colour_only)
        bridge = getattr(env, "bridge", None)
        self.session_id = getattr(bridge, "session_id", None) or uuid.uuid4().hex
        self.state = Steps(out_dir, self.session_id)
        self._verdict = verdict
        self._lock = threading.Lock()
        self._receipts: dict[str, dict[str, Any]] = {}
        self._active_action: str | None = None
        self._primitives._last_obs = obs
        start = {arm: tools._joints(obs.get("state") or {}, arm) for arm in tools.ARMS}
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "move_pose": partial(tools.move_pose, self._primitives, start),
            "grip": partial(tools.grip, self._primitives),
            "home": partial(tools.home, self._primitives, start),
            "locate": partial(tools.locate, self._primitives),
            "back_project": partial(tools.back_project, self._primitives),
            "segment": partial(segment, self._primitives),
        }
        from .robodojo_coordination import handlers

        self._tools.update(handlers(self._primitives))
        self.state.record(obs, self._status())

    def _status(self) -> dict[str, Any]:
        try:
            status = self._primitives.env.get_status()
        except Exception:  # noqa: BLE001 - optional metadata
            return {}
        return status if isinstance(status, dict) else {}

    def _envelope(self, step: int, tool: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        index = self.state.index(step)
        record = self.state.records[index]
        return copy.deepcopy({
            "step": index,
            "task_language": record["instruction"],
            "terminated": False,
            "truncated": record["truncated"],
            "state": {
                "state": record["state"],
                "instruction": record["instruction"],
                "cameras": record["cameras"],
                "image_state": record["image_state"],
            },
            "observation": dict(record["observation"]),
            "log": {"tool": tool, "result": payload},
        })

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> dict[str, Any]:
        return self.execute_tool_with_context(name, input_dict, None)

    def execute_tool_with_context(
        self, name: str, input_dict: dict[str, Any], context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Run one tool; a state-changing one answers with the step it recorded."""

        kwargs = dict(input_dict or {})
        if name == "view_env_state":
            try:
                return self._envelope(int(kwargs.get("step", -1)), name, {})
            except IndexError as exc:
                return {"error": str(exc)}
        handler = self._tools.get(name)
        if handler is None:
            return {"error": f"this RoboDojo host has no tool {name!r}"}
        with self._lock:
            if name in PERCEPTION:
                return self._perceive(name, kwargs)
            stateful = name in STATEFUL
            context = dict(context or {})
            action_id = context.get("action_id") or uuid.uuid4().hex
            context.setdefault("session_id", self.session_id)
            context.setdefault("action_id", action_id)
            if context["session_id"] != self.session_id:
                raise ValueError("action context belongs to another session")
            if stateful and action_id in self._receipts:
                raise ValueError("action identity was already used; reconcile instead of replaying")
            if stateful:
                obs = tools._obs(self._primitives)
                source = obs.get("pyrualean_observation") or {}
                status = self._status()
                stale_source = (
                    context.get("source_id") is not None
                    and context["source_id"] != source.get("source_id")
                )
                stale_step = (
                    context.get("source_native_steps") is not None
                    and status.get("step") is not None
                    and context["source_native_steps"] != status["step"]
                )
                if stale_source or stale_step:
                    step = self.state.record(obs, status)
                    result = self._envelope(step, name, dict(
                        error="stale observation source; refresh before another action",
                        code="stale_observation", reached=False, steps_used=0,
                    ))
                    receipt = dict(
                        schema="pyrualean.action.v1", session_id=self.session_id,
                        action_id=action_id, status="rejected", effect_known=True,
                        execution_complete=True,
                        observation_id=result["observation"]["observation_id"],
                        native_action_ids=[], stop_confirmed=None,
                        reason="stale_observation",
                    )
                    self._receipts[action_id] = receipt
                    result["action_receipt"] = dict(receipt)
                    return result
            before_revision = getattr(self._primitives, "_motion_revision", 0)
            failed = None
            payload = {}
            if stateful:
                self._active_action = action_id
                self._receipts[action_id] = dict(
                    schema="pyrualean.action.v1", session_id=self.session_id,
                    action_id=action_id, status="running", effect_known=False,
                    execution_complete=False, observation_id=context.get("observation_id"),
                    native_action_ids=[], stop_confirmed=None)
                if hasattr(self._primitives.env, "action_context"):
                    self._primitives.env.action_context = context
            try:
                payload = handler(**kwargs)
            except Cancelled:
                payload = {"interrupted": True, "code": "tool_cancelled"}
            except TypeError as exc:
                payload = {"error": f"{name}: {exc}"}
                failed = exc
            except BaseException as exc:
                failed = exc
                raise
            finally:
                if stateful:
                    self._active_action = None
                    if hasattr(self._primitives.env, "action_context"):
                        self._primitives.env.action_context = None
                    self._finalize_receipt(action_id, payload, failed, before_revision)
            if not stateful:
                return payload
            obs = tools._obs(self._primitives)
            step = self.state.record(obs, self._status(), action_id=action_id)
            result = self._envelope(step, name, payload)
            receipt = self._receipts[action_id]
            receipt["observation_id"] = result["observation"]["observation_id"]
            result["observation"]["action_id"] = action_id
            result["action_receipt"] = dict(receipt)
            return result

    def _perceive(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Run every nested perception read against a single persisted sensor packet."""
        observation_id = kwargs.pop("observation_id", None)
        session_id = kwargs.pop("session_id", None)
        observation_step = kwargs.pop("observation_step", None)
        try:
            step = self.state.resolve_observation(observation_id, session_id, observation_step)
            obs = self.state.load_observation(step)
        except (ValueError, IndexError, OSError) as exc:
            return {"error": str(exc), "code": "invalid_observation"}
        metadata = copy.deepcopy(self.state.records[step]["observation"])
        frozen = _SnapshotPrimitives(self._primitives, obs, metadata)
        handler = segment if name == "segment" else getattr(tools, name)
        try:
            payload = handler(frozen, **kwargs)
        except TypeError as exc:
            payload = {"error": f"{name}: {exc}"}
        geometry = payload.get("geometry_provenance") or {}
        camera = kwargs.get("camera", "cam_head")
        used_cameras = geometry.get("used_cameras") or [camera]
        provenance = dict(
            metadata, schema="pyrualean.perception.v1", operation=name,
            camera=camera, method=geometry.get("method", "image_segmentation" if name == "segment"
                                              else "unresolved"),
            frozen=True, is_current_record=step == self.state.index(-1),
            images={key: copy.deepcopy(metadata["images"][key])
                    for key in used_cameras if key in metadata["images"]},
            sensor_manifest="sensors.json", sensor_artifact="sensors.npz",
            image_state_artifact="observation.json", image_state_path="image_state",
        )
        if geometry:
            provenance["geometry"] = copy.deepcopy(geometry)
        payload["provenance"] = provenance
        return payload

    def _finalize_receipt(self, action_id, payload, failed, before_revision):
        receipt = self._receipts[action_id]
        bridge = getattr(self._primitives.env, "bridge", None)
        native = bridge.receipts(action_id) if bridge is not None else []
        attempted = getattr(self._primitives, "_motion_revision", 0) > before_revision
        uncertain = bool(
            failed or str(payload.get("stop_reason", "")).startswith("controller_error")
        )
        if native:
            known = all(r["effect_known"] and r["execution_complete"] for r in native)
            rejected = all(r["status"] == "rejected" for r in native)
        else:
            known = not (uncertain and attempted)
            rejected = not attempted and bool(
                failed or payload.get("error") or payload.get("interrupted")
                or payload.get("status") == "rejected"
            )
        partial = payload.get("status") == "partial" or (
            attempted and (payload.get("reached") is False or payload.get("interrupted"))
        )
        status = "unknown" if not known else "rejected" if rejected else (
            "partial" if partial or uncertain else "completed"
        )
        receipt.update(
            status=status,
            effect_known=known, execution_complete=known,
            native_action_ids=[r["native_action_id"] for r in native],
            native_receipts=native, payload_status=payload.get("status"),
        )
        if failed is not None:
            receipt["error"] = f"{type(failed).__name__}: {failed}"

    def reconcile(self, action_id: str) -> dict[str, Any]:
        receipt = self._receipts.get(action_id)
        if receipt is None:
            return dict(schema="pyrualean.action.v1", session_id=self.session_id,
                        action_id=action_id, status="unknown", effect_known=False,
                        execution_complete=False, error="unknown action identity")
        bridge = getattr(self._primitives.env, "bridge", None)
        if bridge is not None:
            native = bridge.receipts(action_id)
            receipt["native_receipts"] = native
            if (
                receipt["status"] in ("unknown", "running")
                and action_id != self._active_action and native
                and all(r["effect_known"] and r["execution_complete"] for r in native)
            ):
                # A late native completion resolves transport uncertainty, not the
                # high-level motion's target. The interrupted trajectory remains partial.
                receipt.update(effect_known=True, execution_complete=True,
                               status="rejected" if all(r["status"] == "rejected" for r in native)
                               else "partial",
                               observation_id=native[-1].get("observation_id"))
                # The late frame already exists in the bridge. Record that endpoint
                # without issuing capture/hold/replay; artifacts must not remain at the
                # pre-timeout endpoint after reconciliation says execution has ended.
                if self._lock.acquire(blocking=False):
                    try:
                        if self._active_action is None:
                            obs = tools._obs(self._primitives)
                            step = self.state.record(obs, self._status())
                            current = self.state.records[step]["observation"]
                            receipt["current_observation_id"] = current["observation_id"]
                            receipt["current_observation_step"] = step
                    finally:
                        self._lock.release()
        return dict(receipt)

    def refresh_observation(self) -> dict[str, Any]:
        with self._lock:
            env = self._primitives.env
            refresh = getattr(env, "refresh_observation", None)
            obs = refresh() if callable(refresh) else env.get_obs()
            self._primitives._last_obs = obs
            step = self.state.record(obs, self._status())
            return self._envelope(step, "refresh_observation", {})

    def conclude(self) -> None:
        """The policy has finished: let the environment judge the scene as it stands."""

        conclude = getattr(self._primitives.env, "conclude", None)
        if callable(conclude):
            conclude()

    def solved(self) -> bool:
        return bool(self._verdict()) if self._verdict is not None else False

    def episode_over(self) -> bool:
        """Whether the environment's owner has ended the episode (a client that judges it)."""

        over = getattr(self._primitives.env, "episode_over", None)
        return bool(over()) if callable(over) else False

    def cancel_active_and_wait(self) -> dict[str, Any]:
        self._primitives.cancelled.set()
        bridge = getattr(self._primitives.env, "bridge", None)
        if bridge is not None:
            report = bridge.cancel()
            report["confirmed"] = report["confirmed"] and self._active_action is None
            return report
        return dict(requested=True, confirmed=self._active_action is None,
                    scope="tool_dispatch", physical_stop_confirmed=None)

    def close(self) -> None:
        self._primitives.cancelled.set()


class NativeBackend:
    """A :class:`NativeToolkit` as a :class:`pyrualean.Backend`."""

    def __init__(self, toolkit: NativeToolkit) -> None:
        self._toolkit = toolkit

    supports_snapshot_perception = True

    @property
    def toolkit(self) -> NativeToolkit:
        return self._toolkit

    @property
    def session_id(self) -> str:
        return self._toolkit.session_id

    @property
    def colour_only(self) -> bool:
        return bool(self._toolkit._primitives.colour_only)

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        return self._toolkit.execute_tool(name, input_dict)

    def execute_tool_with_context(self, name, input_dict, context):
        return self._toolkit.execute_tool_with_context(name, input_dict, context)

    def observation_metadata(self, step: int = -1) -> dict[str, Any]:
        """Metadata for recorded artifacts; this never pretends they are a live capture."""
        return copy.deepcopy(
            self._toolkit.state.records[self._toolkit.state.index(step)]["observation"]
        )

    def current_observation_source(self) -> dict[str, Any]:
        """Read the environment's current source identity without recording artifacts.

        BridgeEnv returns its latest received snapshot. Other synchronous native
        environments may implement get_obs as acquisition; either way its identity
        is kept separate from the last recorded image and never relabels that image.
        """
        obs = self._toolkit._primitives.env.get_obs()
        source = obs.get("pyrualean_observation") or {}
        status = ((obs.get("additional_info") or {}).get("pyrualean")
                  or self._toolkit._status())
        native_steps = source.get("native_steps", status.get("step"))
        metadata = dict(
            session_id=self.session_id,
            source_id=source.get("source_id"),
            observation_id=source.get("observation_id") or source.get("source_id")
            or f"{self.session_id}:unverified:{native_steps}",
            native_steps=native_steps,
            source_freshness=source.get("source_freshness", "unverified"),
            aligned=source.get("aligned", False),
        )
        if "alignment" in source:
            metadata["alignment"] = copy.deepcopy(source["alignment"])
        return metadata

    def refresh_observation(self) -> dict[str, Any]:
        return self._toolkit.refresh_observation()

    def reconcile(self, action_id: str) -> dict[str, Any]:
        return self._toolkit.reconcile(action_id)

    def conclude(self) -> None:
        self._toolkit.conclude()

    def solved(self) -> bool:
        return self._toolkit.solved()

    def episode_over(self) -> bool:
        return self._toolkit.episode_over()

    def artifact(self, name: str, step: int = -1) -> bytes:
        return self._toolkit.state.load_bytes(name, step=step)

    def env_steps(self) -> int | None:
        step = self._toolkit._status().get("step")
        try:
            return int(step)
        except (TypeError, ValueError):
            return None

    def cancel(self) -> dict[str, Any]:
        return self._toolkit.cancel_active_and_wait()


__all__ = ["CAMERAS", "Cancelled", "NativeBackend", "NativeToolkit", "Steps", "segment"]
