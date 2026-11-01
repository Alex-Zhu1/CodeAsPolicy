# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Between XPolicyLab's two calls and an environment a program steps.

XPolicyLab's evaluation client owns the simulator and asks a policy for actions: it sends an
observation (``update_obs``) and asks what to do (``get_action``).  A PyRUA-Lean episode is the
other way round: a program calls a tool, the tool steps the environment and reads what came
of it.  :class:`Bridge` turns one into the other.  The policy server answers ``get_action``
with the next native action a tool has asked for; the tool's ``step`` returns when the client
has taken it and sent the observation that followed.

A model thinks for minutes between two actions, far longer than the client waits for one
answer, so ``get_action`` never holds the client longer than ``poll_s``: when no tool has
stepped by then it answers :data:`WAIT`, and the adapter's ``deploy.py`` asks again.  An empty
answer means the program has finished.  ``deploy.py`` adds the episode's step count and limit
to each observation (``additional_info["pyrualean"]``) and reports the benchmark's verdict
when the episode is over (the protocol's ``trial_end``, which the server hands to
``on_trial_end``).

Current adapters synchronize acquisition and report explicit alignment evidence. The
bridge binds ``image_state`` to that capture. For older adapters that only report a
``pictures_behind`` hint, it retains the previous state's wrist-camera compensation;
such lagged images cannot become current evidence merely by being read again.

:class:`PolicyModel` is the object XPolicyLab's policy server serves (its ``Model``);
:func:`serve` runs that server in this process.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from typing import Any

#: ``get_action``'s answer while the program is still thinking.
WAIT = [{"pyrualean": "wait"}]
#: The key ``deploy.py`` puts the episode's status under, in an observation's additional_info.
STATUS_KEY = "pyrualean"
OBSERVATION_KEY = "pyrualean_observation"
REQUEST_KEY = "__pyrualean_request__"


def _source_identity_valid(metadata: dict[str, Any]) -> bool:
    """Protocol support requires an actual acquisition identity, not a version claim."""
    captured = metadata.get("captured_at")
    return (
        type(metadata.get("protocol")) is int and metadata["protocol"] == 1
        and all(isinstance(metadata.get(key), str) and metadata[key].strip()
                for key in ("source_session_id", "source_id", "capture_clock"))
        and type(metadata.get("sequence")) is int and metadata["sequence"] >= 1
        and type(captured) in (int, float) and math.isfinite(captured) and captured > 0
        and metadata.get("source_freshness") == "captured"
    )


class EpisodeOver(RuntimeError):
    """The evaluation client has ended the episode: no observation will follow."""


class Bridge:
    """One episode's meeting point of the evaluation client and the tools."""

    def __init__(self, poll_s: float = 20.0) -> None:
        self._cv = threading.Condition()
        self.poll_s = poll_s
        self._obs: dict[str, Any] | None = None
        self._serial = 0
        self._action: dict[str, Any] | None = None
        self._finished = False
        self._over: dict[str, Any] | None = None
        self._status: dict[str, Any] = {}
        self._pictured: Any = None
        self.session_id = uuid.uuid4().hex
        self._source_session: str | None = None
        self._source_sequence = -1
        self._protocol = False
        self._pending: dict[str, Any] | None = None
        self._receipts: dict[str, dict[str, Any]] = {}
        self._refresh_id: str | None = None
        self._refresh_complete: str | None = None
        self.resets = 0

    # -- the evaluation client's side (the policy server calls these) -------------------------

    def reset(self) -> None:
        with self._cv:
            self.resets += 1
            self._cv.notify_all()

    def update_obs(self, obs: dict[str, Any]) -> None:
        with self._cv:
            raw_metadata = obs.get(OBSERVATION_KEY)
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            protocol = _source_identity_valid(metadata)
            if not protocol:
                if self._source_session is not None:
                    return  # malformed/legacy input cannot downgrade a validated source
                metadata = dict(protocol=None, source_session_id=None, source_id=None,
                                captured_at=None, capture_clock=None,
                                source_freshness="unverified")
            source_session = metadata.get("source_session_id")
            if self._source_session is not None and source_session is None:
                return
            if source_session:
                if self._source_session is None:
                    self._source_session = source_session
                if source_session != self._source_session:
                    return  # a different client's late frame cannot replace this scene
                sequence = metadata.get("sequence")
                if not isinstance(sequence, int) or sequence <= self._source_sequence:
                    return  # duplicated/out-of-order captures are not fresh observations
                self._source_sequence = sequence
            self._protocol = protocol
            info = (obs.get("additional_info") or {}).get(STATUS_KEY)
            previous_step = self._status.get("step")
            if isinstance(info, dict):
                self._status = dict(info)
            # An older RoboDojo client hands out the frame rendered before the current
            # physics state: its pictures are of the robot as the previous observation's
            # state has it (a wrist camera's picture fits that end-effector pose to a
            # millimetre, and the current one only while the arm stands still).  deploy.py
            # says which client this is; a current one's pictures show the state they come with.
            state = obs.get("state")
            poses = {
                name: cam.get(key)
                for name, cam in (obs.get("vision") or {}).items()
                for key in ("extrinsic_matrix", "extrinsics_matrix")
                if isinstance(cam, dict) and cam.get(key) is not None
            }
            explicit_alignment = "alignment" in metadata
            alignment = metadata.get("alignment")
            alignment = alignment if isinstance(alignment, dict) else {}
            aligned_capture = (
                protocol and explicit_alignment and alignment.get("status") == "aligned"
                and isinstance(alignment.get("basis"), str) and bool(alignment["basis"].strip())
            )
            # A new adapter reports the result of its acquisition, not an inference
            # from which optional upstream modules happen to be installed. Unknown
            # evidence must never fall through to the legacy default (behind == 0).
            behind = 0 if explicit_alignment else int(self._status.get("pictures_behind") or 0)
            if (
                behind >= 1 and self._obs is not None and previous_step is not None
                and self._status.get("step") == previous_step
            ):
                # An old client's read-only get_obs can reuse the previously rendered
                # frame. Re-reading it does not move its camera or create new evidence.
                obs["image_state"] = self._obs["image_state"]
                image_source = self._obs[OBSERVATION_KEY].get("image_source_id")
                for name, cam in (obs.get("vision") or {}).items():
                    previous_cam = (self._obs.get("vision") or {}).get(name) or {}
                    if isinstance(cam, dict):
                        for key in ("extrinsic_matrix", "extrinsics_matrix"):
                            if previous_cam.get(key) is not None:
                                cam[key] = previous_cam[key]
            elif behind >= 1 and self._pictured is not None:
                # ... and so are the camera poses it sends along (RoboDojo issue 38): a wrist
                # camera's depth image belongs with the pose of the observation before
                obs["image_state"], then, image_source = self._pictured
                for name, cam in (obs.get("vision") or {}).items():
                    if name in then and isinstance(cam, dict):
                        for key in ("extrinsic_matrix", "extrinsics_matrix"):
                            if cam.get(key) is not None:
                                cam[key] = then[name]
            else:
                obs["image_state"] = state
                image_source = metadata.get("source_id") if (
                    aligned_capture if explicit_alignment else behind == 0
                ) else None
            metadata.update(
                session_id=self.session_id,
                observation_id=(metadata.get("source_id")
                                or f"{self.session_id}:legacy:{self._serial}"),
                image_source_id=image_source,
                native_steps=self._status.get("step"),
                aligned=(aligned_capture if explicit_alignment else protocol and behind == 0),
                received_at=time.time(),
            )
            obs[OBSERVATION_KEY] = metadata
            self._pictured = (state, poses, metadata.get("source_id"))
            self._obs = obs
            self._serial += 1
            self._accept_receipt(metadata.get("action_receipt"), metadata)
            if metadata.get("refresh_id") == self._refresh_id:
                self._refresh_complete = self._refresh_id
            self._cv.notify_all()

    def _accept_receipt(self, receipt: Any, observation: dict[str, Any]) -> None:
        if not isinstance(receipt, dict) or receipt.get("session_id") != self.session_id:
            return
        native_id = receipt.get("native_action_id")
        saved = self._receipts.get(native_id)
        if saved is None or receipt.get("action_id") != saved.get("action_id"):
            return
        # Only an exact acknowledgment for a dispatched request establishes its outcome.
        if (
            saved.get("dispatched") and not saved["execution_complete"]
            and receipt.get("status") == "completed"
        ):
            saved.update(
                status="completed", effect_known=True, execution_complete=True,
                observation_id=observation.get("observation_id"),
            )
            if self._pending is saved:
                self._pending = None

    def get_action(self) -> list[dict[str, Any]]:
        with self._cv:
            self._cv.wait_for(lambda: self._action is not None or self._finished, self.poll_s)
            if self._action is not None:
                action, self._action = self._action, None
                if self._pending is not None:
                    self._pending.update(dispatched=True, status="running")
                self._cv.notify_all()
                return [action]
            return [] if self._finished else WAIT

    def episode_end(self, info: dict[str, Any] | None) -> None:
        with self._cv:
            if (info or {}).get("source_session_id", self._source_session) != self._source_session:
                return
            self._over = dict(info or {})
            final = self._over.pop("pyrualean_final_observation", None)
            if isinstance(final, dict):
                self.update_obs(final)
            self._cv.notify_all()

    # -- the tools' side ------------------------------------------------------------------------

    def first_obs(self, timeout_s: float, alive: Any = None) -> dict[str, Any]:
        """The first observation of the episode; ``alive()`` false means the client died."""

        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._obs is None:
                if self._over is not None:
                    raise EpisodeOver("the episode ended before it began")
                if alive is not None and not alive():
                    raise RuntimeError("the evaluation client exited before its first observation")
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError(
                        f"no observation from the evaluation client in {timeout_s:.0f} s"
                    )
                self._cv.wait(min(left, 2.0))
            return self._obs

    def get_obs(self) -> dict[str, Any]:
        with self._cv:
            if self._obs is None:
                raise RuntimeError("no observation yet")
            return self._obs

    def status(self) -> dict[str, Any]:
        with self._cv:
            status = dict(self._status)
            if self._over is not None:
                status["over"] = True
                # nothing more can be done: the budget is what has been spent
                status["step_limit"] = status.get("step", 0)
            return status

    @property
    def over(self) -> dict[str, Any] | None:
        with self._cv:
            return None if self._over is None else dict(self._over)

    def step(
        self, action: dict[str, Any], timeout_s: float = 600.0,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Have the client take one native action; returns the observation that follows.

        A terminal action returns its final observation when the client acknowledges it.
        Raises :class:`EpisodeOver` if the episode ends without an exact completion receipt,
        or had ended before dispatch; the unresolved receipt remains queryable.
        """

        deadline = time.monotonic() + timeout_s
        with self._cv:
            if self._over is not None or self._finished:
                raise EpisodeOver("the episode is over")
            if not self._protocol:
                raise RuntimeError("client does not support action identity protocol v1")
            if (self._pending is not None or self._action is not None
                    or self._refresh_id is not None):
                raise RuntimeError("an unresolved native action blocks further dispatch")
            context = context or {}
            if context.get("session_id", self.session_id) != self.session_id:
                raise ValueError("action belongs to another session")
            native_id = uuid.uuid4().hex
            receipt = dict(
                session_id=self.session_id, native_action_id=native_id,
                action_id=context.get("action_id") or native_id,
                based_on_observation_id=context.get("observation_id"),
                based_on_source_id=(self._obs or {}).get(OBSERVATION_KEY, {}).get("source_id"),
                status="queued", dispatched=False, effect_known=False, execution_complete=False,
            )
            self._receipts[native_id] = receipt
            self._pending = receipt
            self._action = {**action, REQUEST_KEY: dict(receipt)}
            self._cv.notify_all()
            while not receipt["execution_complete"] and self._over is None:
                if receipt.get("cancel_requested"):
                    receipt["status"] = "unknown"
                    raise TimeoutError(
                        "native action cancellation requested; execution may continue, reconcile "
                        "its receipt before another dispatch"
                    )
                left = deadline - time.monotonic()
                if left <= 0:
                    if not receipt["dispatched"]:
                        self._action = self._pending = None
                        receipt.update(
                            status="rejected", effect_known=True, execution_complete=True,
                        )
                    else:
                        receipt["status"] = "unknown"
                    raise TimeoutError(f"the evaluation client did not answer in {timeout_s:.0f} s")
                self._cv.wait(min(left, 2.0))
            if not receipt["execution_complete"]:
                receipt["status"] = "unknown"
                raise EpisodeOver("the episode ended on this action")
            assert self._obs is not None
            return self._obs

    def receipts(self, action_id: str) -> list[dict[str, Any]]:
        with self._cv:
            return [dict(r) for r in self._receipts.values() if r["action_id"] == action_id]

    def refresh(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Request another capture without a physical action or a repeated dispatch."""
        with self._cv:
            if self._over is not None or self._finished:
                return self.get_obs()
            if not self._protocol:
                raise RuntimeError("client does not support read-only capture protocol v1")
            if self._action is not None or self._refresh_id is not None:
                raise RuntimeError("another bridge request is pending")
            refresh_id = uuid.uuid4().hex
            self._refresh_id = refresh_id
            self._action = {"pyrualean": "observe", "refresh_id": refresh_id}
            self._cv.notify_all()
            complete = self._cv.wait_for(
                lambda: self._refresh_complete == refresh_id or self._over is not None, timeout_s)
            self._refresh_id = None
            if not complete:
                if self._action and self._action.get("refresh_id") == refresh_id:
                    self._action = None
                raise TimeoutError("read-only observation capture timed out")
            return self.get_obs()

    def cancel(self) -> dict[str, Any]:
        with self._cv:
            pending = self._pending
            if pending is not None:
                pending["cancel_requested"] = True
            if pending is not None and not pending["dispatched"]:
                self._action = self._pending = None
                pending.update(status="rejected", effect_known=True, execution_complete=True)
            self._cv.notify_all()
            return dict(requested=True, confirmed=self._pending is None,
                        scope="native_dispatch", physical_stop_confirmed=None)

    def finish(self) -> None:
        """The program has no more actions: the client judges the scene and ends the episode."""

        with self._cv:
            self._finished = True
            self._cv.notify_all()

    def abandon(self) -> None:
        """The client has left this episode (it reset): nothing more will come of it.

        A verdict the client has already reported stands.
        """

        with self._cv:
            self._finished = True
            if self._over is None:
                self._over = {"success": False, "abandoned": True}
            self._cv.notify_all()

    def verdict(self, timeout_s: float = 300.0, alive: Any = None) -> dict[str, Any] | None:
        """What the client reported at the end of the episode.

        ``None`` when nothing came in time or ``alive()`` turned false (the client died).
        """

        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._over is None:
                left = deadline - time.monotonic()
                if left <= 0 or (alive is not None and not alive()):
                    return None
                self._cv.wait(min(left, 2.0))
            return dict(self._over)


class BridgeEnv:
    """The environment :mod:`.robodojo_tools` steps, over a :class:`Bridge`."""

    def __init__(self, bridge: Bridge, alive: Any = None) -> None:
        self.bridge = bridge
        self.alive = alive  # false once the evaluation client's process has gone
        self.action_context: dict[str, Any] | None = None

    def get_obs(self) -> dict[str, Any]:
        return self.bridge.get_obs()

    def get_status(self) -> dict[str, Any]:
        return self.bridge.status()

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        try:
            obs = self.bridge.step(action, context=self.action_context)
            done = self.bridge.over is not None
        except EpisodeOver:
            obs, done = self.bridge.get_obs(), True
        return obs, 0.0, done, {"status": self.bridge.status()}

    def refresh_observation(self) -> dict[str, Any]:
        return self.bridge.refresh()

    def episode_over(self) -> bool:
        """The client has ended the episode and reported its verdict."""

        return self.bridge.over is not None

    def conclude(self) -> None:
        self.bridge.finish()
        self.bridge.verdict(alive=self.alive)


class PolicyModel:
    """XPolicyLab's ``Model`` for a PyRUA-Lean episode: every call goes to a :class:`Bridge`."""

    def __init__(self, bridge: Bridge) -> None:
        self.bridge = bridge

    def reset(self) -> None:
        self.bridge.reset()

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.bridge.update_obs(obs)

    def get_action(self) -> list[dict[str, Any]]:
        return self.bridge.get_action()

    def on_trial_end(self, result: dict[str, Any] | None = None) -> bool:
        self.bridge.episode_end(result)
        return True

    def update_obs_batch(self, obs_list: Any) -> None:
        raise NotImplementedError("one conversation drives one environment: eval_batch is false")

    def get_action_batch(self, env_idx_list: Any = None) -> Any:
        raise NotImplementedError("one conversation drives one environment: eval_batch is false")


class Server:
    """XPolicyLab's websocket policy server, run in a thread of this process."""

    def __init__(self, model: Any, host: str, port: int) -> None:
        # XPolicyLab is importable once the host has put its checkout on sys.path
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig

        self._loop = asyncio.new_event_loop()
        self._server = PolicyServer(model, PolicyServerConfig(host=host, port=int(port)))
        self._thread = threading.Thread(
            target=self._run, name="xpolicylab-policy-server", daemon=True
        )
        self.error: BaseException | None = None
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._server.serve_forever())
        except BaseException as exc:  # noqa: BLE001 - reported by the host
            self.error = exc

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._server.stop(), self._loop).result(timeout=10)
        except Exception:  # noqa: BLE001 - best effort during shutdown
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


def serve(bridge: Bridge, host: str, port: int) -> Server:
    """Serve ``bridge`` to an XPolicyLab evaluation client on ``host:port``."""

    return Server(PolicyModel(bridge), host, port)


__all__ = [
    "Bridge",
    "BridgeEnv",
    "EpisodeOver",
    "OBSERVATION_KEY",
    "PolicyModel",
    "REQUEST_KEY",
    "STATUS_KEY",
    "Server",
    "WAIT",
    "serve",
]
