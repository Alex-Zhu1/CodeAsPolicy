"""Deterministic RoboDojo task cards from public observations and capabilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from ..prompt import TaskCard
from ..robodojo import _arm_state, _sequence
from ..task_acceptance import (
    acceptance_sha256,
    default_robodojo_acceptance_path,
    load_acceptance,
)
from .robodojo_coordination import capabilities


def _plain(value):
    """JSON-safe public state; invalid/nonfinite measurements stay missing."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _reference(metadata):
    # Select public observation provenance, never copy an arbitrary native packet.
    return {key: _plain(metadata.get(key)) for key in (
        "schema", "session_id", "observation_id", "source_id", "step", "native_steps",
        "action_id", "captured_at", "capture_clock", "recorded_at", "source_freshness", "aligned",
        "alignment",
    )}


def build_card(*, cell, raw, instruction, images, status, primitives, robot,
               observation=None, image_sources=None, observation_artifact=None,
               acceptance=None):
    """Keep benchmark instructions verbatim; never infer a task or missing state."""
    explicit_acceptance = (
        acceptance
        if acceptance is not None
        else getattr(cell, "acceptance", None) or os.environ.get("PYRUALEAN_TASK_ACCEPTANCE")
    )
    acceptance_source = (
        explicit_acceptance
        if explicit_acceptance is not None
        else default_robodojo_acceptance_path()
    )
    acceptance_contract, acceptance_artifact = load_acceptance(
        acceptance_source, suite=cell.suite, task=cell.task,
    )
    if (
        explicit_acceptance is None
        and not acceptance_contract["requirements"]
        and not acceptance_contract["coverage_source"]
    ):
        acceptance_artifact = None
    facts = {}
    for arm in ("left", "right"):
        pose = _sequence(raw.get(f"{arm}_ee_pose"))
        valid_pose = (len(pose) == 7 and all(map(math.isfinite, pose))
                      and abs(math.hypot(*pose[3:]) - 1) <= 0.001)
        position = angles = "unknown (missing or invalid initial pose)"
        if valid_pose:
            start = _arm_state(raw, arm)
            position = "[" + ", ".join(f"{v:.3f}" for v in start.tcp_pos) + "] m"
            angles = f"{round(start.pitch)} / {round(start.yaw)} / {round(start.roll)} deg"
        facts[f"{arm} arm initial grasp point [x, y, z]"] = position
        facts[f"{arm} gripper initial pitch / yaw / roll"] = angles
        value = raw.get(f"{arm}_ee_joint_state")
        # Native observations may use a scalar, list or a one-element numpy array.
        for _ in range(4):
            if value is None or isinstance(value, (int, float)):
                break
            try:
                value = next(iter(value))
            except (TypeError, StopIteration):
                break
        try:
            opening = float(value)
        except (TypeError, ValueError):
            opening = math.nan
        facts[f"{arm} gripper opening"] = (
            f"{opening:.2f} (1 = open, 0 = closed)"
            if math.isfinite(opening) and 0 <= opening <= 1 and not isinstance(value, bool)
            else "unknown (missing or invalid initial opening)"
        )
    limit = status.get("step_limit")
    try:
        budget = int(limit)
        if isinstance(limit, bool) or budget < 0 or float(limit) != budget:
            raise ValueError("invalid native action budget")
        budget_source = "RoboDojo get_status().step_limit"
    except (TypeError, ValueError, OverflowError):
        budget = cell.max_episode_steps
        budget_source = "configured cell fallback; native step limit unavailable"
    facts["native action budget"] = f"{budget} (a motion call spends one or more)"
    facts["motion capabilities at episode start"] = capabilities(primitives, status=status)
    metadata = observation or {}
    initial = _reference(metadata)
    initial["step"] = 0  # recorded-image index, never an inferred native action count
    image_records = {}
    for camera, path in images.items():
        data = Path(path).read_bytes()
        source = (metadata.get("images") or {}).get(camera) or {}
        image_records[camera] = {
            "path": str(Path(path).resolve()),
            "source_path": (image_sources or {}).get(camera),
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "observation_id": metadata.get("observation_id"), "step": 0,
            **{key: _plain(source.get(key)) for key in (
                "source_id", "image_source_id", "image_state_source_id", "aligned",
                "calibration_source_id", "calibration_method", "calibration_model",
                "artifact", "sensor_artifact", "sensor_manifest", "sensor_path",
                "calibration_artifact", "calibration_path", "depth_artifact", "depth_path",
                "depth_source_id", "image_state_artifact", "image_state_path",
            )},
        }
    # Keep the unrounded robot measurements needed to reproduce the displayed TCP/angles.
    initial_state = {f"{arm}_{field}": _plain(raw.get(f"{arm}_{field}"))
                     for arm in ("left", "right")
                     for field in ("ee_pose", "arm_joint_state", "ee_joint_state")}
    state_bytes = json.dumps(initial_state, sort_keys=True, allow_nan=False).encode()
    native_step = status.get("step")
    if type(native_step) is not int or native_step < 0:
        native_step = None
    artifact = None
    if observation_artifact is not None:
        path = Path(observation_artifact)
        data = path.read_bytes()
        artifact = dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                        size_bytes=len(data))
    if acceptance_contract["requirements"] or acceptance_contract["coverage_source"]:
        label = acceptance_contract["coverage_source"] or "supplied"
        acceptance_label = (
            f"public acceptance specification ({label}); "
            f"coverage={acceptance_contract['coverage']}"
        )
    else:
        acceptance_label = (
            "none supplied; only recorded step-0 instruction is available (coverage=unknown)"
        )
    provenance = dict(
        schema="pyrualean.task_card.v1", assembled_at=time.time(), time_domain="unix_seconds",
        initial_observation=initial, observation_artifact=artifact,
        instruction_sha256=hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
        initial_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
        acceptance_sha256=acceptance_sha256(acceptance_contract),
        acceptance_artifact=acceptance_artifact,
        images=image_records,
        budget_at_card_creation=dict(
            limit=budget, native_steps=native_step,
            remaining=max(0, budget - native_step) if native_step is not None else None,
            source=budget_source,
        ),
        capability_snapshot=dict(native_steps=native_step, source="PyRUA executor status query"),
    )
    return TaskCard(
        suite=cell.suite, task=cell.task, seed=cell.seed,
        task_language=instruction, object_names=[], images=images,
        max_env_steps=budget, robot=robot, facts=facts,
        initial_state=initial_state, provenance=provenance,
        acceptance=acceptance_contract,
        sources={
            "assembly": "PyRUA deterministic formatting; no Astra-generated task or plan",
            "task identifiers": "evaluation cell configuration (suite/task/seed)",
            "task instruction": (
                "RoboDojo recorded step-0 instruction, copied verbatim" if instruction else
                "missing from recorded step 0; no task inferred from the identifier"
            ),
            "task acceptance": acceptance_label,
            "initial state and images": "RoboDojo recorded step 0; PyRUA derives TCP and angles",
            "native action budget": budget_source,
            "motion capabilities": (
                "PyRUA executor at card creation; same fields as robo.motion_capabilities(); "
                "query again for current support"
            ),
        },
    )
