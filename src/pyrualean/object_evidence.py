"""Session-local object references and evidence about a particular grasp."""

from __future__ import annotations

import copy
import uuid

from .perception import perception_summary


class EvidenceResult(dict):
    """Keep full provenance accessible without repeating the camera audit in print()."""

    def __repr__(self):
        values = dict(self)
        if "provenance" in values:
            values["provenance"] = perception_summary(values["provenance"])
        return repr(values)


def baseline_quality(observation, arm):
    """Usable evidence dimensions, independently of whether retention is established."""
    points = observation["landmarks"]
    spatial = [n for n, f in points.items() if f.get("xyz") is not None
               and f.get("uncertainty_m") is not None and f["uncertainty_m"] <= .015]
    camera = f"cam_{arm}_wrist"
    image = [n for n, f in points.items() if camera in f.get("pixels", {})
             and f.get("appearance", {}).get(camera, {}).get("trackable") is True]
    calibrated = observation.get("views", {}).get(camera, {}).get("tcp_from_camera") is not None
    space = "3d" if spatial else "wrist_image_2d" if len(image) >= 2 and calibrated else None
    return dict(ready=space is not None, space=space,
                spatial_features=spatial, image_features=image if calibrated else [])


def _image_evidence(before, after, arm, result):
    """Report projected evidence explicitly; never relabel it as 3D following."""
    import numpy as np

    camera = f"cam_{arm}_wrist"
    va, vb = (o.get("views", {}).get(camera, {}) for o in (before, after))
    if (va.get("tcp_from_camera") is None or vb.get("tcp_from_camera") is None
            or va.get("shape") != vb.get("shape")
            or va.get("intrinsics") != vb.get("intrinsics") or not np.allclose(
                va["tcp_from_camera"], vb["tcp_from_camera"], atol=1e-3)):
        return dict(result, reason=result.get("reason") or "no_comparable_3d_landmarks")
    features = {}
    for name in before["landmarks"].keys() & after["landmarks"].keys():
        a, b = before["landmarks"][name], after["landmarks"][name]
        if (camera not in a.get("pixels", {}) or camera not in b.get("pixels", {})
                or not a.get("appearance", {}).get(camera, {}).get("trackable")):
            continue
        delta = np.asarray(b["pixels"][camera]) - a["pixels"][camera]
        features[name] = dict(camera=camera, delta_rc=delta.tolist(),
                              displacement_px=float(np.linalg.norm(delta)), comparison_bound_px=4.)
    if len(features) < 2:
        return dict(result, reason="insufficient_comparable_wrist_features")
    result.update(evidence_space="wrist_image_2d", image_features=features,
                  coverage="Image projection only: depth and unseen rotations are unobserved.")
    if sum(f["displacement_px"] > 4. for f in features.values()) >= 2:
        return dict(result, state="image_relative_motion", reason="features_shifted_in_wrist_image")
    if any(f["displacement_px"] > 4. for f in features.values()):
        return dict(result, reason="image_features_disagree")
    ta, tb = (np.asarray(o["robot"][arm]["tcp_matrix"]) for o in (before, after))
    translation = float(np.linalg.norm(tb[:3, 3] - ta[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(ta[:3, :3].T @ tb[:3, :3])-1)/2, -1., 1.)))
    if translation < .01 and angle < np.deg2rad(5.):
        return dict(result, reason="insufficient_motion_to_test_following")
    if not result["gripper"]["obstructed_closure"]:
        return dict(result, reason="image_consistency_without_closure_obstruction")
    return dict(result, state="image_consistent", reason="stable_projection_during_tcp_motion")


def assess_hold(before, after, arm):
    """Compare observed features in a measured TCP frame; never certify a grasp."""
    import numpy as np

    a, b = before["robot"][arm], after["robot"][arm]
    obstructed = (b["command"] is not None and b["command"] <= .05
                  and .01 < b["opening"] < .95)
    result = dict(state="unknown", reason=None, arm=arm, grasp_verified=False,
                  identity_verified=False, force_measured=False,
                  gripper=dict(command=b["command"], opening=b["opening"],
                               obstructed_closure=obstructed, command_epoch=b["command_epoch"]),
                  features={},
                  interpretation="Observation evidence, not guaranteed future retention.")
    for obs in (before, after):
        source = obs["provenance"]
        if (source.get("aligned") is not True or source.get("source_freshness") != "captured"
                or not source.get("source_id")):
            return dict(result, reason="unverified_observation")
    if before["provenance"]["source_id"] == after["provenance"]["source_id"]:
        return dict(result, reason="same_observation_not_new_evidence")
    if a["command_epoch"] is None or b["command_epoch"] is None:
        return dict(result, reason="gripper_command_history_unavailable")
    if a["command_epoch"] != b["command_epoch"]:
        return dict(result, reason="grasp_command_changed_reset_baseline", needs_baseline=True)
    if (a["command"] is None or b["command"] is None
            or a["command"] > .05 or b["command"] > .05):
        return dict(result, reason="gripper_not_commanded_closed")
    ta, tb = np.asarray(a["tcp_matrix"]), np.asarray(b["tcp_matrix"])
    movement = []
    changed = False
    for name in before["landmarks"].keys() & after["landmarks"].keys():
        p, q = before["landmarks"][name], after["landmarks"][name]
        if p.get("xyz") is None or q.get("xyz") is None:
            continue
        if max(p["uncertainty_m"], q["uncertainty_m"]) > .015:
            continue
        x, y = np.asarray(p["xyz"]), np.asarray(q["xyz"])
        local_a = ta[:3, :3].T @ (x - ta[:3, 3])
        local_b = tb[:3, :3].T @ (y - tb[:3, 3])
        predicted = tb[:3, :3] @ local_a + tb[:3, 3]
        drift = float(np.linalg.norm(local_b - local_a))
        bound = max(.008, p["uncertainty_m"] + q["uncertainty_m"])
        movement.append(float(np.linalg.norm(predicted - x)))
        changed |= drift > bound
        result["features"][name] = dict(tcp_xyz=local_b.tolist(),
            baseline_tcp_xyz=local_a.tolist(), relative_change_m=drift,
            comparison_bound_m=bound, expected_world_motion_m=movement[-1])
    result["coverage"] = "named_points_only; rotation about unseen axes is unobserved"
    if not movement:
        return _image_evidence(before, after, arm, result)
    result["evidence_space"] = "3d"
    if changed:
        return dict(result, state="relative_motion", reason="object_changed_relative_to_gripper")
    if not any(f["expected_world_motion_m"] >= max(.01, 2 * f["comparison_bound_m"])
               for f in result["features"].values()):
        return _image_evidence(before, after, arm,
                               dict(result, reason="insufficient_motion_to_test_following"))
    if not obstructed:
        return dict(result, state="conflicting_evidence",
                    reason="visual_motion_but_no_closure_obstruction")
    return dict(result, state="following", reason="observed_points_followed_gripper")


class ObjectEvidenceMixin:
    def _object_store(self):
        if not hasattr(self, "_object_refs"):
            self._object_refs = {}
            self._object_observations = {}
            self._object_latest = {}
            self._hold_baselines = {}
            self._hold_captures = {}
            self._object_anchors = {}
            self._grasp_objects = {}
            self._grasp_evidence = {}
            self._grasp_tracks = {}

    def _object_ref(self, reference):
        self._object_store()
        if (not isinstance(reference, dict) or not isinstance(reference.get("name"), str)
                or self._object_refs.get(reference["name"]) != reference):
            raise ValueError("Use an unchanged object_ref from this live robot session")
        return reference

    def _object_observation(self, reference, observation):
        if not isinstance(observation, dict):
            raise ValueError("Use a recorded observe_object result")
        saved = self._object_observations.get(observation.get("object_observation_id"))
        if saved is None or saved != observation or saved["reference"] != reference:
            raise ValueError("Observation changed or belongs to another object/session")
        return copy.deepcopy(saved)

    def object_ref(self, name: str, description: str) -> dict:
        """Register a stable, session-local logical object reference, without moving.

        name is your unique identity; description distinguishes the intended
        object by visible properties. Use a short appearance noun phrase, not
        action instructions or locations that change during motion.
        Same name/description returns the same
        reference; changing its meaning rejects. This is not object detection
        or proof of physical identity. Preserve it through pickup/transfer;
        use a different reference for a different object. No task-specific IDs.
        """
        self._object_store()
        if getattr(self._backend, "supports_snapshot_perception", False) is not True:
            raise NotImplementedError("Object evidence needs a snapshot-perception backend")
        for value, limit in ((name, 80), (description, 300)):
            if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
                raise ValueError("Object name/description must be bounded nonempty text")
        old = self._object_refs.get(name)
        if old is not None:
            if old["description"] != description:
                raise ValueError("object reference identity conflict")
            return copy.deepcopy(old)
        if len(self._object_refs) >= 64:
            raise ValueError("object reference capacity reached")
        ref = dict(name=name, description=description, object_id=uuid.uuid4().hex,
                   session_id=self._interaction_session_id)
        self._object_refs[name] = ref
        return copy.deepcopy(ref)

    def observe_object(self, reference: dict, *, landmarks=None, observation_id=None) -> dict:
        """Observe the SAME object's visible features without assuming it is on a table.

        landmarks maps physical feature names to matching [row, col] pixels:
        {'corner': {'cam_head': [r,c], 'cam_right_wrist': [r,c]}}.
        Select the SAME physical point in each view and keep feature names
        consistent across time. At least two calibrated views are needed for
        3D; poor parallax or inconsistent matches return xyz=None. No RGB
        table-plane depth or full 6D pose is invented for a held object.

        Omit landmarks to acquire distinctive material features inside text
        masks, with mutual descriptor matching across views. Once an anchor
        exists, recover its SAME features by distinctive descriptor/template
        matching against that frozen keyframe, even after empty intermediate
        observations. Failed/ambiguous matches are omitted, never substituted
        with old pixels or a predicted rigid-grasp pose. Masks nominate object
        regions; neither their centres nor silhouettes are triangulated.
        Explicit landmarks={} clears this observation, not a grasp's anchor.

        Returns reference, source provenance, per-view candidates, named 3D
        points with diagnostic uncertainty, robot pose and recorded gripper
        commands. Identity/association remains declared or estimated. Preserve
        this result for observe_hold; require_current(result) before using its
        geometry. Historical reads do not advance the live tracking baseline.
        Read-only: zero native actions, including refresh/analysis.
        """
        return self._observe_object(reference, landmarks=landmarks, observation_id=observation_id)

    def _observe_object(self, reference, *, landmarks=None, observation_id=None,
                        anchor=None, capture=False, arm=None, tracking_id=None,
                        tracking_epoch=None, seed=False):
        ref = self._object_ref(reference)
        if len(self._object_observations) >= 1024:
            raise ValueError("object observation capacity reached")
        source = self._perception_snapshot(observation_id)
        previous = None if capture else self._object_latest.get(ref["object_id"])
        if not capture and anchor is None:
            anchor = self._object_anchors.get(ref["object_id"])
        if previous and source["step"] < previous["provenance"]["step"]:
            previous = None
        if anchor and source["step"] < anchor["provenance"]["step"]:
            anchor = None
        index = len(self._ledger.records)
        payload = self._perception_call("object_observe", dict(
            description=ref["description"], landmarks=copy.deepcopy(landmarks),
            previous=copy.deepcopy(previous), anchor=copy.deepcopy(anchor), capture=capture,
            arm=arm, tracking_id=tracking_id, tracking_epoch=tracking_epoch,
            seed=seed), observation_id=source["observation_id"])
        result = EvidenceResult(payload, reference=copy.deepcopy(ref),
                                object_observation_id=payload["provenance"]["perception_id"])
        self._object_observations[result["object_observation_id"]] = copy.deepcopy(result)
        # Preserve exact evidence even when the policy prints only a compact status.
        self._ledger.records[index].summary = copy.deepcopy(dict(result))
        if self.perception_is_current(result):
            self._object_latest[ref["object_id"]] = copy.deepcopy(result)
            if result["landmarks"] and (capture or landmarks is not None or anchor is None):
                self._object_anchors[ref["object_id"]] = copy.deepcopy(result)
        return result

    def _hold_tracking_id(self, ref, arm, before):
        return f"hold:{ref['object_id']}:{arm}:{before['robot'][arm]['command_epoch']}"

    def _grasp_track(self, grasp_id, *, seed=False):
        """Sample a bound object's native RGB track without changing grasp geometry."""
        self._object_store()
        binding = self._grasp_objects.get(grasp_id)
        if not binding:
            return None
        ref, arm = binding['reference'], binding['arm']
        old = self._grasp_tracks.get(grasp_id)
        latest = self._object_latest.get(ref['object_id'])
        declared = (latest is not None and latest is not old
                    and latest.get('association') == 'declared_correspondence'
                    and bool(latest['landmarks']) and self.perception_is_current(latest)
                    and (old is None or latest['object_observation_id']
                         != old['object_observation_id']))
        if old is None:
            old = latest
        declared_pixels = ({n: f['pixels'] for n, f in latest['landmarks'].items()}
                           if declared else None)
        current = self._observe_object(ref, arm=arm, anchor=old, landmarks=declared_pixels,
            capture=old is None or declared,
            tracking_id=f"grasp:{ref['object_id']}:{grasp_id}", seed=seed)
        self._grasp_tracks[grasp_id] = current
        return current

    def _grasp_tracking_feedback(self, grasp_id, result, *, seed=False):
        try:
            observation = self._grasp_track(grasp_id, seed=seed)
            if observation is None:
                return result
            feedback = dict(observation_id=observation['object_observation_id'],
                source_id=observation['provenance']['source_id'],
                features=len(observation['landmarks']), association=observation['association'],
                quality=baseline_quality(observation, self._grasp_objects[grasp_id]['arm']),
                continuity=observation.get('continuity', {}),
                tracking_failures=observation['tracking_failures'],
                is_hold_evidence=False)
        except Exception as exc:  # Optional perception cannot replay a completed stroke.
            feedback = dict(features=0, error=str(exc), is_hold_evidence=False)
        return dict(result, object_tracking=feedback)

    def observe_hold(self, reference: dict, *, arm: str, observation=None,
                     landmarks=None, observation_id=None, reset=False) -> dict:
        """Gather evidence that named object points follow a particular gripper.

        First call (or reset=True after a NEW grasp) captures fresh object
        material features. baseline_ready becomes true only for usable 3D
        features or at least two trackable wrist-image features. Missing data
        returns baseline_not_ready and recovery with the exact capture ID.
        A later frame cannot silently replace that capture. Repair it using
        landmarks plus observation_id=capture_observation_id; then observe
        current evidence. Historical repair alone is not a current hold claim.

        After an intended probe/rotation, compare again against the SAME anchor.
        Lost intermediate tracks do not destroy it; distinct appearance matches
        can recover the original features. Ambiguity stays unknown. reset=True
        cannot reset an existing baseline within the same gripper-command epoch.
        Reopening/reclosing invalidates it; explicitly reset after the new grasp.
        begin_grasp(..., object_ref=ref) observes while the target is visible and
        tracks native RGB frames through closure. The POST-closure positions,
        never the preclosure positions, establish the baseline. It adds
        object_tracking before closure and object_evidence at close/probe.
        Inspect both; neither a preclosure track nor an empty close is retention.
        Holding tracks may update pixel positions between native frames, but
        comparisons always use the immutable closed baseline. An occluded point
        can return only by matching the original appearance, not text redetection.

        state following/relative_motion concerns 3D points. image_consistent
        and image_relative_motion concern only a rigid wrist camera's 2D image;
        they cannot establish depth, unseen rotations or a 3D hold. Check
        evidence_space, baseline_quality, needs_observation and recovery.
        unknown/conflicting_evidence remain explicit. following means observed
        named points moved consistently with the TCP,
        with closure obstruction; it never proves object identity, full pose,
        grasp stability, contact force or future retention. One point cannot
        reveal rotation about that point. No detection/occlusion means unknown,
        not empty. Changed grip, source, object or uncertain execution is not
        silently accepted. Inspect reason, features and tracking_failures.

        Use after grasp/probe, carry or reorientation and before donor release.
        relative_motion/conflicting_evidence needs fresh observation and strategy
        review; don't automatically open, regrasp or keep transporting. This
        read-only evidence layer never dispatches actions or overrides motion guards.
        """
        ref = self._object_ref(reference)
        if arm not in ("left", "right") or type(reset) is not bool:
            raise ValueError("arm must be left/right; reset must be bool")
        if observation is not None and (landmarks is not None or observation_id is not None):
            raise ValueError("Supply observation OR landmark/snapshot inputs")
        key = (ref["object_id"], arm)
        before = self._hold_baselines.get(key)
        captured = self._hold_captures.get(key)
        if observation is None:
            source = self._perception_snapshot(observation_id)
            epoch = source.get("gripper_command_epochs", {}).get(arm)
        else:
            observation = self._object_observation(ref, observation)
            epoch = observation["robot"][arm]["command_epoch"]
        old = before or captured
        new_grasp = old is not None and old["robot"][arm]["command_epoch"] != epoch
        acquiring = old is None or (reset and new_grasp)
        tracking_id = (self._hold_tracking_id(ref, arm, before)
                       if before is not None and not acquiring else None)
        current = (self._observe_object(ref, landmarks=landmarks,
                    observation_id=source["observation_id"],
                    anchor=None if acquiring else old, capture=acquiring, arm=arm,
                    tracking_id=tracking_id if observation_id is None else None,
                    tracking_epoch=before['robot'][arm]['command_epoch'] if before else None)
                   if observation is None else observation)
        fresh = self.perception_is_current(current)
        quality = baseline_quality(current, arm)
        grip = current["robot"][arm]
        closed = (grip["command"] is not None and grip["command"] <= .05
                  and grip["command_epoch"] is not None and .01 < grip["opening"] < .95)
        repaired = False
        # A missing baseline is repairable only on the originally captured frame.
        # Later motion may already have slipped: never silently move the origin.
        same_capture = (captured is not None and captured["provenance"]["source_id"]
                        == current["provenance"]["source_id"])
        source_ok = (current["provenance"].get("aligned") is True
                     and current["provenance"].get("source_freshness") == "captured")
        if fresh and acquiring and closed:
            before = None
            self._hold_baselines.pop(key, None)
            self._hold_captures[key] = copy.deepcopy(current)
            captured = current
            if quality["ready"]:
                self._hold_baselines[key] = copy.deepcopy(current)
                before = current
        elif before is None and same_capture and closed and source_ok and quality["ready"]:
            self._hold_baselines[key] = copy.deepcopy(current)
            before, repaired = current, True
        result = assess_hold(before or current, current, arm)
        if not fresh:
            reason = ("baseline_repaired_await_current_observation" if repaired
                      else "stale_or_unresolved_observation")
            result.update(state="unknown", reason=reason)
        elif not closed and (before is None or new_grasp):
            result.update(state="unknown", reason="gripper_not_obstructed")
        elif before is None:
            result.update(state="unknown", reason="baseline_not_ready")
        elif before is current:
            result.update(state="unknown", reason="baseline_ready_not_hold_proof")
        ready = before is not None and before["robot"][arm]["command_epoch"] == epoch
        # Start native-frame tracking at the actual post-closure capture. This
        # is an image identity track; before remains the immutable hold baseline.
        if ready and fresh and before is current:
            self._observe_object(ref, arm=arm, anchor=before,
                tracking_id=self._hold_tracking_id(ref, arm, before), tracking_epoch=epoch)
        result.update(reference=copy.deepcopy(ref), current=fresh, baseline_ready=ready,
                      baseline_quality=baseline_quality(before or captured or current, arm),
                      baseline_id=before["object_observation_id"] if before else None,
                      capture_observation_id=(captured or current)["provenance"]["observation_id"],
                      observation_id=current["object_observation_id"],
                      provenance=copy.deepcopy(current["provenance"]),
                      tracking_failures=current["tracking_failures"],
                      association=current["association"],
                      association_details=current.get("association_details", {}))
        result['continuity'] = copy.deepcopy(current.get('continuity', {}))
        result["needs_observation"] = not ready or result["state"] == "unknown"
        if not ready:
            result["recovery"] = dict(action="repair_original_capture_or_establish_new_grasp",
                observation_id=result["capture_observation_id"],
                instruction="Inspect this capture; supply matching material landmarks for it. "
                            "A later frame cannot replace a missing closure baseline.")
        elif result["reason"] in ("no_comparable_3d_landmarks",
                                  "insufficient_comparable_wrist_features"):
            result["recovery"] = dict(action="reassociate_original_features",
                instruction="Original anchor retained. Improve visibility or provide the same "
                            "named features; do not reset the baseline to conceal drift.")
        record = self._ledger.open("hold_evidence", dict(reference=ref, arm=arm,
                                   observation_id=current["provenance"]["observation_id"]),
                                   stateful=False)
        record.ok, record.env_steps = True, 0
        record.session_id = current["provenance"]["session_id"]
        record.observation_id = current["provenance"]["observation_id"]
        record.summary = copy.deepcopy(dict(result))
        return EvidenceResult(result)

    def _grasp_object_feedback(self, grasp_id, stage, result):
        """Attach read-only evidence after a known completed closure/probe."""
        self._object_store()
        binding = self._grasp_objects.get(grasp_id)
        if not binding:
            return result
        if stage in ('open', 'approach', 'align') and result.get('steps_used', 0) > 0:
            return self._grasp_tracking_feedback(grasp_id, result, seed=True)
        if stage not in ("close", "probe") or not result.get("reached"):
            return result
        key = (grasp_id, stage)
        replayed = key in self._grasp_evidence
        if not replayed:
            try:
                closed = self._grasp_track(grasp_id) if stage == 'close' else None
                evidence = self.observe_hold(binding["reference"], arm=binding["arm"],
                                             observation=closed, reset=stage == "close")
            except Exception as exc:
                # A perception error must not erase/replay an already completed stroke.
                evidence = dict(state="unknown", reason="evidence_acquisition_failed",
                                error=str(exc), baseline_ready=False, needs_observation=True)
            self._grasp_evidence[key] = copy.deepcopy(evidence)
        evidence = copy.deepcopy(self._grasp_evidence[key])
        evidence["replayed_evidence"] = replayed
        evidence["current"] = self.perception_is_current(evidence)
        return dict(result, object_evidence=evidence)
