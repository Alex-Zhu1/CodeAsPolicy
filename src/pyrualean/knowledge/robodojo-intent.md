# Operating knowledge for RoboDojo: intent-driven manipulation

## Common operation feedback and continuation
- Motion, local alignment and staged grasp return `result['operation']` with
  schema `pyrualean.operation.v1`. Keep its `ref` (session_id, kind, id).
  `operation_status(ref)` reads existing progress without moving or taking images.
  `resume_operation(ref, max_steps=...)` continues only the already-started phase.
  `resume_motion(id)` uses this same continuation path for compatibility.
- Common states: `ready` needs an explicit next phase; `running`/`paused` can
  continue when `continuation.allowed` is true; `completed` means the operation's
  execution finished. `needs_replan` needs a revised decision; `unknown` requires
  receipt reconciliation and observation, never blind replay. `rejected` means
  this request cannot proceed; `preview` has not registered an executable plan.
- For a grasp, `phase_complete` and `execution_complete` differ. Completed open,
  approach, align or close returns ready/advance. Resume never starts the next
  grasp stage or automatically probes after closing. Choose `grasp_step` explicitly.
  Interrupted refinements retain their adjustment ID and original offset.
- `budget.used` is cumulative across continuations; raising `max_steps` raises
  the TOTAL limit. `receipt` links the last effectful dispatch, including partial
  or unknown effects. Cached calls do not replace it with a zero-action receipt.
  Domain feedback and object_evidence remain available alongside this envelope.
- Execution completion never establishes the visual condition or task success.
  `verification.status='required'` remains so until the planner checks appropriate
  fresh evidence; this protocol does not promote 2D consistency into 3D holding.
  Recorded poses and completed phases are historical. A changed unfinished
  endpoint needs replanning. Cross-session/restart continuation is unsupported.

## Robot conventions and measured geometry
- Two ARX-X5 arms stand at the near edge of the table. World x points across
  the table (left arm at negative x), y away from the robot, z up. Bases are
  at x=-0.30/+0.30, y=-0.45 metres. Table height is z=0.765; read object
  heights from the scene. The benchmark's task instruction is authoritative.
  The task card contains initial observations, source labels and capabilities;
  unknown or stale measurements cannot be used as current targets.
- Targets refer to the middle of the finger pads, TCP, 0.145 m beyond the
  wrist link along the pointing direction. state().arm(arm).tcp_pos reports it;
  eef_pos is the wrist. Pads span about 1.2 cm either side; fingertips extend
  1.3 cm beyond the TCP. Pointing down, keep the TCP at least about 1.5 cm
  above the surface under it and account for the actual finger clearance.
- Custom angles are degrees, not XYZ Euler: pitch=0 is level, 90 down;
  yaw=0 points away, +90 toward the robot's left (-x); roll turns about the
  pointing direction. At pitch=90, yaw sets the closing line from +x seen
  from above. Yaw and roll readings can change together without a physical
  turn; compare quaternion poses. Intent targets preserve the requested pose
  and never substitute a half turn. Position-only targets keep orientation.
- Reach depends on orientation. Pointing down at z=.80 reaches to about y=0;
  at z=.90 about y=-.05; at z=1.0 it is generally unreachable. Pitch=45 reaches
  farther and higher, level farther still. These are kinematic reference
  samples, not guarantees. Preview the actual route and check its result.

## Motion intent and executor responsibilities
- Use `execute_motion(intent_id, left=..., right=..., interaction=...,
  directions=..., holding=..., completion=...)`
  for routine single/dual-arm motion. Astra chooses the goal and constraints;
  the executor chooses waypoint spacing, speed and bounded arrival correction.
  The task card preserves the native instruction and initial facts. `plan()`
  records your current goal; an executable intent is a separate fixed contract.
- Targets are absolute world TCP xyz (preserve initial orientation), a Pose, or
  `{"xyz": xyz, "angles": [pitch,yaw,roll]}`. Omitted arms hold their poses.
  `interaction="free"` asserts a currently observed clear route; `"approach"`
  means interaction only near the endpoint; `"contact"` means interaction
  throughout. Choose this physical meaning, not velocity/acceleration values.
  Each intent follows a Cartesian line. Choose a new staging goal when a route
  or grasp must change; the executor never invents a detour or object recipe.
- `directions={"right":[0,0,-1]}` permits positive motion along that world axis;
  both the target and measured path must remain in its tolerance corridor.
  Omit the constraint for an unrestricted line. Directions do not authorize
  lateral searches, back-and-forth insertion or a changed grasp.
- `holding={"right":"empty"}` asserts an observed empty hand; `"retain"`
  requires the existing closed grip to remain obstructed at every native action;
  `"keep"` preserves the command without asserting object support. No intent
  opens, closes or regrips. Use staged grasp or set_gripper for those operations and
  observe the intended object before declaring retention. A retained opening
  cannot identify an object or detect all slips. In a push with a closed empty
  gripper, declare empty plus contact; no retained-object requirement applies.
- `completion` can set position_m/orientation_deg and a short `verify` condition.
  Robot pose is checked at arrival. The executor independently bounds tracking
  error during travel, then permits at most two progressing target holds at rest
  to meet arrival accuracy. Direction/relative-pose/holding guards still apply
  at every action. `verify` returns as needs_observation;
  inspect the actual scene, or use the existing visual watch contract. TCP
  arrival alone never establishes insertion, intended-object grasp or release.
- Omit `max_steps` and `reserve_steps` normally: all remaining native actions
  are available, with no required reserve for later manipulation or return.
  Cost estimates are advisory, not a veto on progress. If the full motion does
  not fit, the executor tries a shorter part of the same route, ending at rest.
  `paused`/`segment_complete` means partial progress, not final arrival; inspect
  and use `resume_motion(id)` to continue. If you explicitly chose a small cap,
  `resume_motion(id, max_steps=new_total)` increases the cumulative allowance
  without resetting spent steps or corrections. Never weaken geometry or speed
  constraints to fit an estimate. The actual native episode limit still applies.
- Do not abandon an incomplete task, return a held object or home the arms
  solely because forecasts suggest the full remaining sequence might not fit.
  Execute the next useful constrained stage and reobserve. Finish on task
  completion, actual episode termination, or a concrete physical/execution
  blocker. Required cleanup follows manipulation; it is not an early-exit rule.
- Normal checkpoints and progressing corrections stay inside the executor.
  After a known interruption, reconcile receipts and use resume_motion(id).
  It continues the remaining route from measured state with the same budget;
  it does not replay earlier travel or gripper actions. Completed IDs return
  their saved endpoint without movement. Unknown dispatch, external movement
  or a lost host session require review. Read stop_reason when needs_model is
  true. Do not wrap execute_motion/resume_motion in once(); their intent ID is
  already the execution identity. Staged grasps also own their identity;
  keep once() for separate set_gripper and home calls.
- Start with a clear staging intent, then a separately observed approach/contact
  intent where geometry demands it. Avoid unnecessary segmentation inside a
  known-clear route. Low-level movement and speed selection are disabled in
  this policy session. Budget pressure does not change an interaction from
  contact into free space; revise the allocation or physical strategy instead.

## Choose grasps for the remaining task
- Before choosing a grasp, work backwards from the next required use: which
  object region must contact the environment, which region can be held then,
  and what approach/rotation clearance that final hold needs. Choose a stable
  initial grasp that permits a transition to that final hold. A future working
  region may be held temporarily if a planned transfer frees it before use.
  A successful pickup is only one prerequisite; it does not establish that the
  remaining task is feasible.
- When a handoff is needed, choose donor and receiver grasp regions together.
  Leave the receiver region exposed beyond the donor fingers, and preserve
  access to the object's working region after transfer. Choose a mutually
  reachable exchange pose with room for both fingers and the held object.
  Consider staging the empty receiver first and presenting the exposed region
  to it, instead of requiring the receiver to reach the original pickup pose.
  Arm roles and exchange poses come from the current scene, not fixed sides.
- Record the intended use, grasp regions, exchange/working pose and remaining
  budget briefly in plan()'s goal, next_action, done_when and evidence fields.
  Compare a small set of geometrically distinct candidates with read-only
  previews before committing to a costly grasp. Batch independent previews.
  Previews start at the current robot state: they do not simulate a future
  grasp, validate object/inter-arm collisions, or certify a whole sequence.
  Reobserve the actual object-in-hand pose after lifting, then check the next
  route and finger clearance before approaching or releasing.
- If the intended grasp is inaccessible, consider repositioning the object or
  setting it down for a different grasp at a suitable support location. Consider
  the extra manipulation and verification cost without requiring an up-front reserve.
  Do not add repositioning when the existing pose already permits the task.
- A changed grasp region, object slip or changed arm role invalidates the
  assumptions used for transfer and later use. Recheck the receiver access,
  resulting working-region clearance and route before proceeding;
  stable closure alone is insufficient. Astra chooses the revised grasp/path;
  the executor continues to own spacing, timing and bounded corrections.

## Contact, grasp, release and cooperation
- On snapshot backends, register `ref = object_ref(name, description)` once per
  intended object. Preserve the reference across pickup/transfer; it is logical
  identity, not physical identity verification. Pass `object_ref=ref` to
  `begin_grasp` while the object is visible. Preparation and open/approach/align
  return `object_tracking`. Material features are acquired before occlusion and
  tracked through native RGB frames during motion and closure, without extra
  actions or per-frame model calls. Missing preclosure features can be supplied
  with `observe_object(ref, landmarks=...)` on the current visible frame.
  A successful close uses the observed POST-closure feature positions as its
  fixed baseline; preclosure positions are never treated as hold evidence.
  Probe adds a comparison to that fixed capture.
  Inspect the returned `object_evidence` before proceeding to the next stage.
- `baseline_ready` requires usable features. `baseline_quality.space` distinguishes
  `3d` from `wrist_image_2d`. An empty/ambiguous capture returns `baseline_not_ready`
  with `capture_observation_id` and `recovery`; later motion cannot silently
  replace that capture. Inspect its images and repair using
  `observe_hold(ref, arm=..., landmarks=..., observation_id=capture_observation_id)`.
  Landmarks map physical feature names to `[row,col]` in corresponding cameras,
  e.g. `{'corner': {'cam_head': [r,c], 'cam_right_wrist': [r,c]}}`. Then observe
  current evidence again. A historical repair alone is not current confirmation.
  If visibility is inadequate, revise the view/grasp; don't repeatedly probe or
  reset to manufacture a baseline. There is no automatic release or regrasp.
- `observe_hold(ref, arm=...)` after carry/reorientation and before donor release
  follows the SAME features between measured native RGB frames, with two-way
  flow and appearance checks, while retaining the closed comparison baseline.
  Lost features can return through distinctive original appearance, never a
  rigid-grasp prediction or fresh text detection. Intermediate occlusion does
  not erase the anchor. Ambiguous matches
  remain unknown. The original baseline never moves within the same grasp, even
  if reset=True repeats, so small slips accumulate visibly. Reset explicitly only
  after a new close-command epoch; bound grasp closure does this automatically.
- `following`/`relative_motion` compare 3D points. `image_consistent` and
  `image_relative_motion` only describe projections in a calibrated rigid wrist
  camera and cannot resolve depth or unseen rotations. They are not 3D retention
  certificates. Inspect `state`, `reason`, `evidence_space`, `features` or
  `image_features`, `tracking_failures` and `recovery`. Lack of evidence is unknown,
  not empty. Neither closure nor visual following certifies identity, force,
  stable contact or future retention. Do not infer held-object pose using a
  table-height prior. All evidence calls use zero native actions; they do not
  weaken motion guards or consume speculative future-task reserves.
- Use `begin_grasp(id, target, arm=..., approach=..., pregrasp=..., probe=...,
  completion=..., translation_limits=..., rotation_limit_deg=...)` to record a
  grasp without moving. Then `grasp_step(id, stage)` for `open`, `approach`,
  `align`, `close`, `probe` in order. Each stage returns before the next; inspect
  reached, next_stage, actual_pose and stage_result. grasp_complete follows the
  probe only and still requires images to verify the intended object followed.
  The executor plans only the current stage, not an all-or-nothing macro.
- target is the finger-pad centre and orientation at closure; pregrasp may
  specify a distinct approach pose. completion gives alignment accuracy, with
  a default of 1.5 mm / 1 degree. Observe from a useful camera after approach
  and align, before closing thin or partially hidden objects. Claw closure is
  not a way to compensate for an unverified target point.
- For a measured local offset after approach/alignment, call
  `grasp_step(id, 'align', adjustment_id='new_observation_id', offset=[dx,dy,dz],
  rotation_deg=[rx,ry,rz])`. Offsets are FROM the ORIGINAL target, in world axes;
  rotations are axis-angle vectors in degrees, not pitch/yaw/roll. All such
  targets and measured motion stay inside the original translation/rotation
  envelope. A failed alignment may be refined there; no changes after closing.
  A changed offset needs a new adjustment ID; repeating an ID cannot move again.
  If the current pose lies outside that local envelope, use a newly observed
  approach plan instead. No hidden search, automatic reopening or regrasp.
- `align_local(id, offset, reference=pose, frame='world' or 'tcp',
  translation_limits=[...], rotation_limit_deg=..., holding=...)` provides the
  same bounded alignment outside a grasp, for placement or other interactions.
  Keep reference fixed across adjustments; it is an observed estimate, not an
  oracle object frame. Default offset origin is the measured pose frozen on
  first use of that ID. Changing IDs must not silently shift the allowed region.
- Grasp stages preserve cumulative action and correction counts. A known
  interruption continues only the unfinished stage; completed open/close and
  motion stages do not replay. `grasp_status(id)` exposes progress. To increase
  an explicitly small allocation, pass max_steps=new_total to grasp_step. This
  does not reset spending. An empty close stops before probing; an obstructed
  close is not proof of object identity. Inspect before choosing a new grasp.
- Use the measured endpoint for every subsequent offset. A requested goal is
  not the actual pose. At most two progressing corrective holds are allowed
  across the whole intent, including resumes. A stall stops; tracking lag is
  not a contact sensor. Stop reasons distinguish geometry, tracking, budget
  and execution uncertainty. An ended/truncated episode cannot move further.
- For contact, preserve orientation unless the intended interaction requires
  rotation. Observe the resulting object translation/rotation before deciding
  another stroke. A changed contact point or route is an Astra decision.
  Depart from contact along a clear direction before turning or travelling.
- Grasp at a solid accessible region and account for finger dimensions and
  both arms. Previews start from current state and do not simulate a whole
  future sequence. A reachable grasp alone does not prove placement reach.
- set_gripper(arm, CLOSE/OPEN) runs a stroke without moving the arms; use once()
  to retain its identity. CLOSE=1, OPEN=-1. Measured opening instead is 1.0
  open, 0.0 closed on nothing. Width is about .002 + .109 * opening metres.
  opening=0.5 limits an open command to about 5.7 cm for narrow clearances.
  Holding means obstructed closure, not the intended object's identity.
  Verify a short lift in images before transport, turning or donor release.
- execute_motion preserves gripper commands throughout. Use holding='retain'
  for a verified held object, and choose a separate release after observing
  support. After release, retreat from the measured pose with unchanged
  orientation along a clear direction before turning or sweeping sideways.
- One intent can move both arms simultaneously. keep_relative=True constrains
  their relative TCP transform; simultaneity alone does not preserve it.
  Omitted arms hold their starting poses. Plan finger/inter-arm clearance:
  the executor checks sampled arm/table geometry, not all scene collisions.
- home(arm or 'both') is a bounded return primitive retaining gripper commands.
  It guards departure from a recorded release and stops if that retreat fails;
  the rest of the route has no scene collision check. Release onto support and
  clear nearby objects first. Perform return when required by the task or by
  observed clearance needs. Avoid homing after every local action or because
  a forecast says the remaining task might exceed the action budget.
- Read motion_capabilities for actual support. Current adapters do not provide
  force sensing, force regulation, gripper-force or impedance control. Opening
  and tracking lag are not force measurements. The executor's timed trajectories
  are continuous within uninterrupted simulation execution; model waits and
  interruptions do not provide wall-clock continuity.
