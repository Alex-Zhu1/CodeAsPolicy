# Operating knowledge for RoboDojo: intent-driven manipulation

## Common operation feedback and continuation
- Motion, local alignment and staged grasp return `result['operation']` with
  schema `codeaspolicy.operation.v1`. Keep its `ref` (session_id, kind, id).
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
  wrist link along the pointing direction (`state().arm(arm).tcp_pos`;
  `eef_pos` is the end-effector link 0.145 m behind it, where the wrist-camera
  housing projects rigidly above/behind the link). At `tcp_pos`, closed pads
  span ~2.4 cm (`±1.2 cm` around `tcp_pos` along `close_dir`, ~2.0 cm thick
  perpendicular to `close_dir`) and fingertips extend 1.3 cm beyond `tcp_pos`;
  however, ~5–7 cm behind `tcp_pos` at the finger roots, the **fixed jaw-rail
  crossbar** spans ~9–10 cm (`±4.5–5.0 cm` along `close_dir`) **even when the
  gripper is fully closed (`opening=0.0`)**. Pointing down, keep the TCP at
  least about 1.5 cm above the surface under it and account for actual finger,
  jaw-rail crossbar, and housing clearance.
- Custom angles are degrees, not XYZ Euler: pitch=0 is level, 90 down;
  yaw=0 points away, +90 toward the robot's left (-x); roll turns about the
  pointing direction. At pitch=90, yaw sets the closing line from +x seen
  from above. Yaw and roll readings can change together without a physical
  turn; compare quaternion poses. Intent targets preserve the requested pose
  and never substitute a half turn. Position-only targets keep orientation.
  To avoid Euler gimbal lock or manual quaternion algebra, build poses directly
  from physical vectors with `robo.pose_from_axes(xyz, point_dir, close_dir,
  roll_deg=...)` (`point_dir` is `+x_tcp` wrist-to-tips, `close_dir` is `+y_tcp`
  jaw-closing line; `roll_deg=180` swaps empty symmetric jaws), read vectors
  with `pose.axes()`, and generate rigid screw/hinge/shaft arcs with
  `pose.rotated(world_axis, angle_deg, pivot=...)`.
- Reach and continuous-turn headroom depend on orientation and IK branch.
  Pointing down at z=.80 reaches to about y=0; at z=.90 about y=-.05; at z=1.0
  it is generally unreachable. Pitch=45 reaches farther and higher, level
  farther still. Use zero-step local kinematics `robo.fk(arm, joints)` and
  `robo.ik(arm, target, seed=..., jaw_symmetry=..., continuous=...)` in Python
  to verify `ok`, `collides`, and `min_margin_rad` across chained poses (e.g.
  selecting between `0°` and `180°` empty-jaw symmetry before grasping so
  downstream `continuous=True` insertion or rotation stays well away from the
  `±3.1 rad` wrist limit). Whenever an operation involves constrained
  seating/insertion followed by an in-place twist/turn, call `robo.ik()` across
  the **entire pose chain** `(grasp_pose, pre_contact_pose, seated_pose,
  rotated_end_pose)` *before* closing the gripper (0 native steps); never check
  IK only for `grasp_pose` and discover after pickup and transport that
  `seated_pose` or `rotated_end_pose` is IK-unreachable.
- **Align the finger-pointing direction with the required rotation axis**:
  holding a part with a nearly horizontal gripper (`pitch≈0`) and trying to
  twist around a vertical world axis forces the 6-DOF arm to sweep an arc and
  hits wrist singularities or IK rejections. Point along the rotation axis
  (`pitch=90` pointing straight down for a vertical-axis twist) so the turn runs
  purely on the terminal wrist joint (`yaw`, `±178°` range) in place; choose a
  grasp or supported **`place-and-regrasp`** reorientation that puts the working
  axis along the pointing axis before insertion—never pinch a narrow (`<= 3.5–4 mm`)
  distal shaft/tip near ARX-X5's `~2.3 mm` minimum closed-jaw gap (`GRIP_ZERO_M`)
  to pitch-flip (`pitch=90° -> 0°–30°`) a heavier offset head upward in mid-air.
- **Held-object lower-extremity clearance during lateral transport (`object_bottom_z` vs `tcp_z`)**:
  because `tcp_pos` is the finger-pad centre rather than the lowest point of a
  held object, `tcp_z > z_obs` alone is insufficient when part of the held
  object (or fingertips) hangs or protrudes `L_below` below `tcp_pos`. Before
  lateral travel past any elevated fixture, container, or stack of top height
  `z_obs`, lift straight up along a Cartesian line (`+z`) to
  `tcp_z >= z_obs + L_below + 0.03` m so the **lowest point of the held object
  and fingers** (`object_bottom_z = tcp_z - L_below`) clears `z_obs` with margin.

## Motion intent and executor responsibilities
- Use `execute_motion(intent_id, left=..., right=..., waypoints=...,
  interaction=..., directions=..., holding=..., completion=...)`
  for single/dual-arm lines and smooth curves. Astra chooses the goal/curve and
  constraints; the executor chooses spacing, C2 spline timing and bounded
  arrival correction. The task card preserves the native instruction and initial
  facts. `plan()` records your current goal; an executable intent is a separate
  fixed contract.
- Targets are absolute world TCP xyz (preserve initial orientation), a Pose, or
  `{"xyz": xyz, "angles": [pitch,yaw,roll]}`. Omitted arms hold their poses.
  `interaction="free"` asserts a currently observed clear route; `"approach"`
  means interaction only near the endpoint; `"contact"` means interaction
  throughout. Choose this physical meaning, not velocity/acceleration values.
  Single-goal `"approach"`/`"contact"` intents follow a strict Cartesian line;
  unconstrained `"free"` intents follow a Cartesian line and fall back to a
  smooth table-clear joint swing when straight-line SLERP hits a wrist
  singularity or mid-chord boundary. For curved trajectories, pivot arcs, or
  smooth multi-pose paths computed in Python, pass `waypoints=[{"right": p1}, ...]`
  (1..64 poses, executable with `preview=False` and `preview=True`) to run one
  continuous C2 spline without intermediate stops (`following=[...]` is
  `preview=True`-only). Choose a new staging goal when a route or grasp must
  change; the executor never invents an object recipe.
- `directions={"right":[0,0,-1]}` permits positive motion along that world axis;
  both the target and measured path must remain in its tolerance corridor.
  Omit the constraint for an unrestricted line. Directions do not authorize
  lateral searches, back-and-forth insertion or a changed grasp.
- `holding={"right":"empty"}` asserts an observed empty hand; `"retain"`
  requires the gripper to stay commanded closed without collapsing from a wider
  initial grasp (thin `<3.5 mm` parts such as key shafts, coins or cards settle
  at `opening <= 0.01` just like empty jaws, so `retain` permits thin full-closure
  holds while stopping if a wider grip collapses); `"keep"` preserves the
  command without asserting object support. No intent opens, closes or regrips.
  Use staged grasp or set_gripper for those operations and observe the intended
  object after lifting before declaring retention. A retained opening cannot
  identify an object or detect all slips. In a push with a closed empty
  gripper, declare empty plus contact; no retained-object requirement applies.
- `completion` can set position_m/orientation_deg and a short `verify` condition.
  Robot pose is checked at arrival. The executor independently bounds tracking
  error during travel, then permits at most two progressing target holds at rest
  to meet arrival accuracy. Direction/relative-pose/holding guards still apply
  at every action. `verify` returns as needs_observation;
  inspect the actual scene, or use the existing visual watch contract. TCP
  arrival alone never establishes insertion, intended-object grasp or release.
- Treat any episode step limit as Primary Execution Budget (~60%–70%) plus a
  Contingency & Cleanup Reserve (~30%–40% for local alignment/regrasp fixes,
  release verification, and final arm homing). During late-stage manipulation,
  pass `reserve_steps` (e.g., keeping at least 60–100 native actions for final
  release, post-placement adjustment, and homing) or stage `max_steps` so a
  single corrective move cannot exhaust the tail budget. Cost estimates are
  advisory: if a full motion does not fit, the executor runs a shorter segment
  ending at rest. `paused`/`segment_complete` means partial progress; inspect
  and use `resume_motion(id)` (or `max_steps=new_total` to raise an explicit
  cumulative cap without resetting spent steps). Never weaken geometry or speed
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

## Execute a local work group
- Reuse a suitable staging pose across nearby targets. Choose finger orientation,
  opening and contact height for the affected objects and surrounding clearance.
  The gap between fingers can miss a group member; inspect each intended effect.
  Several short contacts and local shifts can share one approach and departure.
- Choose the lowest-order physical action that achieves each member's required
  final state. When a member or group only needs an in-place state or orientation
  change (such as tipping face up, flipping, pushing or pressing) and no
  requirement demands moving it to another zone, perform the contact action in
  place instead of staging a costly pick-carry-place-tip sequence per member.
  Do not pick up and relocate existing row/group members merely to create an
  interior gap for a replacement item; keep existing members at their positions
  and place any replacement/supply item at the designated end or slot of the row.
- **Selective contact / tipping inside a tightly packed row or array**: because
  the fixed jaw-rail crossbar (~5–7 cm behind `tcp_pos`) spans ~9–10 cm
  (`±4.5–5.0 cm` along `close_dir`) even when closed (`opening=0.0`), driving a
  tilted (`pitch=30°–60°`) closed gripper deep into a crowded row lowers that
  `±4.5–5.0 cm` crossbar into the height band of adjacent standing neighbours
  and topples them even when `tcp_pos` itself is centred. When tipping, pushing,
  or pressing individual items while preserving adjacent upright neighbours:
  (1) **point straight down (`pitch=90`)** so the jaw-rail crossbar and camera
  housing stay `>= 6 cm` above `tcp_z`, clear of standing neighbour tops;
  (2) **align `close_dir` along the contact stroke** (e.g. `yaw=±90` for a `±y`
  stroke) when lateral gaps are tight so only the narrow (~2.0 cm) closed-finger
  thickness faces left/right neighbours; (3) **contact one centred item at a
  time with a bounded stroke** (~3–4.5 cm past initial front contact—just past
  its tipping or target threshold—never a multi-item lateral sweep or 7+ cm
  over-travel deep through the row); and (4) **precheck `robo.ik()` for the
  post-contact straight-up (`+z`) retreat** (`dz=+0.04..0.06` m) before
  descending so departure never requires dragging backward through the row.
- Express each changed route as a new `execute_motion` intent and inspect its
  result before dependent motion. Resume an interrupted route under its original
  ID. Group progress is a planner decision over existing operations; the executor
  does not automatically choose contacts, skip members or certify their effects.
- Between contacts, clear fingers or held objects along an observed free route
  only as far as needed for the next local transition. Omit tight `completion`
  tolerances (keep `position_m >= 0.004` and `orientation_deg >= 2.0`) on
  `interaction="free"` retreats and local shifts between members so empty-hand
  transitions stay fast instead of triggering a slow `precision` window; reserve
  sub-4 mm / sub-2 degree tolerances for final contact or alignment endpoints.
  Preserve orientation until clear before turning; a group does not remove
  departure or collision checks. Use combined left/right intents only when their
  routes and object interactions are compatible. Sampled arm/table checks do not
  establish inter-arm clearance.

## Choose grasps for the remaining task
- Before choosing a grasp, work backwards from the next required use: which
  object region must contact the environment, which region can be held then,
  and what approach/rotation clearance that final hold needs. Choose a stable
  initial grasp that permits a transition to that final hold. A future working
  region may be held temporarily if a planned transfer frees it before use.
  A successful pickup is only one prerequisite; it does not establish that the
  remaining task is feasible.
- **Cross-arm transfer (`place-and-regrasp` vs in-air handoff)**: on ARX-X5,
  finger pads span ~2.4 cm (`±1.2 cm` around `tcp_pos`) and each arm's
  wrist-camera housing projects rigidly above/behind `eef_pos` (`0.145 m` behind
  TCP). When a held object exposes less than ~4 cm of unobstructed receiver
  grasping length beyond the donor fingers (compact blocks, tiles, short tools,
  caps, or flat parts `< ~6–7 cm`), an **in-air gripper-to-gripper handoff**
  crowds both finger assemblies and wrist-camera housings into the same
  envelope, causing housing collisions (`tracking_error`), insecure edge pinches,
  IK blocks, or obstructed donor withdrawal. Whenever a clear support patch
  exists in the shared central reach zone and the instruction does not forbid
  intermediate surface support, **prefer a supported central set-down transfer
  (`place-and-regrasp`) over an in-air handoff** for compact objects: place the
  object stably on the clear central surface, open, retreat along a straight
  Cartesian line, and clear/home the donor arm *before* bringing the receiver
  arm in to re-observe (`crop_grid` + `ray_at_z`/`triangulate`) and grasp in
  unobstructed single-arm workspace. Reserve **in-air handoffs** for elongated
  parts exposing ample (`>= 5 cm`) free handle/shaft beyond the donor fingers
  or scenes with no clear central support patch—then choose donor and receiver
  grasp regions together with the receiver region exposed beyond the donor
  fingers, verify a mutually reachable exchange pose, and preserve access to the
  working region after transfer. Arm roles come from the current scene, not
  fixed sides.
- Record the intended use, grasp regions, exchange/working pose and remaining
  step split (primary execution budget vs contingency/cleanup reserve) briefly
  in plan()'s goal, next_action, done_when and evidence fields.
  Compare a small set of geometrically distinct candidates with read-only
  previews before committing to a costly grasp. Batch independent previews.
  Multi-leg previews use `following=[...]` to evaluate chained waypoints under
  active `constraints={"max_tilt_deg": ..., "z_bounds": ..., "keep_out": ..., "corridor_m": ..., "geometry": ...}`
  (including TCP sphere envelopes and `ObjectPose`/`WorldPoint` auto-envelopes).
  Previews start at the current robot state: they do not simulate a future
  grasp or validate unmodeled inter-arm dynamics without explicit constraints.
  Reobserve the actual object-in-hand pose after lifting, then check the next
  route and finger clearance before approaching or releasing.
- A large remaining step budget never justifies higher-order multi-stage detours
  (such as moving objects whose pose/location need not change or laying a
  replacement item flat in an interior arrangement gap): match pickup angle and
  arm-to-arm transfer to the object's thickness, exposed grasp length, and mass
  distribution:
  - **Thin, flat or top-heavy/asymmetric objects** (under ~1 cm thick or
    `opening <= 0.04`, or narrow stems/shafts with an offset head or lateral
    protrusion): closed ARX-X5 pads retain a **`~2.3 mm` minimum closed-jaw gap**
    (`GRIP_ZERO_M`, `opening <= 0.015` is `< 4 mm` with no force control), so
    pinching a narrow distal shaft/tip (or narrow lateral projections) to
    pitch-flip (`pitch=90° -> 0°–30°`) a heavier offset head upward for an
    orthogonal mid-air handoff pendulum-slips under cantilevered gravity torque
    and burns ~80–100 native steps on joint swings, and tilted (`pitch=15°–60°`)
    pinches across a curved perimeter likewise twist out. Pick up vertically
    (`pitch=90°`) centred across the widest head/handle region (`opening > 0.05`),
    translate smoothly without unnecessary wrist rotation, and use a supported
    central **`place-and-regrasp`** transfer (donor sets flat on a clear central
    support patch, opens wide, retreats, and clears/homes before the receiving
    arm enters to re-observe and grasp) over mid-air handoff.
  - **Horizontal-to-vertical reorientation via `place-and-regrasp`, curved-rim
    vs flat-face holds, and asymmetric insertion polarity**: do **not** abandon
    flat-table `place-and-regrasp` on a compact (`< ~6–7 cm`) headed/shafted
    part merely because it rests horizontally on `z_support` while insertion
    requires `pitch=90` (`working axis` along `pointing`). Parallel jaws
    clamping **two flat parallel faces** (`area contact`) resist rotation, while
    jaws closing at `pitch=90` **perpendicular to the shaft** (`close_dir`
    perpendicular to `shaft_dir`) across a rounded/circular head at
    `tcp_z ≈ z_support + 0.018` m (fingertips ~5 mm above `z_support`) leave the
    shaft unobstructed between the open jaw sides so **gravity naturally pivots
    the overhanging shaft `90°` downward on lift into collinear alignment with
    `pointing=[0,0,-1]`** (and upward reaction force during `-z` seating pushes
    the head *deeper* between the finger pads rather than ejecting it, whereas
    clamping lower at `tcp_z <= z_support + 0.0165` m or closing along the shaft
    holds the shaft horizontally). Because a curved-rim or gravity-pivoted hold
    relies on narrow **tangent-line contact** near the fingertips, repeated or
    large (`>= 120°–180°`) in-air wrist yaw spins progressively walk the curved
    rim outward until closure collapses (`opening -> 0.0`): (1) **measure
    directional receptor polarity *before* set-down and receiver pickup**—for
    any target socket/slot with an asymmetric directional feature (one-sided
    slot, notch, tab, or lateral protrusion), localise **two points** along that
    receptor feature (`crop_grid` + `triangulate`/`ray_at_z`) *before* placing
    or regrasping, and select the central set-down heading plus receiver jaw
    sign (`+close_dir` vs `-close_dir`, `roll_deg=0` vs `180`) so the part's
    asymmetric feature **already aligns within `<= 45°–60°` of the receptor**
    at pickup while `robo.ik()` passes across `(grasp_pose, pre_contact_pose,
    seated_pose, rotated_end_pose)`; (2) when calling `ray_at_z` on a
    top-surface pixel of an object of thickness `h` on `z_support`, pass
    **`z = z_support + h`** (not bare `z_support`, which projects obliquely
    behind the true XY centre), and inspect the wrist camera after
    `grasp_step(..., 'align')` to centre `tcp_pos` across the widest chord
    before `close`; and (3) **never execute blind pre-measurement orientation
    spins or `>= 120°–180°` mid-air wrist flips** on a curved-rim or
    gravity-pivoted hold—verify 3D head-to-tip axis and feature facing right
    after `probe` lift (`triangulate`), use one shortest-arc yaw move
    (`<= 60°–90°`), and if polarity is `~180°` reversed, set down on clear
    support and regrasp with the opposite jaw sign (`roll_deg=180`).
  - **Thick, rigid parallel-sided blocks or tiles** (`opening > 0.05` across
    high-friction faces): support firm tilted or side grasps so the placing arm
    can present and stand the item upright directly at its destination rather
    than laying it flat inside a crowded interior arrangement gap; if cross-arm
    transfer is needed on a compact block or tile (`< ~6–7 cm`, exposing `< ~4 cm`
    beyond the donor pads), use **`place-and-regrasp` on an unobstructed central
    support patch** (away from standing group members) rather than crowding both
    wrist-camera housings in mid-air, reserving direct in-air handoff (`preview=True`)
    for elongated bars/shafts exposing `>= 5 cm` beyond the donor fingers or
    scenes without a clear central support patch.
  Before descending to seat, insert or release an object with a required axis,
  slot or face/yaw orientation, verify its actual in-hand alignment (including
  gravity hang, yaw alignment and front/back or tooth direction) in the
  wrist/head cameras in the air to avoid wrong-angle retries.
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
  grasp without moving (`probe` must be 0.5..6 cm; inspect `status` before
  calling `grasp_step`). Then `grasp_step(id, stage)` for `open`, `approach`,
  `align`, `close`, `probe` in order. In a dual-arm handoff, after the receiving
  arm completes `close` and the donor arm opens and withdraws, call
  `grasp_step(id, 'handoff')` to verify stationary receiver hold and reacquire
  state/evidence before `grasp_step(id, 'probe')`. Each stage returns before the next; inspect
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
  does not reset spending. `close` settles to full squeeze even when
  `opening <= 0.01` (because `<3.5 mm` thin parts such as key shafts, coins or
  cards compress to `0.0–0.01` just like empty air); always run `probe` and
  inspect wrist/head images (and `object_evidence`) after lifting to verify
  whether the intended object moved with the fingers before carrying or
  choosing a new grasp.
- Use the measured endpoint for every subsequent offset. A requested goal is
  not the actual pose. At most two progressing corrective holds are allowed
  across the whole intent, including resumes. A stall stops; tracking lag is
  not a contact sensor. Stop reasons distinguish geometry, tracking, budget
  and execution uncertainty. An ended/truncated episode cannot move further.
- For contact, preserve orientation unless the intended interaction requires
  rotation. Observe the resulting object translation/rotation before deciding
  another stroke. A changed contact point or route is an Astra decision.
  Depart from a low contact endpoint (`z < 0.85`) by lifting straight up (`+z`)
  or reversing back along the contact stroke axis before any lateral `free` or
  `approach` move (`post_contact_retreat_clearance`), and verify adjacent
  standing objects remain undisturbed.
- Grasp at a solid accessible region and account for finger dimensions and
  both arms. Previews start from current state and do not simulate a whole
  future sequence. A reachable grasp alone does not prove placement reach.
- set_gripper(arm, CLOSE/OPEN) runs a stroke without moving the arms; use once()
  to retain its identity. CLOSE=1, OPEN=-1. Measured opening instead is 1.0
  open, 0.0 closed on nothing. Width is about .002 + .109 * opening metres
  (`opening <= 0.01` is `<3.4 mm` jaw gap, `2.3 mm` at `0.0`, which occurs for
  both empty closure and `<3.5 mm` thin parts). Never judge thin-part possession
  from `opening` alone; verify a short lift in wrist/head images before
  transport, turning or donor release. opening=0.5 limits an open command to
  about 5.7 cm for narrow clearances.
- execute_motion preserves gripper commands throughout. Use holding='retain'
  for a verified held object, and choose a separate release after observing
  support. When releasing a standing or thin object onto a support surface,
  ensure the object is seated at rest on support first, then open the gripper
  sufficiently wide (normally full `OPEN` / `opening=1.0`, or an opening proven
  wider than the object's full diagonal footprint) before vertical withdrawal;
  never withdraw with a tight partial opening (such as `0.5`–`0.6`) that can
  pinch or drag the standing object upward by pad friction. After release,
  retreat from the measured pose with unchanged orientation along a clear
  direction before turning or sweeping sideways.
- One intent can move both arms simultaneously. keep_relative=True constrains
  their relative TCP transform; simultaneity alone does not preserve it.
  Omitted arms hold their starting poses. Plan finger/inter-arm clearance:
  the executor checks sampled arm/table geometry, not all scene collisions or
  full inter-arm wrist/link volumes. Driving a viewing or receiving arm toward
  table centre (`x ≈ 0`) at the same time as a carrying arm causes mid-air wrist
  collisions and visual occlusion: enforce **sequential single-arm clearance**
  in shared central zones (for a supported set-down transfer, release, retreat,
  and clear/home the donor arm before bringing the receiver arm in; for an
  in-air handoff, park the donor arm stationary at a verified pose before
  advancing only the receiver arm).
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

## Last-mile contact: nudge and visual_align_and_insert

Keep global planning in your program (which object, which arm, grasp, carry,
pre-align). Delegate only the last 1–5 cm of contact or alignment (pressing,
plugging, seating a part in a slot, hanging on a hook, tipping a spout over a rim)
once the views show the held part or the fingers pre-aligned near the goal.

- `robo.nudge(dxyz, arm, dpitch=, dyaw=, droll=, frame="world" | "tcp",
  stop_on_stall=True)` is a guarded compact relative move from the measured pose
  (metres / degrees, at most 6 cm / 20 degrees per call): one native action per
  5 mm waypoint with no spline ramp (about 2–3 actions per cm), preflighted against
  reach, joint limits and the table (a refusal costs 0 actions). `frame="tcp"` with
  `dxyz=(d, 0, 0)` pushes along the fingers. A stall against contact returns
  `status="stopped"`, `stalled=True`, `stop_reason="contact_or_stall"` and raises
  nothing: read it as "contact reached or blocked", check the wrist view, then
  release/withdraw if seated, or back off 3–5 mm with a reverse nudge and correct
  sideways if misaligned. It measures progress, not force.
- **Pre-contact feature-to-target visual alignment (`robo.align_feature`)**:
  never align `tcp_pos` directly to a narrow receptor (`<= 5 mm` clearance:
  aperture/slot insertion, hook/post hanging, connector plugging, precision
  spout alignment): grasp offset, curved-rim tangent pinches, and
  gravity-pivoted suspension displace the held distal working feature (tip,
  pin, shaft, loop, or spout `feature_xyz`) by `3–15 mm` from `tcp_pos`.
  Stage the distal feature to a `1.0–2.0 cm` pre-contact hover upstream of
  `target_xyz` along the engagement `axis` (default `(0, 0, -1)` downward, or a
  horizontal unit vector for side plugging/hanging). In that post-hover
  observation, inspect `crop_grid` on `cam_head` and `cam_<arm>_wrist` to pick
  the 2D pixels of BOTH the **held distal working feature** and the **receptor
  opening/target**; resolve 3D coordinates with two-view `triangulate`
  (preferred: solves true 3D `feature_xyz` without plane-height parallax error;
  if using single-view `ray_at_z`, pass the feature's actual height `z_feature`
  rather than projecting an elevated hovering feature onto `z_target`, which
  introduces oblique ray parallax). Call
  `robo.align_feature(feature_pt, target_pt, arm=..., axis=..., stroke_m=...)`
  (or transverse `robo.align_feature` / `robo.nudge` followed by a tight
  `crop_grid` check when initial offset `> 6 mm`, then axial
  `robo.nudge(..., stop_on_stall=True)`). `align_feature` requires current
  provenance for a measured held `feature_pt` (rejecting stale pre-transport
  measurements in `0` steps) while accepting an earlier unoccluded static
  `target_pt`, and supports `preview=True` for zero-step preflight. If an axial
  contact stroke stalls well short of the expected engagement depth
  (`achieved_stroke_m << stroke_m`), back off `3–5 mm` along `-axis` before
  releasing or re-nudging so rim contact does not strip a lightly held part.
- `robo.visual_align_and_insert(goal, arm=..., max_steps=40, reserve_steps=20)` runs a
  bounded closed loop with a PEAR visual-delta model (head and wrist views plus a
  zoomed wrist crop): each round one clamped relative delta, executed by `nudge`,
  with the physical outcome (reached / stopped / out_of_reach) shown next round. It
  ends on the model's done, a zero delta, a contact stall when
  `stop_on_contact_complete=True`, or the round / step budget. Check `status`,
  `stop_reason`, the last feedback and the images; it is not a success verdict.
- On tight budgets prefer one loop or a few nudges over repeated full re-approach
  motions for millimetre corrections, and keep `reserve_steps` for retreat and home.
