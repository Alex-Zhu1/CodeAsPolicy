# Operating knowledge for RoboDojo: how the tools behave (no task recipes)

## Robot conventions and gripper geometry
- Two 6-DoF ARX-X5 arms (`"left"` at negative `x`, `"right"` at positive `x`) stand at `x = ±0.30, y = -0.45`; `y` points away, `z` up; table top is `z = 0.765` m. `robo.task` is authoritative.
- Targets drive `tcp_pos` (finger-pad centre, `0.145 m` beyond `eef_pos` along `+x_tcp`). Fingertips extend `+1.3 cm` past TCP; closed pads are `~2.4 cm` wide (`±1.2 cm`), the fixed jaw-rail crossbar `5–7 cm` behind TCP spans `±5 cm` even at `opening=0.0`, and the wrist-camera housing protrudes along `+z_link6`.
- Gripper commands are signed (`CLOSE=1`, `OPEN=-1`, `KEEP=0`; `move_both` uses `"close"/"open"/"keep"`), acting from the first native step. Measured `opening` is `1.0` open / `0.0` closed (`width ≈ 0.002 + 0.109 * opening` m; `GRIP_ZERO_M ≈ 2.3 mm`). Never infer thin-part (`< 3.5 mm`) holding from `opening` alone; verify a short lift visually.
- Angles are degrees (`pitch` 0 level / 90 down, `yaw` 0 away / +90 left, `roll` about `+x_tcp`; at `pitch=90`, `yaw` sets the closing line and `roll=0`). Build poses with `robo.pose_from_axes(xyz, point_dir, close_dir, roll_deg=...)` and preflight chained poses with `robo.ik(..., jaw_symmetry=True, twist_rad=..., check_other_arm=True)`.

## Motion, timed splines, and Coarse + Delta controls
- `execute_motion(intent_id, ...)` handles coarse C2 transport to `2.5–4 cm` pre-action hover poses; `delta_grasp`, `align_feature`, and `nudge` handle close-range (`<= 4 cm`) pickup, handoff, alignment, seating, and pressing.
- For low-level calls when `speed_control` is true, `move_to` and `move_both` use C2 timed joint splines with `motion=robo.motion_profile(velocity=1.0, acceleration=4.0, jerk=40.0)` (rad/s, rad/s², rad/s³) or phases (`"native"`, `"transit"`, `"approach"`, `"carry"`, `"precision"`, or `motion_window`). `move_both(left=final_left, right=final_right, via=[...], motion=profile)` joins waypoints continuously in simulation time; `preview=True` checks feasibility without moving.
- Straight-down (`pitch=90`) reach stays low (`y <= 0.0` at `z=0.80`, `y <= -0.05` at `z=0.90`, unreachable near `z=1.0`); tilted/level reaches farther and higher. Point along any required rotation axis (`pitch=90` for a vertical twist) so rotation runs on `joint6`. Lift held objects so `object_bottom_z = tcp_z - L_below` clears obstacles (`+0.03` m) before lateral travel.
- Prefer supported `place-and-regrasp` on a clear central patch for compact uniform objects (`< ~6–7 cm`), reserving in-air handoffs for elongated parts (`>= 5 cm` free span) or orthogonal 90° flat-face head clamps; enforce sequential single-arm clearance in shared central zones.
- Current adapters report `force_control=False`, `force_feedback=False`, `gripper_force_control=False`, and `impedance_control=False`: neither obstructed closure nor tracking lag measures force.
- In tasks with a benchmark-driven third arm, home both arms and step time with `set_gripper(arm, robo.OPEN)` while watching `cam_head` until it finishes.

## The VLA
- `pi0_pick(prompt=None, arm, max_chunks)` runs the frozen Pi0.5 whole-task policy closed loop on all three cameras and both arms (it picks the arm itself; `arm` is a label). Each chunk costs up to 50 native actions.
- `Pick.success` is a provisional lift heuristic that errs both ways: inspect `Pick.left` / `Pick.right` and the wrist view to verify the intended object is held. With `prompt=None` it uses `robo.task`; overrides must distinguish the target and contain no `<...>` markers.
- When Pi0.5 drifts or repeats a failing motion, switch to geometric Coarse + Delta primitives to complete the step or finish the task.

## Perception
- Take `view = robo.snapshot()` (or `refresh=True`) and pass `step=view['step']` to `image` and `observation_id=view['observation_id']` to `locate`, `segment`, `back_project`, `crop_grid`, `ray_at_z`, `triangulate`, and `camera_meta` so images, depth, and calibration match one capture. Always check `robo.require_current(result)` before acting on coordinates; motion invalidates currentness.
- Three 640x480 cameras: `cam_head` (global layout/identity) and `cam_left_wrist` / `cam_right_wrist` (close-up pre-grasp/hold). Pixels are `(row, col)` (row 0 top, col 0 left).
- `locate(text, camera)` uses SAM3 + depth for 3D `center_xyz`, `top_z`, footprint, and `grasp_yaw` (verify the mask against the image; handles/spouts shift `center_xyz`; occlusion shrinks the measured footprint). `segment` + `back_project` measure arbitrary pixels.
- For tight last-mile features, inspect `display(robo.crop_grid(...))` and prefer two-view `triangulate` (true 3D with `miss_m < 0.005` m, no plane-height parallax) over single-view `ray_at_z` (when using `ray_at_z`, pass the true surface height `z = z_support + h`, not bare table height). Move obstructing arms aside and re-observe after every contact.

## Judging progress
- Nothing reports task success while `run` executes: judge progress from fresh images and robot state.
- Confirm grasps by seeing the target leave its place and move with the arm (for named lift amounts, add 2–3 cm for sag/slip). Confirm placements after opening and withdrawing: an object on a rim or beside its target is not placed.
- Final-scene checks are strict: positions within ~1 cm, angles within a few degrees, every named/plural object handled, exact press counts (no extra presses), undisturbed neighbours and upright containers/fixtures, articulated parts pushed to travel end, and liquids poured with the container returned upright. Re-check (`locate` / views) and fix what is off while budget remains.
- Most tasks require BOTH arms back at their starting pose (within `0.15 m` and `20°`) with BOTH grippers open (`opening > 0.8`, occasionally `> 0.9`) once the scene satisfies the instruction. Complete manipulation before cleanup; establish stable support before releasing and homing.

## Safety and recovery
- There is one episode and no reset: inspect the scene after motion, stop if an object is unstable or an arm is obstructed, and recover in place (an object off the table may be unrecoverable).
- After any failed step, name the cause (wrong object, missed grasp, slip, unreachable target, rim contact, premature release) and change one meaningful variable; never repeat the same failing call twice.
- Verify fresh clearance and observations before every recovery move.
