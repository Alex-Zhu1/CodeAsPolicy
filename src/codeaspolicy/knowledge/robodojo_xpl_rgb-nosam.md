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
- In dynamic scenes or tasks with a benchmark-driven third arm, keep both arms at their starting poses and call `robo.wait(steps=..., until_scene_still=True, still_steps=12)` while watching `cam_head` until the third arm both finishes acting and fully retracts to rest; always reserve `>= 25–30` native steps for final `robo.home("both", open_gripper=True)` cleanup.

## Perception
- Take `view = robo.snapshot()` (or `refresh=True`) and pass `step=view['step']` to `image` and `observation_id=view['observation_id']` to `crop_grid`, `camera_meta`, `camera_ray`, `ray_at_z`, `triangulate`, `project_point`, `observe_object`, and `observe_hold` so pixels and derived camera poses match one capture. Call `robo.require_current(result)` before acting on coordinates; motion invalidates currentness.
- Three calibrated 640x480 colour cameras (`cam_head`, `cam_left_wrist`, `cam_right_wrist`) and no depth image (`camera_meta` derives extrinsics from image-time state). Pixels everywhere are `(row, col)` (row 0 top, col 0 left, matching `image[row, col]` — do not swap with OpenCV `(x, y)`).
- Localise with the visual-geometry workflow:
  1. **Verify pixels visually**: overlay candidate `(row, col)` points with `display(robo.crop_grid(camera, box=(r0, c0, r1, c1), points=[(r, c)], step_px=20, scale=3))`; never trust blind colour/contour thresholds alone.
  2. **Prefer two-view `triangulate` for 3D**: one colour view cannot measure unknown height; observe the same feature in two views (moving a wrist overhead at `pitch=90` or `pitch=45..60` if needed) and call `robo.triangulate(cam1, rc1, cam2, rc2, observation_id=obs_id)` (`miss_m < 0.005` m typical; verify via `project_point`).
  3. **Use `ray_at_z` only on known planes**: pass the true surface height `z = z_support + h` (e.g. `0.765` m for tabletop points, `z_support + h` for top-face pixels of height `h` to avoid oblique ray parallax).
  4. **Track holds**: seed `observe_object(ref, landmarks=...)` and verify with `observe_hold(ref, arm=...)`. Move obstructing arms aside and re-observe after every contact.

## Judging progress
- Nothing reports task success while `run` executes: judge progress from fresh images (`crop_grid`, wrist/head views) and robot state.
- Confirm grasps by seeing the target leave its place and move with the arm via `observe_hold` / `triangulate` (for named lift amounts, add 2–3 cm for sag/slip). Confirm placements after opening and withdrawing: an object on a rim or beside its target is not placed.
- Final-scene checks are strict: positions within ~1 cm, angles within a few degrees, every named/plural object handled, exact press counts (no extra presses), undisturbed neighbours and upright containers/fixtures, articulated parts pushed to travel end, and liquids poured with the container returned upright. Re-check views and fix what is off while budget remains.
- Most tasks require BOTH arms back at their starting pose (within `0.15 m` and `20°`) with BOTH grippers open (`opening > 0.8`, occasionally `> 0.9`) once the scene satisfies the instruction. Complete manipulation before cleanup; establish stable support before releasing and homing.

## Safety and recovery
- There is one episode and no reset: inspect the scene after motion, stop if an object is unstable or an arm is obstructed, and recover in place (an object off the table may be unrecoverable).
- After any failed step, name the cause (wrong object, missed grasp, slip, unreachable target, rim contact, premature release) and change one meaningful variable; never repeat the same failing call twice.
- Verify fresh clearance and observations before every recovery move.
