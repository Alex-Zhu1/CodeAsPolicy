"""Tests of the PEAR Visual-Delta distillation pipeline (pure Python; numpy optional)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from codeaspolicy import visual_delta_distill as vd

# ---------------------------------------------------------------------------
# helpers: synthetic states and images
# ---------------------------------------------------------------------------


def quat_wxyz(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    return list(q)


def ee_pose(tcp, pitch=90.0, yaw=0.0, roll=0.0):
    """link6 ``[x, y, z, qw, qx, qy, qz]`` whose grasp point is ``tcp`` (metres)."""

    rot = vd.orientation(pitch, yaw, roll)
    link6 = [tcp[i] - rot[i][0] * vd.TCP_OFFSET for i in range(3)]
    return link6 + quat_wxyz(rot)


def state(right_tcp, left_tcp=(-0.2, 0.0, 0.95), opening=0.8, left_opening=0.8, **angles):
    return {
        "right_ee_pose": ee_pose(right_tcp, **angles),
        "left_ee_pose": ee_pose(left_tcp),
        "right_ee_joint_state": [opening],
        "left_ee_joint_state": [left_opening],
    }


def pose(tcp, pitch=90.0, yaw=0.0, roll=0.0, opening=0.5):
    return vd.ArmPose(
        tcp=tuple(tcp),
        rot=tuple(tuple(r) for r in vd.orientation(pitch, yaw, roll)),
        opening=opening,
    )


def frame(index, tcp, *, phase=None, grip=None, stalled=False, native=None, **angles):
    poses = {"right": pose(tcp, **angles), "left": pose((-0.2, 0.0, 0.95))}
    return vd.Frame(
        index=index,
        poses=poses,
        native_step=native if native is not None else index * 3,
        phase=phase,
        gripper={"right": grip} if grip is not None else {},
        stalled=stalled,
        stop_reason="contact_or_stall" if stalled else None,
    )


def tiny_image(width=8, height=6, seed=0):
    return [[((x * 30 + seed) % 256, (y * 40) % 256, (x + y + seed) % 256) for x in range(width)]
            for y in range(height)]


def parse_reply(text):
    block = text.split("```json", 1)[1].rsplit("```", 1)[0]
    return json.loads(block)


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------


def test_width_cm_matches_grip_fit():
    assert vd.width_cm(0.0) == 0.2
    assert vd.width_cm(0.3) == round((0.0023 + 0.1091 * 0.3) * 100, 1) == 3.5
    assert vd.width_cm(None) is None


def test_gripper_sign_translation_is_explicit():
    assert vd.pear_gripper_to_robodojo(1) == vd.ROBODOJO_OPEN == -1
    assert vd.pear_gripper_to_robodojo(-1) == vd.ROBODOJO_CLOSE == 1
    assert vd.pear_gripper_to_robodojo(0) == vd.ROBODOJO_KEEP == 0
    assert vd.robodojo_gripper_to_pear(-1) == 1
    assert vd.robodojo_gripper_to_pear(1) == -1
    assert vd.robodojo_gripper_to_pear(0) == 0
    assert vd.pear_to_ee_command(1) == 1.0
    assert vd.pear_to_ee_command(-1) == 0.0
    assert vd.pear_to_ee_command(0, current=0.0) == 0.0
    # ee command levels are state; switches only on change
    assert vd.ee_command_to_pear(1.0) == 1
    assert vd.ee_command_to_pear(0.0) == -1
    assert vd.ee_command_to_pear(0.0, previous=1.0) == -1
    assert vd.ee_command_to_pear(1.0, previous=0.0) == 1
    assert vd.ee_command_to_pear(1.0, previous=1.0) == 0
    for g in (1, -1, 0):
        assert vd.robodojo_gripper_to_pear(vd.pear_gripper_to_robodojo(g)) == g


@pytest.mark.parametrize("pitch,yaw,roll", [(90, 0, 0), (30, 20, -15), (0, 0, 0), (45, 170, 10)])
def test_angles_round_trip(pitch, yaw, roll):
    got = vd.angles(vd.orientation(pitch, yaw, roll))
    if pitch == 90:
        assert got[0] == pytest.approx(90.0)
    else:
        assert got == pytest.approx((pitch, yaw, roll), abs=1e-6)


def test_quaternion_matches_rotation():
    rot = vd.orientation(30, 20, -15)
    back = vd.rot_from_quat(quat_wxyz(rot))
    for i in range(3):
        assert back[i] == pytest.approx(rot[i], abs=1e-9)


def test_pure_python_math_matches_host_kinematics():
    np = pytest.importorskip("numpy")
    kin = pytest.importorskip("codeaspolicy.hosts.robodojo_kinematics")
    for p, y, r in [(30, 20, -15), (60, -100, 40), (89.5, 10, 0)]:
        ours = vd.orientation(p, y, r)
        assert np.allclose(np.array(ours), kin.orientation(p, y, r))
        assert vd.angles(ours) == pytest.approx(kin.angles(np.array(ours)), abs=1e-6)
        q = quat_wxyz(ours)
        assert np.allclose(np.array(vd.rot_from_quat(q)), kin.rot_from_quat(q))
    assert vd.TCP_OFFSET == kin.TCP_OFFSET
    # forward kinematics fallback when only joints are known
    q = [0.0, 0.8, 0.9, 0.0, 0.0, 0.0]
    got = vd.arm_pose({"right_arm_joint_state": q}, "right")
    expected = kin.tcp(kin.fk("right", q))
    assert got.tcp == pytest.approx(tuple(float(v) for v in expected), abs=1e-9)


def test_arm_pose_uses_tcp_offset():
    got = vd.arm_pose(state((0.1, 0.05, 0.9)), "right")
    assert got.tcp == pytest.approx((0.1, 0.05, 0.9), abs=1e-9)
    assert got.opening == 0.8
    assert got.angles_deg[0] == pytest.approx(90.0)
    assert vd.arm_pose({}, "right") is None


def test_state_json_units():
    poses = {"right": pose((0.1234, 0.05, 0.9), pitch=30, yaw=20, roll=-15, opening=0.3),
             "left": pose((-0.2, 0.0, 0.95))}
    single = vd.state_json(poses, "right")
    assert single["arm"]["x"] == 12.3 and single["arm"]["z"] == 90.0
    assert single["arm"]["pitch"] == round(math.radians(30), 2)
    assert single["arm"]["roll"] == round(math.radians(-15), 2)
    assert single["gripper"] == {"width": 3.5}
    both = vd.state_json(poses, "both")
    assert set(both) == {"left_arm", "right_arm", "left_gripper", "right_gripper"}


def test_pose_delta_world_and_tcp_frames_round_trip():
    a = pose((0.10, 0.00, 0.90), pitch=30, yaw=20, roll=0)
    b = pose((0.12, 0.01, 0.88), pitch=35, yaw=15, roll=5)
    world = vd.pose_delta(a, b, "world")
    assert world["x"] == pytest.approx(2.0) and world["y"] == pytest.approx(1.0)
    assert world["z"] == pytest.approx(-2.0)
    assert world["pitch"] == pytest.approx(math.radians(5), abs=1e-3)
    assert world["yaw"] == pytest.approx(math.radians(-5), abs=1e-3)
    assert world["roll"] == pytest.approx(math.radians(5), abs=1e-3)
    for frame_name in ("world", "tcp"):
        delta = vd.pose_delta(a, b, frame_name, linear_decimals=6, angular_decimals=9)
        back = vd.apply_delta(a, delta, frame_name)
        assert back.tcp == pytest.approx(b.tcp, abs=1e-7)
        assert back.angles_deg == pytest.approx(b.angles_deg, abs=1e-5)
    # straight down, gripper +x points at world -z: a tcp-frame +x move is a world descent
    down = pose((0.0, 0.0, 0.9), pitch=90)
    lower = pose((0.0, 0.0, 0.88), pitch=90)
    tcp = vd.pose_delta(down, lower, "tcp")
    assert tcp["x"] == pytest.approx(2.0) and abs(tcp["y"]) < 1e-9 and abs(tcp["z"]) < 1e-9
    with pytest.raises(ValueError):
        vd.pose_delta(a, b, "base")


def test_yaw_delta_wraps():
    a = pose((0, 0, 0.9), pitch=30, yaw=175)
    b = pose((0, 0, 0.9), pitch=30, yaw=-175)
    assert vd.pose_delta(a, b)["yaw"] == pytest.approx(math.radians(10), abs=1e-3)


# ---------------------------------------------------------------------------
# macro-step coalescer
# ---------------------------------------------------------------------------


def line(n, step_m=0.005, start=(0.1, 0.0, 0.9), axis=0, **kw):
    out = []
    for i in range(n):
        tcp = list(start)
        tcp[axis] += i * step_m
        out.append(frame(i, tcp, **kw))
    return out


def test_straight_motion_is_one_macro_step():
    frames = line(9, phase="align")  # 8 transitions of 0.5 cm along x
    steps = vd.coalesce(frames)
    assert len(steps) == 1
    step = steps[0]
    assert step.arm == "right"
    assert (step.start, step.end) == (0, 8)
    assert step.action["arm"]["x"] == pytest.approx(4.0)
    assert step.action["gripper"] == {"width": 0}
    assert step.native_steps == 24
    assert step.status == "reached" and step.phase == "align"


def test_displacement_cap_splits_long_moves():
    frames = line(31, step_m=0.005)  # 15 cm
    steps = vd.coalesce(frames, config=vd.CoalesceConfig(max_linear_m=0.06))
    assert len(steps) == 3
    assert all(s.linear_m <= 0.06 + 1e-9 for s in steps)
    assert sum(s.action["arm"]["x"] for s in steps) == pytest.approx(15.0)
    assert [s.start for s in steps[1:]] == [s.end for s in steps[:-1]]


def test_angular_cap_splits_rotations():
    frames = [frame(i, (0.1, 0.0, 0.9), pitch=30, yaw=i * 5.0) for i in range(11)]
    steps = vd.coalesce(frames, config=vd.CoalesceConfig(max_angular_deg=20.0))
    assert len(steps) == 3
    assert sum(s.action["arm"]["yaw"] for s in steps) == pytest.approx(math.radians(50), abs=2e-3)


def test_gripper_switch_closes_a_macro_step():
    frames = line(6, grip=1.0)
    frames[3].gripper = {"right": 0.0}
    frames[4].gripper = {"right": 0.0}
    frames[5].gripper = {"right": 0.0}
    steps = vd.coalesce(frames)
    assert [(s.start, s.end) for s in steps] == [(0, 3), (3, 5)]
    assert steps[0].action["gripper"]["width"] == vd.PEAR_CLOSE
    assert steps[1].action["gripper"]["width"] == vd.PEAR_KEEP


def test_gripper_switch_without_motion_is_kept():
    frames = [frame(i, (0.1, 0.0, 0.9), grip=1.0 if i < 2 else 0.0) for i in range(4)]
    steps = vd.coalesce(frames, arm="right")
    assert len(steps) == 1
    assert steps[0].action["gripper"]["width"] == -1
    assert steps[0].action["arm"]["x"] == 0.0


def test_opening_fallback_when_commands_missing():
    frames = [frame(i, (0.1, 0.0, 0.9)) for i in range(3)]
    frames[2].poses["right"] = pose((0.1, 0.0, 0.9), opening=0.1)
    steps = vd.coalesce(frames, arm="right")
    assert steps[-1].action["gripper"]["width"] == -1


def test_phase_boundary_and_stall_split():
    frames = line(4, phase="approach") + [
        frame(4 + i, (0.1 + 0.015 + 0.002 * (i + 1), 0.0, 0.9), phase="insert") for i in range(3)
    ]
    frames[5].stalled, frames[5].stop_reason = True, "contact_or_stall"
    steps = vd.coalesce(frames)
    assert [(s.start, s.end, s.phase) for s in steps] == [
        (0, 3, "approach"), (3, 5, "insert"), (5, 6, "insert")
    ]
    assert steps[1].status == "stopped" and steps[1].stop_reason == "contact_or_stall"
    assert steps[2].status == "reached"


def test_zero_motion_hold_noise_is_dropped():
    frames = line(4) + [frame(4 + i, (0.1 + 0.015, 0.0, 0.9), phase="hold") for i in range(4)]
    frames[4].poses["right"] = pose((0.1 + 0.0151, 0.0, 0.9))
    steps = vd.coalesce(frames)
    assert len(steps) == 1
    assert steps[0].phase is None


def test_bimanual_selection_and_keys():
    frames = line(5)
    for i, f in enumerate(frames):
        f.poses["left"] = pose((-0.2, 0.01 * i, 0.95))
    assert vd.select_arm(frames) == "both"
    steps = vd.coalesce(frames)
    action = steps[0].action
    assert set(action) == {"left_arm", "right_arm", "left_gripper", "right_gripper"}
    assert action["left_arm"]["y"] == pytest.approx(4.0)
    assert action["right_arm"]["x"] == pytest.approx(2.0)


def test_coalesce_short_inputs():
    assert vd.coalesce([]) == []
    assert vd.coalesce(line(1)) == []


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------


def with_images(frames, tmp_path):
    for f in frames:
        for cam in vd.CAMERAS:
            path = tmp_path / f"{f.index:04d}" / f"{cam}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(vd.encode_png(tiny_image(seed=f.index)))
            f.images[cam] = str(path)
    return frames


def test_conversation_structure(tmp_path):
    frames = with_images(line(4, phase="approach") + [
        frame(4 + i, (0.117 + 0.002 * i, 0.0, 0.9), phase="insert", grip=0.0) for i in range(3)
    ], tmp_path)
    frames[5].stalled, frames[5].stop_reason = True, "contact_or_stall"
    steps = vd.coalesce(frames)
    messages, images = vd.build_conversation(frames, steps, "insert the key", success=True)
    roles = [m["role"] for m in messages]
    assert roles[0] == "system" and roles[1::2] == ["user"] * (len(steps) + 1)
    assert roles[2::2] == ["assistant"] * (len(steps) + 1)
    assert "## ENVIRONMENT" in messages[0]["content"]
    assert "## TASK\ninsert the key" in messages[0]["content"]
    tokens = sum(m["content"].count(vd.IMAGE_TOKEN) for m in messages)
    assert tokens == len(images) == 3 * (len(steps) + 1)
    assert "[cam_head] <image>" in messages[1]["content"]
    assert "none (first turn)" in messages[1]["content"]
    # the stop is reported in the turn after the stopped action
    stopped = [i for i, s in enumerate(steps) if s.status == "stopped"][0]
    assert "stopped (contact_or_stall)" in messages[1 + 2 * (stopped + 1)]["content"]
    first = messages[2]["content"]
    assert first.startswith("<think>") and "</think>" in first
    assert parse_reply(first) == steps[0].action
    final = parse_reply(messages[-1]["content"])
    assert final["done"] is True and final["arm"]["x"] == 0.0
    state_text = messages[1]["content"].split("```json", 1)[1].split("```", 1)[0]
    assert json.loads(state_text)["arm"]["x"] == 10.0


def test_conversation_options_and_crops(tmp_path):
    frames = with_images(line(5), tmp_path)
    steps = vd.coalesce(frames)
    options = vd.DatasetOptions(think=False, cameras=("cam_right_wrist",), done_turn=False)
    def crop(cam, path):
        return str(vd.crop_png(Path(path), tmp_path / "c" / f"{cam}.png"))

    messages, images = vd.build_conversation(frames, steps, "t", options, success=True, crop=crop)
    assert len(messages) == 3
    assert not messages[2]["content"].startswith("<think>")
    assert "[cam_right_wrist_crop] <image>" in messages[1]["content"]
    assert len(images) == 2 and images[1].endswith("cam_right_wrist.png")


def test_rl_samples_split_turns(tmp_path):
    frames = with_images(line(31), tmp_path)
    steps = vd.coalesce(frames)
    messages, images = vd.build_conversation(frames, steps, "t", success=True)
    samples = vd.rl_samples(messages, images)
    assert len(samples) == len(steps) + 1
    assert samples[0]["prompt"][-1]["role"] == "user"
    assert samples[0]["solution"] == steps[0].action
    assert len(samples[1]["images"]) == 6
    assert samples[-1]["solution"]["done"] is True


def test_think_text_words():
    action = {"arm": {"x": 1.2, "y": 0, "z": -0.5, "roll": 0, "pitch": 0, "yaw": 0.1},
              "gripper": {"width": -1}, "done": True}
    text = vd.think_text(action, "world", "insert")
    assert "1.2 cm right" in text and "0.5 cm down" in text and "gripper close" in text
    assert text.startswith("Phase insert:")


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


def test_png_round_trip_and_crop(tmp_path):
    image = tiny_image(10, 8, seed=3)
    data = vd.encode_png(image)
    assert data.startswith(b"\x89PNG")
    assert vd.decode_png(data) == [[tuple(px) for px in row] for row in image]
    src = tmp_path / "a.png"
    src.write_bytes(data)
    dst = vd.crop_png(src, tmp_path / "out" / "a_crop.png", (0.2, 0.25, 0.7, 0.75))
    rows = vd.decode_png(dst.read_bytes())
    assert len(rows) == 4 and len(rows[0]) == 5
    assert rows[0][0] == tuple(image[2][2])
    assert vd.crop_box(10, 8, (0.0, 0.0, 0.01, 0.01)) == (0, 0, 1, 1)


def test_decode_png_filters():
    # rows with Sub, Up, Average and Paeth filters, built by hand
    width, rows = 3, [[(10, 20, 30), (11, 22, 33), (12, 24, 36)] for _ in range(4)]
    raw = bytearray()
    prev = [0] * (width * 3)
    for kind, row in zip((1, 2, 3, 4), rows):
        line_bytes = [v for px in row for v in px]
        raw.append(kind)
        for i, v in enumerate(line_bytes):
            a = line_bytes[i - 3] if i >= 3 else 0
            b = prev[i]
            c = prev[i - 3] if i >= 3 else 0
            pred = {1: a, 2: b, 3: (a + b) >> 1, 4: vd._paeth(a, b, c)}[kind]
            raw.append((v - pred) & 0xFF)
        prev = line_bytes
    import struct
    import zlib

    header = struct.pack(">IIBBBBB", width, 4, 8, 2, 0, 0, 0)
    data = (vd._PNG_SIGNATURE + vd._png_chunk(b"IHDR", header)
            + vd._png_chunk(b"IDAT", zlib.compress(bytes(raw))) + vd._png_chunk(b"IEND", b""))
    assert vd.decode_png(data) == rows
    with pytest.raises(ValueError):
        vd.decode_png(b"not a png")


# ---------------------------------------------------------------------------
# offline episode directories
# ---------------------------------------------------------------------------


def write_episode(root: Path, success=True, with_action_ids=True):
    """A small episode: approach (move_to), close gripper, insert with a contact stop."""

    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({"environment_success": success, "env_steps": 40}))
    (root / "card.json").write_text(json.dumps({"task_language": "insert the key into the lock"}))
    calls = [
        {"index": 0, "tool": "move_to", "kwargs": {"arm": "right"}, "ok": True, "stateful": True,
         "env_steps": 12, "summary": {}, "action_id": "a0"},
        {"index": 1, "tool": "close_gripper", "kwargs": {"arm": "right"}, "ok": True,
         "stateful": True, "env_steps": 4, "summary": {}, "action_id": "a1"},
        {"index": 2, "tool": "nudge", "kwargs": {"arm": "right", "phase": "insert"}, "ok": True,
         "stateful": True, "env_steps": 6,
         "summary": {"status": "stopped", "stalled": True, "stop_reason": "contact_or_stall"},
         "action_id": "a2"},
        {"index": 3, "tool": "nudge", "kwargs": {"arm": "right", "phase": "insert"}, "ok": True,
         "stateful": True, "env_steps": 4, "summary": {"status": "reached"}, "action_id": "a3"},
    ]
    (root / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls) + "garbage\n")
    # (tcp, native step, action id, opening)
    plan = [
        ((0.10, 0.00, 1.00), 0, None, 0.8),
        ((0.10, 0.02, 0.98), 6, "a0", 0.8),
        ((0.10, 0.04, 0.96), 12, "a0", 0.8),
        ((0.10, 0.04, 0.96), 16, "a1", 0.3),
        ((0.10, 0.04, 0.95), 19, "a2", 0.3),
        ((0.10, 0.04, 0.945), 22, "a2", 0.3),
        ((0.10, 0.042, 0.945), 26, "a3", 0.3),
    ]
    for i, (tcp, native, action_id, opening) in enumerate(plan):
        folder = root / "steps" / f"{i:04d}"
        folder.mkdir(parents=True)
        for cam in vd.CAMERAS:
            (folder / f"{cam}.png").write_bytes(vd.encode_png(tiny_image(seed=i)))
        meta = {"native_steps": native, "action_id": action_id if with_action_ids else None,
                "gripper_commands": {}, "images": {c: {"artifact": f"{c}.png"} for c in vd.CAMERAS}}
        record = {"state": state(tcp, opening=opening), "instruction": "", "observation": meta}
        (folder / "observation.json").write_text(json.dumps(record))
    return root


@pytest.mark.parametrize("with_action_ids", [True, False])
def test_load_episode_maps_calls(tmp_path, with_action_ids):
    episode = vd.load_episode(write_episode(tmp_path / "ep", with_action_ids=with_action_ids))
    assert episode.task == "insert the key into the lock"
    assert episode.success is True and episode.native_steps == 40
    assert len(episode.calls) == 4  # malformed line skipped
    phases = [f.phase for f in episode.frames]
    assert phases == [None, "move_to", "move_to", "close_gripper", "insert", "insert", "insert"]
    assert [f.stalled for f in episode.frames] == [False] * 5 + [True, False]
    assert episode.frames[3].gripper == {"right": 0.0}
    assert set(episode.frames[0].images) == set(vd.CAMERAS)


def test_episode_records_full_episode(tmp_path):
    episode = vd.load_episode(write_episode(tmp_path / "ep"))
    records = vd.episode_records(episode)
    assert len(records) == 1
    record = records[0]
    assert record["protocol"] == vd.PROTOCOL and record["arm"] == "right"
    statuses = [s["status"] for s in record["steps"]]
    assert "stopped" in statuses
    closes = [s for s in record["steps"] if s["action"]["gripper"]["width"] == -1]
    assert len(closes) == 1 and closes[0]["phase"] == "close_gripper"
    tokens = sum(m["content"].count(vd.IMAGE_TOKEN) for m in record["messages"])
    assert tokens == len(record["images"])
    assert parse_reply(record["messages"][-1]["content"])["done"] is True


def test_last_mile_and_success_filters(tmp_path):
    episode = vd.load_episode(write_episode(tmp_path / "ep"))
    options = vd.ExtractOptions(last_mile=True, crop=True)
    records = vd.episode_records(episode, options, tmp_path / "out")
    assert len(records) == 1
    record = records[0]
    assert record["source"]["frames"] == [2, 6]
    assert set(record["phases"]) <= {"close_gripper", "insert"}
    assert "Sub-goal" in record["messages"][0]["content"]
    crops = [p for p in record["images"] if p.endswith("_crop.png")]
    assert crops and all(Path(p).is_file() for p in crops)
    failed = vd.load_episode(write_episode(tmp_path / "bad", success=False))
    assert vd.episode_records(failed, vd.ExtractOptions(success_only=True)) == []
    near = vd.episode_records(episode, vd.ExtractOptions(radius_m=0.012))
    assert near and near[0]["source"]["frames"][0] >= 3


def test_rl_format_records(tmp_path):
    episode = vd.load_episode(write_episode(tmp_path / "ep"))
    records = vd.episode_records(episode, vd.ExtractOptions(format="rl"))
    assert len(records) == records[0]["macro_steps"] + 1
    assert [r["turn"] for r in records] == list(range(len(records)))
    assert all("solution" in r and r["prompt"][-1]["role"] == "user" for r in records)


def test_cli_extract_and_stats(tmp_path, capsys):
    write_episode(tmp_path / "runs" / "ep1")
    write_episode(tmp_path / "runs" / "ep2", success=False)
    out = tmp_path / "data" / "train.jsonl"
    assert vd.main(["extract", str(tmp_path / "runs"), "-o", str(out), "--success-only"]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["episodes"] == 2 and counts["used"] == 1 and counts["records"] == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["success"] is True
    assert vd.main(["stats", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["macro_steps"] == counts["macro_steps"]
    assert 0 < summary["contact_stop_ratio"] < 1
    assert summary["gripper_switches"]["close"] == 1
    assert vd.main(["stats", str(tmp_path / "runs")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["episodes"] == 2 and summary["episode_native_steps"] == 80
    rl = tmp_path / "data" / "rl.jsonl"
    assert vd.main(["extract", str(tmp_path / "runs" / "ep1"), "-o", str(rl), "--format", "rl",
                    "--frame", "tcp", "--last-mile", "--no-images"]) == 0
    sample = json.loads(rl.read_text().splitlines()[0])
    assert sample["frame"] == "tcp" and sample["images"] == []


def test_module_entry_point(tmp_path):
    import subprocess
    import sys

    write_episode(tmp_path / "ep")
    out = tmp_path / "x.jsonl"
    src = Path(vd.__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "codeaspolicy.visual_delta_distill", "extract", str(tmp_path / "ep"),
         "-o", str(out)],
        capture_output=True, text=True, env={"PYTHONPATH": str(src), "PATH": ""},
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()


# ---------------------------------------------------------------------------
# live recorder
# ---------------------------------------------------------------------------


class FakeEnv:
    """Moves the right TCP by action["dz"] metres; reports images and a step counter."""

    def __init__(self):
        self.tcp = [0.1, 0.0, 1.0]
        self.opening = 0.8
        self.steps = 0
        self.extra = "delegated"

    def get_obs(self):
        vision = {cam: {"color": tiny_image(seed=self.steps)} for cam in vd.CAMERAS}
        return {"state": state(tuple(self.tcp), opening=self.opening), "vision": vision,
                "instruction": "hang the mug"}

    def get_status(self):
        return {"step": self.steps * 2, "step_limit": 300}

    def step(self, action):
        self.tcp[2] += action.get("dz", 0.0)
        if action.get("right_ee_joint_state") == [0.0]:
            self.opening = 0.3
        self.steps += 1
        return None


def drive(recorder):
    recorder.get_obs()
    with recorder.phase("approach"):
        for _ in range(4):
            recorder.step({"dz": -0.01, "right_ee_joint_state": [1.0]})
            recorder.get_status()
            recorder.get_obs()
    recorder.step({"right_ee_joint_state": [0.0]})
    recorder.get_obs()
    recorder.set_phase("insert")
    for _ in range(2):
        recorder.step({"dz": -0.002, "right_ee_joint_state": [0.0]})
        recorder.get_obs()
    recorder.mark_stop()
    recorder.step({"dz": 0.003, "right_ee_joint_state": [0.0]})


def test_recorder_records_and_flushes(tmp_path):
    env = FakeEnv()
    recorder = vd.VisualDeltaEnvRecorder(env, tmp_path / "rec")
    assert recorder.extra == "delegated"
    drive(recorder)
    recorder.get_obs()
    recorder.get_obs()  # repeated reads without an action are not new frames
    assert len(recorder.frames) == 9
    assert recorder.frames[1].phase == "approach" and recorder.frames[-1].phase == "insert"
    assert recorder.frames[7].stalled and not recorder.frames[8].stalled
    assert recorder.frames[2].native_step == 4  # from get_status
    records = recorder.on_trial_end({"environment_success": True})
    assert len(records) == 1
    record = records[0]
    assert record["task"] == "hang the mug" and record["success"] is True
    phases = [s["phase"] for s in record["steps"]]
    assert phases[0] == "approach" and "insert" in phases
    grips = [s["action"]["gripper"]["width"] for s in record["steps"]]
    assert grips.count(-1) == 1 and 1 not in grips
    assert record["steps"][0]["action"]["arm"]["z"] == pytest.approx(-4.0)
    assert any(s["status"] == "stopped" for s in record["steps"])
    assert all(Path(p).is_file() for p in record["images"])
    lines = (tmp_path / "rec" / "visual_delta.jsonl").read_text().splitlines()
    assert len(lines) == 1
    # the trial directory re-extracts to the same macro steps
    again = vd.episode_records(vd.load_episode(tmp_path / "rec" / "trial_0000"))
    assert [s["action"] for s in again[0]["steps"]] == [s["action"] for s in record["steps"]]
    assert recorder.trial == 1 and recorder.frames == []


def test_recorder_flush_captures_last_obs_and_appends(tmp_path):
    env = FakeEnv()
    recorder = vd.VisualDeltaEnvRecorder(env, tmp_path / "rec", task="t", save_images=False)
    recorder.get_obs()
    recorder.step({"dz": -0.02})
    first = recorder.flush(False)  # no get_obs after the step: flush reads it
    assert first and first[0]["success"] is False and first[0]["images"] == []
    recorder.get_obs()
    recorder.step({"dz": -0.02})
    recorder.on_trial_end()
    lines = (tmp_path / "rec" / "visual_delta.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["source"]["episode"].endswith("trial_0001")


def test_recorder_uses_step_return_obs(tmp_path):
    class ObsEnv(FakeEnv):
        def step(self, action):
            super().step(action)
            return self.get_obs()

    recorder = vd.VisualDeltaEnvRecorder(ObsEnv(), tmp_path / "rec", image_every=2)
    recorder.get_obs()
    for _ in range(3):
        recorder.step({"dz": -0.01})
    assert len(recorder.frames) == 4
    assert recorder.frames[1].images == {} and recorder.frames[2].images


# ---------------------------------------------------------------------------
# near-vertical Euler artefacts and oversize transitions (verifier #9)
# ---------------------------------------------------------------------------


def test_near_vertical_roll_yaw_swap_is_folded():
    # 89.5 deg is inside angles()' vertical band (reported roll 0), 88.8 deg is outside it:
    # two poses less than 2 degrees apart whose Euler angles differ by ~50 deg in yaw and roll
    a = pose((0.1, 0.0, 0.9), pitch=89.5, yaw=40, roll=50)
    b = pose((0.1, 0.0, 0.89), pitch=88.8, yaw=40, roll=50.5)
    assert a.angles_deg[2] == 0.0 and abs(b.angles_deg[2]) > 45  # the artefact
    geodesic = vd.rotation_error(vd._rot(a), vd._rot(b))
    assert geodesic < 2.0
    delta = vd.pose_delta(a, b)
    assert delta["roll"] == 0.0
    assert abs(delta["yaw"]) < math.radians(2) and abs(delta["pitch"]) < math.radians(2)
    reached = vd.apply_delta(a, delta)
    assert vd.rotation_error(vd._rot(reached), vd._rot(b)) <= 2.0
    linear, angular = vd._motion(a, b)
    assert angular == pytest.approx(geodesic) and angular < 2.0
    steps = vd.coalesce([frame(0, a.tcp, pitch=89.5, yaw=40, roll=50),
                         frame(1, b.tcp, pitch=88.8, yaw=40, roll=50.5)], arm="right")
    assert steps[0].angular_deg < 2.0 and not steps[0].oversize
    assert steps[0].action["arm"]["roll"] == 0.0


def test_near_vertical_fold_pointing_up_uses_yaw_plus_roll():
    a = pose((0.1, 0.0, 0.9), pitch=-89.5, yaw=40, roll=50)
    b = pose((0.1, 0.0, 0.9), pitch=-88.8, yaw=40, roll=50.5)
    delta = vd.pose_delta(a, b)
    assert delta["roll"] == 0.0 and abs(delta["yaw"]) < math.radians(2)
    reached = vd.apply_delta(a, delta)
    assert vd.rotation_error(vd._rot(reached), vd._rot(b)) <= 2.0


def test_roll_is_only_folded_beyond_near_vertical_threshold():
    a = pose((0.1, 0.0, 0.9), pitch=80, yaw=0, roll=0)
    b = pose((0.1, 0.0, 0.9), pitch=80, yaw=10, roll=60)
    delta = vd.pose_delta(a, b, angular_decimals=6)
    assert delta["roll"] == pytest.approx(math.radians(60), abs=1e-5)
    assert delta["yaw"] == pytest.approx(math.radians(10), abs=1e-5)
    # both beyond 85 deg: the shared rule (#10) folds deterministically, yaw - sign(p) * roll
    a = pose((0.1, 0.0, 0.9), pitch=87, yaw=0, roll=0)
    b = pose((0.1, 0.0, 0.9), pitch=87, yaw=10, roll=60)
    delta = vd.pose_delta(a, b, angular_decimals=6)
    assert delta["roll"] == 0.0
    assert delta["yaw"] == pytest.approx(math.radians(10 - 60), abs=1e-5)


def test_canonical_angles_and_state_json_near_vertical():
    assert vd.canonical_angles(30, 20, -15) == (30, 20, -15)
    assert vd.canonical_angles(88, 40, 50) == (88, -10, 0.0)
    assert vd.canonical_angles(-88, 40, 50) == (-88, 90, 0.0)
    assert vd.canonical_angles(89, 170, -20) == (89, -170, 0.0)
    near = pose((0.1, 0.0, 0.9), pitch=88.8, yaw=40, roll=50.5)
    arm = vd.state_json({"right": near}, "right")["arm"]
    assert arm["roll"] == 0.0
    assert arm["yaw"] == round(math.radians(40 - 50.5), 2)


def jump_frames():
    """0.5 cm steps, one 8 cm recorded jump (frames 3 -> 4), then 0.5 cm steps."""

    xs = [0.10, 0.105, 0.11, 0.115, 0.195, 0.20, 0.205]
    return [frame(i, (x, 0.0, 0.9), phase="insert") for i, x in enumerate(xs)]


def test_oversize_transition_is_flagged_and_dropped_by_default():
    frames = jump_frames()
    steps = vd.coalesce(frames)
    assert [(s.start, s.end, s.oversize) for s in steps] == [
        (0, 3, False), (3, 4, True), (4, 6, False)
    ]
    assert steps[1].to_dict()["oversize"] is True
    runs = vd.split_runs(steps)
    assert [[(s.start, s.end) for s in run] for run in runs] == [[(0, 3)], [(4, 6)]]
    episode = vd.Episode(path=Path("ep"), task="t", success=True, frames=frames, calls=[],
                         result={})
    records = vd.episode_records(episode)
    assert [r["source"]["frames"] for r in records] == [[0, 3], [4, 6]]
    assert all(r["oversize_steps"] == 0 for r in records)
    assert [r["oversize_dropped"] for r in records] == [1, 0]
    assert all(abs(s["action"]["arm"]["x"]) <= 6.0 for r in records for s in r["steps"])
    # only the run that reaches the end of a successful episode gets the done turn
    assert "done" not in parse_reply(records[0]["messages"][-1]["content"])
    assert parse_reply(records[1]["messages"][-1]["content"])["done"] is True
    summary = vd.stats(records)
    assert summary["oversize_steps"] == 0 and summary["oversize_dropped"] == 1


def test_oversize_keep_mode_flags_only():
    steps = vd.coalesce(jump_frames(), config=vd.CoalesceConfig(oversize="keep"))
    assert [s.oversize for s in steps] == [False, True, False]
    with pytest.raises(ValueError):
        vd.coalesce(jump_frames(), config=vd.CoalesceConfig(oversize="bogus"))


def test_oversize_split_mode_makes_executable_parts(tmp_path):
    frames = with_images(jump_frames(), tmp_path)
    frames[4].gripper = {"right": 0.0}
    frames[4].stalled, frames[4].stop_reason = True, "contact_or_stall"
    for f in frames[:4]:
        f.gripper = {"right": 1.0}
    for f in frames[5:]:
        f.gripper = {"right": 0.0}
    steps = vd.coalesce(frames, config=vd.CoalesceConfig(oversize="split"))
    parts = [s for s in steps if s.part is not None]
    assert [s.part for s in parts] == [(0, 2), (1, 2)]
    assert all(not s.oversize and s.linear_m <= 0.06 for s in steps)
    assert [s.action["arm"]["x"] for s in parts] == pytest.approx([4.0, 4.0])
    assert [s.action["gripper"]["width"] for s in parts] == [0, -1]
    assert [s.status for s in parts] == ["reached", "stopped"]
    assert parts[1].state_poses["right"].tcp[0] == pytest.approx(0.155)
    messages, images = vd.build_conversation(frames, steps, "t")
    second = messages[1 + 2 * steps.index(parts[1])]["content"]
    state_text = second.split("```json", 1)[1].split("```", 1)[0]
    assert json.loads(state_text)["arm"]["x"] == 15.5
    assert sum(m["content"].count(vd.IMAGE_TOKEN) for m in messages) == len(images)


def test_oversize_wrist_spin_split_by_geodesic_angle():
    frames = [frame(0, (0.1, 0.0, 0.9), pitch=30, roll=0.0),
              frame(1, (0.1, 0.0, 0.9), pitch=30, roll=92.6)]
    steps = vd.coalesce(frames, arm="right", config=vd.CoalesceConfig(oversize="split"))
    assert len(steps) == 5
    assert all(s.angular_deg <= 20.0 + 1e-6 and not s.oversize for s in steps)
    total_roll = sum(s.action["arm"]["roll"] for s in steps)
    assert total_roll == pytest.approx(math.radians(92.6), abs=3e-3)


def test_cli_oversize_option(tmp_path, capsys):
    write_episode(tmp_path / "ep")
    out = tmp_path / "o.jsonl"
    assert vd.main(["extract", str(tmp_path / "ep"), "-o", str(out), "--oversize", "split",
                    "--max-linear-m", "0.01"]) == 0
    capsys.readouterr()
    record = json.loads(out.read_text().splitlines()[0])
    assert any("part" in s for s in record["steps"])
    assert all(s["linear_m"] <= 0.01 + 1e-6 for s in record["steps"])
