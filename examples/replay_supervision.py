"""Replay a phase's recorded RGB endpoints through Valen, without a simulator.

The manifest is a JSON list of {step: <native control count>, images: {camera: path}}.
It must contain consecutive primitive endpoints from one episode and one phase.
The contract is JSON with the keyword arguments accepted by watch(). All results
are shadow observations: this script never moves a robot or measures task success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyrualean.supervision import MonitorConfig, VisualSupervisor


class RecordedRobot:
    def __init__(self, frames):
        if not isinstance(frames, list) or not frames:
            raise ValueError("Manifest must contain at least one recorded endpoint")
        cameras = tuple(frames[0]["images"])
        previous = -1
        for frame in frames:
            step = frame["step"]
            if type(step) is not int or step <= previous:
                raise ValueError("Manifest control steps must be nonnegative and increasing")
            if not cameras or set(frame["images"]) != set(cameras):
                raise ValueError("Every endpoint must have the same nonempty camera set")
            for path in frame["images"].values():
                if not Path(path).is_absolute() or not Path(path).is_file():
                    raise ValueError("Image paths must be absolute existing files")
            previous = step
        self.frames = frames
        self.frame = frames[0]
        self.CAMERAS = cameras
        self.SHOW_RESOLUTION = "high"
        self._backend = self

    def env_steps(self):
        return self.frame["step"]

    def artifact(self, camera):
        return Path(self.frame["images"][camera]).read_bytes()

    def _image_name(self, camera, resolution):
        return camera


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--valen-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=30)
    args = parser.parse_args(argv)
    robo = RecordedRobot(json.loads(args.manifest.read_text()))
    if not 0 <= args.start_step < robo.frames[0]["step"]:
        raise ValueError("start-step must precede the first executed endpoint")
    config = MonitorConfig(mode="shadow", url=args.valen_url, timeout_s=args.timeout_s)
    monitor = VisualSupervisor(config, args.out)
    monitor.begin_cell(1)
    monitor.watch(**json.loads(args.contract.read_text()))
    previous = args.start_step
    for frame in robo.frames:
        robo.frame = frame
        monitor.after_action(robo, "recorded_primitive", previous)
        previous = frame["step"]
    summary = dict(replay=True, robot_actions_executed=0, reports=monitor.reports)
    (args.out / "replay.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "endpoints": len(monitor.reports),
                "output": str(args.out),
                "events": [r["event"] for r in monitor.reports],
            }
        )
    )
    return int(any(r["event"] == "monitor_error" for r in monitor.reports))


if __name__ == "__main__":
    raise SystemExit(main())
