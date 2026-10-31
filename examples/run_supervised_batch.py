"""Run a frozen RoboDojo task manifest with bounded GPU concurrency.

Usage: python run_supervised_batch.py /absolute/batch/directory
Requires config.json, frozen code/benchmark/skills, and run_episode.py in that
folder and psutil in the Python environment. Re-running resumes pending jobs,
never repeats finalized task failures.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import io
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def last_event(path):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    except FileNotFoundError:
        pass
    return {}


def collect_result(directory, returncode, error=None):
    result = read_json(directory / "episode/result.json") or {}
    native_paths = list((directory / "episode/bench/eval_result").rglob("_result.json"))
    native = read_json(native_paths[0]) if len(native_paths) == 1 else None
    verdict = result.get("environment_success") if result else None
    if native is not None:
        verdict = native.get("success_rate") == 1.0
    normal = returncode == 0 and result.get("status") == "completed" and error is None
    outcome = ("success" if verdict else "failure") if normal and verdict is not None else "error"
    monitor = directory / "episode/supervision/events.jsonl"
    reviews = errors = 0
    if monitor.exists():
        for line in monitor.read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                event = json.loads(line)
                reviews += event.get("event") == "review"
                errors += event.get("event") == "monitor_error"
    recovery = result.get("stuck_recovery") or {}
    return {
        "outcome": outcome,
        "returncode": returncode,
        "episode_status": result.get("status"),
        "native_success": verdict,
        "native_score": native.get("score") if native else None,
        "env_steps": result.get("env_steps"),
        "policy_wall_s": result.get("policy_wall_s"),
        "planner_finish": result.get("finish"),
        "valen_reviews": reviews,
        "monitor_errors": errors,
        "stuck_recovery_rounds": sum(
            e.get("event") == "recovery_dispatched" for e in recovery.get("events", [])
        ),
        "error": error or result.get("error") or result.get("reason"),
        "result_path": str(directory / "episode/result.json"),
    }


def summary(jobs):
    finished = [j for j in jobs if j["state"] == "finished"]
    evaluated = [j for j in finished if j.get("outcome") in ("success", "failure")]
    successes = sum(j.get("outcome") == "success" for j in evaluated)
    scores = [j["native_score"] for j in evaluated if j.get("native_score") is not None]
    return {
        "total": len(jobs),
        "finished": len(finished),
        "running": sum(j["state"] == "running" for j in jobs),
        "pending": sum(j["state"] == "pending" for j in jobs),
        "evaluated": len(evaluated),
        "successes": successes,
        "task_failures": len(evaluated) - successes,
        "errors": sum(j.get("outcome") == "error" for j in finished),
        "success_rate_evaluated": successes / len(evaluated) if evaluated else None,
        "mean_native_score_evaluated": sum(scores) / len(scores) if scores else None,
    }


class OwnedProcess:
    """Track descendants across their separate sessions; never kill unrelated jobs."""

    def __init__(self, command, **kwargs):
        self.proc = subprocess.Popen(command, start_new_session=True, **kwargs)
        self.identities = {}
        self.started = time.monotonic()
        self.root_created = None
        with contextlib.suppress(psutil.Error):
            self.root_created = psutil.Process(self.proc.pid).create_time()
        self.capture()

    def capture(self):
        with contextlib.suppress(psutil.Error):
            parent = psutil.Process(self.proc.pid)
            if self.root_created is None or parent.create_time() != self.root_created:
                return
            for process in [parent, *parent.children(recursive=True)]:
                with contextlib.suppress(psutil.Error):
                    self.identities[process.pid] = process.create_time()

    def live_owned(self):
        for pid, created in list(self.identities.items()):
            try:
                process = psutil.Process(pid)
                if process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE:
                    yield process
            except psutil.Error:
                pass

    def stop(self):
        self.capture()
        owned = list(self.live_owned())
        for process in reversed(owned):
            with contextlib.suppress(psutil.Error):
                process.terminate()
        _, alive = psutil.wait_procs(owned, timeout=10)
        for process in alive:
            with contextlib.suppress(psutil.Error):
                if process.create_time() == self.identities.get(process.pid):
                    process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(timeout=5)


def still_running(job):
    try:
        process = psutil.Process(job["pid"])
        return (
            process.create_time() == job["process_created"]
            and process.status() != psutil.STATUS_ZOMBIE
        )
    except (psutil.Error, KeyError):
        return False


class Batch:
    def __init__(self, root):
        self.root = root.resolve()
        self.config = json.loads((root / "config.json").read_text())
        old = read_json(root / "state.json")
        self.jobs = (
            old["jobs"]
            if old
            else [
                {"task": task, "state": "pending", "directory": str(self.root / "jobs" / task)}
                for task in self.config["tasks"]
            ]
        )
        if [j["task"] for j in self.jobs] != self.config["tasks"]:
            raise ValueError("Existing state does not match the frozen task manifest")
        for job in self.jobs:
            if job["state"] == "running":
                if still_running(job):
                    raise RuntimeError(
                        f"Existing job still running: {job['task']} pid={job['pid']}"
                    )
                data = collect_result(Path(job["directory"]), 0)
                if data["outcome"] == "error":
                    data["error"] = (
                        "Previous batch was interrupted; inspect before an explicit retry"
                    )
                job.update(data, state="finished", ended_at=now())
        self.active = {}
        self.server = None
        self.server_log = None
        self.stopping = False
        self.state = "starting"
        self.failure = None

    def publish(self):
        for job, _owned, _ in self.active.values():
            event = last_event(Path(job["directory"]) / "episode/supervision/events.jsonl")
            job["latest_step"] = event.get("step")
            job["latest_phase"] = event.get("contract", {}).get("phase")
        metrics = summary(self.jobs)
        write_json(
            self.root / "state.json",
            {
                "state": self.state,
                "updated_at": now(),
                "pid": os.getpid(),
                "failure": self.failure,
                "summary": metrics,
                "jobs": self.jobs,
            },
        )
        write_json(self.root / "summary.json", metrics)
        output = io.StringIO()
        fields = [
            "task",
            "state",
            "outcome",
            "native_success",
            "native_score",
            "env_steps",
            "valen_reviews",
            "stuck_recovery_rounds",
            "episode_status",
            "directory",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.jobs)
        path = self.root / "summary.csv"
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_text(output.getvalue())
        temporary.replace(path)

    def start_valen(self):
        from pyrualean.supervision import CRITERIA, MonitorConfig, ValenClient, decode_answers

        c = self.config
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/predict"
        log_path = self.root / f"valen-{int(time.time())}.log"
        self.server_log = log_path.open("x")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(c["valen_gpu"]),
            HF_HUB_OFFLINE="1",
            HF_MODULES_CACHE=str(self.root / "model-module-cache"),
        )
        self.server = OwnedProcess(
            [
                c["valen_python"],
                c["valen_server"],
                "--model",
                c["valen_model"],
                "--port",
                str(port),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 240
        while "VALEN_MODEL_AND_HTTP_READY" not in log_path.read_text():
            if self.stopping or self.server.proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Valen startup failed or interrupted; inspect its log")
            time.sleep(1)
        questions = {
            "arms": {
                "type": "choice",
                "instructions": "Are robot arms visible?",
                "criteria": CRITERIA,
            }
        }
        packet = {
            "state": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": c["probe_image"]}}],
                    }
                ]
            },
            "questions": questions,
        }
        response = ValenClient(MonitorConfig(mode="pause", url=self.url, timeout_s=60)).predict(
            packet
        )
        values, _ = decode_answers(response, questions, c["monitor_threshold"])
        write_json(
            self.root / "valen_service.json",
            {
                "url": self.url,
                "pid": self.server.proc.pid,
                "model": c["valen_model"],
                "gpu": c["valen_gpu"],
                "probe": values,
                "started_at": now(),
            },
        )

    def launch(self, job, gpu):
        import shutil

        c, directory = self.config, Path(job["directory"])
        if (directory / "episode").exists():
            raise RuntimeError(f"Refusing to overwrite existing episode: {directory}")
        home = directory / "codex-home"
        home.mkdir(parents=True, exist_ok=True)
        for name in ("auth.json", "config.toml"):
            shutil.copy2(Path(c["auth_source"]) / name, home / name)
        (home / "auth.json").chmod(0o600)
        command = [
            c["python"],
            str(self.root / "run_episode.py"),
            "--robot",
            "robodojo_xpl_rgb",
            "--arm",
            "cells",
            "--suite",
            "arx_x5",
            "--task",
            job["task"],
            "--eval-seed",
            str(c["eval_seed"]),
            "--seed",
            str(c["layout"]),
            "--cuda-device",
            str(gpu),
            "--source-root",
            str(self.root / "benchmark"),
            "--runtime",
            "codex",
            "--model",
            c["model"],
            "--reasoning",
            c["reasoning"],
            "--codex-bin",
            c["codex_bin"],
            "--codex-home",
            str(home),
            "--primitives",
            "no-vla",
            "--images",
            "on-demand",
            "--feedback",
            "pure",
            "--monitor-mode",
            "pause",
            "--valen-url",
            self.url,
            "--monitor-timeout-s",
            "30",
            "--monitor-threshold",
            str(c["monitor_threshold"]),
            "--monitor-unknown-limit",
            str(c["monitor_unknown_limit"]),
            "--monitor-max-actions",
            str(c["monitor_max_actions"]),
            "--stuck-recovery-skill",
            str(self.root / "stuck_recovery_skill.md"),
            "--stuck-recovery-rounds",
            str(c["recovery_rounds"]),
            "--max-turns",
            str(c["max_turns"]),
            "--budget-s",
            str(c["budget_s"]),
            "--out",
            str(directory / "episode"),
        ]
        write_json(directory / "command.json", command)
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        log = (directory / "run.log").open("a")
        process = OwnedProcess(
            command,
            cwd=self.root / "code",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        job.update(
            state="running",
            gpu=gpu,
            pid=process.proc.pid,
            process_created=psutil.Process(process.proc.pid).create_time(),
            started_at=now(),
        )
        self.active[gpu] = (job, process, log)
        self.publish()
        print(json.dumps({"event": "started", "task": job["task"], "gpu": gpu}), flush=True)

    def finalize(self, gpu, error=None):
        job, process, log = self.active.pop(gpu)
        returncode = process.proc.poll()
        process.stop()
        log.close()
        job.update(
            collect_result(Path(job["directory"]), returncode, error),
            state="finished",
            ended_at=now(),
        )
        self.publish()
        print(
            json.dumps(
                {
                    "event": "finished",
                    "task": job["task"],
                    "outcome": job["outcome"],
                    "native_success": job["native_success"],
                }
            ),
            flush=True,
        )

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopping", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stopping", True))
        try:
            self.publish()
            if any(j["state"] == "pending" for j in self.jobs):
                self.start_valen()
            self.state = "running"
            while not self.stopping:
                if self.server and self.server.proc.poll() is not None:
                    raise RuntimeError("Shared Valen service exited; pending jobs remain queued")
                for gpu in self.config["simulator_gpus"]:
                    if gpu not in self.active:
                        job = next((j for j in self.jobs if j["state"] == "pending"), None)
                        if job:
                            self.launch(job, gpu)
                if not self.active:
                    break
                for gpu, (_, process, _) in list(self.active.items()):
                    process.capture()
                    if process.proc.poll() is not None:
                        self.finalize(gpu)
                    elif time.monotonic() - process.started > self.config["job_timeout_s"]:
                        self.finalize(gpu, "Batch watchdog timeout")
                self.publish()
                time.sleep(3)
            self.state = "interrupted" if self.stopping else "completed"
        except Exception as exc:
            self.state, self.failure = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for gpu in list(self.active):
                self.finalize(gpu, self.failure or "Batch interrupted")
            if self.server:
                self.server.stop()
            if self.server_log:
                self.server_log.close()
            self.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    with (args.root / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Batch(args.root).run()


if __name__ == "__main__":
    main()
