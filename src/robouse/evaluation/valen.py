"""Replay saved visual checks and evaluate abstention without running a robot."""

from __future__ import annotations

import copy
import json
import math
import time
from collections import Counter
from pathlib import Path

from ..supervision import MonitorConfig, ValenClient, answer_diagnostics, decode_answers


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_samples(audit):
    """Use saved successful checks; error events are not additional observations."""
    audit = Path(audit).resolve()
    samples, seen = [], set()
    for line in (audit / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if not event.get("values"):
            continue
        sample_id = Path(event["artifact_prefix"]).name
        if sample_id in seen:
            raise ValueError(f"Duplicate observation: {sample_id}")
        seen.add(sample_id)
        packet = _read(audit / f"{sample_id}-request.json")
        response = _read(audit / f"{sample_id}-response.json")
        decode_answers(response, packet["questions"], 0.8)
        samples.append(dict(sample_id=sample_id, packet=packet, response=response))
    if not samples:
        raise ValueError("No successful saved visual checks in this audit")
    return samples


def _variant_packet(packet, audit, output, variant):
    """Keep full scenes, appending explicitly selected, labelled local crops."""
    packet = copy.deepcopy(packet)
    for fact, question in variant.get("questions", {}).items():
        if fact in packet["questions"]:
            packet["questions"][fact]["instructions"] = question
    content = packet["state"]["messages"][0]["content"]
    expanded, caption = [], ""
    for block in content:
        if block.get("type") == "text":
            caption = block["text"]
        if block.get("type") != "image_url":
            expanded.append(block)
            continue
        original = Path(block["image_url"]["url"])
        source = audit / original.name
        if not source.is_file():
            raise ValueError(f"Missing audit image: {source}")
        block["image_url"]["url"] = str(source)
        expanded.append(block)
        box = variant.get("crops", {}).get(source.name)
        if box is None:
            continue
        from PIL import Image

        with Image.open(source) as frame:
            if not (0 <= box[0] < box[2] <= frame.width
                    and 0 <= box[1] < box[3] <= frame.height):
                raise ValueError(f"Crop outside image bounds: {source.name}")
            target = output / "crops" / source.name
            target.parent.mkdir(exist_ok=True)
            frame.crop(box).save(target)
        expanded.extend([
            dict(type="text", text=f"Local crop of {caption}; pixel box {box}. "
                 "The full scene above remains the context. This is the same observation."),
            dict(type="image_url", image_url={"url": str(target)}),
        ])
    packet["state"]["messages"][0]["content"] = expanded
    return packet


def _validate_variant(variant, samples):
    if not isinstance(variant, dict) or set(variant) - {"questions", "crops"}:
        raise ValueError("Variant accepts only questions and crops mappings")
    questions, crops = variant.get("questions", {}), variant.get("crops", {})
    if not isinstance(questions, dict) or not isinstance(crops, dict):
        raise ValueError("questions and crops must be mappings")
    known_facts = {fact for s in samples for fact in s["packet"]["questions"]}
    if set(questions) - known_facts:
        raise ValueError("Variant refers to an unknown fact ID")
    if any(not isinstance(q, str) or not q.strip() or len(q) > 1000
           for q in questions.values()):
        raise ValueError("Variant questions must contain 1..1000 characters")
    images = {
        Path(b["image_url"]["url"]).name for s in samples
        for b in s["packet"]["state"]["messages"][0]["content"]
        if b.get("type") == "image_url"
    }
    if set(crops) - images:
        raise ValueError("Variant refers to an unknown image basename")
    for box in crops.values():
        if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError("Crop boxes must be [left, top, right, bottom] integer pixels")


def threshold_metrics(rows, thresholds):
    """No accuracy estimate without independent, question-matched human labels."""
    reports = []
    for threshold in thresholds:
        counts, confusion = Counter(), Counter()
        for row in rows:
            probs = row["probabilities"]
            winner = max(probs, key=probs.get)
            value = winner if probs[winner] >= threshold else "unknown"
            counts[value] += 1
            if row["label"] is not None:
                confusion[(row["label"], value)] += 1
        known_decisions = sum(confusion[(a, b)] for a in ("yes", "no") for b in ("yes", "no"))
        correct = confusion[("yes", "yes")] + confusion[("no", "no")]
        labelled = sum(confusion.values())
        reports.append(dict(
            threshold=threshold, observations=len(rows), decoded=dict(counts),
            unknown_rate=counts["unknown"] / len(rows), labelled=labelled,
            accuracy_on_decided=(correct / known_decisions if known_decisions else None),
            false_yes=confusion[("no", "yes")] if labelled else None,
            false_no=confusion[("yes", "no")] if labelled else None,
            decided_on_unobservable=(confusion[("unknown", "yes")]
                                     + confusion[("unknown", "no")]) if labelled else None,
            confusion={label: {p: confusion[(label, p)] for p in ("yes", "no", "unknown")}
                       for label in ("yes", "no", "unknown")} if labelled else None,
        ))
    return reports


def evaluate(audit, output, *, thresholds=(0.6, 0.7, 0.8, 0.9), labels=None,
             replay_url=None, variant=None, timeout_s=30, client=None):
    """Cached evaluation by default; explicit replay calls only Valen, never a robot.

    Labels are independent human annotations for a specific sample/fact/question.
    Changed questions need matching annotations; labels are never copied from the
    old model's predictions or from the benchmark evaluator.
    """
    if not thresholds or any(isinstance(t, bool) or not math.isfinite(t) or not 0.5 < t <= 1
                             for t in thresholds):
        raise ValueError("Thresholds must be finite numbers in (0.5, 1]")
    audit, output = Path(audit).resolve(), Path(output).resolve()
    samples = load_samples(audit)
    variant = variant or {}
    _validate_variant(variant, samples)
    replay = bool(replay_url or client is not None)
    if variant and not replay:
        raise ValueError("Question/crop variants require --replay-url")
    if labels is not None and not isinstance(labels, list):
        raise ValueError("Labels must be a list of sample/fact/question/label records")
    annotations = {}
    for item in labels or []:
        if (not isinstance(item, dict)
                or not {"sample_id", "fact", "question", "label"} <= item.keys()
                or any(not isinstance(item[k], str) for k in ("sample_id", "fact", "question"))):
            raise ValueError("Each label needs string sample_id, fact, question and a label")
        key = (item["sample_id"], item["fact"])
        if key in annotations:
            raise ValueError(f"Duplicate label: {key}")
        if item["label"] not in (None, "yes", "no", "unknown"):
            raise ValueError("Labels must be yes, no, unknown or null")
        annotations[key] = item
    available = {(s["sample_id"], fact): variant.get("questions", {}).get(
        fact, question["instructions"])
        for s in samples for fact, question in s["packet"]["questions"].items()}
    for key, item in annotations.items():
        if key not in available or item.get("question") != available[key]:
            raise ValueError(f"Label does not match this sample and question: {key}")
    if replay and client is None:
        client = ValenClient(MonitorConfig(mode="shadow", url=replay_url, timeout_s=timeout_s))
    # A new output directory keeps recorded evidence and earlier comparisons immutable.
    output.mkdir(parents=True, exist_ok=False)
    rows, templates, latencies = [], [], []
    for sample in samples:
        sample_id = sample["sample_id"]
        packet = sample["packet"]
        response = sample["response"]
        if replay:
            packet = _variant_packet(packet, audit, output, variant)
            (output / f"{sample_id}-request.json").write_text(json.dumps(packet, indent=2))
            started = time.perf_counter()
            response = client.predict(packet)
            latencies.append(time.perf_counter() - started)
            (output / f"{sample_id}-response.json").write_text(json.dumps(response, indent=2))
        _, probabilities = decode_answers(response, packet["questions"], 0.8)
        diagnostics = answer_diagnostics(probabilities, 0.8)
        images = [b["image_url"]["url"] for b in packet["state"]["messages"][0]["content"]
                  if b.get("type") == "image_url"]
        if not replay:
            images = [str(audit / Path(p).name) for p in images]
        for fact, question in packet["questions"].items():
            label = annotations.get((sample_id, fact), {}).get("label")
            row = dict(sample_id=sample_id, fact=fact, question=question["instructions"],
                       label=label, probabilities=probabilities[fact],
                       decision_at_08=diagnostics[fact])
            rows.append(row)
            templates.append({k: row[k] for k in ("sample_id", "fact", "question", "label")})
            templates[-1]["images"] = images
    summary = dict(
        schema="pyrualean-valen-eval-v1", audit=str(audit), mode="replay" if replay else "cached",
        visual_checks=len(samples), fact_judgments=len(rows), variant=variant,
        replay_url=replay_url, replay_latency_s=latencies,
        thresholds=threshold_metrics(rows, thresholds),
        notes=["Unknown rate alone is not accuracy; threshold selection needs human labels.",
               "Visual fact evaluation does not measure robot task success.",
               "Changed question meanings require new annotations; no threshold is applied live."],
    )
    for name, value in (("summary.json", summary), ("predictions.json", rows),
                        ("labels.json", templates)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return summary


def add_arguments(parser):
    parser.add_argument("--audit", type=Path, required=True, help="episode/supervision directory")
    parser.add_argument("--out", type=Path, required=True, help="new evaluation directory")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--labels", type=Path, help="human-edited labels.json from a prior export")
    parser.add_argument("--replay-url", help="Valen /predict URL; otherwise use saved scores")
    parser.add_argument("--variant", type=Path, help="JSON questions/crops overrides for replay")
    parser.add_argument("--timeout-s", type=float, default=30)


def run(args):
    summary = evaluate(
        args.audit, args.out, thresholds=args.thresholds,
        labels=_read(args.labels) if args.labels else None,
        replay_url=args.replay_url, variant=_read(args.variant) if args.variant else None,
        timeout_s=args.timeout_s,
    )
    print(json.dumps({k: summary[k] for k in ("mode", "visual_checks", "thresholds")}, indent=2))
    return 0
