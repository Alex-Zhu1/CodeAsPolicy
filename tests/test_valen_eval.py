"""Offline comparison must preserve evidence and avoid invented accuracy labels."""

import json
from pathlib import Path

import pytest
from PIL import Image

from pyrualean.evaluation.valen import evaluate


@pytest.fixture
def audit(tmp_path):
    directory = tmp_path / "audit"
    directory.mkdir()
    events = []
    for index, probs in enumerate(((0.6, 0.3, 0.1), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9))):
        sample = f"sample-{index}"
        frame = directory / f"{sample}.png"
        Image.new("RGB", (40, 30), "blue").save(frame)
        packet = dict(state={"messages": [{"role": "user", "content": [
            {"type": "text", "text": f"Observation {index}: cam_head"},
            {"type": "image_url", "image_url": {"url": str(frame)}},
        ]}]}, questions={"held": {"type": "choice", "instructions": "Is the key held?"}})
        response = {"answers": {"held": {"type": "choice", "probabilities":
                    dict(zip(("yes", "no", "unknown"), probs, strict=True))}}}
        (directory / f"{sample}-request.json").write_text(json.dumps(packet))
        (directory / f"{sample}-response.json").write_text(json.dumps(response))
        events.append(dict(artifact_prefix=str(directory / sample), values={"held": "unknown"}))
    (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    return directory


def test_cached_cli_exports_template_without_model_or_robot(audit, tmp_path, monkeypatch):
    from pyrualean.cli import main
    from pyrualean.evaluation import valen

    def unexpected(*args, **kwargs):
        pytest.fail("Cached evaluation must not construct a live Valen client")

    monkeypatch.setattr(valen, "ValenClient", unexpected)
    original = {p.name: p.read_bytes() for p in audit.iterdir()}
    out = tmp_path / "comparison"
    assert main(["valen-eval", "--audit", str(audit), "--out", str(out)]) == 0
    report = json.loads((out / "summary.json").read_text())
    assert report["mode"] == "cached" and report["visual_checks"] == 3
    assert all(t["accuracy_on_decided"] is None and t["false_yes"] is None
               for t in report["thresholds"])
    template = json.loads((out / "labels.json").read_text())
    assert len(template) == 3 and all(row["label"] is None for row in template)
    assert {p.name: p.read_bytes() for p in audit.iterdir()} == original


def test_threshold_metrics_use_independent_labels_and_report_false_passes(audit, tmp_path):
    labels = [dict(sample_id=f"sample-{i}", fact="held", question="Is the key held?", label=label)
              for i, label in enumerate(("no", "yes", "unknown"))]
    report = evaluate(audit, tmp_path / "labelled", thresholds=[0.55, 0.8], labels=labels)
    low, high = report["thresholds"]
    assert low["unknown_rate"] == pytest.approx(1/3) and low["false_yes"] == 1
    assert high["unknown_rate"] == pytest.approx(2/3) and high["false_yes"] == 0
    assert low["false_no"] == high["false_no"] == 1
    assert low["accuracy_on_decided"] == high["accuracy_on_decided"] == 0


def test_replay_keeps_full_scene_and_adds_labelled_crop_and_question_variant(audit, tmp_path):
    class Client:
        def __init__(self):
            self.packets = []

        def predict(self, packet):
            self.packets.append(packet)
            return {"answers": {"held": {"type": "choice", "probabilities":
                    {"yes": 0.9, "no": 0.05, "unknown": 0.05}}}}

    client = Client()
    variant = {"questions": {"held": "Is the round key handle between the left fingers?"},
               "crops": {"sample-0.png": [2, 3, 22, 23]}}
    report = evaluate(audit, tmp_path / "replay", client=client, variant=variant)
    assert report["mode"] == "replay" and len(client.packets) == 3
    packet = client.packets[0]
    assert packet["questions"]["held"]["instructions"] == variant["questions"]["held"]
    content = packet["state"]["messages"][0]["content"]
    paths = [b["image_url"]["url"] for b in content if b.get("type") == "image_url"]
    assert len(paths) == 2 and Path(paths[0]) == audit / "sample-0.png"
    with Image.open(paths[1]) as crop:
        assert crop.size == (20, 20)
    assert "same observation" in content[-2]["text"]
    assert (tmp_path / "replay/sample-0-response.json").is_file()


def test_changed_question_cannot_reuse_old_label(audit, tmp_path):
    labels = [dict(sample_id="sample-0", fact="held", question="Is the key held?", label="yes")]
    with pytest.raises(ValueError, match="does not match"):
        evaluate(audit, tmp_path / "invalid", client=object(), labels=labels,
                 variant={"questions": {"held": "Is the key on the table?"}})
    assert not (tmp_path / "invalid").exists()


def test_variants_require_explicit_replay_and_outputs_cannot_overwrite(audit, tmp_path):
    with pytest.raises(ValueError, match="require --replay-url"):
        evaluate(audit, tmp_path / "a", variant={"questions": {"held": "Is the key visible?"}})
    with pytest.raises(FileExistsError):
        evaluate(audit, audit)


@pytest.mark.parametrize("threshold", [float("nan"), 0.5, 1.01])
def test_bad_threshold_cannot_produce_misleading_metrics(audit, tmp_path, threshold):
    with pytest.raises(ValueError, match="Thresholds"):
        evaluate(audit, tmp_path / "a", thresholds=[threshold])
