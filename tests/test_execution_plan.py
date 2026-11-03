"""Planner notes persist without becoming robot facts or execution permission."""

import hashlib
import json
import textwrap

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.interaction_trace import InteractionTrace


def execute(toolkit, code):
    if toolkit.arm == "cells":
        return toolkit.execute_tool("python", {"code": code})
    return toolkit.execute_tool("run_program", {
        "code": "def run(robo):\n" + textwrap.indent(code, "    "),
    })


@pytest.mark.parametrize("arm", ["cells", "program"])
def test_plan_persists_across_code_calls_with_revision_and_card_reference(
    robo, backend, tmp_path, arm,
):
    card = tmp_path / "card.json"
    original = '{"task_language": "move the block"}\n'
    card.write_text(original)
    trace = InteractionTrace(tmp_path / "events.jsonl")
    toolkit = ArmToolkit(robo, arm=arm, workspace=tmp_path / "ws", images="none",
                         task_card_path=card, trace=trace)
    assert toolkit._plan() is None and not (tmp_path / "execution_plan.json").exists()
    first = execute(toolkit, """\
assert plan() is None
note = plan(goal="Lift the block", next_action="inspect the grasp", done_when="block clears table")
note['goal'] = 'changed only in local copy'
assert plan()['goal'] == 'Lift the block'
""")
    assert not first.is_error and not first.result["moved"] and first.result["env_steps"] == 0
    assert first.result["primitive_calls"] == []
    assert first.result["execution_plan"]["revision"] == 1
    second = execute(toolkit, """\
assert plan()['revision'] == 1
plan(goal="Lift the block", next_action="reinspect", done_when="block clears table",
     evidence="not verified yet", status="blocked")
raise RuntimeError('after the note')
""")
    assert second.is_error and second.result["execution_plan"]["revision"] == 2
    latest = json.loads((tmp_path / "execution_plan.json").read_text())
    history_path = tmp_path / "execution_plan.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert [item["revision"] for item in history] == [1, 2]
    assert history[-1] == latest == toolkit._plan()
    assert latest["author"] == "astra" and latest["kind"] == "planner_note"
    assert latest["task_card"] == {
        "path": str(card), "sha256": hashlib.sha256(original.encode()).hexdigest(),
    }
    assert latest["context"]["turn"] == 2 and latest["session_id"] == trace.session_id
    assert card.read_text() == original and not backend.solved() and toolkit.finish is None
    assert toolkit.turns[-1]["execution_plan"] == second.result["execution_plan"]
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    updates = [event for event in events if event["event"] == "execution_plan_updated"]
    assert [event["plan_revision"] for event in updates] == [1, 2]
    assert all("goal" not in event and "evidence" not in event for event in updates)


def test_plan_uses_cached_observation_and_does_not_refresh_or_claim_task_completion(tmp_path):
    from tests.test_robodojo_trajectory import station

    env, robo, _ = station(tmp_path)
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    observation = robo.observe()
    snapshot = robo.snapshot()
    captures, actions, calls = env.captures, len(env.actions), len(robo._ledger)
    record = toolkit._plan(goal="Inspect", done_when="image read",
                           evidence="reviewed", status="done")
    context = record["context"]
    assert context["observation_id"] == snapshot["observation_id"]
    assert context["source_id"] == snapshot["source_id"]
    assert (env.captures, len(env.actions), len(robo._ledger)) == (captures, actions, calls)
    assert toolkit.finish is None
    robo.move_both(observation.left.shifted(dz=.01), observation.right.shifted(dz=.01))
    assert toolkit._plan() == record  # movement does not silently revalidate an old note
    updated = toolkit._plan(goal="Inspect again", done_when="new image reviewed")
    assert updated["context"]["action_revision"] > context["action_revision"]
    assert updated["context"]["last_action_id"] != context["last_action_id"]
    assert updated["context"]["observation_id"] != context["observation_id"]


@pytest.mark.parametrize("arguments", [
    {"goal": ""}, {"goal": 3}, {"goal": "x" * 1001}, {"done_when": ""},
    {"evidence": None}, {"status": "success"},
])
def test_invalid_plan_does_not_modify_saved_note(robo, tmp_path, arguments):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    original = toolkit._plan(goal="Check", done_when="image inspected")
    with pytest.raises(ValueError):
        toolkit._plan(**(dict(goal="Check", done_when="image inspected") | arguments))
    assert toolkit._plan() == original
    assert len((tmp_path / "execution_plan.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="supply goal"):
        toolkit._plan(status="done")


def test_plan_helper_does_not_bypass_code_turn_limit(robo, tmp_path):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", max_turns=1)
    first = execute(toolkit, 'plan(goal="Inspect", done_when="read image", status="done")')
    assert not first.is_error and toolkit.finish is None
    second = execute(toolkit, 'plan(goal="Move", done_when="at target")')
    assert second.is_error and "budget" in second.result["error"]
    assert toolkit._plan()["goal"] == "Inspect"
