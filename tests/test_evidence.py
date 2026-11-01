"""Persistent declarations must fail atomically and remain bounded."""

import pytest

from pyrualean.evidence import EvidenceLedger


@pytest.mark.parametrize("requires,invalidates", [
    ({"missing": "yes"}, None), ({"held": "unknown"}, None),
    ([], None), ({}, ["missing"]), ({}, ["held", "held"]), ({}, "held"),
])
def test_bad_watch_cannot_erase_prior_gate_or_change_fact(requires, invalidates):
    ledger = EvidenceLedger()
    ledger.configure("carry", {"held": "Is the cup held?"}, {"held": "yes"}, None)
    before = ledger.snapshot()
    with pytest.raises(ValueError):
        ledger.configure("carry", {"new": "Is the tray visible?"}, requires, invalidates)
    assert ledger.snapshot() == before


@pytest.mark.parametrize("options", [
    dict(until="", reason="later"), dict(until="insert", reason=" "),
    dict(until="insert", reason="x" + " " * 500),
    dict(until="insert", reason="later", expected="unknown"),
])
def test_invalid_deferral_cannot_change_requirements(options):
    ledger = EvidenceLedger()
    ledger.configure("align", {"aligned": "Is the key aligned?"}, {}, None)
    before = ledger.snapshot()
    with pytest.raises(ValueError):
        ledger.defer("aligned", **options)
    assert ledger.snapshot() == before


def test_conflicting_gate_cannot_overwrite_existing_dependency():
    ledger = EvidenceLedger()
    ledger.configure("carry", {"held": "Is the cup held?"}, {"held": "yes"}, None)
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="Conflicting"):
        ledger.configure("carry", {"held": "Is the cup held?"}, {"held": "no"}, None)
    with pytest.raises(ValueError, match="Conflicting"):
        ledger.defer("held", until="carry", expected="no", reason="try again")
    assert ledger.snapshot() == before


def test_episode_fact_capacity_fails_without_partial_registration():
    ledger = EvidenceLedger()
    for index in range(ledger.MAX_FACTS):
        ledger.configure("observe", {str(index): f"Is object {index} visible?"}, {}, None)
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="visual facts"):
        ledger.configure("observe", {"overflow": "Is another object visible?"}, {}, None)
    assert ledger.snapshot() == before


def test_source_identity_prevents_repeat_votes_and_unknown_reset():
    ledger = EvidenceLedger()
    ledger.configure("carry", {"held": "Is the cup held?"}, {"held": "yes"}, None)
    unknown = {"held": {"yes": 0, "no": 0, "unknown": 1}}
    yes = {"held": {"yes": 1, "no": 0, "unknown": 0}}
    first = dict(observation_id="request-1", source_id="image-1")
    assert ledger.observe({"held": "unknown"}, unknown, first) == {"held": True}
    ledger.invalidate(None, "action_uncertain", action_id="action-1")
    assert ledger.blockers("carry")["held"]["invalidated_by_action_id"] == "action-1"
    # A request ID is not a new capture, nor permission to flip a prior vote.
    assert ledger.observe({"held": "yes"}, yes, dict(first, observation_id="request-2")) == {
        "held": False,
    }
    assert ledger.facts["held"]["value"] == "unknown"
    assert ledger.facts["held"]["unknown_streak"] == 1
    assert ledger.facts["held"]["invalidated_by_action_id"] is None
    assert ledger.blockers("carry")
    ledger.observe({"held": "unknown"}, unknown, dict(first, source_id="image-2"))
    assert ledger.facts["held"]["unknown_streak"] == 2
    ledger.observe({"held": "yes"}, yes, dict(first, source_id="image-3"))
    assert ledger.facts["held"]["unknown_streak"] == 0
    assert not ledger.blockers("carry")


def test_question_revision_starts_a_new_uncertainty_record():
    ledger = EvidenceLedger()
    ledger.configure("look", {"held": "Is the cup held?"}, {}, None)
    ledger.observe({"held": "unknown"}, {"held": {"yes": 0, "no": 0, "unknown": 1}},
                   dict(observation_id="frame-1"))
    ledger.configure("look", {"held": "Is the red cup held by the left arm?"}, {}, None)
    assert ledger.facts["held"]["version"] == 2
    assert ledger.facts["held"]["unknown_streak"] == 0
    assert ledger.facts["held"]["source_id"] is None
