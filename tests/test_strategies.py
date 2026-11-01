"""Local loops must verify results and yield when their evidence stops improving."""

from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.strategies import GraspObservation, correct, relocalize, verify_grasp


def grasp_obs(version, *, held=True, holding=True, fresh=True):
    return GraspObservation(str(version), version, holding, held, fresh=fresh)


def test_grasp_requires_fresh_object_support_after_probe():
    events = []
    observations = iter([grasp_obs(1), grasp_obs(2)])
    result = verify_grasp(
        lambda: next(observations), lambda i: events.append(("close", i)),
        lambda obs, i: events.append(("probe", obs.state_version)),
        lambda before, after: before.state_version == 1 and after.state_version == 2,
    )
    assert result.status == "verified" and result.attempts == 1
    assert events == [("close", 1), ("probe", 1)]
    assert result.before_probe.state_version == 1 and result.observation.state_version == 2


@pytest.mark.parametrize("after,comparison,expected", [
    (grasp_obs(2, held=None), True, "uncertain"),
    (grasp_obs(2), None, "uncertain"),
    (grasp_obs(2), False, "failed"),
    (grasp_obs(2, holding=False), True, "failed"),
    (grasp_obs(1), True, "uncertain"),
    (grasp_obs(2, fresh=False), True, "uncertain"),
])
def test_holding_flag_alone_and_old_frames_cannot_verify_grasp(after, comparison, expected):
    observations = iter([grasp_obs(1), after])
    repairs = []
    result = verify_grasp(
        lambda: next(observations), lambda i: SimpleNamespace(holding=True),
        lambda o, i: SimpleNamespace(reached=True), lambda a, b: comparison,
        max_attempts=1, repair=lambda o, i: repairs.append(i),
    )
    assert result.status == expected and not repairs


def test_unknown_hands_back_without_repeating_probe_or_repair():
    observations = iter([grasp_obs(1), grasp_obs(2, held=None)])
    actions = []
    result = verify_grasp(
        lambda: next(observations), lambda i: actions.append("close"),
        lambda o, i: actions.append("probe"), lambda a, b: None,
        repair=lambda o, i: actions.append("repair"), max_attempts=3,
    )
    assert result.status == "uncertain" and actions == ["close", "probe"]


def test_failed_close_can_change_grasp_but_retry_is_bounded():
    calls = []
    result = verify_grasp(
        lambda: grasp_obs(1, holding=False, held=False),
        lambda i: calls.append(("close", i)), lambda o, i: calls.append(("probe", i)),
        lambda a, b: True, repair=lambda o, i: calls.append(("repair", i)), max_attempts=2,
    )
    assert result.status == "failed" and result.attempts == 2
    assert calls == [("close", 1), ("repair", 2), ("close", 2)]


def test_grasp_budget_preserves_closed_state_and_does_not_start_probe():
    calls = []
    result = verify_grasp(
        lambda: grasp_obs(1), lambda i: calls.append("close"),
        lambda o, i: calls.append("probe"), lambda a, b: True,
        allow_action=lambda kind, i: kind == "close",
    )
    assert result.status == "uncertain" and result.reason == "phase_budget"
    assert calls == ["close"] and result.observation.holding is True


def test_grasp_pause_never_retries_or_calls_next_callback():
    from pyrualean.supervision import SupervisionPause

    calls = []

    def close(attempt):
        calls.append(attempt)
        raise SupervisionPause("unknown after closure")

    with pytest.raises(SupervisionPause):
        verify_grasp(lambda: None, close, lambda o, i: None, lambda a, b: True)
    assert calls == [1]


def test_grasp_observation_does_not_promote_stale_ledger_fact():
    ledger = {"facts": {"held": {"value": "yes", "fresh": False,
              "observation": {"observation_id": "old", "state_version": 1}}}}
    obs = GraspObservation.from_evidence(ledger, "held", holding=True)
    assert not obs.fresh and obs.object_held is None


def test_grasp_cannot_verify_a_different_predicate_after_probe():
    before = GraspObservation("a", 1, True, True, fresh=True, fact="held", question_version=1)
    after = GraspObservation("b", 2, True, True, fresh=True, fact="held", question_version=2)
    observations = iter([before, after])
    result = verify_grasp(lambda: next(observations), lambda i: None,
                          lambda o, i: None, lambda a, b: True)
    assert result.status == "uncertain" and result.reason == "predicate_changed"


def test_relocalize_observes_after_changing_view():
    events = []
    values = iter([None, (0.1, 0.2, 0.3)])

    def observe():
        events.append("observe")
        return next(values)

    result = relocalize(observe, lambda p: p is not None,
                        prepare_view=lambda p, i: events.append("clear_occlusion"))
    assert result.status == "verified" and result.attempts == 2
    assert result.observation == (0.1, 0.2, 0.3)
    assert events == ["observe", "clear_occlusion", "observe"]


def test_correction_checks_the_last_allowed_action_and_uses_new_evidence():
    observations, actions = iter([3, 2, 1, 0]), []
    result = correct(lambda: next(observations), lambda n: n == 0,
                     lambda n, i: actions.append((n, i)), max_attempts=3)
    assert result.status == "verified" and result.attempts == 3
    assert result.observation == 0 and actions == [(3, 1), (2, 2), (1, 3)]


@pytest.mark.parametrize("verdict,reason,attempts", [
    (None, "observation_uncertain", 0), (False, "no_progress", 2),
])
def test_uncertainty_or_no_progress_hands_back_before_more_actions(verdict, reason, attempts):
    actions = []
    result = correct(lambda: 0.2, lambda _: verdict,
                     lambda p, i: actions.append(i), error=lambda p: p, max_attempts=5)
    assert result.status == "replan" and result.reason == reason
    assert len(actions) == result.attempts == attempts


def test_grasp_uses_observed_holding_not_the_action_return():
    result = correct(lambda: {"holding": False}, lambda s: s["holding"],
                     lambda s, i: {"command_accepted": True}, max_attempts=2)
    assert result.status == "replan" and result.reason == "attempt_limit"
    assert result.last_result["command_accepted"] and result.attempts == 2


def test_primitive_errors_propagate_without_replaying():
    calls = []

    def act(obs, attempt):
        calls.append(attempt)
        raise RuntimeError("controller failed after starting motion")

    with pytest.raises(RuntimeError, match="after starting motion"):
        correct(lambda: False, lambda x: x, act)
    assert calls == [1]


@pytest.mark.parametrize("arm", ["cells", "program"])
def test_helpers_are_available_to_generated_code(robo, tmp_path, arm):
    kit = ArmToolkit(robo, arm=arm, workspace=tmp_path)
    code = "r = strategy.relocalize(lambda: 1, lambda x: x == 1)\nprint(r.status)"
    if arm == "program":
        code = "def run(robo):\n    " + code.replace("\n", "\n    ")
    result = kit.execute_tool("python" if arm == "cells" else "run_program", {"code": code})
    assert not result.is_error and result.result["stdout"] == "verified\n"
