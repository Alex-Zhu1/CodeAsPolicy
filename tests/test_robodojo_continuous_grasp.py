"""Native-frame identity continuity must not become a preclosure hold claim."""
import copy
from types import SimpleNamespace

import pytest

from pyrualean.hosts import robodojo_features as features
from pyrualean.hosts import robodojo_tools as tools
from pyrualean.hosts.robodojo_visual_track import VisualTrack
from tests.test_robodojo_hold_reacquire import Masker, textured
from tests.test_robodojo_staged_grasp import setup, through


def visual_station(tmp_path, *, hidden_at_close=False, slip=False):
    env, robot, host, goal = setup(tmp_path)
    get = env.get_obs
    closed_at = []
    queries = []

    def packet():
        obs = get()
        n = len(env.actions)
        closing = bool(env.actions and env.actions[-1]['right_ee_joint_state'] == [0.])
        if closing and not closed_at:
            closed_at.append(n)
        shift = min(n, 30) if not closed_at else min(closed_at[0], 30)
        if slip and closed_at:
            shift += max(0, n-closed_at[0]-2)*2
        frame = textured(n+1, shift=shift, hidden=closing and hidden_at_close)
        obs['vision'] = {'cam_right_wrist': dict(
            color=frame['vision']['cam_left_wrist']['color'],
            intrinsic_matrix=frame['vision']['cam_left_wrist']['intrinsic_matrix'],
            extrinsic_matrix=tools._pose_of('right', obs['state']['right_arm_joint_state']))}
        obs['image_state'] = copy.deepcopy(obs['state'])
        obs['pyrualean_observation'] = dict(source_id=f'image:{n}', native_steps=n,
            aligned=True, source_freshness='captured')
        return obs

    def segment(image, **kw):
        queries.append(len(env.actions))
        # Recognition ceases at closure, but visible material need not disappear.
        return (SimpleNamespace(found=False) if closed_at else Masker().segment(image, **kw))

    env.get_obs = packet
    host._primitives.sam3_client = SimpleNamespace(segment=segment)
    robot.snapshot(refresh=True)
    ref = robot.object_ref('workpiece', 'textured rigid part')
    result = robot.begin_grasp('tracked', goal, opening=.7, approach=[0,0,.025],
                              probe=[0,0,.015], object_ref=ref)
    assert result['object_tracking']['features'] >= 2
    for stage in ('open', 'approach', 'align'):
        result = robot.grasp_step('tracked', stage)
        assert result['reached'], result
    return env, robot, host, ref, closed_at, queries


def test_preclosure_anchor_crosses_native_frames_and_uses_actual_closed_baseline(tmp_path):
    env, robot, host, ref, closed_at, queries = visual_station(tmp_path)
    pre = robot._grasp_tracks['tracked']
    before = len(env.actions)
    closed = robot.grasp_step('tracked', 'close')
    evidence = closed['object_evidence']
    assert closed['reached'] and evidence['baseline_ready'], closed
    assert evidence['state'] == 'unknown'
    assert evidence['baseline_quality']['space'] == 'wrist_image_2d'
    base = robot._hold_baselines[(ref['object_id'], 'right')]
    assert base['provenance']['native_steps'] == len(env.actions) > before
    assert base['provenance']['source_id'] != pre['provenance']['source_id']
    assert set(base['landmarks']) <= set(pre['landmarks'])
    assert evidence['association'] == 'continuous_visual_tracking'
    assert evidence['continuity']['frames'] >= len(env.actions)
    assert all(n < closed_at[0] for n in queries)  # no text re-detection after closing
    probed = robot.grasp_step('tracked', 'probe')
    assert probed['object_evidence']['state'] == 'image_consistent', probed
    assert probed['object_evidence']['baseline_id'] == evidence['baseline_id']
    assert probed['object_evidence']['continuity']['frames'] > 0
    assert 'checked_frame_flow' in probed['object_evidence']['association_details'].values()
    n = len(env.actions)
    for stage in ('close', 'probe'):
        again = robot.grasp_step('tracked', stage)
        assert again['object_evidence']['replayed_evidence']
    assert len(env.actions) == n
    assert len(host._visual_tracks) == 1
    plain_env, plain, _, _ = setup(tmp_path/'unbound')
    through(plain)
    assert len(plain_env.actions) == n  # tracking cannot buy evidence with extra motions


def test_fully_occluded_close_does_not_turn_preclosure_anchor_into_hold(tmp_path):
    env, robot, host, ref, closed_at, queries = visual_station(tmp_path, hidden_at_close=True)
    result = robot.grasp_step('tracked', 'close')
    assert result['reached']
    assert result['object_evidence']['reason'] == 'baseline_not_ready'
    assert not result['object_evidence']['baseline_ready']
    assert result['object_evidence']['baseline_id'] is None
    assert all(n < closed_at[0] for n in queries)
    assert host._visual_tracks['right'][1].anchor['landmarks']


def test_motion_relative_to_gripper_accumulates_despite_rolling_image_track(tmp_path):
    _, robot, _, ref, _, _ = visual_station(tmp_path, slip=True)
    closed = robot.grasp_step('tracked', 'close')
    assert closed['object_evidence']['baseline_ready']
    base = copy.deepcopy(robot._hold_baselines[(ref['object_id'], 'right')])
    result = robot.grasp_step('tracked', 'probe')
    assert result['object_evidence']['state'] == 'image_relative_motion', result
    assert robot._hold_baselines[(ref['object_id'], 'right')] == base


def make_track():
    old = textured(1)
    source = old['pyrualean_observation']
    anchor = dict(provenance=source, landmarks={
        'material': dict(pixels={'cam_left_wrist':[160,66]})})
    return VisualTrack(anchor, old, source, arm='left', epoch=1)


def test_lost_frame_recovers_original_identity_and_duplicate_frames_do_not_advance():
    track = make_track()
    hidden = textured(2, hidden=True)
    track.advance(hidden, hidden['pyrualean_observation'], epoch=1)
    assert not track.previous['landmarks']
    restored = textured(3, shift=70)
    result = track.read(restored, restored['pyrualean_observation'], epoch=1)
    assert result['points']['material']['cam_left_wrist'] == pytest.approx([160,136], abs=1)
    count = track.frames
    track.read(restored, restored['pyrualean_observation'], epoch=1)
    assert track.frames == count
    assert track.anchor['provenance']['source_id'] == 'capture:1'


@pytest.mark.parametrize('fault', ['unaligned', 'old_frame', 'epoch', 'repeated_pattern'])
def test_invalid_chain_or_ambiguous_reacquisition_never_invents_points(fault):
    track = make_track()
    hidden = textured(2, hidden=True)
    track.advance(hidden, hidden['pyrualean_observation'], epoch=1)
    current = textured(3, repeated=fault == 'repeated_pattern')
    source = current['pyrualean_observation']
    if fault == 'unaligned':
        source['aligned'] = False
    if fault == 'old_frame':
        source['native_steps'] = 0
    result = track.read(current, source, epoch=2 if fault == 'epoch' else 1)
    assert not result['points']
    assert result['failures']


def test_flow_tracks_observed_translation_not_a_rigid_tcp_prediction():
    old, current = textured(1), textured(2, shift=8)
    points, failures = features.flow(dict(landmarks={'p':dict(
        pixels={'cam_left_wrist':[160,66]})}), old, current)
    assert points['p']['cam_left_wrist'] == pytest.approx([160,74], abs=.2)
    assert not failures


def test_tracker_owns_its_camera_buffers_and_failure_cannot_erase_motion(tmp_path, monkeypatch):
    old = textured(1)
    source = old['pyrualean_observation']
    track = VisualTrack(dict(provenance=source, landmarks={'p': dict(
        pixels={'cam_left_wrist':[160,66]})}), old, source, arm='left')
    old['vision']['cam_left_wrist']['color'][:] = 0
    current = textured(2, shift=8)
    assert track.read(current, current['pyrualean_observation'])['points']
    env, robot, host, _, _, _ = visual_station(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('decoder failed')
    monkeypatch.setattr(host._visual_tracks['right'][1], 'advance', fail)
    n = len(env.actions)
    closed = robot.grasp_step('tracked', 'close')
    assert closed['reached'] and len(env.actions) > n
    assert closed['object_evidence']['state'] == 'unknown'
    n = len(env.actions)
    assert robot.grasp_step('tracked', 'close')['reached']
    assert len(env.actions) == n


def test_current_declared_material_points_can_seed_preclosure_identity(tmp_path):
    env, robot, _, ref, _, _ = visual_station(tmp_path)
    shift = min(len(env.actions), 30)
    named = {'surface_a': {'cam_right_wrist': [150, 66+shift]},
             'surface_b': {'cam_right_wrist': [170, 66+shift]}}
    robot.observe_object(ref, landmarks=named)
    closed = robot.grasp_step('tracked', 'close')
    assert closed['object_evidence']['baseline_ready']
    base = robot._hold_baselines[(ref['object_id'], 'right')]
    assert set(base['landmarks']) == set(named)
