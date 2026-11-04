"""Generic closure capture, lost-feature recovery and limited image evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hosts import robodojo_features as features
from pyrualean.hosts import robodojo_tools as tools
from pyrualean.object_evidence import assess_hold
from tests.test_robodojo_object_evidence import (
    packet,
    pixels,
    sample,
    scene,
    translate,
)
from tests.test_robodojo_staged_grasp import setup

# Re-export pytest fixture through import without duplicating the native backend.
__all__ = ['scene']


def textured(sequence=1, shift=0, *, repeated=False, hidden=False):
    cv2 = pytest.importorskip('cv2')
    obs = packet(sequence)
    rng = np.random.default_rng(83)
    texture = cv2.GaussianBlur(rng.integers(20, 240, (64, 64), dtype=np.uint8), (3, 3), .5)
    for camera, center in [('cam_head', 254), ('cam_left_wrist', 66)]:
        image = obs['vision'][camera]['color']
        c = center + shift
        if not hidden:
            lo, hi = max(0, c-32), min(image.shape[1], c+32)
            image[128:192, lo:hi] = texture[:, lo-(c-32):hi-(c-32), None]
            if repeated:
                image[28:92, lo:hi] = texture[:, lo-(c-32):hi-(c-32), None]
    return obs


class Masker:
    def segment(self, image, **kwargs):
        mask = np.asarray(image).max(axis=2) > 0
        return SimpleNamespace(found=bool(mask.any()), mask=mask, score=.99)


def install(scene, obs):
    env, toolkit, backend, robot = scene
    env.packet = obs
    toolkit._primitives.sam3_client = Masker()
    robot.snapshot(refresh=True)


def test_automatic_capture_finds_object_material_and_matching_views(scene):
    env, toolkit, backend, robot = scene
    tools._remember(toolkit._primitives, 'left', 0.)
    install(scene, textured(2))
    ref = robot.object_ref('piece', 'textured object')
    hold = robot.observe_hold(ref, arm='left', reset=True)
    assert hold['baseline_ready'] and hold['baseline_quality']['space'] == '3d'
    assert hold['state'] == 'unknown' and hold['reason'] == 'baseline_ready_not_hold_proof'
    baseline = robot._hold_baselines[(ref['object_id'], 'left')]
    assert any(len(f['pixels']) == 2 and f['xyz'] is not None
               for f in baseline['landmarks'].values())
    assert hold['association'] == 'masked_material_features'
    assert not env.actions
    assert robot._ledger.records[-1].tool == 'hold_evidence'
    assert robot._ledger.records[-1].summary['baseline_ready']


def test_reassociation_survives_empty_intermediate_frame_and_keeps_baseline(scene, monkeypatch):
    env, toolkit, backend, robot = scene
    monkeypatch.setattr(tools, '_pose_of', lambda arm, q: translate(q[0]))
    tools._remember(toolkit._primitives, 'left', 0.)
    install(scene, textured(2))
    ref = robot.object_ref('piece', 'textured object')
    base = robot.observe_hold(ref, arm='left', reset=True)
    assert base['baseline_ready']
    install(scene, textured(3, hidden=True))
    lost = robot.observe_hold(ref, arm='left')
    assert lost['state'] == 'unknown' and lost['baseline_id'] == base['baseline_id']
    obs = textured(4, shift=24)
    for state in ('state', 'image_state'):
        obs[state]['left_arm_joint_state'][0] = 24 * .8 / 300
    install(scene, obs)
    found = robot.observe_hold(ref, arm='left')
    assert found['state'] == 'following', found
    assert found['evidence_space'] == '3d'
    assert found['baseline_id'] == base['baseline_id']
    assert found['association_details']
    # Repeated reset cannot erase displacement within the same grip epoch.
    original = robot._hold_baselines[(ref['object_id'], 'left')]
    obs = textured(5, shift=44)
    for state in ('state', 'image_state'):
        obs[state]['left_arm_joint_state'][0] = 24 * .8 / 300
    install(scene, obs)
    slipped = robot.observe_hold(ref, arm='left', reset=True)
    assert slipped['state'] == 'relative_motion', slipped
    assert robot._hold_baselines[(ref['object_id'], 'left')] == original
    assert not env.actions


def test_reassociation_rejects_ambiguous_appearance_and_occlusion():
    old = textured()
    anchor = dict(landmarks={'point': dict(pixels={'cam_left_wrist': [160, 66]})})
    for current in [textured(2, repeated=True), textured(2, hidden=True)]:
        points, failures, _ = features.reassociate(anchor, old, current)
        assert not points and failures


def test_template_reassociation_recovers_large_translation_without_old_pixel_fallback():
    old = textured()
    anchor = dict(landmarks={'point': dict(pixels={'cam_left_wrist': [160, 66]})})
    points, _, methods = features.reassociate(anchor, old, textured(2, shift=80))
    assert points['point']['cam_left_wrist'] == pytest.approx([160, 146], abs=1)
    assert methods['point/cam_left_wrist'] == 'anchor_template'


def test_empty_closure_capture_is_not_a_baseline_or_silently_replaced(scene):
    env, toolkit, backend, robot = scene
    tools._remember(toolkit._primitives, 'left', 0.)
    install(scene, textured(2, hidden=True))
    ref = robot.object_ref('piece', 'textured object')
    first = robot.observe_hold(ref, arm='left', reset=True)
    assert not first['baseline_ready'] and first['baseline_id'] is None
    assert first['reason'] == 'baseline_not_ready'
    assert first['recovery']['observation_id'] == first['capture_observation_id']
    install(scene, textured(3))
    later = robot.observe_hold(ref, arm='left', reset=True)
    assert not later['baseline_ready'] and later['baseline_id'] is None
    assert later['capture_observation_id'] == first['capture_observation_id']
    # New gripper epoch is a new grasp, so fresh features may establish its own baseline.
    tools._remember(toolkit._primitives, 'left', 1.)
    tools._remember(toolkit._primitives, 'left', 0.)
    install(scene, textured(4))
    new = robot.observe_hold(ref, arm='left', reset=True)
    assert new['baseline_ready']
    assert new['capture_observation_id'] != first['capture_observation_id']
    assert not env.actions


def test_missing_baseline_can_be_repaired_only_on_original_capture(scene):
    env, toolkit, backend, robot = scene
    tools._remember(toolkit._primitives, 'left', 0.)
    install(scene, packet(2))
    ref = robot.object_ref('piece', 'object')
    pending = robot.observe_hold(ref, arm='left', landmarks={}, reset=True)
    install(scene, packet(3))
    later = robot.observe_hold(ref, arm='left', landmarks=pixels(env.packet), reset=True)
    assert not later['baseline_ready']
    repaired = robot.observe_hold(ref, arm='left', landmarks=pixels(env.packet),
                                  observation_id=pending['capture_observation_id'])
    assert repaired['baseline_ready'] and not repaired['current']
    assert repaired['reason'] == 'baseline_repaired_await_current_observation'
    assert repaired['capture_observation_id'] == pending['capture_observation_id']
    assert robot._hold_baselines[(ref['object_id'], 'left')]['provenance'][
        'observation_id'] == pending['capture_observation_id']
    assert not env.actions


def image_sample(source, *, drift=0, motion=.04, opening=.2):
    data = sample(source, tcp=translate(motion), opening=opening)
    camera = 'cam_left_wrist'
    data['landmarks'] = {name: dict(xyz=None, uncertainty_m=None,
        pixels={camera: [40 + i*30 + drift, 60]}, appearance={camera: {'trackable': True}})
        for i, name in enumerate(['a', 'b'])}
    data['views'] = {camera: dict(shape=[160, 160], tcp_from_camera=np.eye(4).tolist(),
                                  intrinsics=[180, 180, 80, 80])}
    return data


@pytest.mark.parametrize('drift,expected', [(0, 'image_consistent'), (10, 'image_relative_motion')])
def test_image_evidence_is_explicitly_distinct_from_3d_hold(drift, expected):
    a, b = image_sample('a', motion=0), image_sample('b', drift=drift)
    out = assess_hold(a, b, 'left')
    assert out['state'] == expected and out['evidence_space'] == 'wrist_image_2d'
    assert not out['grasp_verified'] and 'depth' in out['coverage']


@pytest.mark.parametrize('change', ['same_source', 'no_motion', 'missing_point', 'changed_camera',
                                    'changed_intrinsics', 'epoch', 'empty'])
def test_image_consistency_has_no_shortcuts_to_a_3d_following_claim(change):
    a, b = image_sample('a', motion=0), image_sample('b')
    if change == 'same_source':
        b['provenance']['source_id'] = 'a'
    elif change == 'no_motion':
        b['robot']['left']['tcp_matrix'] = a['robot']['left']['tcp_matrix']
    elif change == 'missing_point':
        b['landmarks'].pop('b')
    elif change == 'changed_camera':
        b['views']['cam_left_wrist']['tcp_from_camera'][0][3] = .02
    elif change == 'changed_intrinsics':
        b['views']['cam_left_wrist']['intrinsics'][0] = 200
    elif change == 'epoch':
        b['robot']['left']['command_epoch'] = 9
    else:
        b['robot']['left']['opening'] = 0
    assert assess_hold(a, b, 'left')['state'] == 'unknown'


def test_bound_grasp_captures_once_after_close(tmp_path, monkeypatch):
    env, robot, host, goal = setup(tmp_path)
    ref = robot.object_ref('piece', 'an object')
    # The existing unbound ID cannot change physical/object identity.
    conflict = robot.begin_grasp('pick', goal, object_ref=ref)
    assert conflict['stop_reason'] == 'grasp_object_identity_conflict'
    robot.begin_grasp('bound', goal, opening=.7, approach=[0, 0, .025],
                      probe=[0, 0, .015], object_ref=ref)
    calls = []
    original = robot.observe_hold
    def record(*args, **kwargs):
        calls.append((len(env.actions), kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(robot, 'observe_hold', record)
    for stage in ('open', 'approach', 'align'):
        assert robot.grasp_step('bound', stage)['reached']
    assert not calls
    closed = robot.grasp_step('bound', 'close')
    assert closed['reached'] and len(calls) == 1
    assert calls[0][0] == len(env.actions) and calls[0][1]['reset'] is True
    n = len(env.actions)
    replay = robot.grasp_step('bound', 'close')
    assert replay['object_evidence']['replayed_evidence']
    assert len(calls) == 1 and len(env.actions) == n
    probed = robot.grasp_step('bound', 'probe')
    assert probed['reached'] and len(calls) == 2 and not calls[-1][1]['reset']


def test_capture_error_preserves_completed_close(tmp_path, monkeypatch):
    env, robot, host, goal = setup(tmp_path)
    ref = robot.object_ref('piece', 'an object')
    robot.begin_grasp('bound', goal, opening=.7, approach=[0, 0, .025], object_ref=ref)
    for stage in ('open', 'approach', 'align'):
        assert robot.grasp_step('bound', stage)['reached']
    def fail(*args, **kwargs):
        raise RuntimeError('image decoder unavailable')
    monkeypatch.setattr(robot, 'observe_hold', fail)
    closed = robot.grasp_step('bound', 'close')
    assert closed['reached']
    assert closed['object_evidence']['reason'] == 'evidence_acquisition_failed'
    n = len(env.actions)
    assert robot.grasp_step('bound', 'close')['reached']
    assert len(env.actions) == n


def test_single_wrist_capture_and_reassociation_give_only_image_evidence(scene, monkeypatch):
    env, toolkit, backend, robot = scene
    monkeypatch.setattr(tools, '_pose_of', lambda arm, q: translate(q[0]))
    tools._remember(toolkit._primitives, 'left', 0.)
    obs = textured(2)
    obs['vision'].pop('cam_head')
    install(scene, obs)
    ref = robot.object_ref('piece', 'textured object')
    baseline = robot.observe_hold(ref, arm='left', reset=True)
    assert baseline['baseline_ready']
    assert baseline['baseline_quality']['space'] == 'wrist_image_2d'
    for seq, shift, expected in [(3, 0, 'image_consistent'), (4, 10, 'image_relative_motion')]:
        obs = textured(seq, shift=shift)
        obs['vision'].pop('cam_head')
        obs['vision']['cam_left_wrist']['extrinsic_matrix'][0, 3] += .04
        for state in ('state', 'image_state'):
            obs[state]['left_arm_joint_state'][0] = .04
        install(scene, obs)
        result = robot.observe_hold(ref, arm='left')
        assert result['state'] == expected, result
        assert result['baseline_id'] == baseline['baseline_id']
        assert result['evidence_space'] == 'wrist_image_2d'
        assert not result['grasp_verified'] and not env.actions
