"""Object evidence uses recorded sensors, not task state or closure alone."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.hosts import robodojo_object_evidence as evidence
from pyrualean.hosts import robodojo_tools as tools
from pyrualean.hosts.robodojo_native import NativeBackend, NativeToolkit
from pyrualean.object_evidence import assess_hold
from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot


def packet(sequence=1):
    state, vision = {}, {}
    for arm in ('left', 'right'):
        state[f'{arm}_arm_joint_state'] = [0.] * 6
        state[f'{arm}_ee_joint_state'] = [.2]
    for camera, x in [('cam_head', -.25), ('cam_left_wrist', .25)]:
        transform = np.eye(4)
        transform[:3, 3] = [x, 0, 2]
        vision[camera] = dict(color=np.zeros((320, 320, 3), np.uint8),
            intrinsic_matrix=np.array([[300., 0, 160], [0, 300, 160], [0, 0, 1]]),
            extrinsic_matrix=transform)
    return dict(state=state, image_state=copy.deepcopy(state), vision=vision,
        instruction='Inspect the object', pyrualean_observation=dict(
            source_id=f'capture:{sequence}', native_steps=sequence-1,
            aligned=True, captured_at=sequence, source_freshness='captured'))


class Environment:
    def __init__(self):
        self.packet = packet()
        self.actions = []

    def get_obs(self):
        return self.packet

    def get_status(self):
        return dict(step=self.packet['pyrualean_observation']['native_steps'], step_limit=100)

    def step(self, action):
        self.actions.append(action)
        raise AssertionError('evidence cannot dispatch actions')


@pytest.fixture
def scene(tmp_path):
    env = Environment()
    sam = SimpleNamespace(segment=lambda image, **kw: SimpleNamespace(
        found=True, mask=np.ones(image.shape[:2], bool), score=.9))
    toolkit = NativeToolkit(env, sam, tmp_path)
    backend = NativeBackend(toolkit)
    robot = RobodojoOfficialColourRobot(backend, primitives='no-vla')
    return env, toolkit, backend, robot


def pixels(obs, point=(0., 0., 1.2)):
    prim = SimpleNamespace(_snapshot_context={})
    return {'corner': {camera: list(tools._perception_view(prim, obs, camera).project(point)[:2])
                       for camera in obs['vision']}}


def advance(scene):
    env, toolkit, backend, robot = scene
    env.packet = packet(env.packet['pyrualean_observation']['native_steps'] + 2)
    robot.snapshot(refresh=True)


def sample(source, point=(.1, 0, 0), tcp=None, epoch=1, opening=.2, uncertainty=.003):
    arm = dict(tcp_matrix=(np.eye(4) if tcp is None else tcp).tolist(),
               command=0., command_epoch=epoch, opening=opening)
    return dict(provenance=dict(aligned=True, source_freshness='captured', source_id=source),
        robot={'left': arm}, landmarks={'corner': dict(xyz=list(point), uncertainty_m=uncertainty)})


def translate(x):
    t = np.eye(4)
    t[0, 3] = x
    return t


def test_following_requires_observed_motion_and_reports_point_coverage():
    a = sample('a')
    b = sample('b', (.15, 0, 0), translate(.05))
    out = assess_hold(a, b, 'left')
    assert out['state'] == 'following'
    assert out['features']['corner']['relative_change_m'] == pytest.approx(0)
    assert not out['grasp_verified'] and not out['identity_verified']
    assert 'unobserved' in out['coverage']
    assert assess_hold(a, sample('b'), 'left')['state'] == 'unknown'
    # Motion smaller than uncertainty cannot distinguish following from a stationary object.
    small = sample('b', (.1, 0, 0), translate(.02), uncertainty=.014)
    assert assess_hold(sample('a', uncertainty=.014), small, 'left')['state'] == 'unknown'


def test_tcp_rotation_is_accounted_for_without_assuming_object_pose():
    turn = np.eye(4)
    turn[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    out = assess_hold(sample('a'), sample('b', (0, .1, 0), turn), 'left')
    assert out['state'] == 'following'
    assert out['features']['corner']['tcp_xyz'] == pytest.approx([.1, 0, 0])


def test_static_object_and_gradual_slip_are_not_reset_away():
    baseline = sample('a')
    assert assess_hold(baseline, sample('b', tcp=translate(.05)), 'left')[
        'state'] == 'relative_motion'
    one = sample('b', (.154, 0, 0), translate(.05))
    two = sample('c', (.209, 0, 0), translate(.1))
    assert assess_hold(baseline, one, 'left')['state'] == 'following'
    assert assess_hold(baseline, two, 'left')['state'] == 'relative_motion'
    assert assess_hold(one, two, 'left')['state'] == 'following'  # rolling baseline would miss slip


@pytest.mark.parametrize('change,reason', [
    ('same', 'same_observation_not_new_evidence'),
    ('unaligned', 'unverified_observation'),
    ('epoch', 'grasp_command_changed_reset_baseline'),
    ('history', 'gripper_command_history_unavailable'),
    ('open', 'gripper_not_commanded_closed'),
    ('missing_command', 'gripper_not_commanded_closed'),
    ('occluded', 'no_comparable_3d_landmarks'),
    ('uncertain', 'no_comparable_3d_landmarks'),
])
def test_missing_or_invalid_evidence_never_becomes_hold(change, reason):
    a, b = sample('a'), sample('b', (.15, 0, 0), translate(.05))
    if change == 'same':
        b['provenance']['source_id'] = 'a'
    elif change == 'unaligned':
        b['provenance']['aligned'] = False
    elif change == 'epoch':
        b['robot']['left']['command_epoch'] = 3
    elif change == 'history':
        b['robot']['left']['command_epoch'] = None
    elif change == 'open':
        b['robot']['left']['command'] = 1.
    elif change == 'missing_command':
        b['robot']['left']['command'] = None
    elif change == 'occluded':
        b['landmarks'] = {}
    else:
        b['landmarks']['corner']['uncertainty_m'] = .1
    out = assess_hold(a, b, 'left')
    assert out['state'] == 'unknown' and out['reason'] == reason


def test_visual_following_without_obstruction_is_conflicting():
    out = assess_hold(sample('a'), sample('b', (.15, 0, 0), translate(.05), opening=0), 'left')
    assert out['state'] == 'conflicting_evidence'


def test_triangulation_is_free_of_table_height_prior(scene):
    env, toolkit, backend, robot = scene
    ref = robot.object_ref('item', 'small red object')
    result = robot.observe_object(ref, landmarks=pixels(env.packet))
    assert result['landmarks']['corner']['xyz'] == pytest.approx([0, 0, 1.2])
    assert result['landmarks']['corner']['uncertainty_m'] > 0
    assert result['geometry_provenance']['height_prior_used'] is False
    assert result['object_pose'] is None and not result['identity_verified']
    assert robot.require_current(result) is result
    assert 'images' in result['provenance'] and 'images' not in repr(result)
    assert not env.actions


@pytest.mark.parametrize('kind,reason', [
    ('single', 'need_two_calibrated_views'), ('parallel', 'insufficient_parallax'),
    ('mismatch', 'inconsistent_correspondence'),
    ('invalid_calibration', 'need_two_calibrated_views'),
    ('invalid_rotation', 'need_two_calibrated_views')])
def test_ambiguous_or_invalid_geometry_stays_unknown(kind, reason):
    obs = packet()
    pts = pixels(obs)['corner']
    if kind == 'single':
        pts.pop('cam_left_wrist')
    elif kind == 'parallel':
        pts['cam_left_wrist'] = pts['cam_head']
    elif kind == 'mismatch':
        pts['cam_left_wrist'][0] += 30
    elif kind == 'invalid_rotation':
        obs['vision']['cam_left_wrist']['extrinsic_matrix'][:3, :3] = 0
    else:
        obs['vision']['cam_left_wrist']['intrinsic_matrix'][:] = 0
    result = evidence._triangulate(SimpleNamespace(_snapshot_context={}), obs, pts, 2.)
    assert result['xyz'] is None and result['reason'] == reason


@pytest.mark.parametrize('landmarks', [[], False, {'tip': {'cam_head': [True, 4]}},
    {'tip': {'cam_head': [float('nan'), 4]}}, {'tip': {'missing': [3, 4]}},
    {'tip': {'cam_head': [999, 4]}}])
def test_invalid_landmarks_reject_without_motion(scene, landmarks):
    from pyrualean._backend import ToolError
    env, toolkit, backend, robot = scene
    with pytest.raises((ToolError, ValueError)):
        robot.observe_object(robot.object_ref('item', 'red cube'), landmarks=landmarks)
    assert not env.actions


def test_masks_are_only_candidates_not_corresponding_points(scene):
    env, toolkit, backend, robot = scene
    result = robot.observe_object(robot.object_ref('item', 'red cube'))
    assert not result['landmarks'] and result['object_pose'] is None
    assert all(v['detection'] == 'text_candidate' and not v['identity_verified']
               for v in result['views'].values())
    hold = robot.observe_hold(result['reference'], arm='left', observation=result)
    assert hold['state'] == 'unknown'
    assert not env.actions


def test_flow_tracks_pixels_but_discards_lost_or_textureless_points():
    pytest.importorskip('cv2')
    old, new = packet(), packet(2)
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    old['vision']['cam_head']['color'] = image
    new['vision']['cam_head']['color'] = np.roll(image, (2, 3), axis=(0, 1))
    previous = {'landmarks': {'corner': {'pixels': {'cam_head': [80, 80]}}}}
    tracked, failures = evidence._track(previous, old, new)
    assert tracked['corner']['cam_head'] == pytest.approx([82, 83], abs=.15)
    assert not failures
    new['vision']['cam_head']['color'][:] = 0
    tracked, failures = evidence._track(previous, old, new)
    assert not tracked and failures
    old['vision']['cam_head']['color'][:] = 0
    assert evidence._track(previous, old, new)[1]['corner/cam_head'] == 'insufficient_texture'


def test_reference_identity_and_observation_cannot_cross_objects_or_sessions(scene, tmp_path):
    env, toolkit, backend, robot = scene
    ref = robot.object_ref('item', 'red cube')
    assert robot.object_ref('item', 'red cube') == ref
    with pytest.raises(ValueError):
        robot.object_ref('item', 'blue cup')
    other = robot.object_ref('second', 'blue cup')
    obs = robot.observe_object(ref, landmarks=pixels(env.packet))
    with pytest.raises(ValueError):
        robot.observe_hold(other, arm='left', observation=obs)
    altered = copy.deepcopy(obs)
    altered['landmarks']['corner']['xyz'][0] = 999
    with pytest.raises(ValueError):
        robot.observe_hold(ref, arm='left', observation=altered)
    second_robot = RobodojoOfficialColourRobot(NativeBackend(
        NativeToolkit(Environment(), None, tmp_path / 'other')), primitives='no-vla')
    with pytest.raises(ValueError):
        second_robot.observe_object(ref)
    assert not env.actions


def test_currentness_historical_reads_and_command_history_are_frozen(scene):
    env, toolkit, backend, robot = scene
    prim = toolkit._primitives
    tools._remember(prim, 'left', 0.)
    advance(scene)
    ref = robot.object_ref('item', 'red cube')
    before = robot.observe_object(ref, landmarks=pixels(env.packet))
    first = robot.observe_hold(ref, arm='left', observation=before)
    assert first['reason'] == 'baseline_ready_not_hold_proof'
    repeated = robot.observe_hold(ref, arm='left')
    assert repeated['reason'] == 'same_observation_not_new_evidence'
    epoch = before['robot']['left']['command_epoch']
    tools._remember(prim, 'left', 0.)  # retention doesn't reset the grasp identity
    assert prim._grip_command_epochs['left'] == epoch
    tools._remember(prim, 'left', 1.)
    tools._remember(prim, 'left', 0.)  # reopen/reclose between observations
    advance(scene)
    after = robot.observe_object(ref, landmarks=pixels(env.packet))
    out = robot.observe_hold(ref, arm='left', observation=after)
    assert out['reason'] == 'grasp_command_changed_reset_baseline'
    assert out['baseline_id'] == before['object_observation_id']
    historic = robot.observe_object(ref, landmarks=pixels(env.packet),
                                    observation_id=before['provenance']['observation_id'])
    assert historic['robot']['left']['command_epoch'] == epoch
    assert historic['robot']['left']['command'] == 0
    assert not robot.perception_is_current(historic)
    assert robot._object_latest[ref['object_id']] == after
    out = robot.observe_hold(ref, arm='left', observation=historic, reset=True)
    assert out['reason'] == 'stale_or_unresolved_observation'
    assert robot._hold_baselines[(ref['object_id'], 'left')] == before
    out = robot.observe_hold(ref, arm='left', observation=after, reset=True)
    assert out['reason'] == 'baseline_ready_not_hold_proof'
    assert out['baseline_id'] == after['object_observation_id']
    assert not env.actions
