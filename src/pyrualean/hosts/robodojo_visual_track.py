"""Read-only, bounded RGB tracks across native actions; never predict a held pose."""
from __future__ import annotations

import copy
from collections import deque

import numpy as np

from . import robodojo_features as features


class VisualTrack:
    """Keep an identity anchor and a rolling image track, not a rolling hold baseline."""

    def __init__(self, observation, packet, source, *, arm, epoch=None):
        self.arm, self.epoch = arm, epoch
        self.anchor = copy.deepcopy(observation)
        self.anchor_packet = self._images(packet)
        self.previous = copy.deepcopy(observation)
        self.packet = self.anchor_packet
        self.source = copy.deepcopy(source)
        self.failures, self.methods = {}, {}
        self.frames = 0
        self.recovered = 0
        self.invalid = None
        self.trail = deque(maxlen=32)

    @staticmethod
    def _images(packet):
        # Native camera buffers may be reused. Keep only detached RGB, never
        # simulator object poses, depth, or the entire sensor/action history.
        return {'vision': {c: {'color': v['color'].copy()}
                           for c, v in packet.get('vision', {}).items()
                           if v.get('color') is not None}}

    def invalidate(self, reason):
        self.invalid = reason
        self.previous = dict(landmarks={})
        self.failures = {'continuity': reason}

    def advance(self, packet, source, *, epoch=None):
        if self.invalid:
            return
        if self.epoch is not None and epoch != self.epoch:
            self.invalidate('grasp_command_changed')
            return
        if (not source.get('source_id') or source.get('aligned') is not True
                or source.get('source_freshness') != 'captured'):
            self.invalidate('unverified_intermediate_frame')
            return
        if source['source_id'] == self.source.get('source_id'):
            return
        old_step, new_step = self.source.get('native_steps'), source.get('native_steps')
        if old_step is None or new_step is None or new_step <= old_step:
            self.invalidate('nonmonotonic_intermediate_frame')
            return
        points, failures = features.flow(self.previous, self.packet, packet)
        self.previous = dict(landmarks={n: dict(pixels=p) for n, p in points.items()})
        self.methods = {f'{n}/{c}': 'checked_frame_flow' for n, p in points.items() for c in p}
        self.failures = failures
        self.packet, self.source = self._images(packet), copy.deepcopy(source)
        self.frames += 1
        self.trail.append(dict(source_id=source['source_id'], native_steps=new_step,
                               features=len(points)))

    def read(self, packet, source, *, epoch=None):
        self.advance(packet, source, epoch=epoch)
        if not self.invalid:
            # A lost track can return only through the original appearance.
            # Re-detecting the text label is not identity recovery.
            found, failures, methods = features.reassociate(
                self.anchor, self.anchor_packet, packet)
            for name, cameras in found.items():
                for camera, pixel in cameras.items():
                    pixels = self.previous['landmarks'].setdefault(name, {'pixels': {}})['pixels']
                    if camera in pixels and np.linalg.norm(np.asarray(pixels[camera])-pixel) > 3.:
                        pixels.pop(camera)
                        self.failures[f'{name}/{camera}'] = 'flow_anchor_disagree'
                    elif camera not in pixels:
                        pixels[camera] = pixel
                        self.recovered += 1
                        self.methods[f'{name}/{camera}'] = methods[f'{name}/{camera}']
            self.previous['landmarks'] = {n: f for n, f in self.previous['landmarks'].items()
                                          if f['pixels']}
            self.failures.update(failures)
            for name, feature in self.previous['landmarks'].items():
                for camera in feature['pixels']:
                    self.failures.pop(f'{name}/{camera}', None)
        return dict(
            points={n: f['pixels'] for n, f in self.previous['landmarks'].items()},
            failures=copy.deepcopy(self.failures), methods=copy.deepcopy(self.methods),
            continuity=dict(anchor_source_id=self.anchor.get('provenance', {}).get('source_id'),
                source_id=self.source.get('source_id'), frames=self.frames,
                recovered_features=self.recovered, invalid=self.invalid,
                recent_frames=list(self.trail), identity_verified=False))
