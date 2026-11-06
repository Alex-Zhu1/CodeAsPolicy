"""Task-independent RGB feature capture and fixed-keyframe reassociation.

No grasp prediction is used to find a match: otherwise assuming retention would
become evidence of retention. Masks nominate an object; they do not certify identity.
"""
from __future__ import annotations

import math

import numpy as np


def _gray(image):
    import cv2
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] < 3 or a.dtype != np.uint8:
        raise ValueError('feature analysis needs recorded uint8 RGB')
    return cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)


def _patch(gray, pixel, radius=8):
    r, c = (int(round(x)) for x in pixel)
    if min(r, c) < radius or r + radius >= gray.shape[0] or c + radius >= gray.shape[1]:
        return None
    p = gray[r-radius:r+radius+1, c-radius:c+radius+1]
    return p if float(p.std()) >= 8. else None


def appearance(obs, pixels):
    """Recorded-pixel trackability; no template or image arrays in public results."""
    out = {}
    try:
        import cv2  # noqa: F401
    except ImportError:
        return out
    for camera, pixel in pixels.items():
        image = (obs.get('vision', {}).get(camera) or {}).get('color')
        if image is not None:
            out[camera] = dict(trackable=_patch(_gray(image), pixel) is not None)
    return out


def flow(previous, old, obs):
    """Follow visible material with two-way optical flow and local appearance.

    No TCP/rigid-object prediction seeds the search. Empty observations stay
    empty; reacquisition belongs to the independently retained anchor.
    """
    try:
        import cv2
    except ImportError:
        return {}, {'continuity': 'opencv_unavailable'}
    points, failures = {}, {}
    for camera, cam in old.get('vision', {}).items():
        entries = [(n, f['pixels'][camera]) for n, f in previous.get('landmarks', {}).items()
                   if camera in f.get('pixels', {})]
        if not entries:
            continue
        before, after = cam.get('color'), (obs.get('vision', {}).get(camera) or {}).get('color')
        if before is None or after is None or before.shape != after.shape:
            failures[camera] = 'camera_unavailable_or_changed'
            continue
        a, b = _gray(before), _gray(after)
        start = np.array([[[p[1], p[0]]] for _, p in entries], np.float32)
        end, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, start, None, winSize=(21, 21), maxLevel=3)
        if end is None:
            failures[camera] = 'flow_lost'
            continue
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, end, None, winSize=(21, 21), maxLevel=3)
        for i, (name, pixel) in enumerate(entries):
            label = f'{name}/{camera}'
            if (not ok[i, 0] or back is None or not back_ok[i, 0]
                    or not np.isfinite(end[i]).all() or not np.isfinite(back[i]).all()
                    or np.linalg.norm(back[i]-start[i]) > 1.5):
                failures[label] = 'flow_inconsistent'
                continue
            candidate = [float(end[i, 0, 1]), float(end[i, 0, 0])]
            pa, pb = _patch(a, pixel), _patch(b, candidate)
            if pa is None or pb is None:
                failures[label] = 'flow_texture_lost'
                continue
            similarity = float(cv2.matchTemplate(pb, pa, cv2.TM_CCOEFF_NORMED)[0, 0])
            if not math.isfinite(similarity) or similarity < .85:
                failures[label] = 'flow_appearance_changed'
                continue
            points.setdefault(name, {})[camera] = candidate
    return points, failures


def _matches(a, b):
    """Mutual, distinctive descriptor matches. Repeated patterns reject."""
    import cv2
    if a is None or b is None or len(b) < 2:
        return {}
    bf = cv2.BFMatcher()
    forward = bf.knnMatch(a, b, k=2)
    reverse = bf.match(b, a)
    back = {m.queryIdx: m.trainIdx for m in reverse}
    return {m.queryIdx: m.trainIdx for pair in forward if len(pair) == 2
            for m, n in [pair] if m.distance < 200. and m.distance < .7 * n.distance
            and back.get(m.trainIdx) == m.queryIdx}


def capture(obs, masks, *, arm=None, identity='capture'):
    """Find material features inside independently segmented object masks."""
    try:
        import cv2
    except ImportError:
        return {}, {'capture': 'opencv_unavailable'}
    sift = cv2.SIFT_create(nfeatures=256)
    detected, reasons = {}, {}
    for camera, mask in masks.items():
        gray = _gray(obs['vision'][camera]['color'])
        if mask.shape != gray.shape or not 9 <= int(mask.sum()) < .75 * mask.size:
            reasons[camera] = 'mask_empty_or_not_object_specific'
            continue
        kp, desc = sift.detectAndCompute(gray, mask.astype(np.uint8) * 255)
        good = []
        for i, k in enumerate(kp):
            pixel = [k.pt[1], k.pt[0]]
            patch = _patch(gray, pixel)
            r, c = (int(round(x)) for x in pixel)
            if patch is not None and float(mask[r-8:r+9, c-8:c+9].mean()) >= .35:
                good.append(i)
        if not good:
            reasons[camera] = 'no_distinctive_object_features'
            continue
        detected[camera] = ([kp[i] for i in good], desc[good])
    preferred = f'cam_{arm}_wrist'
    cameras = sorted(detected, key=lambda c: (c != preferred, -len(detected[c][0]), c))
    points = {}
    for camera in cameras:
        keys, desc = detected[camera]
        matches = {other: _matches(desc, detected[other][1]) for other in cameras
                   if other != camera}
        added = 0
        for i in sorted(range(len(keys)), key=lambda i: -keys[i].response):
            pixel = [keys[i].pt[1], keys[i].pt[0]]
            if any(camera in f['pixels'] and np.linalg.norm(
                    np.asarray(f['pixels'][camera]) - pixel) < 8. for f in points.values()):
                continue
            pixels = {camera: pixel}
            shape = {camera: dict(size=float(keys[i].size), angle=float(keys[i].angle))}
            for other, pairs in matches.items():
                if i in pairs:
                    k = detected[other][0][pairs[i]]
                    pixels[other] = [k.pt[1], k.pt[0]]
                    shape[other] = dict(size=float(k.size), angle=float(k.angle))
            name = f'feature_{identity}_{len(points)}'
            points[name] = dict(pixels=pixels, appearance=shape)
            added += 1
            if len(points) >= 8:
                return points, reasons
            if added >= 4:
                break
    return points, reasons


def _template(old, new, pixel):
    import cv2
    patch = _patch(old, pixel)
    if patch is None:
        return None, 'insufficient_texture'
    scores = cv2.matchTemplate(new, patch, cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(scores).all():
        return None, 'nonfinite_match'
    _, peak, _, loc = cv2.minMaxLoc(scores)
    x, y = loc
    # The second spatially distinct peak must be materially worse. A repeated
    # corner/stripe cannot silently acquire the old feature's identity.
    other = scores.copy()
    other[max(0, y-8):y+9, max(0, x-8):x+9] = -1
    second = float(other.max())
    if peak < .90 or peak - second < .10:
        return None, 'appearance_missing_or_ambiguous'
    return [float(y+8), float(x+8)], None


def reassociate(anchor, old, obs):
    """Find original features after lost intermediate frames, without rebasing."""
    try:
        import cv2
    except ImportError:
        return {}, {'reassociation': 'opencv_unavailable'}, {}
    points, failures, details = {}, {}, {}
    sift = cv2.SIFT_create(nfeatures=512)
    for camera in old.get('vision', {}):
        before = old['vision'][camera].get('color')
        after = (obs.get('vision', {}).get(camera) or {}).get('color')
        features = [(name, f) for name, f in anchor.get('landmarks', {}).items()
                    if camera in f.get('pixels', {})]
        if not features:
            continue
        if before is None or after is None or before.shape != after.shape:
            failures[camera] = 'camera_unavailable_or_changed'
            continue
        a, b = _gray(before), _gray(after)
        queries = []
        for _, f in features:
            r, c = f['pixels'][camera]
            shape = f.get('appearance', {}).get(camera, {})
            queries.append(cv2.KeyPoint(float(c), float(r), float(shape.get('size', 17.)),
                                        float(shape.get('angle', 0.))))
        qk, qd = sift.compute(a, queries)
        keys, desc = sift.detectAndCompute(b, None)
        pairs = _matches(qd, desc)
        for i, (name, feature) in enumerate(features):
            label = f'{name}/{camera}'
            template, reason = _template(a, b, feature['pixels'][camera])
            descriptor = None
            if i in pairs and len(qk) == len(features):
                k = keys[pairs[i]]
                # Pixel offsets use the keypoint's scale/orientation. The
                # original feature is at the keypoint centre, not a mask centre.
                if math.isfinite(k.pt[0]) and math.isfinite(k.pt[1]):
                    descriptor = [float(k.pt[1]), float(k.pt[0])]
            if descriptor is not None and template is not None and np.linalg.norm(
                    np.asarray(descriptor) - template) > 3.:
                failures[label] = 'descriptor_template_disagree'
                continue
            chosen = descriptor if descriptor is not None else template
            if chosen is None:
                failures[label] = reason or 'no_distinctive_match'
            else:
                points.setdefault(name, {})[camera] = chosen
                details[label] = ('anchor_descriptor' if descriptor is not None
                                  else 'anchor_template')
    return points, failures, details
