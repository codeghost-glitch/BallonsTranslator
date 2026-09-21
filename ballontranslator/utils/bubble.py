"""Compact page-space bubble geometry shared by detection and typesetting."""

import math
import sys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .logger import logger as LOGGER


#: Page-space radius that counts a dragged text center as "on" the bubble
#: center. Small enough to avoid surprising jumps, large enough to
#: hit without pixel-hunting. Guides and drag snapping share it so the
#: highlight matches the actual snap behavior.
BUBBLE_CENTER_SNAP_RADIUS = 15.0


def normalize_bubble_polygon(value: object, *, strict: bool = False) -> Optional[List[List[float]]]:
    """Discard malformed optional project geometry without losing the text block.

    >>> normalize_bubble_polygon([[0, 0], [20, 0], [10, 20]])
    [[0.0, 0.0], [20.0, 0.0], [10.0, 20.0]]
    """
    if value is None:
        return None
    try:
        points = np.asarray(value, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 2048:
            raise ValueError('expected 3-2048 coordinate pairs')
        if not np.isfinite(points).all() or (points < 0).any() or (points > 1_000_000).any():
            raise ValueError('coordinates are outside the supported page range')
        if abs(cv2.contourArea(points.astype(np.float32))) < 1:
            raise ValueError('empty polygon')
        return points.tolist()
    except (TypeError, ValueError, OverflowError) as error:
        if strict:
            raise ValueError(f'Invalid bubble polygon: {error}') from error
        LOGGER.warning('Ignoring invalid bubble polygon: %s', error)
        return None


def bubble_polygon_from_mask(
    mask: np.ndarray, source_gray: Optional[np.ndarray] = None,
) -> Optional[List[List[float]]]:
    """Trace segmentation, optionally snapping to a matching enclosed white interior.

    Open borders, dark balloons, or unrelated image regions retain segmentation.

    >>> bubble_polygon_from_mask(np.zeros((8, 8), dtype=np.uint8)) is None
    True
    """
    # Segmentation borders carry pixel stair-steps and 1-2px spurs; a small
    # close-then-open removes them without eroding real geometry (necks,
    # tails), unlike a Gaussian which rounds corners.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(
        cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel),
        cv2.MORPH_OPEN, kernel,
    )
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    snapped = False
    if source_gray is not None:
        if source_gray.shape != mask.shape or source_gray.ndim != 2:
            raise ValueError('Bubble source image must match the mask dimensions.')
        x, y, width, height = cv2.boundingRect(contour)
        margin = max(4, int(min(width, height) * 0.1))
        left, top = max(0, x - margin), max(0, y - margin)
        right = min(mask.shape[1], x + width + margin)
        bottom = min(mask.shape[0], y + height + margin)
        roi = source_gray[top:bottom, left:right]
        predicted = np.zeros(roi.shape, dtype=np.uint8)
        cv2.drawContours(predicted, [contour - [left, top]], -1, 1, cv2.FILLED)
        snapped = False
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (roi >= 200).astype(np.uint8), connectivity=8,
        )
        overlap = np.bincount(labels[predicted != 0], minlength=len(stats))
        overlap[0] = 0
        label = int(overlap.argmax())
        if overlap[label]:
            cx, cy, cw, ch, _ = stats[label]
            # An open balloon can flood the page background. Only enclosed,
            # strongly matching interiors may replace the detector geometry.
            if cx > 0 and cy > 0 and cx + cw < roi.shape[1] and cy + ch < roi.shape[0]:
                borders, _ = cv2.findContours(
                    (labels == label).astype(np.uint8), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                border = max(borders, key=cv2.contourArea)
                enclosed = np.zeros(roi.shape, dtype=np.uint8)
                cv2.drawContours(enclosed, [border], -1, 1, cv2.FILLED)
                intersection = np.count_nonzero(enclosed & predicted)
                union = np.count_nonzero(enclosed | predicted)
                if union and intersection / union >= 0.85:
                    contour = border + [left, top]
                    snapped = True
                else:
                    # Recovery for dashed/touching bubbles: the interior white
                    # leaks to the page through border gaps (failing the
                    # enclosure test) and weak segmentations cut inside the
                    # true border (failing the IoU test). Opening the
                    # component breaks narrow leak necks; the opened interior
                    # is adopted when it contains the segmentation while
                    # staying close to its size — a junk region around stray
                    # text is far larger and still rejected.
                    neck_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (max(9, min(31, min(roi.shape) // 8)),) * 2,
                    )
                    opened = cv2.morphologyEx(
                        (labels == label).astype(np.uint8), cv2.MORPH_OPEN, neck_kernel,
                    )
                    if opened.any():
                        onum, olabels, ostats, _ = cv2.connectedComponentsWithStats(
                            opened, connectivity=8,
                        )
                        ooverlap = np.bincount(olabels[predicted != 0], minlength=onum)
                        ooverlap[0] = 0
                        olabel = int(ooverlap.argmax())
                        if ooverlap[olabel]:
                            ocx, ocy, ocw, och, _ = ostats[olabel]
                            if (ocx > 0 and ocy > 0 and ocx + ocw < roi.shape[1]
                                    and ocy + och < roi.shape[0]):
                                oborders, _ = cv2.findContours(
                                    (olabels == olabel).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                )
                                oborder = max(oborders, key=cv2.contourArea)
                                ofill = np.zeros(roi.shape, dtype=np.uint8)
                                cv2.drawContours(ofill, [oborder], -1, 1, cv2.FILLED)
                                inside = np.count_nonzero(ofill & predicted)
                                if (predicted.any()
                                        and inside / max(1, int(predicted.sum())) >= 0.9
                                        and int(ofill.sum()) <= 1.6 * max(1, int(predicted.sum()))):
                                    contour = oborder + [left, top]
                                    snapped = True
    # The upsampled segmentation boundary carries several-pixel waviness; a
    # light circular low-pass along the contour removes it while large-scale
    # features (necks, tails) survive for splitting and inset geometry.
    points = contour.reshape(-1, 2).astype(np.float64)
    if len(points) > 17 and not snapped:
        # A contour roll-average pulls straight edges inward — a balloon
        # flush against the panel border loses its border edge (page 007
        # teardrop: 121 -> 133). Snapped outlines come from a clean binary
        # interior and stay un-smoothed; approxPolyDP below still strips the
        # 1-2px staircase without moving legitimate straight or diagonal
        # edges.
        smooth_kernel = cv2.getGaussianKernel(17, 4.0).ravel()
        smooth_kernel /= smooth_kernel.sum()
        xs = np.stack([np.roll(points[:, 0], shift) for shift in range(-8, 9)])
        ys = np.stack([np.roll(points[:, 1], shift) for shift in range(-8, 9)])
        contour = np.round(np.stack(
            [smooth_kernel @ xs, smooth_kernel @ ys], axis=1,
        )).astype(np.int32).reshape(-1, 1, 2)
    # Keep the outline as detailed as the stored format allows: smoothing
    # starts small and only coarsens when a huge mask would exceed the point
    # cap, so shallow necks and tail roots survive for splitting and inset
    # geometry instead of being erased by one size-proportional tolerance.
    perimeter = cv2.arcLength(contour, True)
    tolerance = max(1.0, perimeter * 0.001)
    approx = contour
    for _ in range(16):
        approx = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2)
        if len(approx) <= 2048:
            break
        tolerance *= 2.0
    return normalize_bubble_polygon(approx)


def bubble_inner_rect(
    polygon: Sequence[Sequence[float]], padding: Optional[float] = None,
    *, aspect_ratio: Optional[float] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Find an inset rectangle inside a bubble, avoiding its tail and concavities.

    The inset is dynamic: it scales with the bubble (6% sides, 10% top/bottom
    for guide-like vertical emphasis) but clamps absolute pixels (4–26px sides,
    6–40px top/bottom) so huge bubbles keep readable margins instead of huge
    relative ones, and tiny bubbles keep a usable interior. ``aspect_ratio``
    selects the largest contained rectangle of the requested width/height.
    Pass an explicit
    ``padding`` for the legacy uniform relative inset.

    >>> rect = bubble_inner_rect([[0, 0], [100, 0], [100, 100], [0, 100]])
    >>> rect is not None and rect[2] < 100
    True
    """
    if aspect_ratio is not None and (not math.isfinite(aspect_ratio) or aspect_ratio <= 0):
        raise ValueError('Bubble rectangle aspect ratio must be finite and positive.')
    points = np.asarray(normalize_bubble_polygon(polygon, strict=True), dtype=np.float64)
    origin = points.min(axis=0)
    extent = points.max(axis=0) - origin
    scale = min(1.0, 510.0 / max(extent))
    local = np.round((points - origin) * scale).astype(np.int32) + 1
    width, height = np.ceil(extent * scale).astype(int) + 3
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 1)
    if padding is None:
        # Dynamic: relative to the bubble, bounded in absolute pixels.
        # Concave outlines (shout bubbles, tails) get ~1.5x inset since spike
        # roots intrude closer to the maximal rectangle corners.
        try:
            concave = not bool(
                cv2.isContourConvex(
                    np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
                )
            )
        except cv2.error:
            concave = False
        if concave:
            inset_x = min(max(float(extent[0]) * 0.09, 6.0), 36.0)
            inset_y = min(max(float(extent[1]) * 0.14, 8.0), 54.0)
        else:
            inset_x = min(max(float(extent[0]) * 0.06, 4.0), 26.0)
            inset_y = min(max(float(extent[1]) * 0.10, 6.0), 40.0)
        radius_x = max(1, int(round(inset_x * scale)))
        radius_y = max(1, int(round(inset_y * scale)))
        mask = cv2.erode(
            mask,
            np.ones((2 * radius_y + 1, 2 * radius_x + 1), dtype=np.uint8),
        )
    else:
        if not 0 <= padding < 0.5:
            raise ValueError('Bubble padding must be between 0 and 0.5.')
        radius = max(1, int(round(min(extent) * padding * scale)))
        mask = cv2.erode(mask, np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8))
    heights = np.zeros(width, dtype=int)
    best_area = 0
    best = None
    best_balance = float('inf')
    cx, cy = _polygon_centroid(points)
    centroid_x = (cx - origin[0]) * scale + 1
    centroid_y = (cy - origin[1]) * scale + 1
    for row_index, row in enumerate(mask):
        heights = (heights + 1) * row
        stack = []
        for column in range(width + 1):
            current = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current:
                start, bar_height = stack.pop()
                rect_width, rect_height = column - start, bar_height
                if aspect_ratio is not None:
                    rect_width = min(rect_width, rect_height * aspect_ratio)
                    rect_height = min(rect_height, rect_width / aspect_ratio)
                area = rect_width * rect_height
                # Anchor candidates at the outline centroid: eroded ellipse
                # runs bias left/right asymmetrically, so a centered mask
                # bbox (or scan order) still drifts. Prefer larger area,
                # then closeness to the centroid. Keeps "center in bubble"
                # from nudging already-balanced text.
                left_edge = start + (column - start - rect_width) / 2
                top_edge = row_index - bar_height + 1 + (bar_height - rect_height) / 2
                balance = (left_edge + rect_width / 2 - centroid_x) ** 2 \
                    + (top_edge + rect_height / 2 - centroid_y) ** 2
                # Ladder rungs in this scan tie exactly often; among exact
                # ties pick the candidate closest to the outline centroid so
                # "center in bubble" never shifts balanced text sideways.
                if area > best_area or (
                    area == best_area and balance < best_balance
                ):
                    best_area = area
                    best_balance = balance
                    best = (left_edge, top_edge, rect_width, rect_height)
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    if best is None:
        return None
    left, top, rect_width, rect_height = best
    return (
        float(origin[0] + (left - 1) / scale),
        float(origin[1] + (top - 1) / scale),
        float((rect_width - 1) / scale), float((rect_height - 1) / scale),
    )


def _polygon_centroid(points: np.ndarray) -> Tuple[float, float]:
    """Area centroid of a closed polygon (shoelace formula).

    >>> _polygon_centroid(np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))
    (5.0, 5.0)
    """
    x, y = points[:, 0], points[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        mid = (points.max(axis=0) + points.min(axis=0)) / 2.0
        return float(mid[0]), float(mid[1])
    return (
        float(((x + xn) * cross).sum() / (6.0 * area)),
        float(((y + yn) * cross).sum() / (6.0 * area)),
    )


def bubble_inner_center(
    polygon: Sequence[Sequence[float]],
) -> Optional[Tuple[float, float, Tuple[float, float, float, float]]]:
    """Return the visual bubble center plus its typesetting interior.

    The center is the polygon area centroid (image moments) computed from
    the bubble's own size and shape, so it sits on the main body: a tail or
    spike contributes little area and cannot drag it off the way the eroded
    interior rectangle's center or a bounding-box center would.

    Returns ``(center_x, center_y, rect)`` where ``rect`` is the
    ``bubble_inner_rect`` interior (drawn as the safe-area guide) or the
    bounding box when no eroded interior exists. ``None`` for invalid
    polygons.

    >>> center = bubble_inner_center([[0, 0], [100, 0], [100, 100], [0, 100]])
    >>> center is not None and 40 < center[0] < 60 and 40 < center[1] < 60
    True
    >>> tailed = bubble_inner_center(
    ...     [[0, 0], [100, 0], [200, 50], [100, 100], [0, 100]])
    >>> tailed is not None and 60 < tailed[0] < 90 and 40 < tailed[1] < 60
    True
    >>> bubble_inner_center([[0, 0]]) is None
    True
    """
    normalized = normalize_bubble_polygon(polygon)
    if normalized is None:
        return None
    points = np.asarray(normalized, dtype=np.float64)
    try:
        moments = cv2.moments(points.astype(np.float32))
        area = float(moments['m00'])
    except cv2.error:
        area = 0.0
    if area > 1e-9 and np.isfinite(area):
        center_x, center_y = (
            float(moments['m10'] / area), float(moments['m01'] / area)
        )
    else:
        center_x = center_y = float('nan')
    if not np.isfinite([center_x, center_y]).all():
        # Degenerate (zero-area) outline: fall back to the bounding box.
        left, top = (float(value) for value in points.min(axis=0))
        right, bottom = (float(value) for value in points.max(axis=0))
        if not np.isfinite([left, top, right, bottom]).all():
            return None
        if right <= left or bottom <= top:
            return None
        center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
    try:
        rect = bubble_inner_rect(normalized)
    except ValueError:
        rect = None
    if rect is None or min(rect[2:]) < 1:
        left, top = (float(value) for value in points.min(axis=0))
        right, bottom = (float(value) for value in points.max(axis=0))
        if not np.isfinite([left, top, right, bottom]).all():
            return None
        if right <= left or bottom <= top:
            return None
        rect = (left, top, right - left, bottom - top)
    return (center_x, center_y, rect)


def snap_point_to_bubble_center(
    item_x: float,
    item_y: float,
    bubble_x: float,
    bubble_y: float,
    radius: float = BUBBLE_CENTER_SNAP_RADIUS,
) -> Tuple[float, float, bool]:
    """Snap one axis at a time toward the bubble center.

    Each axis snaps independently when it is within ``radius`` so a drag
    can align horizontally first and then vertically, like a crosshair
    magnet. Holding Alt bypasses this in the caller; this helper only
    decides the point.

    >>> snap_point_to_bubble_center(100.0, 103.0, 112.0, 200.0)
    (112.0, 103.0, True)
    >>> snap_point_to_bubble_center(0.0, 0.0, 100.0, 100.0)
    (0.0, 0.0, False)
    """
    try:
        item_x, item_y = float(item_x), float(item_y)
        bubble_x, bubble_y = float(bubble_x), float(bubble_y)
        radius = float(radius)
    except (TypeError, ValueError):
        return (item_x, item_y, False)
    if radius < 0:
        return (item_x, item_y, False)
    snapped = False
    if abs(bubble_x - item_x) <= radius:
        item_x = bubble_x
        snapped = True
    if abs(bubble_y - item_y) <= radius:
        item_y = bubble_y
        snapped = True
    return (item_x, item_y, snapped)


_MAX_CONTOUR_POINTS = 1024

def _turn(
    previous: Tuple[float, float], current: Tuple[float, float],
    following: Tuple[float, float],
) -> float:
    """Cross product of (current - previous) and (following - current)."""
    return (
        (current[0] - previous[0]) * (following[1] - current[1])
        - (current[1] - previous[1]) * (following[0] - current[0])
    )

def _polygon_area(polygon: Sequence[Tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )

def _polygon_area_from_indices(
    polygon: Sequence[Tuple[float, float]], indices: Sequence[int],
) -> float:
    return 0.5 * sum(
        polygon[indices[index]][0] * polygon[indices[(index + 1) % len(indices)]][1]
        - polygon[indices[(index + 1) % len(indices)]][0] * polygon[indices[index]][1]
        for index in range(len(indices))
    )

def _distance_squared(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2

def _point_segment_distance(
    point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float],
) -> float:
    segment = (end[0] - start[0], end[1] - start[1])
    length_squared = segment[0] ** 2 + segment[1] ** 2
    if length_squared <= sys.float_info.epsilon:
        return math.sqrt(_distance_squared(point, start))
    fraction = (
        (point[0] - start[0]) * segment[0] + (point[1] - start[1]) * segment[1]
    ) / length_squared
    fraction = min(max(fraction, 0.0), 1.0)
    return math.sqrt(_distance_squared(
        point, (start[0] + segment[0] * fraction, start[1] + segment[1] * fraction),
    ))

def _point_on_segment(
    start: Tuple[float, float], end: Tuple[float, float],
    point: Tuple[float, float], epsilon: float,
) -> bool:
    return (
        abs(_turn(start, end, point)) <= epsilon
        and min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )

def _segments_intersect(
    first_start: Tuple[float, float], first_end: Tuple[float, float],
    second_start: Tuple[float, float], second_end: Tuple[float, float],
    epsilon: float,
) -> bool:
    first_side_start = _turn(first_start, first_end, second_start)
    first_side_end = _turn(first_start, first_end, second_end)
    second_side_start = _turn(second_start, second_end, first_start)
    second_side_end = _turn(second_start, second_end, first_end)
    crosses = (
        (first_side_start > epsilon and first_side_end < -epsilon)
        or (first_side_start < -epsilon and first_side_end > epsilon)
    ) and (
        (second_side_start > epsilon and second_side_end < -epsilon)
        or (second_side_start < -epsilon and second_side_end > epsilon)
    )
    return crosses or (
        _point_on_segment(first_start, first_end, second_start, epsilon)
        or _point_on_segment(first_start, first_end, second_end, epsilon)
        or _point_on_segment(second_start, second_end, first_start, epsilon)
        or _point_on_segment(second_start, second_end, first_end, epsilon)
    )

def _point_in_polygon(
    polygon: Sequence[Tuple[float, float]], point: Tuple[float, float], epsilon: float,
) -> bool:
    inside = False
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(first, second, point, epsilon):
            return True
        if (first[1] > point[1]) != (second[1] > point[1]):
            crossing = (
                (second[0] - first[0]) * (point[1] - first[1])
                / (second[1] - first[1]) + first[0]
            )
            if point[0] < crossing:
                inside = not inside
    return inside

def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted(set(points))
    if len(ordered) <= 2:
        return ordered
    lower: List[Tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _turn(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _turn(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]

def _distance_to_polygon_boundary(
    point: Tuple[float, float], polygon: Sequence[Tuple[float, float]],
) -> float:
    if len(polygon) < 2:
        return 0.0
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )

def _simplify_chain_indices(
    polygon: Sequence[Tuple[float, float]], chain: Sequence[int], tolerance: float,
) -> List[int]:
    """Iterative Ramer-Douglas-Peucker over original vertex indices."""
    if len(chain) <= 2:
        return list(chain)
    keep = [False] * len(chain)
    keep[0] = keep[-1] = True
    stack = [(0, len(chain) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        anchor_start, anchor_end = polygon[chain[start]], polygon[chain[end]]
        best_position, best_distance = None, tolerance
        for position in range(start + 1, end):
            distance = _point_segment_distance(
                polygon[chain[position]], anchor_start, anchor_end,
            )
            if distance > best_distance:
                best_position, best_distance = position, distance
        if best_position is None:
            continue
        keep[best_position] = True
        stack.append((start, best_position))
        stack.append((best_position, end))
    return [chain[position] for position in range(len(chain)) if keep[position]]

def _simplify_closed_indices(
    polygon: Sequence[Tuple[float, float]], tolerance: float,
) -> List[int]:
    if len(polygon) <= 2:
        return list(range(len(polygon)))
    split = max(
        range(1, len(polygon)),
        key=lambda index: _distance_squared(polygon[0], polygon[index]),
    )
    first = _simplify_chain_indices(polygon, list(range(split + 1)), tolerance)
    second = _simplify_chain_indices(
        polygon, list(range(split, len(polygon))) + [0], tolerance,
    )
    combined = first[:-1] + second[:-1]
    return combined if len(combined) >= 3 else list(range(len(polygon)))

def _valid_diagonal(
    polygon: Sequence[Tuple[float, float]], first: int, second: int, tolerance: float,
) -> bool:
    length = len(polygon)
    if first == second or (first + 1) % length == second or (second + 1) % length == first:
        return False
    start, end = polygon[first], polygon[second]
    epsilon = max(tolerance * tolerance * 0.001, sys.float_info.epsilon)
    for edge in range(length):
        following = (edge + 1) % length
        if edge in (first, second) or following in (first, second):
            continue
        if _segments_intersect(start, end, polygon[edge], polygon[following], epsilon):
            return False
    return all(
        _point_in_polygon(
            polygon,
            (start[0] + (end[0] - start[0]) * fraction,
             start[1] + (end[1] - start[1]) * fraction),
            epsilon,
        )
        for fraction in (0.2, 0.5, 0.8)
    )

def _split_polygon(
    polygon: List[Tuple[float, float]], first: int, second: int,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    if first > second:
        first, second = second, first
    return (
        polygon[first:second + 1],
        polygon[second:] + polygon[:first + 1],
    )

def _clip_half_plane(
    polygon: Sequence[Tuple[float, float]],
    normal: Tuple[float, float],
    offset: float,
) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman clip of ``polygon`` to ``dot(point, normal) <= offset``.

    1:1 port of koharu ``clip_half_plane`` (crates/koharu-renderer/src/bubble.rs);
    keeps points on the boundary (``<= epsilon``) and emits edge intersections.
    """
    if not polygon:
        return []
    epsilon = sys.float_info.epsilon

    def signed(point: Tuple[float, float]) -> float:
        return point[0] * normal[0] + point[1] * normal[1] - offset

    output: List[Tuple[float, float]] = []
    previous = polygon[-1]
    previous_distance = signed(previous)
    for current in polygon:
        current_distance = signed(current)
        previous_inside = previous_distance <= epsilon
        current_inside = current_distance <= epsilon
        if previous_inside != current_inside:
            denominator = previous_distance - current_distance
            if abs(denominator) > epsilon:
                fraction = previous_distance / denominator
                output.append((
                    previous[0] + (current[0] - previous[0]) * fraction,
                    previous[1] + (current[1] - previous[1]) * fraction,
                ))
        if current_inside:
            output.append(current)
        previous = current
        previous_distance = current_distance
    return output

def _anchor_flow_cells(
    width: float,
    height: float,
    contour: Sequence[Tuple[float, float]],
    anchors: Sequence[Tuple[float, float]],
) -> List[List[Tuple[float, float]]]:
    """Partition a contour by perpendicular bisectors between the anchors.

    1:1 port of koharu ``anchor_flow_cells``: anchors closer than 0.25% of the
    frame's smaller side join one cluster (running-mean site); every cluster
    cell starts as the physical contour and is clipped by the bisector against
    each sibling cluster; a cluster holding several anchors is subdivided into
    equal strips along its longer axis. Returns one cell per anchor in input
    order; degenerate cells come back empty like the upstream Rust.
    """
    scale = max(min(width, height), 1.0)
    coincidence_distance_squared = (scale * 0.0025) ** 2
    clusters: List[Tuple[List[float], List[int]]] = []
    for index, anchor in enumerate(anchors):
        for site, indices in clusters:
            dx = anchor[0] - site[0]
            dy = anchor[1] - site[1]
            if dx * dx + dy * dy <= coincidence_distance_squared:
                count = len(indices)
                site[0] = (site[0] * count + anchor[0]) / (count + 1)
                site[1] = (site[1] * count + anchor[1]) / (count + 1)
                indices.append(index)
                break
        else:
            clusters.append(([anchor[0], anchor[1]], [index]))

    cells: List[List[Tuple[float, float]]] = [[] for _ in anchors]
    for cluster_index, (site, indices) in enumerate(clusters):
        # The cell is the editable balloon shape, so fallback bisectors must
        # clip the physical contour rather than just its bounding rectangle.
        cell = (
            list(contour) if len(contour) >= 3
            else [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        )
        for other_index, (other, _) in enumerate(clusters):
            if cluster_index == other_index:
                continue
            normal = (other[0] - site[0], other[1] - site[1])
            offset = (
                other[0] * other[0] + other[1] * other[1]
                - site[0] * site[0] - site[1] * site[1]
            ) * 0.5
            cell = _clip_half_plane(cell, normal, offset)
            if len(cell) < 3:
                break
        if len(indices) == 1:
            cells[indices[0]] = cell
            continue
        # Match the upstream fold so an emptied cell degrades to empty strips
        # instead of raising on min()/max() over no points.
        min_x = min_y = math.inf
        max_x = max_y = -math.inf
        for x, y in cell:
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
        horizontal = max_x - min_x >= max_y - min_y
        minimum, maximum = (min_x, max_x) if horizontal else (min_y, max_y)
        step = (maximum - minimum) / len(indices)
        for order, index in enumerate(indices):
            lower = minimum + step * order
            upper = maximum if order + 1 == len(indices) else lower + step
            lower_normal, upper_normal = (
                ((-1.0, 0.0), (1.0, 0.0)) if horizontal
                else ((0.0, -1.0), (0.0, 1.0))
            )
            strip = _clip_half_plane(cell, lower_normal, -lower)
            cells[index] = _clip_half_plane(strip, upper_normal, upper)
    return cells

def _decompose_lobes(
    polygon: List[Tuple[float, float]],
    anchors: List[Tuple[int, Tuple[float, float]]],
    tolerance: float,
    shared_lobes: bool = False,
) -> Optional[List[Tuple[int, List[Tuple[float, float]]]]]:
    """Split at necks; detector runs may share an unsplittable lobe."""
    if len(anchors) == 1:
        return [(anchors[0][0], polygon)]
    simplified = _simplify_closed_indices(polygon, tolerance)
    if len(simplified) < 4:
        return [(index, polygon) for index, _ in anchors] if shared_lobes else None
    area = _polygon_area_from_indices(polygon, simplified)
    if area == 0.0:
        return None
    orientation = 1.0 if area > 0.0 else -1.0
    hull = _convex_hull([polygon[index] for index in simplified])
    minimum_reflex_cross = tolerance * tolerance * 0.05
    reflex: List[Tuple[int, float]] = []
    for index in range(len(simplified)):
        previous = polygon[simplified[index - 1]]
        current = polygon[simplified[index]]
        following = polygon[simplified[(index + 1) % len(simplified)]]
        cross = _turn(previous, current, following) * orientation
        if cross >= -minimum_reflex_cross:
            continue
        edge_product = math.sqrt(
            _distance_squared(previous, current) * _distance_squared(current, following)
        )
        if edge_product <= sys.float_info.epsilon:
            continue
        # A structural neck connects sharp corners that are recessed from the
        # convex envelope; recession makes outer-wall raster kinks contribute
        # almost no support without a scale-specific rejection threshold.
        recession = _distance_to_polygon_boundary(current, hull)
        reflex.append((simplified[index], -cross / edge_product * recession))
    cross_epsilon = max(tolerance * tolerance * 0.001, sys.float_info.epsilon)
    candidates: List[Tuple[float, float, List, List, List, List]] = []
    for first_index in range(len(reflex)):
        for second_index in range(first_index + 1, len(reflex)):
            first_vertex, first_support = reflex[first_index]
            second_vertex, second_support = reflex[second_index]
            if not _valid_diagonal(polygon, first_vertex, second_vertex, tolerance):
                continue
            first_part, second_part = _split_polygon(polygon, first_vertex, second_vertex)
            if (
                len(first_part) < 3 or len(second_part) < 3
                or abs(_polygon_area(first_part)) <= tolerance * tolerance
                or abs(_polygon_area(second_part)) <= tolerance * tolerance
            ):
                continue
            first_anchors: List[Tuple[int, Tuple[float, float]]] = []
            second_anchors: List[Tuple[int, Tuple[float, float]]] = []
            assigns_cleanly = True
            for anchor in anchors:
                in_first = _point_in_polygon(first_part, anchor[1], cross_epsilon)
                in_second = _point_in_polygon(second_part, anchor[1], cross_epsilon)
                if in_first and not in_second:
                    first_anchors.append(anchor)
                elif in_second and not in_first:
                    second_anchors.append(anchor)
                else:
                    assigns_cleanly = False
                    break
            if not assigns_cleanly or not first_anchors or not second_anchors:
                continue
            length_squared = _distance_squared(
                polygon[first_vertex], polygon[second_vertex]
            )
            if length_squared <= sys.float_info.epsilon:
                continue
            candidates.append((
                min(first_support, second_support) / math.sqrt(length_squared),
                length_squared, first_part, second_part, first_anchors, second_anchors,
            ))
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    for _, _, first_part, second_part, first_anchors, second_anchors in candidates:
        first_cells = _decompose_lobes(first_part, first_anchors, tolerance, shared_lobes)
        if first_cells is None:
            continue
        second_cells = _decompose_lobes(second_part, second_anchors, tolerance, shared_lobes)
        if second_cells is None:
            continue
        return first_cells + second_cells
    # Detector runs in one smooth lobe stay together; fitting still requires
    # one cell per flow and falls back to Voronoi when necks cannot provide it.
    return [(index, polygon) for index, _ in anchors] if shared_lobes else None

def split_connected_bubble(
    polygon: Sequence[Sequence[float]],
    centers: Sequence[Sequence[float]],
) -> Optional[List[List[List[float]]]]:
    """Split one connected outline into one polygon per shared text block.

    Reflex-vertex necks recover the physical lobes first (koharu
    ``topological_flow_cells``). When the outline has no clean neck
    decomposition (smooth shared ovals, centers sharing one lobe), the physical
    contour is partitioned by perpendicular bisectors between the sibling
    centers — koharu ``anchor_flow_cells`` — with coincident centers
    subdivided into strips. Returns one polygon per input center in input
    order, or ``None`` only for malformed geometry or when a cell degenerates,
    so callers fall back to per-block mask layout.

    >>> top = [[0, 0], [100, 0], [100, 100], [40, 100], [40, 140], [100, 140], [100, 240], [0, 240], [0, 140], [60, 140], [60, 100], [0, 100]]
    >>> lobes = split_connected_bubble(top, [(50.0, 50.0), (50.0, 190.0)])
    >>> lobes is not None and len(lobes) == 2
    True
    >>> rect = [[0, 0], [200, 0], [200, 120], [0, 120]]
    >>> cells = split_connected_bubble(rect, [(50.0, 60.0), (150.0, 60.0)])
    >>> all(
    ...     _point_in_polygon([(point[0], point[1]) for point in cell], center, 1e-9)
    ...     for cell, center in zip(cells, [(50.0, 60.0), (150.0, 60.0)])
    ... )
    True
    """
    return _partition_connected_bubble(polygon, centers)

def bubble_partition_cells(
    polygon: Sequence[Sequence[float]],
    centers: Sequence[Sequence[float]],
) -> Optional[List[List[List[float]]]]:
    """Split a joined outline at its physical necks only; one cell per center.

    Unlike :func:`split_connected_bubble` there is no Voronoi fallback: a
    single-lobe outline (no reflex necks) returns ``None``, so callers can
    tell a genuinely joined multi-lobe balloon from one smooth balloon and
    keep its text runs together. Centers within one lobe receive equal cells.

    >>> top = [[0, 0], [100, 0], [100, 100], [40, 100], [40, 140], [100, 140], [100, 240], [0, 240], [0, 140], [60, 140], [60, 100], [0, 100]]
    >>> lobes = bubble_partition_cells(top, [(50.0, 50.0), (30.0, 190.0), (70.0, 190.0)])
    >>> lobes is not None and lobes[0] != lobes[1] == lobes[2]
    True
    >>> bubble_partition_cells([[0, 0], [200, 0], [200, 120], [0, 120]], [(50.0, 60.0), (150.0, 60.0)]) is None
    True
    """
    prepared = _prepare_bubble_partition(polygon, centers)
    if prepared is None:
        return None
    poly, tight, scale = prepared
    cells = _decompose_lobes(poly, list(enumerate(tight)), scale * 0.0075, shared_lobes=True)
    if cells is None or all(cell == poly for _, cell in cells):
        return None
    cells.sort(key=lambda cell: cell[0])
    ordered: List[List[List[float]]] = []
    for _, cell in cells:
        normalized = normalize_bubble_polygon(cell)
        if normalized is None:
            return None
        ordered.append(normalized)
    return ordered

def _prepare_bubble_partition(
    polygon: Sequence[Sequence[float]],
    centers: Sequence[Sequence[float]],
) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float]]:
    try:
        points = np.asarray(
            normalize_bubble_polygon(polygon, strict=True), dtype=np.float64
        )
    except ValueError:
        return None
    if len(centers) < 2 or not 4 <= len(points) <= _MAX_CONTOUR_POINTS:
        return None
    try:
        tight = [(float(center[0]), float(center[1])) for center in centers]
    except (TypeError, ValueError, IndexError):
        return None
    # Non-finite positions would poison every geometric predicate below; the
    # documented failure mode is None, not an exception.
    if not np.isfinite(np.asarray(tight, dtype=np.float64)).all():
        return None
    poly = [(float(point[0]), float(point[1])) for point in points]
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # koharu flow_cells clamps anchors into the frame before partitioning, so
    # a center dragged outside the outline still anchors its nearest region.
    tight = [
        (min(max(x, min_x), max_x), min(max(y, min_y), max_y))
        for x, y in tight
    ]
    scale = max(min(max_x - min_x, max_y - min_y), 1.0)
    return poly, tight, scale

def _partition_connected_bubble(
    polygon: Sequence[Sequence[float]],
    centers: Sequence[Sequence[float]],
) -> Optional[List[List[List[float]]]]:
    prepared = _prepare_bubble_partition(polygon, centers)
    if prepared is None:
        return None
    poly, tight, scale = prepared
    xs = [point[0] for point in poly]
    ys = [point[1] for point in poly]
    cells = _decompose_lobes(poly, list(enumerate(tight)), scale * 0.0075)
    if cells is None:
        # No clean neck decomposition: fall back to the koharu Voronoi
        # partition of the physical contour instead of failing the fit.
        fallback = _anchor_flow_cells(max(xs) - min(xs), max(ys) - min(ys), poly, tight)
        cells = list(enumerate(fallback))
    cells.sort(key=lambda cell: cell[0])
    if len(cells) != len(tight):
        return None
    ordered: List[List[List[float]]] = []
    for _, cell in cells:
        normalized = normalize_bubble_polygon(cell)
        if normalized is None:
            return None
        ordered.append(normalized)
    return ordered
