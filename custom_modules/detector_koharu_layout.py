"""KoharuLayout-RFDETR-Seg-2XL-1152 custom text detector.

Model: https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152
"""
import logging
import math
import os
import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ballontranslator.modules.textdetector.base import (
    DEVICE_SELECTOR,
    ProjImgTrans,
    TextBlock,
    TextDetectorBase,
    register_textdetectors,
)
from ballontranslator.utils.imgproc_utils import xywh2xyxypoly
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.textblock import (
    examine_textblk,
    mit_merge_textlines,
    sort_pnts,
    sort_regions,
)

MODEL_PATH = 'data/models/koharu_layout/model.safetensors'
CLASS_NAMES = ['text', 'onomatopoeia', 'bubble', 'panel']


def _mask_outline(det_mask: np.ndarray) -> Optional[List]:
    """Largest simplified outer contour of one instance mask, as int points.

    Falls back to the caller's box when the mask is empty or too small to
    form a polygon.

    >>> import numpy as np
    >>> _mask_outline(np.zeros((8, 8), np.uint8)) is None
    True
    >>> mask = np.zeros((64, 64), np.uint8)
    >>> mask[10:50, 10:50] = 1
    >>> poly = _mask_outline(mask)
    >>> len(poly) >= 3 and all(len(pt) == 2 for pt in poly)
    True
    """
    contours, _ = cv2.findContours(
        det_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 2.0, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return approx.tolist()


def _containment_ratio(box_a: List, box_b: List) -> float:
    """Overlap over the smaller box's area (1.0 when one contains the other).

    >>> _containment_ratio([0, 0, 10, 10], [5, 5, 15, 15])
    0.25
    >>> _containment_ratio([0, 0, 10, 10], [2, 2, 8, 8])
    1.0
    """
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter = (
        max(0.0, min(ax1, bx1) - max(ax0, bx0))
        * max(0.0, min(ay1, by1) - max(ay0, by0))
    )
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    if inter <= 0 or area_a <= 0 or area_b <= 0:
        return 0.0
    return inter / min(area_a, area_b)


def _selected_bubble_instances(candidates: List[dict], threshold: float) -> List[dict]:
    """Pick one instance per nested cluster: strong, or weak-but-agreed.

    Complex balloons score below the model-card threshold while the head
    fires several nested instances of them; a lone weak instance stays
    rejected, a consistent stack is real. The highest-confidence member
    represents the cluster because its segmentation mask is the tightest
    - weaker instances paint noise strips around the balloon.

    >>> cands = [{'box': [0, 0, 10, 10], 'conf': 0.7},
    ...          {'box': [1, 1, 9, 9], 'conf': 0.3},
    ...          {'box': [50, 50, 60, 60], 'conf': 0.3}]
    >>> [c['conf'] for c in _selected_bubble_instances(cands, 0.5)]
    [0.7]
    """
    clusters: List[List[int]] = []
    for i, cand in enumerate(candidates):
        for group in clusters:
            if any(
                _containment_ratio(cand['box'], candidates[j]['box']) >= 0.6
                for j in group
            ):
                group.append(i)
                break
        else:
            clusters.append([i])
    selected = []
    for group in clusters:
        best = max(group, key=lambda j: candidates[j]['conf'])
        if candidates[best]['conf'] >= threshold or len(group) >= 2:
            selected.append(candidates[best])
    return selected


def _split_two_lobed(outline: List) -> Optional[List[List]]:
    """Cut a joined two-bubble contour at its neck into two closed polygons.

    Touching bubbles merge into one detector instance; cutting the contour
    between its two reflex neck vertices gives one polygon per bubble, each
    closed by the straight chord - the separating line drawn on the canvas.
    Single bubbles (no reflex pair) and shapes without one bubble-sized cut
    stay intact.

    >>> import cv2
    >>> a = cv2.ellipse2Poly((30, 40), (30, 30), 0, 0, 360, 30)
    >>> b = cv2.ellipse2Poly((70, 40), (30, 30), 0, 0, 360, 30)
    >>> m = np.zeros((80, 110), np.uint8)
    >>> _ = cv2.fillPoly(m, [a, b], 255)
    >>> cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    >>> parts = _split_two_lobed(cnts[0].reshape(-1, 2).tolist())
    >>> len(parts)
    2
    >>> _split_two_lobed(a.reshape(-1, 2).tolist()) is None
    True
    """
    pts = np.asarray(outline, np.float32)
    n = len(pts)
    if n < 6:
        return None
    total = abs(cv2.contourArea(pts))
    if total < 1:
        return None
    # Orientation sign: for a CCW contour a convex vertex turns positive.
    x, y = pts[:, 0], pts[:, 1]
    orient = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if orient == 0:
        return None
    reflex = []
    for k in range(n):
        a, b, c = pts[k - 1], pts[k], pts[(k + 1) % n]
        cross = float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        reflex.append(cross * orient < 0)
    # Chord cap relative to the joined outline's size. The cap alone cannot
    # tell a neck from a mid-body slice: true necks over the stored set
    # measure up to 0.97*sqrt(area), and so do slices through a single
    # balloon. Balance and compactness separate them - true necks free two
    # comparable roundish lobes (balance 0.67-0.99, compact 0.6-0.97), a
    # false slice frees an unbalanced (0.48) or flat (<=0.55) piece. So a
    # balanced roundish cut earns the wide cap; anything else keeps the
    # tight cap that rejects the observed mid-body slices.
    tight = 0.79 * float(np.sqrt(total))
    wide = 0.99 * float(np.sqrt(total))
    best = None
    for i in range(n - 2):
        if not reflex[i]:
            continue
        for j in range(i + 2, n):
            if not reflex[j] or (i == 0 and j == n - 1):
                continue
            a, b = pts[i], pts[j]
            chord = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if chord > wide:
                continue
            if cv2.pointPolygonTest(pts, (float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)), False) < 0:
                continue
            part1 = pts[i:j + 1]
            part2 = np.vstack([pts[j:], pts[:i + 1]])
            # Tiny stacked bubbles can be a few percent of the joined area
            # (a small bubble under a big one); the reflex-pair and chord
            # gates keep plain bubbles from shedding noise bumps, so only the
            # minority floor needs to stay low.
            min_part = 250.0
            area1 = abs(cv2.contourArea(part1))
            area2 = abs(cv2.contourArea(part2))
            if area1 < min_part or area2 < min_part:
                continue
            minority = part1 if area1 <= area2 else part2
            marr = np.asarray(minority, np.float32)
            _, _, mw, mh = cv2.boundingRect(marr)
            # A real small bubble fills its box; a crescent shaved off a
            # noise dent does not (measured 0.17-0.28 vs 0.4+ for blobs).
            compact = min(area1, area2) / max(mw * mh, 1)
            if compact < 0.35:
                continue
            # A chord far wider than the lobe it frees slices a balloon
            # between noise dents, not a neck (chord / sqrt(minority area)):
            # the verified pairs measure <= 1.37, slices on unverified or
            # false cuts 1.43+. Deterministic geometry keeps true cuts at
            # their measured value across runs.
            if chord > 1.4 * float(np.sqrt(min(area1, area2))):
                continue
            balance = min(area1, area2) / max(area1, area2)
            if chord > (wide if (balance >= 0.6 and compact >= 0.6) else tight):
                continue
            # Prefer the cut freeing the roundest lobe: a true neck frees a
            # blob (0.4-0.9), noise dents free flatter pieces, and ranking by
            # chord length lets a tiny dent win over the real neck.
            key = (compact, -chord)
            if best is None or key > best[0]:
                best = (key, part1, part2)
    if best is None:
        return None
    return [best[1].astype(int).tolist(), best[2].astype(int).tolist()]


def _outline_boundary(a: np.ndarray, b: np.ndarray):
    """Cut line between two drawn balloons that abut or overlap.

    Adjacent balloons stop at each other's stroke instead of crossing,
    so there is no contour intersection to use; the cut runs through the
    closest approach of the two contours, perpendicular to it - along
    the contact line itself. The line merger glues boxes straddling that
    contact, putting two balloons' speech in one block; this chord is
    what the seam cut separates them by.
    """
    pa = np.asarray(a, np.float32)
    pb = np.asarray(b, np.float32)
    # closest pair of vertices (balloons are ~50 points; naive is fine)
    best_d2, best_p, best_q = None, None, None
    for p in pa:
        d2 = ((pb - p) ** 2).sum(axis=1)
        k = int(np.argmin(d2))
        if best_d2 is None or d2[k] < best_d2:
            best_d2, best_p, best_q = float(d2[k]), p, pb[k]
    if best_p is None:
        return None
    # Direction of the contact: average of the two contours' local
    # tangents at the meeting points. Centroid axes lie when one balloon
    # sits lower than the other, and the gap vector points across the
    # contact, not along it.
    i = int(np.argmin(((pa - best_p) ** 2).sum(axis=1)))
    j = int(np.argmin(((pb - best_q) ** 2).sum(axis=1)))
    ta = pa[(i + 1) % len(pa)] - pa[i]
    tb = pb[(j + 1) % len(pb)] - pb[j]
    na_, nb_ = float(np.hypot(ta[0], ta[1])), float(np.hypot(tb[0], tb[1]))
    if na_ < 1e-6 or nb_ < 1e-6:
        return None
    ta, tb = ta / na_, tb / nb_
    if float(np.dot(ta, tb)) < 0:
        tb = -tb                            # contours may wind opposite
    dvec = (ta + tb) / 2
    nvec = float(np.hypot(dvec[0], dvec[1]))
    if nvec < 1e-6:
        return None
    px, py = float(dvec[0] / nvec), float(dvec[1] / nvec)
    mx, my = (best_p[0] + best_q[0]) / 2, (best_p[1] + best_q[1]) / 2
    # length: the bbox-overlap extent projected on the contact direction
    ax, ay, aw, ah = cv2.boundingRect(pa)
    bx, by, bw, bh = cv2.boundingRect(pb)
    ox0, oy0 = max(ax, bx), max(ay, by)
    ox1, oy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    corners = np.array([[ox0, oy0], [ox1, oy0], [ox1, oy1], [ox0, oy1]], np.float32)
    proj = corners @ np.array([px, py], np.float32)
    length = float(proj.max() - proj.min())
    if length < 8:
        return None
    half = length / 2
    return ((float(mx - px * half), float(my - py * half)),
            (float(mx + px * half), float(my + py * half)))


def _cut_rect(x1, y1, x2, y2, a, b):
    """Split an axis-aligned box across the seam line through points a, b.

    Returns two xywh halves (split along the seam's dominant axis), or None
    when the seam misses the box, grazes a corner, or would leave a sliver.
    A seam shorter than the box along the cut axis cannot center a cut, so
    the midline must fall between the seam's own endpoints.

    >>> halves = _cut_rect(10, 10, 90, 50, (50, 0), (50, 60))
    >>> [[int(v) for v in h] for h in halves]
    [[10, 10, 40, 40], [50, 10, 40, 40]]
    >>> _cut_rect(10, 10, 90, 50, (0, 0), (5, 5)) is None
    True
    >>> _cut_rect(10, 10, 90, 50, (50, 0), (50, 12)) is None
    True
    """
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    if abs(dy) >= abs(dx):
        # Near-vertical seam: cut x at the box's vertical middle, but only
        # when the seam actually spans that height.
        mid_y = (y1 + y2) / 2
        if not min(ay, ay + dy) <= mid_y <= max(ay, ay + dy) or dy == 0:
            return None
        cut = ax + dx * ((mid_y - ay) / dy)
        # Each half must keep a real share of the box. Overlapping bubbles
        # put the seam next to text that belongs to one side, and a fixed
        # pixel sliver survives it: a 38px column cut from a 185px box
        # (20% here) duplicated its neighbour's last word and left 55% of
        # its glyphs outside the detector's text mask, so they never got
        # inpainted.
        if not (x1 + (x2 - x1) * 0.25 < cut < x2 - (x2 - x1) * 0.25):
            return None
        return ([x1, y1, cut - x1, y2 - y1], [cut, y1, x2 - cut, y2 - y1])
    mid_x = (x1 + x2) / 2
    if not min(ax, ax + dx) <= mid_x <= max(ax, ax + dx) or dx == 0:
        return None
    cut = ay + dy * ((mid_x - ax) / dx)
    if not (y1 + (y2 - y1) * 0.25 < cut < y2 - (y2 - y1) * 0.25):
        return None
    return ([x1, y1, x2 - x1, cut - y1], [x1, cut, x2 - x1, y2 - cut])


def _seam_split_blocks(blk_list: List, seams: List, im_w: int, im_h: int,
                        outlines: List = ()) -> List:
    """Cut blocks straddling a split pair's seam, one block per bubble.

    Two joined bubbles each carry their own text, but the text head and the
    line merger both treat the pair as one run. The seam chord runs between
    the two texts, so cutting the block's box along the extended chord gives
    each bubble its own block for OCR. A half whose box misses its own bubble
    (slop into the neighbour) keeps the original block instead of minting an
    empty one.

    >>> class B:
    ...     def __init__(self, x, y, w, h):
    ...         self._b = (x, y, w, h); self.label = 'text'
    ...     def bounding_rect(self): return self._b
    >>> left = [[540, 1400], [630, 1400], [630, 1600], [540, 1600]]
    >>> right = [[630, 1420], [690, 1420], [690, 1560], [630, 1560]]
    >>> out = _seam_split_blocks([B(586, 1457, 77, 127)],
    ...     [((627, 1447), (637, 1551), [left, right])], 1500, 2000)
    >>> len(out), [list(b.bounding_rect()) for b in out]
    (2, [[586, 1457, 48, 127], [634, 1457, 29, 127]])
    """
    if not seams:
        return blk_list
    # The keep/drop test below is outline overlap, so the per-part side of
    # the seam is never needed.
    prepared = [(a, b) for a, b, _parts in seams]
    outline_boxes = []
    for arr in outlines:
        bx, by, bw, bh = cv2.boundingRect(np.asarray(arr, np.float32))
        outline_boxes.append((bx, by, bx + bw, by + bh))
    out = []
    for blk in blk_list:
        boxes = [blk]
        for a, b in prepared:
            next_boxes = []
            for box in boxes:
                x, y, w, h = box.bounding_rect()
                halves = _cut_rect(float(x), float(y), float(x + w), float(y + h), a, b)
                if halves is None:
                    next_boxes.append(box)
                    continue
                kept = []
                for hx, hy, hw, hh in halves:
                    # A half that no balloon reaches would mint an empty
                    # block. Overlap with any outline, not just the seam's
                    # pair: a cut between two balloons often frees a third
                    # balloon's run, and glued boxes extend past their own
                    # bubble's edge, so corner or centre tests on the seam
                    # pair alone reject real cuts.
                    if outline_boxes and not any(
                        min(hx + hw, bx2) - max(hx, bx0) >= 10
                        and min(hy + hh, by2) - max(hy, by0) >= 10
                        for bx0, by0, bx2, by2 in outline_boxes
                    ):
                        kept = None
                        break
                    kept.append((hx, hy, hw, hh))
                if kept is None:
                    next_boxes.append(box)
                    continue
                for xywh in kept:
                    pts = xywh2xyxypoly(np.array([xywh])).reshape(4, 2).tolist()
                    pts_sorted, is_vertical = sort_pnts(pts)
                    nb = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=box.label)
                    nb.vertical = is_vertical
                    nb.adjust_bbox()
                    examine_textblk(nb, im_w, im_h)
                    next_boxes.append(nb)
            boxes = next_boxes
        out.extend(boxes)
    return out


def _unmasked_ink_fraction(gray: np.ndarray, mask: np.ndarray, box) -> Optional[float]:
    """Share of a block box's source ink that the inpaint mask covers.

    ``None`` when the box holds too little ink to judge (a genuine
    punctuation-only bubble). Ink is measured against the box's own median
    so bubble strokes and panel borders are not counted as text.

    A block whose ink mostly escapes the mask is a phantom: a box widened
    past its own detection leaves glyphs that nothing will inpaint.

    >>> g = np.full((40, 40), 240, np.uint8)
    >>> g[10:30, 10:20] = 20
    >>> on_ink = np.zeros((40, 40), np.uint8)
    >>> on_ink[10:30, 10:20] = 255
    >>> round(_unmasked_ink_fraction(g, on_ink, (0, 0, 40, 40)), 2)
    1.0
    >>> off_ink = np.zeros((40, 40), np.uint8)
    >>> off_ink[10:30, 30:40] = 255
    >>> round(_unmasked_ink_fraction(g, off_ink, (0, 0, 40, 40)), 2)
    0.0
    """
    x, y, w, h = box
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = gray[y0:y1, x0:x1]
    ink = patch < float(np.median(patch)) - 30
    total = int(ink.sum())
    if total < 30:
        return None
    return float((ink & (mask[y0:y1, x0:x1] > 0)).sum()) / total


def _drop_contained_detections(items: List[dict]) -> List[dict]:
    """Drop the smaller of two boxes when one mostly sits inside the other.

    rf-detr fires duplicate detections for a text run crossing a joined
    bubble's seam (prefix/full/tail boxes of the same line); one text run
    must become one block. Separate bubbles keep tens of px of gap, so their
    boxes never reach the containment threshold.

    >>> full = {'pts': [[629, 96], [819, 96], [819, 291], [629, 291]]}
    >>> pre = {'pts': [[693, 102], [819, 102], [819, 294], [693, 294]]}
    >>> tail = {'pts': [[628, 102], [680, 102], [680, 248], [628, 248]]}
    >>> other = {'pts': [[394, 228], [580, 228], [580, 372], [394, 372]]}
    >>> [it['pts'][0] for it in _drop_contained_detections([full, pre, tail, other])]
    [[629, 96], [394, 228]]
    """
    if len(items) < 2:
        return items
    boxes = []
    for it in items:
        pts = np.asarray(it['pts'], np.float32)
        boxes.append((
            float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()),
        ))
    dropped = [False] * len(items)
    for i in range(len(items)):
        if dropped[i]:
            continue
        ax0, ay0, ax1, ay1 = boxes[i]
        area_i = (ax1 - ax0) * (ay1 - ay0)
        for j in range(i + 1, len(items)):
            if dropped[i] or dropped[j]:
                continue
            if _containment_ratio(boxes[i], boxes[j]) < 0.6:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            area_j = (bx1 - bx0) * (by1 - by0)
            # Keep the larger box (the full text run), earlier on ties.
            if area_j < area_i:
                dropped[j] = True
            elif area_i < area_j:
                dropped[i] = True
            else:
                dropped[j] = True
    return [it for k, it in enumerate(items) if not dropped[k]]


def _nearby_mask_extent(
    det_mask: Optional[np.ndarray], box: Tuple[int, int, int, int],
    others: Tuple = (),
) -> Optional[Tuple[int, int, int, int]]:
    """Extent of the mask components plausibly belonging to one detection.

    Only fragments near this detection may enlarge its block box. The seg
    head emits stray pixels far from its own box, and a full-mask bbox then
    drags the block across unrelated bubbles: one page of a real chapter
    carries a single mask pixel 970 px above its detection, which stretched a
    bottom-row block all the way into the top panel. Components further away
    than the detection's own size are someone else's text; components within
    that slack still count, because the mask head routinely spills a few dozen
    pixels outside the box head.

    Components whose centre sits on another detection belong to that
    detection, not this one - two joined bubbles then stay two boxes.

    >>> import numpy as np
    >>> mask = np.zeros((64, 64), bool)
    >>> mask[10:20, 10:20] = True
    >>> _nearby_mask_extent(mask, (8, 8, 22, 22))
    (10, 10, 20, 20)
    >>> mask[63, 63] = True
    >>> _nearby_mask_extent(mask, (8, 8, 22, 22))
    (10, 10, 20, 20)
    >>> _nearby_mask_extent(None, (0, 0, 4, 4)) is None
    True
    >>> spill = np.zeros((64, 64), bool)
    >>> spill[10:20, 10:20] = True
    >>> spill[40:50, 40:50] = True
    >>> _nearby_mask_extent(spill, (8, 8, 22, 22), [(36, 36, 54, 54)])
    (10, 10, 20, 20)
    """
    if det_mask is None or not det_mask.any():
        return None
    x1, y1, x2, y2 = box
    slack = max(x2 - x1, y2 - y1)
    # Each per-instance mask is full-page (the caller indexes it into a
    # page mask), so running connected components over the whole page costs
    # O(page area) once per text box. Every component of a mask lies inside
    # the mask's own bounding box, so scan only that box: exact (no component
    # is ever cut) and small, because a text mask's extent is roughly its
    # detection box. A mask carrying a far stray fragment has a large bbox
    # and falls back to near-full-page cost - the rare, already-slow case.
    mask_u8 = det_mask.astype(np.uint8)
    bx0, by0, bw0, bh0 = cv2.boundingRect(mask_u8)
    if bw0 == 0 or bh0 == 0:
        return None
    window = mask_u8[by0:by0 + bh0, bx0:bx0 + bw0]
    count, _, stats, _ = cv2.connectedComponentsWithStats(window, 8)
    extent = None
    for idx in range(1, count):
        left = int(stats[idx, cv2.CC_STAT_LEFT]) + bx0
        top = int(stats[idx, cv2.CC_STAT_TOP]) + by0
        right = left + int(stats[idx, cv2.CC_STAT_WIDTH])
        bottom = top + int(stats[idx, cv2.CC_STAT_HEIGHT])
        dx = max(0, left - x2, x1 - right)
        dy = max(0, top - y2, y1 - bottom)
        if max(dx, dy) > slack:
            continue
        if others:
            # Seg masks bleed across to the neighbouring bubble's glyphs.
            # Claiming that component stretches this box over the neighbour,
            # the containment dedup reads the overlap as one duplicate run,
            # and the smaller bubble's text is dropped - the bubble then
            # never reaches OCR at all.
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            if any(
                x1o <= cx <= x2o and y1o <= cy <= y2o
                for x1o, y1o, x2o, y2o in others
            ):
                continue
        if extent is None:
            extent = [left, top, right, bottom]
        else:
            extent[0] = min(extent[0], left)
            extent[1] = min(extent[1], top)
            extent[2] = max(extent[2], right)
            extent[3] = max(extent[3], bottom)
    if extent is None:
        return None
    return tuple(extent)


class _QuietRFDETRNoise(logging.Filter):
    """Drop rfdetr warnings that are always true for this integration.

    The backbone intentionally rebuilds position embeddings for the 1152/12
    grid with patch size 12 (hence the two dinov2 warnings), and
    optimize_for_inference() is skipped because it torch.jit.traces a 1152
    model whose control flow the tracer cannot follow cleanly.
    """

    _PREFIXES = (
        'Using a different number of positional encodings',
        'Using patch size',
        'Model is not optimized for inference',
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._PREFIXES)


@register_textdetectors('koharu_layout')
class KoharuLayoutDetector(TextDetectorBase):
    """RF-DETR Seg 2XL manga layout detector (text + onomatopoeia classes).

    Example:
        >>> detector = KoharuLayoutDetector()
        >>> detector.name
        'koharu_layout'
    """

    # rfdetr >= 1.6 requires transformers>=5, but the app's OCR/translator/inpaint
    # modules pin transformers==4.57.6 in the same environment; 1.5.2 is the last
    # release that accepts transformers 4.x.
    dependencies = ['torch', 'rfdetr==1.5.2', 'safetensors>=0.5']

    download_file_list = [
        {
            'url': 'https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/resolve/main/model.safetensors',
            'files': MODEL_PATH,
            'sha256_pre_calculated': '9bf6d2cbd7793c956d8c857bb1672a396eb7f100eb0682f86830d05e31168efb',
        }
    ]

    params = {
        'text threshold': {
            'type': 'line_editor', 'value': 0.25, 'display_name': 'Text Threshold',
            'description': 'Confidence threshold for text (model card recommends 0.25).',
        },
        'onomatopoeia threshold': {
            'type': 'line_editor', 'value': 0.20, 'display_name': 'Onomatopoeia Threshold',
            'description': 'Confidence threshold for SFX (model card recommends 0.20; raise to 0.40 for precision).',
        },
        'bubble threshold': {
            'type': 'line_editor', 'value': 0.5, 'display_name': 'Bubble Threshold',
            'description': 'Confidence threshold for bubble outlines drawn on the canvas (model card recommends 0.5).',
        },
        'label': {
            'value': {'text': True, 'onomatopoeia': True, 'bubble': False},
            'type': 'check_group',
            'display_name': 'Labels',
        },
        'merge text lines': {
            'type': 'checkbox', 'value': True, 'display_name': 'Merge Text Lines',
        },
        'font size multiplier': {
            'type': 'line_editor', 'value': 1., 'display_name': 'Font Size Multiplier',
        },
        'font size max': {
            'type': 'line_editor', 'value': -1, 'display_name': 'Font Size Max',
        },
        'font size min': {
            'type': 'line_editor', 'value': -1, 'display_name': 'Font Size Min',
        },
        'mask dilate size': {
            # The seg head under-covers thin glyph rims (measured: ink up to
            # 5 px outside the raw mask on real pages), and lama keeps gray
            # stroke shadows unless the mask clears a stroke by ~4 px:
            # blk1 page 1 residual ink after inpaint was 121/24/0 px at
            # ksize 2/4/6. Smaller values inpaint but leave ghost glyphs;
            # the cost is a wider inpainted band around text (~1.8x mask).
            'type': 'line_editor', 'value': 6, 'display_name': 'Mask Dilate Size',
        },
        'device': {**DEVICE_SELECTOR(), 'display_name': 'Device'},
    }

    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None

    def _load_model(self):
        # Mirrors the hub's load_model.py strict loader; rfdetr is pinned because
        # the constructor/state-dict layout is API-sensitive.

        # Albumentations runs a network update check (and warns) when rfdetr
        # imports it; NO_ALBUMENTATIONS_UPDATE is its documented opt-out.
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
        from rfdetr import RFDETRSeg2XLarge
        from safetensors.torch import load_file

        rfdetr_logger = logging.getLogger('rf-detr')
        if not any(isinstance(f, _QuietRFDETRNoise) for f in rfdetr_logger.filters):
            rfdetr_logger.addFilter(_QuietRFDETRNoise())

        with warnings.catch_warnings():
            try:
                # Warning category only exists in rfdetr >= 1.7.
                from rfdetr.config import PretrainWeightsCompatibilityWarning
                warnings.simplefilter('ignore', PretrainWeightsCompatibilityWarning)
            except ImportError:
                pass
            model = RFDETRSeg2XLarge(
                pretrain_weights=None,
                resolution=1152,
                # rfdetr 1.5.2 keeps the position-embedding grid per variant
                # (64 -> 768 px); the checkpoint was trained at 1152/12 = 96.
                positional_encoding_size=96,
                num_select=160,
                num_classes=len(CLASS_NAMES),
                device=self.get_param_value('device'),
            )
        incompatible = model.model.model.load_state_dict(load_file(MODEL_PATH, device='cpu'), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'Incompatible koharu-layout weights: {incompatible}')
        model.model.class_names = CLASS_NAMES.copy()
        self.model = model

    def get_valid_labels(self) -> set:
        return {k for k, v in self.params['label']['value'].items() if v}

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        im_h, im_w = img.shape[:2]
        mask = np.zeros((im_h, im_w), dtype=np.uint8)

        # Text-bearing classes feed the mask/blocks; bubble detections become
        # canvas outlines only, and panel is unused.
        valid_labels = self.get_valid_labels()
        class_thresholds = {
            cid: float(self.get_param_value(f'{name} threshold'))
            for cid, name in ((0, 'text'), (1, 'onomatopoeia'))
            if name in valid_labels
        }
        want_bubble = 'bubble' in valid_labels
        bubble_threshold = float(self.get_param_value('bubble threshold'))
        # The pipeline detects pages without switching the viewer's page, so
        # the in-flight page key wins over current_img for per-page writes.
        page = None
        if proj is not None:
            page = getattr(proj, 'detecting_page', None) or proj.current_img
        bubble_outlines = []
        seams = []
        if page is not None:
            # Each detect run replaces the stored outlines, so disabling the
            # label clears stale ones.
            proj.set_bubble_outlines(page, [])
        if not class_thresholds and not want_bubble:
            return mask, []

        # Run at the lowest threshold, then filter per class as the model
        # card instructs; rfdetr 1.5.2 resizes to the constructor's resolution
        # (1152) and returns masks at source resolution.
        run_thresholds = dict(class_thresholds)
        if want_bubble:
            run_thresholds[2] = bubble_threshold
        dets = self.model.predict(img, threshold=min(run_thresholds.values()))
        ksize = max(int(self.get_param_value('mask dilate size')), 0)

        detected_items = []
        bubble_candidates = []
        text_boxes = []
        if dets is not None and len(dets) > 0:
            masks = dets.mask
            for cls_id_, conf_, xyxy_ in zip(dets.class_id, dets.confidence, dets.xyxy):
                c = int(cls_id_)
                if c in class_thresholds and float(conf_) >= class_thresholds[c]:
                    bx1, by1, bx2, by2 = xyxy_.astype(int)
                    text_boxes.append((
                        max(bx1, 0), max(by1, 0),
                        min(bx2, im_w), min(by2, im_h),
                    ))
            for i, (cls_id, conf) in enumerate(zip(dets.class_id, dets.confidence)):
                cls_id = int(cls_id)
                if cls_id == 2:
                    if not want_bubble:
                        continue
                    x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, im_w), min(y2, im_h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bubble_candidates.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'conf': float(conf),
                        'mask': masks[i].astype(bool) if masks is not None else None,
                    })
                    continue
                if cls_id not in class_thresholds or conf < class_thresholds[cls_id]:
                    continue
                x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, im_w), min(y2, im_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                det_mask = masks[i].astype(bool) if masks is not None else None
                if det_mask is not None:
                    mask[det_mask] = 255
                else:
                    mask[y1:y2, x1:x2] = 255
                # The block box must contain the mask pixels the pipeline later
                # zeroes when it drops an untranslatable block; the mask head can
                # spill outside the box head, and the final dilation grows it
                # another ksize pixels. Mask fragments far from this detection
                # belong to another block, so they must not stretch the box.
                own_raw = (x1, y1, x2, y2)
                others = tuple(b for b in text_boxes if b != own_raw)
                extent = _nearby_mask_extent(det_mask, (x1, y1, x2, y2), others)
                if extent is not None:
                    x1 = min(x1, extent[0])
                    y1 = min(y1, extent[1])
                    x2 = max(x2, extent[2])
                    y2 = max(y2, extent[3])
                x1, y1 = max(x1 - ksize, 0), max(y1 - ksize, 0)
                x2, y2 = min(x2 + ksize, im_w), min(y2 + ksize, im_h)
                pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
                detected_items.append({'pts': pts, 'label': CLASS_NAMES[cls_id]})

        for cand in _selected_bubble_instances(bubble_candidates, bubble_threshold):
            x1, y1, x2, y2 = cand['box']
            outline = _mask_outline(cand['mask']) if cand['mask'] is not None else None
            if outline is None:
                outline = xywh2xyxypoly(
                    np.array([[x1, y1, x2 - x1, y2 - y1]])
                ).reshape(4, 2).tolist()
            bubble_outlines.append(outline)

        if page is not None:
            # The bubble head emits stacked instances of one balloon at
            # lower thresholds (complex shapes score below the model-card
            # value); collapse nested duplicates so lowering the threshold
            # cannot stack outlines on top of each other.
            deduped = _drop_contained_detections(
                [{'pts': poly} for poly in bubble_outlines]
            )
            # Touching bubbles arrive as one instance; two closed parts read
            # as separate bubbles with a chord between them and give layout
            # one outline per bubble. Split after dedup: split sisters overlap
            # enough (small lobe inside the big one's box) to look contained,
            # and a nested duplicate instance must collapse before both copies
            # split into the same pair.
            bubble_outlines = []
            for item in deduped:
                split_parts = _split_two_lobed(item['pts'])
                if split_parts is not None:
                    bubble_outlines.extend(split_parts)
                    # Chord endpoints plus both parts: a block spanning this
                    # seam belongs to two bubbles and is cut per bubble below.
                    seams.append((split_parts[0][0], split_parts[0][-1], split_parts))
                else:
                    bubble_outlines.append(item['pts'])
            proj.set_bubble_outlines(page, bubble_outlines)
            # Also cut where two drawn balloons meet: the line merger glues
            # boxes straddling neighbouring balloons (three runs across two
            # balloons landed in one 414px box), and a split instance has no
            # crossing to cut along. The cutter's 25% share guard keeps
            # slivers from appearing here.
            for i in range(len(bubble_outlines)):
                for j in range(i + 1, len(bubble_outlines)):
                    a = np.asarray(bubble_outlines[i], np.float32)
                    b = np.asarray(bubble_outlines[j], np.float32)
                    ax, ay, aw, ah = cv2.boundingRect(a)
                    bx, by, bw2, bh2 = cv2.boundingRect(b)
                    if (min(ax + aw, bx + bw2) - max(ax, bx)) < 4 or \
                       (min(ay + ah, by + bh2) - max(ay, by)) < 4:
                        continue
                    chord = _outline_boundary(a, b)
                    if chord is not None:
                        seams.append((chord[0], chord[1], [a.tolist(), b.tolist()]))

        blk_list = []
        if not detected_items:
            return mask, blk_list
        # Nested duplicate detections collapse to their largest box.
        detected_items = _drop_contained_detections(detected_items)
        if self.get_param_value('merge text lines'):
            pts_only_list = [item['pts'] for item in detected_items]
            blk_list = mit_merge_textlines(pts_only_list, width=im_w, height=im_h)
        else:
            for item in detected_items:
                pts_sorted, is_vertical = sort_pnts(item['pts'])
                blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=item['label'])
                blk.vertical = is_vertical
                blk.adjust_bbox()
                examine_textblk(blk, im_w, im_h)
                blk_list.append(blk)

        # Cut after line merging (merging pre-cut boxes would re-join them),
        # so each bubble of a split pair keeps its own block and OCR read.
        blk_list = _seam_split_blocks(blk_list, seams, im_w, im_h, bubble_outlines)

        # Snapshot the detection box: TextBlkItem init rewrites block lines
        # from the stored rich text, and bounding_rect() would then return
        # the previous render's text box instead of the detected region.
        for blk in blk_list:
            blk._detected_bbox = list(blk.bounding_rect())

        blk_list = sort_regions(blk_list)

        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            mask = cv2.dilate(mask, element)

        # A box wider than the detection that produced it (a seam-cut sliver,
        # a duplicate that survived dedup) leaves source ink the mask never
        # covers, and the inpainter will not remove it. Measured across 74
        # blocks of a real chapter, healthy blocks sit at 0.86-1.00 and the
        # one known phantom sat at 0.31, so warn only well below that band.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        for idx, blk in enumerate(blk_list):
            frac = _unmasked_ink_fraction(gray, mask, blk.bounding_rect())
            if frac is not None and frac < 0.5:
                LOGGER.warning(
                    'Block %d box %s has only %.0f%% of its ink in the inpaint '
                    'mask; this box is likely a detection artifact and its '
                    'text will not be inpainted.',
                    idx, blk.bounding_rect(), frac * 100,
                )

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            # RF-DETR records its device at construction, so drop the resident
            # model and let the next run rebuild it on the new device.
            self.unload_model()
