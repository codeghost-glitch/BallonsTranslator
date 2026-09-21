from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .base import DEVICE_SELECTOR, ProjImgTrans, TextBlock, TextDetectorBase, register_textdetectors
from ballontranslator.utils.textblock import examine_textblk, sort_regions
from ballontranslator.utils.bubble import bubble_partition_cells, bubble_polygon_from_mask
from ballontranslator.utils.logger import suppress_model_warnings

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


def _explicit_return_dict(
    module: 'Module', args: tuple, kwargs: dict,
) -> Tuple[tuple, dict]:
    if kwargs.get('return_dict') is None:
        kwargs['return_dict'] = module.config.return_dict
    return args, kwargs

# An onomatopoeia rival evicts a text claim only when it wins by more than
# this margin: sub-threshold classifier noise (one dialogue glyph scored
# ~0.45 as both classes) must not delete dialogue the text threshold already
# accepted, when the winning claim is itself about to be discarded.
_SFX_EVICTION_MARGIN = 0.1

def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    """IoU of two ``xyxy`` boxes. Non-finite coordinates never suppress: every
    comparison against NaN is False, so such boxes stay candidates and are
    rejected later by the explicit finite-value checks in :meth:`_detect`."""
    intersection_width = float(min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = float(min(left[3], right[3]) - max(left[1], right[1]))
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return 0.0
    intersection = intersection_width * intersection_height
    left_area = max(float(left[2] - left[0]), 0.0) * max(float(left[3] - left[1]), 0.0)
    right_area = max(float(right[2] - right[0]), 0.0) * max(float(right[3] - right[1]), 0.0)
    return intersection / (left_area + right_area - intersection)

def _cross_class_loses(
    class_id: int, box: np.ndarray, score: float,
    text_boxes: List[Tuple[np.ndarray, float]],
    sfx_boxes: List[Tuple[np.ndarray, float]],
    detect_sfx: bool,
) -> bool:
    """Whether a text/onomatopoeia candidate yields to a rival of the other
    class claiming the same region.

    The detector often fires both classes over one sound effect with nearly
    identical boxes. The higher-scoring class owns the region (ties go to
    onomatopoeia, the more specific label). With 'detect sound effects'
    disabled, a region the model clearly prefers as onomatopoeia must not
    resurface as a mislabeled text block — but a rival winning by less than
    classifier noise must not evict a text claim the text threshold already
    accepted: one ambiguous dialogue glyph (page 007's `あ`) fired as text
    0.443 and onomatopoeia 0.454, and the discarded onomatopoeia claim would
    otherwise delete the dialogue.

    >>> whirr = np.array([151., 1061., 305., 1824.])
    >>> text = [(whirr, 0.52)]
    >>> sfx = [(whirr, 0.65)]
    >>> _cross_class_loses(0, whirr, 0.52, text, sfx, detect_sfx=False)
    True
    >>> _cross_class_loses(0, whirr, 0.52, text, sfx, detect_sfx=True)
    True
    >>> _cross_class_loses(1, whirr, 0.65, text, sfx, detect_sfx=True)
    False
    >>> _cross_class_loses(1, whirr, 0.65, text, sfx, detect_sfx=False)
    True
    >>> near_tie = np.array([627., 659., 662., 698.])
    >>> _cross_class_loses(0, near_tie, 0.443, None, [(near_tie, 0.454)], detect_sfx=False)
    False
    """
    if class_id == 0:
        return any(
            _intersection_over_union(box, sfx_box) >= 0.5
            and sfx_score >= score + _SFX_EVICTION_MARGIN
            for sfx_box, sfx_score in sfx_boxes
        )
    if not detect_sfx:
        return True
    return any(
        _intersection_over_union(box, text_box) >= 0.5 and text_score > score
        for text_box, text_score in text_boxes
    )

def _containment_duplicate(
    first: int, second: int, xyxy: np.ndarray, masks: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """Return ``(finer, coarser)`` when one same-label box is a merged
    duplicate that already contains the other's glyphs.

    RF-DETR sometimes emits one coarse instance spanning several text columns
    alongside the per-column instances (e.g. a union box over two vertical
    columns). Box IoU cannot see this (a union of two columns
    has ~0.4 IoU with either column), so the pair is a duplicate when the
    smaller box sits inside the bigger one and its mask pixels are already
    covered by the bigger instance's mask. A 0.2 area-ratio floor keeps small
    legitimate instances (furigana, short captions) from evicting the region
    that contains them.

    >>> union = np.zeros((12, 20), dtype=bool); union[1:11, 1:19] = True
    >>> column = np.zeros((12, 20), dtype=bool); column[1:11, 1:9] = True
    >>> _containment_duplicate(
    ...     0, 1, np.array([[1., 1., 19., 11.], [1., 1., 9., 11.]]),
    ...     np.stack([union, column]))
    (1, 0)
    >>> _containment_duplicate(
    ...     0, 1, np.array([[1., 1., 19., 11.], [6., 6., 9., 11.]]),
    ...     np.stack([union, column])) is None
    True
    """
    first_area = _box_area(xyxy[first])
    second_area = _box_area(xyxy[second])
    inner, outer = (
        (second, first) if second_area < first_area else (first, second)
    )
    inner_area = min(first_area, second_area)
    outer_area = max(first_area, second_area)
    if inner_area <= 0.0 or outer_area <= 0.0:
        return None
    if inner_area < 0.2 * outer_area:
        return None
    intersection_width = float(
        min(xyxy[inner][2], xyxy[outer][2]) - max(xyxy[inner][0], xyxy[outer][0])
    )
    intersection_height = float(
        min(xyxy[inner][3], xyxy[outer][3]) - max(xyxy[inner][1], xyxy[outer][1])
    )
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return None
    if intersection_width * intersection_height / inner_area < 0.9:
        return None
    inner_mask = masks[inner].astype(bool)
    inner_pixels = int(inner_mask.sum())
    if not inner_pixels:
        return None
    overlap = int((inner_mask & masks[outer].astype(bool)).sum())
    if overlap / inner_pixels < 0.9:
        return None
    return (inner, outer)

def _box_area(box: np.ndarray) -> float:
    width = max(float(box[2]) - float(box[0]), 0.0)
    height = max(float(box[3]) - float(box[1]), 0.0)
    return width * height

def _non_maximum_suppression(
    xyxy: np.ndarray, class_id: np.ndarray, confidence: np.ndarray,
    masks: Optional[np.ndarray] = None, iou_threshold: float = 0.5,
) -> List[int]:
    """Greedy per-label box NMS; returns kept indices, best score first.

    Dense pages make the detector emit overlapping duplicates, and only
    same-label pairs at or above the IoU threshold are considered duplicates
    (a bubble and a panel may legitimately share a box). Containment
    duplicates also collapse when masks are available: of a coarse/fine pair
    where the fine box sits inside the coarse one, the pair resolves by
    coverage — the coarse instance is a merged duplicate only when the finer
    instances contained in it already cover ~all its ink (a union over two
    columns); otherwise the coarse instance carries unique glyphs no
    finer instance has (the second column of a wrapped sentence) and the
    contained instances are the redundant ones.

    >>> _non_maximum_suppression(
    ...     np.array([[0., 0., 10., 10.], [1., 1., 10., 10.]]),
    ...     np.array([0, 0]), np.array([0.9, 0.8]))
    [0]
    >>> _non_maximum_suppression(
    ...     np.array([[0., 0., 10., 10.], [1., 1., 10., 10.]]),
    ...     np.array([0, 1]), np.array([0.9, 0.8]))
    [0, 1]
    >>> union = np.zeros((12, 20), dtype=bool); union[1:11, 1:19] = True
    >>> column = np.zeros((12, 20), dtype=bool); column[1:11, 1:9] = True
    >>> _non_maximum_suppression(
    ...     np.array([[1., 1., 19., 11.], [1., 1., 9., 11.]]),
    ...     np.array([0, 0]), np.array([0.9, 0.8]),
    ...     masks=np.stack([union, column]))
    [0]
    >>> left = np.zeros((12, 20), dtype=bool); left[1:11, 1:9] = True
    >>> right = np.zeros((12, 20), dtype=bool); right[1:11, 11:19] = True
    >>> both = left | right
    >>> _non_maximum_suppression(
    ...     np.array([[1., 1., 19., 11.], [1., 1., 9., 11.], [11., 1., 19., 11.]]),
    ...     np.array([0, 0, 0]), np.array([0.9, 0.8, 0.8]),
    ...     masks=np.stack([both, left, right]))
    [1, 2]
    """
    order = np.argsort(-confidence, kind='stable')
    kept: List[int] = []
    for index in order:
        if any(
            class_id[other] == class_id[index]
            and _intersection_over_union(xyxy[other], xyxy[index]) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(int(index))
    if masks is None:
        return kept
    # Containment resolution runs after the IoU pass, over the kept set, and
    # decides by coverage of ALL contained finer instances — greedy eviction
    # would drop a coarse instance whose unique glyphs (the second column of
    # a wrapped sentence) only a future candidate covers, silently deleting
    # text. A coarse instance is a merged duplicate only when the finer
    # instances contained in it already cover ~all its ink; otherwise the
    # coarse instance carries unique text and the contained instances (each
    # >=90% covered by it, per _containment_duplicate) are the redundant ones.
    changed = True
    while changed:
        changed = False
        for coarse in sorted(kept, key=lambda index: -_box_area(xyxy[index])):
            fines = [
                fine for fine in kept
                if fine != coarse
                and class_id[fine] == class_id[coarse]
                and _containment_duplicate(fine, coarse, xyxy, masks) == (fine, coarse)
            ]
            if not fines:
                continue
            covered = np.zeros_like(masks[coarse], dtype=bool)
            for fine in fines:
                covered |= masks[fine].astype(bool)
            coarse_pixels = int(masks[coarse].astype(bool).sum())
            if coarse_pixels and int((covered & masks[coarse].astype(bool)).sum()) / coarse_pixels >= 0.9:
                kept.remove(coarse)
            else:
                for fine in fines:
                    kept.remove(fine)
            changed = True
            break
    return kept

def _infer_vertical(mask: np.ndarray) -> Optional[bool]:
    """Infer the writing mode of one instance mask from projection profiles.

    Text lines are separated by near-empty bands along the line-stacking axis:
    vertical manga columns stack along x, horizontal lines along y. The axis
    whose profile concentrates its energy in fewer bins crosses those bands,
    so a sharper per-column profile means the lines stack along x, i.e. the
    text is vertical, from projection profiles at angle zero. Ties within 2%
    (including whole-block squares and rotated sound
    effects) are ambiguous so the caller can fall back to the configured
    default instead of guessing.

    >>> vertical = np.zeros((60, 120), dtype=np.uint8)
    >>> vertical[10:50, 20:36] = 1
    >>> vertical[10:50, 80:96] = 1
    >>> _infer_vertical(vertical)
    True
    >>> horizontal = np.zeros((60, 120), dtype=np.uint8)
    >>> horizontal[20:36, 10:110] = 1
    >>> _infer_vertical(horizontal)
    False
    >>> square = np.zeros((40, 40), dtype=np.uint8)
    >>> square[10:30, 10:30] = 1
    >>> _infer_vertical(square) is None
    True
    >>> _infer_vertical(np.zeros((10, 10), dtype=np.uint8)) is None
    True
    """
    columns = mask.sum(axis=0, dtype=np.float64)
    rows = mask.sum(axis=1, dtype=np.float64)
    pixels = columns.sum()
    if pixels <= 0:
        return None
    column_energy = float(np.square(columns).sum()) / pixels
    row_energy = float(np.square(rows).sum()) / pixels
    strongest = max(column_energy, row_energy)
    if abs(column_energy - row_energy) <= strongest * 0.02:
        return None
    result = column_energy > row_energy
    # Energy alone misreads dense multi-column runs: a run of three vertical
    # columns nearly touching has a per-row profile as
    # sharp as the per-column one, and the run reads horizontal. Bands along
    # the stacking axis disambiguate: vertical columns are taller than the
    # gaps between them are wide, and vice versa for horizontal lines, so the
    # larger mean band aspect (extent along the flow / band width) wins.
    band_aspect_vertical = _band_aspect(mask, vertical=True)
    band_aspect_horizontal = _band_aspect(mask, vertical=False)
    # One axis with two or more bands is the line-stacking signal: vertical
    # columns appear as x-bands, horizontal lines as y-bands. An x-band aspect
    # (height/width) above 1.2 means tall columns; a y-band aspect
    # (width/height) above 2.0 means wide lines — vertical glyph runs are
    # near-square (column width over glyph height <= 2), horizontal lines are
    # not. Square-ish bands on both axes (rotated SFX) carry no signal.
    signals = []
    if band_aspect_vertical is not None and band_aspect_vertical > 1.2:
        signals.append((band_aspect_vertical, True))
    if band_aspect_horizontal is not None and band_aspect_horizontal > 2.0:
        signals.append((band_aspect_horizontal, False))
    if len(signals) == 1:
        return signals[0][1]
    if len(signals) == 2:
        # A 2-D grid of text fires both; the more elongated axis wins.
        return max(signals)[1]
    return result

def _band_aspect(mask: np.ndarray, vertical: bool) -> Optional[float]:
    """Median flow-extent / width ratio of the ink bands along one axis.

    Vertical text stacks columns along x; each band between empty columns is
    one text column whose height over its width is the aspect. Returns None
    when fewer than two bands exist (single blobs carry no stacking signal).

    >>> column = np.zeros((10, 12), dtype=bool)
    >>> column[1:9, 2:5] = True
    >>> column[1:9, 7:10] = True
    >>> round(_band_aspect(column, vertical=True), 1)
    2.7
    """
    ink = mask > 0
    profile = ink.sum(axis=0 if vertical else 1)
    bands: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(profile):
        if value > 0 and start is None:
            start = index
        elif value == 0 and start is not None:
            bands.append((start, index))
            start = None
    if start is not None:
        bands.append((start, len(profile)))
    bands = [band for band in bands if band[1] - band[0] >= 3]
    if len(bands) < 2:
        return None
    # Fragments (furigana rows, stray punctuation) have extreme aspect
    # ratios that would outweigh real columns or lines; ignore bands much
    # narrower than the dominant one.
    widest = max(band[1] - band[0] for band in bands)
    bands = [band for band in bands if band[1] - band[0] >= widest * 0.25]
    if len(bands) < 2:
        return None
    aspects: List[float] = []
    for band_start, band_end in bands:
        window = ink[:, band_start:band_end] if vertical else ink[band_start:band_end, :]
        if not window.any():
            continue
        # Extent of actual ink along the flow axis, not the raw window
        # height: bbox slack must not dilute the elongation signal.
        if vertical:
            inked = np.where(window.any(axis=1))[0]
            aspects.append((inked[-1] - inked[0] + 1) / window.shape[1])
        else:
            inked = np.where(window.any(axis=0))[0]
            aspects.append((inked[-1] - inked[0] + 1) / window.shape[0])
    if not aspects:
        return None
    # Median, not mean: tiny glyph runs (punctuation, small kana) would drag
    # the mean up and read a vertical column as horizontal lines.
    return float(np.median(aspects))


def _estimate_font_size(text_mask: np.ndarray, vertical: bool, fallback: int) -> int:
    """Estimate the glyph size from a text run's segmentation mask.

    Koharu emits one bbox per text run, so ``min(w, h)`` equals a
    multi-line run's column width, not the glyph size, and translations
    rendered several times larger than the source text. Project the mask
    onto the axis the lines stack along: each line or glyph forms an ink
    band whose width is the glyph size, and the median band shrugs off
    furigana and punctuation. ``fallback`` (the bbox minimum dimension)
    keeps the old ceiling for empty or blob-like masks.

    Masks over busy art also bleed sideways, making every column band
    fatter than its glyphs. Each band is therefore refined with the ink
    runs along the reading-flow axis inside it: glyph bodies form runs
    clearly shorter than the band, so their 75th percentile replaces a
    band width that exceeds it by >20%. Tight columns (glyph run as tall
    as the band) keep the band width, which stays the glyph advance.

    >>> mask = np.zeros((40, 100), dtype=bool)
    >>> mask[10:30, 5:25] = True
    >>> mask[10:30, 55:75] = True
    >>> _estimate_font_size(mask, True, 95)
    24
    """
    def snap8(size: int) -> int:
        # Glyph sizes of identical source text must come out identical even
        # when mask bleed/noise shifts band widths a few pixels, or the same
        # Japanese line would render at different translated sizes. Snap to a
        # coarse 8px grid (font-size ladder in manga raster resolutions).
        return (size + 4) // 8 * 8
    ink = text_mask > 0
    if ink.any():
        profile = ink.sum(axis=0 if vertical else 1)
        bands: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for index, value in enumerate(profile):
            if value > 0 and start is None:
                start = index
            elif value == 0 and start is not None:
                bands.append((start, index))
                start = None
        if start is not None:
            bands.append((start, len(profile)))
        refined: List[float] = []
        for band_start, band_end in bands:
            band_width = band_end - band_start
            if band_width < 3:
                continue
            window = ink[:, band_start:band_end] if vertical else ink[band_start:band_end, :]
            # Runs along the reading-flow axis: vertical text reads down a
            # column (y), horizontal text reads across a line (x).
            inner = window.sum(axis=1 if vertical else 0) > 0
            runs: List[int] = []
            run = 0
            for value in inner:
                if value:
                    run += 1
                elif run:
                    runs.append(run)
                    run = 0
            if run:
                runs.append(run)
            kept = [length for length in runs if length >= 4]
            # A band narrower than its own glyph runs only holds sliver
            # fragments (a broken two-column run split into one-glyph
            # halves); skip it. Solid masks (SFX void-fill, or text painted
            # over a dark panel whose mask merged with the background) have
            # no per-glyph gap structure: nearly every flow row/column is
            # inked, and the first run spans almost the whole band. Both
            # states carry no font-size signal, so they fall through to the
            # next band or the fallback.
            if not kept:
                continue
            # Genuine slivers: every glyph is much shorter than the column
            # band (a two-column run split into one-glyph halves).
            if max(kept) < band_width * 0.5:
                continue
            # Floods: a single run spanning most of the flow extent is a
            # mask merged with its background; no glyph structure there.
            if len(kept) == 1 and max(kept) > 0.8 * max(window.shape):
                continue
            inner_size = min(
                float(np.percentile(kept, 75)), float(band_width),
            )
            refined.append(inner_size)
        if refined:
            estimate = int(round(float(np.percentile(refined, 75))))
            if estimate >= 3:
                return snap8(min(estimate, fallback))
    return snap8(fallback)

def _columns_can_merge(first: TextBlock, second: TextBlock) -> bool:
    """Whether two free vertical text columns share one sentence.

    Same-size columns (ratio <= 1.6) whose boxes sit within 0.6 glyph of each
    other and overlap vertically are one sentence wrapped across columns;
    anything else (separate columns can sit hundreds of pixels apart) stays
    independent.

    >>> free = TextBlock(xyxy=[0, 0, 10, 100], src_is_vertical=True)
    >>> free._detected_font_size = 24
    >>> near = TextBlock(xyxy=[18, 2, 28, 98], src_is_vertical=True)
    >>> near._detected_font_size = 24
    >>> _columns_can_merge(free, near)
    True
    >>> far = TextBlock(xyxy=[200, 2, 210, 98], src_is_vertical=True)
    >>> far._detected_font_size = 24
    >>> _columns_can_merge(free, far)
    False
    """
    sizes = (first.detected_font_size, second.detected_font_size)
    if min(sizes) <= 0 or max(sizes) / min(sizes) > 1.6:
        return False
    gap = max(first.xyxy[0], second.xyxy[0]) - min(first.xyxy[2], second.xyxy[2])
    if gap < 0:
        # Overlapping boxes are not side-by-side columns; the x-gap must be
        # positive for two columns of one wrapped sentence.
        return False
    if gap > 0.6 * max(sizes):
        return False
    overlap = min(first.xyxy[3], second.xyxy[3]) - max(first.xyxy[1], second.xyxy[1])
    if overlap < 0.5 * min(first.xyxy[3] - first.xyxy[1], second.xyxy[3] - second.xyxy[1]):
        return False
    return True


def _merge_free_vertical_columns(
    blocks: List[TextBlock], width: int, height: int, page_mask: np.ndarray,
) -> List[TextBlock]:
    """Merge adjacent free vertical text columns of one sentence into one block.

    Koharu emits one instance per text column; a sentence wrapped across two
    columns arrives as two blocks and the OCR
    then translates the halves separately. Bubble-free same-writing columns
    that _columns_can_merge accepts merge so the OCR reads the whole run, and
    the font size re-estimates from the combined mask. Bubble-sharing runs
    merge in _merge_balloon_blocks instead.

    >>> free = TextBlock(xyxy=[0, 0, 10, 10], src_is_vertical=True)
    >>> _merge_free_vertical_columns(
    ...     [free], 100, 100, np.zeros((10, 10), np.uint8))[0] is free
    True
    """
    out: List[TextBlock] = []
    for block in blocks:
        if block.bubble_polygon is not None or not block.src_is_vertical:
            out.append(block)
            continue
        target = None
        for existing in out:
            if existing.bubble_polygon is not None or not existing.src_is_vertical:
                continue
            if _columns_can_merge(existing, block):
                target = existing
                break
        if target is None:
            out.append(block)
            continue
        # _merge_block_group mutates its first member in place, so the merged
        # block replaces the target already stored in out.
        combined = _merge_block_group([target, block], width, height, None)
        # The merged columns share one glyph size; the union mask estimates it
        # better than the minimum of the per-column bands.
        left, top, right, bottom = (int(value) for value in combined.xyxy)
        window = page_mask[top:bottom, left:right] > 0
        fallback = min(right - left, bottom - top)
        combined.font_size = combined._detected_font_size = _estimate_font_size(
            window, True, fallback,
        )
    return out


def _merge_balloon_blocks(
    blocks: List[TextBlock], width: int, height: int,
) -> List[TextBlock]:
    """Merge text runs that share one balloon into one block per flow.

    RF-DETR can emit several runs for one dialogue. Split joined outlines at
    physical necks first, then merge runs within each lobe. Each output keeps
    its lobe outline so fitting and center guides use that lobe, not the whole
    joined balloon. Smooth single-lobe balloons keep one translation flow.

    Only text blocks carrying the same bubble polygon merge, and only when
    they share a writing mode; sound effects, panels, and free text are never
    combined, so unrelated neighbours stay independent.

    >>> free = TextBlock(xyxy=[0, 0, 10, 10], src_is_vertical=True)
    >>> _merge_balloon_blocks([free], 100, 100)[0] is free
    True
    """
    grouped: Dict[Tuple[Tuple[float, float], ...], List[TextBlock]] = {}
    merged: List[TextBlock] = []
    for block in blocks:
        if block.bubble_polygon is None:
            merged.append(block)
            continue
        grouped.setdefault(tuple(tuple(point) for point in block.bubble_polygon), []).append(block)
    for polygon_key, members in grouped.items():
        if len(members) == 1 or len({bool(block.src_is_vertical) for block in members}) > 1:
            merged.extend(members)
            continue
        polygon = [list(point) for point in polygon_key]
        centers = [
            (
                float(block.xyxy[0] + block.xyxy[2]) / 2,
                float(block.xyxy[1] + block.xyxy[3]) / 2,
            )
            for block in members
        ]
        cells = bubble_partition_cells(polygon, centers)
        if cells is None:
            # Single-lobe balloon: one dialogue flow, so every run merges.
            merged.append(_merge_block_group(members, width, height, polygon))
            continue
        # Joined balloon: runs sharing a lobe merge; the lobe becomes each
        # block's outline so fit and the center guide use the lobe, not the
        # whole two-lobe blob.
        lobes: Dict[Tuple[Tuple[float, float], ...], List[TextBlock]] = {}
        for block, cell in zip(members, cells):
            lobes.setdefault(tuple(tuple(point) for point in cell), []).append(block)
        for cell, lobe_members in lobes.items():
            merged.append(
                _merge_block_group(lobe_members, width, height, [list(point) for point in cell])
            )
    return merged

def _merge_block_group(
    members: List[TextBlock], width: int, height: int,
    polygon: List[List[float]],
) -> TextBlock:
    """Collapse one run group into a single block ordered for reading."""
    vertical = bool(members[0].src_is_vertical)
    members.sort(key=lambda block: block.xyxy[0], reverse=vertical)
    primary = members[0]
    primary.lines = [line for block in members for line in block.lines]
    examine_textblk(primary, width, height, sort=True)
    # Member sizes are mask-band estimates now; the minimum is the safest
    # glyph size for a group whose merged lines can hold multi-column runs.
    primary.font_size = primary._detected_font_size = min(
        block.detected_font_size for block in members
    )
    primary.adjust_bbox()
    primary.bubble_polygon = polygon
    if vertical:
        primary.alignment = 1
    else:
        primary.recalulate_alignment()
    return primary

class KoharuPostProcessor:
    """Filter native candidates before projecting one mask at a time to the page.

    ``num_select`` bounds the candidates that can reach mask projection. The
    container postprocessor is replaced, so the cap is owned here instead of
    being configured twice through the container.

    >>> processor = KoharuPostProcessor()
    >>> processor.num_select, processor.thresholds
    (160, {0: 0.25, 2: 0.5})
    """

    def __init__(self, num_select: int = 160) -> None:
        self.num_select = num_select
        self.thresholds = {0: 0.25, 2: 0.5}

    def __call__(
        self, outputs: Dict[str, 'Tensor'], target_sizes: 'Tensor',
    ) -> List[Dict[str, 'Tensor']]:
        import torch
        from torch.nn.functional import interpolate

        logits = outputs['pred_logits']
        probabilities = logits.sigmoid().flatten(1)
        scores, indices = probabilities.topk(min(self.num_select, probabilities.shape[1]), dim=1)
        results = []
        for batch_index, target in enumerate(target_sizes):
            labels = indices[batch_index] % logits.shape[-1]
            keep = torch.zeros_like(labels, dtype=torch.bool)
            for class_id, threshold in self.thresholds.items():
                keep |= (labels == class_id) & (scores[batch_index] >= threshold)
            # Mirrors the container filter that runs right after this one, so masks
            # dropped by the page-level threshold are never resized.
            keep &= scores[batch_index] > min(self.thresholds.values())
            queries = indices[batch_index][keep] // logits.shape[-1]
            boxes = outputs['pred_boxes'][batch_index, queries]
            centers, half_sizes = boxes[:, :2], boxes[:, 2:] / 2
            boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=1)
            height, width = (int(value) for value in target.tolist())
            boxes *= boxes.new_tensor([width, height, width, height])
            masks = torch.empty((len(queries), 1, height, width), dtype=torch.bool, device=logits.device)
            for output_index, query in enumerate(queries):
                native = outputs['pred_masks'][batch_index, query][None, None]
                masks[output_index] = interpolate(
                    native, size=(height, width), mode='bilinear', align_corners=False,
                )[0] > 0
            results.append({
                'scores': scores[batch_index][keep], 'labels': labels[keep],
                'boxes': boxes, 'masks': masks,
            })
        return results


@register_textdetectors('koharu')
class KoharuDetector(TextDetectorBase):
    """KoharuLayout RF-DETR Seg 2XL text detection and segmentation.

    >>> detector = KoharuDetector()
    >>> detector.name, detector.all_model_loaded()
    ('koharu', False)
    """

    dependencies = [
        'torch>=2.2.0', 'torchvision>=0.17.0', 'rfdetr==1.7.0',
        'safetensors>=0.5', 'transformers>=5.1.0,<6.0.0',
    ]
    params = {
        'description': 'KoharuLayout RF-DETR Seg 2XL (1152 px). Detects text with '
                       'segmentation masks; optionally includes sound effects. '
                       'CUDA is recommended. Model: mayocream/koharu-layout-rfdetr-seg-2xl-1152.',
        'device': DEVICE_SELECTOR(not_supported=['privateuseone']),
        'text threshold': {
            'type': 'line_editor', 'value': 0.25,
            'description': 'Minimum text confidence (0.0-1.0).',
        },
        'detect sound effects': {'type': 'checkbox', 'value': False},
        'bubble threshold': {
            'type': 'line_editor', 'value': 0.50,
            'description': 'Minimum bubble confidence for typesetting geometry (0.0-1.0).',
        },
        'sound effect threshold': {
            'type': 'line_editor', 'value': 0.20,
            'description': 'Minimum sound effect confidence (0.0-1.0).',
        },
        'source text is vertical': {'type': 'checkbox', 'value': True},
        'auto detect text orientation': {
            'type': 'checkbox', 'value': True,
            'description': 'Infer per-region vertical or horizontal writing from '
                           'its segmentation mask. Ambiguous regions fall back to '
                           'the "source text is vertical" setting.',
        },
        'fill bubble text boxes': {
            'type': 'checkbox', 'value': True,
            'description': 'Mask the whole detected bubble interior when segmentation '
                           'leaves letter fragments. May remove artwork inside the '
                           'bubble. Free text and sound effects keep their segmentation '
                           'masks.',
        },
        'mask dilate size': {
            'type': 'line_editor', 'value': 2,
            'description': 'Text mask dilation radius in source-image pixels (0-100). '
                           'Letter gaps and holes up to twice the radius are also '
                           'closed without growing the mask further.',
        },
    }
    _load_model_keys = {'model'}
    download_file_list = [{
        'url': 'https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/'
               'resolve/aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2/model.safetensors',
        'files': 'data/models/koharu/model.safetensors',
        'sha256_pre_calculated': '9bf6d2cbd7793c956d8c857bb1672a396eb7f100eb0682f86830d05e31168efb',
    }]

    def _load_model(self) -> None:
        weights = Path('data/models/koharu/model.safetensors')
        if not weights.is_file():
            raise FileNotFoundError(
                f'Koharu weights are missing: {weights}. Run the selected module setup '
                'or download model.safetensors from '
                'mayocream/koharu-layout-rfdetr-seg-2xl-1152 to this path.'
            )
        try:
            from rfdetr import RFDETRSeg2XLarge
            from rfdetr.config import PretrainWeightsCompatibilityWarning
            from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersBackbone
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                'Koharu requires rfdetr==1.7.0, safetensors>=0.5, and '
                'transformers>=5.1.0,<6.0.0. Check the module dependency setup. '
                f'Backend import failed: {error}'
            ) from error

        with warnings.catch_warnings(), suppress_model_warnings(
            'rf-detr', (
                'Using a different number of positional encodings than DINOv2,',
                'Using patch size 12 instead of 14,',
            ),
        ):
            warnings.filterwarnings('ignore', category=PretrainWeightsCompatibilityWarning)
            model = RFDETRSeg2XLarge(
                pretrain_weights=None,
                resolution=1152,
                num_classes=4,
                device=self.get_param_value('device'),
            )
        model.model.model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
        for module in model.model.model.modules():
            if isinstance(module, WindowedDinov2WithRegistersBackbone):
                module.register_forward_pre_hook(_explicit_return_dict, with_kwargs=True)
        model.optimize_for_inference(compile=False)
        model.model.class_names = ['text', 'onomatopoeia', 'bubble', 'panel']
        # The container postprocessor is replaced right below, so the candidate cap
        # lives with KoharuPostProcessor instead of the container configuration.
        model.model.postprocess = KoharuPostProcessor()
        self.model = model

    def _detect(
        self, img: np.ndarray, proj: Optional[ProjImgTrans] = None,
    ) -> Tuple[np.ndarray, List[TextBlock]]:
        if img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) == 0:
            raise ValueError('Koharu requires a non-empty RGB image.')
        thresholds = {}
        for class_id, param_key in (
            (0, 'text threshold'), (1, 'sound effect threshold'), (2, 'bubble threshold'),
        ):
            threshold = float(self.get_param_value(param_key))
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f'{param_key} must be between 0 and 1.')
            thresholds[class_id] = threshold
        # Class 1 stays in the postprocess thresholds even when sound
        # effects are disabled: its candidates are still needed below to
        # resolve regions the model fired on as both text and SFX.
        dilation = float(self.get_param_value('mask dilate size'))
        if not 0 <= dilation <= 100 or not dilation.is_integer():
            raise ValueError('mask dilate size must be an integer between 0 and 100.')
        detect_sfx = bool(self.get_param_value('detect sound effects'))

        # Bubble candidates down to half the lowest other-class threshold
        # stay visible: merged and dim bubbles score low, and their outlines
        # are still recovered for text blocks the kept bubbles cannot link.
        post_thresholds = thresholds.copy()
        post_thresholds[2] = min(thresholds.values()) * 0.5
        self.model.model.postprocess.thresholds = post_thresholds
        detections = self.model.predict(
            img, threshold=min(post_thresholds.values()), shape=(1152, 1152),
            include_source_image=False,
        )
        # The hint pass below needs the raw sub-threshold bubble candidates:
        # the NMS containment resolution keeps a large bled coarse mask over
        # the tight hexagons it contains.
        raw_detections = detections
        if len(detections.xyxy) > 1:
            kept = _non_maximum_suppression(
                detections.xyxy, detections.class_id, detections.confidence,
                masks=detections.mask,
            )
            if len(kept) != len(detections.xyxy):
                detections = detections[np.asarray(kept, dtype=int)]
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        blocks = []
        text_entries: List[Tuple[TextBlock, int]] = []
        bubbles = []
        source_gray = None
        for index, (class_id, score) in enumerate(zip(detections.class_id, detections.confidence)):
            if class_id != 2 or not np.isfinite(score) or score < thresholds[2]:
                continue
            if detections.mask is None or detections.mask[index].shape != (height, width):
                self.logger.warning('Koharu bubble has no valid segmentation mask; ignoring its geometry.')
                continue
            if source_gray is None:
                source_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            polygon = bubble_polygon_from_mask(detections.mask[index], source_gray)
            if polygon is not None:
                bubbles.append((int(detections.mask[index].sum()), polygon, index))
        bubbles.sort(key=lambda bubble: bubble[0])
        bubble_threshold = thresholds[2]
        thresholds.pop(2)
        auto_vertical = bool(self.get_param_value('auto detect text orientation'))
        default_vertical = bool(self.get_param_value('source text is vertical'))
        text_candidates = [
            (detections.xyxy[i], float(detections.confidence[i]))
            for i in range(len(detections.xyxy))
            if detections.class_id[i] == 0
            and np.isfinite(detections.confidence[i])
            and detections.confidence[i] >= thresholds[0]
        ]
        sfx_candidates = [
            (detections.xyxy[i], float(detections.confidence[i]))
            for i in range(len(detections.xyxy))
            if detections.class_id[i] == 1
            and np.isfinite(detections.confidence[i])
            and detections.confidence[i] >= thresholds[1]
        ]
        for index, (box, class_id, score) in enumerate(zip(
            detections.xyxy, detections.class_id, detections.confidence,
        )):
            if class_id not in thresholds or not np.isfinite(score) or score < thresholds[class_id]:
                continue
            if _cross_class_loses(
                int(class_id), box, float(score),
                text_candidates, sfx_candidates, detect_sfx,
            ):
                continue
            if not np.isfinite(box).all():
                # One malformed RF-DETR query must not lose every other
                # detection on the page; skip just this candidate.
                self.logger.warning('Koharu returned a non-finite box; ignoring it.')
                continue
            left, top = np.floor(np.clip(box[:2], [0, 0], [width, height])).astype(int).tolist()
            right, bottom = np.ceil(np.clip(box[2:], [0, 0], [width, height])).astype(int).tolist()
            if right <= left or bottom <= top:
                continue
            if detections.mask is None or detections.mask[index].shape != (height, width):
                raise ValueError('Koharu returned a missing or incorrectly sized text segmentation mask.')
            text_mask = detections.mask[index].astype(bool)
            mask[text_mask] = 255
            if auto_vertical:
                vertical = _infer_vertical(text_mask[top:bottom, left:right])
            else:
                vertical = None
            if vertical is None:
                vertical = default_vertical
            block = TextBlock(
                xyxy=[left, top, right, bottom],
                lines=[[[left, top], [right, top], [right, bottom], [left, bottom]]],
                src_is_vertical=vertical,
                label='text' if class_id == 0 else 'onomatopoeia',
            )
            block.vertical = vertical
            # Link text to the smallest bubble whose mask
            # contains at least 90% of the text mask. A bbox-center test would
            # also attach free text that merely overlaps a bubble, and the
            # assigned polygon then drives destructive bubble-box filling.
            if class_id == 0:
                text_window = text_mask[top:bottom, left:right]
                text_pixels = int(text_window.sum())
                if text_pixels:
                    for _, polygon, bubble_index in bubbles:
                        bubble_window = detections.mask[bubble_index][top:bottom, left:right].astype(bool)
                        if int((text_window & bubble_window).sum()) / text_pixels >= 0.9:
                            block.bubble_polygon = polygon
                            if self.get_param_value('fill bubble text boxes'):
                                # Furigana and ruby sit above/beside the text
                                # rectangle but still inside the balloon;
                                # filling only the text-rect slice leaves
                                # that ink for the inpainter to preserve.
                                # Fill the whole detected interior instead;
                                # the option warns it may remove artwork.
                                mask[detections.mask[bubble_index].astype(bool)] = 255
            # The mask-band estimate keeps multi-line runs at the real
            # glyph size; the bbox minimum dimension alone equals a
            # multi-column run's full width.
            fallback = min(right - left, bottom - top)
            detected_size = _estimate_font_size(
                text_mask[top:bottom, left:right], vertical, fallback,
            )
            block.font_size = block._detected_font_size = detected_size
            if class_id == 0:
                text_entries.append((block, index))
            blocks.append(block)

        blocks = _merge_free_vertical_columns(
            blocks, img.shape[1], img.shape[0], mask,
        )
        # The merge mutates its first member in place; entries of absorbed
        # columns are dead and must not claim hint outlines.
        live_ids = {id(block) for block in blocks}
        text_entries = [
            (block, text_index) for block, text_index in text_entries
            if id(block) in live_ids
        ]

        # Merged and dim bubbles often score below 'bubble threshold' while
        # their text columns detect at full confidence. Text that linked to
        # no kept bubble falls back to the smallest sub-threshold bubble
        # candidate containing >=90% of its mask — one hexagon of a joined
        # bubble beats no outline. This runs after the free-column merge so
        # a sentence's columns merge before any of them gains an outline
        # (the merge skips blocks that already carry one). Candidates much
        # larger than their text are region blobs, not bubble outlines, and
        # stay ignored however confident the text is.
        if text_entries and detections.mask is not None:
            if source_gray is None:
                source_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            hint_floor = min(thresholds.values()) * 0.5
            hints = sorted(
                (
                    (int(raw_detections.mask[i].astype(bool).sum()), i)
                    for i in range(len(raw_detections.xyxy))
                    if raw_detections.class_id[i] == 2
                    and hint_floor <= float(raw_detections.confidence[i]) < bubble_threshold
                    and raw_detections.mask[i].shape == (height, width)
                ),
                key=lambda hint: hint[0],
            )
            hint_boxes = {
                i: float(
                    max(raw_detections.xyxy[i][2] - raw_detections.xyxy[i][0], 0.0)
                    * max(raw_detections.xyxy[i][3] - raw_detections.xyxy[i][1], 0.0)
                )
                for _, i in hints
            }
            for _, hint_index in hints:
                hint_mask = raw_detections.mask[hint_index].astype(bool)
                if not hint_mask.any():
                    continue
                contained = []
                for block, text_index in text_entries:
                    if block.bubble_polygon is not None:
                        continue
                    text_mask = detections.mask[text_index].astype(bool)
                    text_pixels = int(text_mask.sum())
                    if not text_pixels:
                        continue
                    box = block.xyxy
                    text_box_area = float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))
                    if hint_boxes[hint_index] > 4.0 * text_box_area:
                        continue
                    if int((text_mask & hint_mask).sum()) / text_pixels >= 0.9:
                        contained.append(block)
                if not contained:
                    continue
                polygon = bubble_polygon_from_mask(raw_detections.mask[hint_index], source_gray)
                if polygon is None:
                    continue
                for block in contained:
                    block.bubble_polygon = polygon

        blocks = _merge_balloon_blocks(blocks, img.shape[1], img.shape[0])

        if dilation and blocks:
            radius = int(dilation)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            # Closing the page mask (two dilations, one erosion) keeps the
            # same ~radius expansion while also filling glyph holes and
            # inter-letter gaps, which measurably improves inpainting mask IoU.
            mask = cv2.morphologyEx(cv2.dilate(mask, kernel), cv2.MORPH_CLOSE, kernel)
        return mask, sort_regions(blocks)

    def updateParam(self, param_key: str, param_content: object) -> None:
        previous = self.get_param_value(param_key)
        super().updateParam(param_key, param_content)
        if param_key == 'device' and self.get_param_value(param_key) != previous:
            self.unload_model()
