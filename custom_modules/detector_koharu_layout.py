"""KoharuLayout-RFDETR-Seg-2XL-1152 custom text detector.

Model: https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152
"""
import logging
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
    """Keep strong bubble instances and weak ones the head agrees on.

    Complex balloons score below the model-card threshold while the head
    fires several nested instances of the same balloon; a lone weak
    instance stays rejected, a consistent stack is real.

    >>> cands = [{'box': [0, 0, 10, 10], 'conf': 0.7},
    ...          {'box': [1, 1, 9, 9], 'conf': 0.3},
    ...          {'box': [50, 50, 60, 60], 'conf': 0.3}]
    >>> [c['conf'] for c in _selected_bubble_instances(cands, 0.5)]
    [0.7, 0.3]
    """
    return [
        cand for i, cand in enumerate(candidates)
        if cand['conf'] >= threshold or any(
            _containment_ratio(cand['box'], other['box']) >= 0.6
            for j, other in enumerate(candidates) if j != i
        )
    ]


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
    limit = 0.7 * float(np.sqrt(total))
    best = None
    for i in range(n - 2):
        if not reflex[i]:
            continue
        for j in range(i + 2, n):
            if not reflex[j] or (i == 0 and j == n - 1):
                continue
            a, b = pts[i], pts[j]
            chord = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if chord > limit:
                continue
            if cv2.pointPolygonTest(pts, (float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)), False) < 0:
                continue
            part1 = pts[i:j + 1]
            part2 = np.vstack([pts[j:], pts[:i + 1]])
            if abs(cv2.contourArea(part1)) < total * 0.2:
                continue
            if abs(cv2.contourArea(part2)) < total * 0.2:
                continue
            if best is None or chord < best[0]:
                best = (chord, part1, part2)
    if best is None:
        return None
    return [best[1].astype(int).tolist(), best[2].astype(int).tolist()]


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
    det_mask: Optional[np.ndarray], box: Tuple[int, int, int, int]
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
    """
    if det_mask is None or not det_mask.any():
        return None
    x1, y1, x2, y2 = box
    slack = max(x2 - x1, y2 - y1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        det_mask.astype(np.uint8), 8
    )
    extent = None
    for idx in range(1, count):
        left = int(stats[idx, cv2.CC_STAT_LEFT])
        top = int(stats[idx, cv2.CC_STAT_TOP])
        right = left + int(stats[idx, cv2.CC_STAT_WIDTH])
        bottom = top + int(stats[idx, cv2.CC_STAT_HEIGHT])
        dx = max(0, left - x2, x1 - right)
        dy = max(0, top - y2, y1 - bottom)
        if max(dx, dy) > slack:
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
        if dets is not None and len(dets) > 0:
            masks = dets.mask
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
                extent = _nearby_mask_extent(det_mask, (x1, y1, x2, y2))
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
            # Touching bubbles arrive as one instance; two closed parts read
            # as separate bubbles with a chord between them and give layout
            # one outline per bubble.
            split_parts = _split_two_lobed(outline)
            if split_parts is not None:
                bubble_outlines.extend(split_parts)
            else:
                bubble_outlines.append(outline)

        if page is not None:
            # The bubble head emits stacked instances of one balloon at
            # lower thresholds (complex shapes score below the model-card
            # value); collapse nested duplicates so lowering the threshold
            # cannot stack outlines on top of each other.
            deduped = _drop_contained_detections(
                [{'pts': poly} for poly in bubble_outlines]
            )
            bubble_outlines = [item['pts'] for item in deduped]
            proj.set_bubble_outlines(page, bubble_outlines)

        blk_list = []
        if not detected_items:
            return mask, blk_list
        # One text run crossing a joined bubble's seam must stay one block;
        # nested duplicate detections collapse to their largest box.
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

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            # RF-DETR records its device at construction, so drop the resident
            # model and let the next run rebuild it on the new device.
            self.unload_model()
