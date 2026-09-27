"""Regression: koharu block boxes must contain the detector's own mask, and
bubble outlines persist per page without entering the text mask or blocks.

The pipeline zeroes a removed block's bounding_rect() out of the page mask
after OCR; any mask pixel outside every block box survives that step and is
inpainting the removed region as artifacts.
"""
import cv2
import numpy as np

from ballontranslator.modules import TEXTDETECTORS
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


def _synthetic_page() -> np.ndarray:
    img = np.full((1400, 1000, 3), 250, np.uint8)
    cv2.rectangle(img, (40, 40), (960, 660), (20, 20, 20), 6)
    cv2.ellipse(img, (250, 300), (170, 120), 0, 0, 360, (15, 15, 15), 5)
    for i in range(6):
        cv2.line(img, (180 + i * 22, 220), (180 + i * 22, 380), (0, 0, 0), 8)
    return img


def test_mask_stays_inside_block_boxes():
    det = TEXTDETECTORS.resolve_module('koharu_layout')()
    det.updateParam('mask dilate size', 3)
    det.updateParam('label', {'text': True, 'onomatopoeia': True, 'bubble': True})
    proj = ProjImgTrans()
    proj._image_info = {'001.png': {}}
    proj.current_img = '001.png'
    mask, blks = det.detect(_synthetic_page(), proj)
    assert blks, 'detector found nothing on the synthetic page'

    # Bubble detections are outlines only: they never become text blocks or
    # enter the text mask (the stray check below would flag leaked pixels).
    assert all(blk.label != 'bubble' for blk in blks)
    covered = np.zeros(mask.shape, dtype=bool)
    for blk in blks:
        x, y, w, h = blk.bounding_rect()
        covered[y:y + h, x:x + w] = True
    stray = (mask > 0) & ~covered
    assert not stray.any(), f'{int(stray.sum())} mask px outside every block box'

    # Stored outlines are well-formed and inside the page.
    im_h, im_w = mask.shape
    outlines = proj.get_bubble_outlines('001.png')
    for poly in outlines:
        assert len(poly) >= 3, f'degenerate outline polygon: {poly}'
        assert all(0 <= x < im_w and 0 <= y < im_h for x, y in poly), \
            f'outline point outside page: {poly}'

    # Disabling the label clears stale outlines on the next run.
    det.updateParam('label', {'text': True, 'onomatopoeia': True, 'bubble': False})
    det.detect(_synthetic_page(), proj)
    assert proj.get_bubble_outlines('001.png') == []


def test_mask_covers_detected_text_ink():
    """Text ink the detector claims must be inside the inpaint mask.

    The seg head under-covers glyph rims: ink left outside the mask passes
    through lama untouched and resurfaces as ghost glyphs (page 1, blk1).
    The shipped default dilation is what closes those gaps, so this runs on
    default params.
    """
    det = TEXTDETECTORS.resolve_module('koharu_layout')()
    det.updateParam('label', {'text': True, 'onomatopoeia': True, 'bubble': True})
    proj = ProjImgTrans()
    proj._image_info = {'001.png': {}}
    proj.current_img = '001.png'
    img = _synthetic_page()
    mask, blks = det.detect(img, proj)
    assert blks, 'detector found nothing on the synthetic page'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    claimed = np.zeros(gray.shape, dtype=bool)
    for i in range(6):  # the synthetic text lines, exact geometry
        x = 180 + i * 22
        claimed[214:386, max(x - 6, 0):x + 7] = True
    missed = int(((gray < 160) & claimed & ~(mask > 0)).sum())
    assert missed == 0, f'{missed} text ink px left outside the inpaint mask'


def test_stray_mask_fragment_does_not_stretch_block_box():
    """Only mask fragments touching the detection may widen its box.

    Reproduces page 1 of 第4.3話: one stray mask pixel 970 px from its
    detection stretched a bottom-row block into the top panel, so the block
    spanned several bubbles and was labelled outside its own.
    """
    from custom_modules.detector_koharu_layout import _nearby_mask_extent

    mask = np.zeros((1300, 1200), bool)
    mask[1080:1224, 1085:1106] = True   # the detection's own mask
    mask[110, 1116] = True              # stray seg-head fragment, far away
    box = (1085, 1080, 1105, 1223)
    assert _nearby_mask_extent(mask, box) == (1085, 1080, 1106, 1224)

    # Legitimate spill: mask reaching past the box, whether attached to it or
    # a fragment a few dozen pixels away, is kept.
    mask2 = np.zeros((1300, 1200), bool)
    mask2[1070:1240, 1085:1106] = True
    assert _nearby_mask_extent(mask2, box) == (1085, 1070, 1106, 1240)
    mask3 = np.zeros((1300, 1200), bool)
    mask3[1080:1224, 1085:1106] = True
    mask3[1040:1070, 1085:1106] = True   # detached, 10 px above the box
    assert _nearby_mask_extent(mask3, box) == (1085, 1040, 1106, 1224)

    assert _nearby_mask_extent(np.zeros((4, 4), bool), (0, 0, 2, 2)) is None
    assert _nearby_mask_extent(None, box) is None


def test_split_two_lobed_separates_joined_bubbles():
    from custom_modules.detector_koharu_layout import _split_two_lobed

    a = cv2.ellipse2Poly((30, 40), (30, 30), 0, 0, 360, 30)
    b = cv2.ellipse2Poly((70, 40), (30, 30), 0, 0, 360, 30)
    mask = np.zeros((80, 110), np.uint8)
    cv2.fillPoly(mask, [a, b], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    joined = contours[0].reshape(-1, 2).tolist()

    parts = _split_two_lobed(joined)
    assert parts is not None and len(parts) == 2
    total = abs(cv2.contourArea(np.asarray(joined, np.float32)))
    area_sum = sum(abs(cv2.contourArea(np.asarray(p, np.float32))) for p in parts)
    assert abs(area_sum - total) < total * 0.05  # chord has no area: exact partition
    # both parts sit over their own lobe, not the union center
    xs = [int(np.asarray(p)[:, 0].mean()) for p in parts]
    assert min(xs) < 55 < max(xs)
    # a single bubble never splits
    assert _split_two_lobed(a.reshape(-1, 2).tolist()) is None


def test_weak_bubble_kept_only_when_head_stacks_it():
    from custom_modules.detector_koharu_layout import (
        _selected_bubble_instances,
    )

    page11 = [
        {'box': [178, 104, 461, 504], 'conf': 0.715},
        {'box': [914, 949, 1170, 1416], 'conf': 0.387},
        {'box': [892, 946, 1171, 1461], 'conf': 0.301},
        {'box': [905, 949, 1172, 1458], 'conf': 0.289},
        {'box': [903, 945, 1167, 1453], 'conf': 0.210},
    ]
    selected = _selected_bubble_instances(page11, 0.5)
    # one per nested cluster: the strong one, plus the weak stack
    # represented by its highest-confidence (tightest) instance
    assert [c['conf'] for c in selected] == [0.715, 0.387]
    # a lone weak instance stays rejected: no second opinion, no outline
    lone = [{'box': [0, 0, 10, 10], 'conf': 0.387}]
    assert _selected_bubble_instances(lone, 0.5) == []
