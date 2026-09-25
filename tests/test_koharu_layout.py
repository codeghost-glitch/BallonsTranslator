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
