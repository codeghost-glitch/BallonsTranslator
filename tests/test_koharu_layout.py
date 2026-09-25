"""Regression: koharu block boxes must contain the detector's own mask.

The pipeline zeroes a removed block's bounding_rect() out of the page mask
after OCR; any mask pixel outside every block box survives that step and is
inpainting the removed region as artifacts.
"""
import cv2
import numpy as np

from ballontranslator.modules import TEXTDETECTORS


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
    mask, blks = det.detect(_synthetic_page())
    assert blks, 'detector found nothing on the synthetic page'
    covered = np.zeros(mask.shape, dtype=bool)
    for blk in blks:
        x, y, w, h = blk.bounding_rect()
        covered[y:y + h, x:x + w] = True
    stray = (mask > 0) & ~covered
    assert not stray.any(), f'{int(stray.sum())} mask px outside every block box'
