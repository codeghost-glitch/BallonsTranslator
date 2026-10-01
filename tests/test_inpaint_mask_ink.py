import unittest

import cv2
import numpy as np

from ballontranslator.modules.inpaint.base import complete_mask_on_ink


class TestInpaintMaskInkCompletion(unittest.TestCase):

    def test_nearby_missed_stroke_joins_mask_with_clearance(self) -> None:
        img = np.full((40, 60, 3), 255, np.uint8)
        img[10, 33:50] = 0  # glyph: tail the detector missed, body under the mask
        mask = np.zeros((40, 60), np.uint8)
        mask[10, 42:50] = 255
        out = complete_mask_on_ink(img, mask)
        self.assertTrue((out[10, 33:41] >= 128).all())
        # The joined ink must sit clear of the boundary so the model does
        # not fill weakly on it (measured: lama needs ~4 px).
        dist = cv2.distanceTransform((out >= 128).astype(np.uint8), cv2.DIST_L2, 3)
        self.assertGreaterEqual(dist[10, 33:41].min(), 4)

    def test_balloon_stroke_within_the_halo_is_not_absorbed(self) -> None:
        # A balloon stroke passing the text sits well inside the halo but is
        # its own ink run: pulling it in punches a gap in the bubble edge.
        img = np.full((40, 60, 3), 255, np.uint8)
        img[10, 33:50] = 0
        img[17:20, 34] = 0
        mask = np.zeros((40, 60), np.uint8)
        mask[10, 42:50] = 255
        out = complete_mask_on_ink(img, mask)
        self.assertFalse((out[17:20, 34] >= 128).any())

    def test_long_thin_stroke_touching_the_mask_is_not_absorbed(self) -> None:
        # A balloon outline that TOUCHES the mask (the detector's ksize
        # dilation already nudges the text mask onto the bubble edge) is one
        # long, hollow curve. Growing it punches a hole in the drawing. The
        # mask sits just inside the balloon and reaches its stroke, as it does
        # on a real page; the stroke must not be pulled into the mask.
        img = np.full((200, 200, 3), 255, np.uint8)
        # Balloon outline: a large, thin, hollow ellipse.
        cv2.ellipse(img, (100, 100), (70, 80), 0, 0, 360, (0, 0, 0), 3)
        # A text mask inside the balloon whose right edge is dilated onto the
        # balloon's right stroke (stroke sits at x ~167-173).
        mask = np.zeros((200, 200), np.uint8)
        mask[80:120, 140:172] = 255
        out = complete_mask_on_ink(img, mask)
        # Every stroke pixel the base mask did not already cover must stay
        # out: the balloon edge is not erased.
        base_cover = mask > 0
        stroke_pixels = (img[:, :, 0] < 128)
        erased = (out > 0) & stroke_pixels & ~base_cover
        self.assertEqual(int(erased.sum()), 0,
                         'balloon stroke absorbed into the inpaint mask')

    def test_far_ink_is_untouched(self) -> None:
        img = np.full((40, 60, 3), 255, np.uint8)
        img[30, 5:12] = 0
        mask = np.zeros((40, 60), np.uint8)
        mask[10, 42:50] = 255
        out = complete_mask_on_ink(img, mask)
        self.assertFalse((out[30, 5:12] >= 128).any())

    def test_light_on_dark_text_is_detected(self) -> None:
        img = np.full((40, 60, 3), 30, np.uint8)
        img[10, 33:50] = 250
        mask = np.zeros((40, 60), np.uint8)
        mask[10, 42:50] = 255
        out = complete_mask_on_ink(img, mask)
        self.assertTrue((out[10, 33:41] >= 128).all())

    def test_noop_without_mask_or_nearby_ink(self) -> None:
        img = np.full((40, 60, 3), 255, np.uint8)
        mask = np.zeros((40, 60), np.uint8)
        self.assertTrue(np.array_equal(complete_mask_on_ink(img, mask), mask))
        self.assertIsNone(complete_mask_on_ink(img, None))

if __name__ == '__main__':
    unittest.main()
