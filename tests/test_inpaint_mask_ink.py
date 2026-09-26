import doctest
import unittest

import cv2
import numpy as np

from ballontranslator.modules.inpaint import base as inpaint_base
from ballontranslator.modules.inpaint.base import complete_mask_on_ink


class TestInpaintMaskInkCompletion(unittest.TestCase):

    def test_nearby_missed_stroke_joins_mask_with_clearance(self) -> None:
        img = np.full((40, 60, 3), 255, np.uint8)
        img[10, 33:41] = 0  # stroke tail the detector missed
        mask = np.zeros((40, 60), np.uint8)
        mask[10, 42:50] = 255
        out = complete_mask_on_ink(img, mask)
        self.assertTrue((out[10, 33:41] >= 128).all())
        # The joined ink must sit clear of the boundary so the model does
        # not fill weakly on it (measured: lama needs ~4 px).
        dist = cv2.distanceTransform((out >= 128).astype(np.uint8), cv2.DIST_L2, 3)
        self.assertGreaterEqual(dist[10, 33:41].min(), 4)

    def test_far_ink_is_untouched(self) -> None:
        img = np.full((40, 60, 3), 255, np.uint8)
        img[30, 5:12] = 0
        mask = np.zeros((40, 60), np.uint8)
        mask[10, 42:50] = 255
        out = complete_mask_on_ink(img, mask)
        self.assertFalse((out[30, 5:12] >= 128).any())

    def test_light_on_dark_text_is_detected(self) -> None:
        img = np.full((40, 60, 3), 30, np.uint8)
        img[10, 33:41] = 250
        mask = np.zeros((40, 60), np.uint8)
        mask[10, 42:50] = 255
        out = complete_mask_on_ink(img, mask)
        self.assertTrue((out[10, 33:41] >= 128).all())

    def test_noop_without_mask_or_nearby_ink(self) -> None:
        img = np.full((40, 60, 3), 255, np.uint8)
        mask = np.zeros((40, 60), np.uint8)
        self.assertTrue(np.array_equal(complete_mask_on_ink(img, mask), mask))
        self.assertIsNone(complete_mask_on_ink(img, None))

    def test_module_doctests(self) -> None:
        results = doctest.testmod(inpaint_base, verbose=False)
        self.assertEqual(results.failed, 0)


if __name__ == '__main__':
    unittest.main()
