import os
import unittest
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from ballontranslator.utils.textblock import _mask_coverage


class TestMaskCoverage(unittest.TestCase):
    """A degenerate detected box must not produce a NaN coverage score.

    Zero-height or zero-width lines slice the mask to an empty array; their
    mean is NaN, so the score comparison silently passed while numpy warned.
    """

    def test_degenerate_box_is_zero_not_nan(self) -> None:
        mask = np.zeros((10, 10), np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertEqual(_mask_coverage(mask, 2, 2, 2, 8), 0.0)  # zero width
            self.assertEqual(_mask_coverage(mask, 2, 2, 8, 2), 0.0)  # zero height
            self.assertEqual(_mask_coverage(mask, 5, 5, 5, 5), 0.0)  # point

    def test_normal_box_measures_coverage(self) -> None:
        mask = np.zeros((10, 10), np.uint8)
        mask[0:5, 0:5] = 255  # half the box is covered
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            self.assertAlmostEqual(_mask_coverage(mask, 0, 0, 5, 5), 1.0)
            self.assertAlmostEqual(_mask_coverage(mask, 0, 0, 10, 10), 0.25)

    def test_out_of_range_box_does_not_warn(self) -> None:
        # A box past the mask edge clips to empty; still no warning.
        mask = np.zeros((10, 10), np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            _mask_coverage(mask, 20, 20, 30, 30)
