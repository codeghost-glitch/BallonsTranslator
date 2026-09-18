import unittest
from unittest.mock import patch

import numpy as np

from ballontranslator.modules.inpaint.base import InpainterBase
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock import TextBlock


class MaskFillingInpainter(InpainterBase):
    params = {}

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list=None) -> np.ndarray:
        result = img.copy()
        result[mask > 0] = 255
        return result


class PipelineInpaintMaskTests(unittest.TestCase):
    def test_pipeline_covers_mask_outside_boxes_like_manual_and_preserves_input(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[22:28, 22:28] = 255
        mask[70:75, 70:75] = 255
        original = mask.copy()
        inpainter = MaskFillingInpainter()
        with patch.object(pcfg.module, 'check_need_inpaint', False), patch.object(pcfg.module, 'filter_mask_by_bboxes', False):
            manual = inpainter.inpaint(image, mask)
            pipeline = inpainter.inpaint(image, mask, [TextBlock(xyxy=[20, 20, 30, 30])])
        np.testing.assert_array_equal(pipeline, manual)
        np.testing.assert_array_equal(mask, original)

    def test_empty_block_list_still_processes_existing_mask(self) -> None:
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:15, 10:15] = 255
        inpainter = MaskFillingInpainter()
        result = inpainter.inpaint(image, mask, [])
        self.assertTrue(np.all(result[mask > 0] == 255))

    def test_overlapping_blocks_do_not_inpaint_cleared_regions_again(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[12:18, 12:18] = 255
        inpainter = MaskFillingInpainter()
        with patch.object(pcfg.module, 'check_need_inpaint', False), patch.object(pcfg.module, 'filter_mask_by_bboxes', False), patch.object(inpainter, '_inpaint', wraps=inpainter._inpaint) as calls:
            inpainter.inpaint(image, mask, [TextBlock(xyxy=[10, 10, 20, 20]), TextBlock(xyxy=[11, 11, 21, 21])])
        self.assertEqual(calls.call_count, 1)


if __name__ == '__main__':
    unittest.main()
