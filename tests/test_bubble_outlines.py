"""Contract: bubble outlines validate at the project boundary and render as
canvas paths on the current page."""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np

from ballontranslator.utils.proj_imgtrans import ImgnameNotInProjectException, ProjImgTrans
from custom_modules.detector_koharu_layout import _mask_outline


class TestMaskOutline(unittest.TestCase):

    def test_blob_contour_and_empty_fallback(self):
        mask = np.zeros((128, 128), np.uint8)
        cv2.circle(mask, (64, 64), 40, 1, -1)
        poly = _mask_outline(mask)
        self.assertIsNotNone(poly)
        self.assertGreaterEqual(len(poly), 3)
        pts = np.array(poly)
        self.assertGreaterEqual(pts[:, 0].min(), 15)
        self.assertLessEqual(pts[:, 0].max(), 113)
        self.assertIsNone(_mask_outline(np.zeros((8, 8), np.uint8)))


class TestBubbleOutlinesProjectBoundary(unittest.TestCase):

    def _proj(self) -> ProjImgTrans:
        proj = ProjImgTrans()
        proj._image_info = {'001.png': {}}
        return proj

    def test_roundtrip_clear_and_detached_copy(self):
        proj = self._proj()
        poly = [[10, 10], [50, 12], [46, 44], [12, 40]]
        proj.set_bubble_outlines('001.png', [poly])
        self.assertEqual(proj.get_bubble_outlines('001.png'), [poly])
        got = proj.get_bubble_outlines('001.png')
        got[0][0][0] = -1
        self.assertEqual(proj.get_bubble_outlines('001.png'), [poly])
        proj.set_bubble_outlines('001.png', [])
        self.assertEqual(proj.get_bubble_outlines('001.png'), [])

    def test_write_boundary_rejects_malformed(self):
        proj = self._proj()
        for bad in ([[[1, 2]]], 'nope', [1, 2], [[[0, 0], [1]]], [[[0, 0], [1, 1], [True, 2]]]):
            with self.assertRaises(ValueError, msg=f'accepted {bad!r}'):
                proj.set_bubble_outlines('001.png', bad)
        with self.assertRaises(ImgnameNotInProjectException):
            proj.set_bubble_outlines('999.png', [])

    def test_malformed_stored_record_dropped_rest_kept(self):
        proj = self._proj()
        proj._image_info['001.png'] = {'bubble_outlines': [[['x']]], 'finish_code': 3}
        self.assertEqual(proj.get_bubble_outlines('001.png'), [])
        self.assertNotIn('bubble_outlines', proj._image_info['001.png'])
        self.assertEqual(proj._image_info['001.png']['finish_code'], 3)
        self.assertEqual(proj.get_bubble_outlines('999.png'), [])

    def test_prune_drops_bubbles_without_surviving_text(self):
        from ballontranslator.utils.textblock import TextBlock
        proj = self._proj()
        inner = [[0, 0], [100, 0], [100, 100], [0, 100]]
        empty = [[200, 0], [300, 0], [300, 100], [200, 100]]
        proj.set_bubble_outlines('001.png', [inner, empty])
        proj.pages['001.png'] = [TextBlock(xyxy=[10, 10, 60, 60])]
        proj.prune_bubble_outlines('001.png')
        self.assertEqual(proj.get_bubble_outlines('001.png'), [inner])

        # Every block gone (punctuation-only exception) clears all outlines.
        proj.pages['001.png'] = []
        proj.prune_bubble_outlines('001.png')
        self.assertEqual(proj.get_bubble_outlines('001.png'), [])

        # Merged blocks span bubbles: attribution follows line geometry,
        # not the block center (which sits in the other bubble here).
        proj.set_bubble_outlines('001.png', [inner, empty])
        proj.pages['001.png'] = [
            TextBlock(xyxy=[0, 0, 300, 100],
                      lines=[[210, 10, 290, 10, 290, 90, 210, 90]])
        ]
        proj.prune_bubble_outlines('001.png')
        self.assertEqual(proj.get_bubble_outlines('001.png'), [empty])

        # Unknown page is a no-op.
        proj.prune_bubble_outlines('999.png')


class TestCanvasOutlineLayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from qtpy.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_path_follows_current_page(self):
        from ballontranslator.ui.canvas import Canvas
        proj = ProjImgTrans()
        proj._image_info = {'001.png': {}, '002.png': {}}
        proj.set_bubble_outlines('001.png', [[[10, 10], [50, 12], [46, 44], [12, 40]]])
        canvas = Canvas()
        canvas.imgtrans_proj = proj

        proj.current_img = '001.png'
        canvas._refresh_bubble_outlines()
        path = canvas.bubbleOutlineLayer.path()
        self.assertFalse(path.isEmpty())
        self.assertGreaterEqual(path.elementCount(), 4)

        # Page without outlines clears the layer.
        proj.current_img = '002.png'
        canvas._refresh_bubble_outlines()
        self.assertTrue(canvas.bubbleOutlineLayer.path().isEmpty())


if __name__ == '__main__':
    unittest.main()
