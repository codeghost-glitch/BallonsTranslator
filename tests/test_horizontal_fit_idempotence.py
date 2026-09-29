"""Re-running the pipeline must not keep changing a block's font size.

The pipeline saves font_size back into the block and re-uses it as the next
run's starting size. If the fit is not idempotent, every re-run drifts:
feeding a result back in yields a different size. The contract that text
which already fits is left alone is preserved - this only pins stability.
"""
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock import TextBlock

CASES = [
    ("Yeah, I never know when you'll start crying because you're hungry, "
     "Yotsuba-chan.", [143, 146, 165, 266]),
    ('And make sure to stock up on drinking water frequently.',
     [839, 1106, 92, 235]),
    ('Food is important. We should prioritize securing anything we can eat.',
     [969, 1035, 137, 293]),
]


class _Stub:
    auto_textlayout_flag = True
    pairwidget_list: list = []
    addTextBlkItem = staticmethod(lambda item: None)
    canvas = type('C', (), {'textblock_mode': True})()

    _vertical_fit_balloon_box = SceneTextManager._vertical_fit_balloon_box
    _vertical_column_rects = staticmethod(SceneTextManager._vertical_column_rects)
    _layout_textblk_vertical = SceneTextManager._layout_textblk_vertical


class TestHorizontalFitIdempotence(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old = (pcfg.let_fntsize_flag, pcfg.let_autolayout_flag,
                    pcfg.module.translate_source, pcfg.module.translate_target)
        pcfg.let_fntsize_flag = 0
        pcfg.let_autolayout_flag = True
        pcfg.module.translate_source = '日本語'
        pcfg.module.translate_target = 'English'
        self.stub = _Stub()
        self.stub.imgtrans_proj = type('P', (), {})()
        self.stub.imgtrans_proj.current_img = 'p'

    def tearDown(self) -> None:
        (pcfg.let_fntsize_flag, pcfg.let_autolayout_flag,
         pcfg.module.translate_source,
         pcfg.module.translate_target) = self.old

    def _fit(self, text, bbox, start_font):
        bx, by, bw, bh = bbox
        img = np.full((by + bh + 200, bx + bw + 200, 3), 255, np.uint8)
        cv2.ellipse(img, (bx + bw // 2, by + bh // 2),
                    (max(4, bw // 2), max(4, bh // 2)), 0, 0, 360,
                    (0, 0, 0), -1)
        outline = [(bx + bw // 2 + (bw / 2) * np.cos(a),
                   by + bh // 2 + (bh / 2) * np.sin(a))
                   for a in np.linspace(0, 2 * np.pi, 65)[:-1]]
        self.stub.imgtrans_proj.img_array = img
        self.stub.imgtrans_proj.get_bubble_outlines = lambda pg: [outline]

        blk = TextBlock([bx, by, bx + bw, by + bh])
        blk.set_lines_by_xywh([bx, by, bw, bh])
        blk._detected_bbox = [bx, by, bw, bh]
        blk.fontformat.vertical = False
        blk.fontformat.alignment = 1
        blk.src_is_vertical = True
        blk.rich_text = ''
        blk.translation = text
        item = TextBlkItem(blk, 0)
        item.setPlainText('')
        item.setFontSize(start_font)
        rst = SceneTextManager.layout_textblk(self.stub, item, text=text)
        if rst is None:
            item.setPlainText(text)
        return item.font().pointSizeF()

    def test_repeated_runs_do_not_drift(self) -> None:
        # Pipeline behaviour: each run feeds the previous run's stored size
        # back in as the next starting size.
        for text, bbox in CASES:
            with self.subTest(text=text[:24]):
                size = self._fit(text, bbox, 24.0)
                for run in range(2, 6):
                    nxt = self._fit(text, bbox, size)
                    self.assertAlmostEqual(
                        nxt, size, delta=0.51,
                        msg='run %d drifted %.2f -> %.2f' % (run, size, nxt))
                    size = nxt
