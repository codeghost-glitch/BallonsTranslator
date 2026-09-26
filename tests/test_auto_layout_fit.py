import os
import types
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from qtpy.QtGui import QFontMetricsF
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock import TextAlignment, TextBlock

# White ellipse balloon (300x200) on a gray page; the text block sits inside it.
IMG_H, IMG_W = 400, 500
BALLOON_CX, BALLOON_CY, BALLOON_RX, BALLOON_RY = 250, 180, 150, 100
BLOCK_BBOX = [170, 130, 330, 230]
LONG_TEXT = "DIDN'T I TELL YOU TO STOP SLEEPING ON THE FLOOR?"


class TestAutoLayoutFit(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_flags = {
            'let_fntsize_flag': pcfg.let_fntsize_flag,
            'let_autolayout_flag': pcfg.let_autolayout_flag,
        }
        self.old_source = pcfg.module.translate_source
        self.old_target = pcfg.module.translate_target
        pcfg.module.translate_source = 'ja'
        pcfg.module.translate_target = 'en'
        pcfg.let_autolayout_flag = True
        self.img = np.full((IMG_H, IMG_W, 3), 90, np.uint8)
        cv2.ellipse(
            self.img,
            (BALLOON_CX, BALLOON_CY),
            (BALLOON_RX, BALLOON_RY),
            0, 0, 360, (255, 255, 255), -1,
        )

    def tearDown(self) -> None:
        for key, value in self.old_flags.items():
            setattr(pcfg, key, value)
        pcfg.module.translate_source = self.old_source
        pcfg.module.translate_target = self.old_target

    @staticmethod
    def _make_item(alignment: TextAlignment = None) -> TextBlkItem:
        block = TextBlock(BLOCK_BBOX)
        block.set_lines_by_xywh([
            BLOCK_BBOX[0], BLOCK_BBOX[1],
            BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
        ])
        block.fontformat.font_size = 40
        if alignment is not None:
            block.fontformat.alignment = alignment
        return TextBlkItem(block, 0)

    def _layout(self, text: str, fntsize_flag: int, alignment: TextAlignment = None):
        pcfg.let_fntsize_flag = fntsize_flag
        item = self._make_item(alignment)
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(img_array=self.img),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        result = SceneTextManager.layout_textblk(stub, item, text=text)
        self.assertIs(result, True)
        font = item.font()
        fm = QFontMetricsF(font)
        lines = item.toPlainText().split('\n')
        canvas_w = max(fm.horizontalAdvance(ln) for ln in lines)
        canvas_h = int(round(fm.height() * len(lines)))
        return item, font.pointSizeF(), canvas_w, canvas_h

    def test_global_font_setting_is_never_resized(self) -> None:
        # "use global setting" (flag == 1) opts out of adaptive sizing, so the
        # program must keep the font size even though the text overflows.
        _, size, _, canvas_h = self._layout(LONG_TEXT, fntsize_flag=1)
        self.assertEqual(size, 30.0)
        self.assertGreater(canvas_h, 2 * BALLOON_RY)

    def test_decide_by_program_shrinks_text_into_the_balloon(self) -> None:
        for alignment in (None, TextAlignment.Center):
            with self.subTest(alignment=alignment):
                entry = self._make_item(alignment).font().pointSizeF()
                _, size, canvas_w, canvas_h = self._layout(
                    LONG_TEXT, fntsize_flag=0, alignment=alignment
                )
                self.assertLess(size, entry)
                self.assertGreaterEqual(size, entry * 0.3 - 0.5)
                self.assertLessEqual(canvas_h, 2 * BALLOON_RY + 5)
                self.assertLessEqual(canvas_w, 2 * BALLOON_RX + 10)

    def test_text_that_already_fits_is_not_shrunk(self) -> None:
        _, size, _, _ = self._layout('OK', fntsize_flag=0)
        self.assertEqual(size, 30.0)


if __name__ == '__main__':
    unittest.main()
