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
from ballontranslator.utils.imgproc_utils import extract_ballon_region
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
            imgtrans_proj=types.SimpleNamespace(
                img_array=self.img, current_img='001.png',
                get_bubble_outlines=lambda page: [],
            ),
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
                self.assertGreaterEqual(size, entry * 0.15 - 0.5)
                self.assertLessEqual(canvas_h, 2 * BALLOON_RY + 5)
                self.assertLessEqual(canvas_w, 2 * BALLOON_RX + 10)

    def test_text_that_already_fits_is_not_shrunk(self) -> None:
        _, size, _, _ = self._layout('OK', fntsize_flag=0)
        self.assertEqual(size, 30.0)

    def test_persisted_rich_text_does_not_block_fitted_size(self) -> None:
        # Re-runs load the previous render's rich_text, so the document is
        # non-empty when layout runs. The fitted size must still reach the
        # new text and the document default font the next pass reads.
        pcfg.let_fntsize_flag = 0
        block = TextBlock(BLOCK_BBOX)
        block.set_lines_by_xywh([
            BLOCK_BBOX[0], BLOCK_BBOX[1],
            BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
        ])
        block.fontformat.font_size = 40
        block.rich_text = '<p style="font-size: 80pt;">OLD TEXT OLD TEXT</p>'
        item = TextBlkItem(block, 0)
        self.assertFalse(item.document().isEmpty())
        entry = item.font().pointSizeF()
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(
                img_array=self.img, current_img='001.png',
                get_bubble_outlines=lambda page: [],
            ),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        result = SceneTextManager.layout_textblk(stub, item, text=LONG_TEXT)
        self.assertIs(result, True)
        self.assertNotIn('OLD', item.toPlainText())
        self.assertLess(item.font().pointSizeF(), entry)
        self.assertLess(item.document().defaultFont().pointSizeF(), entry)

    def test_floor_shrink_lands_on_floor_when_first_step_overshoots(self) -> None:
        # A big fresh-detected source font makes the first ratio step dip
        # below the readability floor; the fit must clamp onto the floor
        # instead of keeping the unfitted size (real koharu pages hit this).
        pcfg.let_fntsize_flag = 0
        block = TextBlock(BLOCK_BBOX)
        block.set_lines_by_xywh([
            BLOCK_BBOX[0], BLOCK_BBOX[1],
            BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
        ])
        block.fontformat.font_size = 400  # 300pt entry, floor at 45pt
        item = TextBlkItem(block, 0)
        entry = item.font().pointSizeF()
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(
                img_array=self.img, current_img='001.png',
                get_bubble_outlines=lambda page: [],
            ),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        result = SceneTextManager.layout_textblk(stub, item, text=LONG_TEXT * 3)
        self.assertIs(result, True)
        size = item.font().pointSizeF()
        self.assertLess(size, entry)
        self.assertAlmostEqual(size, entry * 0.15, delta=1.0)

    def test_text_is_centered_on_the_balloon_not_the_box(self) -> None:
        # Detection boxes sit off the balloon center; the rendered text box
        # must end up centered on the balloon anyway. The balloon is small
        # and off-center so it stays fully inside the search window (a
        # clipped balloon would skew the mask centroid).
        pcfg.let_fntsize_flag = 0
        img = np.full((IMG_H, IMG_W, 3), 130, np.uint8)
        cv2.ellipse(img, (215, 180), (60, 55), 0, 0, 360, (255, 255, 255), -1)
        block = TextBlock(BLOCK_BBOX)  # center (250, 180) vs balloon (215, 180)
        block.set_lines_by_xywh([
            BLOCK_BBOX[0], BLOCK_BBOX[1],
            BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
        ])
        block.fontformat.font_size = 40
        item = TextBlkItem(block, 0)
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(
                img_array=img, current_img='001.png',
                get_bubble_outlines=lambda page: [],
            ),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        result = SceneTextManager.layout_textblk(stub, item, text=LONG_TEXT)
        self.assertIs(result, True)
        br = item.absBoundingRect(qrect=True)
        self.assertAlmostEqual(br.x() + br.width() / 2, 215, delta=8)
        self.assertAlmostEqual(br.y() + br.height() / 2, 180, delta=8)

    def test_detector_bubble_outline_drives_fit_when_available(self) -> None:
        # On an all-white page the flood fill degenerates to the whole
        # search window; the detector's bubble polygon must drive the fit
        # size and the centering instead.
        pcfg.let_fntsize_flag = 0
        img = np.full((IMG_H, IMG_W, 3), 255, np.uint8)
        poly = [[160, 120], [270, 120], [270, 240], [160, 240]]  # center (215, 180)
        block = TextBlock(BLOCK_BBOX)  # center (250, 180)
        block.set_lines_by_xywh([
            BLOCK_BBOX[0], BLOCK_BBOX[1],
            BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
        ])
        block.fontformat.font_size = 40
        item = TextBlkItem(block, 0)
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(
                img_array=img, current_img='001.png',
                get_bubble_outlines=lambda page: [poly],
            ),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        result = SceneTextManager.layout_textblk(stub, item, text=LONG_TEXT)
        self.assertIs(result, True)
        br = item.absBoundingRect(qrect=True)
        self.assertAlmostEqual(br.x() + br.width() / 2, 215, delta=8)
        self.assertAlmostEqual(br.y() + br.height() / 2, 180, delta=8)
        fm = QFontMetricsF(item.font())
        lines = item.toPlainText().split('\n')
        canvas_w = max(fm.horizontalAdvance(ln) for ln in lines)
        canvas_h = int(round(fm.height() * len(lines)))
        self.assertLessEqual(canvas_w, 110 + 10)
        self.assertLessEqual(canvas_h, 120 + 5)

    def test_elliptical_typesetting_follows_bubble_outline(self) -> None:
        # With elliptical typesetting on, lines curve with the outline: the
        # middle line runs widest, the edge lines stay short.
        pcfg.let_fntsize_flag = 0
        old_elliptic = pcfg.let_elliptic_layout
        pcfg.let_elliptic_layout = True
        try:
            img = np.full((IMG_H, IMG_W, 3), 255, np.uint8)
            poly = cv2.ellipse2Poly(
                (BALLOON_CX, BALLOON_CY), (95, 60), 0, 0, 360, 30
            ).tolist()
            block = TextBlock(BLOCK_BBOX)
            block.set_lines_by_xywh([
                BLOCK_BBOX[0], BLOCK_BBOX[1],
                BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
            ])
            block.fontformat.font_size = 40
            item = TextBlkItem(block, 0)
            stub = types.SimpleNamespace(
                imgtrans_proj=types.SimpleNamespace(
                    img_array=img, current_img='001.png',
                    get_bubble_outlines=lambda page: [poly],
                ),
                pairwidget_list=[],
                auto_textlayout_flag=True,
            )
            result = SceneTextManager.layout_textblk(stub, item, text=LONG_TEXT)
            self.assertIs(result, True)
            lines = item.toPlainText().split('\n')
            self.assertGreaterEqual(len(lines), 3)
            fm = QFontMetricsF(item.font())
            widths = [fm.horizontalAdvance(ln) for ln in lines]
            self.assertGreater(max(widths), widths[0])
            self.assertGreater(max(widths), widths[-1])
            self.assertLessEqual(max(widths), 190 + 10)
        finally:
            pcfg.let_elliptic_layout = old_elliptic

    def test_lines_stay_inside_a_narrow_outline(self) -> None:
        # The first word of a line lands before any width cap runs, and the
        # coverage plateau used to accept lines the collision test just
        # rejected: wide words spilled out of narrow bubbles. Every rendered
        # line endpoint must end up inside the outline.
        pcfg.let_fntsize_flag = 0
        old_elliptic = pcfg.let_elliptic_layout
        old_target = pcfg.module.translate_target
        pcfg.let_elliptic_layout = True
        pcfg.module.translate_target = 'en'
        try:
            img = np.full((IMG_H, IMG_W, 3), 255, np.uint8)
            poly = cv2.ellipse2Poly(
                (BALLOON_CX, BALLOON_CY), (75, 95), 0, 0, 360, 30
            ).tolist()
            block = TextBlock(BLOCK_BBOX)
            block.set_lines_by_xywh([
                BLOCK_BBOX[0], BLOCK_BBOX[1],
                BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
            ])
            block.fontformat.font_size = 40
            item = TextBlkItem(block, 0)
            stub = types.SimpleNamespace(
                imgtrans_proj=types.SimpleNamespace(
                    img_array=img, current_img='001.png',
                    get_bubble_outlines=lambda page: [poly],
                ),
                pairwidget_list=[],
                auto_textlayout_flag=True,
            )
            result = SceneTextManager.layout_textblk(
                stub, item, text="e-cigarette smoking is unquestionably bad"
            )
            self.assertIs(result, True)
            poly_arr = np.asarray(poly, np.float32)
            br = item.absBoundingRect(qrect=True)
            fm = QFontMetricsF(item.font())
            lines = item.toPlainText().split('\n')
            cx = br.x() + br.width() / 2
            for i, ln in enumerate(lines):
                lw = fm.horizontalAdvance(ln)
                row = br.y() + (i + 0.5) * br.height() / len(lines)
                for edge_x in (cx - lw / 2, cx + lw / 2):
                    depth = cv2.pointPolygonTest(
                        poly_arr, (float(edge_x), float(row)), False
                    )
                    self.assertGreaterEqual(
                        depth, 0,
                        f'line {i} {ln!r} endpoint {edge_x:.0f} outside outline',
                    )
        finally:
            pcfg.let_elliptic_layout = old_elliptic
            pcfg.module.translate_target = old_target

    def test_small_starting_font_grows_to_fill_outline(self) -> None:
        # The pre-fit heuristic only shrinks; an outline fit must grow a
        # too-small starting font until the bubble is filled, with the
        # collision probes keeping every grown step inside the outline.
        pcfg.let_fntsize_flag = 0
        img = np.full((IMG_H, IMG_W, 3), 255, np.uint8)
        poly = [[150, 100], [350, 100], [350, 260], [150, 260]]  # 200x160
        block = TextBlock(BLOCK_BBOX)
        block.set_lines_by_xywh([
            BLOCK_BBOX[0], BLOCK_BBOX[1],
            BLOCK_BBOX[2] - BLOCK_BBOX[0], BLOCK_BBOX[3] - BLOCK_BBOX[1],
        ])
        block.fontformat.font_size = 8
        item = TextBlkItem(block, 0)
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(
                img_array=img, current_img='001.png',
                get_bubble_outlines=lambda page: [poly],
            ),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        result = SceneTextManager.layout_textblk(stub, item, text=LONG_TEXT)
        self.assertIs(result, True)
        self.assertGreaterEqual(item.font().pointSizeF(), 12.0)
        poly_arr = np.asarray(poly, np.float32)
        br = item.absBoundingRect(qrect=True)
        fm = QFontMetricsF(item.font())
        lines = item.toPlainText().split('\n')
        cx = br.x() + br.width() / 2
        for i, ln in enumerate(lines):
            lw = fm.horizontalAdvance(ln)
            row = br.y() + (i + 0.5) * br.height() / len(lines)
            for edge_x in (cx - lw / 2, cx + lw / 2):
                self.assertGreaterEqual(
                    cv2.pointPolygonTest(poly_arr, (float(edge_x), float(row)), False),
                    0, f'line {i} {ln!r} outside outline after growth',
                )

    def test_balloon_mask_survives_a_large_search_window(self) -> None:
        # A balloon-sized contour is ~1/4 of the 2.25x window here; the
        # selection must keep the balloon instead of degrading the mask to
        # the whole window (which lets layout accept out-of-balloon text).
        img = np.full((400, 500, 3), 130, np.uint8)
        cv2.ellipse(img, (250, 180), (80, 55), 0, 0, 360, (255, 255, 255), -1)
        rect = [175, 130, 150, 100]  # x, y, w, h inside the balloon
        mask, area, mask_xyxy = extract_ballon_region(img, rect, enlarge_ratio=2.25)
        window_area = (mask_xyxy[2] - mask_xyxy[0]) * (mask_xyxy[3] - mask_xyxy[1])
        self.assertLess(area, window_area * 0.5)
        self.assertGreater(area, 0.5 * np.pi * 80 * 55)
        cx, cy = 250 - mask_xyxy[0], 180 - mask_xyxy[1]
        self.assertGreater(int(mask[cy, cx] > 0), 0)  # balloon center is on
        self.assertEqual(int(mask[0, 0] > 0), 0)       # window corner is off


if __name__ == '__main__':
    unittest.main()
