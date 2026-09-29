import doctest
import os
import types
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from qtpy.QtGui import QFontMetricsF
from qtpy.QtWidgets import QApplication

import ballontranslator.ui.text_engine.editing.manager as M
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

    def test_shape_aware_typesetting_follows_bubble_outline(self) -> None:
        # With shape-aware typesetting on, lines curve with the outline: the
        # middle line runs widest, the edge lines stay short - and the
        # layout must actually be accepted, not pile onto the readability
        # floor where single-word lines look curved by accident.
        pcfg.let_fntsize_flag = 0
        old_shape_aware = pcfg.let_shape_aware_layout
        pcfg.let_shape_aware_layout = True
        try:
            img = np.full((IMG_H, IMG_W, 3), 255, np.uint8)
            # The narrow 95x60 balloon floors this text at every size: the
            # unbreakable words overrun the edge rows and the probes never
            # clear. 120x75 fits, which is what this test is about.
            poly = cv2.ellipse2Poly(
                (BALLOON_CX, BALLOON_CY), (120, 75), 0, 0, 360, 6
            ).tolist()
            cv2.ellipse(img, (BALLOON_CX, BALLOON_CY), (120, 75), 0, 0, 360, (255, 255, 255), -1)
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
            # Accepted: nowhere near the 40 * 0.15 readability floor.
            self.assertGreater(item.font().pointSizeF(), 6.0)
            self.assertGreater(max(widths), widths[0])
            self.assertGreater(max(widths), widths[-1])
            self.assertLessEqual(max(widths), 240 + 10)
        finally:
            pcfg.let_shape_aware_layout = old_shape_aware

    def test_lines_stay_inside_a_narrow_outline(self) -> None:
        # The first word of a line lands before any width cap runs, and the
        # coverage plateau used to accept lines the collision test just
        # rejected: wide words spilled out of narrow bubbles. Every rendered
        # line endpoint must end up inside the outline.
        pcfg.let_fntsize_flag = 0
        old_shape_aware = pcfg.let_shape_aware_layout
        old_target = pcfg.module.translate_target
        pcfg.let_shape_aware_layout = True
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
            pcfg.let_shape_aware_layout = old_shape_aware
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


class TestAdvisoryHyphenation(unittest.TestCase):
    """Hyphenation is a fallback: it must never change a layout that fits.

    The split is derived per attempt and the unhyphenated layout runs
    first, so a plain layout is byte-identical with the feature off, and
    a split appears only where the unhyphenated text cannot fit at any
    readable size.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_target = pcfg.module.translate_target
        self.old_source = pcfg.module.translate_source
        self.old_fntsize = pcfg.let_fntsize_flag
        self.old_autolayout = pcfg.let_autolayout_flag
        pcfg.module.translate_source = 'ja'
        pcfg.module.translate_target = 'English'
        pcfg.let_fntsize_flag = 0
        pcfg.let_autolayout_flag = True

    def tearDown(self) -> None:
        pcfg.module.translate_source = self.old_source
        pcfg.module.translate_target = self.old_target
        pcfg.let_fntsize_flag = self.old_fntsize
        pcfg.let_autolayout_flag = self.old_autolayout

    @staticmethod
    def _fit(img, poly, bbox, lines, text, font_size=40):
        block = TextBlock(bbox)
        block.set_lines_by_xywh(lines)
        block.fontformat.font_size = font_size
        item = TextBlkItem(block, 0)
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(
                img_array=img, current_img='001.png',
                get_bubble_outlines=lambda page: [poly],
            ),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        SceneTextManager.layout_textblk(stub, item, text=text)
        rendered = item.toPlainText()
        return item.font().pointSizeF(), rendered.split('\n'), rendered

    def _wide_balloon(self):
        img = np.full((IMG_H, IMG_W, 3), 90, np.uint8)
        cv2.ellipse(img, (250, 180), (150, 100), 0, 0, 360, (255, 255, 255), -1)
        poly = cv2.ellipse2Poly((250, 180), (150, 100), 0, 0, 360, 6).reshape(-1, 2).tolist()
        return img, poly

    def _narrow_balloon(self):
        # 120x200: the compound still exceeds the line budget at the
        # readability floor, so the unhyphenated layout can never pass.
        img = np.full((IMG_H, IMG_W, 3), 90, np.uint8)
        cv2.ellipse(img, (250, 180), (60, 100), 0, 0, 360, (255, 255, 255), -1)
        poly = cv2.ellipse2Poly((250, 180), (60, 100), 0, 0, 360, 6).reshape(-1, 2).tolist()
        return img, poly

    def test_outline_budget_on_a_low_block_does_not_crash(self) -> None:
        # Window rows are window-local (0..N) but the outline is in page
        # coordinates. A block low on a tall page makes the two disjoint:
        # every row reads "no room", the wrap budget becomes 0.0, and
        # line_is_valid divides by zero.
        img = np.full((1800, 500, 3), 90, np.uint8)
        cv2.ellipse(img, (250, 1600), (150, 100), 0, 0, 360, (255, 255, 255), -1)
        poly = cv2.ellipse2Poly((250, 1600), (150, 100), 0, 0, 360, 6).reshape(-1, 2).tolist()
        size, lines, rendered = self._fit(
            img, poly, [170, 1550, 330, 1650], [170, 1550, 160, 100],
            'The station announced the delay to everyone waiting.',
        )
        self.assertTrue(rendered.strip(), 'text still laid out')
        self.assertGreater(size, 0)

    def test_plain_text_is_identical_with_and_without(self) -> None:
        img, poly = self._wide_balloon()
        text = 'The station announced the delay to everyone waiting.'
        with_h = self._fit(img, poly, [170, 130, 330, 230], [170, 130, 160, 100], text)
        self.assertNotIn('-', with_h[2])

        original = M.hyphenate_long_words
        M.hyphenate_long_words = lambda w, wl, m, l, b: (w, wl)
        try:
            without = self._fit(img, poly, [170, 130, 330, 230], [170, 130, 160, 100], text)
        finally:
            M.hyphenate_long_words = original
        self.assertEqual(with_h, without)

    def test_split_used_only_when_unhyphenated_cannot_fit(self) -> None:
        img, poly = self._narrow_balloon()
        text = 'Donaudampfschiffahrtsgesellschaft'
        size, lines, rendered = self._fit(
            img, poly, [200, 140, 300, 220], [200, 140, 100, 80], text
        )
        self.assertIn('-', rendered)
        self.assertGreater(len(lines), 1)
        self.assertGreater(size, 4.5, 'accepted above the readability floor')

    def test_non_hyphenating_language_is_never_split(self) -> None:
        # pyphen only models space-delimited scripts that use hyphens
        # (Latin/Cyrillic/Greek). Arabic and Thai must be left whole: the
        # old `PYPHEN_LANGS.get(target, 'en')` fallback handed them English
        # rules and force-chopped them mid-word. The splitter is gated on the
        # same validated hyphenator the line-breaker uses, so for these
        # targets it must not run at all.
        img, poly = self._narrow_balloon()
        for target in ('العربية', 'ภาษาไทย'):
            with self.subTest(target=target):
                pcfg.module.translate_target = target
                calls = []
                original = M.hyphenate_long_words
                M.hyphenate_long_words = (
                    lambda w, wl, m, l, b: (calls.append(l), (w, wl))[1]
                )
                try:
                    self._fit(img, poly, [200, 140, 300, 220],
                              [200, 140, 100, 80], 'averyveryverylongunbrokentoken')
                finally:
                    M.hyphenate_long_words = original
                self.assertEqual(calls, [], 'splitter ran for a non-hyphenating target')

        # English (a real hyphenating target) still reaches the splitter, so
        # the assertion above is not passing merely because the gate is dead.
        pcfg.module.translate_target = 'English'
        calls = []
        original = M.hyphenate_long_words
        M.hyphenate_long_words = lambda w, wl, m, l, b: (calls.append(l), (w, wl))[1]
        try:
            self._fit(img, poly, [200, 140, 300, 220],
                      [200, 140, 100, 80], 'averyveryverylongunbrokentoken')
        finally:
            M.hyphenate_long_words = original
        self.assertTrue(calls, 'English target must still reach the splitter')


class TestSharedOutlineFit(unittest.TestCase):
    """Blocks sharing one outline must not grow through each other.

    Each block enlarges its detection box into a window three times its
    size, so two columns in one balloon get overlapping windows and both
    grow into the middle: page 008 of 第2.2話 rendered its two columns
    57 px inside each other.
    """

    # Two narrow columns with a 20px gap in a balloon three times their
    # width - the geometry page 008 of 第2.2話 actually produced.
    LEFT = [160, 90, 290, 270]      # xyxy, first column (130 wide)
    RIGHT = [310, 90, 395, 270]     # xyxy, second column (85 wide)

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old = {
            'source': pcfg.module.translate_source,
            'target': pcfg.module.translate_target,
            'fntsize': pcfg.let_fntsize_flag,
            'autolayout': pcfg.let_autolayout_flag,
        }
        pcfg.module.translate_source = 'ja'
        pcfg.module.translate_target = 'en'
        pcfg.let_fntsize_flag = 0
        pcfg.let_autolayout_flag = True
        self.img = np.full((IMG_H, IMG_W, 3), 90, np.uint8)
        # Wider than the shared fixture's ellipse: the columns sit in its
        # two halves, so each still has room after the split.
        self.centre, self.radii = (250, 180), (170, 150)
        cv2.ellipse(
            self.img, self.centre, self.radii, 0, 0, 360, (255, 255, 255), -1,
        )
        self.poly = cv2.ellipse2Poly(
            self.centre, self.radii, 0, 0, 360, 6
        ).reshape(-1, 2).tolist()

    def tearDown(self) -> None:
        pcfg.module.translate_source = self.old['source']
        pcfg.module.translate_target = self.old['target']
        pcfg.let_fntsize_flag = self.old['fntsize']
        pcfg.let_autolayout_flag = self.old['autolayout']

    @staticmethod
    def _column(box) -> TextBlock:
        blk = TextBlock(box)
        blk.set_lines_by_xywh([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        blk.fontformat.font_size = 40
        blk._detected_bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        return blk

    @staticmethod
    def _ink_span(item):
        """Horizontal extent of the glyphs the item actually renders."""
        fm = QFontMetricsF(item.font())
        width = max(fm.horizontalAdvance(ln) for ln in item.toPlainText().split('\n'))
        rect = item.absBoundingRect(qrect=True)
        align = item.blk.fontformat.alignment
        if align == 2:
            return rect.x() + rect.width() - width, rect.x() + rect.width()
        if align != 1:
            return rect.x(), rect.x() + width
        centre = rect.x() + rect.width() / 2
        return centre - width / 2, centre + width / 2

    def _layout_pair(self):
        blocks = [self._column(self.LEFT), self._column(self.RIGHT)]
        stub = types.SimpleNamespace(
            imgtrans_proj=types.SimpleNamespace(
                img_array=self.img, current_img='001.png',
                get_bubble_outlines=lambda page: [self.poly],
                pages={'001.png': blocks},
            ),
            pairwidget_list=[],
            auto_textlayout_flag=True,
        )
        items = []
        for i, blk in enumerate(blocks):
            item = TextBlkItem(blk, i)
            self.assertIs(
                SceneTextManager.layout_textblk(stub, item, text=LONG_TEXT), True
            )
            items.append(item)
        return items

    def test_columns_keep_their_own_half_of_the_balloon(self) -> None:
        items = self._layout_pair()
        for item in items:
            self.assertTrue(item.toPlainText().strip())
            # Never accepted as a floor-sized sliver: the split leaves each
            # column room, so nothing may collapse onto the 15% floor.
            self.assertGreaterEqual(item.font().pointSizeF(), 40 * 0.15 - 0.5)
        left, right = (self._ink_span(item) for item in items)
        self.assertLessEqual(
            left[1], right[0],
            f'neighbouring columns overlap: {left} vs {right}',
        )

    def test_window_splits_at_the_sibling_midpoint(self) -> None:
        window = [0, 0, 300, 60]
        # Sibling on the left: the window gives up its left half.
        self.assertEqual(
            M._shared_outline_window(
                window, self._column([100, 0, 140, 50]),
                [self._column([0, 0, 60, 50])],
            ),
            [80, 0, 300, 60],
        )
        # Sibling on the same rows: only the vertical axis is split.
        self.assertEqual(
            M._shared_outline_window(
                window, self._column([100, 0, 140, 50]),
                [self._column([90, 60, 150, 110])],
            ),
            [0, 0, 300, 55],
        )
        # Nothing runs the helpers' own examples, so run them here.
        for helper in (M._block_xyxy, M._shared_outline_window):
            runner = doctest.DocTestRunner()
            for test in doctest.DocTestFinder().find(helper):
                runner.run(test)
            self.assertEqual(runner.failures, 0, helper.__name__)


if __name__ == '__main__':
    unittest.main()
