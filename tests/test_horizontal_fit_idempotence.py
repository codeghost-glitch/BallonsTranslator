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

import ballontranslator.ui.text_engine.editing.manager as M
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

    def _fit_through_the_app_save(self, record, runs):
        """Fit a block the way a re-run does: save, reload, fit again.

        Saving is the part that used to leak. ``TextBlkItem`` writes the box it
        just rendered into ``_bounding_rect``, the block goes back to the
        project file, and the next pass reads it. A fit sized by that box is
        sized by the previous pass's *text*, which squeeze keeps shrinking, so
        every re-run made the text smaller and a block that could not fit
        ratcheted toward the readability floor.
        """
        sizes = []
        text = record.get('translation') or ''
        for _ in range(runs):
            blk = TextBlock(**record)
            item = TextBlkItem(blk, 0)
            # addTextBlock fits with the translation parked and restores it
            # after, so a persisted rich_text cannot size the item.
            blk.translation = ''
            blk.rich_text = ''
            if SceneTextManager.layout_textblk(self.stub, item,
                                              text=text) is None:
                item.setPlainText(text)
            sizes.append(item.font().pointSizeF())
            blk.translation = text
            record = blk.to_dict()
            record['_bounding_rect'] = item.absBoundingRect()
        return sizes

    def test_reload_does_not_shrink_the_fit(self) -> None:
        # A page reload is what a second pipeline pass over a saved project
        # starts from, so the fit has to survive it: the block comes back with
        # the rendered text box in _bounding_rect, and that box must not
        # become the window the next pass measures against.
        for text, bbox in CASES:
            with self.subTest(text=text[:24]):
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
                blk.fontformat.vertical = False
                blk.fontformat.alignment = 1
                blk.src_is_vertical = True
                blk.fontformat.font_size = 24.0
                blk.translation = text
                sizes = self._fit_through_the_app_save(blk.to_dict(), 5)
                for run, (first, nxt) in enumerate(zip(sizes, sizes[1:]), 2):
                    self.assertAlmostEqual(
                        nxt, first, delta=0.51,
                        msg='%r shrank on reload %d: %.2f -> %.2f'
                            % (text[:24], run, first, nxt))

    def test_reload_does_not_shrink_a_fit_without_a_detector_outline(self) -> None:
        # The same reload on a page whose detector reported no outline, so the
        # fit has to build its own window from the block box. That is the case
        # the ratchet hides in: with an outline the window is the balloon and
        # the block box never reaches the fit.
        text = 'This disclaimer runs down the page margin.'
        bx, by, bw, bh = 60, 60, 120, 90
        img = np.full((400, 400, 3), 90, np.uint8)
        cv2.ellipse(img, (bx + bw // 2, by + bh // 2), (bw, bh), 0, 0, 360,
                    (255, 255, 255), -1)
        self.stub.imgtrans_proj.img_array = img
        self.stub.imgtrans_proj.get_bubble_outlines = lambda pg: []

        blk = TextBlock([bx, by, bx + bw, by + bh])
        blk.set_lines_by_xywh([bx, by, bw, bh])
        blk.fontformat.vertical = False
        blk.fontformat.alignment = 1
        blk.src_is_vertical = True
        blk.fontformat.font_size = 24.0
        blk.translation = text
        sizes = self._fit_through_the_app_save(blk.to_dict(), 5)
        self.assertTrue(all(s > 1.0 for s in sizes), sizes)
        for run, (first, nxt) in enumerate(zip(sizes, sizes[1:]), 2):
            self.assertAlmostEqual(
                nxt, first, delta=0.51,
                msg='no-outline fit shrank on reload %d: %.2f -> %.2f'
                    % (run, first, nxt))

    def test_the_fit_window_is_the_detected_box_not_the_rendered_one(self) -> None:
        # A save writes the box the pass just rendered into _bounding_rect,
        # and squeeze keeps that box close to the text. Seeding the fit window
        # from it shrinks the window on every re-run until growth has no room
        # left - the ratchet. The window has to come from the geometry the
        # detector found, which survives the round trip.
        blk = TextBlock([10, 20, 40, 80])
        blk.set_lines_by_xywh([10, 20, 30, 60])
        blk._bounding_rect = [14, 30, 22, 40]  # what a squeezed render left
        self.assertEqual(M._fit_window(blk), [10, 20, 30, 60])
        blk._detected_bbox = [8, 16, 36, 70]
        self.assertEqual(M._fit_window(blk), [8, 16, 36, 70])

    def test_the_detected_box_survives_a_project_round_trip(self) -> None:
        # The detector records the box it found and the project file keeps it
        # (page 001 of 第2.3話 carries one per block). Loading must not drop
        # it into deprecated_attributes: without it the fit falls back to the
        # rendered text box, which is the ratchet above.
        blk = TextBlock([10, 20, 40, 80])
        blk.set_lines_by_xywh([10, 20, 30, 60])
        blk._detected_bbox = [12, 18, 36, 66]
        reloaded = TextBlock(**blk.to_dict())
        self.assertEqual(reloaded._detected_bbox, [12, 18, 36, 66])
        self.assertNotIn('_detected_bbox',
                         getattr(reloaded, 'deprecated_attributes', {}))
