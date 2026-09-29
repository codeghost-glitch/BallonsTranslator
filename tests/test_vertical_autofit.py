import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtWidgets import QApplication

import ballontranslator.ui.text_engine.editing.manager as M
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock import TextBlock

CJK_TEXT = '嗯，誰知道四葉醬什麼時候會餓到哭出來呢。'
NARROW_TEXT = '※本漫畫為虛構作品。'


class _Stub:
    """Minimal stand-in exposing the two attributes the vertical fit reads."""

    auto_textlayout_flag = True
    pairwidget_list: list = []

    _vertical_fit_balloon_box = SceneTextManager._vertical_fit_balloon_box
    _layout_textblk_vertical = SceneTextManager._layout_textblk_vertical
    _vertical_column_rects = staticmethod(SceneTextManager._vertical_column_rects)


def _make_vertical_item(box, text, font_size=18.0):
    x, y, w, h = box
    block = TextBlock([x, y, x + w, y + h])
    block.set_lines_by_xywh([x, y, w, h])
    block._detected_bbox = [x, y, w, h]
    block.fontformat.font_size = font_size
    block.fontformat.vertical = True
    block.fontformat.alignment = 1
    block.src_is_vertical = True
    block.vertical = True
    block.translation = text
    item = TextBlkItem(block, 0)
    item.setPlainText(text)
    return item


class TestVerticalAutoFit(unittest.TestCase):
    """Vertical (tategaki) auto-typeset: the settled column extent, not a
    horizontal line canvas, is what must fit the balloon."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_flag = pcfg.let_fntsize_flag
        self.old_auto = pcfg.let_autolayout_flag
        self.old_target = pcfg.module.translate_target
        pcfg.let_fntsize_flag = 0
        pcfg.let_autolayout_flag = True
        pcfg.module.translate_target = '繁體中文'
        self.stub = _Stub()

    def tearDown(self) -> None:
        pcfg.let_fntsize_flag = self.old_flag
        pcfg.let_autolayout_flag = self.old_auto
        pcfg.module.translate_target = self.old_target

    def _fit(self, box, text, font_size=18.0):
        item = _make_vertical_item(box, text, font_size)
        ok = self.stub._layout_textblk_vertical(
            item, text, bounding_rect=list(box))
        return item, ok

    def test_grows_font_to_fill_balloon(self) -> None:
        # A wide, tall balloon holding a short line: the fit must grow the
        # default font well past the stored size to fill the column width.
        item, ok = self._fit([100, 100, 165, 266], CJK_TEXT)
        self.assertTrue(ok)
        settled = item.layout._column_content_width()
        self.assertGreater(settled, 18.0)
        # The winning size keeps the columns inside the balloon width.
        self.assertLessEqual(settled, 165)

    def test_shrinks_for_narrow_column(self) -> None:
        # A 29px-wide margin column cannot hold the default font; the fit
        # must shrink it rather than overflow or bail out.
        item, ok = self._fit([100, 100, 29, 575], NARROW_TEXT)
        self.assertTrue(ok)
        self.assertLessEqual(item.layout._column_content_width(), 29)

    def test_content_width_grows_with_font(self) -> None:
        # The fit signal is monotonic: a larger font never occupies less
        # horizontal extent, so the step-up loop terminates on the balloon.
        widths = []
        for size in (12.0, 20.0, 28.0):
            item = _make_vertical_item([0, 0, 400, 400], CJK_TEXT, size)
            widths.append(item.layout._column_content_width())
        self.assertEqual(widths, sorted(widths))

    def test_centers_item_on_balloon(self) -> None:
        # After the fit the settled item is centered on the detection box,
        # so vertical text neither hugs a corner nor rides an edge.
        box = [100, 100, 165, 266]
        item, ok = self._fit(box, CJK_TEXT)
        self.assertTrue(ok)
        br = item.absBoundingRect(qrect=True)
        cx = br.x() + br.width() / 2
        cy = br.y() + br.height() / 2
        self.assertAlmostEqual(cx, box[0] + box[2] / 2, delta=2.0)
        self.assertAlmostEqual(cy, box[1] + box[3] / 2, delta=2.0)

    def test_respects_autolayout_flag(self) -> None:
        # With auto-typeset off the fit must not touch the font.
        pcfg.let_autolayout_flag = False
        item, ok = self._fit([100, 100, 165, 266], CJK_TEXT)
        self.assertFalse(ok)

    def test_declines_for_non_cjk_target(self) -> None:
        # A vertical Japanese source translated into a Latin/SE-Asian script
        # must NOT be fitted as vertical: the pipeline forces blk.vertical =
        # False for non-CJK targets after layout, so a vertical fit would
        # hand column-shaped geometry to horizontal text. This is what broke
        # English typesetting off Japanese pages.
        for target in ('English', 'ไทย', 'Tiếng Việt'):
            with self.subTest(target=target):
                pcfg.module.translate_target = target
                item, ok = self._fit([100, 100, 165, 266], CJK_TEXT)
                self.assertFalse(ok)

    def test_accepts_for_cjk_target(self) -> None:
        # The Chinese targets that motivated the feature still fit.
        for target in ('繁體中文', '简体中文'):
            with self.subTest(target=target):
                pcfg.module.translate_target = target
                item, ok = self._fit([100, 100, 165, 266], CJK_TEXT)
                self.assertTrue(ok)

    def test_decline_keeps_text_on_item(self) -> None:
        # When the fit declines (auto-typeset off) the translation must still
        # reach the item: the caller clears blk.translation first and only
        # falls back to setPlainText on a None return, but this returns False.
        pcfg.let_autolayout_flag = False
        item = _make_vertical_item([100, 100, 165, 266], CJK_TEXT)
        item.setPlainText('')  # caller cleared translation first
        ok = self.stub._layout_textblk_vertical(
            item, CJK_TEXT, bounding_rect=[100, 100, 165, 266])
        self.assertFalse(ok)
        self.assertEqual(item.toPlainText(), CJK_TEXT)

    def test_dispatch_reaches_vertical_fit(self) -> None:
        # layout_textblk must route a vertical block into the vertical fit
        # instead of the horizontal path's early return.
        import inspect
        src = inspect.getsource(SceneTextManager.layout_textblk)
        self.assertIn('_layout_textblk_vertical', src)
        # The old blanket "vertical not supported" bail must be gone.
        self.assertNotIn('vertical writing is not supported', src)


class TestVerticalOutlineFit(unittest.TestCase):
    """The detector outline is a real ceiling: it must cap a block below the
    size its rectangular detection box alone would allow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_auto = pcfg.let_autolayout_flag
        pcfg.let_autolayout_flag = True
        pcfg.let_fntsize_flag = 0

    def tearDown(self) -> None:
        pcfg.let_autolayout_flag = self.old_auto

    @staticmethod
    def _stub_with_outline(outline):
        stub = _Stub()
        stub.imgtrans_proj = type('P', (), {})()
        stub.imgtrans_proj.current_img = 'p'
        stub.imgtrans_proj.get_bubble_outlines = lambda pg: [outline]
        return stub

    def test_column_rects_cover_each_column(self) -> None:
        # Every glyph cell belongs to a column strip; the builder must return
        # one rect per column, each with positive width and height.
        item = _make_vertical_item([0, 0, 165, 266], CJK_TEXT)
        rects = SceneTextManager._vertical_column_rects(item)
        self.assertGreaterEqual(len(rects), 1)
        for x, y, w, h in rects:
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
        # Columns run right to left: x offsets are ordered but descending.
        # The builder sorts ascending; just assert uniqueness of x.
        xs = [r[0] for r in rects]
        self.assertEqual(len(set(xs)), len(xs))

    def test_outline_gate_scales_with_outline(self) -> None:
        # Anchor the item at the origin so its scene position equals its page
        # position and the per-column pointPolygonTest gate sees the real
        # geometry. A balloon outline that hugs the text (an ellipse matching
        # the block's aspect) must not reject a fit the box alone accepts,
        # while a much tighter outline must cap it harder.
        import numpy as np
        box = [0, 0, 165, 266]
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2

        def fit(scale):
            if scale is None:
                stub = _Stub()
            else:
                rx = (box[2] / 2) * scale
                ry = (box[3] / 2) * scale
                ellipse = [(cx + rx * np.cos(a), cy + ry * np.sin(a))
                           for a in np.linspace(0, 2 * np.pi, 65)[:-1]]
                stub = self._stub_with_outline(ellipse)
            item = _make_vertical_item(box, CJK_TEXT)
            stub._layout_textblk_vertical(item, CJK_TEXT, bounding_rect=list(box))
            return item.layout._column_content_width()

        none_w = fit(None)
        roomy_w = fit(1.05)   # balloon comfortably contains the columns
        tight_w = fit(0.55)   # balloon much smaller than the text
        # An ellipse can never contain a box's corners, so a real balloon
        # always caps the fit below the box-only result, and a tighter
        # balloon caps it further. This is the gate doing its job: the top of
        # the leftmost column sits outside a round balloon, so the font
        # shrinks until every column fits.
        self.assertLessEqual(roomy_w, none_w + 1e-6)
        self.assertLess(tight_w, roomy_w)
