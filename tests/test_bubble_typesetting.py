import json
import importlib.util
import math
import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QRectF, Qt
from qtpy.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QTextCharFormat, QTextCursor, QTextDocument
from qtpy.QtWidgets import QApplication, QPlainTextEdit

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.bubble_layout import (
    _current_font,
    _hyphenator,
    _longest_word_width,
    fit_text_to_bubble,
    hyphenate_document,
)
from ballontranslator.ui.text_engine.annotations import AnnotationProperty
from ballontranslator.ui.text_engine.editing.commands import AutoLayoutCommand
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.bubble import (
    bubble_inner_center,
    bubble_inner_rect,
    bubble_polygon_from_mask,
    split_connected_bubble,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import px2pt
from ballontranslator.utils.proj_imgtrans import TextBlkEncoder
from ballontranslator.utils.text_layout import layout_text
from ballontranslator.utils.text_processing import seg_text
from ballontranslator.utils.textblock import TextBlock


def assert_inside_target(
    testcase, item, polygon, tolerance=2.0,
) -> None:
    """Check the whole fitted rectangle against the physical outline."""
    box = item.absBoundingRect(qrect=True)
    contour = np.asarray(polygon, np.float32)
    for x in np.linspace(box.left(), box.right(), max(2, int(box.width()) + 1)):
        for y in np.linspace(box.top(), box.bottom(), max(2, int(box.height()) + 1)):
            testcase.assertGreaterEqual(cv2.pointPolygonTest(contour, (float(x), float(y)), True), -tolerance)


class _QtItemCase(unittest.TestCase):
    """Shared QApplication setup and TextBlkItem factory for Qt test classes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def make_item(self, text: str, vertical: bool | None = None) -> TextBlkItem:
        block = TextBlock(
            xyxy=[50, 50, 140, 110], translation=text,
            bubble_polygon=[[0, 0], [220, 0], [220, 180], [0, 180]],
        )
        if vertical is not None:
            block.vertical = vertical
        block._bounding_rect = [50, 50, 90, 60]
        block.fontformat.font_family = 'Arial'
        block.fontformat.font_size = 32
        block.fontformat.line_spacing = 1.0
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        return item


class BubbleGeometryTests(unittest.TestCase):
    def test_invalid_optional_geometry_does_not_discard_translation(self) -> None:
        for polygon in ('bad', [[0, 0]], [[0, 0], [float('nan'), 10], [10, 20]]):
            block = TextBlock(translation='Keep this', bubble_polygon=polygon)
            self.assertEqual(block.translation, 'Keep this')
            self.assertIsNone(block.bubble_polygon)

    def test_polygon_survives_project_json_and_old_blocks_default_to_none(self) -> None:
        polygon = [[0, 0], [120, 0], [120, 100], [0, 100]]
        block = TextBlock(translation='Text', bubble_polygon=polygon)
        restored = TextBlock(**json.loads(json.dumps(block, cls=TextBlkEncoder)))
        self.assertEqual(restored.bubble_polygon, polygon)
        self.assertIsNone(TextBlock().bubble_polygon)

    def test_inner_rectangle_avoids_concave_tail(self) -> None:
        polygon = [[10, 10], [110, 10], [110, 100], [65, 100], [20, 155], [30, 100], [10, 100]]
        rect = bubble_inner_rect(polygon)
        self.assertIsNotNone(rect)
        left, top, width, height = rect
        contour = np.asarray(polygon, np.float32)
        for horizontal in np.linspace(left, left + width, 20):
            for vertical in np.linspace(top, top + height, 20):
                self.assertGreater(cv2.pointPolygonTest(contour, (horizontal, vertical), False), 0)
        self.assertLess(top + height, 110)

    def test_outline_reaches_border_clipped_balloon_edge(self) -> None:
        # A balloon flush against the panel border: its white interior is
        # bounded on the border side by the border ink, and the traced
        # outline must keep that full extent. The old contour roll-average
        # pulled sparse straight edges inward (page 007 teardrop: 121 -> 133),
        # leaving a white gap between the highlight and the panel border.
        page = np.full((160, 240), 255, dtype=np.uint8)
        cv2.rectangle(page, (42, 42), (138, 118), 0, 3)      # ink outline
        cv2.rectangle(page, (38, 36), (42, 124), 0, -1)      # panel border, left
        mask = np.zeros((160, 240), dtype=np.uint8)
        cv2.rectangle(mask, (46, 46), (134, 114), 255, -1)   # interior bubble mask
        polygon = bubble_polygon_from_mask(mask, page)
        self.assertIsNotNone(polygon)
        xs = np.asarray([point[0] for point in polygon], dtype=np.float64)
        # The white interior starts where the border ink ends (x=45).
        self.assertLessEqual(xs.min(), 47.0)

    def test_source_border_repairs_mask_notch_without_flooding_open_balloon(self) -> None:
        source = np.full((120, 140), 255, dtype=np.uint8)
        cv2.rectangle(source, (20, 20), (120, 100), 0, 2)
        predicted = np.zeros_like(source)
        predicted[22:99, 22:119] = 1
        predicted[50:65, 105:119] = 0
        refined = bubble_polygon_from_mask(predicted, source)
        self.assertGreater(cv2.pointPolygonTest(
            np.asarray(refined, dtype=np.float32), (112.0, 57.0), False,
        ), 0)
        self.assertLess(cv2.pointPolygonTest(
            np.asarray(refined, dtype=np.float32), (125.0, 57.0), False,
        ), 0)
        source[50:65, 118:123] = 255
        self.assertEqual(
            bubble_polygon_from_mask(predicted, source),
            bubble_polygon_from_mask(predicted),
        )

    def test_invalid_live_geometry_is_rejected_on_save(self) -> None:
        block = TextBlock()
        block.bubble_polygon = [[0, 0]]
        with self.assertRaisesRegex(ValueError, 'Invalid bubble polygon'):
            json.dumps(block, cls=TextBlkEncoder)


class BubbleTypesettingTests(_QtItemCase):
    def shaped_inner_rect(self, polygon):
        # Reference center for symmetric fixtures, not a font-size ceiling.
        return QRectF(*bubble_inner_rect(polygon))

    def assert_centered_interior(self, item: TextBlkItem, target: QRectF) -> None:
        box = item.absBoundingRect(qrect=True)
        assert_inside_target(self, item, item.blk.bubble_polygon)
        self.assertEqual(int(item.fontformat.alignment), 1)
        self.assertLessEqual(abs(box.center().x() - target.center().x()), 3.0)
        needed_height, needed_width = item.layout.minSize()
        pad = max(0.0, float(item.padding())) * 2.0
        text_pad = max(0.0, float(item.layout.text_padding or 0.0))
        # Stay-inside minimal path may exceed the interior by a couple px of
        # glyph overhang; the erosion margin to the bubble border absorbs it
        # while the logical box itself stays contained (checked above).
        # text_padding derives from vertical ink extents, so the width check
        # excludes it exactly like the fit search does.
        self.assertLessEqual(needed_height + pad, box.height() + 3.0)
        self.assertLessEqual(needed_width - text_pad + pad, box.width() + 3.0)

    def test_fitting_uses_bubble_interior_and_preserves_text(self) -> None:
        text = 'An extraordinarily interesting conversation about typography.'
        item = self.make_item(text)
        original_size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        # Breaks are baked in like the mask fallback path, so compare words;
        # dictionary-rescued words carry literal hyphens but keep every
        # letter.
        self.assertEqual(
            item.toPlainText().replace('-', '').replace('\u00ad', '').replace('\n', '').replace(' ', ''),
            text.replace(' ', ''),
        )
        self.assertLessEqual(item.document().begin().begin().fragment().charFormat().fontPointSize(), original_size)
        expected = self.shaped_inner_rect(item.blk.bubble_polygon)
        self.assert_centered_interior(item, expected)

    def test_short_dialogue_uses_wide_interior_without_stranding_pronoun(self) -> None:
        item = self.make_item('Probably\nIkebukuro,\nI think.')
        item.blk.bubble_polygon = [
            [100 + 100 * math.cos(math.radians(d)), 150 + 150 * math.sin(math.radians(d))]
            for d in range(0, 360, 5)
        ]
        fit_text_to_bubble(item, None, language='English',
                           target_rect=bubble_inner_rect(item.blk.bubble_polygon))
        narrow_size = _current_font(item).pointSizeF()
        item.setPlainText('Probably\nIkebukuro,\nI think.')
        fit_text_to_bubble(item, None, language='English')
        # Words may carry dictionary hyphens in the rescue, but letters
        # survive and the wide interior grows the text.
        normalized = item.toPlainText().replace('-', '').replace('\n', '').replace(' ', '')
        self.assertEqual(normalized, 'ProbablyIkebukuro,Ithink.')
        self.assertGreater(_current_font(item).pointSizeF(), narrow_size + 1.0)
        # The outline path rides the balloon's taper like the mask layout:
        # the widest fitted line may poke a few px past the detected ink
        # outline, which sits inside the balloon's real wall.
        assert_inside_target(self, item, item.blk.bubble_polygon, tolerance=12.0)

    def test_vertical_and_rotated_items_are_unchanged(self) -> None:
        item = self.make_item('Translation')
        original = item.toHtml()
        item.setRotation(15)
        self.assertFalse(fit_text_to_bubble(item, None, language='English'))
        self.assertEqual(item.toHtml(), original)

    def test_tapered_bubble_rides_wide_mid_rows_for_short_text(self) -> None:
        # A pinch-waisted bubble has a narrow maximal interior, but short
        # dialogue rides its wider rows like the lettering that actually
        # fits the bubble (page 007: `Kunieda...` pinned to 92px while the
        # middle rows are 148px). The fitted line must also stay on one
        # visual line at the fitted width — an anywhere-wrap would strand
        # the trailing period (`KUNIEDA..` / `.`).
        from ballontranslator.ui.text_engine.bubble_layout import (
            _current_font,
            fit_text_to_bubble,
        )

        polygon = [
            [0, 0], [160, 0], [130, 60], [130, 140], [160, 200],
            [0, 200], [30, 140], [30, 60],
        ]
        interior_item = self.make_item('Kunieda...')
        interior_item.blk.bubble_polygon = polygon
        self.assertTrue(fit_text_to_bubble(
            interior_item, None, language='English',
            target_rect=bubble_inner_rect(polygon),
        ))
        interior_size = _current_font(interior_item).pointSizeF()

        item = self.make_item('Kunieda...')
        item.blk.bubble_polygon = polygon
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        # Letters survive any dictionary rescue splits.
        normalized = item.toPlainText().replace('-', '').replace('\n', '').replace(' ', '')
        self.assertEqual(normalized, 'Kunieda...')
        size = _current_font(item).pointSizeF()
        self.assertGreaterEqual(size, interior_size * 1.2)
        # Fitted lines must not tear at the fitted box width (a wrap would
        # strand the trailing period on its own line).
        from qtpy.QtGui import QTextLayout, QTextOption
        box = item.absBoundingRect(qrect=True)
        for line_text in item.toPlainText().split('\n'):
            layout = QTextLayout(line_text, _current_font(item))
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WrapAnywhere)
            layout.setTextOption(option)
            layout.beginLayout()
            line = layout.createLine()
            line.setLineWidth(max(1.0, box.width()))
            second = layout.createLine()
            layout.endLayout()
            self.assertFalse(second.isValid())

    def test_collapsed_interior_returns_block_to_mask_layout(self) -> None:
        # A diagonal teardrop balloon collapses the axis-aligned interior
        # (page 007: 25x72 against 69x208). The bubble fit must decline so
        # the mask layout fits the source region instead of shrinking to
        # the floor, and it must leave the item untouched for that fallback.
        item = self.make_item("Don't give me that \"Sensei~\"")
        item.blk.lines = [[[[139, 710], [208, 710], [208, 918], [139, 918]]]]
        item.blk.bubble_polygon = [
            [134, 661], [133, 665], [133, 683], [144, 739], [159, 795],
            [178, 854], [204, 920], [222, 950], [226, 946], [236, 916],
            [244, 875], [246, 837], [245, 801], [242, 777], [237, 759],
            [237, 750], [243, 744], [244, 741], [241, 741], [236, 737],
            [229, 717], [217, 696], [204, 681], [193, 671], [176, 661],
            [168, 659], [152, 658],
        ]
        before = item.absBoundingRect(qrect=True)
        self.assertFalse(fit_text_to_bubble(item, None, language='English'))
        after = item.absBoundingRect(qrect=True)
        self.assertEqual(before, after)

    def test_asymmetric_balloon_centers_text_on_balloon_center(self) -> None:
        # A tall balloon whose maximal interior sits above its visual center
        # (oval body + tail below): the fit must slide the box onto the same
        # interior-centroid guide the manual Center-in-bubble command uses,
        # not leave it at the interior rectangle's position (page 007: the
        # phone balloon's text sat at 40% of the balloon height).
        oval = [[150 + 50 * math.cos(math.radians(d)), 120 + 90 * math.sin(math.radians(d))]
                for d in range(0, 360, 6)]
        polygon = oval[:16] + [[150, 208], [138, 242], [150, 210]] + oval[16:]
        guide = bubble_inner_center(polygon)
        center_y = float(guide[1])
        item = self.make_item('Hello there, friend.')
        item.blk.bubble_polygon = polygon
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        box = item.absBoundingRect(qrect=True)
        self.assertAlmostEqual(box.center().y(), center_y, delta=4.0)
        # The slide routes through the logical-position setter so the block
        # model (and the saved project) keeps the centered box; a plain
        # setPos would move only the graphics item and revert on reload.
        bx1, by1, bx2, by2 = item.blk.xyxy
        self.assertAlmostEqual((by1 + by2) / 2.0, center_y, delta=4.0)
        assert_inside_target(self, item, polygon, tolerance=12.0)

    def test_small_bubble_text_grows_to_fit_and_repeated_fitting_is_stable(self) -> None:
        item = self.make_item('Oh!')
        item.blk.bubble_polygon = [[0, 0], [65, 0], [65, 100], [0, 100]]
        item.setFontSize(6)
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        fitted_size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertGreater(fitted_size, 6)
        target = self.shaped_inner_rect(item.blk.bubble_polygon)
        self.assert_centered_interior(item, target)
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        self.assertAlmostEqual(item.document().begin().begin().fragment().charFormat().fontPointSize(), fitted_size)
        self.assertEqual(item.toPlainText(), 'Oh!')

    def test_auto_fit_responds_to_bubble_size_not_starting_font_size(self) -> None:
        sizes = []
        for extent, initial_size in ((80, 40), (160, 6), (160, 30)):
            item = self.make_item('A short reply')
            item.blk.bubble_polygon = [[0, 0], [extent, 0], [extent, extent], [0, extent]]
            item.setFontSize(initial_size)
            self.assertTrue(fit_text_to_bubble(item, None, language='English'))
            sizes.append(item.document().begin().begin().fragment().charFormat().fontPointSize())
            target = self.shaped_inner_rect(item.blk.bubble_polygon)
            self.assert_centered_interior(item, target)
        self.assertGreater(sizes[1], sizes[0])
        self.assertAlmostEqual(sizes[1], sizes[2])

    @unittest.skipUnless(importlib.util.find_spec('pyphen'), 'Optional Pyphen is not installed')
    def test_soft_hyphens_preserve_utf16_and_formatting_and_are_idempotent(self) -> None:
        document = QTextDocument()
        document.setPlainText('😀 extraordinary conversation')
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        char_format = QTextCharFormat()
        char_format.setForeground(QColor('red'))
        cursor.mergeCharFormat(char_format)
        hyphenate_document(document, 'English')
        once = document.toPlainText()
        self.assertEqual(once.replace('\u00ad', ''), '😀 extraordinary conversation')
        self.assertIn('\u00ad', once)
        hyphenate_document(document, 'English')
        self.assertEqual(document.toPlainText(), once)
        cursor.setPosition(4)
        self.assertEqual(cursor.charFormat().foreground().color(), QColor('red'))

    def test_missing_optional_dictionary_package_keeps_content(self) -> None:
        document = QTextDocument()
        document.setPlainText('extraordinary conversation')
        _hyphenator.cache_clear()
        try:
            with patch.dict('sys.modules', {'pyphen': None}):
                hyphenate_document(document, 'English')
            self.assertEqual(document.toPlainText(), 'extraordinary conversation')
        finally:
            _hyphenator.cache_clear()

    @unittest.skipUnless(importlib.util.find_spec('pyphen'), 'Optional Pyphen is not installed')
    def test_app_language_names_select_their_dictionaries(self) -> None:
        for language in ('Français', 'Deutsch', 'Español', 'Português', 'Brazilian Portuguese'):
            with self.subTest(language=language):
                self.assertIsNotNone(_hyphenator(language))

    def test_annotated_runs_are_not_split(self) -> None:
        document = QTextDocument()
        document.setPlainText('extraordinary')
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        char_format = QTextCharFormat()
        char_format.setProperty(AnnotationProperty.RUBY_ID, 'group')
        cursor.mergeCharFormat(char_format)
        hyphenate_document(document, 'English')
        self.assertEqual(document.toPlainText(), 'extraordinary')

    def test_unsupported_language_keeps_content(self) -> None:
        document = QTextDocument()
        document.setPlainText('日本語のテキスト')
        hyphenate_document(document, '日本語')
        self.assertEqual(document.toPlainText(), '日本語のテキスト')

    def test_fit_undo_redo_keeps_paired_editor_synchronized(self) -> None:
        item = self.make_item('An extraordinarily interesting conversation.')
        old_html = item.toHtml()
        old_rect = item.absBoundingRect(qrect=True)
        old_alignment = int(item.fontformat.alignment)
        original_text = item.toPlainText()
        editor = QPlainTextEdit()
        self.addCleanup(editor.deleteLater)
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        self.assertEqual(int(item.fontformat.alignment), 1)
        fitted_text = item.toPlainText()
        fitted_rect = item.absBoundingRect(qrect=True)
        command = AutoLayoutCommand([item], [old_rect], [old_html], [editor], [old_alignment])
        command.redo()
        command.undo()
        self.assertEqual(item.toPlainText(), original_text)
        self.assertEqual(editor.toPlainText(), original_text)
        self.assertEqual(item.absBoundingRect(qrect=True), old_rect)
        self.assertEqual(int(item.fontformat.alignment), old_alignment)
        command.redo()
        self.assertEqual(item.toPlainText(), fitted_text)
        self.assertEqual(editor.toPlainText(), fitted_text)
        self.assertEqual(item.absBoundingRect(qrect=True), fitted_rect)
        self.assertEqual(int(item.fontformat.alignment), 1)

    def test_elliptic_bubble_centers_text_with_equal_margins(self) -> None:
        ellipse = [[110 + 100 * math.cos(math.radians(d)),
                    90 + 80 * math.sin(math.radians(d))] for d in range(0, 360, 10)]
        item = self.make_item('I still don\'t know much about the human world.')
        item.blk.bubble_polygon = ellipse
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        target = self.shaped_inner_rect(ellipse)
        self.assert_centered_interior(item, target)

    def test_narrow_bubble_keeps_words_intact_without_hyphenation(self) -> None:
        item = self.make_item('COME ON! HAND IT OVER.')
        item.blk.bubble_polygon = [[0, 0], [110, 0], [110, 400], [0, 400]]
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        # Words may fit into a shaped retry rectangle wider than the default
        # interior; they must never exceed the rectangle they were fitted to.
        self.assertLessEqual(
            _longest_word_width(item), item.absBoundingRect(qrect=True).width() + 0.5
        )

    def test_fitting_keeps_trailing_comma_on_its_word_line(self) -> None:
        item = self.make_item('Probably\nIkebukuro,\nI think.')
        item.setFontSize(20.0)
        font = item.document().begin().begin().fragment().charFormat().font()
        metrics = QFontMetricsF(font)
        width = (metrics.horizontalAdvance('Ikebukuro')
                 + metrics.horizontalAdvance('Ikebukuro,')) / 2.0
        self.assertTrue(fit_text_to_bubble(
            item, None, language='English',
            target_rect=(0.0, 0.0, width, 300.0),
        ))
        block = item.document().begin()
        while block.isValid():
            self.assertEqual(block.layout().lineCount(), 1, block.text())
            block = block.next()
        self.assertEqual(item.toPlainText(), 'Probably\nIkebukuro,\nI think.')

    def test_shared_oval_bisector_fallback_fits_each_sibling(self) -> None:
        # A smooth oval has no clean neck: the koharu Voronoi fallback must
        # partition the physical contour so each sibling fits its own cell.
        polygon = [
            [120 + 110 * math.cos(math.radians(d)),
             200 + 190 * math.sin(math.radians(d))]
            for d in range(0, 360, 6)
        ]
        centers = [(120.0, 90.0), (120.0, 310.0)]
        cells = split_connected_bubble(polygon, centers)
        self.assertIsNotNone(cells)
        assert cells is not None
        self.assertEqual(len(cells), 2)
        header = self.make_item('HEY!! YU!!')
        body = self.make_item('CUT IT OUT!! THESE BALLS ARE NOT FREE, YOU KNOW!')
        for item, cell, center in zip((header, body), cells, centers):
            self.assertGreaterEqual(
                cv2.pointPolygonTest(
                    np.asarray(cell, dtype=np.float32),
                    (float(center[0]), float(center[1])), False,
                ), 0.0,
            )
            item.blk.bubble_polygon = cell
            self.assertTrue(
                fit_text_to_bubble(item, None, language='English')
            )
        header_box = header.absBoundingRect(qrect=True)
        body_box = body.absBoundingRect(qrect=True)
        self.assertLessEqual(header_box.bottom(), body_box.top() + 2.0)

    def test_coincident_anchors_subdivide_into_strips(self) -> None:
        # Two blocks sharing one center on a smooth oval: the bisector
        # fallback splits the contour into equal strips along the longer axis.
        polygon = [
            [120 + 110 * math.cos(math.radians(d)),
             200 + 190 * math.sin(math.radians(d))]
            for d in range(0, 360, 6)
        ]
        cells = split_connected_bubble(polygon, [(120.0, 200.0), (120.0, 200.0)])
        self.assertIsNotNone(cells)
        assert cells is not None
        self.assertEqual(len(cells), 2)
        for cell in cells:
            self.assertGreaterEqual(
                cv2.pointPolygonTest(
                    np.asarray(cell, dtype=np.float32), (120.0, 200.0), False,
                ), 0.0,
            )
        top_max_y = max(point[1] for point in cells[0])
        bottom_min_y = min(point[1] for point in cells[1])
        # The taller-than-wide contour partitions along y; strips meet at the
        # shared boundary without crossing.
        self.assertLessEqual(top_max_y, bottom_min_y + 1.0)

    def test_connected_lobes_fit_their_own_bubble(self) -> None:
        polygon = [
            [0, 0], [100, 0], [100, 100], [40, 100], [40, 140],
            [100, 140], [100, 240], [0, 240], [0, 140], [60, 140],
            [60, 100], [0, 100],
        ]
        centers = [(50.0, 50.0), (50.0, 190.0)]
        lobes = split_connected_bubble(polygon, centers)
        self.assertIsNotNone(lobes)
        assert lobes is not None
        header = self.make_item('HEY!! YU!!')
        body = self.make_item('CUT IT OUT!! THESE BALLS ARE NOT FREE, YOU KNOW!')
        for item, lobe in zip((header, body), lobes):
            item.blk.bubble_polygon = lobe
            self.assertTrue(
                fit_text_to_bubble(item, None, language='English')
            )
        header_box = header.absBoundingRect(qrect=True)
        body_box = body.absBoundingRect(qrect=True)
        self.assertLessEqual(header_box.bottom(), body_box.top() + 2.0)
        for item, lobe in zip((header, body), lobes):
            assert_inside_target(self, item, lobe)

    def test_long_translation_stays_inside_with_readable_size(self) -> None:
        long_text = ('Without even knowing it, I became a genius at conquering girls '
                     'in dating sims, and thus I was dubbed the God of Conquest. ' * 3).strip()
        item = self.make_item(long_text)
        item.blk.bubble_polygon = [[0, 0], [220, 0], [220, 180], [0, 180]]
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        target = self.shaped_inner_rect(item.blk.bubble_polygon)
        self.assert_centered_interior(item, target)
        fitted_size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertGreaterEqual(fitted_size, 4.0)

    def test_selected_bubble_block_highlights_balloon_not_text_box(self) -> None:
        # The dashed selection guide of a detected-bubble block follows the
        # balloon outline: the fitted text box can be a sliver in a tall
        # balloon or an oversized slab over a diagonal one, and its dashed
        # rect then floats over artwork. Rotated, transformed, unselected,
        # and bubble-less blocks keep the fitted text rectangle.
        canvas = Canvas()
        item = self.make_item('Text')
        canvas.attach_text_item(item)
        canvas.setSceneRect(QRectF(0, 0, 300, 200))
        bubble = [[50, 50], [220, 70], [220, 160], [50, 160]]
        try:
            item.blk.bubble_polygon = bubble
            item.setSelected(True)
            outline = item._ui_guide_outline()
            self.assertAlmostEqual(outline.boundingRect().width(), 170.0, delta=0.5)
            self.assertAlmostEqual(outline.boundingRect().height(), 110.0, delta=0.5)
            self.assertNotEqual(
                outline.boundingRect().size(),
                item.geometry_controller.visual_outline_in_item().boundingRect().size(),
            )
            item.setRotation(15)
            self.assertEqual(
                item._ui_guide_outline(),
                item.geometry_controller.visual_outline_in_item(),
            )
            item.setRotation(0)
            item.blk.bubble_polygon = None
            self.assertEqual(
                item._ui_guide_outline(),
                item.geometry_controller.visual_outline_in_item(),
            )
            item.blk.bubble_polygon = bubble
            item.setSelected(False)
            self.assertEqual(
                item._ui_guide_outline(),
                item.geometry_controller.visual_outline_in_item(),
            )
        finally:
            canvas.removeItem(item)
            canvas.gv.deleteLater()
            canvas.deleteLater()

    def test_overlay_releases_regions_and_does_not_change_scene_export(self) -> None:
        canvas = Canvas()
        item = self.make_item('Text')
        canvas.attach_text_item(item)
        canvas.gv.add_bubble_region(item)
        canvas.setSceneRect(QRectF(0, 0, 220, 180))

        def render_scene() -> QImage:
            result = QImage(220, 180, QImage.Format.Format_ARGB32)
            result.fill(Qt.GlobalColor.white)
            painter = QPainter(result)
            canvas.render(painter)
            painter.end()
            return result

        try:
            with patch.object(pcfg, 'show_detected_bubbles', False):
                without_overlay = render_scene()
            with patch.object(pcfg, 'show_detected_bubbles', True):
                self.assertEqual(render_scene(), without_overlay)
                preview = QImage(220, 180, QImage.Format.Format_ARGB32)
                preview.fill(Qt.GlobalColor.transparent)
                painter = QPainter(preview)
                canvas.gv.drawForeground(painter, canvas.sceneRect())
                painter.end()
                self.assertGreater(preview.pixelColor(100, 100).alpha(), 0)
            canvas.removeItem(item)
            self.assertFalse(canvas.gv.bubble_paths)
        finally:
            canvas.gv.deleteLater()
            canvas.deleteLater()


class TallNarrowBubbleTests(_QtItemCase):
    """Tall 1:2-1:3 bubbles from dense pages with vertical source text.

    Covers tall narrow ovals like a ~140x320 bubble: the fit must
    stay inside, keep words whole, and settle at a readable size instead of
    pinning to the 4pt floor.
    """

    @staticmethod
    def tall_ellipse(width: float, height: float):
        return [[width / 2 + width / 2 * math.cos(math.radians(d)),
                 height / 2 + height / 2 * math.sin(math.radians(d))]
                for d in range(0, 360, 10)]

    def assert_readable_interior(
        self, item: TextBlkItem, polygon, minimum_size: float = 6.0,
        tolerance: float = 2.0,
    ) -> None:
        assert_inside_target(self, item, polygon, tolerance=tolerance)
        size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertGreaterEqual(size, minimum_size)

    def test_tall_oval_fits_english_translation(self) -> None:
        text = 'This is a test, this is a test! Wait a second!'
        item = self.make_item(text)
        polygon = self.tall_ellipse(140.0, 320.0)
        item.blk.bubble_polygon = polygon
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        # `second!` spans the 82px interior on its own, so the fit stops
        # where the longest word still holds rather than at the floor.
        self.assert_readable_interior(item, polygon, tolerance=12.0)
        self.assertEqual(
            sorted(item.toPlainText().replace('\u00ad', '').split()),
            sorted(text.split()),
        )

    def test_tall_oval_fits_vertical_japanese(self) -> None:
        text = 'これはテストです！ちょっと待って！'
        item = self.make_item(text, vertical=True)
        polygon = self.tall_ellipse(140.0, 320.0)
        item.blk.bubble_polygon = polygon
        self.assertTrue(fit_text_to_bubble(item, None, language='日本語'))
        # Spaceless scripts wrap per character; the whole string must not
        # count as one unbreakable word pinning the fit to the floor.
        self.assert_readable_interior(item, polygon, minimum_size=10.0)

    def test_short_text_grows_to_twice_source_size(self) -> None:
        # Growth caps at twice the detected source size so short dialogue
        # fills more of a roomy balloon without unbounded blowup.
        item = self.make_item('Oh!')
        item.blk._detected_font_size = 40
        item.blk.bubble_polygon = [[0, 0], [400, 0], [400, 400], [0, 400]]
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertLessEqual(size, px2pt(40) * 2.0 + 0.01)
        self.assert_readable_interior(item, item.blk.bubble_polygon)

    def test_explicit_size_override_skips_source_cap(self) -> None:
        item = self.make_item('Oh!')
        item.blk._detected_font_size = 40
        item.blk.bubble_polygon = [[0, 0], [400, 0], [400, 400], [0, 400]]
        with patch.object(pcfg, 'let_fntsize_flag', 1):
            self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertGreater(size, px2pt(40) * 1.25)

    def test_tall_rect_fits_long_translation(self) -> None:
        text = 'What is it? What the hell did you just do to me?'
        item = self.make_item(text)
        polygon = [[0, 0], [120, 0], [120, 360], [0, 360]]
        item.blk.bubble_polygon = polygon
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        self.assert_readable_interior(item, polygon)

    def test_underfilled_fit_stays_intact(self) -> None:
        # Hyphenation stays a rescue for sentences: a clean fit that reads
        # at normal dialogue size keeps words intact, and a one/two-word
        # bubble never splits at all — the word IS the bubble, so an
        # underfilled narrow interior keeps `SERIOUSLY?` whole instead of
        # shredding it for size.
        dense = self.make_item(
            'the quick brown fox jumps over the lazy dog again and again '
            'until every single line is completely full of words'
        )
        self.assertTrue(fit_text_to_bubble(dense, None, language='English'))
        self.assertNotIn('\u00ad', dense.toPlainText())
        self.assertNotIn('-', dense.toPlainText())
        sparse = self.make_item('…CUTE? SERIOUSLY?')
        sparse.blk.bubble_polygon = [[0, 0], [60, 0], [60, 300], [0, 300]]
        self.assertTrue(fit_text_to_bubble(sparse, None, language='English'))
        self.assertNotIn('\u00ad', sparse.toPlainText())
        self.assertNotIn('-', sparse.toPlainText())
        self.assertEqual(
            sparse.toPlainText().replace('\n', ' '), '…CUTE? SERIOUSLY?'
        )
        size = sparse.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertLess(size, 7.0)

    def test_minimum_size_fallback_keeps_dp_breaks(self) -> None:
        # An interior nothing fits still centers at minimum size, but with
        # the DP's balanced breaks rather than one unbroken line.
        text = ' '.join(f'word{index}' for index in range(30))
        item = self.make_item(text)
        item.blk.bubble_polygon = [[0, 0], [20, 0], [20, 20], [0, 20]]
        self.assertTrue(fit_text_to_bubble(item, None, language='English'))
        size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertEqual(size, 4.0)
        self.assertIn('\n', item.toPlainText())


class MaskFallbackLayoutTests(_QtItemCase):
    """Mask-fallback auto-layout centers like the bubble path.

    Blocks without a detected outline used to keep their Left alignment
    while bubble-fitted siblings centered.
    """

    def test_mask_fallback_centers_alignment(self) -> None:

        block = TextBlock(
            xyxy=[50, 50, 140, 110], translation='I know everything already.',
        )
        block._bounding_rect = [50, 50, 90, 60]
        block.bubble_polygon = None
        block.fontformat.font_family = 'Arial'
        block.fontformat.font_size = 32
        block.fontformat.alignment = 0
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        self.assertEqual(int(item.fontformat.alignment), 0)

        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (260, 260), (0, 0, 0), 3)
        stub = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(img_array=img),
            pairwidget_list=[],
            auto_textlayout_flag=False,
            _bubble_counts=Counter(),
        )
        with patch.object(
            pcfg.module, 'translate_target', 'English'
        ), patch.object(pcfg.module, 'translate_source', 'English'):
            result = SceneTextManager.layout_textblk(stub, item, text=block.translation)
        self.assertTrue(result)
        self.assertEqual(int(item.fontformat.alignment), 1)

    def test_mask_fallback_caps_growth_at_source_size(self) -> None:

        block = TextBlock(
            xyxy=[50, 50, 140, 110], translation='Oh!',
        )
        block._bounding_rect = [50, 50, 90, 60]
        block.bubble_polygon = None
        # Oversized current font with a small detected raw: without the cap
        # the heuristic keeps 64pt, with it the fit returns near the raw.
        block._detected_font_size = 20
        block.fontformat.font_family = 'Arial'
        block.fontformat.font_size = 64
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)

        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (260, 260), (0, 0, 0), 3)
        stub = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(img_array=img),
            pairwidget_list=[],
            auto_textlayout_flag=True,
            _bubble_counts=Counter(),
        )
        with patch.object(
            pcfg.module, 'translate_target', 'English'
        ), patch.object(pcfg.module, 'translate_source', 'English'), patch.object(
            pcfg, 'let_fntsize_flag', 0
        ), patch.object(pcfg, 'let_autolayout_flag', True):
            result = SceneTextManager.layout_textblk(stub, item, text=block.translation)
        self.assertTrue(result)
        size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertLessEqual(size, px2pt(20) * 1.25 + 0.5)

    def test_mask_fallback_centers_on_bubble(self) -> None:
        # A block whose bubble fit bailed (collapsed interior) still has a
        # detected outline: the mask layout must land it on the bubble's
        # inner center, where the manual Center-in-bubble command puts it.
        block = TextBlock(
            xyxy=[50, 50, 140, 110], translation='I know everything already.',
        )
        block._bounding_rect = [50, 50, 90, 60]
        polygon = [[0, 0], [20, 0], [20, 30], [0, 30]]
        block.bubble_polygon = polygon
        block.set_lines_by_xywh([50, 50, 90, 60])
        block.fontformat.font_family = 'Arial'
        block.fontformat.font_size = 32
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)

        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (40, 40), (260, 260), (0, 0, 0), 3)
        stub = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(img_array=img),
            pairwidget_list=[],
            auto_textlayout_flag=False,
            _bubble_counts=Counter(),
        )
        with patch.object(
            pcfg.module, 'translate_target', 'English'
        ), patch.object(pcfg.module, 'translate_source', 'English'):
            result = SceneTextManager.layout_textblk(stub, item, text=block.translation)
        self.assertTrue(result)
        guide = bubble_inner_center(polygon)
        box = item.absBoundingRect(qrect=True)
        self.assertAlmostEqual(box.center().x(), guide[0], delta=2.0)
        self.assertAlmostEqual(box.center().y(), guide[1], delta=2.0)
