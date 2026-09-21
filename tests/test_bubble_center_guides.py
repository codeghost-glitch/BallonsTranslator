import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest

from qtpy.QtCore import QCoreApplication, QEvent, QPointF, QRectF, Qt
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.commands import CenterInBubbleCommand
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.bubble import (
    BUBBLE_CENTER_SNAP_RADIUS,
    bubble_inner_center,
    snap_point_to_bubble_center,
)
from ballontranslator.utils.textblock import TextBlock


def make_block(translation='Hello'):
    block = TextBlock(
        xyxy=[50, 50, 140, 110],
        translation=translation,
        bubble_polygon=[[0, 0], [220, 0], [220, 180], [0, 180]],
    )
    block._bounding_rect = [50, 50, 90, 60]
    block.fontformat.font_family = 'Arial'
    return block


class BubbleInnerCenterTests(unittest.TestCase):
    def test_center_matches_interior_rect(self) -> None:
        center = bubble_inner_center([[0, 0], [100, 0], [100, 100], [0, 100]])
        self.assertIsNotNone(center)
        assert center is not None
        x, y, rect = center
        left, top, width, height = rect
        self.assertAlmostEqual(x, left + width / 2.0)
        self.assertAlmostEqual(y, top + height / 2.0)

    def test_invalid_polygon_returns_none(self) -> None:
        self.assertIsNone(bubble_inner_center([[0, 0]]))
        self.assertIsNone(bubble_inner_center(None))

    def test_tailed_bubble_centroid_stays_on_body(self) -> None:
        # A spike to x=200 must not drag the center off the 0-100 body the
        # way a bounding-box center (x=100) would.
        import cv2
        import numpy as np

        polygon = [[0, 0], [100, 0], [200, 50], [100, 100], [0, 100]]
        center = bubble_inner_center(polygon)
        self.assertIsNotNone(center)
        assert center is not None
        x, y, _rect = center
        self.assertLess(x, 90.0)
        self.assertGreater(x, 60.0)
        contour = np.asarray(polygon, dtype=np.float32)
        self.assertGreaterEqual(
            cv2.pointPolygonTest(contour, (float(x), float(y)), False), 0.0
        )


class SnapHelperTests(unittest.TestCase):
    def test_snaps_per_axis(self) -> None:
        snapped = snap_point_to_bubble_center(100.0, 103.0, 112.0, 200.0)
        self.assertEqual(snapped, (112.0, 103.0, True))

    def test_far_point_does_not_snap(self) -> None:
        self.assertEqual(
            snap_point_to_bubble_center(0.0, 0.0, 100.0, 100.0),
            (0.0, 0.0, False),
        )

    def test_radius_is_shared_constant(self) -> None:
        self.assertGreater(BUBBLE_CENTER_SNAP_RADIUS, 0)


class BubbleCenterItemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def make_item(self, translation='Hello') -> TextBlkItem:
        item = TextBlkItem(make_block(translation), 0)
        self.addCleanup(item.deleteLater)
        return item

    def test_center_command_moves_and_sets_alignment_with_undo(self) -> None:
        item = self.make_item()
        guide = bubble_inner_center(item.blk.bubble_polygon)
        self.assertIsNotNone(guide)
        assert guide is not None
        bubble_x, bubble_y, _rect = guide
        size = item.geometry_controller.logical_rect().size()
        target = QPointF(
            bubble_x - size.width() / 2.0, bubble_y - size.height() / 2.0
        )
        before = QPointF(item.logical_position())
        command = CenterInBubbleCommand([item], [before], [target], [0], [1])
        self.assertFalse(command.isObsolete())
        command.redo()
        self.assertEqual(item.logical_position(), target)
        self.assertEqual(int(item.fontformat.alignment), 1)
        command.undo()
        self.assertEqual(item.logical_position(), before)
        self.assertEqual(int(item.fontformat.alignment), 0)

    def test_center_command_obsolete_without_change(self) -> None:
        item = self.make_item()
        pos = QPointF(item.logical_position())
        command = CenterInBubbleCommand([item], [pos], [QPointF(pos)], [1], [1])
        self.assertTrue(command.isObsolete())

    def test_drag_snap_triggers_near_center(self) -> None:
        canvas = Canvas()
        self.addCleanup(canvas.deleteLater)
        self.addCleanup(canvas.gv.deleteLater)
        item = self.make_item()
        canvas.attach_text_item(item)
        item.setSelected(True)

        class _Move:
            def modifiers(self):
                return Qt.KeyboardModifier.NoModifier

        item.set_logical_position(QPointF(70, 65))
        self.assertTrue(item._maybe_snap_to_bubble_center(_Move()))
        guide = bubble_inner_center(item.blk.bubble_polygon)
        assert guide is not None
        bubble_x, bubble_y, _rect = guide
        size = item.geometry_controller.logical_rect().size()
        self.assertAlmostEqual(
            item.logical_position().x(), bubble_x - size.width() / 2.0
        )
        self.assertAlmostEqual(
            item.logical_position().y(), bubble_y - size.height() / 2.0
        )

    def test_drag_snap_ignores_far_and_alt(self) -> None:
        canvas = Canvas()
        self.addCleanup(canvas.deleteLater)
        self.addCleanup(canvas.gv.deleteLater)
        item = self.make_item()
        canvas.attach_text_item(item)
        item.setSelected(True)

        class _Move:
            def modifiers(self):
                return Qt.KeyboardModifier.NoModifier

        class _AltMove:
            def modifiers(self):
                return Qt.KeyboardModifier.AltModifier

        item.set_logical_position(QPointF(0, 0))
        self.assertFalse(item._maybe_snap_to_bubble_center(_Move()))
        self.assertEqual(item.logical_position(), QPointF(0, 0))
        item.set_logical_position(QPointF(70, 65))
        self.assertFalse(item._maybe_snap_to_bubble_center(_AltMove()))
        self.assertEqual(item.logical_position(), QPointF(70, 65))

    def test_drag_snap_skips_multi_selection(self) -> None:
        canvas = Canvas()
        self.addCleanup(canvas.deleteLater)
        self.addCleanup(canvas.gv.deleteLater)
        first = self.make_item('First')
        second = self.make_item('Second')
        canvas.attach_text_item(first)
        canvas.attach_text_item(second)
        first.setSelected(True)
        second.setSelected(True)

        class _Move:
            def modifiers(self):
                return Qt.KeyboardModifier.NoModifier

        first.set_logical_position(QPointF(70, 65))
        self.assertFalse(first._maybe_snap_to_bubble_center(_Move()))

    def test_fallback_keeps_longest_word_intact(self) -> None:
        # A 24px-wide bubble has a ~16px interior while COME at the 4pt
        # floor measures ~20px. The anywhere-wrap would visually split it
        # into COM/E; instead the box widens symmetrically to the word.
        from ballontranslator.ui.text_engine.bubble_layout import (
            _longest_word_width,
            fit_text_to_bubble,
        )

        block = make_block('COME ON.')
        block.bubble_polygon = [[0, 0], [24, 0], [24, 300], [0, 300]]
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        self.assertTrue(
            fit_text_to_bubble(item, None, language='English')
        )
        box = item.absBoundingRect(qrect=True)
        self.assertGreaterEqual(
            box.width(), _longest_word_width(item) - 0.5
        )
        self.assertIn('COME', item.toPlainText().split('\n'))

    def test_success_path_never_exceeds_word_width(self) -> None:
        # The fit search must shrink until the longest word truly fits the
        # box; even a 1-2px overflow is visually split mid-word.
        from ballontranslator.ui.text_engine.bubble_layout import (
            _longest_word_width,
            fit_text_to_bubble,
        )

        block = make_block('COME ON.')
        block.bubble_polygon = [[0, 0], [30, 0], [30, 300], [0, 300]]
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        self.assertTrue(
            fit_text_to_bubble(item, None, language='English')
        )
        box = item.absBoundingRect(qrect=True)
        self.assertLessEqual(
            _longest_word_width(item), box.width() + 0.5
        )

    def test_floor_fit_stays_intact(self) -> None:
        # A clean fit pinned to the 4pt floor grows via dictionary splits:
        # the fit never inserts soft hyphens and never drops letters, and
        # every split carries a visible hyphen.
        from ballontranslator.ui.text_engine.bubble_layout import (
            fit_text_to_bubble,
        )

        block = make_block('KUNIEDA! SMARTPHONES ARE PROHIBITED.')
        block.bubble_polygon = [[0, 0], [70, 0], [70, 380], [0, 380]]
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        self.assertTrue(
            fit_text_to_bubble(item, None, language='English')
        )
        self.assertNotIn('\u00ad', item.toPlainText())
        self.assertEqual(
            item.toPlainText().replace('-', '').replace('\n', '').replace(' ', ''),
            'KUNIEDA!SMARTPHONESAREPROHIBITED.',
        )
        size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertLess(size, 6.0)

    def test_unfittable_bubble_keeps_words_intact(self) -> None:
        # A bubble too narrow for whole words gets the typesetter rescue:
        # words split at dictionary points with literal hyphens so letters
        # are never dropped and soft hyphens are never inserted.
        from ballontranslator.ui.text_engine.bubble_layout import (
            fit_text_to_bubble,
        )

        block = make_block('KUNIEDA! SMARTPHONES ARE PROHIBITED.')
        block.bubble_polygon = [[0, 0], [50, 0], [50, 380], [0, 380]]
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        self.assertTrue(
            fit_text_to_bubble(item, None, language='English')
        )
        self.assertNotIn('\u00ad', item.toPlainText())
        self.assertEqual(
            item.toPlainText().replace('-', '').replace('\n', '').replace(' ', ''),
            'KUNIEDA!SMARTPHONESAREPROHIBITED.',
        )
        size = item.document().begin().begin().fragment().charFormat().fontPointSize()
        self.assertGreater(size, 3.99)
        self.assertLessEqual(size, 8.0)

    def test_selected_guide_paints_center_cross(self) -> None:
        canvas = Canvas()
        self.addCleanup(canvas.deleteLater)
        self.addCleanup(canvas.gv.deleteLater)
        item = self.make_item()
        canvas.attach_text_item(item)
        item.setSelected(True)
        # Center the box so the snap highlight path is also exercised.
        guide = bubble_inner_center(item.blk.bubble_polygon)
        assert guide is not None
        bubble_x, bubble_y, _rect = guide
        size = item.geometry_controller.logical_rect().size()
        item.set_logical_position(
            QPointF(
                bubble_x - size.width() / 2.0,
                bubble_y - size.height() / 2.0,
            )
        )
        image = QImage(300, 300, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            canvas.gv.drawForeground(painter, QRectF(0, 0, 300, 300))
        finally:
            painter.end()
        self.assertGreater(image.pixelColor(110, 90).alpha(), 0)
        self.assertTrue(canvas.gv.bubble_guides)


if __name__ == '__main__':
    unittest.main()
