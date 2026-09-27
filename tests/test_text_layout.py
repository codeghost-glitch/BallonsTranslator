import unittest

from ballontranslator.utils.text_layout import (
    Line,
    hyphenate_long_words,
    line_is_valid,
    row_width_profile,
)

try:
    import pyphen  # noqa: F401
    HAS_PYPHEN = True
except Exception:
    HAS_PYPHEN = False


class TestHyphenateLongWords(unittest.TestCase):

    def test_short_tokens_pass_through(self) -> None:
        words, wl = ['OK', 'yes'], [30, 40]
        self.assertEqual(
            hyphenate_long_words(words, wl, lambda s: len(s) * 10, 'en', 60),
            (words, wl),
        )

    def test_long_token_splits_at_linguistic_point(self) -> None:
        if not HAS_PYPHEN:
            self.skipTest('pyphen not installed')
        measure = lambda s: len(s) * 10
        words, wl = hyphenate_long_words(
            ['unquestionably'], [140], measure, 'en', 80
        )
        self.assertGreater(len(words), 1)
        self.assertTrue(words[0].endswith('-'))
        self.assertEqual(
            ''.join(w[:-1] if w.endswith('-') else w for w in words),
            'unquestionably',
        )
        self.assertEqual(wl, [measure(w) for w in words])
        self.assertLessEqual(max(wl), 80)

    def test_progress_guaranteed_on_unbreakable_token(self) -> None:
        # No linguistic point (e.g. a URL-like run): the token stays whole
        # instead of looping forever.
        words, wl = ['x' * 40], [400]
        self.assertEqual(
            hyphenate_long_words(words, wl, lambda s: len(s) * 10, 'en', 80),
            (words, wl),
        )


class TestRowProfileLineBudget(unittest.TestCase):

    def test_budget_shrinks_away_from_center(self) -> None:
        # Same idea as the old ellipse budget, now measured from the
        # outline: a circle 200 tall and 200 wide centred on row 100 is 200
        # wide at the middle and ~97 at 35px off-centre.
        import numpy as np
        circle = [(100.0 + 100 * np.cos(t), 100 + 100 * np.sin(t))
                  for t in np.linspace(0, 2 * np.pi, 720, endpoint=False)]
        left, right, y0 = row_width_profile(np.array(circle, np.float32), 0, 200)
        center_row = Line('abc', 0, 90, 60, 0)   # row center 100, width 200
        edge_row = Line('abc', 0, 160, 60, 0)    # row center 170, width ~143
        # 500 fits the middle row's budget but overruns the edge row's; the
        # grow-balance branch accepts a short line up to max_width/new_len.
        self.assertTrue(
            line_is_valid(center_row, 500, 0, 10000, 0, None, 0, 20,
                          row_profile=(left, right, y0))
        )
        self.assertFalse(
            line_is_valid(edge_row, 500, 0, 10000, 0, None, 0, 20,
                          row_profile=(left, right, y0))
        )

    def test_row_outside_outline_has_no_room(self) -> None:
        import numpy as np
        from ballontranslator.utils.text_layout import row_width_profile
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)
        left, right, y0 = row_width_profile(square, 0, 101)
        inside = Line('abc', 0, 40, 60, 0)
        below = Line('abc', 0, 140, 60, 0)
        self.assertTrue(
            line_is_valid(inside, 90, 0, 10000, 0, None, 0, 20,
                          row_profile=(left, right, y0))
        )
        self.assertFalse(
            line_is_valid(below, 90, 0, 10000, 0, None, 0, 20,
                          row_profile=(left, right, y0))
        )

    def test_rectangular_layout_unchanged_without_profile(self) -> None:
        line = Line('abc', 0, 90, 60, 0)
        self.assertTrue(line_is_valid(line, 110, 0, 10000, 0, None, 0, 20))
        self.assertFalse(line_is_valid(line, 200, 0, 100, 0, None, 0, 20))


if __name__ == '__main__':
    unittest.main()


class TestOpticalHyphenation(unittest.TestCase):

    def test_line_end_hyphen_fills_narrow_budget(self) -> None:
        if not HAS_PYPHEN:
            self.skipTest('pyphen not installed')
        import numpy as np
        from ballontranslator.utils.textblock import TextBlock
        from ballontranslator.utils.text_layout import layout_text

        measure = lambda s: len(s) * 10
        blk = TextBlock(xyxy=[0, 0, 100, 40])
        blk.set_lines_by_xywh([0, 0, 100, 40])
        mask = np.full((120, 300), 255, np.uint8)
        words = ['malls', 'evacuating', 'shopping']
        wl = [measure(w) for w in words]
        hyphenator = __import__('pyphen').Pyphen(lang='en')
        text, xywh, _, _ = layout_text(
            blk, mask, [0, 0, 300, 120], [10, 60],
            list(words), list(wl), ' ', measure(' '), 30,
            max_central_width=120, src_is_cjk=False, tgt_is_cjk=False,
            hyphenator=hyphenator, measure=measure,
        )
        lines = text.split('\n')
        self.assertTrue(any(ln.endswith('-') for ln in lines), text)
        # nothing lost: stripping the added hyphens rejoins the input
        rejoined = ''.join(ln[:-1] if ln.endswith('-') else ln for ln in lines)
        rejoined = rejoined.replace(' ', '')
        self.assertEqual(rejoined, ''.join(words))

    def test_hyphenation_off_without_hyphenator(self) -> None:
        import numpy as np
        from ballontranslator.utils.textblock import TextBlock
        from ballontranslator.utils.text_layout import layout_text

        measure = lambda s: len(s) * 10
        blk = TextBlock(xyxy=[0, 0, 100, 40])
        blk.set_lines_by_xywh([0, 0, 100, 40])
        mask = np.full((120, 300), 255, np.uint8)
        words = ['malls', 'evacuating', 'shopping']
        wl = [measure(w) for w in words]
        text, _, _, _ = layout_text(
            blk, mask, [0, 0, 300, 120], [10, 60],
            list(words), list(wl), ' ', measure(' '), 30,
            max_central_width=120, src_is_cjk=False, tgt_is_cjk=False,
        )
        self.assertNotIn('-', text.replace(' ', ''))


class TestCenterLayoutHyphenOrder(unittest.TestCase):

    def test_hyphenation_preserves_word_order_center_alignment(self) -> None:
        # The center path builds lines outward in both directions; a head /
        # tail swap there reorders text ('radio' -> 'dio ... ra-'). Whatever
        # the wrap does, the joined output must equal the input words.
        if not HAS_PYPHEN:
            self.skipTest('pyphen not installed')
        import numpy as np
        from ballontranslator.utils.textblock import TextBlock, TextAlignment
        from ballontranslator.utils.text_layout import layout_text

        measure = lambda s: len(s) * 10
        blk = TextBlock(xyxy=[0, 0, 400, 80])
        blk.set_lines_by_xywh([0, 0, 400, 80])
        blk.alignment = TextAlignment.Center
        mask = np.full((140, 400), 255, np.uint8)
        words = ['An', 'analog', 'radio', 'signal', 'came', 'through']
        wl = [measure(w) for w in words]
        hyphenator = __import__('pyphen').Pyphen(lang='en')
        text, _, _, _ = layout_text(
            blk, mask, [0, 0, 400, 140], [200, 70],
            list(words), list(wl), ' ', measure(' '), 30,
            max_central_width=130, src_is_cjk=False, tgt_is_cjk=False,
            hyphenator=hyphenator, measure=measure,
        )
        lines = text.split('\n')
        rejoined = ''.join(
            ln[:-1] if ln.endswith('-') else ln for ln in lines
        ).replace(' ', '')
        self.assertEqual(rejoined, ''.join(words))
