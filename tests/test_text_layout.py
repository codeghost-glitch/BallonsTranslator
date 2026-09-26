import unittest

from ballontranslator.utils.text_layout import (
    Line,
    hyphenate_long_words,
    line_is_valid,
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


class TestEllipseLineBudget(unittest.TestCase):

    def test_budget_shrinks_away_from_center(self) -> None:
        # Ellipse (cx=0, cy=100, a=60, b=80); line_height=20.
        ellipse = (0, 100, 60, 80)
        center_row = Line('abc', 0, 90, 60, 0)   # row center: 100 -> cap 120
        edge_row = Line('abc', 0, 160, 60, 0)    # row center: 170 -> cap ~58
        self.assertTrue(
            line_is_valid(center_row, 110, 0, 10000, 0, None, 0, 20,
                          ellipse=ellipse)
        )
        self.assertFalse(
            line_is_valid(edge_row, 110, 0, 10000, 0, None, 0, 20,
                          ellipse=ellipse)
        )

    def test_rectangular_layout_unchanged_without_ellipse(self) -> None:
        line = Line('abc', 0, 90, 60, 0)
        self.assertTrue(line_is_valid(line, 110, 0, 10000, 0, None, 0, 20))
        self.assertFalse(line_is_valid(line, 200, 0, 100, 0, None, 0, 20))


if __name__ == '__main__':
    unittest.main()
