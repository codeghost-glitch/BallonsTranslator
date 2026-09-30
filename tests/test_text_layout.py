import unittest

from ballontranslator.utils.text_layout import (
    _MIN_PIECE as MIN_PIECE,
    Line,
    hyphenate_long_words,
    line_is_valid,
    row_width_profile,
    split_run_tokens,
)

try:
    import pyphen  # noqa: F401
    HAS_PYPHEN = True
except Exception:
    HAS_PYPHEN = False



# text_layout marks the breaks it makes with a private sentinel; the split
# results carry it until _join_hyphen_runs restores the visible hyphen.
BREAK = '\x1e'

class TestHyphenateLongWords(unittest.TestCase):

    def test_short_tokens_pass_through(self) -> None:
        words, wl = ['OK', 'yes'], [30, 40]
        self.assertEqual(
            hyphenate_long_words(words, wl, lambda s: len(s) * 10, 'en', 60),
            (words, wl),
        )

    def test_interjection_is_not_hyphenated(self) -> None:
        # A tight balloon holding a 5-character interjection used to
        # force-break it into 'Wha-t?'. Words under the typographic floor
        # stay whole; the fit shrinks them instead.
        words, wl = ['What?'], [400]
        self.assertEqual(
            hyphenate_long_words(words, wl, lambda s: len(s) * 10, 'en', 60),
            (words, wl),
        )

    def test_word_at_the_floor_still_splits(self) -> None:
        # The floor must not swallow real hyphenation: a 6-character word
        # too wide for the budget still breaks. The budget has to admit the
        # only legal split of a six-letter word, 3+3 - 'can-' is four
        # characters wide, so anything below 40 here would be asking for the
        # stub that _MIN_PIECE exists to refuse.
        if not HAS_PYPHEN:
            self.skipTest('pyphen not installed')
        measure = lambda s: len(s) * 10
        words, _ = hyphenate_long_words(['cannot'], [60], measure, 'en', 40)
        self.assertEqual(words, ['can' + BREAK, 'not'])

    def test_long_token_splits_at_linguistic_point(self) -> None:
        if not HAS_PYPHEN:
            self.skipTest('pyphen not installed')
        measure = lambda s: len(s) * 10
        words, wl = hyphenate_long_words(
            ['unquestionably'], [140], measure, 'en', 80
        )
        self.assertGreater(len(words), 1)
        self.assertTrue(words[0].endswith(BREAK))
        self.assertEqual(
            ''.join(w[:-1] if w.endswith(BREAK) else w for w in words),
            'unquestionably',
        )
        self.assertEqual(wl, [measure(w) for w in words])
        self.assertLessEqual(max(wl), 80)

    def test_unbreakable_token_is_force_broken_and_terminates(self) -> None:
        # No linguistic point (a URL-like run): rather than stay whole and
        # overflow the line - pinning the fit small and leaving a tall
        # balloon's height unused - it is force-broken into line-fitting
        # chunks, and the loop still terminates.
        measure = lambda s: len(s) * 10
        words, wl = hyphenate_long_words(['x' * 40], [400], measure, 'en', 80)
        self.assertGreater(len(words), 1)
        self.assertEqual(
            ''.join(w[:-1] if w.endswith(BREAK) else w for w in words),
            'x' * 40,
        )
        self.assertLessEqual(max(wl), 80)


    def test_no_piece_is_shorter_than_the_readable_minimum(self) -> None:
        # The complaint that produced _MIN_PIECE: a word twice its budget
        # was cut three times and rendered 'fre- quentl- y.'. Every piece
        # on both sides of every break must stay readable, over a range of
        # budgets wide enough to force both the pyphen and forced paths.
        if not HAS_PYPHEN:
            self.skipTest('pyphen not installed')
        measure = lambda s: len(s) * 10
        for word in ('frequently.', 'extraordinary', 'unquestionably', 'x' * 24):
            for budget in range(20, 200, 7):
                words, _ = hyphenate_long_words([word], [400], measure, 'en', budget)
                for piece in words:
                    text = piece[:-1] if piece.endswith(BREAK) else piece
                    self.assertGreaterEqual(
                        len(text), MIN_PIECE,
                        f'{word!r} at {budget}px produced a stub {piece!r}',
                    )
                # Pieces are still the original word, and the loop terminated.
                self.assertEqual(
                    ''.join(w[:-1] if w.endswith(BREAK) else w for w in words),
                    word,
                )

    def test_a_word_too_tight_to_split_is_left_whole(self) -> None:
        # Refusing a stub must not shrink the split to a two-letter wing
        # either: 'mother' at 20px cannot become 'mo-ther' (the tail is four)
        # nor 'm-other' (the head is one), so it stays whole and overhangs.
        measure = lambda s: len(s) * 10
        self.assertEqual(
            hyphenate_long_words(['mother'], [60], measure, 'en', 20),
            (['mother'], [60]),
        )


    def test_a_run_of_words_is_never_broken_across_its_space(self) -> None:
        # seg_eng glues a one- or two-letter word to its neighbour so a line
        # can carry them as one unit, and the result is a single token that
        # is longer than the floor. Splitting it rendered 'sure-' / ' to'.
        measure = lambda s: len(s) * 10
        for run in ('sure to', 'up on', 'a tod'):
            self.assertEqual(
                hyphenate_long_words([run], [400], measure, 'en', 20),
                ([run], [400]),
            )

    def test_a_linguistic_word_is_cut_at_most_twice(self) -> None:
        # Cutting a word until it fits the budget shredded it: at 60px
        # 'frequently.' came out as 'fre-' / 'quentl-' / 'y.' with a
        # one-letter tail. Two cuts still divide a word far wider than the
        # line; what is left is closed by the line-breaker's own in-line
        # hyphenation.
        measure = lambda s: len(s) * 10
        words, _ = hyphenate_long_words(['frequently.'], [110], measure, 'en', 60)
        self.assertLessEqual(len(words), 3)
        self.assertTrue(all(len(w.rstrip(BREAK)) >= MIN_PIECE for w in words))

    def test_a_token_with_no_break_point_is_still_cut_until_it_fits(self) -> None:
        # The cut limit protects typography, not width: a run pyphen has no
        # point for has nothing to protect, and leaving it whole would
        # overflow its line.
        measure = lambda s: len(s) * 10
        words, wl = hyphenate_long_words(['x' * 40], [400], measure, 'en', 80)
        self.assertLessEqual(max(wl), 80)


class TestSplitRunTokens(unittest.TestCase):

    def test_a_run_wider_than_the_budget_gives_its_spaces_back(self) -> None:
        # The wrap cannot break a token, so a glued run wider than the line
        # sets the line width and with it the font: "Treating me like a
        # toddler..." fitted at 8.6pt in a 127x174 balloon, 40% of the
        # height, because 'me like a' was 125px wide.
        self.assertEqual(
            split_run_tokens(['Treating', 'me like a', 'toddler...'],
                             [102, 114, 127], lambda s: len(s) * 10, 90),
            (['Treating', 'me', 'like', 'a', 'toddler...'], [102, 20, 40, 10, 127]),
        )

    def test_a_run_that_fits_keeps_its_glue(self) -> None:
        # The glue is what stops "a" from being orphaned on a line of its
        # own, so a run inside the budget must not be taken apart.
        words, wl = ['And', 'sure to', 'up on'], [30, 70, 60]
        self.assertEqual(
            split_run_tokens(words, wl, lambda s: len(s) * 10, 90), (words, wl),
        )

    def test_a_plain_word_is_never_split(self) -> None:
        # Only a run has a space to break at; a lone word is the
        # hyphenation pre-pass's business, not this one's.
        words, wl = ['frequently.'], [138]
        self.assertEqual(
            split_run_tokens(words, wl, lambda s: len(s) * 10, 90), (words, wl),
        )

    def test_no_budget_leaves_every_run_alone(self) -> None:
        words, wl = ['me like a'], [114]
        self.assertEqual(
            split_run_tokens(words, wl, lambda s: len(s) * 10, 0), (words, wl),
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

    def test_translator_hyphen_is_not_swallowed(self) -> None:
        # A '-' the translator typed ends a real token. Treating it as one
        # of our own breaks deleted it and welded the words together.
        import numpy as np
        from ballontranslator.utils.textblock import TextBlock
        from ballontranslator.utils.text_layout import layout_text

        measure = lambda s: len(s) * 10
        blk = TextBlock(xyxy=[0, 0, 100, 40])
        blk.set_lines_by_xywh([0, 0, 100, 40])
        mask = np.full((120, 300), 255, np.uint8)
        words = ['fine-', "I'll", 'go']
        wl = [measure(w) for w in words]
        text, _, _, _ = layout_text(
            blk, mask, [0, 0, 300, 120], [10, 60],
            list(words), list(wl), ' ', measure(' '), 30,
            max_central_width=300, src_is_cjk=False, tgt_is_cjk=False,
        )
        self.assertEqual(text, "fine- I'll go")

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
