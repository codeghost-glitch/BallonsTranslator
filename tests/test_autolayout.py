import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from ballontranslator.utils.autolayout import (
    ComicBalloon,
    LineProfile,
    SegmentMeasure,
    balloon_from_mask,
    balloon_from_polygon,
    build_segments,
    comic_break_penalty,
    comic_line_breaks,
    exact_profiled_line_breaks,
    largest_fitting_font_size,
    line_breaks_with_policy,
)
from ballontranslator.utils.text_layout import layout_text, seg_text_paragraphs
from ballontranslator.utils.textblock import TextBlock


def _rect_balloon(width: float, height: float, air: float = 4.0) -> ComicBalloon:
    return ComicBalloon(
        width, height,
        [[(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]],
        air,
    )


def _word_segments(words, width=40.0, delimiter_width=5.0):
    text = ' '.join(words)
    return build_segments(words, [width] * len(words), delimiter_width, ' ', text)


class BreakPenaltyTests(unittest.TestCase):
    def test_sentence_end_is_free(self) -> None:
        self.assertEqual(comic_break_penalty('Hello. World', 6), 0.0)

    def test_dangling_article_costs_more_than_clean_break(self) -> None:
        self.assertLess(
            comic_break_penalty('Hello the world', len('Hello')),
            comic_break_penalty('Hello the world', len('Hello the')),
        )

    def test_dp_avoids_stranding_articles(self) -> None:
        balloon = _rect_balloon(100.0, 40.0)
        segments = _word_segments(['aa', 'the', 'bb'], width=30.0)
        result = comic_line_breaks(
            segments, balloon, False, 12.0, 10.0, (4.0, 4.0), False,
        )
        self.assertFalse(result.overflowed)
        self.assertEqual(result.breaks, [1, 3])

    def test_dp_keeps_subject_with_verb(self) -> None:
        segments = build_segments(['Probably', 'Ikebukuro,', 'I', 'think.'],
                                  [65.0, 80.0, 6.0, 40.0], 5.0, ' ',
                                  'Probably Ikebukuro, I think.')
        result = comic_line_breaks(segments, _rect_balloon(100.0, 60.0, 0.0),
                                   False, 16.0, 14.0, (0.0, 0.0), False)
        self.assertEqual(result.breaks, [1, 2, 4])


class BalloonProfileTests(unittest.TestCase):
    def test_rectangular_contour_shares_one_axis(self) -> None:
        balloon = _rect_balloon(120.0, 80.0)
        sets = balloon.line_profile_candidates(False, 3, 12.0, 10.0, 4.0, 4.0)
        self.assertEqual(len(sets), 1)
        axes = {round(profile.center_offset, 6) for profile in sets[0]}
        self.assertEqual(len(axes), 1)

    def test_ellipse_fallback_is_widest_in_the_middle(self) -> None:
        balloon = ComicBalloon(120.0, 80.0, [], 4.0)
        profiles = balloon.line_profile_candidates(False, 3, 12.0, 10.0, 4.0, 4.0)[0]
        widths = [profile.width for profile in profiles]
        self.assertGreaterEqual(widths[1], widths[0])
        self.assertGreaterEqual(widths[1], widths[2])

    def test_ellipse_rows_keep_their_own_widths(self) -> None:
        # Elliptic shaping: end rows stay narrower than the middle row
        # instead of every line clamping to the narrowest span.
        balloon = ComicBalloon(120.0, 80.0, [], 4.0)
        profiles = balloon.line_profile_candidates(False, 3, 12.0, 10.0, 4.0, 4.0)[0]
        widths = [profile.width for profile in profiles]
        self.assertGreater(widths[1], widths[0])
        self.assertGreater(widths[1], widths[2])
        self.assertGreater(widths[0], 0.0)

    def test_rect_contour_breaks_fewer_lines_than_tapered_ellipse(self) -> None:
        # Rectangular Qt placement should use the full interior width for
        # every line instead of pinching end lines: fewer lines at one size
        # leaves headroom for a bigger font. Realistic advances keep this
        # independent of whatever font the test box substitutes.
        words = ['I', 'still', "don't", 'know', 'much', 'about', 'the',
                 'human', 'world.']
        advances = [8.0, 28.0, 38.0, 32.0, 30.0, 36.0, 22.0, 38.0, 40.0]
        segments = build_segments(words, advances, 4.0, ' ', ' '.join(words))
        tapered = comic_line_breaks(
            segments, ComicBalloon(164.0, 66.0, [], 0.0),
            False, 14.0, 11.0, (0.0, 0.0), False,
        )
        rect = comic_line_breaks(
            segments,
            ComicBalloon(164.0, 66.0,
                         [[(0.0, 0.0), (164.0, 0.0), (164.0, 66.0), (0.0, 66.0)]],
                         0.0),
            False, 14.0, 11.0, (0.0, 0.0), False,
        )
        self.assertFalse(rect.overflowed)
        self.assertLessEqual(len(rect.breaks), len(tapered.breaks))
        self.assertTrue(
            all(profile.width == 164.0 for profile in rect.profiles)
        )

    def test_malformed_polygon_falls_back_to_ellipse(self) -> None:
        balloon = balloon_from_polygon('bad', (0.0, 0.0), 50.0, 40.0)
        self.assertEqual(balloon.contours, [])
        self.assertEqual((balloon.width, balloon.height), (50.0, 40.0))

    def test_mask_balloon_matches_mask_extent(self) -> None:
        balloon = balloon_from_mask(np.ones((10, 20), dtype=np.uint8) * 255)
        self.assertEqual((balloon.width, balloon.height), (20.0, 10.0))


class FontSearchTests(unittest.TestCase):
    def test_largest_first_search_skips_failing_sizes(self) -> None:
        best = largest_fitting_font_size(
            9.0, 24.0, lambda size: size, lambda size: 10.0 <= size <= 12.0,
        )
        self.assertIsNotNone(best)
        assert best is not None
        self.assertLessEqual(best, 12.0)
        self.assertGreater(best, 11.9)

    def test_last_resort_keeps_clean_setting_when_it_fits(self) -> None:
        segments = [SegmentMeasure(10.0)]
        result = line_breaks_with_policy(segments, 12.0, True)
        self.assertFalse(result.overflowed)
        self.assertEqual(result.breaks, [1])

    def test_profiled_dp_reports_overflow(self) -> None:
        profiles = [LineProfile(width=10.0)]
        result = exact_profiled_line_breaks([SegmentMeasure(40.0)], profiles, False)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.overflowed)


class LayoutTextTests(unittest.TestCase):
    def test_single_line_text_stays_on_one_line(self) -> None:
        blk = TextBlock(xyxy=[0, 0, 100, 60])
        mask = np.ones((60, 100), dtype=np.uint8) * 255
        text, xywh = layout_text(
            blk, mask, [0, 0, 100, 60], [50, 30],
            ['hello', 'world'], [30, 30], ' ', 4, 12,
        )
        self.assertEqual(text, 'hello world')
        self.assertEqual((xywh[2], xywh[3]), (64, 12))

    def test_narrow_balloon_wraps_whole_words(self) -> None:
        blk = TextBlock(xyxy=[0, 0, 40, 60])
        mask = np.ones((60, 40), dtype=np.uint8) * 255
        text, xywh = layout_text(
            blk, mask, [0, 0, 40, 60], [20, 30],
            ['hello', 'world'], [30, 30], ' ', 4, 12,
        )
        self.assertEqual(text, 'hello\nworld')
        self.assertGreater(xywh[2], 0)
        self.assertEqual(xywh[3], 24)

    def test_vertical_layout_transposes_canvas(self) -> None:
        blk = TextBlock(xyxy=[0, 0, 100, 60])
        blk.vertical = True
        mask = np.ones((60, 100), dtype=np.uint8) * 255
        text, xywh = layout_text(
            blk, mask, [0, 0, 100, 60], [50, 30],
            ['hello', 'world'], [30, 30], ' ', 4, 12, vertical=True,
        )
        self.assertIn('\n', text)
        self.assertEqual((xywh[2], xywh[3]), (24, 30))

    def test_paragraph_breaks_are_preserved(self) -> None:
        units, advances, delimiter, glue = seg_text_paragraphs(
            'hello\nworld', 'English', len,
        )
        self.assertEqual(delimiter, ' ')
        self.assertIn('\n', units)
        self.assertEqual(len(units), len(advances))
        self.assertEqual(len(units), len(glue))

    def test_full_width_spaces_break_like_ascii_spaces(self) -> None:
        units, advances, delimiter, glue = seg_text_paragraphs(
            'AH　HA　HA', 'English', len,
        )
        self.assertEqual(units, ['AH', 'HA', 'HA'])
        self.assertEqual(advances, [2.0, 2.0, 2.0])
        self.assertEqual(glue, [False, False, False])

    def test_dash_compounds_split_with_glue(self) -> None:
        units, advances, delimiter, glue = seg_text_paragraphs(
            'EHH—BUT CUTE', 'English', len,
        )
        self.assertEqual(units, ['EHH—', 'BUT', 'CUTE'])
        self.assertEqual(advances, [4.0, 3.0, 4.0])
        self.assertEqual(glue, [False, True, False])

    def test_break_before_dash_is_penalized(self) -> None:
        # `Hayate-` / `kun` is correct; `Hayate` / `-kun` is not.
        segments = build_segments(
            ['Go', '-go'], [20.0, 20.0], 4.0, ' ', 'Go -go',
        )
        self.assertEqual(
            [round(segment.break_penalty, 1) for segment in segments],
            [1100.0, 100.0],
        )

    def test_dash_split_enables_taller_breaks(self) -> None:        # 'EHH—BUT' (65px) cannot share a 60px line, but its dash pieces can,
        # unlocking a clean layout instead of an overflow.
        balloon = ComicBalloon(
            60.0, 60.0, [[(0.0, 0.0), (60.0, 0.0), (60.0, 60.0), (0.0, 60.0)]],
            0.0,
        )
        split = comic_line_breaks(
            build_segments(
                ['EHH—', 'BUT', 'CUTE'], [35.0, 30.0, 30.0], 5.0, ' ',
                'EHH—BUTCUTE', glue=[False, True, False],
            ),
            balloon, False, 12.0, 10.0, (0.0, 0.0), False,
        )
        self.assertFalse(split.overflowed)
        glued = comic_line_breaks(
            build_segments(
                ['EHH—BUT', 'CUTE'], [65.0, 30.0], 5.0, ' ',
                'EHH—BUT CUTE',
            ),
            balloon, False, 12.0, 10.0, (0.0, 0.0), False,
        )
        self.assertTrue(glued.overflowed)

    def test_tall_narrow_mask_wraps_lines(self) -> None:
        blk = TextBlock(xyxy=[0, 0, 60, 300])
        mask = np.ones((300, 60), dtype=np.uint8) * 255
        text, xywh = layout_text(
            blk, mask, [0, 0, 60, 300], [30, 150],
            ['this', 'is', 'bad', 'wait', 'a', 'second'],
            [24, 12, 24, 24, 8, 36], ' ', 4, 12,
        )
        self.assertGreaterEqual(text.count('\n'), 1)
        self.assertGreater(xywh[2], 0)
        self.assertGreater(xywh[3], 12)


if __name__ == '__main__':
    unittest.main()
