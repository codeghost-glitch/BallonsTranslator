"""Comic balloon auto-layout in pure Python.

Balloon-aware line breaking and font-size search; shaping stays with Qt:

- Each candidate line count gets per-line usable widths sampled across the
  glyph ink band from the balloon contour (polygon scanline) or an ellipse
  fallback, so elliptic balloons get narrow end lines and wide middles, and
  all rows share one visual axis so centering placement stays inside the ink.
  Placement centers every line in its rectangle, which is what keeps the
  paragraph coherent instead of a drifting per-line axis.
- Breaks are chosen by dynamic programming minimizing squared slack plus a
  heavy overflow penalty, a hyphen penalty, and a linguistic break penalty
  (sentence ends cheapest, dangling articles most expensive).
- Font sizes are searched largest-first (non-monotonic balloon fits) with a
  probe plus binary refine, and hyphenation is last-resort: clean setting
  wins unless hyphenation recovers a visible pixel.

>>> isinstance(LINE_BREAK_HYPHEN_PENALTY, float)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

LINE_BREAK_HYPHEN_PENALTY = 2000.0
LINE_BREAK_OVERFLOW_MULTIPLIER = 10000.0
COMIC_LINE_OVERFLOW_PENALTY = 1000000.0
COMIC_MAX_LINES = 64

T = TypeVar('T')


@dataclass
class SegmentMeasure:
    """One breakable unit (word/run) with linguistic break metadata.

    >>> SegmentMeasure(advance=10.0).trailing_advance
    0.0
    """

    advance: float
    trailing_advance: float = 0.0
    break_suffix_advance: float = 0.0
    break_penalty: float = 0.0
    is_mandatory: bool = False


@dataclass
class LineProfile:
    """Usable width and placement for one laid-out line.

    >>> LineProfile(width=10.0, center_offset=0.0, block_baseline=0.0).width
    10.0
    """

    width: float
    center_offset: float = 0.0
    block_baseline: float = 0.0


@dataclass
class LineBreakResult:
    """Chosen break offsets plus the profiles they were fitted against.

    >>> LineBreakResult(breaks=[1], profiles=[], overflowed=False, cost=0.0).overflowed
    False
    """

    breaks: List[int]
    profiles: List[LineProfile]
    overflowed: bool
    cost: float


@dataclass
class ComicBalloon:
    """Balloon bounds plus optional inset contours in layout-local coordinates.

    ``contours`` holds physical wall polygons (page-space); every wall owns
    its full air margin and walls are intersected when several are given.

    >>> ComicBalloon(100.0, 60.0, [], 4.0).air(10.0)
    10.0
    """

    width: float
    height: float
    contours: List[List[Tuple[float, float]]] = field(default_factory=list)
    minimum_air: float = 0.0

    def air(self, ink_thickness: float) -> float:
        """Padding owed to one axis: at least the glyph ink band.

        >>> ComicBalloon(10.0, 10.0, [], 6.0).air(2.0)
        6.0
        """
        return max(self.minimum_air, ink_thickness)

    def inline_extent(self, vertical: bool, air_x: float, air_y: float) -> float:
        """Widest usable inline run ignoring contour taper.

        >>> ComicBalloon(100.0, 60.0, [], 0.0).inline_extent(False, 4.0, 5.0)
        92.0
        """
        if vertical:
            return max(1.0, self.height - air_y * 2.0)
        return max(1.0, self.width - air_x * 2.0)

    def line_profile_candidates(
        self, vertical: bool, line_count: int, line_height: float,
        ink_thickness: float, air_x: float, air_y: float,
    ) -> List[List[LineProfile]]:
        """Yield centered profile sets for one line count (usually one).

        >>> balloon = ComicBalloon(120.0, 80.0, [], 4.0)
        >>> sets = balloon.line_profile_candidates(False, 3, 12.0, 10.0, 4.0, 4.0)
        >>> [round(p.width, 1) for p in sets[0]][1] >= [round(p.width, 1) for p in sets[0]][0]
        True
        """
        return _balloon_line_profile_candidates(
            self, vertical, line_count, line_height, ink_thickness, air_x, air_y,
        )


def comic_break_penalty(text: str, boundary: int) -> float:
    """Prefer breaks after sentence ends; punish dangling articles.

    Break penalties in tiers: sentence
    punctuation is free, commas cheap, conjunctions moderate, articles and
    prepositions expensive, everything else in between.

    >>> comic_break_penalty('Hello. World', 6)
    0.0
    >>> comic_break_penalty('the cat', 3) > comic_break_penalty('Hello, World', 6)
    True
    """
    boundary = min(max(0, boundary), len(text))
    before = text[:boundary].rstrip()
    after = text[boundary:].lstrip()
    last = before[-1:] if before else ''
    if last in ('.', '!', '?', '…', '‼', '⁇', '⁈', '⁉'):
        return 0.0
    if last in (',', ';', ':', '—', '–'):
        return 20.0

    def _word(chunk: str, reverse: bool) -> str:
        parts = chunk.split()
        if not parts:
            return ''
        raw = parts[-1] if reverse else parts[0]
        return ''.join(ch for ch in raw if ch.isalpha()).lower()

    if _word(after, False) in {
        'and', 'but', 'or', 'so', 'because', 'although', 'while', 'then',
    }:
        return 40.0
    if _word(before, True) in {
        'a', 'an', 'the', 'to', 'of', 'for', 'in', 'on', 'at', 'with', 'from',
        'i', 'you', 'he', 'she', 'we', 'they', 'it',
    }:
        return 300.0
    return 100.0


def largest_fitting_font_size(
    minimum: float,
    maximum: float,
    layout_at: Callable[[float], T],
    fits: Callable[[T], bool],
) -> Optional[T]:
    """Search largest-first so non-monotonic balloon fits still resolve.

    A smaller font can reflow into more lines whose tapered ends fit worse,
    so a plain binary search can miss a larger fitting size. Probe down from
    the maximum in fixed steps, then binary-refine between the first fit and
    the larger miss.

    >>> 11.9 < largest_fitting_font_size(
    ...     9.0, 24.0, lambda s: s, lambda s: 10.0 <= s <= 12.0) <= 12.0
    True
    """
    probes = 8
    if maximum - minimum <= 1e-9:
        candidate = layout_at(maximum)
        return candidate if fits(candidate) else None
    step = (maximum - minimum) / float(probes)
    larger_non_fit: Optional[float] = None
    for probe in range(probes + 1):
        size = minimum if probe == probes else maximum - step * probe
        candidate = layout_at(size)
        if not fits(candidate):
            larger_non_fit = size
            continue
        if larger_non_fit is None:
            return candidate
        low, high = size, larger_non_fit
        best = candidate
        for _ in range(10):
            if high - low <= 0.01:
                break
            midpoint = (low + high) * 0.5
            candidate = layout_at(midpoint)
            if fits(candidate):
                best = candidate
                low = midpoint
            else:
                high = midpoint
        return best
    return None


def line_break_badness(line_advance: float, max_extent: float) -> float:
    """Cubic slack/overflow cost for uniform (non-balloon) breaking.

    >>> line_break_badness(5.0, 10.0) < line_break_badness(11.0, 10.0)
    True
    """
    if line_advance <= max_extent:
        return (max_extent - line_advance) ** 3
    return (line_advance - max_extent) ** 3 * LINE_BREAK_OVERFLOW_MULTIPLIER


def optimal_uniform_line_breaks(
    segments: Sequence[SegmentMeasure], max_extent: float, allow_hyphenation: bool,
) -> LineBreakResult:
    """Break against one fixed width with hyphen and linguistic penalties.

    >>> result = optimal_uniform_line_breaks(
    ...     [SegmentMeasure(10.0), SegmentMeasure(10.0)], 12.0, False)
    >>> result.breaks
    [1, 2]
    """
    total = len(segments)
    if total == 0:
        return LineBreakResult(breaks=[], profiles=[], overflowed=False, cost=0.0)
    if not np.isfinite(max_extent) or max_extent <= 0:
        return LineBreakResult(
            breaks=[total],
            profiles=[LineProfile(width=max_extent)],
            overflowed=False,
            cost=0.0,
        )
    dp = [float('inf')] * (total + 1)
    prev: List[Optional[int]] = [None] * (total + 1)
    dp[0] = 0.0
    for start in range(total):
        if not np.isfinite(dp[start]):
            continue
        advance = 0.0
        for end in range(start + 1, total + 1):
            advance += segments[end - 1].advance
            suffix = segments[end - 1].break_suffix_advance if end < total else 0.0
            line_advance = advance - segments[end - 1].trailing_advance + suffix
            hyphenated = end < total and suffix > 0.0
            if hyphenated and not allow_hyphenation:
                continue
            cost = dp[start] + line_break_badness(line_advance, max_extent)
            if hyphenated:
                cost += LINE_BREAK_HYPHEN_PENALTY
            if end < total:
                cost += segments[end - 1].break_penalty
            if cost < dp[end]:
                dp[end] = cost
                prev[end] = start
            if segments[end - 1].is_mandatory or advance > max_extent:
                break
    if not np.isfinite(dp[total]):
        widest = sum(seg.advance for seg in segments) - segments[-1].trailing_advance
        return LineBreakResult(
            breaks=[total],
            profiles=[LineProfile(width=max_extent)],
            overflowed=widest > max_extent,
            cost=float('inf'),
        )
    breaks: List[int] = []
    index = total
    while index > 0:
        breaks.append(index)
        parent = prev[index]
        if parent is None:
            return LineBreakResult(
                breaks=[total],
                profiles=[LineProfile(width=max_extent)],
                overflowed=True,
                cost=float('inf'),
            )
        index = parent
    breaks.reverse()
    widths = [max_extent]
    return LineBreakResult(
        breaks=breaks,
        profiles=[LineProfile(width=max_extent) for _ in breaks],
        overflowed=breaks_overflow(segments, breaks, widths),
        cost=dp[total],
    )


def line_breaks_with_policy(
    segments: Sequence[SegmentMeasure], max_extent: float,
    allow_hyphenation: bool, last_resort_hyphen: bool = True,
) -> LineBreakResult:
    """Try clean breaks first when hyphenation is last-resort.

    >>> line_breaks_with_policy([SegmentMeasure(10.0)], 12.0, True).overflowed
    False
    """
    if allow_hyphenation and last_resort_hyphen:
        clean = optimal_uniform_line_breaks(segments, max_extent, False)
        if not clean.overflowed:
            return clean
    return optimal_uniform_line_breaks(segments, max_extent, allow_hyphenation)


def exact_profiled_line_breaks(
    segments: Sequence[SegmentMeasure], profiles: Sequence[LineProfile], allow_hyphenation: bool,
) -> Optional[LineBreakResult]:
    """Fit segments into exact per-line widths via dynamic programming.

    Cost per line is squared relative slack plus a heavy squared overflow
    term; the total prefers fewer lines.

    >>> profiles = [LineProfile(width=12.0), LineProfile(width=12.0)]
    >>> result = exact_profiled_line_breaks(
    ...     [SegmentMeasure(10.0), SegmentMeasure(10.0)], profiles, False)
    >>> result is not None and result.breaks
    [1, 2]
    """
    total = len(segments)
    line_count = len(profiles)
    if line_count == 0 or line_count > total:
        return None
    dp = [[float('inf')] * (total + 1) for _ in range(line_count + 1)]
    prev: List[List[Optional[int]]] = [[None] * (total + 1) for _ in range(line_count + 1)]
    dp[0][0] = 0.0
    for line in range(line_count):
        remaining = line_count - line - 1
        for start in range(line, total):
            if not np.isfinite(dp[line][start]):
                continue
            advance = 0.0
            last_end = total - remaining
            for end in range(start + 1, last_end + 1):
                advance += segments[end - 1].advance
                suffix = segments[end - 1].break_suffix_advance if end < total else 0.0
                hyphenated = end < total and suffix > 0.0
                if hyphenated and not allow_hyphenation:
                    continue
                line_advance = advance - segments[end - 1].trailing_advance + suffix
                width = max(1.0, profiles[line].width)
                overflow = max(0.0, line_advance - width) / width
                slack = max(0.0, width - line_advance) / width
                cost = (
                    dp[line][start]
                    + slack * slack * 1000.0
                    + overflow * overflow * COMIC_LINE_OVERFLOW_PENALTY
                )
                if hyphenated:
                    cost += LINE_BREAK_HYPHEN_PENALTY
                if end < total:
                    cost += segments[end - 1].break_penalty
                if cost < dp[line + 1][end]:
                    dp[line + 1][end] = cost
                    prev[line + 1][end] = start
                if segments[end - 1].is_mandatory or advance > width:
                    break
    cost = dp[line_count][total]
    if not np.isfinite(cost):
        return None
    cost = cost / float(line_count) + float(line_count) * 8.0
    breaks: List[int] = []
    end = total
    for line in range(line_count, 0, -1):
        breaks.append(end)
        parent = prev[line][end]
        if parent is None:
            return None
        end = parent
    if end != 0:
        return None
    breaks.reverse()
    widths = [profile.width for profile in profiles]
    return LineBreakResult(
        breaks=breaks,
        profiles=list(profiles),
        overflowed=breaks_overflow(segments, breaks, widths),
        cost=cost,
    )


def breaks_overflow(
    segments: Sequence[SegmentMeasure], breaks: Sequence[int], widths: Sequence[float],
) -> bool:
    """Report whether any broken line exceeds its allotted width.

    >>> breaks_overflow([SegmentMeasure(20.0)], [1], [10.0])
    True
    """
    start = 0
    for line, end in enumerate(breaks):
        advance = sum(seg.advance for seg in segments[start:end])
        advance -= segments[end - 1].trailing_advance
        if end < len(segments):
            advance += segments[end - 1].break_suffix_advance
        width = widths[line] if line < len(widths) else (widths[0] if widths else 0.0)
        if advance > width + 1e-6:
            return True
        start = end
    return False


def comic_line_breaks(
    segments: Sequence[SegmentMeasure],
    balloon: ComicBalloon,
    vertical: bool,
    line_height: float,
    ink_thickness: float,
    air: Tuple[float, float],
    allow_hyphenation: bool,
    last_resort_hyphen: bool = True,
) -> LineBreakResult:
    """Choose breaks across candidate line counts inside a balloon.

    Every line count yields centered profiles (widest in the middle for an
    ellipse, contour-limited for polygons); the winner is the fewest
    non-overflowing lines, then the cheapest cost. Hyphenation follows the
    same last-resort rule as :func:`line_breaks_with_policy`.

    >>> balloon = ComicBalloon(120.0, 60.0, [], 4.0)
    >>> result = comic_line_breaks(
    ...     [SegmentMeasure(30.0), SegmentMeasure(30.0)], balloon,
    ...     False, 12.0, 10.0, (4.0, 4.0), True)
    >>> result.breaks and not result.overflowed
    True
    """
    air_x, air_y = air
    block_extent = balloon.width if vertical else balloon.height
    block_air = air_x if vertical else air_y
    usable_block = max(0.0, block_extent - block_air * 2.0)
    if ink_thickness <= usable_block and line_height > 0:
        maximum_lines = 1 + int((usable_block - ink_thickness) // line_height)
    else:
        maximum_lines = 0
    maximum_lines = min(maximum_lines, COMIC_MAX_LINES, len(segments))
    if maximum_lines <= 0:
        fallback = line_breaks_with_policy(
            segments, balloon.inline_extent(vertical, air_x, air_y),
            allow_hyphenation, last_resort_hyphen,
        )
        fallback.overflowed = True
        return fallback

    def _select(allow_hyphens: bool) -> Optional[LineBreakResult]:
        best: Optional[LineBreakResult] = None
        for line_count in range(1, maximum_lines + 1):
            for profiles in balloon.line_profile_candidates(
                vertical, line_count, line_height, ink_thickness, air_x, air_y,
            ):
                candidate = exact_profiled_line_breaks(segments, profiles, allow_hyphens)
                if candidate is None:
                    continue
                if best is None or (
                    (best.overflowed, len(best.profiles), best.cost)
                    > (candidate.overflowed, len(candidate.profiles), candidate.cost)
                ):
                    best = candidate
        return best

    if allow_hyphenation and last_resort_hyphen:
        clean = _select(False)
        if clean is not None and not clean.overflowed:
            return clean
    best = _select(allow_hyphenation)
    if best is not None:
        return best
    fallback = line_breaks_with_policy(
        segments, balloon.inline_extent(vertical, air_x, air_y),
        allow_hyphenation, last_resort_hyphen,
    )
    fallback.overflowed = True
    return fallback


def polygon_inline_spans(
    contour: Sequence[Tuple[float, float]], vertical: bool, block: float,
) -> List[Tuple[float, float]]:
    """Intersect a polygon with one scanline across the block axis.

    Vertical text scans along x (columns); horizontal text scans along y
    (rows).

    >>> polygon_inline_spans([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], False, 5.0)
    [(0.0, 10.0)]
    """
    if len(contour) < 3:
        return []
    hits: List[float] = []
    total = len(contour)
    for index in range(total):
        first = contour[index]
        second = contour[(index + 1) % total]
        if vertical:
            first_block, first_inline = first[0], first[1]
            second_block, second_inline = second[0], second[1]
        else:
            first_block, first_inline = first[1], first[0]
            second_block, second_inline = second[1], second[0]
        if (first_block <= block < second_block) or (second_block <= block < first_block):
            fraction = (block - first_block) / (second_block - first_block)
            hits.append(first_inline + (second_inline - first_inline) * fraction)
    hits.sort()
    return [(hits[i], hits[i + 1]) for i in range(0, len(hits) - 1, 2)]


def _balloon_line_profile_candidates(
    balloon: ComicBalloon,
    vertical: bool,
    line_count: int,
    line_height: float,
    ink_thickness: float,
    air_x: float,
    air_y: float,
) -> List[List[LineProfile]]:
    """Yield centered profile sets for one line count (usually one).

    The block of lines is centered in the usable block extent; each line
    samples its ink band at five heights and keeps the tightest span, then
    all lines share one clamped center so the paragraph keeps a single
    visual axis instead of zig-zagging.

    >>> balloon = ComicBalloon(120.0, 80.0, [], 4.0)
    >>> profiles = _balloon_line_profile_candidates(balloon, False, 3, 12.0, 10.0, 4.0, 4.0)[0]
    >>> [round(p.width, 1) for p in profiles][1] >= [round(p.width, 1) for p in profiles][0]
    True
    """
    if vertical:
        block_extent, inline_extent, block_air, inline_air = (
            balloon.width, balloon.height, air_x, air_y,
        )
    else:
        block_extent, inline_extent, block_air, inline_air = (
            balloon.height, balloon.width, air_y, air_x,
        )
    inline_radius = inline_extent * 0.5 - inline_air
    if inline_radius <= 0:
        return []
    block_size = ink_thickness + max(0, line_count - 1) * line_height
    block_origin = _centered_block_origin(balloon, vertical, block_extent, block_air, block_size)
    if block_origin is None:
        return []
    single = _line_profiles_at_origin(
        balloon, vertical, line_count, line_height, ink_thickness,
        block_extent, inline_extent, block_air, inline_air, block_origin,
    )
    return [single] if single else []


def _centered_block_origin(
    balloon: ComicBalloon, vertical: bool, block_extent: float, block_air: float, block_size: float,
) -> Optional[float]:
    first, last = block_air, block_extent - block_air
    # Every contour owns its full air margin; intersect the walls of all
    # valid contours (producers currently pass one cell or the mask ellipse).
    for contour in balloon.contours:
        if len(contour) < 3:
            continue
        blocks = [p[0] if vertical else p[1] for p in contour]
        first = max(first, min(blocks) + block_air)
        last = min(last, max(blocks) - block_air)
    if last - first + 1e-6 < block_size:
        return None
    return first + (last - first - block_size) * 0.5


def _inline_span(
    balloon: ComicBalloon, vertical: bool, block: float,
    block_extent: float, inline_extent: float, block_air: float, inline_air: float,
) -> Optional[Tuple[float, float]]:
    if any(len(contour) >= 3 for contour in balloon.contours):
        return _contour_inline_span(balloon, vertical, block, inline_air)
    block_radius = block_extent * 0.5 - block_air
    inline_radius = inline_extent * 0.5 - inline_air
    if block_radius <= 0:
        return None
    normalized = max(-1.0, min(1.0, (block - block_extent * 0.5) / block_radius))
    half = inline_radius * max(0.0, 1.0 - normalized * normalized) ** 0.5
    if half <= 0:
        return None
    center = inline_extent * 0.5
    return (center - half, center + half)


def _contour_inline_span(
    balloon: ComicBalloon, vertical: bool, block: float, inline_air: float,
) -> Optional[Tuple[float, float]]:
    spans: Optional[List[Tuple[float, float]]] = None
    # Intersect spans across all valid contours (producers currently pass one
    # cell or the mask ellipse).
    for contour in balloon.contours:
        if len(contour) < 3:
            continue
        narrowed = [
            (left + inline_air, right - inline_air)
            for left, right in polygon_inline_spans(contour, vertical, block)
            if right - inline_air > left + inline_air
        ]
        if spans is None:
            spans = narrowed
        else:
            merged = [
                (max(left, other_left), min(right, other_right))
                for left, right in spans
                for other_left, other_right in narrowed
                if min(right, other_right) > max(left, other_left)
            ]
            spans = merged
        if not spans:
            return None
    if not spans:
        return None
    return max(spans, key=lambda span: span[1] - span[0])


def _line_profiles_at_origin(
    balloon: ComicBalloon, vertical: bool, line_count: int, line_height: float,
    ink_thickness: float, block_extent: float, inline_extent: float,
    block_air: float, inline_air: float, block_origin: float,
) -> Optional[List[LineProfile]]:
    inline_center = inline_extent * 0.5
    # Ascenders/descenders live between baselines; sample only the ink band
    # so tapered balloon ends do not steal leading space.
    ink_before = ink_thickness * 0.8
    spans: List[Tuple[float, float, float]] = []
    for line in range(line_count):
        block_index = line_count - line - 1 if vertical else line
        baseline = block_origin + ink_before + block_index * line_height
        band_start = baseline - ink_before
        left, right = float('-inf'), float('inf')
        for sample in range(5):
            block = band_start + ink_thickness * sample / 4.0
            span = _inline_span(
                balloon, vertical, block, block_extent, inline_extent, block_air, inline_air,
            )
            if span is None:
                return None
            left = max(left, span[0])
            right = min(right, span[1])
        if right <= left:
            return None
        spans.append((left, right, baseline))
    # A phrase has one visual axis: share a single center across rows so
    # centering placement keeps every line inside its own span. Widths keep
    # their per-row taper around that shared center instead of collapsing to
    # the narrowest row.
    common_left = max(span[0] for span in spans)
    common_right = min(span[1] for span in spans)
    if common_right <= common_left:
        return None
    contour_center = sum((left + right) * 0.5 for left, right, _ in spans) / len(spans)
    shared_center = min(max(contour_center, common_left), common_right)
    profiles: List[LineProfile] = []
    for left, right, baseline in spans:
        half_width = min(shared_center - left, right - shared_center)
        if half_width <= 0.0:
            return None
        profiles.append(LineProfile(
            width=half_width * 2.0, center_offset=shared_center - inline_center,
            block_baseline=baseline,
        ))
    return profiles


#: Breaking before these leaves stranded punctuation (`SERIOUSLY` / `?`).
WIDOW_PUNCTUATION = frozenset('?!.…,;:‼⁇⁈⁉」』）］｝”’～〜')
#: Strong nudge to keep closing punctuation with the previous word.
WIDOW_BREAK_PENALTY = 1000.0
#: Strong nudge against starting a line with a dash (`Hayate-` / `kun`,
#: never `Hayate` / `-kun`).
DASH_START_BREAK_PENALTY = 1000.0

#: Dashes that offer a break *after* them (`EHH—` / `BUT`). Unlike soft
#: hyphens they render literally, so baked breaks need no dictionary.
DASH_BREAK_CHARS = frozenset('-\u2010\u2011\u2012\u2013\u2014\u2015～〜')


def split_dash_compounds(word: str) -> List[str]:
    """Split a word after interior dashes followed by more word content.

    A leading dash (bullets, `-5`, `—Bonjour`) and dash runs (`――`) stay
    whole; only `EHH—BUT`-style joints split, rejoining with no space.

    >>> split_dash_compounds('EHH—BUT')
    ['EHH—', 'BUT']
    >>> split_dash_compounds('well-known')
    ['well-', 'known']
    >>> split_dash_compounds('-5')
    ['-5']
    >>> split_dash_compounds('X――Y')
    ['X――Y']
    """
    pieces: List[str] = []
    start = 0
    for index in range(1, len(word) - 1):
        if (
            word[index] in DASH_BREAK_CHARS
            and word[index - 1] not in DASH_BREAK_CHARS
            and word[index + 1].isalnum()
        ):
            pieces.append(word[start:index + 1])
            start = index + 1
    pieces.append(word[start:])
    return [piece for piece in pieces if piece]


def join_layout_words(
    words: Sequence[str], delimiter: str, glue: Optional[Sequence[bool]] = None,
) -> str:
    """Join words, attaching glued dash pieces with no space.

    >>> join_layout_words(['EHH—', 'BUT', 'CUTE'], ' ', [False, True, False])
    'EHH—BUT CUTE'
    """
    if not words:
        return ''
    if glue is None:
        return delimiter.join(words)
    parts = [words[0]]
    for word, attached in zip(words[1:], glue[1:]):
        parts.append(word if attached else delimiter + word)
    return ''.join(parts)


def build_segments(
    words: Sequence[str],
    advances: Sequence[float],
    delimiter_width: float,
    delimiter: str,
    full_text: str,
    hyphen_width: float = 0.0,
    word_break: bool = False,
    glue: Optional[Sequence[bool]] = None,
) -> List[SegmentMeasure]:
    """Convert laid-out words into DP segments with linguistic penalties.

    Each word carries its advance plus the following delimiter; explicit
    newlines in ``words`` become mandatory breaks. ``glue[i]`` marks a dash
    piece attached to its predecessor with no space, so it carries no
    trailing delimiter. Hyphen suffixes apply to
    long words only, so short words never split mid-line. A break before
    closing punctuation costs extra so `?` never dangles on its own line
    when joining it still fits.

    >>> build_segments(['hi', 'there'], [10.0, 20.0], 4.0, ' ', 'hi there')[0].advance
    14.0
    >>> [round(segment.break_penalty, 1) for segment in
    ...  build_segments(['Wow', '?'], [20.0, 5.0], 4.0, ' ', 'Wow ?')]
    [1100.0, 0.0]
    >>> [round(segment.advance, 1) for segment in
    ...  build_segments(['EHH—', 'BUT'], [20.0, 15.0], 4.0, ' ', 'EHH—BUT',
    ...                 glue=[False, True])]
    [20.0, 19.0]
    """
    clean_words: List[str] = []
    clean_advances: List[float] = []
    clean_glue: List[bool] = []
    mandatory: List[bool] = []
    cursor = 0
    for position, (word, advance) in enumerate(zip(words, advances)):
        if word in ('\n', '\r\n'):
            if mandatory:
                mandatory[-1] = True
            cursor += len(word)
            continue
        clean_words.append(word)
        clean_advances.append(advance)
        # ``glue`` shares the original indexing (markers carry False).
        clean_glue.append(bool(glue[position]) if glue is not None else False)
        mandatory.append(False)
        cursor += len(word) + (len(delimiter) if delimiter else 0)
    segments: List[SegmentMeasure] = []
    cursor = 0
    for index, (word, advance) in enumerate(zip(clean_words, clean_advances)):
        attached_next = index + 1 < len(clean_words) and clean_glue[index + 1]
        trailing = 0.0 if attached_next else (delimiter_width if delimiter else 0.0)
        suffix = 0.0
        if hyphen_width > 0 and word_break and len(word.strip()) >= 5:
            suffix = hyphen_width
        penalty = comic_break_penalty(full_text, cursor + len(word))
        if (
            index + 1 < len(clean_words)
            and clean_words[index + 1][:1] in WIDOW_PUNCTUATION
        ):
            penalty += WIDOW_BREAK_PENALTY
        if (
            index + 1 < len(clean_words)
            and clean_words[index + 1][:1] in DASH_BREAK_CHARS
        ):
            penalty += DASH_START_BREAK_PENALTY
        segments.append(SegmentMeasure(
            advance=float(advance) + trailing,
            trailing_advance=trailing,
            break_suffix_advance=suffix,
            break_penalty=penalty,
            is_mandatory=mandatory[index],
        ))
        cursor += len(word) + (0 if attached_next else (len(delimiter) if delimiter else 0))
    return segments


def balloon_from_mask(
    mask: np.ndarray, minimum_air: float = 4.0,
) -> ComicBalloon:
    """Build an ellipse-fallback balloon from a binary interior mask.

    >>> balloon_from_mask(np.ones((10, 20), dtype=np.uint8) * 255).width
    20.0
    """
    height, width = mask.shape[:2]
    return ComicBalloon(float(width), float(height), [], max(0.0, float(minimum_air)))


def balloon_from_polygon(
    polygon: Sequence[Sequence[float]],
    origin: Tuple[float, float],
    width: float,
    height: float,
    minimum_air: float = 4.0,
) -> ComicBalloon:
    """Shift a page-space contour into layout-local balloon coordinates.

    Malformed outlines fall back to the plain ellipse model instead of
    aborting the layout, keeping optional geometry passive.

    >>> balloon = balloon_from_polygon([[0, 0], [10, 0], [10, 10], [0, 10]], (0, 0), 10, 10)
    >>> len(balloon.contours)
    1
    """
    try:
        points = [(float(p[0]) - origin[0], float(p[1]) - origin[1]) for p in polygon]
        if len(points) < 3 or width <= 0 or height <= 0:
            raise ValueError('degenerate contour')
        if any(not np.isfinite(x) or not np.isfinite(y) for x, y in points):
            raise ValueError('non-finite contour')
    except (TypeError, ValueError, IndexError):
        return ComicBalloon(float(width), float(height), [], max(0.0, float(minimum_air)))
    return ComicBalloon(float(width), float(height), [points], max(0.0, float(minimum_air)))
