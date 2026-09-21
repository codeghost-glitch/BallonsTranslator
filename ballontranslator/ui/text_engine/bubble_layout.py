"""Detected-bubble fitting through the existing editable text item layout."""

from functools import lru_cache
import re
import cv2
import numpy as np
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QFontMetricsF, QTextCursor, QTextDocument, QTextLayout, QTextOption

from ballontranslator.utils.bubble import bubble_inner_center, bubble_inner_rect
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import TextAlignment, px2pt
from ballontranslator.utils.autolayout import (
    ComicBalloon,
    build_segments,
    comic_line_breaks,
    join_layout_words,
    largest_fitting_font_size,
    polygon_inline_spans,
)
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.text_layout import seg_text_paragraphs
from ballontranslator.utils.text_processing import is_cjk
from .item import TextBlkItem
from .annotations import AnnotationProperty
from .rendering.indexing import _utf16_boundaries

if TYPE_CHECKING:
    from pyphen import Pyphen


@lru_cache(maxsize=16)
def _hyphenator(language: str) -> Optional['Pyphen']:
    try:
        import pyphen
    except ImportError:
        LOGGER.warning('Automatic hyphenation needs the optional Pyphen package; keeping words unchanged.')
        return None
    codes = {
        'English': 'en_US', 'German': 'de_DE', 'French': 'fr_FR',
        'Spanish': 'es_ES', 'Italian': 'it_IT', 'Portuguese': 'pt_PT',
        'Brazilian Portuguese': 'pt_BR', 'Russian': 'ru_RU', 'Ukrainian': 'uk_UA',
        'Polish': 'pl_PL', 'Dutch': 'nl_NL', 'Turkish': 'tr_TR',
        'Indonesian': 'id_ID', 'Swedish': 'sv_SE', 'Finnish': 'fi_FI',
        'čeština': 'cs_CZ', 'Nederlands': 'nl_NL', 'Français': 'fr_FR',
        'Deutsch': 'de_DE', 'magyar nyelv': 'hu_HU', 'Italiano': 'it_IT',
        'Polski': 'pl_PL', 'Português': 'pt_PT', 'limba română': 'ro_RO',
        'русский язык': 'ru_RU', 'Español': 'es_ES', 'Türk dili': 'tr_TR',
        'украї́нська мо́ва': 'uk_UA', 'Hindi': 'hi_IN', 'Malayalam': 'ml_IN',
        'Tamil': 'ta_IN', 'Thai': 'th_TH',
    }
    code = pyphen.language_fallback(codes.get(language, language))
    if code is None:
        LOGGER.warning('No hyphenation dictionary for %s; keeping words unchanged.', language)
        return None
    return pyphen.Pyphen(lang=code)


def hyphenate_document(document: QTextDocument, language: str) -> None:
    """Insert discretionary breaks without changing letters or inline formatting.

    >>> callable(hyphenate_document)
    True
    """
    dictionary = _hyphenator(language)
    if dictionary is None:
        return
    text = document.toPlainText()
    boundaries = _utf16_boundaries(text)
    positions = []
    for match in re.finditer(r'[^\W\d_]+(?:\u00ad[^\W\d_]+)*', text, re.UNICODE):
        word = match.group()
        if '\u00ad' in word or len(word) < 6:
            continue
        for offset in dictionary.positions(word):
            if getattr(offset, 'data', None) is not None:
                continue
            positions.append(boundaries[match.start() + int(offset)])
    cursor = QTextCursor(document)
    cursor.beginEditBlock()
    try:
        for position in reversed(positions):
            cursor.setPosition(position)
            char_format = cursor.charFormat()
            if char_format.hasProperty(AnnotationProperty.RUBY_ID) or char_format.hasProperty(AnnotationProperty.TEXT_COMBINE_ID):
                continue
            cursor.insertText('\u00ad')
    finally:
        cursor.endEditBlock()


def _current_font(item: TextBlkItem):
    """Return the live font of the first text fragment, if any.

    >>> callable(_current_font)
    True
    """
    document = item.document()
    block = document.firstBlock()
    while block.isValid():
        fragment = block.begin()
        while not fragment.atEnd():
            current = fragment.fragment()
            if current.length() > 0 and current.text().strip():
                return current.charFormat().font()
            fragment += 1
        block = block.next()
    return item.font()


def _longest_word_width(item: TextBlkItem, *, cjk: bool = False) -> float:
    """Measure the widest unsplittable run at the live font size.

    Attached punctuation stays with its token so it cannot wrap onto its own row.
    CJK measures the widest single character instead: spaceless scripts wrap
    per character, so the whole string is never one unbreakable word.
    An empty document measures zero and never constrains the fit.

    >>> callable(_longest_word_width)
    True
    """
    text = item.toPlainText()
    metrics = QFontMetricsF(_current_font(item))
    if cjk:
        chars = [char for char in text if not char.isspace()]
        if not chars:
            return 0.0
        return max(metrics.horizontalAdvance(char) for char in chars)
    tokens = text.split()
    if not tokens:
        return 0.0
    return max(metrics.horizontalAdvance(token) for token in tokens)


def _split_oversized_tokens(
    units: List[str], advances: List[float], glue: List[bool],
    extent: float, measure: Callable[[str], float], language: str,
) -> Tuple[List[str], List[float], List[bool]]:
    """Split tokens wider than the inline extent at dictionary hyphen points.

    Returns new units/advances/glue where every token either fits the extent
    or keeps its current shape (no dictionary, or no usable break point).
    Split pieces carry a literal hyphen on every non-final piece and glue to
    their continuation with no space, so the DP can place ``PROHIB-`` /
    ``ITED`` across lines without Qt splitting a word mid-letter. This is
    the typesetter move: one long word no longer pins the whole bubble to a
    tiny size, and the hyphen shows only where the line actually breaks.

    >>> callable(_split_oversized_tokens)
    True
    """
    out_units: List[str] = []
    out_advances: List[float] = []
    out_glue: List[bool] = []
    for unit, advance, flag in zip(units, advances, glue):
        if unit in ('\n',) or advance <= extent:
            out_units.append(unit)
            out_advances.append(advance)
            out_glue.append(flag)
            continue
        # Editorial hyphenation rule: only split words the dictionary can break
        # in tall balloons when the intact fit is small. Very short words
        # (under 5 letters) never split; anything longer gets a dictionary
        # point, so `teacher'.` cannot pin a tall balloon to 8pt when
        # `teach-` / `er'.` at 14pt reads fine (page 007 blk4/blk5).
        letters = sum(char.isalpha() for char in unit)
        if letters < 5:
            out_units.append(unit)
            out_advances.append(advance)
            out_glue.append(flag)
            continue
        pieces = _hyphen_pieces(unit, extent, measure, language)
        if len(pieces) == 1:
            out_units.append(unit)
            out_advances.append(advance)
            out_glue.append(flag)
            continue
        for index, piece in enumerate(pieces):
            out_units.append(piece)
            out_advances.append(float(measure(piece)))
            # The original unit started a new word; continuations attach
            # with no space so a same-line join reads `PROHIB-ITED`.
            out_glue.append(flag if index == 0 else True)
    return out_units, out_advances, out_glue


def _hyphen_pieces(
    word: str, extent: float, measure: Callable[[str], float], language: str,
) -> List[str]:
    """Break ``word`` at dictionary hyphen points into pieces fitting width.

    Greedy from the left: take the longest prefix that fits ``extent`` (the
    hyphen glyph included) and ends at a dictionary break, append ``-``,
    repeat on the rest. Hyphenation points come from the letter core, so a
    trailing tilde or ellipsis (``teacher~``, ``confiscated.``) cannot
    strangle the dictionary; the suffix rides on the final piece. No
    dictionary or no usable point keeps the whole word intact (one element).

    >>> callable(_hyphen_pieces)
    True
    """
    dictionary = _hyphenator(language)
    core = word
    suffix = ''
    if word:
        last_letter = 0
        for index, char in enumerate(word):
            if char.isalpha():
                last_letter = index + 1
        core, suffix = word[:last_letter], word[last_letter:]
    pieces: List[str] = []
    rest = core
    guard = 0
    while float(measure(rest)) > extent:
        if dictionary is None:
            return [word]
        guard += 1
        if guard > 16:
            return [word]
        cut: Optional[int] = None
        for position in dictionary.positions(rest):
            # The hyphen glyph rides on the piece, so the prefix plus '-'
            # must both fit; a 1px-over piece would overflow every line.
            if float(measure(rest[:position] + '-')) > extent:
                break
            if 0 < position < len(rest):
                cut = position
        if cut is None:
            return [word]
        pieces.append(rest[:cut] + '-')
        rest = rest[cut:]
    pieces.append(rest + suffix)
    return pieces


def _local_wall_profile(
    polygon: List[List[float]], rect: Tuple[float, float, float, float],
    vertical: bool,
) -> Optional[List[List[float]]]:
    """Sample the outline into a two-sided wall loop in ``rect``'s local frame.

    The DP only needs each scanline's inline span, so sampling the polygon
    every few pixels and rebuilding a closed left-wall/right-wall loop gives
    it the balloon's true taper for a fraction of the cost of scanning the
    raw outline (a 350-point resample would multiply every probe). A sparse
    outline instead lets the DP under-measure a row whose wall runs between
    two distant vertices: page 007 blk5's 49px interior rectangle reported
    ~36px rows where the real wall holds 76px, pinning ``confiscated.`` to
    8.6pt when the outline itself admits ~14.6pt. Rows with several spans
    (tails, notches) bridge to the outer walls; the candidate corner checks
    still keep adopted boxes honest.

    >>> callable(_local_wall_profile)
    True
    """
    points = [(float(point[0]), float(point[1])) for point in polygon]
    if len(points) < 3:
        return None
    blocks = [point[0] if vertical else point[1] for point in points]
    first, last = min(blocks), max(blocks)
    if last - first < 2.0:
        return None
    step = max(1.0, (last - first) / 32.0)
    near_wall: List[Tuple[float, float]] = []
    far_wall: List[Tuple[float, float]] = []
    block = first
    while block <= last + 1e-6:
        spans = polygon_inline_spans(points, vertical, float(block))
        if spans:
            near_wall.append((min(span[0] for span in spans), block))
            far_wall.append((max(span[1] for span in spans), block))
        block += step
    if len(near_wall) < 2:
        return None
    # (inline, block) pairs walking the near wall down and the far wall back.
    loop = near_wall + far_wall[::-1]
    origin_x, origin_y = float(rect[0]), float(rect[1])
    if vertical:
        return [[block - origin_x, inline - origin_y] for inline, block in loop]
    return [[inline - origin_x, block - origin_y] for inline, block in loop]


def _inscribed_outline_box(
    polygon: List[List[float]], minimum_width: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Widest box inside the outline that keeps a usable inline run everywhere.

    Sampling the outline's row spans picks the box whose every row holds a
    run at least ``minimum_width`` wide, maximizing area. A balloon's bbox
    rectangle instead lets a fitted line reach the single widest scanline and
    drag its corners onto the wall: page 007 blk5 has an 82px bbox but its
    corners sit on rows a few px wide, so the box must hug the 76px middle
    rows instead. Returns ``(x, y, width, height)`` or ``None``.

    >>> _inscribed_outline_box([[0, 0], [10, 0], [10, 10], [0, 10]], 6.0)[2:]
    (10.0, 10.0)
    """
    points = [(float(point[0]), float(point[1])) for point in polygon]
    if len(points) < 3:
        return None
    ys = [point[1] for point in points]
    first_y, last_y = min(ys), max(ys)
    if last_y - first_y < 2.0:
        return None
    step = max(1.0, (last_y - first_y) / 96.0)
    rows: List[Tuple[float, float, float]] = []
    sample_y = first_y
    while sample_y <= last_y + 1e-6:
        spans = polygon_inline_spans(points, False, float(sample_y))
        if spans:
            left, right = max(spans, key=lambda span: span[1] - span[0])
            rows.append((sample_y, left, right))
        sample_y += step
    if not rows:
        return None
    best: Optional[Tuple[float, float, float, float, float]] = None
    for start in range(len(rows)):
        left, right = rows[start][1], rows[start][2]
        if right - left < minimum_width:
            continue
        for end in range(start, len(rows)):
            left = max(left, rows[end][1])
            right = min(right, rows[end][2])
            if right - left < minimum_width:
                break
            height = rows[end][0] - rows[start][0] + step
            area = (right - left) * height
            if best is None or area > best[0]:
                best = (area, left, rows[start][0], right - left, height)
    if best is None:
        return None
    return (float(best[1]), float(best[2]), float(best[3]), float(best[4]))


def _balloon_prebreak(
    base_text: str, language: str, font_metrics: QFontMetricsF,
    balloon: ComicBalloon, line_height: float, ink: float, *, vertical: bool,
    word_break: bool = False, contour: Optional[List[List[float]]] = None,
) -> Tuple[str, bool]:
    """Break text with the balloon DP against the interior width.

    Breaks usually stay on whole-word boundaries; with ``word_break``,
    tokens wider than the inline extent are first split at dictionary
    hyphen points (``PROHIB-`` / ``ITED``), so one oversized word no longer
    pins every other line to a tiny size. The split renders as a literal
    hyphen in the broken plain text, like the mask path does.

    ``balloon`` is already in balloon-local coordinates (origin at the
    interior rect). When ``contour`` is given it is translated into that same
    frame so the DP sees the balloon's real per-row widths: a uniform
    rectangle under-measures a tapering outline's middle rows and pins tall
    dialogue to an unreadable size (page 007 blk5: 8.6pt rect vs 16pt
    outline).

    >>> callable(_balloon_prebreak)
    True
    """
    units, advances, delimiter, glue = seg_text_paragraphs(
        base_text, language, font_metrics.horizontalAdvance,
    )
    extent = balloon.inline_extent(vertical, 0.0, 0.0)
    if word_break:
        units, advances, glue = _split_oversized_tokens(
            units, advances, glue, extent, font_metrics.horizontalAdvance, language,
        )
    words = [unit for unit in units if unit != '\n']
    if not words:
        return base_text, False
    word_glue = [flag for unit, flag in zip(units, glue) if unit != '\n']
    delimiter_width = font_metrics.horizontalAdvance(delimiter) if delimiter else 0.0
    segments = build_segments(
        units, advances, delimiter_width, delimiter,
        join_layout_words(words, delimiter, word_glue), glue=glue,
    )
    if not segments:
        return base_text, False
    if contour:
        # Already rebased into this interior rectangle's local frame by
        # _local_wall_profile, matching the balloon's own (0, 0) origin.
        walls = [[(float(x), float(y)) for x, y in contour]]
    else:
        walls = [[(0.0, 0.0), (balloon.width, 0.0),
                  (balloon.width, balloon.height), (0.0, balloon.height)]]
    rect_balloon = ComicBalloon(
        balloon.width, balloon.height, walls, balloon.minimum_air,
    )
    result = comic_line_breaks(
        segments, rect_balloon, vertical, max(1.0, line_height), max(1.0, ink),
        (0.0, 0.0), allow_hyphenation=False,
    )
    breaks = result.breaks or [len(segments)]
    lines, start = [], 0
    for end in breaks:
        lines.append(join_layout_words(
            words[start:end], delimiter, word_glue[start:end],
        ))
        start = end
    return '\n'.join(lines), result.overflowed


def _best_font_for_target(
    item: TextBlkItem, target: QRectF, *, language: str, vertical: bool = False,
    preferred_max: Optional[float] = None, allow_split: bool = False,
    polygon: Optional[List[List[float]]] = None, slack: float = 0.0,
) -> Tuple[Optional[Tuple[float, str]], str]:
    """Search largest-first for the biggest fitting size and its breaks.

    Probes down from the maximum (balloon fits are non-monotonic: a smaller
    font can reflow into more lines with worse tapered ends) then refines,
    mirroring the largest-first search in
    :mod:`ballontranslator.utils.autolayout`. Each probe re-breaks
    the text with the balloon DP at that size and reserves effect padding on
    every side; the DP keeps words intact, so narrow bubbles shrink instead
    of breaking `COME` into `COM`/`E`. For Latin text, a second search lets
    the DP split one oversized word (``PROHIBITED`` in a narrow bubble) and
    is adopted only when it wins a visible point over the intact fit — the
    typesetter move that keeps a long word from pinning the whole bubble to
    an unreadable size. ``preferred_max`` caps growth at the
    source size plus a little (scanlation practice matches the raw instead
    of filling bubbles); shrinking below it is unaffected. Returns the size
    plus its explicitly broken text (or ``None`` when even minimum size
    overflows) alongside the DP-broken text at minimum size, so the fallback
    still carries balanced breaks instead of Qt's greedy wrap.

    >>> callable(_best_font_for_target)
    True
    """
    upper = max(4.0, px2pt(max(target.width(), target.height())))
    lower = 4.0
    if preferred_max is not None and preferred_max > lower:
        upper = min(upper, preferred_max)
    base_text = item.toPlainText()
    probe_needed: Optional[tuple[float, float, float, float]] = None
    # The outline in this rectangle's local frame, so the DP measures the
    # balloon's real per-row widths instead of a uniform box.
    contour = (
        _local_wall_profile(polygon, (target.x(), target.y(), target.width(), target.height()), vertical)
        if polygon else None
    )

    def layout_at(size: float, word_break: bool = False) -> Tuple[str, float, float, bool]:
        item.setFontSize(size)
        metrics = QFontMetricsF(_current_font(item))
        pad = max(0.0, float(item.padding())) * 2.0
        spacing = float(getattr(item.fontformat, 'line_spacing', 1.0) or 1.0)
        line_height = max(1.0, metrics.height() * spacing)
        ink = max(1.0, metrics.ascent() + metrics.descent())
        balloon = ComicBalloon(
            max(1.0, target.width() - pad + slack),
            max(1.0, target.height() - pad + slack), [], 0.0,
        )
        broken, overflowed = _balloon_prebreak(
            base_text, language, metrics, balloon, line_height, ink,
            vertical=vertical, word_break=word_break, contour=contour,
        )
        item.setPlainText(broken)
        # The fit holds an outer undo edit block, which defers Qt's layout
        # rebuild past setRect; settle synchronously instead (this is what
        # documentChanged would run outside an edit block).
        item.layout.reLayoutEverything()
        item.setRect(target, padding=False)
        needed_height, needed_width = item.layout.minSize()
        return broken, needed_width, needed_height, overflowed

    def fits(candidate: Tuple[str, float, float, bool]) -> bool:
        broken, needed_width, needed_height, _overflowed = candidate
        pad = max(0.0, float(item.padding())) * 2.0
        text_pad = max(0.0, float(getattr(item.layout, 'text_padding', 0.0) or 0.0))
        # 2px tolerance covers normal glyph overhang; the erosion margin to
        # the bubble border (typically 8px+) absorbs it. The width term keeps
        # vertical layout's column stack inside the interior (a column's
        # per-line height check cannot see the total width); for horizontal
        # text it is always satisfied once the DP lines fit. The DP lines
        # themselves are re-measured explicitly: minimum size alone is blind
        # to a line Qt had to tear mid-word (`PROHIBITED` in a narrow bubble
        # reads as eight stacked `H`s), so any DP line wider than the
        # interior rejects the size outright.
        fits_size = (
            needed_width - text_pad + pad <= target.width() + 2.0 + slack
            and needed_height + pad <= target.height() + 2.0
        )
        line_width = (target.height() if vertical else target.width()) - pad + slack
        # Ask Qt itself whether each DP line survives at this width: metrics
        # and shaping can disagree by a few px, and an over-wide tail like
        # `Kunieda...` would tear at the last period (`KUNIEDA..` / `.`) on
        # a line `QFontMetricsF` vouched for.
        font = _current_font(item)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAnywhere)
        lines_fit = True
        for line in broken.split('\n'):
            layout = QTextLayout(line, font)
            layout.setTextOption(option)
            layout.beginLayout()
            first = layout.createLine()
            if first.isValid():
                first.setLineWidth(max(1.0, line_width))
            second = layout.createLine()
            layout.endLayout()
            if second.isValid():
                lines_fit = False
                break
        return fits_size and lines_fit

    def best_at(word_break: bool) -> Optional[Tuple[float, str]]:
        sized = largest_fitting_font_size(
            lower, upper,
            lambda size: (size, layout_at(size, word_break=word_break)),
            lambda candidate: fits(candidate[1]),
        )
        return None if sized is None else (sized[0], sized[1][0])

    first = layout_at(lower, word_break=allow_split)
    probe_needed = (first[1], first[2], max(0.0, float(item.padding())) * 2.0,
                    _longest_word_width(item, cjk=is_cjk(language)))
    best: Optional[Tuple[float, str]] = None
    min_broken = first[0]
    if fits(first):
        best = best_at(allow_split)
    if best is None:
        if probe_needed is not None:
            needed_width, needed_height, pad, longest = probe_needed
            LOGGER.warning(
                'Bubble fit rejected at 4pt: target=%.1fx%.1f needed=%.1fx%.1f '
                'pad=%.1f longest=%.1f.',
                target.width(), target.height(), needed_width, needed_height,
                pad, longest,
            )
        return None, min_broken
    return best, min_broken


def fit_text_to_bubble(
    item: TextBlkItem, text: Optional[str], *, language: str,
    target_rect: Optional[tuple] = None,
) -> bool:
    """Fit text centered within the dynamic bubble interior.

    The interior is the maximal rectangle inside the detected shape with a
    dynamic inset (relative to the bubble, capped in absolute pixels, vertical
    emphasis like the guide) while tails and concavities are avoided. Callers
    fitting several blocks that share one outline pass ``target_rect`` with the
    block's pre-split slice so siblings stay inside the same bubble without
    overlapping. Fitted text uses Center
    alignment, keeps the interior width for stable wrapping, and centers
    vertically so top/bottom and left/right margins match. Line breaks come
    from the balloon DP (balanced lines, linguistic penalties) over the full
    interior width,
    and the search reserves
    effect padding for stroked/bold ink and shrinks narrow bubbles until the
    longest word fits instead of splitting `COME` into `COM`/`E`. Latin text
    gets a typesetter rescue: when the intact fit stays below dialogue size
    (or fails entirely), long words (5+ letters, dictionary-gated) split at hyphen
    points with a literal hyphen, so one oversized word no longer pins the
    whole bubble to an unreadable size. Splits are adopted only when they
    buy a real gain; otherwise words stay intact and the block overflows
    centered at minimum size ("Hyphenate text" remains a manual opt-in).
    Closing punctuation never dangles alone when joining it still fits.
    Vertical text
    breaks into columns with the same DP transposed. When even minimum size
    overflows, the text stays centered at minimum size on the bubble
    instead of falling back to a mask layout outside of it, widening past
    the interior only as far as the longest word needs to stay intact.

    >>> callable(fit_text_to_bubble)
    True
    """
    if item.blk.bubble_polygon is None and target_rect is None:
        LOGGER.info('Bubble fit skipped: no detected bubble polygon.')
        return False
    if item.rotation() != 0:
        LOGGER.info('Bubble fit skipped: rotated item keeps its own layout.')
        return False
    if not item._text_transform_is_neutral():
        LOGGER.info('Bubble fit skipped: transformed item keeps its own layout.')
        return False
    polygon = item.blk.bubble_polygon
    rect = tuple(target_rect) if target_rect is not None else (
        bubble_inner_rect(polygon) if polygon is not None else None
    )
    if rect is None or min(rect[2:]) < 4:
        return False
    # A diagonal or heavily concave balloon collapses the axis-aligned
    # interior to a sliver far smaller than the region the source text
    # occupied (page 007: 25x72 against 69x208), and fitting into it only
    # shrinks toward the 4pt floor. Hand the block back to the mask layout,
    # which hugs diagonal walls per scanline, instead of "successfully"
    # fitting unreadable text into the sliver. Both axes must collapse:
    # a thin wide interior renders short horizontal dialogue just fine.
    lines = item.blk.lines_array()
    if len(lines):
        source_w = float(lines[..., 0].max() - lines[..., 0].min())
        source_h = float(lines[..., 1].max() - lines[..., 1].min())
        if rect[2] < 0.6 * source_w and rect[3] < 0.6 * source_h:
            LOGGER.info(
                'Bubble fit skipped: interior %.0fx%.0f collapsed against '
                'source text %.0fx%.0f; using the mask layout.',
                rect[2], rect[3], source_w, source_h,
            )
            return False
    vertical = bool(item.blk.vertical)
    original_alignment = int(item.fontformat.alignment)
    cursor = QTextCursor(item.document())
    cursor.beginEditBlock()
    try:
        if text is not None:
            item.setPlainText(text)
        if not item.toPlainText().strip():
            return False
        # The fit never hyphenates on its own and dehyphenates like real
        # typesetting software on reflow: a hyphen at a line end was a baked
        # artifact of a previous fit (`Teach-` / `er~`), so remove it and
        # rejoin the word before searching — otherwise every re-fit keeps
        # the old hyphens forever. Soft hyphens from the manual "Hyphenate
        # text" action are stripped too.
        clean_base = re.sub(r'-\n', '', item.toPlainText())
        clean_base = clean_base.replace('\u00ad', '')
        item.setPlainText(clean_base)
        # Centered lines read as one centered block inside the bubble.
        if original_alignment != int(TextAlignment.Center):
            item.setAlignment(int(TextAlignment.Center), repaint_background=False)
        target = QRectF(*rect)
        # Grow up to twice the detected source size so short dialogue fills
        # more of a roomy balloon (users read the old match-the-raw cap as
        # "too small"); shrinking stays untouched and the interior rectangle
        # remains the real ceiling. Blocks without a detection keep the full
        # search range, as does an explicit global-size override.
        detected = max(0.0, float(getattr(item.blk, '_detected_font_size', -1) or -1))
        if detected > 0 and pcfg.let_fntsize_flag == 1:
            detected = 0.0
        preferred_max = px2pt(detected) * 2.0 if detected > 0 else None
        def search_all(allow_split: bool):
            """Run every fitting candidate (interior, outline, shaped bands,
            slab) with words intact or with the dictionary-split DP active."""
            def _ink_box_inside(
                size: float, broken: str, rect: QRectF, tolerance: float = -3.0,
            ) -> bool:
                """The laid-out ink box, moved onto ``rect``, stays inside the
                outline. Corner-only checks pass while a box's edges poke
                through tapered walls (page 007 blk1/blk8: a wide box whose
                corners sit on wide rows but whose middle rows are narrower
                than the text), so sample the full perimeter."""
                item.setFontSize(size)
                item.setPlainText(broken)
                item.layout.reLayoutEverything()
                item.setRect(rect, padding=False)
                needed_height, _ = item.layout.minSize()
                pad = max(0.0, float(item.padding())) * 2.0
                height = min(max(needed_height, 1.0) + pad, rect.height())
                box = QRectF(
                    rect.x(), rect.center().y() - height / 2.0,
                    rect.width(), height,
                )
                shape_contour = np.asarray(polygon, dtype=np.float32)
                points = [
                    (box.left() + box.width() * fraction, box.top())
                    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
                ] + [
                    (box.left() + box.width() * fraction, box.bottom())
                    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
                ] + [
                    (box.left(), box.top() + box.height() * fraction)
                    for fraction in (0.25, 0.5, 0.75)
                ] + [
                    (box.right(), box.top() + box.height() * fraction)
                    for fraction in (0.25, 0.5, 0.75)
                ]
                return all(
                    cv2.pointPolygonTest(
                        shape_contour, (float(px), float(py)), True,
                    ) >= tolerance
                    for px, py in points
                )

            target = QRectF(*rect)
            best, min_broken = _best_font_for_target(
                item, target, language=language, vertical=vertical,
                preferred_max=preferred_max, allow_split=allow_split,
            )
            if best is not None and target_rect is None and polygon is not None and not vertical:
                # The maximal interior rectangle is narrower than the balloon's
                # own span, so tapering dialogue (blk5: 49px interior vs 82px
                # outline) fits unreadably small inside a sliver even though
                # the outline itself holds the same text two sizes larger.
                # Measure against the balloon's own box with the contour walls,
                # keeping the wider envelope only while the ink still sits in
                # the outline; the mask path has always done this.
                points = np.asarray(polygon, dtype=np.float32)
                bbox_rect = QRectF(
                    float(points[:, 0].min()), float(points[:, 1].min()),
                    float(points[:, 0].max() - points[:, 0].min()),
                    float(points[:, 1].max() - points[:, 1].min()),
                )
                # A tapering balloon's maximal interior is far narrower than
                # the balloon itself (page 007 blk5: 49px interior vs 82px
                # outline), so its sliver fit pins dialogue to an unreadable
                # size while the outline's wide middle rows hold the same text
                # two sizes larger. Re-fit against the balloon's own box with
                # the erosion margin as slack, and adopt only when the
                # interior fit is unreadable and the outline fit wins a real
                # gain. No corner gate: the mask layout this mirrors allows
                # the same taper overflow.
                if (
                    best[0] < 11.0
                    and bbox_rect.width() > target.width() * 1.4
                    and bbox_rect.height() > 4
                ):
                    item.setPlainText(clean_base)
                    outline_probe, _ = _best_font_for_target(
                        item, bbox_rect, language=language, vertical=vertical,
                        preferred_max=preferred_max, allow_split=allow_split,
                        slack=min(8.0, (bbox_rect.width() - target.width()) / 2.0),
                    )
                    if outline_probe is not None and outline_probe[0] > best[0] * 1.15:
                        # The box must hold the widest fitted line or Qt
                        # re-wraps it; widen symmetrically past the outline
                        # only as far as the line needs (the mask layout this
                        # mirrors overflows the same way).
                        content_pad = max(0.0, float(item.padding())) * 2.0
                        item.setFontSize(outline_probe[0])
                        metrics = QFontMetricsF(_current_font(item))
                        box_w = max(
                            bbox_rect.width(),
                            max(
                                metrics.horizontalAdvance(line)
                                for line in outline_probe[1].split('\n')
                            ) + content_pad,
                        )
                        item.setPlainText(outline_probe[1])
                        item.layout.reLayoutEverything()
                        item.setRect(
                            QRectF(
                                bbox_rect.center().x() - box_w / 2.0,
                                bbox_rect.y(), box_w, bbox_rect.height(),
                            ),
                            padding=False,
                        )
                        needed_height, _ = item.layout.minSize()
                        box_h = min(max(needed_height, 1.0) + content_pad, bbox_rect.height())
                        guide = bubble_inner_center(polygon)
                        if guide is not None:
                            probe_x, probe_y = float(guide[0]), float(guide[1])
                        else:
                            probe_x, probe_y = bbox_rect.center().x(), bbox_rect.center().y()
                        # The outline path mirrors the mask layout, which lets
                        # text ride a tapering balloon's wide rows. Adopt only
                        # when the laid-out ink stays within the balloon's
                        # taper, bounded by the erosion margin the interior
                        # rectangle threw away (capped at 12px): the box may
                        # poke past the detected ink outline at the narrow end
                        # rows, still inside the balloon's real wall. Check at
                        # the guide — the tail slide's target — not the
                        # rectangle's own center.
                        if _ink_box_inside(
                            outline_probe[0], outline_probe[1],
                            QRectF(
                                probe_x - box_w / 2.0, probe_y - box_h / 2.0,
                                box_w, box_h,
                            ),
                            tolerance=-min(
                                12.0, (bbox_rect.width() - target.width()) / 2.0
                            ),
                        ):
                            best, target = outline_probe, QRectF(
                                probe_x - box_w / 2.0, probe_y - box_h / 2.0,
                                box_w, box_h,
                            )
            if best is not None and target_rect is None and polygon is not None and not vertical:
                # Maximum-area interiors favor tall rectangles. Short dialogue
                # can fit at a larger size in a wider, shallower interior.
                item.setFontSize(best[0])
                metrics = QFontMetricsF(_current_font(item))
                lines = best[1].split('\n')
                width = max(metrics.horizontalAdvance(line) for line in lines)
                height = metrics.height() * float(item.fontformat.line_spacing or 1.0) * len(lines)
                # The guide's generous inset is not a text-size ceiling. Keep a
                # 3% contour margin here; Qt separately reserves effect padding.
                # Candidate 1: a rectangle shaped like the fitted lines. Short
                # dialogue in a wide balloon gains nothing from tall interiors.
                # Candidate 2: the outline's own aspect — ellipse-like balloons
                # hold more text in that shape. Keep the best-scoring one.
                def _shaped_best(rect: tuple):
                    if rect is None:
                        return None
                    item.setPlainText(clean_base)
                    probe, _ = _best_font_for_target(
                        item, QRectF(*rect), language=language,
                        preferred_max=preferred_max, polygon=polygon,
                    )
                    return probe

                shaped_rect = bubble_inner_rect(polygon, padding=0.03, aspect_ratio=width / max(height, 1.0))
                shaped_candidate = _shaped_best(shaped_rect)
                xs = [point[0] for point in polygon]
                ys = [point[1] for point in polygon]
                extent_x, extent_y = max(xs) - min(xs), max(ys) - min(ys)
                if extent_x > 1 and extent_y > 1:
                    poly_rect = bubble_inner_rect(
                        polygon, padding=0.03, aspect_ratio=extent_x / extent_y,
                    )
                    poly_candidate = _shaped_best(poly_rect)
                    if poly_candidate is not None and (
                        shaped_candidate is None
                        or poly_candidate[0] > shaped_candidate[0]
                    ):
                        shaped_rect, shaped_candidate = poly_rect, poly_candidate
                if shaped_candidate is not None and shaped_candidate[0] > best[0]:
                    # A shaped interior that cannot sit centered on the balloon
                    # (a wide/short band in the balloon's wider low half) trades
                    # centering for a marginal font bump — the text then hugs a
                    # jagged band (page 007: "Ehhhhh." at 65% height). Adopt the
                    # shaped interior only when its ink box, moved onto the
                    # balloon's guide (the tail slide's target), still stays
                    # inside the outline.
                    shaped = QRectF(*shaped_rect)
                    guide = bubble_inner_center(polygon)
                    if guide is not None:
                        check_x, check_y = float(guide[0]), float(guide[1])
                    else:
                        shape_contour = np.asarray(polygon, dtype=np.float32)
                        check_x = float(
                            shape_contour[:, 0].min() + shape_contour[:, 0].max()
                        ) / 2.0
                        check_y = float(
                            shape_contour[:, 1].min() + shape_contour[:, 1].max()
                        ) / 2.0
                    if _ink_box_inside(
                        shaped_candidate[0], shaped_candidate[1],
                        shaped.translated(
                            check_x - shaped.center().x(),
                            check_y - shaped.center().y(),
                        ),
                    ):
                        best, target = shaped_candidate, shaped
                # Candidate 3: the outline's fattest horizontal slab. A max-area
                # rectangle dodges concavities and tails, so a tapering bubble
                # can pin `Kunieda...` to a 92px interior while its middle rows
                # are 148px wide. Single-line dialogue rides that slab, centered
                # where it occurs. Adopt only on a real gain, and only when the
                # slab's corners stay inside the physical outline.
                contour = np.asarray(polygon, dtype=np.float32)
                if len(best[1].split('\n')) <= 2:
                    row_spans: dict = {}
                    for scan_y in range(
                        int(contour[:, 1].min()), int(contour[:, 1].max()) + 1,
                    ):
                        spans = polygon_inline_spans(list(polygon), False, float(scan_y))
                        if spans:
                            left, right = max(spans, key=lambda s: s[1] - s[0])
                            row_spans[scan_y] = (float(left), float(right))
                    if row_spans:
                        rows = sorted(row_spans)
                        best_win = None
                        for height in (24, 36, 48, 64, 96):
                            for start_index in range(len(rows)):
                                window_rows = rows[start_index:start_index + height]
                                if len(window_rows) < max(2, min(height, 24)):
                                    break
                                width = min(
                                    row_spans[y][1] - row_spans[y][0]
                                    for y in window_rows
                                )
                                if best_win is None or width > best_win[0]:
                                    best_win = (width, window_rows, height)
                        if best_win is not None:
                            span_w, window_rows, span_h = best_win
                            center_x = (
                                min(row_spans[y][0] for y in window_rows)
                                + max(row_spans[y][1] for y in window_rows)
                            ) / 2.0
                            band = QRectF(
                                center_x - span_w / 2.0,
                                float(window_rows[0]),
                                span_w, float(len(window_rows)),
                            )
                            if span_w > rect[2] * 1.25:
                                band_candidate = _shaped_best(
                                    (band.x(), band.y(), band.width(), band.height()),
                                )
                                if band_candidate is not None and band_candidate[0] >= best[0] * 1.25:
                                    if _ink_box_inside(
                                        band_candidate[0], band_candidate[1], band,
                                    ):
                                        best, target = band_candidate, band
            return best, target, min_broken

        best, target, min_broken = search_all(False)
        # One global split decision: whole words across every candidate
        # first. A one/two-word bubble is never split — the word IS the
        # bubble (`Teacher...`, `Kunieda...` read fine whole, and hyphenating
        # them is exactly the shredding users reject). A sentence splits
        # only when its whole fit is below the ~11pt readability floor and
        # the split buys at least 25% more (the tall-balloon rescue). CJK
        # targets never split: spaceless scripts wrap per character.
        if (
            (best is None or best[0] < 11.0)
            and len(clean_base.split()) >= 3
            and not is_cjk(language)
        ):
            split_best, split_target, split_broken = search_all(True)
            if split_best is not None and (best is None or split_best[0] > best[0]):
                best, target, min_broken = split_best, split_target, split_broken
        if best is None:
            # Even the interior overflows at minimum size. Stay centered at
            # minimum size inside the bubble instead of returning False to a
            # mask layout outside of it, keeping the DP breaks from the
            # minimum-size probe instead of Qt's greedy wrap.
            # A narrow interior can still be thinner than the longest word;
            # the anywhere-wrap would then split it mid-word (`COM`/`E`).
            # Widen symmetrically to the word width so it stays intact: the
            # bubble erosion margin usually still covers the overflow, and an
            # intact overflowing word reads better than a split one.
            item.setFontSize(4.0)
            item.setPlainText(min_broken)
            item.layout.reLayoutEverything()
            item.setRect(target, padding=False)
            needed_height, _ = item.layout.minSize()
            centered_height = min(max(needed_height, 1.0), target.height())
            centered_top = target.y() + (target.height() - centered_height) / 2.0
            intact_pad = max(0.0, float(item.padding())) * 2.0
            longest = _longest_word_width(item, cjk=is_cjk(language))
            if vertical:
                want_height = max(centered_height, longest + intact_pad)
                item.setRect(
                    QRectF(
                        target.x(),
                        target.center().y() - want_height / 2.0,
                        target.width(),
                        want_height,
                    ),
                    padding=False,
                )
            else:
                want_width = max(target.width(), longest + intact_pad)
                item.setRect(
                    QRectF(
                        target.center().x() - want_width / 2.0,
                        centered_top,
                        want_width,
                        centered_height,
                    ),
                    padding=False,
                )
            item.repaint_background()
            LOGGER.warning(
                'Translation exceeds the bubble interior even at minimum size; '
                'placed centered at minimum size.'
            )
            return True
        size, broken = best
        item.setFontSize(size)
        item.setPlainText(broken)
        item.layout.reLayoutEverything()
        item.setRect(target, padding=False)
        needed_height, _needed_width = item.layout.minSize()
        # Keep the full interior width so wrapping cannot reflow, then squeeze
        # the height to the laid-out content (plus visual padding) and center
        # vertically for equal top/bottom margins. No re-measure inside the
        # squeezed box: a re-layout there can re-wrap wider and force a
        # revert, which shoved short text back to the top of tall balloons.
        content_pad = max(0.0, float(item.padding())) * 2.0
        centered_height = min(max(needed_height, 1.0) + content_pad, target.height())
        item.setRect(
            QRectF(
                target.x(),
                target.center().y() - centered_height / 2.0,
                target.width(),
                centered_height,
            ),
            padding=False,
        )
        # Slide the fitted box onto the balloon's visual center. The maximal
        # interior rectangle sits off-center inside asymmetric balloons (tall
        # ovals with tails, jagged outlines), so centering within it leaves
        # the text hugging the top or a jagged band — exactly the
        # not-centered look users fix by hand. Prefer the interior centroid
        # guide (what Center-in-bubble uses) so the automatic fit lands
        # where the manual command does; fall back to the outline's bbox
        # center. The manual command slides unconditionally, so we do too —
        # a corner-poke test silently left text pinned to a band or the top
        # of the interior on tapering bubbles (page 007: `Kunieda...` sat
        # 24px above the bubble center).
        target_x = target_y = float('nan')
        if target_rect is None:
            contour = np.asarray(polygon, dtype=np.float32)
            guide = bubble_inner_center(polygon)
            if guide is not None:
                target_x, target_y = float(guide[0]), float(guide[1])
            else:
                target_x = float(contour[:, 0].min() + contour[:, 0].max()) / 2.0
                target_y = float(contour[:, 1].min() + contour[:, 1].max()) / 2.0
            box = item.absBoundingRect(qrect=True)
            shift_x = target_x - box.center().x()
            shift_y = target_y - box.center().y()
            if abs(shift_x) > 0.5 or abs(shift_y) > 0.5:
                # setPos alone moves the graphics item without syncing the
                # block model: the box would render centered but revert to
                # the saved position on the next page load. Route the move
                # through the logical-position setter, which syncs xyxy.
                item.set_logical_position(
                    item.logical_position() + QPointF(shift_x, shift_y)
                )
        item.repaint_background()
        box = item.absBoundingRect(qrect=True)
        LOGGER.info(
            'Bubble fit ok: target=%.0fx%.0f best=%.1fpt box=%.0fx%.0f at (%.0f,%.0f) '
            'guide=(%.0f,%.0f).',
            target.width(), target.height(), size, box.width(), box.height(),
            box.center().x(), box.center().y(), target_x, target_y,
        )
        return True
    finally:
        cursor.endEditBlock()
