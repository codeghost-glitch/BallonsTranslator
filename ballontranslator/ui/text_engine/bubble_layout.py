"""Detected-bubble fitting through the existing editable text item layout."""

from functools import lru_cache
import math
import re
import cv2
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QFontMetricsF, QTextCursor, QTextDocument

from ballontranslator.utils.bubble import bubble_inner_rect
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.fontformat import TextAlignment, px2pt
from ballontranslator.utils.autolayout import (
    ComicBalloon,
    build_segments,
    comic_line_breaks,
    join_layout_words,
    largest_fitting_font_size,
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


def _balloon_prebreak(
    base_text: str, language: str, font_metrics: QFontMetricsF,
    balloon: ComicBalloon, line_height: float, ink: float, *, vertical: bool,
) -> Tuple[str, bool]:
    """Break text with the balloon DP against the interior width.

    Breaks stay on whole-word boundaries (soft hyphens remain Qt's job for
    intra-word splits); the DP only chooses *where* phrases wrap, balancing
    lines with linguistic penalties. The balloon carries a rectangular
    contour on purpose: placement stays a Qt rectangle inside the inset
    interior, so tapered end-line widths would only over-wrap into extra
    lines whose height pressure shrinks readable text for free. (The free
    mask-fallback placement keeps tapered profiles to hug the balloon.)
    Returns the explicitly broken text and whether any line overflows.

    >>> callable(_balloon_prebreak)
    True
    """
    units, advances, delimiter, glue = seg_text_paragraphs(
        base_text, language, font_metrics.horizontalAdvance,
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
    rect_balloon = ComicBalloon(
        balloon.width, balloon.height,
        [[(0.0, 0.0), (balloon.width, 0.0),
          (balloon.width, balloon.height), (0.0, balloon.height)]],
        balloon.minimum_air,
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
    allow_shy_word_split: bool = False, preferred_max: Optional[float] = None,
) -> Tuple[Optional[Tuple[float, str]], str]:
    """Search largest-first for the biggest fitting size and its breaks.

    Probes down from the maximum (balloon fits are non-monotonic: a smaller
    font can reflow into more lines with worse tapered ends) then refines,
    mirroring the largest-first search in
    :mod:`ballontranslator.utils.autolayout`. Each probe re-breaks
    the text with the balloon DP at that size and reserves effect padding on
    every side; the DP never splits words, so narrow bubbles shrink instead
    of breaking `COME` into `COM`/`E`. ``preferred_max`` caps growth at the
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

    def layout_at(size: float) -> Tuple[str, float, float, bool]:
        item.setFontSize(size)
        metrics = QFontMetricsF(_current_font(item))
        pad = max(0.0, float(item.padding())) * 2.0
        spacing = float(getattr(item.fontformat, 'line_spacing', 1.0) or 1.0)
        line_height = max(1.0, metrics.height() * spacing)
        ink = max(1.0, metrics.ascent() + metrics.descent())
        balloon = ComicBalloon(
            max(1.0, target.width() - pad), max(1.0, target.height() - pad), [], 0.0,
        )
        broken, overflowed = _balloon_prebreak(
            base_text, language, metrics, balloon, line_height, ink, vertical=vertical,
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
        _, needed_width, needed_height, _overflowed = candidate
        pad = max(0.0, float(item.padding())) * 2.0
        text_pad = max(0.0, float(getattr(item.layout, 'text_padding', 0.0) or 0.0))
        # 2px tolerance covers normal glyph overhang; the erosion margin to
        # the bubble border (typically 8px+) absorbs it. text_padding derives
        # from vertical ink extents, so it is kept for the height check only:
        # counting it against the width rejects words that visibly fit.
        fits_size = (
            needed_width - text_pad + pad <= target.width() + 2.0
            and needed_height + pad <= target.height() + 2.0
        )
        # The DP only decides where phrases wrap; placement stays rectangular,
        # so size follows Qt measurement plus the interior extent, not the
        # tapered profile widths (which would shrink readable text for free).
        # Without hyphenation, still require the longest run to fit so narrow
        # bubbles shrink instead of splitting `COME` into `COM`/`E`. Columns
        # run along the height in vertical text, so that is the constraining
        # extent there. This check is strict (no overhang tolerance): the
        # layout wraps at word-boundary-or-anywhere, so even a word 1px wider
        # than the box is visually split mid-word. Paint overhang stays
        # covered by the bubble erosion margin instead.
        fits_words = allow_shy_word_split or (
            _longest_word_width(item, cjk=is_cjk(language))
            <= (target.height() if vertical else target.width()) - pad
        )
        return fits_size and fits_words

    first = layout_at(lower)
    probe_needed = (first[1], first[2], max(0.0, float(item.padding())) * 2.0,
                    _longest_word_width(item, cjk=is_cjk(language)))
    if not fits(first):
        best: Optional[Tuple[float, str]] = None
    else:
        sized = largest_fitting_font_size(
            lower, upper,
            lambda size: (size, layout_at(size)),
            lambda candidate: fits(candidate[1]),
        )
        best = None if sized is None else (sized[0], sized[1][0])
    if best is None:
        if probe_needed is not None:
            needed_width, needed_height, pad, longest = probe_needed
            LOGGER.warning(
                'Bubble fit rejected at 4pt: target=%.1fx%.1f needed=%.1fx%.1f '
                'pad=%.1f longest=%.1f hyphenate=%s.',
                target.width(), target.height(), needed_width, needed_height,
                pad, longest, allow_shy_word_split,
            )
        return None, first[0]
    return best, first[0]


def fit_text_to_bubble(
    item: TextBlkItem, text: Optional[str], *, hyphenate: bool, language: str,
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
    effect padding for stroked/bold ink and, without hyphenation, shrinks narrow
    bubbles until the longest word fits instead of splitting `COME` into
    `COM`/`E`. Hyphenation is a true last resort: the clean search runs first
    and soft hyphens are only considered when it misses entirely, so mid-word
    splits never trade readability for size. A clean fit at any size (even
    pinned to the 4pt floor) stays intact, and the hyphenated search stays
    small instead of chasing a giant shredded size.
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
    vertical = bool(item.blk.vertical)
    original_alignment = int(item.fontformat.alignment)
    cursor = QTextCursor(item.document())
    cursor.beginEditBlock()
    try:
        if text is not None:
            item.setPlainText(text)
        if not item.toPlainText().strip():
            return False
        # Hyphenation is a true last resort: the clean search runs
        # first and soft hyphens are only considered when it misses entirely.
        # Mid-word splits like ABAYAS/HI are never worth a slightly bigger
        # size. Stripping first keeps the clean search honest when a previous
        # fit left soft hyphens.
        clean_base = item.toPlainText().replace('\u00ad', '')
        item.setPlainText(clean_base)
        # Centered lines read as one centered block inside the bubble.
        if original_alignment != int(TextAlignment.Center):
            item.setAlignment(int(TextAlignment.Center), repaint_background=False)
        target = QRectF(*rect)
        # Scanlation practice matches the raw instead of filling bubbles:
        # cap growth a little above the detected source size while leaving
        # shrinking untouched. Blocks without a detection keep the full
        # search range, as does an explicit global-size override.
        detected = max(0.0, float(getattr(item.blk, '_detected_font_size', -1) or -1))
        if detected > 0 and pcfg.let_fntsize_flag == 1:
            detected = 0.0
        preferred_max = px2pt(detected) * 1.25 if detected > 0 else None
        best, min_broken = _best_font_for_target(
            item, target, language=language, vertical=vertical,
            preferred_max=preferred_max,
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
                    preferred_max=preferred_max,
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
                best, target = shaped_candidate, QRectF(*shaped_rect)
        used_hyphens = False
        if hyphenate and best is None:
            # Hyphenation is a fitting tool, never a growth tool. When the
            # clean search fits at any size (even pinned to the 4pt floor),
            # keep words intact: shredding them only buys a bigger size at
            # the cost of syllable stacks like `KUNIE-DA!` / `PHONE-S`.
            # The hyphenated search also stays small so a tall bubble can't
            # trade one-syllable lines for a giant shredded size.
            hyphenated_base = clean_base
            item.setPlainText(hyphenated_base)
            hyphenate_document(item.document(), language)
            hyphenated_base = item.toPlainText()
            if '\u00ad' in hyphenated_base:
                hyphen_cap = 6.0
                if preferred_max is not None:
                    hyphen_cap = min(preferred_max, hyphen_cap)
                hyphenated, _ = _best_font_for_target(
                    item, target, language=language, vertical=vertical,
                    allow_shy_word_split=True, preferred_max=hyphen_cap,
                )
                if hyphenated is not None and (
                    best is None or math.floor(hyphenated[0]) > math.floor(best[0])
                ):
                    best = hyphenated
                    used_hyphens = True
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
        # Keep the full interior width so wrapping does not reflow, then
        # squeeze the height to content and center vertically for equal
        # top/bottom margins. Horizontal centering comes from the Center
        # alignment set above. If rounding makes the squeezed box overflow,
        # keep the verified full interior instead of crossing the border.
        centered_height = min(max(needed_height, 1.0), target.height())
        centered_top = target.y() + (target.height() - centered_height) / 2.0
        centered = QRectF(target.x(), centered_top, target.width(), centered_height)
        item.setRect(centered, padding=False)
        final_height, final_width = item.layout.minSize()
        final_pad = max(0.0, float(item.padding())) * 2.0
        if (
            final_width + final_pad > target.width() + 2.0
            or final_height + final_pad > centered.height() + 2.0
        ):
            item.setRect(target, padding=False)
        # The manual guide and drag snap anchor at the outline centroid; the
        # eroded interior rectangle can sit a few px off it on ellipses.
        # Slide the fitted box onto that same anchor only when the shifted
        # box still stays inside the physical outline — otherwise the fitted
        # rectangle already hugs the border and the move would clip glyphs.
        if target_rect is None:
            from ballontranslator.utils.bubble import bubble_inner_center
            guide = bubble_inner_center(polygon)
            if guide is not None:
                box = item.absBoundingRect(qrect=True)
                shift_x = guide[0] - box.center().x()
                shift_y = guide[1] - box.center().y()
                shifted = box.translated(shift_x, shift_y)
                contour = np.asarray(polygon, dtype=np.float32)
                corners_inside = all(
                    cv2.pointPolygonTest(contour, (float(x), float(y)), True) >= -2.0
                    for x in (shifted.left(), shifted.right())
                    for y in (shifted.top(), shifted.bottom())
                )
                if corners_inside:
                    item.setPos(item.pos() + QPointF(shift_x, shift_y))
        item.repaint_background()
        LOGGER.info(
            'Bubble fit ok: target=%.0fx%.0f best=%.1fpt box=%.0fx%.0f hyphenate=%s.',
            target.width(), target.height(), size,
            item.absBoundingRect(qrect=True).width(),
            item.absBoundingRect(qrect=True).height(), used_hyphens,
        )
        return True
    finally:
        cursor.endEditBlock()
