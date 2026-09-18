from typing import Callable, List, Optional, Tuple
import re

import numpy as np

from .autolayout import (
    balloon_from_mask,
    balloon_from_polygon,
    build_segments,
    comic_line_breaks,
    join_layout_words,
    split_dash_compounds,
)
from .textblock import TextBlock, TextAlignment


def seg_text_paragraphs(
    text: str, lang: str, measure: Callable[[str], float],
) -> Tuple[List[str], List[float], str, List[bool]]:
    """Segment text paragraph by paragraph, keeping explicit line breaks.

    ``seg_text`` strips newlines, so split first and re-insert ``'\\n'``
    markers (measured as zero width) for the DP's mandatory breaks. Words
    are additionally split on whitespace into atomic units: ``seg_eng``
    glues short words (``SERIOUSLY ?`` becomes one unit) which would hide
    break opportunities from the DP and let Qt emergency-wrap punctuation
    onto its own line. The DP's cost model already discourages awkward
    micro-lines, so atomic units are strictly more expressive. Dash
    compounds split after interior dashes (``EHH—BUT``); ``glue`` marks
    pieces that rejoin with no space.

    >>> seg_text_paragraphs('hi there', 'English', len)
    (['hi', 'there'], [2.0, 5.0], ' ', [False, False])
    >>> seg_text_paragraphs('really ?', 'English', len)
    (['really', '?'], [6.0, 1.0], ' ', [False, False])
    >>> seg_text_paragraphs('EHH—BUT CUTE', 'English', len)
    (['EHH—', 'BUT', 'CUTE'], [4.0, 3.0, 4.0], ' ', [False, True, False])
    >>> seg_text_paragraphs('AH　HA　HA', 'English', len)
    (['AH', 'HA', 'HA'], [2.0, 2.0, 2.0], ' ', [False, False, False])
    """
    from .text_processing import seg_text

    # Machine-translated text routinely carries full-width spaces, no-break
    # spaces, and tabs, which seg_eng would keep inside one unbreakable unit
    # (overflowing any interior) while Qt measures words across all Unicode
    # whitespace. Normalize exotic runs to ASCII spaces first; paragraph
    # newlines are split before this and never reach the substitution.
    exotic_whitespace = re.compile(r'[\t\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+')

    units: List[str] = []
    advances: List[float] = []
    glue: List[bool] = []
    delimiter = ' '
    first = True
    for paragraph in text.split('\n'):
        if not first:
            units.append('\n')
            advances.append(0.0)
            glue.append(False)
        first = False
        paragraph = exotic_whitespace.sub(' ', paragraph)
        words, delimiter = seg_text(paragraph, lang)
        for word in words:
            parts = word.split(' ') if delimiter == ' ' else [word]
            for part in parts:
                if not part:
                    continue
                for position, dash_piece in enumerate(split_dash_compounds(part)):
                    units.append(dash_piece)
                    advances.append(float(measure(dash_piece)))
                    # Only the first piece of each space-separated part starts
                    # a new word; dash pieces glue to their predecessor.
                    glue.append(position > 0)
    return units, advances, delimiter, glue


def _join_breaks(
    seg_words: List[str], breaks: List[int], delimiter: str,
    glue: Optional[List[bool]] = None,
) -> List[str]:
    """Join segmented words at DP break offsets into display lines.

    Glued dash pieces reattach with no space.

    >>> _join_breaks(['hello', 'world'], [1, 2], ' ')
    ['hello', 'world']
    >>> _join_breaks(['EHH—', 'BUT', 'CUTE'], [2, 3], ' ', [False, True, False])
    ['EHH—BUT', 'CUTE']
    """
    lines: List[str] = []
    start = 0
    for end in breaks:
        if glue is None:
            lines.append(delimiter.join(seg_words[start:end]))
        else:
            lines.append(join_layout_words(seg_words[start:end], delimiter, glue[start:end]))
        start = end
    return lines


def layout_text(
    blk: TextBlock,
    mask: np.ndarray,
    mask_xyxy: List,
    centroid: List,
    words: List[str],
    wl_list: List[int],
    delimiter: str,
    delimiter_len: int,
    line_height: int,
    max_central_width=np.inf,
    vertical: bool = False,
    glue: Optional[List[bool]] = None,
) -> Tuple[str, List]:
    """Break words into balloon-aware lines with a balloon-fitting DP.

    Replaces the former greedy center-out walk (which probed single mask
    columns and zig-zagged when the balloon tapered) with
    :mod:`ballontranslator.utils.autolayout`: per-line usable widths
    sampled across the glyph ink band from the bubble contour (or an ellipse
    fallback from the mask), one shared visual axis, and a DP minimizing
    slack plus linguistic break penalties. Breaks stay whole-word; intra-word
    soft-hyphen splits remain Qt's job at render time, since explicit
    ``'\\n'`` joins cannot show mid-word hyphen breaks. ``glue`` marks dash
    pieces (``EHH—``/``BUT``) that rejoin with no space; breaks after dashes
    need no dictionary and render literally, unlike soft hyphens.

    ``vertical`` runs the same DP transposed (columns break along height).

    >>> blk = TextBlock(xyxy=[0, 0, 100, 60])
    >>> mask = (np.ones((60, 100), dtype=np.uint8) * 255)
    >>> text, xywh = layout_text(
    ...     blk, mask, [0, 0, 100, 60], [50, 30],
    ...     ['hello', 'world'], [30, 30], ' ', 4, 12)
    >>> text.count(chr(10)) >= 0 and xywh[2] > 0 and xywh[3] > 0
    True
    """
    vertical = bool(vertical or getattr(blk, 'vertical', False))
    alignment = blk.alignment

    if max_central_width == np.inf:
        max_central_width = mask.shape[1] if not vertical else mask.shape[0]

    # ``words``/``wl_list`` stay aligned: plain ``seg_text`` callers never
    # contain breaks, while ``seg_text_paragraphs`` callers pre-split them
    # into zero-width '\n' units. Anything else embedded is split defensively.
    seg_words: List[str] = []
    unit_advances: List[float] = []
    unit_glue: List[bool] = []
    for position, (word, advance) in enumerate(zip(words, wl_list)):
        # Incoming glue attaches this word to the previous unit, unless a
        # paragraph break sits between them.
        attached = (
            bool(glue[position]) if glue is not None else False
        ) and bool(seg_words) and seg_words[-1] != '\n'
        if word == '\n':
            seg_words.append(word)
            unit_advances.append(0.0)
            unit_glue.append(False)
        elif '\n' in word:
            parts = word.split('\n')
            share = float(advance) / max(1, len([p for p in parts if p]))
            for index, part in enumerate(parts):
                if index > 0:
                    seg_words.append('\n')
                    unit_advances.append(0.0)
                    unit_glue.append(False)
                if part:
                    seg_words.append(part)
                    unit_advances.append(share)
                    unit_glue.append(False)
        else:
            seg_words.append(word)
            unit_advances.append(float(advance))
            unit_glue.append(attached)
    full_text = join_layout_words(
        [w for w in seg_words if w != '\n'], delimiter,
        [g for w, g in zip(seg_words, unit_glue) if w != '\n'],
    )

    if not seg_words:
        return '', [0, 0, 0, 0]

    segments = build_segments(
        seg_words, unit_advances, float(delimiter_len), delimiter, full_text,
        glue=unit_glue,
    )
    if not segments:
        return '', [0, 0, 0, 0]
    # Break offsets index the filtered segments, so join/measure without
    # the '\n' markers that only carried mandatory-break flags.
    break_words = [word for word in seg_words if word != '\n']
    break_advances = [adv for word, adv in zip(seg_words, unit_advances) if word != '\n']
    break_glue = [flag for word, flag in zip(seg_words, unit_glue) if word != '\n']

    ink_thickness = max(1.0, float(line_height) / max(1.0, float(getattr(blk, 'line_spacing', 1.0) or 1.0)))
    minimum_air = max(4.0, float(line_height) * 0.25)
    origin = (float(mask_xyxy[0]), float(mask_xyxy[1]))
    if getattr(blk, 'bubble_polygon', None):
        balloon = balloon_from_polygon(
            blk.bubble_polygon, origin, float(mask.shape[1]), float(mask.shape[0]), minimum_air,
        )
        if not balloon.contours:
            balloon = balloon_from_mask(mask, minimum_air)
    else:
        balloon = balloon_from_mask(mask, minimum_air)
    if np.isfinite(max_central_width):
        cap = float(max_central_width)
        if vertical:
            balloon.height = min(balloon.height, cap)
        else:
            balloon.width = min(balloon.width, cap)
    air = (balloon.air(ink_thickness), balloon.air(ink_thickness))

    result = comic_line_breaks(
        segments, balloon, vertical, float(line_height), ink_thickness, air,
        allow_hyphenation=False,
    )
    breaks = result.breaks or [len(segments)]
    lines = _join_breaks(break_words, breaks, delimiter, break_glue)

    widths: List[float] = []
    cursor = 0
    for end in breaks:
        span = break_advances[cursor:end]
        gaps = sum(
            1 for position in range(cursor + 1, end) if not break_glue[position]
        )
        widths.append(float(sum(span)) + float(delimiter_len) * max(0, gaps))
        cursor = end
    canvas_w = int(max(widths)) if widths else 0
    canvas_h = int(len(lines) * line_height)
    if vertical:
        canvas_w, canvas_h = int(len(lines) * line_height), int(max(widths)) if widths else 0

    centroid_x, centroid_y = centroid
    center_x = mask_xyxy[0] + centroid_x
    center_y = mask_xyxy[1] + centroid_y
    if alignment == TextAlignment.Center or int(getattr(alignment, 'value', alignment)) == 1:
        abs_x = int(round(center_x - canvas_w / 2))
        abs_y = int(round(center_y - canvas_h / 2))
    else:
        abs_x, abs_y = int(round(origin[0])), int(round(origin[1]))

    return '\n'.join(lines), [abs_x, abs_y, canvas_w, canvas_h]
