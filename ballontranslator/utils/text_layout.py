from typing import List, Optional, Tuple
import numpy as np

from .imgproc_utils import rotate_image
from .textblock import TextBlock, TextAlignment

class Line:

    def __init__(self, text: str = '', pos_x: int = 0, pos_y: int = 0, length: float = 0, spacing: int = 0) -> None:
        self.text = text
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.length = int(length)
        self.num_words = 0
        if text:
            self.num_words += 1
        self.spacing = 0
        self.add_spacing(spacing)

    def append_right(self, word: str, w_len: int, delimiter: str = ''):
        self.text = self.text + delimiter + word
        if word:
            self.num_words += 1
        self.length += w_len

    def append_left(self, word: str, w_len: int, delimiter: str = ''):
        self.text = word + delimiter + self.text
        if word:
            self.num_words += 1
        self.length += w_len

    def add_spacing(self, spacing: int):
        self.spacing = spacing
        self.pos_x -= spacing
        self.length += 2 * spacing

    def strip_spacing(self):
        self.length -= self.spacing * 2
        self.pos_x += self.spacing
        self.spacing = 0

def row_width_profile(poly_arr: np.ndarray, y_start: int, y_end: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Horizontal extent of a polygon at every integer row, exactly.

    Line budgets read from this profile follow the real outline, so
    rectangular, lopsided, and concave bubbles budget correctly instead of
    being rounded off by an ellipse fitted to their bounding box. That
    approximation needed fudge factors to stay inside the bubble; measuring
    the outline itself needs none.

    Returns ``(left, right, y_start)`` in page coordinates. Rows the polygon
    does not cover come back inverted (left > right), which callers read as
    "no room".

    >>> poly = np.array([[0, 0], [100, 0], [50, 100]], np.float32)
    >>> left, right, y0 = row_width_profile(poly, 0, 101)
    >>> y0, int(left[0]), int(right[0])
    (0, 0, 100)
    >>> int(left[50]), int(right[50])
    (25, 75)
    >>> bool(left[100] > right[100])
    True
    """
    y0 = int(y_start)
    rows = np.arange(y0, int(y_end), dtype=np.float64)
    x1 = poly_arr[:, 0].astype(np.float64)
    y1 = poly_arr[:, 1].astype(np.float64)
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    hit = ((np.minimum(y1, y2)[:, None] <= rows[None, :])
           & (rows[None, :] < np.maximum(y1, y2)[:, None]))
    with np.errstate(divide='ignore', invalid='ignore'):
        xs = x1[:, None] + (rows[None, :] - y1[:, None]) * (x2 - x1)[:, None] / (y2 - y1)[:, None]
    ok = hit & np.isfinite(xs)
    left = np.where(ok, xs, np.inf).min(axis=0)
    right = np.where(ok, xs, -np.inf).max(axis=0)
    return left, right, y0


def line_is_valid(line: Line, new_len: int, delimiter_len, max_width, words_length, srcline_wlist, line_no: int, line_height, ref_src_lines: bool = False, row_profile=None):
    """Whether a line may grow to ``new_len`` under the width budget.

    >>> line = Line('word', 0, 0, 60)
    >>> line_is_valid(line, 80, 0, 100, 80, None, 0, 20)
    True
    >>> line_is_valid(line, 80, 0, 0, 80, None, 0, 20)
    False
    >>> prof = (np.array([20.0, 20.0, 20.0]), np.array([80.0, 80.0, 80.0]), 0)
    >>> line_is_valid(Line('word', 0, -10, 60), 80, 0, 100, 80, None, 0, 20, row_profile=prof)
    False
    """
    if row_profile is not None:
        # Shape-aware typesetting: the budget is the outline's own width at
        # this line's row, so line lengths follow the bubble instead of its
        # bounding box. A row outside the outline has no room.
        p_left, p_right, p_y0 = row_profile
        row = int(line.pos_y + line_height / 2) - p_y0
        ecap = (p_right[row] - p_left[row]) if 0 <= row < len(p_left) else 1.0
        max_width = min(max_width, max(ecap, 1.0))
    if ref_src_lines:
        # if line_no >= 0 and line_no < len(srcline_wlist):
        #     _max_width = min(srcline_wlist[line_no], max_width)
        # else:
        #     _max_width = max_width
        if line_no >= 0 and line_no < len(srcline_wlist):
            _max_width = srcline_wlist[line_no] * words_length
        else:
            _max_width = np.inf
            _max_width = max(srcline_wlist) * words_length
        _max_width = _max_width + delimiter_len * line.num_words
        max_width = min(max_width, _max_width)

    if max_width <= 0:
        # No budget: a stale outline row, a zero-width source line, or an
        # empty profile. The line cannot grow, and the ratio comparison
        # below would divide by it.
        return False
    if new_len < max_width:
        return True
    else:
        if line.length / max_width < max_width / new_len:
            return True
        else:
            return False

# Stands in for the hyphen at a break this module made. A '-' the
# translator typed is a real character and must survive untouched, so the
# joiner can only swallow a break it can see the mark on. It never reaches
# the canvas: _join_hyphen_runs runs on every rendered line and drops the
# mark, restoring a visible hyphen where the break ends the line.
_HYPHEN_BREAK = '\x1e'

# Typographic minimum for splitting a word. Below it a 2+2 break leaves a
# fragment that reads as a typo ("Wha-t?"), and forcing one costs more than
# the slightly smaller unsplit word costs in a tight balloon. Shared by the
# advisory pre-split and the in-line break, which must agree or a word that
# the pre-split leaves whole still gets broken later.
#
# The same rule covers a token that is not a word at all. seg_eng glues a
# word of one or two letters to a neighbour so a line can carry them as one
# unit ("sure to", "up on", "a tod-"), and the line breaker then treats that
# run as indivisible. A hyphen is only ever legal inside a word, so such a
# token passes through whole: splitting it would render "sure-" / " to".
_MIN_HYPHEN_WORD = 6

# The shortest piece a break may leave on either side, head and tail alike.
# Three characters is the shortest fragment that still reads as a syllable;
# one or two read as a typo on the page ("fre- quentl- y."), and a broken word
# is worse than a word that overhangs its line by a glyph. Shared with the
# in-line break for the same reason as _MIN_HYPHEN_WORD.
_MIN_PIECE = 3
# Balloon lettering breaks a word once or twice, not repeatedly: the
# line-breaker closes a line with an in-line hyphenation head for whatever
# still does not fit, so a third advisory cut only adds a hyphen the reader
# has to read past. Measured on a 167x287 balloon, one cut per word cost
# 13.1pt -> 10.9pt against two, for no readable gain.
_MAX_CUTS_PER_WORD = 2


def _hyphen_head_for_line(
    line: Line, word: str, hyphenator, measure, delimiter_len,
    line_no: int, max_width, words_length, srcline_wlist, line_height,
    ref_src_lines, row_profile,
):
    """Widest pyphen head that still grows this line legally.

    Returns (head, head_width, tail, tail_width) or None when no split
    fits or the word has no hyphen points.
    """
    if hyphenator is None or measure is None:
        return None
    best = None
    # See _MIN_HYPHEN_WORD: a short word, or a run of words seg_eng glued
    # into one token, has no legal break point at all.
    if len(word) < _MIN_HYPHEN_WORD or ' ' in word:
        return None
    for p in hyphenator.positions(word):
        if p <= 0 or p >= len(word):
            continue
        bare = word[:-1] if word.endswith(_HYPHEN_BREAK) else word
        head = bare[:p] + _HYPHEN_BREAK
        hw = measure(bare[:p] + '-')
        if line_is_valid(line, line.length + hw + delimiter_len, delimiter_len,
                         max_width, words_length, srcline_wlist, line_no,
                         line_height, ref_src_lines, row_profile):
            best = (head, hw, bare[p:], measure(bare[p:]))
        elif best is not None:
            break  # positions ascend: later heads are only wider
    return best


def layout_lines_aligncenter(
    blk: TextBlock,
    mask: np.ndarray, 
    words: List[str], 
    centroid: List[int],
    wl_list: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    max_central_width: float = np.inf,
    word_break: bool = False,
    ref_src_lines = False,
    srcline_wlist=None,
    start_from_top=False,
    row_profile=None,
    hyphenator=None,
    measure=None
)->List[Line]:
    
    lh_pad = 0
    if blk.line_spacing > 1:
        lh_pad = int(np.ceil(line_height - line_height / blk.line_spacing))

    centroid_x, centroid_y = centroid
    adjust_x = adjust_y = 0

    border_thr = 220
    
    # layout the central line, the center word is approximately aligned with the centroid of the mask
    num_words = len(words)
    len_left, len_right = [], []
    wlst_left, wlst_right = [], []
    sum_left, sum_right = 0, 0
    words_length = sum(wl_list)
    if num_words > 1:
        wl_array = np.array(wl_list, dtype=np.float64)
        wl_cumsums = np.cumsum(wl_array)
        wl_cumsums = wl_cumsums - wl_cumsums[-1] / 2 - wl_array / 2
        central_index = np.argmin(np.abs(wl_cumsums))

        if central_index > 0:
            wlst_left = words[:central_index]
            len_left = wl_list[:central_index]
            sum_left = np.sum(len_left)
        if central_index < num_words - 1:
            wlst_right = words[central_index + 1:]
            len_right = wl_list[central_index + 1:]
            sum_right = np.sum(len_right)
    else:
        central_index = 0

    pos_y = centroid_y - line_height // 2
    pos_x = centroid_x - wl_list[central_index] // 2

    bh, bw = mask.shape[:2]
    central_line = Line(words[central_index], pos_x, pos_y, wl_list[central_index], spacing)
    line_bottom = pos_y + line_height
    while (sum_left > 0 or sum_right > 0) and not start_from_top:
        left_valid, right_valid = False, False

        if sum_left > 0:
            new_len_l = central_line.length + len_left[-1] + delimiter_len
            new_x_l = centroid_x - new_len_l // 2
            new_r_l = new_x_l + new_len_l
            if (new_x_l > 0 and new_r_l < bw):
                left_col = mask[pos_y: line_bottom - lh_pad, new_x_l]
                right_col = mask[pos_y: line_bottom - lh_pad, new_r_l]
                if left_col.size and right_col.size and \
                    left_col.mean() > border_thr and right_col.mean() > border_thr:
                    left_valid = True
        if sum_right > 0:
            new_len_r = central_line.length + len_right[0] + delimiter_len
            new_x_r = centroid_x - new_len_r // 2 - line_height // 2
            new_r_r = centroid_x + new_len_r // 2 + line_height // 2
            if (new_x_r > 0 and new_r_r < bw):
                left_col = mask[pos_y: line_bottom - lh_pad, new_x_r]
                right_col = mask[pos_y: line_bottom - lh_pad, new_r_r]
                if left_col.size and right_col.size and \
                    left_col.mean() > border_thr and right_col.mean() > border_thr:
                    right_valid = True

        insert_left = False
        if left_valid and right_valid:
            if sum_left > sum_right:
                insert_left = True
        elif left_valid:
            insert_left = True
        elif not right_valid:
            break

        if insert_left:
            new_len = central_line.length + len_left[-1] + delimiter_len
        else:
            new_len = central_line.length + len_right[0] + delimiter_len

        line_valid = line_is_valid(central_line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, -1, line_height, ref_src_lines, row_profile=row_profile)
        if ref_src_lines and not line_valid and len(srcline_wlist) == 1:
            if new_len < max_central_width:
                line_valid = True
        if not line_valid:
            break

        if insert_left:
            central_line.append_left(wlst_left.pop(-1), len_left[-1] + delimiter_len, delimiter)
            sum_left -= len_left.pop(-1)
            central_line.pos_x = new_x_l
        else:
            central_line.append_right(wlst_right.pop(0), len_right[0] + delimiter_len, delimiter)
            sum_right -= len_right.pop(0)
            central_line.pos_x = new_x_r

    line_right_no = line_left_no = 0
    if ref_src_lines:
        nl = len(srcline_wlist)
        if nl % 2 == 0:
            line_right_no = nl // 2
            line_left_no = nl // 2 - 1
        else:
            line_right_no = nl // 2 + 1
            line_left_no = nl // 2 - 1

    if not start_from_top:
        central_line.strip_spacing()
        lines = [central_line]
    else:
        lines = []
        sum_right = sum(wl_list)
        sum_left = 0
        wlst_right = words
        len_right = wl_list
        line_right_no = 0

    # layout bottom half
    if sum_right > 0:
        w, wl = wlst_right.pop(0), len_right.pop(0)
        pos_x = centroid_x - wl // 2
        if start_from_top:
            pos_y = centroid_y - int(blk.bounding_rect()[3] / 2)
        else:
            pos_y = centroid_y + line_height // 2
        pos_y = max(0, min(pos_y, mask.shape[0] - 1))
        top_mean = mask[pos_y, :].mean()
        x_mean = mask.mean(axis=1)
        base_mean = x_mean.max() / 2
        if top_mean < base_mean:
            available_y = np.where(
                x_mean[pos_y:] > base_mean
            )[0]
            if len(available_y) > 0:
                adjust_y = min(available_y[0], line_height)
                pos_y = pos_y + adjust_y
        line_bottom = pos_y + line_height
        line = Line(w, pos_x, pos_y, wl, spacing)
        lines.append(line)
        sum_right -= wl
        while sum_right > 0:
            w, wl = wlst_right.pop(0), len_right.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2 - line_height // 2
            right_x = new_x + new_len + line_height // 2
            if new_x < 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom - lh_pad, new_x].mean() < border_thr or\
                mask[pos_y: line_bottom - lh_pad, right_x].mean() < border_thr:
                line_valid = False
                if ref_src_lines and (len(wl_list) == 1 or line_right_no + 1 >= len(srcline_wlist)) and \
                    line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_right_no, line_height, ref_src_lines, row_profile=row_profile):
                    line_valid = True
            else:
                line_valid = True
            if line_valid:
                # Optical fill: when the full word breaks the width budget,
                # close this line with its hyphen-fitting head and wrap the
                # tail instead of leaving a ragged gap.
                if hyphenator is not None and measure is not None and new_len > max_central_width:
                    split = _hyphen_head_for_line(line, w, hyphenator, measure, delimiter_len, line_right_no, max_central_width, words_length, srcline_wlist, line_height, ref_src_lines, row_profile)
                    if split is not None:
                        head, hw, tail, tw = split
                        h_len = line.length + hw + delimiter_len
                        line.append_right(head, hw + delimiter_len, delimiter)
                        line.pos_x = centroid_x - h_len // 2 - line_height // 2
                        w, wl = tail, tw
                        line_valid = False
                if line_valid:
                    line.append_right(w, wl+delimiter_len, delimiter)
                    line.pos_x = new_x
                    line_valid = line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_right_no, line_height, ref_src_lines, row_profile=row_profile)
                    if not line_valid:
                        if sum_right > 0:
                            w, wl = wlst_right.pop(0), len_right.pop(0)
                            sum_right -= wl
                        else:
                            line.strip_spacing()
                            break

            if not line_valid:
                pos_x = centroid_x - wl // 2
                pos_y = line_bottom
                line_bottom += line_height
                line.strip_spacing()
                line = Line(w, pos_x, pos_y, wl, spacing)
                lines.append(line)
                line_right_no += 1

    # layout top half
    if sum_left > 0:
        w, wl = wlst_left.pop(-1), len_left.pop(-1)
        pos_x = centroid_x - wl // 2
        pos_y = centroid_y - line_height // 2 - line_height
        pos_y = max(0, min(pos_y, mask.shape[0] - 1))
        line_bottom = pos_y + line_height
        line = Line(w, pos_x, pos_y, wl, spacing)
        lines.insert(0, line)
        sum_left -= wl
        while sum_left > 0:
            w, wl = wlst_left.pop(-1), len_left.pop(-1)
            sum_left -= wl
            new_len = line.length + wl + delimiter_len
            new_x = centroid_x - new_len // 2 - line_height // 2
            right_x = new_x + new_len + line_height // 2
            if new_x <= 0 or right_x >= bw:
                line_valid = False
            elif mask[pos_y: line_bottom - lh_pad, new_x].mean() < border_thr or\
                mask[pos_y: line_bottom - lh_pad, right_x].mean() < border_thr:
                line_valid = False
                if ref_src_lines and line_left_no - 1 < 0 and \
                    line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_left_no, line_height, ref_src_lines, row_profile=row_profile):
                    line_valid = True
            else:
                line_valid = True
            if line_valid:
                line.append_left(w, wl+delimiter_len, delimiter)
                line.pos_x = new_x
                line_valid = line_is_valid(line, new_len, delimiter_len, max_central_width, words_length, srcline_wlist, line_left_no, line_height, ref_src_lines, row_profile=row_profile)
                if not line_valid:
                    if sum_left > 0:
                        w, wl = wlst_left.pop(-1), len_left.pop(-1)
                        sum_left -= wl
                    else:
                        line.strip_spacing()
                        break

            if not line_valid :
                pos_x = centroid_x - wl // 2
                pos_y -= line_height
                line_bottom = pos_y + line_height
                line.strip_spacing()
                line = Line(w, pos_x, pos_y, wl, spacing)
                lines.insert(0, line)
                line_left_no -= 1
    
    return lines, (adjust_x, adjust_y)

def layout_lines_alignside(
    blk: TextBlock,
    mask: np.ndarray, 
    words: List[str], 
    origin: List[int],
    wl_list: List[int], 
    delimiter_len: int, 
    line_height: int,
    spacing: int = 0,
    delimiter: str = ' ',
    word_break: bool = False,
    max_width: int = np.inf,
    ref_src_lines = False,
    srcline_wlist=None,
    row_profile=None,
    hyphenator=None,
    measure=None,
)->List[Line]:

    align_right = blk.fontformat.alignment == TextAlignment.Right

    ox, oy = origin
    bh, bw = mask.shape[:2]
    num_words = len(words)
    blk_rect = blk.bounding_rect()
    blk_width = blk_rect[2]
    lines = []
    words_length = sum(wl_list)

    lh_pad = 0
    if blk.line_spacing > 1:
        lh_pad = int(np.ceil(line_height - line_height / blk.line_spacing))

    if num_words > 0:
        sum_right = np.array(wl_list).sum()
        w, wl = words.pop(0), wl_list.pop(0)
        line = Line(w, ox, oy, wl)
        lines.append(line)
        sum_right -= wl
        line_bottom = oy + line_height
        pos_y = oy
        line_id = 0
        while sum_right > 0:
            w, wl = words.pop(0), wl_list.pop(0)
            sum_right -= wl
            new_len = line.length + wl + delimiter_len
            if align_right:
                new_x = ox + blk_width - new_len - line_height // 2
            else:
                new_x = ox + new_len + line_height // 2
            line_valid = False
            if new_x < bw and new_x > 0:
                # A zero-height probe means this line sits outside the mask
                # window: oy comes from the block's first line in absolute
                # page coords, so a block whose text starts above the window
                # crop drives both clip bounds to 0. .mean() of that is nan
                # and nan > 240 is False, which already reads as "no ink
                # here"; keep that verdict but stop numpy warning on it.
                _probe = mask[np.clip(pos_y, 0, bh - 1): np.clip(line_bottom - lh_pad, 0, bh), new_x]
                if _probe.size and _probe.mean() > 240:
                    line_valid = True
                else:
                    if ref_src_lines and line_id + 1 >= len(srcline_wlist) and line_is_valid(line, new_len, delimiter_len, max_width, words_length, srcline_wlist, line_id, line_height, ref_src_lines, row_profile=row_profile):
                        line_valid = True
            if line_valid:
                line_valid = line_is_valid(line, new_len, delimiter_len, max_width, words_length, srcline_wlist, line_id, line_height, ref_src_lines, row_profile=row_profile)
                if hyphenator is not None and measure is not None and new_len > max_width:
                    split = _hyphen_head_for_line(line, w, hyphenator, measure, delimiter_len, line_id, max_width, words_length, srcline_wlist, line_height, ref_src_lines, row_profile)
                    if split is not None:
                        head, hw, tail, tw = split
                        line.append_right(head, hw + delimiter_len, delimiter)
                        w, wl = tail, tw
                        line_valid = False
            if line_valid:
                line.append_right(w, wl+delimiter_len, delimiter)
            else:
                pos_y = line_bottom
                line_bottom += line_height
                line = Line(w, ox, pos_y, wl)
                line_id += 1
                lines.append(line)
    return lines, (0, 0)



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
    spacing: int = 0,
    max_central_width=np.inf,
    src_is_cjk=False,
    tgt_is_cjk=False,
    ref_src_lines = False,
    row_profile=None,
    hyphenator=None,
    measure=None
) -> Tuple[str, List]:

    angle = blk.angle
    alignment = blk.alignment

    start_from_top = False
    srcline_wlist = None

    if ref_src_lines:
        srcline_wlist, srcline_width = blk.normalizd_width_list(normalize=False)
        # tgtline_width = sum(wl_list) + delimiter_len * max(len(wl_list) - 1, 0)
        # if tgtline_width < srcline_width:
        #     min_bbox = blk.min_rect(rotate_back=True)[0]
        #     x1, y1 = min_bbox[0]
        #     x2, y2 = min_bbox[2]
        #     w = x2 - x1
        #     max_central_width = min(max_central_width, w)
        #     pass

        if alignment == TextAlignment.Center and \
        len(srcline_wlist) > 1:
            if len(srcline_wlist) == 2:
                start_from_top = True
            else:
                nw = len(srcline_wlist)
                # nl = min(nw // 2, 2)
                nl = 1
                sum_top = sum(srcline_wlist[:nl])
                sum_btn = sum(srcline_wlist[-nl:])
                start_from_top = sum_top / sum_btn > 1.2 and srcline_wlist[0] / max(srcline_wlist) > 0.9

        srcline_wlist = np.array(srcline_wlist) / srcline_width
        srcline_wlist = srcline_wlist.tolist()
        # line_height = min((blk.detected_font_size), line_height)

    # if ref_src_lines:
    #     mask = np.ones_like(mask) * 255

    if max_central_width == np.inf:
        max_central_width = mask.shape[1]

    centroid_x, centroid_y = centroid
    center_x = mask_xyxy[0] + centroid_x
    center_y = mask_xyxy[1] + centroid_y
    shifted_x, shifted_y = 0, 0
    if abs(angle) > 0:

        old_h, old_w = mask.shape[:2]
        old_origin = (old_w // 2, old_h // 2)
        rel_cx, rel_cy = centroid[0] - old_origin[0], centroid[1] - old_origin[1]
        
        mask = rotate_image(mask, angle)
        rad = np.deg2rad(angle)
        r_sin, r_cos = np.sin(rad), np.cos(rad)
        new_rel_cy =  -rel_cx * r_sin + rel_cy * r_cos
        new_rel_cx =  rel_cy * r_sin + rel_cx * r_cos

        shifted_x, shifted_y = new_rel_cx - rel_cx, new_rel_cy - rel_cy
        
        new_h, new_w = mask.shape[:2]
        new_origin = (new_w // 2, new_h // 2)
        new_cx, new_cy = new_origin[0] + new_rel_cx, new_origin[1] + new_rel_cy
        centroid = [int(new_cx), int(new_cy)]

    if alignment == TextAlignment.Center:
        lines, adjust_xy = layout_lines_aligncenter(blk, mask, words, centroid, wl_list, delimiter_len, line_height, spacing, delimiter, 
                                         max_central_width, ref_src_lines=ref_src_lines, srcline_wlist=srcline_wlist,
                                         start_from_top=start_from_top, row_profile=row_profile,
                                         hyphenator=hyphenator, measure=measure)
    else:
        lines, adjust_xy = layout_lines_alignside(blk, mask, words, centroid, wl_list, delimiter_len, line_height, spacing, delimiter, False, max_central_width, 
                                       ref_src_lines=ref_src_lines, srcline_wlist=srcline_wlist, row_profile=row_profile,
                                       hyphenator=hyphenator, measure=measure)
    
    concated_text = []
    pos_x_lst, pos_right_lst = [], []
    for line in lines:
        pos_x_lst.append(line.pos_x)
        pos_right_lst.append(max(line.pos_x, 0) + line.length)
        concated_text.append(_join_hyphen_runs(line.text))
    concated_text = '\n'.join(concated_text)

    pos_x_lst = np.array(pos_x_lst)
    pos_right_lst = np.array(pos_right_lst)
    canvas_l, canvas_r = pos_x_lst.min(), pos_right_lst.max()
    canvas_t, canvas_b = lines[0].pos_y, lines[-1].pos_y + line_height

    canvas_h = int(canvas_b - canvas_t)
    canvas_w = int(canvas_r - canvas_l)

    if alignment == 1:
        abs_x = int(round(center_x - canvas_w / 2))
        abs_y = int(round(center_y - canvas_h / 2))
    elif abs(angle) > 0:
        abs_x = shifted_x
        abs_y = shifted_y
    else:
        # canvas_l/canvas_t are mask-relative; return window coordinates to
        # match the center path so callers can test placement against the mask.
        abs_x = int(canvas_l + mask_xyxy[0])
        abs_y = int(canvas_t + mask_xyxy[1])

    return concated_text, [abs_x, abs_y, canvas_w, canvas_h], start_from_top, adjust_xy


def _join_hyphen_runs(line_text: str) -> str:
    """Reassemble hyphen-split segments that landed on one line.

    The wrap treats the pieces of one word as separate words, so it joins
    them with spaces and marks every piece it broke with ``_HYPHEN_BREAK``
    (the examples spell it as an escape). Only a mark is swallowed:
    interior pieces merge silently, a mark left at line end becomes a real
    hyphen, and a '-' the translator typed is left alone.

    >>> _join_hyphen_runs("it's im\x1e pos\x1e")
    "it's impos-"
    >>> _join_hyphen_runs('im\x1e pos\x1e sible~~~!')
    'impossible~~~!'
    >>> _join_hyphen_runs('possible~~~!')
    'possible~~~!'
    >>> _join_hyphen_runs("fine- I'll go")
    "fine- I'll go"
    """
    merged = []
    for token in line_text.split(' '):
        if merged and merged[-1].endswith(_HYPHEN_BREAK):
            merged[-1] = merged[-1][:-1] + token
        else:
            merged.append(token)
    if merged and merged[-1].endswith(_HYPHEN_BREAK):
        # The line ends here, so the break is a real line-ending hyphen.
        merged[-1] = merged[-1][:-1] + '-'
    return ' '.join(merged)


def _widest_fitting_prefix(word: str, max_width: float, measure, min_tail: int) -> Optional[int]:
    r"""Largest split point whose head (plus a hyphen) fits the line.

    Character fallback for tokens pyphen has no point for - URLs, sound
    effects, foreign runs. Returns None when not even the shortest head
    fits, so the caller can keep the token whole and let it overflow.

    >>> _widest_fitting_prefix('hahahahaha', 60, lambda s: len(s) * 10, 2)
    5
    >>> _widest_fitting_prefix('ab', 60, lambda s: len(s) * 10, 2) is None
    True
    """
    lo, hi, best = 1, len(word) - min_tail, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if measure(word[:mid] + '-') <= max_width:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def hyphenate_long_words(
    words: List[str],
    wl_list: List[int],
    measure,
    lang: str,
    max_width: int,
) -> Tuple[List[str], List[int]]:
    r"""Split Latin-script tokens wider than the line budget to hyphenation
    points, falling back to a character break when pyphen has none.

    Candidates are advisory: the caller derives them per layout attempt from
    its pristine word list and tries the unhyphenated layout first, so a
    break that was not needed never survives into a passing layout. Tokens
    shorter than six characters stay whole, every break keeps at least three
    characters on each side, and a word is cut at most
    ``_MAX_CUTS_PER_WORD`` times. Those are all typographic minimums: a
    shorter word split 2+2 leaves a fragment that reads as a typo, a word cut
    until only a stub is left ("fre- quentl- y.") is worse still, and a third
    cut turns one long word into a hyphen the reader has to read past twice.
    A word that cannot be cut cleanly stays whole and overhangs its line.

    A token pyphen cannot split at a linguistic point (a URL, a sound
    effect, a foreign run) is force-broken at the widest prefix that fits,
    as often as it takes: such a token has no typography to protect, only a
    width to meet. Letting it stay whole overflows the line and pins the fit
    small, so a tall balloon's spare height goes unused; the break lets
    balloon typesetting grow the font to fill it. The forced head carries the
    same internal break marker as a pyphen head, so it is hidden on a shared
    line and hyphenated only at a real line end.

    Unknown languages fall back through pyphen's language mapping; without
    pyphen the input passes through unchanged. ``measure`` maps a string to
    its pixel width in the current layout font.

    >>> hyphenate_long_words(['OK'], [30], lambda s: len(s) * 10, 'en', 60)
    (['OK'], [30])
    >>> hyphenate_long_words(['extraordinary'], [110], lambda s: len(s) * 10, 'en', 60)
    (['extra\x1e', 'ordin\x1e', 'ary'], [60, 60, 30])
    >>> hyphenate_long_words(['mother'], [60], lambda s: len(s) * 10, 'en', 20)
    (['mother'], [60])
    >>> hyphenate_long_words(['onto'], [400], lambda s: len(s) * 10, 'en', 60)
    (['onto'], [400])
    >>> hyphenate_long_words(['What?'], [400], lambda s: len(s) * 10, 'en', 60)
    (['What?'], [400])
    """
    if not words or max_width <= 0:
        return words, wl_list
    try:
        import pyphen
        dic = pyphen.Pyphen(lang=pyphen.language_fallback(lang or 'en'))
    except Exception:
        return words, wl_list
    out_words, out_wl = [], []
    for word, width in zip(words, wl_list):
        # See _MIN_HYPHEN_WORD: a short word stays whole and simply shrinks,
        # and so does a run of words seg_eng glued into one token - there is
        # no legal break point inside "sure to".
        if len(word) < _MIN_HYPHEN_WORD or ' ' in word:
            out_words.append(word)
            out_wl.append(width)
            continue
        # Cut the word into as many segments as it takes to fit the line.
        # Segments that land on one line are joined back together at render
        # (see _join_hyphen_runs); only a segment that actually ends the
        # line shows its hyphen, so the extra cuts cost nothing visually.
        # A cut that would leave a piece under _MIN_PIECE is refused: the
        # word then stays whole and overhangs its line by whatever is left,
        # which reads far better than a stub on the page.
        cuts = 0
        while width > max_width and len(word) >= 2 * _MIN_PIECE:
            positions = [
                p for p in dic.positions(word)
                if _MIN_PIECE <= p <= len(word) - _MIN_PIECE
            ]
            split = None
            for p in positions:
                head_w = measure(word[:p] + '-')
                if head_w <= max_width:
                    split = p  # widest head that still fits a line
            if split is None:
                # pyphen has no usable point (URLs, sound effects, foreign
                # runs). Keeping the token whole lets it overflow the line
                # and pins the fit small: balloon typesetting grows toward
                # height by breaking whatever blocks its width, so a run
                # pyphen cannot split would leave a tall balloon's spare
                # axis unused. Force a character break instead; the head
                # carries _HYPHEN_BREAK, so _join_hyphen_runs hides it on a
                # shared line and shows a hyphen only at a real line end.
                forced = _widest_fitting_prefix(word, max_width, measure, _MIN_PIECE)
                # _widest_fitting_prefix already guarantees the tail; the
                # head is the remaining side. Nothing splits this word into
                # two readable pieces, so leave it whole and let the line
                # overhang rather than print a stub.
                if forced is None or forced < _MIN_PIECE:
                    break  # not even one glyph fits: leave it to overflow
                out_words.append(word[:forced] + _HYPHEN_BREAK)
                out_wl.append(measure(word[:forced] + '-'))
                word = word[forced:]
                width = measure(word)
                # Not counted against _MAX_CUTS_PER_WORD: such a token has no
                # typography to protect, only a width to meet.
                continue
            if cuts >= _MAX_CUTS_PER_WORD:
                # A linguistic word stops after _MAX_CUTS_PER_WORD cuts. What
                # is left is closed by the line-breaker's own in-line
                # hyphenation, and a further cut buys room at the price of
                # another hyphen the reader has to read past.
                break
            out_words.append(word[:split] + _HYPHEN_BREAK)
            out_wl.append(measure(word[:split] + '-'))
            word = word[split:]
            width = measure(word)
            cuts += 1
        out_words.append(word)
        out_wl.append(width)
    return out_words, out_wl


def split_run_tokens(
    words: List[str],
    wl_list: List[int],
    measure,
    max_width: int,
) -> Tuple[List[str], List[int]]:
    r"""Give a run of glued words its spaces back when the run is too wide.

    ``seg_eng`` glues a one or two letter word to its neighbour ("me" + "like"
    + "a" become the single token "me like a") so a short word never sits
    alone on a line. The wrap treats a token as indivisible, so once such a
    run is wider than the line budget it is the run, not the longest word in
    it, that sets the line width - and with it the font the fit may use. A
    tall narrow balloon then gets a short stack of wide lines with most of its
    height unused, which is the block looking like a rectangle dropped into
    an ellipse.

    Only runs wider than the budget are split: a run that fits keeps the glue
    that stops "a" from being orphaned, which is the whole point of it.
    Splitting at the space is an ordinary word break, so nothing is
    hyphenated here. ``measure`` maps a string to its pixel width in the
    current layout font.

    >>> split_run_tokens(['And', 'me like a', 'toddler...'],
    ...                  [30, 114, 89], lambda s: len(s) * 10, 91)
    (['And', 'me', 'like', 'a', 'toddler...'], [30, 20, 40, 10, 89])
    >>> split_run_tokens(['OK'], [20], lambda s: len(s) * 10, 91)
    (['OK'], [20])
    """
    if not words or max_width <= 0:
        return words, wl_list
    out_words, out_wl = [], []
    for word, width in zip(words, wl_list):
        if ' ' not in word or width <= max_width:
            out_words.append(word)
            out_wl.append(width)
            continue
        for part in word.split(' '):
            out_words.append(part)
            out_wl.append(measure(part))
    return out_words, out_wl
