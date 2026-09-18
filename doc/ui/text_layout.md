# Text layout

Read [Text engine](text_engine.md) first. This guide records the behavior and
ownership shared by shaping, wrapping, vertical flow, painting, and editing.
Implementation-specific algorithms belong in code comments and focused tests.

## Mental model

```text
QTextDocument rich text (Qt UTF-16 positions)
  -> SceneTextLayout fragment metrics
  -> horizontal lines or vertical columns
  -> settled placement
  -> fill, effects, annotations, cursor, selection, and hit testing
  -> TextItemGeometryController bounds and visual mapping
```

Qt remains the editable text model and shaper. The custom layouts place Qt
`QTextLine`s; they do not create a second text representation.

## Core contract

- `FontFormat` supplies item-wide writing mode, alignment, and compatibility
  defaults. `QTextDocument` formats own range-bound typography and
  paragraph-bound line spacing.
- Placement records, ink bounds, and caches are derived. Rebuild them together
  for one settled layout generation and never persist them.
- Fill, effects, annotations, cursor, selection, hit testing, and visual bounds
  must consume the same settled cells and transforms.
- Qt positions are UTF-16 code units. Use the shared UTF-16 and grapheme helpers
  wherever Python strings meet Qt positions; never expose a caret inside a
  surrogate pair or combined run.
- Effect padding and visible ink overflow belong to source geometry, not the
  persistent logical rectangle.

`TextBlock.text_layout_version` versions item-wide layout semantics. Missing or
version-zero vertical blocks migrate to right alignment, matching their earlier
effective placement. Inline HTML extensions remain versionless and follow the
compatibility rules in [Text engine](text_engine.md).

## Writing modes

### Horizontal

`HorizontalTextDocumentLayout` keeps Qt shaping, glyph runs, cursor behavior,
and word-boundary wrapping. It adds only the geometry Qt does not expose in the
form the editor needs. In particular, overflowing trailing U+0020 spaces stay
in the document but receive derived continuation-row cells so wrapping, box
growth, cursor, selection, and hit testing agree. Other Unicode separators keep
Qt behavior.

Character spacing and font features are applied per range. Identity spacing is
left unset when common ligatures should remain available because an explicit Qt
spacing property may suppress optional ligatures. Version-specific feature-tag
handling stays inside the layout/annotation boundary.

### Vertical

`VerticalTextDocumentLayout` normally creates one cell per grapheme and places
columns from right to left. Punctuation orientation and alignment are semantic
classes near the top of `vertical_layout.py`; extend those classes instead of
adding paint-time glyph exceptions.

Standard Roman mode keeps proportional Roman glyphs upright and centered. The
alternate mode rotates them clockwise and uses the Chinese mixed-layout
punctuation path. Compact punctuation shortens eligible punctuation cells
without clipping their ink. Repeated dashes, bars, leaders, and ellipses form
indivisible runs, with character spacing applied after the run.

Tate-chu-yoko is a horizontal Qt run occupying one vertical flow cell. Its
layout ignores authored letter spacing and uses the font's half-width
punctuation plus matching half-, third-, or quarter-width feature when
available. Standard Roman mode keeps that shaped run's natural horizontal
width; the alternate mode horizontally scales any remaining excess to one em.
The resulting visible ink is centered without changing the stored text. Glyph
ink may overhang the column, but that overhang affects only painting and
interaction bounds, never neighboring columns.

Ruby/furigana is attached layout content, not a detached overlay. Group Ruby is
indivisible; mono Ruby may wrap only between base/reading pairs. Each unit uses
the larger of its base and annotation advances, and the shorter run is spaced
within that cell. Horizontal Ruby appears above or below; vertical Ruby remains
upright on the right or left. The same cells own wrapping, paint, selection,
cursor, hit testing, effects, and visible bounds. Ruby and tate-chu-yoko cannot
overlap, and automatic Ruby overhang is not supported.

## Flow and spacing

Whitespace remains document content and must consume explicit editable cells.
Horizontal and vertical layouts may represent those cells differently, but
neither may move whitespace into a second text model or drop it from cursor and
hit geometry. Vertical whitespace contributes flow advance, not the ink bounds
used to center the neighboring glyph in its column.

Character spacing is a trailing advance for the affected glyph or joined run;
W3C tate-chu-yoko composition ignores it. On a squeezed single-column vertical
item, increasing it may grow the logical height to preserve that column;
multi-column items keep normal fixed-area reflow, and automatic growth never
silently shrinks the box.

Line spacing is owned by the destination row or column. The first visual row or
column stays anchored without leading spacing; each later one uses its
paragraph's spacing value and mode. Paragraph boundaries do not restart this
visual leading-edge rule.

Settle a layout as one transaction. Wrapping, whitespace, annotations,
fragment metrics, UTF-16 positions, ink bounds, and interaction geometry are
coupled and must be published only when complete.

## Alignment and resize

Vertical alignment translates settled columns horizontally:

| Alignment | Fixed growth anchor | Added-width movement |
| --- | --- | --- |
| Left | Top-left | Columns stay fixed and grow rightward |
| Center | Top-center | Columns move by half and grow evenly |
| Right | Top-right | Columns move with the right edge and grow leftward |

Alignment changes every placement and ink-bound record together but does not
reshape text or change document content. A width-only resize may reuse that
translation when the settled content still fits; height, padding, or flow
changes require full layout. The geometry controller preserves the matching
scene-space anchor, so layout and scene movement must not both compensate for
the same resize.

## Detected bubbles and hyphenation

`TextBlock.bubble_polygon` stores an optional page-space bubble outline.
`utils/bubble.py` owns validation and interior geometry: old projects default
to no outline, malformed optional data is discarded with a warning, and
invalid live data fails at serialization. Outlines stay anchored to the source
page when text moves; pasted blocks discard the source bubble association.
[Koharu detection](../modules/koharu_detector.md) supplies these outlines.

Auto layout and the pipeline always fit text through
`text_engine/bubble_layout.py`. It finds the maximal inset rectangle inside the
detected shape with a dynamic inset (scales with the bubble, capped in absolute
pixels, taller top/bottom like the guide), avoiding tails and concavities.
Short dialogue retries against two content-shaped interiors — a rectangle
shaped like the fitted lines and one shaped like the outline itself (both
with a small uniform margin) — keeping whichever fits at the larger size, so
readable size is not pinned by the interior that favors tall rectangles;
vertical text and shared-outline siblings keep their existing rectangles.
The fitted box is then anchored to the outline centroid (the same center the
guide and drag snap use) whenever the shifted box still stays inside the
physical outline, so automatic layout and "center in bubble" agree.
`utils/autolayout.py` (balanced lines, linguistic penalties that avoid
stranding articles, one shared visual axis) over the full interior width,
baked into the document like the mask fallback path (which shares the
same engine over the balloon mask or outline contour with tapered profiles
to hug the balloon, and keeps authored
paragraph breaks as mandatory breaks); Qt measurement
stays the ground truth for the font-size search, which probes largest-first
(balloon fits are non-monotonic) and refines. Fitted text uses Center
alignment, keeps the interior width for stable wrapping, and centers
vertically so top/bottom and left/right margins match. This is rectangular
interior fitting, not per-line contour
wrapping. The font grows or shrinks to the largest fitting size rather
than being capped by the detector's source-text estimate. The search reserves
effect padding for stroked ink. Hyphenation during fitting is a last resort:
the clean search runs first and soft hyphens are only considered when it
pins to (or misses) the readable floor, so mid-word
splits never trade readability for size. Dash compounds split after interior
dashes (`EHH—`/`BUT`) with no dictionary and render literally, and lines
never start with a dash (`Hayate-`/`kun`, never `Hayate`/`-kun`). Growth caps
a little above the detected source size instead of filling bubbles, while
shrinking is unaffected. Breaks avoid stranding closing
punctuation on its own line when joining it still fits. The mask fallback
path centers like the bubble path. Missing
geometry, rotated text, and transformed text retain the previous
layout path; vertical text breaks into columns with the same DP transposed.
Blocks sharing one connected outline first try a neck split so each
uses its own lobe interior (a `HEY!! YU!!` header stays in the top lobe instead
of the body's box); without a clean neck the shared outline is partitioned by
perpendicular bisectors between the sibling centers, clipped to the physical
contour (koharu-style Voronoi fallback), with coincident siblings subdivided
into strips, so siblings stay inside the shared outline without overlapping
each other; shared outlines are still indexed once per layout batch. When even minimum size overflows, the text
stays centered at minimum size inside the bubble instead of falling back to a
mask layout outside of it. Manual fitting and hyphenation use canvas undo,
including alignment, and keep the paired editor in sync on undo/redo.

The canvas **Hyphenate text** action inserts U+00AD discretionary breaks
using the target-language Pyphen
dictionary. Pyphen is optional; a missing package or unsupported language logs
a warning and leaves text unchanged. Dictionary use is local and never
downloads or installs packages. Qt shows a hyphen only at a line break;
document content retains soft hyphens through editing and serialization.
Insertion preserves inline formats and UTF-16 positions, avoids annotated
ruby/combined runs and nonstandard spelling-changing breaks, and is idempotent.
Hyphenation uses canvas undo and keeps the paired editor in sync on undo/redo.

The canvas context menu's **Highlight detected bubbles** toggle shows dashed outlines
associated with live text blocks. `CustomGV` caches shared paths by outline,
updates them on item attachment/removal, and paints them in the view foreground.
They are page-space guides and never enter scene image exports or inpainting
masks. Bubbles without an associated text block are not retained as guides.

Focused verification: `tests/test_bubble_typesetting.py` covers geometry,
project recovery, fitting, hyphenation, undo, and overlay export/lifetime;
`tests/test_koharu_detector.py` covers detection association and mask exclusion.

## Painting and interaction

`vertical_line_placement()` is the shared boundary for rotated glyphs,
tate-chu-yoko, emphasis, Glyph Slant, and effects. Cursor, selection, and hit
testing must use the same placement. Ligatures and joined glyphs may change
shaping, but they do not change the logical UTF-16 editing range.

Document backgrounds paint below selection, and glyph ink paints above it.
Foreground and effect layouts must reuse the same settled offsets. Caches tied
to placement must be invalidated with the layout generation and must not retain
records from a replaced document layout.

## Invalidation and verification

The normal path is:

```text
document or format change
  -> rebuild fragment metrics and position maps
  -> settle lines or columns
  -> update draw offsets and ink bounds
  -> publish size and refresh geometry/effects
```

Test relationships rather than exact font-dependent pixels. Cover the affected
writing modes, alignments, spacing, annotations, effects, UTF-16 text, editing,
resize, and mode switches. Focused coverage lives in:

- `tests/test_horizontal_whitespace.py`
- `tests/test_vertical_alignment.py`
- `tests/test_vertical_interaction.py`
- `tests/test_vertical_roman_alignment.py`
- `tests/test_rich_text_annotations.py`
- `tests/test_ruby_furigana.py`

Run both PyQt5 and PyQt6 when layout lifetime, shaping, cursor geometry, or
painting behavior changes.
