# Bubble outlines

Canvas overlay that shows speech balloons detected by the text detector. The
koharu-layout custom detector populates it; any detector can honor the same
contract.

## Ownership

| Concern | Owner | File |
| --- | --- | --- |
| Capture: instance-mask contour (box fallback) | detector module | [`custom_modules/detector_koharu_layout.py`](../../custom_modules/detector_koharu_layout.py) |
| Persistence, validation | `ProjImgTrans`, per-page `image_info['bubble_outlines']` | [`utils/proj_imgtrans.py`](../../ballontranslator/utils/proj_imgtrans.py) |
| Rendering | `Canvas.bubbleOutlineLayer`, rebuilt by `updateCanvas()` | [`ui/canvas.py`](../../ballontranslator/ui/canvas.py) |

## Contract

- Shape: list of polygons, each polygon a list of at least three `[x, y]`
  page-pixel points. Only `set_bubble_outlines()` writes; `get_bubble_outlines()`
  returns a detached copy.
- Capture is keyed by `begin_detection()`'s page (`detecting_page`), with
  `current_img` as the fallback for direct calls: the pipeline detects pages
  without switching the viewer's page, so keying by `current_img` writes
  every page's outlines onto whichever page was open.
- Passive project loading is permissive: malformed records log a warning, the
  invalid value alone is dropped, and the rest of the page keeps loading.
- Each koharu detect run clears then rewrites the current page's outlines, so
  disabling the `bubble` label removes stale data on the next run.
- The layer is decorative: it ignores mouse buttons, sits between the drawing
  and text layers, and refreshes from `updateCanvas()` — page switches and
  pipeline finish need no extra wiring. Headless mode has no canvas; outlines
  still persist in the project.
- Auto layout consumes outlines as balloon geometry (attribution, collision,
  elliptical typesetting); that contract lives in
  [Text engine](text_engine.md#auto-layout-fit).

## Enable

Detector params → Labels → `bubble` checkbox (default off) plus
Bubble Threshold (model-card default 0.5).

## Verification

`tests/test_bubble_outlines.py` covers validation, write boundaries, and the
offscreen canvas layer. `tests/test_koharu_layout.py` runs the real model and
checks outlines stay well-formed, inside the page, and clear when disabled.
