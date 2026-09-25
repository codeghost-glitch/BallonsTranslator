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
- Passive project loading is permissive: malformed records log a warning, the
  invalid value alone is dropped, and the rest of the page keeps loading.
- Each koharu detect run clears then rewrites the current page's outlines, so
  disabling the `bubble` label removes stale data on the next run.
- The layer is decorative: it ignores mouse buttons, sits between the drawing
  and text layers, and refreshes from `updateCanvas()` — page switches and
  pipeline finish need no extra wiring. Headless mode has no canvas; outlines
  still persist in the project.

## Enable

Detector params → Labels → `bubble` checkbox (default off) plus
Bubble Threshold (model-card default 0.5).

## Verification

`tests/test_bubble_outlines.py` covers validation, write boundaries, and the
offscreen canvas layer. `tests/test_koharu_layout.py` runs the real model and
checks outlines stay well-formed, inside the page, and clear when disabled.
