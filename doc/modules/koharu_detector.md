# Koharu detector

Select `koharu` in the text detector settings. It uses the
[KoharuLayout RF-DETR Seg 2XL model](https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152)
directly in Python.

The detector owns a memory-bounded RF-DETR postprocessor: it filters the top 160
query/class candidates using the selected thresholds, then resizes one mask
at a time. This avoids expanding rejected candidates into page-sized float
tensors. Accepted masks retain RF-DETR's bilinear interpolation and zero-logit
threshold. CPU inference at 1152 px can still be slow.

The loader enables RF-DETR's uncompiled inference optimization (which retains
an additional model copy) and explicitly passes the backbone's `return_dict`
setting for Transformers 5 compatibility. Expected stock-backbone initialization
warnings are filtered only while constructing the custom-checkpoint model;
strict checkpoint loading still reports missing or incompatible weights.
Hayai loads its processor from cache first and filters the misleading
Transformers tensor-sharding warning during its single-device model load.
Other warnings and errors remain visible.

The pipeline logs detection, OCR, and inpainting start/finish timings. A
missing completion message identifies the last active stage, but is not by
itself proof of a native crash. The block inpainting path preserves its input
mask, skips already processed crops, and processes residual masked regions
outside text-block crops. Manual and pipeline runs still differ in crop
context and the optional uniform-background shortcut.

The existing module preparation flow checks optional `rfdetr==1.7.0`,
`safetensors>=0.5`, `transformers>=5.1.0,<6.0.0`, PyTorch, and torchvision dependencies and downloads the
checksum-verified weights to `data/models/koharu/model.safetensors` when the
module is first run. Merely selecting the detector in settings does not load
models or download files. CUDA is recommended; CPU inference is supported but
slow. Headless runs use the same detector and module preparation path.

RF-DETR 1.7 requires Transformers 5; an existing Transformers 4 installation
must be updated before inference. Modules that require Transformers 4 cannot
share that dependency version. Hayai OCR v2 supports both Transformers 4.57.6
and 5.x, so it can run alongside the Koharu detector without downgrading Transformers.
Restart Ballons after changing installed Transformers versions so Python does
not retain the previous version in memory. The plugin does not modify installed packages
on import or selection; use the existing dependency setup deliberately.

Text regions become ordinary Ballons `TextBlock` objects. The inpainting mask
combines their instance segmentation masks, with configurable dilation that
also closes letter gaps and holes up to twice the radius without growing the
mask further.
**Fill bubble text boxes** is enabled by default to avoid leftover letter
fragments: it masks the whole detected bubble interior, which also covers
furigana and ruby that sit above or beside the text rectangle. It does not
fill free text or sound-effect rectangles, and it only fills detected bubble
masks. It can also erase artwork inside bubbles, so inspect the mask and
disable it when inpainting illustrated bubbles if needed.
Existing saved choices are preserved.
Rerun detection on the original page to regenerate the mask after changing
detector settings; rerunning inpainting alone reuses the saved mask.
Sound effects are opt-in and have a separate confidence threshold. Overlapping
same-label duplicates are suppressed with per-label non-maximum suppression at
IoU 0.5. Bubbles
and panels are excluded from both text blocks and erasure masks. Each text
region retains the outline of the smallest detected bubble whose mask contains
at least 90% of the text region's segmentation mask; the bubble confidence
threshold is independently configurable. Joined outlines are partitioned at
physical necks before merging text runs. Runs sharing a lobe and writing mode
merge into one block, ordered right-to-left for vertical text; smooth balloons
keep one flow. Each block retains its lobe outline for fitting and center
guides, keeping side-lobe dialogue separate from the main lobe.
For an enclosed white balloon, the saved outline is refined against the raw
page interior only when it strongly overlaps the segmentation. Open borders,
dark interiors, and mismatched regions keep the detector outline; no ellipse
or convex hull is forced. Existing saved outlines require detection to be rerun.
See [bubble typesetting](../ui/text_layout.md#detected-bubbles-and-hyphenation)
for fitting and highlight behavior. Source text
orientation defaults to vertical Japanese manga text, and each region's
writing direction is inferred from its segmentation mask, falling back to the
configured default when ambiguous.
The detector provides region boxes rather than individual text lines, so initial
font sizes are estimates. Continue using a separate OCR and translator.

The loader follows the model author's strict SafeTensors loading contract at
revision `aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2`, without downloading or
executing a remote Python loader. Model license and training-data terms are
documented in the linked model card.

Verify conversion, mask filtering, bubble association, lazy metadata, and failure handling with
`python -m unittest discover -s tests -p test_koharu_detector.py`.
Real model validation should include a Japanese page with dialogue and sound
effects, checking masks before inpainting with sound effects disabled/enabled.
