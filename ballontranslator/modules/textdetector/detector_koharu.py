from pathlib import Path
import warnings
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .base import DEVICE_SELECTOR, ProjImgTrans, TextBlock, TextDetectorBase, register_textdetectors
from ballontranslator.utils.textblock import examine_textblk, sort_regions
from ballontranslator.utils.bubble import bubble_partition_cells, bubble_polygon_from_mask
from ballontranslator.utils.logger import suppress_model_warnings

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


def _explicit_return_dict(
    module: 'Module', args: tuple, kwargs: dict,
) -> Tuple[tuple, dict]:
    if kwargs.get('return_dict') is None:
        kwargs['return_dict'] = module.config.return_dict
    return args, kwargs

def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    """IoU of two ``xyxy`` boxes. Non-finite coordinates never suppress: every
    comparison against NaN is False, so such boxes stay candidates and are
    rejected later by the explicit finite-value checks in :meth:`_detect`."""
    intersection_width = float(min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = float(min(left[3], right[3]) - max(left[1], right[1]))
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return 0.0
    intersection = intersection_width * intersection_height
    left_area = max(float(left[2] - left[0]), 0.0) * max(float(left[3] - left[1]), 0.0)
    right_area = max(float(right[2] - right[0]), 0.0) * max(float(right[3] - right[1]), 0.0)
    return intersection / (left_area + right_area - intersection)

def _containment_duplicate(
    first: int, second: int, xyxy: np.ndarray, masks: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """Return ``(finer, coarser)`` when one same-label box is a merged
    duplicate that already contains the other's glyphs.

    RF-DETR sometimes emits one coarse instance spanning several text columns
    alongside the per-column instances (e.g. page 15: a union box over two
    vertical scream columns). Box IoU cannot see this (a union of two columns
    has ~0.4 IoU with either column), so the pair is a duplicate when the
    smaller box sits inside the bigger one and its mask pixels are already
    covered by the bigger instance's mask. A 0.2 area-ratio floor keeps small
    legitimate instances (furigana, short captions) from evicting the region
    that contains them.

    >>> union = np.zeros((12, 20), dtype=bool); union[1:11, 1:19] = True
    >>> column = np.zeros((12, 20), dtype=bool); column[1:11, 1:9] = True
    >>> _containment_duplicate(
    ...     0, 1, np.array([[1., 1., 19., 11.], [1., 1., 9., 11.]]),
    ...     np.stack([union, column]))
    (1, 0)
    >>> _containment_duplicate(
    ...     0, 1, np.array([[1., 1., 19., 11.], [6., 6., 9., 11.]]),
    ...     np.stack([union, column])) is None
    True
    """
    first_area = _box_area(xyxy[first])
    second_area = _box_area(xyxy[second])
    inner, outer = (
        (second, first) if second_area < first_area else (first, second)
    )
    inner_area = min(first_area, second_area)
    outer_area = max(first_area, second_area)
    if inner_area <= 0.0 or outer_area <= 0.0:
        return None
    if inner_area < 0.2 * outer_area:
        return None
    intersection_width = float(
        min(xyxy[inner][2], xyxy[outer][2]) - max(xyxy[inner][0], xyxy[outer][0])
    )
    intersection_height = float(
        min(xyxy[inner][3], xyxy[outer][3]) - max(xyxy[inner][1], xyxy[outer][1])
    )
    if intersection_width <= 0.0 or intersection_height <= 0.0:
        return None
    if intersection_width * intersection_height / inner_area < 0.9:
        return None
    inner_mask = masks[inner].astype(bool)
    inner_pixels = int(inner_mask.sum())
    if not inner_pixels:
        return None
    overlap = int((inner_mask & masks[outer].astype(bool)).sum())
    if overlap / inner_pixels < 0.9:
        return None
    return (inner, outer)

def _box_area(box: np.ndarray) -> float:
    width = max(float(box[2]) - float(box[0]), 0.0)
    height = max(float(box[3]) - float(box[1]), 0.0)
    return width * height

def _non_maximum_suppression(
    xyxy: np.ndarray, class_id: np.ndarray, confidence: np.ndarray,
    masks: Optional[np.ndarray] = None, iou_threshold: float = 0.5,
) -> List[int]:
    """Greedy per-label box NMS; returns kept indices, best score first.

    Dense pages make the detector emit overlapping duplicates, and only
    same-label pairs at or above the IoU threshold are considered duplicates
    (a bubble and a panel may legitimately share a box). Containment
    duplicates also collapse when masks are available: of a coarse/fine pair
    where the fine box sits inside the coarse one with the same glyphs, the
    finer instance is kept so per-column text is not doubled by a merged
    detection.

    >>> _non_maximum_suppression(
    ...     np.array([[0., 0., 10., 10.], [1., 1., 10., 10.]]),
    ...     np.array([0, 0]), np.array([0.9, 0.8]))
    [0]
    >>> _non_maximum_suppression(
    ...     np.array([[0., 0., 10., 10.], [1., 1., 10., 10.]]),
    ...     np.array([0, 1]), np.array([0.9, 0.8]))
    [0, 1]
    >>> union = np.zeros((12, 20), dtype=bool); union[1:11, 1:19] = True
    >>> column = np.zeros((12, 20), dtype=bool); column[1:11, 1:9] = True
    >>> _non_maximum_suppression(
    ...     np.array([[1., 1., 19., 11.], [1., 1., 9., 11.]]),
    ...     np.array([0, 0]), np.array([0.9, 0.8]),
    ...     masks=np.stack([union, column]))
    [1]
    """
    order = np.argsort(-confidence, kind='stable')
    kept: List[int] = []
    for index in order:
        if any(
            class_id[other] == class_id[index]
            and _intersection_over_union(xyxy[other], xyxy[index]) >= iou_threshold
            for other in kept
        ):
            continue
        if masks is not None:
            # A surviving candidate may itself be the coarser duplicate of an
            # already-kept finer instance; evict the coarse one so only the
            # finer instance of those glyphs remains.
            suppressed = False
            for other in list(kept):
                if class_id[other] != class_id[index]:
                    continue
                duplicate = _containment_duplicate(index, other, xyxy, masks)
                if duplicate is None:
                    continue
                if duplicate[1] == other:
                    kept.remove(other)
                else:
                    # The candidate is the coarser duplicate of a kept finer
                    # instance.
                    suppressed = True
                    break
            if not suppressed:
                kept.append(int(index))
            continue
        kept.append(int(index))
    return kept

def _infer_vertical(mask: np.ndarray) -> Optional[bool]:
    """Infer the writing mode of one instance mask from projection profiles.

    Text lines are separated by near-empty bands along the line-stacking axis:
    vertical manga columns stack along x, horizontal lines along y. The axis
    whose profile concentrates its energy in fewer bins crosses those bands,
    so a sharper per-column profile means the lines stack along x, i.e. the
    text is vertical, from projection profiles at angle zero. Ties within 2%
    (including whole-block squares and rotated sound
    effects) are ambiguous so the caller can fall back to the configured
    default instead of guessing.

    >>> vertical = np.zeros((60, 120), dtype=np.uint8)
    >>> vertical[10:50, 20:36] = 1
    >>> vertical[10:50, 80:96] = 1
    >>> _infer_vertical(vertical)
    True
    >>> horizontal = np.zeros((60, 120), dtype=np.uint8)
    >>> horizontal[20:36, 10:110] = 1
    >>> _infer_vertical(horizontal)
    False
    >>> square = np.zeros((40, 40), dtype=np.uint8)
    >>> square[10:30, 10:30] = 1
    >>> _infer_vertical(square) is None
    True
    >>> _infer_vertical(np.zeros((10, 10), dtype=np.uint8)) is None
    True
    """
    columns = mask.sum(axis=0, dtype=np.float64)
    rows = mask.sum(axis=1, dtype=np.float64)
    pixels = columns.sum()
    if pixels <= 0:
        return None
    column_energy = float(np.square(columns).sum()) / pixels
    row_energy = float(np.square(rows).sum()) / pixels
    strongest = max(column_energy, row_energy)
    if abs(column_energy - row_energy) <= strongest * 0.02:
        return None
    return column_energy > row_energy


def _estimate_font_size(text_mask: np.ndarray, vertical: bool, fallback: int) -> int:
    """Estimate the glyph size from a text run's segmentation mask.

    Koharu emits one bbox per text run, so ``min(w, h)`` equals a
    multi-line run's column width, not the glyph size, and translations
    rendered several times larger than the source text. Project the mask
    onto the axis the lines stack along: each line or glyph forms an ink
    band whose width is the glyph size, and the median band shrugs off
    furigana and punctuation. ``fallback`` (the bbox minimum dimension)
    keeps the old ceiling for empty or blob-like masks.

    >>> mask = np.zeros((40, 100), dtype=bool)
    >>> mask[10:30, 5:25] = True
    >>> mask[10:30, 55:75] = True
    >>> _estimate_font_size(mask, True, 95)
    20
    """
    ink = text_mask > 0
    if ink.any():
        profile = ink.sum(axis=0 if vertical else 1)
        bands: List[int] = []
        run = 0
        for value in profile:
            if value > 0:
                run += 1
            elif run:
                bands.append(run)
                run = 0
        if run:
            bands.append(run)
        if bands:
            estimate = int(round(float(np.median(bands))))
            if estimate >= 3:
                return min(estimate, fallback)
    return fallback

def _merge_balloon_blocks(
    blocks: List[TextBlock], width: int, height: int,
) -> List[TextBlock]:
    """Merge text runs that share one balloon into one block per flow.

    RF-DETR can emit several runs for one dialogue. Split joined outlines at
    physical necks first, then merge runs within each lobe. Each output keeps
    its lobe outline so fitting and center guides use that lobe, not the whole
    joined balloon. Smooth single-lobe balloons keep one translation flow.

    Only text blocks carrying the same bubble polygon merge, and only when
    they share a writing mode; sound effects, panels, and free text are never
    combined, so unrelated neighbours stay independent.

    >>> free = TextBlock(xyxy=[0, 0, 10, 10], src_is_vertical=True)
    >>> _merge_balloon_blocks([free], 100, 100)[0] is free
    True
    """
    grouped: Dict[Tuple[Tuple[float, float], ...], List[TextBlock]] = {}
    merged: List[TextBlock] = []
    for block in blocks:
        if block.bubble_polygon is None:
            merged.append(block)
            continue
        grouped.setdefault(tuple(tuple(point) for point in block.bubble_polygon), []).append(block)
    for polygon_key, members in grouped.items():
        if len(members) == 1 or len({bool(block.src_is_vertical) for block in members}) > 1:
            merged.extend(members)
            continue
        polygon = [list(point) for point in polygon_key]
        centers = [
            (
                float(block.xyxy[0] + block.xyxy[2]) / 2,
                float(block.xyxy[1] + block.xyxy[3]) / 2,
            )
            for block in members
        ]
        cells = bubble_partition_cells(polygon, centers)
        if cells is None:
            # Single-lobe balloon: one dialogue flow, so every run merges.
            merged.append(_merge_block_group(members, width, height, polygon))
            continue
        # Joined balloon: runs sharing a lobe merge; the lobe becomes each
        # block's outline so fit and the center guide use the lobe, not the
        # whole two-lobe blob.
        lobes: Dict[Tuple[Tuple[float, float], ...], List[TextBlock]] = {}
        for block, cell in zip(members, cells):
            lobes.setdefault(tuple(tuple(point) for point in cell), []).append(block)
        for cell, lobe_members in lobes.items():
            merged.append(
                _merge_block_group(lobe_members, width, height, [list(point) for point in cell])
            )
    return merged

def _merge_block_group(
    members: List[TextBlock], width: int, height: int,
    polygon: List[List[float]],
) -> TextBlock:
    """Collapse one run group into a single block ordered for reading."""
    vertical = bool(members[0].src_is_vertical)
    members.sort(key=lambda block: block.xyxy[0], reverse=vertical)
    primary = members[0]
    primary.lines = [line for block in members for line in block.lines]
    examine_textblk(primary, width, height, sort=True)
    # Member sizes are mask-band estimates now; the minimum is the safest
    # glyph size for a group whose merged lines can hold multi-column runs.
    primary.font_size = primary._detected_font_size = min(
        block.detected_font_size for block in members
    )
    primary.adjust_bbox()
    primary.bubble_polygon = polygon
    if vertical:
        primary.alignment = 1
    else:
        primary.recalulate_alignment()
    return primary

class KoharuPostProcessor:
    """Filter native candidates before projecting one mask at a time to the page.

    ``num_select`` bounds the candidates that can reach mask projection. The
    container postprocessor is replaced, so the cap is owned here instead of
    being configured twice through the container.

    >>> processor = KoharuPostProcessor()
    >>> processor.num_select, processor.thresholds
    (160, {0: 0.25, 2: 0.5})
    """

    def __init__(self, num_select: int = 160) -> None:
        self.num_select = num_select
        self.thresholds = {0: 0.25, 2: 0.5}

    def __call__(
        self, outputs: Dict[str, 'Tensor'], target_sizes: 'Tensor',
    ) -> List[Dict[str, 'Tensor']]:
        import torch
        from torch.nn.functional import interpolate

        logits = outputs['pred_logits']
        probabilities = logits.sigmoid().flatten(1)
        scores, indices = probabilities.topk(min(self.num_select, probabilities.shape[1]), dim=1)
        results = []
        for batch_index, target in enumerate(target_sizes):
            labels = indices[batch_index] % logits.shape[-1]
            keep = torch.zeros_like(labels, dtype=torch.bool)
            for class_id, threshold in self.thresholds.items():
                keep |= (labels == class_id) & (scores[batch_index] >= threshold)
            # Mirrors the container filter that runs right after this one, so masks
            # dropped by the page-level threshold are never resized.
            keep &= scores[batch_index] > min(self.thresholds.values())
            queries = indices[batch_index][keep] // logits.shape[-1]
            boxes = outputs['pred_boxes'][batch_index, queries]
            centers, half_sizes = boxes[:, :2], boxes[:, 2:] / 2
            boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=1)
            height, width = (int(value) for value in target.tolist())
            boxes *= boxes.new_tensor([width, height, width, height])
            masks = torch.empty((len(queries), 1, height, width), dtype=torch.bool, device=logits.device)
            for output_index, query in enumerate(queries):
                native = outputs['pred_masks'][batch_index, query][None, None]
                masks[output_index] = interpolate(
                    native, size=(height, width), mode='bilinear', align_corners=False,
                )[0] > 0
            results.append({
                'scores': scores[batch_index][keep], 'labels': labels[keep],
                'boxes': boxes, 'masks': masks,
            })
        return results


@register_textdetectors('koharu')
class KoharuDetector(TextDetectorBase):
    """KoharuLayout RF-DETR Seg 2XL text detection and segmentation.

    >>> detector = KoharuDetector()
    >>> detector.name, detector.all_model_loaded()
    ('koharu', False)
    """

    dependencies = [
        'torch>=2.2.0', 'torchvision>=0.17.0', 'rfdetr==1.7.0',
        'safetensors>=0.5', 'transformers>=5.1.0,<6.0.0',
    ]
    params = {
        'description': 'KoharuLayout RF-DETR Seg 2XL (1152 px). Detects text with '
                       'segmentation masks; optionally includes sound effects. '
                       'CUDA is recommended. Model: mayocream/koharu-layout-rfdetr-seg-2xl-1152.',
        'device': DEVICE_SELECTOR(not_supported=['privateuseone']),
        'text threshold': {
            'type': 'line_editor', 'value': 0.25,
            'description': 'Minimum text confidence (0.0-1.0).',
        },
        'detect sound effects': {'type': 'checkbox', 'value': False},
        'bubble threshold': {
            'type': 'line_editor', 'value': 0.50,
            'description': 'Minimum bubble confidence for typesetting geometry (0.0-1.0).',
        },
        'sound effect threshold': {
            'type': 'line_editor', 'value': 0.20,
            'description': 'Minimum sound effect confidence (0.0-1.0).',
        },
        'source text is vertical': {'type': 'checkbox', 'value': True},
        'auto detect text orientation': {
            'type': 'checkbox', 'value': True,
            'description': 'Infer per-region vertical or horizontal writing from '
                           'its segmentation mask. Ambiguous regions fall back to '
                           'the "source text is vertical" setting.',
        },
        'fill bubble text boxes': {
            'type': 'checkbox', 'value': True,
            'description': 'Mask the whole detected bubble interior when segmentation '
                           'leaves letter fragments. May remove artwork inside the '
                           'bubble. Free text and sound effects keep their segmentation '
                           'masks.',
        },
        'mask dilate size': {
            'type': 'line_editor', 'value': 2,
            'description': 'Text mask dilation radius in source-image pixels (0-100). '
                           'Letter gaps and holes up to twice the radius are also '
                           'closed without growing the mask further.',
        },
    }
    _load_model_keys = {'model'}
    download_file_list = [{
        'url': 'https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/'
               'resolve/aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2/model.safetensors',
        'files': 'data/models/koharu/model.safetensors',
        'sha256_pre_calculated': '9bf6d2cbd7793c956d8c857bb1672a396eb7f100eb0682f86830d05e31168efb',
    }]

    def _load_model(self) -> None:
        weights = Path('data/models/koharu/model.safetensors')
        if not weights.is_file():
            raise FileNotFoundError(
                f'Koharu weights are missing: {weights}. Run the selected module setup '
                'or download model.safetensors from '
                'mayocream/koharu-layout-rfdetr-seg-2xl-1152 to this path.'
            )
        try:
            from rfdetr import RFDETRSeg2XLarge
            from rfdetr.config import PretrainWeightsCompatibilityWarning
            from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersBackbone
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                'Koharu requires rfdetr==1.7.0, safetensors>=0.5, and '
                'transformers>=5.1.0,<6.0.0. Check the module dependency setup. '
                f'Backend import failed: {error}'
            ) from error

        with warnings.catch_warnings(), suppress_model_warnings(
            'rf-detr', (
                'Using a different number of positional encodings than DINOv2,',
                'Using patch size 12 instead of 14,',
            ),
        ):
            warnings.filterwarnings('ignore', category=PretrainWeightsCompatibilityWarning)
            model = RFDETRSeg2XLarge(
                pretrain_weights=None,
                resolution=1152,
                num_classes=4,
                device=self.get_param_value('device'),
            )
        model.model.model.load_state_dict(load_file(str(weights), device='cpu'), strict=True)
        for module in model.model.model.modules():
            if isinstance(module, WindowedDinov2WithRegistersBackbone):
                module.register_forward_pre_hook(_explicit_return_dict, with_kwargs=True)
        model.optimize_for_inference(compile=False)
        model.model.class_names = ['text', 'onomatopoeia', 'bubble', 'panel']
        # The container postprocessor is replaced right below, so the candidate cap
        # lives with KoharuPostProcessor instead of the container configuration.
        model.model.postprocess = KoharuPostProcessor()
        self.model = model

    def _detect(
        self, img: np.ndarray, proj: Optional[ProjImgTrans] = None,
    ) -> Tuple[np.ndarray, List[TextBlock]]:
        if img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) == 0:
            raise ValueError('Koharu requires a non-empty RGB image.')
        thresholds = {}
        for class_id, param_key in (
            (0, 'text threshold'), (1, 'sound effect threshold'), (2, 'bubble threshold'),
        ):
            threshold = float(self.get_param_value(param_key))
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f'{param_key} must be between 0 and 1.')
            thresholds[class_id] = threshold
        dilation = float(self.get_param_value('mask dilate size'))
        if not 0 <= dilation <= 100 or not dilation.is_integer():
            raise ValueError('mask dilate size must be an integer between 0 and 100.')
        if not self.get_param_value('detect sound effects'):
            thresholds.pop(1)

        self.model.model.postprocess.thresholds = thresholds.copy()
        detections = self.model.predict(
            img, threshold=min(thresholds.values()), shape=(1152, 1152),
            include_source_image=False,
        )
        if len(detections.xyxy) > 1:
            kept = _non_maximum_suppression(
                detections.xyxy, detections.class_id, detections.confidence,
                masks=detections.mask,
            )
            if len(kept) != len(detections.xyxy):
                detections = detections[np.asarray(kept, dtype=int)]
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        blocks = []
        bubbles = []
        source_gray = None
        for index, (class_id, score) in enumerate(zip(detections.class_id, detections.confidence)):
            if class_id != 2 or not np.isfinite(score) or score < thresholds[2]:
                continue
            if detections.mask is None or detections.mask[index].shape != (height, width):
                self.logger.warning('Koharu bubble has no valid segmentation mask; ignoring its geometry.')
                continue
            if source_gray is None:
                source_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            polygon = bubble_polygon_from_mask(detections.mask[index], source_gray)
            if polygon is not None:
                bubbles.append((int(detections.mask[index].sum()), polygon, index))
        bubbles.sort(key=lambda bubble: bubble[0])
        thresholds.pop(2)
        auto_vertical = bool(self.get_param_value('auto detect text orientation'))
        default_vertical = bool(self.get_param_value('source text is vertical'))
        for index, (box, class_id, score) in enumerate(zip(
            detections.xyxy, detections.class_id, detections.confidence,
        )):
            if class_id not in thresholds or not np.isfinite(score) or score < thresholds[class_id]:
                continue
            if not np.isfinite(box).all():
                # One malformed RF-DETR query must not lose every other
                # detection on the page; skip just this candidate.
                self.logger.warning('Koharu returned a non-finite box; ignoring it.')
                continue
            left, top = np.floor(np.clip(box[:2], [0, 0], [width, height])).astype(int).tolist()
            right, bottom = np.ceil(np.clip(box[2:], [0, 0], [width, height])).astype(int).tolist()
            if right <= left or bottom <= top:
                continue
            if detections.mask is None or detections.mask[index].shape != (height, width):
                raise ValueError('Koharu returned a missing or incorrectly sized text segmentation mask.')
            text_mask = detections.mask[index].astype(bool)
            mask[text_mask] = 255
            if auto_vertical:
                vertical = _infer_vertical(text_mask[top:bottom, left:right])
            else:
                vertical = None
            if vertical is None:
                vertical = default_vertical
            block = TextBlock(
                xyxy=[left, top, right, bottom],
                lines=[[[left, top], [right, top], [right, bottom], [left, bottom]]],
                src_is_vertical=vertical,
                label='text' if class_id == 0 else 'onomatopoeia',
            )
            block.vertical = vertical
            # Link text to the smallest bubble whose mask
            # contains at least 90% of the text mask. A bbox-center test would
            # also attach free text that merely overlaps a bubble, and the
            # assigned polygon then drives destructive bubble-box filling.
            if class_id == 0:
                text_window = text_mask[top:bottom, left:right]
                text_pixels = int(text_window.sum())
                if text_pixels:
                    for _, polygon, bubble_index in bubbles:
                        bubble_window = detections.mask[bubble_index][top:bottom, left:right].astype(bool)
                        if int((text_window & bubble_window).sum()) / text_pixels >= 0.9:
                            block.bubble_polygon = polygon
                            if self.get_param_value('fill bubble text boxes'):
                                # Furigana and ruby sit above/beside the text
                                # rectangle but still inside the balloon;
                                # filling only the text-rect slice leaves
                                # that ink for the inpainter to preserve.
                                # Fill the whole detected interior instead;
                                # the option warns it may remove artwork.
                                mask[detections.mask[bubble_index].astype(bool)] = 255
            # The mask-band estimate keeps multi-line runs at the real
            # glyph size; the bbox minimum dimension alone equals a
            # multi-column run's full width.
            fallback = min(right - left, bottom - top)
            detected_size = _estimate_font_size(
                text_mask[top:bottom, left:right], vertical, fallback,
            )
            block.font_size = block._detected_font_size = detected_size
            blocks.append(block)

        blocks = _merge_balloon_blocks(blocks, img.shape[1], img.shape[0])

        if dilation and blocks:
            radius = int(dilation)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            # Closing the page mask (two dilations, one erosion) keeps the
            # same ~radius expansion while also filling glyph holes and
            # inter-letter gaps, which measurably improves inpainting mask IoU.
            mask = cv2.morphologyEx(cv2.dilate(mask, kernel), cv2.MORPH_CLOSE, kernel)
        return mask, sort_regions(blocks)

    def updateParam(self, param_key: str, param_content: object) -> None:
        previous = self.get_param_value(param_key)
        super().updateParam(param_key, param_content)
        if param_key == 'device' and self.get_param_value(param_key) != previous:
            self.unload_model()
