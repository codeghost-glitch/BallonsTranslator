"""KoharuLayout-RFDETR-Seg-2XL-1152 custom text detector.

Model: https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152
"""
import logging
import os
import warnings
from typing import List, Tuple

import cv2
import numpy as np

from ballontranslator.modules.textdetector.base import (
    DEVICE_SELECTOR,
    ProjImgTrans,
    TextBlock,
    TextDetectorBase,
    register_textdetectors,
)
from ballontranslator.utils.imgproc_utils import xywh2xyxypoly
from ballontranslator.utils.textblock import (
    examine_textblk,
    mit_merge_textlines,
    sort_pnts,
    sort_regions,
)

MODEL_PATH = 'data/models/koharu_layout/model.safetensors'
CLASS_NAMES = ['text', 'onomatopoeia', 'bubble', 'panel']


class _QuietRFDETRNoise(logging.Filter):
    """Drop rfdetr warnings that are always true for this integration.

    The backbone intentionally rebuilds position embeddings for the 1152/12
    grid with patch size 12 (hence the two dinov2 warnings), and
    optimize_for_inference() is skipped because it torch.jit.traces a 1152
    model whose control flow the tracer cannot follow cleanly.
    """

    _PREFIXES = (
        'Using a different number of positional encodings',
        'Using patch size',
        'Model is not optimized for inference',
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._PREFIXES)


@register_textdetectors('koharu_layout')
class KoharuLayoutDetector(TextDetectorBase):
    """RF-DETR Seg 2XL manga layout detector (text + onomatopoeia classes).

    Example:
        >>> detector = KoharuLayoutDetector()
        >>> detector.name
        'koharu_layout'
    """

    # rfdetr >= 1.6 requires transformers>=5, but the app's OCR/translator/inpaint
    # modules pin transformers==4.57.6 in the same environment; 1.5.2 is the last
    # release that accepts transformers 4.x.
    dependencies = ['torch', 'rfdetr==1.5.2', 'safetensors>=0.5']

    download_file_list = [
        {
            'url': 'https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/resolve/main/model.safetensors',
            'files': MODEL_PATH,
            'sha256_pre_calculated': '9bf6d2cbd7793c956d8c857bb1672a396eb7f100eb0682f86830d05e31168efb',
        }
    ]

    params = {
        'text threshold': {
            'type': 'line_editor', 'value': 0.25, 'display_name': 'Text Threshold',
            'description': 'Confidence threshold for text (model card recommends 0.25).',
        },
        'onomatopoeia threshold': {
            'type': 'line_editor', 'value': 0.20, 'display_name': 'Onomatopoeia Threshold',
            'description': 'Confidence threshold for SFX (model card recommends 0.20; raise to 0.40 for precision).',
        },
        'label': {
            'value': {'text': True, 'onomatopoeia': True},
            'type': 'check_group',
            'display_name': 'Labels',
        },
        'merge text lines': {
            'type': 'checkbox', 'value': True, 'display_name': 'Merge Text Lines',
        },
        'font size multiplier': {
            'type': 'line_editor', 'value': 1., 'display_name': 'Font Size Multiplier',
        },
        'font size max': {
            'type': 'line_editor', 'value': -1, 'display_name': 'Font Size Max',
        },
        'font size min': {
            'type': 'line_editor', 'value': -1, 'display_name': 'Font Size Min',
        },
        'mask dilate size': {
            # Glyph-shaped masks grow ~radially: ksize 2 merges dense text and
            # drove measured precision down to 0.52 on dense pages; 1 keeps
            # recall ~0.96 with precision ~0.89 (ink-based per-instance check).
            'type': 'line_editor', 'value': 1, 'display_name': 'Mask Dilate Size',
        },
        'device': {**DEVICE_SELECTOR(), 'display_name': 'Device'},
    }

    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None

    def _load_model(self):
        # Mirrors the hub's load_model.py strict loader; rfdetr is pinned because
        # the constructor/state-dict layout is API-sensitive.

        # Albumentations runs a network update check (and warns) when rfdetr
        # imports it; NO_ALBUMENTATIONS_UPDATE is its documented opt-out.
        os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
        from rfdetr import RFDETRSeg2XLarge
        from safetensors.torch import load_file

        rfdetr_logger = logging.getLogger('rf-detr')
        if not any(isinstance(f, _QuietRFDETRNoise) for f in rfdetr_logger.filters):
            rfdetr_logger.addFilter(_QuietRFDETRNoise())

        with warnings.catch_warnings():
            try:
                # Warning category only exists in rfdetr >= 1.7.
                from rfdetr.config import PretrainWeightsCompatibilityWarning
                warnings.simplefilter('ignore', PretrainWeightsCompatibilityWarning)
            except ImportError:
                pass
            model = RFDETRSeg2XLarge(
                pretrain_weights=None,
                resolution=1152,
                # rfdetr 1.5.2 keeps the position-embedding grid per variant
                # (64 -> 768 px); the checkpoint was trained at 1152/12 = 96.
                positional_encoding_size=96,
                num_select=160,
                num_classes=len(CLASS_NAMES),
                device=self.get_param_value('device'),
            )
        incompatible = model.model.model.load_state_dict(load_file(MODEL_PATH, device='cpu'), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'Incompatible koharu-layout weights: {incompatible}')
        model.model.class_names = CLASS_NAMES.copy()
        self.model = model

    def get_valid_labels(self) -> set:
        return {k for k, v in self.params['label']['value'].items() if v}

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        im_h, im_w = img.shape[:2]
        mask = np.zeros((im_h, im_w), dtype=np.uint8)

        # Only text-bearing classes feed the mask/blocks; bubble and panel
        # detections are dropped because no bubble association stage consumes them.
        class_thresholds = {
            cid: float(self.get_param_value(f'{name} threshold'))
            for cid, name in ((0, 'text'), (1, 'onomatopoeia'))
            if name in self.get_valid_labels()
        }
        if not class_thresholds:
            return mask, []

        # Run at the lowest class threshold, then filter per class as the model
        # card instructs; rfdetr 1.5.2 resizes to the constructor's resolution
        # (1152) and returns masks at source resolution.
        dets = self.model.predict(img, threshold=min(class_thresholds.values()))
        ksize = max(int(self.get_param_value('mask dilate size')), 0)

        detected_items = []
        if dets is not None and len(dets) > 0:
            masks = dets.mask
            for i, (cls_id, conf) in enumerate(zip(dets.class_id, dets.confidence)):
                cls_id = int(cls_id)
                if cls_id not in class_thresholds or conf < class_thresholds[cls_id]:
                    continue
                x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, im_w), min(y2, im_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                det_mask = masks[i].astype(bool) if masks is not None else None
                if det_mask is not None:
                    mask[det_mask] = 255
                else:
                    mask[y1:y2, x1:x2] = 255
                # The block box must contain every mask pixel the pipeline later
                # zeroes when it drops an untranslatable block; the mask head can
                # spill outside the box head, and the final dilation grows it
                # another ksize pixels. Stray fragments left outside the box get
                # inpainted as artifacts.
                if det_mask is not None:
                    ys, xs = np.nonzero(det_mask)
                    if ys.size:
                        x1 = min(x1, int(xs.min()))
                        y1 = min(y1, int(ys.min()))
                        x2 = max(x2, int(xs.max()) + 1)
                        y2 = max(y2, int(ys.max()) + 1)
                x1, y1 = max(x1 - ksize, 0), max(y1 - ksize, 0)
                x2, y2 = min(x2 + ksize, im_w), min(y2 + ksize, im_h)
                pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
                detected_items.append({'pts': pts, 'label': CLASS_NAMES[cls_id]})

        blk_list = []
        if not detected_items:
            return mask, blk_list
        if self.get_param_value('merge text lines'):
            pts_only_list = [item['pts'] for item in detected_items]
            blk_list = mit_merge_textlines(pts_only_list, width=im_w, height=im_h)
        else:
            for item in detected_items:
                pts_sorted, is_vertical = sort_pnts(item['pts'])
                blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=item['label'])
                blk.vertical = is_vertical
                blk.adjust_bbox()
                examine_textblk(blk, im_w, im_h)
                blk_list.append(blk)

        blk_list = sort_regions(blk_list)

        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
            mask = cv2.dilate(mask, element)

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            # RF-DETR records its device at construction, so drop the resident
            # model and let the next run rebuild it on the new device.
            self.unload_model()
