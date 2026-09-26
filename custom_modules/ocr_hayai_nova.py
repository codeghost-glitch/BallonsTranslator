"""Hayai OCR v2.5 Nova custom OCR module.

Model: https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova
"""
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModel,
    PreTrainedTokenizerFast,
    Siglip2ImageProcessor,
)

from ballontranslator.modules.ocr.base import DEVICE_SELECTOR, OCRBase, register_OCR


MODEL_PATH = "data/models/hayai-ocr-v25-nova"
VISION_MODEL_ID = "google/siglip2-base-patch16-naflex"


class _QuietHayaiLoadNoise(logging.Filter):
    """Drop third-party load-time warnings that hold on every load.

    - urllib3 retries when Hugging Face advertises HTTP/3 it cannot negotiate;
      the HTTP/1 fallback succeeds.
    - tokenizer_config.json declares "TokenizersBackend" while this module
      loads through PreTrainedTokenizerFast (class-name skew, works either way).
    - transformers verifies tensor-parallel plans even for models that define
      none, so it always reports every layer as unsharded here.
    - the 226-class IDS head is trained but unused at inference, so its
      checkpoint weights are always reported as not used.
    """

    _SUBSTRINGS = (
        "MustDowngradeError",
        "The tokenizer class you load from this checkpoint",
        "The following layers were not sharded",
        "were not used when initializing HayaiModel",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(s in record.getMessage() for s in self._SUBSTRINGS)


_QUIET_LOGGERS = (
    "urllib3.connectionpool",
    "transformers.tokenization_utils_base",
    "transformers.integrations.tensor_parallel",
    "transformers.modeling_utils",
)


@register_OCR("hayai_ocr_v25_nova")
class HayaiOCRV25Nova(OCRBase):
    """Crop-level Hayai OCR v2.5 Nova backend.

    v2.5 replaces the v2 MLP projector with a DSCProjector that packs 4 patches
    into 1 decoder token, so a 384-patch budget feeds the decoder 96 vision
    tokens instead of 384.

    >>> HayaiOCRV25Nova.params["max_num_patches"]["value"]
    384
    """

    dependencies = ["torch", "transformers==4.57.6"]
    params = {
        "max_num_patches": {
            "type": "selector",
            "options": [256, 384, 512],
            "value": 384,
            "display_name": "Max Num Patches",
            "description": (
                "Maximum image patches. 256 for throughput, 384 for balanced "
                "quality, 512 for small or stylized glyphs."
            ),
        },
        "device": DEVICE_SELECTOR(),
        "description": "Hayai OCR v2.5 Nova crop recognition model.",
    }
    download_file_list = [{
        "url": (
            "https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova/"
            "resolve/e34d7755ed11e626c5ba39544af5d66f20ee57cc/"
        ),
        "save_dir": MODEL_PATH,
        "files": [
            "config.json",
            "configuration_hayai.py",
            "model.safetensors",
            "modeling_hayai.py",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "sha256_pre_calculated": [
            "880c99360c77733032fe06e94de8123dca21610cab782c6df43e4e26c9d37c41",
            "47abd38cf1bae7aef27d01f5b8b4aa0960a7bc625a8afad79c4762ff5e5ed970",
            "ac63cd177c5b68bd91e18dd4408e0400461c7cf19f484b0da1643e232d1ffc3a",
            "a6be06426c77db2f521928a792a057d13ab17fe477b4df6f1b88861b2fc4da36",
            "f8a0a909c628a684fe463094614e236a8b1d3609e7770f77e7beafaf1056bf13",
            "6fb6c69afaedf1275872d3e62e276fd4467bd00da7a84cbbb5566a2cd28f58f6",
        ],
        "concatenate_url_filename": 1,
    }]
    _load_model_keys = {"model", "processor", "tokenizer"}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model = None
        self.processor = None
        self.tokenizer = None

    @property
    def device(self) -> str:
        return self.get_param_value("device")

    @property
    def max_num_patches(self) -> int:
        value = self.get_param_value("max_num_patches")
        if value <= 0:
            raise ValueError("max_num_patches must be greater than zero")
        return value

    def _load_model(self) -> None:
        for logger_name in _QUIET_LOGGERS:
            target = logging.getLogger(logger_name)
            if not any(isinstance(f, _QuietHayaiLoadNoise) for f in target.filters):
                target.addFilter(_QuietHayaiLoadNoise())

        processor = Siglip2ImageProcessor.from_pretrained(VISION_MODEL_ID)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
        # Hayai defines custom block-causal attention and 2D mRoPE classes.
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
        ).to(self.device).eval()

        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model

    def ocr_img(self, img: np.ndarray, **kwargs: Any) -> str:
        image = Image.fromarray(img).convert("RGB")
        inputs = self.processor(
            images=[image],
            max_num_patches=self.max_num_patches,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            texts = self.model.generate(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
                tokenizer=self.tokenizer,
                max_new_tokens=128,
                # CJK relies on legitimate repetition; the model card requires 1.0.
                repetition_penalty=1.0,
            )
        return texts[0]

    def updateParam(self, param_key: str, param_content: Any) -> None:
        if (
            param_key == "device"
            and self.model is not None
            and self.device != param_content
        ):
            self.model.to(param_content)
        super().updateParam(param_key, param_content)
