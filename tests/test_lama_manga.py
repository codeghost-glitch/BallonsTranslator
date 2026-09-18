import os
import unittest
from unittest.mock import patch

import numpy as np

from ballontranslator.modules.inpaint.base import INPAINTERS
from ballontranslator.modules.inpaint.inpaint_default import LamaManga

WEIGHTS = r'data/models/lama_manga.safetensors'
needs_weights = unittest.skipUnless(
    os.path.isfile(WEIGHTS), 'mayocream lama-manga weights not downloaded',
)


class LamaMangaRegistrationTests(unittest.TestCase):
    def test_registered_under_lama_manga_key(self) -> None:
        self.assertIs(INPAINTERS.module_dict['lama_manga'], LamaManga)
        self.assertEqual(LamaManga().name, 'lama_manga')

    def test_lazy_spec_is_complete(self) -> None:
        from ballontranslator.modules.lazy_registry import validate_lazy_module_specs
        spec = INPAINTERS.get_spec('lama_manga')
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.params)
        self.assertEqual(validate_lazy_module_specs([spec]), [])

    def test_params_surface_mayocream_checkpoint(self) -> None:
        params = LamaManga().params
        self.assertEqual(params['inpaint_size']['value'], 1536)
        self.assertIn('precision', params)
        self.assertIn('mayocream/lama-manga', params['description'])

    def test_download_points_at_mayocream_safetensors(self) -> None:
        files = LamaManga.download_file_list
        self.assertEqual(len(files), 1)
        self.assertEqual(
            files[0]['url'],
            'https://huggingface.co/mayocream/lama-manga/resolve/main/lama-manga.safetensors',
        )
        self.assertEqual(files[0]['files'], WEIGHTS)

    def test_missing_weights_raise_helpful_error(self) -> None:
        inpainter = LamaManga()
        with patch(
            'ballontranslator.modules.inpaint.inpaint_default.osp.isfile',
            return_value=False,
        ):
            with self.assertRaisesRegex(FileNotFoundError, 'lama_manga.safetensors'):
                inpainter._load_model()


class LamaMangaPreprocessTests(unittest.TestCase):
    def test_preprocess_pads_to_64_aligned_square(self) -> None:
        import torch
        from ballontranslator.modules.inpaint.lama import LamaFourier
        inpainter = LamaManga()
        # Position encoding is pure numpy; random init is enough for shapes.
        inpainter.model = LamaFourier(
            build_discriminator=False, use_mpe=False, large_arch=True,
        )
        img = np.full((100, 150, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 150), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        out = inpainter.inpaint_preprocess(img, mask)
        img_torch, mask_torch = out[0], out[1]
        self.assertIsInstance(img_torch, torch.Tensor)
        self.assertEqual(tuple(img_torch.shape), (1, 3, 192, 192))
        self.assertEqual(tuple(mask_torch.shape), (1, 1, 192, 192))
        self.assertTrue(bool(((mask_torch == 0) | (mask_torch == 1)).all()))
        self.assertLessEqual(float(img_torch.max()), 1.0)
        self.assertGreaterEqual(float(img_torch.min()), 0.0)


class LamaMangaWeightsTests(unittest.TestCase):
    @needs_weights
    def test_generator_forward_shape_and_range(self) -> None:
        import torch
        from ballontranslator.modules.inpaint.lama import load_lama_manga
        model = load_lama_manga(WEIGHTS, 'cpu')
        image = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        with torch.no_grad():
            output = model(image, mask)
        self.assertEqual(tuple(output.shape), (1, 3, 64, 64))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertLessEqual(float(output.max()), 1.0)
        self.assertGreaterEqual(float(output.min()), 0.0)

    @needs_weights
    def test_inpaint_keeps_unmasked_pixels(self) -> None:
        inpainter = LamaManga()
        inpainter._load_model()
        rng = np.random.RandomState(7)
        img = rng.randint(0, 256, size=(128, 128, 3)).astype(np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[48:80, 48:80] = 255
        result = inpainter._inpaint(img, mask)
        self.assertEqual(result.shape, img.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(bool((result == img)[mask == 0].all()))
        self.assertTrue(bool((result.astype(int) != img.astype(int))[48:80, 48:80].any()))


if __name__ == '__main__':
    unittest.main()
