from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.lazy_registry import _scan_file, validate_lazy_module_specs
from ballontranslator.modules.textdetector.detector_koharu import KoharuDetector, KoharuPostProcessor

class _SuppressibleDetections(SimpleNamespace):
    """Duck-typed supervision.Detections that supports integer-array selection."""

    def __getitem__(self, index):
        index = np.asarray(index)
        return _SuppressibleDetections(
            xyxy=self.xyxy[index], class_id=self.class_id[index],
            confidence=self.confidence[index], mask=self.mask[index],
        )


class KoharuDetectorTests(unittest.TestCase):
    def test_postprocess_filters_candidates_before_resizing_masks_one_at_a_time(self) -> None:
        import torch
        from torch.nn.functional import interpolate

        logits = torch.full((1, 3, 5), -20.0)
        logits[0, 0, 0] = 9
        logits[0, 1, 2] = 8
        logits[0, 2, 3] = 7
        native_masks = torch.tensor([[[[-1., 1.], [1., -1.]], [[1., -1.], [-1., 1.]], [[1., 1.], [1., 1.]]]])
        outputs = {
            'pred_logits': logits,
            'pred_boxes': torch.tensor([[[.5, .5, .4, .6]]]).expand(1, 3, 4),
            'pred_masks': native_masks,
        }
        processor = KoharuPostProcessor()
        with patch('torch.nn.functional.interpolate', wraps=interpolate) as resize:
            result = processor(outputs, torch.tensor([[20, 30]]))[0]
        self.assertEqual(result['labels'].tolist(), [0, 2])
        self.assertEqual(resize.call_count, 2)
        self.assertTrue(all(call.args[0].shape[:2] == (1, 1) for call in resize.call_args_list))
        expected_masks = interpolate(native_masks[0, :2, None], size=(20, 30), mode='bilinear', align_corners=False) > 0
        torch.testing.assert_close(result['masks'], expected_masks)
        torch.testing.assert_close(result['boxes'], torch.tensor([[9., 4., 21., 16.]]).expand(2, 4))
        torch.testing.assert_close(result['scores'], torch.tensor([9., 8.]).sigmoid())

    def test_postprocess_empty_result_does_not_resize_masks(self) -> None:
        import torch

        processor = KoharuPostProcessor()
        with patch('torch.nn.functional.interpolate') as resize:
            result = processor({
                'pred_logits': torch.full((1, 3, 5), -20.0),
                'pred_boxes': torch.zeros((1, 3, 4)),
                'pred_masks': torch.zeros((1, 3, 2, 2)),
            }, torch.tensor([[20, 30]]))[0]
        self.assertEqual(tuple(result['masks'].shape), (0, 1, 20, 30))
        resize.assert_not_called()

    def setUp(self) -> None:
        self.params_patch = patch.object(KoharuDetector, 'params', deepcopy(KoharuDetector.params))
        self.params_patch.start()
        self.addCleanup(self.params_patch.stop)
        self.detector = KoharuDetector()
        self.detector.set_param_value('mask dilate size', 0)
        self.detector.set_param_value('fill bubble text boxes', False)
        self.image = np.zeros((20, 30, 3), dtype=np.uint8)
        self.image[..., 0] = 200

    def predictions(self) -> SimpleNamespace:
        masks = np.zeros((4, 20, 30), dtype=bool)
        masks[0, 2:7, 3:8] = True
        masks[1, 10:15, 20:25] = True
        masks[2:, :, :] = True
        return SimpleNamespace(
            xyxy=np.array([[2, 1, 9, 8], [19, 9, 26, 16], [0, 0, 30, 20], [0, 0, 30, 20]]),
            class_id=np.array([0, 1, 2, 3]),
            confidence=np.array([0.8, 0.3, 0.9, 0.9]),
            mask=masks,
        )

    def test_text_mask_excludes_sound_effects_bubbles_and_panels(self) -> None:
        predictions = self.predictions()
        self.detector.model = Mock()
        self.detector.model.predict.return_value = predictions
        mask, blocks = self.detector.detect(self.image)
        np.testing.assert_array_equal(mask, predictions.mask[0].astype(np.uint8) * 255)
        self.assertEqual([block.label for block in blocks], ['text'])
        self.assertEqual(blocks[0].xyxy, [2, 1, 9, 8])
        self.assertEqual(blocks[0].det_model, 'koharu')
        self.assertTrue(blocks[0].src_is_vertical)
        self.assertIsNotNone(blocks[0].bubble_polygon)
        self.assertIs(self.detector.model.predict.call_args.args[0], self.image)
        self.assertEqual(self.detector.model.predict.call_args.kwargs['shape'], (1152, 1152))

    def test_runs_sharing_one_balloon_merge_into_a_single_block(self) -> None:
        # Smooth balloons keep one dialogue, ordered right-to-left.
        masks = np.zeros((4, 20, 30), dtype=bool)
        masks[0, 2:18, 20:26] = True
        masks[1, 2:18, 12:18] = True
        masks[2, 2:18, 4:10] = True
        masks[3, 2:18, 4:26] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[20, 2, 26, 18], [12, 2, 18, 18], [4, 2, 10, 18], [4, 2, 26, 18]]),
            class_id=np.array([0, 0, 0, 2]),
            confidence=np.array([0.9, 0.9, 0.9, 0.9]),
            mask=masks,
        )
        self.detector.set_param_value('source text is vertical', True)
        self.detector.set_param_value('auto detect text orientation', False)
        _, blocks = self.detector.detect(self.image)
        merged = [block for block in blocks if block.xyxy[2] <= 26]
        self.assertEqual(len(merged), 1)
        block = merged[0]
        self.assertEqual(len(block.lines), 3)
        # Vertical columns merge right to left: the rightmost run reads first.
        self.assertEqual(
            [line[0][0] for line in block.lines],
            [20, 12, 4],
        )
        self.assertEqual(block.src_is_vertical, True)
        self.assertEqual(block.alignment, 1)
        self.assertEqual(block.xyxy, [4, 2, 26, 18])
        self.assertTrue(block.bubble_polygon)

    def test_joined_balloon_merges_runs_only_within_each_lobe(self) -> None:
        from ballontranslator.modules.textdetector.detector_koharu import _merge_balloon_blocks
        from ballontranslator.utils.bubble import bubble_inner_rect, split_connected_bubble
        from ballontranslator.utils.textblock import TextBlock

        polygon = [[0, 0], [100, 0], [100, 40], [140, 40], [140, 0],
                   [300, 0], [300, 160], [140, 160], [140, 60],
                   [100, 60], [100, 100], [0, 100]]
        runs = []
        for left in (40, 170, 210, 250):
            line = [[left, 20], [left + 15, 20], [left + 15, 80], [left, 80]]
            runs.append(TextBlock(
                xyxy=[left, 20, left + 15, 80], lines=[line],
                src_is_vertical=True, bubble_polygon=polygon,
            ))
        centers = [(left + 7.5, 50) for left in (40, 170, 210, 250)]
        blocks = sorted(_merge_balloon_blocks(runs, 300, 160), key=lambda block: block.xyxy[0])
        self.assertEqual([len(block.lines) for block in blocks], [1, 3])
        self.assertEqual([line[0][0] for line in blocks[1].lines], [250, 210, 170])
        small, large = [bubble_inner_rect(block.bubble_polygon) for block in blocks]
        self.assertLess(small[0] + small[2], large[0])
        for block, center in zip(blocks, (centers[0], centers[-1])):
            import cv2
            self.assertGreater(cv2.pointPolygonTest(np.asarray(block.bubble_polygon, np.float32), center, False), 0)
        # Legacy shared-flow fitting must still assign distinct cells, not
        # overlap multiple translations in the unsplittable large lobe.
        cells = split_connected_bubble(polygon, centers)
        self.assertIsNotNone(cells)
        self.assertEqual(len({tuple(map(tuple, cell)) for cell in cells}), 4)

    def test_sound_effects_opt_in_and_independent_threshold(self) -> None:
        self.detector.model = Mock()
        self.detector.model.predict.return_value = self.predictions()
        self.detector.set_param_value('detect sound effects', True)
        self.detector.set_param_value('source text is vertical', False)
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual({block.label for block in blocks}, {'text', 'onomatopoeia'})
        self.assertTrue(all(not block.src_is_vertical for block in blocks))
        self.assertEqual(np.count_nonzero(mask), 50)
        self.detector.set_param_value('sound effect threshold', 0.4)
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(np.count_nonzero(mask), 25)

    def test_bubble_threshold_only_changes_typesetting_geometry(self) -> None:
        self.detector.model = Mock()
        self.detector.model.predict.return_value = self.predictions()
        self.detector.set_param_value('bubble threshold', 0.95)
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(np.count_nonzero(mask), 25)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].bubble_polygon)

    def test_bubble_text_box_fill_covers_missed_strokes_and_furigana(self) -> None:
        # The whole detected bubble interior is masked so strokes that sit
        # above or beside the text rectangle (furigana, ruby) cannot survive
        # inpainting; the fill is still limited to the bubble's own mask.
        predictions = self.predictions()
        self.detector.model = Mock()
        self.detector.model.predict.return_value = predictions
        self.detector.set_param_value('fill bubble text boxes', True)
        mask, blocks = self.detector.detect(self.image)
        expected = predictions.mask[2] | predictions.mask[0]
        np.testing.assert_array_equal(mask, expected.astype(np.uint8) * 255)
        self.assertEqual(len(blocks), 1)

    def test_box_fill_keeps_segmentation_when_no_bubble_is_detected(self) -> None:
        self.detector.model = Mock()
        self.detector.model.predict.return_value = self.predictions()
        self.detector.set_param_value('fill bubble text boxes', True)
        self.detector.set_param_value('bubble threshold', 0.95)
        mask, _ = self.detector.detect(self.image)
        np.testing.assert_array_equal(mask, self.predictions().mask[0].astype(np.uint8) * 255)

    def test_empty_predictions_return_empty_mask(self) -> None:
        self.detector.model = Mock()
        self.detector.model.predict.return_value = SimpleNamespace(
            xyxy=np.empty((0, 4)), class_id=np.array([]), confidence=np.array([]), mask=None,
        )
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(blocks, [])
        self.assertEqual(mask.shape, (20, 30))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertFalse(mask.any())

    def test_dilation_expands_only_the_selected_text_mask(self) -> None:
        self.detector.model = Mock()
        self.detector.model.predict.return_value = self.predictions()
        self.detector.set_param_value('mask dilate size', 1)
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(len(blocks), 1)
        self.assertGreater(np.count_nonzero(mask), 25)
        self.assertFalse(mask[10:15, 20:25].any())

    def test_duplicate_text_detections_are_suppressed_per_label(self) -> None:
        masks = np.zeros((3, 20, 30), dtype=bool)
        masks[0, 2:7, 3:8] = True
        masks[1, 3:8, 4:9] = True
        masks[2, :, :] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[2, 1, 9, 8], [2, 1, 9, 8], [0, 0, 30, 20]]),
            class_id=np.array([0, 0, 2]),
            confidence=np.array([0.8, 0.9, 0.9]),
            mask=masks,
        )
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual([block.xyxy for block in blocks], [[2, 1, 9, 8]])
        np.testing.assert_array_equal(mask, masks[1].astype(np.uint8) * 255)

    def test_shared_box_across_labels_survives_nms(self) -> None:
        masks = np.zeros((2, 20, 30), dtype=bool)
        masks[0, 2:7, 3:8] = True
        masks[1, 10:15, 20:25] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[2, 1, 9, 8], [2, 1, 9, 8]]),
            class_id=np.array([0, 3]),
            confidence=np.array([0.8, 0.9]),
            mask=masks,
        )
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual([block.label for block in blocks], ['text'])
        np.testing.assert_array_equal(mask, masks[0].astype(np.uint8) * 255)

    def test_merged_coarse_duplicate_of_column_text_is_suppressed(self) -> None:
        # Page 15 doubling: the model emitted one merged instance over two
        # vertical columns alongside the per-column instances. Box IoU between
        # the union and either column is only ~0.4, so the union must be
        # dropped through mask containment instead — the finer per-column
        # instances survive.
        masks = np.zeros((2, 20, 30), dtype=bool)
        masks[0, 2:18, 2:28] = True
        masks[1, 2:18, 2:12] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[2, 2, 28, 18], [2, 2, 12, 18]]),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            mask=masks,
        )
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual([block.xyxy for block in blocks], [[2, 2, 12, 18]])
        np.testing.assert_array_equal(mask, masks[1].astype(np.uint8) * 255)

    def test_small_instance_inside_text_region_survives_nms(self) -> None:
        # A small instance nested in a big region (furigana, caption) must not
        # evict it: the 0.2 area-ratio floor blocks the containment rule.
        masks = np.zeros((2, 20, 30), dtype=bool)
        masks[0, 2:18, 2:28] = True
        masks[1, 4:10, 4:9] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[2, 2, 28, 18], [4, 4, 9, 10]]),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.8]),
            mask=masks,
        )
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(
            sorted(tuple(block.xyxy) for block in blocks),
            [(2, 2, 28, 18), (4, 4, 9, 10)],
        )

    def test_bubble_association_requires_mask_containment(self) -> None:
        masks = np.zeros((2, 20, 30), dtype=bool)
        masks[0, 2:8, 10:16] = True
        masks[1, :, :15] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[10, 2, 16, 8], [0, 0, 15, 20]]),
            class_id=np.array([0, 2]),
            confidence=np.array([0.8, 0.9]),
            mask=masks,
        )
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(len(blocks), 1)
        # The bbox center sits inside the bubble polygon, but only 5/6 of the
        # text mask is contained, so no bubble geometry is assigned.
        self.assertIsNone(blocks[0].bubble_polygon)
        np.testing.assert_array_equal(mask, masks[0].astype(np.uint8) * 255)

    def test_orientation_is_inferred_from_the_text_mask(self) -> None:
        masks = np.zeros((2, 20, 30), dtype=bool)
        masks[0, 8:12, 4:26] = True
        masks[1, 2:18, 24:28] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[4, 8, 26, 12], [24, 2, 28, 18]]),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.9]),
            mask=masks,
        )
        _, blocks = self.detector.detect(self.image)
        blocks = sorted(blocks, key=lambda block: block.xyxy[0])
        self.assertEqual([block.src_is_vertical for block in blocks], [False, True])
        self.detector.set_param_value('auto detect text orientation', False)
        _, blocks = self.detector.detect(self.image)
        blocks = sorted(blocks, key=lambda block: block.xyxy[0])
        self.assertEqual([block.src_is_vertical for block in blocks], [True, True])

    def test_mask_closing_fills_letter_holes_without_extra_growth(self) -> None:
        masks = np.zeros((1, 20, 30), dtype=bool)
        masks[0, 5:14, 10:19] = True
        masks[0, 8:11, 13:16] = False
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[10, 5, 19, 14]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            mask=masks,
        )
        self.detector.set_param_value('mask dilate size', 1)
        mask, blocks = self.detector.detect(self.image)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(mask[9, 14], 255)
        self.assertFalse(mask[3, 8])
        self.detector.set_param_value('mask dilate size', 0)
        mask, _ = self.detector.detect(self.image)
        np.testing.assert_array_equal(mask, masks[0].astype(np.uint8) * 255)

    def test_missing_weights_report_local_setup_path(self) -> None:
        with patch('pathlib.Path.is_file', return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, 'model.safetensors'):
                self.detector.load_model()
        self.assertFalse(self.detector.all_model_loaded())

    def test_clipping_and_degenerate_boxes(self) -> None:
        predictions = self.predictions()
        predictions.xyxy[0] = [-2, -1, 35, 25]
        self.detector.model = Mock()
        self.detector.model.predict.return_value = predictions
        _, blocks = self.detector.detect(self.image)
        self.assertEqual(blocks[0].xyxy, [0, 0, 30, 20])
        predictions.xyxy[0] = [31, 0, 40, 10]
        mask, blocks = self.detector.detect(self.image)
        self.assertFalse(mask.any())
        self.assertEqual(blocks, [])

    def test_missing_mask_and_inference_errors_are_not_empty_success(self) -> None:
        self.detector.model = Mock()
        predictions = self.predictions()
        predictions.mask = None
        self.detector.model.predict.return_value = predictions
        with self.assertRaisesRegex(ModuleRunError, 'segmentation mask'):
            self.detector.detect(self.image)
        self.detector.model.predict.side_effect = RuntimeError('inference failed')
        with self.assertRaisesRegex(ModuleRunError, 'inference failed'):
            self.detector.detect(self.image)

    def test_invalid_runtime_parameters_fail_before_inference(self) -> None:
        self.detector.model = Mock()
        for key, value in [('text threshold', float('nan')), ('text threshold', 1.1),
                           ('mask dilate size', -1), ('mask dilate size', 2.5)]:
            with self.subTest(key=key, value=value):
                previous = self.detector.get_param_value(key)
                self.detector.set_param_value(key, value, convert_dtype=False)
                with self.assertRaises(ModuleRunError):
                    self.detector.detect(self.image)
                self.detector.set_param_value(key, previous, convert_dtype=False)
        self.detector.model.predict.assert_not_called()

    def test_device_changes_unload_without_eager_reload(self) -> None:
        self.detector.model = Mock()
        original = self.detector.model
        self.detector.updateParam('device', self.detector.get_param_value('device'))
        self.assertIs(self.detector.model, original)
        with patch.object(self.detector, 'load_model') as load:
            self.detector.updateParam('device', 'cuda:1')
        self.assertFalse(self.detector.all_model_loaded())
        load.assert_not_called()

    def test_loader_uses_strict_local_weights_without_pretrained_download(self) -> None:
        factory = Mock()
        factory.return_value.model.model.modules.return_value = []
        load_file = Mock(return_value={'weights': 'fixture'})
        with patch.dict('sys.modules', {
            'rfdetr': SimpleNamespace(RFDETRSeg2XLarge=factory),
            'rfdetr.config': SimpleNamespace(PretrainWeightsCompatibilityWarning=UserWarning),
            'rfdetr.models.backbone.dinov2_with_windowed_attn': SimpleNamespace(
                WindowedDinov2WithRegistersBackbone=type('Backbone', (), {}),
            ),
            'safetensors.torch': SimpleNamespace(load_file=load_file),
        }), patch('pathlib.Path.is_file', return_value=True):
            self.detector.load_model()
        self.assertIsNone(factory.call_args.kwargs['pretrain_weights'])
        self.assertEqual(factory.call_args.kwargs['num_classes'], 4)
        factory.return_value.model.model.load_state_dict.assert_called_once_with(
            {'weights': 'fixture'}, strict=True,
        )
        self.assertTrue(self.detector.all_model_loaded())
        factory.return_value.optimize_for_inference.assert_called_once_with(compile=False)

    def test_lazy_metadata_contains_setup_without_loading_model(self) -> None:
        specs = _scan_file('ballontranslator/modules/textdetector/detector_koharu.py', 'textdetector')
        self.assertEqual(validate_lazy_module_specs(specs), [])
        self.assertEqual(specs[0].key, 'koharu')
        self.assertIn('rfdetr==1.7.0', specs[0].dependencies)
        self.assertIn('transformers>=5.1.0,<6.0.0', specs[0].dependencies)
        self.assertFalse(specs[0].params['detect sound effects']['value'])
        self.assertTrue(specs[0].params['fill bubble text boxes']['value'])
        self.assertEqual(len(specs[0].download_file_list), 1)
        self.assertFalse(self.detector.all_model_loaded())

    def test_hayai_and_koharu_accept_the_same_transformers_version(self) -> None:
        from packaging.requirements import Requirement

        requirements = []
        for path, module_type in (
            ('ballontranslator/modules/textdetector/detector_koharu.py', 'textdetector'),
            ('ballontranslator/modules/ocr/ocr_hayai.py', 'ocr'),
        ):
            spec = _scan_file(path, module_type)[0]
            requirement = next(
                Requirement(value) for value in spec.dependencies
                if Requirement(value).name == 'transformers'
            )
            requirements.append(requirement)
        for requirement in requirements:
            self.assertIn('5.16.1', requirement.specifier)
            self.assertNotIn('6.0.0', requirement.specifier)
        self.assertIn('4.57.6', requirements[1].specifier)


    def test_estimate_font_size_prefers_mask_bands_over_bbox_min_dimension(self) -> None:
        from ballontranslator.modules.textdetector.detector_koharu import _estimate_font_size

        # Two vertical columns of 20px glyphs inside a 95px-wide bbox: the
        # bbox minimum dimension (95) equals the whole column pair, not the
        # glyph size.
        two_columns = np.zeros((120, 95), dtype=bool)
        two_columns[10:110, 10:30] = True
        two_columns[10:110, 55:75] = True
        self.assertEqual(_estimate_font_size(two_columns, True, 95), 20)

        # Horizontal lines plus a smaller furigana band: the median ignores it.
        lines = np.zeros((80, 200), dtype=bool)
        lines[5:13, 20:180] = True
        lines[30:48, 20:180] = True
        lines[60:78, 20:180] = True
        self.assertEqual(_estimate_font_size(lines, False, 80), 18)

        # Empty masks keep the bbox-based fallback.
        self.assertEqual(_estimate_font_size(np.zeros((40, 40), dtype=bool), False, 40), 40)

        # Blob-like masks tighten to their single band when it is smaller.
        blob = np.zeros((30, 30), dtype=bool)
        blob[5:25, 5:25] = True
        self.assertEqual(_estimate_font_size(blob, False, 30), 20)

    def test_detect_font_size_uses_mask_band_not_bbox_min_dimension(self) -> None:
        masks = np.zeros((1, 60, 120), dtype=bool)
        # One text run of two 8px-wide columns in a 32x40 bbox: the old
        # min-dimension estimate (32) is 4x the real glyph size.
        masks[0, 10:50, 24:32] = True
        masks[0, 10:50, 48:56] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[24, 10, 56, 50]]),
            class_id=np.array([0]),
            confidence=np.array([0.9]),
            mask=masks,
        )
        self.detector.set_param_value('source text is vertical', True)
        self.detector.set_param_value('auto detect text orientation', False)
        _, blocks = self.detector.detect(np.zeros((60, 120, 3), dtype=np.uint8))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].font_size, 8)
        self.assertEqual(blocks[0]._detected_font_size, 8)

    def test_merged_balloon_font_size_takes_min_member_estimate(self) -> None:
        masks = np.zeros((3, 60, 120), dtype=bool)
        # Right run: one 6px column. Left run: one 12px column. The shared
        # bubble merges them; the merged size must be the smaller glyph.
        masks[0, 10:50, 90:96] = True
        masks[1, 10:50, 20:32] = True
        masks[2, 10:50, 14:100] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[90, 10, 96, 50], [20, 10, 32, 50], [14, 10, 100, 50]]),
            class_id=np.array([0, 0, 2]),
            confidence=np.array([0.9, 0.9, 0.9]),
            mask=masks,
        )
        self.detector.set_param_value('source text is vertical', True)
        self.detector.set_param_value('auto detect text orientation', False)
        _, blocks = self.detector.detect(np.zeros((60, 120, 3), dtype=np.uint8))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].font_size, 6)
        self.assertEqual(blocks[0]._detected_font_size, 6)

    def test_non_finite_box_skips_candidate_without_losing_detections(self) -> None:
        masks = np.zeros((2, 60, 120), dtype=bool)
        masks[0, 10:14, 10:14] = True
        masks[1, 10:50, 24:32] = True
        self.detector.model = Mock()
        self.detector.model.predict.return_value = _SuppressibleDetections(
            xyxy=np.array([[np.nan, 10, 14, 14], [24, 10, 32, 50]]),
            class_id=np.array([0, 0]),
            confidence=np.array([0.9, 0.9]),
            mask=masks,
        )
        self.detector.set_param_value('source text is vertical', True)
        self.detector.set_param_value('auto detect text orientation', False)
        mask, blocks = self.detector.detect(np.zeros((60, 120, 3), dtype=np.uint8))
        self.assertEqual([block.xyxy for block in blocks], [[24, 10, 32, 50]])
        np.testing.assert_array_equal(mask, masks[1].astype(np.uint8) * 255)

if __name__ == '__main__':
    unittest.main()
