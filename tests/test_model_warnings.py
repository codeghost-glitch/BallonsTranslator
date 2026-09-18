import logging
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ballontranslator.utils.logger import suppress_model_warnings
from ballontranslator.modules.textdetector.detector_koharu import _explicit_return_dict


class ModelWarningTests(unittest.TestCase):
    def test_filter_is_thread_scoped_and_removed_after_failure(self) -> None:
        target = logging.getLogger('test.model_warnings')
        with self.assertLogs(target, level='WARNING') as captured:
            with self.assertRaises(RuntimeError):
                with suppress_model_warnings(target.name, ('Expected warning',)):
                    target.warning('Expected warning in owner')
                    worker = Thread(target=target.warning, args=('Expected warning in worker',))
                    worker.start()
                    worker.join()
                    target.warning('Unexpected warning')
                    target.error('Expected warning but error severity')
                    raise RuntimeError('load failed')
            target.warning('Expected warning after context')
        self.assertEqual([record.getMessage() for record in captured.records], [
            'Expected warning in worker', 'Unexpected warning',
            'Expected warning but error severity', 'Expected warning after context',
        ])

    def test_return_dict_hook_preserves_explicit_values(self) -> None:
        module = SimpleNamespace(config=SimpleNamespace(return_dict=False))
        self.assertEqual(_explicit_return_dict(module, (), {})[1], {'return_dict': False})
        self.assertEqual(_explicit_return_dict(module, (), {'return_dict': True})[1], {'return_dict': True})

    def test_hayai_processor_uses_cache_and_only_falls_back_on_cache_miss(self) -> None:
        from ballontranslator.modules.ocr.ocr_hayai import HayaiOCRV2, VISION_MODEL_ID

        for cache_missing in (False, True):
            with self.subTest(cache_missing=cache_missing), patch(
                'ballontranslator.modules.ocr.ocr_hayai.Siglip2ImageProcessor.from_pretrained',
            ) as processor, patch(
                'ballontranslator.modules.ocr.ocr_hayai.PreTrainedTokenizerFast.from_pretrained',
            ), patch('ballontranslator.modules.ocr.ocr_hayai.AutoModel.from_pretrained'):
                if cache_missing:
                    processor.side_effect = [OSError('cache miss'), Mock()]
                HayaiOCRV2()._load_model()
                self.assertEqual(processor.call_args_list[0].args, (VISION_MODEL_ID,))
                self.assertEqual(processor.call_args_list[0].kwargs, {'local_files_only': True})
                self.assertEqual(processor.call_count, 2 if cache_missing else 1)


if __name__ == '__main__':
    unittest.main()
