"""Right-to-left language handling.

Qt auto-detects bidi for glyph shaping, so the app-level work is
recognising an RTL target and mirroring the alignment that would otherwise
be derived from the (left-to-right) source geometry.
"""
import unittest

from ballontranslator.utils.text_processing import is_rtl, is_cjk
from ballontranslator.utils.textblock import TextBlock, TextAlignment


class TestRTLDetection(unittest.TestCase):

    def test_display_names_used_by_translator_modules(self) -> None:
        # These are the exact strings the target dropdown stores for
        # pcfg.module.translate_target, not native-script names.
        for name in ('Arabic', 'Hebrew', 'Persian', 'Urdu', 'Yiddish'):
            self.assertTrue(is_rtl(name), name)
        for name in ('English', '日本語', '简体中文', 'Français', 'ไทย'):
            self.assertFalse(is_rtl(name), name)

    def test_native_script_spellings_also_match(self) -> None:
        # A custom translator module may store a native name instead.
        self.assertTrue(is_rtl('العربية'))
        self.assertTrue(is_rtl('עברית'))

    def test_rtl_and_cjk_sets_are_disjoint(self) -> None:
        # A language must never be both: layout picks wrapping style from
        # is_cjk and edge mirroring from is_rtl.
        from ballontranslator.utils.text_processing import LANGSET_CJK, LANGSET_RTL
        self.assertFalse(LANGSET_CJK & LANGSET_RTL)


class TestRTLMirroring(unittest.TestCase):

    def test_left_and_right_swap_center_unchanged(self) -> None:
        block = TextBlock([0, 0, 40, 20])
        block.alignment = TextAlignment.Left
        block.mirror_alignment_for_rtl()
        self.assertEqual(block.alignment, TextAlignment.Right)

        block.alignment = TextAlignment.Right
        block.mirror_alignment_for_rtl()
        self.assertEqual(block.alignment, TextAlignment.Left)

        block.alignment = TextAlignment.Center
        block.mirror_alignment_for_rtl()
        self.assertEqual(block.alignment, TextAlignment.Center)


class TestSpacelessScriptSegmentation(unittest.TestCase):
    """Languages written without spaces must still wrap across lines."""

    def test_thai_segments_into_words(self) -> None:
        from ballontranslator.utils.text_processing import seg_text
        try:
            import pythainlp  # noqa: F401
        except ImportError:
            self.skipTest('pythainlp not installed')
        words, _ = seg_text('นี่คือการทดสอบ', 'Thai')
        # Real word boundaries, not one unbreakable blob.
        self.assertGreater(len(words), 1)
        self.assertEqual(''.join(words), 'นี่คือการทดสอบ')

    def test_other_spaceless_scripts_do_not_become_one_blob(self) -> None:
        from ballontranslator.utils.text_processing import seg_text
        for lang, text in (('Khmer', 'នេះជាការសាកល្បង'),
                           ('Burmese', 'ဤသည်မှာစမ်းသပ်မှု')):
            with self.subTest(lang=lang):
                words, _ = seg_text(text, lang)
                self.assertGreater(len(words), 1,
                                   f'{lang} collapsed to one unbreakable blob')

    def test_space_languages_unchanged(self) -> None:
        from ballontranslator.utils.text_processing import seg_text
        words, delim = seg_text('hello world', 'English')
        self.assertEqual(words, ['hello', 'world'])
        self.assertEqual(delim, ' ')


if __name__ == '__main__':
    unittest.main()
