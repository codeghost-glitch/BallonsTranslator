import os
import tempfile
import unittest
import zipfile

from ballontranslator.utils.proj_imgtrans import ProjImgTrans

PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
JPEG_MAGIC = b'\xff\xd8\xff'


def make_project():
    project = ProjImgTrans()
    project.directory = tempfile.mkdtemp()
    return project


class CbzNamingTests(unittest.TestCase):
    def test_archive_takes_project_folder_name(self) -> None:
        project = make_project()
        self.assertEqual(
            os.path.basename(project.cbz_path()),
            os.path.basename(project.directory) + '.cbz',
        )
        self.assertEqual(
            os.path.dirname(project.cbz_path()), project.directory,
        )

    def test_encoding_follows_original_extension(self) -> None:
        self.assertEqual(ProjImgTrans.cbz_encode_params('01.jpg'), ('JPEG', 95))
        self.assertEqual(ProjImgTrans.cbz_encode_params('02.jpeg'), ('JPEG', 95))
        self.assertEqual(ProjImgTrans.cbz_encode_params('03.PNG'), ('PNG', -1))
        self.assertEqual(ProjImgTrans.cbz_encode_params('04.webp'), ('PNG', -1))


class DumpCbzTests(unittest.TestCase):
    def test_round_trip_preserves_names_order_and_bytes(self) -> None:
        project = make_project()
        entries = [
            ('01.png', PNG_MAGIC + b'first'),
            ('02.jpg', JPEG_MAGIC + b'second'),
        ]
        path = project.dump_cbz(entries)
        self.assertEqual(path, project.cbz_path())
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(
                archive.namelist(), [name for name, _ in entries],
            )
            for name, payload in entries:
                self.assertEqual(archive.read(name), payload)

    def test_explicit_path_is_used(self) -> None:
        project = make_project()
        target = os.path.join(project.directory, 'custom.cbz')
        returned = project.dump_cbz([('01.png', PNG_MAGIC)], target)
        self.assertEqual(returned, target)
        self.assertTrue(os.path.exists(target))

    def test_empty_export_is_rejected(self) -> None:
        project = make_project()
        with self.assertRaisesRegex(ValueError, 'no rendered pages'):
            project.dump_cbz([])


if __name__ == '__main__':
    unittest.main()
