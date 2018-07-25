import os
import unittest
from tempfile import NamedTemporaryFile

from Orange.widgets.utils.filedialogs import RecentPath


class TestRecentPath(unittest.TestCase):
    def test_resolve(self):
        temp_file = NamedTemporaryFile(dir=os.getcwd(), delete=False)
        file_name = temp_file.name
        temp_file.close()
        base_name = os.path.basename(file_name)
        try:
            recent_path = RecentPath(
                os.path.join("temp/datasets", base_name), "",
                os.path.join("datasets", base_name)
            )
            search_paths = [("basedir", os.getcwd())]
            self.assertIsNotNone(recent_path.resolve(search_paths))
        finally:
            os.remove(file_name)
