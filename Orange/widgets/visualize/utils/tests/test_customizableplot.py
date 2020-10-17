import unittest
from unittest.mock import patch, Mock

from Orange.widgets.visualize.utils import customizableplot


class TestFonts(unittest.TestCase):
    def test_available_font_families(self):
        with patch.object(customizableplot, "QFont") as font, \
                patch.object(customizableplot, "QFontDatabase") as db:
            font.return_value = Mock()
            font.return_value.family = Mock(return_value="mock regular")

            db.return_value = Mock()
            db.return_value.families = Mock(
                return_value=["a", ".d", "e", ".b", "mock regular", "c"])
            self.assertEqual(customizableplot.available_font_families(),
                             ["mock regular", "", "a", ".b", "c", ".d", "e"])

            db.return_value = Mock()
            db.return_value.families = Mock(
                return_value=["a", ".d", "e", ".b", "mock regular",
                              "mock bold", "mock italic", "c", "mock semi"])
            self.assertEqual(customizableplot.available_font_families(),
                             ["mock regular", "mock bold", "mock italic",
                              "mock semi", "",
                              "a", ".b", "c", ".d", "e"])

            # It seems it's possible that default font family does not exist
            # (see gh-5036)
            db.return_value.families.return_value = ["a", ".d", "e", ".b", "c"]
            self.assertEqual(customizableplot.available_font_families(),
                             ["mock regular", "", "a", ".b", "c", ".d", "e"])
            self.assertIn(customizableplot.default_font_family(),
                          customizableplot.available_font_families())

if __name__ == "__main__":
    unittest.main()
