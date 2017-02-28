# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
from Orange.canvas.application.addons import cleanup
from Orange.canvas.gui import test


class TestSchemeInfo(test.QAppTestCase):
    def test_cleanup_name(self):
        """Test 'cleaning up' of potential names for addons"""
        names = ["Orange3-Data-Fusion", "Orange3 - Text", "Orange3-spark",
                 "Orange-Bioinformatics", "Orange3-ImageAnalytics", "Associate"]
        for name in names:
            name = cleanup(name)
            self.assertIsNotNone(name)
            self.assertNotEqual("", name)
