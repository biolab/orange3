import unittest


class TestAxisItem(unittest.TestCase):
    def test_scalable_axis_item(self):
        from pyqtgraph import __version__
        # When upgraded to 0.11.1 (or bigger) check if resizing AxisItem font
        # works and overwritten functions generateDrawSpecs and
        # _updateMaxTextSize are no longer needed.
        self.assertLess(__version__, "0.11.1")


if __name__ == '__main__':
    unittest.main()
