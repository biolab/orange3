import unittest


class TestAxisItem(unittest.TestCase):
    def test_remove_old_pyqtgraph_support(self):
        from pyqtgraph import __version__
        # When 0.11.2 is released there is probably time to drop support
        # for pyqtgraph <= 0.11.0:
        # - remove AxisItem.generateDrawSpecs
        # - remove AxisItem._updateMaxTextSize
        self.assertLess(__version__, "0.11.2")


if __name__ == '__main__':
    unittest.main()
