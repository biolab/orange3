import unittest
from Orange.widgets.utils.localization import pl


class TestEn(unittest.TestCase):
    def test_pl(self):
        self.assertEqual(pl(0, "cat"), "cats")
        self.assertEqual(pl(1, "cat"), "cat")
        self.assertEqual(pl(2, "cat"), "cats")
        self.assertEqual(pl(100, "cat"), "cats")
        self.assertEqual(pl(101, "cat"), "cats")

        self.assertEqual(pl(0, "cat|cats"), "cats")
        self.assertEqual(pl(1, "cat|cats"), "cat")
        self.assertEqual(pl(2, "cat|cats"), "cats")
        self.assertEqual(pl(100, "cat|cats"), "cats")
        self.assertEqual(pl(101, "cat|cats"), "cats")


if __name__ == "__main__":
    unittest.main()
