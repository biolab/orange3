import unittest
from unittest.mock import patch
from Orange.widgets.utils.userinput import (
    _get_points, numbers_from_list)


class TestPointsFromList(unittest.TestCase):
    @patch("Orange.widgets.utils.userinput._numbers_from_no_dots")
    @patch("Orange.widgets.utils.userinput._numbers_from_dots")
    def test_points_from_list(self, from_dots, no_dots):
        numbers_from_list("1 2 3", int)
        no_dots.assert_called()
        no_dots.reset_mock()
        from_dots.assert_not_called()

        numbers_from_list("5, 9", int)
        no_dots.assert_called()
        no_dots.reset_mock()
        from_dots.assert_not_called()

        numbers_from_list("5.13", int)
        no_dots.assert_called()
        no_dots.reset_mock()
        from_dots.assert_not_called()

        numbers_from_list("1 ... 3", int)
        no_dots.assert_not_called()
        from_dots.assert_called()
        from_dots.reset_mock()

        numbers_from_list("1, 2, 3...5", int)
        no_dots.assert_not_called()
        from_dots.assert_called()
        from_dots.reset_mock()

        numbers_from_list("...3, 4, 5", int)
        no_dots.assert_not_called()
        from_dots.assert_called()
        from_dots.reset_mock()

        numbers_from_list("3, 4, 5...", int)
        no_dots.assert_not_called()
        from_dots.assert_called()
        from_dots.reset_mock()

        numbers_from_list("3, 4, ...,5...", int)
        no_dots.assert_not_called()
        from_dots.assert_called()
        from_dots.reset_mock()

    def test_get_points(self):
        self.assertEqual(_get_points(["1", "2", "3"], int), (1, 2, 3))
        self.assertEqual(_get_points(["1", "2", "3"], float), (1, 2, 3))
        self.assertEqual(_get_points(["10.5", "2.25", "3"], float), (10.5, 2.25, 3))

        with self.assertRaisesRegex(ValueError, "3.3"):
            _get_points(["1", "2", "3.3", "4"], int)

        with self.assertRaisesRegex(ValueError, "asdf"):
            _get_points(["1", "2", "asdf", "4"], int)

    def test_numbers_from_no_dots(self):
        self.assertEqual(numbers_from_list("1 2 3", int), (1, 2, 3))
        self.assertEqual(numbers_from_list("1 2 3", float), (1, 2, 3))
        self.assertEqual(numbers_from_list("10.5 2.25 3", float), (2.25, 3, 10.5))

        with self.assertRaisesRegex(ValueError, "3.3"):
            numbers_from_list("1 2 3.3 4", int)

        with self.assertRaisesRegex(ValueError, "asdf"):
            numbers_from_list("1 2 asdf 4", int)

        with self.assertRaisesRegex(ValueError, "value must be at least 2"):
            numbers_from_list("1", int, 2)

        with self.assertRaisesRegex(ValueError, "value must be at most 2"):
            numbers_from_list("3", int, None, 2)

        with self.assertRaisesRegex(ValueError, "value must be between 2 and 3"):
            numbers_from_list("1 4 2 3", int, 2, 3)

    def test_numbers_from_dots(self):
        def check(minimum, maximum, tests, typ=int, enforce_range=True):
            for text, *expected in tests:
                if not expected:
                    with self.assertRaises(ValueError):
                        numbers_from_list(text, typ, minimum, maximum,
                                          enforce_range)
                else:
                    self.assertEqual(
                        numbers_from_list(text, typ, minimum, maximum,
                                          enforce_range),
                        expected[0],
                        f"for {text}")

        check(None, None, [
            ("1, 2, ..., 5", (1, 2, 3, 4, 5)),
            ("1, 2, 3, ..., 5, 6, 7", (1, 2, 3, 4, 5, 6, 7)),
            ("3, ..., 5, 6", (3, 4, 5, 6)),
            ("..., 5, 6", ),
            ("5, 6, ...", ),
            ("1, 2, 3, 4, 5, ...", ),
            ("1, ..., 5", ),
            ("1, 2, ..., 5, 6, ..., 8", )])

        # 5 to 10
        check(5, 10, [
            ("4, 5, ..., 8", ),
            ("5, 6, ..., 12", ),
            ("5, 6, ..., 9", (5, 6, 7, 8, 9)),
            ("6, 7, ..., 9", (6, 7, 8, 9)),
            ("6, 7, ..., 8, 9", (6, 7, 8, 9)),
            ("..., 8, 9", (5, 6, 7, 8, 9)),
            ("6, 7, ...", (6, 7, 8, 9, 10)),
            ("6, 7, 8, 9, ...", (6, 7, 8, 9, 10)),
            ("..., 8, 9", (5, 6, 7, 8, 9)),
            ])

        check(5, None, [
            ("4, 5, ..., 8", ),
            ("5, 6, ..., 12", (5, 6, 7, 8, 9, 10, 11, 12)),
            ("6, 7, ..., 9", (6, 7, 8, 9)),
            ("6, 7, ..., 8, 9", (6, 7, 8, 9)),
            ("..., 8, 9", (5, 6, 7, 8, 9)),
            ("6, 7, ...", )
            ])

        check(None, 10, [
            ("4, 5, ..., 8", (4, 5, 6, 7, 8)),
            ("5, 6, ..., 12", ),
            ("5, 6, ..., 9", (5, 6, 7, 8, 9)),
            ("6, 7, ..., 9", (6, 7, 8, 9)),
            ("..., 8, 9", ),
            ("6, 7, ...", (6, 7, 8, 9, 10)),
            ("6, 7, 8, 9, ...", (6, 7, 8, 9, 10)),
            ("..., 8, 9", )])

        check(5, 10, [
            ("5, 6..., 8", (5, 6, 7, 8)),
            ("5,6...,8", (5, 6, 7, 8)),
            ("5,6...8", (5, 6, 7, 8)),
            ("5,   6  ...  8", (5, 6, 7, 8)),
            ("5,   6  ...  8", (5, 6, 7, 8)),
            ("5    6  ...  ", (5, 6, 7, 8, 9, 10)),
            ("..., 7, 8", (5, 6, 7, 8)),
            ("..., 7, 8, ...", ),
            ("5, 6, ..., 7, 8, ...", ),
            ("5, 6, ..., 7, 8, ...", ),
            ("5  6  8, ...", ),
            ("5, 6, 8, ...", ),
            ("5, 6, ..., 8, 10", ),
            ("5, 7, ..., 8, 10", ),
            ("8, 7, 6, ...", ),
            ("5, 6, 7, ..., 7, 8", )])

        check(5, 10, [
            ("3, 4, 5, 6..., 8", (3, 4, 5, 6, 7, 8)),
            ("5, ..., 9, 11, 13", (5, 7, 9, 11, 13)),
            ("5, 6, ..., ", (5, 6, 7, 8, 9, 10)),
            ("..., 9, 10, 11", (5, 6, 7, 8, 9, 10, 11))],
            int, False)

        check(1, None, [
            ("5, 6..., 8", (5, 6, 7, 8)),
            ("..., 7, 9", (1, 3, 5, 7, 9)),
            ("..., 8, 10", (2, 4, 6, 8, 10)),
            ("..., 7, 10", (1, 4, 7, 10)),
            ("..., 6, 10", (2, 6, 10)),
            ("..., 5, 10", (5, 10)),
            ("..., 4, 10", (4, 10)),
        ])

        check(None, 10, [
            ("2, 4, ...", (2, 4, 6, 8, 10)),
            ("1, 3, ...", (1, 3, 5, 7, 9)),
            ("1, 4, ...", (1, 4, 7, 10)),
            ("1, 5, ...", (1, 5, 9)),
            ("1, 6, ...", (1, 6)),
        ])

        check(None, None, [
            ("1.3, 1.6, ..., 2.5", (1.3, 1.6, 1.9, 2.2, 2.5)),
            ("1.3, 1.6, ..., 2.55", )],
            float)


if __name__ == "__main__":
    unittest.main()
