import unittest

from Orange.data import ContinuousVariable, guess_data_type


class TestIoUtil(unittest.TestCase):
    def test_guess_continuous_w_nans(self):
        self.assertIs(
            guess_data_type(["9", "", "98", "?", "98", "98", "98"])[2],
            ContinuousVariable)


if __name__ == '__main__':
    unittest.main()
