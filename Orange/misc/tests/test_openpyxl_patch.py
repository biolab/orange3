import unittest

import openpyxl


class TestRemoveTemporarySolution(unittest.TestCase):
    def test_remove_openpyxl_temp_solution(self):
        """
        When this test starts to fail revert https://github.com/biolab/orange3/pull/6737
        """
        self.assertLessEqual(
            [int(x) for x in openpyxl.__version__.split(".")], [3, 1, 2]
        )


if __name__ == "__main__":
    unittest.main()
