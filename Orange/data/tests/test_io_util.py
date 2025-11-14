import os.path
import unittest
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
from numpy.testing import assert_array_equal

from Orange.data import (
    ContinuousVariable,
    guess_data_type,
    Table,
    Domain,
    StringVariable,
    DiscreteVariable,
)
from Orange.data.io_util import update_origin, array_strptime, to_datetime


class TestIoUtil(unittest.TestCase):
    def test_guess_continuous_w_nans(self):
        self.assertIs(
            guess_data_type(["9", "", "98", "?", "98", "98", "98"])[2],
            ContinuousVariable)

    def test_array_strptime(self):
        a = array_strptime(["NaT", "2015"], "%Y").astype(object)
        assert_array_equal(a, [None, datetime(2015, 1, 1)])
        a = array_strptime(["nan", "1402-04-06"], "%Y-%d-%m").astype(object)
        assert_array_equal(a, [None, datetime(1402, 6, 4)])
        with self.assertRaises(ValueError):
            array_strptime(["this is not a date"], "%Y")
        assert_array_equal(
            array_strptime(["this is not a date"], "%Y", errors="coerce").astype(object),
            [None]
        )

    def test_to_datetime(self):
        def assert_equal(a, b):
            # convert to array of datetime objects
            a = a.astype(object)
            assert_array_equal(a, b)
        assert_equal(
            to_datetime(["1/1/2020", "2/1/2020"]),
            [datetime(2020, 1, 1), datetime(2020, 2, 1)]
        )

        assert_equal(
            to_datetime(["NaT", "1/1/2020", "2/1/2020"]),
            [None, datetime(2020, 1, 1), datetime(2020, 2, 1)]
        )

        assert_equal(
            to_datetime(["1/1/2020", "2/1/2020", "1/1/1400"]),
            [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(1400, 1, 1)]
        )

        assert_equal(
            to_datetime(["1|12|7000"], format="%d|%m|%Y"),
            [datetime(7000, 12, 1)]
        )

        with self.assertRaises(ValueError):
            to_datetime(["1012l0awd7"], format="%d|%m|%Y", errors="raise"),

        assert_equal(
            to_datetime(["1012l0awd7"], format="%d|%m|%Y", errors="coerce"),
            [None]
        )


class TestUpdateOrigin(unittest.TestCase):
    FILE_NAMES = ["file1.txt", "file2.txt", "file3.txt"]

    def setUp(self) -> None:
        self.alt_dir = TemporaryDirectory()  # pylint: disable=consider-using-with

        self.var_string = var = StringVariable("Files")
        files = self.FILE_NAMES + [var.Unknown]
        self.table_string = Table.from_list(
            Domain([], metas=[var]), np.array(files).reshape((-1, 1))
        )
        self.var_discrete = var = DiscreteVariable("Files", values=self.FILE_NAMES)
        files = self.FILE_NAMES + [var.Unknown]
        self.table_discrete = Table.from_list(
            Domain([], metas=[var]), np.array(files).reshape((-1, 1))
        )

    def tearDown(self) -> None:
        self.alt_dir.cleanup()

    def __create_files(self):
        for f in self.FILE_NAMES:
            f = os.path.join(self.alt_dir.name, f)
            with open(f, "w", encoding="utf8"):
                pass
            self.assertTrue(os.path.exists(f))

    def test_origin_not_changed(self):
        """
        Origin exist; keep it unchanged, even though dataset path also includes
        files from column.
        """
        with TemporaryDirectory() as dir_name:
            self.var_string.attributes["origin"] = dir_name
            update_origin(self.table_string, self.alt_dir.name)
            self.assertEqual(
                self.table_string.domain[self.var_string].attributes["origin"], dir_name
            )

    def test_origin_subdir(self):
        """
        Origin is wrong but last dir in origin exit in the dataset file's path
        """
        images_dir = os.path.join(self.alt_dir.name, "subdir")
        os.mkdir(images_dir)

        self.var_string.attributes["origin"] = "/a/b/subdir"
        update_origin(self.table_string, os.path.join(self.alt_dir.name, "data.csv"))
        self.assertEqual(
            self.table_string.domain[self.var_string].attributes["origin"], images_dir
        )

    def test_origin_parents_subdir(self):
        """
        Origin is wrong but last dir in origin exit in the dataset file
        parent's directory
        """
        # make the dir where dataset is placed
        images_dir = os.path.join(self.alt_dir.name, "subdir")
        os.mkdir(images_dir)

        self.var_string.attributes["origin"] = "/a/b/subdir"
        update_origin(self.table_string, os.path.join(images_dir, "data.csv"))
        self.assertEqual(
            self.table_string.domain[self.var_string].attributes["origin"], images_dir
        )

    def test_column_paths_subdir(self):
        """
        Origin dir not exiting but paths from column exist in dataset's dir
        """
        self.__create_files()

        self.var_string.attributes["origin"] = "/a/b/non-exiting-dir"
        update_origin(self.table_string, os.path.join(self.alt_dir.name, "data.csv"))
        self.assertEqual(
            self.table_string.domain[self.var_string].attributes["origin"],
            self.alt_dir.name,
        )

        self.var_discrete.attributes["origin"] = "/a/b/non-exiting-dir"
        update_origin(self.table_discrete, os.path.join(self.alt_dir.name, "data.csv"))
        self.assertEqual(
            self.table_discrete.domain[self.var_discrete].attributes["origin"],
            self.alt_dir.name,
        )

    def test_column_paths_parents_subdir(self):
        """
        Origin dir not exiting but paths from column exist in dataset parent's dir
        """
        # make the dir where dataset is placed
        dataset_dir = os.path.join(self.alt_dir.name, "subdir")
        self.__create_files()

        self.var_string.attributes["origin"] = "/a/b/non-exiting-dir"
        update_origin(self.table_string, os.path.join(dataset_dir, "data.csv"))
        self.assertEqual(
            self.table_string.domain[self.var_string].attributes["origin"],
            self.alt_dir.name,
        )

        self.var_discrete.attributes["origin"] = "/a/b/non-exiting-dir"
        update_origin(self.table_discrete, os.path.join(dataset_dir, "data.csv"))
        self.assertEqual(
            self.table_discrete.domain[self.var_discrete].attributes["origin"],
            self.alt_dir.name,
        )


if __name__ == '__main__':
    unittest.main()
