from numpy.testing import assert_array_equal
import unittest

import os
import io
import csv

import numpy as np

from Orange.widgets.tests.base import WidgetTest, GuiTest

from Orange.widgets.data import owcsvimport
from Orange.widgets.data.owcsvimport import (
    pandas_to_table, ColumnType, RowSpec
)


class TestOWCSVFileImport(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(owcsvimport.OWCSVFileImport)

    def tearDown(self):
        self.widgets.remove(self.widget)
        self.widget.onDeleteWidget()
        self.widget = None

    def test_basic(self):
        w = self.widget
        w.activate_recent(0)
        w.cancel()


class TestImportDialog(GuiTest):
    def test_dialog(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "grep_file.txt")
        d = owcsvimport.CSVImportDialog()
        d.setPath(path)
        opts = owcsvimport.Options(
            encoding="utf-8",
            dialect=owcsvimport.textimport.Dialect(
                " ", "\"", "\\", True, True
            ),
            columntypes=[
                (range(0, 2), ColumnType.Numeric),
                (range(2, 3), ColumnType.Categorical)
            ],
            rowspec=[
                (range(0, 4), RowSpec.Skipped),
                (range(4, 5), RowSpec.Header),
                (range(8, 13), RowSpec.Skipped),
            ]
        )
        d.setOptions(opts)
        d.restoreDefaults()
        opts1 = d.options()
        d.reset()
        opts1 = d.options()


class TestUtils(unittest.TestCase):
    def test_load_csv(self):
        contents = (
            b'1/1/1990,1.0,[,one,\n'
            b'1/1/1990,2.0,],two,\n'
            b'1/1/1990,3.0,{,three,'
        )
        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=csv.excel(),
            columntypes=[
                (range(0, 1), ColumnType.Time),
                (range(1, 2), ColumnType.Numeric),
                (range(2, 3), ColumnType.Text),
                (range(3, 4), ColumnType.Categorical),
                (range(4, 5), ColumnType.Auto),
            ],
            rowspec=[]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        self.assertEqual(df.shape, (3, 5))
        self.assertSequenceEqual(
            list(df.dtypes),
            [np.dtype("M8[ns]"), np.dtype(float), np.dtype(object),
             "category", np.dtype(float)],
        )
        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=csv.excel(),
            columntypes=[
                (range(0, 1), ColumnType.Skip),
                (range(1, 2), ColumnType.Numeric),
                (range(2, 3), ColumnType.Skip),
                (range(3, 4), ColumnType.Categorical),
                (range(4, 5), ColumnType.Skip),
            ],
            rowspec=[
                (range(1, 2), RowSpec.Skipped)
            ]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        self.assertEqual(df.shape, (2, 2))
        self.assertSequenceEqual(
            list(df.dtypes), [np.dtype(float), "category"]
        )
        self.assertSequenceEqual(
            list(df.iloc[:, 0]), [1.0, 3.0]
        )
        self.assertSequenceEqual(
            list(df.iloc[:, 1]), ["one", "three"]
        )

    def test_convert(self):
        contents = (
            b'I, J,  K\n'
            b' , A,   \n'
            b'B,  ,  1\n'
            b'?, ., NA'
        )

        class dialect(csv.excel):
            skipinitialspace = True

        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=dialect(),
            columntypes=[
                (range(0, 1), ColumnType.Text),
                (range(1, 2), ColumnType.Categorical),
                (range(2, 3), ColumnType.Text),

            ],
            rowspec=[(range(0, 1), RowSpec.Header)]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        tb = pandas_to_table(df)

        assert_array_equal(tb.metas[:, 0], ["", "B", "?"])
        assert_array_equal(tb.metas[:, 1], ["", "1", "NA"])
        assert_array_equal(tb.X[:, 0], [0.0, np.nan, np.nan])

        opts = owcsvimport.Options(
            encoding="ascii",
            dialect=dialect(),
            columntypes=[
                (range(0, 1), ColumnType.Categorical),
                (range(1, 2), ColumnType.Categorical),
                (range(2, 3), ColumnType.Numeric),
            ],
            rowspec=[(range(0, 1), RowSpec.Header)]
        )
        df = owcsvimport.load_csv(io.BytesIO(contents), opts)
        tb = pandas_to_table(df)

        assert_array_equal(tb.X[:, 0], [np.nan, 0, np.nan])
        assert_array_equal(tb.X[:, 1], [0, np.nan, np.nan])
        assert_array_equal(tb.X[:, 2], [np.nan, 1, np.nan])
