import unittest
import csv
import io

from AnyQt.QtWidgets import QComboBox, QWidget
from AnyQt.QtTest import QSignalSpy

from Orange.widgets.utils import textimport
from Orange.widgets.tests.base import GuiTest

ColumnTypes = textimport.ColumnType

DATA1 = b",A,B,C"
DATA2 = b""
DATA3 = b"A"
DATA4 = b'A, B, C, D\n' \
        b'a, 1, 2, *\n' \
        b'b, 2, 4, *'
DATA5 = b'a\tb\n' * 1000


class WidgetsTests(GuiTest):
    def test_options_widget(self):
        w = textimport.CSVOptionsWidget()
        schanged = QSignalSpy(w.optionsChanged)
        sedited = QSignalSpy(w.optionsEdited)
        w.setDialect(csv.excel())
        self.assertEqual(len(schanged), 1)
        self.assertEqual(len(sedited), 0)
        w.setSelectedEncoding("iso8859-1")

        self.assertEqual(len(schanged), 2)
        self.assertEqual(len(sedited), 0)

        d = w.dialect()
        self.assertEqual(d.delimiter, csv.excel.delimiter)
        self.assertEqual(d.doublequote, csv.excel.doublequote)
        self.assertEqual(w.encoding(), "iso8859-1")

        d = textimport.Dialect("a", "b", "c", True, True)
        w.setDialect(d)

        cb = w.findChild(QComboBox, "delimiter-combo-box")
        self.assertEqual(cb.currentIndex(),
                         textimport.CSVOptionsWidget.DelimiterOther)
        le = w.findChild(QWidget, "custom-delimiter-edit")
        self.assertEqual(le.text(), "a")

        cb = w.findChild(QWidget, "quote-edit-combo-box")
        self.assertEqual(cb.currentText(), "b")
        d1 = w.dialect()
        self.assertEqual(d.delimiter, d1.delimiter)
        self.assertEqual(d.quotechar, d1.quotechar)

    def test_import_widget(self):
        w = textimport.CSVImportWidget()
        w.setDialect(csv.excel())
        w.setSampleContents(io.BytesIO(DATA1))
        view = w.dataview
        model = view.model()
        self.assertEqual(model.columnCount(), 4)
        self.assertEqual(model.rowCount(), 1)
        self.assertEqual(model.canFetchMore(), False)
        w.setSampleContents(io.BytesIO(DATA2))
        model = view.model()
        self.assertEqual(model.columnCount(), 0)
        self.assertEqual(model.rowCount(), 0)
        self.assertEqual(model.canFetchMore(), False)
        w.setSampleContents(io.BytesIO(DATA4))
        model = view.model()
        self.assertEqual(model.columnCount(), 4)
        self.assertEqual(model.rowCount(), 3)

        types = {
            0: ColumnTypes.Categorical,
            1: ColumnTypes.Numeric,
            2: ColumnTypes.Text,
            3: ColumnTypes.Time,
        }
        w.setColumnTypes(types)
        self.assertEqual(w.columnTypes(), types)
        rs = w.rowStates()
        self.assertEqual(rs, {})
        w.setStateForRow(0, textimport.TablePreview.Header)
        w.setRowStates({0: textimport.TablePreview.Header})
        self.assertEqual(w.rowStates(), {0: textimport.TablePreview.Header})
        w.setStateForRow(1, textimport.TablePreview.Skipped)
        view.grab()

        w.setSampleContents(io.BytesIO(DATA5))
        model = view.model()
        self.assertEqual(model.columnCount(), 1)
        w.setDialect(csv.excel_tab())
        w.setSampleContents(io.BytesIO(DATA5))
        model = view.model()
        self.assertEqual(model.columnCount(), 2)
        self.assertTrue(model.canFetchMore())
        rows = model.rowCount()
        spy = QSignalSpy(model.rowsInserted)
        model.fetchMore()
        self.assertGreater(model.rowCount(), rows)
        self.assertEqual(len(spy), 1)


if __name__ == "__main__":
    unittest.main(__name__)
