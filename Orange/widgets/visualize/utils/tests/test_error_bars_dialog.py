# pylint: disable=protected-access
import unittest
from unittest.mock import Mock

from orangewidget.tests.base import GuiTest

from Orange.data import Table
from Orange.widgets.visualize.utils.error_bars_dialog import ErrorBarsDialog


class TestErrorBarsDialog(GuiTest):
    def setUp(self) -> None:
        self._dlg = ErrorBarsDialog(None)

    def test_init(self):
        form = self._dlg.layout().itemAt(0)

        self.assertEqual(form.itemAt(0).widget().text(), "Upper:")
        self.assertEqual(form.itemAt(1).widget().currentText(), "(None)")

        self.assertEqual(form.itemAt(2).widget().text(), "Lower:")
        self.assertEqual(form.itemAt(3).widget().currentText(), "(None)")

        self.assertEqual(form.itemAt(4).widget().text(),
                         "Difference from plotted value")
        self.assertTrue(form.itemAt(4).widget().isChecked())

        self.assertEqual(form.itemAt(5).widget().text(),
                         "Absolute position on the plot")
        self.assertFalse(form.itemAt(5).widget().isChecked())

    def test_get_data(self):
        upper_var, lower_var, is_abs = self._dlg.get_data()
        self.assertIsNone(upper_var)
        self.assertIsNone(lower_var)
        self.assertFalse(is_abs)

    def test_set_data(self):
        data = Table("iris")

        self._dlg._set_data(data.domain, data.domain.attributes[2],
                            data.domain.attributes[1], True)
        upper_var, lower_var, is_abs = self._dlg.get_data()
        self.assertIs(upper_var, data.domain.attributes[2])
        self.assertIs(lower_var, data.domain.attributes[1])
        self.assertTrue(is_abs)

        self._dlg._set_data(data.domain, None, None, True)
        upper_var, lower_var, is_abs = self._dlg.get_data()
        self.assertIsNone(upper_var)
        self.assertIsNone(lower_var)
        self.assertTrue(is_abs)

    def test_set_data_none(self):
        self._dlg._set_data(None, None, None, False)
        upper_var, lower_var, is_abs = self._dlg.get_data()
        self.assertIsNone(upper_var)
        self.assertIsNone(lower_var)
        self.assertFalse(is_abs)

    def test_set_data_err(self):
        data = Table("iris")
        self.assertRaises(ValueError, self._dlg._set_data, data.domain,
                          data.domain.class_var, data.domain.class_var, False)

    def test_changed(self):
        data = Table("iris")
        mock = Mock()
        self._dlg.changed.connect(mock)
        self._dlg._set_data(data.domain, data.domain.attributes[2],
                            data.domain.attributes[1], True)

        self._dlg._ErrorBarsDialog__upper_combo.setCurrentIndex(1)
        mock.assert_called_once()

        mock.reset_mock()
        self._dlg._ErrorBarsDialog__lower_combo.setCurrentIndex(0)
        mock.assert_called_once()

        mock.reset_mock()
        self._dlg._ErrorBarsDialog__radio_buttons.buttons()[0].click()
        mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
