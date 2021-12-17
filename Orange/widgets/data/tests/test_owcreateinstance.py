# pylint: disable=missing-docstring,protected-access
from unittest.mock import Mock

import numpy as np

from AnyQt.QtCore import QDateTime, QDate, QTime, QPoint, QObject
from AnyQt.QtWidgets import QWidget, QLineEdit, QStyleOptionViewItem, QMenu, \
    QPushButton

from orangewidget.tests.base import GuiTest

from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable, \
    TimeVariable
from Orange.widgets.data.owcreateinstance import OWCreateInstance, \
    DiscreteVariableEditor, ContinuousVariableEditor, StringVariableEditor, \
    TimeVariableEditor, VariableDelegate, VariableItemModel, ValueRole
from Orange.widgets.tests.base import WidgetTest, datasets


class TestOWCreateInstance(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCreateInstance)
        self.data = Table("iris")

    def test_output(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.controls.append_to_data.setChecked(False)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 1)
        self.assertEqual(output.name, "created")
        self.assertEqual(output.domain, self.data.domain)
        array = np.round(np.median(self.data.X, axis=0), 1).reshape(1, 4)
        np.testing.assert_array_equal(output.X, array)

    def test_output_append_data(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.controls.append_to_data.setChecked(True)

        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 151)

        np.testing.assert_array_equal(output.X[:150], self.data.X)
        np.testing.assert_array_equal(output.Y[:150], self.data.Y)
        array = np.zeros((150, 1), dtype=object)
        np.testing.assert_array_equal(output.metas[:150], array)

        array = np.round(np.median(self.data.X, axis=0), 1).reshape(1, 4)
        np.testing.assert_array_equal(output.X[150:], array)
        np.testing.assert_array_equal(output.Y[150:], np.array([0]))
        np.testing.assert_array_equal(output.metas[150:], np.array([[1]]))

        self.assertEqual(output.domain.attributes, self.data.domain.attributes)
        self.assertEqual(output.domain.class_vars, self.data.domain.class_vars)
        self.assertIn("Source ID", [m.name for m in output.domain.metas])
        self.assertTupleEqual(output.domain.metas[0].values,
                              ("iris", "created"))

    def _get_init_buttons(self, widget=None):
        if not widget:
            widget = self.widget
        return widget.findChild(QObject, "buttonBox").findChildren(QPushButton)

    def test_initialize_buttons(self):
        self.widget.controls.append_to_data.setChecked(False)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.reference, self.data[:1])
        output = self.get_output(self.widget.Outputs.data).copy()

        buttons = self._get_init_buttons()

        buttons[3].click()  # Input
        output_input = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output_input, self.data[:1])

        buttons[0].click()  # Median
        output_median = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output_median, output)

        buttons[1].click()  # Mean
        output_mean = self.get_output(self.widget.Outputs.data)
        with output.unlocked():
            output.X = np.round(np.mean(self.data.X, axis=0), 1).reshape(1, 4)
        self.assert_table_equal(output_mean, output)

        buttons[2].click()  # Random
        output_random = self.get_output(self.widget.Outputs.data)
        self.assertTrue((self.data.X.max(axis=0) >= output_random.X).all())
        self.assertTrue((self.data.X.min(axis=0) <= output_random.X).all())

        self.send_signal(self.widget.Inputs.data, self.data[9:10])
        buttons[2].click()  # Random
        output_random = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output_random, self.data[9:10])

        self.send_signal(self.widget.Inputs.reference, None)
        buttons[3].click()  # Input
        output = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output_random, output)

    def test_initialize_buttons_commit_once(self):
        self.widget.commit.deferred = self.widget.commit.now = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.reference, self.data[:1])
        self.widget.commit.now.assert_called_once()

        self.widget.commit.now.reset_mock()
        buttons = self._get_init_buttons()
        buttons[3].click()  # Input
        self.widget.commit.deferred.assert_called_once()

    def test_table(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertEqual(self.widget.view.model().rowCount(), 5)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

        data = Table("zoo")
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(self.widget.view.model().rowCount(), 18)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.view.model().rowCount(), 0)
        self.assertEqual(self.widget.view.horizontalHeader().count(), 2)

    def test_table_data_changed(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        index = self.widget.model.index(0, 1)
        self.widget.model.setData(index, 7, role=ValueRole)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 151)
        self.assertEqual(output.X[150, 0], 7)

    def test_datasets(self):
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)

    def test_missing_values(self):
        domain = Domain([ContinuousVariable("c")],
                        class_vars=[DiscreteVariable("m", ("a", "b"))])
        data = Table(domain, np.array([[np.nan], [np.nan]]),
                     np.array([np.nan, np.nan]))
        self.widget.controls.append_to_data.setChecked(False)
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output, data[:1])
        self.assertTrue(self.widget.Information.nans_removed.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.nans_removed.is_shown())

    def test_missing_values_reference(self):
        reference = self.data[:1].copy()
        with reference.unlocked():
            reference[:] = np.nan
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.reference, reference)
        output1 = self.get_output(self.widget.Outputs.data)
        buttons = self._get_init_buttons()
        buttons[3].click()  # Input
        output2 = self.get_output(self.widget.Outputs.data)
        self.assert_table_equal(output1, output2)

    def test_saved_workflow(self):
        data = self.data
        with data.unlocked():
            data.X[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        buttons = self._get_init_buttons()
        buttons[2].click()  # Random
        output1 = self.get_output(self.widget.Outputs.data)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWCreateInstance, stored_settings=settings)
        self.send_signal(widget.Inputs.data, data, widget=widget)
        output2 = self.get_output(widget.Outputs.data)
        self.assert_table_equal(output1, output2)

    def test_commit_once(self):
        self.widget.commit.now = self.widget.commit.deferred = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.commit.now.assert_called_once()

        self.widget.commit.now.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.commit.deferred.assert_called_once()

        self.widget.commit.deferred.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.commit.deferred.assert_called_once()

    def test_context_menu(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.send_signal(self.widget.Inputs.reference, self.data[:1])
        output1 = self.get_output(self.widget.Outputs.data)
        self.widget.view.customContextMenuRequested.emit(QPoint(0, 0))
        menu = [w for w in self.widget.children() if isinstance(w, QMenu)][0]
        self.assertEqual(len(menu.actions()), 4)

        menu.actions()[3].trigger()  # Input
        output2 = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(output2.X[:, 1:], output1.X[:, 1:])
        np.testing.assert_array_equal(output2.X[150:, :1], self.data.X[:1, :1])

    def test_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()

    def test_sparse(self):
        data = self.data.to_sparse()
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.reference, data)


class TestDiscreteVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.editor = DiscreteVariableEditor(
            self.parent, ["Foo", "Bar"], self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, 0)
        self.assertEqual(self.editor._combo.currentText(), "Foo")
        self.callback.assert_not_called()

    def test_edit(self):
        """ Edit combo by user. """
        self.editor._combo.setCurrentText("Bar")
        self.assertEqual(self.editor.value, 1)
        self.assertEqual(self.editor._combo.currentText(), "Bar")
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set combo box value. """
        self.editor.value = 1
        self.assertEqual(self.editor.value, 1)
        self.assertEqual(self.editor._combo.currentText(), "Bar")
        self.callback.assert_called_once()


class TestContinuousVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        data = Table("iris")
        values = data.get_column_view(data.domain[0])[0]
        self.min_value = np.min(values)
        self.max_value = np.max(values)
        self.editor = ContinuousVariableEditor(
            self.parent, data.domain[0], self.min_value,
            self.max_value, self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, self.min_value)
        self.assertEqual(self.editor._slider.value(), self.min_value * 10)
        self.assertEqual(self.editor._spin.value(), self.min_value)
        self.callback.assert_not_called()

    def test_edit_slider(self):
        """ Edit slider by user. """
        self.editor._slider.setValue(int(self.max_value * 10))
        self.assertEqual(self.editor.value, self.max_value)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 2
        self.editor._slider.setValue(int(value * 10))
        self.assertEqual(self.editor.value, value)
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.callback.assert_called_once()

    def test_edit_spin(self):
        """ Edit spin by user. """
        self.editor._spin.setValue(self.max_value)
        self.assertEqual(self.editor.value, self.max_value)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        self.editor._spin.setValue(self.max_value + 1)
        self.assertEqual(self.editor.value, self.max_value + 1)
        self.assertEqual(self.editor._slider.value(), self.max_value * 10)
        self.assertEqual(self.editor._spin.value(), self.max_value + 1)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 2
        self.editor._spin.setValue(value)
        self.assertEqual(self.editor.value, value)
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set slider/spin value. """
        self.editor.value = -2
        self.assertEqual(self.editor._slider.value(), self.min_value * 10)
        self.assertEqual(self.editor._spin.value(), -2)
        self.assertEqual(self.editor.value, -2)
        self.callback.assert_called_once()

        self.callback.reset_mock()
        value = self.min_value + (self.max_value - self.min_value) / 4
        self.editor.value = value
        self.assertEqual(self.editor._slider.value(), value * 10)
        self.assertEqual(self.editor._spin.value(), value)
        self.assertEqual(self.editor.value, value)
        self.callback.assert_called_once()

    def test_missing_values(self):
        var = ContinuousVariable("var")
        self.assertRaises(ValueError, ContinuousVariableEditor, self.parent,
                          var, np.nan, np.nan, Mock())

    def test_overflow(self):
        var = ContinuousVariable("var", number_of_decimals=10)
        editor = ContinuousVariableEditor(
            self.parent, var, -100000, 1, self.callback
        )
        self.assertLess(editor._n_decimals, 10)

    def test_spin_selection_after_init(self):
        edit: QLineEdit = self.editor._spin.lineEdit()
        edit.selectAll()
        self.assertEqual(edit.selectedText(), "")
        self.assertIs(self.editor.focusProxy(), edit.parent())


class TestStringVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.editor = StringVariableEditor(self.parent, self.callback)

    def test_init(self):
        self.assertEqual(self.editor.value, "")
        self.assertEqual(self.editor._edit.text(), "")
        self.callback.assert_not_called()

    def test_edit(self):
        """ Set lineedit by user. """
        self.editor._edit.setText("Foo")
        self.assertEqual(self.editor.value, "Foo")
        self.assertEqual(self.editor._edit.text(), "Foo")
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set lineedit value. """
        self.editor.value = "Foo"
        self.assertEqual(self.editor.value, "Foo")
        self.assertEqual(self.editor._edit.text(), "Foo")
        self.callback.assert_called_once()


class TestTimeVariableEditor(GuiTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.parent = QWidget()

    def setUp(self):
        self.callback = Mock()
        self.editor = TimeVariableEditor(
            self.parent, TimeVariable("var", have_date=1), self.callback
        )

    def test_init(self):
        self.assertEqual(self.editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1)))
        self.callback.assert_not_called()

    def test_edit(self):
        """ Edit datetimeedit by user. """
        datetime = QDateTime(QDate(2001, 9, 9))
        self.editor._edit.setDateTime(datetime)
        self.assertEqual(self.editor.value, 999993600)
        self.assertEqual(self.editor._edit.dateTime(), datetime)
        self.callback.assert_called_once()

    def test_set_value(self):
        """ Programmatically set datetimeedit value. """
        value = 999993600
        self.editor.value = value
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(2001, 9, 9)))
        self.assertEqual(self.editor.value, value)
        self.callback.assert_called_once()

    def test_have_date_have_time(self):
        callback = Mock()
        editor = TimeVariableEditor(
            self.parent, TimeVariable("var", have_date=1, have_time=1),
            callback
        )
        self.assertEqual(editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1), QTime(0, 0, 0)))
        self.callback.assert_not_called()

        datetime = QDateTime(QDate(2001, 9, 9), QTime(1, 2, 3))
        editor._edit.setDateTime(datetime)
        self.assertEqual(editor._edit.dateTime(), datetime)
        self.assertEqual(editor.value, 999993600 + 3723)
        callback.assert_called_once()

    def test_have_time(self):
        callback = Mock()
        editor = TimeVariableEditor(
            self.parent, TimeVariable("var", have_time=1), callback
        )
        self.assertEqual(editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1), QTime(0, 0, 0)))
        self.callback.assert_not_called()

        datetime = QDateTime(QDate(1900, 1, 1), QTime(1, 2, 3))
        editor._edit.setDateTime(datetime)
        self.assertEqual(editor._edit.dateTime(), datetime)
        self.assertEqual(editor.value, 3723)
        callback.assert_called_once()

    def test_no_date_no_time(self):
        callback = Mock()
        editor = TimeVariableEditor(self.parent, TimeVariable("var"), callback)
        self.assertEqual(editor.value, 0)
        self.assertEqual(self.editor._edit.dateTime(),
                         QDateTime(QDate(1970, 1, 1), QTime(0, 0, 0)))
        self.callback.assert_not_called()

        datetime = QDateTime(QDate(2001, 9, 9), QTime(1, 2, 3))
        editor._edit.setDateTime(datetime)
        self.assertEqual(editor._edit.dateTime(), datetime)
        self.assertEqual(editor.value, 999993600 + 3723)
        callback.assert_called_once()


class TestVariableDelegate(GuiTest):
    def setUp(self):
        self.data = Table("iris")
        self.model = model = VariableItemModel()
        model.set_data(self.data)
        widget = OWCreateInstance()
        self.delegate = VariableDelegate(widget)
        self.parent = QWidget()
        self.opt = QStyleOptionViewItem()

    def test_create_editor(self):
        index = self.model.index(0, 1)
        editor = self.delegate.createEditor(self.parent, self.opt, index)
        self.assertIsInstance(editor, ContinuousVariableEditor)

        index = self.model.index(4, 1)
        editor = self.delegate.createEditor(self.parent, self.opt, index)
        self.assertIsInstance(editor, DiscreteVariableEditor)

    def test_set_editor_data(self):
        index = self.model.index(0, 1)
        editor = self.delegate.createEditor(self.parent, self.opt, index)
        self.delegate.setEditorData(editor, index)
        self.assertEqual(editor.value, np.median(self.data.X[:, 0]))

    def test_set_model_data(self):
        index = self.model.index(0, 1)
        editor = self.delegate.createEditor(self.parent, self.opt, index)
        editor.value = 7.5
        self.delegate.setModelData(editor, self.model, index)
        self.assertEqual(self.model.data(index, ValueRole), 7.5)

    def test_editor_geometry(self):
        index = self.model.index(0, 1)
        editor = self.delegate.createEditor(self.parent, self.opt, index)
        self.delegate.updateEditorGeometry(editor, self.opt, index)
        self.assertGreaterEqual(editor.geometry().width(),
                                self.opt.rect.width())

        size = self.delegate.sizeHint(self.opt, index)
        self.assertEqual(size.width(), editor.geometry().width())
        self.assertEqual(size.height(), 40)


if __name__ == "__main__":
    import unittest
    unittest.main()
