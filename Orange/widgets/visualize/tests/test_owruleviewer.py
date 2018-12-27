# pylint: disable=missing-docstring,protected-access
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.classification import CN2Learner
from Orange.widgets.visualize.owruleviewer import OWRuleViewer


class TestOWRuleViewer(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.titanic = Table('titanic')
        cls.learner = CN2Learner()
        cls.classifier = cls.learner(cls.titanic)
        # CN2Learner does not add `instances` attribute to the model, but
        # the Rules widget does. We simulate the model we get from the widget.
        cls.classifier.instances = cls.titanic

        cls.signal_name = "Classifier"
        cls.signal_data = cls.classifier
        cls.data = cls.titanic

    def setUp(self):
        self.widget = self.create_widget(OWRuleViewer)

    def test_set_data(self):
        # data must be None before assignment
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # assign None data
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # assign data
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertEqual(self.titanic, self.widget.data)

        # output signal should not be sent without a classifier
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # remove data
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_set_classifier(self):
        # classifier must be None before assignment
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.widget.classifier)
        self.assertIsNone(self.widget.selected)

        # assign the classifier
        self.send_signal(self.widget.Inputs.classifier, self.classifier)
        self.assertIsNone(self.widget.data)
        self.assertIsNotNone(self.widget.classifier)
        self.assertIsNone(self.widget.selected)

        # without data also set, the output should be None
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_filtered_data_output(self):
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.send_signal(self.widget.Inputs.classifier, self.classifier)

        # select the last rule (TRUE)
        selection_model = self.widget.view.selectionModel()
        selection_model.select(
            self.widget.proxy_model.index(
                len(self.classifier.rule_list) - 1, 0),
            selection_model.Select | selection_model.Rows)

        # the number of output data instances (filtered)
        # must match the size of titanic data-set
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(self.titanic), len(output))

        # clear selection,
        selection_model.clearSelection()

        # output should now be None
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_copy_to_clipboard(self):
        self.send_signal(self.widget.Inputs.classifier, self.classifier)

        # select the last rule (TRUE)
        selection_model = self.widget.view.selectionModel()
        selection_model.select(
            self.widget.proxy_model.index(
                len(self.classifier.rule_list) - 1, 0),
            selection_model.Select | selection_model.Rows)

        # copy the selection and test if correct
        self.widget.copy_to_clipboard()
        clipboard_contents = QApplication.clipboard().text()
        self.assertTrue(self.classifier.rule_list[-1].__str__() ==
                        clipboard_contents)

    def test_restore_original_order(self):
        self.send_signal(self.widget.Inputs.classifier, self.classifier)
        bottom_row = len(self.classifier.rule_list) - 1

        # sort the table
        self.widget.proxy_model.sort(0, Qt.AscendingOrder)

        # bottom row QIndex
        q_index = self.widget.proxy_model.index(bottom_row, 0)
        self.assertEqual(bottom_row, q_index.row())

        # translate to TableModel QIndex
        q_index = self.widget.proxy_model.mapToSource(q_index)

        # the row indices do NOT match
        self.assertNotEqual(bottom_row, q_index.row())

        # restore original order
        self.widget.restore_original_order()

        # repeat the process
        q_index = self.widget.proxy_model.index(bottom_row, 0)
        self.assertEqual(bottom_row, q_index.row())

        # translate to TableModel QIndex
        q_index = self.widget.proxy_model.mapToSource(q_index)

        # the row indices now match
        self.assertEqual(bottom_row, q_index.row())

    def test_selection_compact_view(self):
        self.send_signal(self.widget.Inputs.classifier, self.classifier)

        # test that selection persists through view change
        selection_model = self.widget.view.selectionModel()
        selection_model.select(self.widget.proxy_model.index(0, 0),
                               selection_model.Select | selection_model.Rows)

        self.widget._save_selected(actual=True)
        temp = self.widget.selected

        # update (compact view)
        self.widget.on_update()
        self.widget._save_selected(actual=True)

        # test that the selection persists
        self.assertEqual(temp, self.widget.selected)

    def _select_data(self):
        selection_model = self.widget.view.selectionModel()
        selection_model.select(self.widget.proxy_model.index(2, 0),
                               selection_model.Select | selection_model.Rows)
        return list(range(586, 597))
