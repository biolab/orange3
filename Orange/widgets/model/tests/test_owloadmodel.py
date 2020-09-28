# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring
import os
import pickle
from tempfile import NamedTemporaryFile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from orangewidget.utils.filedialogs import RecentPath
from Orange.data import Table
from Orange.classification.naive_bayes import NaiveBayesLearner
from Orange.widgets.model.owloadmodel import OWLoadModel, OWLoadModelDropHandler
from Orange.widgets.tests.base import WidgetTest


class TestOWLoadModel(WidgetTest):
    # Attribute used to store event data so it does not get garbage
    # collected before event is processed.
    event_data = None

    def setUp(self):
        self.widget = self.create_widget(OWLoadModel)  # type: OWLoadModel
        data = Table("iris")
        self.model = NaiveBayesLearner()(data)
        with NamedTemporaryFile(suffix=".pkcls", delete=False) as f:
            self.filename = f.name
            pickle.dump(self.model, f)

    def tearDown(self):
        os.remove(self.filename)

    def test_browse_file_opens_file(self):
        w = self.widget
        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   Mock(return_value=(self.filename, "*.pkcls"))):
            w.browse_file()
            model = self.get_output(w.Outputs.model)
            np.testing.assert_equal(
                model.log_cont_prob, self.model.log_cont_prob)

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   Mock(return_value=("", "*.pkcls"))):
            w.browse_file()
            # Keep the same model on output
            model2 = self.get_output(w.Outputs.model)
            self.assertIs(model2, model)

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   Mock(return_value=(self.filename, "*.pkcls"))):
            w.reload()
            model2 = self.get_output(w.Outputs.model)
            self.assertIsNot(model2, model)

    @patch("pickle.load")
    def test_select_file(self, load):
        w = self.widget
        with NamedTemporaryFile(suffix=".pkcls") as f2, \
                NamedTemporaryFile(suffix=".pkcls", delete=False) as f3:
            w.add_path(self.filename)
            w.add_path(f2.name)
            w.add_path(f3.name)
            w.open_file()
            args = load.call_args[0][0]
            self.assertEqual(args.name, f3.name.replace("\\", "/"))
            w.select_file(2)
            args = load.call_args[0][0]
            self.assertEqual(args.name, self.filename.replace("\\", "/"))

    def test_load_error(self):
        w = self.widget
        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   Mock(return_value=(self.filename, "*.pkcls"))):
            with patch("pickle.load", side_effect=pickle.UnpicklingError):
                w.browse_file()
                self.assertTrue(w.Error.load_error.is_shown())
                self.assertIsNone(self.get_output(w.Outputs.model))

            w.reload()
            self.assertFalse(w.Error.load_error.is_shown())
            model = self.get_output(w.Outputs.model)
            self.assertIsNotNone(model)

            with patch.object(w, "last_path", Mock(return_value="")), \
                    patch("pickle.load") as load:
                w.reload()
                load.assert_not_called()
                self.assertFalse(w.Error.load_error.is_shown())
                self.assertIs(self.get_output(w.Outputs.model), model)

            with patch("pickle.load", side_effect=pickle.UnpicklingError):
                w.reload()
                self.assertTrue(w.Error.load_error.is_shown())
                self.assertIsNone(self.get_output(w.Outputs.model))

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileName",
                   Mock(return_value=("foo", "*.pkcls"))):
            w.browse_file()
            self.assertTrue(w.Error.load_error.is_shown())
            self.assertIsNone(self.get_output(w.Outputs.model))

    def test_no_last_path(self):
        self.widget = \
            self.create_widget(OWLoadModel,
                               stored_settings={"recent_paths": []})
        # Doesn't crash and contains a single item, (none).
        self.assertEqual(self.widget.file_combo.count(), 1)

    @patch("Orange.widgets.widget.OWWidget.workflowEnv",
           Mock(return_value={"basedir": os.getcwd()}))
    @patch("pickle.load")
    def test_open_moved_workflow(self, load):
        """
        Test opening workflow that has been moved to another location
        (i.e. sent by email), considering data file is stored in the same
        directory as the workflow.
        """
        temp_file = NamedTemporaryFile(dir=os.getcwd(), delete=False)
        file_name = temp_file.name
        temp_file.close()
        base_name = os.path.basename(file_name)
        try:
            recent_path = RecentPath(
                os.path.join("temp/models", base_name), "",
                os.path.join("models", base_name)
            )
            stored_settings = {"recent_paths": [recent_path]}
            w = self.create_widget(OWLoadModel,
                                   stored_settings=stored_settings)
            w.open_file()
            self.assertEqual(w.file_combo.count(), 1)
            args = load.call_args[0][0]
            self.assertEqual(args.name, file_name.replace("\\", "/"))
        finally:
            os.remove(file_name)


class TestOWLoadModelDropHandler(unittest.TestCase):
    def test_canDropFile(self):
        handler = OWLoadModelDropHandler()
        self.assertTrue(handler.canDropFile("test.pkcls"))
        self.assertFalse(handler.canDropFile("test.txt"))

    def test_parametersFromFile(self):
        handler = OWLoadModelDropHandler()
        res = handler.parametersFromFile("test.pkcls")
        self.assertEqual(res["recent_paths"][0].basename, "test.pkcls")


if __name__ == "__main__":
    unittest.main()
