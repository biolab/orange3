# pylint: disable=protected-access,unused-import
from contextlib import contextmanager
import os
import pickle
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp

from AnyQt.QtGui import QFont, QTextDocumentFragment
from AnyQt.QtCore import QRectF, QPointF
from AnyQt.QtTest import QSignalSpy
from AnyQt.QtWidgets import (
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider
)

from orangewidget.widget import StateInfo
from orangewidget.tests.base import (
    GuiTest, WidgetTest as WidgetTestBase, DummySignalManager, DEFAULT_TIMEOUT
)
from Orange.base import SklModel, Model
from Orange.classification.base_classification import (
    LearnerClassification, ModelClassification
)
from Orange.data import (
    Table, Domain, DiscreteVariable, ContinuousVariable, Variable
)
from Orange.modelling import Fitter
from Orange.preprocess import RemoveNaNColumns, Randomize, Continuize
from Orange.preprocess.preprocess import PreprocessorList
from Orange.regression.base_regression import (
    LearnerRegression, ModelRegression
)
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.annotated_data import (
    ANNOTATED_DATA_FEATURE_NAME, ANNOTATED_DATA_SIGNAL_NAME
)
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.visualize.utils.plotutils import AnchorItem
from Orange.widgets.widget import OWWidget


class WidgetTest(WidgetTestBase):
    def assert_table_equal(self, table1, table2):
        if table1 is None or table2 is None:
            self.assertIs(table1, table2)
            return
        self.assert_domain_equal(table1.domain, table2.domain)
        np.testing.assert_array_equal(table1.X, table2.X)
        np.testing.assert_array_equal(table1.Y, table2.Y)
        np.testing.assert_array_equal(table1.metas, table2.metas)

    def assert_domain_equal(self, domain1, domain2):
        """
        Test domains for equality.

        Unlike in domain1 == domain2 uses `Variable.__eq__`, which in case of
        DiscreteVariable ignores `values`, this method also checks that both
        domain have equal `values`.
        """
        for var1, var2 in zip(domain1.variables + domain1.metas,
                              domain2.variables + domain2.metas):
            self.assertEqual(type(var1), type(var2))
            self.assertEqual(var1.name, var2.name)
            if var1.is_discrete:
                self.assertEqual(var1.values, var2.values)


class TestWidgetTest(WidgetTest):
    """Meta tests for widget test helpers"""

    def test_process_events_handles_timeouts(self):
        with self.assertRaises(TimeoutError):
            self.process_events(until=lambda: False, timeout=0)

    def test_minimum_size(self):
        return  # skip this test


class BaseParameterMapping:
    """Base class for mapping between gui components and learner's parameters
    when testing learner widgets.

    Parameters
    ----------
    name : str
        Name of learner's parameter.

    gui_element : QWidget
        Gui component who's corresponding parameter is to be tested.

    values: list
        List of values to be tested.

    getter: function
        It gets component's value.

    setter: function
        It sets component's value.
    """

    def __init__(self, name, gui_element, values, getter, setter,
                 problem_type="both"):
        self.name = name
        self.gui_element = gui_element
        self.values = values
        self.get_value = getter
        self.set_value = setter
        self.problem_type = problem_type

    def __str__(self):
        if self.problem_type == "both":
            return self.name
        else:
            return "%s (%s)" % (self.name, self.problem_type)


class DefaultParameterMapping(BaseParameterMapping):
    """Class for mapping between gui components and learner's parameters
    when testing unchecked properties and therefore default parameters
    should be used.

    Parameters
    ----------
    name : str
        Name of learner's parameter.

    default_value: str, int,
        Value that should be used by default.
    """

    def __init__(self, name, default_value):
        super().__init__(name, None, [default_value],
                         lambda: default_value, lambda x: None)


class ParameterMapping(BaseParameterMapping):
    """Class for mapping between gui components and learner parameters
    when testing learner widgets

    Parameters
    ----------
    name : str
        Name of learner's parameter.

    gui_element : QWidget
        Gui component who's corresponding parameter is to be tested.

    values: list, mandatory for ComboBox, optional otherwise
        List of values to be tested. When None, it is set according to
        component's type.

    getter: function, optional
        It gets component's value. When None, it is set according to
        component's type.

    setter: function, optional
        It sets component's value. When None, it is set according to
        component's type.
    """

    def __init__(self, name, gui_element, values=None,
                 getter=None, setter=None, **kwargs):
        super().__init__(
            name, gui_element,
            values or self._default_values(gui_element),
            getter or self._default_get_value(gui_element, values),
            setter or self._default_set_value(gui_element, values),
            **kwargs)

    @staticmethod
    def get_gui_element(widget, attribute):
        return widget.controlled_attributes[attribute][0].control

    @classmethod
    def from_attribute(cls, widget, attribute, parameter=None):
        return cls(parameter or attribute, cls.get_gui_element(widget, attribute))

    @staticmethod
    def _default_values(gui_element):
        if isinstance(gui_element, (QSpinBox, QDoubleSpinBox, QSlider)):
            return [gui_element.minimum(), gui_element.maximum()]
        else:
            raise TypeError("{} is not supported".format(gui_element))

    @staticmethod
    def _default_get_value(gui_element, values):
        if isinstance(gui_element, (QSpinBox, QDoubleSpinBox, QSlider)):
            return gui_element.value
        elif isinstance(gui_element, QComboBox):
            return lambda: values[gui_element.currentIndex()]
        else:
            raise TypeError("{} is not supported".format(gui_element))

    @staticmethod
    def _default_set_value(gui_element, values):
        if isinstance(gui_element, (QSpinBox, QDoubleSpinBox, QSlider)):
            return lambda val: gui_element.setValue(val)
        elif isinstance(gui_element, QComboBox):
            def fun(val):
                value = values.index(val)
                gui_element.activated.emit(value)
                gui_element.setCurrentIndex(value)

            return fun
        else:
            raise TypeError("{} is not supported".format(gui_element))


class WidgetLearnerTestMixin:
    """Base class for widget learner tests.

    Contains init method to set up testing parameters and test methods.

    All widget learner tests should extend it (beside extending WidgetTest
    class as well). Learners with extra parameters, which can be set on the
    widget, should override self.parameters list in the setUp method. The
    list should contain mapping: learner parameter - gui component.
    """

    widget = None  # type: OWBaseLearner

    def init(self):
        cls_ds = Table(datasets.path("testing_dataset_cls"))
        reg_ds = Table(datasets.path("testing_dataset_reg"))

        if issubclass(self.widget.LEARNER, Fitter):
            self.data = cls_ds
            self.valid_datasets = (cls_ds, reg_ds)
            self.inadequate_dataset = ()
            self.learner_class = Fitter
            self.model_class = Model
            self.model_name = 'Model'
        elif issubclass(self.widget.LEARNER, LearnerClassification):
            self.data = cls_ds
            self.valid_datasets = (cls_ds,)
            self.inadequate_dataset = (reg_ds,)
            self.learner_class = LearnerClassification
            self.model_class = ModelClassification
            self.model_name = 'Classifier'
        else:
            self.data = reg_ds
            self.valid_datasets = (reg_ds,)
            self.inadequate_dataset = (cls_ds,)
            self.learner_class = LearnerRegression
            self.model_class = ModelRegression
            self.model_name = 'Predictor'

        self.parameters = []

    def test_has_unconditional_apply(self):
        self.assertTrue(hasattr(self.widget, "unconditional_apply"))

    def test_input_data(self):
        """Check widget's data with data on the input"""
        self.assertEqual(self.widget.data, None)
        self.send_signal("Data", self.data)
        self.assertEqual(self.widget.data, self.data)
        self.wait_until_stop_blocking()

    def test_input_data_disconnect(self):
        """Check widget's data and model after disconnecting data from input"""
        self.send_signal("Data", self.data)
        self.assertEqual(self.widget.data, self.data)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        self.send_signal("Data", None)
        self.wait_until_stop_blocking()
        self.assertEqual(self.widget.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.model))

    def test_input_data_learner_adequacy(self):
        """Check if error message is shown with inadequate data on input"""
        for inadequate in self.inadequate_dataset:
            self.send_signal("Data", inadequate)
            self.widget.apply_button.button.click()
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.Error.data_error.is_shown())
        for valid in self.valid_datasets:
            self.send_signal("Data", valid)
            self.wait_until_stop_blocking()
            self.assertFalse(self.widget.Error.data_error.is_shown())

    def test_input_preprocessor(self):
        """Check learner's preprocessors with an extra pp on input"""
        randomize = Randomize()
        self.send_signal("Preprocessor", randomize)
        self.assertEqual(
            randomize, self.widget.preprocessors,
            'Preprocessor not added to widget preprocessors')
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        self.assertEqual(
            (randomize,), self.widget.learner.preprocessors,
            'Preprocessors were not passed to the learner')

    def test_input_preprocessors(self):
        """Check multiple preprocessors on input"""
        pp_list = PreprocessorList([Randomize(), RemoveNaNColumns()])
        self.send_signal("Preprocessor", pp_list)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        self.assertEqual(
            (pp_list,), self.widget.learner.preprocessors,
            '`PreprocessorList` was not added to preprocessors')

    def test_input_preprocessor_disconnect(self):
        """Check learner's preprocessors after disconnecting pp from input"""
        randomize = Randomize()
        self.send_signal("Preprocessor", randomize)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        self.assertEqual(randomize, self.widget.preprocessors)

        self.send_signal("Preprocessor", None)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.preprocessors,
                          'Preprocessors not removed on disconnect.')

    def test_output_learner(self):
        """Check if learner is on output after apply"""
        initial = self.get_output("Learner")
        self.assertIsNotNone(initial, "Does not initialize the learner output")
        self.widget.apply_button.button.click()
        newlearner = self.get_output("Learner")
        self.assertIsNot(initial, newlearner,
                         "Does not send a new learner instance on `Apply`.")
        self.assertIsNotNone(newlearner)
        self.assertIsInstance(newlearner, self.widget.LEARNER)

    def test_output_model(self):
        """Check if model is on output after sending data and apply"""
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.widget.apply_button.button.click()
        self.assertIsNone(self.get_output(self.widget.Outputs.model))
        self.send_signal('Data', self.data)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, self.widget.LEARNER.__returns__)
        self.assertIsInstance(model, self.model_class)

    def test_output_learner_name(self):
        """Check if learner's name properly changes"""
        new_name = "Learner Name"
        self.widget.apply_button.button.click()
        self.assertEqual(self.widget.learner.name,
                         self.widget.name_line_edit.text()
                         or self.widget.name_line_edit.placeholderText())
        self.widget.name_line_edit.setText(new_name)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        self.assertEqual(self.get_output("Learner").name, new_name)

    def test_output_model_name(self):
        """Check if model's name properly changes"""
        new_name = "Model Name"
        self.widget.name_line_edit.setText(new_name)
        self.send_signal("Data", self.data)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        self.assertEqual(self.get_output(self.widget.Outputs.model).name, new_name)

    def test_output_model_picklable(self):
        """Check if model can be pickled"""
        self.send_signal("Data", self.data)
        self.widget.apply_button.button.click()
        self.wait_until_stop_blocking()
        model = self.get_output(self.widget.Outputs.model)
        self.assertIsNotNone(model)
        pickle.dumps(model)

    @staticmethod
    def _get_param_value(learner, param):
        if isinstance(learner, Fitter):
            # Both is just a was to indicate to the tests, fitters don't
            # actually support this
            if param.problem_type == "both":
                problem_type = learner.CLASSIFICATION
            else:
                problem_type = param.problem_type
            return learner.get_params(problem_type).get(param.name)
        else:
            return learner.params.get(param.name)

    def test_parameters_default(self):
        """Check if learner's parameters are set to default (widget's) values
        """
        for dataset in self.valid_datasets:
            self.send_signal("Data", dataset)
            self.widget.apply_button.button.click()
            self.wait_until_stop_blocking()
            for parameter in self.parameters:
                # Skip if the param isn't used for the given data type
                if self._should_check_parameter(parameter, dataset):
                    self.assertEqual(
                        self._get_param_value(self.widget.learner, parameter),
                        parameter.get_value())

    def test_parameters(self):
        """Check learner and model for various values of all parameters"""
        # Test params on every valid dataset, since some attributes may apply
        # to only certain problem types
        for dataset in self.valid_datasets:
            self.send_signal("Data", dataset)
            self.wait_until_stop_blocking()

            for parameter in self.parameters:
                # Skip if the param isn't used for the given data type
                if not self._should_check_parameter(parameter, dataset):
                    continue

                assert isinstance(parameter, BaseParameterMapping)

                for value in parameter.values:
                    parameter.set_value(value)
                    self.widget.apply_button.button.click()
                    self.wait_until_stop_blocking()
                    param = self._get_param_value(self.widget.learner, parameter)
                    self.assertEqual(
                        param, parameter.get_value(),
                        "Mismatching setting for parameter '%s'" % parameter)
                    self.assertEqual(
                        param, value,
                        "Mismatching setting for parameter '%s'" % parameter)
                    param = self._get_param_value(self.get_output("Learner"), parameter)
                    self.assertEqual(
                        param, value,
                        "Mismatching setting for parameter '%s'" % parameter)

                    if issubclass(self.widget.LEARNER, SklModel):
                        model = self.get_output(self.widget.Outputs.model)
                        if model is not None:
                            self.assertEqual(self._get_param_value(model, parameter), value)
                            self.assertFalse(self.widget.Error.active)
                        else:
                            self.assertTrue(self.widget.Error.active)

    def test_params_trigger_settings_changed(self):
        """Check that the learner gets updated whenever a param is changed."""
        for dataset in self.valid_datasets:
            self.send_signal("Data", dataset)
            self.wait_until_stop_blocking()

            for parameter in self.parameters:
                # Skip if the param isn't used for the given data type
                if not self._should_check_parameter(parameter, dataset):
                    continue

                assert isinstance(parameter, BaseParameterMapping)
                # Set the mock here so we can include the param name in the
                # error message, so if any test fails, we see where
                # We mock `apply` and not `settings_changed` since that's
                # sometimes connected with Qt signals, which are not directly
                # called
                self.widget.apply = Mock(name="apply(%s)" % parameter)
                # Since the settings only get updated when the value actually
                # changes, find a value that isn't the same as the current
                # value and try with that
                new_value = [x for x in parameter.values
                             if x != parameter.get_value()][0]
                parameter.set_value(new_value)
                # wait for asynchronous calls
                self.process_events(lambda: self.widget.apply.call_args is not None)
                self.widget.apply.assert_called_once()

    @staticmethod
    def _should_check_parameter(parameter, data):
        """Should the param be passed into the learner given the data"""
        return ((parameter.problem_type == "classification" and
                 data.domain.has_discrete_class) or
                (parameter.problem_type == "regression" and
                 data.domain.has_continuous_class) or
                (parameter.problem_type == "both"))

    def test_send_report(self, timeout=DEFAULT_TIMEOUT):
        """Test report"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.report_button.click()
        self.wait_until_finished(timeout=timeout)
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()


class WidgetOutputsTestMixin:
    """Class for widget's outputs testing.

    Contains init method to set up testing parameters and a test method, which
    checks Selected Data and (Annotated) Data outputs.

    Since widgets have different ways of selecting data instances, _select_data
    method should be implemented when subclassed. The method should assign
    value to selected_indices parameter.

    If output's expected domain differs from input's domain, parameter
    same_input_output_domain should be set to False.

    If Selected Data and Data domains differ, override method
    _compare_selected_annotated_domains.
    """

    def init(self, same_table_attributes=True, output_all_on_no_selection=False):
        self.data = Table("iris")
        self.same_input_output_domain = True
        self.same_table_attributes = same_table_attributes
        self.output_all_on_no_selection = output_all_on_no_selection

    def test_outputs(self, timeout=DEFAULT_TIMEOUT):
        self.send_signal(self.signal_name, self.signal_data)

        self.wait_until_finished(timeout=timeout)

        # check selected data output
        output = self.get_output("Selected Data")
        if self.output_all_on_no_selection:
            self.assertEqual(output, self.signal_data)
        else:
            self.assertIsNone(output)

        # check annotated data output
        feature_name = ANNOTATED_DATA_FEATURE_NAME
        annotated = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(0, np.sum([i[feature_name] for i in annotated]))

        # select data instances
        selected_indices = self._select_data()

        # check selected data output
        selected = self.get_output("Selected Data")
        n_sel, n_attr = len(selected), len(self.data.domain.attributes)
        self.assertGreater(n_sel, 0)
        self.assertEqual(selected.domain == self.data.domain,
                         self.same_input_output_domain)
        np.testing.assert_array_equal(selected.X[:, :n_attr],
                                      self.data.X[selected_indices])
        if self.same_table_attributes:
            self.assertEqual(selected.attributes, self.data.attributes)

        # check annotated data output
        annotated = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        self.assertEqual(n_sel, np.sum([i[feature_name] for i in annotated]))
        if self.same_table_attributes:
            self.assertEqual(annotated.attributes, self.data.attributes)

        # compare selected and annotated data domains
        self._compare_selected_annotated_domains(selected, annotated)

        # check output when data is removed
        self.send_signal(self.signal_name, None)
        self.assertIsNone(self.get_output("Selected Data"))
        self.assertIsNone(self.get_output(ANNOTATED_DATA_SIGNAL_NAME))

    def _select_data(self):
        raise NotImplementedError("Subclasses should implement select_data")

    def _compare_selected_annotated_domains(self, selected, annotated):
        selected_vars = selected.domain.variables + selected.domain.metas
        annotated_vars = annotated.domain.variables + annotated.domain.metas
        self.assertLess(set(selected_vars), set(annotated_vars))


class ProjectionWidgetTestMixin:
    """Class for projection widget testing"""

    def init(self):
        self.data = Table("iris")

    def _select_data(self):
        rect = QRectF(QPointF(-20, -20), QPointF(20, 20))
        self.widget.graph.select_by_rectangle(rect)
        return self.widget.graph.get_selection()

    def _compare_selected_annotated_domains(self, selected, annotated):
        selected_vars = selected.domain.variables
        annotated_vars = annotated.domain.variables
        self.assertLessEqual(set(selected_vars), set(annotated_vars))

    def test_setup_graph(self, timeout=DEFAULT_TIMEOUT):
        """Plot should exist after data has been sent in order to be
        properly set/updated"""
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assertTrue(
            self.signal_manager.wait_for_finished(self.widget, timeout),
            f"Did not finish in the specified {timeout}ms timeout"
        )

        self.assertIsNotNone(self.widget.graph.scatterplot_item)

    def test_default_attrs(self, timeout=DEFAULT_TIMEOUT):
        """Check default values for 'Color', 'Shape', 'Size' and 'Label'"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIs(self.widget.attr_color, self.data.domain.class_var)
        self.assertIsNone(self.widget.attr_label)
        self.assertIsNone(self.widget.attr_shape)
        self.assertIsNone(self.widget.attr_size)
        self.wait_until_finished(timeout=timeout)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.attr_color)

    def test_attr_models(self):
        """Check possible values for 'Color', 'Shape', 'Size' and 'Label'"""
        self.send_signal(self.widget.Inputs.data, self.data)
        controls = self.widget.controls
        self.assertEqual(len(controls.attr_color.model()), 8)
        self.assertEqual(len(controls.attr_shape.model()), 3)
        self.assertTrue(5 < len(controls.attr_size.model()) < 8)
        self.assertEqual(len(controls.attr_label.model()), 8)

        # color and label should contain all variables
        # size should contain only continuous variables
        # shape should contain only discrete variables
        for var in self.data.domain.variables + self.data.domain.metas:
            self.assertIn(var, controls.attr_color.model())
            self.assertIn(var, controls.attr_label.model())
            if var.is_continuous:
                self.assertIn(var, controls.attr_size.model())
                self.assertNotIn(var, controls.attr_shape.model())
            if var.is_discrete:
                self.assertNotIn(var, controls.attr_size.model())
                self.assertIn(var, controls.attr_shape.model())

    def test_attr_label_metas(self, timeout=DEFAULT_TIMEOUT):
        """Set 'Label' from string meta attribute"""
        cont = Continuize(multinomial_treatment=Continuize.AsOrdinal)
        data = cont(Table("zoo"))
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished(timeout=timeout)
        simulate.combobox_activate_item(self.widget.controls.attr_label,
                                        data.domain[-1].name)

    def test_handle_primitive_metas(self):
        """Set 'Color' from continuous meta attribute"""
        d, attrs = self.data.domain, self.data.domain.attributes
        data = self.data.transform(Domain(attrs[:2], d.class_vars, attrs[2:]))
        self.send_signal(self.widget.Inputs.data, data)
        simulate.combobox_activate_item(self.widget.controls.attr_color,
                                        data.domain.metas[0].name)

    def test_datasets(self, timeout=DEFAULT_TIMEOUT):
        """Test widget for datasets with missing values and constant features"""
        for ds in datasets.datasets():
            self.send_signal(self.widget.Inputs.data, ds)
            self.wait_until_finished(timeout=timeout)

    def test_none_data(self):
        """Test widget for empty dataset"""
        self.send_signal(self.widget.Inputs.data, self.data[:0])

    def test_plot_once(self, timeout=DEFAULT_TIMEOUT):
        """Test if data is plotted only once but committed on every input change"""
        table = Table("heart_disease")
        self.widget.setup_plot = Mock()
        self.widget.commit.now = self.widget.commit.deferred = Mock()
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.now.assert_called_once()

        self.wait_until_finished(timeout=timeout)
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.now.assert_called_once()

        self.widget.commit.now.reset_mock()
        self.send_signal(self.widget.Inputs.data_subset, table[::10])
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.now.assert_called_once()

    def test_subset_data_color(self, timeout=DEFAULT_TIMEOUT):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTrue(
            self.signal_manager.wait_for_finished(self.widget, timeout),
            f"Did not finish in the specified {timeout}ms timeout"
        )
        self.send_signal(self.widget.Inputs.data_subset, self.data[:10])
        subset = [brush.color().name() == "#46befa" for brush in
                  self.widget.graph.scatterplot_item.data['brush'][:10]]
        other = [brush.color().name() == "#000000" for brush in
                 self.widget.graph.scatterplot_item.data['brush'][10:]]
        self.assertTrue(all(subset))
        self.assertTrue(all(other))

    def test_class_density(self, timeout=DEFAULT_TIMEOUT):
        """Check class density update"""
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.cb_class_density.click()
        self.wait_until_finished(timeout=timeout)
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.cb_class_density.click()

    def test_dragging_tooltip(self):
        """Dragging tooltip depends on data being jittered"""
        text = QTextDocumentFragment.fromHtml(self.widget.graph.tiptexts[0]).toPlainText()
        self.send_signal(self.widget.Inputs.data, Table("heart_disease"))
        self.assertEqual(self.widget.graph.tip_textitem.toPlainText(), text)

    def test_sparse_data(self, timeout=DEFAULT_TIMEOUT):
        """Test widget for sparse data"""
        table = Table("iris").to_sparse()
        self.assertTrue(sp.issparse(table.X))
        self.send_signal(self.widget.Inputs.data, table)
        self.wait_until_finished(timeout=timeout)
        self.send_signal(self.widget.Inputs.data_subset, table[::30])
        self.assertEqual(len(self.widget.subset_data), 5)

    def test_invalidated_embedding(self, timeout=DEFAULT_TIMEOUT):
        """Check if graph has been replotted when sending same data"""
        self.widget.graph.update_coordinates = Mock()
        self.widget.graph.update_point_props = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished(timeout=timeout)

        self.widget.graph.update_coordinates.assert_called()
        self.widget.graph.update_point_props.assert_called()

        self.widget.graph.update_coordinates.reset_mock()
        self.widget.graph.update_point_props.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished(timeout=timeout)

        self.widget.graph.update_coordinates.assert_not_called()
        self.widget.graph.update_point_props.assert_called_once()

    def test_saved_selection(self, timeout=DEFAULT_TIMEOUT):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTrue(
            self.signal_manager.wait_for_finished(self.widget, timeout),
            f"Did not finish in the specified {timeout}ms timeout"
        )

        self.widget.graph.select_by_indices(list(range(0, len(self.data), 10)))
        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(self.widget.__class__, stored_settings=settings)

        self.send_signal(self.widget.Inputs.data, self.data, widget=w)
        self.assertTrue(
            self.signal_manager.wait_for_finished(w, timeout),
            f"Did not finish in the specified {timeout}ms timeout"
        )

        self.assertEqual(np.sum(w.graph.selection), 15)
        np.testing.assert_equal(self.widget.graph.selection, w.graph.selection)

    def test_send_report(self, timeout=DEFAULT_TIMEOUT):
        """Test report """
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.report_button.click()
        self.wait_until_finished(timeout=timeout)
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()

    def test_hidden_effective_variables(self, timeout=DEFAULT_TIMEOUT):
        hidden_var1 = ContinuousVariable("c1")
        hidden_var1.attributes["hidden"] = True
        hidden_var2 = ContinuousVariable("c2")
        hidden_var2.attributes["hidden"] = True
        class_vars = [DiscreteVariable("cls", values=("a", "b"))]
        table = Table(Domain([hidden_var1, hidden_var2], class_vars),
                      np.array([[0., 1.], [2., 3.]]),
                      np.array([[0.], [1.]]))
        self.send_signal(self.widget.Inputs.data, table)
        self.wait_until_finished(timeout=timeout)
        self.send_signal(self.widget.Inputs.data, table)

    def test_visual_settings(self, timeout=DEFAULT_TIMEOUT):
        graph = self.widget.graph
        font = QFont()
        font.setItalic(True)
        font.setFamily("Helvetica")

        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished(timeout=timeout)

        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Title", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(20)
        self.assertFontEqual(graph.parameter_setter.title_item.item.font(), font)

        key, value = ("Fonts", "Label", "Font size"), 10
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Label", "Italic"), True
        self.widget.set_visual_settings(key, value)
        simulate.combobox_activate_item(self.widget.controls.attr_label,
                                        self.data.domain[0].name)
        font.setPointSize(10)
        self.assertFontEqual(graph.labels[0].textItem.font(), font)

        key, value = ("Fonts", "Categorical legend", "Font size"), 14
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Categorical legend", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(14)
        legend_item = list(graph.parameter_setter.cat_legend_items)[0]
        self.assertFontEqual(legend_item[1].item.font(), font)

        key, value = ("Fonts", "Numerical legend", "Font size"), 12
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Numerical legend", "Italic"), True
        self.widget.set_visual_settings(key, value)
        simulate.combobox_activate_item(self.widget.controls.attr_color,
                                        self.data.domain[0].name)
        variables = self.data.domain.variables + self.data.domain.metas
        discrete_var = next(
            (x for x in variables if isinstance(x, DiscreteVariable)), None
        )
        if discrete_var:
            # activate item only when discrete variable available
            simulate.combobox_activate_item(
                self.widget.controls.attr_shape, discrete_var.name
            )
        font.setPointSize(12)
        self.assertFontEqual(graph.parameter_setter.num_legend.items[0][0].font, font)

        key, value = ("Annotations", "Title", "Title"), "Foo"
        self.widget.set_visual_settings(key, value)
        self.assertEqual(graph.parameter_setter.title_item.item.toPlainText(), "Foo")
        self.assertEqual(graph.parameter_setter.title_item.text, "Foo")

    def assertFontEqual(self, font1, font2):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())


class AnchorProjectionWidgetTestMixin(ProjectionWidgetTestMixin):
    def test_embedding_missing_values(self):
        table = Table("heart_disease")
        with table.unlocked():
            table.X[0] = np.nan
        self.send_signal(self.widget.Inputs.data, table)
        self.assertFalse(np.all(self.widget.valid_data))
        output = self.get_output(ANNOTATED_DATA_SIGNAL_NAME)
        embedding_mask = np.all(np.isnan(output.metas[:, :2]), axis=1)
        np.testing.assert_array_equal(~embedding_mask, self.widget.valid_data)
        # reload
        self.send_signal(self.widget.Inputs.data, table)

    def test_sparse_data(self, timeout=DEFAULT_TIMEOUT):
        table = Table("iris")
        with table.unlocked():
            table.X = sp.csr_matrix(table.X)
        self.assertTrue(sp.issparse(table.X))
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.sparse_data.is_shown())
        self.send_signal(self.widget.Inputs.data_subset, table[::30])
        self.assertEqual(len(self.widget.subset_data), 5)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.sparse_data.is_shown())

    def test_manual_move(self):
        data = self.data.copy()
        with data.unlocked():
            data[1, 0] = np.nan
        nvalid, nsample = len(self.data) - 1, self.widget.SAMPLE_SIZE
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.graph.select_by_indices(list(range(0, len(data), 10)))

        # remember state
        selection = self.widget.graph.selection.copy()

        # simulate manual move
        self.widget._manual_move_start()
        self.widget._manual_move(0, 1, 1)
        self.assertEqual(len(self.widget.graph.scatterplot_item.data), nsample)
        self.widget._manual_move_finish(0, 1, 2)

        # check new state
        self.assertEqual(len(self.widget.graph.scatterplot_item.data), nvalid)
        np.testing.assert_equal(self.widget.graph.selection, selection)

    def test_visual_settings(self, timeout=DEFAULT_TIMEOUT):
        super().test_visual_settings(timeout)

        graph = self.widget.graph
        font = QFont()
        font.setItalic(True)
        font.setFamily("Helvetica")

        key, value = ("Fonts", "Anchor", "Font size"), 10
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Anchor", "Italic"), True
        self.widget.set_visual_settings(key, value)
        font.setPointSize(10)
        for item in graph.anchor_items:
            if isinstance(item, AnchorItem):
                item = item._label
            self.assertFontEqual(item.textItem.font(), font)


class datasets:
    @staticmethod
    def path(filename):
        dirname = os.path.join(os.path.dirname(__file__), "datasets")
        return os.path.join(dirname, filename)

    @classmethod
    def missing_data_1(cls):
        """
        Dataset with 3 continuous features (X{1,2,3}) where all the columns
        and rows contain at least one NaN value.

        One discrete class D with NaN values
        Mixed continuous/discrete/string metas ({X,D,S}M)

        Returns
        -------
        data : Orange.data.Table
        """
        return Table(cls.path("missing_data_1.tab"))

    @classmethod
    def missing_data_2(cls):
        """
        Dataset with 3 continuous features (X{1,2,3}) where all the columns
        and rows contain at least one NaN value and X1, X2 are constant.

        One discrete constant class D with NaN values.
        Mixed continuous/discrete/string class metas ({X,D,S}M)

        Returns
        -------
        data : Orange.data.Table
        """
        return Table(cls.path("missing_data_2.tab"))

    @classmethod
    def missing_data_3(cls):
        """
        Dataset with 3 discrete features D{1,2,3} where all the columns and
        rows contain at least one NaN value

        One discrete class D with NaN values
        Mixes continuous/discrete/string metas ({X,D,S}M)

        Returns
        -------
        data : Orange.data.Table
        """
        return Table(cls.path("missing_data_3.tab"))

    @classmethod
    def data_one_column_vals(cls, value=np.nan):
        """
        Dataset with two continuous features and one discrete. One continuous
        columns has custom set values (default nan).

        Returns
        -------
        data : Orange.data.Table
        """
        table = Table.from_list(
            Domain(
                [ContinuousVariable("a"),
                 ContinuousVariable("b"),
                 DiscreteVariable("c", values=("y", "n"))]
            ),
            list(zip(
                [42.48, 16.84, 15.23, 23.8],
                ["", "", "", ""],
                "ynyn"
            )))
        with table.unlocked():
            table[:, 1] = value
        return table

    @classmethod
    def data_one_column_nans(cls):
        """
        Dataset with two continuous features and one discrete. One continuous
        columns has missing values (NaN).

        Returns
        -------
        data : Orange.data.Table
        """
        return cls.data_one_column_vals(value=np.nan)

    @classmethod
    def data_one_column_infs(cls):
        return cls.data_one_column_vals(value=np.inf)

    @classmethod
    def datasets(cls):
        """
        Yields multiple datasets.

        Returns
        -------
        data : Generator of Orange.data.Table
        """
        ds_cls = Table(cls.path("testing_dataset_cls"))
        ds_reg = Table(cls.path("testing_dataset_reg"))
        for ds in (ds_cls, ds_reg):
            d, a = ds.domain, ds.domain.attributes
            for i in range(0, len(a), 2):
                yield ds.transform(Domain(a[i: i + 2], d.class_vars, d.metas))
            yield ds.transform(Domain(a[:2] + a[8: 10], d.class_vars, d.metas))
        yield cls.missing_data_1()
        yield cls.missing_data_2()
        yield cls.missing_data_3()
        yield cls.data_one_column_nans()
        yield ds_cls
        yield ds_reg


@contextmanager
def open_widget_classes():
    with patch.object(OWWidget, "__init_subclass__"):
        yield
