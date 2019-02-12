import unittest
from unittest.mock import patch
import numpy as np

from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.preprocess import Preprocess, Normalize
from Orange.projection.manifold import TSNE
from Orange.widgets.tests.base import (
    WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
)
from Orange.widgets.unsupervised import owtsne
from Orange.widgets.unsupervised.owtsne import OWtSNE


class TestOWtSNE(WidgetTest, ProjectionWidgetTestMixin,
                 WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.same_input_output_domain = False

        cls.signal_name = "Data"
        cls.signal_data = cls.data

    def setUp(self):
        def fit(*args, **_):
            return np.ones((len(args[1]), 2), float)

        def transform(*args, **_):
            return np.ones((len(args[1]), 2), float)

        def optimize(*_, **__):
            return TSNE()()

        self._fit = owtsne.TSNE.fit
        self._transform = owtsne.TSNEModel.transform
        self._optimize = owtsne.TSNEModel.optimize
        owtsne.TSNE.fit = fit
        owtsne.TSNEModel.transform = transform
        owtsne.TSNEModel.optimize = optimize

        self.widget = self.create_widget(OWtSNE,
                                         stored_settings={"multiscale": False})

        self.class_var = DiscreteVariable('Stage name', values=['STG1', 'STG2'])
        self.attributes = [ContinuousVariable('GeneName' + str(i)) for i in range(5)]
        self.domain = Domain(self.attributes, class_vars=self.class_var)
        self.empty_domain = Domain([], class_vars=self.class_var)

    def tearDown(self):
        self.restore_mocked_functions()

    def restore_mocked_functions(self):
        owtsne.TSNE.fit = self._fit
        owtsne.TSNEModel.transform = self._transform
        owtsne.TSNEModel.optimize = self._optimize

    def test_wrong_input(self):
        # no data
        self.data = None
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNone(self.widget.data)

        # <2 rows
        self.data = Table(self.domain, [[1, 2, 3, 4, 5, 'STG1']])
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_rows.is_shown())

        # no attributes
        self.data = Table(self.empty_domain, [['STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.no_attributes.is_shown())

        # constant data
        self.data = Table(self.domain, [[1, 2, 3, 4, 5, 'STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.constant_data.is_shown())

        # correct input
        self.data = Table(self.domain, [[1, 2, 3, 4, 5, 'STG1'],
                                        [5, 4, 3, 2, 1, 'STG1']])
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertIsNotNone(self.widget.data)
        self.assertFalse(self.widget.Error.not_enough_rows.is_shown())
        self.assertFalse(self.widget.Error.no_attributes.is_shown())
        self.assertFalse(self.widget.Error.constant_data.is_shown())

    def test_input(self):
        self.data = Table(self.domain, [[1, 1, 1, 1, 1, 'STG1'],
                                        [2, 2, 2, 2, 2, 'STG1'],
                                        [4, 4, 4, 4, 4, 'STG2'],
                                        [5, 5, 5, 5, 5, 'STG2']])

        self.send_signal(self.widget.Inputs.data, self.data)

    def test_attr_models(self):
        """Check possible values for 'Color', 'Shape', 'Size' and 'Label'"""
        self.send_signal(self.widget.Inputs.data, self.data)
        controls = self.widget.controls
        for var in self.data.domain.class_vars + self.data.domain.metas:
            self.assertIn(var, controls.attr_color.model())
            self.assertIn(var, controls.attr_label.model())
            if var.is_continuous:
                self.assertIn(var, controls.attr_size.model())
                self.assertNotIn(var, controls.attr_shape.model())
            if var.is_discrete:
                self.assertNotIn(var, controls.attr_size.model())
                self.assertIn(var, controls.attr_shape.model())

    def test_output_preprocessor(self):
        # To test the validity of the preprocessor, we'll have to actually
        # compute the projections
        self.restore_mocked_functions()

        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking(wait=20000)
        output_data = self.get_output(self.widget.Outputs.annotated_data)

        # We send the same data to the widget, we expect the point locations to
        # be fairly close to their original ones
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsInstance(pp, Preprocess)

        transformed_data = pp(self.data)
        self.assertIsInstance(transformed_data, Table)
        self.assertEqual(transformed_data.X.shape, (len(self.data), 2))
        np.testing.assert_allclose(transformed_data.X, output_data.metas[:, :2],
                                   rtol=1, atol=3)
        self.assertEqual([a.name for a in transformed_data.domain.attributes],
                         [m.name for m in output_data.domain.metas[:2]])

    def test_multiscale_changed(self):
        self.assertFalse(self.widget.controls.multiscale.isChecked())
        self.assertTrue(self.widget.perplexity_spin.isEnabled())
        self.widget.controls.multiscale.setChecked(True)
        self.assertFalse(self.widget.perplexity_spin.isEnabled())

        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWtSNE, stored_settings=settings)
        self.assertTrue(w.controls.multiscale.isChecked())
        self.assertFalse(w.perplexity_spin.isEnabled())

    def test_normalize_data(self):
        # Normalization should be checked by default
        self.assertTrue(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_called_once()

        # Disable checkbox
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.controls.normalize.isChecked())
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_not_called()

        # Normalization shouldn't work on sparse data
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.controls.normalize.isChecked())

        sparse_data = self.data.to_sparse()
        with patch("Orange.preprocess.preprocess.Normalize", wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, sparse_data)
            self.assertFalse(self.widget.controls.normalize.isEnabled())
            normalize.assert_not_called()

    @patch("Orange.projection.manifold.TSNEModel.optimize")
    def test_exaggeration_is_passed_through_properly(self, optimize):
        def _check_exaggeration(call, exaggeration):
            # Check the last call to `optimize`, so we catch one during the
            # regular regime
            _, _, kwargs = call.mock_calls[-1]
            self.assertIn("exaggeration", kwargs)
            self.assertEqual(kwargs["exaggeration"], exaggeration)

        # Set value to 1
        self.widget.controls.exaggeration.setValue(1)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.commit_and_wait()
        _check_exaggeration(optimize, 1)

        # Reset and clear state
        optimize.reset_mock()
        self.send_signal(self.widget.Inputs.data, None)

        # Change to 3
        self.widget.controls.exaggeration.setValue(3)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.commit_and_wait()
        _check_exaggeration(optimize, 3)


if __name__ == '__main__':
    unittest.main()
