import unittest
import numpy as np

from AnyQt.QtTest import QSignalSpy

from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.preprocess import Preprocess
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
        self.reset_tsne()

    def reset_tsne(self):
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
        self.reset_tsne()
        self.send_signal(self.widget.Inputs.data, self.data)
        if self.widget.isBlocking():
            spy = QSignalSpy(self.widget.blockingStateChanged)
            self.assertTrue(spy.wait(20000))
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertIsInstance(pp, Preprocess)
        transformed = pp(self.data)
        self.assertIsInstance(transformed, Table)
        self.assertEqual(transformed.X.shape, (len(self.data), 2))
        output = self.get_output(self.widget.Outputs.annotated_data)
        np.testing.assert_allclose(transformed.X, output.metas[:, :2],
                                   rtol=1, atol=1)
        self.assertEqual([a.name for a in transformed.domain.attributes],
                         [m.name for m in output.domain.metas[:2]])

    def test_multiscale_changed(self):
        self.assertFalse(self.widget.controls.multiscale.isChecked())
        self.assertTrue(self.widget.perplexity_spin.isEnabled())
        self.widget.controls.multiscale.setChecked(True)
        self.assertFalse(self.widget.perplexity_spin.isEnabled())

        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWtSNE, stored_settings=settings)
        self.assertTrue(w.controls.multiscale.isChecked())
        self.assertFalse(w.perplexity_spin.isEnabled())


if __name__ == '__main__':
    unittest.main()
