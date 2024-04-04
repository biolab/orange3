import unittest

from Orange.data import Table, Domain, StringVariable
from Orange.widgets.model.owpls import OWPLS
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, \
    ParameterMapping


class TestOWPLS(WidgetTest, WidgetLearnerTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._data = Table("housing")
        cls._data = cls._data.add_column(StringVariable("Foo"),
                                         ["Bar"] * len(cls._data),
                                         to_metas=True)
        class_vars = [cls._data.domain.class_var,
                      cls._data.domain.attributes[0]]
        domain = Domain(cls._data.domain.attributes[1:], class_vars,
                        cls._data.domain.metas)
        cls._data_multi_target = cls._data.transform(domain)

    def setUp(self):
        self.widget = self.create_widget(OWPLS,
                                         stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            ParameterMapping('max_iter', self.widget.controls.max_iter),
            ParameterMapping('n_components', self.widget.controls.n_components)
        ]

    def test_output_coefsdata(self):
        self.send_signal(self.widget.Inputs.data, self._data)
        coefsdata = self.get_output(self.widget.Outputs.coefsdata)
        self.assertEqual(coefsdata.X.shape, (13, 1))
        self.assertEqual(coefsdata.Y.shape, (13, 0))
        self.assertEqual(coefsdata.metas.shape, (13, 1))

    def test_output_coefsdata_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        coefsdata = self.get_output(self.widget.Outputs.coefsdata)
        self.assertEqual(coefsdata.X.shape, (12, 2))
        self.assertEqual(coefsdata.Y.shape, (12, 0))
        self.assertEqual(coefsdata.metas.shape, (12, 1))

    def test_output_data(self):
        self.send_signal(self.widget.Inputs.data, self._data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(output.X.shape, (506, 13))
        self.assertEqual(output.Y.shape, (506,))
        self.assertEqual(output.metas.shape, (506, 8))
        self.assertEqual([v.name for v in self._data.domain.variables],
                         [v.name for v in output.domain.variables])
        metas = ["PLS U1", "PLS U2", "PLS T1", "PLS T2",
                 "Sample Quantiles (MEDV)", "Theoretical Quantiles (MEDV)",
                 "DModX"]
        self.assertEqual([v.name for v in self._data.domain.metas] + metas,
                         [v.name for v in output.domain.metas])

    def test_output_data_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(output.X.shape, (506, 12))
        self.assertEqual(output.Y.shape, (506, 2))
        self.assertEqual(output.metas.shape, (506, 10))
        orig_domain = self._data_multi_target.domain
        self.assertEqual([v.name for v in orig_domain.variables],
                         [v.name for v in output.domain.variables])
        metas = ["PLS U1", "PLS U2", "PLS T1", "PLS T2",
                 "Sample Quantiles (MEDV)", "Theoretical Quantiles (MEDV)",
                 "Sample Quantiles (CRIM)", "Theoretical Quantiles (CRIM)",
                 "DModX"]
        self.assertEqual([v.name for v in orig_domain.metas] + metas,
                         [v.name for v in output.domain.metas])

    def test_output_components(self):
        self.send_signal(self.widget.Inputs.data, self._data)
        components = self.get_output(self.widget.Outputs.components)
        self.assertEqual(components.X.shape, (4, 13))
        self.assertEqual(components.Y.shape, (4,))
        self.assertEqual(components.metas.shape, (4, 1))

    def test_output_components_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        components = self.get_output(self.widget.Outputs.components)
        self.assertEqual(components.X.shape, (4, 12))
        self.assertEqual(components.Y.shape, (4, 2))
        self.assertEqual(components.metas.shape, (4, 1))

    def test_output_loadings(self):
        self.send_signal(self.widget.Inputs.data, self._data)
        loadings = self.get_output(self.widget.Outputs.loadings)
        self.assertEqual(loadings.name, "Loadings")
        self.assertEqual(loadings.X.shape, (14, 2))
        self.assertEqual(loadings.Y.shape, (14, 0,))
        self.assertEqual(loadings.metas.shape, (14, 1))
        self.assertEqual(["Loading 1", "Loading 2"],
                         [v.name for v in loadings.domain.attributes])
        self.assertEqual(["Feature name"],
                         [v.name for v in loadings.domain.metas])
        self.assertAlmostEqual(loadings.X[0, 0], 0.237, 3)
        self.assertAlmostEqual(loadings.X[13, 0], -0.304, 3)

    def test_output_loadings_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        loadings = self.get_output(self.widget.Outputs.loadings)
        self.assertEqual(loadings.name, "Loadings")
        self.assertEqual(loadings.X.shape, (14, 2))
        self.assertEqual(loadings.Y.shape, (14, 0))
        self.assertEqual(loadings.metas.shape, (14, 1))
        self.assertEqual(["Loading 1", "Loading 2"],
                         [v.name for v in loadings.domain.attributes])
        self.assertEqual(["Feature name"],
                         [v.name for v in loadings.domain.metas])
        metas = [[v.name] for v in self._data_multi_target.domain.variables]
        self.assertTrue((loadings.metas == metas).all())
        self.assertAlmostEqual(loadings.X[0, 0], -0.198, 3)
        self.assertAlmostEqual(loadings.X[12, 0], -0.288, 3)
        self.assertAlmostEqual(loadings.X[13, 0], 0.243, 3)


if __name__ == "__main__":
    unittest.main()
