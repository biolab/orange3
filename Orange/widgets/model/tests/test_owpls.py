import unittest
import numpy as np

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
        self.assertEqual(coefsdata.name, "Coefficients and Loadings")
        self.assertEqual(coefsdata.X.shape, (14, 3))
        self.assertEqual(coefsdata.Y.shape, (14, 0))
        self.assertEqual(coefsdata.metas.shape, (14, 2))

        self.assertEqual(["coef (MEDV)", "Loading 1", "Loading 2"],
                         [v.name for v in coefsdata.domain.attributes])
        self.assertEqual(["Variable name", "Variable role"],
                         [v.name for v in coefsdata.domain.metas])
        metas = [v.name for v in self._data.domain.variables]
        self.assertTrue((coefsdata.metas[:, 0] == metas).all())
        self.assertTrue((coefsdata.metas[:-1, 1] == 0).all())
        self.assertTrue((coefsdata.metas[-1, 1] == 1))
        self.assertAlmostEqual(coefsdata.X[0, 1], 0.237, 3)
        self.assertAlmostEqual(coefsdata.X[13, 1], -0.304, 3)

    def test_output_coefsdata_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        coefsdata = self.get_output(self.widget.Outputs.coefsdata)
        self.assertEqual(coefsdata.name, "Coefficients and Loadings")
        self.assertEqual(coefsdata.X.shape, (14, 4))
        self.assertEqual(coefsdata.Y.shape, (14, 0))
        self.assertEqual(coefsdata.metas.shape, (14, 2))

        attr_names = ["coef (MEDV)", "coef (CRIM)", "Loading 1", "Loading 2"]
        self.assertEqual(attr_names,
                         [v.name for v in coefsdata.domain.attributes])
        self.assertEqual(["Variable name", "Variable role"],
                         [v.name for v in coefsdata.domain.metas])
        metas = [v.name for v in self._data_multi_target.domain.variables]
        self.assertTrue((coefsdata.metas[:, 0] == metas).all())
        self.assertTrue((coefsdata.metas[:-2, 1] == 0).all())
        self.assertTrue((coefsdata.metas[-2:, 1] == 1).all())
        self.assertAlmostEqual(coefsdata.X[0, 2], -0.198, 3)
        self.assertAlmostEqual(coefsdata.X[12, 2], -0.288, 3)
        self.assertAlmostEqual(coefsdata.X[13, 2], 0.243, 3)

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
        self.assertEqual(components.X.shape, (2, 13))
        self.assertEqual(components.Y.shape, (2,))
        self.assertEqual(components.metas.shape, (2, 1))

    def test_output_components_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        components = self.get_output(self.widget.Outputs.components)
        self.assertEqual(components.X.shape, (2, 12))
        self.assertEqual(components.Y.shape, (2, 2))
        self.assertEqual(components.metas.shape, (2, 1))

    def test_missing_target(self):
        data = self._data[:5].copy()
        data.Y[[0, 4]] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertFalse(np.isnan(output.metas[:, 3:].astype(float)).any())
        self.assertTrue(np.isnan(output.metas[0, 1:3].astype(float)).all())
        self.assertTrue(np.isnan(output.metas[4, 1:3].astype(float)).all())
        self.assertFalse(np.isnan(output.metas[1:4, 1:3].astype(float)).any())

        with data.unlocked(data.Y):
            data.Y[:] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))


if __name__ == "__main__":
    unittest.main()
