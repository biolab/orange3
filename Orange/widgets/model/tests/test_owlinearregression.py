from unittest.mock import patch

import numpy as np

from Orange.data import Table

from Orange.widgets.model.owlinearregression import OWLinearRegression
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin


class TestOWLinearRegression(WidgetTest, WidgetLearnerTestMixin):
    def setUp(self):
        self.widget: OWLinearRegression = \
            self.create_widget(OWLinearRegression,
                               stored_settings={"auto_apply": False})
        self.init()

    @patch("Orange.regression.linear.OLSRegressionLearner._compute_pvalues")
    def test_p_values(self, _compute_pvalues):
        self.widget.auto_apply = True

        p_values = np.array([0.1, 0.2, 0.3])
        _compute_pvalues.return_value = p_values

        x = np.array([
            [8, 9, 9, 9, 3],
            [8, -6, 0, -6, 3],
            [-10, -1, 2, -4, -4],
            [-4, 1, 6, 0, -8],
            [-4, -10, -6, 6, 4],
            [7, 6, -5, -1, -1],
            [-3, -10, -4, -3, -8],
            [8, -4, -6, -2, 3],
            [8, 9, 4, 2, 0],
            [-7, 5, -9, 7, -6]])
        y = np.array(
            [95.09214324, 8.95689637, -57.17925557, -38.71883822,
             38.33283747, -17.89237331, -71.98402682, 8.92551392,
             27.29567782, -47.20940499])

        self.widget.reg_type = OWLinearRegression.OLS
        data = Table.from_numpy(None, x, y)
        self.send_signal(self.widget.Inputs.data, data)
        out = self.get_output(self.widget.Outputs.coefficients)
        self.assertEqual(len(out.domain.attributes), 2)
        np.testing.assert_array_equal(out.get_column(1), p_values)

        self.widget.reg_type = OWLinearRegression.Ridge
        self.widget._reg_type_changed()
        out = self.get_output(self.widget.Outputs.coefficients)
        self.assertEqual(len(out.domain.attributes), 1)

        self.widget.reg_type = OWLinearRegression.Lasso
        self.widget._reg_type_changed()
        out = self.get_output(self.widget.Outputs.coefficients)
        self.assertEqual(len(out.domain.attributes), 1)


if __name__ == "__main__":  # pragma: no cover
    import unittest
    unittest.main()