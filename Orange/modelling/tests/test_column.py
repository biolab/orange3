import unittest
from unittest.mock import patch

import numpy as np

from Orange.data import DiscreteVariable, ContinuousVariable, Table, Domain
from Orange.modelling.column import _check_column_combinations, \
    valid_prob_range, valid_value_sets, ColumnLearner, ColumnModel


class TestBase(unittest.TestCase):
    def setUp(self):
        self.disc_a = DiscreteVariable("a", values=("a", "b", "c"))
        self.disc_b = DiscreteVariable("b", values=("c", "a", "b"))
        self.disc_c = DiscreteVariable("c", values=("c", "b"))
        self.cont_e = ContinuousVariable("e")
        self.cont_f = ContinuousVariable("f")
        self.cont_g = ContinuousVariable("g")


class TestColumnLearner(TestBase):
    @patch("Orange.modelling.column._check_column_combinations")
    def test_column_regressor(self, check):
        data = Table.from_numpy(
            Domain([self.disc_a, self.cont_e, self.cont_f],
                   self.cont_g),
            np.array([[0, 1, -6],
                      [1, 2, -4],
                      [2, 4, 0],
                      [np.nan, 6, 4],
                      [0, 3, -2]]),
            np.array([0, 1, 3, 5, 2]))

        model = ColumnLearner(self.cont_g, self.cont_e, True)(data)
        check.assert_called()
        self.assertIs(model.class_var, self.cont_g)
        self.assertIs(model.column, self.cont_e)
        self.assertAlmostEqual(model.intercept, -1)
        self.assertAlmostEqual(model.coefficient, 1)

        model = ColumnLearner(self.cont_g, self.cont_f, True)(data)
        self.assertIs(model.class_var, self.cont_g)
        self.assertIs(model.column, self.cont_f)
        self.assertAlmostEqual(model.intercept, 3)
        self.assertAlmostEqual(model.coefficient, 0.5)

        check.reset_mock()
        model = ColumnLearner(self.cont_g, self.cont_f)(data)
        check.assert_called()
        self.assertIs(model.class_var, self.cont_g)
        self.assertIs(model.column, self.cont_f)
        self.assertIsNone(model.intercept)
        self.assertIsNone(model.coefficient)

    @patch("Orange.modelling.column._check_column_combinations")
    def test_column_classifier_from_numeric(self, check):
        data = Table.from_numpy(
            Domain([self.disc_a, self.disc_b, self.cont_e],
                   self.disc_c),
            np.array([[0, 1, 0],
                      [1, 0, 1],
                      [2, 1, 1],
                      [0, 2, 0],
                      [0, 0, 1]]),
            np.array([0, 1, 0, 1, 0]))

        model = ColumnLearner(self.disc_c, self.cont_e, True)(data)
        check.assert_called()
        self.assertIs(model.class_var, self.disc_c)
        self.assertIs(model.column, self.cont_e)
        # These values were not computed manually
        self.assertAlmostEqual(model.intercept, -0.3127646959895215)
        self.assertAlmostEqual(model.coefficient, -0.15535275317811897)

    @patch("Orange.modelling.column._check_column_combinations")
    def test_column_classifier_from_discrete(self, check):
        data = Table.from_numpy(
            Domain([self.disc_b, self.disc_c, self.cont_e],
                   self.disc_a),
            np.array([[0, 1, 0],
                      [1, 0, 1],
                      [2, 1, 1],
                      [0, 0, 0],
                      [0, 0, 1]]),
        np.array([0, 1, 2, 1, 0]))
        model = ColumnLearner(self.disc_a, self.disc_c)(data)
        check.assert_called()
        self.assertIs(model.class_var, self.disc_a)
        self.assertIs(model.column, self.disc_c)
        self.assertIsNone(model.intercept)
        self.assertIsNone(model.coefficient)

    def test_class_mismatch(self):
        data = Table.from_numpy(
            Domain([self.disc_b, self.disc_c, self.cont_e],
                   self.disc_a),
            np.array([[0, 1, 0],
                      [1, 0, 1],
                      [2, 1, 1],
                      [0, 0, 0],
                      [0, 0, 1]]),
        np.array([0, 1, 2, 1, 0]))

        self.assertRaises(
            ValueError,
            ColumnLearner(self.disc_b, self.disc_c),
            data)


class TestModel(TestBase):
    def test_str(self):
        model = ColumnModel(self.disc_a, self.disc_c)
        self.assertEqual(str(model), "ColumnModel c")

        model = ColumnModel(self.disc_c, self.cont_e, 1, 2)
        self.assertEqual(str(model), "ColumnModel e (1, 2)")

    def test_mapping(self):
        model = ColumnModel(self.disc_a,
                            DiscreteVariable("a2", self.disc_a.values))
        self.assertIsNone(model.value_mapping)
        model = ColumnModel(self.disc_c, self.cont_e)
        self.assertIsNone(model.value_mapping)
        model = ColumnModel(self.disc_c, self.cont_e, 1, 2)
        self.assertIsNone(model.value_mapping)
        model = ColumnModel(self.disc_b, self.disc_c)
        np.testing.assert_equal(model.value_mapping, [0, 2])
        model = ColumnModel(self.disc_a, self.disc_c)
        np.testing.assert_equal(model.value_mapping, [2, 1])
        model = ColumnModel(self.disc_b, self.disc_a)
        np.testing.assert_equal(model.value_mapping, [1, 2, 0])

    @patch("Orange.modelling.column._check_column_combinations")
    def test_check_validity(self, check):
        ColumnModel(self.disc_c, self.cont_e)
        check.assert_called()

        with self.assertRaises(ValueError):
            ColumnModel(self.disc_b, self.cont_e, 0, None)
        with self.assertRaises(ValueError):
            ColumnModel(self.disc_b, self.cont_e, None, 1)

    def test_name(self):
        self.assertEqual(
            ColumnModel(self.disc_c, self.cont_e).name,
            "column 'e'")

        self.assertEqual(
            ColumnModel(self.disc_c, self.cont_e, -1, 2).name,
            "column 'e' (-1, 2)")

    @patch("Orange.modelling.column.ColumnModel._predict_discrete")
    @patch("Orange.modelling.column.ColumnModel._predict_continuous")
    def test_predict_storage(self, predict_cont, predict_disc):
        model = ColumnModel(self.cont_e, self.cont_f)
        data = Table.from_numpy(
            Domain([self.cont_f], self.cont_e),
            np.array([[1], [2], [3]]),
            np.array([0, 1, 2]))
        model.predict_storage(data)
        predict_cont.assert_called()
        predict_cont.reset_mock()
        predict_disc.assert_not_called()

        model = ColumnModel(self.disc_a, self.disc_c)
        data = Table.from_numpy(
            Domain([self.disc_c], self.disc_a),
            np.array([[0], [1], [0]]),
            np.array([0, 1, 0]))
        model.predict_storage(data)
        predict_disc.assert_called()
        predict_cont.assert_not_called()

    def test_predict_disc_from_disc_w_mapping(self):
        data = Table.from_numpy(
            Domain([self.disc_a, self.disc_c, self.cont_e],
                   self.disc_b),
            np.array([[0, 1, 0],  # a, b, 0
                      [1, 0, 1],  # b c 1
                      [2, 1, 1],  # c b 1
                      [0, 0, 0],  # a c 0
                      [0, 0, 1]]),  # a c 1
            np.array([0, 1, 0, 1, 0]))

        model = ColumnModel(self.disc_b, self.disc_c)
        vals, probs = model.predict_storage(data)
        np.testing.assert_equal(vals, [2, 0, 2, 0, 0])
        np.testing.assert_almost_equal(probs, [[0, 0, 1],
                                               [1, 0, 0],
                                               [0, 0, 1],
                                               [1, 0, 0],
                                               [1, 0, 0]])

        model = ColumnModel(self.disc_b, self.disc_a)
        vals, probs = model.predict_storage(data)
        np.testing.assert_equal(vals, [1, 2, 0, 1, 1])
        np.testing.assert_almost_equal(probs, [[0, 1, 0],
                                               [0, 0, 1],
                                               [1, 0, 0],
                                               [0, 1, 0],
                                               [0, 1, 0]])

    def test_predict_disc_from_disc_wout_mapping(self):
        data = Table.from_numpy(
            Domain([self.disc_a, self.disc_c, self.cont_e],
                   DiscreteVariable("a1", values=self.disc_a.values)),
            np.array([[0, 1, 0],
                      [1, 0, 1],
                      [2, 1, 1],
                      [0, 0, 0],
                      [0, 0, 1]]),
            np.array([0, 1, 0, 1, 0]))

        model = ColumnModel(data.domain.class_var, self.disc_a)
        vals, probs = model.predict_storage(data)
        np.testing.assert_equal(vals, [0, 1, 2, 0, 0])
        np.testing.assert_almost_equal(probs, [[1, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 1],
                                               [1, 0, 0],
                                               [1, 0, 0]])

    def test_predict_disc_from_cont(self):
        data = Table.from_numpy(
            Domain([self.cont_e, self.cont_f],
                   self.disc_c),
            np.array([[0, -1],
                      [1, 0],
                      [0.45, 1],
                      [np.nan, 2],
                      [0.52, np.nan]]),
            np.array([0, 1, 0, 1, 0]))

        model = ColumnModel(data.domain.class_var, self.cont_e)
        vals, probs = model.predict_storage(data)
        p1 = np.array([0, 1, 0.45, np.nan, 0.52])
        np.testing.assert_equal(vals, [0, 1, 0, np.nan, 1])
        np.testing.assert_almost_equal(probs, np.vstack((1 - p1, p1)).T)

        model = ColumnModel(self.disc_c, self.cont_e, -0.1, 5)
        vals, probs = model.predict_storage(data)
        p1e = 1 / (1 + np.exp(+0.1 - 5 * p1))
        np.testing.assert_equal(vals, [0, 1, 1, np.nan, 1])
        np.testing.assert_almost_equal(probs, np.vstack((1 - p1e, p1e)).T)

        with self.assertRaises(ValueError):
            # values outside [0, 1] range
            ColumnModel(self.disc_c, self.cont_f)(data)

        model = ColumnModel(self.disc_c, self.cont_f, -1, 5)
        vals, probs = model.predict_storage(data)
        p1 = 1 / (1 + np.exp(1 - 5 * data.X[:, 1]))
        np.testing.assert_equal(vals, [0, 0, 1, 1, np.nan])
        np.testing.assert_almost_equal(probs, np.vstack((1 - p1, p1)).T)

    def test_predict_cont(self):
        data = Table.from_numpy(
            Domain([self.cont_e, self.cont_f],
                   self.cont_g),
            np.array([[0, -1],
                      [1, 0],
                      [0.45, 1],
                      [np.nan, 2],
                      [0.52, np.nan]]),
            np.array([0, 1, 0.45, 2, 0.52]))

        model = ColumnModel(self.cont_g, self.cont_e)
        np.testing.assert_equal(model.predict_storage(data), data.X[:, 0])

        model = ColumnModel(self.cont_g, self.cont_f, -0.1, 5)
        np.testing.assert_almost_equal(
            model.predict_storage(data),
            -0.1 + 5 * data.X[:, 1])


class Test(TestBase):
    def test_checks(self):
        def check(class_var, column, fit=False):
            _check_column_combinations(class_var, column, fit)

        def value_error(*args):
            self.assertRaises(ValueError, check, *args)

        value_error(self.cont_e, self.disc_a)  # regression from discrete column
        value_error(self.disc_a, self.cont_e)  # non-binary class from numeric
        value_error(self.disc_c, self.disc_a)  # column has vales not in class
        value_error(self.disc_a, self.disc_b, True) # fitting from discrete column

        check(self.cont_e, self.cont_f)
        check(self.cont_e, self.cont_f, True)
        check(self.disc_a, self.disc_b)
        check(self.disc_a, self.disc_c)

    def test_valid_prob_range(self):
        self.assertTrue(valid_prob_range(np.array([1, 0, 0.5])))
        self.assertFalse(valid_prob_range(np.array([-0.1, 0.5, 1])))
        self.assertFalse(valid_prob_range(np.array([0, 1.1, 0.5])))

    def test_valid_value_sets(self):
        self.assertTrue(valid_value_sets(self.disc_a, self.disc_b))
        self.assertTrue(valid_value_sets(self.disc_a, self.disc_c))
        self.assertFalse(valid_value_sets(self.disc_c, self.disc_a))


if __name__ == '__main__':
    unittest.main()
