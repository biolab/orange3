# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

from Orange.data import (Table, Domain, StringVariable,
                         DiscreteVariable, ContinuousVariable)
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.utils.annotated_data import (ANNOTATED_DATA_FEATURE_NAME)
from Orange.widgets.visualize.owvenndiagram import (reshape_wide,
                                                    table_concat,
                                                    varying_between,
                                                    drop_columns,
                                                    OWVennDiagram,
                                                    group_table_indices)
from Orange.tests import test_filename


class TestVennDiagram(unittest.TestCase):
    def add_metas(self, table, meta_attrs, meta_data):
        domain = Domain(table.domain.attributes,
                        table.domain.class_vars,
                        table.domain.metas + meta_attrs)
        metas = np.hstack((table.metas, meta_data))
        return Table(domain, table.X, table.Y, metas)

    def test_reshape_wide(self):
        class_var = DiscreteVariable("c", values=["x"])
        item_id_var = StringVariable("item_id")
        source_var = StringVariable("source")
        c1, c, item_id, ca, cb = np.random.randint(10, size=5)
        data = Table(Domain([ContinuousVariable("c1")], [class_var],
                            [DiscreteVariable("c(a)", class_var.values),
                             DiscreteVariable("c(b)", class_var.values),
                             source_var, item_id_var]),
                     np.array([[c1], [c1]], dtype=object),
                     np.array([[c], [c]], dtype=object),
                     np.array([[ca, np.nan, "a", item_id],
                               [np.nan, cb, "b", item_id]], dtype=object))

        data = reshape_wide(data, [], [item_id_var], [source_var])
        self.assertFalse(any(np.isnan(data.metas.astype(np.float32)[0])))
        self.assertEqual(len(data), 1)
        np.testing.assert_equal(data.metas, np.array([[ca, cb, item_id]],
                                                     dtype=object))

    def test_reshape_wide_missing_vals(self):
        data = Table(test_filename("test9.tab"))
        reshaped_data = reshape_wide(data, [], [data.domain[0]],
                                     [data.domain[0]])
        self.assertEqual(2, len(reshaped_data))

    def test_varying_between_missing_vals(self):
        data = Table(test_filename("test9.tab"))
        self.assertEqual(6, len(varying_between(data, data.domain[0])))

    def test_venn_diagram(self):
        sources = ["SVM Learner", "Naive Bayes", "Random Forest"]
        item_id_var = StringVariable("item_id")
        source_var = StringVariable("source")
        table = Table("zoo")
        class_var = table.domain.class_var
        cv = np.random.randint(len(class_var.values), size=(3, len(sources)))

        tables = []
        for i in range(len(sources)):
            temp_table = Table.from_table(table.domain, table,
                                          [0 + i, 1 + i, 2 + i])
            temp_d = (DiscreteVariable("%s(%s)" % (class_var.name,
                                                   sources[0 + i]),
                                       class_var.values),
                      source_var, item_id_var)
            temp_m = np.array([[cv[0, i], sources[i], table.metas[0 + i, 0]],
                               [cv[1, i], sources[i], table.metas[1 + i, 0]],
                               [cv[2, i], sources[i], table.metas[2 + i, 0]]],
                              dtype=object)
            temp_table = self.add_metas(temp_table, temp_d, temp_m)
            tables.append(temp_table)

        data = table_concat(tables)
        varying = varying_between(data, item_id_var)
        if source_var in varying:
            varying.remove(source_var)
        data = reshape_wide(data, varying, [item_id_var], [source_var])
        data = drop_columns(data, [item_id_var])

        result = np.array([[table.metas[0, 0], cv[0, 0], np.nan, np.nan],
                           [table.metas[1, 0], cv[1, 0], cv[0, 1], np.nan],
                           [table.metas[2, 0], cv[2, 0], cv[1, 1], cv[0, 2]],
                           [table.metas[3, 0], np.nan, cv[2, 1], cv[1, 2]],
                           [table.metas[4, 0], np.nan, np.nan, cv[2, 2]]],
                          dtype=object)

        for i in range(len(result)):
            for j in range(len(result[0])):
                val = result[i][j]
                if isinstance(val, float) and np.isnan(val):
                    self.assertTrue(np.isnan(data.metas[i][j]))
                else:
                    np.testing.assert_equal(data.metas[i][j], result[i][j])


class TestOWVennDiagram(WidgetTest, WidgetOutputsTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)

        cls.signal_data = cls.data[:25]

    def setUp(self):
        self.widget = self.create_widget(OWVennDiagram)
        self.signal_name = self.widget.Inputs.data

    def _select_data(self):
        self.widget.vennwidget.vennareas()[1].setSelected(True)
        return list(range(len(self.signal_data)))

    def test_multiple_input(self):
        self.send_signal(self.signal_name, self.data[:100], 1)
        self.send_signal(self.signal_name, self.data[50:], 2)

        # check selected data output
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # check annotated data output
        feature_name = ANNOTATED_DATA_FEATURE_NAME
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(0, np.sum([i[feature_name] for i in annotated]))

        # select data instances
        self.widget.vennwidget.vennareas()[3].setSelected(True)
        selected_indices = list(range(50, 100))

        # check selected data output
        selected = self.get_output(self.widget.Outputs.selected_data)
        n_sel, n_attr = len(selected), len(self.data.domain.attributes)
        self.assertGreater(n_sel, 0)
        self.assertEqual(selected.domain == self.data.domain,
                         self.same_input_output_domain)
        np.testing.assert_array_equal(selected.X[:, :n_attr],
                                      self.data.X[selected_indices])

        # check annotated data output
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(n_sel, np.sum([i[feature_name] for i in annotated]))

        # compare selected and annotated data domains
        self._compare_selected_annotated_domains(selected, annotated)

        # check output when data is removed
        self.send_signal(self.signal_name, None, 1)
        self.send_signal(self.signal_name, None, 2)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_no_data(self):
        """Check that the widget doesn't crash on empty data"""
        self.send_signal(self.signal_name, self.data[:0], 1)
        self.send_signal(self.signal_name, self.data[:100], 2)
        self.send_signal(self.signal_name, self.data[50:], 3)

        for i in range(1, 4):
            self.send_signal(self.signal_name, None, i)

        self.send_signal(self.signal_name, self.data[:100], 1)
        self.send_signal(self.signal_name, self.data[:0], 1)
        self.send_signal(self.signal_name, self.data[50:], 3)

        for i in range(1, 4):
            self.send_signal(self.signal_name, None, i)

        self.send_signal(self.signal_name, self.data[:100], 1)
        self.send_signal(self.signal_name, self.data[50:], 2)
        self.send_signal(self.signal_name, self.data[:0], 3)


class GroupTableIndicesTest(unittest.TestCase):

    def test_varying_between_combined(self):
        X = np.array([[0, 0, 0, 0, 0, 1,],
                      [0, 0, 1, 1, 0, 1,],
                      [0, 0, 0, 2, np.nan, np.nan,],
                      [0, 1, 0, 0, 0, 0,],
                      [0, 1, 0, 2, 0, 0,],
                      [0, 1, 0, 0, np.nan, 0,]])

        M = np.array([["A", 0, 0, 0, 0, 0, 1,],
                      ["A", 0, 0, 1, 1, 0, 1,],
                      ["A", 0, 0, 0, 2, np.nan, np.nan,],
                      ["B", 0, 1, 0, 0, 0, 0,],
                      ["B", 0, 1, 0, 2, 0, 0,],
                      ["B", 0, 1, 0, 0, np.nan, 0,]], dtype=str)

        variables = [ContinuousVariable(name="F%d" % j) for j in range(X.shape[1])]
        metas = [StringVariable(name="M%d" % j) for j in range(M.shape[1])]
        domain = Domain(attributes=variables, metas=metas)

        data = Table.from_numpy(X=X, domain=domain, metas=M)

        self.assertEqual(varying_between(data, idvar=data.domain.metas[0]),
                         [variables[2], variables[3], metas[3], metas[4], metas[5], metas[6]])

        data = Table.from_numpy(X=sp.csr_matrix(X), domain=domain, metas=M)
        self.assertEqual(varying_between(data, idvar=data.domain.metas[0]),
                         [variables[2], variables[3], metas[3], metas[4], metas[5], metas[6]])


    def test_group_table_indices(self):
        table = Table(test_filename("test9.tab"))
        dd = defaultdict(list)
        dd["1"] = [0, 1]
        dd["huh"] = [2]
        dd["hoy"] = [3]
        dd["?"] = [4]
        dd["2"] = [5]
        dd["oh yeah"] = [6]
        dd["3"] = [7]
        self.assertEqual(dd, group_table_indices(table, "g"))
