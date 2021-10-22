# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object
import unittest
from unittest.mock import patch
from pickle import loads, dumps

import numpy as np

from AnyQt.QtCore import Qt, QPoint
from AnyQt.QtTest import QTest

from Orange.data import (Table, Domain, ContinuousVariable as Cv,
                         StringVariable as sv, DiscreteVariable as Dv,
                         TimeVariable as Tv)
from Orange.widgets.data.owpivot import (OWPivot, Pivot,
                                         AggregationFunctionsEnum)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate


class TestOWPivot(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPivot)
        self.agg_checkboxes = self.widget.aggregation_checkboxes
        self.assertGreater(len(self.agg_checkboxes), 0)
        self.iris = Table("iris")
        self.heart_disease = Table("heart_disease")
        self.zoo = Table("zoo")

    def test_comboboxes(self):
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        controls = self.widget.controls
        name = self.heart_disease.domain.class_var.name
        self.assertEqual(controls.row_feature.currentText(), name)
        self.assertEqual(controls.col_feature.currentText(), "(Same as rows)")
        self.assertEqual(controls.val_feature.currentText(), "age")

        self.assertEqual(len(controls.row_feature.model()), 15)
        self.assertEqual(len(controls.col_feature.model()), 11)
        self.assertEqual(len(controls.val_feature.model()), 17)

        domain = self.heart_disease.domain
        for var in domain.variables + domain.metas:
            self.assertIn(var, controls.val_feature.model())
            if var.is_continuous:
                self.assertIn(var, controls.row_feature.model())
                self.assertNotIn(var, controls.col_feature.model())
            elif var.is_discrete:
                self.assertIn(var, controls.row_feature.model())
                self.assertIn(var, controls.col_feature.model())

    def test_feature_combinations(self):
        for cb in self.agg_checkboxes[1:]:
            cb.click()
        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_run_through_all(self.widget.controls.row_feature)
        simulate.combobox_run_through_all(self.widget.controls.col_feature)
        simulate.combobox_run_through_all(self.widget.controls.val_feature)

    def test_output_grouped_data(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.agg_checkboxes[Pivot.Sum.value].click()
        grouped = self.get_output(self.widget.Outputs.grouped_data)
        names = ["iris", "(count)", "sepal length (sum)", "sepal width (sum)",
                 "petal length (sum)", "petal width (sum)"]
        self.assertListEqual(names, [a.name for a in grouped.domain.variables])
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.grouped_data))

    def test_output_grouped_data_time_var(self):
        domain = Domain([Dv("d1", ("a", "b")), Tv("t1", have_date=1)])
        X = np.array([[0, 1e9], [0, 1e8], [1, 2e8], [1, np.nan]])
        data = Table(domain, X)
        self.send_signal(self.widget.Inputs.data, data)
        self.agg_checkboxes[Pivot.Functions.Mean.value].click()
        grouped = self.get_output(self.widget.Outputs.grouped_data)
        str_grouped = "[[a, 2, 1987-06-06],\n [b, 2, 1976-05-03]]"
        self.assertEqual(str(grouped), str_grouped)

    def test_output_filtered_data(self):
        self.agg_checkboxes[Pivot.Functions.Sum.value].click()
        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_activate_item(self.widget.controls.row_feature,
                                        self.iris.domain.attributes[0].name)
        simulate.combobox_activate_item(self.widget.controls.col_feature,
                                        self.iris.domain.class_var.name)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.attributes[1].name)
        self.assertIsNone(self.get_output(self.widget.Outputs.filtered_data))

        self.widget.table_view.set_selection(set([(11, 0), (11, 1), (12, 0),
                                                  (12, 1), (13, 0), (13, 1),
                                                  (14, 0), (14, 1)]))
        self.widget.table_view.selection_changed.emit()
        output = self.get_output(self.widget.Outputs.filtered_data)
        self.assertEqual(output.X.shape, (20, 4))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.filtered_data))

    def test_output_pivot_table(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.attributes[0].name)
        table = self.get_output(self.widget.Outputs.pivot_table)
        names = ["iris", "Aggregate", "Iris-setosa",
                 "Iris-versicolor", "Iris-virginica"]
        self.assertListEqual(names, [a.name for a in table.domain.variables])
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.pivot_table))

    def test_pivot_table_cont_row(self):
        for cb in self.agg_checkboxes[1:]:
            cb.click()
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Warning.cannot_aggregate.is_shown())
        simulate.combobox_activate_item(self.widget.controls.row_feature,
                                        self.iris.domain.attributes[0].name)
        self.assertTrue(self.widget.Warning.no_col_feature.is_shown())
        simulate.combobox_activate_item(self.widget.controls.col_feature,
                                        self.iris.domain.class_var.name)
        self.assertFalse(self.widget.Warning.no_col_feature.is_shown())

        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.attributes[1].name)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.class_var.name)

    def test_pivot_table_disc_row(self):
        for cb in self.agg_checkboxes[1:]:
            cb.click()
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Warning.cannot_aggregate.is_shown())
        simulate.combobox_activate_item(self.widget.controls.col_feature,
                                        self.iris.domain.class_var.name)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.attributes[1].name)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.class_var.name)

        self.send_signal(self.widget.Inputs.data, self.zoo)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.zoo.domain.metas[0].name)
        simulate.combobox_activate_item(self.widget.controls.col_feature,
                                        self.zoo.domain.attributes[0].name)

    def test_aggregations(self):
        # agg: Count, feature: Continuous
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertFalse(self.widget.Warning.cannot_aggregate.is_shown())
        # agg: Count, Sum, feature: Continuous
        self.agg_checkboxes[Pivot.Sum.value].click()
        self.assertFalse(self.widget.Warning.cannot_aggregate.is_shown())
        # agg: Count, Sum, Majority, feature: Continuous
        self.agg_checkboxes[Pivot.Majority.value].click()
        self.assertTrue(self.widget.Warning.cannot_aggregate.is_shown())
        # agg: Count, Majority, feature: Continuous
        self.agg_checkboxes[Pivot.Sum.value].click()
        self.assertTrue(self.widget.Warning.cannot_aggregate.is_shown())
        # agg: Count, Majority, feature: Discrete
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.class_var.name)
        self.assertFalse(self.widget.Warning.cannot_aggregate.is_shown())
        # agg: Count, Majority, feature: None
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        "(None)")
        self.assertTrue(self.widget.Warning.cannot_aggregate.is_shown())
        # agg: Count, Majority, feature: None, row: Continuous
        simulate.combobox_activate_item(self.widget.controls.row_feature,
                                        self.iris.domain.attributes[1].name)
        self.assertFalse(self.widget.Warning.cannot_aggregate.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.cannot_aggregate.is_shown())

    @patch("Orange.widgets.data.owpivot.Pivot._initialize",
           return_value=(None, None))
    def test_group_table_created_once(self, initialize):
        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_activate_item(self.widget.controls.row_feature,
                                        self.iris.domain.attributes[0].name)
        simulate.combobox_activate_item(self.widget.controls.col_feature,
                                        self.iris.domain.class_var.name)
        initialize.assert_called_with(set([Pivot.Count]),
                                      self.iris.domain.attributes[0])
        initialize.reset_mock()
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.attributes[1].name)
        initialize.assert_not_called()

    def test_saved_workflow(self):
        self.agg_checkboxes[Pivot.Functions.Sum.value].click()
        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_activate_item(self.widget.controls.row_feature,
                                        self.iris.domain.attributes[0].name)
        simulate.combobox_activate_item(self.widget.controls.col_feature,
                                        self.iris.domain.class_var.name)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.iris.domain.attributes[1].name)
        self.widget.table_view.set_selection(set([(11, 0), (11, 1), (12, 0),
                                                  (12, 1), (13, 0), (13, 1),
                                                  (14, 0), (14, 1)]))
        self.widget.table_view.selection_changed.emit()
        output = self.get_output(self.widget.Outputs.filtered_data)
        self.assertEqual(output.X.shape, (20, 4))

        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(self.widget.__class__, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.iris, widget=w)
        output = self.get_output(self.widget.Outputs.filtered_data)
        self.assertEqual(output.X.shape, (20, 4))
        self.assertSetEqual(self.widget.selection, w.selection)

    def test_select_by_click(self):
        view = self.widget.table_view
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        self.agg_checkboxes[Pivot.Functions.Sum.value].click()
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        self.heart_disease.domain[0].name)

        # column in a group
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=QPoint(208, 154))
        self.assertSetEqual({(3, 0), (2, 0)}, view.get_selection())

        # column
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=QPoint(340, 40))
        self.assertSetEqual({(0, 1), (3, 1), (1, 1), (2, 1)},
                            view.get_selection())

        # group
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=QPoint(155, 75))
        self.assertSetEqual({(0, 1), (1, 0), (0, 0), (1, 1)},
                            view.get_selection())

        # all
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=QPoint(400, 198))
        self.assertSetEqual({(0, 1), (0, 0), (3, 0), (3, 1), (2, 1), (2, 0),
                             (1, 0), (1, 1)}, view.get_selection())

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.report_button.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.report_button.click()

    def test_renaming_warning(self):
        data = Table('iris')
        cls_var = data.domain.class_var.copy(name='Aggregate')
        data.domain = Domain(data.domain.attributes, (cls_var,))
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.renamed_vars.is_shown())

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertFalse(self.widget.Warning.renamed_vars.is_shown())

    @patch("Orange.widgets.data.owpivot.OWPivot.MAX_VALUES", 2)
    def test_max_values(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Warning.too_many_values.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.too_many_values.is_shown())

    def test_table_values(self):
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        domain = self.heart_disease.domain
        self.agg_checkboxes[Pivot.Functions.Majority.value].click()
        simulate.combobox_activate_item(self.widget.controls.col_feature,
                                        domain["gender"].name)
        simulate.combobox_activate_item(self.widget.controls.val_feature,
                                        domain["thal"].name)

        model = self.widget.table_view.model()
        self.assertEqual(model.data(model.index(2, 3)), "72.0")
        self.assertEqual(model.data(model.index(3, 3)), "normal")
        self.assertEqual(model.data(model.index(4, 3)), "25.0")
        self.assertEqual(model.data(model.index(5, 3)), "reversable defect")
        self.assertEqual(model.data(model.index(2, 4)), "92.0")
        self.assertEqual(model.data(model.index(3, 4)), "normal")
        self.assertEqual(model.data(model.index(4, 4)), "114.0")
        self.assertEqual(model.data(model.index(5, 4)), "reversable defect")

    def test_only_metas_table(self):
        self.send_signal(self.widget.Inputs.data, self.zoo[:, 17:])
        self.assertTrue(self.widget.Warning.no_variables.is_shown())

        data = self.zoo.transform(Domain([], metas=self.zoo.domain.attributes))
        self.send_signal(self.widget.Inputs.data, data)
        self.assertFalse(self.widget.Warning.no_variables.is_shown())

    def test_empty_table(self):
        data = self.heart_disease[:, :0]
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.no_variables.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.no_variables.is_shown())

        data = self.heart_disease
        self.send_signal(self.widget.Inputs.data, data)

        zoo_domain = self.zoo.domain
        data = self.zoo.transform(Domain([], metas=zoo_domain.metas))
        self.send_signal(self.widget.Inputs.data, data)

        domain = Domain([], zoo_domain.class_vars, metas=zoo_domain.metas)
        data = self.zoo.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)


class TestAggregationFunctionsEnum(unittest.TestCase):
    def test_pickle(self):
        self.assertIs(AggregationFunctionsEnum.Sum,
                      loads(dumps(AggregationFunctionsEnum.Sum)))

    def test_sort(self):
        af = AggregationFunctionsEnum
        self.assertEqual(sorted([af.Sum, af.Min]), sorted([af.Min, af.Sum]))


class TestPivot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        domain = Domain([Dv("d1", ("a", "b")),
                         Dv("d2", ("c", "d", "e")), Cv("c1")])
        X = np.array([[0, 0, 1], [0, 1, 2], [0, np.nan, 3], [0, 0, 4],
                      [1, 0, 5], [1, 0, 6], [1, 1, np.nan], [1, 2, 7],
                      [np.nan, 0, 8]])
        cls.table = Table(domain, X)

        domain = Domain([Cv("c0"), Dv("d1", ("a", "b")), Cv("c1"),
                         Dv("d2", ("a", "b")), Cv("c2")],
                        Dv("cls", ("a", "b")), [sv("m1"), sv("m2")])
        X = np.array([[np.nan, 0, 1, 0, 2], [np.nan, 1, 2, np.nan, 3],
                      [np.nan, 0, 3, 1, np.nan]])
        M = np.array([["aa", "dd"], ["bb", "ee"], ["cc", ""]], dtype=object)
        cls.table1 = Table(domain, X, np.array([0, 0, 1]), M)

    def test_group_table(self):
        domain = self.table.domain
        pivot = Pivot(self.table, Pivot.Functions, domain[0], domain[1])
        group_tab = pivot.group_table
        atts = (Cv("(count)"), Cv("d1 (count defined)"),
                Dv("d1 (majority)", ["a", "b"]),
                Cv("d2 (count defined)"), Dv("d2 (majority)", ["c", "d", "e"]),
                Cv("c1 (count defined)"), Cv("c1 (sum)"),
                Cv("c1 (mean)"), Cv("c1 (min)"), Cv("c1 (max)"),
                Cv("c1 (mode)"), Cv("c1 (median)"), Cv("c1 (var)"))
        X = np.array(
            [[0, 0, 2, 2, 0, 2, 0, 2, 5, 2.5, 1, 4, 1, 2.5, 2.25],
             [0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0],
             [0, 2, 0, 0, np.nan, 0, np.nan, 0, 0, np.nan,
              np.nan, np.nan, np.nan, np.nan, np.nan],
             [1, 0, 2, 2, 1, 2, 0, 2, 11, 5.5, 5, 6, 5, 5.5, 0.25],
             [1, 1, 1, 1, 1, 1, 1, 0, 0, np.nan, np.nan,
              np.nan, np.nan, np.nan, np.nan],
             [1, 2, 1, 1, 1, 1, 2, 1, 7, 7, 7, 7, 7, 7, 0]])
        self.assert_table_equal(group_tab, Table(Domain(domain[:2] + atts), X))

    def test_group_table_time_var(self):
        domain = Domain([Dv("d1", ("a", "b")), Tv("t1", have_date=1)])
        X = np.array([[0, 1e9], [0, 1e8], [1, 2e8], [1, np.nan]])
        table = Table(domain, X)
        pivot = Pivot(table, Pivot.Functions, domain[0], val_var=domain[1])
        str_grouped = \
            "[[a, 2, 2, a, 2, 1.1e+09, 1987-06-06, 1973-03-03, " \
            "2001-09-09, 1973-03-03, 1987-06-06, 2.025e+17],\n " \
            "[b, 2, 2, b, 1, 2e+08, 1976-05-03, 1976-05-03, " \
            "1976-05-03, 1976-05-03, 1976-05-03, 0]]"
        self.assertEqual(str(pivot.group_table), str_grouped)

    def test_group_table_metas(self):
        domain = Domain([Dv("d1", ("a", "b")), Cv("c1"),
                         Dv("d2", ("a", "b")), Cv("c2")])
        X = np.array([[0, 1, 0, 2], [1, 2, np.nan, 3], [0, 3, 1, np.nan]])
        table = Table(domain, X).transform(
            Domain(domain.attributes[:2], metas=domain.attributes[2:])).copy()
        with table.unlocked():
            table.metas = table.metas.astype(object)

        pivot = Pivot(table, Pivot.Functions, table.domain[-1])
        group_tab = pivot.group_table
        atts = (table.domain[-1], Cv("(count)"), Cv("d1 (count defined)"),
                Dv("d1 (majority)", ["a", "b"]),
                Cv("c1 (count defined)"), Cv("c1 (sum)"), Cv("c1 (mean)"),
                Cv("c1 (min)"), Cv("c1 (max)"), Cv("c1 (mode)"),
                Cv("c1 (median)"), Cv("c1 (var)"), Cv("d2 (count defined)"),
                Dv("d2 (majority)", ["a", "b"]), Cv("c2 (count defined)"),
                Cv("c2 (sum)"), Cv("c2 (mean)"), Cv("c2 (min)"), Cv("c2 (max)"),
                Cv("c2 (mode)"), Cv("c2 (median)"), Cv("c2 (var)"))
        X = np.array([[0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                       1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 0],
                      [1, 1, 1, 0, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, np.nan,
                       np.nan, np.nan, np.nan, np.nan, np.nan]], dtype=float)
        self.assert_table_equal(group_tab, Table(Domain(atts), X))

    @patch("Orange.widgets.data.owpivot.Pivot.Count.func",
           side_effect=Pivot.Count.func)
    @patch("Orange.widgets.data.owpivot.Pivot.Sum.func",
           side_effect=Pivot.Sum.func)
    def test_group_table_use_cached(self, count_func, sum_func):
        domain = self.table.domain
        pivot = Pivot(self.table, [Pivot.Count, Pivot.Sum], domain[0], domain[1])
        group_tab = pivot.group_table
        count_func.reset_mock()
        sum_func.reset_mock()

        pivot.update_group_table(Pivot.Functions)
        count_func.assert_not_called()
        sum_func.assert_not_called()
        atts = (Cv("(count)"), Cv("d1 (count defined)"),
                Dv("d1 (majority)", ["a", "b"]),
                Cv("d2 (count defined)"), Dv("d2 (majority)", ["c", "d", "e"]),
                Cv("c1 (count defined)"), Cv("c1 (sum)"), Cv("c1 (mean)"),
                Cv("c1 (min)"), Cv("c1 (max)"), Cv("c1 (mode)"),
                Cv("c1 (median)"), Cv("c1 (var)"))
        X = np.array(
            [[0, 0, 2, 2, 0, 2, 0, 2, 5, 2.5, 1, 4, 1, 2.5, 2.25],
             [0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0],
             [0, 2, 0, 0, np.nan, 0, np.nan, 0, 0, np.nan,
              np.nan, np.nan, np.nan, np.nan, np.nan],
             [1, 0, 2, 2, 1, 2, 0, 2, 11, 5.5, 5, 6, 5, 5.5, 0.25],
             [1, 1, 1, 1, 1, 1, 1, 0, 0, np.nan, np.nan,
              np.nan, np.nan, np.nan, np.nan],
             [1, 2, 1, 1, 1, 1, 2, 1, 7, 7, 7, 7, 7, 7, 0]])
        self.assert_table_equal(pivot.group_table,
                                Table(Domain(domain[:2] + atts), X))

        pivot.update_group_table([Pivot.Count, Pivot.Sum])
        count_func.assert_not_called()
        sum_func.assert_not_called()
        self.assert_table_equal(pivot.group_table, group_tab)

    def test_group_table_no_col_var(self):
        domain = self.table.domain
        pivot = Pivot(self.table, Pivot.Functions, domain[0])
        group_tab = pivot.group_table
        atts = (Cv("(count)"), Cv("d1 (count defined)"),
                Dv("d1 (majority)", ["a", "b"]),
                Cv("d2 (count defined)"), Dv("d2 (majority)", ["c", "d", "e"]),
                Cv("c1 (count defined)"), Cv("c1 (sum)"),
                Cv("c1 (mean)"), Cv("c1 (min)"), Cv("c1 (max)"),
                Cv("c1 (mode)"), Cv("c1 (median)"), Cv("c1 (var)"))
        domain = Domain(domain[:1] + atts)
        X = np.array([[0, 4, 4, 0, 3, 0, 4, 10, 2.5, 1, 4, 1, 2.5, 1.25],
                      [1, 4, 4, 1, 4, 0, 3, 18, 6, 5, 7, 5, 6, 2 / 3]],
                     dtype=float)
        self.assert_table_equal(group_tab, Table(Domain(domain[:1] + atts), X))

        pivot = Pivot(self.table, Pivot.Functions, domain[0], domain[0])
        group_tab_same_vars = pivot.group_table
        self.assert_table_equal(group_tab, group_tab_same_vars)

    def test_group_table_no_col_var_metas(self):
        for var in self.table1.domain.metas:
            self.assertRaises(TypeError, Pivot, self.table1, var)

        domain = Domain([Dv("d1", ("a", "b")), Cv("c1"),
                         Dv("d2", ("a", "b")), Cv("c2")])
        X = np.array([[0, 1, 0, 2], [1, 2, np.nan, 3], [0, 3, 1, np.nan]])
        table = Table(domain, X).transform(
            Domain(domain.attributes[:2], metas=domain.attributes[2:]))

        pivot = Pivot(table, Pivot.Functions, table.domain[-1])
        group_tab = pivot.group_table
        atts = (table.domain[-1], Cv("(count)"), Cv("d1 (count defined)"),
                Dv("d1 (majority)", ["a", "b"]),
                Cv("c1 (count defined)"), Cv("c1 (sum)"), Cv("c1 (mean)"),
                Cv("c1 (min)"), Cv("c1 (max)"), Cv("c1 (mode)"),
                Cv("c1 (median)"), Cv("c1 (var)"), Cv("d2 (count defined)"),
                Dv("d2 (majority)", ["a", "b"]), Cv("c2 (count defined)"),
                Cv("c2 (sum)"), Cv("c2 (mean)"), Cv("c2 (min)"), Cv("c2 (max)"),
                Cv("c2 (mode)"), Cv("c2 (median)"), Cv("c2 (var)"))
        X = np.array([[0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                       1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 0],
                      [1, 1, 1, 0, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0, 0, np.nan,
                       np.nan, np.nan, np.nan, np.nan, np.nan]], dtype=float)
        self.assert_table_equal(group_tab, Table(Domain(atts), X))

    def test_group_table_update(self):
        domain = self.table.domain
        atts = (Cv("(count)"), Cv("d1 (count defined)"),
                Dv("d1 (majority)", ["a", "b"]),
                Cv("d2 (count defined)"), Dv("d2 (majority)", ["c", "d", "e"]),
                Cv("c1 (count defined)"), Cv("c1 (sum)"), Cv("c1 (mean)"),
                Cv("c1 (min)"), Cv("c1 (max)"), Cv("c1 (mode)"),
                Cv("c1 (median)"), Cv("c1 (var)"))
        X = np.array(
            [[0, 0, 2, 2, 0, 2, 0, 2, 5, 2.5, 1, 4, 1, 2.5, 2.25],
             [0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0],
             [0, 2, 0, 0, np.nan, 0, np.nan, 0, 0, np.nan,
              np.nan, np.nan, np.nan, np.nan, np.nan],
             [1, 0, 2, 2, 1, 2, 0, 2, 11, 5.5, 5, 6, 5, 5.5, 0.25],
             [1, 1, 1, 1, 1, 1, 1, 0, 0, np.nan, np.nan,
              np.nan, np.nan, np.nan, np.nan],
             [1, 2, 1, 1, 1, 1, 2, 1, 7, 7, 7, 7, 7, 7, 0]])
        table = Table(Domain(domain[:2] + atts), X)

        agg = [Pivot.Functions.Count, Pivot.Functions.Sum]
        pivot = Pivot(self.table, agg, domain[0], domain[1])
        group_tab = pivot.group_table
        pivot.update_group_table(Pivot.Functions)
        self.assert_table_equal(pivot.group_table, table)
        pivot.update_group_table(agg)
        self.assert_table_equal(group_tab, pivot.group_table)

    def test_group_table_1(self):
        var = self.table1.domain.variables[1]
        domain = Domain(
            [var, Cv("(count)"), Cv("c0 (count defined)"), Cv("c0 (sum)"),
             Cv("c0 (mean)"), Cv("c0 (min)"), Cv("c0 (max)"), Cv("c0 (mode)"),
             Cv("c0 (median)"), Cv("c0 (var)"), Cv("d1 (count defined)"),
             Dv("d1 (majority)", ["a", "b"]), Cv("c1 (count defined)"),
             Cv("c1 (sum)"), Cv("c1 (mean)"), Cv("c1 (min)"), Cv("c1 (max)"),
             Cv("c1 (mode)"), Cv("c1 (median)"), Cv("c1 (var)"),
             Cv("d2 (count defined)"), Dv("d2 (majority)", ["a", "b"]),
             Cv("c2 (count defined)"), Cv("c2 (sum)"), Cv("c2 (mean)"),
             Cv("c2 (min)"), Cv("c2 (max)"), Cv("c2 (mode)"),
             Cv("c2 (median)"), Cv("c2 (var)"), Cv("cls (count defined)"),
             Dv("cls (majority)", ["a", "b"]), Cv("m1 (count defined)"),
             Cv("m2 (count defined)")])
        X = np.array([[0, 2, 0, 0, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, 2, 0, 2, 4, 2, 1, 3, 1, 2, 1,
                       2, 0, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 1],
                      [1, 1, 0, 0, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0,
                       np.nan, 1, 3, 3, 3, 3, 3, 3, 0, 1, 0, 1, 1]])
        pivot = Pivot(self.table1, Pivot.Functions, var)
        group_tab = pivot.group_table
        self.assert_table_equal(group_tab, Table(domain, X))

    def test_group_sparse_data(self):
        var = self.table1.domain.variables[1]
        dense = Pivot(self.table1, Pivot.Functions, var)
        sparse_data = self.table1.to_sparse()
        var = sparse_data.domain.variables[1]
        sparse = Pivot(sparse_data, Pivot.Functions, var)
        self.assert_table_equal(dense.group_table, sparse.group_table)

    def test_pivot(self):
        domain = self.table.domain
        pivot = Pivot(self.table, Pivot.Functions, domain[0], domain[1], domain[2])
        pivot_tab = pivot.pivot_table
        atts = (Dv("Aggregate", ["Count", "Count defined", "Sum", "Mean",
                                 "Min", "Max", "Mode", "Median", "Var"]),
                Cv("c"), Cv("d"), Cv("e"))
        X = np.array([[0, 0, 2, 1, 0],
                      [0, 1, 2, 1, 0],
                      [0, 2, 5, 2, 0],
                      [0, 3, 2.5, 2, np.nan],
                      [0, 4, 1, 2, np.nan],
                      [0, 5, 4, 2, np.nan],
                      [0, 6, 1, 2, np.nan],
                      [0, 7, 2.5, 2, np.nan],
                      [0, 8, 2.25, 0, np.nan],
                      [1, 0, 2, 1, 1],
                      [1, 1, 2, 0, 1],
                      [1, 2, 11, 0, 7],
                      [1, 3, 5.5, np.nan, 7],
                      [1, 4, 5, np.nan, 7],
                      [1, 5, 6, np.nan, 7],
                      [1, 6, 5, np.nan, 7],
                      [1, 7, 5.5, np.nan, 7],
                      [1, 8, 0.25, np.nan, 0]])
        self.assert_table_equal(pivot_tab, Table(Domain(domain[:1] + atts), X))

    def test_pivot_total(self):
        domain = self.table.domain
        pivot = Pivot(self.table, [Pivot.Functions.Count, Pivot.Functions.Sum],
                      domain[0], domain[1], domain[2])

        atts = (Dv(domain[0].name, ["Total"]),
                Dv("Aggregate", ["Count", "Sum"]), Cv("c"), Cv("d"), Cv("e"))
        X = np.array([[0, 0, 4, 2, 1], [0, 1, 16, 2, 7]])
        table = Table(Domain(atts), X)

        self.assert_table_equal(pivot.pivot_total_h, table)
        table = Table(Domain((Cv("Total"),)), np.array([[3], [7], [4], [18]]))
        self.assert_table_equal(pivot.pivot_total_v, table)

        table = Table(Domain((Cv("Total"),)), np.array([[7], [25]]))
        self.assert_table_equal(pivot.pivot_total, table)

    def test_pivot_no_col_var(self):
        domain = self.table.domain
        pivot = Pivot(self.table, Pivot.Functions, domain[0], None, domain[2])
        pivot_tab = pivot.pivot_table
        atts = (Dv("Aggregate",
                   ["Count", "Count defined", "Sum", "Mean",
                    "Min", "Max", "Mode", "Median", "Var"]),
                Cv("a"), Cv("b"))
        X = np.array([[0, 0, 4, 0],
                      [0, 1, 4, 0],
                      [0, 2, 10, 0],
                      [0, 3, 2.5, np.nan],
                      [0, 4, 1, np.nan],
                      [0, 5, 4, np.nan],
                      [0, 6, 1, np.nan],
                      [0, 7, 2.5, np.nan],
                      [0, 8, 1.25, np.nan],
                      [1, 0, 0, 4],
                      [1, 1, 0, 3],
                      [1, 2, 0, 18],
                      [1, 3, np.nan, 6],
                      [1, 4, np.nan, 5],
                      [1, 5, np.nan, 7],
                      [1, 6, np.nan, 5],
                      [1, 7, np.nan, 6],
                      [1, 8, np.nan, 2 / 3]])
        self.assert_table_equal(pivot_tab, Table(Domain(domain[:1] + atts), X))

    def test_pivot_no_val_var(self):
        domain = self.table.domain
        pivot = Pivot(self.table, Pivot.Functions, domain[0], domain[1])
        pivot_tab = pivot.pivot_table
        atts = (Dv("Aggregate", ["Count"]),
                Cv("c"), Cv("d"), Cv("e"))
        X = np.array([[0, 0, 2, 1, 0], [1, 0, 2, 1, 1]])
        self.assert_table_equal(pivot_tab, Table(Domain(domain[:1] + atts), X))

    def test_pivot_disc_val_var(self):
        domain = self.table.domain
        pivot = Pivot(self.table, [Pivot.Count_defined, Pivot.Majority],
                      domain[2], domain[0], domain[1])
        pivot_tab = pivot.pivot_table
        atts = (domain[2], Dv("Aggregate", ["Count defined", "Majority"]),
                Dv("a", ["0.0", "1.0", "c", "d"]),
                Dv("b", ["0.0", "1.0", "c", "e"]))
        X = np.array([[1, 0, 1, 0],
                      [1, 1, 2, np.nan],
                      [2, 0, 1, 0],
                      [2, 1, 3, np.nan],
                      [3, 0, 0, 0],
                      [3, 1, np.nan, np.nan],
                      [4, 0, 1, 0],
                      [4, 1, 2, np.nan],
                      [5, 0, 0, 1],
                      [5, 1, np.nan, 2],
                      [6, 0, 0, 1],
                      [6, 1, np.nan, 2],
                      [7, 0, 0, 1],
                      [7, 1, np.nan, 3],
                      [8, 0, 0, 0],
                      [8, 1, np.nan, np.nan]])
        self.assert_table_equal(pivot_tab, Table(Domain(atts), X))

    def test_pivot_time_val_var(self):
        domain = Domain([Dv("d1", ("a", "b")), Dv("d2", ("c", "d")),
                         Tv("t1", have_date=1)])
        X = np.array([[0, 1, 1e9], [0, 0, 1e8], [1, 0, 2e8], [1, 1, np.nan]])
        table = Table(domain, X)

        # Min
        pivot = Pivot(table, [Pivot.Min],
                      domain[0], domain[1], domain[2])
        atts = (domain[0], Dv("Aggregate", ["Min"]),
                Tv("c", have_date=1), Tv("d", have_date=1))
        X = np.array([[0, 0, 1e8, 1e9],
                      [1, 0, 2e8, np.nan]])
        self.assert_table_equal(pivot.pivot_table, Table(Domain(atts), X))

        # Min, Max
        pivot = Pivot(table, [Pivot.Min, Pivot.Max],
                      domain[0], domain[1], domain[2])
        atts = (domain[0], Dv("Aggregate", ["Min", "Max"]),
                Tv("c", have_date=1), Tv("d", have_date=1))
        X = np.array([[0, 0, 1e8, 1e9],
                      [0, 1, 1e8, 1e9],
                      [1, 0, 2e8, np.nan],
                      [1, 1, 2e8, np.nan]])
        self.assert_table_equal(pivot.pivot_table, Table(Domain(atts), X))

        # Count defined, Sum
        pivot = Pivot(table, [Pivot.Count_defined, Pivot.Sum],
                      domain[0], domain[1], domain[2])
        atts = (domain[0], Dv("Aggregate", ["Count defined", "Sum"]),
                Cv("c"), Cv("d"))
        X = np.array([[0, 0, 1, 1],
                      [0, 1, 1e8, 1e9],
                      [1, 0, 1, 0],
                      [1, 1, 2e8, 0]])
        self.assert_table_equal(pivot.pivot_table, Table(Domain(atts), X))

        # Count defined, Max
        pivot = Pivot(table, [Pivot.Count_defined, Pivot.Max],
                      domain[0], domain[1], domain[2])
        atts = (domain[0], Dv("Aggregate", ["Count defined", "Max"]),
                Dv("c", ["1.0", "1973-03-03", "1976-05-03"]),
                Dv("d", ["0.0", "1.0", "2001-09-09"]))
        X = np.array([[0, 0, 0, 1],
                      [0, 1, 1, 2],
                      [1, 0, 0, 0],
                      [1, 1, 2, np.nan]])
        self.assert_table_equal(pivot.pivot_table, Table(Domain(atts), X))

    def test_pivot_attr_combinations(self):
        domain = self.table1.domain
        for var1, var2, var3 in ((domain[1], domain[3], domain[5]),  # d d d
                                 (domain[1], domain[3], domain[4]),  # d d c
                                 (domain[1], domain[3], domain[-1]),  # d d s
                                 (domain[2], domain[3], domain[5]),  # c d d
                                 (domain[2], domain[3], domain[4]),  # c d c
                                 (domain[2], domain[3], domain[-1])):  # c d s
            pivot = Pivot(self.table1, Pivot.Functions, var1, var2, var3)
            pivot_tab = pivot.pivot_table
            self.assertGreaterEqual(pivot_tab.X.shape[0], 4)
            self.assertGreaterEqual(pivot_tab.X.shape[1], 4)
        for var1, var2 in ((domain[1], domain[2]),  # d c
                           (domain[1], domain[-2]),  # d s
                           (domain[2], domain[4]),  # c
                           (domain[-1], domain[1])):  # s
            self.assertRaises(TypeError, Pivot, self.table1, var1, var2)

    def test_pivot_update(self):
        domain = self.table.domain
        pivot = Pivot(self.table, [Pivot.Functions.Count], domain[0],
                      domain[1], domain[2])
        pivot_tab1 = pivot.pivot_table
        pivot.update_pivot_table(domain[1])
        pivot.update_pivot_table(domain[2])
        self.assert_table_equal(pivot_tab1, pivot.pivot_table)

    def test_pivot_data_subset(self):
        data = Table("iris")
        cls_var = data.domain.class_var
        pivot = Pivot(data[:100], Pivot.Functions, cls_var, None, cls_var)
        atts = (cls_var, Dv("Aggregate", ["Count", "Count defined", "Majority"]),
                Dv("Iris-setosa", ["0.0", "50.0", "Iris-setosa"]),
                Dv("Iris-versicolor", ["0.0", "50.0", "Iris-versicolor"]))
        domain = Domain(atts)
        self.assert_domain_equal(domain, pivot.pivot_table.domain)

    def test_pivot_renaming_domain(self):
        data = Table("iris")
        cls_var = data.domain.class_var.copy(name='Aggregate')
        data.domain = Domain(data.domain.attributes, (cls_var,))
        pivot = Pivot(data, [Pivot.Functions.Sum], cls_var, None, None)

        renamed_var = data.domain.class_var.copy(name='Aggregate (1)')
        self.assertTrue(renamed_var in pivot.pivot_table.domain)
        renamed_var = data.domain.class_var.copy(name='Aggregate (2)')
        self.assertTrue(renamed_var in pivot.pivot_table.domain)

    def assert_table_equal(self, table1, table2):
        self.assert_domain_equal(table1.domain, table2.domain)
        np.testing.assert_array_equal(table1.X, table2.X)
        np.testing.assert_array_equal(table1.Y, table2.Y)
        np.testing.assert_array_equal(table1.metas, table2.metas)

    def assert_domain_equal(self, domain1, domain2):
        for var1, var2 in zip(domain1.variables + domain1.metas,
                              domain2.variables + domain2.metas):
            self.assertEqual(type(var1), type(var2))
            self.assertEqual(var1.name, var2.name)
            if var1.is_discrete:
                self.assertEqual(var1.values, var2.values)


if __name__ == "__main__":
    unittest.main()
