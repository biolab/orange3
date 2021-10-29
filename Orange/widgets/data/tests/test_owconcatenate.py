# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring, abstract-method, protected-access
import unittest
from unittest.mock import patch, Mock

import numpy as np
from numpy.testing import assert_array_equal

from Orange.data import (
    Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
)
from Orange.preprocess.transformation import Identity
from Orange.widgets.data.owconcatenate import OWConcatenate
from Orange.widgets.tests.base import WidgetTest


class TestOWConcatenate(WidgetTest):

    class DummyTable(Table):

        pass

    def setUp(self):
        self.widget = self.create_widget(OWConcatenate)
        self.iris = Table("iris")
        self.titanic = Table("titanic")

    def test_no_input(self):
        self.widget.commit.now()
        self.widget.controls.append_source_column.toggle()
        self.widget.commit.now()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_single_input(self):
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.send_signal(self.widget.Inputs.primary_data, self.iris)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(list(output), list(self.iris))
        self.send_signal(self.widget.Inputs.primary_data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.send_signal(self.widget.Inputs.additional_data, self.iris)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(list(output), list(self.iris))
        self.send_signal(self.widget.Inputs.additional_data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_two_inputs_union(self):
        self.send_signal(self.widget.Inputs.additional_data, self.iris, 0)
        self.send_signal(self.widget.Inputs.additional_data, self.titanic, 1)
        output = self.get_output(self.widget.Outputs.data)
        # needs to contain all instances
        self.assertEqual(len(output), len(self.iris) + len(self.titanic))
        # needs to contain all variables
        outvars = output.domain.variables
        self.assertLess(set(self.iris.domain.variables), set(outvars))
        self.assertLess(set(self.titanic.domain.variables), set(outvars))
        # the first part of the dataset is iris, the second part is titanic
        np.testing.assert_equal(self.iris.X, output.X[:len(self.iris), :-3])
        self.assertTrue(np.isnan(output.X[:len(self.iris), -3:]).all())
        np.testing.assert_equal(self.titanic.X, output.X[len(self.iris):, -3:])
        self.assertTrue(np.isnan(output.X[len(self.iris):, :-3]).all())

    def test_two_inputs_intersection(self):
        self.send_signal(self.widget.Inputs.additional_data, self.iris, 0)
        self.send_signal(self.widget.Inputs.additional_data, self.titanic, 1)
        self.widget.controls.merge_type.buttons[1].click()
        output = self.get_output(self.widget.Outputs.data)
        # needs to contain all instances
        self.assertEqual(len(output), len(self.iris) + len(self.titanic))
        # no common variables
        outvars = output.domain.variables
        self.assertEqual(0, len(outvars))

    def test_source(self):
        self.send_signal(self.widget.Inputs.additional_data, self.iris, 0)
        self.send_signal(self.widget.Inputs.additional_data, self.titanic, 1)
        outputb = self.get_output(self.widget.Outputs.data)
        outvarsb = outputb.domain.variables
        def get_source():
            output = self.get_output(self.widget.Outputs.data)
            outvars = output.domain.variables + output.domain.metas
            return (set(outvars) - set(outvarsb)).pop()
        # test adding source
        self.widget.controls.append_source_column.toggle()
        source = get_source()
        self.assertEqual(source.name, "Source ID")
        # test name changing
        self.widget.controls.source_attr_name.setText("Source")
        self.widget.controls.source_attr_name.callback()
        source = get_source()
        self.assertEqual(source.name, "Source")
        # test source_column role
        places = ["class_vars", "attributes", "metas"]
        for i, place in enumerate(places):
            self.widget.source_column_role = i
            self.widget.commit.now()
            source = get_source()
            output = self.get_output(self.widget.Outputs.data)
            self.assertTrue(source in getattr(output.domain, place))
            data = output.transform(Domain([source]))
            np.testing.assert_equal(data[:len(self.iris)].X, 0)
            np.testing.assert_equal(data[len(self.iris):].X, 1)

    def test_singleclass_source_class(self):
        self.send_signal(self.widget.Inputs.primary_data, self.iris)
        # add source into a class variable
        self.widget.controls.append_source_column.toggle()

    def test_disable_merging_on_primary(self):
        self.assertTrue(self.widget.mergebox.isEnabled())
        self.send_signal(self.widget.Inputs.primary_data, self.iris)
        self.assertFalse(self.widget.mergebox.isEnabled())
        self.send_signal(self.widget.Inputs.primary_data, None)
        self.assertTrue(self.widget.mergebox.isEnabled())

    def test_unconditional_commit_on_new_signal(self):
        with patch.object(self.widget.commit, 'now') as apply:
            self.widget.auto_commit = False
            apply.reset_mock()
            self.send_signal(self.widget.Inputs.primary_data, self.iris)
            apply.assert_called()

    def test_type_compatibility(self):
        # result is on the Output for compatible types
        self.send_signal(self.widget.Inputs.primary_data, self.iris)
        self.send_signal(self.widget.Inputs.additional_data, self.iris)
        self.assertIsNotNone(self.widget.Outputs.data)
        self.assertFalse(self.widget.Error.bow_concatenation.is_shown())
        # test incompatible type error
        self.send_signal(self.widget.Inputs.primary_data, self.iris)
        self.send_signal(self.widget.Inputs.additional_data, self.DummyTable())
        self.assertTrue(self.widget.Error.bow_concatenation.is_shown())

    def test_same_var_name(self):
        widget = self.widget

        var1 = DiscreteVariable(name="x", values=list("abcd"))
        data1 = Table.from_numpy(Domain([var1]),
                                 np.arange(4).reshape(4, 1), np.zeros((4, 0)))
        var2 = DiscreteVariable(name="x", values=list("def"))
        data2 = Table.from_numpy(Domain([var2]),
                                 np.arange(3).reshape(3, 1), np.zeros((3, 0)))

        self.send_signal(widget.Inputs.additional_data, data1, 1)
        self.send_signal(widget.Inputs.additional_data, data2, 2)
        output = self.get_output(widget.Outputs.data)
        np.testing.assert_equal(output.X,
                                np.array([0, 1, 2, 3, 3, 4, 5]).reshape(7, 1))

    def test_duplicated_id_column(self):
        widget = self.widget

        var1 = DiscreteVariable(name="x", values=list("abcd"))
        data1 = Table.from_numpy(Domain([var1]),
                                 np.arange(4).reshape(4, 1), np.zeros((4, 0)))
        widget.append_source_column = True
        widget.source_column_role = 0
        widget.source_attr_name = "x"
        self.send_signal(widget.Inputs.primary_data, data1)
        out = self.get_output(widget.Outputs.data)
        self.assertEqual(out.domain.attributes[0].name, "x")
        self.assertEqual(out.domain.class_var.name, "x (1)")

    def test_domain_intersect(self):
        widget = self.widget
        widget.merge_type = OWConcatenate.MergeIntersection

        X1, X2, X3 = map(ContinuousVariable, ["X1", "X2", "X3"])
        D1, D2, D3 = map(lambda n: DiscreteVariable(n, values=("a", "b")),
                         ["D1", "D2", "D3"])
        S1, S2 = map(StringVariable, ["S1", "S2"])
        domain1 = Domain([X1, X2], [D1], [S1])
        domain2 = Domain([X3], [D2], [S2])

        res = widget.merge_domains([domain1, domain2])
        self.assertSequenceEqual(res.attributes, [])
        self.assertSequenceEqual(res.class_vars, [])
        self.assertSequenceEqual(res.metas, [])

        domain2 = Domain([X2, X3], [D1, D2, D3], [S1, S2])
        res = widget.merge_domains([domain1, domain2])
        self.assertSequenceEqual(res.attributes, [X2])
        self.assertSequenceEqual(res.class_vars, [D1])
        self.assertSequenceEqual(res.metas, [S1])

        res = widget.merge_domains([domain1, domain1])
        self.assertSequenceEqual(res.attributes, domain1.attributes)
        self.assertSequenceEqual(res.class_vars, domain1.class_vars)
        self.assertSequenceEqual(res.metas, domain1.metas)

    def test_domain_union(self):
        widget = self.widget
        widget.merge_type = OWConcatenate.MergeUnion

        X1, X2, X3 = map(ContinuousVariable, ["X1", "X2", "X3"])
        D1, D2, D3 = map(lambda n: DiscreteVariable(n, values=("a", "b")),
                         ["D1", "D2", "D3"])
        S1, S2 = map(StringVariable, ["S1", "S2"])
        domain1 = Domain([X1, X2], [D1], [S1])
        domain2 = Domain([X3], [D2], [S2])
        res = widget.merge_domains([domain1, domain2])

        self.assertSequenceEqual(res.attributes, [X1, X2, X3])
        self.assertSequenceEqual(res.class_vars, [D1, D2])
        self.assertSequenceEqual(res.metas, [S1, S2])

        domain2 = Domain([X3, X2], [D2, D1, D3], [S2, S1])
        res = widget.merge_domains([domain1, domain2])
        self.assertSequenceEqual(res.attributes, [X1, X2, X3])
        self.assertSequenceEqual(res.class_vars, [D1, D2, D3])
        self.assertSequenceEqual(res.metas, [S1, S2])

        res = widget.merge_domains([domain1, domain1])
        self.assertSequenceEqual(res.attributes, domain1.attributes)
        self.assertSequenceEqual(res.class_vars, domain1.class_vars)
        self.assertSequenceEqual(res.metas, domain1.metas)

    def test_domain_union_duplicated_names(self):
        widget = self.widget
        widget.merge_type = OWConcatenate.MergeUnion

        X1, X2, X3 = map(ContinuousVariable, ["X1", "X2", "X3"])
        D1, D2 = map(lambda n: DiscreteVariable(n, values=["a", "b"]),
                     ["D1", "X2"])
        S1, S2 = map(StringVariable, ["S1", "X1"])
        domain1 = Domain([X1, X2], [D1], [S1])
        domain2 = Domain([X3], [D2], [S2])
        res = widget.merge_domains([domain1, domain2])

        attributes = res.attributes
        class_vars = res.class_vars
        metas = res.metas

        self.assertEqual([var.name for var in attributes],
                         ["X1 (1)", "X2 (1)", "X3"])
        self.assertEqual([var.name for var in class_vars],
                         ["D1", "X2 (2)"])
        self.assertEqual([var.name for var in metas],
                         ["S1", "X1 (2)"])

        x21_val_from = attributes[1].compute_value
        self.assertIsInstance(x21_val_from, Identity)
        self.assertIsInstance(x21_val_from.variable, ContinuousVariable)
        self.assertEqual(x21_val_from.variable.name, "X2")

        x22_val_from = class_vars[1].compute_value
        self.assertIsInstance(x22_val_from, Identity)
        self.assertIsInstance(x22_val_from.variable, DiscreteVariable)
        self.assertEqual(x22_val_from.variable.name, "X2")

        x11_val_from = attributes[0].compute_value
        self.assertIsInstance(x11_val_from, Identity)
        self.assertIsInstance(x11_val_from.variable, ContinuousVariable)
        self.assertEqual(x11_val_from.variable.name, "X1")

        x12_val_from = metas[1].compute_value
        self.assertIsInstance(x12_val_from, Identity)
        self.assertIsInstance(x12_val_from.variable, StringVariable)
        self.assertEqual(x12_val_from.variable.name, "X1")

    def test_get_part_union(self):
        get_part = OWConcatenate._get_part  # pylint: disable=protected-access

        X1, X2, X3, X4 = map(ContinuousVariable, ["X1", "X2", "X3", "X4"])
        D1, D2, D3 = map(lambda n: DiscreteVariable(n, values=["a", "b"]),
                         ["D1", "D2", "D3"])
        S1, S2, S3 = map(StringVariable, ["S1", "S2", "S3"])
        domain1 = Domain([X1, X2], [D1], [S1, S3])
        domain2 = Domain([X3, X2], [D2, D1], [S2, S3, S1])
        domain3 = Domain([X3, X2, X4], [D2, D1, D3], [S2, S1, S3])

        self.assertEqual(
            get_part([domain1, domain2], set.union, "attributes"),
            [X1, X2, X3]
        )
        self.assertEqual(
            get_part([domain3, domain1, domain2], set.union, "attributes"),
            [X3, X2, X4, X1]
        )
        self.assertEqual(
            get_part([domain1, domain2], set.union, "class_vars"),
            [D1, D2]
        )
        self.assertEqual(
            get_part([domain3, domain1, domain2], set.union, "class_vars"),
            [D2, D1, D3]
        )
        self.assertEqual(
            get_part([domain3, domain1, domain2], set.union, "class_vars"),
            [D2, D1, D3]
        )
        self.assertEqual(
            get_part([domain1, domain2], set.union, "metas"),
            [S1, S3, S2]
        )
        self.assertEqual(
            get_part([domain2, domain1], set.union, "metas"),
            [S2, S3, S1]
        )
        self.assertEqual(
            get_part([domain3, domain2, domain1], set.union, "metas"),
            [S2, S1, S3]
        )

    def test_get_part_intersection(self):
        get_part = OWConcatenate._get_part  # pylint: disable=protected-access

        X1, X2, X3, X4 = map(ContinuousVariable, ["X1", "X2", "X3", "X4"])
        D1, D2, D3 = map(lambda n: DiscreteVariable(n, values=["a", "b"]),
                         ["D1", "D2", "D3"])
        S1, S2, S3 = map(StringVariable, ["S1", "S2", "S3"])
        domain1 = Domain([X1, X2], [D1], [S1, S3])
        domain2 = Domain([X3, X2], [D2, D1], [S2, S3, S1])
        domain3 = Domain([X3, X2, X4], [D2, D1, D3], [S2, S1, S3])

        self.assertEqual(
            get_part([domain1, domain2], set.intersection, "attributes"),
            [X2]
        )
        self.assertEqual(
            get_part([domain1, domain2, domain3], set.intersection, "class_vars"),
            [D1]
        )
        self.assertEqual(
            get_part([domain3, domain1, domain2], set.intersection, "metas"),
            [S1, S3]
        )
        self.assertEqual(
            get_part([domain2, domain1, domain3], set.intersection, "metas"),
            [S3, S1]
        )

    def test_get_unique_vars(self):
        X1, X1a, X2, X2a = map(ContinuousVariable, ["X1", "X1", "X2", "X2"])
        X2.number_of_decimals = 3
        X2a.number_of_decimals = 4
        D1 = DiscreteVariable("X1", values=("a", "b", "c"))
        D1a = DiscreteVariable("X1", values=("e", "b", "d"))
        D2 = DiscreteVariable("X2", values=("a", "b", "c"))
        S1 = StringVariable("X1")

        # pylint: disable=unbalanced-tuple-unpacking,protected-access
        uX1, uX2, uD1, uD2, uS1 =\
            OWConcatenate._unique_vars([X1, X1a, X2, X2a, D1, D2, D1a, S1])

        self.assertIs(X1, uX1)

        self.assertEqual(X2, uX2)
        self.assertEqual(X2a, uX2)
        self.assertEqual(X2.number_of_decimals, 3)
        self.assertEqual(X2a.number_of_decimals, 4)
        self.assertEqual(uX2.number_of_decimals, 4)

        self.assertEqual(D1.values, tuple("abc"))
        self.assertEqual(D1a.values, tuple("ebd"))
        self.assertEqual(uD1, D1)
        self.assertEqual(uD1, D1a)
        self.assertEqual(uD1.values, tuple("abced"))

        self.assertIs(uD2, D2)

        self.assertIs(S1, uS1)

    def test_different_number_decimals(self):
        widget = self.widget

        x1 = ContinuousVariable("x", number_of_decimals=3)
        x2 = ContinuousVariable("x", number_of_decimals=4)
        data1 = Table.from_numpy(Domain([x1]), np.array([[1], [2], [3]]))
        data2 = Table.from_numpy(Domain([x2]), np.array([[1], [2], [3]]))
        for d1, d2, id1, id2 in ((data1, data2, 1, 2), (data1, data2, 2, 1),
                                 (data2, data1, 1, 2), (data2, data1, 2, 1)):
            self.send_signal(widget.Inputs.additional_data, d1, id1)
            self.send_signal(widget.Inputs.additional_data, d2, id2)
            out_dom = self.get_output(widget.Outputs.data).domain
            self.assertEqual(len(out_dom.attributes), 1)
            x = out_dom.attributes[0]
            self.assertEqual(x.number_of_decimals, 4)

    def _create_compute_values(self):
        a1, a2, a3, a4, c1 = self.iris.domain.variables

        def times2(*_):
            return 2

        na1 = a1.copy()
        na2 = a2.copy(compute_value=times2)
        na3 = a3.copy(compute_value=lambda *_: 3)
        na4 = a4.copy(compute_value=lambda *_: 4)
        nc1 = c1.copy(compute_value=lambda *_: 5)

        ma1 = a1.copy()
        ma2 = a2.copy(compute_value=times2)
        ma3 = a3.copy(compute_value=lambda x: 6)
        ma4 = a4.copy(compute_value=lambda x: 7)

        table_n = self.iris.transform(Domain([na1, na2, na3, na4], nc1))
        table_m = self.iris.transform(Domain([ma1, ma2, ma3], None, [ma4]))
        return table_n, table_m

    def test_dumb_tables(self):
        self.widget.commit.deferred = Mock()
        table_n, table_m = self._create_compute_values()
        na1, na2, na3, na4, nc1 = table_n.domain.variables
        ma1, ma2, ma3 = table_m.domain.attributes
        ma4 = table_m.domain.metas[0]

        self.send_signal(self.widget.Inputs.additional_data, table_n, 1)
        self.send_signal(self.widget.Inputs.additional_data, table_m, 2)

        # pylint: disable=unbalanced-tuple-unpacking
        dtable_n, dtable_m = self.widget._dumb_tables()
        dna1, dna2, dna3, dna4, dnc1 = dtable_n.domain.variables
        dma1, dma2, dma3 = dtable_m.domain.attributes
        dma4 = dtable_m.domain.metas[0]

        # No copying: same name and no compute value
        self.assertIs(na1, dna1)
        self.assertIs(ma1, dma1)

        # No copying: same name and same compute value
        self.assertIs(na2, dna2)
        self.assertIs(ma2, dma2)

        # Copy: same name and different compute value
        self.assertIsNot(na3, dna3)
        self.assertIsNot(ma3, dma3)
        self.assertIsNone(dna3.compute_value)
        self.assertIsNone(dma3.compute_value)

        # No copying: same name and different compute value, but different part
        self.assertIs(na4, dna4)
        self.assertIs(ma4, dma4)

        # No copying: does not appear in the other table
        self.assertIs(nc1, dnc1)

        np.testing.assert_equal(table_m.X, dtable_m.X)
        np.testing.assert_equal(table_m.Y, dtable_m.Y)
        np.testing.assert_equal(table_n.X, dtable_n.X)
        np.testing.assert_equal(table_n.metas, dtable_n.metas)

    def test_dont_ignore_compute_value(self):
        table_n, table_m = self._create_compute_values()
        na1, na2, na3, na4, nc1 = table_n.domain.variables
        ma3 = table_m.domain.attributes[2]
        ma4 = table_m.domain.metas[0]

        self.send_signal(self.widget.Inputs.additional_data, table_n, 1)
        self.send_signal(self.widget.Inputs.additional_data, table_m, 2)

        self.widget.ignore_compute_value = False
        self.widget.commit.now()

        output = self.get_output(self.widget.Outputs.data)
        attributes = output.domain.attributes
        self.assertEqual(len(attributes), 5)
        self.assertIs(attributes[0], na1)
        self.assertIs(attributes[1], na2)
        self.assertIs(attributes[2].compute_value.variable, na3)  # renamed
        self.assertIs(attributes[3].compute_value.variable, na4)  # renamed
        self.assertIs(attributes[4].compute_value.variable, ma3)  # renamed

        self.assertIs(output.domain.class_var, nc1)

        self.assertEqual(len(output.domain.metas), 1)
        self.assertIs(output.domain.metas[0].compute_value.variable, ma4)

    def test_ignore_compute_value(self):
        table_n, table_m = self._create_compute_values()
        na1, na2, na3, na4, nc1 = table_n.domain.variables
        ma3 = table_m.domain.attributes[2]
        ma4 = table_m.domain.metas[0]

        self.send_signal(self.widget.Inputs.additional_data, table_n, 1)
        self.send_signal(self.widget.Inputs.additional_data, table_m, 2)

        self.widget.ignore_compute_value = True
        self.widget.commit.now()

        output = self.get_output(self.widget.Outputs.data)
        attributes = output.domain.attributes
        self.assertEqual(len(attributes), 4)
        self.assertIs(attributes[0], na1)
        self.assertIs(attributes[1], na2)
        self.assertIsNot(attributes[2], na3)
        self.assertIsNot(attributes[2], ma3)
        self.assertIsNone(attributes[2].compute_value, ma3)  # renamed
        self.assertEqual(attributes[2].name, na3.name)
        self.assertIs(attributes[3].compute_value.variable, na4)  # renamed

        self.assertIs(output.domain.class_var, nc1)

        self.assertEqual(len(output.domain.metas), 1)
        self.assertIs(output.domain.metas[0].compute_value.variable, ma4)  # renamed

    def test_explicit_closing(self):
        w = self.widget
        self.send_signal(w.Inputs.additional_data, self.iris[:1], 0)
        self.send_signal(w.Inputs.additional_data, self.iris[1:2], 1)
        self.send_signal(w.Inputs.additional_data, self.iris[2:3], 2)

        def assert_output_equal(expected: np.ndarray):
            out = self.get_output(w.Outputs.data)
            assert_array_equal(out.X, expected)

        assert_output_equal(self.iris[:3].X)
        self.send_signal(w.Inputs.additional_data, None, 1)
        assert_output_equal(self.iris[:3:2].X)
        self.send_signal(w.Inputs.additional_data, self.iris[1:2], 1)
        assert_output_equal(self.iris[:3].X)
        self.send_signal(w.Inputs.additional_data,
                         w.Inputs.additional_data.closing_sentinel, 1)
        assert_output_equal(self.iris[:3:2].X)
        self.send_signal(w.Inputs.additional_data, self.iris[1:2], 1)
        assert_output_equal(np.vstack((self.iris[:3:2].X, self.iris[1:2].X)))


if __name__ == "__main__":
    unittest.main()
