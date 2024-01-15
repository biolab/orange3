from unittest.mock import patch

import random
import unittest

import numpy as np

from Orange.data import Table, Domain, StringVariable, DiscreteVariable, \
    ContinuousVariable
from Orange.data.filter import SameValue
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, create_groups_table, ANNOTATED_DATA_FEATURE_NAME,
    lazy_annotated_table, lazy_groups_table, domain_with_annotation_column
)


class TestAnnotatedData(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.zoo = Table("zoo")

    def test_domain_with_annotation_column(self):
        a, b, c = (ContinuousVariable(x) for x in "abc")

        x = [[1, 2, 3], [4, 5, 6]]

        for data in (dabc := Domain([a, b, c]), Table.from_list(dabc, x)):
            dom, var = domain_with_annotation_column(data)
            self.assertEqual(dom.attributes, (a, b, c))
            self.assertIs(dom.class_var, var)
            self.assertEqual(var.name, ANNOTATED_DATA_FEATURE_NAME)
            self.assertEqual(var.values, ("No", "Yes"))

            dom, var = domain_with_annotation_column(
                data, values=tuple("xyz"), var_name="d")
            self.assertEqual(dom.attributes, (a, b, c))
            self.assertIs(dom.class_var, var)
            self.assertEqual(var.name, "d")
            self.assertEqual(var.values, tuple("xyz"))

        for data in (dabc := Domain([a, b], c), Table.from_list(dabc, x)):
            dom, var = domain_with_annotation_column(
                data, values=tuple("xyz"), var_name="d")
            self.assertEqual(dom.attributes, (a, b))
            self.assertIs(dom.class_var, c)
            self.assertEqual(dom.metas, (var, ))
            self.assertEqual(var.name, "d")
            self.assertEqual(var.values, tuple("xyz"))

            dom, var = domain_with_annotation_column(
                data, values=tuple("xyz"), var_name="c")
            self.assertEqual(dom.attributes, (a, b))
            self.assertIs(dom.class_var, c)
            self.assertEqual(dom.metas, (var, ))
            self.assertEqual(var.name, "c (1)")
            self.assertEqual(var.values, tuple("xyz"))

    def test_create_annotated_table(self):
        annotated = create_annotated_table(self.zoo, list(range(10)))

        # check annotated table domain
        self.assertEqual(annotated.domain.variables, self.zoo.domain.variables)
        self.assertEqual(2, len(annotated.domain.metas))
        self.assertIn(self.zoo.domain.metas[0], annotated.domain.metas)
        self.assertIn(ANNOTATED_DATA_FEATURE_NAME,
                      [m.name for m in annotated.domain.metas])

        # check annotated table data
        np.testing.assert_array_equal(annotated.X, self.zoo.X)
        np.testing.assert_array_equal(annotated.Y, self.zoo.Y)
        np.testing.assert_array_equal(annotated.metas[:, 0].ravel(),
                                      self.zoo.metas.ravel())
        self.assertEqual(
            10, np.sum([i[ANNOTATED_DATA_FEATURE_NAME] for i in annotated]))

    def test_create_annotated_table_selected(self):
        # check annotated column for no selected indices
        annotated = create_annotated_table(self.zoo, [])
        self.assertEqual(len(annotated), len(self.zoo))
        self.assertEqual(
            0, np.sum([i[ANNOTATED_DATA_FEATURE_NAME] for i in annotated]))

        # check annotated column fol all selectes indices
        annotated = create_annotated_table(self.zoo, list(range(len(self.zoo))))
        self.assertEqual(len(annotated), len(self.zoo))
        self.assertEqual(
            len(self.zoo),
            np.sum([i[ANNOTATED_DATA_FEATURE_NAME] for i in annotated]))

    def test_create_annotated_table_none_data(self):
        self.assertIsNone(create_annotated_table(None, None))

    def test_create_annotated_table_none_indices(self):
        annotated = create_annotated_table(self.zoo, None)
        self.assertEqual(len(annotated), len(self.zoo))
        self.assertEqual(
            0, np.sum([i[ANNOTATED_DATA_FEATURE_NAME] for i in annotated]))

    def _renamed_zoo_meta(self, name, name0=None):
        zoo = self.zoo
        zood = self.zoo.domain
        var0 = zood.attributes[0]
        name0 = name0 or var0.name
        domain = Domain(
            (DiscreteVariable(name0, var0.values), ) + zood.attributes[1:],
            zood.class_var, [StringVariable(name)])
        return Table(domain, zoo.X, zoo.Y, zoo.metas)

    def test_cascade_annotated_tables(self):
        # check cascade of annotated tables
        data = self._renamed_zoo_meta(ANNOTATED_DATA_FEATURE_NAME)
        first_meta = data.domain.metas[0]
        for i in range(5):
            data = create_annotated_table(
                data, random.sample(range(0, len(data)), 20))
            self.assertEqual(2 + i, len(data.domain.metas))
            self.assertIn(first_meta, data.domain.metas)
            self.assertIn(ANNOTATED_DATA_FEATURE_NAME,
                          [m.name for m in data.domain.metas])
            for j in range(1, i + 2):
                self.assertIn("{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, j),
                              [m.name for m in data.domain.metas])

    def test_cascade_annotated_tables_with_missing_middle_feature(self):
        # check table for domain [..., "Feature", "Selected", "Selected (3)] ->
        # [..., "Feature", "Selected", "Selected (3), "Selected (4)"]
        data = self._renamed_zoo_meta(f"{ANNOTATED_DATA_FEATURE_NAME} (3)",
                                      ANNOTATED_DATA_FEATURE_NAME)
        data = create_annotated_table(
            data, random.sample(range(0, len(self.zoo)), 20))
        self.assertEqual(2, len(data.domain.metas))
        self.assertEqual(data.domain.attributes[0].name,
                         ANNOTATED_DATA_FEATURE_NAME)
        self.assertEqual(data.domain.metas[0].name,
                         "{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, 3))
        self.assertEqual(data.domain.metas[1].name,
                         "{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, 4))

    def test_cascade_annotated_tables_with_missing_annotated_feature(self):
        # check table for domain [..., "Feature", "Selected (3)] ->
        # [..., "Feature", "Selected (3), "Selected (4)"]
        data = self._renamed_zoo_meta(f"{ANNOTATED_DATA_FEATURE_NAME} (3)")
        data = create_annotated_table(
            data, random.sample(range(0, len(self.zoo)), 20))
        self.assertEqual(2, len(data.domain.metas))
        self.assertEqual(data.domain.metas[0].name,
                         "{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, 3))
        self.assertEqual(data.domain.metas[1].name,
                         "{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, 4))

    def test_create_groups_table_include_unselected(self):
        group_indices = random.sample(range(0, len(self.zoo)), 20)
        selection = np.zeros(len(self.zoo), dtype=np.uint8)
        selection[group_indices[:10]] = 1
        selection[group_indices[10:]] = 2
        table = create_groups_table(self.zoo, selection)
        selvar = table.domain["Selected"]
        self.assertEqual(
            len(SameValue(selvar, "Unselected")(table)),
            len(self.zoo) - len(group_indices)
        )
        self.assertEqual(selvar.values, ("G1", "G2", "Unselected"))

    def test_create_groups_table_set_values(self):
        group_indices = random.sample(range(0, len(self.zoo)), 20)
        selection = np.zeros(len(self.zoo), dtype=np.uint8)
        selection[group_indices[:10]] = 1
        selection[group_indices[10:]] = 2
        values = ("this", "that", "rest")
        table = create_groups_table(self.zoo, selection, values=values)
        self.assertEqual(tuple(table.domain["Selected"].values), values)

    @patch("Orange.widgets.utils.annotated_data.create_annotated_table")
    def test_lazy_annotated_table(self, creator):
        selected_indices = np.array([1, 2, 3])
        lazy_table = lazy_annotated_table(self.zoo, selected_indices)
        self.assertEqual(lazy_table.length, len(self.zoo))
        self.assertEqual(lazy_table.domain.attributes, self.zoo.domain.attributes)
        self.assertEqual(lazy_table.domain.class_var, self.zoo.domain.class_var)
        self.assertEqual(len(lazy_table.domain.metas), 2)
        var = lazy_table.domain.metas[1]
        self.assertIsInstance(var, DiscreteVariable)
        self.assertEqual(var.name, ANNOTATED_DATA_FEATURE_NAME)
        creator.assert_not_called()
        self.assertIs(lazy_table.get_value(), creator.return_value)

    @patch("Orange.widgets.utils.annotated_data.create_groups_table")
    def test_lazy_groups_table(self, creator):
        group_indices = np.zeros(len(self.zoo), dtype=int)
        group_indices[10:15] = 1

        lazy_table = lazy_groups_table(self.zoo, group_indices)
        self.assertEqual(lazy_table.length, len(self.zoo))
        self.assertEqual(lazy_table.domain.attributes, self.zoo.domain.attributes)
        self.assertEqual(lazy_table.domain.class_var, self.zoo.domain.class_var)
        self.assertEqual(len(lazy_table.domain.metas), 2)
        var = lazy_table.domain.metas[1]
        self.assertIsInstance(var, DiscreteVariable)
        self.assertEqual(var.name, ANNOTATED_DATA_FEATURE_NAME)
        creator.assert_not_called()
        self.assertIs(lazy_table.get_value(), creator.return_value)
        creator.reset_mock()

        lazy_table = lazy_groups_table(
            self.zoo, group_indices, include_unselected=False, var_name="foo",
            values=("bar", "baz"))
        self.assertEqual(lazy_table.length, 5)
        self.assertEqual(lazy_table.domain.attributes, self.zoo.domain.attributes)
        self.assertEqual(lazy_table.domain.class_var, self.zoo.domain.class_var)
        self.assertEqual(len(lazy_table.domain.metas), 2)
        var = lazy_table.domain.metas[1]
        self.assertIsInstance(var, DiscreteVariable)
        self.assertEqual(var.name, "foo")
        self.assertEqual(var.values, ("bar", "baz"))
        creator.assert_not_called()
        self.assertIs(lazy_table.get_value(), creator.return_value)


if __name__ == "__main__":
    unittest.main()
