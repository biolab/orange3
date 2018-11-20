import random
import unittest

import numpy as np

from Orange.data import Table, Variable
from Orange.data.filter import SameValue
from Orange.data.util import get_unique_names
from Orange.widgets.utils.annotated_data import (
    create_annotated_table, get_next_name,
    create_groups_table, ANNOTATED_DATA_FEATURE_NAME
)


class TestGetNextName(unittest.TestCase):
    def test_get_var_name(self):
        self.assertEqual(get_next_name(["a"], "XX"), "XX")
        self.assertEqual(get_next_name(["a", "XX"], "XX"), "XX (1)")
        self.assertEqual(get_next_name(["a", "XX (4)"], "XX"), "XX (5)")
        self.assertEqual(get_next_name(["a", "XX", "XX (4)"], "XX"), "XX (5)")


class TestAnnotatedData(unittest.TestCase):
    def setUp(self):
        Variable._clear_all_caches()  # pylint: disable=protected-access
        random.seed(42)
        self.zoo = Table("zoo")

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

    def test_cascade_annotated_tables(self):
        # check cascade of annotated tables
        data = self.zoo
        data.domain.metas[0].name = ANNOTATED_DATA_FEATURE_NAME
        for i in range(5):
            data = create_annotated_table(
                data, random.sample(range(0, len(self.zoo)), 20))
            self.assertEqual(2 + i, len(data.domain.metas))
            self.assertIn(self.zoo.domain.metas[0], data.domain.metas)
            self.assertIn(ANNOTATED_DATA_FEATURE_NAME,
                          [m.name for m in data.domain.metas])
            for j in range(1, i + 2):
                self.assertIn("{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, j),
                              [m.name for m in data.domain.metas])

    def test_cascade_annotated_tables_with_missing_middle_feature(self):
        # check table for domain [..., "Feature", "Selected", "Selected (3)] ->
        # [..., "Feature", "Selected", "Selected (3), "Selected (4)"]
        data = self.zoo
        data.domain.attributes[0].name = ANNOTATED_DATA_FEATURE_NAME
        data.domain.metas[0].name = "{} ({})".format(
            ANNOTATED_DATA_FEATURE_NAME, 3)
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
        data = self.zoo
        data.domain.metas[0].name = "{} ({})".format(
            ANNOTATED_DATA_FEATURE_NAME, 3)
        data = create_annotated_table(
            data, random.sample(range(0, len(self.zoo)), 20))
        self.assertEqual(2, len(data.domain.metas))
        self.assertEqual(data.domain.metas[0].name,
                         "{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, 3))
        self.assertEqual(data.domain.metas[1].name,
                         "{} ({})".format(ANNOTATED_DATA_FEATURE_NAME, 4))

    def test_get_unique_names(self):
        names = ["charlie", "bravo", "charlie (2)", "charlie (3)", "bravo (2)", "charlie (4)",
                 "bravo (3)"]
        self.assertEqual(get_unique_names(names, ["bravo", "charlie"]),
                         ["bravo (5)", "charlie (5)"])

    def test_create_groups_table_include_unselected(self):
        group_indices = random.sample(range(0, len(self.zoo)), 20)
        selection = np.zeros(len(self.zoo), dtype=np.uint8)
        selection[group_indices[:10]] = 1
        selection[group_indices[10:]] = 2
        table = create_groups_table(self.zoo, selection)
        self.assertEqual(
            len(SameValue(table.domain["Selected"], "Unselected")(table)),
            len(self.zoo) - len(group_indices)
        )
