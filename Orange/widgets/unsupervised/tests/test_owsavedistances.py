import unittest
from unittest.mock import Mock

import numpy as np

from Orange.data import Table, Domain, StringVariable
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.tests import named_file
from Orange.widgets.unsupervised.owsavedistances import OWSaveDistances
from Orange.widgets.utils.save.tests.test_owsavebase import \
    SaveWidgetsTestBaseMixin
from Orange.widgets.tests.base import WidgetTest


class OWSaveTestBase(WidgetTest, SaveWidgetsTestBaseMixin):
    def setUp(self):
        self.widget = self.create_widget(OWSaveDistances)
        self.distances = Euclidean(Table("iris"))

    def _save_and_load(self):
        widget = self.widget
        widget.auto_save = False

        with named_file("", suffix=".dst") as filename:
            widget.get_save_filename = Mock(
                return_value=(filename, widget.filters[0]))

            self.send_signal(widget.Inputs.distances, self.distances)
            widget.save_file_as()
            return DistMatrix.from_file(filename)

    def test_save_part_labels_as_table(self):
        widget = self.widget

        # This should be how DistMatrix from table comes
        assert self.distances.row_items is not None
        assert self.distances.col_items is None

        for labels_in in ("rows", "columns"):
            distances = self._save_and_load()
            np.testing.assert_almost_equal(distances, self.distances)
            self.assertIsNone(distances.row_items,
                              msg=f"failed when labels in {labels_in}")
            self.assertIsNone(distances.col_items,
                              msg=f"failed when labels in {labels_in}")
            self.assertFalse(widget.Warning.table_not_saved.is_shown(),
                             msg=f"failed when labels in {labels_in}")
            self.assertTrue(widget.Warning.part_not_saved.is_shown(),
                            msg=f"failed when labels in {labels_in}")
            self.assertTrue(
                labels_in in widget.Warning.part_not_saved.formatted,
                msg=f"failed when labels in {labels_in}")

            # switch from columns to rows for the second test
            self.distances.col_items = self.distances.row_items
            self.distances.row_items = None

    def test_save_both_labels_as_table(self):
        widget = self.widget

        # This should be how DistMatrix from table comes
        assert self.distances.row_items is not None
        assert self.distances.col_items is None

        self.distances.col_items = self.distances.row_items

        distances = self._save_and_load()
        np.testing.assert_almost_equal(distances, self.distances)
        self.assertIsNone(distances.row_items)
        self.assertIsNone(distances.col_items)
        self.assertTrue(widget.Warning.table_not_saved.is_shown())
        self.assertFalse(widget.Warning.part_not_saved.is_shown())

    def test_save_no_labels(self):
        widget = self.widget
        self.distances.row_items = self.distances.col_items = None
        distances = self._save_and_load()
        np.testing.assert_almost_equal(distances, self.distances)
        self.assertIsNone(distances.row_items)
        self.assertIsNone(distances.col_items)
        self.assertFalse(widget.Warning.table_not_saved.is_shown())
        self.assertFalse(widget.Warning.part_not_saved.is_shown())

    def test_save_trivial_labels(self):
        widget = self.widget
        domain = Domain([], [], [StringVariable("label")])
        n = len(self.distances)
        col_labels = Table.from_list(domain, [[str(x)] for x in range(n)])
        row_labels = Table.from_list(domain,
                                     [[str(x)] for x in range(1, 1 + n)])

        for self.distances.col_items, self.distances.row_items in (
                (col_labels, None), (None, row_labels),
                (col_labels, row_labels)):
            distances = self._save_and_load()
            np.testing.assert_almost_equal(distances, self.distances)
            self.assert_table_equal(distances.row_items,
                                    self.distances.row_items)
            self.assert_table_equal(distances.col_items,
                                    self.distances.col_items)
            self.assertFalse(widget.Warning.table_not_saved.is_shown())
            self.assertFalse(widget.Warning.part_not_saved.is_shown())

    def test_send_report(self):
        widget = self.widget

        widget.send_report()

        widget.filename = "test.dst"
        widget.send_report()

        self.send_signal(widget.Inputs.distances, self.distances)
        widget.send_report()


if __name__ == "__main__":
    unittest.main()
