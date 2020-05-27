import unittest

import numpy as np

from orangewidget.settings import SettingProvider
from orangewidget.widget import OWBaseWidget
from Orange.data import DiscreteVariable, ContinuousVariable, StringVariable, \
    TimeVariable, Domain, Table
from Orange.widgets.tests.base import GuiTest
from Orange.widgets.utils.domaineditor import DomainEditor, Column


class MockWidget(OWBaseWidget):
    name = "mock"
    domain_editor = SettingProvider(DomainEditor)

    def __init__(self):
        self.domain_editor = DomainEditor(self)


class DomainEditorTest(GuiTest):
    def setUp(self):
        self.widget = MockWidget()
        self.editor = self.widget.domain_editor

        self.orig_variables = [
            ["d1", DiscreteVariable, 0, "x, y, z, ...", False],
            ["d2", DiscreteVariable, 0, "1, 2, 3, ...", True],
            ["c1", ContinuousVariable, 0, "", True],
            ["d3", DiscreteVariable, 1, "4, 3, 6, ...", True],
            ["s", StringVariable, 2, "", False],
            ["t", TimeVariable, 2, "", True]
        ]
        self.domain = Domain(
            [DiscreteVariable("d1", values=list("xyzw")),
             DiscreteVariable("d2", values=list("12345")),
             ContinuousVariable("c1")],
            DiscreteVariable("d3", values=list("4368")),
            [StringVariable("s"),
             TimeVariable("t")])

    def test_deduplication(self):
        editor = self.editor
        model = editor.model()
        model.set_orig_variables(self.orig_variables)
        model.reset_variables()

        data = Table.from_numpy(
            self.domain,
            np.zeros((1, 3)), np.zeros((1, 1)), np.array([["foo", 42]]))

        # No duplicates

        domain, _ = \
            editor.get_domain(self.domain, data)
        self.assertEqual([var.name for var in domain.attributes],
                         ["d1", "d2", "c1"])
        self.assertEqual([var.name for var in domain.class_vars],
                         ["d3"])
        self.assertEqual([var.name for var in domain.metas],
                         ["s", "t"])

        domain, _, renamed = \
            editor.get_domain(self.domain, data, deduplicate=True)
        self.assertEqual([var.name for var in domain.attributes],
                         ["d1", "d2", "c1"])
        self.assertEqual([var.name for var in domain.class_vars],
                         ["d3"])
        self.assertEqual([var.name for var in domain.metas],
                         ["s", "t"])
        self.assertEqual(renamed, [])


        # Duplicates

        model.setData(model.index(3, Column.name), "d2")
        model.setData(model.index(5, Column.name), "s")

        domain, _, renamed = \
            editor.get_domain(self.domain, data, deduplicate=True)
        self.assertEqual([var.name for var in domain.attributes],
                         ["d1", "d2 (1)", "c1"])
        self.assertEqual([var.name for var in domain.class_vars],
                         ["d2 (2)"])
        self.assertEqual([var.name for var in domain.metas],
                         ["s (1)", "s (2)"])
        self.assertEqual(renamed, ["d2", "s"])


        # Duplicates, some skipped

        model.setData(model.index(5, Column.place), "skip")

        domain, _, renamed = \
            editor.get_domain(self.domain, data, deduplicate=True)
        self.assertEqual([var.name for var in domain.attributes],
                         ["d1", "d2 (1)", "c1"])
        self.assertEqual([var.name for var in domain.class_vars],
                         ["d2 (2)"])
        self.assertEqual([var.name for var in domain.metas],
                         ["s"])
        self.assertEqual(renamed, ["d2"])


if __name__ == "__main__":
    unittest.main()
