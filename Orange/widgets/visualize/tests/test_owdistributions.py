# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from Orange.data import Table, Domain, DiscreteVariable
from Orange.data.table import dataset_dirs
from Orange.tests import test_dirname
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.visualize.owdistributions import OWDistributions


class TestOWDistributions(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        dataset_dirs.append(test_dirname())
        cls.data = Table("test9.tab")
        cls.iris = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWDistributions)

    def test_metas(self):
        self.send_signal("Data", self.data)

        # check metas in list views
        for meta in self.data.domain.metas:
            if meta.is_discrete or meta.is_continuous:
                self.assertIn(meta, self.widget.varmodel)
        for meta in self.data.domain.metas:
            if meta.is_discrete:
                self.assertIn(meta, self.widget.groupvarmodel)

        # select meta attribute
        self.widget.cb_disc_cont.setChecked(True)
        self.widget.variable_idx = 2
        self.widget._setup()

    def test_remove_data(self):
        """Check widget when data is removed"""
        self.send_signal("Data", self.iris)
        self.assertEqual(self.widget.cb_prob.count(), 5)
        self.assertEqual(self.widget.groupvarview.count(), 2)
        self.send_signal("Data", None)
        self.assertEqual(self.widget.cb_prob.count(), 0)
        self.assertEqual(self.widget.groupvarview.count(), 0)

    def test_discretize_meta(self):
        """The widget discretizes continuous meta attributes"""
        domain = self.iris.domain
        mdomain = Domain(domain.attributes[:-1], domain.class_var,
                         metas=domain.attributes[-1:])
        miris = Table(mdomain, self.iris)
        self.send_signal("Data", miris)
        widget = self.widget
        widget.disc_cont = True
        widget.varview.selectionModel().select(
            widget.varview.model().index(4, 0))
        self.assertIsInstance(widget.var, DiscreteVariable)
        self.assertEqual(widget.var.name, mdomain.metas[0].name)
