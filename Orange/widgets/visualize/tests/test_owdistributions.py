# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,protected-access

from Orange.data import Table, Domain, DiscreteVariable
from Orange.data.table import dataset_dirs
from Orange.tests import test_dirname
from Orange.widgets.tests.base import WidgetTest, datasets
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
        self.send_signal(self.widget.Inputs.data, self.data)

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
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.cb_prob.count(), 5)
        self.assertEqual(self.widget.groupvarview.count(), 2)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.cb_prob.count(), 0)
        self.assertEqual(self.widget.groupvarview.count(), 0)

    def test_discretize_meta(self):
        """The widget discretizes continuous meta attributes"""
        domain = self.iris.domain
        mdomain = Domain(domain.attributes[:-1], domain.class_var,
                         metas=domain.attributes[-1:])
        miris = Table(mdomain, self.iris)
        self.send_signal(self.widget.Inputs.data, miris)
        widget = self.widget
        widget.disc_cont = True
        widget.varview.selectionModel().select(
            widget.varview.model().index(4, 0))
        self.assertIsInstance(widget.var, DiscreteVariable)
        self.assertEqual(widget.var.name, mdomain.metas[0].name)

    def test_variable_group_combinations(self):
        """Check widget for all combinations of variable and group for dataset
        with constant columns and missing data"""
        self.send_signal(self.widget.Inputs.data, Table(datasets.path("testing_dataset_cls")))
        for groupvar_idx in range(len(self.widget.groupvarmodel)):
            self.widget.groupvar_idx = groupvar_idx
            for var_idx in range(len(self.widget.varmodel)):
                self.widget.variable_idx = var_idx
                self.widget._setup()

    def test_no_distributions(self):
        """
        Do not fail when there is no data and when sb clicks
        "Show relative frequencies".
        GH-2383
        GH-2428
        """
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.cb_rel_freq.click()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.cb_rel_freq.setChecked(False)
        self.widget.cb_rel_freq.click()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.cb_rel_freq.setChecked(True)
        self.widget.cb_rel_freq.click()
