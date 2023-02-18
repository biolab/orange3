# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring,unsubscriptable-object,protected-access
import unittest
from itertools import chain
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import Qt, QModelIndex, QItemSelectionModel
from AnyQt.QtTest import QSignalSpy

from orangewidget.tests.base import GuiTest

from Orange.data import Table, DiscreteVariable, ContinuousVariable, Domain
from Orange.widgets.data.owcontinuize import OWContinuize, DefaultKey, \
    ContinuousOptions, Normalize, Continuize, DiscreteOptions, \
    ContDomainModel, DefaultContModel, ListViewSearch
from Orange.widgets.tests.base import WidgetTest
from orangewidget.utils.itemmodels import SeparatedListDelegate


class TestOWContinuize(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWContinuize)

    def test_empty_data(self):
        """No crash on empty data"""
        data = Table("iris")
        widget = self.widget

        self.send_signal(self.widget.Inputs.data, data)
        widget.commit.now()

        self.send_signal(self.widget.Inputs.data,
                         Table.from_domain(data.domain))
        widget.commit.now()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

        self.send_signal(self.widget.Inputs.data, None)
        widget.commit.now()
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_continuous(self):
        table = Table("housing")
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.commit.now()

    def test_one_column_equal_values(self):
        table = Table("iris")
        table = table[:, 1].copy()
        with table.unlocked():
            table[:] = 42.0
        self.send_signal(self.widget.Inputs.data, table)
        # Normalize.NormalizeBySD
        self.widget.continuous_treatment = 2
        self.widget.commit.now()

    def test_one_column_nan_values_normalize_sd(self):
        table = Table("iris")
        with table.unlocked():
            table[:, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        # Normalize.NormalizeBySD
        self.widget.continuous_treatment = 2
        self.widget.commit.now()

        table = Table("iris")
        with table.unlocked():
            table[1, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.commit.now()

    def test_one_column_nan_values_normalize_span(self):
        table = Table("iris")
        with table.unlocked():
            table[:, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        # Normalize.NormalizeBySpan
        self.widget.continuous_treatment = 1
        self.widget.commit.now()

        table = Table("iris")
        with table.unlocked():
            table[1, 2] = np.NaN
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.commit.now()

    def test_commit_calls_prepare_output(self):
        # This test ensures that commit returns the result of _prepare_output,
        # so further tests can just check the latter. If this is changed, the
        # test will fail, which is OK - test can be removed, but other tests
        # then have to check the output and not just _prepare_output.
        out = object()
        self.widget._prepare_output = lambda: out
        self.widget.Outputs.data.send = Mock()
        self.widget.commit.now()
        self.widget.Outputs.data.send.assert_called_with(out)

    def test_check_unsuppoerted_sparse_continuous(self):
        # This test checks response at two points:
        # - when scaling sparse data with a method that does not support it,
        #   the wiget must show an error and output nothing
        # - the above is tested via method _unsupported_sparse, so we also
        #   directly check this method
        w = self.widget
        hints = w.cont_var_hints
        iris = Table("iris")
        sparse_iris = iris.to_sparse()

        # input dense
        for key in (DefaultKey, "sepal length"):
            hints[DefaultKey] = Normalize.Leave
            for hints[key], desc in ContinuousOptions.items():
                if desc.id_ == Normalize.Default:
                    continue
                msg = f"at {desc.label}, " + ("default" if key is DefaultKey else key)

                self.send_signal(w.Inputs.data, iris)
                self.assertFalse(w._unsupported_sparse(), msg)
                self.assertFalse(w.Error.unsupported_sparse.is_shown(), msg)
                self.assertIsNotNone(self.get_output(w.Outputs.data), msg)

                self.send_signal(w.Inputs.data, sparse_iris)
                self.assertIsNot(w._unsupported_sparse(), desc.supports_sparse, msg)
                self.assertIsNot(w.Error.unsupported_sparse.is_shown(), desc.supports_sparse, msg)
                if desc.supports_sparse:
                    self.assertIsNotNone(self.get_output(w.Outputs.data), msg)
                else:
                    self.assertIsNone(self.get_output(w.Outputs.data), msg)
                    self.send_signal(w.Inputs.data, None)
                    self.assertFalse(w.Error.unsupported_sparse.is_shown(), msg)

    def test_check_unsuppoerted_sparse_discrete(self):
        # This test checks response at two points:
        # - when scaling sparse data with a method that does not support it,
        #   the wiget must show an error and output nothing
        # - the above is tested via method _unsupported_sparse, so we also
        #   directly check this method
        w = self.widget
        hints = w.disc_var_hints
        zoo = Table("zoo")
        sparse_zoo = zoo.to_sparse()

        # input dense
        for key in (DefaultKey, "legs"):
            hints[DefaultKey] = Continuize.Leave
            for hints[key], desc in DiscreteOptions.items():
                if desc.id_ == Continuize.Default:
                    continue
                msg = f"at {desc.label}, " + ("default" if key is DefaultKey else key)

                self.send_signal(w.Inputs.data, zoo)
                self.assertFalse(w._unsupported_sparse(), msg)
                self.assertFalse(w.Error.unsupported_sparse.is_shown(), msg)
                self.assertIsNotNone(self.get_output(w.Outputs.data), msg)

                self.send_signal(w.Inputs.data, sparse_zoo)
                self.assertIsNot(w._unsupported_sparse(), desc.supports_sparse, msg)
                self.assertIsNot(w.Error.unsupported_sparse.is_shown(), desc.supports_sparse, msg)
                if desc.supports_sparse:
                    self.assertIsNotNone(self.get_output(w.Outputs.data), msg)
                else:
                    self.assertIsNone(self.get_output(w.Outputs.data), msg)
                    self.send_signal(w.Inputs.data, None)
                    self.assertFalse(w.Error.unsupported_sparse.is_shown(), msg)

    def test_update_cont_radio_buttons(self):
        w = self.widget
        w.disc_var_hints[DefaultKey] = Continuize.AsOrdinal
        w.disc_var_hints["chest pain"] = w.disc_var_hints["rest ECG"] = Continuize.Remove
        w.disc_var_hints["exerc ind ang"] = Continuize.FirstAsBase

        w.cont_var_hints[DefaultKey] = Normalize.Center
        w.cont_var_hints["cholesterol"] = Normalize.Scale

        self.send_signal(w.Inputs.data, Table("heart_disease"))

        dview = w.disc_view
        dmod = dview.model()
        dselmod = dview.selectionModel()
        dgroup = w.disc_group

        with patch.object(w, "_update_radios") as upd:
            w._on_var_selection_changed(dview)
            upd.assert_not_called()

        dselmod.select(dmod.index(1, 0), QItemSelectionModel.ClearAndSelect)  # chest_pain
        self.assertEqual(dgroup.checkedId(), Continuize.Remove)
        self.assertTrue(dgroup.button(99).isEnabled())

        dselmod.select(dmod.index(2, 0), QItemSelectionModel.ClearAndSelect)  # blood sugar
        self.assertEqual(dgroup.checkedId(), Continuize.Default)

        dselmod.select(dmod.index(3, 0), QItemSelectionModel.ClearAndSelect)  # rest ECG
        self.assertEqual(dgroup.checkedId(), Continuize.Remove)

        dselmod.select(dmod.index(4, 0), QItemSelectionModel.ClearAndSelect)  # exerc ind ang
        self.assertEqual(dgroup.checkedId(), Continuize.FirstAsBase)

        dselmod.select(dmod.index(3, 0), QItemSelectionModel.Select)  # read ECG and exerc ind ang
        self.assertEqual(dgroup.checkedId(), -1)

        dview.select_default()
        self.assertEqual(dgroup.checkedId(), Continuize.AsOrdinal)
        self.assertFalse(dgroup.button(99).isEnabled())

        cview = w.cont_view
        cmod = cview.model()
        cselmod = cview.selectionModel()
        cgroup = w.cont_group

        cselmod.select(cmod.index(2, 0), QItemSelectionModel.ClearAndSelect)  # cholesterol
        self.assertEqual(cgroup.checkedId(), Normalize.Scale)
        self.assertEqual(dgroup.checkedId(), Continuize.AsOrdinal)
        self.assertTrue(cgroup.button(99).isEnabled())

        cview.select_default()
        self.assertEqual(cgroup.checkedId(), Normalize.Center)
        self.assertEqual(dgroup.checkedId(), Continuize.AsOrdinal)
        self.assertFalse(cgroup.button(99).isEnabled())

        w._uncheck_all_buttons(cgroup)
        self.assertEqual(cgroup.checkedId(), -1)
        self.assertEqual(dgroup.checkedId(), Continuize.AsOrdinal)

        w._uncheck_all_buttons(dgroup)
        self.assertEqual(dgroup.checkedId(), -1)

    def test_set_hints_on_new_data(self):
        w = self.widget
        domain = Domain([ContinuousVariable(c) for c in "abc"] +
                        [DiscreteVariable("m", values=tuple("xy"))],
                        ContinuousVariable("d"),
                        [ContinuousVariable(c) for c in "ef"])
        data = Table.from_list(domain, [[0] * 6])

        w.cont_var_hints["b"] = Normalize.Leave
        w.cont_var_hints["f"] = Normalize.Normalize11
        w.cont_var_hints["x"] = Normalize.Normalize11

        self.send_signal(w.Inputs.data, None)
        self.send_signal(w.Inputs.data, data)

        model = w.cont_view.model()
        self.assertEqual(model.index(0, 0).data(model.HintRole), ("preset", False))
        self.assertEqual(model.index(1, 0).data(model.HintRole),
                         (ContinuousOptions[Normalize.Leave].short_desc, True))
        self.assertEqual(model.index(5, 0).data(model.HintRole),
                         (ContinuousOptions[Normalize.Normalize11].short_desc, True))
        self.assertNotIn("x", w.cont_var_hints)

    def test_change_hints_disc(self):
        w = self.widget
        w.disc_var_hints[DefaultKey] = Continuize.AsOrdinal
        w.disc_var_hints["chest pain"] = w.disc_var_hints["rest ECG"] = Continuize.Remove
        w.disc_var_hints["exerc ind ang"] = Continuize.FirstAsBase

        dview = w.disc_view
        dmod = dview.model()
        dselmod = dview.selectionModel()
        dgroup = w.disc_group

        self.send_signal(w.Inputs.data, Table("heart_disease"))
        self.assertEqual(
            dmod.index(3, 0).data(dmod.HintRole),
            (DiscreteOptions[Continuize.Remove].short_desc, True))

        dselmod.select(dmod.index(1, 0), QItemSelectionModel.ClearAndSelect)  # chest pain
        dselmod.select(dmod.index(4, 0), QItemSelectionModel.Select)  # exerc ind ang
        dgroup.button(Continuize.AsOrdinal).setChecked(True)
        dgroup.idClicked.emit(Continuize.AsOrdinal)

        self.assertFalse("gender" in w.disc_var_hints)
        self.assertEqual(w.disc_var_hints["chest pain"], Continuize.AsOrdinal)
        self.assertEqual(w.disc_var_hints["exerc ind ang"], Continuize.AsOrdinal)
        self.assertEqual(w.disc_var_hints["rest ECG"], Continuize.Remove)

        dselmod.select(dmod.index(1, 0), QItemSelectionModel.ClearAndSelect)  # chest pain
        dselmod.select(dmod.index(0, 0), QItemSelectionModel.Select)  # gender
        dgroup.button(99).setChecked(True)
        dgroup.idClicked.emit(99)
        self.assertFalse("chest pain" in w.disc_var_hints)
        self.assertFalse("gender" in w.disc_var_hints)
        self.assertEqual(w.disc_var_hints["rest ECG"], Continuize.Remove)

        self.assertEqual(dmod.index(0, 0).data(dmod.HintRole), ("preset", False))
        self.assertEqual(
            dmod.index(3, 0).data(dmod.HintRole),
            (DiscreteOptions[Continuize.Remove].short_desc, True))

        dview.select_default()
        dgroup.button(Continuize.AsOrdinal).setChecked(True)
        dgroup.idClicked.emit(Continuize.AsOrdinal)
        self.assertEqual(w.disc_var_hints[DefaultKey], Continuize.AsOrdinal)

    def test_change_hints_cont(self):
        w = self.widget
        w.cont_var_hints[DefaultKey] = Normalize.Center
        w.cont_var_hints["cholesterol"] = Normalize.Scale

        self.send_signal(w.Inputs.data, Table("heart_disease"))

        cview = w.cont_view
        cmod = cview.model()
        cselmod = cview.selectionModel()
        cgroup = w.cont_group

        cselmod.select(cmod.index(2, 0), QItemSelectionModel.ClearAndSelect)  # cholesterol
        cselmod.select(cmod.index(3, 0), QItemSelectionModel.Select)  # max HR
        cgroup.button(Normalize.Normalize11).setChecked(True)
        cgroup.idClicked.emit(Normalize.Normalize11)

        self.assertFalse("age" in w.cont_var_hints)
        self.assertEqual(w.cont_var_hints["cholesterol"], Normalize.Normalize11)
        self.assertEqual(w.cont_var_hints["max HR"], Normalize.Normalize11)

        cselmod.select(cmod.index(2, 0), QItemSelectionModel.ClearAndSelect)  # cholesterol
        cselmod.select(cmod.index(0, 0), QItemSelectionModel.Select)  # age
        cgroup.button(99).setChecked(True)
        cgroup.idClicked.emit(99)
        self.assertFalse("age" in w.cont_var_hints)
        self.assertFalse("cholesterol" in w.cont_var_hints)
        self.assertEqual(w.cont_var_hints["max HR"], Normalize.Normalize11)

        self.assertEqual(cmod.index(0, 0).data(cmod.HintRole), ("preset", False))
        self.assertEqual(
            cmod.index(3, 0).data(cmod.HintRole),
            (ContinuousOptions[Normalize.Normalize11].short_desc, True))

    def test_transformations(self):
        domain = Domain([DiscreteVariable(c, values="abc")
                         for c in ("default", "leave", "first", "frequent",
                                   "one-hot", "remove-if", "remove", "ordinal",
                                   "normordinal")],
                         DiscreteVariable("y", values="abc"),
                        [ContinuousVariable(c)
                         for c in ("cdefault", "cleave",
                                   "cstandardize", "ccenter", "cscale",
                                   "cnormalize11", "cnormalize01")]
                        )
        data = Table.from_list(domain,
                               [[x] * 17 for x in range(3)] + [[2] * 17])

        w = self.widget
        w.disc_var_hints = {
            var.name: id_
            for var, id_ in zip(domain.attributes, DiscreteOptions)
            if id_ != 99
        }
        w.disc_var_hints[DefaultKey] = Continuize.FrequentAsBase

        w.cont_var_hints = {
            var.name: id_
            for var, id_ in zip(domain.metas, ContinuousOptions)
            if id_ != 99
        }
        w.cont_var_hints[DefaultKey] = Normalize.Center

        w.continuize_class = True

        self.send_signal(w.Inputs.data, data)
        outp = self.get_output(w.Outputs.data)

        np.testing.assert_almost_equal(
            outp.X,
            [[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
             [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0.5],
             [0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 2, 1],
             [0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 2, 1],
             ]
        )
        np.testing.assert_almost_equal(
            outp.Y,
            [0, 1, 2, 2]
        )
        np.testing.assert_almost_equal(
            outp.metas,
            [[-1.25, 0, -1.50755672, -1.25, 0, -1, 0],
             [-0.25, 1, -0.30151134, -0.25, 1.20604538, 0, 0.5],
             [0.75, 2, 0.90453403, 0.75, 2.41209076, 1, 1],
             [0.75, 2, 0.90453403, 0.75, 2.41209076, 1, 1],
             ]
        )

    def test_send_report(self):
        w = self.widget
        self.send_signal(w.Inputs.data, Table("heart_disease"))
        self.widget.send_report()

        w.disc_var_hints[DefaultKey] = Continuize.AsOrdinal
        w.disc_var_hints["chest pain"] = w.disc_var_hints["rest ECG"] = Continuize.Remove
        w.disc_var_hints["exerc ind ang"] = Continuize.FirstAsBase

        self.send_signal(w.Inputs.data, Table("heart_disease"))
        self.widget.send_report()

        w.cont_var_hints[DefaultKey] = Normalize.Center
        w.cont_var_hints["cholesterol"] = Normalize.Scale

        self.send_signal(w.Inputs.data, Table("heart_disease"))
        self.widget.send_report()

        w.continuize_class = True
        w.disc_var_hints[DefaultKey] = Continuize.AsOrdinal
        w.disc_var_hints["chest pain"] = w.disc_var_hints["rest ECG"] = Continuize.Remove
        w.disc_var_hints["exerc ind ang"] = Continuize.FirstAsBase

        w.cont_var_hints[DefaultKey] = Normalize.Center
        w.cont_var_hints["cholesterol"] = Normalize.Scale

        self.send_signal(w.Inputs.data, Table("heart_disease"))
        self.widget.send_report()

    def test_migrate_settings_to_v3(self):
        Normalize = OWContinuize.Normalize

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=0))
        self.assertEqual(widget.cont_var_hints[DefaultKey], Normalize.Leave)

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=1, zero_based=True))
        self.assertEqual(widget.cont_var_hints[DefaultKey], Normalize.Normalize01)

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=1, zero_based=False))
        self.assertEqual(widget.cont_var_hints[DefaultKey], Normalize.Normalize11)

        widget = self.create_widget(
            OWContinuize,
            stored_settings=dict(continuous_treatment=2))
        self.assertEqual(widget.cont_var_hints[DefaultKey], Normalize.Standardize)


class TestModelsAndViews(GuiTest):
    def test_contmodel(self):
        domain = Domain([ContinuousVariable(c) for c in "abc"],
                        ContinuousVariable("y"))
        model = ContDomainModel(ContinuousVariable)
        model.set_domain(domain)

        ind = model.index(0, 0)
        self.assertEqual(ind.data()[0], "a")
        self.assertEqual(ind.data(model.FilterRole)[0], "a")
        self.assertIsNone(ind.data(Qt.ToolTipRole))

        ind = model.index(1, 0)
        model.setData(ind, ("mega encoding", True), model.HintRole)
        self.assertEqual(ind.data(), ("b", "mega encoding", True))
        self.assertEqual(ind.data(model.HintRole), ("mega encoding", True))
        self.assertIn("b", ind.data(model.FilterRole))
        self.assertIn("mega encoding", ind.data(model.FilterRole))
        self.assertNotIn("bmega encoding", ind.data(model.FilterRole))
        self.assertIsNone(ind.data(Qt.ToolTipRole))

        ind = model.index(3, 0)  # separator
        self.assertIsNone(ind.data())
        self.assertIsNone(ind.data(model.HintRole))
        self.assertIsNone(ind.data(model.FilterRole))

    def test_defaultcontmodel(self):
        model = DefaultContModel()
        self.assertEqual(1, model.rowCount(QModelIndex()))
        self.assertEqual(1, model.columnCount(QModelIndex()))
        ind = model.index(0, 0)
        model.setMethod("mega encoding")
        self.assertEqual(ind.data(), "Preset: mega encoding")
        self.assertIsNotNone(ind.data(Qt.DecorationRole))
        self.assertIsNotNone(ind.data(Qt.ToolTipRole))


class TestListViewDelegate(unittest.TestCase):
    def test_displaytext(self):
        delegate = ListViewSearch.Delegate()
        self.assertEqual(delegate.displayText(("a", "foo", False), Mock()), "a")
        self.assertEqual(delegate.displayText(("a", "foo", True), Mock()), "a: foo")
        delegate.set_default_hints(True)
        self.assertEqual(delegate.displayText(("a", "foo", False), Mock()), "a: foo")
        delegate.set_default_hints(False)
        self.assertEqual(delegate.displayText(("a", "foo", False), Mock()), "a")

    @patch.object(SeparatedListDelegate, "initStyleOption")
    def test_bold(self, _):
        delegate = ListViewSearch.Delegate()
        option = Mock()
        index = Mock()
        index.data = lambda role: ("foo", True) if role == ContDomainModel.HintRole else None
        delegate.initStyleOption(option, index)
        option.font.setBold.assert_called_with(True)
        index.data = lambda role: ("foo", False) if role == ContDomainModel.HintRole else None
        delegate.initStyleOption(option, index)
        option.font.setBold.assert_called_with(False)
        index.data = lambda role: None if role == ContDomainModel.HintRole else None
        delegate.initStyleOption(option, index)
        option.font.setBold.assert_called_with(False)

if __name__ == "__main__":
    unittest.main()
