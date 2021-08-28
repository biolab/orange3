# Tests test protected methods
# pylint: disable=protected-access

import unittest
from typing import Optional, Union
from unittest.mock import Mock, patch

import numpy as np
from scipy import sparse as sp

from orangewidget.settings import ContextSetting

from Orange.data import (
    DiscreteVariable, ContinuousVariable, StringVariable, Domain, Table
)
from Orange.widgets.widget import OWWidget
from Orange.widgets.tests.base import WidgetTest

from Orange.widgets.data import owmelt


def data_without_commit(f=None, *, sparse=False):
    def wrapped(self):
        with patch("Orange.widgets.data.owmelt.OWMelt.commit"):
            data = self.data
            if sparse:
                data = Table.from_numpy(
                    data.domain, sp.csr_matrix(data.X), None, data.metas)
            self.send_signal(self.widget.Inputs.data, data)
            f(self)
    if f is None:
        return lambda g: data_without_commit(g, sparse=sparse)
    return wrapped


def names(variables):
    return [var.name for var in variables]


class TestOWMeltBase(WidgetTest):

    # Tests use this table:
    #
    #           attributes                       metas
    # -----------------------------------   -----------------
    # gender   age     pretzels   telezka   name     greeting
    # disc     cont    cont       disc      string   string
    # -------------------------------------------------------
    # 0        25      3                    ana      hi
    # 0        26      0          1         berta    hello
    # 0        27                 0         cilka
    # 1        28                                    hi
    # 1                2                    evgen    foo
    # -------------------------------------------------------

    def setUp(self):
        self.widget = self.create_widget(owmelt.OWMelt)
        n = np.nan

        attributes = [
            DiscreteVariable("gender", values=("f", "m")),
            ContinuousVariable("age"),
            ContinuousVariable("pretzels"),
            DiscreteVariable("telezka", values=("big", "small"))
        ]
        metas = [
            StringVariable("name"),
            StringVariable("greeting")]

        x = np.array([[0, 25, 3, n],
                      [0, 26, 0, 1],
                      [0, 27, n, 0],
                      [1, 28, n, n],
                      [1, n, 2, n]])
        m = np.array([["ana", "hi"],
                      ["berta", "hello"],
                      ["cilka", ""],
                      ["", "hi"],
                      ["evgen", "foo"]])

        self.data = Table.from_numpy(Domain(attributes, [], metas), x, None, m)
        self.data_no_metas = Table.from_numpy(Domain(attributes, []), x, None)
        self.data_only_meta_id = Table.from_numpy(
            Domain(attributes[:-1], [], metas), x[:, :-1], None, m)


class TestOWMeltFunctional(TestOWMeltBase):
    @data_without_commit
    def test_idvar_model(self):
        widget = self.widget

        telezka = self.data.domain.attributes[-1]
        name = self.data.domain.metas[0]

        self.send_signal(widget.Inputs.data, self.data)
        self.assertSequenceEqual(self.widget.idvar_model, [None, telezka, name])
        self.assertIsNone(widget.idvar)

        self.send_signal(widget.Inputs.data, self.data_no_metas)
        self.assertSequenceEqual(self.widget.idvar_model, [None, telezka])
        self.assertIs(widget.idvar, telezka)

        self.send_signal(widget.Inputs.data, self.data_only_meta_id)
        self.assertSequenceEqual(self.widget.idvar_model, [None, name])
        self.assertIs(widget.idvar, name)

        self.send_signal(widget.Inputs.data, Table("iris"))
        self.assertSequenceEqual(self.widget.idvar_model, [None])
        self.assertIsNone(widget.idvar)

    def test_context_and_no_data(self):
        widget = self.widget

        self.send_signal(widget.Inputs.data, self.data)
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(widget.idvar)

        widget.idvar = widget.idvar_model[2]

        self.send_signal(widget.Inputs.data, None)
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(widget.idvar)

        self.send_signal(widget.Inputs.data, self.data)
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertIs(widget.idvar, widget.idvar_model[2])

    def test_context_disregards_none(self):
        # By default, widget selects None in case of multiple candidates
        widget = self.create_widget(owmelt.OWMelt)
        self.send_signal(widget.Inputs.data, self.data)
        self.assertIsNone(widget.idvar)

        # Start with a new context, so we don't get a perfect match to the above
        widget = self.create_widget(owmelt.OWMelt)
        self.send_signal(widget.Inputs.data, self.data_no_metas)
        self.assertIsNotNone(widget.idvar)
        expected = widget.idvar

        self.send_signal(widget.Inputs.data, self.data)
        self.assertIs(widget.idvar, expected)

    def test_no_suitable_features(self):
        widget = self.widget
        heart = Table("heart_disease")
        self.send_signal(self.widget.Inputs.data, self.data)

        self.assertFalse(widget.Information.no_suitable_features.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))

        # Sending unsuitable data shows the warning, resets output
        self.send_signal(widget.Inputs.data, heart)
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertSequenceEqual(widget.idvar_model, [None])
        self.assertTrue(widget.Information.no_suitable_features.is_shown())
        self.assertIsNone(widget.idvar)

        # Suitable data clears it, gives output
        self.send_signal(widget.Inputs.data, self.data)
        self.assertFalse(widget.Information.no_suitable_features.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertNotEqual(list(widget.idvar_model), [None])
        self.assertIsNone(widget.idvar)

        # Sending unsuitable data again shows the information, resets output
        self.send_signal(widget.Inputs.data, heart)
        self.assertTrue(widget.Information.no_suitable_features.is_shown())
        self.assertIsNotNone(self.get_output(widget.Outputs.data))
        self.assertSequenceEqual(widget.idvar_model, [None])
        self.assertIsNone(widget.idvar)

        # Removing data resets information, but still no output
        self.send_signal(widget.Inputs.data, None)
        self.assertFalse(widget.Information.no_suitable_features.is_shown())
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertIsNone(widget.idvar)
        self.assertSequenceEqual(widget.idvar_model, [None])

    def test_invalidates(self):
        widget = self.widget
        mock_return = Table("heart_disease")
        widget._reshape_to_long = lambda *_: mock_return
        widget.Outputs.data.send = send = Mock()

        self.send_signal(self.widget.Inputs.data, self.data)
        send.assert_called_with(mock_return)
        send.reset_mock()

        widget.controls.only_numeric.click()
        send.assert_called_with(mock_return)
        send.reset_mock()

        widget.controls.exclude_zeros.click()
        send.assert_called_with(mock_return)
        send.reset_mock()

        widget.controls.idvar.activated.emit(1)
        send.assert_called_with(mock_return)
        send.reset_mock()

    def test_report(self):
        widget = self.widget

        self.send_signal(widget.Inputs.data, self.data)
        self.assertIsNone(widget.idvar)
        self.assertIsNotNone(widget._output_desc)
        widget.send_report()

        self.send_signal(widget.Inputs.data, self.data_no_metas)
        self.assertIsNotNone(widget.idvar)
        self.assertIsNotNone(widget._output_desc)
        widget.send_report()

        self.send_signal(widget.Inputs.data, None)
        self.assertIsNone(widget._output_desc)
        widget.send_report()


class TestOWMeltUnit(TestOWMeltBase):
    @data_without_commit
    def test_is_unique(self):
        domain = self.data.domain
        widget = self.widget

        self.assertTrue(widget._is_unique(domain["name"]))
        self.assertTrue(widget._is_unique(domain["telezka"]))
        self.assertFalse(widget._is_unique(domain["gender"]))
        self.assertFalse(widget._is_unique(domain["greeting"]))

    def test_nonnan_mask(self):
        for arr in ([1., 2, np.nan, 0], ["Ana", "Berta", "", "Dani"]):
            np.testing.assert_equal(
                self.widget._notnan_mask(np.array(arr)),
                [True, True, False, True])

    @data_without_commit
    def test_get_useful_vars(self):
        def assert_useful(expected):
            self.assertEqual(
                [var.name for
                 var, useful in zip(domain.attributes,
                                    widget._get_useful_vars())
                 if useful],
                expected)

        domain = self.data.domain
        widget = self.widget

        widget.idvar = domain["name"]
        widget.only_numeric = False
        assert_useful(["gender", "age", "pretzels", "telezka"])

        widget.idvar = domain["name"]
        widget.only_numeric = True
        assert_useful(["age", "pretzels"])

        widget.idvar = domain["telezka"]
        widget.only_numeric = False
        assert_useful(["gender", "age", "pretzels"])

        widget.idvar = domain["telezka"]
        widget.only_numeric = True
        assert_useful(["age", "pretzels"])

    @data_without_commit
    def test_get_item_names(self):
        self.assertEqual(
            self.widget._get_item_names(np.array([False, True, False, True])),
            ("age", "telezka")
        )

    @data_without_commit
    def test_prepare_domain_names(self):
        domain = self.data.domain
        widget = self.widget

        widget.only_numeric = True

        widget.idvar = domain["name"]
        widget.item_var_name = "the item"
        widget.value_var_name = "the value"
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        idvar, itemvar = outdomain.attributes
        self.assertEqual(idvar.name, "name")
        self.assertEqual(itemvar.name, "the item")
        self.assertEqual(outdomain.class_var.name, "the value")

        widget.idvar = domain["telezka"]
        widget.item_var_name = ""
        widget.value_var_name = ""
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        idvar, itemvar = outdomain.attributes
        self.assertEqual(idvar.name, "telezka")
        self.assertEqual(itemvar.name, owmelt.DEFAULT_ITEM_NAME)
        self.assertEqual(outdomain.class_var.name, owmelt.DEFAULT_VALUE_NAME)

    @data_without_commit
    def test_prepare_domain_renames(self):
        # Renaming is pretty generic, so we basically test that the widget
        # calls it. We tests two scenarios, not all God-knows-how-many
        widget = self.widget

        # The user proposes a name that matches idvar's name
        widget.idvar = None
        widget.item_var_name = owmelt.DEFAULT_NAME_FOR_ROW
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        _, itemvar = outdomain.attributes
        self.assertNotEqual(itemvar.name, owmelt.DEFAULT_NAME_FOR_ROW)
        self.assertTrue(
            itemvar.name.startswith(owmelt.DEFAULT_NAME_FOR_ROW))

        # Idvar's name is the same as the default for value
        svar = DiscreteVariable(owmelt.DEFAULT_VALUE_NAME, ("a", "b"))
        sdata = Table.from_numpy(Domain([svar], []), np.arange(5).reshape(5, 1))
        self.send_signal(widget.Inputs.data, sdata)
        widget.idvar = svar
        widget.item_value_var_name = ""
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        value_var_name = outdomain.class_var.name
        self.assertNotEqual(value_var_name, owmelt.DEFAULT_VALUE_NAME)
        self.assertTrue(
            value_var_name.startswith(owmelt.DEFAULT_VALUE_NAME))

    @data_without_commit
    def test_prepare_domain_values(self):
        domain = self.data.domain
        widget = self.widget

        widget.only_numeric = True

        widget.idvar = domain["name"]
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], ["Ana", "Berta", "Dani"])
        idvar, itemvar = outdomain.attributes
        self.assertEqual(idvar.values, ("Ana", "Berta", "Dani"))
        self.assertEqual(itemvar.values, ("age", "pretzels"))
        self.assertIsInstance(outdomain.class_var, ContinuousVariable)

        widget.idvar = domain["telezka"]
        outdomain = self.widget._prepare_domain(
            ["age", "pretzels"], None)
        idvar, itemvar = outdomain.attributes
        self.assertIs(idvar, widget.idvar)
        self.assertEqual(itemvar.values, ("age", "pretzels"))
        self.assertIsInstance(outdomain.class_var, ContinuousVariable)

    @data_without_commit
    def test_reshape_dense_by_meta(self):
        domain = self.data.domain
        widget = self.widget
        widget.idvar = domain["name"]

        widget.only_numeric = True
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 25], [0, 1, 3],
             [1, 0, 26],
             [2, 0, 27],
             [3, 1, 2]]
        )

        widget.only_numeric = True
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 25], [0, 1, 3],
             [1, 0, 26], [1, 1, 0],
             [2, 0, 27],
             [3, 1, 2]]
        )

        widget.only_numeric = False
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 0], [0, 1, 25], [0, 2, 3],
             [1, 0, 0], [1, 1, 26], [1, 3, 1],
             [2, 0, 0], [2, 1, 27], [2, 3, 0],
             [3, 0, 1], [3, 2, 2]]
        )

        widget.only_numeric = False
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 0], [0, 1, 25], [0, 2, 3],
             [1, 0, 0], [1, 1, 26], [1, 2, 0], [1, 3, 1],
             [2, 0, 0], [2, 1, 27], [2, 3, 0],
             [3, 0, 1], [3, 2, 2]]
        )

    @data_without_commit
    def test_reshape_dense_by_attr(self):
        domain = self.data.domain
        widget = self.widget
        widget.idvar = domain["telezka"]

        widget.only_numeric = True
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 26],
             [0, 0, 27]]
        )

        widget.only_numeric = True
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 26], [1, 1, 0],
             [0, 0, 27]]
        )

        widget.only_numeric = False
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 0], [1, 1, 26],
             [0, 0, 0], [0, 1, 27]]
        )

        widget.only_numeric = False
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[1, 0, 0], [1, 1, 26], [1, 2, 0],
             [0, 0, 0], [0, 1, 27]]
        )

    @data_without_commit
    def test_reshape_dense_by_row_number(self):
        widget = self.widget
        widget.idvar = None

        widget.exclude_zeros = True
        widget.only_numeric = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 25], [0, 1, 3],
             [1, 0, 26],
             [2, 0, 27],
             [3, 0, 28],
             [4, 1, 2]]
        )

        widget.only_numeric = True
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 25], [0, 1, 3],
             [1, 0, 26], [1, 1, 0],
             [2, 0, 27],
             [3, 0, 28],
             [4, 1, 2]]
        )

        widget.only_numeric = False
        widget.exclude_zeros = True
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 0], [0, 1, 25], [0, 2, 3],
             [1, 0, 0], [1, 1, 26], [1, 3, 1],
             [2, 0, 0], [2, 1, 27], [2, 3, 0],
             [3, 0, 1], [3, 1, 28],
             [4, 0, 1], [4, 2, 2]]
        )

        widget.only_numeric = False
        widget.exclude_zeros = False
        out = widget._reshape_to_long()
        np.testing.assert_equal(
            np.hstack((out.X, np.atleast_2d(out.Y).T)),
            [[0, 0, 0], [0, 1, 25], [0, 2, 3],
             [1, 0, 0], [1, 1, 26], [1, 2, 0], [1, 3, 1],
             [2, 0, 0], [2, 1, 27], [2, 3, 0],
             [3, 0, 1], [3, 1, 28],
             [4, 0, 1], [4, 2, 2]]
        )

    @data_without_commit(sparse=True)
    def test_reshape_sparse_by_meta(self):
        domain = self.data.domain
        widget = self.widget
        widget.idvar = domain["name"]
        assert sp.issparse(widget.data.X)

        widget.only_numeric = True
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[0, 0, 25], [0, 1, 3],
                 [1, 0, 26],
                 [2, 0, 27],
                 [3, 1, 2]]
            )

        widget.only_numeric = False
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[0, 1, 25], [0, 2, 3],
                 [1, 1, 26], [1, 3, 1],
                 [2, 1, 27],
                 [3, 0, 1], [3, 2, 2]]
            )

    @data_without_commit(sparse=True)
    def test_reshape_sparse_by_attr(self):
        domain = self.data.domain
        widget = self.widget
        widget.idvar = domain["telezka"]

        widget.only_numeric = True
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[1, 0, 26],
                 [0, 0, 27]]
            )

        widget.only_numeric = False
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[1, 1, 26],
                 [0, 1, 27]]
            )

    @data_without_commit(sparse=True)
    def test_reshape_sparse_by_row_number(self):
        widget = self.widget
        widget.idvar = None

        widget.only_numeric = True
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[0, 0, 25], [0, 1, 3],
                 [1, 0, 26],
                 [2, 0, 27],
                 [3, 0, 28],
                 [4, 1, 2]]
            )

        widget.only_numeric = False
        for widget.exclude_zeros in (True, False):
            out = widget._reshape_to_long()
            np.testing.assert_equal(
                np.hstack((out.X, np.atleast_2d(out.Y).T)),
                [[0, 1, 25], [0, 2, 3],
                 [1, 1, 26], [1, 3, 1],
                 [2, 1, 27],
                 [3, 0, 1], [3, 1, 28],
                 [4, 0, 1], [4, 2, 2]]
            )


class TestContextHandler(WidgetTest):
    # Context handler is tested within the real-world context
    # Here are the only specific, unit tests, which (at the time of writing)
    # can't occur in practice, but may in the future

    def setUp(self):
        class MockWidget(OWWidget):
            settingsHandler = owmelt.MeltContextHandler()

            idvar: Union[DiscreteVariable, StringVariable, None] \
                = ContextSetting(None)
            not_idvar: Optional[DiscreteVariable] = ContextSetting(None)

        self.widget = self.create_widget(MockWidget)

    base = owmelt.MeltContextHandler.__bases__[0]

    @patch.object(base, "decode_setting")
    def test_decode_calls_super(self, super_decode):
        handler = self.widget.settingsHandler

        handler.decode_setting(handler.known_settings["idvar"], None, [])
        super_decode.assert_not_called()
        handler.decode_setting(handler.known_settings["not_idvar"], None, [])
        super_decode.assert_called()

    @patch.object(base, "encode_setting")
    def test_encode_calls_super(self, super_encode):
        handler = self.widget.settingsHandler
        context = handler.new_context([])

        handler.encode_setting(context, handler.known_settings["idvar"], None)
        super_encode.assert_not_called()
        handler.encode_setting(context, handler.known_settings["not_idvar"], None)
        super_encode.assert_called()


# Avoid reports about missing tests
del WidgetTest

if __name__ == "__main__":
    unittest.main()
