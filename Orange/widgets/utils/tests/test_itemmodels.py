# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import patch

import numpy as np

from AnyQt.QtCore import Qt, QModelIndex

from Orange.data import \
    Domain, \
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable
from Orange.widgets.utils.itemmodels import \
    AbstractSortTableModel, PyTableModel,\
    PyListModel, VariableListModel, DomainModel,\
    _as_contiguous_range
from Orange.widgets.gui import TableVariable


class TestUtils(unittest.TestCase):
    def test_as_contiguous_range(self):
        self.assertEqual(_as_contiguous_range(slice(1, 8), 20), (1, 8, 1))
        self.assertEqual(_as_contiguous_range(slice(1, 8), 6), (1, 6, 1))
        self.assertEqual(_as_contiguous_range(slice(8, 1, -1), 6), (2, 6, 1))
        self.assertEqual(_as_contiguous_range(slice(8), 6), (0, 6, 1))
        self.assertEqual(_as_contiguous_range(slice(8, None, -1), 6), (0, 6, 1))
        self.assertEqual(_as_contiguous_range(slice(7, None, -1), 9), (0, 8, 1))
        self.assertEqual(_as_contiguous_range(slice(None, None, -1), 9),
                         (0, 9, 1))


class TestPyTableModel(unittest.TestCase):
    def setUp(self):
        self.model = PyTableModel([[1, 4],
                                   [2, 3]])

    def test_init(self):
        self.model = PyTableModel()
        self.assertEqual(self.model.rowCount(), 0)

    def test_rowCount(self):
        self.assertEqual(self.model.rowCount(), 2)
        self.assertEqual(len(self.model), 2)

    def test_columnCount(self):
        self.assertEqual(self.model.columnCount(), 2)

    def test_data(self):
        mi = self.model.index(0, 0)
        self.assertEqual(self.model.data(mi), '1')
        self.assertEqual(self.model.data(mi, Qt.EditRole), 1)

    def test_editable(self):
        editable_model = PyTableModel([[0]], editable=True)
        self.assertFalse(int(self.model.flags(self.model.index(0, 0)) & Qt.ItemIsEditable))
        self.assertTrue(int(editable_model.flags(editable_model.index(0, 0)) & Qt.ItemIsEditable))

    def test_sort(self):
        self.model.sort(1)
        self.assertEqual(self.model.index(0, 0).data(Qt.EditRole), 2)

    def test_setHeaderLabels(self):
        self.model.setHorizontalHeaderLabels(['Col 1', 'Col 2'])
        self.assertEqual(self.model.headerData(1, Qt.Horizontal), 'Col 2')
        self.assertEqual(self.model.headerData(1, Qt.Vertical), 2)

    def test_removeRows(self):
        self.model.removeRows(0, 1)
        self.assertEqual(len(self.model), 1)
        self.assertEqual(self.model[0][1], 3)

    def test_removeColumns(self):
        self.model.removeColumns(0, 1)
        self.assertEqual(self.model.columnCount(), 1)
        self.assertEqual(self.model[1][0], 3)

    def test_insertRows(self):
        self.model.insertRows(0, 1)
        self.assertEqual(self.model[1][0], 1)

    def test_insertColumns(self):
        self.model.insertColumns(0, 1)
        self.assertEqual(self.model[0], ['', 1, 4])

    def test_wrap(self):
        self.model.wrap([[0]])
        self.assertEqual(self.model.rowCount(), 1)
        self.assertEqual(self.model.columnCount(), 1)

    def test_clear(self):
        self.model.clear()
        self.assertEqual(self.model.rowCount(), 0)

    def test_append(self):
        self.model.append([5, 6])
        self.assertEqual(self.model[2][1], 6)
        self.assertEqual(self.model.rowCount(), 3)

    def test_extend(self):
        self.model.extend([[5, 6]])
        self.assertEqual(self.model[2][1], 6)
        self.assertEqual(self.model.rowCount(), 3)

    def test_insert(self):
        self.model.insert(0, [5, 6])
        self.assertEqual(self.model[0][1], 6)
        self.assertEqual(self.model.rowCount(), 3)

    def test_remove(self):
        self.model.remove([2, 3])
        self.assertEqual(self.model.rowCount(), 1)

    def test_other_roles(self):
        self.model.append([2, 3])
        self.model.setData(self.model.index(2, 0),
                           Qt.AlignCenter,
                           Qt.TextAlignmentRole)
        del self.model[1]
        self.assertTrue(Qt.AlignCenter &
                        self.model.data(self.model.index(1, 0),
                                        Qt.TextAlignmentRole))


class TestVariableListModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.disc = DiscreteVariable("gender", values=("M", "F"))
        cls.cont = ContinuousVariable("age")
        cls.string = StringVariable("name")
        cls.time = TimeVariable("birth")
        cls.model = VariableListModel([
            cls.cont, None, "Foo", cls.disc, cls.string, cls.time])

    def test_placeholder(self):
        model = self.model
        self.assertEqual(model.data(model.index(1)), "None")
        model.placeholder = "Bar"
        self.assertEqual(model.data(model.index(1)), "Bar")
        model.placeholder = "None"

    def test_displayrole(self):
        data, index = self.model.data, self.model.index
        self.assertEqual(data(index(0)), "age")
        self.assertEqual(data(index(1)), "None")
        self.assertEqual(data(index(2)), "Foo")
        self.assertEqual(data(index(3)), "gender")
        self.assertEqual(data(index(4)), "name")
        self.assertEqual(data(index(5)), "birth")

    def test_tooltip(self):
        def get_tooltip(i):
            return self.model.data(self.model.index(i), Qt.ToolTipRole)

        text = get_tooltip(0)
        self.assertIn("age", text)
        self.assertIn("Numeric", text)

        self.assertIsNone(get_tooltip(1))
        self.assertIsNone(get_tooltip(2))

        text = get_tooltip(3)
        self.assertIn("gender", text)
        self.assertIn("M", text)
        self.assertIn("F", text)
        self.assertIn("2", text)
        self.assertIn("Categorical", text)

        text = get_tooltip(4)
        self.assertIn("name", text)
        self.assertIn("Text", text)

        text = get_tooltip(5)
        self.assertIn("birth", text)
        self.assertIn("Time", text)

        self.cont.attributes = {"foo": "bar"}
        text = get_tooltip(0)
        self.assertIn("foo", text)
        self.assertIn("bar", text)

    def test_table_variable(self):
        self.assertEqual(
            [self.model.data(self.model.index(i), TableVariable)
             for i in range(self.model.rowCount())],
            [self.cont, None, None, self.disc, self.string, self.time])

    def test_other_roles(self):
        with patch.object(PyListModel, "data") as data:
            index = self.model.index(0)
            _ = self.model.data(index, Qt.BackgroundRole)
            self.assertEqual(data.call_args[0][1:], (index, Qt.BackgroundRole))

    def test_invalid_index(self):
        self.assertIsNone(self.model.data(self.model.index(0).parent()))


class TestDomainModel(unittest.TestCase):
    def test_init_with_single_section(self):
        model = DomainModel(order=DomainModel.CLASSES)
        self.assertEqual(model.order, (DomainModel.CLASSES, ))

    def test_separators(self):
        attrs = [ContinuousVariable(n) for n in "abg"]
        classes = [ContinuousVariable(n) for n in "deh"]
        metas = [ContinuousVariable(n) for n in "ijf"]

        model = DomainModel()
        sep = [model.Separator]
        model.set_domain(Domain(attrs, classes, metas))
        self.assertEqual(list(model), classes + sep + metas + sep + attrs)

        model = DomainModel()
        model.set_domain(Domain(attrs, [], metas))
        self.assertEqual(list(model), metas + sep + attrs)

        model = DomainModel()
        model.set_domain(Domain([], [], metas))
        self.assertEqual(list(model), metas)

        model = DomainModel(placeholder="foo")
        model.set_domain(Domain([], [], metas))
        self.assertEqual(list(model), [None] + sep + metas)

        model = DomainModel(placeholder="foo")
        model.set_domain(Domain(attrs, [], metas))
        self.assertEqual(list(model), [None] + sep + metas + sep + attrs)

    def test_placeholder_placement(self):
        model = DomainModel(placeholder="foo")
        sep = model.Separator
        self.assertEqual(model.order, (None, sep) + model.SEPARATED)

        model = DomainModel(order=("bar", ), placeholder="foo")
        self.assertEqual(model.order, (None, "bar"))

        model = DomainModel(order=("bar", None, "baz"), placeholder="foo")
        self.assertEqual(model.order, ("bar", None, "baz"))

        model = DomainModel(order=("bar", sep, "baz"),
                            placeholder="foo")
        self.assertEqual(model.order, (None, sep, "bar", sep, "baz"))

    def test_subparts(self):
        attrs = [ContinuousVariable(n) for n in "abg"]
        classes = [ContinuousVariable(n) for n in "deh"]
        metas = [ContinuousVariable(n) for n in "ijf"]

        m = DomainModel
        sep = m.Separator
        model = DomainModel(
            order=(m.ATTRIBUTES | m.METAS, sep, m.CLASSES))
        model.set_domain(Domain(attrs, classes, metas))
        self.assertEqual(list(model), attrs + metas + [sep] + classes)

        m = DomainModel
        sep = m.Separator
        model = DomainModel(
            order=(m.ATTRIBUTES | m.METAS, sep, m.CLASSES),
            alphabetical=True)
        model.set_domain(Domain(attrs, classes, metas))
        self.assertEqual(list(model),
                         sorted(attrs + metas, key=lambda x: x.name) +
                         [sep] +
                         sorted(classes, key=lambda x: x.name))

    def test_filtering(self):
        cont = [ContinuousVariable(n) for n in "abc"]
        disc = [DiscreteVariable(n) for n in "def"]
        attrs = cont + disc

        model = DomainModel(valid_types=(ContinuousVariable, ))
        model.set_domain(Domain(attrs))
        self.assertEqual(list(model), cont)

        model = DomainModel(valid_types=(DiscreteVariable, ))
        model.set_domain(Domain(attrs))
        self.assertEqual(list(model), disc)

        disc[0].attributes["hidden"] = True
        model.set_domain(Domain(attrs))
        self.assertEqual(list(model), disc[1:])

        model = DomainModel(valid_types=(DiscreteVariable, ),
                            skip_hidden_vars=False)
        model.set_domain(Domain(attrs))
        self.assertEqual(list(model), disc)

    def test_no_separators(self):
        """
        GH-2697
        """
        attrs = [ContinuousVariable(n) for n in "abg"]
        classes = [ContinuousVariable(n) for n in "deh"]
        metas = [ContinuousVariable(n) for n in "ijf"]

        model = DomainModel(order=DomainModel.SEPARATED, separators=False)
        model.set_domain(Domain(attrs, classes, metas))
        self.assertEqual(list(model), classes + metas + attrs)

        model = DomainModel(order=DomainModel.SEPARATED, separators=True)
        model.set_domain(Domain(attrs, classes, metas))
        self.assertEqual(
            list(model),
            classes + [PyListModel.Separator] + metas + [PyListModel.Separator] + attrs)

    def test_read_only(self):
        model = DomainModel()
        domain = Domain([ContinuousVariable(x) for x in "abc"])
        model.set_domain(domain)
        index = model.index(0, 0)

        self.assertRaises(TypeError, model.append, 42)
        self.assertRaises(TypeError, model.extend, [42])
        self.assertRaises(TypeError, model.insert, 0, 42)
        self.assertRaises(TypeError, model.remove, 0)
        self.assertRaises(TypeError, model.pop)
        self.assertRaises(TypeError, model.clear)
        self.assertRaises(TypeError, model.reverse)
        self.assertRaises(TypeError, model.sort)
        with self.assertRaises(TypeError):
            model[0] = 1
        with self.assertRaises(TypeError):
            del model[0]

        self.assertFalse(model.setData(index, domain[0], Qt.EditRole))
        self.assertTrue(model.setData(index, "foo", Qt.ToolTipRole))

        self.assertFalse(model.setItemData(index, {Qt.EditRole: domain[0],
                                                   Qt.ToolTipRole: "foo"}))
        self.assertTrue(model.setItemData(index, {Qt.ToolTipRole: "foo"}))

        self.assertFalse(model.insertRows(0, 1))
        self.assertSequenceEqual(model, domain)
        self.assertFalse(model.removeRows(0, 1))
        self.assertSequenceEqual(model, domain)


if __name__ == "__main__":
    unittest.main()
