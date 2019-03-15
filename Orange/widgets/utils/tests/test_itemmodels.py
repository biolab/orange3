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
    _argsort, _as_contiguous_range
from Orange.widgets.gui import TableVariable


class TestArgsort(unittest.TestCase):
    def test_argsort(self):
        self.assertEqual(_argsort("dacb"), [1, 3, 2, 0])
        self.assertEqual(_argsort("dacb", reverse=True), [0, 2, 3, 1])
        self.assertEqual(_argsort([3, -1, 0, 2], key=abs), [2, 1, 3, 0])
        self.assertEqual(
            _argsort([3, -1, 0, 2], key=abs, reverse=True), [0, 3, 1, 2])
        self.assertEqual(
            _argsort([3, -1, 0, 2],
                     cmp=lambda x, y: (abs(x) > abs(y)) - (abs(x) < abs(y))),
            [2, 1, 3, 0])
        self.assertEqual(
            _argsort([3, -1, 0, 2],
                     cmp=lambda x, y: (abs(x) > abs(y)) - (abs(x) < abs(y)),
                     reverse=True),
            [0, 3, 1, 2])

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


class TestAbstractSortTableModel(unittest.TestCase):
    def test_sorting(self):
        assert issubclass(PyTableModel, AbstractSortTableModel)
        model = PyTableModel([[1, 4],
                              [2, 2],
                              [3, 3]])
        model.sort(1, Qt.AscendingOrder)
        # mapToSourceRows
        self.assertSequenceEqual(model.mapToSourceRows(...).tolist(), [1, 2, 0])
        self.assertEqual(model.mapToSourceRows(1).tolist(), 2)
        self.assertSequenceEqual(model.mapToSourceRows([1, 2]).tolist(), [2, 0])
        self.assertSequenceEqual(model.mapToSourceRows([]), [])
        self.assertSequenceEqual(model.mapToSourceRows(np.array([], dtype=int)).tolist(), [])
        self.assertRaises(IndexError, model.mapToSourceRows, np.r_[0.])

        # mapFromSourceRows
        self.assertSequenceEqual(model.mapFromSourceRows(...).tolist(), [2, 0, 1])
        self.assertEqual(model.mapFromSourceRows(1).tolist(), 0)
        self.assertSequenceEqual(model.mapFromSourceRows([1, 2]).tolist(), [0, 1])
        self.assertSequenceEqual(model.mapFromSourceRows([]), [])
        self.assertSequenceEqual(model.mapFromSourceRows(np.array([], dtype=int)).tolist(), [])
        self.assertRaises(IndexError, model.mapFromSourceRows, np.r_[0.])

        model.sort(1, Qt.DescendingOrder)
        self.assertSequenceEqual(model.mapToSourceRows(...).tolist(), [0, 2, 1])
        self.assertSequenceEqual(model.mapFromSourceRows(...).tolist(), [0, 2, 1])


# Tests test _is_index_valid and access model._other_data. The latter tests
# implementation, but it would be cumbersome and less readable to test function
# pylint: disable=protected-access
class TestPyListModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PyListModel([1, 2, 3, 4])

    def test_wrap(self):
        model = PyListModel()
        s = [1, 2]
        model.wrap(s)
        self.assertSequenceEqual(model, [1, 2])
        model.append(3)
        self.assertEqual(s, [1, 2, 3])
        self.assertEqual(len(model._other_data), 3)

        s.append(5)
        self.assertRaises(RuntimeError, model._is_index_valid, 0)

    def test_is_index_valid(self):
        self.assertTrue(self.model._is_index_valid(0))
        self.assertTrue(self.model._is_index_valid(2))
        self.assertTrue(self.model._is_index_valid(-1))
        self.assertTrue(self.model._is_index_valid(-4))

        self.assertFalse(self.model._is_index_valid(-5))
        self.assertFalse(self.model._is_index_valid(5))

    def test_index(self):
        index = self.model.index(2, 0)
        self.assertTrue(index.isValid())
        self.assertEqual(index.row(), 2)
        self.assertEqual(index.column(), 0)

        self.assertFalse(self.model.index(5, 0).isValid())
        self.assertFalse(self.model.index(-5, 0).isValid())
        self.assertFalse(self.model.index(0, 1).isValid())

    def test_headerData(self):
        self.assertEqual(self.model.headerData(3, Qt.Vertical), "3")

    def test_rowCount(self):
        self.assertEqual(self.model.rowCount(), len(self.model))
        self.assertEqual(self.model.rowCount(self.model.index(2, 0)), 0)

    def test_columnCount(self):
        self.assertEqual(self.model.columnCount(), 1)
        self.assertEqual(self.model.columnCount(self.model.index(2, 0)), 0)

    def test_indexOf(self):
        self.assertEqual(self.model.indexOf(3), 2)

    def test_data(self):
        mi = self.model.index(2)
        self.assertEqual(self.model.data(mi), 3)
        self.assertEqual(self.model.data(mi, Qt.EditRole), 3)

        self.assertIsNone(self.model.data(self.model.index(5)))

    def test_itemData(self):
        model = PyListModel([1, 2, 3, 4])
        mi = model.index(2)
        model.setItemData(mi, {Qt.ToolTipRole: "foo"})
        self.assertEqual(model.itemData(mi)[Qt.ToolTipRole], "foo")

        self.assertEqual(model.itemData(model.index(5)), {})

    def test_mimeData(self):
        model = PyListModel([1, 2])
        model._other_data[:] = [{Qt.UserRole: "a"}, {}]
        mime = model.mimeData([model.index(0), model.index(1)])
        self.assertTrue(mime.hasFormat(PyListModel.MIME_TYPE))

    def test_dropMimeData(self):
        model = PyListModel([1, 2])
        model.setData(model.index(0), "a", Qt.UserRole)
        mime = model.mimeData([model.index(0)])
        self.assertTrue(
            model.dropMimeData(mime, Qt.CopyAction, 2, -1, model.index(-1, -1))
        )
        self.assertEqual(len(model), 3)
        self.assertEqual(
            model.itemData(model.index(2)),
            {Qt.DisplayRole: 1, Qt.EditRole: 1, Qt.UserRole: "a"}
        )

    def test_parent(self):
        self.assertFalse(self.model.parent(self.model.index(2)).isValid())

    def test_set_data(self):
        model = PyListModel([1, 2, 3, 4])
        model.setData(model.index(0), None, Qt.EditRole)
        self.assertIs(model.data(model.index(0), Qt.EditRole), None)

        model.setData(model.index(1), "This is two", Qt.ToolTipRole)
        self.assertEqual(model.data(model.index(1), Qt.ToolTipRole),
                         "This is two",)

        self.assertFalse(model.setData(model.index(5), "foo"))

    def test_setitem(self):
        model = PyListModel([1, 2, 3, 4])
        model[1] = 42
        self.assertSequenceEqual(model, [1, 42, 3, 4])
        model[-1] = 42
        self.assertSequenceEqual(model, [1, 42, 3, 42])

        with self.assertRaises(IndexError):
            model[4]  # pylint: disable=pointless-statement

        with self.assertRaises(IndexError):
            model[-5]  # pylint: disable=pointless-statement

        model = PyListModel([1, 2, 3, 4])
        model[0:0] = [-1, 0]
        self.assertSequenceEqual(model, [-1, 0, 1, 2, 3, 4])

        model = PyListModel([1, 2, 3, 4])
        model[len(model):len(model)] = [5, 6]
        self.assertSequenceEqual(model, [1, 2, 3, 4, 5, 6])

        model = PyListModel([1, 2, 3, 4])
        model[0:2] = (-1, -2)
        self.assertSequenceEqual(model, [-1, -2, 3, 4])

        model = PyListModel([1, 2, 3, 4])
        model[-2:] = [-3, -4]
        self.assertSequenceEqual(model, [1, 2, -3, -4])

        model = PyListModel([1, 2, 3, 4])
        with self.assertRaises(IndexError):
            # non unit strides currently not supported
            model[0:-1:2] = [3, 3]

    def test_getitem(self):
        self.assertEqual(self.model[0], 1)
        self.assertEqual(self.model[2], 3)
        self.assertEqual(self.model[-1], 4)
        self.assertEqual(self.model[-4], 1)

        with self.assertRaises(IndexError):
            self.model[4]    # pylint: disable=pointless-statement

        with self.assertRaises(IndexError):
            self.model[-5]  # pylint: disable=pointless-statement

    def test_delitem(self):
        model = PyListModel([1, 2, 3, 4])
        model._other_data = list("abcd")
        del model[1]
        self.assertSequenceEqual(model, [1, 3, 4])
        self.assertSequenceEqual(model._other_data, "acd")

        model = PyListModel([1, 2, 3, 4])
        model._other_data = list("abcd")
        del model[1:3]

        self.assertSequenceEqual(model, [1, 4])
        self.assertSequenceEqual(model._other_data, "ad")

        model = PyListModel([1, 2, 3, 4])
        model._other_data = list("abcd")
        del model[:]
        self.assertSequenceEqual(model, [])
        self.assertEqual(len(model._other_data), 0)

        model = PyListModel([1, 2, 3, 4])
        with self.assertRaises(IndexError):
            # non unit strides currently not supported
            del model[0:-1:2]
        self.assertEqual(len(model), len(model._other_data))

    def test_add(self):
        model2 = self.model + [5, 6]
        self.assertSequenceEqual(model2, [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(model2), len(model2._other_data))

    def test_iadd(self):
        model = PyListModel([1, 2, 3, 4])
        model += [5, 6]
        self.assertSequenceEqual(model, [1, 2, 3, 4, 5, 6])
        self.assertEqual(len(model), len(model._other_data))

    def test_list_specials(self):
        # Essentially tested in other tests, but let's do it explicitly, too
        # __len__
        self.assertEqual(len(self.model), 4)

        # __contains__
        self.assertTrue(2 in self.model)
        self.assertFalse(5 in self.model)

        # __iter__
        self.assertSequenceEqual(self.model, [1, 2, 3, 4])

        # __bool__
        self.assertTrue(bool(self.model))
        self.assertFalse(bool(PyListModel()))

    def test_insert_delete_rows(self):
        model = PyListModel([1, 2, 3, 4])
        success = model.insertRows(0, 3)

        self.assertIs(success, True)
        self.assertSequenceEqual(model, [None, None, None, 1, 2, 3, 4])

        success = model.removeRows(3, 4)
        self.assertIs(success, True)
        self.assertSequenceEqual(model, [None, None, None])

        self.assertFalse(model.insertRows(0, 1, model.index(0)))
        self.assertFalse(model.removeRows(0, 1, model.index(0)))

    def test_extend(self):
        model = PyListModel([])
        model.extend([1, 2, 3, 4])
        self.assertSequenceEqual(model, [1, 2, 3, 4])

        model.extend([5, 6])
        self.assertSequenceEqual(model, [1, 2, 3, 4, 5, 6])

        self.assertEqual(len(model), len(model._other_data))

    def test_append(self):
        model = PyListModel([])
        model.append(1)
        self.assertSequenceEqual(model, [1])

        model.append(2)
        self.assertSequenceEqual(model, [1, 2])

        self.assertEqual(len(model), len(model._other_data))

    def test_insert(self):
        model = PyListModel()
        model.insert(0, 1)
        self.assertSequenceEqual(model, [1])
        self.assertEqual(len(model._other_data), 1)
        model._other_data = ["a"]

        model.insert(0, 2)
        self.assertSequenceEqual(model, [2, 1])
        self.assertEqual(model._other_data[1], "a")
        self.assertNotEqual(model._other_data[0], "a")
        model._other_data[0] = "b"

        model.insert(1, 3)
        self.assertSequenceEqual(model, [2, 3, 1])
        self.assertEqual(model._other_data[0], "b")
        self.assertEqual(model._other_data[2], "a")
        self.assertNotEqual(model._other_data[1], "b")
        self.assertNotEqual(model._other_data[1], "a")
        model._other_data[1] = "c"

        model.insert(3, 4)
        self.assertSequenceEqual(model, [2, 3, 1, 4])
        self.assertSequenceEqual(model._other_data[:3], ["b", "c", "a"])
        model._other_data[3] = "d"

        model.insert(-1, 5)
        self.assertSequenceEqual(model, [2, 3, 1, 5, 4])
        self.assertSequenceEqual(model._other_data[:3], ["b", "c", "a"])
        self.assertEqual(model._other_data[4], "d")
        self.assertEqual(len(model), len(model._other_data))

    def test_remove(self):
        model = PyListModel([1, 2, 3, 2, 4])
        model._other_data = list("abcde")
        model.remove(2)
        self.assertSequenceEqual(model, [1, 3, 2, 4])
        self.assertSequenceEqual(model._other_data, "acde")

    def test_pop(self):
        model = PyListModel([1, 2, 3, 2, 4])
        model._other_data = list("abcde")
        model.pop(1)
        self.assertSequenceEqual(model, [1, 3, 2, 4])
        self.assertSequenceEqual(model._other_data, "acde")

    def test_clear(self):
        model = PyListModel([1, 2, 3, 2, 4])
        model.clear()
        self.assertSequenceEqual(model, [])
        self.assertEqual(len(model), len(model._other_data))

        model.clear()
        self.assertSequenceEqual(model, [])
        self.assertEqual(len(model), len(model._other_data))

    def test_reverse(self):
        model = PyListModel([1, 2, 3, 4])
        model._other_data = list("abcd")
        model.reverse()
        self.assertSequenceEqual(model, [4, 3, 2, 1])
        self.assertSequenceEqual(model._other_data, "dcba")

    def test_sort(self):
        model = PyListModel([3, 1, 4, 2])
        model._other_data = list("abcd")
        model.sort()
        self.assertSequenceEqual(model, [1, 2, 3, 4])
        self.assertSequenceEqual(model._other_data, "bdac")

    def test_moveRows(self):
        model = PyListModel([1, 2, 3, 4])
        for i in range(model.rowCount()):
            model.setData(model.index(i), str(i + 1), Qt.UserRole)

        def modeldata(role):
            return [model.index(i).data(role)
                    for i in range(model.rowCount())]

        def userdata():
            return modeldata(Qt.UserRole)

        def editdata():
            return modeldata(Qt.EditRole)

        r = model.moveRows(QModelIndex(), 1, 1, QModelIndex(), 0)
        self.assertIs(r, True)
        self.assertSequenceEqual(editdata(), [2, 1, 3, 4])
        self.assertSequenceEqual(userdata(), ["2", "1", "3", "4"])
        r = model.moveRows(QModelIndex(), 1, 2, QModelIndex(), 4)
        self.assertIs(r, True)
        self.assertSequenceEqual(editdata(), [2, 4, 1, 3])
        self.assertSequenceEqual(userdata(), ["2", "4", "1", "3"])
        r = model.moveRows(QModelIndex(), 3, 1, QModelIndex(), 0)
        self.assertIs(r, True)
        self.assertSequenceEqual(editdata(), [3, 2, 4, 1])
        self.assertSequenceEqual(userdata(), ["3", "2", "4", "1"])
        r = model.moveRows(QModelIndex(), 2, 1, QModelIndex(), 2)
        self.assertIs(r, False)
        model = PyListModel([])
        r = model.moveRows(QModelIndex(), 0, 0, QModelIndex(), 0)
        self.assertIs(r, False)


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

    @unittest.skip
    def test_decoration(self):
        decorations = [self.model.data(self.model.index(i), Qt.DecorationRole)
                       for i in range(self.model.rowCount())]
        self.assertIs(decorations[1], decorations[2])
        del decorations[2]
        for i, dec1 in enumerate(decorations):
            for dec2 in decoreations[i]:
                self.assertIsNot(dec1, dec2)

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
