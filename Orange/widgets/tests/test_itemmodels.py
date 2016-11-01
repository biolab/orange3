# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

from unittest import TestCase

from AnyQt.QtCore import Qt

from Orange.data import Domain, ContinuousVariable
from Orange.widgets.utils.itemmodels import \
    PyTableModel, PyListModel, DomainModel, _argsort


class TestArgsort(TestCase):
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


class TestPyTableModel(TestCase):
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
        self.assertEqual(self.model[0][0], 2)

    def test_setHeaderLabels(self):
        self.model.setHorizontalHeaderLabels(['Col 1', 'Col 2'])
        self.assertEqual(self.model.headerData(1, Qt.Horizontal), 'Col 2')
        self.assertEqual(self.model.headerData(1, Qt.Vertical), '1')

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


class TestPyListModel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PyListModel([1, 2, 3, 4])

    def test_indexOf(self):
        self.assertEqual(self.model.indexOf(3), 2)

    def test_data(self):
        mi = self.model.index(2)
        self.assertEqual(self.model.data(mi), 3)
        self.assertEqual(self.model.data(mi, Qt.EditRole), 3)

    def test_set_data(self):
        model = PyListModel([1, 2, 3, 4])
        model.setData(model.index(0), None, Qt.EditRole)
        self.assertIs(model.data(model.index(0), Qt.EditRole), None)

        model.setData(model.index(1), "This is two", Qt.ToolTipRole)
        self.assertEqual(model.data(model.index(1), Qt.ToolTipRole),
                         "This is two",)

    def test_setitem(self):
        model = PyListModel([1, 2, 3, 4])
        model[1] = 42
        self.assertSequenceEqual(model, [1, 42, 3, 4])
        model[-1] = 42
        self.assertSequenceEqual(model, [1, 42, 3, 42])

        with self.assertRaises(IndexError):
            model[4]

        with self.assertRaises(IndexError):
            model[-5]

        model = PyListModel([1, 2, 3, 4])
        model[0:0] = [-1, 0]
        self.assertSequenceEqual(model, [-1, 0, 1, 2, 3, 4])

        model = PyListModel([1, 2, 3, 4])
        model[len(model):len(model)] = [5, 6]
        self.assertSequenceEqual(model, [1, 2, 3, 4, 5, 6])

        model = PyListModel([1, 2, 3, 4])
        model[0:2] = [-1, -2]
        self.assertSequenceEqual(model, [-1, -2, 3, 4])

        model = PyListModel([1, 2, 3, 4])
        model[-2:] = [-3, -4]
        self.assertSequenceEqual(model, [1, 2, -3, -4])

        model = PyListModel([1, 2, 3, 4])
        with self.assertRaises(IndexError):
            # non unit strides currently not supported
            model[0:-1:2] = [3, 3]

    def test_delitem(self):
        model = PyListModel([1, 2, 3, 4])
        del model[1]
        self.assertSequenceEqual(model, [1, 3, 4])

        model = PyListModel([1, 2, 3, 4])
        del model[1:3]

        self.assertSequenceEqual(model, [1, 4])
        model = PyListModel([1, 2, 3, 4])
        del model[:]
        self.assertSequenceEqual(model, [])

        model = PyListModel([1, 2, 3, 4])
        with self.assertRaises(IndexError):
            # non unit strides currently not supported
            del model[0:-1:2]

    def test_insert_delete_rows(self):
        model = PyListModel([1, 2, 3, 4])
        success = model.insertRows(0, 3)

        self.assertIs(success, True)
        self.assertSequenceEqual(model, [None, None, None, 1, 2, 3, 4])

        success = model.removeRows(3, 4)
        self.assertIs(success, True)
        self.assertSequenceEqual(model, [None, None, None])


class TestDomainModel(TestCase):
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
