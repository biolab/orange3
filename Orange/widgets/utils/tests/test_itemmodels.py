# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest
from unittest.mock import patch

from AnyQt.QtCore import Qt, QModelIndex
from AnyQt.QtTest import QSignalSpy

from Orange.data import \
    Domain, \
    ContinuousVariable, DiscreteVariable, StringVariable, TimeVariable
from Orange.widgets.utils import colorpalettes
from Orange.widgets.utils.itemmodels import \
    PyTableModel, PyListModel, PyListModelTooltip,\
    VariableListModel, DomainModel, ContinuousPalettesModel, \
    _as_contiguous_range
from Orange.widgets.gui import TableVariable
from orangewidget.tests.base import GuiTest


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

    def test_init_wrap_empty(self):
        # pylint: disable=protected-access
        t = []
        model = PyTableModel(t)
        self.assertIs(model._table, t)
        t.append([1, 2, 3])
        self.assertEqual(list(model), [[1, 2, 3]])

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

    def test_set_iten_slice(self):
        self.model[:1] = [[10, 11], [12, 13], [14, 15]]
        self.assertEqual(list(self.model), [[10, 11], [12, 13], [14, 15], [2, 3]])

        self.model[1:3] = []
        self.assertEqual(list(self.model), [[10, 11], [2, 3]])

        self.model[:] = [[20, 21]]
        self.assertEqual(list(self.model), [[20, 21]])

        self.model[1:] = [[10, 11], [2, 3]]
        self.assertEqual(list(self.model), [[20, 21], [10, 11], [2, 3]])

    def test_emits_column_changes_on_row_insert(self):
        inserted = []
        removed = []
        model = PyTableModel()
        model.columnsInserted.connect(inserted.append)
        model.columnsRemoved.connect(removed.append)
        inserted = QSignalSpy(model.columnsInserted)
        removed = QSignalSpy(model.columnsRemoved)
        model.append([2])
        self.assertEqual(list(inserted)[-1][1:], [0, 0])
        model.append([2, 3])
        self.assertEqual(list(inserted)[-1][1:], [1, 1])
        del model[:]
        self.assertEqual(list(removed)[0][1:], [0, 1])
        model.extend([[0, 1], [0, 2]])
        self.assertEqual(list(inserted)[-1][1:], [0, 1])
        model.clear()
        self.assertEqual(list(removed)[0][1:], [0, 1])
        model[:] = [[1], [2]]
        self.assertEqual(list(inserted)[-1][1:], [0, 0])


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
        self.assertEqual(list(model), list(domain.variables))
        self.assertFalse(model.removeRows(0, 1))
        self.assertEqual(list(model), list(domain.variables))


class TestContinuousPalettesModel(GuiTest):
    def setUp(self):
        self.palette1, self.palette2 = \
            list(colorpalettes.ContinuousPalettes.values())[:2]

    def test_all_categories(self):
        model = ContinuousPalettesModel()
        shown = {palette.name for palette in model.items
                 if isinstance(palette, colorpalettes.Palette)}
        expected = {palette.name
                    for palette in colorpalettes.ContinuousPalettes.values()}
        self.assertEqual(expected, shown)

        shown = {name for name in model.items if isinstance(name, str)}
        expected = {palette.category
                    for palette in colorpalettes.ContinuousPalettes.values()}
        self.assertEqual(expected, shown)

    def test_category_selection(self):
        categories = ('Diverging', 'Linear')
        model = ContinuousPalettesModel(categories=categories)
        shown = {palette.name
                 for palette in model.items
                 if isinstance(palette, colorpalettes.Palette)}
        expected = {palette.name
                    for palette in colorpalettes.ContinuousPalettes.values()
                    if palette.category in categories}
        self.assertEqual(expected, shown)
        self.assertIn("Diverging", model.items)
        self.assertIn("Linear", model.items)

    def test_single_category(self):
        category = 'Diverging'
        model = ContinuousPalettesModel(categories=(category, ))
        shown = {palette.name
                 for palette in model.items
                 if isinstance(palette, colorpalettes.Palette)}
        expected = {palette.name
                    for palette in colorpalettes.ContinuousPalettes.values()
                    if palette.category == category}
        self.assertEqual(expected, shown)
        self.assertEqual(len(model.items), len(shown))

    def test_count(self):
        model = ContinuousPalettesModel()
        model.items = [self.palette1, self.palette1]
        self.assertEqual(model.rowCount(QModelIndex()), 2)
        self.assertEqual(model.columnCount(QModelIndex()), 1)

    def test_data(self):
        model = ContinuousPalettesModel()
        model.items = ["Palettes", self.palette1, self.palette2]
        data = model.data
        index = model.index

        self.assertEqual(data(index(0, 0), Qt.EditRole), "Palettes")
        self.assertEqual(data(index(1, 0), Qt.EditRole),
                         self.palette1.friendly_name)
        self.assertEqual(data(index(2, 0), Qt.EditRole),
                         self.palette2.friendly_name)

        self.assertEqual(data(index(0, 0), Qt.DisplayRole), "Palettes")
        self.assertEqual(data(index(1, 0), Qt.DisplayRole),
                         self.palette1.friendly_name)
        self.assertEqual(data(index(2, 0), Qt.DisplayRole),
                         self.palette2.friendly_name)

        self.assertIsNone(data(index(0, 0), Qt.DecorationRole))
        with patch.object(self.palette1, "color_strip") as color_strip:
            self.assertIs(data(index(1, 0), Qt.DecorationRole),
                          color_strip.return_value)
        with patch.object(self.palette2, "color_strip") as color_strip:
            self.assertIs(data(index(2, 0), Qt.DecorationRole),
                          color_strip.return_value)

        self.assertIsNone(data(index(0, 0), Qt.UserRole))
        self.assertIs(data(index(1, 0), Qt.UserRole), self.palette1)
        self.assertIs(data(index(2, 0), Qt.UserRole), self.palette2)

        self.assertIsNone(data(index(2, 0), Qt.FontRole))

    def test_select_flags(self):
        model = ContinuousPalettesModel()
        model.items = ["Palettes", self.palette1, self.palette2]
        self.assertFalse(model.flags(model.index(0, 0)) & Qt.ItemIsSelectable)
        self.assertTrue(model.flags(model.index(1, 0)) & Qt.ItemIsSelectable)
        self.assertTrue(model.flags(model.index(2, 0)) & Qt.ItemIsSelectable)

    def testIndexOf(self):
        model = ContinuousPalettesModel()
        model.items = ["Palettes", self.palette1, self.palette2]
        self.assertEqual(model.indexOf(self.palette1), 1)
        self.assertEqual(model.indexOf(self.palette1.name), 1)
        self.assertEqual(model.indexOf(self.palette1.friendly_name), 1)
        self.assertEqual(model.indexOf(self.palette2), 2)
        self.assertEqual(model.indexOf(self.palette2.name), 2)
        self.assertEqual(model.indexOf(self.palette2.friendly_name), 2)
        self.assertIsNone(model.indexOf(42))


class TestPyListModelTooltip(GuiTest):
    def test_tooltips_size(self):
        def data(i):
            return model.data(model.index(i, 0))

        def tip(i):
            return model.data(model.index(i, 0), Qt.ToolTipRole)

        # Not enough tooptips - return None
        model = PyListModelTooltip(["foo", "bar", "baz"], ["footip", "bartip"])
        self.assertEqual(data(1), "bar")
        self.assertEqual(data(2), "baz")
        self.assertIsNone(data(3))
        self.assertEqual(tip(1), "bartip")
        self.assertIsNone(tip(2))

        # No tooltips
        model = PyListModelTooltip(["foo", "bar", "baz"])
        self.assertIsNone(tip(1))
        self.assertIsNone(tip(2))

        # Too many tooltips
        model = PyListModelTooltip(["foo", "bar"], ["footip", "bartip", "btip"])
        self.assertEqual(data(0), "foo")
        self.assertEqual(data(1), "bar")
        self.assertIsNone(data(2))
        self.assertEqual(tip(1), "bartip")
        self.assertEqual(tip(2), "btip")

    def test_tooltip_arg(self):
        def tip(i):
            return model.data(model.index(i, 0), Qt.ToolTipRole)

        # Allow generators
        s = dict(a="ta", b="tb")
        model = PyListModelTooltip(s, s.values())
        self.assertEqual(tip(0), "ta")
        self.assertEqual(tip(1), "tb")

        # Basically backward compatibility; this behaviour diverges from
        # behaviour of data role
        s = []
        model = PyListModelTooltip(["foo"], s)
        self.assertIsNone(tip(0))

        s += ["footip"]
        self.assertEqual(tip(1), "footip")


if __name__ == "__main__":
    unittest.main()
