from unittest import TestCase
from unittest.mock import Mock

from AnyQt.QtCore import Qt

from Orange.data import Table, ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.data.contexthandlers import \
    SelectAttributesDomainContextHandler
from Orange.widgets.settings import ContextSetting
from Orange.widgets.utils import vartype
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.data.owselectcolumns \
    import OWSelectAttributes, VariablesListItemModel

Continuous = vartype(ContinuousVariable())
Discrete = vartype(DiscreteVariable())


class TestSelectAttributesDomainContextHandler(TestCase):
    def setUp(self):
        self.domain = Domain(
            attributes=[ContinuousVariable('c1'),
                        DiscreteVariable('d1', values='abc'),
                        DiscreteVariable('d2', values='def')],
            class_vars=[DiscreteVariable('d3', values='ghi')],
            metas=[ContinuousVariable('c2'),
                   DiscreteVariable('d4', values='jkl')]
        )
        self.args = (self.domain,
                     {'c1': Continuous, 'd1': Discrete,
                      'd2': Discrete, 'd3': Discrete},
                     {'c2': Continuous, 'd4': Discrete, })

        self.handler = SelectAttributesDomainContextHandler()
        self.handler.read_defaults = lambda: None

    def test_open_context(self):
        self.handler.bind(SimpleWidget)
        context = Mock(
            attributes=self.args[1], metas=self.args[2], values=dict(
                domain_role_hints=({('d1', Discrete): ('available', 0),
                                    ('d2', Discrete): ('meta', 0),
                                    ('c1', Continuous): ('attribute', 0),
                                    ('d3', Discrete): ('attribute', 1),
                                    ('d4', Discrete): ('attribute', 2),
                                    ('c2', Continuous): ('class', 0)}, -2),
                with_metas=[('d1', Discrete), ('d2', Discrete)]
            ))
        self.handler.global_contexts = \
            [Mock(values={}), context, Mock(values={})]

        widget = SimpleWidget()
        self.handler.initialize(widget)
        self.handler.open_context(widget, self.args[0])
        self.assertEqual(widget.domain_role_hints,
                         {('d1', Discrete): ('available', 0),
                          ('d2', Discrete): ('meta', 0),
                          ('c1', Continuous): ('attribute', 0),
                          ('d3', Discrete): ('attribute', 1),
                          ('d4', Discrete): ('attribute', 2),
                          ('c2', Continuous): ('class', 0)})

    def test_open_context_with_imperfect_match(self):
        self.handler.bind(SimpleWidget)
        context = Mock(values=dict(
            domain_role_hints=({('d1', Discrete): ('available', 0),
                                ('d2', Discrete): ('meta', 0),
                                ('c1', Continuous): ('attribute', 0),
                                ('d6', Discrete): ('attribute', 1),
                                ('d7', Discrete): ('attribute', 2),
                                ('c2', Continuous): ('class', 0)}, -2)
        ))
        self.handler.global_contexts = \
            [Mock(values={}), context, Mock(values={})]

        widget = SimpleWidget()
        self.handler.initialize(widget)
        self.handler.open_context(widget, self.args[0])

        self.assertEqual(widget.domain_role_hints,
                         {('d1', Discrete): ('available', 0),
                          ('d2', Discrete): ('meta', 0),
                          ('c1', Continuous): ('attribute', 0),
                          ('c2', Continuous): ('class', 0)})

    def test_open_context_with_no_match(self):
        self.handler.bind(SimpleWidget)
        context = Mock(values=dict(
            domain_role_hints=({('d1', Discrete): ('available', 0),
                               ('d2', Discrete): ('meta', 0),
                               ('c1', Continuous): ('attribute', 0),
                               ('d3', Discrete): ('attribute', 1),
                               ('d4', Discrete): ('attribute', 2),
                               ('c2', Continuous): ('class', 0)}, -2),
            required=('g1', Continuous),
        ))
        self.handler.global_contexts = [context]
        widget = SimpleWidget()
        self.handler.initialize(widget)
        self.handler.open_context(widget, self.args[0])
        self.assertEqual(widget.domain_role_hints, {})


class TestModel(TestCase):
    def test_drop_mime(self):
        iris = Table("iris")
        m = VariablesListItemModel(iris.domain.variables)
        mime = m.mimeData([m.index(1, 0)])
        self.assertTrue(mime.hasFormat(VariablesListItemModel.MIME_TYPE))
        assert m.dropMimeData(mime, Qt.MoveAction, 5, 0, m.index(-1, -1))
        self.assertIs(m[5], m[1])
        assert m.dropMimeData(mime, Qt.MoveAction, -1, -1, m.index(-1, -1))
        self.assertIs(m[6], m[1])

    def test_flags(self):
        m = VariablesListItemModel([ContinuousVariable("X")])
        index = m.index(0)
        flags = m.flags(m.index(0))
        self.assertTrue(flags & Qt.ItemIsDragEnabled)
        self.assertFalse(flags & Qt.ItemIsDropEnabled)
        # 'invalid' index is drop enabled -> indicates insertion capability
        flags = m.flags(m.index(-1, -1))
        self.assertTrue(flags & Qt.ItemIsDropEnabled)


class SimpleWidget:
    domain_role_hints = ContextSetting({})
    required = ContextSetting("", required=ContextSetting.REQUIRED)

    def retrieveSpecificSettings(self):
        pass

    def storeSpecificSettings(self):
        pass


class TestOWSelectAttributes(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSelectAttributes)

    def assertVariableCountsEqual(self, available, used, classattrs):
        self.assertEqual(len(self.widget.available_attrs), available)
        self.assertEqual(len(self.widget.used_attrs), used)
        self.assertEqual(len(self.widget.class_attrs), classattrs)

    def test_multiple_target_variable(self):
        """
        More than one target variable can be moved to a box for target variables
        at the same time and moved back as well.
        GH-2100
        GH-2086
        """
        iris = Table("iris")
        self.send_signal(self.widget.Inputs.data, iris)
        self.assertVariableCountsEqual(available=0, used=4, classattrs=1)

        self.widget.move_class_button.click()
        self.assertVariableCountsEqual(available=0, used=4, classattrs=1)

        self.widget.used_attrs_view.selectAll()
        self.widget.move_selected(self.widget.used_attrs_view)
        self.assertVariableCountsEqual(available=4, used=0, classattrs=1)

        self.widget.available_attrs_view.selectAll()
        self.widget.move_selected(self.widget.class_attrs_view)
        self.assertVariableCountsEqual(available=0, used=0, classattrs=5)
