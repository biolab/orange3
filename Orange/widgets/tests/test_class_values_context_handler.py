from unittest import TestCase
from unittest.mock import Mock
from Orange.data import DiscreteVariable, Domain
from Orange.data import ContinuousVariable
from Orange.widgets.settings import ContextSetting, ClassValuesContextHandler
from Orange.widgets.utils import vartype

Continuous = vartype(ContinuousVariable())
Discrete = vartype(DiscreteVariable())


class TestClassValuesContextHandler(TestCase):
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
        self.handler = ClassValuesContextHandler()
        self.handler.read_defaults = lambda: None

    def test_open_context(self):
        self.handler.bind(SimpleWidget)
        context = Mock(
            classes=['g', 'h', 'i'], values=dict(
                text='u',
                with_metas=[('d1', Discrete), ('d2', Discrete)]
            ))
        self.handler.global_contexts = \
            [Mock(values={}), context, Mock(values={})]

        widget = SimpleWidget()
        self.handler.initialize(widget)
        self.handler.open_context(widget, self.args[0].class_var)
        self.assertEqual(widget.text, 'u')
        self.assertEqual(widget.with_metas, [('d1', Discrete),
                                             ('d2', Discrete)])

    def test_open_context_with_no_match(self):
        self.handler.bind(SimpleWidget)
        context = Mock(
            classes=['g', 'h', 'i'], values=dict(
                text='u',
                with_metas=[('d1', Discrete), ('d2', Discrete)]
            ))
        self.handler.global_contexts = \
            [Mock(values={}), context, Mock(values={})]
        widget = SimpleWidget()
        self.handler.initialize(widget)
        widget.text = 'u'

        self.handler.open_context(widget, self.args[0][1])

        context = widget.current_context
        self.assertEqual(context.classes, ['a', 'b', 'c'])
        self.assertEqual(widget.text, 'u')
        self.assertEqual(widget.with_metas, [])


class SimpleWidget:
    text = ContextSetting("", not_attribute=True)
    with_metas = ContextSetting([])
    required = ContextSetting("", required=ContextSetting.REQUIRED)
    if_selected = ContextSetting([], required=ContextSetting.IF_SELECTED,
                                 selected='selected')
    selected = ""

    def retrieveSpecificSettings(self):
        pass
    def storeSpecificSettings(self):
        pass
