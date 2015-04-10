from unittest import TestCase
from unittest.mock import Mock
from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.data.contexthandlers import \
    SelectAttributesDomainContextHandler
from Orange.widgets.settings import ContextSetting
from Orange.widgets.utils import vartype


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

        self.handler = SelectAttributesDomainContextHandler(metas_in_res=True)
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


class SimpleWidget:
    domain_role_hints = ContextSetting({}, exclude_metas=False)
    required = ContextSetting("", required=ContextSetting.REQUIRED)

    def retrieveSpecificSettings(self):
        pass

    def storeSpecificSettings(self):
        pass
