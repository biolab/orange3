# Test methods with descriptive names can omit docstrings
# pylint: disable=missing-docstring

from unittest import TestCase
from unittest.mock import Mock

from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.settings import ContextSetting, PerfectDomainContextHandler, Context, Setting
from Orange.widgets.utils import vartype

Continuous = vartype(ContinuousVariable())
Discrete = vartype(DiscreteVariable())


class TestPerfectDomainContextHandler(TestCase):
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
                     (('c1', Continuous), ('d1', Discrete), ('d2', Discrete)),
                     (('d3', Discrete),),
                     (('c2', Continuous), ('d4', Discrete)))
        self.args_match_all = (self.domain,
                               (('c1', Continuous), ('d1', list('abc')), ('d2', list('def'))),
                               (('d3', list('ghi')),),
                               (('c2', Continuous), ('d4', list('jkl'))))
        self.handler = PerfectDomainContextHandler()
        self.handler.read_defaults = lambda: None
        self.handler.bind(SimpleWidget)
        self.widget = SimpleWidget()
        self.handler.initialize(self.widget)

    def test_new_context(self):
        context = self.handler.new_context(*self.args)
        _, attrs, class_vars, metas = self.args

        self.assertEqual(context.attributes, attrs)
        self.assertEqual(context.class_vars, class_vars)
        self.assertEqual(context.metas, metas)

    def test_open_context(self):
        context = Context()
        context.attributes = ()
        context.class_vars = ()
        context.metas = ()
        self.handler.new_context = Mock(return_value=context)
        self.handler.open_context(self.widget, self.domain)
        self.handler.new_context.assert_called_with(*self.args)

    def test_encode_domain_simple(self):
        attrs, class_vars, metas = self.handler.encode_domain(self.domain)

        self.assertEqual(attrs, (('c1', Continuous), ('d1', Discrete), ('d2', Discrete)))
        self.assertEqual(class_vars, (('d3', Discrete),))
        self.assertEqual(metas, (('c2', Continuous), ('d4', Discrete)))

    def test_encode_domain_match_values(self):
        self.handler.match_values = self.handler.MATCH_VALUES_ALL
        attrs, class_vars, metas = self.handler.encode_domain(self.domain)

        self.assertEqual(attrs, (('c1', Continuous), ('d1', list('abc')), ('d2', list('def'))))
        self.assertEqual(class_vars, (('d3', list('ghi')),))
        self.assertEqual(metas, (('c2', Continuous), ('d4', list('jkl'))))

    def test_match_simple(self):
        domain, attrs, class_vars, metas = self.args
        context = self._create_context(attrs, class_vars, metas)

        self.assertEqual(self.handler.match(context, *self.args),
                         self.handler.PERFECT_MATCH)

        attrs2 = list(attrs)
        attrs2[:2] = attrs[1::-1]
        self.assertEqual(self.handler.match(context,
                                            domain, attrs2, class_vars, metas),
                         self.handler.NO_MATCH)

        attrs3 = list(attrs)
        attrs3.append(attrs[0])
        self.assertEqual(self.handler.match(context,
                                            domain, attrs3, class_vars, metas),
                         self.handler.NO_MATCH)

        metas2 = list(metas)
        metas2.append(attrs[0])
        self.assertEqual(self.handler.match(context,
                                            domain, attrs, class_vars, metas2),
                         self.handler.NO_MATCH)

    def test_match_values(self):
        domain, attrs, class_vars, metas = self.args_match_all
        context = self._create_context(attrs, class_vars, metas)

        self.handler.match_values = self.handler.MATCH_VALUES_ALL
        self.assertEqual(self.handler.match(context, *self.args_match_all),
                         self.handler.PERFECT_MATCH)

        attrs2 = list(attrs)
        attrs2[:2] = attrs[1::-1]
        self.assertEqual(self.handler.match(context,
                                            domain, attrs2, class_vars, metas),
                         self.handler.NO_MATCH)

        attrs3 = list(attrs)
        attrs3.append(attrs[0])
        self.assertEqual(self.handler.match(context,
                                            domain, attrs2, class_vars, metas),
                         self.handler.NO_MATCH)

    def test_encode_setting(self):
        _, attrs, class_vars, metas = self.args
        context = self._create_context(attrs, class_vars, metas)
        encoded_setting = self.handler.encode_setting(
            context, SimpleWidget.setting, "d1")
        self.assertEqual(encoded_setting, ("d1", -2))

        encoded_setting = self.handler.encode_setting(
            context, SimpleWidget.text, "d1")
        self.assertEqual(encoded_setting, ("d1", -1))

        encoded_setting = self.handler.encode_setting(
            context, SimpleWidget.with_metas, "d4")
        self.assertEqual(encoded_setting, ("d4", 1))

    def _create_context(self, attrs, class_vars, metas):
        context = Context()
        context.attributes = attrs
        context.class_vars = class_vars
        context.metas = metas
        return context


class SimpleWidget:
    setting = Setting("foo")
    text = ContextSetting("", exclude_attributes=True, exclude_metas=True)
    with_metas = ContextSetting([])

    def retrieveSpecificSettings(self):
        pass

    def storeSpecificSettings(self):
        pass
