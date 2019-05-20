import warnings
from distutils.version import LooseVersion
from unittest import TestCase
from unittest.mock import Mock

import Orange
from Orange.data import Domain, DiscreteVariable
from Orange.data import ContinuousVariable
from Orange.util import OrangeDeprecationWarning
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils import vartype

Continuous = 100 + vartype(ContinuousVariable())
Discrete = 100 + vartype(DiscreteVariable())


class TestDomainContextHandler(TestCase):
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
                     {'c1': Continuous - 100, 'd1': Discrete - 100,
                      'd2': Discrete - 100, 'd3': Discrete - 100},
                     {'c2': Continuous - 100, 'd4': Discrete - 100, })
        self.handler = DomainContextHandler()
        self.handler.read_defaults = lambda: None

    def test_encode_domain_with_match_none(self):
        handler = DomainContextHandler(
            match_values=DomainContextHandler.MATCH_VALUES_NONE)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes,
                         {'c1': Continuous - 100, 'd1': Discrete - 100,
                          'd2': Discrete - 100, 'd3': Discrete - 100})
        self.assertEqual(encoded_metas,
                         {'c2': Continuous - 100, 'd4': Discrete - 100, })

    def test_encode_domain_with_match_class(self):
        handler = DomainContextHandler(
            match_values=DomainContextHandler.MATCH_VALUES_CLASS)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes,
                         {'c1': Continuous - 100, 'd1': Discrete - 100,
                          'd2': Discrete - 100,
                          'd3': list('ghi')})
        self.assertEqual(encoded_metas,
                         {'c2': Continuous - 100, 'd4': Discrete - 100})

    def test_encode_domain_with_match_all(self):
        handler = DomainContextHandler(
            match_values=DomainContextHandler.MATCH_VALUES_ALL)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes,
                         {'c1': Continuous - 100, 'd1': list('abc'),
                          'd2': list('def'), 'd3': list('ghi')})
        self.assertEqual(encoded_metas,
                         {'c2': Continuous - 100, 'd4': list('jkl')})

    def test_match_returns_2_on_perfect_match(self):
        context = Mock(
            attributes=self.args[1], metas=self.args[2], values={})
        self.assertEqual(2., self.handler.match(context, *self.args))

    def test_match_returns_1_if_everything_matches(self):
        self.handler.bind(SimpleWidget)

        # Attributes in values
        context = Mock(values=dict(
            with_metas=('d1', Discrete),
            required=('d1', Discrete)))
        self.assertEqual(1., self.handler.match(context, *self.args))

        # Metas in values
        context = Mock(values=dict(
            with_metas=('d4', Discrete),
            required=('d1', Discrete)))
        self.assertEqual(1., self.handler.match(context, *self.args))

        # Attributes in lists
        context = Mock(values=dict(
            with_metas=[("d1", Discrete)]
        ))
        self.assertEqual(1., self.handler.match(context, *self.args))

        # Metas in lists
        context = Mock(values=dict(
            with_metas=[("d4", Discrete)]
        ))
        self.assertEqual(1., self.handler.match(context, *self.args))

    def test_match_returns_point_1_when_nothing_to_match(self):
        self.handler.bind(SimpleWidget)

        context = Mock(values={})
        self.assertEqual(0.1, self.handler.match(context, *self.args))

    def test_match_returns_zero_on_incompatible_context(self):
        self.handler.bind(SimpleWidget)

        # required
        context = Mock(values=dict(required=('u', Discrete),
                                   with_metas=('d1', Discrete)))
        self.assertEqual(0, self.handler.match(context, *self.args))

    def test_clone_context(self):
        self.handler.bind(SimpleWidget)
        context = self.create_context(self.domain, dict(
            text=('u', -2),
            with_metas=[('d1', Discrete), ('d1', Continuous),
                        ('c1', Continuous), ('c1', Discrete)],
            required=('u', Continuous)
        ))

        new_values = self.handler.clone_context(context, *self.args).values

        self.assertEqual(new_values['text'], ('u', -2))
        self.assertEqual([('d1', Discrete), ('c1', Continuous)],
                         new_values['with_metas'])
        self.assertNotIn('required', new_values)

    def test_open_context(self):
        self.handler.bind(SimpleWidget)
        context = self.create_context(self.domain, dict(
            text=('u', -2),
            with_metas=[('d1', Discrete), ('d2', Discrete)]
        ))
        self.handler.global_contexts = \
            [Mock(values={}), context, Mock(values={})]

        widget = SimpleWidget()
        self.handler.initialize(widget)
        old_metas_list = widget.with_metas
        self.handler.open_context(widget, self.args[0])

        context = widget.current_context
        self.assertEqual(context.attributes, self.args[1])
        self.assertEqual(context.metas, self.args[2])
        self.assertIs(old_metas_list, widget.with_metas)

        self.assertEqual(widget.text, 'u')
        self.assertEqual(widget.with_metas, [('d1', Discrete),
                                             ('d2', Discrete)])

    def test_open_context_with_imperfect_match(self):
        self.handler.bind(SimpleWidget)
        context = self.create_context(None, dict(
            text=('u', -2),
            with_metas=[('d1', Discrete), ('d1', Continuous),
                        ('c1', Continuous), ('c1', Discrete)]
        ))
        self.handler.global_contexts = \
            [Mock(values={}), context, Mock(values={})]

        widget = SimpleWidget()
        self.handler.initialize(widget)
        self.handler.open_context(widget, self.args[0])

        context = widget.current_context
        self.assertEqual(context.attributes, self.args[1])
        self.assertEqual(context.metas, self.args[2])

        self.assertEqual(widget.text, 'u')
        self.assertEqual(widget.with_metas, [('d1', Discrete),
                                             ('c1', Continuous)])

    def test_open_context_with_no_match(self):
        self.handler.bind(SimpleWidget)
        widget = SimpleWidget()
        self.handler.initialize(widget)
        widget.text = 'u'

        self.handler.open_context(widget, self.args[0])

        self.assertEqual(widget.text, 'u')
        self.assertEqual(widget.with_metas, [])
        context = widget.current_context
        self.assertEqual(context.attributes, self.args[1])
        self.assertEqual(context.metas, self.args[2])
        self.assertEqual(context.values['text'], ('u', -2))

    def test_filter_value(self):
        setting = ContextSetting([])
        setting.name = "value"

        def test_filter(before_value, after_value):
            data = dict(value=before_value)
            self.handler.filter_value(setting, data, *self.args)
            self.assertEqual(data.get("value", None), after_value)

        # filter list values
        test_filter([], [])
        # When list contains attributes asa tuple of (name, type),
        # Attributes not present in domain should be filtered out
        test_filter([("d1", Discrete), ("d1", Continuous),
                     ("c1", Continuous), ("c1", Discrete)],
                    [("d1", Discrete), ("c1", Continuous)])
        # All other values in list should remain
        test_filter([0, [1, 2, 3], "abcd", 5.4], [0, [1, 2, 3], "abcd", 5.4])

    def test_encode_setting(self):
        setting = ContextSetting(None)

        var = self.domain[0]
        val = self.handler.encode_setting(None, setting, var)
        self.assertEqual(val, (var.name, 100 + vartype(var)))

        # Should not crash on anonymous variables
        var.name = ""
        val = self.handler.encode_setting(None, setting, var)
        self.assertEqual(val, (var.name, 100 + vartype(var)))

    def test_encode_list_settings(self):
        setting = ContextSetting(None)

        var1, var2 = self.domain[:2]
        val = self.handler.encode_setting(None, setting, [None, var1, var2])
        self.assertEqual(
            val,
            ([None,
              (var1.name, 100 + vartype(var1)),
              (var2.name, 100 + vartype(var2))], -3))

        a_list = [1, 2, 3]
        val = self.handler.encode_setting(None, setting, a_list)
        self.assertEqual(val, [1, 2, 3])
        self.assertIsNot(val, a_list)

        a_list = []
        val = self.handler.encode_setting(None, setting, a_list)
        self.assertEqual(val, [])
        self.assertIsNot(val, a_list)

        a_list = [None, None]
        val = self.handler.encode_setting(None, setting, a_list)
        self.assertEqual(val, [None, None])
        self.assertIsNot(val, a_list)

    def test_decode_setting(self):
        setting = ContextSetting(None)

        var = self.domain[0]
        val = self.handler.decode_setting(setting, (var.name, 100 + vartype(var)),
                                          self.domain)
        self.assertIs(val, var)

        all_metas_domain = Domain([], metas=[var])
        val = self.handler.decode_setting(setting, (var.name, 100 + vartype(var)),
                                          all_metas_domain)
        self.assertIs(val, var)

    def test_decode_list_setting(self):
        setting = ContextSetting(None)

        var1, var2 = self.domain[:2]
        val = self.handler.decode_setting(
            setting,
            ([None,
              (var1.name, 100 + vartype(var1)),
              (var2.name, 100 + vartype(var2))], -3),
            self.domain)
        self.assertEqual(val, [None, var1, var2])

        val = self.handler.decode_setting(setting, [1, 2, 3], self.domain)
        self.assertEqual(val, [1, 2, 3])

    def test_backward_compatible_params(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DomainContextHandler(metas_in_res=True)
            self.assertIn(OrangeDeprecationWarning,
                          [x.category for x in w])

    def test_deprecated_str_as_var(self):
        if LooseVersion(Orange.__version__) >= LooseVersion("3.26"):
            # pragma: no cover
            self.fail("Remove support for variables stored as string settings "
                      "and this test.")

        context = Mock()
        context.attributes = {"foo": 2}
        context.metas = {}
        setting = ContextSetting("")
        setting.name = "setting_name"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DomainContextHandler.encode_setting(context, setting, "foo")
            self.assertIn("setting_name", w[0].message.args[0])


    def create_context(self, domain, values):
        if domain is None:
            domain = Domain([])

        context = self.handler.new_context(domain,
                                           *self.handler.encode_domain(domain))
        context.values = values
        return context


class SimpleWidget:
    text = ContextSetting("")
    with_metas = ContextSetting([], required=ContextSetting.OPTIONAL)
    required = ContextSetting("", required=ContextSetting.REQUIRED)

    def retrieveSpecificSettings(self):
        pass

    def storeSpecificSettings(self):
        pass
