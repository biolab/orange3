from unittest import TestCase
from unittest.mock import Mock
from Orange.data import Domain, DiscreteVariable
from Orange.data import ContinuousVariable
from Orange.widgets.settings import DomainContextHandler, ContextSetting
from Orange.widgets.utils import vartype

Continuous = vartype(ContinuousVariable())
Discrete = vartype(DiscreteVariable())


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
                     {'c1': Continuous, 'd1': Discrete,
                      'd2': Discrete, 'd3': Discrete},
                     {'c2': Continuous, 'd4': Discrete, })
        self.handler = DomainContextHandler(metas_in_res=True)
        self.handler.read_defaults = lambda: None

    def test_encode_domain_with_match_none(self):
        handler = DomainContextHandler(
            match_values=DomainContextHandler.MATCH_VALUES_NONE,
            metas_in_res=True)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes,
                         {'c1': Continuous, 'd1': Discrete,
                          'd2': Discrete, 'd3': Discrete})
        self.assertEqual(encoded_metas, {'c2': Continuous, 'd4': Discrete, })

    def test_encode_domain_with_match_class(self):
        handler = DomainContextHandler(
            match_values=DomainContextHandler.MATCH_VALUES_CLASS,
            metas_in_res=True)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes,
                         {'c1': Continuous, 'd1': Discrete, 'd2': Discrete,
                          'd3': tuple('ghi')})
        self.assertEqual(encoded_metas, {'c2': Continuous, 'd4': Discrete})

    def test_encode_domain_with_match_all(self):
        handler = DomainContextHandler(
            match_values=DomainContextHandler.MATCH_VALUES_ALL,
            metas_in_res=True)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes,
                         {'c1': Continuous, 'd1': tuple('abc'),
                          'd2': tuple('def'), 'd3': tuple('ghi')})
        self.assertEqual(encoded_metas,
                         {'c2': Continuous, 'd4': tuple('jkl')})

    def test_encode_domain_with_false_attributes_in_res(self):
        handler = DomainContextHandler(attributes_in_res=False,
                                       metas_in_res=True)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes, {})
        self.assertEqual(encoded_metas, {'c2': Continuous, 'd4': Discrete})

    def test_encode_domain_with_false_metas_in_res(self):
        handler = DomainContextHandler(attributes_in_res=True,
                                       metas_in_res=False)

        encoded_attributes, encoded_metas = handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes,
                         {'c1': Continuous, 'd1': Discrete,
                          'd2': Discrete, 'd3': Discrete})
        self.assertEqual(encoded_metas, {})

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

        # selected if_selected
        context = Mock(values=dict(with_metas=('d1', Discrete),
                                   if_selected=[('u', Discrete)],
                                   selected=[0]))
        self.assertEqual(0, self.handler.match(context, *self.args))

        # unselected if_selected
        context = Mock(values=dict(with_metas=('d1', Discrete),
                                   if_selected=[('u', Discrete),
                                                ('d1', Discrete)],
                                   selected=[1]))
        self.assertAlmostEqual(0.667, self.handler.match(context, *self.args),
                               places=2)

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
        self.handler.open_context(widget, self.args[0])

        context = widget.current_context
        self.assertEqual(context.attributes, self.args[1])
        self.assertEqual(context.metas, self.args[2])
        self.assertSequenceEqual(context.ordered_domain,
                                 (('c1', Continuous), ('d1', Discrete),
                                  ('d2', Discrete), ('d3', Discrete),
                                  ('c2', Continuous), ('d4', Discrete)))

        self.assertEqual(widget.text, 'u')
        self.assertEqual(widget.with_metas, [('d1', Discrete),
                                             ('d2', Discrete)])

    def test_open_context_with_imperfect_match(self):
        self.handler.bind(SimpleWidget)
        context = self.create_context(None, dict(
            text=('u', -2),
            with_metas=[('d1', Discrete), ('d1', Continuous),
                        ('c1', Continuous), ('c1', Discrete)],
            if_selected=[('c1', Discrete), ('c1', Continuous),
                         ('d1', Discrete), ('d1', Continuous)],
            selected=[2],
        ))
        self.handler.global_contexts = \
            [Mock(values={}), context, Mock(values={})]

        widget = SimpleWidget()
        self.handler.initialize(widget)
        self.handler.open_context(widget, self.args[0])

        context = widget.current_context
        self.assertEqual(context.attributes, self.args[1])
        self.assertEqual(context.metas, self.args[2])
        self.assertSequenceEqual(context.ordered_domain,
                                 (('c1', Continuous), ('d1', Discrete),
                                  ('d2', Discrete), ('d3', Discrete),
                                  ('c2', Continuous), ('d4', Discrete)))

        self.assertEqual(widget.text, 'u')
        self.assertEqual(widget.with_metas, [('d1', Discrete),
                                             ('c1', Continuous)])
        self.assertEqual(widget.if_selected, [('c1', Continuous),
                                              ('d1', Discrete)])
        self.assertEqual(widget.selected, [1])

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
        self.assertSequenceEqual(context.ordered_domain,
                                 (('c1', Continuous), ('d1', Discrete),
                                  ('d2', Discrete), ('d3', Discrete),
                                  ('c2', Continuous), ('d4', Discrete)))
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

    def test_decode_setting(self):
        setting = ContextSetting(None)

        var = self.domain[0]
        val = self.handler.decode_setting(setting, (var.name, 100 + vartype(var)), self.domain)
        self.assertIs(val, var)

    def create_context(self, domain, values):
        if not domain:
            domain = Domain([])

        context = self.handler.new_context(domain,
                                           *self.handler.encode_domain(domain))
        context.values = values
        return context


class SimpleWidget:
    text = ContextSetting("", not_attribute=True)
    with_metas = ContextSetting([], exclude_metas=False)
    required = ContextSetting("", required=ContextSetting.REQUIRED)
    if_selected = ContextSetting([], required=ContextSetting.IF_SELECTED,
        selected='selected')
    selected = ""

    def retrieveSpecificSettings(self):
        pass

    def storeSpecificSettings(self):
        pass
