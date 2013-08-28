import unittest
from mock import Mock
from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.settings import DomainContextHandler, ContextSetting

VarTypes = ContinuousVariable.VarTypes


class DomainContextSettingsHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = DomainContextHandler(attributes_in_res=True,
                                            metas_in_res=True)
        self.handler.read_defaults = lambda: None  # Disable reading from disk
        self.domain = self._create_domain()

    def test_encode_domain_with_match_none(self):
        self.handler.match_values = self.handler.MATCH_VALUES_NONE

        encoded_attributes, encoded_metas = \
            self.handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes, {
            'cf1': VarTypes.Continuous,
            'df1': VarTypes.Discrete,
            'df2': VarTypes.Discrete,
            'dc1': VarTypes.Discrete,
        })
        self.assertEqual(encoded_metas, {
            'cm1': VarTypes.Continuous,
            'dm1': VarTypes.Discrete,
        })

    def test_encode_domain_with_match_class(self):
        self.handler.match_values = self.handler.MATCH_VALUES_CLASS

        encoded_attributes, encoded_metas = \
            self.handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes, {
            'cf1': VarTypes.Continuous,
            'df1': VarTypes.Discrete,
            'df2': VarTypes.Discrete,
            'dc1': ["g", "h", "i"],
        })
        self.assertEqual(encoded_metas, {
            'cm1': VarTypes.Continuous,
            'dm1': VarTypes.Discrete,
        })

    def test_encode_domain_with_match_all(self):
        self.handler.match_values = self.handler.MATCH_VALUES_ALL

        encoded_attributes, encoded_metas = \
            self.handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes, {
            'cf1': VarTypes.Continuous,
            'df1': ["a", "b", "c"],
            'df2': ["d", "e", "f"],
            'dc1': ["g", "h", "i"],
        })
        self.assertEqual(encoded_metas, {
            'cm1': VarTypes.Continuous,
            'dm1': ["j", "k", "l"],
        })

    def test_encode_domain_with_false_attributes_in_res(self):
        self.handler = DomainContextHandler(attributes_in_res=False,
                                            metas_in_res=True)
        encoded_attributes, encoded_metas = \
            self.handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes, {})
        self.assertEqual(encoded_metas, {
            'cm1': VarTypes.Continuous,
            'dm1': VarTypes.Discrete,
        })

    def test_encode_domain_with_false_metas_in_res(self):
        self.handler = DomainContextHandler(attributes_in_res=True,
                                            metas_in_res=False)
        encoded_attributes, encoded_metas = \
            self.handler.encode_domain(self.domain)

        self.assertEqual(encoded_attributes, {
            'cf1': VarTypes.Continuous,
            'df1': VarTypes.Discrete,
            'df2': VarTypes.Discrete,
            'dc1': VarTypes.Discrete,
        })
        self.assertEqual(encoded_metas, {})

    def test_settings_from_widget(self):
        widget = MockWidget()
        widget.current_context.attributes, widget.current_context.metas = \
            self.handler.encode_domain(self.domain)
        self.add_setting(widget, "string_setting", ContextSetting("abc"))
        self.add_setting(widget, "list_setting", ContextSetting([1, 2, 3]))
        self.add_setting(widget, "dict_setting", ContextSetting({1: 2}))
        self.add_setting(widget, "continuous_setting", ContextSetting("cf1"))
        self.add_setting(widget, "discrete_setting", ContextSetting("df1"))
        self.add_setting(widget, "class_setting", ContextSetting("dc1"))
        self.add_setting(widget, "excluded_meta_setting", ContextSetting("dm1"))
        self.add_setting(widget, "meta_setting",
                         ContextSetting("dm1", exclude_metas=False))

        self.handler.settings_from_widget(widget)

        self.assertEqual(widget.current_context.values, dict(
            string_setting=("abc", -2),
            list_setting=[1, 2, 3],
            dict_setting=({1: 2}, -2),
            continuous_setting=("cf1", VarTypes.Continuous),
            discrete_setting=("df1", VarTypes.Discrete),
            class_setting=("dc1", VarTypes.Discrete),
            excluded_meta_setting=("dm1", -2),
            meta_setting=("dm1", VarTypes.Discrete),
        ))

    def test_settings_to_widget(self):
        widget = MockWidget()
        widget.current_context.attributes, widget.current_context.metas = \
            self.handler.encode_domain(self.domain)
        self.add_setting(widget, "string_setting", ContextSetting(""))
        self.add_setting(widget, "continuous_setting", ContextSetting(""))
        self.add_setting(widget, "discrete_setting", ContextSetting(""))
        self.add_setting(widget, "list_setting", ContextSetting([]))
        self.add_setting(widget, "attr_list_setting",
                         ContextSetting([], selected="selection1"))
        self.add_setting(widget, "attr_tuple_list_setting",
                         ContextSetting([], selected="selection2",
                                        exclude_metas=False))
        widget.current_context.values = dict(
            string_setting=("abc", -2),
            continuous_setting=("cf1", VarTypes.Continuous),
            discrete_setting=("df1", VarTypes.Discrete),
            list_setting=[1, 2, 3],
            attr_list_setting=["df1", "dc1"],
            selection1=[0],
            attr_tuple_list_setting=[("dm1", VarTypes.Discrete),
                                     ("cm1", VarTypes.Continuous)],
            selection2=[1],
        )

        self.handler.settings_to_widget(widget)

        self.assertEqual(widget.string_setting, "abc")
        self.assertEqual(widget.continuous_setting, "cf1")
        self.assertEqual(widget.discrete_setting, "df1")
        self.assertEqual(widget.list_setting, [1, 2, 3])
        self.assertEqual(widget.attr_list_setting, ["df1", "dc1"])
        self.assertEqual(widget.attr_tuple_list_setting,
                         ["dm1", "cm1"])
        self.assertEqual(widget.selection1, [0])
        self.assertEqual(widget.selection2, [1])

    def test_settings_to_widget_filters_selections(self):
        widget = MockWidget()
        widget.current_context.attributes, widget.current_context.metas = \
            self.handler.encode_domain(self.domain)
        self.add_setting(widget, "attr_list_setting",
                         ContextSetting([], selected="selection"))
        widget.current_context.values = dict(
            string_setting=("abc", -2),
            continuous_setting=("cf1", VarTypes.Continuous),
            discrete_setting=("df1", VarTypes.Discrete),
            list_setting=[1, 2, 3],
            attr_list_setting=["dm1", "df1", "cm1", "dc1"],
            selection=[1, 2],
        )

        self.handler.settings_to_widget(widget)

        self.assertEqual(widget.attr_list_setting, ["df1", "dc1"])
        self.assertEqual(widget.selection, [0])

    def test_perfect_match_returns_2(self):
        attrs, metas = self.handler.encode_domain(self.domain)
        mock_context = Mock(attributes=attrs, metas=metas, values={})

        self.assertEqual(self.handler.match(mock_context, attrs, metas), 2.)

    def test_match_when_nothing_to_match_returns_point_1(self):
        attrs, metas = self.handler.encode_domain(self.domain)
        mock_context = Mock(values={})

        self.assertEqual(self.handler.match(mock_context, attrs, metas), 0.1)

    def test_match_if_all_values_match_returns_1(self):
        attrs, metas = self.handler.encode_domain(self.domain)
        mock_context = Mock(values={})
        self.add_setting(mock_context, "setting", ContextSetting(""))
        self.add_setting(mock_context, "required_setting",
                         ContextSetting("", required=ContextSetting.REQUIRED))
        mock_context.values["setting"] = ("df1", VarTypes.Discrete)
        mock_context.values["required_setting"] = ("df1", VarTypes.Discrete)

        self.assertEqual(self.handler.match(mock_context, attrs, metas), 1.)

    def test_match_if_all_list_values_match_returns_1(self):
        attrs, metas = self.handler.encode_domain(self.domain)
        mock_context = Mock(values={})
        self.add_setting(mock_context, "setting", ContextSetting(""))
        mock_context.values["setting"] = [("df1", VarTypes.Discrete)]

        self.assertEqual(self.handler.match(mock_context, attrs, metas), 1.)

    def test_match_if_all_required_list_values_match_returns_1(self):
        attrs, metas = self.handler.encode_domain(self.domain)
        mock_context = Mock(values={})
        self.add_setting(mock_context, "required_setting",
                         ContextSetting("", required=ContextSetting.REQUIRED))
        mock_context.values["required_setting"] = [("df1", VarTypes.Discrete)]

        self.assertEqual(self.handler.match(mock_context, attrs, metas), 1.)

    def add_setting(self, widget, name, setting):
        setting.name = name
        setattr(widget, name, setting.default)
        self.handler.settings[name] = setting

    def _create_domain(self):
        features = [
            ContinuousVariable(name="cf1"),
            DiscreteVariable(name="df1", values=["a", "b", "c"]),
            DiscreteVariable(name="df2", values=["d", "e", "f"])
        ]
        class_vars = [
            DiscreteVariable(name="dc1", values=["g", "h", "i"])
        ]
        metas = [
            ContinuousVariable(name="cm1"),
            DiscreteVariable(name="dm1", values=["j", "k", "l"]),
        ]
        return Domain(features, class_vars, metas)


class MockWidget:
    storeSpecificSettings = lambda x: None
    retrieveSpecificSettings = lambda x: None
    getattr_deep = lambda self, name: getattr(self, name)

    def __init__(self):
        self.current_context = Mock()


if __name__ == '__main__':
    unittest.main(verbosity=2)
