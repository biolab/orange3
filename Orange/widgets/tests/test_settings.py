import unittest

from unittest.mock import Mock

from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.settings import DomainContextHandler, ContextSetting, Setting, \
    SettingProvider
from Orange.widgets.utils import vartype


CONTINOUS_ATTR = "cf1"
DISCRETE_ATTR_ABC = "df1"
DISCRETE_ATTR_DEF = "df2"
DISCRETE_CLASS_GHI = "dc1"
CONTINUOUS_META = "cm1"
DISCRETE_META_JKL = "dm1"

UNKNOWN_TYPE = -2
VALUE = "abc"

Continuous = vartype(ContinuousVariable())
Discrete = vartype(DiscreteVariable())

domain = Domain(
    attributes=[
        ContinuousVariable(name=CONTINOUS_ATTR),
        DiscreteVariable(name=DISCRETE_ATTR_ABC, values=["a", "b", "c"]),
        DiscreteVariable(name=DISCRETE_ATTR_DEF, values=["d", "e", "f"])
    ],
    class_vars=[
        DiscreteVariable(name=DISCRETE_CLASS_GHI, values=["g", "h", "i"])
    ],
    metas=[
        ContinuousVariable(name=CONTINUOUS_META),
        DiscreteVariable(name=DISCRETE_META_JKL, values=["j", "k", "l"]),
    ]
)



class MockComponent():
    setting = Setting("")


class MockWidget:
    name = "MockWidget"

    storeSpecificSettings = lambda x: None
    retrieveSpecificSettings = lambda x: None

    ordinary_setting = Setting("")
    string_setting = ContextSetting("")
    list_setting = ContextSetting([])
    dict_setting = ContextSetting({})
    continuous_setting = ContextSetting("")
    discrete_setting = ContextSetting("")
    class_setting = ContextSetting("")
    excluded_meta_setting = ContextSetting("")
    meta_setting = ContextSetting("", exclude_metas=False)

    attr_list_setting = ContextSetting([], selected="selection1")
    selection1 = ()

    attr_tuple_list_setting = ContextSetting([], selected="selection2", exclude_metas=False)
    selection2 = ()

    required_setting = ContextSetting("", required=ContextSetting.REQUIRED)

    subprovider = SettingProvider(MockComponent)

    def __init__(self):
        self.current_context = Mock()

        self.subprovider = MockComponent()


class DomainContextSettingsHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = DomainContextHandler(attributes_in_res=True,
                                            metas_in_res=True)
        self.handler.read_defaults = lambda: None  # Disable reading settings from disk
        self.handler.bind(MockWidget)
        self.widget = MockWidget()
        encoded_attributes, encoded_metas = self.handler.encode_domain(domain)
        self.widget.current_context.attributes = encoded_attributes
        self.widget.current_context.metas = encoded_metas
        self.handler.initialize(self.widget)
        self.handler.initialize(self.widget.subprovider)
        self.handler.open_context(self.widget, domain)

    def test_settings_from_widget(self):
        widget = self.widget
        widget.ordinary_setting = VALUE
        widget.string_setting = VALUE
        widget.list_setting = [1, 2, 3]
        widget.dict_setting = {1: 2}
        widget.continuous_setting = CONTINOUS_ATTR
        widget.discrete_setting = DISCRETE_ATTR_ABC
        widget.class_setting = DISCRETE_CLASS_GHI
        widget.excluded_meta_setting = DISCRETE_META_JKL
        widget.meta_setting = DISCRETE_META_JKL

        self.handler.settings_from_widget(widget)

        values = widget.current_context.values
        self.assertEqual((VALUE, UNKNOWN_TYPE), values['ordinary_setting'])
        self.assertEqual((VALUE, UNKNOWN_TYPE), values['string_setting'])
        self.assertEqual([1, 2, 3], values['list_setting'])
        self.assertEqual(({1: 2}, UNKNOWN_TYPE), values['dict_setting'])
        self.assertEqual((CONTINOUS_ATTR, Continuous), values['continuous_setting'])
        self.assertEqual((DISCRETE_ATTR_ABC, Discrete), values['discrete_setting'])
        self.assertEqual((DISCRETE_CLASS_GHI, Discrete), values['class_setting'])
        self.assertEqual((DISCRETE_META_JKL, UNKNOWN_TYPE), values['excluded_meta_setting'])
        self.assertEqual((DISCRETE_META_JKL, Discrete), values['meta_setting'])

    def test_settings_to_widget(self):
        self.widget.current_context.values = dict(
            string_setting=(VALUE, -2),
            continuous_setting=(CONTINOUS_ATTR, Continuous),
            discrete_setting=(DISCRETE_ATTR_ABC, Discrete),
            list_setting=[1, 2, 3],
            attr_list_setting=[DISCRETE_ATTR_ABC, DISCRETE_CLASS_GHI],
            selection1=[0],
            attr_tuple_list_setting=[(DISCRETE_META_JKL, Discrete),
                                     (CONTINUOUS_META, Continuous)],
            selection2=[1],
        )

        self.handler.settings_to_widget(self.widget)

        self.assertEqual(self.widget.string_setting, VALUE)
        self.assertEqual(self.widget.continuous_setting, CONTINOUS_ATTR)
        self.assertEqual(self.widget.discrete_setting, DISCRETE_ATTR_ABC)
        self.assertEqual(self.widget.list_setting, [1, 2, 3])
        self.assertEqual(self.widget.attr_list_setting, [DISCRETE_ATTR_ABC, DISCRETE_CLASS_GHI])
        self.assertEqual(self.widget.attr_tuple_list_setting,
                         [DISCRETE_META_JKL, CONTINUOUS_META])
        self.assertEqual(self.widget.selection1, [0])
        self.assertEqual(self.widget.selection2, [1])

    def test_settings_to_widget_filters_selections(self):
        self.widget.current_context.values = dict(
            attr_list_setting=[DISCRETE_META_JKL, DISCRETE_ATTR_ABC,
                               CONTINUOUS_META, DISCRETE_CLASS_GHI],
            selection1=[1, 2],
        )

        self.handler.settings_to_widget(self.widget)

        self.assertEqual(self.widget.attr_list_setting, [DISCRETE_ATTR_ABC, DISCRETE_CLASS_GHI])
        self.assertEqual(self.widget.selection1, [0])

    def test_clone_context(self):
        mock_context = Mock(values=dict(
            required_setting=(DISCRETE_ATTR_ABC, Discrete)
        ))
        attrs, metas = self.handler.encode_domain(domain)
        cloned_context = self.handler.clone_context(mock_context, domain, attrs, metas)
        self.assertEqual(cloned_context.values, mock_context.values)

    def add_setting(self, widget, name, setting):
        setting.name = name
        setattr(widget, name, setting.default)
        self.handler.provider.settings[name] = setting

    def match(self, context):
        attrs, metas = self.handler.encode_domain(domain)
        return self.handler.match(context, None, attrs, metas)

    def test_initialize_sets_current_context(self):
        self.widget = MockWidget()
        del self.widget.current_context
        self.handler.initialize(self.widget)
        self.assertIs(self.widget.current_context, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
