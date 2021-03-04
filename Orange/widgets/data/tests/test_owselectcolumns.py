# pylint: disable=unsubscriptable-object
import unittest
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from AnyQt.QtCore import QMimeData, QPoint, Qt
from AnyQt.QtGui import QDragEnterEvent


from Orange.data import Table, ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.settings import ContextSetting
from Orange.widgets.utils import vartype
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.data.owselectcolumns \
    import OWSelectAttributes, VariablesListItemModel, \
    SelectAttributesDomainContextHandler
from Orange.widgets.data.owrank import OWRank
from Orange.widgets.widget import AttributeList

Continuous = vartype(ContinuousVariable("c"))
Discrete = vartype(DiscreteVariable("d"))


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

        self.handler = SelectAttributesDomainContextHandler(first_match=False)
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
        domain = self.args[0]
        self.handler.open_context(widget, domain)
        self.assertEqual(widget.domain_role_hints,
                         {domain['d1']: ('available', 0),
                          domain['d2']: ('meta', 0),
                          domain['c1']: ('attribute', 0),
                          domain['d3']: ('attribute', 1),
                          domain['d4']: ('attribute', 2),
                          domain['c2']: ('class', 0)})

    def test_open_context_with_imperfect_match(self):
        self.handler.bind(SimpleWidget)
        context1 = Mock(values=dict(
            domain_role_hints=({('d1', Discrete): ('attribute', 0),
                                ('m2', Discrete): ('meta', 0)}, -2)
        ))
        context = Mock(values=dict(
            domain_role_hints=({('d1', Discrete): ('available', 0),
                                ('d2', Discrete): ('meta', 0),
                                ('c1', Continuous): ('attribute', 0),
                                ('d6', Discrete): ('attribute', 1),
                                ('d7', Discrete): ('attribute', 2),
                                ('c2', Continuous): ('class', 0)}, -2)
        ))
        self.handler.global_contexts = \
            [Mock(values={}), context1, context, Mock(values={})]

        widget = SimpleWidget()
        self.handler.initialize(widget)
        domain = self.args[0]
        self.handler.open_context(widget, domain)

        self.assertEqual(widget.domain_role_hints,
                         {domain['d1']: ('available', 0),
                          domain['d2']: ('meta', 0),
                          domain['c1']: ('attribute', 0),
                          domain['c2']: ('class', 0)})


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

    def assertVariableCountsEqual(self, available, used, classattrs, metas=0):
        self.assertEqual(len(self.widget.available_attrs), available)
        self.assertEqual(len(self.widget.used_attrs), used)
        self.assertEqual(len(self.widget.class_attrs), classattrs)
        self.assertEqual(len(self.widget.meta_attrs), metas)

    def assertControlsEnabled(self, _list, button, box, widget=None):
        if widget is None:
            widget = self.widget
        control = widget.use_features_box
        self.assertEqual(control.button.isEnabled(), button)
        self.assertEqual(control.isVisibleTo(widget), box)
        self.assertEqual(widget.used_attrs_view.isEnabled(), _list)
        self.assertEqual(widget.move_attr_button.isEnabled(), _list)
        if button:
            control.button.click()
            self.assertEqual(control.button.isEnabled(), False)

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
        self.assertVariableCountsEqual(0, 0, 5)

    def test_input_features(self):
        data = Table("zoo")
        in_features = AttributeList(data.domain.attributes)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, False, True)
        self.assertVariableCountsEqual(0, 16, 1, 1)
        self.assertListEqual(self.get_output(self.widget.Outputs.features),
                             list(data.domain.attributes))

    def test_input_features_by_name(self):
        data = Table("zoo")
        in_features = AttributeList([DiscreteVariable(attr.name, attr.values)
                                     for attr in data.domain.attributes])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, False, True)
        self.assertVariableCountsEqual(0, 16, 1, 1)
        self.assertListEqual(self.get_output(self.widget.Outputs.features),
                             list(data.domain.attributes))

    def test_input_features_same_domain(self):
        data = Table("zoo")
        in_features = AttributeList(data.domain.variables + data.domain.metas)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, False, True)
        self.assertVariableCountsEqual(0, 16, 1, 1)
        self.assertListEqual(self.get_output(self.widget.Outputs.features),
                             list(data.domain.attributes))

    def test_input_features_sub_domain(self):
        data = Table("zoo")
        in_features = AttributeList(data.domain.attributes[::3])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, True, True)
        self.assertVariableCountsEqual(10, 6, 1, 1)
        self.assertListEqual(self.get_output(self.widget.Outputs.features),
                             in_features)

    def test_input_features_by_name_sub_domain(self):
        data = Table("zoo")
        in_features = AttributeList([DiscreteVariable(attr.name, attr.values)
                                     for attr in data.domain.attributes[:5]])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, True, True)
        self.assertVariableCountsEqual(11, 5, 1, 1)
        self.assertListEqual(self.get_output(self.widget.Outputs.features),
                             list(data.domain.attributes[:5]))

    def test_input_features_diff_domain(self):
        zoo = Table("zoo")
        in_features = AttributeList(Table("iris").domain.attributes)
        self.send_signal(self.widget.Inputs.data, zoo)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(0, 16, 1, 1)
        self.assertListEqual(self.get_output(self.widget.Outputs.features),
                             list(zoo.domain.attributes))
        self.assertTrue(self.widget.Warning.mismatching_domain.is_shown())
        self.send_signal(self.widget.Inputs.features, None)
        self.assertFalse(self.widget.Warning.mismatching_domain.is_shown())

    def test_input_features_no_data(self):
        data = Table("zoo")
        in_features = AttributeList(data.domain.variables + data.domain.metas)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(0, 0, 0, 0)
        self.assertIsNone(self.get_output(self.widget.Outputs.features))

    def test_input_combinations(self):
        data = Table("iris")
        in_features = AttributeList(data.domain.attributes[:2])

        # check initial state
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(0, 0, 0, 0)
        self.assertIsNone(self.get_output(self.widget.Outputs.features))

        # send data
        self.send_signal(self.widget.Inputs.data, data)
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(0, 4, 1, 0)
        self.assertEqual(len(self.get_output(self.widget.Outputs.features)), 4)

        # send features
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, True, True)
        self.assertVariableCountsEqual(2, 2, 1, 0)
        self.assertEqual(len(self.get_output(self.widget.Outputs.features)), 2)

        # remove data
        self.send_signal(self.widget.Inputs.data, None)
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(0, 0, 0, 0)
        self.assertIsNone(self.get_output(self.widget.Outputs.features))

        # remove features
        self.send_signal(self.widget.Inputs.features, None)
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(0, 0, 0, 0)
        self.assertIsNone(self.get_output(self.widget.Outputs.features))

        # send features
        self.send_signal(self.widget.Inputs.features, in_features)
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(0, 0, 0, 0)
        self.assertIsNone(self.get_output(self.widget.Outputs.features))

        # send data
        self.send_signal(self.widget.Inputs.data, data)
        self.assertControlsEnabled(True, False, True)
        self.assertVariableCountsEqual(2, 2, 1, 0)
        self.assertEqual(len(self.get_output(self.widget.Outputs.features)), 2)

        # remove features
        self.send_signal(self.widget.Inputs.features, None)
        self.assertControlsEnabled(True, False, False)
        self.assertVariableCountsEqual(2, 2, 1, 0)
        self.assertEqual(len(self.get_output(self.widget.Outputs.features)), 2)

    def test_input_features_from_rank(self):
        data = Table("iris")
        owrank = self.create_widget(OWRank)
        self.send_signal(owrank.Inputs.data, data, widget=owrank)
        rank_features = self.get_output(owrank.Outputs.features, widget=owrank)
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features, rank_features)
        self.assertControlsEnabled(True, True, True)
        features = self.get_output(self.widget.Outputs.features)
        self.assertListEqual(rank_features, features)

    def test_use_features_checked(self):
        data = Table("iris")
        attrs = data.domain.attributes

        # prepare stored settings (check "Use input features")
        in_features = AttributeList(attrs[:2])
        self.send_signal(self.widget.Inputs.data, data)
        self.send_signal(self.widget.Inputs.features, in_features)
        self.widget.use_features_box.checkbox.setChecked(True)
        self.assertControlsEnabled(False, False, True)
        out_features = self.get_output(self.widget.Outputs.features)
        self.assertListEqual(out_features, in_features)
        settings = self.widget.settingsHandler.pack_data(self.widget)

        # "Use input features" is checked by default
        widget = self.create_widget(OWSelectAttributes, settings)
        in_features = AttributeList(attrs[:3])
        self.send_signal(widget.Inputs.data, data, widget=widget)
        self.send_signal(widget.Inputs.features, in_features, widget=widget)
        self.assertControlsEnabled(False, False, True, widget)
        out_features = self.get_output(widget.Outputs.features, widget=widget)
        self.assertListEqual(out_features, in_features)

        # reset "Features"
        widget.reset()
        out_features = self.get_output(widget.Outputs.features, widget=widget)
        self.assertFalse(widget.use_features_box.checkbox.isChecked())
        self.assertListEqual(out_features, AttributeList(attrs))
        self.assertControlsEnabled(True, True, True, widget)

    def test_used_attrs_supported_types(self):
        data = Table("zoo")
        event = self._drag_enter_event(data.domain[:1])
        self.widget.used_attrs_view.dragEnterEvent(event)
        self.assertTrue(event.isAccepted())

        event = self._drag_enter_event(data.domain.metas)
        self.widget.used_attrs_view.dragEnterEvent(event)
        self.assertFalse(event.isAccepted())

    def _drag_enter_event(self, variables):
        # pylint: disable=attribute-defined-outside-init
        self.event_data = mime = QMimeData()
        mime.setProperty("_items", variables)
        return QDragEnterEvent(QPoint(0, 0), Qt.MoveAction, mime,
                               Qt.NoButton, Qt.NoModifier)

    def test_move_rows(self):
        data = Table("iris")[:5]
        w = self.widget
        self.send_signal(w.Inputs.data, data)
        view = w.used_attrs_view
        model = view.model()
        selmodel = view.selectionModel()
        midx = model.index(1, 0)
        selmodel.select(midx, selmodel.ClearAndSelect)

        w.move_up(view)
        d1 = self.get_output(w.Outputs.data, w)
        self.assertEqual(
            d1.domain.attributes,
            data.domain.attributes[:2][::-1] + data.domain.attributes[2:]
        )
        w.move_down(view)
        d1 = self.get_output(w.Outputs.data, w)
        self.assertEqual(
            d1.domain.attributes,
            data.domain.attributes
        )

    def test_domain_new_feature(self):
        """ Test scenario when new attribute is added at position 0 """
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)

        data1 = Table(
            Domain(
                (ContinuousVariable("a"),) + data.domain.attributes,
                data.domain.class_var),
            np.hstack((np.ones((len(data), 1)), data.X)),
            data.Y
        )
        self.send_signal(self.widget.Inputs.data, data1)

    def test_select_new_features(self):
        """
        When ignore_new_features unchecked new attributes must appear in one of
        selected columns. Test with fist make context remember attributes of
        reduced domain and then testing with full domain. Features in missing
        in reduced domain must appears as seleceted.
        """
        data = Table("iris")
        domain = data.domain

        # data with one feature missing
        new_domain = Domain(
            domain.attributes[:-1], domain.class_var, domain.metas
        )
        new_data = Table.from_table(new_domain, data)

        # make context remember features in reduced domain
        self.send_signal(self.widget.Inputs.data, new_data)
        output = self.get_output(self.widget.Outputs.data)

        self.assertTupleEqual(
            new_data.domain.attributes, output.domain.attributes
        )
        self.assertTupleEqual(new_data.domain.metas, output.domain.metas)
        self.assertEqual(new_data.domain.class_var, output.domain.class_var)

        # send full domain
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)

        # if select_new_features checked all new features goes in the selected
        # features columns - domain equal original
        self.assertFalse(self.widget.ignore_new_features)
        self.assertTupleEqual(data.domain.attributes, output.domain.attributes)
        self.assertTupleEqual(data.domain.metas, output.domain.metas)
        self.assertEqual(data.domain.class_var, output.domain.class_var)

    def test_unselect_new_features(self):
        """
        When ignore_new_features checked new attributes must appear in one
        available attributes column. Test with fist make context remember
        attributes of reduced domain and then testing with full domain.
        Features in missing in reduced domain must appears as not seleceted.
        """
        data = Table("iris")
        domain = data.domain

        # data with one feature missing
        new_domain = Domain(
            domain.attributes[:-1], domain.class_var, domain.metas
        )
        new_data = Table.from_table(new_domain, data)

        # make context remember features in reduced domain
        self.send_signal(self.widget.Inputs.data, new_data)
        # select ignore_new_features
        self.widget.controls.ignore_new_features.click()
        self.assertTrue(self.widget.ignore_new_features)
        output = self.get_output(self.widget.Outputs.data)

        self.assertTupleEqual(
            new_data.domain.attributes, output.domain.attributes
        )
        self.assertTupleEqual(new_data.domain.metas, output.domain.metas)
        self.assertEqual(new_data.domain.class_var, output.domain.class_var)

        # send full domain
        self.send_signal(self.widget.Inputs.data, data)
        output = self.get_output(self.widget.Outputs.data)

        # if ignore_new_features checked all new features goes in the
        # available attributes column
        self.assertTrue(self.widget.ignore_new_features)
        self.assertTupleEqual(new_domain.attributes, output.domain.attributes)
        self.assertTupleEqual(new_domain.metas, output.domain.metas)
        self.assertEqual(new_domain.class_var, output.domain.class_var)
        # test if new attribute was added to unselected attributes
        self.assertEqual(
            domain.attributes[-1], list(self.widget.available_attrs)[0]
        )


if __name__ == "__main__":
    unittest.main()
