import unittest

from Orange.data import Table, Domain, StringVariable
from Orange.widgets.model.owpls import OWPLS
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, \
    ParameterMapping


class TestOWPLS(WidgetTest, WidgetLearnerTestMixin):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._data = Table("housing")
        cls._data = cls._data.add_column(StringVariable("Foo"),
                                         ["Bar"] * len(cls._data),
                                         to_metas=True)
        class_vars = [cls._data.domain.class_var,
                      cls._data.domain.attributes[0]]
        domain = Domain(cls._data.domain.attributes[1:], class_vars,
                        cls._data.domain.metas)
        cls._data_multi_target = cls._data.transform(domain)

    def setUp(self):
        self.widget = self.create_widget(OWPLS,
                                         stored_settings={"auto_apply": False})
        self.init()
        self.parameters = [
            ParameterMapping('max_iter', self.widget.controls.max_iter),
            ParameterMapping('n_components', self.widget.controls.n_components)
        ]

    def test_output_coefsdata(self):
        self.send_signal(self.widget.Inputs.data, self._data)
        coefsdata = self.get_output(self.widget.Outputs.coefsdata)
        self.assertEqual(coefsdata.X.shape, (13, 1))
        self.assertEqual(coefsdata.Y.shape, (13, 0))
        self.assertEqual(coefsdata.metas.shape, (13, 1))

    def test_output_coefsdata_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        coefsdata = self.get_output(self.widget.Outputs.coefsdata)
        self.assertEqual(coefsdata.X.shape, (12, 2))
        self.assertEqual(coefsdata.Y.shape, (12, 0))
        self.assertEqual(coefsdata.metas.shape, (12, 1))

    def test_output_data(self):
        self.send_signal(self.widget.Inputs.data, self._data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(output.X.shape, (506, 13))
        self.assertEqual(output.Y.shape, (506,))
        self.assertEqual(output.metas.shape, (506, 5))

    def test_output_data_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(output.X.shape, (506, 12))
        self.assertEqual(output.Y.shape, (506, 2))
        self.assertEqual(output.metas.shape, (506, 5))

    def test_output_components(self):
        self.send_signal(self.widget.Inputs.data, self._data)
        components = self.get_output(self.widget.Outputs.components)
        self.assertEqual(components.X.shape, (2, 13))
        self.assertEqual(components.Y.shape, (2,))
        self.assertEqual(components.metas.shape, (2, 1))

    def test_output_components_multi_target(self):
        self.send_signal(self.widget.Inputs.data, self._data_multi_target)
        components = self.get_output(self.widget.Outputs.components)
        self.assertEqual(components.X.shape, (2, 12))
        self.assertEqual(components.Y.shape, (2, 2))
        self.assertEqual(components.metas.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
