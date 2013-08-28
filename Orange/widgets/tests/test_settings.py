import unittest
from Orange.data import ContinuousVariable, DiscreteVariable, Domain
from Orange.widgets.settings import DomainContextHandler

VarTypes = ContinuousVariable.VarTypes


class DomainContextSettingsHandlerTests(unittest.TestCase):
    def setUp(self):
        self.handler = DomainContextHandler(attributes_in_res=True,
                                            metas_in_res=True)
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

        self.assertEqual(encoded_attributes, None)
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
        self.assertEqual(encoded_metas, None)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
