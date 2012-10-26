import unittest
import numpy as np

from Orange import data
import Orange.classification.naive_bayes as nb

class BayesTest(unittest.TestCase):

    def test_dummyPass(self):
        self.assertEqual(1,1)

    def test_dummyFail(self):
        self.assertEqual(1,0)

    def test_runBayes(self):
        nrows = 10
        attributes = ["Feature %i" % i for i in range(10)]
        class_vars = ["Class %i" % i for i in range(3)]
        metas = ["Meta %i" % i for i in range(5)]
        x = np.random.random((nrows, len(attributes)))
        y = np.random.random((nrows, len(class_vars)))
        meta_data = np.random.random((nrows, len(metas)))

        attr_vars = [data.ContinuousVariable(name=a) if isinstance(a, str) else a for a in attributes]
        class_vars = [data.ContinuousVariable(name=c) if isinstance(c, str) else c for c in class_vars]
        meta_vars = [data.StringVariable(name=m) if isinstance(m, str) else m for m in metas]

        domain = data.Domain(attr_vars, class_vars)
        domain.metas = meta_vars

        t = data.Table(domain, x[:-1], y[:-1], meta_data)

        learn = nb.BayesLearner()
        clf = learn(t)
        print (clf(x[-1]), y[-1])
