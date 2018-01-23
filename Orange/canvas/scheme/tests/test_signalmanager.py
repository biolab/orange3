import unittest

from Orange.canvas.scheme import signalmanager


class TestSCC(unittest.TestCase):
    def test_scc(self):
        E1 = {}
        scc = signalmanager.strongly_connected_components(E1, E1.__getitem__)
        self.assertEqual(scc, [])

        E2 = {1: []}
        scc = signalmanager.strongly_connected_components(E2, E2.__getitem__)
        self.assertEqual(scc, [[1]])

        T1 = {1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [], 5: [], 6: [], 7: []}
        scc = signalmanager.strongly_connected_components(T1, T1.__getitem__)
        self.assertEqual(scc, [[4], [5], [2], [6], [7], [3], [1]])

        C1 = {1: [2], 2: [3], 3: [1]}
        scc = signalmanager.strongly_connected_components(C1, C1.__getitem__)
        self.assertEqual(scc, [[1, 2, 3]])

        G1 = {1: [2, 3], 2: [3, 5], 3: [], 5: [2]}
        scc = signalmanager.strongly_connected_components(G1, G1.__getitem__)
        self.assertEqual(scc, [[3], [2, 5], [1]])

        DAG1 = {1: [2, 3], 2: [3], 3: [4], 4: []}
        scc = signalmanager.strongly_connected_components(
            DAG1, DAG1.__getitem__)
        self.assertEqual(scc, [[4], [3], [2], [1]])

        G2 = {1: [2], 2: [1, 5], 3: [4], 4: [3, 5], 5: [6],
              6: [7], 7: [8], 8: [6, 9], 9: []}
        scc = signalmanager.strongly_connected_components(G2, G2.__getitem__)
        self.assertEqual(scc, [[9], [6, 7, 8], [5], [1, 2], [3, 4]])

        G3 = {1: [2], 2: [3], 3: [1],
              4: [5, 3], 5: [4, 6],
              6: [3, 7], 7: [6],
              8: [8]}
        scc = signalmanager.strongly_connected_components(G3, G3.__getitem__)
        self.assertEqual(scc, [[1, 2, 3], [6, 7], [4, 5], [8]])


def suite():
    test_suite = unittest.TestSuite()
    for ui in (TestSCC, ):
        test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(ui))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
