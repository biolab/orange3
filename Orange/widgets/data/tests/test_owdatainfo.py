import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse as sp

from Orange.data import \
    Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets.data.owdatainfo import OWDataInfo
from Orange.widgets.tests.base import WidgetTest


class TestOWDataInfo(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDataInfo)

    def test_data(self):
        # I guess we don't want to test specific output tests, just different
        # combinations that must not crash
        a, b, c = (DiscreteVariable(n) for n in "abc")
        x, y, z = (ContinuousVariable(n) for n in "xyz")
        m, n = (StringVariable(n) for n in "nm")
        self.widget.send_report()
        for attrs, classes, metas in (((a, b, c), (), ()),
                                      ((a, b, c, x), (y,), ()),
                                      ((a, b, c), (y, x), (m, )),
                                      ((a, b), (y, x, c), (m, )),
                                      ((a, ), (b, c), (m, )),
                                      ((a, b, x), (c, ), (m, y)),
                                      ((), (c, ), (m, y))):
            data = Table.from_numpy(
                Domain(attrs, classes, metas),
                np.zeros((3, len(attrs))),
                np.zeros((3, len(classes))),
                np.full((3, len(metas)), object()))
            data.attributes = {"att 1": 1, "att 2": True, "att 3": 3}
            if metas:
                data.name = "name"
            self.send_signal(self.widget.Inputs.data, data)
            self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()

        data.attributes = {"foo": "bar"}
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()

    def test_sparse(self):
        x, y, z, u, w = (ContinuousVariable(n) for n in "xyzuw")
        data = Table.from_numpy(
            Domain([x, y], z, [u, w]),
            sp.csc_matrix(np.random.randint(0, 1, (5, 2))),
            sp.csc_matrix(np.random.randint(0, 1, (5, 1))),
            sp.csc_matrix(np.random.randint(0, 1, (5, 2))))
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.send_report()

    def test_sql(self):
        class SqlTable(Table):
            connection_params = {"foo": "bar"}

        class Thread:
            def __init__(self, target):
                self.target = target

            def start(self):
                self.target()

        w = self.widget

        domain = Domain([ContinuousVariable("y")])

        with patch("Orange.widgets.data.owdatainfo.SqlTable", new=SqlTable), \
                patch("threading.Thread", new=Thread), \
                patch.object(self.widget, "_p_size", wraps=self.widget._p_size) as p_size:

            self.send_signal(w.Inputs.data, Table.from_numpy(domain, [[42]]))
            p_size.assert_called_once()
            p_size.reset_mock()

            d = SqlTable.from_numpy(domain, [[42]])
            self.send_signal(w.Inputs.data, d)
            self.assertEqual(p_size.call_count, 2)
            self.assertEqual(p_size.call_args_list[0], ((d, ),))
            self.assertEqual(p_size.call_args_list[1], ((d, ), dict(exact=True)))


if __name__ == "__main__":
    unittest.main()
