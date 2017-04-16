from timetest import *

class TestLoad(TimeTest):
        
    def test_adult(self):
        Orange.data.Table("adult")

    def test_iris(self):
        Orange.data.Table("iris")

class TestDataAccess(TimeTest):
    
    def setUp(self):
        self.data = Orange.data.Table("adult")

    def test_str(self):
        for a in self.data:
            str(a)
    
    def test_read_values(self):
        for a in self.data:
            for b in a:
                c = b

    def test_read_values_str(self):
        for a in self.data:
            for b in a:
                c = str(b)

    def test_modify(self):
        for a in self.data:
            a[0] = 42.

    def test_copy_same(self):
        Orange.data.Table(self.data.domain, self.data)

    def test_copy_different_domain(self):
        dom = Orange.data.Domain(self.data.domain.attributes[4:10], self.data.domain.class_var)
        d = Orange.data.Table(dom, self.data)

if __name__ == '__main__':
    unittest.main()
