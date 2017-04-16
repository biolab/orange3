from timetest import *

if ORANGE3:
    SVM = Orange.classification.NuSVMLearner(nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001)
else:
    from Orange.classification.svm import SVMLearner
    SVM = SVMLearner(svm_type=SVMLearner.Nu_SVC, 
        kernel_type=SVMLearner.RBF, kernel_func=None, gamma=0.0, degree=3, normalization=False, eps=0.001, nu=0.5)

class TestSimpleTree(TimeTest):
    
    def setUp(self):
        self.data = Orange.data.Table("iris.tab")

    def test_simpleTreeUse(self):
        SVM(self.data)

class TestSimpleTreeA(TimeTest):
    
    def setUp(self):
        self.data = Orange.data.Table("adult_sample.tab")

    def test_simpleTreeUse(self):
        SVM(self.data)

class TestSimpleTreeC(TimeTest):
    
    def setUp(self):
        self.data = Orange.data.Table("car.tab")

    def test_simpleTreeUse(self):
        SVM(self.data)

if __name__ == '__main__':
    unittest.main()
