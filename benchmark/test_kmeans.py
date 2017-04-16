from timetest import *

class TestSimpleTree(TimeTest):
    
    def setUp(self):
        self.data = Orange.data.Table("adult.tab")
        if ORANGE3:
            continuizer = Orange.preprocess.Continuize()
            self.cl = Orange.clustering.KMeans(n_clusters=3)
            self.datac = continuizer(self.data)
        else:
            continuizer = Orange.data.continuization.DomainContinuizer()
            domain0 = continuizer(self.data)
            self.cl = Orange.clustering.kmeans.Clustering(centroids=3)
            self.datac = Orange.data.Table(domain0, self.data)

    def test_kmeans_continuized(self):
        self.cl(self.datac)

    def test_kmeans(self):
        self.cl(self.data)

if __name__ == '__main__':
    unittest.main()
