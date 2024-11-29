import unittest
from greedy_series import GreedySeries


class TestPearson(unittest.TestCase):

    scores = GreedySeries([0.2, 0.7, 0.5])

    # Default implementation of pd.Series.corr() method
    def test_pearson_near_one_a(self):
        lables_a = GreedySeries([0, 2, 1])
        correlation = self.scores.corr(lables_a, method='pearson')
        self.assertAlmostEqual(correlation, 0.9933, places=3)
        print('pearson_near_one_a:', correlation)

    def test_pearson_near_one_b(self):
        lables_b = GreedySeries([0, 1, 1])
        correlation = self.scores.corr(lables_b, method='pearson')
        self.assertAlmostEqual(correlation, 0.9176, places=3)
        print('pearson_near_one_b:', correlation)

    def test_kendal_not_one_c(self):
        lables_c = GreedySeries([0, 0, 1])
        correlation = self.scores.corr(lables_c, method='pearson')
        self.assertAlmostEqual(correlation, 0.11470786693528091, places=3)
        print('kendal_not_one_c:', correlation)

    # Greedy implementation of pd.Series.corr() method
    def test_pearson_near_one_a_greedy(self):
        lables_a = GreedySeries([0, 2, 1])
        correlation = self.scores.corr(lables_a, method='pearson-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('pearson_near_one_a_greedy:', correlation)

    def test_pearson_near_one_b_greedy(self):
        lables_b = GreedySeries([0, 1, 1])
        correlation = self.scores.corr(lables_b, method='pearson-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('pearson_near_one_b_greedy:', correlation)

    def test_kendal_not_one_c_greedy(self):
        lables_c = GreedySeries([0, 0, 1])
        correlation = self.scores.corr(lables_c, method='pearson-greedy')
        self.assertAlmostEqual(correlation, 0.4999, places=3)
        print('kendal_not_one_c_greedy:', correlation)

    # scores = GreedySeries([0.2, 0.7, 0.8, 0.5, 0.3])

    # # def test_pearson_near_one_a(self):
    # #     correlation = self.scores.corr(self.scores, method='pearson')
    # #     self.assertAlmostEqual(correlation, 1.0, places=3)
    # #     print('pearson_near_one_a:', correlation)

    # def test_pearson_near_one_b(self):
    #     scores_b = GreedySeries([0, 2, 2, 1, 0])
    #     correlation = self.scores.corr(scores_b, method='pearson')
    #     self.assertAlmostEqual(correlation, 0.9805, places=3)
    #     print('pearson_near_one_b:', correlation)

    # def test_pearson_near_one_c(self):
    #     scores_c = GreedySeries([0, 1, 2, 1, 0])
    #     correlation = self.scores.corr(scores_c, method='pearson')
    #     self.assertAlmostEqual(correlation, 0.9376, places=3)
    #     print('pearson_near_one_c:', correlation)

    # def test_pearson_near_one_d(self):
    #     scores_d = GreedySeries([0, 0, 1, 0, 0])
    #     correlation = self.scores.corr(scores_d, method='pearson')
    #     self.assertAlmostEqual(correlation, 0.6577, places=3)
    #     print('pearson_near_one_d:', correlation)

    # def test_pearson_not_one_e(self):
    #     scores_d = GreedySeries([0, 0, 1, 1, 0])
    #     correlation = self.scores.corr(scores_d, method='pearson')
    #     self.assertAlmostEqual(correlation, 0.5370, places=3)
    #     print('pearson_not_one_e:', correlation)

    # # def test_pearson_near_one_a_greedy(self):
    # #     correlation = self.scores.corr(self.scores, method='pearson-greedy')
    # #     self.assertAlmostEqual(correlation, 1.0, places=3)
    # #     print('pearson_near_one_a_greedy:', correlation)

    # def test_pearson_near_one_b_greedy(self):
    #     scores_b = GreedySeries([0, 2, 2, 1, 0])
    #     correlation = self.scores.corr(scores_b, method='pearson-greedy')
    #     self.assertAlmostEqual(correlation, 1.0, places=3)
    #     print('pearson_near_one_b_greedy:', correlation)

    # def test_pearson_near_one_c_greedy(self):
    #     scores_c = GreedySeries([0, 1, 2, 1, 0])
    #     correlation = self.scores.corr(scores_c, method='pearson-greedy')
    #     self.assertAlmostEqual(correlation, 1.0, places=3)
    #     print('pearson_near_one_c_greedy:', correlation)

    # def test_pearson_near_one_d_greedy(self):
    #     scores_d = GreedySeries([0, 0, 1, 0, 0])
    #     correlation = self.scores.corr(scores_d, method='pearson-greedy')
    #     self.assertAlmostEqual(correlation, 1.0, places=3)
    #     print('pearson_near_one_d_greedy:', correlation)

    # def test_pearson_not_one_e_greedy(self):
    #     scores_d = GreedySeries([0, 0, 1, 1, 0])
    #     correlation = self.scores.corr(scores_d, method='pearson-greedy')
    #     self.assertAlmostEqual(correlation, 0.6666, places=3)
    #     print('pearson_not_one_e_greedy:', correlation)


if __name__ == '__main__':
    unittest.main()
