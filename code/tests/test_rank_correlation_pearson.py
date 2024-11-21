import unittest
import pandas as pd


class TestPearson(unittest.TestCase):

    scores = pd.Series([0.2, 0.7, 0.8, 0.5, 0.3])

    def test_pearson_near_one_a(self):
        correlation = self.scores.corr(self.scores, method='pearson')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('pearson_near_one_a:', correlation)

    def test_pearson_near_one_b(self):
        scores_b = pd.Series([0, 2, 2, 1, 0])
        correlation = self.scores.corr(scores_b, method='pearson')
        self.assertAlmostEqual(correlation, 0.9805, places=3)
        print('pearson_near_one_b:', correlation)

    def test_pearson_near_one_c(self):
        scores_c = pd.Series([0, 1, 2, 1, 0])
        correlation = self.scores.corr(scores_c, method='pearson')
        self.assertAlmostEqual(correlation, 0.9376, places=3)
        print('pearson_near_one_c:', correlation)

    def test_pearson_near_one_d(self):
        scores_d = pd.Series([0, 0, 1, 0, 0])
        correlation = self.scores.corr(scores_d, method='pearson')
        self.assertAlmostEqual(correlation, 0.6577, places=3)
        print('pearson_near_one_d:', correlation)


if __name__ == '__main__':
    unittest.main()
