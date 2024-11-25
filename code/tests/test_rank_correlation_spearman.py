import unittest
from greedy_series import GreedySeries


class TestSpearman(unittest.TestCase):

    scores = GreedySeries([0.2, 0.7, 0.8, 0.5, 0.3])

    def test_spearman_near_one_a(self):
        correlation = self.scores.corr(self.scores, method='spearman')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('spearman_near_one_a:', correlation)

    def test_spearman_near_one_b(self):
        scores_b = GreedySeries([0, 2, 2, 1, 0])
        correlation = self.scores.corr(scores_b, method='spearman')
        self.assertAlmostEqual(correlation, 0.9486, places=3)
        print('spearman_near_one_b:', correlation)

    def test_spearman_near_one_c(self):
        scores_c = GreedySeries([0, 1, 2, 1, 0])
        correlation = self.scores.corr(scores_c, method='spearman')
        self.assertAlmostEqual(correlation, 0.9486, places=3)
        print('spearman_near_one_c:', correlation)

    def test_spearman_near_one_d(self):
        scores_d = GreedySeries([0, 0, 1, 0, 0])
        correlation = self.scores.corr(scores_d, method='spearman')
        self.assertAlmostEqual(correlation, 0.7071, places=3)
        print('spearman_near_one_d:', correlation)

    def test_spearman_near_one_a_greedy(self):
        correlation = self.scores.corr(self.scores, method='spearman-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('spearman_near_one_a:', correlation)

    def test_spearman_near_one_b_greedy(self):
        scores_b = GreedySeries([0, 2, 2, 1, 0])
        correlation = self.scores.corr(scores_b, method='spearman-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('spearman_near_one_b_greedy:', correlation)

    def test_spearman_near_one_c_greedy(self):
        scores_c = GreedySeries([0, 1, 2, 1, 0])
        correlation = self.scores.corr(scores_c, method='spearman-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('spearman_near_one_c_greedy:', correlation)

    def test_spearman_near_one_d_greedy(self):
        scores_d = GreedySeries([0, 0, 1, 0, 0])
        correlation = self.scores.corr(scores_d, method='spearman-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('spearman_near_one_d_greedy:', correlation)


if __name__ == '__main__':
    unittest.main()
