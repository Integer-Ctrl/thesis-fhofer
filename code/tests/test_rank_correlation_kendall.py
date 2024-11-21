import unittest
import pandas as pd

# class GreedySeries(pd.Series):
#     def corr(self, other, method = ..., min_periods = ...):
#         if '-greedy' in method:
#            return self.greedy_corr(other, method.replace('-greedy', ''), min_periods)
#         else:
#             return super().corr(other, method, min_periods)
        
#     def greedy_corr(self, other, method, min_periods):
#         thresholds = []
#         #
#         for i in range(0, len(self)):
#             if self[i] < other[i]:
#                 self[i] = 0


class TestKendall(unittest.TestCase):

    scores = pd.Series([0.2, 0.7, 0.8, 0.5, 0.3])

    def test_kendall_near_one_a(self):
        correlation = self.scores.corr(self.scores, method='kendall')
        self.assertGreaterEqual(correlation, 0.90)
        print('kendall_near_one_a:', correlation)

    def test_kendall_near_one_b(self):
        scores_b = pd.Series([0, 2, 2, 1, 0])
        correlation = self.scores.corr(scores_b, method='kendall')
        self.assertGreaterEqual(correlation, 0.90)
        print('kendall_near_one_b:', correlation)

    def test_kendall_near_one_c(self):
        scores_c = pd.Series([0, 1, 2, 1, 0])
        correlation = self.scores.corr(scores_c, method='kendall')
        self.assertGreaterEqual(correlation, 0.90)
        print('kendall_near_one_c:', correlation)

    def test_kendall_near_one_d(self):
        scores_d = pd.Series([0, 0, 1, 0, 0])
        correlation = self.scores.corr(scores_d, method='kendall')
        self.assertGreaterEqual(correlation, 0.90)
        print('kendall_near_one_d:', correlation)

    # scores_a = pd.Series([0.2, 0.4, 0.8])  # ground truth
    # scores_b = pd.Series([0.4, 0.8, 0.9])  # high correlation with scores_a
    # scores_c = pd.Series([0.9, 0.6, 0.3])  # negative correlation with scores_a

    # # Test cases for kendall rank correlation
    # def test_kendall_near_one_ab(self):
    #     correlation = self.scores_a.corr(self.scores_b, method='kendall')
    #     self.assertGreaterEqual(correlation, 0.95)
    #     print('test_kendall_near_one_ab:', correlation)

    # def test_kendall_near_one_ba(self):
    #     correlation = self.scores_b.corr(self.scores_a, method='kendall')
    #     self.assertGreaterEqual(correlation, 0.95)
    #     print('test_kendall_near_one_ba:', correlation)

    # def test_kendall_near_negative_one_ac(self):
    #     correlation = self.scores_a.corr(self.scores_c, method='kendall')
    #     self.assertGreaterEqual(-0.95, correlation)
    #     print('test_kendall_near_negative_one_ac:', correlation)

    # def test_kendall_near_negative_one_ca(self):
    #     correlation = self.scores_c.corr(self.scores_a, method='kendall')
    #     self.assertGreaterEqual(-0.95, correlation)
    #     print('test_kendall_near_negative_one_ca:', correlation)


if __name__ == '__main__':
    unittest.main()
