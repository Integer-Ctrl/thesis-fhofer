import unittest
from greedy_series import GreedySeries


class TestKendall(unittest.TestCase):

    scores = GreedySeries([0.2, 0.7, 0.5])

    # Default implementation of pd.Series.corr() method
    def test_kendall_near_one_a(self):
        lables_a = GreedySeries([0, 2, 1])
        correlation = self.scores.corr(lables_a, method='kendall')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('kendall_near_one_a:', correlation)

    def test_kendall_near_one_b(self):
        lables_b = GreedySeries([0, 1, 1])
        correlation = self.scores.corr(lables_b, method='kendall')
        self.assertAlmostEqual(correlation, 0.8164, places=3)
        print('kendall_near_one_b:', correlation)

    def test_kendal_not_one_c(self):
        lables_c = GreedySeries([0, 0, 1])
        correlation = self.scores.corr(lables_c, method='kendall')
        self.assertAlmostEqual(correlation, 0.0, places=3)
        print('kendal_not_one_c:', correlation)

    # Greedy implementation of pd.Series.corr() method
    def test_kendall_near_one_a_greedy(self):
        lables_a = GreedySeries([0, 2, 1])
        correlation = self.scores.corr(lables_a, method='kendall-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('kendall_near_one_a_greedy:', correlation)

    def test_kendall_near_one_b_greedy(self):
        lables_b = GreedySeries([0, 1, 1])
        correlation = self.scores.corr(lables_b, method='kendall-greedy')
        self.assertAlmostEqual(correlation, 1.0, places=3)
        print('kendall_near_one_b_greedy:', correlation)

    def test_kendal_not_one_c_greedy(self):
        lables_c = GreedySeries([0, 0, 1])
        correlation = self.scores.corr(lables_c, method='kendall-greedy')
        self.assertAlmostEqual(correlation, 0.4999, places=3)
        print('kendal_not_one_c_greedy:', correlation)

    # scores = GreedySeries([0.2, 0.7, 0.8, 0.5, 0.3])

    # # def test_kendall_near_one_a(self):
    # #     correlation = self.scores.corr(self.scores, method='kendall')
    # #     self.assertAlmostEqual(correlation, 1.0, places=3)
    # #     print('kendall_near_one_a:', correlation)

    # def test_kendall_near_one_b(self):
    #     scores_b = GreedySeries([0, 2, 2, 1, 0])
    #     correlation = self.scores.corr(scores_b, method='kendall')
    #     self.assertAlmostEqual(correlation, 0.8944, places=3)
    #     print('kendall_near_one_b:', correlation)

    # def test_kendall_near_one_c(self):
    #     scores_c = GreedySeries([0, 1, 2, 1, 0])
    #     correlation = self.scores.corr(scores_c, method='kendall')
    #     self.assertAlmostEqual(correlation, 0.8944, places=3)
    #     print('kendall_near_one_c:', correlation)

    # def test_kendall_near_one_d(self):
    #     scores_d = GreedySeries([0, 0, 1, 0, 0])
    #     correlation = self.scores.corr(scores_d, method='kendall')
    #     self.assertAlmostEqual(correlation, 0.6324, places=3)
    #     print('kendall_near_one_d:', correlation)

    # def test_kendall_not_one_e(self):
    #     scores_d = GreedySeries([0, 0, 1, 1, 0])
    #     correlation = self.scores.corr(scores_d, method='kendall')
    #     self.assertAlmostEqual(correlation, 0.5163, places=3)
    #     print('kendall_not_one_e:', correlation)

    # # def test_kendall_near_one_a_greedy(self):
    # #     correlation = self.scores.corr(self.scores, method='kendall-greedy')
    # #     self.assertAlmostEqual(correlation, 1.0, places=3)
    # #     print('kendall_near_one_a_greedy:', correlation)

    # def test_kendall_near_one_b_greedy(self):
    #     scores_b = GreedySeries([0, 2, 2, 1, 0])
    #     correlation = self.scores.corr(scores_b, method='kendall-greedy')
    #     self.assertAlmostEqual(correlation, 1.0, places=3)
    #     print('kendall_near_one_b_greedy:', correlation)

    # def test_kendall_near_one_c_greedy(self):
    #     scores_c = GreedySeries([0, 1, 2, 1, 0])
    #     correlation = self.scores.corr(scores_c, method='kendall-greedy')
    #     self.assertAlmostEqual(correlation, 1.0, places=3)
    #     print('kendall_near_one_c_greedy:', correlation)

    # def test_kendall_near_one_d_greedy(self):
    #     scores_d = GreedySeries([0, 0, 1, 0, 0])
    #     correlation = self.scores.corr(scores_d, method='kendall-greedy')
    #     self.assertAlmostEqual(correlation, 1.0, places=3)
    #     print('kendall_near_one_d_greedy:', correlation)

    # def test_kendall_not_one_e_greedy(self):
    #     scores_d = GreedySeries([0, 0, 1, 1, 0])
    #     correlation = self.scores.corr(scores_d, method='kendall-greedy')
    #     self.assertAlmostEqual(correlation, 0.6666, places=3)
    #     print('kendall_not_one_e_greedy:', correlation)


if __name__ == '__main__':
    unittest.main()
