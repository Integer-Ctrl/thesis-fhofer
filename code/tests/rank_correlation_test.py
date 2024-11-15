# tests/test_module1.py

import unittest
import pandas as pd


class TestModule1(unittest.TestCase):

    scores_a = pd.Series([0.2, 0.4, 0.8])
    scores_b = pd.Series([0.4, 0.8, 0.9])
    scores_c = pd.Series([0.9, 0.6, 0.3])

    # Test cases for kendall rank correlation
    def test_kendall_near_one_ab(self):
        correlation = self.scores_a.corr(self.scores_b, method='kendall')
        self.assertEqual(correlation, 1.0)

    def test_kendall_near_one_ba(self):
        correlation = self.scores_b.corr(self.scores_a, method='kendall')
        self.assertEqual(correlation, 1.0)

    def test_kendall_near_negative_one_ac(self):
        correlation = self.scores_a.corr(self.scores_c, method='kendall')
        self.assertEqual(correlation, -1.0)

    def test_kendall_near_negative_one_ca(self):
        correlation = self.scores_c.corr(self.scores_a, method='kendall')
        self.assertEqual(correlation, -1.0)

    # Test cases for pearson rank correlation
    def test_pearson_near_one_ab(self):
        correlation = self.scores_a.corr(self.scores_b, method='pearson')
        # 0.85 <= correlation <= 0.9
        self.assertGreaterEqual(correlation, 0.85)
        self.assertGreaterEqual(0.9, correlation)

    def test_pearson_near_one_ba(self):
        correlation = self.scores_b.corr(self.scores_a, method='pearson')
        # 0.85 <= correlation <= 0.9
        self.assertGreaterEqual(correlation, 0.85)
        self.assertGreaterEqual(0.9, correlation)

    def test_pearson_near_zero_ac(self):
        correlation = self.scores_a.corr(self.scores_c, method='pearson')
        # -1.0 <= correlation <= -0.95
        self.assertGreaterEqual(correlation, -1.0)
        self.assertGreaterEqual(-0.95, correlation)

    def test_pearson_near_negative_one_ca(self):
        correlation = self.scores_c.corr(self.scores_a, method='pearson')
        # -1.0 <= correlation <= -0.95
        self.assertGreaterEqual(correlation, -1.0)
        self.assertGreaterEqual(-0.95, correlation)

    # Test cases for spearman rank correlation
    def test_spearman_near_one_ab(self):
        correlation = self.scores_a.corr(self.scores_b, method='spearman')
        self.assertEqual(correlation, 1.0)

    def test_spearman_near_one_ba(self):
        correlation = self.scores_b.corr(self.scores_a, method='spearman')
        self.assertEqual(correlation, 1.0)

    def test_spearman_near_negative_one_ac(self):
        correlation = self.scores_a.corr(self.scores_c, method='spearman')
        self.assertEqual(correlation, -1.0)

    def test_spearman_near_negative_one_ca(self):
        correlation = self.scores_c.corr(self.scores_a, method='spearman')
        self.assertEqual(correlation, -1.0)


if __name__ == '__main__':
    unittest.main()
