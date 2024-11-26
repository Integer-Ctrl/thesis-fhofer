import unittest
from sklearn.metrics import cohen_kappa_score


class TestCohensKappa(unittest.TestCase):

    labels = [0, 1, 1, 2, 0]

    def test_kappa_near_one_a(self):
        kappa = cohen_kappa_score(self.labels, self.labels)
        print('kappa_near_one_a:', kappa)
        self.assertGreaterEqual(kappa, 0.90)

    def test_kappa_near_one_b(self):
        labels_b = [0, 1, 1, 1, 0]
        kappa = cohen_kappa_score(self.labels, labels_b)
        print('kappa_near_one_b:', kappa)
        self.assertGreaterEqual(kappa, 0.90)

    def test_kappa_near_one_c(self):
        labels_c = [1, 1, 1, 2, 0]
        kappa = cohen_kappa_score(self.labels, labels_c)
        print('kappa_near_one_c:', kappa)
        self.assertGreaterEqual(kappa, 0.90)

    def test_kappa_near_one_d(self):
        labels_d = [0, 1, 1, 2, 1]
        kappa = cohen_kappa_score(self.labels, labels_d)
        print('kappa_near_one_d:', kappa)
        self.assertGreaterEqual(kappa, 0.90)

    # Define common rater inputs
    # rater_a = [0, 1, 2]
    # rater_b = [0, 0, 1]
    # rater_c = [2, 1, 0]

    # def test_kappa_near_one_aa(self):
    #     kappa = cohen_kappa_score(self.rater_a, self.rater_a)
    #     print('test_kappa_near_one_aa:', kappa)
    #     self.assertGreaterEqual(kappa, 0.7)

    # def test_kappa_near_negative_one_ac(self):
    #     kappa = cohen_kappa_score(self.rater_a, self.rater_c)
    #     print('test_kappa_near_negative_one_ac:', kappa)
    #     self.assertGreaterEqual(kappa, -0.7)

    # def test_kappa_near_negative_one_ca(self):
    #     kappa = cohen_kappa_score(self.rater_a, self.rater_c)
    #     print('test_kappa_near_negative_one_ca:', kappa)
    #     self.assertGreaterEqual(kappa, -0.7)

    # def test_kappanear_zero_ab(self):
    #     kappa = cohen_kappa_score(self.rater_a, self.rater_c)
    #     print('test_kappa_near_zero_ab:', kappa)
    #     self.assertAlmostEqual(kappa, 0.0, places=1)

    # def test_kappa_near_zero_ba(self):
    #     kappa = cohen_kappa_score(self.rater_a, self.rater_c)
    #     print('test_kappa_near_zero_ba:', kappa)
    #     self.assertAlmostEqual(kappa, 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
