import unittest

import numpy as np

from distribution_compare import overlap_and_kl


class TestDistributionCompare(unittest.TestCase):

    def test_identical_sets_overlap_fully(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=2000)

        oa, kl = overlap_and_kl(a, a.copy())

        self.assertAlmostEqual(oa, 1.0, places=6)
        self.assertAlmostEqual(kl, 0.0, places=6)

    def test_shifted_set_overlaps_less(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=2000)

        oa, kl = overlap_and_kl(a, a + 3.0)

        self.assertLess(oa, 0.5)
        self.assertGreater(kl, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
