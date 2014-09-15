import unittest
import numpy as np
import anhima.ped

# build parents with combinations of genotypes
href_parent = np.zeros((1, 1, 2))
halt_parent = np.ones((1, 1, 2), dtype='int')
het_parent = np.array([0, 1], dtype='int').reshape((1, 1, 2))

# give progeny 4 genotypes, 00, 01, 11, --
progeny = np.array([[0, 0], [0, 1], [1, 1], [-1, -1]]).reshape(1, -1, 2)


class TestMendelianError(unittest.TestCase):
    def test_test(self):
        self.assertTrue(True)

    def test_function_exists(self):
        self.assertTrue('is_non_mendelian_diploid' in dir(anhima.ped))

    def test_homref_homref(self):
        parents = np.hstack([href_parent, href_parent])
        non_mendelian = anhima.ped.is_non_mendelian_diploid(
            parental_genotypes=parents,
            progeny_genotypes=progeny
        )
        print non_mendelian[0]
        self.assertTrue(np.array_equal(non_mendelian[0],
                                       [0, 1, 2, 0]))

    def test_homref_het(self):
        parents = np.hstack([href_parent, het_parent])
        non_mendelian = anhima.ped.is_non_mendelian_diploid(
            parental_genotypes=parents,
            progeny_genotypes=progeny
        )
        print non_mendelian[0]
        self.assertTrue(np.array_equal(non_mendelian[0], [0, 0, 1, 0]))

    def test_homref_homalt(self):
        parents = np.hstack([halt_parent, href_parent])
        non_mendelian = anhima.ped.is_non_mendelian_diploid(
            parental_genotypes=parents,
            progeny_genotypes=progeny,
        )
        print non_mendelian[0]
        self.assertTrue(np.array_equal(non_mendelian[0], [1, 0, 1, 0]))

    def test_homalt_het(self):
        parents = np.hstack([halt_parent, het_parent])
        non_mendelian = anhima.ped.is_non_mendelian_diploid(
            parental_genotypes=parents,
            progeny_genotypes=progeny,
        )
        print non_mendelian[0]
        self.assertTrue(np.array_equal(non_mendelian[0], [1, 0, 0, 0]))

    def test_homalt_homalt(self):
        parents = np.hstack([halt_parent, halt_parent])
        non_mendelian = anhima.ped.is_non_mendelian_diploid(
            parental_genotypes=parents,
            progeny_genotypes=progeny
        )
        print non_mendelian[0]
        self.assertTrue(np.array_equal(non_mendelian[0], [2, 1, 0, 0]))

    def test_error_multiallelic(self):
        parents = np.hstack([href_parent, halt_parent])
        progeny_multi = (
            np.array([[0, 2], [0, 1], [1, 2], [-1, -1]])
            .reshape(1, -1, 2)
        )

        self.assertRaises(
            AssertionError,
            anhima.ped.is_non_mendelian_diploid,
            parental_genotypes=parents,
            progeny_genotypes=progeny_multi
        )
