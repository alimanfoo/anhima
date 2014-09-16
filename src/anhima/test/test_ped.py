from __future__ import division, print_function, unicode_literals, \
    absolute_import


import unittest
import numpy as np
import anhima.ped


# build parents with combinations of genotypes
href_parent = np.zeros((1, 1, 2), dtype='int')
halt_parent = np.ones((1, 1, 2), dtype='int')
het_parent = np.array([0, 1], dtype='int').reshape((1, 1, 2))


# give progeny 4 genotypes, 00, 01, 11, --
progeny = np.array([[0, 0], [0, 1], [1, 1], [-1, -1]]).reshape(1, -1, 2)


class TestMendelianError(unittest.TestCase):

    def test_function_exists(self):
        self.assertTrue('diploid_mendelian_error' in dir(anhima.ped))

    def test_homref_homref(self):
        parents = np.hstack([href_parent, href_parent])
        non_mendelian = anhima.ped.diploid_mendelian_error(
            parental_genotypes=parents,
            progeny_genotypes=progeny
        )
        self.assertTrue(np.array_equal(non_mendelian[0],
                                       [0, 1, 2, 0]))

    def test_homref_het(self):
        parents = np.hstack([href_parent, het_parent])
        non_mendelian = anhima.ped.diploid_mendelian_error(
            parental_genotypes=parents,
            progeny_genotypes=progeny
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [0, 0, 1, 0]))

    def test_homref_homalt(self):
        parents = np.hstack([halt_parent, href_parent])
        non_mendelian = anhima.ped.diploid_mendelian_error(
            parental_genotypes=parents,
            progeny_genotypes=progeny,
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [1, 0, 1, 0]))

    def test_homalt_het(self):
        parents = np.hstack([halt_parent, het_parent])
        non_mendelian = anhima.ped.diploid_mendelian_error(
            parental_genotypes=parents,
            progeny_genotypes=progeny,
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [1, 0, 0, 0]))

    def test_homalt_homalt(self):
        parents = np.hstack([halt_parent, halt_parent])
        non_mendelian = anhima.ped.diploid_mendelian_error(
            parental_genotypes=parents,
            progeny_genotypes=progeny
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [2, 1, 0, 0]))

    def test_multiple_variants(self):
        parents = np.vstack([np.hstack([href_parent, href_parent]),
                             np.hstack([halt_parent, halt_parent]),
                             np.hstack([het_parent, halt_parent]),
                             np.hstack([het_parent, het_parent])])

        multiv_progeny = np.repeat(progeny, 4, axis=0)

        non_mendelian = anhima.ped.diploid_mendelian_error(
            parental_genotypes=parents,
            progeny_genotypes=multiv_progeny
        )

        expected_result = np.array([[0, 1, 2, 0],
                                    [2, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0]])
        self.assertTrue(np.array_equal(parents.shape, (4, 2, 2)))
        self.assertTrue(np.array_equal(non_mendelian.shape,
                                       expected_result.shape))

        self.assertTrue(np.array_equal(non_mendelian, expected_result))

    def test_error_multiallelic(self):
        parents = np.hstack([href_parent, halt_parent])
        progeny_multi = (
            np.array([[0, 2], [0, 1], [1, 2], [-1, -1]])
            .reshape((1, -1, 2))
        )

        self.assertRaises(
            AssertionError,
            anhima.ped.diploid_mendelian_error,
            parental_genotypes=parents,
            progeny_genotypes=progeny_multi
        )
