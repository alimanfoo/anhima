# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


import unittest
import numpy as np
from anhima.ped import diploid_mendelian_error_biallelic, \
    diploid_mendelian_error_multiallelic


class TestMendelianErrorBiallelic(unittest.TestCase):

    def setUp(self):

        # give progeny 4 genotypes, 00, 01, 11, --
        self.progeny = np.array([[0, 0], [0, 1], [1, 1], [-1, -1]])\
            .reshape(1, -1, 2)

        # build parents with combinations of genotypes
        self.href_parent = np.zeros((1, 1, 2), dtype='int')
        self.halt_parent = np.ones((1, 1, 2), dtype='int')
        self.het_parent = np.array([0, 1], dtype='int').reshape((1, 1, 2))

    def test_homref_homref(self):
        parents = np.hstack([self.href_parent, self.href_parent])
        non_mendelian = diploid_mendelian_error_biallelic(
            parental_genotypes=parents,
            progeny_genotypes=self.progeny
        )
        self.assertTrue(np.array_equal(non_mendelian[0],
                                       [0, 1, 2, 0]))

    def test_homref_het(self):
        parents = np.hstack([self.href_parent, self.het_parent])
        non_mendelian = diploid_mendelian_error_biallelic(
            parental_genotypes=parents,
            progeny_genotypes=self.progeny
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [0, 0, 1, 0]))

    def test_homref_homalt(self):
        parents = np.hstack([self.halt_parent, self.href_parent])
        non_mendelian = diploid_mendelian_error_biallelic(
            parental_genotypes=parents,
            progeny_genotypes=self.progeny,
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [1, 0, 1, 0]))

    def test_homalt_het(self):
        parents = np.hstack([self.halt_parent, self.het_parent])
        non_mendelian = diploid_mendelian_error_biallelic(
            parental_genotypes=parents,
            progeny_genotypes=self.progeny,
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [1, 0, 0, 0]))

    def test_homalt_homalt(self):
        parents = np.hstack([self.halt_parent, self.halt_parent])
        non_mendelian = diploid_mendelian_error_biallelic(
            parental_genotypes=parents,
            progeny_genotypes=self.progeny
        )
        self.assertTrue(np.array_equal(non_mendelian[0], [2, 1, 0, 0]))

    def test_multiple_variants(self):
        parents = np.vstack([np.hstack([self.href_parent, self.href_parent]),
                             np.hstack([self.halt_parent, self.halt_parent]),
                             np.hstack([self.het_parent, self.halt_parent]),
                             np.hstack([self.het_parent, self.het_parent])])

        multiv_progeny = np.repeat(self.progeny, 4, axis=0)

        non_mendelian = diploid_mendelian_error_biallelic(
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


class TestMendelianErrorMultiallelic(unittest.TestCase):

    def _test(self, genotypes, expect):
        actual = diploid_mendelian_error_multiallelic(genotypes[:, 0:2],
                                                      genotypes[:, 2:], 4)
        assert np.array_equal(expect, actual)

        # swap parents, should have no affect
        actual = diploid_mendelian_error_multiallelic(genotypes[:, [1, 0]],
                                                      genotypes[:, 2:], 4)
        assert np.array_equal(expect, actual)

        # swap alleles, should have no effect
        actual = diploid_mendelian_error_multiallelic(genotypes[:, 0:2, ::-1],
                                                      genotypes[:, 2:, ::-1],
                                                      4)
        assert np.array_equal(expect, actual)

    def test_consistent(self):
        genotypes = np.array([
            # aa x aa -> aa
            [[0, 0], [0, 0], [0, 0], [-1, -1], [-1, -1], [-1, -1]],
            [[1, 1], [1, 1], [1, 1], [-1, -1], [-1, -1], [-1, -1]],
            [[2, 2], [2, 2], [2, 2], [-1, -1], [-1, -1], [-1, -1]],
            # aa x ab -> aa or ab
            [[0, 0], [0, 1], [0, 0], [0, 1], [-1, -1], [-1, -1]],
            [[0, 0], [0, 2], [0, 0], [0, 2], [-1, -1], [-1, -1]],
            [[1, 1], [0, 1], [1, 1], [0, 1], [-1, -1], [-1, -1]],
            # aa x bb -> ab
            [[0, 0], [1, 1], [0, 1], [-1, -1], [-1, -1], [-1, -1]],
            [[0, 0], [2, 2], [0, 2], [-1, -1], [-1, -1], [-1, -1]],
            [[1, 1], [2, 2], [1, 2], [-1, -1], [-1, -1], [-1, -1]],
            # aa x bc -> ab or ac
            [[0, 0], [1, 2], [0, 1], [0, 2], [-1, -1], [-1, -1]],
            [[1, 1], [0, 2], [0, 1], [1, 2], [-1, -1], [-1, -1]],
            # ab x ab -> aa or ab or bb
            [[0, 1], [0, 1], [0, 0], [0, 1], [1, 1], [-1, -1]],
            [[1, 2], [1, 2], [1, 1], [1, 2], [2, 2], [-1, -1]],
            [[0, 2], [0, 2], [0, 0], [0, 2], [2, 2], [-1, -1]],
            # ab x bc -> ab or ac or bb or bc
            [[0, 1], [1, 2], [0, 1], [0, 2], [1, 1], [1, 2]],
            [[0, 1], [0, 2], [0, 0], [0, 1], [0, 1], [1, 2]],
            # ab x cd -> ac or ad or bc or bd
            [[0, 1], [2, 3], [0, 2], [0, 3], [1, 2], [1, 3]],
        ])
        expect = np.zeros((17, 4))
        self._test(genotypes, expect)

    def test_error_nonparental(self):
        genotypes = np.array([
            # aa x aa -> ab or ac or bb or cc
            [[0, 0], [0, 0], [0, 1], [0, 2], [1, 1], [2, 2]],
            [[1, 1], [1, 1], [0, 1], [1, 2], [0, 0], [2, 2]],
            [[2, 2], [2, 2], [0, 2], [1, 2], [0, 0], [1, 1]],
            # aa x ab -> ac or bc or cc
            [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
            [[0, 0], [0, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
            [[1, 1], [0, 1], [1, 2], [0, 2], [2, 2], [2, 2]],
            # aa x bb -> ac or bc or cc
            [[0, 0], [1, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
            [[0, 0], [2, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
            [[1, 1], [2, 2], [0, 1], [0, 2], [0, 0], [0, 0]],
            # ab x ab -> ac or bc or cc
            [[0, 1], [0, 1], [0, 2], [1, 2], [2, 2], [2, 2]],
            [[0, 2], [0, 2], [0, 1], [1, 2], [1, 1], [1, 1]],
            [[1, 2], [1, 2], [0, 1], [0, 2], [0, 0], [0, 0]],
            # ab x bc -> ad or bd or cd or dd
            [[0, 1], [1, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
            [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
            [[0, 2], [1, 2], [0, 3], [1, 3], [2, 3], [3, 3]],
            # ab x cd -> ae or be or ce or de
            [[0, 1], [2, 3], [0, 4], [1, 4], [2, 4], [3, 4]],
        ])
        expect = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 2],
            [1, 1, 1, 1],
        ])
        self._test(genotypes, expect)

    def test_error_hemiparental(self):
        genotypes = np.array([
            # aa x ab -> bb
            [[0, 0], [0, 1], [1, 1], [-1, -1]],
            [[0, 0], [0, 2], [2, 2], [-1, -1]],
            [[1, 1], [0, 1], [0, 0], [-1, -1]],
            # ab x bc -> aa or cc
            [[0, 1], [1, 2], [0, 0], [2, 2]],
            [[0, 1], [0, 2], [1, 1], [2, 2]],
            [[0, 2], [1, 2], [0, 0], [1, 1]],
            # ab x cd -> aa or bb or cc or dd
            [[0, 1], [2, 3], [0, 0], [1, 1]],
            [[0, 1], [2, 3], [2, 2], [3, 3]],
        ])
        expect = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ])
        self._test(genotypes, expect)

    def test_error_uniparental(self):
        genotypes = np.array([
            # aa x bb -> aa or bb
            [[0, 0], [1, 1], [0, 0], [1, 1]],
            [[0, 0], [2, 2], [0, 0], [2, 2]],
            [[1, 1], [2, 2], [1, 1], [2, 2]],
            # ab x cd -> ab or cd
            [[0, 1], [2, 3], [0, 1], [2, 3]],
        ])
        expect = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ])
        self._test(genotypes, expect)

    def test_parent_missing(self):
        genotypes = np.array([
            [[-1, -1], [0, 0], [0, 0], [1, 1]],
            [[0, 0], [-1, -1], [0, 0], [2, 2]],
            [[-1, -1], [-1, -1], [1, 1], [2, 2]],
        ])
        expect = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
        ])
        self._test(genotypes, expect)
