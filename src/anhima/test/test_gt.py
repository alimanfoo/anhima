from __future__ import division, print_function, unicode_literals, \
    absolute_import


import unittest
import numpy as np
from anhima.gt import as_012, as_n_alt, as_allele_counts, pack_diploid, \
    unpack_diploid, BHOM00, BHET01, BHOM11, BMISSING


class TestTransformations(unittest.TestCase):

    def setUp(self):
        self.genotypes = np.array(
            [[[0, 0], [0, 1]],
             [[1, 1], [-1, -1]],
             [[0, 2], [1, 2]],
             [[2, 2], [2, 0]]],
            dtype='i1'
        )

    def test_as_012(self):
        expect = np.array(
            [[0, 1],
             [2, -1],
             [1, 1],
             [2, 1]]
        )
        actual = as_012(self.genotypes)
        assert np.array_equal(expect, actual)

    def test_as_nalt(self):
        expect = np.array(
            [[0, 1],
             [2, 0],
             [1, 2],
             [2, 1]]
        )
        actual = as_n_alt(self.genotypes)
        assert np.array_equal(expect, actual)

    def test_as_allele_counts(self):
        expect = np.array(
            [[[2, 0, 0], [1, 1, 0]],
             [[0, 2, 0], [0, 0, 0]],
             [[1, 0, 1], [0, 1, 1]],
             [[0, 0, 2], [1, 0, 1]]]
        )
        actual = as_allele_counts(self.genotypes)
        assert np.array_equal(expect, actual)


class TestPackUnpackDiploid(unittest.TestCase):

    def test_pack_unpack_diploid(self):
        """Check that genotypes survive pack/unpack round trip."""

        genotypes = np.array([[[0, 0], [0, 1]],
                              [[1, 1], [-1, -1]],
                              [[6, 6], [7, 7]],
                              [[10, 10], [14, 14]]], dtype='i1')
        packed = pack_diploid(genotypes)
        self.assertEqual(BHOM00, packed[0, 0])
        self.assertEqual(BHET01, packed[0, 1])
        self.assertEqual(BHOM11, packed[1, 0])
        self.assertEqual(BMISSING, packed[1, 1])
        actual = unpack_diploid(packed)
        self.assertTrue(np.array_equal(genotypes, actual))

    def test_pack_diploid_allele_range(self):
        """Check that alleles outside of the supported range raise an error."""

        genotypes = np.array([[[-2, -2]]], dtype='i1')
        self.assertRaises(AssertionError,
                          pack_diploid,
                          genotypes)

        genotypes = np.array([[[15, 15]]], dtype='i1')
        self.assertRaises(AssertionError,
                          pack_diploid,
                          genotypes)

    def test_pack_diploid_ref_zero(self):
        """Check that hom ref calls are encoded as 0 for sparse matrices."""

        genotypes = np.array([[[0, 0]]], dtype='i1')
        packed = pack_diploid(genotypes)
        self.assertEqual(0, packed[0, 0])

