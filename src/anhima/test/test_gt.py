from __future__ import division, print_function, unicode_literals, \
    absolute_import


import unittest
import numpy as np
import anhima


class TestPackUnpackDiploid(unittest.TestCase):

    def test_pack_unpack_diploid(self):
        """Check that genotypes survive pack/unpack round trip."""

        genotypes = np.array([[[0, 0], [0, 1]],
                              [[1, 1], [-1, -1]],
                              [[6, 6], [7, 7]],
                              [[10, 10], [14, 14]]], dtype='i1')
        packed = anhima.gt.pack_diploid(genotypes)
        self.assertEqual(anhima.gt.BHOM00, packed[0, 0])
        self.assertEqual(anhima.gt.BHET01, packed[0, 1])
        self.assertEqual(anhima.gt.BHOM11, packed[1, 0])
        self.assertEqual(anhima.gt.BMISSING, packed[1, 1])
        actual = anhima.gt.unpack_diploid(packed)
        self.assertTrue(np.array_equal(genotypes, actual))

    def test_pack_diploid_allele_range(self):
        """Check that alleles outside of the supported range raise an error."""

        genotypes = np.array([[[-2, -2]]], dtype='i1')
        self.assertRaises(AssertionError,
                          anhima.gt.pack_diploid,
                          genotypes)

        genotypes = np.array([[[15, 15]]], dtype='i1')
        self.assertRaises(AssertionError,
                          anhima.gt.pack_diploid,
                          genotypes)

    def test_pack_diploid_ref_zero(self):
        """Check that hom ref calls are encoded as 0 for sparse matrices."""

        genotypes = np.array([[[0, 0]]], dtype='i1')
        packed = anhima.gt.pack_diploid(genotypes)
        self.assertEqual(0, packed[0, 0])

