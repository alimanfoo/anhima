from __future__ import division, print_function, unicode_literals, \
    absolute_import

__author__ = 'Alistair Miles'

import unittest
import os
import anhima.h5
import h5py


class TestTped(unittest.TestCase):

    def setUp(self):

        ff = os.path.join('fixture', 'test32.h5')
        self.assertTrue(os.path.isfile(ff))

        self.h5_file = h5py.File(ff)
        self.samples = None
        self.start = 300000
        self.stop = 400000

    def test_load_nosamples(self):

        self.var, self.gt = anhima.h5.load_region(
            self.h5_file, '2L', self.start, self.stop,
            variants_fields=['POS', 'QD', 'DP', 'AC', 'AF'],
            calldata_fields=['genotype', 'AD', 'DP', 'GQ'],
            samples=self.samples)

        self.assertItemsEqual(self.var['POS'].shape, self.var['QD'].shape)

    def test_load_samples(self):

        self.samples = ['AC0090-C', 'AC0091-C', 'AC0092-C']
        self.var, self.gt = anhima.h5.load_region(
            self.h5_file, '2L', self.start, self.stop,
            variants_fields=['POS', 'QD', 'DP', 'AC', 'AF'],
            calldata_fields=['genotype', 'AD', 'DP', 'GQ'],
            samples=self.samples)

        self.assertItemsEqual(self.var['POS'].shape, self.var['QD'].shape)
        self.assertEqual(self.var['POS'].size, 11)

        self.assertItemsEqual(self.gt['genotype'].shape, (11, 3, 2))