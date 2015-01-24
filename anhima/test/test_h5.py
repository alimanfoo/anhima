# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


import unittest
import os
import anhima.h5
import h5py
import logging


logger = logging.getLogger(__name__)
debug = logger.debug


class TestLoadRegion(unittest.TestCase):

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

        self.assertTupleEqual(self.var['POS'].shape, self.var['QD'].shape)

    def test_load_samples(self):

        self.samples = ['AC0090-C', 'AC0091-C', 'AC0092-C']
        self.var, self.gt = anhima.h5.load_region(
            self.h5_file, '2L', self.start, self.stop,
            variants_fields=['POS', 'QD', 'DP', 'AC', 'AF'],
            calldata_fields=['genotype', 'AD', 'DP', 'GQ'],
            samples=self.samples)

        self.assertTupleEqual(self.var['POS'].shape, self.var['QD'].shape)
        self.assertEqual(self.var['POS'].size, 11)

        self.assertTupleEqual(self.gt['genotype'].shape, (11, 3, 2))
