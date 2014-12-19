from __future__ import division, print_function, unicode_literals, \
    absolute_import


import unittest
import numpy as np
import os
import anhima.sim
import anhima.io
import scipy.stats
import tempfile


class TestTped(unittest.TestCase):

    def setUp(self):

        n_variants = 1000
        self.ref = np.random.choice(['A', 'C', 'G', 'T'], n_variants)
        self.alt = np.random.choice(['A', 'C', 'G', 'T'], n_variants)
        self.pos = np.random.choice(range(n_variants*100), n_variants, False)
        self.pos.sort()

        # simulate genotypes
        self.n_samples = 100
        ploidy = 2
        af_dist = scipy.stats.beta(a=.4, b=.6)
        p_missing = .1
        self.genotypes = anhima.sim.simulate_biallelic_genotypes(
            n_variants, self.n_samples, af_dist, p_missing, ploidy)

        self.n_variants = n_variants

    def test_file_created(self):

        path = tempfile.NamedTemporaryFile(delete=False)
        print(path.name)
        anhima.io.save_tped(path.name, self.genotypes, self.ref, self.alt,
                            self.pos)
        self.assertTrue(os.path.isfile(path.name))

        # read in file
        with open(path.name) as f:
            content = f.readlines()

        self.assertEqual(self.n_variants, len(content))

        # test that content is good...by taking first line
        line = content[0].split("\t")
        self.assertEquals(self.n_samples + 4, len(line))
        self.assertEquals('_'.join(line[0:4]),
                          '_'.join(['0', 'snp' + str(self.pos[0]),
                                    '0.0', str(self.pos[0])]))
    # test that we can load from hdf5 and create tped

    # test that we can write to hdf5 ok
